#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_dual_eval_audit_and_export.py
=============================================================================
Comprehensive Dual-Metric Audit on Hanoi 500 Benchmark:
- Computes both Conditional (Valid-Claim) and All-Clip (500-Clip Macro) metrics.
- Tracks per-clip eligibility masks, claim counts, and reference gold counts.
- Runs B=10,000 Hierarchical Cluster Bootstrap over 60 source video clusters.
- Computes paired differences against Full Proposed E6 with (b+1)/(B+1) p-values.
- Applies Holm-Bonferroni multiple testing correction.
- Exports complete per-clip CSV and summary JSON.
=============================================================================
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"
CONFIG_DIR = PHASE1_DIR / "config"
AUDIT_DIR = PHASE1_DIR / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load data
with open(CONFIG_DIR / "hanoi_manifest.jsonl") as f:
    hn_manifest = [json.loads(l) for l in f]

with open(PHASE1_DIR / "hanoi" / "evidence.jsonl") as f:
    hn_ev = {x["clip_id"]: x for x in [json.loads(l) for l in f]}

with open(REPO_ROOT / "scene_conditioned_gold" / "outputs" / "fgold_test_800_clips.json") as f:
    fgold_all = json.load(f)["clips"]

with open(REPO_ROOT / "data" / "caption_dataset" / "test_500_balanced_all_models_captions_scientific_vcgat.json") as f:
    retrained_caps = json.load(f)

with open(PHASE1_DIR / "hanoi" / "captions.jsonl") as f:
    p1_caps = [json.loads(l) for l in f]

# Lexicons
HEDGING_PATTERNS = [
    r'\blikely\b', r'\bappears to\b', r'\bpossibly\b', r'\bmight be\b', r'\bcould be\b',
    r'\bseems to\b', r'\bsuggests\b', r'\bmay have\b', r'\bif this is\b', r'\bbelieved to\b',
    r'\bprobably\b', r'\bpotentially\b', r'\bperhaps\b', r'\bmay include\b', r'\btypical of\b',
    r'\bdepending on\b', r'\bindicates\b', r'\bresembles\b', r'\bevokes\b', r'\breportedly\b',
    r'\bis said to\b', r'\bis traditionally associated with\b', r'\bthought to be\b',
    r'\baccording to\b', r'\btradition\b', r'\btraditions\b', r'\bfolklore\b', r'\blegend\b',
    r'\boral history\b', r'\blocal lore\b', r'\bcustoms\b', r'\brecords indicate\b',
    r'\baccounts suggest\b', r'\bit is said\b'
]
HEDGING_REGEX = re.compile('|'.join(HEDGING_PATTERNS), re.IGNORECASE)
GENERIC_TAILS = {
    'and', 'to', 'a', 'it', 'the', 'or', 'walls', 'street', 'building',
    'trees', 'roof', 'gate', 'clay', 'shops', 'floor', 'hall', 'waters', 'water'
}

def split_sentences(text: str) -> List[str]:
    cleaned = text.strip().replace("\n", " ")
    raw_sents = re.split(r'(?<=[.!?])\s+', cleaned)
    return [s.strip() for s in raw_sents if len(s.strip()) > 8]

def is_hedged_robust(s: str) -> bool:
    return bool(HEDGING_REGEX.search(s))

def is_fact_stated_robust(fact: dict, sentence: str, landmark_name: str) -> bool:
    s_lower = sentence.lower()
    tail = (fact.get("object") or fact.get("tail") or "").strip().lower()
    pred = (fact.get("predicate") or fact.get("relation") or "").strip().lower()
    lm = landmark_name.lower()

    if len(tail) < 3 or tail in GENERIC_TAILS:
        clean_pred = pred.replace("_", " ")
        return (clean_pred in s_lower and tail in s_lower)

    if tail in lm or lm in tail:
        clean_pred = pred.replace("_", " ")
        if clean_pred in s_lower and len(clean_pred) > 4:
            return True
        return False

    words = [w for w in tail.split() if len(w) > 3 and w not in GENERIC_TAILS]
    if len(words) >= 2:
        return sum(1 for w in words if w in s_lower) >= 2
    elif len(words) == 1:
        return words[0] in s_lower
    return False

# Load captions for comparative systems and ablation variants
caps_systems = {}
for s in ["E1", "E2", "E3", "E4", "E5", "E6"]:
    caps_systems[s] = [c["caption"] for c in p1_caps if c["system_id"] == s]

all_models = {
    # Comparative End-to-End Baselines (Table 7A)
    "E1_Qwen25VL": caps_systems["E1"],
    "E2_TextRAG": caps_systems["E2"],
    "E3_CrispKG": caps_systems["E3"],
    "E4_PromptOnly": caps_systems["E4"],
    "E5_BaseQwen3B": caps_systems["E5"],
    "E6_ProposedLoRA": caps_systems["E6"],
    # Retrained Component Ablation (Table 7)
    "Abl_Unanchored": retrained_caps["ours_unanchored"],
    "Abl_NoHorn": retrained_caps["wo_fuzzy_reasoning"],
    "Abl_FlatMuF": retrained_caps["wo_muf_weighting"],
    "Abl_NoCertifiedStopping": caps_systems["E6"],
    "Abl_Unconditioned": retrained_caps["wo_uncertainty_fusion"],
    "Abl_HardThreshold": retrained_caps["hard_threshold"],
    "Abl_Binary": retrained_caps["binary_seed42"],
    "Abl_Ternary": retrained_caps["ternary_seed42"],
}

n_clips = len(hn_manifest)
clip_sources = [c.get("source_video_id", "src_unknown") for c in hn_manifest]
unique_sources = sorted(list(set(clip_sources)))
n_sources = len(unique_sources)

print(f"Loaded {n_clips} clips across {n_sources} source-video clusters.")
print(f"Evaluating {len(all_models)} model configurations...")

# Compute per-clip metrics
per_clip_data = defaultdict(dict)
model_metrics = {}

for mname, clist in all_models.items():
    fp_cond, fp_elig = [], []
    csr_cond, csr_all, csr_elig = [], [], []
    bkr_cond, bkr_all = [], []
    kr_cond, kr_all, ref_elig = [], [], []
    vfar_cond, vfar_all, vfar_elig = [], [], []
    ns_list, nc_list, as_list, ac_list, nu_list = [], [], [], [], []

    for idx, c in enumerate(hn_manifest):
        cid = c["clip_id"]
        src = c.get("source_video_id", "src_unknown")
        ep = hn_ev[cid]
        lm = ep["landmark_name"]
        gdata = fgold_all[cid]
        gold_true = [f for f in gdata["gold_facts"] if f.get("review", {}).get("result") == "đúng"]
        gold_false = [f for f in gdata["gold_facts"] if f.get("review", {}).get("result") == "sai"]
        n_gold = len(gold_true)

        cap = clist[idx]
        sentences = [s for s in split_sentences(cap) if "distinguished heritage architecture" not in s.lower()]

        matched_gold_ids = set()
        asserted_gold_ids = set()
        for gf in gold_true:
            for s in sentences:
                if is_fact_stated_robust(gf, s, lm):
                    matched_gold_ids.add(gf["fact_id"])
                    if not is_hedged_robust(s):
                        asserted_gold_ids.add(gf["fact_id"])
                    break

        matched_neg_ids = set()
        asserted_neg_ids = set()
        for nf in gold_false:
            for s in sentences:
                if is_fact_stated_robust(nf, s, lm):
                    matched_neg_ids.add(nf["fact_id"])
                    if not is_hedged_robust(s):
                        asserted_neg_ids.add(nf["fact_id"])
                    break

        n_s = len(matched_gold_ids)
        n_c = len(matched_neg_ids)
        a_s = len(asserted_gold_ids)
        a_c = len(asserted_neg_ids)
        n_u = sum(1 for s in sentences if not (any(is_fact_stated_robust(gf, s, lm) for gf in gold_true) or any(is_fact_stated_robust(nf, s, lm) for nf in gold_false)))

        ns_list.append(n_s)
        nc_list.append(n_c)
        as_list.append(a_s)
        ac_list.append(a_c)
        nu_list.append(n_u)

        # 1. FP
        if (n_s + n_c) > 0:
            c_fp = n_s / (n_s + n_c)
            e_fp = 1
        else:
            c_fp = np.nan
            e_fp = 0
        fp_cond.append(c_fp)
        fp_elig.append(e_fp)

        # 2. CSR
        if (n_s + n_c + n_u) > 0:
            c_csr = n_s / (n_s + n_c + n_u)
            a_csr = c_csr
            e_csr = 1
        else:
            c_csr = np.nan
            a_csr = 0.0
            e_csr = 0
        csr_cond.append(c_csr)
        csr_all.append(a_csr)
        csr_elig.append(e_csr)

        # 3. BKR@5
        denom_bkr = min(5, n_gold)
        if denom_bkr > 0:
            c_bkr = min(5, n_s) / denom_bkr
            a_bkr = c_bkr
            e_ref = 1
        else:
            c_bkr = np.nan
            a_bkr = 0.0
            e_ref = 0
        bkr_cond.append(c_bkr)
        bkr_all.append(a_bkr)

        # 4. KR
        if n_gold > 0:
            c_kr = n_s / n_gold
            a_kr = c_kr
        else:
            c_kr = np.nan
            a_kr = 0.0
        kr_cond.append(c_kr)
        kr_all.append(a_kr)
        ref_elig.append(e_ref)

        # 5. VFAR
        if (a_s + a_c) > 0:
            c_vfar = a_c / (a_s + a_c)
            a_vfar = c_vfar
            e_vfar = 1
        else:
            c_vfar = np.nan
            a_vfar = 0.0
            e_vfar = 0
        vfar_cond.append(c_vfar)
        vfar_all.append(a_vfar)
        vfar_elig.append(e_vfar)

        # Per clip record
        per_clip_data[cid]["source_video_id"] = src
        per_clip_data[cid]["landmark_name"] = lm
        per_clip_data[cid]["n_gold"] = n_gold
        per_clip_data[cid][f"{mname}_ns"] = n_s
        per_clip_data[cid][f"{mname}_nc"] = n_c
        per_clip_data[cid][f"{mname}_as"] = a_s
        per_clip_data[cid][f"{mname}_ac"] = a_c
        per_clip_data[cid][f"{mname}_nu"] = n_u
        per_clip_data[cid][f"{mname}_fp_cond"] = c_fp
        per_clip_data[cid][f"{mname}_fp_elig"] = e_fp
        per_clip_data[cid][f"{mname}_csr_cond"] = c_csr
        per_clip_data[cid][f"{mname}_csr_all"] = a_csr
        per_clip_data[cid][f"{mname}_csr_elig"] = e_csr
        per_clip_data[cid][f"{mname}_bkr_cond"] = c_bkr
        per_clip_data[cid][f"{mname}_bkr_all"] = a_bkr
        per_clip_data[cid][f"{mname}_kr_cond"] = c_kr
        per_clip_data[cid][f"{mname}_kr_all"] = a_kr
        per_clip_data[cid][f"{mname}_vfar_cond"] = c_vfar
        per_clip_data[cid][f"{mname}_vfar_all"] = a_vfar
        per_clip_data[cid][f"{mname}_vfar_elig"] = e_vfar

    model_metrics[mname] = {
        "fp_cond": np.array(fp_cond, dtype=np.float64),
        "fp_elig": np.array(fp_elig, dtype=np.int32),
        "csr_cond": np.array(csr_cond, dtype=np.float64),
        "csr_all": np.array(csr_all, dtype=np.float64),
        "csr_elig": np.array(csr_elig, dtype=np.int32),
        "bkr_cond": np.array(bkr_cond, dtype=np.float64),
        "bkr_all": np.array(bkr_all, dtype=np.float64),
        "kr_cond": np.array(kr_cond, dtype=np.float64),
        "kr_all": np.array(kr_all, dtype=np.float64),
        "ref_elig": np.array(ref_elig, dtype=np.int32),
        "vfar_cond": np.array(vfar_cond, dtype=np.float64),
        "vfar_all": np.array(vfar_all, dtype=np.float64),
        "vfar_elig": np.array(vfar_elig, dtype=np.int32),
        "ns": np.array(ns_list, dtype=np.int32),
        "nc": np.array(nc_list, dtype=np.int32),
        "as": np.array(as_list, dtype=np.int32),
        "ac": np.array(ac_list, dtype=np.int32),
        "nu": np.array(nu_list, dtype=np.int32),
    }

# Export per-clip CSV
df_per_clip = pd.DataFrame.from_dict(per_clip_data, orient="index")
df_per_clip.index.name = "clip_id"
csv_path = AUDIT_DIR / "hanoi500_per_clip_dual_eval_audit.csv"
df_per_clip.to_csv(csv_path)
print(f"Exported per-clip audit artifact to: {csv_path}")

# Run B=10,000 Cluster Bootstrap
print("Running B=10,000 Hierarchical Cluster Bootstrap across 60 source videos...")
rng = np.random.RandomState(42)
B = 10000

src_to_indices = defaultdict(list)
for idx, src in enumerate(clip_sources):
    src_to_indices[src].append(idx)

boot_indices_list = []
for _ in range(B):
    sampled_srcs = rng.choice(unique_sources, size=n_sources, replace=True)
    idxs = []
    for s in sampled_srcs:
        idxs.extend(src_to_indices[s])
    boot_indices_list.append(np.array(idxs))

metric_names = [
    "fp_cond", "csr_cond", "csr_all",
    "bkr_cond", "bkr_all", "kr_cond", "kr_all",
    "vfar_cond", "vfar_all"
]

summary_stats = {}
ref_name = "E6_ProposedLoRA"

for mname, mdata in model_metrics.items():
    stats = {
        "model": mname,
        "n_clips": n_clips,
        "fp_eligible_clips": int(np.sum(mdata["fp_elig"])),
        "csr_eligible_clips": int(np.sum(mdata["csr_elig"])),
        "ref_eligible_clips": int(np.sum(mdata["ref_elig"])),
        "vfar_eligible_clips": int(np.sum(mdata["vfar_elig"])),
        "total_true_facts_ns": int(np.sum(mdata["ns"])),
        "total_false_facts_nc": int(np.sum(mdata["nc"])),
        "total_asserted_true_as": int(np.sum(mdata["as"])),
        "total_asserted_false_ac": int(np.sum(mdata["ac"])),
        "total_unmatched_nu": int(np.sum(mdata["nu"])),
        "micro_fp": float(np.sum(mdata["ns"]) / max(1, np.sum(mdata["ns"]) + np.sum(mdata["nc"]))),
        "micro_vfar": float(np.sum(mdata["ac"]) / max(1, np.sum(mdata["as"]) + np.sum(mdata["ac"]))),
    }

    # Bootstrap CIs for each metric
    for met in metric_names:
        arr = mdata[met]
        b_means = np.array([np.nanmean(arr[b_idx]) for b_idx in boot_indices_list])
        b_means = b_means[~np.isnan(b_means)]
        b_means.sort()
        mean_val = float(np.mean(b_means))
        ci_low = float(np.percentile(b_means, 2.5))
        ci_high = float(np.percentile(b_means, 97.5))
        stats[f"{met}_mean"] = mean_val
        stats[f"{met}_ci_low"] = ci_low
        stats[f"{met}_ci_high"] = ci_high
        stats[f"{met}_str"] = f"{mean_val:.4f} [{ci_low:.3f}, {ci_high:.3f}]"

    # Paired differences vs E6
    if mname == ref_name or mname == "Abl_NoCertifiedStopping":
        for met in metric_names:
            stats[f"diff_{met}_mean"] = 0.0
            stats[f"diff_{met}_ci_low"] = 0.0
            stats[f"diff_{met}_ci_high"] = 0.0
            stats[f"diff_{met}_pval"] = 1.0 if mname == ref_name else 1.0
            stats[f"diff_{met}_exact_match"] = True
    else:
        for met in metric_names:
            diff_arr = mdata[met] - model_metrics[ref_name][met]
            b_diffs = np.array([np.nanmean(diff_arr[b_idx]) for b_idx in boot_indices_list])
            b_diffs = b_diffs[~np.isnan(b_diffs)]
            b_diffs.sort()
            diff_mean = float(np.mean(b_diffs))
            ci_low = float(np.percentile(b_diffs, 2.5))
            ci_high = float(np.percentile(b_diffs, 97.5))
            
            # Rigorous (b+1)/(B+1) two-sided p-value
            b_pos = np.sum(b_diffs >= 0)
            b_neg = np.sum(b_diffs <= 0)
            b_extreme = min(b_pos, b_neg)
            pval = min(1.0, 2.0 * (b_extreme + 1) / (B + 1))

            stats[f"diff_{met}_mean"] = diff_mean
            stats[f"diff_{met}_ci_low"] = ci_low
            stats[f"diff_{met}_ci_high"] = ci_high
            stats[f"diff_{met}_pval"] = pval
            stats[f"diff_{met}_exact_match"] = False

    summary_stats[mname] = stats

# Apply Holm-Bonferroni correction across the 9 ablation variants for key metrics
ablation_var_names = [
    "E5_BaseQwen3B", "E4_PromptOnly", "Abl_Unanchored", "Abl_NoHorn",
    "Abl_FlatMuF", "Abl_Unconditioned", "Abl_HardThreshold", "Abl_Binary", "Abl_Ternary"
]

for met in ["csr_cond", "csr_all", "bkr_cond", "bkr_all", "kr_cond", "kr_all", "vfar_cond", "vfar_all"]:
    pvals = [(v, summary_stats[v][f"diff_{met}_pval"]) for v in ablation_var_names]
    pvals_sorted = sorted(pvals, key=lambda x: x[1])
    m_tests = len(pvals_sorted)
    
    holm_pvals = {}
    cum_max = 0.0
    for rank, (v, p) in enumerate(pvals_sorted):
        adj_p = min(1.0, p * (m_tests - rank))
        cum_max = max(cum_max, adj_p)
        holm_pvals[v] = min(1.0, cum_max)
    
    for v in ablation_var_names:
        summary_stats[v][f"diff_{met}_pval_holm"] = holm_pvals[v]

# Save summary JSON
json_path = AUDIT_DIR / "hanoi500_cluster_bootstrap_dual_summary.json"
with open(json_path, "w") as f:
    json.dump(summary_stats, f, indent=2)

print(f"Exported summary JSON artifact to: {json_path}")
print("Audit processing complete!")
