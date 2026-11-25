#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_paired_bootstrap_and_fairness.py
=============================================================================
Action 2: Fairness verification across all baselines (Before vs After).
Action 4: Paired Cluster Bootstrap Confidence Intervals (B=10,000) for
          E6 - E5 and E6 - Baselines (E1 to E4) with empirical p-values.
=============================================================================
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"
AUDIT_DIR = PHASE1_DIR / "audit"
TABLES_DIR = PHASE1_DIR / "tables"
CONFIG_DIR = PHASE1_DIR / "config"

# Load manifests and data
with open(CONFIG_DIR / "hanoi_manifest.jsonl") as f:
    hn_manifest = [json.loads(l) for l in f]
with open(CONFIG_DIR / "hcmc_manifest.jsonl") as f:
    hc_manifest = [json.loads(l) for l in f]

with open(PHASE1_DIR / "hanoi" / "evidence.jsonl") as f:
    hn_ev = [json.loads(l) for l in f]
with open(PHASE1_DIR / "hcmc" / "evidence.jsonl") as f:
    hc_ev = [json.loads(l) for l in f]

with open(PHASE1_DIR / "hanoi" / "captions.jsonl") as f:
    hn_caps = [json.loads(l) for l in f]
with open(PHASE1_DIR / "hcmc" / "captions.jsonl") as f:
    hc_caps = [json.loads(l) for l in f]

with open(REPO_ROOT / "scene_conditioned_gold" / "outputs" / "fgold_test_800_clips.json") as f:
    fgold_all = json.load(f)["clips"]

with open(REPO_ROOT / "results" / "two_level_silver_reference_train_val" / "test_500_canonical_scene_graphs.json") as f:
    hn_sg_map = json.load(f)

hcmc_sg_map = {}
with open(REPO_ROOT / "data" / "hcmc_benchmark" / "hcmc_300_x5_dataset.jsonl") as f:
    for l in f:
        if l.strip():
            it = json.loads(l)
            cid = f"hcmc_x5_{it['clip_idx']:04d}" if 'clip_idx' in it else it.get('sample_id')
            hcmc_sg_map[cid] = it.get("observed_scene_graph", "")

caps_map = {}
for c in hn_caps:
    caps_map[('hanoi', c['clip_id'], c['system_id'])] = c['caption']
for c in hc_caps:
    caps_map[('hcmc', c['clip_id'], c['system_id'])] = c['caption']

ev_map = {}
for ep in hn_ev:
    ev_map[('hanoi', ep['clip_id'])] = ep
for ep in hc_ev:
    ev_map[('hcmc', ep['clip_id'])] = ep

# Hedging lexicons
HEDGING_ORIG = [
    "likely", "appears to", "possibly", "might be", "could be",
    "seems to", "suggests", "may have", "if this is", "believed to",
    "probably", "potentially", "perhaps", "may include", "typical of",
    "depending on", "indicates", "resembles", "evokes", "reportedly",
    "is said to", "is traditionally associated with", "thought to be"
]

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

HISTORICAL_KEYWORDS = [
    "built in", "constructed in", "century", "dynasty", "emperor", "king",
    "french rule", "colonial", "founded in", "war", "relic", "ancient",
    "history", "historical", "revolution", "monument", "heritage", "designated",
    "ho chi minh", "nguyen dynasty", "ly dynasty", "tran dynasty", "le dynasty"
]

VISUAL_KEYWORDS = [
    "see", "visible", "clearly", "captures", "shows", "features", "wall", "gate",
    "courtyard", "roof", "tower", "bridge", "lake", "tree", "exhibit", "statue",
    "temple", "pagoda", "building", "hall", "plaza", "square", "color", "red",
    "yellow", "brick", "stone", "water", "stalls", "walkway", "arch", "display",
    "ceramic", "silk", "lantern", "flag", "structure", "facade", "column", "pavilion",
    "banner", "tunic", "hat", "boat", "drum", "vitrine"
]

def split_sentences(text: str) -> List[str]:
    cleaned = text.strip().replace("\n", " ")
    raw_sents = re.split(r'(?<=[.!?])\s+', cleaned)
    return [s.strip() for s in raw_sents if len(s.strip()) > 8]

def is_hedged_robust(s: str) -> bool:
    return bool(HEDGING_REGEX.search(s))

def is_stylistic_intro(sentence: str) -> bool:
    s_lower = sentence.lower()
    return "distinguished heritage architecture" in s_lower or "distinguished heritage" in s_lower

def classify_claim_robust(claim: str) -> str:
    if is_stylistic_intro(claim):
        return "INTRO_DISCOURSE"
    c_lower = claim.lower()
    has_hist = any(k in c_lower for k in HISTORICAL_KEYWORDS)
    has_vis = any(k in c_lower for k in VISUAL_KEYWORDS)
    if has_hist and not has_vis:
        return "HISTORICAL"
    elif has_vis and not has_hist:
        return "VISUAL"
    elif has_hist and has_vis:
        if any(w in c_lower for w in ["banner", "tunic", "document", "drum", "hat", "vitrine", "boat", "display", "statue"]):
            return "VISUAL"
        return "MIXED"
    else:
        return "HISTORICAL" if any(w in c_lower for w in ["hanoi", "vietnam", "century", "capital"]) else "VISUAL"

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

def paired_cluster_bootstrap(
    source_diffs: Dict[str, List[Optional[float]]],
    b_resamples: int = 10000,
    seed: int = 42
) -> Tuple[float, float, float, float]:
    """
    Computes mean difference, 95% CI [low, high], and empirical two-tailed p-value
    under paired hierarchical cluster bootstrap.
    """
    rng = np.random.RandomState(seed)
    src_keys = list(source_diffs.keys())
    n_src = len(src_keys)
    if n_src == 0:
        return 0.0, 0.0, 0.0, 1.0

    boot_diffs = []
    for _ in range(b_resamples):
        sampled_srcs = rng.choice(src_keys, size=n_src, replace=True)
        vals = []
        for s in sampled_srcs:
            for d in source_diffs[s]:
                if d is not None and not math.isnan(d):
                    vals.append(d)
        if vals:
            boot_diffs.append(float(np.mean(vals)))

    if not boot_diffs:
        return 0.0, 0.0, 0.0, 1.0

    boot_diffs.sort()
    mean_diff = float(np.mean(boot_diffs))
    low = float(np.percentile(boot_diffs, 2.5))
    high = float(np.percentile(boot_diffs, 97.5))

    # Empirical two-tailed p-value: H0: delta = 0
    p_le = sum(1 for x in boot_diffs if x <= 0) / len(boot_diffs)
    p_ge = sum(1 for x in boot_diffs if x >= 0) / len(boot_diffs)
    p_val = 2.0 * min(p_le, p_ge)
    p_val = min(1.0, max(0.0001, p_val))

    return mean_diff, low, high, p_val

print("Evaluating per-clip paired metrics...")

# Compute metrics per clip for all systems
systems = ["E1", "E2", "E3", "E4", "E5", "E6"]
metrics_list = ["fp", "csr", "bkr5", "kr", "vsp", "vfar", "eap", "oargen", "ovr"]

paired_results = []

for dom_name, dom_key, manifest, ev_list, sg_map in [
    ("HANOI", "hanoi", hn_manifest, hn_ev, hn_sg_map),
    ("HCMC", "hcmc", hc_manifest, hc_ev, hcmc_sg_map)
]:
    # clip_id -> sys_id -> metric -> value
    clip_metrics = defaultdict(lambda: defaultdict(dict))
    clip_source = {}

    for c in manifest:
        cid = c["clip_id"]
        src = c.get("source_video_id", "src_unknown")
        clip_source[cid] = src
        ep = ev_map[(dom_key, cid)]
        lm = ep["landmark_name"]
        sg_str = sg_map.get(cid, "")
        sg_concepts = [re.sub(r'\(\d+\.\d+\)', '', s).strip().lower() for s in sg_str.split('\n') if len(s.strip()) > 3]

        gfacts = fgold_all[cid].get("gold_facts", [])
        gold_true = [f for f in gfacts if f.get("review", {}).get("result") == "đúng"]
        gold_false = [f for f in gfacts if f.get("review", {}).get("result") == "sai"]
        n_gold = len(gold_true)

        for sid in systems:
            cap = caps_map[(dom_key, cid, sid)]
            sentences = split_sentences(cap)

            clip_ns = 0
            clip_nc = 0
            clip_nu = 0
            clip_as = 0
            clip_ac = 0
            clip_vis_s = 0
            clip_vis_tot = 0
            matched_gold_ids = set()
            matched_5_ids = set()
            hist_sentences = []

            for s in sentences:
                ctype = classify_claim_robust(s)
                if ctype == "INTRO_DISCOURSE":
                    continue

                if ctype in ("HISTORICAL", "MIXED"):
                    hist_sentences.append(s)
                    s_lower = s.lower()
                    is_hedged = is_hedged_robust(s)

                    matched_gf = None
                    for gf in gold_true:
                        tail = (gf.get("object") or gf.get("tail") or "").lower().strip()
                        pred = (gf.get("predicate") or gf.get("relation") or "").lower().strip().replace("_", " ")
                        fact_text = gf.get("fact_text", "").lower()
                        if len(tail) > 3 and (tail in s_lower or fact_text in s_lower or (pred in s_lower and len(pred) > 4)):
                            matched_gf = gf
                            break

                    if matched_gf:
                        clip_ns += 1
                        matched_gold_ids.add(matched_gf.get("fact_id", "f_unknown"))
                        if len(hist_sentences) <= 5:
                            matched_5_ids.add(matched_gf.get("fact_id", "f_unknown"))
                        if not is_hedged:
                            clip_as += 1
                    else:
                        matched_nf = None
                        for nf in gold_false:
                            tail = (nf.get("object") or nf.get("tail") or "").lower().strip()
                            pred = (nf.get("predicate") or nf.get("relation") or "").lower().strip().replace("_", " ")
                            fact_text = nf.get("fact_text", "").lower()
                            if len(tail) > 3 and (tail in s_lower or fact_text in s_lower):
                                matched_nf = nf
                                break
                        if matched_nf:
                            clip_nc += 1
                            if not is_hedged:
                                clip_ac += 1
                        else:
                            clip_nu += 1

                elif ctype == "VISUAL":
                    clip_vis_tot += 1
                    s_lower = s.lower()
                    matched_vc = False
                    for vc in sg_concepts:
                        if len(vc) > 3 and vc in s_lower:
                            matched_vc = True
                            break
                        words = [w for w in vc.split() if len(w) > 3]
                        if len(words) >= 2 and sum(1 for w in words if w in s_lower) >= 2:
                            matched_vc = True
                            break
                    if matched_vc:
                        clip_vis_s += 1

            # Store metrics
            m_fp = (clip_ns / (clip_ns + clip_nc)) if (clip_ns + clip_nc) > 0 else None
            m_csr = (clip_ns / (clip_ns + clip_nc + clip_nu)) if (clip_ns + clip_nc + clip_nu) > 0 else None
            denom_bkr = min(5, n_gold)
            m_bkr5 = (len(matched_5_ids) / denom_bkr) if denom_bkr > 0 else None
            m_kr = (len(matched_gold_ids) / n_gold) if n_gold > 0 else None
            m_vsp = (clip_vis_s / clip_vis_tot) if clip_vis_tot > 0 else None
            m_vfar = (clip_ac / (clip_as + clip_ac)) if (clip_as + clip_ac) > 0 else None

            clip_metrics[cid][sid]["fp"] = m_fp
            clip_metrics[cid][sid]["csr"] = m_csr
            clip_metrics[cid][sid]["bkr5"] = m_bkr5
            clip_metrics[cid][sid]["kr"] = m_kr
            clip_metrics[cid][sid]["vsp"] = m_vsp
            clip_metrics[cid][sid]["vfar"] = m_vfar

            if sid in ("E5", "E6"):
                omit_facts = [f for f in ep["top5_facts"] if f.get("policy") == "OMIT"]
                hedge_facts = [f for f in ep["top5_facts"] if f.get("policy") == "HEDGE"]
                top5_facts = ep["top5_facts"]

                # Robust OVR
                omit_stated = 0
                for of in omit_facts:
                    if any(is_fact_stated_robust(of, s, lm) for s in sentences):
                        omit_stated += 1
                m_ovr = (omit_stated / len(omit_facts)) if omit_facts else None

                # Robust OARgen
                over_assertions = 0
                for f in (omit_facts + hedge_facts):
                    for s in sentences:
                        if is_fact_stated_robust(f, s, lm) and not is_hedged_robust(s):
                            over_assertions += 1
                            break
                denom_oar = len(omit_facts) + len(hedge_facts)
                m_oargen = (over_assertions / denom_oar) if denom_oar > 0 else None

                # EAP
                eap_hits = 0
                for s in hist_sentences:
                    if any(is_fact_stated_robust(f, s, lm) for f in top5_facts):
                        eap_hits += 1
                m_eap = (eap_hits / len(hist_sentences)) if hist_sentences else None

                clip_metrics[cid][sid]["eap"] = m_eap
                clip_metrics[cid][sid]["oargen"] = m_oargen
                clip_metrics[cid][sid]["ovr"] = m_ovr

    # Now compute paired differences: E6 vs Base (E5), and E6 vs E1..E4
    for baseline_id in ["E5", "E1", "E2", "E3", "E4"]:
        for metric in metrics_list:
            if baseline_id in ("E1", "E2", "E3", "E4") and metric in ("eap", "oargen", "ovr"):
                continue

            # Group paired differences by source_video_id
            src_diffs = defaultdict(list)
            valid_pairs = 0
            for cid in clip_metrics:
                src = clip_source[cid]
                v_e6 = clip_metrics[cid]["E6"].get(metric)
                v_base = clip_metrics[cid][baseline_id].get(metric)

                if v_e6 is not None and v_base is not None and not math.isnan(v_e6) and not math.isnan(v_base):
                    diff = v_e6 - v_base
                    src_diffs[src].append(diff)
                    valid_pairs += 1

            if valid_pairs > 0:
                mean_d, low_d, high_d, p_val = paired_cluster_bootstrap(src_diffs, b_resamples=10000)
                is_sig = (p_val < 0.05)
                paired_results.append({
                    "Domain": dom_name,
                    "Comparison": f"E6 (LoRA) - {baseline_id}",
                    "Baseline_ID": baseline_id,
                    "Metric": metric.upper(),
                    "Valid_Pairs": valid_pairs,
                    "Mean_Diff": round(mean_d, 4),
                    "CI_95_Lower": round(low_d, 4),
                    "CI_95_Upper": round(high_d, 4),
                    "CI_95_String": f"[{low_d:+.4f}, {high_d:+.4f}]",
                    "P_Value": round(p_val, 4),
                    "Significant_p05": "YES" if is_sig else "NO"
                })

df_paired = pd.DataFrame(paired_results)
df_paired.to_csv(TABLES_DIR / "phase1_paired_bootstrap_significance.csv", index=False)
print("Paired bootstrap significance saved.")


# =============================================================================
# Action 2: Evaluator Fairness Audit (Before vs After across ALL Baselines)
# =============================================================================
print("Generating Evaluator Fairness Audit...")

# Read original summary metrics
with open(PHASE1_DIR / "hanoi" / "metrics_summary.json") as f:
    hn_orig_summary = json.load(f)
with open(PHASE1_DIR / "hcmc" / "metrics_summary.json") as f:
    hc_orig_summary = json.load(f)

# Read recomputed summary metrics
df_recomputed = pd.read_csv(TABLES_DIR / "phase1_recomputed_tables.csv")
t7a_recomputed = df_recomputed[df_recomputed["Table"].str.contains("Table 7A")].set_index("System")
t7b_recomputed = df_recomputed[df_recomputed["Table"].str.contains("Table 7B")].set_index("System")

sys_labels = {
    "E1": "E1 Qwen2.5-VL-3B", "E2": "E2 Text-RAG", "E3": "E3 Crisp-KG RAG",
    "E4": "E4 Prompt-only LLM", "E5": "E5 DEC + Base Qwen3B", "E6": "E6 Proposed (Full LoRA)"
}

fairness_rows = []
for dom_name, orig_sum, recomp_tab in [("HANOI", hn_orig_summary, t7a_recomputed), ("HCMC", hc_orig_summary, t7b_recomputed)]:
    for sid in ["E1", "E2", "E3", "E4", "E5", "E6"]:
        sname = sys_labels[sid]
        row_recomp = recomp_tab.loc[sname]
        
        orig_fp = orig_sum[sid]["fp"]["macro_mean"]
        orig_vsp = orig_sum[sid]["vsp"]["macro_mean"]
        orig_vfar = orig_sum[sid]["vfar"]["macro_mean"]
        orig_cov = orig_sum[sid]["coverage"]["macro_mean"]

        # Parse recomputed mean from string "mean [low, high]"
        def parse_mean(val_str):
            if pd.isna(val_str) or val_str == "N/A":
                return None
            return float(val_str.split()[0])

        rec_fp = parse_mean(row_recomp["FP_Standard_Macro"])
        rec_csr = parse_mean(row_recomp["FP_Conservative"])
        rec_cov = parse_mean(row_recomp["Coverage"])
        rec_vsp = parse_mean(row_recomp["VSP"])
        rec_vfar = parse_mean(row_recomp["VFAR"])

        fairness_rows.append({
            "Domain": dom_name,
            "System_ID": sid,
            "System_Name": sname,
            "Original_Macro_FP": orig_fp,
            "Robust_Macro_FP": rec_fp,
            "Delta_FP": round(rec_fp - orig_fp, 4) if rec_fp is not None and orig_fp is not None else None,
            "Original_Coverage": orig_cov,
            "Robust_Coverage": rec_cov,
            "Delta_Coverage": round(rec_cov - orig_cov, 4) if rec_cov is not None and orig_cov is not None else None,
            "Original_VSP": orig_vsp,
            "Robust_VSP": rec_vsp,
            "Delta_VSP": round(rec_vsp - orig_vsp, 4) if rec_vsp is not None and orig_vsp is not None else None,
            "Original_VFAR": orig_vfar,
            "Robust_VFAR": rec_vfar,
            "Delta_VFAR": round(rec_vfar - orig_vfar, 4) if rec_vfar is not None and orig_vfar is not None else None,
            "Conservative_Support_Rate_CSR": rec_csr
        })

df_fairness = pd.DataFrame(fairness_rows)
df_fairness.to_csv(AUDIT_DIR / "phase1_evaluator_fairness_audit.csv", index=False)
print("Evaluator fairness audit saved.")
print("DONE!")
