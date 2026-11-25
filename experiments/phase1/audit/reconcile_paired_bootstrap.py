#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reconcile_paired_bootstrap.py
=============================================================================
Reconciles Paired Cluster Bootstrap Significance with Table 7A-7D.
Evaluates:
  1. Valid Pairs (conditional on non-None values in both models)
  2. All Clips (zero-filled across entire N=500 Hanoi / N=300 HCMC)
  3. Direct comparison with Table 7 marginal means.
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"
AUDIT_DIR = PHASE1_DIR / "audit"
TABLES_DIR = PHASE1_DIR / "tables"
CONFIG_DIR = PHASE1_DIR / "config"

with open(CONFIG_DIR / "hanoi_manifest.jsonl") as f:
    hn_manifest = [json.loads(l) for l in f]
with open(CONFIG_DIR / "hcmc_manifest.jsonl") as f:
    hc_manifest = [json.loads(l) for l in f]

with open(PHASE1_DIR / "hanoi" / "evidence.jsonl") as f:
    hn_ev = {x["clip_id"]: x for x in [json.loads(l) for l in f]}
with open(PHASE1_DIR / "hcmc" / "evidence.jsonl") as f:
    hc_ev = {x["clip_id"]: x for x in [json.loads(l) for l in f]}

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
            cidx = it.get("clip_idx")
            cid = f"hcmc_x5_{cidx:04d}" if cidx is not None else it.get("sample_id")
            hcmc_sg_map[cid] = it.get("observed_scene_graph", "")

caps_map = {}
for c in hn_caps:
    caps_map[("hanoi", c["clip_id"], c["system_id"])] = c["caption"]
for c in hc_caps:
    caps_map[("hcmc", c["clip_id"], c["system_id"])] = c["caption"]

HEDGING_PATTERNS = [
    r"\blikely\b", r"\bappears to\b", r"\bpossibly\b", r"\bmight be\b", r"\bcould be\b",
    r"\bseems to\b", r"\bsuggests\b", r"\bmay have\b", r"\bif this is\b", r"\bbelieved to\b",
    r"\bprobably\b", r"\bpotentially\b", r"\bperhaps\b", r"\bmay include\b", r"\btypical of\b",
    r"\bdepending on\b", r"\bindicates\b", r"\bresembles\b", r"\bevokes\b", r"\breportedly\b",
    r"\bis said to\b", r"\bis traditionally associated with\b", r"\bthought to be\b",
    r"\baccording to\b", r"\btradition\b", r"\btraditions\b", r"\bfolklore\b", r"\blegend\b",
    r"\boral history\b", r"\blocal lore\b", r"\bcustoms\b", r"\brecords indicate\b",
    r"\baccounts suggest\b", r"\bit is said\b"
]
HEDGING_REGEX = re.compile("|".join(HEDGING_PATTERNS), re.IGNORECASE)

GENERIC_TAILS = {
    "and", "to", "a", "it", "the", "or", "walls", "street", "building",
    "trees", "roof", "gate", "clay", "shops", "floor", "hall", "waters", "water"
}

def split_sentences(text: str) -> List[str]:
    cleaned = text.strip().replace("\n", " ")
    raw_sents = re.split(r"(?<=[.!?])\s+", cleaned)
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

def fast_paired_bootstrap(diffs_by_src: Dict[str, List[float]], b_resamples: int = 5000, seed: int = 42):
    rng = np.random.RandomState(seed)
    src_keys = list(diffs_by_src.keys())
    n_src = len(src_keys)
    if n_src == 0:
        return 0.0, 0.0, 0.0, 1.0
    boot_diffs = []
    for _ in range(b_resamples):
        sampled = rng.choice(src_keys, size=n_src, replace=True)
        vals = []
        for s in sampled:
            vals.extend(diffs_by_src[s])
        if vals:
            boot_diffs.append(float(np.mean(vals)))
    if not boot_diffs:
        return 0.0, 0.0, 0.0, 1.0
    boot_diffs.sort()
    mean_d = float(np.mean(boot_diffs))
    low = float(np.percentile(boot_diffs, 2.5))
    high = float(np.percentile(boot_diffs, 97.5))
    p_le = sum(1 for x in boot_diffs if x <= 0) / len(boot_diffs)
    p_ge = sum(1 for x in boot_diffs if x >= 0) / len(boot_diffs)
    p_val = min(1.0, max(0.0001, 2.0 * min(p_le, p_ge)))
    return mean_d, low, high, p_val

systems = ["E1", "E2", "E3", "E4", "E5", "E6"]
metrics = ["csr", "bkr5", "kr", "vsp", "vfar"]

results_rows = []

for dom_name, dom_key, manifest, ev_dict, sg_map in [
    ("HANOI", "hanoi", hn_manifest, hn_ev, hn_sg_map),
    ("HCMC", "hcmc", hc_manifest, hc_ev, hcmc_sg_map)
]:
    total_clips = len(manifest)
    clip_dict = defaultdict(lambda: defaultdict(dict))
    clip_src = {}

    for c in manifest:
        cid = c["clip_id"]
        src = c.get("source_video_id", "src_unknown")
        clip_src[cid] = src
        ep = ev_dict[cid]
        lm = ep["landmark_name"]
        sg_str = sg_map.get(cid, "")
        sg_concepts = [re.sub(r"\(\d+\.\d+\)", "", s).strip().lower() for s in sg_str.split("\n") if len(s.strip()) > 3]

        gdata = fgold_all[cid]
        gold_true = [f for f in gdata["gold_facts"] if f.get("review", {}).get("result") == "đúng"]
        gold_false = [f for f in gdata["gold_facts"] if f.get("review", {}).get("result") == "sai"]
        n_gold = len(gold_true)

        for sid in systems:
            cap = caps_map[(dom_key, cid, sid)]
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
            n_u = max(0, len(sentences) - n_s - n_c)

            vis_tot = len(sentences)
            vis_s = 0
            for s in sentences:
                s_l = s.lower()
                matched_vc = False
                for vc in sg_concepts:
                    if len(vc) > 3 and vc in s_l:
                        matched_vc = True
                        break
                    words = [w for w in vc.split() if len(w) > 3]
                    if len(words) >= 2 and sum(1 for w in words if w in s_l) >= 2:
                        matched_vc = True
                        break
                if matched_vc:
                    vis_s += 1

            clip_dict[cid][sid]["csr"] = (n_s / (n_s + n_c + n_u)) if (n_s + n_c + n_u) > 0 else None
            denom_bkr = min(5, n_gold)
            clip_dict[cid][sid]["bkr5"] = (min(5, n_s) / denom_bkr) if denom_bkr > 0 else None
            clip_dict[cid][sid]["kr"] = (n_s / n_gold) if n_gold > 0 else None
            clip_dict[cid][sid]["vsp"] = (vis_s / vis_tot) if vis_tot > 0 else None
            clip_dict[cid][sid]["vfar"] = (a_c / (a_s + a_c)) if (a_s + a_c) > 0 else None

    # Compute paired comparisons for E6 vs all others
    for base_id in ["E5", "E1", "E2", "E3", "E4"]:
        for m in metrics:
            valid_diffs = defaultdict(list)
            valid_count = 0
            all_diffs = defaultdict(list)

            for cid in clip_dict:
                src = clip_src[cid]
                v6 = clip_dict[cid]["E6"][m]
                vb = clip_dict[cid][base_id][m]

                if v6 is not None and vb is not None:
                    valid_diffs[src].append(v6 - vb)
                    valid_count += 1
                
                v6_z = v6 if v6 is not None else 0.0
                vb_z = vb if vb is not None else 0.0
                all_diffs[src].append(v6_z - vb_z)

            m_val, low_val, high_val, p_val = fast_paired_bootstrap(valid_diffs)
            m_all, low_all, high_all, p_all = fast_paired_bootstrap(all_diffs)

            results_rows.append({
                "Domain": dom_name,
                "Comparison": f"E6 - {base_id}",
                "Metric": m.upper(),
                "Valid_Pairs_Count": valid_count,
                "Total_Clips": total_clips,
                "Valid_Pairs_Diff": round(m_val, 4),
                "Valid_Pairs_CI95": f"[{low_val:+.4f}, {high_val:+.4f}]",
                "Valid_Pairs_P_Val": round(p_val, 4),
                "All_Clips_ZeroFilled_Diff": round(m_all, 4),
                "All_Clips_CI95": f"[{low_all:+.4f}, {high_all:+.4f}]",
                "All_Clips_P_Val": round(p_all, 4),
            })

df_res = pd.DataFrame(results_rows)
df_res.to_csv(AUDIT_DIR / "phase1_reconciled_paired_bootstrap.csv", index=False)
print("Saved reconciled paired bootstrap to experiments/phase1/audit/phase1_reconciled_paired_bootstrap.csv")

print("\n--- HANOI E6 - E5 ---")
print(df_res[(df_res["Domain"] == "HANOI") & (df_res["Comparison"] == "E6 - E5")][["Metric", "Valid_Pairs_Count", "Valid_Pairs_Diff", "Valid_Pairs_CI95", "Valid_Pairs_P_Val", "All_Clips_ZeroFilled_Diff", "All_Clips_CI95"]])
print("\n--- HCMC E6 - E5 ---")
print(df_res[(df_res["Domain"] == "HCMC") & (df_res["Comparison"] == "E6 - E5")][["Metric", "Valid_Pairs_Count", "Valid_Pairs_Diff", "Valid_Pairs_CI95", "Valid_Pairs_P_Val", "All_Clips_ZeroFilled_Diff", "All_Clips_CI95"]])
