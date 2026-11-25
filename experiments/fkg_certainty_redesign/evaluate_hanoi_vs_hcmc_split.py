#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_hanoi_vs_hcmc_split.py
=============================================================================
Geographic Held-Out Evaluation Protocol:
Separating Hanoi (In-Domain Dev/Test, 30 Landmarks) vs HCMC (Dedicated Held-Out Evaluation Set, 5 Landmarks).

Strict Rule:
  - Zero HCMC data used for tuning, calibration, or schema development.
  - HCMC serves strictly as a dedicated, untouched post-finalization evaluation set.

Evaluates M0 (Baseline), M2_V1 (Initial V1), and M2_V2 (Proposed V2) across:
  1. KB_Base Facts (Hanoi N=5,543 vs HCMC Held-Out N=1,545 vs All N=7,088)
  2. Horn_Inferred Facts (Hanoi N=7,011 vs HCMC Held-Out N=2,248 vs All N=9,259)
  3. All Unique Facts (Hanoi N=12,554 vs HCMC Held-Out N=3,793 vs All N=16,347)
  4. Video Test Clip Instances (Hanoi N=9,099 vs HCMC Held-Out N=24,054 vs All N=33,153)
  5. Detailed Precision & Threshold Sweep on Dedicated HCMC Held-Out Test Clips.
=============================================================================
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")
sys.path.insert(0, str(REPO_ROOT))
from experiments.fkg_certainty_redesign.evaluate_fkg_ablation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("EvalHanoiVsHCMC")

V1_VERIFIED_PATH = REPO_ROOT / "experiments" / "outputs" / "fkg_facts_verified.json"
V2_VERIFIED_PATH = REPO_ROOT / "experiments" / "outputs" / "fkg_facts_verified_v2.json"
TEST_PAIRS_PATH = REPO_ROOT / "experiments" / "outputs" / "mu_f_vs_fref_test_pairs.json"
OUTPUT_REPORT_PATH = REPO_ROOT / "experiments" / "outputs" / "hanoi_vs_hcmc_held_out_evaluation_report.json"

HCMC_LANDMARKS = {
    "saigon_central_post_office",
    "independence_palace",
    "ben_thanh_market",
    "saigon_opera_house",
    "ba_thien_hau_temple"
}


def run_split_evaluation():
    LOGGER.info(f"Loading V1 facts from {V1_VERIFIED_PATH}...")
    with open(V1_VERIFIED_PATH, "r", encoding="utf-8") as f:
        v1_facts = json.load(f)

    LOGGER.info(f"Loading V2 facts from {V2_VERIFIED_PATH}...")
    with open(V2_VERIFIED_PATH, "r", encoding="utf-8") as f:
        v2_facts = json.load(f)

    LOGGER.info(f"Loading test clip instances from {TEST_PAIRS_PATH}...")
    with open(TEST_PAIRS_PATH, "r", encoding="utf-8") as f:
        test_pairs = json.load(f)

    # 1. Fact Partitions
    all_fids = list(v2_facts.keys())
    hanoi_fids = [fid for fid in all_fids if v2_facts[fid].get("landmark_id") not in HCMC_LANDMARKS]
    hcmc_fids = [fid for fid in all_fids if v2_facts[fid].get("landmark_id") in HCMC_LANDMARKS]

    LOGGER.info(f"Unique facts partition: Hanoi={len(hanoi_fids):,}, HCMC (Held-Out)={len(hcmc_fids):,}, Total={len(all_fids):,}")

    def eval_subset(fids: List[str], score_type: str = "joint"):
        subset_v2 = [v2_facts[fid] for fid in fids]
        subset_v1 = [v1_facts.get(fid, {}) for fid in fids]
        y_arr = [f["y_true"] for f in subset_v2]

        # M0
        s_m0 = [f.get("old_mu_F", 0.0) for f in subset_v2]

        # M2_V1
        if score_type == "base":
            s_v1 = [f.get("dst", {}).get("w_base", 0.0) for f in subset_v1]
            s_v2 = [f.get("w_base", 0.0) for f in subset_v2]
        elif score_type == "horn":
            s_v1 = [f.get("mu_F_models", {}).get("M2_nli_dst_horn", 0.0) for f in subset_v1]
            s_v2 = [f.get("mu_horn", 0.0) for f in subset_v2]
        else:  # joint mu_F
            s_v1 = [f.get("mu_F_models", {}).get("M2_nli_dst_horn", 0.0) for f in subset_v1]
            s_v2 = [f.get("mu_F_v2", 0.0) for f in subset_v2]

        m_m0 = compute_metrics(y_arr, s_m0)
        m_v1 = compute_metrics(y_arr, s_v1)
        m_v2 = compute_metrics(y_arr, s_v2)

        return {
            "n": len(fids),
            "base_rate": round(sum(y_arr) / len(y_arr) * 100.0, 2) if y_arr else 0.0,
            "M0_baseline": m_m0,
            "M2_V1": m_v1,
            "M2_V2": m_v2
        }

    # Helper for table printing
    def print_benchmark_table(title: str, res_dict: Dict[str, Any]):
        print("\n" + "=" * 95)
        print(title)
        print("=" * 95)
        print(f"{'Split / Domain':<28} | {'Model':<12} | {'AUROC':<7} | {'AUPRC':<7} | {'P@0.8':<7} | {'P@0.9':<7} | {'ECE':<7}")
        print("-" * 95)
        for split_key, split_name in [
            ("hanoi", "Hanoi (In-Domain Dev/Test)"),
            ("hcmc", "HCMC (Held-Out Evaluation Set)"),
            ("combined", "Combined (All Landmarks)"),
        ]:
            data = res_dict[split_key]
            n_val = data["n"]
            br_val = data["base_rate"]
            for m_key, m_lbl in [("M0_baseline", "M0 Baseline"), ("M2_V1", "M2 (V1)"), ("M2_V2", "M2 (V2 Ours)")]:
                m = data[m_key]
                p08 = m["sweep"].get("tau_0.8", {}).get("precision", 0.0)
                p09 = m["sweep"].get("tau_0.9", {}).get("precision", 0.0)
                lbl_full = f"{split_name} [N={n_val:,}]" if m_key == "M0_baseline" else ""
                print(f"{lbl_full:<28} | {m_lbl:<12} | {m['auroc']:<7.4f} | {m['auprc']:<7.4f} | {p08:<7.2f} | {p09:<7.2f} | {m['ece']:<7.4f}")
            print("-" * 95)

    # A. KB_Base Facts
    kb_hanoi = [fid for fid in hanoi_fids if v2_facts[fid].get("provenance_type") == "KB_Base"]
    kb_hcmc = [fid for fid in hcmc_fids if v2_facts[fid].get("provenance_type") == "KB_Base"]
    kb_all = [fid for fid in all_fids if v2_facts[fid].get("provenance_type") == "KB_Base"]

    res_kb_base = {
        "hanoi": eval_subset(kb_hanoi, "base"),
        "hcmc": eval_subset(kb_hcmc, "base"),
        "combined": eval_subset(kb_all, "base")
    }
    print_benchmark_table("1. KB_BASE FACTS (EVIDENTIAL VERIFICATION w_base)", res_kb_base)

    # B. Horn_Inferred Facts
    horn_hanoi = [fid for fid in hanoi_fids if v2_facts[fid].get("provenance_type") != "KB_Base"]
    horn_hcmc = [fid for fid in hcmc_fids if v2_facts[fid].get("provenance_type") != "KB_Base"]
    horn_all = [fid for fid in all_fids if v2_facts[fid].get("provenance_type") != "KB_Base"]

    res_horn_inf = {
        "hanoi": eval_subset(horn_hanoi, "horn"),
        "hcmc": eval_subset(horn_hcmc, "horn"),
        "combined": eval_subset(horn_all, "horn")
    }
    print_benchmark_table("2. HORN_INFERRED FACTS (SYMBOLIC DERIVATION mu_horn)", res_horn_inf)

    # C. All Unique Facts
    res_all_unique = {
        "hanoi": eval_subset(hanoi_fids, "joint"),
        "hcmc": eval_subset(hcmc_fids, "joint"),
        "combined": eval_subset(all_fids, "joint")
    }
    print_benchmark_table("3. ALL UNIQUE ONTOLOGY FACTS (JOINT MEMBERSHIP mu_F)", res_all_unique)

    # D. Video Test Clip Instances
    LOGGER.info("Evaluating Video Test Clip Instances split by Hanoi vs HCMC...")
    hanoi_test_pairs = []
    hcmc_test_pairs = []

    for p in test_pairs:
        fid = p["fact_id"]
        vf = v2_facts.get(fid, {})
        lmid = vf.get("landmark_id", "")
        if lmid in HCMC_LANDMARKS:
            hcmc_test_pairs.append(p)
        else:
            hanoi_test_pairs.append(p)

    def eval_test_pairs(pairs: List[Dict[str, Any]]):
        y_arr = [p["y_true"] for p in pairs]
        s_m0 = [v2_facts.get(p["fact_id"], {}).get("old_mu_F", 0.0) for p in pairs]
        s_v1 = [v1_facts.get(p["fact_id"], {}).get("mu_F_models", {}).get("M2_nli_dst_horn", 0.0) for p in pairs]
        s_v2 = [v2_facts.get(p["fact_id"], {}).get("mu_F_v2", 0.0) for p in pairs]

        m_m0 = compute_metrics(y_arr, s_m0)
        m_v1 = compute_metrics(y_arr, s_v1)
        m_v2 = compute_metrics(y_arr, s_v2)

        return {
            "n": len(pairs),
            "base_rate": round(sum(y_arr) / len(y_arr) * 100.0, 2) if y_arr else 0.0,
            "M0_baseline": m_m0,
            "M2_V1": m_v1,
            "M2_V2": m_v2
        }

    res_video_test = {
        "hanoi": eval_test_pairs(hanoi_test_pairs),
        "hcmc": eval_test_pairs(hcmc_test_pairs),
        "combined": eval_test_pairs(test_pairs)
    }
    print_benchmark_table("4. VIDEO TEST CLIP INSTANCES (GROUND TRUTH SCENE VALIDATION)", res_video_test)

    # E. Detailed Sweep for Dedicated HCMC Held-Out Test Set
    print("\n" + "=" * 95)
    print("5. DEDICATED HCMC HELD-OUT TEST CLIPS: THRESHOLD SWEEP (N = 24,054)")
    print("=" * 95)
    print(f"{'Threshold tau':<15} | {'Retained Count':<16} | {'Retained %':<12} | {'Precision %':<12} | {'Recall %':<12} | {'FAR %':<10}")
    print("-" * 95)
    hcmc_sweep = res_video_test["hcmc"]["M2_V2"]["sweep"]
    for tau in [0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
        t_key = f"tau_{tau}"
        sw = hcmc_sweep.get(t_key, {})
        print(f"tau >= {tau:<8.2f} | {sw.get('retained', 0):<16,d} | {sw.get('retained_pct', 0.0):<12.2f} | {sw.get('precision', 0.0):<12.2f} | {sw.get('recall', 0.0):<12.2f} | {sw.get('far', 0.0):<10.2f}")
    print("=" * 95 + "\n")

    # Save full report
    full_report = {
        "protocol": "Geographic Held-Out (Zero HCMC data used for tuning or calibration)",
        "landmarks": {
            "hanoi_count": 30,
            "hcmc_count": 5,
            "hcmc_list": sorted(list(HCMC_LANDMARKS))
        },
        "kb_base_facts": res_kb_base,
        "horn_inferred_facts": res_horn_inf,
        "all_unique_facts": res_all_unique,
        "video_test_clips": res_video_test,
    }

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Saved full held-out report to {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    run_split_evaluation()
