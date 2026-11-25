#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_fkg_ablation.py
=============================================================================
Definitive Comparative Evaluation & Ablation Benchmark:
Compares 5 Certainty Formulations:
  - M0: FKG Baseline (Raw extractor confidence / mock string match)
  - M1: NLI + DST Base (Semantic entailment + Dempster-Shafer on base facts)
  - M2: NLI + DST + Weighted Horn (Full Proposed Pipeline)
  - M3: NLI + t-Conorm + Horn (Ablation: Noisy-OR t-conorm fusion)
  - M4: NLI Direct Verification (Ablation: direct NLI without Horn reasoning)

Evaluates on:
  1. All 16,347 Unique Ontology Facts (Overall, KB_Base, Horn_Inferred)
  2. 33,153 Test Set Clip Instances (Scene-grounded evaluation benchmark)

Computes:
  - AUROC, AUPRC
  - Precision, Recall, Coverage, FAR at tau in {0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0}
  - ECE (10-bin equal width), Brier Score, Spearman rank correlation rho.
=============================================================================
"""

import json
import logging
import math
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("EvaluateFKGAblation")

REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")
VERIFIED_PATH = REPO_ROOT / "experiments" / "outputs" / "fkg_facts_verified.json"
TEST_PAIRS_PATH = REPO_ROOT / "experiments" / "outputs" / "mu_f_vs_fref_test_pairs.json"
OUTPUT_REPORT_PATH = REPO_ROOT / "experiments" / "outputs" / "fkg_certainty_redesign_ablation_report.json"


def compute_metrics(y_true: List[int], scores: List[float]) -> Dict[str, Any]:
    n = len(y_true)
    n_pos = sum(y_true)
    n_neg = n - n_pos

    if n == 0 or n_pos == 0 or n_neg == 0:
        return {
            "n": n, "n_pos": n_pos, "n_neg": n_neg, "base_rate": (n_pos / n) if n > 0 else 0.0,
            "auroc": 0.0, "auprc": 0.0, "brier": 0.0, "ece": 0.0, "spearman_rho": 0.0
        }

    # 1. AUROC (Mann-Whitney U)
    pairs = sorted(zip(scores, y_true), key=lambda x: x[0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and pairs[j][0] == pairs[j + 1][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    sum_ranks_pos = sum(ranks[k] for k in range(n) if pairs[k][1] == 1)
    u_stat = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    auroc = u_stat / (n_pos * n_neg)

    # 2. AUPRC (Average Precision under PR)
    sorted_pairs = sorted(zip(scores, y_true), key=lambda x: -x[0])
    distinct_scores = sorted(list(set(scores)), reverse=True)
    idx = 0
    cum_tp = 0
    cum_fp = 0
    auprc = 0.0
    for s in distinct_scores:
        delta_tp = 0
        delta_fp = 0
        while idx < n and sorted_pairs[idx][0] == s:
            if sorted_pairs[idx][1] == 1:
                delta_tp += 1
            else:
                delta_fp += 1
            idx += 1
        if delta_tp > 0:
            rec_prev = cum_tp / n_pos
            cum_tp += delta_tp
            cum_fp += delta_fp
            rec_curr = cum_tp / n_pos
            prec_curr = cum_tp / (cum_tp + cum_fp)
            auprc += (rec_curr - rec_prev) * prec_curr
        else:
            cum_fp += delta_fp

    # 3. Brier Score
    brier = sum((s - y) ** 2 for s, y in zip(scores, y_true)) / n

    # 4. ECE (10-bin equal width)
    ece = 0.0
    n_bins = 10
    bin_width = 1.0 / n_bins
    for b in range(n_bins):
        low = b * bin_width
        high = (b + 1) * bin_width
        if b == n_bins - 1:
            in_bin = [(s, y) for s, y in zip(scores, y_true) if low <= s <= high]
        else:
            in_bin = [(s, y) for s, y in zip(scores, y_true) if low <= s < high]
        if in_bin:
            bin_acc = sum(y for _, y in in_bin) / len(in_bin)
            bin_conf = sum(s for s, _ in in_bin) / len(in_bin)
            ece += (len(in_bin) / n) * abs(bin_acc - bin_conf)

    # 5. Spearman rank correlation rho
    rank_scores = ranks
    # Ranks for y_true
    y_pairs = sorted(enumerate(y_true), key=lambda x: x[1])
    y_ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and y_pairs[j][1] == y_pairs[j + 1][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            y_ranks[y_pairs[k][0]] = avg_rank
        i = j + 1

    # Map rank_scores back to original order
    orig_order_scores = [0.0] * n
    for k in range(n):
        orig_order_scores[k] = ranks[k]

    mean_rs = (n + 1) / 2.0
    mean_ry = (n + 1) / 2.0
    cov = sum((rs - mean_rs) * (ry - mean_ry) for rs, ry in zip(orig_order_scores, y_ranks))
    std_rs = math.sqrt(sum((rs - mean_rs) ** 2 for rs in orig_order_scores))
    std_ry = math.sqrt(sum((ry - mean_ry) ** 2 for ry in y_ranks))
    spearman_rho = (cov / (std_rs * std_ry)) if (std_rs * std_ry) > 0 else 0.0

    # 6. Threshold sweep
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    sweep = {}
    for tau in thresholds:
        retained = [(s, y) for s, y in zip(scores, y_true) if s >= tau]
        n_ret = len(retained)
        if n_ret > 0:
            tp = sum(y for _, y in retained)
            fp = n_ret - tp
            prec = round(tp / n_ret * 100.0, 2)
            rec = round(tp / n_pos * 100.0, 2)
            far = round(fp / n_neg * 100.0, 2)
            acc = round((tp + (n_neg - fp)) / n * 100.0, 2)
        else:
            prec, rec, far, acc = 0.0, 0.0, 0.0, round(n_neg / n * 100.0, 2)
        sweep[f"tau_{tau}"] = {
            "retained": n_ret,
            "retained_pct": round(n_ret / n * 100.0, 2),
            "precision": prec,
            "recall": rec,
            "far": far,
            "accuracy": acc
        }

    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "base_rate": round(n_pos / n * 100.0, 2),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "spearman_rho": round(spearman_rho, 4),
        "sweep": sweep
    }


def run_comprehensive_ablation():
    LOGGER.info(f"Loading verified facts from {VERIFIED_PATH}...")
    with open(VERIFIED_PATH, "r", encoding="utf-8") as f:
        verified_facts = json.load(f)

    LOGGER.info(f"Loading test pairs from {TEST_PAIRS_PATH}...")
    with open(TEST_PAIRS_PATH, "r", encoding="utf-8") as f:
        test_pairs = json.load(f)

    models = [
        ("M0_baseline", "M0: Baseline FKG mu_F (Mock/GLiNER)"),
        ("M1_nli_dst_base", "M1: NLI + DST (Base Facts)"),
        ("M2_nli_dst_horn", "M2: NLI + DST + Weighted Horn (Proposed)"),
        ("M3_nli_tconorm_horn", "M3: NLI + t-Conorm + Horn (Ablation)"),
        ("M4_nli_only", "M4: NLI Direct Verification (Ablation)"),
    ]

    report = {
        "evaluation_timestamp": str(Path(VERIFIED_PATH).stat().st_mtime),
        "total_unique_facts": len(verified_facts),
        "total_test_instances": len(test_pairs),
        "unique_facts_benchmarks": {},
        "test_instances_benchmarks": {},
        "subgroup_kb_base_benchmarks": {},
        "subgroup_horn_inferred_benchmarks": {},
    }

    # 1. Unique facts evaluation
    LOGGER.info("Evaluating on 16,347 unique ontology facts...")
    y_unique = [f["y_true"] for f in verified_facts.values()]

    for m_key, m_name in models:
        scores_m = [f["mu_F_models"][m_key] for f in verified_facts.values()]
        report["unique_facts_benchmarks"][m_key] = {
            "name": m_name,
            "metrics": compute_metrics(y_unique, scores_m)
        }

    # 2. Subgroup: KB_Base vs Horn_Inferred on unique facts
    LOGGER.info("Evaluating subgroups (KB_Base vs Horn_Inferred) on unique facts...")
    kb_base_facts = [f for f in verified_facts.values() if f.get("provenance_type") == "KB_Base"]
    horn_inf_facts = [f for f in verified_facts.values() if f.get("provenance_type") != "KB_Base"]

    y_kb_base = [f["y_true"] for f in kb_base_facts]
    y_horn_inf = [f["y_true"] for f in horn_inf_facts]

    for m_key, m_name in models:
        scores_kb = [f["mu_F_models"][m_key] for f in kb_base_facts]
        report["subgroup_kb_base_benchmarks"][m_key] = {
            "name": m_name,
            "metrics": compute_metrics(y_kb_base, scores_kb)
        }
        scores_horn = [f["mu_F_models"][m_key] for f in horn_inf_facts]
        report["subgroup_horn_inferred_benchmarks"][m_key] = {
            "name": m_name,
            "metrics": compute_metrics(y_horn_inf, scores_horn)
        }

    # 3. Test Set Clips Evaluation (33,153 scene instances)
    LOGGER.info("Evaluating on 33,153 test scene instances...")
    y_test = [p["y_true"] for p in test_pairs]

    for m_key, m_name in models:
        scores_test = []
        for p in test_pairs:
            fid = p["fact_id"]
            if fid in verified_facts:
                scores_test.append(verified_facts[fid]["mu_F_models"][m_key])
            else:
                scores_test.append(0.0)

        report["test_instances_benchmarks"][m_key] = {
            "name": m_name,
            "metrics": compute_metrics(y_test, scores_test)
        }

    # Save output report
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Report saved to {OUTPUT_REPORT_PATH}")

    # Print summary table
    print("\n" + "=" * 80)
    print("FKG CERTAINTY REDESIGN: DEFINITIVE BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Model':<35} | {'AUROC':<7} | {'AUPRC':<7} | {'P@0.8':<7} | {'P@0.9':<7} | {'ECE':<7} | {'Brier':<7}")
    print("-" * 80)
    print("--- 1. Unique Ontology Facts (N = 16,347) ---")
    for m_key, _ in models:
        m = report["unique_facts_benchmarks"][m_key]["metrics"]
        p08 = m["sweep"]["tau_0.8"]["precision"]
        p09 = m["sweep"]["tau_0.9"]["precision"]
        print(f"{m_key:<35} | {m['auroc']:<7.4f} | {m['auprc']:<7.4f} | {p08:<7.2f} | {p09:<7.2f} | {m['ece']:<7.4f} | {m['brier']:<7.4f}")

    print("\n--- 2. Subgroup: KB_Base Unique Facts (N = 7,088) ---")
    for m_key, _ in models:
        m = report["subgroup_kb_base_benchmarks"][m_key]["metrics"]
        p08 = m["sweep"]["tau_0.8"]["precision"]
        p09 = m["sweep"]["tau_0.9"]["precision"]
        print(f"{m_key:<35} | {m['auroc']:<7.4f} | {m['auprc']:<7.4f} | {p08:<7.2f} | {p09:<7.2f} | {m['ece']:<7.4f} | {m['brier']:<7.4f}")

    print("\n--- 3. Subgroup: Horn_Inferred Unique Facts (N = 9,259) ---")
    for m_key, _ in models:
        m = report["subgroup_horn_inferred_benchmarks"][m_key]["metrics"]
        p08 = m["sweep"]["tau_0.8"]["precision"]
        p09 = m["sweep"]["tau_0.9"]["precision"]
        print(f"{m_key:<35} | {m['auroc']:<7.4f} | {m['auprc']:<7.4f} | {p08:<7.2f} | {p09:<7.2f} | {m['ece']:<7.4f} | {m['brier']:<7.4f}")

    print("\n--- 4. Test Set Scene Instances (N = 33,153) ---")
    for m_key, _ in models:
        m = report["test_instances_benchmarks"][m_key]["metrics"]
        p08 = m["sweep"]["tau_0.8"]["precision"]
        p09 = m["sweep"]["tau_0.9"]["precision"]
        print(f"{m_key:<35} | {m['auroc']:<7.4f} | {m['auprc']:<7.4f} | {p08:<7.2f} | {p09:<7.2f} | {m['ece']:<7.4f} | {m['brier']:<7.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_comprehensive_ablation()
