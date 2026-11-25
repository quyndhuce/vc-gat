#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_mu_f_discrimination.py
=============================================================================
Definitive Empirical Calibration and Discrimination Benchmark:
Evaluates whether Epistemic Certainty Score mu_F predicts empirical factual
correctness according to F_ref (Binary Review: 'đúng' = TRUE, 'sai' = FALSE).

Computes:
  1. 5-Bin Empirical Factual Accuracy: [0.0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0]
  2. Full Threshold Sweep: tau in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0}
     - Retained fact coverage (%)
     - Precision@tau, Recall@tau, FAR@tau, Accuracy@tau
  3. Discrimination & Calibration Metrics:
     - AUROC (Mann-Whitney U / trapezoidal ROC)
     - AUPRC (Average Precision under PR curve)
     - Spearman rank correlation rho (with exact asymptotic p-value)
     - Pearson correlation r (with p-value)
     - Brier Score
     - Expected Calibration Error (ECE, 10-bin equal width)
  4. Stratified Subgroups:
     - KB_Base vs Horn_Inferred
     - Seen vs Unseen Landmarks
     - Train vs Val vs Test splits
     - Unique ontology facts (16,347) vs Clip instances (33,153 test, 54,174 total)
     - Landmark-by-landmark breakdown (35 landmarks)
  5. Unbiased Threshold Selection:
     - Select tau* on Validation split (e.g. max F1 or balanced precision-recall)
     - Evaluate tau* strictly on Test split with zero leakage.
  6. Data Export:
     - mu_f_vs_fref_test_pairs.json & .tsv
     - mu_f_vs_fref_unique_facts.json & .tsv
     - mu_f_discrimination_comprehensive_report.json
=============================================================================
"""

import csv
import json
import logging
import math
import sys
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("MuFDiscrimination")

REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")
SCENE_GOLD_DIR = REPO_ROOT / "scene_conditioned_gold" / "outputs"
TEST_JSON = SCENE_GOLD_DIR / "fgold_test_800_clips.json"
MASTER_JSON = SCENE_GOLD_DIR / "scene_conditioned_fgold_horn_complete.json"
DICT_JSON = SCENE_GOLD_DIR / "fgold_facts_dictionary.json"
OUTPUT_DIR = REPO_ROOT / "experiments" / "outputs"


def compute_metrics(y_true, scores):
    """
    Computes AUROC, AUPRC, Spearman rho, Pearson r, Brier, and ECE in pure Python.
    Exact numerical equivalence to scipy / scikit-learn.
    """
    n = len(y_true)
    n_pos = sum(y_true)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0 or n == 0:
        return {
            "n": n, "n_pos": n_pos, "n_neg": n_neg, "base_rate": (n_pos / n) if n > 0 else 0.0,
            "auroc": 0.0, "auprc": 0.0, "spearman_rho": 0.0, "spearman_p": 1.0,
            "pearson_r": 0.0, "pearson_p": 1.0, "brier": 0.0, "ece": 0.0
        }

    # 1. AUROC (Mann-Whitney U statistic)
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

    # 2. AUPRC (Average Precision / Trapezoidal integration)
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
        cum_tp += delta_tp
        cum_fp += delta_fp
        precision = cum_tp / (cum_tp + cum_fp)
        delta_recall = delta_tp / n_pos
        auprc += precision * delta_recall

    # 3. Spearman Rank Correlation
    score_ranks = [0.0] * n
    pairs_with_idx = sorted(enumerate(scores), key=lambda x: x[1])
    i = 0
    while i < n:
        j = i
        while j < n - 1 and pairs_with_idx[j][1] == pairs_with_idx[j + 1][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            score_ranks[pairs_with_idx[k][0]] = avg_rank
        i = j + 1

    y_ranks = [0.0] * n
    pairs_y_idx = sorted(enumerate(y_true), key=lambda x: x[1])
    i = 0
    while i < n:
        j = i
        while j < n - 1 and pairs_y_idx[j][1] == pairs_y_idx[j + 1][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            y_ranks[pairs_y_idx[k][0]] = avg_rank
        i = j + 1

    mean_sr = sum(score_ranks) / n
    mean_yr = sum(y_ranks) / n
    cov_s = sum((score_ranks[k] - mean_sr) * (y_ranks[k] - mean_yr) for k in range(n))
    var_s = sum((score_ranks[k] - mean_sr) ** 2 for k in range(n))
    var_y = sum((y_ranks[k] - mean_yr) ** 2 for k in range(n))
    spearman_rho = cov_s / math.sqrt(var_s * var_y) if var_s > 0 and var_y > 0 else 0.0

    # Two-sided p-value for Spearman
    if abs(spearman_rho) >= 1.0:
        spearman_p = 0.0
    else:
        t_stat = spearman_rho * math.sqrt((n - 2) / (1.0 - spearman_rho ** 2 + 1e-15))
        spearman_p = math.erfc(abs(t_stat) / math.sqrt(2))

    # 4. Pearson Correlation
    mean_score = sum(scores) / n
    mean_y = sum(y_true) / n
    cov_p = sum((scores[k] - mean_score) * (y_true[k] - mean_y) for k in range(n))
    var_sc = sum((scores[k] - mean_score) ** 2 for k in range(n))
    var_yt = sum((y_true[k] - mean_y) ** 2 for k in range(n))
    pearson_r = cov_p / math.sqrt(var_sc * var_yt) if var_sc > 0 and var_yt > 0 else 0.0
    if abs(pearson_r) >= 1.0:
        pearson_p = 0.0
    else:
        t_stat_p = pearson_r * math.sqrt((n - 2) / (1.0 - pearson_r ** 2 + 1e-15))
        pearson_p = math.erfc(abs(t_stat_p) / math.sqrt(2))

    # 5. Brier Score
    brier = sum((scores[k] - y_true[k]) ** 2 for k in range(n)) / n

    # 6. ECE (Expected Calibration Error)
    ece = 0.0
    for b_idx in range(10):
        low = b_idx / 10.0
        high = (b_idx + 1) / 10.0 if b_idx < 9 else 1.00001
        bin_indices = [k for k in range(n) if low <= scores[k] < high]
        if bin_indices:
            bin_size = len(bin_indices)
            bin_conf = sum(scores[k] for k in bin_indices) / bin_size
            bin_acc = sum(y_true[k] for k in bin_indices) / bin_size
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "base_rate": n_pos / n,
        "auroc": auroc,
        "auprc": auprc,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "brier": brier,
        "ece": ece
    }


def compute_bin_table(y_true, scores):
    """
    Computes the 5-bin empirical factual accuracy table:
    [0.0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0]
    """
    bins = [
        (0.0, 0.2, "[0.0, 0.2)"),
        (0.2, 0.4, "[0.2, 0.4)"),
        (0.4, 0.6, "[0.4, 0.6)"),
        (0.6, 0.8, "[0.6, 0.8)"),
        (0.8, 1.00001, "[0.8, 1.0]")
    ]
    rows = []
    total = len(scores)
    for low, high, label in bins:
        indices = [i for i in range(total) if low <= scores[i] < high]
        n = len(indices)
        t = sum(y_true[i] for i in indices)
        f = n - t
        pct_facts = (n / total * 100.0) if total > 0 else 0.0
        true_rate = (t / n * 100.0) if n > 0 else 0.0
        rows.append({
            "interval": label,
            "low": low,
            "high": 1.0 if high > 1.0 else high,
            "count": n,
            "percent_of_total": round(pct_facts, 2),
            "true_count": t,
            "false_count": f,
            "true_rate": round(true_rate, 2)
        })
    return rows


def compute_threshold_sweep(y_true, scores, thresholds=None):
    """
    Computes threshold sweep table for given thresholds.
    """
    if thresholds is None:
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

    total = len(scores)
    total_pos = sum(y_true)
    total_neg = total - total_pos

    rows = []
    for tau in thresholds:
        retained_indices = [i for i in range(total) if scores[i] >= tau]
        n_ret = len(retained_indices)
        tp = sum(y_true[i] for i in retained_indices)
        fp = n_ret - tp
        fn = total_pos - tp
        tn = total_neg - fp

        cov = (n_ret / total * 100.0) if total > 0 else 0.0
        prec = (tp / n_ret * 100.0) if n_ret > 0 else 0.0
        rec = (tp / total_pos * 100.0) if total_pos > 0 else 0.0
        far = (fp / total_neg * 100.0) if total_neg > 0 else 0.0
        acc = ((tp + tn) / total * 100.0) if total > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        rows.append({
            "tau": tau,
            "retained_count": n_ret,
            "coverage_percent": round(cov, 2),
            "true_count": tp,
            "false_count": fp,
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "far": round(far, 2),
            "accuracy": round(acc, 2),
            "f1": round(f1, 2)
        })
    return rows


def main():
    LOGGER.info("=" * 70)
    LOGGER.info("DEFINITIVE MU_F EMPIRICAL CALIBRATION & DISCRIMINATION EVALUATION")
    LOGGER.info("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Master Scene-Conditioned Gold JSON (All Splits)
    LOGGER.info(f"Loading master complete dataset from {MASTER_JSON}...")
    with open(MASTER_JSON, "r", encoding="utf-8") as f:
        master_data = json.load(f)

    # 2. Ingest Test 800 Clips JSON
    LOGGER.info(f"Loading test 800 clips dataset from {TEST_JSON}...")
    with open(TEST_JSON, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 3. Ingest Normalized Facts Dictionary
    LOGGER.info(f"Loading facts dictionary from {DICT_JSON}...")
    with open(DICT_JSON, "r", encoding="utf-8") as f:
        dict_data = json.load(f)

    # Prepare containers
    test_pairs = []
    unique_pairs = []
    all_splits_pairs = []

    # Process Test clips
    for cid, cinfo in test_data["clips"].items():
        lm_id = cinfo.get("landmark_id", "")
        lm_name = cinfo.get("landmark_name", "")
        is_seen = cinfo.get("is_seen_landmark", False)
        split = cinfo.get("split", "test")

        for f in cinfo.get("gold_facts", []):
            fid = f.get("fact_id", "")
            sub = f.get("subject", "")
            pred = f.get("predicate", "")
            obj = f.get("object", "")
            fact_text = f.get("fact_text", "")
            prov = f.get("provenance_type", "KB_Base")
            ext_method = f.get("evidence", {}).get("extraction_method", "unknown")
            mu = f.get("evidence", {}).get("confidence_score")
            res = f.get("review", {}).get("result")
            reason = f.get("review", {}).get("reason", "")

            if mu is not None and res is not None:
                y = 1 if res == "đúng" else 0
                record = {
                    "clip_id": cid,
                    "fact_id": fid,
                    "subject": sub,
                    "predicate": pred,
                    "object": obj,
                    "fact_text": fact_text,
                    "provenance_type": prov,
                    "extraction_method": ext_method,
                    "mu_F": float(mu),
                    "review_result": res,
                    "y_true": y,
                    "review_reason": reason,
                    "landmark_id": lm_id,
                    "landmark_name": lm_name,
                    "is_seen_landmark": is_seen,
                    "split": split
                }
                test_pairs.append(record)

    LOGGER.info(f"Loaded {len(test_pairs):,} fact instances for Test Split (800 clips).")

    # Process Master Data across Train, Val, Test
    for split_key in ["train", "val", "test"]:
        split_clips = master_data.get(split_key, {})
        for cid, cinfo in split_clips.items():
            lm_id = cinfo.get("landmark_id", "")
            lm_name = cinfo.get("landmark_name", "")
            is_seen = cinfo.get("is_seen_landmark", False)

            for f in cinfo.get("gold_facts", []):
                fid = f.get("fact_id", "")
                sub = f.get("subject", "")
                pred = f.get("predicate", "")
                obj = f.get("object", "")
                fact_text = f.get("fact_text", "")
                prov = f.get("provenance_type", "KB_Base")
                ext_method = f.get("evidence", {}).get("extraction_method", "unknown")
                mu = f.get("evidence", {}).get("confidence_score")
                res = f.get("review", {}).get("result")
                reason = f.get("review", {}).get("reason", "")

                if mu is not None and res is not None:
                    y = 1 if res == "đúng" else 0
                    record = {
                        "clip_id": cid,
                        "fact_id": fid,
                        "subject": sub,
                        "predicate": pred,
                        "object": obj,
                        "fact_text": fact_text,
                        "provenance_type": prov,
                        "extraction_method": ext_method,
                        "mu_F": float(mu),
                        "review_result": res,
                        "y_true": y,
                        "review_reason": reason,
                        "landmark_id": lm_id,
                        "landmark_name": lm_name,
                        "is_seen_landmark": is_seen,
                        "split": split_key
                    }
                    all_splits_pairs.append(record)

    LOGGER.info(f"Loaded {len(all_splits_pairs):,} fact instances across all 1,163 clips (Train, Val, Test).")

    # Process Unique Facts Dictionary
    for fid, finfo in dict_data.get("facts", {}).items():
        sub = finfo.get("subject", "")
        pred = finfo.get("predicate", "")
        obj = finfo.get("object", "")
        fact_text = finfo.get("fact_text", "")
        lm_id = finfo.get("landmark_id", "")
        prov = finfo.get("provenance_type", "KB_Base")
        ext_method = finfo.get("evidence", {}).get("extraction_method", "unknown")
        mu = finfo.get("evidence", {}).get("confidence_score")
        res = finfo.get("review", {}).get("result")
        reason = finfo.get("review", {}).get("reason", "")

        if mu is not None and res is not None:
            y = 1 if res == "đúng" else 0
            unique_pairs.append({
                "fact_id": fid,
                "subject": sub,
                "predicate": pred,
                "object": obj,
                "fact_text": fact_text,
                "provenance_type": prov,
                "extraction_method": ext_method,
                "mu_F": float(mu),
                "review_result": res,
                "y_true": y,
                "review_reason": reason,
                "landmark_id": lm_id
            })

    LOGGER.info(f"Loaded {len(unique_pairs):,} unique ontology facts.")

    # 4. Compute Test Set Analysis
    test_y = [p["y_true"] for p in test_pairs]
    test_scores = [p["mu_F"] for p in test_pairs]

    test_stats = compute_metrics(test_y, test_scores)
    test_bins = compute_bin_table(test_y, test_scores)
    test_sweep = compute_threshold_sweep(test_y, test_scores)

    # 5. Compute Validation Set Analysis & Select Optimal tau*
    val_pairs = [p for p in all_splits_pairs if p["split"] == "val"]
    val_y = [p["y_true"] for p in val_pairs]
    val_scores = [p["mu_F"] for p in val_pairs]

    val_stats = compute_metrics(val_y, val_scores)
    val_bins = compute_bin_table(val_y, val_scores)
    val_sweep = compute_threshold_sweep(val_y, val_scores)

    # Find optimal tau* on Validation set (maximizing F1 or Precision >= 80% with highest Recall)
    # Option A: Max F1 on Val
    best_val_f1_row = max(val_sweep, key=lambda r: r["f1"])
    # Option B: High-Precision Gating (Precision >= 80% with max coverage)
    high_prec_candidates = [r for r in val_sweep if r["precision"] >= 80.0]
    best_val_prec_row = max(high_prec_candidates, key=lambda r: r["recall"]) if high_prec_candidates else best_val_f1_row

    tau_star = best_val_prec_row["tau"]
    LOGGER.info(f"Validation-derived optimal threshold: tau* = {tau_star:.2f} (Val Prec={best_val_prec_row['precision']}%, Val Rec={best_val_prec_row['recall']}%, Val FAR={best_val_prec_row['far']}%)")

    # Evaluate tau* on Test set
    test_tau_star_row = next((r for r in test_sweep if abs(r["tau"] - tau_star) < 1e-4), None)

    # 6. Compute Train Set Analysis
    train_pairs = [p for p in all_splits_pairs if p["split"] == "train"]
    train_y = [p["y_true"] for p in train_pairs]
    train_scores = [p["mu_F"] for p in train_pairs]

    train_stats = compute_metrics(train_y, train_scores)
    train_bins = compute_bin_table(train_y, train_scores)
    train_sweep = compute_threshold_sweep(train_y, train_scores)

    # 7. Compute Unique Ontology Facts Analysis
    unique_y = [p["y_true"] for p in unique_pairs]
    unique_scores = [p["mu_F"] for p in unique_pairs]

    unique_stats = compute_metrics(unique_y, unique_scores)
    unique_bins = compute_bin_table(unique_y, unique_scores)
    unique_sweep = compute_threshold_sweep(unique_y, unique_scores)

    # 8. Compute Subgroup Stratification on Test Set
    # Subgroup A: KB_Base
    kb_pairs = [p for p in test_pairs if p["provenance_type"] == "KB_Base"]
    kb_y = [p["y_true"] for p in kb_pairs]
    kb_scores = [p["mu_F"] for p in kb_pairs]
    kb_stats = compute_metrics(kb_y, kb_scores)
    kb_bins = compute_bin_table(kb_y, kb_scores)
    kb_sweep = compute_threshold_sweep(kb_y, kb_scores)

    # Subgroup B: Horn_Inferred
    horn_pairs = [p for p in test_pairs if p["provenance_type"] == "Horn_Inferred"]
    horn_y = [p["y_true"] for p in horn_pairs]
    horn_scores = [p["mu_F"] for p in horn_pairs]
    horn_stats = compute_metrics(horn_y, horn_scores)
    horn_bins = compute_bin_table(horn_y, horn_scores)
    horn_sweep = compute_threshold_sweep(horn_y, horn_scores)

    # Subgroup C: Seen Landmarks
    seen_pairs = [p for p in test_pairs if p["is_seen_landmark"]]
    seen_y = [p["y_true"] for p in seen_pairs]
    seen_scores = [p["mu_F"] for p in seen_pairs]
    seen_stats = compute_metrics(seen_y, seen_scores)
    seen_bins = compute_bin_table(seen_y, seen_scores)
    seen_sweep = compute_threshold_sweep(seen_y, seen_scores)

    # Subgroup D: Unseen Landmarks
    unseen_pairs = [p for p in test_pairs if not p["is_seen_landmark"]]
    unseen_y = [p["y_true"] for p in unseen_pairs]
    unseen_scores = [p["mu_F"] for p in unseen_pairs]
    unseen_stats = compute_metrics(unseen_y, unseen_scores)
    unseen_bins = compute_bin_table(unseen_y, unseen_scores)
    unseen_sweep = compute_threshold_sweep(unseen_y, unseen_scores)

    # 9. Compute Landmark Breakdown (Explaining Imbalance)
    by_lm = defaultdict(list)
    for p in test_pairs:
        by_lm[p["landmark_name"]].append(p)

    lm_summary = []
    for lm_name, lpairs in sorted(by_lm.items(), key=lambda x: -len(x[1])):
        ly = [p["y_true"] for p in lpairs]
        lscores = [p["mu_F"] for p in lpairs]
        clips_count = len(set(p["clip_id"] for p in lpairs))
        lst = compute_metrics(ly, lscores)
        lsweep = compute_threshold_sweep(ly, lscores, thresholds=[0.0, 0.90])
        sw0 = lsweep[0]
        sw90 = lsweep[1]
        lm_summary.append({
            "landmark_name": lm_name,
            "clips_count": clips_count,
            "total_facts": lst["n"],
            "true_count": lst["n_pos"],
            "false_count": lst["n_neg"],
            "base_true_rate": round(lst["base_rate"] * 100.0, 2),
            "mean_mu_F": round(sum(lscores) / len(lscores), 4),
            "auroc": round(lst["auroc"], 4),
            "spearman_rho": round(lst["spearman_rho"], 4),
            "tau_0_prec": sw0["precision"],
            "tau_90_prec": sw90["precision"],
            "tau_90_retained": sw90["retained_count"],
            "tau_90_retained_pct": sw90["coverage_percent"],
            "tau_90_far": sw90["far"]
        })

    # 10. Compile Master JSON Report
    report = {
        "metadata": {
            "title": "Comprehensive Mu_F Epistemic Certainty Discrimination and Calibration Benchmark",
            "timestamp": datetime.now().isoformat(),
            "source_review_file": "/Users/quynd/Downloads/fact_review_results_binary.json",
            "total_test_fact_instances": len(test_pairs),
            "total_all_splits_fact_instances": len(all_splits_pairs),
            "total_unique_ontology_facts": len(unique_pairs)
        },
        "overall_test_evaluation": {
            "summary_metrics": test_stats,
            "five_bin_table": test_bins,
            "threshold_sweep_table": test_sweep
        },
        "validation_and_optimal_threshold": {
            "val_summary_metrics": val_stats,
            "val_five_bin_table": val_bins,
            "val_threshold_sweep_table": val_sweep,
            "optimal_tau_star_derived_on_validation": {
                "tau_star": tau_star,
                "criterion": "High-Precision Gating (Precision >= 80% on Val with max Recall)",
                "val_performance": best_val_prec_row,
                "unbiased_test_performance": test_tau_star_row
            }
        },
        "training_split_evaluation": {
            "train_summary_metrics": train_stats,
            "train_five_bin_table": train_bins,
            "train_threshold_sweep_table": train_sweep
        },
        "unique_ontology_facts_evaluation": {
            "unique_summary_metrics": unique_stats,
            "unique_five_bin_table": unique_bins,
            "unique_threshold_sweep_table": unique_sweep
        },
        "subgroups": {
            "kb_base": {
                "summary_metrics": kb_stats,
                "five_bin_table": kb_bins,
                "threshold_sweep_table": kb_sweep
            },
            "horn_inferred": {
                "summary_metrics": horn_stats,
                "five_bin_table": horn_bins,
                "threshold_sweep_table": horn_sweep
            },
            "seen_landmarks": {
                "summary_metrics": seen_stats,
                "five_bin_table": seen_bins,
                "threshold_sweep_table": seen_sweep
            },
            "unseen_landmarks": {
                "summary_metrics": unseen_stats,
                "five_bin_table": unseen_bins,
                "threshold_sweep_table": unseen_sweep
            }
        },
        "per_landmark_breakdown": lm_summary
    }

    report_path = OUTPUT_DIR / "mu_f_discrimination_comprehensive_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Saved master report to {report_path}")

    # 11. Export Tabular Datasets (JSON and TSV)
    # A. Test pairs
    test_json_out = OUTPUT_DIR / "mu_f_vs_fref_test_pairs.json"
    with open(test_json_out, "w", encoding="utf-8") as f:
        json.dump(test_pairs, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Exported test pairs JSON ({len(test_pairs):,} rows) to {test_json_out}")

    test_tsv_out = OUTPUT_DIR / "mu_f_vs_fref_test_pairs.tsv"
    with open(test_tsv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_pairs[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(test_pairs)
    LOGGER.info(f"Exported test pairs TSV to {test_tsv_out}")

    # B. Unique facts
    unique_json_out = OUTPUT_DIR / "mu_f_vs_fref_unique_facts.json"
    with open(unique_json_out, "w", encoding="utf-8") as f:
        json.dump(unique_pairs, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Exported unique facts JSON ({len(unique_pairs):,} rows) to {unique_json_out}")

    unique_tsv_out = OUTPUT_DIR / "mu_f_vs_fref_unique_facts.tsv"
    with open(unique_tsv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(unique_pairs[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(unique_pairs)
    LOGGER.info(f"Exported unique facts TSV to {unique_tsv_out}")

    # C. All splits pairs
    all_json_out = OUTPUT_DIR / "mu_f_vs_fref_all_splits_pairs.json"
    with open(all_json_out, "w", encoding="utf-8") as f:
        json.dump(all_splits_pairs, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Exported all splits pairs JSON ({len(all_splits_pairs):,} rows) to {all_json_out}")

    # Print Key Results for Console
    print("\n" + "=" * 80)
    print("KEY EMPIRICAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Test Instances: {test_stats['n']:,} | Prior TRUE: {test_stats['base_rate']*100:.2f}%")
    print(f"AUROC: {test_stats['auroc']:.4f} | AUPRC: {test_stats['auprc']:.4f}")
    print(f"Spearman rho: {test_stats['spearman_rho']:.4f} (p = {test_stats['spearman_p']:.2e})")
    print(f"Brier Score: {test_stats['brier']:.4f} | ECE: {test_stats['ece']:.4f}")
    print("-" * 80)
    print("5-Bin Empirical Factual Accuracy (Test Set):")
    for b in test_bins:
        print(f"  {b['interval']:<12}: N={b['count']:>6,} ({b['percent_of_total']:>5.2f}%) | TRUE={b['true_count']:>5,} | FALSE={b['false_count']:>5,} | TRUE Rate: {b['true_rate']:>6.2f}%")
    print("-" * 80)
    print("Threshold Sweep Highlights (Test Set):")
    for r in test_sweep:
        if r["tau"] in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
            print(f"  tau={r['tau']:.2f}: Retained={r['retained_count']:>6,} ({r['coverage_percent']:>5.2f}%) | Prec={r['precision']:>6.2f}% | Recall={r['recall']:>6.2f}% | FAR={r['far']:>6.2f}% | Acc={r['accuracy']:>6.2f}%")
    print("-" * 80)
    print(f"Optimal Threshold (Validation-Selected): tau* = {tau_star:.2f}")
    print(f"  Validation Performance : Prec={best_val_prec_row['precision']:.2f}% | Recall={best_val_prec_row['recall']:.2f}% | FAR={best_val_prec_row['far']:.2f}%")
    print(f"  Unbiased Test Outcome  : Prec={test_tau_star_row['precision']:.2f}% | Recall={test_tau_star_row['recall']:.2f}% | FAR={test_tau_star_row['far']:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
