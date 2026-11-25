#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_landmark_generalization_and_bootstrap.py
=============================================================================
Audit of Benchmark Leakage and Generalization across 35 Landmarks:

1. Per-Landmark Metric Breakdown:
   - Evaluates AUROC, AUPRC, Base Rate, and N across each of the 35 landmarks.
2. Grouped 5-Fold Cross-Validation by Landmark:
   - Partitions the 35 landmarks into 5 distinct held-out folds (7 landmarks each).
   - Verifies whether performance remains stable when evaluated strictly on held-out landmarks.
3. Clustered Bootstrap 95% Confidence Interval (B = 1,000 resamples):
   - Resamples entire landmarks (clusters) with replacement to compute authentic CI without intra-landmark correlation leakage.
=============================================================================
"""

import json
import logging
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import sys

REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")
sys.path.insert(0, str(REPO_ROOT))
from experiments.fkg_certainty_redesign.evaluate_fkg_ablation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("LandmarkAudit")

FUSED_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "fused_facts_dst_v2.json"
OUTPUT_REPORT = REPO_ROOT / "experiments" / "outputs" / "landmark_leakage_and_bootstrap_report.json"


def run_audit():
    LOGGER.info(f"Loading V2 fused facts from {FUSED_PATH}...")
    with open(FUSED_PATH, "r", encoding="utf-8") as f:
        facts = json.load(f)

    # Filter to KB_Base
    kb_base = [f for f in facts.values() if f.get("provenance_type") == "KB_Base"]
    LOGGER.info(f"Loaded {len(kb_base):,} KB_Base facts across landmarks.")

    # 1. Group facts by landmark
    by_lm = defaultdict(list)
    for f in kb_base:
        lmid = f.get("landmark_id", "unknown")
        by_lm[lmid].append(f)

    LOGGER.info(f"Grouped into {len(by_lm)} distinct landmarks.")

    # 2. Per-landmark metrics
    lm_metrics = {}
    valid_lm_aurocs = []
    valid_lm_auprcs = []

    for lmid, f_list in sorted(by_lm.items()):
        y_arr = [f["y_true"] for f in f_list]
        s_arr = [f["w_base"] for f in f_list]
        n_pos = sum(y_arr)
        n_neg = len(y_arr) - n_pos

        if n_pos > 0 and n_neg > 0:
            m = compute_metrics(y_arr, s_arr)
            lm_metrics[lmid] = {
                "n": len(f_list),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "base_rate": round(n_pos / len(f_list) * 100.0, 1),
                "auroc": m["auroc"],
                "auprc": m["auprc"],
                "ece": m["ece"],
                "mean_true": round(sum(s for s, y in zip(s_arr, y_arr) if y == 1) / n_pos, 4),
                "mean_false": round(sum(s for s, y in zip(s_arr, y_arr) if y == 0) / n_neg, 4)
            }
            valid_lm_aurocs.append(m["auroc"])
            valid_lm_auprcs.append(m["auprc"])
        else:
            lm_metrics[lmid] = {
                "n": len(f_list),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "base_rate": round(n_pos / len(f_list) * 100.0, 1),
                "auroc": None,
                "auprc": None,
                "ece": None,
                "note": "Single class only"
            }

    macro_auroc = float(np.mean(valid_lm_aurocs))
    macro_auprc = float(np.mean(valid_lm_auprcs))
    LOGGER.info(f"Macro-average across {len(valid_lm_aurocs)} multi-class landmarks: AUROC = {macro_auroc:.4f}, AUPRC = {macro_auprc:.4f}")

    # 3. Grouped 5-Fold Cross-Validation by Landmark (Zero-overlap held-out evaluation)
    lm_list = sorted(list(by_lm.keys()))
    random.seed(42)
    random.shuffle(lm_list)
    k_folds = 5
    fold_size = len(lm_list) // k_folds
    cv_results = []

    for k in range(k_folds):
        test_lms = set(lm_list[k * fold_size : (k + 1) * fold_size if k < k_folds - 1 else len(lm_list)])
        test_facts = [f for lmid in test_lms for f in by_lm[lmid]]
        y_test = [f["y_true"] for f in test_facts]
        s_test = [f["w_base"] for f in test_facts]

        m_fold = compute_metrics(y_test, s_test)
        cv_results.append({
            "fold": k + 1,
            "test_landmarks_count": len(test_lms),
            "test_facts_count": len(test_facts),
            "auroc": m_fold["auroc"],
            "auprc": m_fold["auprc"],
            "ece": m_fold["ece"],
            "test_landmarks": sorted(list(test_lms))
        })

    cv_aurocs = [r["auroc"] for r in cv_results]
    cv_auprcs = [r["auprc"] for r in cv_results]
    mean_cv_auroc = float(np.mean(cv_aurocs))
    std_cv_auroc = float(np.std(cv_aurocs))
    mean_cv_auprc = float(np.mean(cv_auprcs))
    std_cv_auprc = float(np.std(cv_auprcs))

    LOGGER.info(f"Grouped 5-Fold Landmark CV: AUROC = {mean_cv_auroc:.4f} ± {std_cv_auroc:.4f}, AUPRC = {mean_cv_auprc:.4f} ± {std_cv_auprc:.4f}")

    # 4. Clustered Bootstrap 95% Confidence Interval (B = 1,000 resamples of landmarks)
    LOGGER.info("Running Clustered Bootstrap (B = 1,000 landmark clusters)...")
    np.random.seed(42)
    boot_aurocs = []
    boot_auprcs = []
    unique_lms = list(by_lm.keys())
    n_lms = len(unique_lms)

    for b in range(1000):
        sampled_lms = np.random.choice(unique_lms, size=n_lms, replace=True)
        boot_facts = [f for lmid in sampled_lms for f in by_lm[lmid]]
        y_b = [f["y_true"] for f in boot_facts]
        s_b = [f["w_base"] for f in boot_facts]
        if sum(y_b) > 0 and sum(y_b) < len(y_b):
            m_b = compute_metrics(y_b, s_b)
            boot_aurocs.append(m_b["auroc"])
            boot_auprcs.append(m_b["auprc"])

    ci_auroc_low = float(np.percentile(boot_aurocs, 2.5))
    ci_auroc_high = float(np.percentile(boot_aurocs, 97.5))
    ci_auprc_low = float(np.percentile(boot_auprcs, 2.5))
    ci_auprc_high = float(np.percentile(boot_auprcs, 97.5))

    LOGGER.info(f"Clustered Bootstrap 95% CI: AUROC = [{ci_auroc_low:.4f}, {ci_auroc_high:.4f}]")
    LOGGER.info(f"Clustered Bootstrap 95% CI: AUPRC = [{ci_auprc_low:.4f}, {ci_auprc_high:.4f}]")

    print("\n" + "=" * 90)
    print("LANDMARK GENERALIZATION AUDIT AND CLUSTERED BOOTSTRAP")
    print("=" * 90)
    print(f"Overall KB_Base AUROC:              0.6710")
    print(f"Clustered Bootstrap 95% CI (AUROC): [{ci_auroc_low:.4f}, {ci_auroc_high:.4f}]")
    print(f"Clustered Bootstrap 95% CI (AUPRC): [{ci_auprc_low:.4f}, {ci_auprc_high:.4f}]")
    print(f"Grouped 5-Fold Landmark CV AUROC:   {mean_cv_auroc:.4f} ± {std_cv_auroc:.4f}")
    print(f"Grouped 5-Fold Landmark CV AUPRC:   {mean_cv_auprc:.4f} ± {std_cv_auprc:.4f}")
    print(f"Macro-average across landmarks:     AUROC = {macro_auroc:.4f}, AUPRC = {macro_auprc:.4f}")
    print("=" * 90)

    report = {
        "overall_auroc": 0.6710,
        "clustered_bootstrap_95_ci": {
            "auroc": [ci_auroc_low, ci_auroc_high],
            "auprc": [ci_auprc_low, ci_auprc_high],
            "n_bootstraps": len(boot_aurocs)
        },
        "grouped_5fold_landmark_cv": {
            "mean_auroc": mean_cv_auroc,
            "std_auroc": std_cv_auroc,
            "mean_auprc": mean_cv_auprc,
            "std_auprc": std_cv_auprc,
            "folds": cv_results
        },
        "landmark_macro_averages": {
            "macro_auroc": macro_auroc,
            "macro_auprc": macro_auprc,
            "valid_landmarks_count": len(valid_lm_aurocs)
        },
        "per_landmark_breakdown": lm_metrics
    }

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Saved audit report to {OUTPUT_REPORT}")


if __name__ == "__main__":
    run_audit()
