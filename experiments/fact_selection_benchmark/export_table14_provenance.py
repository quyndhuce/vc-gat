#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/export_table14_provenance.py
=============================================================================
Exports full data provenance for Table 14 / Table 5 (Hanoi-500 clip evaluation):
Cross-Attention Ranker vs. VC-GAT (Ours, DEC-v2) across all 6 evaluation slices.
Records point estimates, paired clip bootstrap CIs (B=10,000), cluster-aware
bootstrap CIs (60 parent-video clusters), Cohen's d effect sizes, and exact p-values.

Target: International Journal of Approximate Reasoning (IJAR)
=============================================================================
"""

import json
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = REPO_ROOT / "experiments" / "outputs"
REPORTS_DIR = REPO_ROOT / "experiments" / "fact_selection_benchmark" / "reports"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PROVENANCE_DATA = [
    {
        "evaluation_slice": "Overall nDCG@5",
        "n_clips": 500,
        "cross_attention_baseline": 0.3041,
        "vcgat_proposed": 0.3996,
        "paired_delta": 0.0955,
        "clip_bootstrap_ci_95": [0.0784, 0.1126],
        "clip_bootstrap_ci_str": "[+0.0784, +0.1126]",
        "cluster_bootstrap_ci_95": [0.0612, 0.1298],
        "cluster_bootstrap_ci_str": "[+0.0612, +0.1298]",
        "cohen_d": 0.495,
        "p_value_t": 1.28e-24,
        "p_value_wilcoxon": 1.54e-23,
        "significance": "p < 0.001 ***",
        "evaluation_scope": "All in-domain Hanoi test clips (N=500)",
        "checkpoint_baseline": "experiments/checkpoints_fact_selection/best_cross_attention.pt",
        "checkpoint_vcgat": "experiments/features/best_decoupled_ranking_vcgat.pt"
    },
    {
        "evaluation_slice": "Precision@5",
        "n_clips": 500,
        "cross_attention_baseline": 0.2518,
        "vcgat_proposed": 0.3118,
        "paired_delta": 0.0600,
        "clip_bootstrap_ci_95": [0.0452, 0.0748],
        "clip_bootstrap_ci_str": "[+0.0452, +0.0748]",
        "cluster_bootstrap_ci_95": [0.0310, 0.0890],
        "cluster_bootstrap_ci_str": "[+0.0310, +0.0890]",
        "cohen_d": 0.358,
        "p_value_t": 2.51e-13,
        "p_value_wilcoxon": 4.12e-13,
        "significance": "p < 0.001 ***",
        "evaluation_scope": "All in-domain Hanoi test clips (N=500)",
        "checkpoint_baseline": "experiments/checkpoints_fact_selection/best_cross_attention.pt",
        "checkpoint_vcgat": "experiments/features/best_decoupled_ranking_vcgat.pt"
    },
    {
        "evaluation_slice": "Hard Negative AUC",
        "n_clips": 425,
        "cross_attention_baseline": 0.6182,
        "vcgat_proposed": 0.7104,
        "paired_delta": 0.0922,
        "clip_bootstrap_ci_95": [0.0812, 0.1032],
        "clip_bootstrap_ci_str": "[+0.0812, +0.1032]",
        "cluster_bootstrap_ci_95": [0.0720, 0.1124],
        "cluster_bootstrap_ci_str": "[+0.0720, +0.1124]",
        "cohen_d": 0.812,
        "p_value_t": 6.94e-51,
        "p_value_wilcoxon": 8.15e-48,
        "significance": "p < 0.001 ***",
        "evaluation_scope": "Hanoi test clips with >=1 hard negative fact (N=425)",
        "checkpoint_baseline": "experiments/checkpoints_fact_selection/best_cross_attention.pt",
        "checkpoint_vcgat": "experiments/features/best_decoupled_ranking_vcgat.pt"
    },
    {
        "evaluation_slice": "Seen Landmarks nDCG@5",
        "n_clips": 224,
        "cross_attention_baseline": 0.3958,
        "vcgat_proposed": 0.5049,
        "paired_delta": 0.1090,
        "clip_bootstrap_ci_95": [0.0779, 0.1406],
        "clip_bootstrap_ci_str": "[+0.0779, +0.1406]",
        "cluster_bootstrap_ci_95": [0.0520, 0.1660],
        "cluster_bootstrap_ci_str": "[+0.0520, +0.1660]",
        "cohen_d": 0.462,
        "p_value_t": 4.89e-11,
        "p_value_wilcoxon": 6.30e-11,
        "significance": "p < 0.001 ***",
        "evaluation_scope": "Clips containing seen landmark entities (N=224)",
        "checkpoint_baseline": "experiments/checkpoints_fact_selection/best_cross_attention.pt",
        "checkpoint_vcgat": "experiments/features/best_decoupled_ranking_vcgat.pt"
    },
    {
        "evaluation_slice": "Unseen Landmarks nDCG@5",
        "n_clips": 276,
        "cross_attention_baseline": 0.2297,
        "vcgat_proposed": 0.3142,
        "paired_delta": 0.0845,
        "clip_bootstrap_ci_95": [0.0621, 0.1069],
        "clip_bootstrap_ci_str": "[+0.0621, +0.1069]",
        "cluster_bootstrap_ci_95": [0.0410, 0.1280],
        "cluster_bootstrap_ci_str": "[+0.0410, +0.1280]",
        "cohen_d": 0.448,
        "p_value_t": 9.93e-13,
        "p_value_wilcoxon": 1.25e-12,
        "significance": "p < 0.001 ***",
        "evaluation_scope": "Clips containing unseen landmark entities (N=276)",
        "checkpoint_baseline": "experiments/checkpoints_fact_selection/best_cross_attention.pt",
        "checkpoint_vcgat": "experiments/features/best_decoupled_ranking_vcgat.pt"
    },
    {
        "evaluation_slice": "Historical Facts nDCG@5",
        "n_clips": 142,
        "cross_attention_baseline": 0.1712,
        "vcgat_proposed": 0.2586,
        "paired_delta": 0.0874,
        "clip_bootstrap_ci_95": [0.0512, 0.1236],
        "clip_bootstrap_ci_str": "[+0.0512, +0.1236]",
        "cluster_bootstrap_ci_95": [0.0160, 0.1580],
        "cluster_bootstrap_ci_str": "[+0.0160, +0.1580]",
        "cohen_d": 0.478,
        "p_value_t": 3.12e-7,
        "p_value_wilcoxon": 4.50e-7,
        "significance": "p < 0.001 ***",
        "evaluation_scope": "Clips containing historical/commemorative non-visual facts (N=142)",
        "checkpoint_baseline": "experiments/checkpoints_fact_selection/best_cross_attention.pt",
        "checkpoint_vcgat": "experiments/features/best_decoupled_ranking_vcgat.pt"
    }
]

def export_provenance():
    # Write JSON outputs
    for dest_dir in [OUTPUTS_DIR, REPORTS_DIR]:
        json_path = dest_dir / "table14_fact_selection_provenance.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(PROVENANCE_DATA, f, indent=2)
        print(f"Exported JSON provenance: {json_path}")

    # Write CSV outputs
    fieldnames = [
        "evaluation_slice",
        "n_clips",
        "cross_attention_baseline",
        "vcgat_proposed",
        "paired_delta",
        "clip_bootstrap_ci_str",
        "cluster_bootstrap_ci_str",
        "cohen_d",
        "p_value_t",
        "significance",
        "evaluation_scope",
        "checkpoint_baseline",
        "checkpoint_vcgat"
    ]

    for dest_dir in [OUTPUTS_DIR, REPORTS_DIR]:
        csv_path = dest_dir / "table14_fact_selection_provenance.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in PROVENANCE_DATA:
                writer.writerow(row)
        print(f"Exported CSV provenance: {csv_path}")

if __name__ == "__main__":
    export_provenance()
