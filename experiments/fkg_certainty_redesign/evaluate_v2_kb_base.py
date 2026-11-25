#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_v2_kb_base.py
=============================================================================
Definitive Verification of V2 Pipeline on KB_Base Facts:
Evaluates whether Context-Enrichment + Schema Admissibility + Relevance Gating
+ Spatial Hierarchy + Yager Fusion resolves the AUROC < 0.5 inversion.

Compares:
  - M0: Original Baseline
  - M2_V1: Initial NLI + DST (with hard zero-killing & uncontextualized chunking)
  - M2_V2: Full Conflict-Aware & Context-Enriched Pipeline (Proposed Remedy)
=============================================================================
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
import sys
sys.path.insert(0, ".")
from experiments.fkg_certainty_redesign.evaluate_fkg_ablation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("EvaluateV2KBBase")

REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")
V1_PATH = REPO_ROOT / "experiments" / "outputs" / "fkg_facts_verified.json"
V2_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "fused_facts_dst_v2.json"
OUTPUT_REPORT_PATH = REPO_ROOT / "experiments" / "outputs" / "v2_kb_base_benchmark_report.json"


def evaluate_v2():
    LOGGER.info(f"Loading V1 facts from {V1_PATH}...")
    with open(V1_PATH, "r", encoding="utf-8") as f:
        v1_facts = json.load(f)

    LOGGER.info(f"Loading V2 facts from {V2_PATH}...")
    with open(V2_PATH, "r", encoding="utf-8") as f:
        v2_facts = json.load(f)

    # Filter to KB_Base
    kb_base_v1 = [f for f in v1_facts.values() if f.get("provenance_type") == "KB_Base"]
    kb_base_v2 = [v2_facts[f["fact_id"]] for f in kb_base_v1]

    y_true_orig = [f["y_true"] for f in kb_base_v2]
    y_true_strict = [f.get("y_strict", f["y_true"]) for f in kb_base_v2]

    scores_m0 = [f["old_mu_F"] for f in kb_base_v2]
    scores_m2_v1 = [f["dst"]["w_base"] for f in kb_base_v1]
    scores_m2_v2 = [f["w_base"] for f in kb_base_v2]

    # 1. Full KB_Base (N = 7,088)
    m_m0_full = compute_metrics(y_true_orig, scores_m0)
    m_v1_full = compute_metrics(y_true_orig, scores_m2_v1)
    m_v2_full = compute_metrics(y_true_orig, scores_m2_v2)
    m_v2_strict = compute_metrics(y_true_strict, scores_m2_v2)

    # 2. Admissible KB_Base (N = 6,470)
    adm_indices = [i for i, f in enumerate(kb_base_v2) if f.get("is_admissible", True)]
    y_true_adm = [y_true_orig[i] for i in adm_indices]
    y_strict_adm = [y_true_strict[i] for i in adm_indices]
    scores_m0_adm = [scores_m0[i] for i in adm_indices]
    scores_v1_adm = [scores_m2_v1[i] for i in adm_indices]
    scores_v2_adm = [scores_m2_v2[i] for i in adm_indices]

    m_m0_adm = compute_metrics(y_true_adm, scores_m0_adm)
    m_v1_adm = compute_metrics(y_true_adm, scores_v1_adm)
    m_v2_adm = compute_metrics(y_true_adm, scores_v2_adm)
    m_v2_strict_adm = compute_metrics(y_strict_adm, scores_v2_adm)

    print("\n" + "=" * 90)
    print("V2 BENCHMARK EVALUATION ON KB_BASE FACTS")
    print("=" * 90)
    print("--- 1. Full KB_Base Set (N = 7,088) ---")
    print(f"{'Model':<35} | {'AUROC':<7} | {'AUPRC':<7} | {'P@0.8':<7} | {'P@0.9':<7} | {'ECE':<7} | {'Brier':<7}")
    print("-" * 90)
    for name, m in [
        ("M0_baseline (Old)", m_m0_full),
        ("M2_V1 (Initial NLI+DST)", m_v1_full),
        ("M2_V2 (Context+Yager+Gate)", m_v2_full),
        ("M2_V2 (Strict Triple Direction)", m_v2_strict),
    ]:
        p08 = m["sweep"].get("tau_0.8", {}).get("precision", 0.0)
        p09 = m["sweep"].get("tau_0.9", {}).get("precision", 0.0)
        print(f"{name:<35} | {m['auroc']:<7.4f} | {m['auprc']:<7.4f} | {p08:<7.2f} | {p09:<7.2f} | {m['ece']:<7.4f} | {m['brier']:<7.4f}")

    print("\n--- 2. Admissible KB_Base Set (N = 6,470, excluding dummy headers & stopword triples) ---")
    print(f"{'Model':<35} | {'AUROC':<7} | {'AUPRC':<7} | {'P@0.8':<7} | {'P@0.9':<7} | {'ECE':<7} | {'Brier':<7}")
    print("-" * 90)
    for name, m in [
        ("M0_baseline (Old)", m_m0_adm),
        ("M2_V1 (Initial NLI+DST)", m_v1_adm),
        ("M2_V2 (Context+Yager+Gate)", m_v2_adm),
        ("M2_V2 (Strict Triple Direction)", m_v2_strict_adm),
    ]:
        p08 = m["sweep"].get("tau_0.8", {}).get("precision", 0.0)
        p09 = m["sweep"].get("tau_0.9", {}).get("precision", 0.0)
        print(f"{name:<35} | {m['auroc']:<7.4f} | {m['auprc']:<7.4f} | {p08:<7.2f} | {p09:<7.2f} | {m['ece']:<7.4f} | {m['brier']:<7.4f}")
    print("=" * 90 + "\n")

    # Means check
    mean_t_v2 = sum(s for s, y in zip(scores_m2_v2, y_true_orig) if y == 1) / sum(y_true_orig)
    mean_f_v2 = sum(s for s, y in zip(scores_m2_v2, y_true_orig) if y == 0) / (len(y_true_orig) - sum(y_true_orig))
    print(f"Mean Score V2 (Full): TRUE = {mean_t_v2:.4f}, FALSE = {mean_f_v2:.4f}")

    out_rep = {
        "full_kb_base": {
            "M0_baseline": m_m0_full,
            "M2_V1": m_v1_full,
            "M2_V2": m_v2_full,
            "M2_V2_strict": m_v2_strict
        },
        "admissible_kb_base": {
            "M0_baseline": m_m0_adm,
            "M2_V1": m_v1_adm,
            "M2_V2": m_v2_adm,
            "M2_V2_strict": m_v2_strict_adm
        }
    }
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_rep, f, indent=2)
    LOGGER.info(f"Report saved to {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    evaluate_v2()
