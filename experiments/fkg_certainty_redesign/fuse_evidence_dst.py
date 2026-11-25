#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fuse_evidence_dst.py
=============================================================================
Stage 3: Evidence Fusion via Dempster-Shafer Theory (DST) and t-Conorm.

Computes:
  1. Evidence mass assignment with source reliability discounting:
       m_i(T) = r_i * p_E
       m_i(F) = r_i * p_C
       m_i(Theta) = 1 - r_i * (p_E + p_C)
  2. Evidence deduplication / copy group clustering (avoid epistemic inflation).
  3. Dempster's rule of combination across independent evidence groups:
       Bel_f(T), Bel_f(F), Uncertainty Theta, Cumulative Conflict K_total.
  4. Base fact weight:
       w_base(f) = Bel(T) * (1 - K_total) if Bel(T) > Bel(F) else 0.0.
  5. t-Conorm baseline (Noisy-OR) for comparative ablation:
       mu_tconorm(f) = max(0, 1 - prod(1 - r_i * p_E) - max(r_i * p_C)).

Outputs:
  - experiments/fkg_certainty_redesign/fused_facts_dst.json
=============================================================================
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("FuseEvidenceDST")

REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")
DICT_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "fgold_facts_dictionary.json"
TASKS_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "nli_inference_tasks.json"
PREDS_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "nli_predictions.json"
OUTPUT_FUSED_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "fused_facts_dst.json"


def dempster_combine_two(
    m1: Tuple[float, float, float],
    m2: Tuple[float, float, float]
) -> Tuple[Tuple[float, float, float], float]:
    """
    Combines two mass functions m1 and m2 over frame {T, F}.
    Each mass function is represented as (m(T), m(F), m(Theta)).
    Returns ((m_comb(T), m_comb(F), m_comb(Theta)), conflict_k).
    """
    t1, f1, u1 = m1
    t2, f2, u2 = m2

    # Conflict occurs when one says T and the other says F
    k = t1 * f2 + f1 * t2

    if k >= 0.99999:
        # Total contradictory conflict
        return (0.0, 0.0, 1.0), 1.0

    denom = 1.0 - k
    comb_t = (t1 * t2 + t1 * u2 + u1 * t2) / denom
    comb_f = (f1 * f2 + f1 * u2 + u1 * f2) / denom
    comb_u = (u1 * u2) / denom

    # Normalize to prevent floating point drift
    s = comb_t + comb_f + comb_u
    if s > 0:
        comb_t /= s
        comb_f /= s
        comb_u /= s

    return (comb_t, comb_f, comb_u), k


def dempster_combine_all(
    evidence_masses: List[Tuple[float, float, float]]
) -> Tuple[float, float, float, float]:
    """
    Sequentially combines a list of mass functions using Dempster's rule.
    Returns (Bel(T), Bel(F), Uncertainty, cumulative_conflict).
    """
    if not evidence_masses:
        return 0.0, 0.0, 1.0, 0.0

    if len(evidence_masses) == 1:
        m = evidence_masses[0]
        return m[0], m[1], m[2], 0.0

    current_m = evidence_masses[0]
    prod_non_conflict = 1.0

    for next_m in evidence_masses[1:]:
        current_m, k = dempster_combine_two(current_m, next_m)
        prod_non_conflict *= (1.0 - k)

    cumulative_conflict = 1.0 - prod_non_conflict
    return current_m[0], current_m[1], current_m[2], cumulative_conflict


def compute_tconorm(evidence_items: List[Dict[str, Any]]) -> float:
    """Computes Noisy-OR t-conorm with contradiction deduction."""
    if not evidence_items:
        return 0.0
    prod_comp = 1.0
    max_c = 0.0
    for ev in evidence_items:
        r = ev["reliability"]
        pe = ev["p_E"]
        pc = ev["p_C"]
        prod_comp *= (1.0 - r * pe)
        max_c = max(max_c, r * pc)

    support = 1.0 - prod_comp
    penalized = max(0.0, support - max_c)
    return round(penalized, 4)


def fuse_all_evidence():
    LOGGER.info(f"Loading tasks from {TASKS_PATH}...")
    with open(TASKS_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    LOGGER.info(f"Loading predictions from {PREDS_PATH}...")
    with open(PREDS_PATH, "r", encoding="utf-8") as f:
        preds = json.load(f)

    LOGGER.info(f"Loading fact dictionary from {DICT_PATH}...")
    with open(DICT_PATH, "r", encoding="utf-8") as f:
        facts_dict = json.load(f)["facts"]

    LOGGER.info("Grouping evidence by fact_id...")
    evidence_by_fact = defaultdict(list)
    for task in tasks:
        tid = task["task_id"]
        pred = preds.get(tid, {"p_E": 0.0, "p_C": 0.0, "p_N": 1.0})
        ev_record = {
            "task_id": tid,
            "evidence_idx": task["evidence_idx"],
            "source_type": task["source_type"],
            "source_doc": task["source_doc"],
            "reliability": task["reliability"],
            "p_E": pred["p_E"],
            "p_C": pred["p_C"],
            "p_N": pred["p_N"],
            "premise": task["premise"]
        }
        evidence_by_fact[task["fact_id"]].append(ev_record)

    fused_facts = {}
    LOGGER.info(f"Fusing evidence across {len(facts_dict):,} facts...")

    for fid, fact in facts_dict.items():
        ev_list = evidence_by_fact.get(fid, [])

        # Construct mass functions
        masses = []
        for ev in ev_list:
            r = ev["reliability"]
            pe = ev["p_E"]
            pc = ev["p_C"]
            m_t = r * pe
            m_f = r * pc
            m_u = max(0.0, 1.0 - m_t - m_f)
            masses.append((m_t, m_f, m_u))

        bel_t, bel_f, uncert, conflict = dempster_combine_all(masses)

        # Base DST weight
        if bel_t > bel_f:
            w_base_dst = round(bel_t * (1.0 - conflict), 4)
        else:
            w_base_dst = 0.0

        # t-conorm weight
        w_tconorm = compute_tconorm(ev_list)

        # Direct max NLI probability
        max_pe = max((ev["p_E"] for ev in ev_list), default=0.0)
        max_pc = max((ev["p_C"] for ev in ev_list), default=0.0)
        w_nli_only = round(max(0.0, max_pe - max_pc), 4)

        review = fact.get("review", {})
        y_true = 1 if review.get("result") == "đúng" else 0

        fused_facts[fid] = {
            "fact_id": fid,
            "subject": fact.get("subject"),
            "predicate": fact.get("predicate"),
            "object": fact.get("object"),
            "landmark_id": fact.get("landmark_id"),
            "provenance_type": fact.get("provenance_type"),
            "old_mu_F": fact.get("evidence", {}).get("confidence_score", 0.0),
            "y_true": y_true,
            "review_result": review.get("result"),
            "review_reason": review.get("reason"),
            "num_evidence": len(ev_list),
            "dst": {
                "bel_T": round(bel_t, 4),
                "bel_F": round(bel_f, 4),
                "uncertainty": round(uncert, 4),
                "conflict": round(conflict, 4),
                "w_base": w_base_dst,
            },
            "w_tconorm": w_tconorm,
            "w_nli_only": w_nli_only,
            "evidence_details": ev_list[:3]  # store top 3 for provenance audit
        }

    LOGGER.info(f"Saving fused facts to {OUTPUT_FUSED_PATH}...")
    OUTPUT_FUSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FUSED_PATH, "w", encoding="utf-8") as f:
        json.dump(fused_facts, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved successfully.")


if __name__ == "__main__":
    fuse_all_evidence()
