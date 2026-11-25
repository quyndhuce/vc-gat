#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fuse_evidence_dst_v2.py
=============================================================================
Stage 3 (V2): Conflict-Aware Evidence Fusion with Yager/Dubois-Prade Redistribution.

Key Principles:
  1. No Hard Zero-Killing: w_base = Bel(T).
  2. Spatial Hierarchy Compatible: Omission/Parent Mention != Contradiction.
  3. Yager / Dubois-Prade Fusion: Conflicting mass K is transferred to Uncertainty Theta.
  4. Ontology Schema Admissibility: Inadmissible triples receive w_base = 0.0.
=============================================================================
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("FuseEvidenceDSTV2")

REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")
DICT_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "fgold_facts_dictionary.json"
TASKS_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "nli_inference_tasks_v2.json"
PREDS_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "nli_predictions_v2.json"
OUTPUT_FUSED_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "fused_facts_dst_v2.json"

STOPWORDS = {"the", "a", "an", "and", "or", "in", "on", "at", "it", "this", "that", "there", ".", ",", "-", ";"}
DUMMY_HEADERS = {
    "characteristic physical features", "visual components", "physical features",
    "named places", "sections", "parts", "exterior", "inside", "performance", "design"
}
FOOD_ENTITIES = {
    "pho", "noodles", "rice noodles", "beef pho", "bun cha", "cha ca", "banh cuon",
    "broth", "fresh herbs", "street-side stools", "bowls of pho", "bun cha grills",
    "traditional southern vietnamese street food"
}

SPATIAL_PARENTS = {
    "ba dinh": {"hanoi", "ha noi", "vietnam", "viet nam"},
    "ba dinh ward": {"ba dinh", "ba dinh district", "hanoi", "ha noi", "vietnam"},
    "ba dinh district": {"hanoi", "ha noi", "vietnam"},
    "hoan kiem": {"hanoi", "ha noi", "vietnam"},
    "hoan kiem district": {"hanoi", "ha noi", "vietnam"},
    "tay ho": {"hanoi", "ha noi", "vietnam"},
    "tay ho district": {"hanoi", "ha noi", "vietnam"},
    "dong da": {"hanoi", "ha noi", "vietnam"},
    "dong da district": {"hanoi", "ha noi", "vietnam"},
    "ha dong": {"hanoi", "ha noi", "vietnam"},
    "ha dong district": {"hanoi", "ha noi", "vietnam"},
    "cho lon": {"ho chi minh city", "ho chi minh", "hcmc", "saigon", "sai gon", "district 5", "district 5, ho chi minh city", "vietnam"},
    "cholon": {"ho chi minh city", "ho chi minh", "hcmc", "saigon", "sai gon", "district 5", "vietnam"},
    "district 1": {"ho chi minh city", "ho chi minh", "hcmc", "saigon", "sai gon", "vietnam"},
    "district 5": {"ho chi minh city", "ho chi minh", "hcmc", "saigon", "sai gon", "vietnam"},
    "hanoi": {"vietnam", "viet nam"},
    "ha noi": {"vietnam", "viet nam"},
    "ho chi minh city": {"vietnam", "viet nam"},
    "saigon": {"vietnam", "viet nam"},
    "thanh hoa": {"vietnam", "viet nam"},
    "hue": {"vietnam", "viet nam"},
    "da nang": {"vietnam", "viet nam"}
}


def is_admissible_triple(s: str, p: str, o: str) -> bool:
    s_c = s.strip().lower()
    o_c = o.strip().lower()
    p_c = p.strip().lower()

    if s_c in STOPWORDS or o_c in STOPWORDS or len(s_c) <= 1 or len(o_c) <= 1:
        return False
    if s_c in DUMMY_HEADERS or o_c in DUMMY_HEADERS:
        return False
    if s_c in FOOD_ENTITIES and p_c in {"part_of", "located_in"} and o_c in {"hanoi", "ha noi", "ho chi minh city", "saigon", "vietnam"}:
        return False
    macro_locs = {"hanoi", "ha noi", "vietnam", "ho chi minh city", "saigon", "cholon"}
    if s_c in macro_locs and p_c in {"part_of", "has_component", "located_inside"}:
        return False
    return True


def yager_combine_all(masses: List[Tuple[float, float, float]]) -> Tuple[float, float, float, float]:
    """
    Yager / Dubois-Prade Combination on frame {T, F}:
    Assigns conflicting mass k directly to Uncertainty Theta.
    """
    if not masses:
        return 0.0, 0.0, 1.0, 0.0
    if len(masses) == 1:
        return masses[0][0], masses[0][1], masses[0][2], 0.0

    curr_t, curr_f, curr_u = masses[0]
    total_conflict = 0.0

    for next_m in masses[1:]:
        t2, f2, u2 = next_m
        k = curr_t * f2 + curr_f * t2
        new_t = curr_t * t2 + curr_t * u2 + curr_u * t2
        new_f = curr_f * f2 + curr_f * u2 + curr_u * f2
        new_u = curr_u * u2 + k  # Transfer conflict to Theta
        curr_t, curr_f, curr_u = new_t, new_f, new_u
        total_conflict += k

    return curr_t, curr_f, curr_u, total_conflict


def fuse_all_evidence_v2():
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
    LOGGER.info(f"Fusing evidence across all {len(facts_dict):,} facts...")

    for fid, fact in facts_dict.items():
        s = (fact.get("subject") or "").strip()
        p = (fact.get("predicate") or "").strip()
        o = (fact.get("object") or "").strip()
        o_l = o.lower()

        is_adm = is_admissible_triple(s, p, o)
        if not is_adm:
            # Inadmissible triples get zero weight
            review = fact.get("review", {})
            y_true = 1 if review.get("result") == "đúng" else 0
            fused_facts[fid] = {
                "fact_id": fid,
                "subject": s, "predicate": p, "object": o,
                "landmark_id": fact.get("landmark_id"),
                "provenance_type": fact.get("provenance_type"),
                "old_mu_F": fact.get("evidence", {}).get("confidence_score", 0.0),
                "y_true": y_true,
                "review_result": review.get("result"),
                "review_reason": review.get("reason"),
                "is_admissible": False,
                "num_evidence": 0,
                "dst": {"bel_T": 0.0, "bel_F": 0.0, "uncertainty": 1.0, "conflict": 0.0, "w_base": 0.0},
                "w_base": 0.0,
                "evidence_details": []
            }
            continue

        ev_list = evidence_by_fact.get(fid, [])

        # Construct mass functions with Spatial Hierarchy handling
        masses = []
        for ev in ev_list:
            r = ev["reliability"]
            pe = ev["p_E"]
            pc = ev["p_C"]
            p_text = ev["premise"].lower()

            # Spatial hierarchy check: parent mention != contradiction
            if p.lower() in {"located_in", "located_inside", "in"}:
                for parent_cand in SPATIAL_PARENTS.get(o_l, set()):
                    if parent_cand in p_text and pc > 0.5:
                        pc = 0.0
                        break

            m_t = r * pe
            m_f = r * pc
            m_u = max(0.0, 1.0 - m_t - m_f)
            masses.append((m_t, m_f, m_u))

        bel_t, bel_f, uncert, conflict = yager_combine_all(masses)

        # Base weight formulation: w_base = Bel(T) (No hard zero-killing!)
        w_base = round(bel_t, 4)

        review = fact.get("review", {})
        y_true = 1 if review.get("result") == "đúng" else 0

        # Strict triple directionality check
        s_l = s.lower()
        p_l = p.lower()
        reason = (review.get("reason") or "").lower()
        if p_l == "part_of" and any(macro in s_l for macro in ["hanoi", "ha noi", "cholon", "saigon", "vietnam"]):
            y_strict = 0
        elif "đảo chiều" in reason:
            y_strict = 0
        else:
            y_strict = y_true

        fused_facts[fid] = {
            "fact_id": fid,
            "subject": s, "predicate": p, "object": o,
            "landmark_id": fact.get("landmark_id"),
            "provenance_type": fact.get("provenance_type"),
            "old_mu_F": fact.get("evidence", {}).get("confidence_score", 0.0),
            "y_true": y_true,
            "y_strict": y_strict,
            "review_result": review.get("result"),
            "review_reason": review.get("reason"),
            "is_admissible": True,
            "num_evidence": len(ev_list),
            "dst": {
                "bel_T": round(bel_t, 4),
                "bel_F": round(bel_f, 4),
                "uncertainty": round(uncert, 4),
                "conflict": round(conflict, 4),
                "w_base": w_base,
            },
            "w_base": w_base,
            "evidence_details": ev_list[:3]
        }

    LOGGER.info(f"Saving fused facts V2 to {OUTPUT_FUSED_PATH}...")
    OUTPUT_FUSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FUSED_PATH, "w", encoding="utf-8") as f:
        json.dump(fused_facts, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved successfully.")


if __name__ == "__main__":
    fuse_all_evidence_v2()
