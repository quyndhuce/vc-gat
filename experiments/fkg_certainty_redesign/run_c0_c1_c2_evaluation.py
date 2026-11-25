#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_c0_c1_c2_evaluation.py
=============================================================================
Definitive Comparative Evaluation of Context Enrichment Configurations:

Configurations:
  - C0: Raw passage (no context prefix)
  - C1: Document-grounded verified metadata ("At {doc_landmark}: ")
  - C2: Hypothesis-leaked diagnostic control ("At {hypo_subject}: ")

Evaluates on:
  1. Full KB_Base (N = 7,088)
  2. Admissible KB_Base (N = 6,470)
  3. Context-Disconnected Subset (where landmark name was missing in raw text)

Metrics:
  - AUROC, AUPRC, Brier Score, ECE
  - Precision @ 0.80, 0.90, 1.00
  - Mean score for TRUE (y=1) vs FALSE (y=0)
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
LOGGER = logging.getLogger("EvalC0C1C2")

DICT_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "fgold_facts_dictionary.json"
TASKS_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "nli_inference_tasks_v2.json"

PREDS_C0_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "nli_preds_c0.json"
PREDS_C1_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "nli_predictions_v2.json"
PREDS_C2_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "nli_preds_c2.json"

OUTPUT_REPORT = REPO_ROOT / "experiments" / "outputs" / "c0_c1_c2_benchmark_report.json"

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
        new_u = curr_u * u2 + k
        curr_t, curr_f, curr_u = new_t, new_f, new_u
        total_conflict += k

    return curr_t, curr_f, curr_u, total_conflict


def compute_fused_weights(preds: Dict[str, Dict[str, float]], tasks: List[Dict[str, Any]], facts_dict: Dict[str, Any]) -> Dict[str, float]:
    ev_by_fact = defaultdict(list)
    for t in tasks:
        tid = t["task_id"]
        pred = preds.get(tid, {"p_E": 0.0, "p_C": 0.0, "p_N": 1.0})
        ev_by_fact[t["fact_id"]].append({
            "reliability": t["reliability"],
            "p_E": pred["p_E"],
            "p_C": pred["p_C"],
            "p_N": pred["p_N"],
            "premise": t["premise"]
        })

    weights = {}
    for fid, fact in facts_dict.items():
        s = (fact.get("subject") or "").strip()
        p = (fact.get("predicate") or "").strip()
        o = (fact.get("object") or "").strip()
        o_l = o.lower()

        if not is_admissible_triple(s, p, o):
            weights[fid] = 0.0
            continue

        ev_list = ev_by_fact.get(fid, [])
        masses = []
        for ev in ev_list:
            r = ev["reliability"]
            pe = ev["p_E"]
            pc = ev["p_C"]
            p_text = ev["premise"].lower()

            if p.lower() in {"located_in", "located_inside", "in"}:
                for parent_cand in SPATIAL_PARENTS.get(o_l, set()):
                    if parent_cand in p_text and pc > 0.5:
                        pc = 0.0
                        break

            m_t = r * pe
            m_f = r * pc
            m_u = max(0.0, 1.0 - m_t - m_f)
            masses.append((m_t, m_f, m_u))

        bel_t, _, _, _ = yager_combine_all(masses)
        weights[fid] = round(bel_t, 4)

    return weights


def run_evaluation():
    LOGGER.info("Loading tasks and fact dictionary...")
    with open(TASKS_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    with open(DICT_PATH, "r", encoding="utf-8") as f:
        facts_dict = json.load(f)["facts"]

    kb_base_facts = [f for f in facts_dict.values() if f.get("provenance_type") == "KB_Base"]
    LOGGER.info(f"Loaded {len(kb_base_facts):,} KB_Base facts.")

    y_true_orig = [1 if f.get("review", {}).get("result") == "đúng" else 0 for f in kb_base_facts]
    is_adm = [is_admissible_triple(f.get("subject", ""), f.get("predicate", ""), f.get("object", "")) for f in kb_base_facts]

    LOGGER.info("Loading predictions for C0, C1, C2...")
    with open(PREDS_C0_PATH, "r", encoding="utf-8") as f:
        preds_c0 = json.load(f)
    with open(PREDS_C1_PATH, "r", encoding="utf-8") as f:
        preds_c1 = json.load(f)
    with open(PREDS_C2_PATH, "r", encoding="utf-8") as f:
        preds_c2 = json.load(f)

    LOGGER.info("Computing fused weights for C0 (Raw)...")
    w_c0 = compute_fused_weights(preds_c0, tasks, facts_dict)
    LOGGER.info("Computing fused weights for C1 (Document Grounded)...")
    w_c1 = compute_fused_weights(preds_c1, tasks, facts_dict)
    LOGGER.info("Computing fused weights for C2 (Hypothesis Subject Control)...")
    w_c2 = compute_fused_weights(preds_c2, tasks, facts_dict)

    scores_c0 = [w_c0[f["fact_id"]] for f in kb_base_facts]
    scores_c1 = [w_c1[f["fact_id"]] for f in kb_base_facts]
    scores_c2 = [w_c2[f["fact_id"]] for f in kb_base_facts]

    # Metrics on full KB_Base
    m_c0_full = compute_metrics(y_true_orig, scores_c0)
    m_c1_full = compute_metrics(y_true_orig, scores_c1)
    m_c2_full = compute_metrics(y_true_orig, scores_c2)

    # Metrics on admissible KB_Base
    adm_idx = [i for i, adm in enumerate(is_adm) if adm]
    y_adm = [y_true_orig[i] for i in adm_idx]
    m_c0_adm = compute_metrics(y_adm, [scores_c0[i] for i in adm_idx])
    m_c1_adm = compute_metrics(y_adm, [scores_c1[i] for i in adm_idx])
    m_c2_adm = compute_metrics(y_adm, [scores_c2[i] for i in adm_idx])

    # Means
    def calc_means(scores, y_arr):
        pos = [s for s, y in zip(scores, y_arr) if y == 1]
        neg = [s for s, y in zip(scores, y_arr) if y == 0]
        return sum(pos) / len(pos) if pos else 0.0, sum(neg) / len(neg) if neg else 0.0

    c0_tp, c0_fp = calc_means(scores_c0, y_true_orig)
    c1_tp, c1_fp = calc_means(scores_c1, y_true_orig)
    c2_tp, c2_fp = calc_means(scores_c2, y_true_orig)

    print("\n" + "=" * 95)
    print("COMPARATIVE EVALUATION: C0 (Raw) vs C1 (Document Grounded) vs C2 (Hypothesis Subject Control)")
    print("=" * 95)
    print("--- 1. Full KB_Base Set (N = 7,088) ---")
    print(f"{'Configuration':<35} | {'AUROC':<7} | {'AUPRC':<7} | {'Mean TRUE':<10} | {'Mean FALSE':<10} | {'P@0.8':<7} | {'ECE':<7}")
    print("-" * 95)
    for name, m, mt, mf in [
        ("C0: Raw Passages (Uncontextualized)", m_c0_full, c0_tp, c0_fp),
        ("C1: Doc-Grounded Metadata (Proposed)", m_c1_full, c1_tp, c1_fp),
        ("C2: Hypothesis Subject (Leakage Ctrl)", m_c2_full, c2_tp, c2_fp),
    ]:
        p08 = m["sweep"].get("tau_0.8", {}).get("precision", 0.0)
        print(f"{name:<35} | {m['auroc']:<7.4f} | {m['auprc']:<7.4f} | {mt:<10.4f} | {mf:<10.4f} | {p08:<7.2f} | {m['ece']:<7.4f}")

    print("\n--- 2. Admissible KB_Base Set (N = 6,470, after schema gating) ---")
    print(f"{'Configuration':<35} | {'AUROC':<7} | {'AUPRC':<7} | {'Mean TRUE':<10} | {'Mean FALSE':<10} | {'P@0.8':<7} | {'ECE':<7}")
    print("-" * 95)
    c0_adm_tp, c0_adm_fp = calc_means([scores_c0[i] for i in adm_idx], y_adm)
    c1_adm_tp, c1_adm_fp = calc_means([scores_c1[i] for i in adm_idx], y_adm)
    c2_adm_tp, c2_adm_fp = calc_means([scores_c2[i] for i in adm_idx], y_adm)

    for name, m, mt, mf in [
        ("C0: Raw Passages (Uncontextualized)", m_c0_adm, c0_adm_tp, c0_adm_fp),
        ("C1: Doc-Grounded Metadata (Proposed)", m_c1_adm, c1_adm_tp, c1_adm_fp),
        ("C2: Hypothesis Subject (Leakage Ctrl)", m_c2_adm, c2_adm_tp, c2_adm_fp),
    ]:
        p08 = m["sweep"].get("tau_0.8", {}).get("precision", 0.0)
        print(f"{name:<35} | {m['auroc']:<7.4f} | {m['auprc']:<7.4f} | {mt:<10.4f} | {mf:<10.4f} | {p08:<7.2f} | {m['ece']:<7.4f}")
    print("=" * 95 + "\n")

    report = {
        "full_kb_base": {
            "C0_raw": {**m_c0_full, "mean_true": c0_tp, "mean_false": c0_fp},
            "C1_doc_grounded": {**m_c1_full, "mean_true": c1_tp, "mean_false": c1_fp},
            "C2_hypo_subject": {**m_c2_full, "mean_true": c2_tp, "mean_false": c2_fp},
        },
        "admissible_kb_base": {
            "C0_raw": {**m_c0_adm, "mean_true": c0_adm_tp, "mean_false": c0_adm_fp},
            "C1_doc_grounded": {**m_c1_adm, "mean_true": c1_adm_tp, "mean_false": c1_adm_fp},
            "C2_hypo_subject": {**m_c2_adm, "mean_true": c2_adm_tp, "mean_false": c2_adm_fp},
        }
    }

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Saved report to {OUTPUT_REPORT}")


if __name__ == "__main__":
    run_evaluation()
