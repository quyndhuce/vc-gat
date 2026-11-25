#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_progressive_ablation_kb_base.py
=============================================================================
Step-by-Step Ablation on KB_Base (7,088 Facts):
Following User's Exact Scientific Protocol:

  A0: Current M2 (Baseline with hard zero-killing and unconstrained evidence)
  A1: A0 + Remove hard zero-killing rule: w_base = Bel(T)
  A2: A1 + Evidence Relevance Gate (lexical mention of Subject or Object in secondary passages)
  A3: A2 + Spatial Hierarchy Handling (parent region omission != contradiction)
  A4: A3 + Yager / Dubois-Prade Fusion (assign conflict mass K to Uncertainty Theta)
  A5: A4 + Ontology Schema Gate (domain/range constraints, reject syntactic fragments & stopword entities)
  A6: A5 + Strict Triple Correctness (triple_correct vs lenient related_fact_correct)

Computes for each version:
  - AUROC, AUPRC, Brier Score, ECE
  - Precision@0.8, Precision@0.9, Precision@1.0
  - Mean score for TRUE vs FALSE
  - Count of zero scores for TRUE vs FALSE
=============================================================================
"""

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("ProgressiveAblation")

REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")
VERIFIED_PATH = REPO_ROOT / "experiments" / "outputs" / "fkg_facts_verified.json"
OUTPUT_REPORT_PATH = REPO_ROOT / "experiments" / "outputs" / "progressive_kb_base_ablation_report.json"

STOPWORDS = {"the", "a", "an", "and", "or", "in", "on", "at", "it", "this", "that", "there", ".", ",", "-", ";"}
DUMMY_HEADERS = {
    "characteristic physical features", "visual components", "physical features",
    "named places", "sections", "parts", "exterior", "inside", "performance", "design"
}

# Known spatial containment hierarchy
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

FOOD_ENTITIES = {
    "pho", "noodles", "rice noodles", "beef pho", "bun cha", "cha ca", "banh cuon",
    "broth", "fresh herbs", "street-side stools", "bowls of pho", "bun cha grills",
    "traditional southern vietnamese street food"
}


def is_spatial_compatible(child: str, parent: str) -> bool:
    c_clean = child.strip().lower()
    p_clean = parent.strip().lower()
    if c_clean == p_clean:
        return True
    if c_clean in SPATIAL_PARENTS:
        if p_clean in SPATIAL_PARENTS[c_clean]:
            return True
        for intermediate in SPATIAL_PARENTS[c_clean]:
            if intermediate in SPATIAL_PARENTS and p_clean in SPATIAL_PARENTS[intermediate]:
                return True
    return False


def is_ontology_schema_admissible(s: str, p: str, o: str) -> bool:
    s_c = s.strip().lower()
    o_c = o.strip().lower()
    p_c = p.strip().lower()

    # 1. Stopwords / Punctuation
    if s_c in STOPWORDS or o_c in STOPWORDS or len(s_c) <= 1 or len(o_c) <= 1:
        return False

    # 2. Dummy template headers
    if s_c in DUMMY_HEADERS or o_c in DUMMY_HEADERS:
        return False

    # 3. Domain / Range checks
    # Food cannot be part_of a city / geographic region
    if s_c in FOOD_ENTITIES and p_c in {"part_of", "located_in"}:
        if o_c in {"hanoi", "ha noi", "ho chi minh city", "saigon", "vietnam"}:
            return False

    # Macro-location cannot be part_of a single building or park
    macro_locs = {"hanoi", "ha noi", "vietnam", "ho chi minh city", "saigon", "cholon"}
    if s_c in macro_locs and p_c in {"part_of", "has_component", "located_inside"}:
        return False

    return True


def dempster_combine_two(m1, m2):
    t1, f1, u1 = m1
    t2, f2, u2 = m2
    k = t1 * f2 + f1 * t2
    if k >= 0.99999:
        return (0.0, 0.0, 1.0), 1.0
    denom = 1.0 - k
    comb_t = (t1 * t2 + t1 * u2 + u1 * t2) / denom
    comb_f = (f1 * f2 + f1 * u2 + u1 * f2) / denom
    comb_u = (u1 * u2) / denom
    s = comb_t + comb_f + comb_u
    if s > 0:
        comb_t /= s
        comb_f /= s
        comb_u /= s
    return (comb_t, comb_f, comb_u), k


def dempster_combine_all(masses):
    if not masses:
        return 0.0, 0.0, 1.0, 0.0
    if len(masses) == 1:
        return masses[0][0], masses[0][1], masses[0][2], 0.0
    curr = masses[0]
    prod_non_conflict = 1.0
    for nxt in masses[1:]:
        curr, k = dempster_combine_two(curr, nxt)
        prod_non_conflict *= (1.0 - k)
    return curr[0], curr[1], curr[2], (1.0 - prod_non_conflict)


def yager_combine_all(masses):
    """
    Yager / Dubois-Prade Combination on 2-element frame {T, F}:
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
        new_u = curr_u * u2 + k  # Conflict transferred to uncertainty Theta!
        curr_t, curr_f, curr_u = new_t, new_f, new_u
        total_conflict += k

    return curr_t, curr_f, curr_u, total_conflict


def compute_metrics(y_true, scores):
    n = len(y_true)
    n_pos = sum(y_true)
    n_neg = n - n_pos
    if n == 0 or n_pos == 0 or n_neg == 0:
        return {"n": n, "auroc": 0.0, "auprc": 0.0, "brier": 0.0, "ece": 0.0, "sweep": {}}

    # AUROC
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

    # AUPRC
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

    brier = sum((s - y) ** 2 for s, y in zip(scores, y_true)) / n

    # ECE
    ece = 0.0
    n_bins = 10
    bin_width = 1.0 / n_bins
    for b in range(n_bins):
        low = b * bin_width
        high = (b + 1) * bin_width
        in_bin = [(s, y) for s, y in zip(scores, y_true) if (low <= s <= high if b == n_bins - 1 else low <= s < high)]
        if in_bin:
            bin_acc = sum(y for _, y in in_bin) / len(in_bin)
            bin_conf = sum(s for s, _ in in_bin) / len(in_bin)
            ece += (len(in_bin) / n) * abs(bin_acc - bin_conf)

    # Sweep
    sweep = {}
    for tau in [0.5, 0.7, 0.8, 0.9, 1.0]:
        ret = [(s, y) for s, y in zip(scores, y_true) if s >= tau]
        n_ret = len(ret)
        tp = sum(y for _, y in ret)
        prec = (tp / n_ret * 100.0) if n_ret > 0 else 0.0
        rec = (tp / n_pos * 100.0) if n_pos > 0 else 0.0
        sweep[f"tau_{tau}"] = {"ret": n_ret, "prec": round(prec, 2), "rec": round(rec, 2)}

    return {
        "n": n, "n_pos": n_pos, "n_neg": n_neg,
        "auroc": round(auroc, 4), "auprc": round(auprc, 4),
        "brier": round(brier, 4), "ece": round(ece, 4),
        "sweep": sweep
    }


def run_progressive_ablation():
    LOGGER.info(f"Loading verified facts from {VERIFIED_PATH}...")
    with open(VERIFIED_PATH, "r", encoding="utf-8") as f:
        facts = json.load(f)

    kb_base = [f for f in facts.values() if f.get("provenance_type") == "KB_Base"]
    LOGGER.info(f"Loaded {len(kb_base):,} KB_Base facts.")

    y_true_orig = [f["y_true"] for f in kb_base]

    # Pre-extract text fields
    records = []
    for f in kb_base:
        s = f.get("subject", "").strip()
        p = f.get("predicate", "").strip()
        o = f.get("object", "").strip()
        ev_list = f.get("evidence_details", [])
        records.append({
            "fact_id": f["fact_id"],
            "subject": s,
            "predicate": p,
            "object": o,
            "y_true": f["y_true"],
            "review_reason": f.get("review_reason", ""),
            "evidence_details": ev_list,
            "old_mu_F": f.get("old_mu_F", 0.0),
            "m2_w_base": f["dst"]["w_base"],
            "bel_T": f["dst"]["bel_T"],
        })

    results = {}

    # -------------------------------------------------------------
    # A0: Current M2 Baseline
    # -------------------------------------------------------------
    scores_a0 = [r["m2_w_base"] for r in records]
    results["A0_current_m2"] = compute_metrics(y_true_orig, scores_a0)
    results["A0_current_m2"]["name"] = "A0: Current M2 (Hard Zero-Kill on Bel(F) >= Bel(T))"

    # -------------------------------------------------------------
    # A1: A0 + Remove Hard Zero-Kill: w_base = Bel(T)
    # -------------------------------------------------------------
    scores_a1 = [r["bel_T"] for r in records]
    results["A1_bel_t_only"] = compute_metrics(y_true_orig, scores_a1)
    results["A1_bel_t_only"]["name"] = "A1: A0 + w_base = Bel(T) (No Hard Zero-Kill)"

    # -------------------------------------------------------------
    # A2: A1 + Evidence Relevance Gate
    # Only admit secondary evidence if it mentions Subject or Object
    # -------------------------------------------------------------
    scores_a2 = []
    for r in records:
        s_l = r["subject"].lower()
        o_l = r["object"].lower()
        admitted = []
        for idx, ev in enumerate(r["evidence_details"]):
            p_text = ev["premise"].lower()
            if idx == 0 or (s_l in p_text or o_l in p_text):
                rel = ev["reliability"]
                pe = ev["p_E"]
                pc = ev["p_C"]
                admitted.append((rel * pe, rel * pc, max(0.0, 1.0 - rel * (pe + pc))))
        bt, bf, bu, conf = dempster_combine_all(admitted)
        scores_a2.append(bt)
    results["A2_relevance_gate"] = compute_metrics(y_true_orig, scores_a2)
    results["A2_relevance_gate"]["name"] = "A2: A1 + Evidence Relevance Gate (Lexical Filter)"

    # -------------------------------------------------------------
    # A3: A2 + Spatial Hierarchy Handling
    # If premise mentions parent region, clamp spurious contradiction to 0
    # -------------------------------------------------------------
    scores_a3 = []
    for r in records:
        s_l = r["subject"].lower()
        o_l = r["object"].lower()
        admitted = []
        for idx, ev in enumerate(r["evidence_details"]):
            p_text = ev["premise"].lower()
            if idx == 0 or (s_l in p_text or o_l in p_text):
                rel = ev["reliability"]
                pe = ev["p_E"]
                pc = ev["p_C"]

                # Spatial hierarchy check
                if r["predicate"] in {"located_in", "located_inside", "in"}:
                    for parent_cand in SPATIAL_PARENTS.get(o_l, set()):
                        if parent_cand in p_text and pc > 0.5:
                            # Spurious contradiction due to parent mention!
                            pc = 0.0
                            break

                admitted.append((rel * pe, rel * pc, max(0.0, 1.0 - rel * (pe + pc))))
        bt, bf, bu, conf = dempster_combine_all(admitted)
        scores_a3.append(bt)
    results["A3_spatial_hierarchy"] = compute_metrics(y_true_orig, scores_a3)
    results["A3_spatial_hierarchy"]["name"] = "A3: A2 + Spatial Hierarchy Conflict Dampening"

    # -------------------------------------------------------------
    # A4: A3 + Yager / Dubois-Prade Fusion
    # Assign conflict mass K to Uncertainty Theta instead of normalizing away
    # -------------------------------------------------------------
    scores_a4 = []
    for r in records:
        s_l = r["subject"].lower()
        o_l = r["object"].lower()
        admitted = []
        for idx, ev in enumerate(r["evidence_details"]):
            p_text = ev["premise"].lower()
            if idx == 0 or (s_l in p_text or o_l in p_text):
                rel = ev["reliability"]
                pe = ev["p_E"]
                pc = ev["p_C"]

                if r["predicate"] in {"located_in", "located_inside", "in"}:
                    for parent_cand in SPATIAL_PARENTS.get(o_l, set()):
                        if parent_cand in p_text and pc > 0.5:
                            pc = 0.0
                            break

                admitted.append((rel * pe, rel * pc, max(0.0, 1.0 - rel * (pe + pc))))
        bt, bf, bu, conf = yager_combine_all(admitted)
        scores_a4.append(bt)
    results["A4_yager_fusion"] = compute_metrics(y_true_orig, scores_a4)
    results["A4_yager_fusion"]["name"] = "A4: A3 + Yager / Dubois-Prade Conflict Redistribution"

    # -------------------------------------------------------------
    # A5: A4 + Ontology Schema Gate (Filter out invalid triples)
    # -------------------------------------------------------------
    adm_indices = [
        i for i, r in enumerate(records)
        if is_ontology_schema_admissible(r["subject"], r["predicate"], r["object"])
    ]
    y_true_adm = [y_true_orig[i] for i in adm_indices]
    scores_a5 = [scores_a4[i] for i in adm_indices]
    results["A5_ontology_schema_gate"] = compute_metrics(y_true_adm, scores_a5)
    results["A5_ontology_schema_gate"]["name"] = f"A5: A4 + Ontology Schema Gate (N={len(adm_indices):,})"

    # -------------------------------------------------------------
    # A6: A5 + Strict Triple Directionality (triple_correct)
    # If a macro-region is stated as part_of a building, triple is strictly false
    # -------------------------------------------------------------
    y_true_strict = []
    for i in adm_indices:
        r = records[i]
        orig_y = r["y_true"]
        s_l = r["subject"].lower()
        p_l = r["predicate"].lower()
        o_l = r["object"].lower()
        reason = r["review_reason"].lower()

        # Inverted part_of
        if p_l == "part_of" and any(macro in s_l for macro in ["hanoi", "ha noi", "cholon", "saigon", "vietnam"]):
            strict_y = 0
        elif "đảo chiều" in reason:
            strict_y = 0
        else:
            strict_y = orig_y
        y_true_strict.append(strict_y)

    results["A6_strict_triple_correctness"] = compute_metrics(y_true_strict, scores_a5)
    results["A6_strict_triple_correctness"]["name"] = "A6: A5 + Strict Triple Correctness (No Lenient Ground Truth)"

    # Print summary table
    print("\n" + "=" * 85)
    print("PROGRESSIVE ABLATION BENCHMARK ON KB_BASE FACTS")
    print("=" * 85)
    print(f"{'Version':<30} | {'AUROC':<7} | {'AUPRC':<7} | {'P@0.8':<7} | {'P@0.9':<7} | {'ECE':<7} | {'Brier':<7}")
    print("-" * 85)
    for v_key in ["A0_current_m2", "A1_bel_t_only", "A2_relevance_gate", "A3_spatial_hierarchy", "A4_yager_fusion", "A5_ontology_schema_gate", "A6_strict_triple_correctness"]:
        res = results[v_key]
        p08 = res["sweep"].get("tau_0.8", {}).get("prec", 0.0)
        p09 = res["sweep"].get("tau_0.9", {}).get("prec", 0.0)
        print(f"{v_key:<30} | {res['auroc']:<7.4f} | {res['auprc']:<7.4f} | {p08:<7.2f} | {p09:<7.2f} | {res['ece']:<7.4f} | {res['brier']:<7.4f}")
    print("=" * 85 + "\n")

    # Save to JSON
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Report saved to {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    run_progressive_ablation()
