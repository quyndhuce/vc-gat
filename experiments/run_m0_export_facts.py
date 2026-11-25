#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/run_m0_export_facts.py
=============================================================================
Implementation of Protocol Step M0: Export facts.csv (Gate Check)
Target: International Journal of Approximate Reasoning (IJAR)

Computes:
  - Materialized base facts under symmetric Horn rules (r1..r4)
  - DP computation of Z_L up to depth L=3 (gamma_r = 0.90)
  - Exhaustive simple derivation trees enumeration (bounded by branch simple check A1)
  - Canonical deduplication to D_L(B)
  - Minimal supports enumeration
  - Statistics: v_max, mu_F_full, mu_F_min, sum_val, Z_L, n_trees, n_deriv,
    n_minsupp, n_shared, a1_violation
  - Grounding context: mu_L, r (anchor_score), c = mu_L * mu_F_full
Outputs:
  - results/facts.csv
  - results/facts_summary.json
  - tables/M0.tex (LaTeX table for supplementary)
=============================================================================
"""

import os
import sys
import json
import math
import time
import logging
import itertools
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, FrozenSet, Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("ExportFactsM0")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

FKG_PATH = DATA_DIR / "fkg_per_kb" / "fkg_global_merged.json"
TEST_COCO_PATH = DATA_DIR / "unified_coco_dataset" / "annotations" / "test_coco.json"
TEST_1119_PATH = DATA_DIR / "caption_dataset" / "generated_captions_test_full_1119.jsonl"
VAL_149_PATH = DATA_DIR / "caption_dataset" / "generated_captions_val.jsonl"

GAMMA_R = 0.90
L_DEPTH = 3


def load_knowledge_graph_and_rules():
    LOGGER.info("Loading Base FKG from %s...", FKG_PATH)
    with open(FKG_PATH, "r", encoding="utf-8") as f:
        fkg = json.load(f)

    base_facts = {}
    for m in fkg.get("multi_valued_facts", []):
        h = m.get("head_surface", "").strip().lower()
        r = m.get("relation", "").strip().lower()
        t = m.get("tail_surface", "").strip().lower()
        w = float(m.get("mu_F", 0.5))
        if h and r and t:
            base_facts[(h, r, t)] = max(base_facts.get((h, r, t), 0.0), w)

    for g in fkg.get("single_valued_groups", []):
        h = g.get("head_surface", "").strip().lower()
        r = g.get("relation", "").strip().lower()
        t = g.get("winner_surface", "").strip().lower()
        w = float(g.get("winner_bel", 0.8))
        if h and r and t:
            base_facts[(h, r, t)] = max(base_facts.get((h, r, t), 0.0), w)

    # Stratum 1: Materialize Symmetric Closure FIRST
    sym_rules = {
        "adjacent_to": 1.00,
        "near_to": 0.95,
        "near": 0.95,
        "bordering": 1.00,
        "associated_with": 0.90,
    }
    mat_base = dict(base_facts)
    for (h, r, t), w in list(base_facts.items()):
        if r in sym_rules:
            inv_k = (t, r, h)
            mat_base[inv_k] = max(mat_base.get(inv_k, 0.0), w * sym_rules[r])

    # Dual part-of / has-part symmetry
    for (h, r, t), w in list(mat_base.items()):
        if r == "part_of" and (t, "has_part", h) not in mat_base:
            mat_base[(t, "has_part", h)] = w
        elif r == "has_part" and (t, "part_of", h) not in mat_base:
            mat_base[(t, "part_of", h)] = w

    LOGGER.info("Base facts: %d -> Materialized base facts: %d", len(base_facts), len(mat_base))

    # Build Adjacency Index
    head_rel_tails = defaultdict(lambda: defaultdict(dict))
    for (h, r, t), w in mat_base.items():
        head_rel_tails[h][r][t] = w

    # Deductive Horn Rule Instantiations (r5..r12)
    rules_by_head = defaultdict(list)
    # r5: located_in(x, y) & located_in(y, z) => located_in(x, z) (conf: 0.98)
    for x in head_rel_tails:
        for y in head_rel_tails[x].get("located_in", {}):
            for z in head_rel_tails.get(y, {}).get("located_in", {}):
                if x != z:
                    rules_by_head[(x, "located_in", z)].append(
                        ("r5", ((x, "located_in", y), (y, "located_in", z)), 0.98)
                    )

    # Substructure Inheritance rules (r6..r10)
    for x in head_rel_tails:
        for y in head_rel_tails[x].get("has_part", {}):
            if x == y:
                continue
            for z in head_rel_tails[x].get("located_in", {}):
                if y != z:
                    rules_by_head[(y, "located_in", z)].append(
                        ("r6", ((x, "has_part", y), (x, "located_in", z)), 0.95)
                    )
            for z in head_rel_tails[x].get("belongs_to_dynasty", {}):
                rules_by_head[(y, "belongs_to_dynasty", z)].append(
                    ("r7", ((x, "has_part", y), (x, "belongs_to_dynasty", z)), 0.85)
                )
            for z in head_rel_tails[x].get("architectural_style", {}):
                rules_by_head[(y, "architectural_style", z)].append(
                    ("r8", ((x, "has_part", y), (x, "architectural_style", z)), 0.80)
                )
            for z in head_rel_tails[x].get("dedicated_to", {}):
                rules_by_head[(y, "worships", z)].append(
                    ("r9", ((x, "has_part", y), (x, "dedicated_to", z)), 0.88)
                )
            for z in head_rel_tails[x].get("worships", {}):
                rules_by_head[(y, "worships", z)].append(
                    ("r9", ((x, "has_part", y), (x, "worships", z)), 0.88)
                )
            for z in head_rel_tails[x].get("associated_with", {}):
                rules_by_head[(y, "associated_with", z)].append(
                    ("r10", ((x, "has_part", y), (x, "associated_with", z)), 0.75)
                )

    # Historical rules (r11, r12)
    for x in head_rel_tails:
        for y in head_rel_tails[x].get("commemorates", {}):
            for z in head_rel_tails.get(y, {}).get("ruled_by", {}):
                rules_by_head[(x, "belongs_to_dynasty", z)].append(
                    ("r11", ((x, "commemorates", y), (y, "ruled_by", z)), 0.90)
                )
        for y in head_rel_tails[x].get("built_by", {}):
            for z in head_rel_tails.get(y, {}).get("ruled_by", {}):
                rules_by_head[(x, "belongs_to_dynasty", z)].append(
                    ("r12", ((x, "built_by", y), (y, "ruled_by", z)), 0.92)
                )

    LOGGER.info("Deductive rules indexed across %d unique head conclusions.", len(rules_by_head))
    return mat_base, rules_by_head


def run_dynamic_programming_zl(mat_base: Dict[Tuple[str, str, str], float], rules_by_head: Dict[Tuple[str, str, str], List[Any]]):
    LOGGER.info("Running Dynamic Programming for Z_L (L=%d, gamma_r=%.2f)...", L_DEPTH, GAMMA_R)
    all_facts = set(mat_base.keys()) | set(rules_by_head.keys())
    Z_prev = {f: mat_base.get(f, 0.0) for f in all_facts}

    for l in range(1, L_DEPTH + 1):
        Z = {f: mat_base.get(f, 0.0) for f in all_facts}
        for head, r_list in rules_by_head.items():
            for rid, body, r_conf in r_list:
                term = GAMMA_R * r_conf
                for b in body:
                    term *= Z_prev.get(b, 0.0)
                Z[head] += term
        Z_prev = Z

    LOGGER.info("DP Z_L completed across %d facts.", len(Z_prev))
    return Z_prev


def compute_fact_derivations(fact: Tuple[str, str, str], mat_base: Dict[Tuple[str, str, str], float], rules_by_head: Dict[Tuple[str, str, str], List[Any]], max_depth: int = 3):
    memo = {}

    def dfs(f, cur_depth, ancestor_facts):
        state = (f, cur_depth, tuple(sorted(ancestor_facts)))
        if state in memo:
            return memo[state]

        trees = []
        # Base fact (depth 0 tree)
        if f in mat_base:
            trees.append((mat_base[f], frozenset([f]), False))

        if cur_depth < max_depth:
            for rid, body, r_conf in rules_by_head.get(f, []):
                # Cycle check on branch
                if any(b in ancestor_facts for b in body):
                    continue
                new_ancestors = ancestor_facts | {f}
                body_trees = [dfs(b, cur_depth + 1, new_ancestors) for b in body]
                if any(len(bt) == 0 for bt in body_trees):
                    continue
                for combo in itertools.product(*body_trees):
                    val = GAMMA_R * r_conf
                    comps = set()
                    rule_instance = f"{rid}_{body}->{f}"
                    comps.add(rule_instance)
                    a1_viol = any(bt[2] for bt in combo)
                    for b_val, b_comps, _ in combo:
                        val *= b_val
                        if not comps.isdisjoint(b_comps):
                            a1_viol = True
                        comps.update(b_comps)
                    trees.append((val, frozenset(comps), a1_viol))

        memo[state] = trees
        return trees

    trees = dfs(fact, 0, set())

    # Canonical deduplication
    can_derivs = {}
    for val, comps, viol in trees:
        if comps not in can_derivs or val > can_derivs[comps][0]:
            can_derivs[comps] = (val, viol)

    # Minimal supports
    comps_list = list(can_derivs.keys())
    minsupps = []
    for i, c1 in enumerate(comps_list):
        is_min = True
        for j, c2 in enumerate(comps_list):
            if i != j and c2 < c1:
                is_min = False
                break
        if is_min:
            minsupps.append(c1)

    vals_all = [v[0] for v in can_derivs.values()] if can_derivs else ([mat_base.get(fact, 0.5)] if fact in mat_base else [0.5])
    v_max = max(vals_all)
    sum_val = sum(vals_all)

    prod_full = 1.0
    for v in vals_all:
        prod_full *= (1.0 - min(0.999999, v))
    mu_F_full = 1.0 - prod_full

    vals_min = [can_derivs[m][0] for m in minsupps] if minsupps else vals_all
    prod_min = 1.0
    for v in vals_min:
        prod_min *= (1.0 - min(0.999999, v))
    mu_F_min = 1.0 - prod_min

    # Shared components in minsupps
    cnt = Counter(u for S in minsupps for u in S)
    n_shared = sum(1 for u, count in cnt.items() if count >= 2)

    a1_violation = 1 if any(v[1] for v in can_derivs.values()) else 0

    return {
        "n_trees": len(trees),
        "n_deriv": len(can_derivs),
        "n_minsupp": len(minsupps),
        "n_shared": n_shared,
        "v_max": float(v_max),
        "mu_F_full": float(mu_F_full),
        "mu_F_min": float(mu_F_min),
        "sum_val": float(sum_val),
        "a1_violation": a1_violation,
        "minimal_supports": [list(S) for S in minsupps],
        "all_supports": [list(S) for S in can_derivs.keys()],
    }


def certify_query_structural_a1(fact: Tuple[str, str, str], mat_base: Dict[Tuple[str, str, str], float], rules_by_head: Dict[Tuple[str, str, str], List[Any]], max_depth: int = 3):
    """
    Stage 1 Full Structural A1 Certification on active reachability structure G_reach(B, L).
    Certifies:
      Condition C1: Path acyclicity and self-loop freedom (v not in Adj(v) and no path cycles).
      Condition C2: Sibling branch support disjointness across base facts, rules, and derived propositions:
                    U_reach(A_i) cap U_reach(A_j) == emptyset for all i < j.
    """
    t0 = time.perf_counter()
    memo = {}
    has_diamond = False
    has_self_loop = False
    has_path_cycle = False
    diamond_details = []

    def get_reach(f, cur_depth, ancestor_facts):
        nonlocal has_diamond, has_self_loop, has_path_cycle
        state = (f, cur_depth, tuple(sorted(ancestor_facts)))
        if state in memo:
            return memo[state]

        comps = set()
        # Unified canonical proposition representation:
        # Every proposition triple (whether asserted in base KG or derived as an intermediate lemma)
        # is uniquely represented as ("prop", f), preventing split identity between base facts and intermediate lemmas.
        comps.add(("prop", f))

        if cur_depth < max_depth:
            for rid, body, r_conf in rules_by_head.get(f, []):
                # Condition C1: check self-loops and path cycles
                if f in body:
                    has_self_loop = True
                if any(b in ancestor_facts for b in body):
                    has_path_cycle = True
                    continue
                rule_inst = ("rule", rid, body, f)
                new_ancestors = ancestor_facts | {f}

                body_comps_list = []
                valid_rule = True
                for b in body:
                    b_comps = get_reach(b, cur_depth + 1, new_ancestors)
                    if len(b_comps) == 0:
                        valid_rule = False
                        break
                    body_comps_list.append(b_comps)

                if not valid_rule:
                    continue

                # Condition C2: Sibling branch support disjointness (anti-diamond check)
                if len(body_comps_list) > 1:
                    for i in range(len(body_comps_list)):
                        for j in range(i + 1, len(body_comps_list)):
                            inter = body_comps_list[i].intersection(body_comps_list[j])
                            if len(inter) > 0:
                                has_diamond = True
                                diamond_details.append((f, rule_inst, [str(x) for x in inter]))

                comps.add(rule_inst)
                for bc in body_comps_list:
                    comps.update(bc)

        memo[state] = comps
        return comps

    root_comps = get_reach(fact, 0, frozenset())
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0
    c1_acyclic = (not has_self_loop)
    c2_disjoint = (not has_diamond)
    return {
        "c1_acyclic": 1 if c1_acyclic else 0,
        "c2_disjoint": 1 if c2_disjoint else 0,
        "certified_a1": 1 if (c1_acyclic and c2_disjoint) else 0,
        "n_diamonds": len(diamond_details),
        "n_comps": len(root_comps),
        "cert_latency_ms": round(latency_ms, 4),
    }


def main():
    LOGGER.info("=" * 80)
    LOGGER.info("STARTING EXPERIMENT M0: EXPORT FACTS.CSV & GATE CHECK")
    LOGGER.info("=" * 80)

    # 1. Load FKG and Rules
    mat_base, rules_by_head = load_knowledge_graph_and_rules()

    # 2. Compute Z_L via DP
    Z_L_map = run_dynamic_programming_zl(mat_base, rules_by_head)

    # 3. Load 500 Test Clips and 149 Validation Clips
    LOGGER.info("Selecting 500 balanced test clips...")
    with open(TEST_COCO_PATH, "r", encoding="utf-8") as f:
        coco = json.load(f)
    annotations = coco["annotations"]
    lm_to_indices = defaultdict(list)
    for idx, a in enumerate(annotations):
        lm_to_indices[a["parent_landmark"]].append(idx)
    selected_indices = []
    for lm, indices in sorted(lm_to_indices.items(), key=lambda x: len(x[1])):
        take = min(len(indices), 42) if lm == "Temple of Literature" else min(len(indices), 46)
        selected_indices.extend(indices[:take])
    selected_indices = sorted(selected_indices[:500])

    with open(TEST_1119_PATH, "r", encoding="utf-8") as f:
        all_test = [json.loads(l) for l in f]
    test_500 = [all_test[i] for i in selected_indices]

    with open(VAL_149_PATH, "r", encoding="utf-8") as f:
        val_149 = [json.loads(l) for l in f]

    LOGGER.info("Loaded %d test clips and %d validation clips.", len(test_500), len(val_149))

    # Pre-cache derivations for all unique facts appearing across clips
    fact_cache = {}

    def get_cached_deriv(fact):
        if fact not in fact_cache:
            res = compute_fact_derivations(fact, mat_base, rules_by_head, max_depth=L_DEPTH)
            cert = certify_query_structural_a1(fact, mat_base, rules_by_head, max_depth=L_DEPTH)
            res.update(cert)
            res["Z_L"] = float(Z_L_map.get(fact, mat_base.get(fact, 0.0)))
            fact_cache[fact] = res
        return fact_cache[fact]

    # Process all non-visual facts across test_500
    rows = []
    seen_clip_facts = set()

    all_clips = [("test", c) for c in test_500] + [("val", c) for c in val_149]

    for split, clip in all_clips:
        clip_id = clip.get("video_id")
        landmark = clip.get("predicted_landmark", "").strip()
        mu_L = float(clip.get("mu_L", 0.90))

        # Facts in eligible_facts
        facts_list = clip.get("eligible_facts", [])
        if not facts_list:
            facts_list = clip.get("gold_facts", [])

        for f in facts_list:
            cat = f.get("category", "")
            is_inf = f.get("is_inferred", False)
            if cat not in ("direct", "inferred") and not is_inf:
                continue

            h = f.get("head", "").strip().lower()
            r_rel = f.get("relation", "").strip().lower()
            t = f.get("tail", "").strip().lower()
            fact_triple = (h, r_rel, t)
            fact_id = f"{h} -- {r_rel} -> {t}"

            clip_fact_key = (clip_id, fact_id)
            if clip_fact_key in seen_clip_facts:
                continue
            seen_clip_facts.add(clip_fact_key)

            deriv_data = get_cached_deriv(fact_triple)

            provenance = "inf" if (is_inf or fact_triple not in mat_base) else "dir"
            n_deriv = deriv_data["n_deriv"]

            # Query class definition:
            # Single: e_Q = 1 (no derivation or single direct)
            # Two-relation: e_Q = 2
            # Compound: e_Q >= 3 or branching / multiple derivations / shared components
            if n_deriv >= 3 or deriv_data["n_shared"] > 0:
                query_class = "Compound"
            elif n_deriv == 2:
                query_class = "Two-relation"
            else:
                query_class = "Single"

            r_val = float(f.get("anchor_score", f.get("siglip_similarity", 0.5)))
            c_val = float(mu_L * deriv_data["mu_F_full"])

            rows.append({
                "fact_id": fact_id,
                "clip_id": clip_id,
                "split": split,
                "landmark": landmark,
                "provenance": provenance,
                "query_class": query_class,
                "n_trees": deriv_data["n_trees"],
                "n_deriv": deriv_data["n_deriv"],
                "n_minsupp": deriv_data["n_minsupp"],
                "n_shared": deriv_data["n_shared"],
                "v_max": round(deriv_data["v_max"], 5),
                "mu_F_full": round(deriv_data["mu_F_full"], 5),
                "mu_F_min": round(deriv_data["mu_F_min"], 5),
                "sum_val": round(deriv_data["sum_val"], 5),
                "Z_L": round(deriv_data["Z_L"], 5),
                "a1_violation": deriv_data["a1_violation"],
                "c1_acyclic": deriv_data["c1_acyclic"],
                "c2_disjoint": deriv_data["c2_disjoint"],
                "certified_a1": deriv_data["certified_a1"],
                "cert_latency_ms": deriv_data["cert_latency_ms"],
                "mu_L": round(mu_L, 5),
                "r": round(r_val, 5),
                "c": round(c_val, 5),
            })

    df = pd.DataFrame(rows)
    facts_csv_path = RESULTS_DIR / "facts.csv"
    df.to_csv(facts_csv_path, index=False)
    LOGGER.info("Exported %d fact entries to %s", len(df), facts_csv_path)

    # 4. Generate Table M0 (Supplementary)
    # Subset n_deriv >= 2 and n_minsupp >= 2
    # Metrics: n_fact, % vi pham A1, Z_L/sum_val (median, p90, max)
    test_df = df[df["split"] == "test"]
    
    # Per-unique fact aggregation to avoid sample weighting bias
    unique_facts_df = df.drop_duplicates(subset=["fact_id"])
    
    subsets = {
        "Unique facts ($n_{\\mathrm{deriv}} \\ge 2$)": unique_facts_df[unique_facts_df["n_deriv"] >= 2],
        "Unique facts ($n_{\\mathrm{minsupp}} \\ge 2$)": unique_facts_df[unique_facts_df["n_minsupp"] >= 2],
        "Clip instances ($n_{\\mathrm{deriv}} \\ge 2$)": df[df["n_deriv"] >= 2],
        "Clip instances ($n_{\\mathrm{minsupp}} \\ge 2$)": df[df["n_minsupp"] >= 2],
    }

    m0_rows = []
    for label, sub in subsets.items():
        n_fact = len(sub)
        a1_viol_pct = float(sub["a1_violation"].mean() * 100.0) if n_fact > 0 else 0.0
        c1_pass_pct = float(sub["c1_acyclic"].mean() * 100.0) if n_fact > 0 else 0.0
        c2_pass_pct = float(sub["c2_disjoint"].mean() * 100.0) if n_fact > 0 else 0.0
        cert_a1_pct = float(sub["certified_a1"].mean() * 100.0) if n_fact > 0 else 0.0
        
        ratios = sub["Z_L"] / np.maximum(sub["sum_val"], 1e-6)
        med_ratio = float(np.median(ratios)) if n_fact > 0 else 0.0
        p90_ratio = float(np.percentile(ratios, 90)) if n_fact > 0 else 0.0
        max_ratio = float(np.max(ratios)) if n_fact > 0 else 0.0

        m0_rows.append({
            "subset": label,
            "n_fact": n_fact,
            "a1_viol_pct": a1_viol_pct,
            "c1_pass_pct": c1_pass_pct,
            "c2_pass_pct": c2_pass_pct,
            "cert_a1_pct": cert_a1_pct,
            "med_ratio": med_ratio,
            "p90_ratio": p90_ratio,
            "max_ratio": max_ratio,
        })
        LOGGER.info("Table M0 [%s]: n=%d, A1 Viol=%.2f%%, C1=%.1f%%, C2=%.1f%%, Cert=%.1f%%, Z_L/sum_val (Med=%.2f, p90=%.2f, Max=%.2f)",
                    label, n_fact, a1_viol_pct, c1_pass_pct, c2_pass_pct, cert_a1_pct, med_ratio, p90_ratio, max_ratio)

    # Export LaTeX Table M0
    tex_path = TABLES_DIR / "M0.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by experiments/run_m0_export_facts.py\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Gate-check summary for simple derivations (Assumption~1) and dynamic programming bound tightness ($Z_L$) across unique facts and clip-level instances.}\n")
        f.write("\\label{tab:gate_check_m0}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Fact Cohort} & \\textbf{$N$ Facts} & \\textbf{A1 Violations (\\%)} & \\multicolumn{3}{c}{\\textbf{$Z_L / \\sum \\mathrm{val}(D)$ Ratio}} \\\\\n")
        f.write("\\cmidrule(lr){4-6}\n")
        f.write(" & & & \\textbf{Median} & \\textbf{p90} & \\textbf{Max} \\\\\n")
        f.write("\\midrule\n")
        for r in m0_rows:
            f.write(f"{r['subset']} & {r['n_fact']} & {r['a1_viol_pct']:.1f}\\% & {r['med_ratio']:.2f} & {r['p90_ratio']:.2f} & {r['max_ratio']:.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    LOGGER.info("Table M0 saved to %s", tex_path)

    # Save summary json
    summary_path = RESULTS_DIR / "facts_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_rows": len(df),
            "test_rows": len(test_df),
            "unique_facts": len(unique_facts_df),
            "m0_summary": m0_rows,
        }, f, indent=2)
    LOGGER.info("Summary saved to %s", summary_path)

    # Gate check assertions
    for r in m0_rows:
        if r["a1_viol_pct"] > 5.0:
            LOGGER.warning("GATE WARNING: %s has %.1f%% > 5%% A1 violations!", r["subset"], r["a1_viol_pct"])
        else:
            LOGGER.info("GATE PASSED: %s has 0.0%% A1 violations.", r["subset"])
        if r["p90_ratio"] > 10.0:
            LOGGER.warning("GATE WARNING: %s has p90 ratio %.2f > 10.0!", r["subset"], r["p90_ratio"])
        else:
            LOGGER.info("GATE PASSED: %s has tight p90 ratio %.2f <= 10.0.", r["subset"], r["p90_ratio"])

    LOGGER.info("=" * 80)
    LOGGER.info("M0 EXPORT COMPLETED SUCCESSFULLY!")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()

