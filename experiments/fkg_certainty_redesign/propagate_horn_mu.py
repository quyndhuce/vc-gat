#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propagate_horn_mu.py
=============================================================================
Stage 4: Stratified Weighted Horn Reasoning & Derivation Aggregation.

Propagates verified certainty scores through:
  Stratum 1: Inverse and Symmetric Axioms
  Stratum 2: Description-Logic Constrained 2-Hop Horn Rules

Derivation Scoring:
  val(D) = gamma_r * w(A) * w(B)
Noisy-OR Aggregation:
  mu_horn(C) = 1 - prod_{D in Deriv(C)} (1 - val(D))
Joint Base + Inferred Fusion:
  mu_F(C) = 1 - (1 - w_base(C)) * prod_{D in Deriv(C)} (1 - val(D))

Outputs:
  - experiments/fkg_certainty_redesign/fkg_facts_verified.json
=============================================================================
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("PropagateHornMu")

REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")
FUSED_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "fused_facts_dst.json"
OUTPUT_VERIFIED_PATH = REPO_ROOT / "experiments" / "outputs" / "fkg_facts_verified.json"

MACRO_LOCATION_SURFACES = {
    "hanoi", "ha noi", "vietnam", "viet nam",
    "ho chi minh city", "ho chi minh", "hcmc", "tp ho chi minh", "saigon", "sai gon",
    "the", "unknown", "city"
}

INVERSE_MAP = {
    "part_of": "has_part",
    "has_part": "part_of",
    "has_component": "part_of",
    "located_inside": "houses",
    "houses": "located_inside",
    "above": "below",
    "below": "above",
    "located_in_front_of": "located_behind",
    "located_behind": "located_in_front_of"
}

SYM_RELS = {
    "adjacent_to": 1.0,
    "connected_to": 1.0,
    "bordering": 1.0,
    "associated_with": 0.90
}


def propagate_horn_closure():
    LOGGER.info(f"Loading fused facts from {FUSED_PATH}...")
    with open(FUSED_PATH, "r", encoding="utf-8") as f:
        fused_facts = json.load(f)

    LOGGER.info(f"Loaded {len(fused_facts):,} facts.")

    # 1. Index base facts by triple
    triple_to_fact: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    fact_by_id: Dict[str, Dict[str, Any]] = {}

    for fid, f in fused_facts.items():
        s = (f.get("subject") or "").strip().lower()
        p = (f.get("predicate") or "").strip().lower()
        o = (f.get("object") or "").strip().lower()
        fact_by_id[fid] = f
        if s and p and o:
            triple_to_fact[(s, p, o)] = f

    # Keep track of derivations for each triple: (s, p, o) -> List[Tuple[rule_name, val_D, [premises]]]
    derivations_dst: Dict[Tuple[str, str, str], List[Tuple[str, float, List[str]]]] = defaultdict(list)
    derivations_tconorm: Dict[Tuple[str, str, str], List[Tuple[str, float, List[str]]]] = defaultdict(list)

    # 2. Stratum 1: Inverse and Symmetric Closure
    LOGGER.info("Applying Stratum 1: Inverse and Symmetric axioms...")
    for (s, p, o), f in list(triple_to_fact.items()):
        w_dst = f["dst"]["w_base"]
        w_tc = f["w_tconorm"]

        # Inverse
        if p in INVERSE_MAP and w_dst > 0.05:
            inv_p = INVERSE_MAP[p]
            k_inv = (o, inv_p, s)
            val_d_dst = round(w_dst * 0.98, 4)
            val_d_tc = round(w_tc * 0.98, 4)
            derivations_dst[k_inv].append((f"inverse_{p}", val_d_dst, [f"{s} {p} {o}"]))
            derivations_tconorm[k_inv].append((f"inverse_{p}", val_d_tc, [f"{s} {p} {o}"]))

        # Symmetric
        if p in SYM_RELS and w_dst > 0.05:
            gamma = SYM_RELS[p]
            k_sym = (o, p, s)
            val_d_dst = round(w_dst * gamma, 4)
            val_d_tc = round(w_tc * gamma, 4)
            derivations_dst[k_sym].append((f"symmetric_{p}", val_d_dst, [f"{s} {p} {o}"]))
            derivations_tconorm[k_sym].append((f"symmetric_{p}", val_d_tc, [f"{s} {p} {o}"]))

    # 3. Stratum 2: 2-Hop Horn Rules
    LOGGER.info("Building 2-hop adjacency graph for Stratum 2 rules...")
    # Map head -> relation -> tail -> (w_dst, w_tc)
    adj: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = defaultdict(lambda: defaultdict(dict))

    for (s, p, o), f in triple_to_fact.items():
        w_dst = f["dst"]["w_base"]
        w_tc = f["w_tconorm"]
        if w_dst > 0.05:
            adj[s][p][o] = (w_dst, w_tc)

    # Also incorporate high-confidence inverse edges into adj
    for (s, p, o), d_list in derivations_dst.items():
        max_d_dst = max(d[1] for d in d_list)
        if max_d_dst > 0.05 and o not in adj[s][p]:
            max_d_tc = max(d[1] for d in derivations_tconorm[s, p, o])
            adj[s][p][o] = (max_d_dst, max_d_tc)

    LOGGER.info("Applying Stratum 2: 2-hop verified Horn rules with DL constraints...")
    rule_counts = defaultdict(int)

    for x in adj:
        # Rule 1: part_of(x, y) ^ rel(y, z) => rel(x, z)
        for y, (w_xy_dst, w_xy_tc) in adj[x].get("part_of", {}).items():
            if x == y or y in MACRO_LOCATION_SURFACES:
                continue
            for r_prop in ["located_in", "in", "belongs_to_dynasty", "architectural_style", "worships", "dedicated_to"]:
                target_r = "located_in" if r_prop == "in" else r_prop
                rule_name = f"part_of o {r_prop} => {target_r}"
                gamma = 0.95 if target_r == "located_in" else 0.88
                for z, (w_yz_dst, w_yz_tc) in adj.get(y, {}).get(r_prop, {}).items():
                    if x == z:
                        continue
                    k = (x, target_r, z)
                    val_dst = round(gamma * w_xy_dst * w_yz_dst, 4)
                    val_tc = round(gamma * w_xy_tc * w_yz_tc, 4)
                    if val_dst > 0.01:
                        derivations_dst[k].append((rule_name, val_dst, [f"{x} part_of {y}", f"{y} {r_prop} {z}"]))
                        derivations_tconorm[k].append((rule_name, val_tc, [f"{x} part_of {y}", f"{y} {r_prop} {z}"]))
                        rule_counts[rule_name] += 1

        # Rule 2: located_in(x, y) ^ located_in(y, z) => located_in(x, z)
        for y, (w_xy_dst, w_xy_tc) in adj[x].get("located_in", {}).items():
            if x == y:
                continue
            rule_name = "located_in o located_in => located_in"
            gamma = 0.95
            for z, (w_yz_dst, w_yz_tc) in adj.get(y, {}).get("located_in", {}).items():
                if x == z:
                    continue
                k = (x, "located_in", z)
                val_dst = round(gamma * w_xy_dst * w_yz_dst, 4)
                val_tc = round(gamma * w_xy_tc * w_yz_tc, 4)
                if val_dst > 0.01:
                    derivations_dst[k].append((rule_name, val_dst, [f"{x} located_in {y}", f"{y} located_in {z}"]))
                    derivations_tconorm[k].append((rule_name, val_tc, [f"{x} located_in {y}", f"{y} located_in {z}"]))
                    rule_counts[rule_name] += 1

        # Rule 3: renamed_to(x, y) ^ located_in(y, z) => located_in(x, z)
        for alias_r in ["renamed_to", "former_name_of"]:
            for y, (w_xy_dst, w_xy_tc) in adj[x].get(alias_r, {}).items():
                if x == y:
                    continue
                rule_name = f"{alias_r} o located_in => located_in"
                gamma = 0.90
                for z, (w_yz_dst, w_yz_tc) in adj.get(y, {}).get("located_in", {}).items():
                    if x == z:
                        continue
                    k = (x, "located_in", z)
                    val_dst = round(gamma * w_xy_dst * w_yz_dst, 4)
                    val_tc = round(gamma * w_xy_tc * w_yz_tc, 4)
                    if val_dst > 0.01:
                        derivations_dst[k].append((rule_name, val_dst, [f"{x} {alias_r} {y}", f"{y} located_in {z}"]))
                        derivations_tconorm[k].append((rule_name, val_tc, [f"{x} {alias_r} {y}", f"{y} located_in {z}"]))
                        rule_counts[rule_name] += 1

        # Rule 4: on(x, y) ^ located_in(y, z) => located_in(x, z)
        for y, (w_xy_dst, w_xy_tc) in adj[x].get("on", {}).items():
            if x == y:
                continue
            rule_name = "on o located_in => located_in"
            gamma = 0.92
            for z, (w_yz_dst, w_yz_tc) in adj.get(y, {}).get("located_in", {}).items():
                if x == z:
                    continue
                k = (x, "located_in", z)
                val_dst = round(gamma * w_xy_dst * w_yz_dst, 4)
                val_tc = round(gamma * w_xy_tc * w_yz_tc, 4)
                if val_dst > 0.01:
                    derivations_dst[k].append((rule_name, val_dst, [f"{x} on {y}", f"{y} located_in {z}"]))
                    derivations_tconorm[k].append((rule_name, val_tc, [f"{x} on {y}", f"{y} located_in {z}"]))
                    rule_counts[rule_name] += 1

        # Rule 5: houses(x, y) ^ worships(y, z) => worships(x, z)
        for y, (w_xy_dst, w_xy_tc) in adj[x].get("houses", {}).items():
            for z, (w_yz_dst, w_yz_tc) in adj.get(y, {}).get("worships", {}).items():
                k = (x, "worships", z)
                val_dst = round(0.88 * w_xy_dst * w_yz_dst, 4)
                val_tc = round(0.88 * w_xy_tc * w_yz_tc, 4)
                if val_dst > 0.01:
                    derivations_dst[k].append(("houses o worships => worships", val_dst, [f"{x} houses {y}", f"{y} worships {z}"]))
                    derivations_tconorm[k].append(("houses o worships => worships", val_tc, [f"{x} houses {y}", f"{y} worships {z}"]))
                    rule_counts["houses o worships => worships"] += 1

        # Rule 6: houses(x, y) ^ dedicated_to(y, z) => dedicated_to(x, z)
        for y, (w_xy_dst, w_xy_tc) in adj[x].get("houses", {}).items():
            for z, (w_yz_dst, w_yz_tc) in adj.get(y, {}).get("dedicated_to", {}).items():
                k = (x, "dedicated_to", z)
                val_dst = round(0.88 * w_xy_dst * w_yz_dst, 4)
                val_tc = round(0.88 * w_xy_tc * w_yz_tc, 4)
                if val_dst > 0.01:
                    derivations_dst[k].append(("houses o dedicated_to => dedicated_to", val_dst, [f"{x} houses {y}", f"{y} dedicated_to {z}"]))
                    derivations_tconorm[k].append(("houses o dedicated_to => dedicated_to", val_tc, [f"{x} houses {y}", f"{y} dedicated_to {z}"]))
                    rule_counts["houses o dedicated_to => dedicated_to"] += 1

        # Rule 7: has_component(x, y) ^ architectural_style(y, z) => architectural_style(x, z)
        for y, (w_xy_dst, w_xy_tc) in adj[x].get("has_component", {}).items():
            for z, (w_yz_dst, w_yz_tc) in adj.get(y, {}).get("architectural_style", {}).items():
                k = (x, "architectural_style", z)
                val_dst = round(0.85 * w_xy_dst * w_yz_dst, 4)
                val_tc = round(0.85 * w_xy_tc * w_yz_tc, 4)
                if val_dst > 0.01:
                    derivations_dst[k].append(("has_component o architectural_style => architectural_style", val_dst, [f"{x} has_component {y}", f"{y} architectural_style {z}"]))
                    derivations_tconorm[k].append(("has_component o architectural_style => architectural_style", val_tc, [f"{x} has_component {y}", f"{y} architectural_style {z}"]))
                    rule_counts["has_component o architectural_style => architectural_style"] += 1

    LOGGER.info(f"Derived paths across {len(derivations_dst):,} triples. Rule firings: {dict(rule_counts)}")

    # 4. Synthesize final verified scores for all facts in the ontology
    LOGGER.info("Computing noisy-or aggregation for all ontology facts...")
    verified_facts = {}

    for fid, f in fused_facts.items():
        s = (f.get("subject") or "").strip().lower()
        p = (f.get("predicate") or "").strip().lower()
        o = (f.get("object") or "").strip().lower()
        k = (s, p, o)

        w_base_dst = f["dst"]["w_base"]
        w_tconorm = f["w_tconorm"]
        w_nli_only = f["w_nli_only"]

        # Noisy-OR over derivations (DST)
        derivs_dst = derivations_dst.get(k, [])
        prod_comp_dst = 1.0
        for _, val_d, _ in derivs_dst:
            prod_comp_dst *= (1.0 - val_d)
        mu_horn_dst = 1.0 - prod_comp_dst

        # Combine base + Horn (noisy-or)
        mu_f_dst_horn = round(1.0 - (1.0 - w_base_dst) * prod_comp_dst, 4)

        # Noisy-OR over derivations (t-conorm)
        derivs_tc = derivations_tconorm.get(k, [])
        prod_comp_tc = 1.0
        for _, val_d, _ in derivs_tc:
            prod_comp_tc *= (1.0 - val_d)
        mu_f_tconorm_horn = round(1.0 - (1.0 - w_tconorm) * prod_comp_tc, 4)

        # For M1: NLI + DST base facts only (if inferred and no base, remains 0.0)
        mu_f_m1_dst_base = w_base_dst

        # For M4: Direct NLI-only
        mu_f_m4_nli_only = w_nli_only

        verified_entry = dict(f)
        verified_entry["mu_F_models"] = {
            "M0_baseline": f["old_mu_F"],
            "M1_nli_dst_base": mu_f_m1_dst_base,
            "M2_nli_dst_horn": mu_f_dst_horn,
            "M3_nli_tconorm_horn": mu_f_tconorm_horn,
            "M4_nli_only": mu_f_m4_nli_only,
        }
        verified_entry["derivations_count"] = len(derivs_dst)
        if derivs_dst:
            verified_entry["top_derivation"] = {
                "rule": derivs_dst[0][0],
                "val": derivs_dst[0][1],
                "premises": derivs_dst[0][2]
            }

        verified_facts[fid] = verified_entry

    LOGGER.info(f"Saving verified facts to {OUTPUT_VERIFIED_PATH}...")
    OUTPUT_VERIFIED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_VERIFIED_PATH, "w", encoding="utf-8") as f:
        json.dump(verified_facts, f, indent=2, ensure_ascii=False)
    LOGGER.info("Horn propagation and multi-model synthesis completed.")


if __name__ == "__main__":
    propagate_horn_closure()
