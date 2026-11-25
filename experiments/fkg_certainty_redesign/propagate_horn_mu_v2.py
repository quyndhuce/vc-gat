#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propagate_horn_mu_v2.py
=============================================================================
Stage 4 (V2): Stratified Horn Reasoning & Derivation Aggregation on V2 Graph.

Propagates conflict-aware base certainties through:
  Stratum 1: Inverse & Symmetric Axioms
  Stratum 2: Description-Logic Constrained 2-Hop Horn Rules
Noisy-OR Aggregation:
  mu_horn(C) = 1 - prod_{D in Deriv(C)} (1 - val(D))
Joint Base + Inferred Fusion:
  mu_F(C) = 1 - (1 - w_base(C)) * prod_{D in Deriv(C)} (1 - val(D))

Outputs:
  - experiments/outputs/fkg_facts_verified_v2.json
=============================================================================
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("PropagateHornMuV2")

REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")
FUSED_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "fused_facts_dst_v2.json"
OUTPUT_VERIFIED_PATH = REPO_ROOT / "experiments" / "outputs" / "fkg_facts_verified_v2.json"

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


def propagate_horn_closure_v2():
    LOGGER.info(f"Loading V2 fused facts from {FUSED_PATH}...")
    with open(FUSED_PATH, "r", encoding="utf-8") as f:
        fused_facts = json.load(f)

    LOGGER.info(f"Loaded {len(fused_facts):,} facts.")

    # Index base facts by triple
    triple_to_fact: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for fid, f in fused_facts.items():
        s = (f.get("subject") or "").strip().lower()
        p = (f.get("predicate") or "").strip().lower()
        o = (f.get("object") or "").strip().lower()
        if s and p and o:
            triple_to_fact[(s, p, o)] = f

    derivations: Dict[Tuple[str, str, str], List[Tuple[str, float, List[str]]]] = defaultdict(list)

    # 1. Stratum 1: Inverse & Symmetric Axioms
    LOGGER.info("Applying Stratum 1: Inverse and Symmetric axioms...")
    for (s, p, o), f in list(triple_to_fact.items()):
        w_base = f["w_base"]
        if w_base > 0.05:
            # Inverse
            if p in INVERSE_MAP:
                inv_p = INVERSE_MAP[p]
                k_inv = (o, inv_p, s)
                val_d = round(w_base * 0.98, 4)
                derivations[k_inv].append((f"inverse_{p}", val_d, [f"{s} {p} {o}"]))

            # Symmetric
            if p in SYM_RELS:
                gamma = SYM_RELS[p]
                k_sym = (o, p, s)
                val_d = round(w_base * gamma, 4)
                derivations[k_sym].append((f"symmetric_{p}", val_d, [f"{s} {p} {o}"]))

    # 2. Stratum 2: 2-Hop Horn Rules
    LOGGER.info("Building 2-hop adjacency graph for Stratum 2 rules...")
    adj: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    for (s, p, o), f in triple_to_fact.items():
        w_base = f["w_base"]
        if w_base > 0.05:
            adj[s][p][o] = w_base

    for (s, p, o), d_list in derivations.items():
        max_d = max(d[1] for d in d_list)
        if max_d > 0.05 and o not in adj[s][p]:
            adj[s][p][o] = max_d

    LOGGER.info("Applying Stratum 2: 2-hop verified Horn rules with DL constraints...")
    rule_counts = defaultdict(int)

    for x in adj:
        # Rule 1: part_of(x, y) ^ rel(y, z) => rel(x, z)
        for y, w_xy in adj[x].get("part_of", {}).items():
            if x == y or y in MACRO_LOCATION_SURFACES:
                continue
            for r_prop in ["located_in", "in", "belongs_to_dynasty", "architectural_style", "worships", "dedicated_to"]:
                target_r = "located_in" if r_prop == "in" else r_prop
                rule_name = f"part_of o {r_prop} => {target_r}"
                gamma = 0.95 if target_r == "located_in" else 0.88
                for z, w_yz in adj.get(y, {}).get(r_prop, {}).items():
                    if x == z:
                        continue
                    k = (x, target_r, z)
                    val_d = round(gamma * w_xy * w_yz, 4)
                    if val_d > 0.01:
                        derivations[k].append((rule_name, val_d, [f"{x} part_of {y}", f"{y} {r_prop} {z}"]))
                        rule_counts[rule_name] += 1

        # Rule 2: located_in(x, y) ^ located_in(y, z) => located_in(x, z)
        for y, w_xy in adj[x].get("located_in", {}).items():
            if x == y:
                continue
            rule_name = "located_in o located_in => located_in"
            gamma = 0.95
            for z, w_yz in adj.get(y, {}).get("located_in", {}).items():
                if x == z:
                    continue
                k = (x, "located_in", z)
                val_d = round(gamma * w_xy * w_yz, 4)
                if val_d > 0.01:
                    derivations[k].append((rule_name, val_d, [f"{x} located_in {y}", f"{y} located_in {z}"]))
                    rule_counts[rule_name] += 1

        # Rule 3: renamed_to(x, y) ^ located_in(y, z) => located_in(x, z)
        for alias_r in ["renamed_to", "former_name_of"]:
            for y, w_xy in adj[x].get(alias_r, {}).items():
                if x == y:
                    continue
                rule_name = f"{alias_r} o located_in => located_in"
                gamma = 0.90
                for z, w_yz in adj.get(y, {}).get("located_in", {}).items():
                    if x == z:
                        continue
                    k = (x, "located_in", z)
                    val_d = round(gamma * w_xy * w_yz, 4)
                    if val_d > 0.01:
                        derivations[k].append((rule_name, val_d, [f"{x} {alias_r} {y}", f"{y} located_in {z}"]))
                        rule_counts[rule_name] += 1

        # Rule 4: on(x, y) ^ located_in(y, z) => located_in(x, z)
        for y, w_xy in adj[x].get("on", {}).items():
            if x == y:
                continue
            rule_name = "on o located_in => located_in"
            gamma = 0.92
            for z, w_yz in adj.get(y, {}).get("located_in", {}).items():
                if x == z:
                    continue
                k = (x, "located_in", z)
                val_d = round(gamma * w_xy * w_yz, 4)
                if val_d > 0.01:
                    derivations[k].append((rule_name, val_d, [f"{x} on {y}", f"{y} located_in {z}"]))
                    rule_counts[rule_name] += 1

        # Rule 5: houses(x, y) ^ worships(y, z) => worships(x, z)
        for y, w_xy in adj[x].get("houses", {}).items():
            for z, w_yz in adj.get(y, {}).get("worships", {}).items():
                k = (x, "worships", z)
                val_d = round(0.88 * w_xy * w_yz, 4)
                if val_d > 0.01:
                    derivations[k].append(("houses o worships => worships", val_d, [f"{x} houses {y}", f"{y} worships {z}"]))
                    rule_counts["houses o worships => worships"] += 1

        # Rule 6: houses(x, y) ^ dedicated_to(y, z) => dedicated_to(x, z)
        for y, w_xy in adj[x].get("houses", {}).items():
            for z, w_yz in adj.get(y, {}).get("dedicated_to", {}).items():
                k = (x, "dedicated_to", z)
                val_d = round(0.88 * w_xy * w_yz, 4)
                if val_d > 0.01:
                    derivations[k].append(("houses o dedicated_to => dedicated_to", val_d, [f"{x} houses {y}", f"{y} dedicated_to {z}"]))
                    rule_counts["houses o dedicated_to => dedicated_to"] += 1

        # Rule 7: has_component(x, y) ^ architectural_style(y, z) => architectural_style(x, z)
        for y, w_xy in adj[x].get("has_component", {}).items():
            for z, w_yz in adj.get(y, {}).get("architectural_style", {}).items():
                k = (x, "architectural_style", z)
                val_d = round(0.85 * w_xy * w_yz, 4)
                if val_d > 0.01:
                    derivations[k].append(("has_component o architectural_style => architectural_style", val_d, [f"{x} has_component {y}", f"{y} architectural_style {z}"]))
                    rule_counts["has_component o architectural_style => architectural_style"] += 1

    LOGGER.info(f"Derived paths across {len(derivations):,} triples.")

    # 3. Final synthesis
    verified_facts_v2 = {}
    for fid, f in fused_facts.items():
        s = (f.get("subject") or "").strip().lower()
        p = (f.get("predicate") or "").strip().lower()
        o = (f.get("object") or "").strip().lower()
        k = (s, p, o)

        w_base = f["w_base"]

        # Noisy-OR over derivations
        derivs = derivations.get(k, [])
        prod_comp = 1.0
        for _, val_d, _ in derivs:
            prod_comp *= (1.0 - val_d)
        mu_horn = 1.0 - prod_comp

        # Joint fusion
        mu_f_final = round(1.0 - (1.0 - w_base) * prod_comp, 4)

        entry = dict(f)
        entry["mu_F_v2"] = mu_f_final
        entry["mu_horn"] = round(mu_horn, 4)
        entry["derivations_count"] = len(derivs)
        verified_facts_v2[fid] = entry

    LOGGER.info(f"Saving V2 verified facts to {OUTPUT_VERIFIED_PATH}...")
    OUTPUT_VERIFIED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_VERIFIED_PATH, "w", encoding="utf-8") as f:
        json.dump(verified_facts_v2, f, indent=2, ensure_ascii=False)
    LOGGER.info("V2 Horn propagation completed.")


if __name__ == "__main__":
    propagate_horn_closure_v2()
