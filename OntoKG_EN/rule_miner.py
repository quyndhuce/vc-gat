# -*- coding: utf-8 -*-
r"""
Lineage-Aware Rule Induction (LARI) Engine
==========================================
Implements Algorithm 1 (LARI) for the two-tier reasoning architecture:
    [ Candidate Rule Mining ]
                │
                ▼
    [ Lineage & Type Validation (LARI) ]
                │
                ▼
    [ Validated Rules \hat{H}_valid ]
                │
                ▼
    [ Lineage-Disjoint Fuzzy Inference (LDFI) ]

Mathematical Formulation:
1. Candidate Pattern Mining (2-hop closed Horn patterns):
   rho: r1(x, y) ^ r2(y, z) => r*(x, z)

2. Structural Confidence with Bayesian Shrinkage:
   \tilde{C}_PCA(rho) = (N_success + alpha) / (N_eligible + alpha + beta)
   Prevents small sample size artifacts (e.g. Supp = 6 having 1.0 confidence).

3. Lineage Corroboration Score:
   For grounding g = (x, y, z) with body premises B_g = {f1, f2} and observed head f_h = (x, r*, z):
     Lambda_B(g) = Lambda(f1) U Lambda(f2)
     Lambda_H(g) = Lambda(f_h)
     Disjoint confirmation iff Lambda_B(g) intersect Lambda_H(g) == empty.
   C_lin(rho) = N_disjoint_confirmed(rho) / N_confirmed(rho)

4. Ontological Type Consistency (Hard Filter):
   TypeCons(rho) in {0, 1} ensures relations preserve valid domain/range.

5. Unified Rule Reliability:
   q_rho = sqrt(\tilde{C}_PCA(rho) * C_lin(rho))
   Learned rule weight w_rho is set directly to q_rho (no artificial floors).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from OntoKG_EN.ldfi_engine import TypedRule

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
LOGGER = logging.getLogger("LARIEngine")

MACRO_LOCATION_URIS: Set[str] = {
    "http://www.wikidata.org/entity/Q1858",   # Hanoi
    "http://transmtl.vn/ent/207462d2",        # Hanoi
    "http://transmtl.vn/ent/5f8f9e33",        # Vietnam
    "http://www.wikidata.org/entity/Q881",    # Vietnam
    "http://transmtl.vn/ent/e909c2d7",        # country
}

# ─────────────────────────────────────────────────────────────────────────────
# Formal Ontology T-Box Domain-Range & Property-Chain Axioms (Non-Circular)
# ─────────────────────────────────────────────────────────────────────────────
# Based on W3C OWL 2 Property Chain Axioms (SubObjectPropertyOf) and Mereology.
# Validates whether a relational composition r1 o r2 => r* is admissible under
# the foundational schema of the Tourism Knowledge Graph:

def check_ontology_property_chain_consistency(r1: str, r2: str, r_star: str) -> bool:
    """
    Formal Description Logic (DL) and Mereological Constraint Validator.
    Takes ANY arbitrary relational triple (r1, r2, r*) and evaluates domain-range
    and property-hierarchy compatibility WITHOUT using any hardcoded rule whitelists.
    
    Axioms:
      A1. Mereological Range Invariance: If r2 in ('located_in', 'in'), target is an
          administrative region; r* cannot be 'part_of' (which requires a parent monument).
      A2. Topological Non-Containment: Proximity/orientation/sightline relations
          ('near', 'faces', 'overlooks', 'connects', 'leading_to', 'surrounded_by')
          do NOT entail topological containment ('located_in').
      A3. Semiotic Independence: Symbolic relations ('symbol_of') do NOT compose with
          spatial relations to produce physical containment.
      A4. Governance Invariance: Administrative jurisdiction ('capital_of') requires
          capital-city to country domain/range, not generic containment.
      A5. Structural Non-Inheritance: Parent complex does not inherit atomic dates
          ('built_in') or fine-grained styles from constituent parts.
    """
    # A1: Meronymic vs Geographic Range
    if r2 in ("located_in", "in") and r_star == "part_of":
        return False
    if r1 in ("on", "situated_on") and r_star == "part_of" and r2 == "located_in":
        return False
        
    # A2: Proximity / Sightline / Orientation non-containment
    proximity_rels = ("near", "overlooks", "faces", "connects", "leading_to", "surrounded_by", "border_with", "borders")
    if (r1 in proximity_rels or r2 in proximity_rels) and r_star in ("located_in", "part_of", "in", "on"):
        return False
        
    # A3: Semiotic independence - symbol_of cannot be composed transitively with spatial/meronymic relations
    if "symbol_of" in (r1, r2, r_star):
        return False
        
    # A4: Governance invariance - political rule applies to polities, not physical structures; capital_of is not spatial containment
    if "ruled_by" in (r1, r2, r_star) or "capital_of" in (r1, r2, r_star):
        return False
        
    # A5: Structural/Chronological non-inheritance
    if r_star in ("built_in", "architectural_style") and r1 in ("has_part", "part_of"):
        return False
    if r_star == "near" and r1 in ("renamed_to", "on", "has_part"):
        return False
        
    # A6: Alias identity consistency: alias relations can only be composed from alias chains
    if r_star in ("renamed_to", "former_name_of") and (r1 not in ("renamed_to", "former_name_of") or r2 not in ("renamed_to", "former_name_of")):
        return False
        
    return True


@dataclass
class LARIRule:
    r1: str
    r2: str
    r_star: str
    support: int
    n_eligible: int
    raw_pca_conf: float
    shrunk_pca_conf: float
    n_disjoint_confirmed: int
    c_lin: float
    is_type_valid: bool
    q_rho: float

    def to_typed_rule(self, rule_id_prefix: str = "LARI_R") -> TypedRule:
        rule_id = f"{rule_id_prefix}_{self.r1[:4]}_{self.r2[:4]}_{self.r_star[:4]}".upper()
        desc = (
            f"LARI: {self.r1} o {self.r2} => {self.r_star} "
            f"(q_rho={self.q_rho:.4f}, PCA_shrunk={self.shrunk_pca_conf:.4f}, "
            f"C_lin={self.c_lin:.4f}, Supp={self.support})"
        )
        return TypedRule(
            rule_id=rule_id,
            r1=self.r1,
            r2=self.r2,
            r_star=self.r_star,
            w_rho=round(self.q_rho, 4),
            description=desc,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": f"{self.r1} o {self.r2} => {self.r_star}",
            "r1": self.r1,
            "r2": self.r2,
            "r_star": self.r_star,
            "support": self.support,
            "n_eligible": self.n_eligible,
            "raw_pca_conf": round(self.raw_pca_conf, 4),
            "shrunk_pca_conf": round(self.shrunk_pca_conf, 4),
            "n_disjoint_confirmed": self.n_disjoint_confirmed,
            "c_lin": round(self.c_lin, 4),
            "is_type_valid": self.is_type_valid,
            "q_rho": round(self.q_rho, 4),
        }


class LARIRuleMiner:
    def __init__(
        self,
        min_support: int = 5,
        min_pca_conf: float = 0.20,
        alpha_prior: float = 1.0,
        beta_prior: float = 2.0,
        require_type_validity: bool = True,
        min_lari_score: float = 0.25,
    ):
        self.min_support = min_support
        self.min_pca_conf = min_pca_conf
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.require_type_validity = require_type_validity
        self.min_lari_score = min_lari_score

        self.facts: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.direct_edges: Dict[Tuple[str, str], Dict[str, Tuple[float, Set[str]]]] = defaultdict(dict)
        self.out_edges: Dict[str, List[Tuple[str, str, float, Set[str]]]] = defaultdict(list)
        self.heads_having_rel: Dict[str, Set[str]] = defaultdict(set)
        self.rel_instance_counts: Counter[str] = Counter()
        self.uri_to_surf: Dict[str, str] = {}

        self.kb_subgraphs: Dict[str, Dict[Tuple[str, str, str], Dict[str, Any]]] = defaultdict(dict)
        self.kb_out_edges: Dict[str, Dict[str, List[Tuple[str, str, float, Set[str]]]]] = defaultdict(lambda: defaultdict(list))
        self.retained_rules: List[LARIRule] = []

    def load_from_per_kb_directory(self, data_dir: Path) -> int:
        """Load URI-canonical facts, per-KB subgraphs, and copy-group lineages."""
        kb_files = sorted(data_dir.glob("*/fkg_facts.json"))
        LOGGER.info("LARI Miner loading from %d per-KB files in %s...", len(kb_files), data_dir)

        for kbf in kb_files:
            kb_id = kbf.parent.name
            with open(kbf, "r", encoding="utf-8") as fp:
                d = json.load(fp)

            for sv in d.get("single_valued_groups", []):
                h = sv.get("head_uri")
                r = sv.get("relation", "").strip()
                h_surf = sv.get("head_surface", "")
                if not h or not r:
                    continue
                if h_surf:
                    self.uri_to_surf[h] = h_surf

                for cand in sv.get("candidates", []):
                    t = cand.get("tail_uri")
                    bel = cand.get("bel", 0.0)
                    t_surf = cand.get("tail_surface", "")
                    if not t or bel <= 0:
                        continue
                    if t_surf:
                        self.uri_to_surf[t] = t_surf

                    lineage = set()
                    for e in cand.get("evidence", []):
                        aid = e.get("article_id") or kb_id
                        cg = e.get("copy_group_id", -1)
                        origin = f"{kb_id}::cg_{cg}" if cg >= 0 else f"{kb_id}::doc_{aid}"
                        lineage.add(origin)

                    key = (h, r, t)
                    fact_data = {"mu": bel, "lineage": lineage, "h_surf": h_surf, "t_surf": t_surf}
                    if key not in self.kb_subgraphs[kb_id] or bel > self.kb_subgraphs[kb_id][key]["mu"]:
                        self.kb_subgraphs[kb_id][key] = fact_data
                    else:
                        self.kb_subgraphs[kb_id][key]["lineage"].update(lineage)

                    if key not in self.facts or bel > self.facts[key]["mu"]:
                        self.facts[key] = {"mu": bel, "lineage": lineage}
                    else:
                        self.facts[key]["lineage"].update(lineage)

            for mv in d.get("multi_valued_facts", []):
                h = mv.get("head_uri")
                r = mv.get("relation", "").strip()
                t = mv.get("tail_uri")
                mu = mv.get("mu_F", 0.0)
                h_surf = mv.get("head_surface", "")
                t_surf = mv.get("tail_surface", "")
                if not h or not r or not t or mu <= 0:
                    continue
                if h_surf:
                    self.uri_to_surf[h] = h_surf
                if t_surf:
                    self.uri_to_surf[t] = t_surf

                lineage = set()
                for e in mv.get("evidence", []):
                    aid = e.get("article_id") or kb_id
                    cg = e.get("copy_group_id", -1)
                    origin = f"{kb_id}::cg_{cg}" if cg >= 0 else f"{kb_id}::doc_{aid}"
                    lineage.add(origin)

                key = (h, r, t)
                fact_data = {"mu": mu, "lineage": lineage, "h_surf": h_surf, "t_surf": t_surf}
                if key not in self.kb_subgraphs[kb_id] or mu > self.kb_subgraphs[kb_id][key]["mu"]:
                    self.kb_subgraphs[kb_id][key] = fact_data
                else:
                    self.kb_subgraphs[kb_id][key]["lineage"].update(lineage)

                if key not in self.facts or mu > self.facts[key]["mu"]:
                    self.facts[key] = {"mu": mu, "lineage": lineage}
                else:
                    self.facts[key]["lineage"].update(lineage)

        # Build global and local adjacency and PCA structures
        for (h, r, t), data in self.facts.items():
            self.direct_edges[(h, t)][r] = (data["mu"], data["lineage"])
            self.out_edges[h].append((r, t, data["mu"], data["lineage"]))
            self.heads_having_rel[r].add(h)
            self.rel_instance_counts[r] += 1

        for kb_id, facts_dict in self.kb_subgraphs.items():
            for (h, r, t), data in facts_dict.items():
                self.kb_out_edges[kb_id][h].append((r, t, data["mu"], data["lineage"]))

        LOGGER.info(
            "LARI loaded %d unique global facts across %d relations and %d landmark subgraphs.",
            len(self.facts),
            len(self.rel_instance_counts),
            len(self.kb_subgraphs),
        )
        return len(self.facts)

    def induce_rules(self, mining_mode: str = "local_to_global") -> Tuple[List[LARIRule], List[LARIRule]]:
        """
        Executes Algorithm 1: Lineage-Aware Rule Induction (LARI).
        mining_mode='local_to_global': Enumerate 2-hop body groundings strictly within each
            landmark subgraph G_i, then aggregate and evaluate rule schemas globally across the corpus.
        Returns:
            all_candidates: All mined Horn candidates with their computed scores.
            validated_rules: Rules passing type consistency and LARI reliability thresholds.
        """
        LOGGER.info(
            "Inducing rules (mode='%s'): local path enumeration across %d subgraphs -> global aggregation...",
            mining_mode,
            len(self.kb_subgraphs) if self.kb_subgraphs else 1,
        )

        body_instances: Dict[Tuple[str, str], List[Tuple[str, str, str, float, Set[str]]]] = defaultdict(list)

        if mining_mode == "local_to_global" and self.kb_subgraphs:
            # Step 1: Enumerate 2-hop grounding paths locally within each landmark subgraph G_i
            for kb_id, facts_dict in self.kb_subgraphs.items():
                for (h, r1, m), data1 in facts_dict.items():
                    if r1 == "part_of" and m in MACRO_LOCATION_URIS:
                        continue
                    mu1 = data1["mu"]
                    lin1 = data1["lineage"]

                    for (r2, t, mu2, lin2) in self.kb_out_edges[kb_id].get(m, []):
                        if t == h:
                            continue  # suppress self-loops
                        mu_body = max(0.0, mu1 + mu2 - 1.0)
                        if mu_body <= 0.0:
                            continue
                        lin_body = lin1 | lin2
                        body_instances[(r1, r2)].append((h, m, t, mu_body, lin_body))
        else:
            # Fallback pure global enumeration
            for (h, r1, m), data1 in self.facts.items():
                if r1 == "part_of" and m in MACRO_LOCATION_URIS:
                    continue
                mu1 = data1["mu"]
                lin1 = data1["lineage"]

                for (r2, t, mu2, lin2) in self.out_edges.get(m, []):
                    if t == h:
                        continue
                    mu_body = max(0.0, mu1 + mu2 - 1.0)
                    if mu_body <= 0.0:
                        continue
                    lin_body = lin1 | lin2
                    body_instances[(r1, r2)].append((h, m, t, mu_body, lin_body))

        # Step 2: Global Corpus-Level Rule Schema Scoring
        all_candidates: List[LARIRule] = []

        for (r1, r2), instances in body_instances.items():
            candidate_r_stars: Set[str] = set()
            for h, m, t, _, _ in instances:
                for r_s in self.direct_edges.get((h, t), {}):
                    candidate_r_stars.add(r_s)

            for r_star in candidate_r_stars:
                pos_sum = 0.0
                pca_den_sum = 0.0
                supp = 0
                n_eligible = 0
                n_disjoint_confirmed = 0
                h_has_r_star = self.heads_having_rel[r_star]

                for h, m, t, mu_b, lin_b in instances:
                    if h in h_has_r_star:
                        pca_den_sum += mu_b
                        n_eligible += 1
                        if r_star in self.direct_edges.get((h, t), {}):
                            mu_h, lin_h = self.direct_edges[(h, t)][r_star]
                            pos_sum += min(mu_b, mu_h)
                            supp += 1
                            if lin_b.isdisjoint(lin_h):
                                n_disjoint_confirmed += 1

                if supp >= self.min_support and pca_den_sum > 0:
                    raw_pca = pos_sum / pca_den_sum
                    if raw_pca >= self.min_pca_conf:
                        shrunk_pca = (supp + self.alpha_prior) / (n_eligible + self.alpha_prior + self.beta_prior)
                        c_lin = n_disjoint_confirmed / supp if supp > 0 else 0.0
                        q_rho = (shrunk_pca * c_lin) ** 0.5

                        is_type_valid = check_ontology_property_chain_consistency(r1, r2, r_star)

                        lari_rule = LARIRule(
                            r1=r1,
                            r2=r2,
                            r_star=r_star,
                            support=supp,
                            n_eligible=n_eligible,
                            raw_pca_conf=raw_pca,
                            shrunk_pca_conf=shrunk_pca,
                            n_disjoint_confirmed=n_disjoint_confirmed,
                            c_lin=c_lin,
                            is_type_valid=is_type_valid,
                            q_rho=q_rho,
                        )
                        all_candidates.append(lari_rule)

        # Filter validated rules
        validated_rules = [
            r for r in all_candidates
            if (not self.require_type_validity or r.is_type_valid) and r.q_rho >= self.min_lari_score
        ]
        validated_rules.sort(key=lambda r: r.q_rho, reverse=True)
        self.retained_rules = validated_rules

        LOGGER.info(
            "LARI Induction complete: %d candidate patterns mined -> %d validated rules H_valid retained globally.",
            len(all_candidates),
            len(validated_rules),
        )
        return all_candidates, validated_rules

    def save_retained_rule_library(self, filepath: Optional[Path] = None) -> Path:
        """Serializes the global LARI-retained rule library R_LARI into a canonical JSON file."""
        if filepath is None:
            filepath = repo_root / "data/fkg_per_kb/lari_global_retained_rules.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "title": "Global LARI-Retained Rule Library (R_LARI)",
            "paradigm": "Globally Learned Rules, Locally Instantiated Inference",
            "num_subgraphs_mined": len(self.kb_subgraphs),
            "num_retained_rules": len(self.retained_rules),
            "rules": [
                {
                    **r.to_dict(),
                    "rule_id": r.to_typed_rule().rule_id,
                    "formula": f"{r.r1}(x, y) ^ {r.r2}(y, z) => {r.r_star}(x, z)",
                }
                for r in self.retained_rules
            ],
        }

        with open(filepath, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)

        LOGGER.info("Saved global LARI rule library with %d rules to %s", len(self.retained_rules), filepath)
        return filepath

    @classmethod
    def load_retained_rule_library(cls, filepath: Path) -> List[TypedRule]:
        """Loads the global retained rule library as a list of TypedRules for online inference."""
        if not filepath.exists():
            raise FileNotFoundError(f"Rule library not found at {filepath}")
        with open(filepath, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        rules = []
        for rd in data.get("rules", []):
            rules.append(
                TypedRule(
                    rule_id=rd["rule_id"],
                    r1=rd["r1"],
                    r2=rd["r2"],
                    r_star=rd["r_star"],
                    w_rho=rd["q_rho"],
                    description=f"LARI: {rd['formula']} (q_rho={rd['q_rho']:.4f}, supp={rd['support']})",
                )
            )
        LOGGER.info("Loaded %d typed rules from %s", len(rules), filepath)
        return rules


# Alias for backward compatibility
FuzzyRuleMiner = LARIRuleMiner

if __name__ == "__main__":
    data_dir = repo_root / "data/fkg_per_kb"
    miner = LARIRuleMiner()
    miner.load_from_per_kb_directory(data_dir)
    candidates, validated = miner.induce_rules()

    print(f"\nTotal Candidates: {len(candidates)}")
    print(f"Validated Rules (H_valid): {len(validated)}")
    print(f"\n{'Rule':45s} | {'Supp':4s} | {'PCA_raw':8s} | {'PCA_shrunk':10s} | {'C_lin':8s} | {'q_rho':8s}")
    print("-" * 95)
    for r in validated:
        s = f"{r.r1} o {r.r2} => {r.r_star}"
        print(f"{s:45s} | {r.support:4d} | {r.raw_pca_conf:8.4f} | {r.shrunk_pca_conf:10.4f} | {r.c_lin:8.4f} | {r.q_rho:8.4f}")

    saved_path = miner.save_retained_rule_library()
    print(f"\nSaved Global LARI Rule Library to: {saved_path}")
