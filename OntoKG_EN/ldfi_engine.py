# -*- coding: utf-8 -*-
"""
Lineage-Disjoint Fuzzy Inference (LDFI) Engine
==============================================
Implementation of the LDFI operator for Approximate Reasoning on Fuzzy Knowledge Graphs:
  1. URI-Canonical Identity: Nodes are identified strictly by URI (e.g. http://transmtl.vn/kb/...),
     preventing spurious entity collisions from identical surface names (e.g., 'main hall', 'gate').
  2. Source-Dependence Lineage: Lineage is tracked at copy-group level (origin = kb_id::cg_{copy_group_id}),
     falling back to article_id only when copy group is unavailable.
  3. Typed Fuzzy Composition Rules: rho = <r1, r2, r*, Dx, Dy, Dz, w_rho> with strict ontological typing
     (preventing macro-locations like Hanoi/Vietnam from acting as intermediate complex parents).
  4. Conservative Path Support via Lukasiewicz t-norm:
       s(P) = max(0, w_rho + mu_F(f1) + mu_F(f2) - 2)
  5. Derivation Lineage: Lambda(P) = Lambda(f1) U Lambda(f2)
  6. Lineage-Disjoint Aggregation:
       mu_I(c) = max_{A in A_C(c)} [ 1 - prod_{P in A} (1 - s(P)) ]
     subject to: |A| <= C and Lambda(Pi) intersect Lambda(Pj) = empty for all i != j.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
LOGGER = logging.getLogger("LDFIEngine")

MACRO_LOCATION_URIS: Set[str] = {
    "http://www.wikidata.org/entity/Q1858",   # Hanoi
    "http://transmtl.vn/ent/207462d2",        # Hanoi
    "http://transmtl.vn/ent/5f8f9e33",        # Vietnam
    "http://www.wikidata.org/entity/Q881",    # Vietnam
    "http://transmtl.vn/ent/e909c2d7",        # country
}


@dataclass(frozen=True)
class TypedRule:
    rule_id: str
    r1: str
    r2: str
    r_star: str
    w_rho: float
    description: str = ""


@dataclass
class Derivation:
    premise1_uri: Tuple[str, str, str]  # (h_uri, r1, m_uri)
    premise2_uri: Tuple[str, str, str]  # (m_uri, r2, t_uri)
    intermediate_surf: str
    rule: TypedRule
    mu1: float
    mu2: float
    s_P: float
    lineage: Set[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "premise1_uri": list(self.premise1_uri),
            "premise2_uri": list(self.premise2_uri),
            "intermediate_surface": self.intermediate_surf,
            "rule_id": self.rule.rule_id,
            "rule": f"{self.rule.r1} o {self.rule.r2} => {self.rule.r_star}",
            "mu1": round(self.mu1, 4),
            "mu2": round(self.mu2, 4),
            "w_rho": self.rule.w_rho,
            "s_P": round(self.s_P, 4),
            "lineage": sorted(list(self.lineage)),
        }


@dataclass
class InferredConclusion:
    head_uri: str
    relation: str
    tail_uri: str
    head_surf: str = ""
    tail_surf: str = ""
    derivations: List[Derivation] = field(default_factory=list)
    mu_max: float = 0.0
    mu_naive: float = 0.0
    mu_ldfi: float = 0.0
    selected_pair_indices: Optional[Tuple[int, int]] = None
    is_disjoint_pair: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conclusion_uri": [self.head_uri, self.relation, self.tail_uri],
            "conclusion_surface": [self.head_surf, self.relation, self.tail_surf],
            "total_derivations": len(self.derivations),
            "mu_max_C1": round(self.mu_max, 4),
            "mu_naive_all_path": round(self.mu_naive, 4),
            "mu_ldfi_C2": round(self.mu_ldfi, 4),
            "is_disjoint_corroboration": self.is_disjoint_pair,
            "selected_pair_indices": list(self.selected_pair_indices) if self.selected_pair_indices else None,
            "derivations": [d.to_dict() for d in self.derivations],
        }


class LDFIEngine:
    def __init__(self, rules: Optional[List[TypedRule]] = None, cap_C: int = 2):
        self.cap_C = cap_C
        if rules is not None:
            self.rules = rules
        else:
            self.rules = [
                TypedRule("R1_LOC", "part_of", "located_in", "located_in", 0.95, "Child monument inherits location of parent complex"),
                TypedRule("R2_MAT", "has_part", "made_of", "incorporates_material", 0.92, "Structure incorporates element material"),
                TypedRule("R3_DYNASTY_BUILT", "built_in", "dated_to", "belongs_to_dynasty", 0.95, "Era built translates to historical dynasty"),
                TypedRule("R4_DYNASTY_PERSON", "dedicated_to", "associated_with", "related_dynasty", 0.88, "Deity dedication relates to dynasty"),
            ]
        # (head_uri, relation, tail_uri) -> {'mu': float, 'lineage': Set[str], 'h_surf': str, 't_surf': str}
        self.base_facts: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.uri_to_surf: Dict[str, str] = {}

    def load_from_per_kb_directory(self, data_dir: Path) -> int:
        """Load ground-truth facts with strict URI canonicalization and copy-group lineages."""
        kb_files = sorted(data_dir.glob("*/fkg_facts.json"))
        LOGGER.info("Loading URI-canonical facts and copy-group lineages from %d per-KB files...", len(kb_files))

        loaded_count = 0
        for kbf in kb_files:
            kb_id = kbf.parent.name
            with open(kbf, "r", encoding="utf-8") as fp:
                d = json.load(fp)

            # Single-valued groups
            for sv in d.get("single_valued_groups", []):
                h_uri = sv.get("head_uri")
                h_surf = sv.get("head_surface", "")
                r = sv.get("relation", "").strip()
                if not h_uri or not r:
                    continue
                if h_surf:
                    self.uri_to_surf[h_uri] = h_surf

                for cand in sv.get("candidates", []):
                    t_uri = cand.get("tail_uri")
                    t_surf = cand.get("tail_surface", "")
                    bel = cand.get("bel", 0.0)
                    if not t_uri or bel <= 0:
                        continue
                    if t_surf:
                        self.uri_to_surf[t_uri] = t_surf

                    lineage = set()
                    for e in cand.get("evidence", []):
                        aid = e.get("article_id") or kb_id
                        cg = e.get("copy_group_id", -1)
                        origin = f"{kb_id}::cg_{cg}" if cg >= 0 else f"{kb_id}::doc_{aid}"
                        lineage.add(origin)

                    key = (h_uri, r, t_uri)
                    if key not in self.base_facts or bel > self.base_facts[key]["mu"]:
                        self.base_facts[key] = {"mu": bel, "lineage": lineage, "h_surf": h_surf, "t_surf": t_surf, "type": "SV"}
                    else:
                        self.base_facts[key]["lineage"].update(lineage)
                    loaded_count += 1

            # Multi-valued facts
            for mv in d.get("multi_valued_facts", []):
                h_uri = mv.get("head_uri")
                h_surf = mv.get("head_surface", "")
                r = mv.get("relation", "").strip()
                t_uri = mv.get("tail_uri")
                t_surf = mv.get("tail_surface", "")
                mu = mv.get("mu_F", 0.0)
                if not h_uri or not r or not t_uri or mu <= 0:
                    continue
                if h_surf:
                    self.uri_to_surf[h_uri] = h_surf
                if t_surf:
                    self.uri_to_surf[t_uri] = t_surf

                lineage = set()
                for e in mv.get("evidence", []):
                    aid = e.get("article_id") or kb_id
                    cg = e.get("copy_group_id", -1)
                    origin = f"{kb_id}::cg_{cg}" if cg >= 0 else f"{kb_id}::doc_{aid}"
                    lineage.add(origin)

                key = (h_uri, r, t_uri)
                if key not in self.base_facts or mu > self.base_facts[key]["mu"]:
                    self.base_facts[key] = {"mu": mu, "lineage": lineage, "h_surf": h_surf, "t_surf": t_surf, "type": "MV"}
                else:
                    self.base_facts[key]["lineage"].update(lineage)
                loaded_count += 1

        LOGGER.info("Unified base graph loaded: %d distinct URI-canonical facts across %d KBs.", len(self.base_facts), len(kb_files))
        return len(self.base_facts)

    @staticmethod
    def aggregate_derivations(derivs: List[Any], cap_C: int = 2) -> float:
        """Static aggregator for a list of Derivations or derivation dicts under cap_C."""
        if not derivs:
            return 0.0
        
        # Normalize s_P and lineage
        parsed = []
        for d in derivs:
            sp = d.s_P if hasattr(d, "s_P") else d["s_P"]
            lin = d.lineage if hasattr(d, "lineage") else d["lineage"]
            lin_set = lin if isinstance(lin, set) else set(lin)
            parsed.append((float(sp), lin_set))
            
        mu_max = max(p[0] for p in parsed)
        if cap_C == 1 or len(parsed) == 1:
            return float(mu_max)
            
        best = mu_max
        M = len(parsed)
        for i in range(M):
            for j in range(i + 1, M):
                if parsed[i][1].isdisjoint(parsed[j][1]):
                    c = 1.0 - (1.0 - parsed[i][0]) * (1.0 - parsed[j][0])
                    if c > best:
                        best = c
                        
        if cap_C is not None and (cap_C >= 3 or cap_C == float("inf")) and M >= 3:
            for i in range(M):
                for j in range(i + 1, M):
                    for k in range(j + 1, M):
                        if (parsed[i][1].isdisjoint(parsed[j][1]) and 
                            parsed[i][1].isdisjoint(parsed[k][1]) and 
                            parsed[j][1].isdisjoint(parsed[k][1])):
                            c = 1.0 - (1.0 - parsed[i][0]) * (1.0 - parsed[j][0]) * (1.0 - parsed[k][0])
                            if c > best:
                                best = c
        return float(best)

    def infer(self) -> Dict[Tuple[str, str, str], InferredConclusion]:
        """Execute Lineage-Disjoint Fuzzy Inference (LDFI) with type constraints and copy-group lineages."""
        LOGGER.info("Executing LDFI inference over %d typed rules...", len(self.rules))

        # Adjacency index: head_uri -> list of (relation, tail_uri, mu, lineage)
        adj: Dict[str, List[Tuple[str, str, float, Set[str]]]] = defaultdict(list)
        for (h_uri, r, t_uri), data in self.base_facts.items():
            adj[h_uri].append((r, t_uri, data["mu"], data["lineage"]))

        grouped_derivations: Dict[Tuple[str, str, str], List[Derivation]] = defaultdict(list)

        for (h_uri, r1, t1_uri), data1 in self.base_facts.items():
            # Constraint: In part_of rules, intermediate node must not be a macro location
            if r1 == "part_of" and t1_uri in MACRO_LOCATION_URIS:
                continue

            for (r2, t2_uri, mu2, lin2) in adj.get(t1_uri, []):
                if t2_uri == h_uri:
                    continue  # suppress self-loops

                for rule in self.rules:
                    if r1 == rule.r1 and r2 == rule.r2:
                        s_P = max(0.0, rule.w_rho + data1["mu"] + mu2 - 2.0)
                        if s_P > 0.0:
                            lin_P = data1["lineage"] | lin2
                            deriv = Derivation(
                                premise1_uri=(h_uri, r1, t1_uri),
                                premise2_uri=(t1_uri, r2, t2_uri),
                                intermediate_surf=self.uri_to_surf.get(t1_uri, t1_uri),
                                rule=rule,
                                mu1=data1["mu"],
                                mu2=mu2,
                                s_P=s_P,
                                lineage=lin_P,
                            )
                            grouped_derivations[(h_uri, rule.r_star, t2_uri)].append(deriv)

        LOGGER.info("Discovered %d candidate conclusion triples across derivations.", len(grouped_derivations))

        conclusions: Dict[Tuple[str, str, str], InferredConclusion] = {}

        for (h_uri, r_star, t_uri), derivs in grouped_derivations.items():
            derivs.sort(key=lambda d: d.s_P, reverse=True)

            mu_max = max(d.s_P for d in derivs)

            prod = 1.0
            for d in derivs:
                prod *= (1.0 - d.s_P)
            mu_naive = 1.0 - prod

            best_ldfi = mu_max
            best_pair = None
            is_disjoint = False

            if self.cap_C == 1 or len(derivs) == 1:
                best_ldfi = mu_max
            else:
                for i in range(len(derivs)):
                    for j in range(i + 1, len(derivs)):
                        if derivs[i].lineage.isdisjoint(derivs[j].lineage):
                            s_comb = 1.0 - (1.0 - derivs[i].s_P) * (1.0 - derivs[j].s_P)
                            if s_comb > best_ldfi:
                                best_ldfi = s_comb
                                best_pair = (i, j)
                                is_disjoint = True

            concl = InferredConclusion(
                head_uri=h_uri,
                relation=r_star,
                tail_uri=t_uri,
                head_surf=self.uri_to_surf.get(h_uri, h_uri),
                tail_surf=self.uri_to_surf.get(t_uri, t_uri),
                derivations=derivs,
                mu_max=mu_max,
                mu_naive=mu_naive,
                mu_ldfi=best_ldfi,
                selected_pair_indices=best_pair,
                is_disjoint_pair=is_disjoint,
            )
            conclusions[(h_uri, r_star, t_uri)] = concl

        LOGGER.info("LDFI complete: %d semantic conclusions computed under URI identity.", len(conclusions))
        return conclusions


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data/fkg_per_kb"

    engine = LDFIEngine()
    engine.load_from_per_kb_directory(data_dir)
    results = engine.infer()

    print(f"\nTotal computed conclusions: {len(results)}")
    multi_deriv = [c for c in results.values() if len(c.derivations) > 1]
    print(f"Conclusions with multiple paths (M > 1): {len(multi_deriv)}")
    disjoint_corroborated = [c for c in multi_deriv if c.is_disjoint_pair]
    print(f"Conclusions with genuine copy-group disjoint corroboration: {len(disjoint_corroborated)}")
    print(f"Conclusions with shared copy-group redundancy: {len(multi_deriv) - len(disjoint_corroborated)}")
