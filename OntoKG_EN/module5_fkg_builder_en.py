# module5_fkg_builder_en.py
"""
Module 5 EN — Fuzzy Knowledge Graph (FKG) Global Builder
=========================================================

Builds a **global Fuzzy Knowledge Graph** from per-article triples and entities,
using a hybrid Dempster–Shafer / t-conorm evidence fusion framework.

Theoretical foundation
──────────────────────
Nodes and edges carry *membership degrees* μ ∈ [0, 1].

  • **Single-valued relations** (e.g. built_in, has_height):
    Only one value is correct ⇒ candidates are *mutually exclusive*.
    We use **Dempster's rule of combination** over a frame of discernment
    Θ = {t₁, t₂, …, tₙ, other}.

    Each evidence j supporting value tⱼ produces mass function:
        mⱼ({tⱼ}) = aⱼ · bⱼ       (aⱼ = source authority, bⱼ = extraction confidence)
        mⱼ(Θ)    = 1 − aⱼ · bⱼ   (residual ignorance)

    Combination:  m₁₂(A) = [1/(1−K)] Σ_{B∩C=A} m₁(B)·m₂(C)
    Conflict:     K       = Σ_{B∩C=∅} m₁(B)·m₂(C)

    Result:  Bel(t) = m({t}),  Pl(t) = m({t}) + m(Θ)
    Fact membership = Bel(t) for the winning candidate.

  • **Multi-valued relations** (e.g. dedicated_to, part_of):
    Many values can be simultaneously true ⇒ each fact is independent.
    We use the **product t-conorm**:
        μ_F(f) = 1 − ∏ⱼ (1 − aⱼ · bⱼ)

  • **Storage principle**: we store the *evidence list*, NOT the cached
    membership degrees. Degrees are recomputed whenever evidence changes.

Pipeline
────────
  Step 1: Entity Unification — verify cross-article URI consistency,
          merge internal URIs with cosine(embedding) ≥ threshold,
          log ambiguous pairs for human review.
  Step 2: Triple Merging — group identical (head, rel, tail) facts,
          keep per-occurrence evidence.
  Step 3: Copy Detection — pairwise sentence similarity to detect
          non-independent evidence (near-duplicate copying).
  Step 4: Hybrid Fusion — Dempster for single-valued groups,
          t-conorm for multi-valued facts.  Flag high-conflict facts.

Input
─────
  OntoKG_EN/*_triples_en.json      (Module 4 output)
  OntoKG_EN/*_linked_output.json   (Module 3 output)
  OntoKG_EN/entity_registry_en.pkl (shared registry)

Output
──────
  data/fkg_en/fkg_facts.json          — all facts with evidence & membership
  data/fkg_en/fkg_stats.json          — statistics
  data/fkg_en/conflict_report.txt     — facts needing human review
  data/fkg_en/entity_index_en.pkl     — unified entity index
  data/fkg_en/fkg_networkx.pkl        — NetworkX graph with fuzzy weights
  data/fkg_en/ambiguous_pairs.json    — entity pairs in the ambiguous zone
"""
from __future__ import annotations

import json
import logging
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    nx = None  # type: ignore[assignment]
    HAS_NX = False

# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────
LOGGER = logging.getLogger("module5_fkg_builder_en")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(_h)


# ══════════════════════════════════════════════════════════
# 1 ▸ CONSTANTS
# ══════════════════════════════════════════════════════════

# Relations where only ONE tail value is correct for a given head.
# Dempster's rule of combination is applied.
SINGLE_VALUED_RELATIONS: Set[str] = {
    "capital_of", "built_in", "founded_in", "inaugurated_in",
    "gifted_in", "has_height", "has_width", "has_length",
    "has_area", "recognized_in", "symbol_of", "former_name_of",
    "dated_to", "demolished_in", "renovated_in",
    "has_number_of_levels", "has_number_of_columns", "has_shape",
    "has_roof_type", "has_roof_shape",
}

# Metadata / structural relations — deduplicated but NOT fused.
METADATA_RELATIONS: Set[str] = {"rdf:type", "hasEntity", "belongsTo"}

# Default source authority when no per-article authority is configured.
DEFAULT_AUTHORITY_BY_METHOD: Dict[str, float] = {
    "metadata":               1.0,
    "landmark_meronym_anchor": 0.95,
    "gliner_relex":           0.80,
}

# The special key representing ignorance in Dempster mass functions.
THETA = "__theta__"

# Copy detection threshold: SequenceMatcher ratio.
COPY_SIM_THRESHOLD = 0.85

# Dempster conflict warning threshold.
CONFLICT_K_THRESHOLD = 0.50

# Entity embedding cosine similarity thresholds.
MERGE_SIM_THRESHOLD    = 0.92   # auto-merge above this
AMBIGUOUS_SIM_LOWER    = 0.85   # log-for-review between this and MERGE
AMBIGUOUS_SIM_UPPER    = MERGE_SIM_THRESHOLD

# URI source base confidence for node membership.
URI_SOURCE_BASE_CONFIDENCE: Dict[str, float] = {
    "wikidata":        0.95,
    "embedding_match": 0.85,
    "alias_match":     0.90,
    "surface_match":   0.88,
    "new":             0.70,
}


# ══════════════════════════════════════════════════════════
# 2 ▸ DATA CLASSES
# ══════════════════════════════════════════════════════════

@dataclass
class Evidence:
    """A single piece of evidence supporting a triple."""
    article_id:     str
    confidence:     float       # bⱼ : extraction confidence (GLiNER score)
    authority:      float       # aⱼ : source authority
    evidence_text:  str         # sentence / context string
    head_surface:   str
    tail_surface:   str
    window_idx:     int   = -1
    method:         str   = "gliner_relex"
    copy_group_id:  int   = -1     # -1 = not assigned yet
    is_representative: bool = True # False ⇒ eclipsed by a higher-authority copy

    @property
    def mass(self) -> float:
        """mⱼ({tⱼ}) = aⱼ · bⱼ"""
        return self.authority * self.confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "article_id":       self.article_id,
            "confidence":       round(self.confidence, 6),
            "authority":        round(self.authority, 4),
            "evidence_text":    self.evidence_text,
            "head_surface":     self.head_surface,
            "tail_surface":     self.tail_surface,
            "window_idx":       self.window_idx,
            "method":           self.method,
            "copy_group_id":    self.copy_group_id,
            "is_representative": self.is_representative,
            "mass":             round(self.mass, 6),
        }


@dataclass
class FactCandidate:
    """One candidate tail value for a single-valued (head, relation)."""
    tail_uri:     str
    tail_surface: str
    evidence:     List[Evidence] = field(default_factory=list)
    bel:          float = 0.0
    pl:           float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tail_uri":     self.tail_uri,
            "tail_surface": self.tail_surface,
            "bel":          round(self.bel, 6),
            "pl":           round(self.pl, 6),
            "n_evidence":   len(self.evidence),
            "evidence":     [e.to_dict() for e in self.evidence],
        }


@dataclass
class FactGroup:
    """All candidates for a single-valued (head, relation) pair."""
    head_uri:     str
    relation:     str
    head_surface: str
    candidates:   List[FactCandidate] = field(default_factory=list)
    conflict_K:   float  = 0.0
    needs_review: bool   = False
    winner_uri:   str    = ""
    winner_bel:   float  = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type":         "single_valued",
            "head_uri":     self.head_uri,
            "relation":     self.relation,
            "head_surface": self.head_surface,
            "conflict_K":   round(self.conflict_K, 6),
            "needs_review": self.needs_review,
            "winner_uri":   self.winner_uri,
            "winner_bel":   round(self.winner_bel, 6),
            "candidates":   [c.to_dict() for c in self.candidates],
        }


@dataclass
class FuzzyFact:
    """A multi-valued fact: (head, relation, tail) with t-conorm membership."""
    head_uri:     str
    relation:     str
    tail_uri:     str
    head_surface: str
    tail_surface: str
    evidence:     List[Evidence] = field(default_factory=list)
    mu_F:         float = 0.0       # cached t-conorm membership

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type":         "multi_valued",
            "head_uri":     self.head_uri,
            "relation":     self.relation,
            "tail_uri":     self.tail_uri,
            "head_surface": self.head_surface,
            "tail_surface": self.tail_surface,
            "mu_F":         round(self.mu_F, 6),
            "n_evidence":   len(self.evidence),
            "evidence":     [e.to_dict() for e in self.evidence],
        }


@dataclass
class MetadataFact:
    """A deduplicated structural triple (rdf:type, hasEntity, belongsTo)."""
    head_uri:     str
    relation:     str
    tail_uri:     str
    head_surface: str
    tail_surface: str
    n_occurrences: int = 1
    article_ids:   List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type":          "metadata",
            "head_uri":      self.head_uri,
            "relation":      self.relation,
            "tail_uri":      self.tail_uri,
            "head_surface":  self.head_surface,
            "tail_surface":  self.tail_surface,
            "n_occurrences": self.n_occurrences,
            "article_ids":   self.article_ids,
        }


# ══════════════════════════════════════════════════════════
# 3 ▸ CORE ALGORITHMS — DEMPSTER'S RULE
# ══════════════════════════════════════════════════════════

def dempster_combine_two(
    m1: Dict[str, float],
    m2: Dict[str, float],
) -> Tuple[Dict[str, float], float]:
    """Combine two mass functions via Dempster's rule.

    Each mass function maps hypothesis keys to mass values.
    Singletons are tail-URI strings; ``THETA`` represents ignorance.

    Intersection semantics (simple model — singletons + Θ):
      • {tᵢ} ∩ {tⱼ}  = ∅   when i ≠ j   → contributes to K
      • {tᵢ} ∩ {tᵢ}  = {tᵢ}
      • {tᵢ} ∩ Θ     = {tᵢ}
      • Θ    ∩ Θ     = Θ

    Returns
    -------
    combined : dict  — combined mass function
    K        : float — degree of conflict (0 = no conflict, 1 = total)
    """
    combined: Dict[str, float] = defaultdict(float)
    K = 0.0

    for a, ma in m1.items():
        if ma == 0.0:
            continue
        for b, mb in m2.items():
            if mb == 0.0:
                continue
            product = ma * mb
            if a == THETA and b == THETA:
                combined[THETA] += product
            elif a == THETA:
                combined[b] += product
            elif b == THETA:
                combined[a] += product
            elif a == b:
                combined[a] += product
            else:
                # Singletons that disagree → conflict
                K += product

    # Normalize by 1/(1−K)
    if K >= 1.0 - 1e-12:
        # Total conflict — rule is undefined
        LOGGER.warning("Dempster: total conflict K=%.6f ≈ 1; returning raw sums", K)
        return dict(combined), K

    norm = 1.0 / (1.0 - K)
    for key in combined:
        combined[key] *= norm

    return dict(combined), K


def dempster_combine_all(
    mass_functions: List[Dict[str, float]],
) -> Tuple[Dict[str, float], float]:
    """Sequentially combine a list of mass functions via Dempster's rule.

    Returns (final_mass, cumulative_K).

    Because the rule is *commutative and associative*, the result does not
    depend on the order of combination — only on the set of evidence.
    """
    if not mass_functions:
        return {THETA: 1.0}, 0.0
    if len(mass_functions) == 1:
        return dict(mass_functions[0]), 0.0

    current = mass_functions[0]
    cumulative_K = 0.0
    for mf in mass_functions[1:]:
        current, K_step = dempster_combine_two(current, mf)
        # Track conflict: 1 − (1−K_cum)(1−K_step) gives the overall
        # fraction of evidence-product mass that was conflicting.
        cumulative_K = 1.0 - (1.0 - cumulative_K) * (1.0 - K_step)
    return current, cumulative_K


def build_mass_function(tail_uri: str, authority: float, confidence: float) -> Dict[str, float]:
    """Build a basic mass function for one piece of evidence.

    mⱼ({tⱼ}) = aⱼ · bⱼ
    mⱼ(Θ)    = 1 − aⱼ · bⱼ
    """
    ab = authority * confidence
    ab = max(0.0, min(ab, 1.0))  # clamp
    return {tail_uri: ab, THETA: 1.0 - ab}


def compute_bel_pl(
    combined_mass: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    """From a combined mass function, compute Bel and Pl for each value.

    Bel(t) = m({t})
    Pl(t)  = m({t}) + m(Θ)
    """
    theta_mass = combined_mass.get(THETA, 0.0)
    result: Dict[str, Tuple[float, float]] = {}
    for key, mass in combined_mass.items():
        if key == THETA:
            continue
        bel = mass
        pl = mass + theta_mass
        result[key] = (bel, pl)
    return result


# ══════════════════════════════════════════════════════════
# 4 ▸ CORE ALGORITHMS — t-CONORM
# ══════════════════════════════════════════════════════════

def t_conorm_combine(evidence_pairs: List[Tuple[float, float]]) -> float:
    """Product t-conorm for independent evidence.

    μ_F = 1 − ∏ⱼ (1 − aⱼ · bⱼ)

    Args:
        evidence_pairs: list of (authority, confidence) tuples
    Returns:
        Combined membership degree in [0, 1]
    """
    if not evidence_pairs:
        return 0.0
    product = 1.0
    for a, b in evidence_pairs:
        product *= (1.0 - a * b)
    return 1.0 - product


# ══════════════════════════════════════════════════════════
# 5 ▸ COPY DETECTION
# ══════════════════════════════════════════════════════════

def _normalise_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip for comparison."""
    return " ".join(text.lower().split())


def detect_copy_groups(
    evidence_list: List[Evidence],
    threshold: float = COPY_SIM_THRESHOLD,
) -> List[int]:
    """Detect copy groups via pairwise sentence similarity.

    Uses ``difflib.SequenceMatcher`` (pure-Python, no external dep).
    Returns a list of group IDs (Union-Find canonical representatives).
    Two evidence items in the same group are considered non-independent.
    """
    n = len(evidence_list)
    if n <= 1:
        return list(range(n))

    # Union-Find with path compression
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    normalised = [_normalise_text(e.evidence_text) for e in evidence_list]

    for i in range(n):
        if not normalised[i]:
            continue
        for j in range(i + 1, n):
            if not normalised[j]:
                continue
            sim = SequenceMatcher(None, normalised[i], normalised[j]).ratio()
            if sim >= threshold:
                union(i, j)

    return [find(i) for i in range(n)]


def filter_independent_evidence(
    evidence_list: List[Evidence],
    threshold: float = COPY_SIM_THRESHOLD,
) -> List[Evidence]:
    """Assign copy-group IDs and mark non-representative copies.

    Within each copy group, only the evidence with the highest
    ``authority`` is kept as representative.  The others are still
    stored (for provenance) but marked ``is_representative = False``
    and are excluded from fusion calculations.

    Modifies evidence in-place and returns the list.
    """
    if len(evidence_list) <= 1:
        for e in evidence_list:
            e.copy_group_id = 0
            e.is_representative = True
        return evidence_list

    group_ids = detect_copy_groups(evidence_list, threshold)

    # Assign group IDs
    for i, gid in enumerate(group_ids):
        evidence_list[i].copy_group_id = gid

    # Within each group, keep only the highest-authority member
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, gid in enumerate(group_ids):
        groups[gid].append(i)

    for gid, members in groups.items():
        if len(members) == 1:
            evidence_list[members[0]].is_representative = True
            continue
        # Sort by authority descending, break ties by confidence
        members.sort(
            key=lambda idx: (evidence_list[idx].authority,
                             evidence_list[idx].confidence),
            reverse=True,
        )
        for rank, idx in enumerate(members):
            evidence_list[idx].is_representative = (rank == 0)

    return evidence_list


def get_representative_evidence(evidence_list: List[Evidence]) -> List[Evidence]:
    """Return only representative (independent) evidence for fusion."""
    return [e for e in evidence_list if e.is_representative]


# ══════════════════════════════════════════════════════════
# 6 ▸ ENTITY UNIFICATION
# ══════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class EntityUnifier:
    """Cross-article entity unification with ambiguity logging.

    Uses the 4-tier architecture already in Module 3:
      Tier 0: alias dictionary (already applied)
      Tier 1: surface_to_uri registry (already applied)
      Tier 2: Wikidata ID (already applied)
      Tier 3: embedding cosine similarity (post-hoc verification)

    This class performs a *post-hoc* verification pass on the entity
    registry to find internal URIs that should be merged.
    """

    def __init__(
        self,
        merge_threshold:    float = MERGE_SIM_THRESHOLD,
        ambiguous_lower:    float = AMBIGUOUS_SIM_LOWER,
    ):
        self.merge_threshold = merge_threshold
        self.ambiguous_lower = ambiguous_lower

        self.uri_redirect: Dict[str, str] = {}       # old_uri → canonical
        self.ambiguous_pairs: List[Dict[str, Any]] = []

    def unify(
        self,
        entity_data: Dict[str, Dict[str, Any]],
    ) -> Dict[str, str]:
        """Build a URI redirect map by checking embedding similarity
        among internal URIs with the same NER label.

        Args:
            entity_data:  uri → {info: {...}, embedding: [...], ...}

        Returns:
            uri_redirect: old_uri → canonical_uri  (identity for unchanged)
        """
        # Group internal URIs by label
        label_groups: Dict[str, List[str]] = defaultdict(list)
        embeddings: Dict[str, np.ndarray] = {}

        for uri, data in entity_data.items():
            info = data.get("info", {})
            source = info.get("uri_source", "")
            # Only check internal/new URIs — Wikidata URIs are authoritative
            if "wikidata.org" in uri:
                continue
            emb = data.get("embedding")
            if emb is None:
                continue
            emb_arr = np.array(emb, dtype=np.float32) if not isinstance(emb, np.ndarray) else emb
            if emb_arr.shape[0] < 10:
                continue
            label = info.get("label", "MISC")
            label_groups[label].append(uri)
            embeddings[uri] = emb_arr

        # Union-Find for merges
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                # Keep shorter URI as canonical (Wikidata > internal)
                if len(ry) < len(rx):
                    parent[rx] = ry
                else:
                    parent[ry] = rx

        # Pairwise similarity within label groups
        for label, uris in label_groups.items():
            for i in range(len(uris)):
                for j in range(i + 1, len(uris)):
                    u1, u2 = uris[i], uris[j]
                    sim = cosine_similarity(embeddings[u1], embeddings[u2])
                    if sim >= self.merge_threshold:
                        LOGGER.info(
                            "MERGE: %s ↔ %s  (sim=%.4f, label=%s)", u1, u2, sim, label
                        )
                        union(u1, u2)
                    elif sim >= self.ambiguous_lower:
                        self.ambiguous_pairs.append({
                            "uri_1":      u1,
                            "uri_2":      u2,
                            "similarity": round(sim, 4),
                            "label":      label,
                            "surfaces_1": entity_data[u1].get("info", {}).get(
                                "surface_forms", []),
                            "surfaces_2": entity_data[u2].get("info", {}).get(
                                "surface_forms", []),
                            "action":     "NEEDS_REVIEW",
                        })

        # Build redirect map (every URI maps to its canonical)
        all_uris = set(entity_data.keys())
        for uri in all_uris:
            canonical = find(uri)
            if canonical != uri:
                self.uri_redirect[uri] = canonical
            # else: identity, no redirect needed

        n_merged = len(self.uri_redirect)
        n_ambiguous = len(self.ambiguous_pairs)
        LOGGER.info(
            "Entity unification: %d merges, %d ambiguous pairs logged",
            n_merged, n_ambiguous,
        )
        return self.uri_redirect

    def apply_redirect(self, uri: str) -> str:
        """Map a URI through the redirect table."""
        return self.uri_redirect.get(uri, uri)


# ══════════════════════════════════════════════════════════
# 7 ▸ FKG BUILDER
# ══════════════════════════════════════════════════════════

class FKGBuilder:
    """Orchestrates the 4-step FKG construction pipeline."""

    def __init__(
        self,
        *,
        authority_map:          Optional[Dict[str, float]] = None,
        copy_sim_threshold:     float = COPY_SIM_THRESHOLD,
        conflict_K_threshold:   float = CONFLICT_K_THRESHOLD,
        merge_sim_threshold:    float = MERGE_SIM_THRESHOLD,
        ambiguous_sim_lower:    float = AMBIGUOUS_SIM_LOWER,
    ):
        # Per-article_id authority overrides  (article_id → authority)
        self.authority_map = authority_map or {}
        self.copy_sim_threshold   = copy_sim_threshold
        self.conflict_K_threshold = conflict_K_threshold

        # Entity unifier
        self.unifier = EntityUnifier(
            merge_threshold=merge_sim_threshold,
            ambiguous_lower=ambiguous_sim_lower,
        )

        # Intermediate state
        self._raw_triples:    List[Dict[str, Any]] = []
        self._entity_data:    Dict[str, Dict[str, Any]] = {}

        # Results
        self.fact_groups:   List[FactGroup]    = []   # single-valued
        self.fuzzy_facts:   List[FuzzyFact]    = []   # multi-valued
        self.metadata_facts: List[MetadataFact] = []  # structural
        self.entity_index:  Dict[str, Dict[str, Any]] = {}
        self.uri_redirect:  Dict[str, str]     = {}

    # ──────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────

    def _load_triples(self, triple_files: List[str]) -> None:
        """Load all triple JSON files."""
        for path in triple_files:
            p = Path(path)
            if not p.exists():
                LOGGER.warning("Triple file not found: %s", path)
                continue
            with open(p, "r", encoding="utf-8") as f:
                triples = json.load(f)
            self._raw_triples.extend(triples)
            LOGGER.info("Loaded %d triples from %s", len(triples), p.name)
        LOGGER.info("Total raw triples loaded: %d", len(self._raw_triples))

    def _load_entities(self, linked_files: List[str], registry_path: Optional[str] = None) -> None:
        """Load entities from linked output files and/or registry."""
        entity_data: Dict[str, Dict[str, Any]] = {}

        # From linked output files
        for path in linked_files:
            p = Path(path)
            if not p.exists():
                LOGGER.warning("Linked output not found: %s", path)
                continue
            with open(p, "r", encoding="utf-8") as f:
                record = json.load(f)

            article_id = record.get("article_id", p.stem)
            for ent in record.get("ner_entities", []):
                uri = ent.get("uri", "")
                if not uri:
                    continue
                if uri not in entity_data:
                    emb = ent.get("embedding")
                    entity_data[uri] = {
                        "info": {
                            "uri":            uri,
                            "label":          ent.get("label", "MISC"),
                            "label_name":     ent.get("label_name", ""),
                            "fine_grained_label": ent.get("fine_grained_label", ""),
                            "uri_source":     ent.get("uri_source", "new"),
                            "wikidata_id":    ent.get("wikidata_id"),
                            "surface_forms":  set(),
                            "article_ids":    set(),
                            "occurrence_count": 0,
                            "ner_confidences": [],
                        },
                        "embedding": emb,
                    }
                info = entity_data[uri]["info"]
                surface = ent.get("surface", "")
                if surface:
                    info["surface_forms"].add(surface)
                info["article_ids"].add(article_id)
                info["occurrence_count"] += 1
                conf = ent.get("confidence", 0.5)
                info["ner_confidences"].append(conf)
                # Update embedding by running average
                new_emb = ent.get("embedding")
                if new_emb is not None and entity_data[uri]["embedding"] is not None:
                    old = np.array(entity_data[uri]["embedding"], dtype=np.float32)
                    new = np.array(new_emb, dtype=np.float32)
                    if old.shape == new.shape:
                        n = info["occurrence_count"]
                        entity_data[uri]["embedding"] = ((old * (n - 1) + new) / n).tolist()

            # Concepts as entities
            for con in record.get("concept_mentions", []):
                uri = con.get("uri", "")
                if not uri:
                    continue
                if uri not in entity_data:
                    entity_data[uri] = {
                        "info": {
                            "uri":          uri,
                            "label":        "MISC",
                            "label_name":   "Concept",
                            "uri_source":   con.get("uri_source", "new"),
                            "wikidata_id":  None,
                            "surface_forms": set(),
                            "article_ids":  set(),
                            "occurrence_count": 0,
                            "ner_confidences": [],
                        },
                        "embedding": con.get("embedding"),
                    }
                info = entity_data[uri]["info"]
                concept_name = con.get("concept", "")
                if concept_name:
                    info["surface_forms"].add(concept_name)
                info["article_ids"].add(article_id)
                info["occurrence_count"] += 1

        # Also load from registry pkl
        if registry_path and Path(registry_path).exists():
            with open(registry_path, "rb") as f:
                reg = pickle.load(f)
            uri_info = reg.get("uri_info", {})
            reg_embeddings = reg.get("embeddings", {})
            for uri, info in uri_info.items():
                if uri not in entity_data:
                    emb = reg_embeddings.get(uri)
                    entity_data[uri] = {
                        "info": {
                            "uri":          uri,
                            "label":        info.get("label", "MISC"),
                            "label_name":   info.get("label_name", ""),
                            "uri_source":   info.get("uri_source", "new"),
                            "wikidata_id":  info.get("wikidata_id"),
                            "surface_forms": set(info.get("surface_forms", [])),
                            "article_ids":  set(),
                            "occurrence_count": info.get("occurrence_count", 1),
                            "ner_confidences": [],
                        },
                        "embedding": emb.tolist() if isinstance(emb, np.ndarray) else emb,
                    }

        # Fallback: ensure all URIs referenced in _raw_triples exist in entity_data
        for t in self._raw_triples:
            for role in ("head", "tail"):
                uri = t.get(f"{role}_uri", "")
                surf = t.get(f"{role}_surface", "")
                if uri and uri not in entity_data:
                    entity_data[uri] = {
                        "info": {
                            "uri":                uri,
                            "label":              t.get(f"{role}_type", "MISC").upper(),
                            "label_name":         t.get(f"{role}_type", ""),
                            "fine_grained_label": t.get(f"{role}_type", ""),
                            "uri_source":         "extracted",
                            "wikidata_id":        t.get(f"{role}_wikidata"),
                            "surface_forms":      {surf} if surf else set(),
                            "article_ids":        {t.get("article_id", "")},
                            "occurrence_count":   1,
                            "ner_confidences":    [t.get("confidence", 0.5)],
                        },
                        "embedding": None,
                    }
                elif uri in entity_data and surf:
                    sforms = entity_data[uri]["info"]["surface_forms"]
                    if isinstance(sforms, set):
                        sforms.add(surf)
                    else:
                        entity_data[uri]["info"]["surface_forms"] = set(sforms) | {surf}

        # Convert sets to lists for serialisation
        for uri, data in entity_data.items():
            info = data["info"]
            info["surface_forms"]  = sorted(info["surface_forms"])  if isinstance(info["surface_forms"], set) else info["surface_forms"]
            info["article_ids"]    = sorted(info["article_ids"])    if isinstance(info["article_ids"], set) else info["article_ids"]

        self._entity_data = entity_data
        LOGGER.info("Loaded %d unique entities", len(entity_data))

    # ──────────────────────────────────────────────────
    # Step 1: Entity Unification
    # ──────────────────────────────────────────────────

    def step1_unify_entities(self) -> Dict[str, str]:
        """Verify cross-article entity consistency and merge where needed."""
        LOGGER.info("═══ Step 1: Entity Unification ═══")
        self.uri_redirect = self.unifier.unify(self._entity_data)

        # Merge entity_data entries for redirected URIs
        for old_uri, new_uri in self.uri_redirect.items():
            if old_uri in self._entity_data and new_uri in self._entity_data:
                old_info = self._entity_data[old_uri]["info"]
                new_info = self._entity_data[new_uri]["info"]
                # Merge surfaces and article_ids
                if isinstance(new_info["surface_forms"], list):
                    new_info["surface_forms"] = list(
                        set(new_info["surface_forms"]) | set(old_info.get("surface_forms", []))
                    )
                if isinstance(new_info["article_ids"], list):
                    new_info["article_ids"] = list(
                        set(new_info["article_ids"]) | set(old_info.get("article_ids", []))
                    )
                new_info["occurrence_count"] = (
                    new_info.get("occurrence_count", 0) + old_info.get("occurrence_count", 0)
                )
            elif old_uri in self._entity_data:
                self._entity_data[new_uri] = self._entity_data[old_uri]

        # Remove redirected entries
        for old_uri in self.uri_redirect:
            self._entity_data.pop(old_uri, None)

        return self.uri_redirect

    # ──────────────────────────────────────────────────
    # Step 2: Triple Merging
    # ──────────────────────────────────────────────────

    def step2_merge_triples(self) -> None:
        """Group identical (head, rel, tail) triples, keeping evidence lists.

        Applies URI redirects from Step 1.
        Separates into: single-valued FactGroups, multi-valued FuzzyFacts,
        and MetadataFacts.
        """
        LOGGER.info("═══ Step 2: Triple Merging ═══")

        # Apply URI redirect
        redirect = self.unifier.apply_redirect

        # ── Separate metadata vs semantic triples ──
        metadata_counter: Dict[Tuple[str, str, str], MetadataFact] = {}
        # key=(head_uri, relation, tail_uri) → list of evidence dicts
        semantic_evidence: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

        skipped_self_loops = 0

        for t in self._raw_triples:
            head_uri = redirect(t.get("head_uri", ""))
            tail_uri = redirect(t.get("tail_uri", ""))
            relation = t.get("relation", "")

            if not (head_uri and tail_uri and relation):
                continue

            # Skip self-loops
            if head_uri == tail_uri and relation not in METADATA_RELATIONS:
                skipped_self_loops += 1
                continue

            if relation in METADATA_RELATIONS:
                key = (head_uri, relation, tail_uri)
                if key not in metadata_counter:
                    metadata_counter[key] = MetadataFact(
                        head_uri=head_uri,
                        relation=relation,
                        tail_uri=tail_uri,
                        head_surface=t.get("head_surface", ""),
                        tail_surface=t.get("tail_surface", ""),
                        n_occurrences=0,
                        article_ids=[],
                    )
                mf = metadata_counter[key]
                mf.n_occurrences += 1
                aid = t.get("article_id", "")
                if aid and aid not in mf.article_ids:
                    mf.article_ids.append(aid)
            else:
                key = (head_uri, relation, tail_uri)
                semantic_evidence[key].append(t)

        self.metadata_facts = list(metadata_counter.values())
        LOGGER.info(
            "Metadata: %d unique facts (from %d raw), skipped %d self-loops",
            len(self.metadata_facts),
            sum(mf.n_occurrences for mf in self.metadata_facts),
            skipped_self_loops,
        )

        # ── Build evidence objects for semantic triples ──
        # Determine authority for each article
        def _get_authority(article_id: str, method: str) -> float:
            if article_id in self.authority_map:
                return self.authority_map[article_id]
            return DEFAULT_AUTHORITY_BY_METHOD.get(method, 0.80)

        # Group semantic triples into FactGroups vs FuzzyFacts
        # First, build all evidence keyed by (head, rel, tail)
        fact_evidence: Dict[Tuple[str, str, str], List[Evidence]] = {}
        for key, raw_list in semantic_evidence.items():
            evidences = []
            for t in raw_list:
                head_uri, relation, tail_uri = key
                e = Evidence(
                    article_id    = t.get("article_id", ""),
                    confidence    = t.get("confidence", 0.5),
                    authority     = _get_authority(
                        t.get("article_id", ""), t.get("method", "gliner_relex")
                    ),
                    evidence_text = t.get("evidence", ""),
                    head_surface  = t.get("head_surface", ""),
                    tail_surface  = t.get("tail_surface", ""),
                    window_idx    = t.get("window_idx", -1),
                    method        = t.get("method", "gliner_relex"),
                )
                evidences.append(e)
            fact_evidence[key] = evidences

        # ── Single-valued: group by (head, relation) ──
        sv_groups: Dict[Tuple[str, str], FactGroup] = {}
        mv_facts: Dict[Tuple[str, str, str], FuzzyFact] = {}

        for (h, r, t), evidences in fact_evidence.items():
            if r in SINGLE_VALUED_RELATIONS:
                gkey = (h, r)
                if gkey not in sv_groups:
                    sv_groups[gkey] = FactGroup(
                        head_uri=h,
                        relation=r,
                        head_surface=evidences[0].head_surface if evidences else "",
                    )
                # Find or create candidate for this tail
                fg = sv_groups[gkey]
                candidate = None
                for c in fg.candidates:
                    if c.tail_uri == t:
                        candidate = c
                        break
                if candidate is None:
                    candidate = FactCandidate(
                        tail_uri=t,
                        tail_surface=evidences[0].tail_surface if evidences else "",
                    )
                    fg.candidates.append(candidate)
                candidate.evidence.extend(evidences)
            else:
                fkey = (h, r, t)
                if fkey not in mv_facts:
                    mv_facts[fkey] = FuzzyFact(
                        head_uri=h,
                        relation=r,
                        tail_uri=t,
                        head_surface=evidences[0].head_surface if evidences else "",
                        tail_surface=evidences[0].tail_surface if evidences else "",
                    )
                mv_facts[fkey].evidence.extend(evidences)

        self.fact_groups = list(sv_groups.values())
        self.fuzzy_facts = list(mv_facts.values())

        LOGGER.info(
            "Semantic: %d single-valued groups, %d multi-valued facts",
            len(self.fact_groups), len(self.fuzzy_facts),
        )

    # ──────────────────────────────────────────────────
    # Step 3: Copy Detection
    # ──────────────────────────────────────────────────

    def step3_detect_copies(self) -> int:
        """Run copy detection on all evidence lists.

        Returns the total number of copy groups detected (where group size > 1).
        """
        LOGGER.info("═══ Step 3: Copy Detection ═══")
        n_copy_groups = 0

        # Single-valued fact groups
        for fg in self.fact_groups:
            for candidate in fg.candidates:
                filter_independent_evidence(
                    candidate.evidence, self.copy_sim_threshold
                )
                groups = set(e.copy_group_id for e in candidate.evidence)
                sizes = Counter(e.copy_group_id for e in candidate.evidence)
                n_copy_groups += sum(1 for sz in sizes.values() if sz > 1)

        # Multi-valued facts
        for ff in self.fuzzy_facts:
            filter_independent_evidence(ff.evidence, self.copy_sim_threshold)
            sizes = Counter(e.copy_group_id for e in ff.evidence)
            n_copy_groups += sum(1 for sz in sizes.values() if sz > 1)

        LOGGER.info("Copy groups detected (size > 1): %d", n_copy_groups)
        return n_copy_groups

    # ──────────────────────────────────────────────────
    # Step 4: Hybrid Fusion
    # ──────────────────────────────────────────────────

    def step4_compute_fusion(self) -> None:
        """Compute membership degrees for all facts.

        Single-valued → Dempster's rule with conflict K.
        Multi-valued  → product t-conorm.
        """
        LOGGER.info("═══ Step 4: Hybrid Fusion ═══")
        n_conflicts = 0

        # ── Single-valued: Dempster ──
        for fg in self.fact_groups:
            if not fg.candidates:
                continue

            # Collect mass functions from representative evidence across all candidates
            mass_functions: List[Dict[str, float]] = []
            for candidate in fg.candidates:
                rep_evidence = get_representative_evidence(candidate.evidence)
                for e in rep_evidence:
                    mf = build_mass_function(candidate.tail_uri, e.authority, e.confidence)
                    mass_functions.append(mf)

            if not mass_functions:
                continue

            combined, K = dempster_combine_all(mass_functions)
            fg.conflict_K = K
            fg.needs_review = (K >= self.conflict_K_threshold)
            if fg.needs_review:
                n_conflicts += 1

            # Assign Bel and Pl to each candidate
            bel_pl = compute_bel_pl(combined)
            for candidate in fg.candidates:
                if candidate.tail_uri in bel_pl:
                    candidate.bel, candidate.pl = bel_pl[candidate.tail_uri]

            # Determine winner
            if fg.candidates:
                winner = max(fg.candidates, key=lambda c: c.bel)
                fg.winner_uri = winner.tail_uri
                fg.winner_bel = winner.bel

        # ── Multi-valued: t-conorm ──
        for ff in self.fuzzy_facts:
            rep_evidence = get_representative_evidence(ff.evidence)
            if not rep_evidence:
                ff.mu_F = 0.0
                continue
            pairs = [(e.authority, e.confidence) for e in rep_evidence]
            ff.mu_F = t_conorm_combine(pairs)

        LOGGER.info(
            "Fusion complete: %d single-valued groups (K ≥ %.2f flagged: %d), "
            "%d multi-valued facts",
            len(self.fact_groups), self.conflict_K_threshold, n_conflicts,
            len(self.fuzzy_facts),
        )

    # ──────────────────────────────────────────────────
    # Node membership degrees
    # ──────────────────────────────────────────────────

    def _compute_node_memberships(self) -> None:
        """Compute μ_node for each entity using t-conorm over NER evidence.

        Each mention of an entity is treated as evidence for its existence.
        μ_node = 1 − ∏ⱼ (1 − a_source · confidence_j)
        """
        for uri, data in self._entity_data.items():
            info = data.get("info", {})
            source = info.get("uri_source", "new")
            base_conf = URI_SOURCE_BASE_CONFIDENCE.get(source, 0.70)
            ner_confs = info.get("ner_confidences", [])

            if ner_confs:
                pairs = [(base_conf, c) for c in ner_confs]
                mu_node = t_conorm_combine(pairs)
            else:
                mu_node = base_conf

            data["mu_node"] = round(mu_node, 6)

    # ──────────────────────────────────────────────────
    # Build entity index
    # ──────────────────────────────────────────────────

    def _build_entity_index(self) -> None:
        """Build the final entity index from processed entity data."""
        self._compute_node_memberships()
        self.entity_index = {}
        for uri, data in self._entity_data.items():
            info = data.get("info", {})
            # Remove ner_confidences from info to keep it clean
            clean_info = {k: v for k, v in info.items() if k != "ner_confidences"}
            self.entity_index[uri] = {
                "info":      clean_info,
                "embedding": data.get("embedding"),
                "mu_node":   data.get("mu_node", 0.5),
            }

    # ──────────────────────────────────────────────────
    # Save outputs
    # ──────────────────────────────────────────────────

    def _save_outputs(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. FKG facts JSON (including unified nodes with mu_node)
        nodes_list = []
        for uri, data in sorted(
            self.entity_index.items(), key=lambda x: -x[1].get("mu_node", 0.0)
        ):
            info = data.get("info", {})
            surfaces = info.get("surface_forms", [])
            canonical = surfaces[0] if surfaces else ""
            nodes_list.append({
                "uri":                uri,
                "label":              info.get("label", ""),
                "label_name":         info.get("label_name", ""),
                "fine_grained_label": info.get("fine_grained_label", ""),
                "canonical_surface":  canonical,
                "surface_forms":      surfaces,
                "mu_node":            data.get("mu_node", 0.5),
                "uri_source":         info.get("uri_source", "new"),
                "wikidata_id":        info.get("wikidata_id"),
                "occurrence_count":   info.get("occurrence_count", 1),
                "article_ids":        info.get("article_ids", []),
            })

        fkg = {
            "version": "1.0",
            "fusion_method": "hybrid_dempster_tconorm",
            "params": {
                "copy_sim_threshold":   self.copy_sim_threshold,
                "conflict_K_threshold": self.conflict_K_threshold,
            },
            "nodes":                nodes_list,
            "single_valued_groups": [fg.to_dict() for fg in self.fact_groups],
            "multi_valued_facts":   [ff.to_dict() for ff in self.fuzzy_facts],
            "metadata_facts":       [mf.to_dict() for mf in self.metadata_facts],
        }
        facts_path = out / "fkg_facts.json"
        facts_path.write_text(
            json.dumps(fkg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("FKG facts → %s (%d nodes, %d SV groups, %d MV facts)",
                    facts_path, len(nodes_list), len(self.fact_groups), len(self.fuzzy_facts))

        # 2. Statistics
        n_sv_evidence = sum(
            len(e) for fg in self.fact_groups for c in fg.candidates for e in [c.evidence]
        )
        n_mv_evidence = sum(len(ff.evidence) for ff in self.fuzzy_facts)

        stats = {
            "n_entities":               len(self.entity_index),
            "n_single_valued_groups":   len(self.fact_groups),
            "n_multi_valued_facts":     len(self.fuzzy_facts),
            "n_metadata_facts":         len(self.metadata_facts),
            "n_evidence_single_valued": n_sv_evidence,
            "n_evidence_multi_valued":  n_mv_evidence,
            "n_conflicts_flagged":      sum(1 for fg in self.fact_groups if fg.needs_review),
            "n_uri_merges":             len(self.uri_redirect),
            "n_ambiguous_pairs":        len(self.unifier.ambiguous_pairs),
            "relation_summary": {},
        }

        # Relation-level summary
        rel_stats: Dict[str, Dict[str, Any]] = {}
        for fg in self.fact_groups:
            r = fg.relation
            if r not in rel_stats:
                rel_stats[r] = {"type": "single_valued", "n_groups": 0,
                                "mean_K": 0.0, "max_K": 0.0, "n_flagged": 0}
            rel_stats[r]["n_groups"] += 1
            rel_stats[r]["mean_K"] += fg.conflict_K
            rel_stats[r]["max_K"] = max(rel_stats[r]["max_K"], fg.conflict_K)
            if fg.needs_review:
                rel_stats[r]["n_flagged"] += 1
        for r, rs in rel_stats.items():
            if rs["n_groups"] > 0:
                rs["mean_K"] = round(rs["mean_K"] / rs["n_groups"], 4)
                rs["max_K"] = round(rs["max_K"], 4)

        for ff in self.fuzzy_facts:
            r = ff.relation
            if r not in rel_stats:
                rel_stats[r] = {"type": "multi_valued", "n_facts": 0,
                                "mean_mu": 0.0, "max_mu": 0.0}
            rel_stats[r]["n_facts"] = rel_stats[r].get("n_facts", 0) + 1
            rel_stats[r]["mean_mu"] = rel_stats[r].get("mean_mu", 0.0) + ff.mu_F
            rel_stats[r]["max_mu"] = max(rel_stats[r].get("max_mu", 0.0), ff.mu_F)
        for r, rs in rel_stats.items():
            if rs.get("n_facts", 0) > 0:
                rs["mean_mu"] = round(rs["mean_mu"] / rs["n_facts"], 4)
                rs["max_mu"] = round(rs["max_mu"], 4)

        stats["relation_summary"] = rel_stats

        stats_path = out / "fkg_stats.json"
        stats_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("FKG stats → %s", stats_path)

        # 3. Conflict report
        report_lines = [
            "=" * 72,
            "FUZZY KNOWLEDGE GRAPH — CONFLICT & REVIEW REPORT",
            "=" * 72,
            "",
        ]
        flagged = [fg for fg in self.fact_groups if fg.needs_review]
        if flagged:
            report_lines.append(f"⚠  {len(flagged)} SINGLE-VALUED GROUPS NEED REVIEW (K ≥ {self.conflict_K_threshold})")
            report_lines.append("")
            for fg in flagged:
                report_lines.append(f"  ({fg.head_surface})  --[{fg.relation}]-->  ?")
                report_lines.append(f"  Head URI  : {fg.head_uri}")
                report_lines.append(f"  Conflict K: {fg.conflict_K:.4f}")
                report_lines.append(f"  Candidates:")
                for c in sorted(fg.candidates, key=lambda x: -x.bel):
                    report_lines.append(
                        f"    • {c.tail_surface:<30}  Bel={c.bel:.4f}  Pl={c.pl:.4f}"
                        f"  ({len(c.evidence)} evidence)"
                    )
                report_lines.append("")
        else:
            report_lines.append("✓  No single-valued conflicts above threshold.")
            report_lines.append("")

        # All fact groups summary
        report_lines.append("-" * 72)
        report_lines.append("ALL SINGLE-VALUED FACT GROUPS:")
        report_lines.append("-" * 72)
        for fg in self.fact_groups:
            flag = " ⚠ REVIEW" if fg.needs_review else ""
            report_lines.append(
                f"  [{fg.relation}] ({fg.head_surface}) → {fg.winner_uri}"
                f"  Bel={fg.winner_bel:.4f}  K={fg.conflict_K:.4f}{flag}"
            )
        report_lines.append("")
        report_lines.append("-" * 72)
        report_lines.append("MULTI-VALUED FACTS (top-20 by μ_F):")
        report_lines.append("-" * 72)
        for ff in sorted(self.fuzzy_facts, key=lambda x: -x.mu_F)[:20]:
            report_lines.append(
                f"  [{ff.relation}] ({ff.head_surface}) → ({ff.tail_surface})"
                f"  μ_F={ff.mu_F:.4f}  ({len(ff.evidence)} evidence)"
            )

        # Ambiguous entities
        if self.unifier.ambiguous_pairs:
            report_lines.append("")
            report_lines.append("-" * 72)
            report_lines.append(f"AMBIGUOUS ENTITY PAIRS ({len(self.unifier.ambiguous_pairs)}):")
            report_lines.append("-" * 72)
            for pair in self.unifier.ambiguous_pairs:
                report_lines.append(
                    f"  {pair['uri_1']}  ↔  {pair['uri_2']}"
                    f"  sim={pair['similarity']:.4f}  label={pair['label']}"
                )
                report_lines.append(
                    f"    surfaces 1: {pair['surfaces_1']}"
                )
                report_lines.append(
                    f"    surfaces 2: {pair['surfaces_2']}"
                )

        report_lines.append("")
        report_lines.append("=" * 72)

        report_path = out / "conflict_report.txt"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        LOGGER.info("Conflict report → %s", report_path)

        # 4. Entity index pkl
        idx_path = out / "entity_index_en.pkl"
        with open(idx_path, "wb") as f:
            pickle.dump(self.entity_index, f)
        LOGGER.info("Entity index → %s  (%d entities)", idx_path, len(self.entity_index))

        # 5. Ambiguous pairs JSON
        if self.unifier.ambiguous_pairs:
            amb_path = out / "ambiguous_pairs.json"
            amb_path.write_text(
                json.dumps(self.unifier.ambiguous_pairs, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            LOGGER.info("Ambiguous pairs → %s", amb_path)

        # 6. NetworkX graph
        if HAS_NX:
            G = nx.MultiDiGraph()
            # Add entity nodes with mu_node
            for uri, data in self.entity_index.items():
                info = data.get("info", {})
                G.add_node(
                    uri,
                    label=", ".join(info.get("surface_forms", [])[:2]),
                    entity_label=info.get("label", ""),
                    uri_source=info.get("uri_source", ""),
                    mu_node=data.get("mu_node", 0.5),
                )
            # Single-valued: add the winner
            for fg in self.fact_groups:
                if fg.winner_uri:
                    G.add_edge(
                        fg.head_uri, fg.winner_uri,
                        relation=fg.relation,
                        mu=fg.winner_bel,
                        conflict_K=fg.conflict_K,
                        fusion="dempster",
                    )
            # Multi-valued
            for ff in self.fuzzy_facts:
                G.add_edge(
                    ff.head_uri, ff.tail_uri,
                    relation=ff.relation,
                    mu=ff.mu_F,
                    fusion="t_conorm",
                )
            # Metadata
            for mf in self.metadata_facts:
                G.add_edge(
                    mf.head_uri, mf.tail_uri,
                    relation=mf.relation,
                    mu=1.0,
                    fusion="metadata",
                )
            nx_path = out / "fkg_networkx.pkl"
            with open(nx_path, "wb") as f:
                pickle.dump(G, f)
            LOGGER.info(
                "NetworkX graph → %s  (%d nodes, %d edges)",
                nx_path, G.number_of_nodes(), G.number_of_edges(),
            )

        # 7. RDF Turtle (.ttl) export
        ttl_path = out / "fkg_global.ttl"
        self._export_rdf_turtle(ttl_path)

    def _export_rdf_turtle(self, ttl_path: Path) -> None:
        """Export the complete Fuzzy Knowledge Graph to standard W3C Turtle (.ttl).

        Nodes carry:
          - rdf:type onto:<Class>
          - rdfs:label "..."@en
          - fkg:muNode "0.9500"^^xsd:decimal
          - fkg:occurrenceCount N
          - fkg:uriSource "wikidata"|"new"|...
          - fkg:wikidataId "Q..." (if available)

        Edges (Triples) are exported in two complementary forms:
          1. Direct triple: <head> onto:<rel> <tail> (for standard SPARQL path traversal)
          2. Fuzzy reification: _:stmt a fkg:FuzzyStatement with fkg:membershipDegree,
             fkg:fusionMethod, and fkg:conflictK (for fuzzy reasoning & queries).
        """
        try:
            from rdflib import Graph, URIRef, Literal, Namespace, BNode
            from rdflib.namespace import RDF, RDFS, XSD
            HAS_RDFLIB = True
        except ImportError:
            HAS_RDFLIB = False

        if HAS_RDFLIB:
            g = Graph()
            FKG = Namespace("http://transmtl.vn/fkg/")
            ONTO = Namespace("http://transmtl.vn/onto/")
            g.bind("fkg", FKG)
            g.bind("onto", ONTO)
            g.bind("rdfs", RDFS)
            g.bind("xsd", XSD)

            def _clean_uri(u: str) -> URIRef:
                u = u.strip().replace(" ", "_")
                if not (u.startswith("http://") or u.startswith("https://")):
                    u = f"http://transmtl.vn/ent/{u}"
                return URIRef(u)

            def _clean_name(n: str) -> str:
                n = re.sub(r"[^\w\-]", "_", str(n).strip())
                return n or "Entity"

            def _get_rel_uri(rel: str) -> URIRef:
                if rel.startswith("http://") or rel.startswith("https://"):
                    return URIRef(rel)
                if rel == "rdf:type":
                    return RDF.type
                return ONTO[_clean_name(rel)]

            # ── Export Nodes ──
            for uri, data in self.entity_index.items():
                node_ref = _clean_uri(uri)
                info = data.get("info", {})
                label = _clean_name(info.get("label", "Entity"))
                g.add((node_ref, RDF.type, ONTO[label]))

                surfaces = info.get("surface_forms", [])
                if surfaces:
                    g.add((node_ref, RDFS.label, Literal(surfaces[0], lang="en")))
                    for alt in surfaces[1:]:
                        g.add((node_ref, RDFS.label, Literal(alt, lang="en")))

                mu_node = data.get("mu_node", 0.5)
                g.add((node_ref, FKG.muNode, Literal(round(mu_node, 4), datatype=XSD.decimal)))
                g.add((node_ref, FKG.occurrenceCount, Literal(info.get("occurrence_count", 1), datatype=XSD.integer)))
                g.add((node_ref, FKG.uriSource, Literal(info.get("uri_source", "new"))))
                if info.get("wikidata_id"):
                    g.add((node_ref, FKG.wikidataId, Literal(info["wikidata_id"])))

            # ── Export Single-Valued Facts (Dempster winners) ──
            for fg in self.fact_groups:
                if not fg.winner_uri:
                    continue
                h_ref = _clean_uri(fg.head_uri)
                t_ref = _clean_uri(fg.winner_uri)
                r_ref = _get_rel_uri(fg.relation)

                # Direct triple
                g.add((h_ref, r_ref, t_ref))

                # Fuzzy statement reification
                stmt = BNode()
                g.add((stmt, RDF.type, FKG.FuzzyStatement))
                g.add((stmt, RDF.subject, h_ref))
                g.add((stmt, RDF.predicate, r_ref))
                g.add((stmt, RDF.object, t_ref))
                g.add((stmt, FKG.membershipDegree, Literal(round(fg.winner_bel, 4), datatype=XSD.decimal)))
                g.add((stmt, FKG.fusionMethod, Literal("dempster")))
                g.add((stmt, FKG.conflictK, Literal(round(fg.conflict_K, 4), datatype=XSD.decimal)))

            # ── Export Multi-Valued Facts (t-conorm) ──
            for ff in self.fuzzy_facts:
                h_ref = _clean_uri(ff.head_uri)
                t_ref = _clean_uri(ff.tail_uri)
                r_ref = _get_rel_uri(ff.relation)

                # Direct triple
                g.add((h_ref, r_ref, t_ref))

                # Fuzzy statement reification
                stmt = BNode()
                g.add((stmt, RDF.type, FKG.FuzzyStatement))
                g.add((stmt, RDF.subject, h_ref))
                g.add((stmt, RDF.predicate, r_ref))
                g.add((stmt, RDF.object, t_ref))
                g.add((stmt, FKG.membershipDegree, Literal(round(ff.mu_F, 4), datatype=XSD.decimal)))
                g.add((stmt, FKG.fusionMethod, Literal("t_conorm")))

            # ── Export Metadata Facts ──
            for mf in self.metadata_facts:
                h_ref = _clean_uri(mf.head_uri)
                t_ref = _clean_uri(mf.tail_uri)
                r_ref = _get_rel_uri(mf.relation)
                g.add((h_ref, r_ref, t_ref))

            g.serialize(destination=str(ttl_path), format="turtle")
            LOGGER.info("RDF Turtle export → %s  (%d RDF triples)", ttl_path, len(g))
        else:
            # Fallback pure-Python generator
            lines = [
                "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
                "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
                "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
                "@prefix onto: <http://transmtl.vn/onto/> .",
                "@prefix fkg: <http://transmtl.vn/fkg/> .",
                "",
            ]
            for uri, data in self.entity_index.items():
                info = data.get("info", {})
                label = re.sub(r"[^\w\-]", "_", str(info.get("label", "Entity")).strip()) or "Entity"
                surfaces = info.get("surface_forms", [])
                name = surfaces[0].replace('"', '\\"') if surfaces else ""
                mu = data.get("mu_node", 0.5)
                lines.append(f"<{uri}> a onto:{label} ;")
                if name:
                    lines.append(f'    rdfs:label "{name}"@en ;')
                lines.append(f'    fkg:muNode "{mu:.4f}"^^xsd:decimal .')
                lines.append("")
            ttl_path.write_text("\n".join(lines), encoding="utf-8")
            LOGGER.info("RDF Turtle export (fallback) → %s", ttl_path)

    # ──────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────

    def _print_summary(self) -> None:
        print()
        print("=" * 62)
        print("  MODULE 5 — FUZZY KNOWLEDGE GRAPH SUMMARY")
        print("=" * 62)
        print(f"  Entities (unified)          : {len(self.entity_index):,}")
        print(f"  URI merges applied          : {len(self.uri_redirect):,}")
        print(f"  Ambiguous entity pairs      : {len(self.unifier.ambiguous_pairs):,}")
        print()
        print(f"  Single-valued fact groups   : {len(self.fact_groups):,}")
        n_conflicts = sum(1 for fg in self.fact_groups if fg.needs_review)
        print(f"    Conflicts (K ≥ {self.conflict_K_threshold:.2f})     : {n_conflicts}")
        if self.fact_groups:
            mean_K = sum(fg.conflict_K for fg in self.fact_groups) / len(self.fact_groups)
            print(f"    Mean conflict K          : {mean_K:.4f}")
        print()
        print(f"  Multi-valued facts          : {len(self.fuzzy_facts):,}")
        if self.fuzzy_facts:
            mean_mu = sum(ff.mu_F for ff in self.fuzzy_facts) / len(self.fuzzy_facts)
            print(f"    Mean μ_F                 : {mean_mu:.4f}")
        print()
        print(f"  Metadata facts (dedup)      : {len(self.metadata_facts):,}")
        print()

        # Top single-valued groups
        if self.fact_groups:
            print("  Single-valued winners:")
            for fg in self.fact_groups:
                flag = " ⚠" if fg.needs_review else ""
                print(
                    f"    [{fg.relation}] {fg.head_surface} → "
                    f"{fg.winner_uri}  Bel={fg.winner_bel:.4f}  K={fg.conflict_K:.4f}{flag}"
                )
            print()

        # Top multi-valued facts
        if self.fuzzy_facts:
            print("  Top multi-valued facts (by μ_F):")
            for ff in sorted(self.fuzzy_facts, key=lambda x: -x.mu_F)[:10]:
                print(
                    f"    [{ff.relation}] {ff.head_surface} → "
                    f"{ff.tail_surface}  μ_F={ff.mu_F:.4f}"
                )

        print("=" * 62)
        print()

    # ──────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────

    def build(
        self,
        triple_files:    List[str],
        linked_files:    List[str],
        registry_path:   Optional[str] = None,
        output_dir:      str = "./data/fkg_en",
    ) -> None:
        """Run the full 4-step FKG construction pipeline."""
        LOGGER.info("╔══════════════════════════════════════════════════╗")
        LOGGER.info("║  Module 5 EN — FKG Builder START                ║")
        LOGGER.info("╚══════════════════════════════════════════════════╝")

        # Load data
        self._load_triples(triple_files)
        self._load_entities(linked_files, registry_path)

        # Step 1: Entity Unification
        self.step1_unify_entities()

        # Step 2: Triple Merging
        self.step2_merge_triples()

        # Step 3: Copy Detection
        self.step3_detect_copies()

        # Step 4: Fusion
        self.step4_compute_fusion()

        # Build entity index with node memberships
        self._build_entity_index()

        # Save outputs
        self._save_outputs(output_dir)
        self._print_summary()

        LOGGER.info("╔══════════════════════════════════════════════════╗")
        LOGGER.info("║  Module 5 EN — FKG Builder DONE                 ║")
        LOGGER.info("╚══════════════════════════════════════════════════╝")


# ══════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════

def _discover_files(base_dir: Path) -> Tuple[List[str], List[str], Optional[str]]:
    """Auto-discover triple, linked-output, and registry files."""
    triple_files = sorted(str(p) for p in base_dir.glob("*_triples_en.json"))
    linked_files = sorted(str(p) for p in base_dir.glob("*_linked_output.json"))
    registry = base_dir / "entity_registry_en.pkl"
    registry_path = str(registry) if registry.exists() else None
    return triple_files, linked_files, registry_path


if __name__ == "__main__":
    import sys

    base = Path(__file__).resolve().parent
    triple_files, linked_files, registry_path = _discover_files(base)

    if not triple_files:
        print("ERROR: No *_triples_en.json files found in", base)
        sys.exit(1)

    LOGGER.info("Discovered %d triple files, %d linked files", len(triple_files), len(linked_files))

    builder = FKGBuilder()
    builder.build(
        triple_files=triple_files,
        linked_files=linked_files,
        registry_path=registry_path,
        output_dir=str(base.parent / "data" / "fkg_en"),
    )
