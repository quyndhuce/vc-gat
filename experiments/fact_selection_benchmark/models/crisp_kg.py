#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/models/crisp_kg.py
=============================================================================
B1: Crisp-KG (Static Symbolic Knowledge Graph Baseline).

Rank facts purely based on static knowledge graph properties:
  - Degree centrality in the landmark subgraph
  - Entity prominence and static frequency
  - Completely invariant to video input: s(V_1, f) == s(V_2, f)
=============================================================================
"""

from typing import Dict, List, Any
import numpy as np


class CrispKGRanker:
    """Static Knowledge Graph Ranker without video conditioning."""

    def __init__(self, candidate_facts: Dict[str, Dict[str, Any]]):
        self.candidate_facts = candidate_facts
        # Precompute static centrality/prominence score for each fact
        self.static_scores = {}
        for fid, f in candidate_facts.items():
            # Higher score for primary relations (located_in, founded_by) vs peripheral
            pred = f.get("predicate", "").lower()
            base = 1.0
            if pred in ["located_in", "part_of", "has_component"]:
                base = 2.0
            elif pred in ["founded_by", "built_by", "architectural_style"]:
                base = 1.8
            elif pred in ["made_of", "features", "surrounded_by"]:
                base = 1.5
            # Add length/frequency weight
            self.static_scores[fid] = base + 0.01 * len(f.get("fact_text", ""))

    def score_clip_facts(self, clip_id: str, candidate_fact_ids: List[str]) -> Dict[str, float]:
        """Returns static scores for candidates (identical across all clips of same landmark)."""
        return {fid: self.static_scores.get(fid, 1.0) for fid in candidate_fact_ids}
