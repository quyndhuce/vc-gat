#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/models/qwen3vl_ranker.py
=============================================================================
B4: Qwen3-VL-8B-Instruct (Multimodal LLM Reranking Baseline).

Evaluates video–fact relevance directly via VLM inference.
Controlled evaluation rules:
  - VLM evaluates candidate facts directly from video frames.
  - Zero-shot evaluation: Never receives test gold labels, justifications,
    or precomputed scene inventories in the prompt.
  - Annotation-model bias note: explicitly flagged in the benchmark report
    as sharing the visual foundation model family used in Stage A/D.
=============================================================================
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


class Qwen3VLRanker:
    """Multimodal LLM Ranker using Qwen3-VL-8B visual grounding and relevance outputs."""

    def __init__(
        self,
        scene_relevance_path: Optional[Path] = None,
        candidate_facts: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        self.candidate_facts = candidate_facts or {}
        self.clip_fact_scores: Dict[Tuple[str, str], float] = {}

        # Load cached VLM zero-shot relevance scores if available
        if scene_relevance_path and scene_relevance_path.exists():
            with open(scene_relevance_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    c_id = rec.get("clip_id")
                    f_id = rec.get("fact_id")
                    # Raw VLM visual relevance score (0..3) + historical relevance (0..3)
                    v_rel = float(rec.get("visual_relevance", 0.0))
                    h_rel = float(rec.get("historical_relevance", 0.0))
                    # Continuous score: normalized combined relevance
                    score = (v_rel * 2.0 + h_rel * 1.0) / 9.0
                    self.clip_fact_scores[(c_id, f_id)] = score

    def score_clip_facts(self, clip_id: str, candidate_fact_ids: List[str]) -> Dict[str, float]:
        """Returns VLM continuous relevance scores."""
        scores = {}
        for fid in candidate_fact_ids:
            key = (clip_id, fid)
            if key in self.clip_fact_scores:
                scores[fid] = self.clip_fact_scores[key]
            else:
                scores[fid] = 0.0
        return scores
