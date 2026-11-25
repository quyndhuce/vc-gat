#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/models/siglip2_ranker.py
=============================================================================
B2: SigLIP2 (Vision-Language Dual Encoder Retrieval Baseline).

Supports:
  1. Zero-shot Mode: Direct cosine similarity between video embedding v in R^768
     and fact embedding t in R^768: s_ZS(v, t) = <v, t>
  2. Fine-Tuned / Supervised Mode: Multimodal ranking projection head trained on
     Gold Train clips (BCE / Margin Ranking Loss):
     s_FT(v, t) = MLP([v; t; v * t; |v - t|])
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


class SigLIP2RankingHead(nn.Module):
    """Supervised ranking head over frozen or fine-tuned dual-encoder features."""

    def __init__(self, d_model: int = 768, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        # Input: [v (768), t (768), v * t (768), |v - t| (768)] = 3072
        self.mlp = nn.Sequential(
            nn.Linear(4 * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        v: [B, D] or [1, D]
        t: [N, D]
        Returns: [N] logits
        """
        if v.dim() == 1:
            v = v.unsqueeze(0)
        N = t.size(0)
        v_exp = v.expand(N, -1)
        feat = torch.cat([v_exp, t, v_exp * t, torch.abs(v_exp - t)], dim=-1)
        return self.mlp(feat).squeeze(-1)


class SigLIP2Ranker:
    """Inference wrapper for SigLIP2 Zero-shot and Supervised modes."""

    def __init__(
        self,
        video_features: Dict[str, np.ndarray],
        fact_features: Dict[str, np.ndarray],
        head_model: Optional[SigLIP2RankingHead] = None,
        device: torch.device = torch.device("cpu")
    ):
        self.video_features = video_features
        self.fact_features = fact_features
        self.head_model = head_model
        self.device = device
        if self.head_model is not None:
            self.head_model.to(device).eval()

    def score_clip_facts(
        self,
        clip_id: str,
        candidate_fact_ids: List[str],
        mode: str = "zero_shot"
    ) -> Dict[str, float]:
        v = self.video_features.get(clip_id)
        if v is None:
            return {fid: 0.0 for fid in candidate_fact_ids}

        # Collect fact vectors
        t_list = []
        valid_fids = []
        for fid in candidate_fact_ids:
            if fid in self.fact_features:
                t_list.append(self.fact_features[fid])
                valid_fids.append(fid)

        if not t_list:
            return {fid: 0.0 for fid in candidate_fact_ids}

        t_arr = np.array(t_list, dtype=np.float32)

        if mode == "zero_shot" or self.head_model is None:
            # Cosine similarity
            v_norm = v / (np.linalg.norm(v) + 1e-8)
            t_norm = t_arr / (np.linalg.norm(t_arr, axis=1, keepdims=True) + 1e-8)
            scores = np.dot(t_norm, v_norm)
        else:
            # Supervised ranking head
            with torch.no_grad():
                v_t = torch.from_numpy(v).float().to(self.device)
                t_t = torch.from_numpy(t_arr).float().to(self.device)
                logits = self.head_model(v_t, t_t)
                scores = logits.cpu().numpy()

        res = {}
        for fid, s in zip(valid_fids, scores):
            res[fid] = float(s)
        # Any missing facts get 0.0
        for fid in candidate_fact_ids:
            if fid not in res:
                res[fid] = 0.0
        return res
