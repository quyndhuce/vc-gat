#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/models/internvideo2_ranker.py
=============================================================================
B3: InternVideo2 (Video Foundation Model with Temporal Dynamics Baseline).

Captures temporal frame transitions and cross-frame attention for video-text matching:
  1. Zero-shot Mode: Multi-frame temporal cosine retrieval s_ZS(V, f).
  2. Fine-Tuned / Supervised Mode: Temporal aggregation + ranking projection head:
     s_FT(V, f) = MLP([TemporalPool(V_frames); t_f; Interaction]).
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


class InternVideo2RankingHead(nn.Module):
    """Supervised temporal ranking head over video frame sequence and fact claims."""

    def __init__(self, d_model: int = 768, num_heads: int = 4, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        # Temporal self-attention over video frames
        self.temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(self.temporal_layer, num_layers=1)
        
        # Scorer: [v_temporal (768), t (768), v_temporal * t (768)] = 2304
        self.scorer = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, v_frames: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        v_frames: [K, D] (e.g. 8 frames)
        t: [N, D] (candidate fact embeddings)
        """
        if v_frames.dim() == 2:
            v_frames = v_frames.unsqueeze(0)  # [1, K, D]
        # Temporal contextualization
        v_temp = self.temporal_encoder(v_frames)  # [1, K, D]
        v_pool = v_temp.mean(dim=1)  # [1, D]
        v_pool = F.normalize(v_pool, p=2, dim=-1)

        N = t.size(0)
        v_exp = v_pool.expand(N, -1)
        feat = torch.cat([v_exp, t, v_exp * t], dim=-1)
        return self.scorer(feat).squeeze(-1)


class InternVideo2Ranker:
    """Inference wrapper for InternVideo2 Zero-shot and Supervised modes."""

    def __init__(
        self,
        video_frames: Dict[str, np.ndarray],
        fact_features: Dict[str, np.ndarray],
        head_model: Optional[InternVideo2RankingHead] = None,
        device: torch.device = torch.device("cpu")
    ):
        self.video_frames = video_frames
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
        frames = self.video_frames.get(clip_id)
        if frames is None:
            return {fid: 0.0 for fid in candidate_fact_ids}

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
            # Temporal mean pooling with frame-maximum semantic alignment
            # For each frame, compute dot product, then take max or mean across frames
            # max-pooling across frames captures temporally localized facts
            f_norm = frames / (np.linalg.norm(frames, axis=1, keepdims=True) + 1e-8)  # [8, D]
            t_norm = t_arr / (np.linalg.norm(t_arr, axis=1, keepdims=True) + 1e-8)   # [N, D]
            sim_matrix = np.dot(t_norm, f_norm.T)  # [N, 8]
            # Combination of temporal mean and temporal peak
            scores = 0.6 * np.max(sim_matrix, axis=1) + 0.4 * np.mean(sim_matrix, axis=1)
        else:
            with torch.no_grad():
                v_t = torch.from_numpy(frames).float().to(self.device)
                t_t = torch.from_numpy(t_arr).float().to(self.device)
                logits = self.head_model(v_t, t_t)
                scores = logits.cpu().numpy()

        res = {}
        for fid, s in zip(valid_fids, scores):
            res[fid] = float(s)
        for fid in candidate_fact_ids:
            if fid not in res:
                res[fid] = 0.0
        return res
