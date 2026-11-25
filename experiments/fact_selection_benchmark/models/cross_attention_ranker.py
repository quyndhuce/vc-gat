#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/models/cross_attention_ranker.py
=============================================================================
B5: Cross-Attention Ranker (Supervised Non-Graph Multimodal Baseline).

Performs fine-grained token-level cross-attention between fact claim tokens
and video temporal frame representations:
  1. Fact query attends to temporal video frames:
     Q = W_q * t_f, K = W_k * V_frames, V = W_v * V_frames
  2. Multi-Head Cross Attention captures fine-grained visual interactions.
  3. Contextualized fact representation feeds into scoring MLP.
  4. Supervised on Gold Train (702 clips) with Margin / BCE Loss.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


class CrossAttentionRankerModel(nn.Module):
    """Supervised Cross-Attention Ranker Network."""

    def __init__(self, d_model: int = 768, num_heads: int = 4, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Scorer: [t_f (768), context (768), t_f * context (768)] = 2304
        self.scorer = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, v_frames: torch.Tensor, t_facts: torch.Tensor) -> torch.Tensor:
        """
        v_frames: [K, D] (8 frames)
        t_facts: [N, D] (N candidate facts)
        Returns: [N] logits
        """
        # Batch dimension: [1, K, D] and [N, 1, D]
        if v_frames.dim() == 2:
            v_frames = v_frames.unsqueeze(0)  # [1, K, D]
        N = t_facts.size(0)
        q = t_facts.unsqueeze(1)             # [N, 1, D]
        kv = v_frames.expand(N, -1, -1)      # [N, K, D]

        # Cross attention: query attends to video frames
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)  # [N, 1, D]
        ctx = self.norm1(attn_out.squeeze(1) + t_facts)            # [N, D]

        feat = torch.cat([t_facts, ctx, t_facts * ctx], dim=-1)   # [N, 3D]
        return self.scorer(feat).squeeze(-1)


class CrossAttentionRanker:
    """Inference wrapper for Cross-Attention Ranker."""

    def __init__(
        self,
        video_frames: Dict[str, np.ndarray],
        fact_features: Dict[str, np.ndarray],
        model: Optional[CrossAttentionRankerModel] = None,
        device: torch.device = torch.device("cpu")
    ):
        self.video_frames = video_frames
        self.fact_features = fact_features
        self.model = model
        self.device = device
        if self.model is not None:
            self.model.to(device).eval()

    def score_clip_facts(self, clip_id: str, candidate_fact_ids: List[str]) -> Dict[str, float]:
        frames = self.video_frames.get(clip_id)
        if frames is None:
            return {fid: 0.0 for fid in candidate_fact_ids}

        t_list = []
        valid_fids = []
        for fid in candidate_fact_ids:
            if fid in self.fact_features:
                t_list.append(self.fact_features[fid])
                valid_fids.append(fid)

        if not t_list or self.model is None:
            return {fid: 0.0 for fid in candidate_fact_ids}

        t_arr = np.array(t_list, dtype=np.float32)

        with torch.no_grad():
            v_t = torch.from_numpy(frames).float().to(self.device)
            t_t = torch.from_numpy(t_arr).float().to(self.device)
            logits = self.model(v_t, t_t)
            scores = logits.cpu().numpy()

        res = {}
        for fid, s in zip(valid_fids, scores):
            res[fid] = float(s)
        for fid in candidate_fact_ids:
            if fid not in res:
                res[fid] = 0.0
        return res
