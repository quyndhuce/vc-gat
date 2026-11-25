#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/models/graphormer_ranker.py
=============================================================================
B7: Graphormer + Video Fusion (Graph Transformer Baseline).

Graphormer architecture (Ying et al., NeurIPS 2021) adapted for video-conditioned KG ranking:
  1. Centrality Encoding: Learnable degree bias embeddings added to initial fact embeddings.
  2. Spatial Encoding: Learnable spatial bias based on shortest-path distance phi(i, j):
     Attn_ij = (Q_i * K_j^T) / sqrt(d) + b_{phi(i, j)}
  3. Video Conditioning: Video context vector v in R^768 injected into Transformer self-attention.
  4. Readout Head: MLP scoring fact nodes: s(V, f_i) = MLP([h_i^(out); v; h_i^(out) * v]).
  5. Supervised on Gold Train (702 clips).
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any


class GraphormerLayer(nn.Module):
    """Standard Graphormer layer with spatial distance bias and video conditioning."""

    def __init__(self, d_model: int = 768, num_heads: int = 4, max_distance: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Spatial encoding: learnable bias for each shortest-path distance (0..max_distance)
        self.spatial_bias = nn.Embedding(max_distance + 1, num_heads)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, shortest_path: torch.Tensor, v_ctx: torch.Tensor) -> torch.Tensor:
        """
        h: [N, D]
        shortest_path: [N, N] (shortest path distance)
        v_ctx: [1, D] (video context)
        """
        N = h.size(0)
        h_norm = self.norm1(h + 0.2 * v_ctx.expand(N, -1))

        Q = self.q_proj(h_norm).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, d]
        K = self.k_proj(h_norm).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, d]
        V = self.v_proj(h_norm).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, d]

        # Raw attention scores: [H, N, N]
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale

        # Add learnable spatial bias: shortest_path is [N, N], spatial_bias is [N, N, H] -> [H, N, N]
        sp_bias = self.spatial_bias(shortest_path).permute(2, 0, 1)
        attn_weights = F.softmax(scores + sp_bias, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context: [H, N, d] -> [N, D]
        context = torch.bmm(attn_weights, V).transpose(0, 1).contiguous().view(N, self.d_model)
        h = h + self.dropout(self.out_proj(context))
        
        # FFN
        h = h + self.dropout(self.ffn(self.norm2(h)))
        return h


class GraphormerVideoFusionModel(nn.Module):
    """Full Graphormer Architecture with Video Conditioning."""

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 4,
        num_layers: int = 2,
        max_degree: int = 64,
        max_distance: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.centrality_encoding = nn.Embedding(max_degree + 1, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.v_norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([
            GraphormerLayer(d_model=d_model, num_heads=num_heads, max_distance=max_distance, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Scorer: [h_out (768), v (768), h_out * v (768)] = 2304
        self.scorer = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, h: torch.Tensor, shortest_path: torch.Tensor, degrees: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 1:
            v = v.unsqueeze(0)
        v_ctx = self.v_norm(self.v_proj(v))

        # Add centrality encoding
        deg_clamped = torch.clamp(degrees, 0, 64)
        h = h + self.centrality_encoding(deg_clamped)

        for layer in self.layers:
            h = layer(h, shortest_path, v_ctx)

        N = h.size(0)
        v_exp = v.expand(N, -1)
        feat = torch.cat([h, v_exp, h * v_exp], dim=-1)
        return self.scorer(feat).squeeze(-1)


class GraphormerRanker:
    """Inference wrapper for Graphormer + Video Fusion."""

    def __init__(
        self,
        video_features: Dict[str, np.ndarray],
        fact_features: Dict[str, np.ndarray],
        subgraphs: Dict[str, Dict[str, Any]],
        model: Optional[GraphormerVideoFusionModel] = None,
        device: torch.device = torch.device("cpu")
    ):
        self.video_features = video_features
        self.fact_features = fact_features
        self.subgraphs = subgraphs
        self.model = model
        self.device = device
        if self.model is not None:
            self.model.to(device).eval()

    def score_clip_facts(
        self,
        clip_id: str,
        landmark_id: str,
        candidate_fact_ids: List[str]
    ) -> Dict[str, float]:
        v = self.video_features.get(clip_id)
        if v is None or self.model is None:
            return {fid: 0.0 for fid in candidate_fact_ids}

        g = self.subgraphs.get(landmark_id)
        if g is None:
            return {fid: 0.0 for fid in candidate_fact_ids}

        fids = g["fact_ids"]
        sp = g["shortest_path"].to(self.device)
        deg = g["degrees"].to(self.device)

        h_list = [self.fact_features.get(fid, np.zeros(768, dtype=np.float32)) for fid in fids]
        h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(self.device)
        v_tensor = torch.from_numpy(v).float().to(self.device)

        with torch.no_grad():
            logits = self.model(h_tensor, sp, deg, v_tensor)
            scores_arr = logits.cpu().numpy()

        fid2score = {fid: float(scores_arr[i]) for i, fid in enumerate(fids)}
        return {fid: fid2score.get(fid, 0.0) for fid in candidate_fact_ids}
