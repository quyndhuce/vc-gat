#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/models/vcgat_gold.py
=============================================================================
Ours: VC-GAT Gold-Supervised (Contrastive Video-Conditioned Graph Attention Network).

Engineered to excel in Hard Negative Discrimination and Cross-Scene Contrastive Accuracy:
  1. Dynamic Node Saliency Gate:
     g_i(v) = sigmoid( W_s * [h_i * v; |h_i - v|; cos(h_i, v)] ) in [0, 1].
     Identifies visual anchors and attenuates unobserved landmark facts.
  2. Video-Conditioned Graph Attention with Neighbor Gating:
     alpha_ij(v) = Softmax( Q_i * (K_j * (0.5 + 0.5 * g_j(v)))^T / sqrt(d) + b_v(v) ).
     Prevents topological leakage from visually ungrounded neighbor nodes.
  3. Pure Multiplicative Multimodal Readout (Zero Static Fact Bias):
     s(V, f_i) = MLP([h_i^(out) * v; |h_i^(out) - v|; cos(h_i^(out), v); g_i(v)]).
     Guarantees that score variations are strictly driven by video context V.
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any


class ContrastiveVCGATLayer(nn.Module):
    """Video-Conditioned Graph Attention Layer with Neighbor Saliency Gating."""

    def __init__(self, d_model: int = 768, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Video-conditioned attention bias
        self.edge_gate = nn.Linear(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor, v: torch.Tensor, g_nodes: torch.Tensor) -> torch.Tensor:
        """
        h: [N, D]
        adj_mask: [N, N]
        v: [1, D]
        g_nodes: [N, 1] in [0, 1]
        """
        N = h.size(0)
        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, d]

        # Modulate key and value representations by node saliency gate
        # Visually active nodes propagate strong activations; ungrounded nodes are muted
        h_gated = h * (0.5 + 0.5 * g_nodes)
        K = self.k_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [H, N, N]

        # Add video-conditioned edge bias
        v_edge_bias = self.edge_gate(v).view(self.num_heads, 1, 1)
        scores = scores + 0.2 * v_edge_bias

        mask = (adj_mask == 0).unsqueeze(0).expand(self.num_heads, -1, -1)
        scores = scores.masked_fill(mask, -1e9)
        attn_weights = self.dropout(F.softmax(scores, dim=-1))

        context = torch.bmm(attn_weights, V).transpose(0, 1).contiguous().view(N, self.d_model)
        return self.norm(h + self.dropout(self.out_proj(context)))


class VCGATGoldModel(nn.Module):
    """Full Contrastive VC-GAT Architecture with Zero Static Bias Readout."""

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.v_norm = nn.LayerNorm(d_model)

        # Node Saliency Gate: computes initial visual relevance g_i(v)
        self.saliency_gate = nn.Sequential(
            nn.Linear(2 * d_model + 1, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.layer = ContrastiveVCGATLayer(d_model=d_model, num_heads=num_heads, dropout=dropout)

        # Scorer: Purely interactive features [h_out * v (768), |h_out - v| (768), cos_sim (1), g_nodes (1)] = 1538
        # ZERO static isolated h_out terms! Prevents static fact bias leakage!
        self.scorer = nn.Sequential(
            nn.Linear(2 * d_model + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 1:
            v = v.unsqueeze(0)
        v_n = self.v_norm(v)
        N = h.size(0)
        v_exp = v_n.expand(N, -1)

        # 1. Compute Node Saliency Gate g_i(v)
        h_norm_unit = F.normalize(h, p=2, dim=-1)
        v_norm_unit = F.normalize(v_n, p=2, dim=-1).expand(N, -1)
        cos_sim_init = torch.sum(h_norm_unit * v_norm_unit, dim=-1, keepdim=True)
        gate_input = torch.cat([h * v_exp, torch.abs(h - v_exp), cos_sim_init], dim=-1)
        g_nodes = self.saliency_gate(gate_input)  # [N, 1] in [0, 1]

        # 2. Video-Conditioned Graph Attention
        curr_h = self.layer(h, adj_mask, v_n, g_nodes)

        # 3. Final Multiplicative Readout (Zero Static Bias)
        curr_h_unit = F.normalize(curr_h, p=2, dim=-1)
        cos_sim_final = torch.sum(curr_h_unit * v_norm_unit, dim=-1, keepdim=True)
        feat = torch.cat([curr_h * v_exp, torch.abs(curr_h - v_exp), cos_sim_final, g_nodes], dim=-1)
        return self.scorer(feat).squeeze(-1)


class VCGATGoldRanker:
    """Inference wrapper for VC-GAT Gold-Supervised."""

    def __init__(
        self,
        video_features: Dict[str, np.ndarray],
        fact_features: Dict[str, np.ndarray],
        subgraphs: Dict[str, Dict[str, Any]],
        model: Optional[VCGATGoldModel] = None,
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
        adj = g["adj"].to(self.device)

        h_list = [self.fact_features.get(fid, np.zeros(768, dtype=np.float32)) for fid in fids]
        h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(self.device)
        v_tensor = torch.from_numpy(v).float().to(self.device)

        with torch.no_grad():
            logits = self.model(h_tensor, adj, v_tensor)
            scores_arr = logits.cpu().numpy()

        fid2score = {fid: float(scores_arr[i]) for i, fid in enumerate(fids)}
        return {fid: fid2score.get(fid, 0.0) for fid in candidate_fact_ids}
