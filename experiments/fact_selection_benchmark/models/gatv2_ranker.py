#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/models/gatv2_ranker.py
=============================================================================
B6: GATv2 + Video Fusion (Dynamic Graph Attention Network Baseline).

Addresses the static attention limitation of traditional GAT (Brody et al., 2021):
  1. Triple KG -> Graph Representation:
     - Nodes: Candidate facts in landmark L with text embeddings h_i in R^768.
     - Edges: Shared entity connections (subject/object co-occurrence).
  2. Video Conditioning:
     - Video embedding v in R^768 is projected and fused into node states.
  3. Dynamic Attention Message Passing:
     - 2-layer GATv2Conv where attention weights dynamically adapt:
       e_ij = a^T LeakyReLU(W_s h_i + W_t h_j + W_v v)
  4. Scoring Head:
     - Readout: s(V, f_i) = MLP([h_i^(2); v; h_i^(2) * v]).
  5. Supervised on Gold Train (702 clips).
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np
from typing import Dict, List, Optional, Any


class GATv2VideoFusionModel(nn.Module):
    """GATv2 with Video Feature Injection and Dynamic Attention."""

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        head_dim = d_model // num_heads

        # Layer 1
        self.conv1 = GATv2Conv(
            in_channels=d_model,
            out_channels=head_dim,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            share_weights=False
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Layer 2
        self.conv2 = GATv2Conv(
            in_channels=d_model,
            out_channels=head_dim,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            share_weights=False
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Video context projection
        self.v_proj = nn.Linear(d_model, d_model)
        self.v_norm = nn.LayerNorm(d_model)

        # Scorer: [h_i^(2) (768), v (768), h_i^(2) * v (768)] = 2304
        self.scorer = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        h: [N, D] (node fact embeddings)
        edge_index: [2, E] (graph edges)
        v: [D] or [1, D] (video embedding)
        Returns: [N] continuous logits
        """
        if v.dim() == 1:
            v = v.unsqueeze(0)
        N = h.size(0)
        v_ctx = self.v_norm(self.v_proj(v)).expand(N, -1)

        # Inject video context additively into node representations prior to dynamic attention
        x = h + 0.3 * v_ctx
        
        # Layer 1: GATv2 dynamic message passing
        h1 = self.conv1(x, edge_index)
        h1 = self.norm1(F.gelu(h1) + h)

        # Layer 2
        h2 = self.conv2(h1, edge_index)
        h2 = self.norm2(F.gelu(h2) + h1)

        # Final video fusion readout
        v_exp = v.expand(N, -1)
        feat = torch.cat([h2, v_exp, h2 * v_exp], dim=-1)
        return self.scorer(feat).squeeze(-1)


class GATv2Ranker:
    """Inference wrapper for GATv2 + Video Fusion."""

    def __init__(
        self,
        video_features: Dict[str, np.ndarray],
        fact_features: Dict[str, np.ndarray],
        subgraphs: Dict[str, Dict[str, Any]],
        model: Optional[GATv2VideoFusionModel] = None,
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

        # Look up landmark subgraph
        g = self.subgraphs.get(landmark_id)
        if g is None:
            # Fallback uniform
            return {fid: 0.0 for fid in candidate_fact_ids}

        fids = g["fact_ids"]
        edge_index = g["edge_index"].to(self.device)
        
        # Node features
        h_list = [self.fact_features.get(fid, np.zeros(768, dtype=np.float32)) for fid in fids]
        h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(self.device)
        v_tensor = torch.from_numpy(v).float().to(self.device)

        with torch.no_grad():
            logits = self.model(h_tensor, edge_index, v_tensor)
            scores_arr = logits.cpu().numpy()

        fid2score = {fid: float(scores_arr[i]) for i, fid in enumerate(fids)}
        return {fid: fid2score.get(fid, 0.0) for fid in candidate_fact_ids}
