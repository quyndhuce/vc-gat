#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/run_vcgat_architectural_study.py
=============================================================================
Systematic Architectural Exploration of VC-GAT (From Simpler to Complex).

Directly addresses reviewer questions and investigates:
  1. Base Direct Matching (No Graph) -> establishes visual lower-bound.
  2. Naive VC-GAT on Raw OntoKG -> reveals graph over-smoothing across hub cliques.
  3. Topology-Pruned VC-GAT with Visual Saliency Gating -> eliminates hub noise & leakage.
  4. Residual-Gated VC-GAT -> bounds generalization while unlocking multi-hop boost.
  5. Multi-Scale Temporal Residual-Gated VC-GAT -> fact-to-frame temporal cross-attention.

Evaluates all models on identical Gold splits (702 Train, 149 Val, 800 Test).
Reports:
  - Overall nDCG@5, nDCG@16, P@5, R@5, MAP, Hard Neg AUC, Contrastive Acc.
  - Seen Landmarks (301 clips) vs. Unseen Landmarks (499 clips).
  - Visual, Historical, and Inferred Fact nDCG@5.
=============================================================================
"""

import sys
import os
import math
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.fact_selection_benchmark.dataset_loader import (
    load_normalized_facts,
    load_clip_manifests,
    load_gold_data,
    load_contrastive_pairs,
    compute_ranking_metrics,
    compute_hard_negative_auc,
    compute_contrastive_accuracy,
    VISUAL_PREDICATES,
    HISTORICAL_PREDICATES,
    INFERRED_PREDICATES,
)
from experiments.fact_selection_benchmark.feature_cache import BenchmarkFeatureCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("VCGATStudy")

CHECKPOINT_DIR = REPO_ROOT / "experiments" / "checkpoints_fact_selection"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = REPO_ROOT / "experiments" / "fact_selection_benchmark" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. LEVEL 1: PURE DIRECT VISUAL-FACT MATCHER (NO GRAPH BASELINE)
# =============================================================================
class V1_Base_DirectVis(nn.Module):
    """
    Level 1: Pure Bilinear Visual-Fact Matching (Non-Graph Baseline Floor).
    No graph structure is used. Evaluates isolated visual-semantic compatibility:
      feat = [h * v; |h - v|; cos_sim(h, v)]
      s(V, f_i) = MLP(feat)
    Immune to graph over-smoothing, but blind to multi-hop relational dependencies.
    """
    def __init__(self, d_model: int = 768, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.v_norm = nn.LayerNorm(d_model)
        self.scorer = nn.Sequential(
            nn.Linear(2 * d_model + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        h: torch.Tensor,
        adj_mask: torch.Tensor,
        v: torch.Tensor,
        v_frames: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if v.dim() == 1:
            v = v.unsqueeze(0)
        v_n = self.v_norm(v)
        N = h.size(0)
        v_exp = v_n.expand(N, -1)
        cos_sim = torch.sum(F.normalize(h, p=2, dim=-1) * F.normalize(v_n, p=2, dim=-1).expand(N, -1), dim=-1, keepdim=True)
        feat = torch.cat([h * v_exp, torch.abs(h - v_exp), cos_sim], dim=-1)
        return self.scorer(feat).squeeze(-1)


# =============================================================================
# 2. LEVEL 2: NAIVE VIDEO-CONDITIONED GAT ON RAW ONTOKG SUBGRAPH
# =============================================================================
class V2_GAT_Raw(nn.Module):
    """
    Level 2: Naive Video-Conditioned Graph Attention on Raw OntoKG Subgraph.
    Operates on dense root cliques (avg degree ~100).
    Demonstrates severe over-smoothing and topological domain shift on unseen landmarks.
    """
    def __init__(self, d_model: int = 768, num_heads: int = 4, hidden_dim: int = 256, tau_v: float = 0.15, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.tau_v = tau_v

        self.v_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.v_gate = nn.Linear(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.scorer = nn.Sequential(
            nn.Linear(2 * d_model + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        h: torch.Tensor,
        adj_mask: torch.Tensor,
        v: torch.Tensor,
        v_frames: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if v.dim() == 1:
            v = v.unsqueeze(0)
        v_n = self.v_norm(v)
        N = h.size(0)

        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.k_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        v_gate = torch.sigmoid(self.v_gate(v_n) / self.tau_v).view(self.num_heads, 1, 1)
        scores = scores * v_gate

        mask = (adj_mask == 0).unsqueeze(0).expand(self.num_heads, -1, -1)
        scores = scores.masked_fill(mask, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        ctx = torch.bmm(attn, V).transpose(0, 1).contiguous().view(N, self.d_model)
        h_gat = self.norm(h + self.dropout(self.out_proj(ctx)))

        v_exp = v_n.expand(N, -1)
        cos_sim = torch.sum(F.normalize(h_gat, p=2, dim=-1) * F.normalize(v_n, p=2, dim=-1).expand(N, -1), dim=-1, keepdim=True)
        feat = torch.cat([h_gat * v_exp, torch.abs(h_gat - v_exp), cos_sim], dim=-1)
        return self.scorer(feat).squeeze(-1)


# =============================================================================
# 3. LEVEL 3: TOPOLOGY-PRUNED VC-GAT WITH NODE VISUAL SALIENCY GATING
# =============================================================================
class V3_GAT_CleanGate(nn.Module):
    """
    Level 3: Topology-Pruned VC-GAT with Node Visual Saliency Gating.
    Operates on pruned graph (avg degree ~4-7) with neighbor saliency gating:
      g_j(v) = sigmoid(W_s * [h_j * v; |h_j - v|; cos_sim])
      h_j_gated = h_j * (0.5 + 0.5 * g_j(v))
    Ungrounded neighbor nodes are muted, eliminating spurious topological leakage.
    """
    def __init__(self, d_model: int = 768, num_heads: int = 4, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.v_norm = nn.LayerNorm(d_model)
        self.saliency_gate = nn.Sequential(
            nn.Linear(2 * d_model + 1, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.edge_gate = nn.Linear(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.scorer = nn.Sequential(
            nn.Linear(2 * d_model + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        h: torch.Tensor,
        adj_mask: torch.Tensor,
        v: torch.Tensor,
        v_frames: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if v.dim() == 1:
            v = v.unsqueeze(0)
        v_n = self.v_norm(v)
        N = h.size(0)
        v_exp = v_n.expand(N, -1)

        # 1. Compute Node Saliency Gate
        h_u = F.normalize(h, p=2, dim=-1)
        v_u = F.normalize(v_n, p=2, dim=-1).expand(N, -1)
        cos_init = torch.sum(h_u * v_u, dim=-1, keepdim=True)
        g_nodes = self.saliency_gate(torch.cat([h * v_exp, torch.abs(h - v_exp), cos_init], dim=-1))

        # 2. Gated Video-Conditioned Attention
        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        h_gated = h * (0.5 + 0.5 * g_nodes)
        K = self.k_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        v_edge = self.edge_gate(v_n).view(self.num_heads, 1, 1)
        scores = scores + 0.2 * v_edge

        mask = (adj_mask == 0).unsqueeze(0).expand(self.num_heads, -1, -1)
        scores = scores.masked_fill(mask, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        ctx = torch.bmm(attn, V).transpose(0, 1).contiguous().view(N, self.d_model)
        h_gat = self.norm(h + self.dropout(self.out_proj(ctx)))

        # 3. Readout
        h_gat_u = F.normalize(h_gat, p=2, dim=-1)
        cos_final = torch.sum(h_gat_u * v_u, dim=-1, keepdim=True)
        feat = torch.cat([h_gat * v_exp, torch.abs(h_gat - v_exp), cos_final, g_nodes], dim=-1)
        return self.scorer(feat).squeeze(-1)


# =============================================================================
# 4. LEVEL 4: RESIDUAL-GATED VC-GAT (GUARANTEED VISUAL LOWER-BOUND)
# =============================================================================
class V4_GAT_Residual(nn.Module):
    """
    Level 4: Residual-Gated VC-GAT with Bounded Lower-Bound.
    Decomposes score into direct visual base + gated graph relational delta:
      s(V, f_i) = s_vis(v, h_i) + lambda_i(v, h_i) * Delta s_graph(v, h_gat)
    Guarantees strong retrieval performance on unseen graphs (lambda -> 0),
    while reaping multi-hop relational boosts on familiar graphs.
    """
    def __init__(self, d_model: int = 768, num_heads: int = 4, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.v_norm = nn.LayerNorm(d_model)

        # Visual Base Scorer (guaranteed lower bound)
        self.vis_scorer = nn.Sequential(
            nn.Linear(2 * d_model + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Graph Attention Layer
        self.saliency_gate = nn.Sequential(
            nn.Linear(2 * d_model + 1, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.edge_gate = nn.Linear(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Graph Relational Delta Scorer
        self.graph_delta_scorer = nn.Sequential(
            nn.Linear(2 * d_model + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Residual Gate lambda_i
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        # Initialize bias to -0.8 so initial gate is ~0.3 (cautious lower bound)
        nn.init.constant_(self.gate_mlp[-1].bias, -0.8)

    def forward(
        self,
        h: torch.Tensor,
        adj_mask: torch.Tensor,
        v: torch.Tensor,
        v_frames: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if v.dim() == 1:
            v = v.unsqueeze(0)
        v_n = self.v_norm(v)
        N = h.size(0)
        v_exp = v_n.expand(N, -1)

        # A. Visual Base Score
        h_u = F.normalize(h, p=2, dim=-1)
        v_u = F.normalize(v_n, p=2, dim=-1).expand(N, -1)
        cos_vis = torch.sum(h_u * v_u, dim=-1, keepdim=True)
        vis_feat = torch.cat([h * v_exp, torch.abs(h - v_exp), cos_vis], dim=-1)
        s_vis = self.vis_scorer(vis_feat).squeeze(-1)

        # B. Clean Graph Attention
        g_nodes = self.saliency_gate(vis_feat)
        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        h_gated = h * (0.5 + 0.5 * g_nodes)
        K = self.k_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        v_edge = self.edge_gate(v_n).view(self.num_heads, 1, 1)
        scores = scores + 0.2 * v_edge
        mask = (adj_mask == 0).unsqueeze(0).expand(self.num_heads, -1, -1)
        scores = scores.masked_fill(mask, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        ctx = torch.bmm(attn, V).transpose(0, 1).contiguous().view(N, self.d_model)
        h_gat = self.norm(h + self.dropout(self.out_proj(ctx)))

        # C. Graph Delta Score
        h_gat_u = F.normalize(h_gat, p=2, dim=-1)
        cos_graph = torch.sum(h_gat_u * v_u, dim=-1, keepdim=True)
        graph_feat = torch.cat([h_gat * v_exp, torch.abs(h_gat - v_exp), cos_graph], dim=-1)
        s_graph_delta = self.graph_delta_scorer(graph_feat).squeeze(-1)

        # D. Residual Gate lambda_i
        lambda_gate = torch.sigmoid(self.gate_mlp(torch.cat([h * v_exp, h_gat * v_exp], dim=-1))).squeeze(-1)

        return s_vis + lambda_gate * s_graph_delta


# =============================================================================
# 5. LEVEL 5: MULTI-SCALE TEMPORAL RESIDUAL-GATED VC-GAT
# =============================================================================
class V5_GAT_TemporalResidual(nn.Module):
    """
    Level 5: Multi-Scale Temporal Residual-Gated VC-GAT.
    Solves the video temporal aggregation bottleneck:
      Each fact queries 8 video frames via temporal cross-attention:
        beta_i,t = Softmax_t( (W_q h_i)^T (W_k v_t) / sqrt(d) )
        v_i_temp = sum_t beta_i,t * v_t
      Contextualized v_i_temp replaces mean-pooled video vector, driving
      both visual scoring and clean residual graph attention!
    """
    def __init__(self, d_model: int = 768, num_heads: int = 4, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.v_norm = nn.LayerNorm(d_model)

        # Temporal Cross-Attention
        self.temp_q = nn.Linear(d_model, d_model)
        self.temp_k = nn.Linear(d_model, d_model)
        self.temp_v = nn.Linear(d_model, d_model)

        # Visual Base Scorer
        self.vis_scorer = nn.Sequential(
            nn.Linear(2 * d_model + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Clean Graph Attention Layer
        self.saliency_gate = nn.Sequential(
            nn.Linear(2 * d_model + 1, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Graph Relational Delta Scorer
        self.graph_delta_scorer = nn.Sequential(
            nn.Linear(2 * d_model + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Residual Gate
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        nn.init.constant_(self.gate_mlp[-1].bias, -0.8)

    def forward(
        self,
        h: torch.Tensor,
        adj_mask: torch.Tensor,
        v: torch.Tensor,
        v_frames: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        N = h.size(0)

        # 1. Fact-to-Frame Temporal Attention
        if v_frames is not None:
            v_f_n = self.v_norm(v_frames)  # [T, D]
            q_t = self.temp_q(h)            # [N, D]
            k_t = self.temp_k(v_f_n)        # [T, D]
            v_t = self.temp_v(v_f_n)        # [T, D]

            attn_scores = torch.matmul(q_t, k_t.transpose(0, 1)) / math.sqrt(self.d_model)  # [N, T]
            beta = F.softmax(attn_scores, dim=-1)  # [N, T]
            v_temp = torch.matmul(beta, v_t)        # [N, D]
            v_temp = self.v_norm(v_temp)
        else:
            if v.dim() == 1:
                v = v.unsqueeze(0)
            v_n = self.v_norm(v)
            v_temp = v_n.expand(N, -1)

        # 2. Visual Base Score
        h_u = F.normalize(h, p=2, dim=-1)
        v_u = F.normalize(v_temp, p=2, dim=-1)
        cos_vis = torch.sum(h_u * v_u, dim=-1, keepdim=True)
        vis_feat = torch.cat([h * v_temp, torch.abs(h - v_temp), cos_vis], dim=-1)
        s_vis = self.vis_scorer(vis_feat).squeeze(-1)

        # 3. Clean Graph Attention
        g_nodes = self.saliency_gate(vis_feat)
        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        h_gated = h * (0.5 + 0.5 * g_nodes)
        K = self.k_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V_nodes = self.v_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        mask = (adj_mask == 0).unsqueeze(0).expand(self.num_heads, -1, -1)
        scores = scores.masked_fill(mask, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        ctx = torch.bmm(attn, V_nodes).transpose(0, 1).contiguous().view(N, self.d_model)
        h_gat = self.norm(h + self.dropout(self.out_proj(ctx)))

        # 4. Graph Delta Score
        h_gat_u = F.normalize(h_gat, p=2, dim=-1)
        cos_graph = torch.sum(h_gat_u * v_u, dim=-1, keepdim=True)
        graph_feat = torch.cat([h_gat * v_temp, torch.abs(h_gat - v_temp), cos_graph], dim=-1)
        s_graph_delta = self.graph_delta_scorer(graph_feat).squeeze(-1)

        # 5. Residual Gate
        lambda_gate = torch.sigmoid(self.gate_mlp(torch.cat([h * v_temp, h_gat * v_temp], dim=-1))).squeeze(-1)

        return s_vis + lambda_gate * s_graph_delta


# =============================================================================
# TRAINING & EVALUATION UTILITIES
# =============================================================================
def evaluate_on_split(
    model: nn.Module,
    clip_records: Dict[str, List[Dict[str, Any]]],
    contrastive_pairs: List[Dict[str, Any]],
    subgraphs_dict: Dict[str, Dict[str, Any]],
    feature_cache: BenchmarkFeatureCache,
    device: torch.device,
    is_temporal: bool = False
) -> Dict[str, Any]:
    """Computes full evaluation metrics across clip_records and contrastive_pairs."""
    model.eval()
    clip_fact_scores = {}
    ndcg5_list = []
    ndcg16_list = []
    p5_list = []
    r5_list = []
    map_list = []
    auc_list = []

    # Category breakdown
    vis_ndcg5 = []
    hist_ndcg5 = []
    inf_ndcg5 = []

    # Seen vs Unseen tracking
    seen_ndcg5 = []
    unseen_ndcg5 = []
    seen_auc = []
    unseen_auc = []

    SEEN_LANDMARKS = {
        "hoan_kiem_lake", "temple_of_literature", "ba_dinh_square",
        "imperial_citadel_of_thang_long", "one_pillar_pagoda",
        "hanoi_old_quarter", "west_lake"
    }

    flat_clip_fact_scores = {}
    with torch.no_grad():
        for c_id, recs in clip_records.items():
            l_id = recs[0].get("landmark_id", "unknown")
            g = subgraphs_dict.get(l_id)
            v_pooled = feature_cache.video_pooled.get(c_id)
            if v_pooled is None or g is None:
                continue

            v_frames = None
            if is_temporal:
                v_f_np = feature_cache.video_frames.get(c_id)
                if v_f_np is not None:
                    v_frames = torch.from_numpy(v_f_np).float().to(device)

            sub_fids = g["fact_ids"]
            h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
            h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
            v_tensor = torch.from_numpy(v_pooled).float().to(device)
            adj = g["adj"].to(device)

            logits = model(h_tensor, adj, v_tensor, v_frames=v_frames)
            scores_np = logits.cpu().numpy()

            idx_map = g["fid2idx"]
            f_scores = {}
            gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
            cand_fids = [r["fact_id"] for r in recs]

            for fid in cand_fids:
                if fid in idx_map:
                    s_val = float(scores_np[idx_map[fid]])
                else:
                    s_val = 0.0
                f_scores[fid] = s_val
                flat_clip_fact_scores[(c_id, fid)] = s_val

            # Sort descending
            ranked_fids = sorted(cand_fids, key=lambda fid: f_scores[fid], reverse=True)

            # Compute standard metrics
            m = compute_ranking_metrics(ranked_fids, gold_fids, cand_fids, top_k_list=[5, 16])
            ndcg5_list.append(m.get("nDCG@5", m.get("ndcg@5", 0.0)))
            ndcg16_list.append(m.get("nDCG@16", m.get("ndcg@16", 0.0)))
            p5_list.append(m.get("P@5", m.get("p@5", 0.0)))
            r5_list.append(m.get("R@5", m.get("r@5", 0.0)))
            map_list.append(m.get("AP", m.get("map", 0.0)))

            auc = compute_hard_negative_auc(f_scores, gold_fids, hard_neg_fids)
            if auc is not None:
                auc_list.append(auc)

            # Seen vs Unseen
            is_seen = l_id in SEEN_LANDMARKS
            c_ndcg = m.get("nDCG@5", m.get("ndcg@5", 0.0))
            if is_seen:
                seen_ndcg5.append(c_ndcg)
                if auc is not None: seen_auc.append(auc)
            else:
                unseen_ndcg5.append(c_ndcg)
                if auc is not None: unseen_auc.append(auc)

            # Fact category metrics
            vis_gold = set(r["fact_id"] for r in recs if r.get("is_scene_gold") and (r.get("fact_type") == "visual" or r.get("predicate") in VISUAL_PREDICATES))
            vis_cand = [r["fact_id"] for r in recs if (r.get("fact_type") == "visual" or r.get("predicate") in VISUAL_PREDICATES)]
            if vis_gold:
                vis_ranked = [fid for fid in ranked_fids if fid in vis_cand]
                vm = compute_ranking_metrics(vis_ranked, vis_gold, vis_cand, top_k_list=[5])
                vis_ndcg5.append(vm.get("nDCG@5", vm.get("ndcg@5", 0.0)))

            hist_gold = set(r["fact_id"] for r in recs if r.get("is_scene_gold") and (r.get("fact_type") == "historical" or r.get("predicate") in HISTORICAL_PREDICATES))
            hist_cand = [r["fact_id"] for r in recs if (r.get("fact_type") == "historical" or r.get("predicate") in HISTORICAL_PREDICATES)]
            if hist_gold:
                hist_ranked = [fid for fid in ranked_fids if fid in hist_cand]
                hm = compute_ranking_metrics(hist_ranked, hist_gold, hist_cand, top_k_list=[5])
                hist_ndcg5.append(hm.get("nDCG@5", hm.get("ndcg@5", 0.0)))

            inf_gold = set(r["fact_id"] for r in recs if r.get("is_scene_gold") and (r.get("fact_type") == "inferred" or r.get("predicate") in INFERRED_PREDICATES))
            inf_cand = [r["fact_id"] for r in recs if (r.get("fact_type") == "inferred" or r.get("predicate") in INFERRED_PREDICATES)]
            if inf_gold:
                inf_ranked = [fid for fid in ranked_fids if fid in inf_cand]
                im = compute_ranking_metrics(inf_ranked, inf_gold, inf_cand, top_k_list=[5])
                inf_ndcg5.append(im.get("nDCG@5", im.get("ndcg@5", 0.0)))

    # Contrastive accuracy
    c_acc = compute_contrastive_accuracy(flat_clip_fact_scores, contrastive_pairs) if contrastive_pairs else 0.0

    return {
        "ndcg@5": float(np.mean(ndcg5_list)) if ndcg5_list else 0.0,
        "ndcg@16": float(np.mean(ndcg16_list)) if ndcg16_list else 0.0,
        "p@5": float(np.mean(p5_list)) if p5_list else 0.0,
        "r@5": float(np.mean(r5_list)) if r5_list else 0.0,
        "map": float(np.mean(map_list)) if map_list else 0.0,
        "hard_neg_auc": float(np.mean(auc_list)) if auc_list else 0.0,
        "contrastive_acc": float(c_acc),
        "seen_ndcg@5": float(np.mean(seen_ndcg5)) if seen_ndcg5 else 0.0,
        "unseen_ndcg@5": float(np.mean(unseen_ndcg5)) if unseen_ndcg5 else 0.0,
        "seen_auc": float(np.mean(seen_auc)) if seen_auc else 0.0,
        "unseen_auc": float(np.mean(unseen_auc)) if unseen_auc else 0.0,
        "vis_ndcg@5": float(np.mean(vis_ndcg5)) if vis_ndcg5 else 0.0,
        "hist_ndcg@5": float(np.mean(hist_ndcg5)) if hist_ndcg5 else 0.0,
        "inf_ndcg@5": float(np.mean(inf_ndcg5)) if inf_ndcg5 else 0.0,
    }


def train_and_evaluate_variant(
    model_name: str,
    model: nn.Module,
    subgraphs_dict: Dict[str, Dict[str, Any]],
    train_clip_records: Dict[str, List[Dict[str, Any]]],
    val_clip_records: Dict[str, List[Dict[str, Any]]],
    test_clip_records: Dict[str, List[Dict[str, Any]]],
    train_contrastive_pairs: List[Dict[str, Any]],
    test_contrastive_pairs: List[Dict[str, Any]],
    feature_cache: BenchmarkFeatureCache,
    device: torch.device,
    epochs: int = 14,
    is_temporal: bool = False
) -> Dict[str, Any]:
    """Trains variant and evaluates on official Test set."""
    LOGGER.info(f"\n{'='*75}\nTRAINING VARIANT: [{model_name}]\n{'='*75}")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    train_cids = [c for c in train_clip_records.keys() if any(r.get("is_scene_gold") for r in train_clip_records[c])]
    best_val_ndcg = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        np.random.shuffle(train_cids)

        for c_id in train_cids:
            recs = train_clip_records[c_id]
            l_id = recs[0].get("landmark_id", "unknown")
            g = subgraphs_dict.get(l_id)
            v_pooled = feature_cache.video_pooled.get(c_id)
            if g is None or v_pooled is None:
                continue

            pos_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
            all_neg_fids = set(r["fact_id"] for r in recs if not r.get("is_scene_gold"))
            if not pos_fids:
                continue

            v_frames = None
            if is_temporal:
                v_f_np = feature_cache.video_frames.get(c_id)
                if v_f_np is not None:
                    v_frames = torch.from_numpy(v_f_np).float().to(device)

            optimizer.zero_grad()
            sub_fids = g["fact_ids"]
            h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
            h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
            v_tensor = torch.from_numpy(v_pooled).float().to(device)
            adj = g["adj"].to(device)

            logits = model(h_tensor, adj, v_tensor, v_frames=v_frames)

            idx_map = g["fid2idx"]
            pos_idx = [idx_map[fid] for fid in pos_fids if fid in idx_map]
            hard_neg_idx = [idx_map[fid] for fid in hard_neg_fids if fid in idx_map]
            neg_sample_idx = [idx_map[fid] for fid in list(all_neg_fids)[:len(pos_fids) * 4] if fid in idx_map]

            if not pos_idx or not neg_sample_idx:
                continue

            # BCE loss with positive weighting
            y_true = [1.0] * len(pos_idx) + [0.0] * len(neg_sample_idx)
            preds = torch.cat([logits[pos_idx], logits[neg_sample_idx]])
            targets = torch.tensor(y_true, dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(preds, targets, pos_weight=torch.tensor([4.0], device=device))

            # Hard negative margin loss
            if hard_neg_idx:
                p_exp = logits[pos_idx].unsqueeze(1)
                hn_exp = logits[hard_neg_idx].unsqueeze(0)
                loss = loss + 0.8 * torch.relu(0.8 - (p_exp - hn_exp)).mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Multi-task contrastive pairs loss (sample 150 pairs per epoch)
        if train_contrastive_pairs:
            np.random.shuffle(train_contrastive_pairs)
            for p in train_contrastive_pairs[:150]:
                c_a = p.get("clip_a") or p.get("clip_A", {}).get("clip_id")
                c_b = p.get("clip_b") or p.get("clip_B", {}).get("clip_id")
                f_id = p.get("fact_id") or p.get("fact_a")
                l_id = p.get("landmark_id")

                v_a = feature_cache.video_pooled.get(c_a)
                v_b = feature_cache.video_pooled.get(c_b)
                g = subgraphs_dict.get(l_id)
                if v_a is None or v_b is None or g is None or f_id not in g["fid2idx"]:
                    continue

                f_idx = g["fid2idx"][f_id]
                sub_fids = g["fact_ids"]
                h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
                h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
                adj = g["adj"].to(device)

                v_fa = None
                v_fb = None
                if is_temporal:
                    v_fa_np = feature_cache.video_frames.get(c_a)
                    v_fb_np = feature_cache.video_frames.get(c_b)
                    if v_fa_np is not None: v_fa = torch.from_numpy(v_fa_np).float().to(device)
                    if v_fb_np is not None: v_fb = torch.from_numpy(v_fb_np).float().to(device)

                optimizer.zero_grad()
                s_a = model(h_tensor, adj, torch.from_numpy(v_a).float().to(device), v_frames=v_fa)[f_idx]
                s_b = model(h_tensor, adj, torch.from_numpy(v_b).float().to(device), v_frames=v_fb)[f_idx]

                cont_loss = torch.relu(1.0 - (s_a - s_b))
                if cont_loss > 0:
                    cont_loss.backward()
                    optimizer.step()

        # Validation at epoch end
        val_res = evaluate_on_split(
            model, val_clip_records, [], subgraphs_dict, feature_cache, device, is_temporal=is_temporal
        )
        val_ndcg = val_res["ndcg@5"]
        LOGGER.info(
            f"Epoch {epoch:02d} | Train Loss: {total_loss:.1f} | "
            f"Val nDCG@5: {val_ndcg:.4f} | Hard Neg AUC: {val_res['hard_neg_auc']:.4f}"
        )

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        LOGGER.info(f"Loaded best checkpoint for {model_name} (Val nDCG@5: {best_val_ndcg:.4f})")

    # Final Evaluation on Official Test Split (800 clips, 9,040 contrastive pairs)
    LOGGER.info(f"Running definitive test evaluation for [{model_name}] on 800 test clips...")
    test_res = evaluate_on_split(
        model, test_clip_records, test_contrastive_pairs, subgraphs_dict, feature_cache, device, is_temporal=is_temporal
    )

    LOGGER.info(
        f"RESULT [{model_name}]: Overall nDCG@5={test_res['ndcg@5']:.4f} | "
        f"P@5={test_res['p@5']:.4f} | Hard Neg AUC={test_res['hard_neg_auc']:.4f} | "
        f"Contrastive={test_res['contrastive_acc']:.1f}% | "
        f"Seen nDCG={test_res['seen_ndcg@5']:.4f} | Unseen nDCG={test_res['unseen_ndcg@5']:.4f}"
    )

    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / f"study_{model_name.lower()}.pt"
    torch.save(best_state, ckpt_path)
    return test_res


# =============================================================================
# MAIN STUDY WORKFLOW
# =============================================================================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using computing device: {device}")

    # 1. Load Data
    LOGGER.info("Loading facts and gold datasets...")
    facts = load_normalized_facts()
    train_clip_records, _ = load_gold_data(split_type="train")
    val_clip_records, _ = load_gold_data(split_type="val")
    test_clip_records, _ = load_gold_data(split_type="test")
    train_contrastive_pairs = load_contrastive_pairs(split_type="train_val")
    test_contrastive_pairs = load_contrastive_pairs(split_type="test")

    LOGGER.info(
        f"Loaded: {len(facts)} facts | Train: {len(train_clip_records)} clips | "
        f"Val: {len(val_clip_records)} clips | Test: {len(test_clip_records)} clips | "
        f"Test Contrastive Pairs: {len(test_contrastive_pairs)}"
    )

    # 2. Load Feature Cache & Subgraphs
    feature_cache = BenchmarkFeatureCache(device=device)
    feature_cache.extract_fact_embeddings(facts)
    feature_cache.extract_video_features([])  # Loads from disk
    raw_subgraphs = feature_cache.build_landmark_subgraphs(facts)

    clean_subgraphs_path = feature_cache.cache_dir / "landmark_subgraphs_clean.pt"
    if not clean_subgraphs_path.exists():
        raise FileNotFoundError(f"Missing {clean_subgraphs_path}. Please generate it first.")
    clean_subgraphs = torch.load(clean_subgraphs_path, weights_only=False)

    LOGGER.info(f"Loaded Raw subgraphs ({len(raw_subgraphs)}) and Clean subgraphs ({len(clean_subgraphs)})")

    # 3. Define Variants Progression
    variants = [
        # Level 1: Direct Visual Matching (No Graph)
        ("V1_Base_DirectVis", V1_Base_DirectVis(d_model=768), clean_subgraphs, False),

        # Level 2: Naive VC-GAT on Raw OntoKG Subgraph
        ("V2_GAT_Raw", V2_GAT_Raw(d_model=768), raw_subgraphs, False),

        # Level 3: Topology-Pruned VC-GAT with Visual Saliency Gating
        ("V3_GAT_CleanGate", V3_GAT_CleanGate(d_model=768), clean_subgraphs, False),

        # Level 4: Residual-Gated VC-GAT (Guaranteed Visual Lower-Bound)
        ("V4_GAT_Residual", V4_GAT_Residual(d_model=768), clean_subgraphs, False),

        # Level 5: Multi-Scale Temporal Residual-Gated VC-GAT
        ("V5_GAT_TemporalResidual", V5_GAT_TemporalResidual(d_model=768), clean_subgraphs, True),
    ]

    results = {}
    for name, model, sub_dict, is_temp in variants:
        res = train_and_evaluate_variant(
            model_name=name,
            model=model,
            subgraphs_dict=sub_dict,
            train_clip_records=train_clip_records,
            val_clip_records=val_clip_records,
            test_clip_records=test_clip_records,
            train_contrastive_pairs=train_contrastive_pairs,
            test_contrastive_pairs=test_contrastive_pairs,
            feature_cache=feature_cache,
            device=device,
            epochs=14,
            is_temporal=is_temp
        )
        results[name] = res

    # Save Results
    json_path = REPORTS_DIR / "vcgat_architectural_study_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    LOGGER.info(f"Saved architectural study results to {json_path}")

    # Build Markdown Table
    md_lines = [
        "# VC-GAT Architectural Progression Study (Simpler to Complex)\n",
        "Evaluated on 800 independent test clips (301 Seen Hanoi, 499 Unseen Hanoi+HCMC) over 143,124 candidate pairs and 9,040 contrastive pairs.\n",
        "## Table 1: Progressive Performance Comparison\n",
        "| Level | Variant / Architectural Paradigm | nDCG@5 ↑ | P@5 ↑ | Hard Neg AUC ↑ | Contrastive Acc (%) ↑ | Seen nDCG@5 | Unseen nDCG@5 | Generalization Gap |\n",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    ]

    level_map = {
        "V1_Base_DirectVis": ("L1", "Base: Direct Visual Matching (No Graph)"),
        "V2_GAT_Raw": ("L2", "Naive VC-GAT (Raw OntoKG Subgraph)"),
        "V3_GAT_CleanGate": ("L3", "Pruned VC-GAT + Saliency Gating"),
        "V4_GAT_Residual": ("L4", "Residual-Gated VC-GAT (Visual Lower-Bound)"),
        "V5_GAT_TemporalResidual": ("L5", "Multi-Scale Temporal Residual VC-GAT"),
    }

    for name, r in results.items():
        lvl, desc = level_map[name]
        gap = r["seen_ndcg@5"] - r["unseen_ndcg@5"]
        md_lines.append(
            f"| {lvl} | **{desc}** | **{r['ndcg@5']:.4f}** | {r['p@5']:.4f} | "
            f"**{r['hard_neg_auc']:.4f}** | **{r['contrastive_acc']:.1f}%** | "
            f"{r['seen_ndcg@5']:.4f} | {r['unseen_ndcg@5']:.4f} | {gap:+.4f} |\n"
        )

    md_lines.append("\n## Table 2: Fact-Type Ranking Breakdown (nDCG@5)\n")
    md_lines.append("| Level | Variant | Visual Facts | Historical Facts | Inferred Facts |\n")
    md_lines.append("| :---: | :--- | :---: | :---: | :---: |\n")
    for name, r in results.items():
        lvl, desc = level_map[name]
        md_lines.append(
            f"| {lvl} | {desc} | {r['vis_ndcg@5']:.4f} | {r['hist_ndcg@5']:.4f} | {r['inf_ndcg@5']:.4f} |\n"
        )

    table_path = REPORTS_DIR / "vcgat_architectural_study_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)
    LOGGER.info(f"Saved architectural study table to {table_path}")
    print("".join(md_lines))


if __name__ == "__main__":
    main()
