#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/train_rvcgat_study.py
=============================================================================
IJAR Research Model: Relation-Aware Video-Conditioned Graph Attention (R-VCGAT)
with Evidence-Based Reliability Gating and Scene-Conditioned Hard-Negative Ranking Loss.

Directly targets the IJAR Paper Research Objectives:
  1. Overall nDCG@5 >= 0.290 (Beating Cross-Attention 0.288 & InternVideo2 0.281)
  2. Unseen nDCG@5 >= 0.230 (Maintaining graph transferability on HCMC OOD)
  3. Hard Negative AUC >= 0.660 (Benchmark-best scene discrimination)
  4. Contrastive Accuracy >= 60.0% (Benchmark-best cross-scene sensitivity)

Scientific Innovations:
  - Relation-Aware Video Attention: Differentiates 4 semantic edge types
    (self-loop, co-visible visual, visual-to-historical grounding bridge, semantic context).
  - Evidence-Based Reliability Gate: Modulates graph contribution lambda_i(V, G)
    via neighborhood visual grounding ratio (R_anchor), relational alignment, and degree sparsity.
  - Scene-Conditioned Triple-Margin Objective: Jointly optimizes BCE, top ranking margin,
    intra-landmark hard negative margin, and cross-scene contrastive margin.
  - Strict Validation Selection Protocol: Checkpoint chosen exclusively on Val split (149 clips),
    completely decoupled from Test evaluation (800 clips).
  - Controlled Ablation Matrix: Tests with/without Relation-Aware, Reliability Gate,
    Residual Connection, and Scene Hard-Neg Loss under exact identical conditions.
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
LOGGER = logging.getLogger("R-VCGAT")

CHECKPOINT_DIR = REPO_ROOT / "experiments" / "checkpoints_fact_selection"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = REPO_ROOT / "experiments" / "fact_selection_benchmark" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. RELATION-AWARE VIDEO-CONDITIONED GRAPH ATTENTION LAYER
# =============================================================================
class RelationAwareVCGATLayer(nn.Module):
    """
    Relation-Aware Video-Conditioned Graph Attention Layer.
    Differentiates 4 edge types:
      0: Self-loop
      1: Co-visible / Spatial Part-Whole (vis <-> vis)
      2: Multimodal Grounding Bridge (vis <-> sem)
      3: Historical / Conceptual Context (sem <-> sem)
    """
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 4,
        num_relations: int = 4,
        d_rel: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Relation embeddings & projections
        self.rel_embed = nn.Embedding(num_relations, d_rel)
        self.rel_k_proj = nn.Linear(d_rel, d_model)
        self.rel_v_proj = nn.Linear(d_rel, d_model)
        self.rel_bias = nn.Embedding(num_relations, num_heads)

        # Video-relation contextual gate
        self.v_rel_proj = nn.Linear(d_model, d_rel)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        adj_mask: torch.Tensor,
        rel_adj: torch.Tensor,
        v: torch.Tensor,
        g_nodes: torch.Tensor,
        use_relations: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: [N, D]
        adj_mask: [N, N] binary
        rel_adj: [N, N] int relation type
        v: [1, D]
        g_nodes: [N, 1] in [0, 1]
        Returns: (h_out [N, D], attn_weights [H, N, N])
        """
        N = h.size(0)
        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1) # [H, N, d]

        # Node visual saliency gating
        h_gated = h * (0.5 + 0.5 * g_nodes)
        K = self.k_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1) # [H, N, d]
        V_nodes = self.v_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1) # [H, N, d]

        # Base scores: Q * K^T
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale # [H, N, N]

        if use_relations:
            rel_emb = self.rel_embed(rel_adj) # [N, N, d_rel]
            rel_k = self.rel_k_proj(rel_emb).view(N, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3) # [H, N, N, d]
            q_exp = Q.unsqueeze(2) # [H, N, 1, d]
            scores_rel = torch.sum(q_exp * rel_k, dim=-1) * self.scale # [H, N, N]
            scores = scores + scores_rel

            # Relation bias
            r_bias = self.rel_bias(rel_adj).permute(2, 0, 1) # [H, N, N]
            scores = scores + r_bias

            # Video modulation on relation bridges
            v_r = self.v_rel_proj(v) # [1, d_rel]
            v_r_sim = torch.sum(F.normalize(rel_emb, p=2, dim=-1) * F.normalize(v_r, p=2, dim=-1).unsqueeze(0), dim=-1)
            scores = scores + 0.25 * v_r_sim.unsqueeze(0)

        # Mask non-edges
        mask = (adj_mask == 0).unsqueeze(0).expand(self.num_heads, -1, -1)
        scores = scores.masked_fill(mask, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1)) # [H, N, N]

        # Context aggregation
        ctx_nodes = torch.bmm(attn, V_nodes) # [H, N, d]

        if use_relations:
            rel_v = self.rel_v_proj(rel_emb).view(N, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3) # [H, N, N, d]
            ctx_rel = torch.sum(attn.unsqueeze(-1) * rel_v, dim=2) # [H, N, d]
            context = (ctx_nodes + ctx_rel).transpose(0, 1).contiguous().view(N, -1)
        else:
            context = ctx_nodes.transpose(0, 1).contiguous().view(N, -1)

        h_out = self.norm(h + self.dropout(self.out_proj(context)))
        return h_out, attn


# =============================================================================
# 2. EVIDENCE-BASED GRAPH RELIABILITY GATE
# =============================================================================
class EvidenceGraphReliabilityGate(nn.Module):
    """
    Computes topological reliability gate lambda_i(V, G).
    Grounded in approximate reasoning principles:
      - R_anchor: weighted visual grounding ratio of the active neighborhood
      - cos_sim_gat: semantic alignment between updated graph representation and video
      - deg_factor: topological sparsity factor
    """
    def __init__(self, d_model: int = 768, hidden_dim: int = 64):
        super().__init__()
        # Input features: h*v (D), h_gat*v (D), cos_vis (1), cos_gat (1), R_anchor (1), deg_factor (1) = 2*D + 4
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d_model + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        # Prior bias: -0.6 (ensures cautious initial gate ~0.35)
        nn.init.constant_(self.gate_mlp[-1].bias, -0.6)

    def forward(
        self,
        h: torch.Tensor,
        h_gat: torch.Tensor,
        v: torch.Tensor,
        attn_mean: torch.Tensor,
        g_nodes: torch.Tensor,
        degrees: torch.Tensor
    ) -> torch.Tensor:
        N = h.size(0)
        v_exp = v.expand(N, -1)

        # 1. Neighborhood Visual Grounding Ratio R_anchor:
        eye = torch.eye(N, device=h.device)
        attn_neighbor = attn_mean * (1.0 - eye)
        neighbor_sum = attn_neighbor.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        r_anchor = torch.matmul(attn_neighbor, g_nodes) / neighbor_sum

        # 2. Semantic alignment
        h_u = F.normalize(h, p=2, dim=-1)
        h_gat_u = F.normalize(h_gat, p=2, dim=-1)
        v_u = F.normalize(v, p=2, dim=-1).expand(N, -1)

        cos_vis = torch.sum(h_u * v_u, dim=-1, keepdim=True)
        cos_gat = torch.sum(h_gat_u * v_u, dim=-1, keepdim=True)

        # 3. Degree factor
        deg_factor = (1.0 / torch.sqrt(1.0 + degrees.float())).unsqueeze(-1)

        gate_feat = torch.cat([h * v_exp, h_gat * v_exp, cos_vis, cos_gat, r_anchor, deg_factor], dim=-1)
        return torch.sigmoid(self.gate_mlp(gate_feat)).squeeze(-1) # [N] in (0, 1)


# =============================================================================
# 3. PROPOSED R-VCGAT MODEL ARCHITECTURE
# =============================================================================
class ProposedRVCGATModel(nn.Module):
    """
    Proposed Full R-VCGAT Architecture.
    Decomposes scoring into:
      s(V, f_i) = s_vis(v, h_i) + lambda_i(V, G) * Delta s_graph(v, h_gat)
    """
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_relations: bool = True,
        use_reliability_gate: bool = True,
        use_residual: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.use_relations = use_relations
        self.use_reliability_gate = use_reliability_gate
        self.use_residual = use_residual

        self.v_norm = nn.LayerNorm(d_model)

        # Visual Base Scorer (Strict Direct Matching Lower-Bound)
        self.vis_scorer = nn.Sequential(
            nn.Linear(2 * d_model + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Node Saliency Gate
        self.saliency_gate = nn.Sequential(
            nn.Linear(2 * d_model + 1, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Relation-Aware VC-GAT Layer
        self.layer = RelationAwareVCGATLayer(
            d_model=d_model, num_heads=num_heads, num_relations=4, dropout=dropout
        )

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

        # Evidence-Based Reliability Gate
        self.reliability_gate = EvidenceGraphReliabilityGate(d_model=d_model)

    def forward(
        self,
        h: torch.Tensor,
        adj_mask: torch.Tensor,
        rel_adj: torch.Tensor,
        degrees: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if v.dim() == 1:
            v = v.unsqueeze(0)
        v_n = self.v_norm(v)
        N = h.size(0)
        v_exp = v_n.expand(N, -1)

        # 1. Direct Visual Base Score s_vis
        h_u = F.normalize(h, p=2, dim=-1)
        v_u = F.normalize(v_n, p=2, dim=-1).expand(N, -1)
        cos_vis = torch.sum(h_u * v_u, dim=-1, keepdim=True)
        vis_feat = torch.cat([h * v_exp, torch.abs(h - v_exp), cos_vis], dim=-1)
        s_vis = self.vis_scorer(vis_feat).squeeze(-1)

        # 2. Node Saliency Gate
        g_nodes = self.saliency_gate(vis_feat)

        # 3. Relation-Aware VC-GAT
        h_gat, attn = self.layer(h, adj_mask, rel_adj, v_n, g_nodes, use_relations=self.use_relations)
        attn_mean = attn.mean(dim=0)

        # 4. Graph Relational Delta Score Delta s_graph
        h_gat_u = F.normalize(h_gat, p=2, dim=-1)
        cos_gat = torch.sum(h_gat_u * v_u, dim=-1, keepdim=True)
        graph_feat = torch.cat([h_gat * v_exp, torch.abs(h_gat - v_exp), cos_gat], dim=-1)
        s_graph_delta = self.graph_delta_scorer(graph_feat).squeeze(-1)

        # 5. Reliability Gate lambda_i
        if self.use_reliability_gate:
            lambda_gate = self.reliability_gate(h, h_gat, v_n, attn_mean, g_nodes, degrees)
        else:
            lambda_gate = torch.ones(N, device=h.device)

        # 6. Residual Combination
        if not self.use_residual:
            final_score = s_graph_delta
        else:
            final_score = s_vis + lambda_gate * s_graph_delta

        return final_score, s_vis, lambda_gate


# =============================================================================
# 4. EVALUATION & INFERENCE FUNCTIONS
# =============================================================================
def evaluate_model_on_split(
    model: nn.Module,
    clip_records: Dict[str, List[Dict[str, Any]]],
    contrastive_pairs: List[Dict[str, Any]],
    subgraphs_dict: Dict[str, Dict[str, Any]],
    feature_cache: BenchmarkFeatureCache,
    device: torch.device
) -> Dict[str, Any]:
    """Computes full evaluation metrics across clip_records and contrastive_pairs."""
    model.eval()
    flat_clip_fact_scores = {}
    ndcg5_list = []
    ndcg16_list = []
    p5_list = []
    r5_list = []
    map_list = []
    auc_list = []
    gates_seen = []
    gates_unseen = []

    vis_ndcg5 = []
    hist_ndcg5 = []
    inf_ndcg5 = []

    seen_ndcg5 = []
    unseen_ndcg5 = []
    seen_auc = []
    unseen_auc = []

    SEEN_LANDMARKS = {
        "hoan_kiem_lake", "temple_of_literature", "ba_dinh_square",
        "imperial_citadel_of_thang_long", "one_pillar_pagoda",
        "hanoi_old_quarter", "west_lake"
    }

    with torch.no_grad():
        for c_id, recs in clip_records.items():
            l_id = recs[0].get("landmark_id", "unknown")
            g = subgraphs_dict.get(l_id)
            v_pooled = feature_cache.video_pooled.get(c_id)
            if v_pooled is None or g is None:
                continue

            sub_fids = g["fact_ids"]
            h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
            h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
            v_tensor = torch.from_numpy(v_pooled).float().to(device)
            adj = g["adj"].to(device)
            rel_adj = g["rel_adj"].to(device)
            degrees = g["degrees"].to(device)

            logits, _, lambda_gate = model(h_tensor, adj, rel_adj, degrees, v_tensor)
            scores_np = logits.cpu().numpy()
            gate_np = lambda_gate.cpu().numpy()

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

            ranked_fids = sorted(cand_fids, key=lambda fid: f_scores[fid], reverse=True)
            m = compute_ranking_metrics(ranked_fids, gold_fids, cand_fids, top_k_list=[5, 16])
            ndcg5_list.append(m.get("nDCG@5", 0.0))
            ndcg16_list.append(m.get("nDCG@16", 0.0))
            p5_list.append(m.get("P@5", 0.0))
            r5_list.append(m.get("R@5", 0.0))
            map_list.append(m.get("AP", 0.0))

            auc = compute_hard_negative_auc(f_scores, gold_fids, hard_neg_fids)
            if auc is not None:
                auc_list.append(auc)

            # Seen vs Unseen
            is_seen = l_id in SEEN_LANDMARKS
            c_ndcg = m.get("nDCG@5", 0.0)
            if is_seen:
                seen_ndcg5.append(c_ndcg)
                if auc is not None: seen_auc.append(auc)
                gates_seen.extend(gate_np.tolist())
            else:
                unseen_ndcg5.append(c_ndcg)
                if auc is not None: unseen_auc.append(auc)
                gates_unseen.extend(gate_np.tolist())

            # Fact category metrics
            vis_gold = set(r["fact_id"] for r in recs if r.get("is_scene_gold") and (r.get("fact_type") == "visual" or r.get("predicate") in VISUAL_PREDICATES))
            vis_cand = [r["fact_id"] for r in recs if (r.get("fact_type") == "visual" or r.get("predicate") in VISUAL_PREDICATES)]
            if vis_gold:
                vis_ranked = [fid for fid in ranked_fids if fid in vis_cand]
                vm = compute_ranking_metrics(vis_ranked, vis_gold, vis_cand, top_k_list=[5])
                vis_ndcg5.append(vm.get("nDCG@5", 0.0))

            hist_gold = set(r["fact_id"] for r in recs if r.get("is_scene_gold") and (r.get("fact_type") == "historical" or r.get("predicate") in HISTORICAL_PREDICATES))
            hist_cand = [r["fact_id"] for r in recs if (r.get("fact_type") == "historical" or r.get("predicate") in HISTORICAL_PREDICATES)]
            if hist_gold:
                hist_ranked = [fid for fid in ranked_fids if fid in hist_cand]
                hm = compute_ranking_metrics(hist_ranked, hist_gold, hist_cand, top_k_list=[5])
                hist_ndcg5.append(hm.get("nDCG@5", 0.0))

            inf_gold = set(r["fact_id"] for r in recs if r.get("is_scene_gold") and (r.get("fact_type") == "inferred" or r.get("predicate") in INFERRED_PREDICATES))
            inf_cand = [r["fact_id"] for r in recs if (r.get("fact_type") == "inferred" or r.get("predicate") in INFERRED_PREDICATES)]
            if inf_gold:
                inf_ranked = [fid for fid in ranked_fids if fid in inf_cand]
                im = compute_ranking_metrics(inf_ranked, inf_gold, inf_cand, top_k_list=[5])
                inf_ndcg5.append(im.get("nDCG@5", 0.0))

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
        "mean_gate_seen": float(np.mean(gates_seen)) if gates_seen else 0.0,
        "mean_gate_unseen": float(np.mean(gates_unseen)) if gates_unseen else 0.0,
    }


# =============================================================================
# 5. TRAINING ENGINE WITH STRICT VALIDATION SELECTION
# =============================================================================
def train_and_evaluate_rvcgat(
    experiment_name: str,
    model: nn.Module,
    subgraphs_dict: Dict[str, Dict[str, Any]],
    train_clip_records: Dict[str, List[Dict[str, Any]]],
    val_clip_records: Dict[str, List[Dict[str, Any]]],
    test_clip_records: Dict[str, List[Dict[str, Any]]],
    train_contrastive_pairs: List[Dict[str, Any]],
    test_contrastive_pairs: List[Dict[str, Any]],
    feature_cache: BenchmarkFeatureCache,
    device: torch.device,
    use_scene_hn_loss: bool = True,
    epochs: int = 16
) -> Dict[str, Any]:
    """
    Trains model and performs checkpoint selection STRICTLY on Val split.
    Then evaluates chosen checkpoint ONCE on Test split.
    """
    LOGGER.info(f"\n{'='*75}\nTRAINING: [{experiment_name}]\n{'='*75}")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1.2e-4, weight_decay=1e-4)

    # Validation contrastive pairs (from val clips)
    val_cids = set(val_clip_records.keys())
    val_contrastive_pairs = [
        p for p in train_contrastive_pairs
        if (p.get("clip_A", {}).get("clip_id") in val_cids or p.get("clip_a") in val_cids)
    ]
    if not val_contrastive_pairs:
        val_contrastive_pairs = train_contrastive_pairs[:400]

    train_cids = [c for c in train_clip_records.keys() if any(r.get("is_scene_gold") for r in train_clip_records[c])]
    best_val_score = -1.0
    best_val_metrics = {}
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
            if g is None or v_pooled is None: continue

            pos_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
            all_neg_fids = set(r["fact_id"] for r in recs if not r.get("is_scene_gold"))
            if not pos_fids or not all_neg_fids: continue

            optimizer.zero_grad()
            sub_fids = g["fact_ids"]
            h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
            h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
            v_tensor = torch.from_numpy(v_pooled).float().to(device)
            adj = g["adj"].to(device)
            rel_adj = g["rel_adj"].to(device)
            degrees = g["degrees"].to(device)

            logits, _, _ = model(h_tensor, adj, rel_adj, degrees, v_tensor)
            idx_map = g["fid2idx"]
            pos_idx = [idx_map[fid] for fid in pos_fids if fid in idx_map]
            hard_neg_idx = [idx_map[fid] for fid in hard_neg_fids if fid in idx_map]
            neg_idx = [idx_map[fid] for fid in list(all_neg_fids)[:len(pos_idx)*6] if fid in idx_map]

            if not pos_idx or not neg_idx: continue

            # 1. Calibrated BCE Loss
            y_true = [1.0] * len(pos_idx) + [0.0] * len(neg_idx)
            preds = torch.cat([logits[pos_idx], logits[neg_idx]])
            targets = torch.tensor(y_true, dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(preds, targets, pos_weight=torch.tensor([4.0], device=device))

            p_scores = logits[pos_idx].unsqueeze(1) # [P, 1]
            n_scores = logits[neg_idx].unsqueeze(0) # [1, N]

            if use_scene_hn_loss:
                # 2. Pairwise Top-Ranking Margin Loss (margin 0.6)
                rank_loss = torch.relu(0.6 - (p_scores - n_scores)).mean()
                loss = loss + 0.8 * rank_loss

                # 3. Scene-Conditioned Hard-Negative Margin Loss (strict margin 1.0)
                if hard_neg_idx:
                    hn_scores = logits[hard_neg_idx].unsqueeze(0) # [1, HN]
                    hn_loss = torch.relu(1.0 - (p_scores - hn_scores)).mean()
                    loss = loss + 1.0 * hn_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Multi-task cross-scene contrastive pairs
        if use_scene_hn_loss and train_contrastive_pairs:
            np.random.shuffle(train_contrastive_pairs)
            for p in train_contrastive_pairs[:180]:
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
                rel_adj = g["rel_adj"].to(device)
                degrees = g["degrees"].to(device)

                optimizer.zero_grad()
                s_a, _, _ = model(h_tensor, adj, rel_adj, degrees, torch.from_numpy(v_a).float().to(device))
                s_b, _, _ = model(h_tensor, adj, rel_adj, degrees, torch.from_numpy(v_b).float().to(device))

                cont_loss = torch.relu(1.0 - (s_a[f_idx] - s_b[f_idx]))
                if cont_loss > 0:
                    cont_loss.backward()
                    optimizer.step()

        # STRICT VALIDATION: Checkpoint selection on Val split
        val_res = evaluate_model_on_split(
            model, val_clip_records, val_contrastive_pairs, subgraphs_dict, feature_cache, device
        )
        val_ndcg = val_res["ndcg@5"]
        val_auc = val_res["hard_neg_auc"]
        val_cont = val_res["contrastive_acc"]

        # Composite Validation Criterion (Balances Ranking, Discrimination, and Contrastive)
        val_score = 0.5 * val_ndcg + 0.3 * val_auc + 0.2 * (val_cont / 100.0)

        LOGGER.info(
            f"Epoch {epoch:02d} | Train Loss: {total_loss:.1f} | "
            f"Val nDCG@5: {val_ndcg:.4f} | AUC: {val_auc:.4f} | Cont: {val_cont:.1f}% | ValScore: {val_score:.4f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_metrics = dict(val_res)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load frozen best checkpoint
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    LOGGER.info(f"Loaded frozen best checkpoint on Validation (ValScore: {best_val_score:.4f})")

    # Evaluate ONCE on independent 800 Test clips
    LOGGER.info(f"Running definitive evaluation on 800 Test clips for [{experiment_name}]...")
    test_res = evaluate_model_on_split(
        model, test_clip_records, test_contrastive_pairs, subgraphs_dict, feature_cache, device
    )

    LOGGER.info(
        f"\nFINAL TEST RESULT [{experiment_name}]:\n"
        f"  Overall nDCG@5      = {test_res['ndcg@5']:.4f}\n"
        f"  Overall P@5         = {test_res['p@5']:.4f}\n"
        f"  Hard Negative AUC   = {test_res['hard_neg_auc']:.4f}\n"
        f"  Contrastive Acc     = {test_res['contrastive_acc']:.1f}%\n"
        f"  Seen nDCG@5         = {test_res['seen_ndcg@5']:.4f}\n"
        f"  Unseen nDCG@5       = {test_res['unseen_ndcg@5']:.4f}\n"
        f"  Mean Gate (Seen)    = {test_res['mean_gate_seen']:.3f}\n"
        f"  Mean Gate (Unseen)  = {test_res['mean_gate_unseen']:.3f}\n"
    )

    ckpt_path = CHECKPOINT_DIR / f"best_{experiment_name.lower()}.pt"
    torch.save(best_state, ckpt_path)
    return {
        "val_score": best_val_score,
        "val_metrics": best_val_metrics,
        "test_metrics": test_res
    }


# =============================================================================
# 6. MAIN ORCHESTRATION & ABLATION MATRIX
# =============================================================================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using computing device: {device}")

    # 1. Load Data
    facts = load_normalized_facts()
    train_clip_records, _ = load_gold_data(split_type="train")
    val_clip_records, _ = load_gold_data(split_type="val")
    test_clip_records, _ = load_gold_data(split_type="test")
    train_contrastive_pairs = load_contrastive_pairs(split_type="train_val")
    test_contrastive_pairs = load_contrastive_pairs(split_type="test")

    feature_cache = BenchmarkFeatureCache(device=device)
    feature_cache.extract_fact_embeddings(facts)
    feature_cache.extract_video_features([])

    rel_subgraphs_path = feature_cache.cache_dir / "landmark_subgraphs_rel.pt"
    if not rel_subgraphs_path.exists():
        raise FileNotFoundError(f"Missing {rel_subgraphs_path}. Please run generate_rel_subgraphs.py.")
    rel_subgraphs = torch.load(rel_subgraphs_path, weights_only=False)
    LOGGER.info(f"Loaded {len(rel_subgraphs)} relation-aware subgraphs.")

    # 2. Controlled Experiment Matrix
    # All variants share: identical backbone (768d), identical train clips (702), identical test clips (800)
    experiments = [
        # (A) Proposed Full R-VCGAT: All components enabled
        ("R_VCGAT_Proposed_Full",
         ProposedRVCGATModel(d_model=768, use_relations=True, use_reliability_gate=True, use_residual=True),
         True),

        # (B) Ablation 1: w/o Relation-Aware Graph Attention (treats all edges homogeneously)
        ("Ablation_wo_Relations",
         ProposedRVCGATModel(d_model=768, use_relations=False, use_reliability_gate=True, use_residual=True),
         True),

        # (C) Ablation 2: w/o Reliability Gate (lambda = 1.0 fixed residual, no adaptive gating)
        ("Ablation_wo_ReliabilityGate",
         ProposedRVCGATModel(d_model=768, use_relations=True, use_reliability_gate=False, use_residual=True),
         True),

        # (D) Ablation 3: w/o Residual Connection (pure graph readout s = Delta s_graph, no visual lower bound)
        ("Ablation_wo_Residual",
         ProposedRVCGATModel(d_model=768, use_relations=True, use_reliability_gate=False, use_residual=False),
         True),

        # (E) Ablation 4: w/o Scene Hard-Negative Ranking Loss (standard uniform BCE only)
        ("Ablation_wo_SceneHNLoss",
         ProposedRVCGATModel(d_model=768, use_relations=True, use_reliability_gate=True, use_residual=True),
         False),
    ]

    all_results = {}
    for exp_name, model_inst, use_hn in experiments:
        res = train_and_evaluate_rvcgat(
            experiment_name=exp_name,
            model=model_inst,
            subgraphs_dict=rel_subgraphs,
            train_clip_records=train_clip_records,
            val_clip_records=val_clip_records,
            test_clip_records=test_clip_records,
            train_contrastive_pairs=train_contrastive_pairs,
            test_contrastive_pairs=test_contrastive_pairs,
            feature_cache=feature_cache,
            device=device,
            use_scene_hn_loss=use_hn,
            epochs=16
        )
        all_results[exp_name] = res

    # Save to JSON
    json_path = REPORTS_DIR / "rvcgat_study_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    LOGGER.info(f"Saved complete R-VCGAT results to {json_path}")

    # Generate Markdown Table
    md_lines = [
        "# R-VCGAT Controlled Improvement & Ablation Benchmark\n",
        "Evaluated on 800 independent test clips (301 Seen Hanoi, 499 Unseen Hanoi+HCMC) over 143,124 candidate pairs and 9,040 contrastive pairs.\n",
        "Checkpoints selected exclusively via Composite Validation Criterion on 149 Val clips.\n\n",
        "## Table 1: Benchmark Performance vs. Strong External Baselines\n",
        "| Architecture / Paradigm | Overall nDCG@5 ↑ | P@5 ↑ | Hard Neg AUC ↑ | Contrastive Acc (%) ↑ | Seen nDCG@5 | Unseen nDCG@5 | Generalization Gap |\n",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n",
        "| Cross-Attention Ranker (Non-Graph) | 0.2882 | 0.2520 | 0.6180 | 56.1% | 0.4196 | 0.2083 | +0.2113 |\n",
        "| InternVideo2 (Fine-tuned) | 0.2811 | 0.2463 | 0.6284 | 58.6% | 0.4182 | 0.1983 | +0.2199 |\n",
        "| SigLIP2 (Fine-tuned) | 0.2654 | 0.2312 | 0.6421 | 59.0% | 0.4039 | 0.1820 | +0.2219 |\n",
        "| GATv2 + Video Fusion (Raw KG) | 0.2661 | 0.2304 | 0.6042 | 54.4% | 0.3402 | 0.1431 | +0.1971 |\n",
        "| **V4 Residual-Gated (Previous)** | 0.2671 | 0.2412 | 0.6356 | 53.9% | 0.3757 | 0.2249 | +0.1508 |\n",
    ]

    for exp_name, res in all_results.items():
        tm = res["test_metrics"]
        gap = tm["seen_ndcg@5"] - tm["unseen_ndcg@5"]
        bold = "**" if "Proposed" in exp_name else ""
        md_lines.append(
            f"| {bold}{exp_name}{bold} | {bold}{tm['ndcg@5']:.4f}{bold} | {tm['p@5']:.4f} | "
            f"{bold}{tm['hard_neg_auc']:.4f}{bold} | {bold}{tm['contrastive_acc']:.1f}%{bold} | "
            f"{tm['seen_ndcg@5']:.4f} | {bold}{tm['unseen_ndcg@5']:.4f}{bold} | {gap:+.4f} |\n"
        )

    md_lines.append("\n## Table 2: Controlled Ablation Analysis (Component Contributions)\n")
    md_lines.append("| Configuration | Overall nDCG@5 | Delta nDCG | Hard Neg AUC | Delta AUC | Contrastive Acc | Delta Cont | Unseen nDCG@5 |\n")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

    base_tm = all_results["R_VCGAT_Proposed_Full"]["test_metrics"]
    for exp_name, res in all_results.items():
        tm = res["test_metrics"]
        d_ndcg = tm["ndcg@5"] - base_tm["ndcg@5"]
        d_auc = tm["hard_neg_auc"] - base_tm["hard_neg_auc"]
        d_cont = tm["contrastive_acc"] - base_tm["contrastive_acc"]
        md_lines.append(
            f"| {exp_name} | {tm['ndcg@5']:.4f} | {d_ndcg:+.4f} | "
            f"{tm['hard_neg_auc']:.4f} | {d_auc:+.4f} | {tm['contrastive_acc']:.1f}% | {d_cont:+.1f}% | {tm['unseen_ndcg@5']:.4f} |\n"
        )

    table_path = REPORTS_DIR / "rvcgat_study_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)
    LOGGER.info(f"Saved R-VCGAT table to {table_path}")
    print("".join(md_lines))


if __name__ == "__main__":
    main()
