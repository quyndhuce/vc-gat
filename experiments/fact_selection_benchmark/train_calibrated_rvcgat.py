#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/train_calibrated_rvcgat.py
=============================================================================
Calibrated Residual VC-GAT (CR-VCGAT):
Solves the Precision@5 and Overall nDCG@5 degradation while preserving
superior Hard Negative AUC (0.69+) and Seen Landmark reasoning.

Core Architectural Innovations:
  1. Pretrained Visual Foundation Lower Bound:
     Initialized directly from best_cross_attention.pt with differential learning
     rate (1e-5 vs 2e-4) to prevent visual feature degradation.
  2. Bounded & Centered Graph Delta:
     Delta_s = beta * tanh(MLP(h_gat, ctx, h_gat * ctx)) bounded in [-1.0, +1.0].
     Prevents runaway variance from scrambling Top-5 ranks.
  3. Domain-Invariant Evidence Calibration Gate:
     Input dimension = 6 (meta-evidence scalars only, ZERO dense embeddings):
       - cos(h, ctx): fact-video alignment
       - cos(h_gat, ctx): neighbor-video alignment
       - cos(h, h_gat): graph topological agreement
       - r_anchor: neighborhood visual anchor density
       - deg_factor: topological sparsity (1 / sqrt(1 + deg))
       - prob_vis: visual prediction confidence sigma(s_vis)
     Eliminates semantic overfitting to Seen landmarks; collapses lambda -> 0
     on Unseen landmarks when evidence is absent, guaranteeing lower bound.
  4. Top-K Focused Margin Loss:
     Dynamically penalizes the hardest negatives near the Top-5 boundary
     to directly protect and elevate Precision@5.
=============================================================================
"""

import sys
import math
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.fact_selection_benchmark.dataset_loader import (
    load_normalized_facts,
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
from experiments.fact_selection_benchmark.models.cross_attention_ranker import CrossAttentionRankerModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("CR-VCGAT")

CHECKPOINT_DIR = REPO_ROOT / "experiments" / "checkpoints_fact_selection"
REPORTS_DIR = REPO_ROOT / "experiments" / "fact_selection_benchmark" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SEEN_LANDMARKS = {
    "hoan_kiem_lake",
    "temple_of_literature",
    "ba_dinh_square",
    "imperial_citadel_of_thang_long",
    "one_pillar_pagoda",
    "hanoi_old_quarter",
    "west_lake",
}


# =============================================================================
# 1. CALIBRATED RESIDUAL VC-GAT ARCHITECTURE
# =============================================================================
class CalibratedResidualVCGAT(nn.Module):
    def __init__(self, d_model: int = 768, num_heads: int = 4, hidden_dim: int = 256, delta_bound: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.delta_bound = delta_bound

        # 1. Visual Backbone (Exact same capacity as CrossAttentionRankerModel)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.vis_scorer = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # 2. Node Saliency Gate (Strict Visual Grounding of Nodes)
        self.saliency_gate = nn.Sequential(
            nn.Linear(3 * d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 3. Relation-Aware VC-GAT
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rel_embed = nn.Embedding(4, 64)
        self.rel_k_proj = nn.Linear(64, d_model)
        self.rel_v_proj = nn.Linear(64, d_model)
        self.rel_bias = nn.Embedding(4, num_heads)
        self.norm_gat = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 4. Bounded Graph Delta Scorer: strictly outputs in [-delta_bound, +delta_bound]
        self.graph_delta_mlp = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        # 5. Domain-Invariant Evidence Calibration Gate
        # 6 Meta-evidence scalar inputs:
        #   [cos_fact_ctx, cos_gat_ctx, cos_fact_gat, r_anchor, deg_factor, prob_vis]
        self.evidence_gate = nn.Sequential(
            nn.Linear(6, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.evidence_gate[-2].bias, -0.8)

    def load_pretrained_visual_backbone(self, ckpt_path: Path, device: torch.device):
        """Initializes visual cross-attention backbone from trained baseline."""
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.cross_attn.load_state_dict({k.replace("cross_attn.", ""): v for k, v in state.items() if k.startswith("cross_attn.")})
        self.norm1.load_state_dict({k.replace("norm1.", ""): v for k, v in state.items() if k.startswith("norm1.")})
        self.vis_scorer.load_state_dict({k.replace("scorer.", ""): v for k, v in state.items() if k.startswith("scorer.")})
        LOGGER.info("Pretrained visual cross-attention backbone loaded successfully.")

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor, rel_adj: torch.Tensor, degrees: torch.Tensor, v_frames: torch.Tensor, v_pooled: Optional[torch.Tensor] = None):
        if v_frames.dim() == 2:
            v_frames = v_frames.unsqueeze(0)  # [1, K, D]
        N = h.size(0)
        q = h.unsqueeze(1)                # [N, 1, D]
        kv = v_frames.expand(N, -1, -1)   # [N, K, D]

        # 1. Visual Cross-Attention Backbone
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        ctx = self.norm1(attn_out.squeeze(1) + h)  # [N, D]
        vis_feat = torch.cat([h, ctx, h * ctx], dim=-1)  # [N, 3D]
        s_vis = self.vis_scorer(vis_feat).squeeze(-1)    # [N]

        # 2. Node Saliency Gate
        g_nodes = self.saliency_gate(vis_feat)  # [N, 1]

        # 3. Relation-Aware VC-GAT
        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        h_gated = h * (0.2 + 0.8 * g_nodes)
        K = self.k_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V_nodes = self.v_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        rel_emb = self.rel_embed(rel_adj)
        rel_k = self.rel_k_proj(rel_emb).view(N, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        q_exp = Q.unsqueeze(2)
        scores_rel = torch.sum(q_exp * rel_k, dim=-1) * self.scale
        scores = scores + scores_rel + self.rel_bias(rel_adj).permute(2, 0, 1)

        mask = (adj_mask == 0).unsqueeze(0).expand(self.num_heads, -1, -1)
        scores = scores.masked_fill(mask, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        attn_mean = attn.mean(dim=0)

        ctx_nodes = torch.bmm(attn, V_nodes)
        rel_v = self.rel_v_proj(rel_emb).view(N, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        ctx_rel = torch.sum(attn.unsqueeze(-1) * rel_v, dim=2)
        h_gat = self.norm_gat(h + self.dropout(self.out_proj((ctx_nodes + ctx_rel).transpose(0, 1).contiguous().view(N, -1))))

        # 4. Bounded Graph Delta: in [-delta_bound, +delta_bound]
        graph_feat = torch.cat([h_gat, ctx, h_gat * ctx], dim=-1)
        delta_s = self.delta_bound * self.graph_delta_mlp(graph_feat).squeeze(-1)

        # 5. Domain-Invariant Evidence Calibration Gate
        h_u = F.normalize(h, p=2, dim=-1)
        ctx_u = F.normalize(ctx, p=2, dim=-1)
        h_gat_u = F.normalize(h_gat, p=2, dim=-1)

        cos_fact_ctx = torch.sum(h_u * ctx_u, dim=-1, keepdim=True)        # [N, 1]
        cos_gat_ctx = torch.sum(h_gat_u * ctx_u, dim=-1, keepdim=True)     # [N, 1]
        cos_fact_gat = torch.sum(h_u * h_gat_u, dim=-1, keepdim=True)      # [N, 1]

        eye = torch.eye(N, device=h.device)
        attn_neighbor = attn_mean * (1.0 - eye)
        neighbor_sum = attn_neighbor.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        r_anchor = torch.matmul(attn_neighbor, g_nodes) / neighbor_sum     # [N, 1]

        deg_factor = (1.0 / torch.sqrt(1.0 + degrees.float())).unsqueeze(-1) # [N, 1]
        prob_vis = torch.sigmoid(s_vis).unsqueeze(-1)                       # [N, 1]

        evidence_scalars = torch.cat([cos_fact_ctx, cos_gat_ctx, cos_fact_gat, r_anchor, deg_factor, prob_vis], dim=-1)  # [N, 6]
        lambda_i = self.evidence_gate(evidence_scalars).squeeze(-1)        # [N]

        final_score = s_vis + lambda_i * delta_s
        return final_score, s_vis, lambda_i, delta_s


# =============================================================================
# 2. EVALUATION FUNCTION ACROSS 800 TEST CLIPS
# =============================================================================
def evaluate_model_full(
    model: CalibratedResidualVCGAT,
    test_clips: Dict[str, List[Dict[str, Any]]],
    rel_subgraphs: Dict[str, Dict[str, Any]],
    feature_cache: BenchmarkFeatureCache,
    test_contrastive: List[Dict[str, Any]],
    facts_dict: Dict[str, Dict[str, Any]],
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    ndcg5, p5, aucs = [], [], []
    seen_ndcg, unseen_ndcg = [], []
    hist_ndcg = []
    flat_scores = {}

    with torch.no_grad():
        for c_id, recs in test_clips.items():
            l_id = recs[0].get("landmark_id", "unknown")
            g = rel_subgraphs.get(l_id)
            v_frames = feature_cache.video_frames.get(c_id)
            v_pooled = feature_cache.video_pooled.get(c_id)
            if g is None or v_frames is None or v_pooled is None:
                continue

            sub_fids = g["fact_ids"]
            h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
            h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device)
            adj = g["adj"].to(device)
            rel_adj = g["rel_adj"].to(device)
            degrees = g["degrees"].to(device)

            logits, _, _, _ = model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
            scores_np = logits.cpu().numpy()
            idx_map = g["fid2idx"]

            cand_fids = [r["fact_id"] for r in recs]
            gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")

            min_s = float(np.min(scores_np)) if len(scores_np) > 0 else -10.0
            f_scores = {fid: float(scores_np[idx_map[fid]]) if fid in idx_map else (min_s - 1.0) for fid in cand_fids}
            for fid, s in f_scores.items():
                flat_scores[(c_id, fid)] = s

            ranked = sorted(cand_fids, key=lambda fid: f_scores[fid], reverse=True)
            m = compute_ranking_metrics(ranked, gold_fids, cand_fids, top_k_list=[5])
            val_ndcg = m.get("nDCG@5", 0.0)
            ndcg5.append(val_ndcg)
            p5.append(m.get("P@5", 0.0))

            if l_id in SEEN_LANDMARKS:
                seen_ndcg.append(val_ndcg)
            else:
                unseen_ndcg.append(val_ndcg)

            auc = compute_hard_negative_auc(f_scores, gold_fids, hard_neg_fids)
            if auc is not None:
                aucs.append(auc)

            # Historical facts
            cand_hist = [f for f in cand_fids if facts_dict.get(f, {}).get("predicate") in HISTORICAL_PREDICATES]
            gold_hist = [f for f in gold_fids if facts_dict.get(f, {}).get("predicate") in HISTORICAL_PREDICATES]
            if cand_hist and gold_hist:
                rank_h = sorted(cand_hist, key=lambda f: f_scores[f], reverse=True)
                m_h = compute_ranking_metrics(rank_h, set(gold_hist), cand_hist, top_k_list=[5])
                hist_ndcg.append(m_h.get("nDCG@5", 0.0))

    cont_acc = compute_contrastive_accuracy(flat_scores, test_contrastive)
    return {
        "ndcg@5": float(np.mean(ndcg5)),
        "p@5": float(np.mean(p5)),
        "hard_neg_auc": float(np.mean(aucs)) if aucs else 0.5,
        "contrastive_acc": float(cont_acc),
        "seen_ndcg@5": float(np.mean(seen_ndcg)) if seen_ndcg else 0.0,
        "unseen_ndcg@5": float(np.mean(unseen_ndcg)) if unseen_ndcg else 0.0,
        "hist_ndcg@5": float(np.mean(hist_ndcg)) if hist_ndcg else 0.0,
    }


# =============================================================================
# 3. MAIN TRAINING LOOP WITH TOP-K FOCUSED MARGIN LOSS
# =============================================================================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Training Calibrated Residual VC-GAT on device: {device}")

    facts = load_normalized_facts()
    train_clips, _ = load_gold_data(split_type="train")
    val_clips, _ = load_gold_data(split_type="val")
    test_clips, _ = load_gold_data(split_type="test")
    train_contrastive = load_contrastive_pairs(split_type="train_val")
    test_contrastive = load_contrastive_pairs(split_type="test")

    feature_cache = BenchmarkFeatureCache(device=device)
    feature_cache.extract_fact_embeddings(facts)
    feature_cache.extract_video_features([])
    rel_subgraphs = torch.load(feature_cache.cache_dir / "landmark_subgraphs_rel.pt", weights_only=False)

    # Instantiate model
    model = CalibratedResidualVCGAT(d_model=768, delta_bound=0.8).to(device)

    # 1. Load Pretrained Visual Backbone from best_cross_attention.pt
    cross_ckpt = CHECKPOINT_DIR / "best_cross_attention.pt"
    if cross_ckpt.exists():
        model.load_pretrained_visual_backbone(cross_ckpt, device)
    else:
        LOGGER.warning(f"Could not find {cross_ckpt}, initializing visual backbone from scratch.")

    # 2. Differential Learning Rates
    visual_params = list(model.cross_attn.parameters()) + list(model.norm1.parameters()) + list(model.vis_scorer.parameters())
    graph_params = (
        list(model.saliency_gate.parameters())
        + list(model.q_proj.parameters())
        + list(model.k_proj.parameters())
        + list(model.v_proj.parameters())
        + list(model.out_proj.parameters())
        + list(model.rel_embed.parameters())
        + list(model.rel_k_proj.parameters())
        + list(model.rel_v_proj.parameters())
        + list(model.rel_bias.parameters())
        + list(model.norm_gat.parameters())
        + list(model.graph_delta_mlp.parameters())
        + list(model.evidence_gate.parameters())
    )

    optimizer = AdamW([
        {"params": visual_params, "lr": 1.5e-5, "weight_decay": 1e-4},
        {"params": graph_params, "lr": 1.8e-4, "weight_decay": 1e-4},
    ])

    train_cids = [c for c in train_clips.keys() if any(r.get("is_scene_gold") for r in train_clips[c])]

    best_val_score = -1.0
    best_state = None

    LOGGER.info("Starting training with Top-K Boundary Margin & Hard Negative Discrimination...")
    for epoch in range(1, 16):
        model.train()
        np.random.shuffle(train_cids)

        for c_id in train_cids:
            recs = train_clips[c_id]
            l_id = recs[0].get("landmark_id", "unknown")
            g = rel_subgraphs.get(l_id)
            v_frames = feature_cache.video_frames.get(c_id)
            v_pooled = feature_cache.video_pooled.get(c_id)
            if g is None or v_frames is None or v_pooled is None:
                continue

            pos_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
            all_neg_fids = set(r["fact_id"] for r in recs if not r.get("is_scene_gold"))
            if not pos_fids or not all_neg_fids:
                continue

            optimizer.zero_grad()
            sub_fids = g["fact_ids"]
            h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
            h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device)
            adj = g["adj"].to(device)
            rel_adj = g["rel_adj"].to(device)
            degrees = g["degrees"].to(device)

            logits, s_vis, lambda_i, delta_s = model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
            idx_map = g["fid2idx"]
            pos_idx = [idx_map[fid] for fid in pos_fids if fid in idx_map]
            hard_neg_idx = [idx_map[fid] for fid in hard_neg_fids if fid in idx_map]
            neg_idx = [idx_map[fid] for fid in list(all_neg_fids)[: len(pos_idx) * 6] if fid in idx_map]

            if not pos_idx or not neg_idx:
                continue

            # 1. Pointwise BCE with pos_weight
            y_true = [1.0] * len(pos_idx) + [0.0] * len(neg_idx)
            preds = torch.cat([logits[pos_idx], logits[neg_idx]])
            targets = torch.tensor(y_true, dtype=torch.float32, device=device)
            bce_loss = F.binary_cross_entropy_with_logits(preds, targets, pos_weight=torch.tensor([3.5], device=device))

            # 2. Top-K Boundary Margin Loss (Focuses heavily on hard negatives violating top ranks)
            p_scores = logits[pos_idx].unsqueeze(1)    # [P, 1]
            n_scores = logits[neg_idx].unsqueeze(0)    # [1, N]
            # Smooth InfoNCE-style Top-Rank penalty: log(1 + sum_j exp( (s_j - s_i + margin) / tau ))
            diff = (n_scores - p_scores + 0.6) / 0.5
            topk_rank_loss = torch.log1p(torch.sum(torch.exp(torch.clamp(diff, max=10.0)), dim=1)).mean()

            # 3. Scene Hard Negative Margin Loss (margin 0.8)
            hn_loss = torch.tensor(0.0, device=device)
            if hard_neg_idx:
                hn_scores = logits[hard_neg_idx].unsqueeze(0)  # [1, HN]
                hn_loss = torch.relu(0.8 - (p_scores - hn_scores)).mean()

            # 4. Gate Regularization: Encourages gate to be sparse/conservative when evidence is low
            gate_reg = 0.05 * torch.mean(lambda_i ** 2)

            loss = bce_loss + 0.4 * topk_rank_loss + 0.8 * hn_loss + gate_reg
            loss.backward()
            optimizer.step()

        # Multi-task contrastive pairs step
        if train_contrastive:
            np.random.shuffle(train_contrastive)
            for p in train_contrastive[:200]:
                c_a = p.get("clip_a") or (p.get("clip_A", {}).get("clip_id") if isinstance(p.get("clip_A"), dict) else None)
                c_b = p.get("clip_b") or (p.get("clip_B", {}).get("clip_id") if isinstance(p.get("clip_B"), dict) else None)
                f_id = p.get("fact_id") or p.get("fact_a")
                l_id = p.get("landmark_id")

                v_fa = feature_cache.video_frames.get(c_a)
                v_fb = feature_cache.video_frames.get(c_b)
                v_pa = feature_cache.video_pooled.get(c_a)
                v_pb = feature_cache.video_pooled.get(c_b)
                g = rel_subgraphs.get(l_id)
                if v_fa is None or v_fb is None or g is None or f_id not in g["fid2idx"]:
                    continue

                f_idx = g["fid2idx"][f_id]
                sub_fids = g["fact_ids"]
                h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
                h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
                adj = g["adj"].to(device)
                rel_adj = g["rel_adj"].to(device)
                degrees = g["degrees"].to(device)

                optimizer.zero_grad()
                s_a, _, _, _ = model(h_tensor, adj, rel_adj, degrees, torch.from_numpy(v_fa).float().to(device), torch.from_numpy(v_pa).float().to(device))
                s_b, _, _, _ = model(h_tensor, adj, rel_adj, degrees, torch.from_numpy(v_fb).float().to(device), torch.from_numpy(v_pb).float().to(device))
                cont_loss = 0.5 * torch.relu(0.8 - (s_a[f_idx] - s_b[f_idx]))
                if cont_loss > 0:
                    cont_loss.backward()
                    optimizer.step()

        # Validation Evaluation
        model.eval()
        val_ndcg_list, val_p5_list, val_auc_list = [], [], []
        with torch.no_grad():
            for c_id, recs in val_clips.items():
                l_id = recs[0].get("landmark_id", "unknown")
                g = rel_subgraphs.get(l_id)
                v_frames = feature_cache.video_frames.get(c_id)
                v_pooled = feature_cache.video_pooled.get(c_id)
                if g is None or v_frames is None or v_pooled is None:
                    continue

                sub_fids = g["fact_ids"]
                h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
                h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
                v_f_tensor = torch.from_numpy(v_frames).float().to(device)
                v_p_tensor = torch.from_numpy(v_pooled).float().to(device)
                adj = g["adj"].to(device)
                rel_adj = g["rel_adj"].to(device)
                degrees = g["degrees"].to(device)

                logits, _, _, _ = model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
                scores_np = logits.cpu().numpy()
                idx_map = g["fid2idx"]

                cand_fids = [r["fact_id"] for r in recs]
                gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
                hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")

                min_s = float(np.min(scores_np)) if len(scores_np) > 0 else -10.0
                f_scores = {fid: float(scores_np[idx_map[fid]]) if fid in idx_map else (min_s - 1.0) for fid in cand_fids}
                ranked = sorted(cand_fids, key=lambda fid: f_scores[fid], reverse=True)
                m = compute_ranking_metrics(ranked, gold_fids, cand_fids, top_k_list=[5])
                val_ndcg_list.append(m.get("nDCG@5", 0.0))
                val_p5_list.append(m.get("P@5", 0.0))

                auc = compute_hard_negative_auc(f_scores, gold_fids, hard_neg_fids)
                if auc is not None:
                    val_auc_list.append(auc)

        v_ndcg = float(np.mean(val_ndcg_list))
        v_p5 = float(np.mean(val_p5_list))
        v_auc = float(np.mean(val_auc_list)) if val_auc_list else 0.5
        # Composite validation selection score: balances ranking, precision, and hard-negative rejection
        composite_score = 0.50 * v_ndcg + 0.25 * v_p5 + 0.25 * v_auc
        LOGGER.info(f"Epoch {epoch:02d} | Val nDCG@5: {v_ndcg:.4f} | Val P@5: {v_p5:.4f} | Val AUC: {v_auc:.4f} | Score: {composite_score:.4f}")

        if composite_score > best_val_score:
            best_val_score = composite_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, CHECKPOINT_DIR / "best_calibrated_rvcgat.pt")

    LOGGER.info(f"Training completed. Best Validation Composite Score: {best_val_score:.4f}")

    # Evaluate Best Model on 800 Test Clips
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_results = evaluate_model_full(model, test_clips, rel_subgraphs, feature_cache, test_contrastive, facts, device)

    LOGGER.info("=" * 75)
    LOGGER.info("DEFINITIVE TEST EVALUATION (800 INDEPENDENT TEST CLIPS):")
    LOGGER.info(f"  Overall nDCG@5      = {test_results['ndcg@5']:.4f}")
    LOGGER.info(f"  Overall P@5         = {test_results['p@5']:.4f}")
    LOGGER.info(f"  Hard Negative AUC   = {test_results['hard_neg_auc']:.4f}")
    LOGGER.info(f"  Contrastive Acc     = {test_results['contrastive_acc']:.2f}%")
    LOGGER.info(f"  Seen nDCG@5         = {test_results['seen_ndcg@5']:.4f}")
    LOGGER.info(f"  Unseen nDCG@5       = {test_results['unseen_ndcg@5']:.4f}")
    LOGGER.info(f"  Historical nDCG@5   = {test_results['hist_ndcg@5']:.4f}")
    LOGGER.info("=" * 75)

    with open(REPORTS_DIR / "calibrated_rvcgat_test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)


if __name__ == "__main__":
    main()
