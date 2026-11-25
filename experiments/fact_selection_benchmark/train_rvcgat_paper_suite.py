#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/train_rvcgat_paper_suite.py
=============================================================================
Definitive Paper Benchmark & Controlled Ablation Suite for IJAR:
Relation-Aware Video-Conditioned Graph Attention (R-VCGAT) with
Evidence-Based Reliability Gating and Scene-Conditioned Hard-Negative Margin.

Directly targets the IJAR Paper Research Objectives:
  1. Overall nDCG@5 >= 0.290 (Beating Cross-Attention 0.2882 & InternVideo2 0.2811)
  2. Unseen nDCG@5 >= 0.230 (Maintaining graph transferability on HCMC OOD)
  3. Hard Negative AUC >= 0.660 (Benchmark-best scene discrimination)
  4. Contrastive Accuracy >= 60.0% (Benchmark-best cross-scene sensitivity)

Controlled Ablation Matrix:
  - R_VCGAT_Proposed_Full
  - Ablation_wo_Relations
  - Ablation_wo_ReliabilityGate
  - Ablation_wo_Residual
  - Ablation_wo_SceneHNLoss
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
LOGGER = logging.getLogger("R-VCGAT-Suite")

CHECKPOINT_DIR = REPO_ROOT / "experiments" / "checkpoints_fact_selection"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
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
# MODEL ARCHITECTURE: R-VCGAT WITH CROSS-ATTENTION VISUAL BACKBONE
# =============================================================================
class FullUnifiedRVCGAT(nn.Module):
    """
    Relation-Aware Video-Conditioned Graph Attention (R-VCGAT).
    Decomposes scoring into:
      s(V, f_i) = s_vis(V_frames, f_i) + lambda_i(V, G) * Delta s_graph(V_frames, G_L^*, f_i)
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
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_relations = use_relations
        self.use_reliability_gate = use_reliability_gate
        self.use_residual = use_residual

        # 1. Cross-Attention Visual Backbone (Exact same capacity as B5)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(d_model)
        self.vis_scorer = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # 2. Node Saliency Gate
        self.saliency_gate = nn.Sequential(
            nn.Linear(3 * d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 3. Relation-Aware Graph Attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if self.use_relations:
            self.rel_embed = nn.Embedding(4, 64)
            self.rel_k_proj = nn.Linear(64, d_model)
            self.rel_v_proj = nn.Linear(64, d_model)
            self.rel_bias = nn.Embedding(4, num_heads)

        self.norm_gat = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 4. Graph Relational Delta Scorer
        self.graph_delta_scorer = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # 5. Evidence-Based Reliability Gate
        if self.use_reliability_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * d_model + 3, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, 1)
            )
            # Prior bias: -0.6 (ensures cautious initial gate ~0.35)
            nn.init.constant_(self.gate_mlp[-1].bias, -0.6)

    def forward(
        self,
        h: torch.Tensor,
        adj_mask: torch.Tensor,
        rel_adj: torch.Tensor,
        degrees: torch.Tensor,
        v_frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        h: [N, D] fact features
        adj_mask: [N, N] binary adjacency
        rel_adj: [N, N] relation types 0..3
        degrees: [N] node degrees
        v_frames: [K=8, D] video frames
        """
        if v_frames.dim() == 2:
            v_frames = v_frames.unsqueeze(0)  # [1, K, D]
        N = h.size(0)
        q = h.unsqueeze(1)                # [N, 1, D]
        kv = v_frames.expand(N, -1, -1)   # [N, K, D]

        # A. Visual Cross-Attention Backbone
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        ctx = self.norm_cross(attn_out.squeeze(1) + h)  # [N, D]
        vis_feat = torch.cat([h, ctx, h * ctx], dim=-1)  # [N, 3D]
        s_vis = self.vis_scorer(vis_feat).squeeze(-1)    # [N]

        # B. Node Saliency Gate
        g_nodes = self.saliency_gate(vis_feat)  # [N, 1]

        # C. Graph Attention (Relation-Aware or Homogeneous)
        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, d]
        h_gated = h * (0.5 + 0.5 * g_nodes)
        K = self.k_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, d]
        V_nodes = self.v_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, d]

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [H, N, N]

        if self.use_relations:
            rel_emb = self.rel_embed(rel_adj)  # [N, N, 64]
            rel_k = self.rel_k_proj(rel_emb).view(N, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
            q_exp = Q.unsqueeze(2)  # [H, N, 1, d]
            scores_rel = torch.sum(q_exp * rel_k, dim=-1) * self.scale
            scores = scores + scores_rel + self.rel_bias(rel_adj).permute(2, 0, 1)

        mask = (adj_mask == 0).unsqueeze(0).expand(self.num_heads, -1, -1)
        scores = scores.masked_fill(mask, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))  # [H, N, N]
        attn_mean = attn.mean(dim=0)  # [N, N]

        ctx_nodes = torch.bmm(attn, V_nodes)  # [H, N, d]
        if self.use_relations:
            rel_v = self.rel_v_proj(rel_emb).view(N, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
            ctx_rel = torch.sum(attn.unsqueeze(-1) * rel_v, dim=2)
            context = (ctx_nodes + ctx_rel).transpose(0, 1).contiguous().view(N, -1)
        else:
            context = ctx_nodes.transpose(0, 1).contiguous().view(N, -1)

        h_gat = self.norm_gat(h + self.dropout(self.out_proj(context)))

        # D. Graph Relational Delta Scorer
        graph_feat = torch.cat([h_gat, ctx, h_gat * ctx], dim=-1)
        s_graph_delta = self.graph_delta_scorer(graph_feat).squeeze(-1)

        # E. Evidence-Based Reliability Gate
        if self.use_reliability_gate:
            eye = torch.eye(N, device=h.device)
            attn_neighbor = attn_mean * (1.0 - eye)
            neighbor_sum = attn_neighbor.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            r_anchor = torch.matmul(attn_neighbor, g_nodes) / neighbor_sum

            h_gat_u = F.normalize(h_gat, p=2, dim=-1)
            ctx_u = F.normalize(ctx, p=2, dim=-1)
            cos_gat = torch.sum(h_gat_u * ctx_u, dim=-1, keepdim=True)
            deg_factor = (1.0 / torch.sqrt(1.0 + degrees.float())).unsqueeze(-1)

            gate_feat = torch.cat([h * ctx, h_gat * ctx, cos_gat, r_anchor, deg_factor], dim=-1)
            lambda_gate = torch.sigmoid(self.gate_mlp(gate_feat)).squeeze(-1)
        else:
            lambda_gate = torch.ones(N, device=h.device)

        # F. Residual Combination
        if not self.use_residual:
            final_score = s_graph_delta
        else:
            final_score = s_vis + lambda_gate * s_graph_delta

        return final_score, s_vis, lambda_gate


# =============================================================================
# EVALUATION ROUTINE
# =============================================================================
def evaluate_model_on_records(
    model: nn.Module,
    clip_records: Dict[str, List[Dict[str, Any]]],
    contrastive_pairs: List[Dict[str, Any]],
    rel_subgraphs: Dict[str, Dict[str, Any]],
    feature_cache: BenchmarkFeatureCache,
    device: torch.device,
    facts_dict: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, float]:
    """Evaluates the model over clips and contrastive pairs."""
    model.eval()
    flat_scores = {}
    ndcg5, p5, aucs = [], [], []
    seen_ndcg, unseen_ndcg = [], []
    seen_aucs, unseen_aucs = [], []
    vis_ndcg, hist_ndcg, inf_ndcg = [], [], []

    with torch.no_grad():
        for c_id, recs in clip_records.items():
            l_id = recs[0].get("landmark_id", "unknown")
            g = rel_subgraphs.get(l_id)
            v_frames = feature_cache.video_frames.get(c_id)
            if g is None or v_frames is None:
                continue

            sub_fids = g["fact_ids"]
            h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
            h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            adj = g["adj"].to(device)
            rel_adj = g["rel_adj"].to(device)
            degrees = g["degrees"].to(device)

            logits, _, _ = model(h_tensor, adj, rel_adj, degrees, v_f_tensor)
            scores_np = logits.cpu().numpy()

            idx_map = g["fid2idx"]
            f_scores = {}
            gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
            cand_fids = [r["fact_id"] for r in recs]

            for fid in cand_fids:
                s_val = float(scores_np[idx_map[fid]]) if fid in idx_map else -999.0
                f_scores[fid] = s_val
                flat_scores[(c_id, fid)] = s_val

            ranked_fids = sorted(cand_fids, key=lambda fid: f_scores[fid], reverse=True)
            m = compute_ranking_metrics(ranked_fids, gold_fids, cand_fids, top_k_list=[5, 16])
            c_ndcg = m.get("nDCG@5", 0.0)
            ndcg5.append(c_ndcg)
            p5.append(m.get("P@5", 0.0))

            auc = compute_hard_negative_auc(f_scores, gold_fids, hard_neg_fids)
            if auc is not None:
                aucs.append(auc)
                if l_id in SEEN_LANDMARKS:
                    seen_aucs.append(auc)
                else:
                    unseen_aucs.append(auc)

            if l_id in SEEN_LANDMARKS:
                seen_ndcg.append(c_ndcg)
            else:
                unseen_ndcg.append(c_ndcg)

            # Predicate breakdowns
            if facts_dict is not None:
                for pred_type, pred_list, res_list in [
                    ("visual", VISUAL_PREDICATES, vis_ndcg),
                    ("historical", HISTORICAL_PREDICATES, hist_ndcg),
                    ("inferred", INFERRED_PREDICATES, inf_ndcg),
                ]:
                    sub_cand = [f for f in cand_fids if facts_dict.get(f, {}).get("predicate") in pred_list]
                    sub_gold = [f for f in gold_fids if facts_dict.get(f, {}).get("predicate") in pred_list]
                    if sub_cand and sub_gold:
                        sub_ranked = sorted(sub_cand, key=lambda f: f_scores[f], reverse=True)
                        sub_m = compute_ranking_metrics(sub_ranked, set(sub_gold), sub_cand, top_k_list=[5])
                        res_list.append(sub_m.get("nDCG@5", 0.0))

    cont_acc = compute_contrastive_accuracy(flat_scores, contrastive_pairs)

    return {
        "ndcg@5": float(np.mean(ndcg5)) if ndcg5 else 0.0,
        "p@5": float(np.mean(p5)) if p5 else 0.0,
        "hard_neg_auc": float(np.mean(aucs)) if aucs else 0.0,
        "contrastive_acc": float(cont_acc),
        "seen_ndcg@5": float(np.mean(seen_ndcg)) if seen_ndcg else 0.0,
        "unseen_ndcg@5": float(np.mean(unseen_ndcg)) if unseen_ndcg else 0.0,
        "seen_auc": float(np.mean(seen_aucs)) if seen_aucs else 0.0,
        "unseen_auc": float(np.mean(unseen_aucs)) if unseen_aucs else 0.0,
        "vis_ndcg@5": float(np.mean(vis_ndcg)) if vis_ndcg else 0.0,
        "hist_ndcg@5": float(np.mean(hist_ndcg)) if hist_ndcg else 0.0,
        "inf_ndcg@5": float(np.mean(inf_ndcg)) if inf_ndcg else 0.0,
    }


# =============================================================================
# TRAINING ENGINE WITH SCENE-CONDITIONED MULTI-TASK LOSS
# =============================================================================
def train_and_evaluate_variant(
    config_name: str,
    model_kwargs: Dict[str, Any],
    use_scene_hn_loss: bool,
    train_clips: Dict[str, List[Dict[str, Any]]],
    val_clips: Dict[str, List[Dict[str, Any]]],
    test_clips: Dict[str, List[Dict[str, Any]]],
    train_contrastive: List[Dict[str, Any]],
    val_contrastive: List[Dict[str, Any]],
    test_contrastive: List[Dict[str, Any]],
    rel_subgraphs: Dict[str, Dict[str, Any]],
    feature_cache: BenchmarkFeatureCache,
    facts_dict: Dict[str, Dict[str, Any]],
    device: torch.device,
    epochs: int = 15,
    lr: float = 1.0e-4,
    contrastive_samples_per_epoch: int = 500,
    checkpoint_path: Optional[Path] = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Trains a model variant with strict validation selection and evaluates on test.
    Returns: (val_metrics, test_metrics)
    """
    LOGGER.info(f"================================================================")
    LOGGER.info(f"STARTING EXPERIMENT: {config_name}")
    LOGGER.info(f"Model args: {model_kwargs} | use_scene_hn_loss: {use_scene_hn_loss}")
    LOGGER.info(f"================================================================")

    model = FullUnifiedRVCGAT(**model_kwargs).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_cids = [c for c in train_clips.keys() if any(r.get("is_scene_gold") for r in train_clips[c])]

    best_val_score = -1.0
    best_state_dict = None
    best_val_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(train_cids)

        # 1. Main Clip Ranking Loop
        for c_id in train_cids:
            recs = train_clips[c_id]
            l_id = recs[0].get("landmark_id", "unknown")
            g = rel_subgraphs.get(l_id)
            v_frames = feature_cache.video_frames.get(c_id)
            if g is None or v_frames is None:
                continue

            pos_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
            all_neg_fids = set(r["fact_id"] for r in recs if not r.get("is_scene_gold"))
            if not pos_fids or not all_neg_fids:
                continue

            sub_fids = g["fact_ids"]
            h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
            h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            adj = g["adj"].to(device)
            rel_adj = g["rel_adj"].to(device)
            degrees = g["degrees"].to(device)

            optimizer.zero_grad()
            logits, _, _ = model(h_tensor, adj, rel_adj, degrees, v_f_tensor)
            idx_map = g["fid2idx"]

            pos_idx = [idx_map[fid] for fid in pos_fids if fid in idx_map]
            hard_neg_idx = [idx_map[fid] for fid in hard_neg_fids if fid in idx_map]
            neg_idx = [idx_map[fid] for fid in list(all_neg_fids)[: len(pos_idx) * 6] if fid in idx_map]

            if not pos_idx or not neg_idx:
                continue

            # A. Pointwise BCE Loss with positive class re-weighting
            y_true = [1.0] * len(pos_idx) + [0.0] * len(neg_idx)
            preds = torch.cat([logits[pos_idx], logits[neg_idx]])
            targets = torch.tensor(y_true, dtype=torch.float32, device=device)
            bce_loss = F.binary_cross_entropy_with_logits(
                preds, targets, pos_weight=torch.tensor([4.0], device=device)
            )

            # B. Top-Ranking Margin Loss (margin 0.6)
            p_scores = logits[pos_idx].unsqueeze(1)
            n_scores = logits[neg_idx].unsqueeze(0)
            rank_loss = torch.relu(0.6 - (p_scores - n_scores)).mean()

            total_loss = bce_loss + 0.8 * rank_loss

            # C. Scene-Conditioned Hard-Negative Margin Loss (margin 0.8)
            if use_scene_hn_loss and hard_neg_idx:
                hn_scores = logits[hard_neg_idx].unsqueeze(0)
                hn_loss = torch.relu(0.8 - (p_scores - hn_scores)).mean()
                total_loss = total_loss + 0.8 * hn_loss

            total_loss.backward()
            optimizer.step()

        # 2. Dense Intra-Landmark Cross-Scene Contrastive Alignment
        if train_contrastive and contrastive_samples_per_epoch > 0:
            np.random.shuffle(train_contrastive)
            sampled_pairs = train_contrastive[:contrastive_samples_per_epoch]

            for p in sampled_pairs:
                c_a = p.get("clip_a") or (p.get("clip_A", {}).get("clip_id") if isinstance(p.get("clip_A"), dict) else None)
                c_b = p.get("clip_b") or (p.get("clip_B", {}).get("clip_id") if isinstance(p.get("clip_B"), dict) else None)
                f_id = p.get("fact_id") or p.get("fact_a")
                l_id = p.get("landmark_id")

                if not c_a or not c_b or not f_id or not l_id:
                    continue

                v_fa = feature_cache.video_frames.get(c_a)
                v_fb = feature_cache.video_frames.get(c_b)
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
                s_a, _, _ = model(h_tensor, adj, rel_adj, degrees, torch.from_numpy(v_fa).float().to(device))
                s_b, _, _ = model(h_tensor, adj, rel_adj, degrees, torch.from_numpy(v_fb).float().to(device))

                # Contrastive margin 0.8 with balanced scaling
                cont_loss = 0.5 * torch.relu(0.8 - (s_a[f_idx] - s_b[f_idx]))
                if cont_loss > 0:
                    cont_loss.backward()
                    optimizer.step()

        # 3. Validation Evaluation & Strict Model Selection
        val_m = evaluate_model_on_records(
            model, val_clips, val_contrastive, rel_subgraphs, feature_cache, device, facts_dict=facts_dict
        )

        # Primary Selection Criterion: 0.70 nDCG@5 + 0.30 HardNegAUC (Ranking first, scene discrimination second)
        val_score = 0.70 * val_m["ndcg@5"] + 0.30 * val_m["hard_neg_auc"]

        LOGGER.info(
            f"[{config_name}] Epoch {epoch:02d} | Val nDCG@5: {val_m['ndcg@5']:.4f} | "
            f"Val AUC: {val_m['hard_neg_auc']:.4f} | Val ContAcc: {val_m['contrastive_acc']:.1f}% | "
            f"ValScore: {val_score:.4f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_metrics = val_m
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            LOGGER.info(f"  >>> New Best Checkpoint at Epoch {epoch:02d} (ValScore: {val_score:.4f})")

    # Save best checkpoint
    if checkpoint_path and best_state_dict is not None:
        torch.save(best_state_dict, checkpoint_path)
        LOGGER.info(f"Saved best model checkpoint to {checkpoint_path}")

    # Load best checkpoint and evaluate once on TEST
    model.load_state_dict(best_state_dict)
    test_m = evaluate_model_on_records(
        model, test_clips, test_contrastive, rel_subgraphs, feature_cache, device, facts_dict=facts_dict
    )

    LOGGER.info(
        f"[{config_name}] FINAL TEST EVALUATION: "
        f"nDCG@5: {test_m['ndcg@5']:.4f} | P@5: {test_m['p@5']:.4f} | "
        f"HardNeg AUC: {test_m['hard_neg_auc']:.4f} | ContAcc: {test_m['contrastive_acc']:.1f}% | "
        f"Seen nDCG: {test_m['seen_ndcg@5']:.4f} | Unseen nDCG: {test_m['unseen_ndcg@5']:.4f}"
    )

    return best_val_metrics, test_m


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all", choices=["all", "full_only", "ablation_only"])
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1.2e-4)
    parser.add_argument("--cont_samples", type=int, default=600)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Initializing R-VCGAT Paper Benchmark on Device: {device} | Mode: {args.mode}")

    facts = load_normalized_facts()
    facts_dict = facts
    LOGGER.info(f"Loaded {len(facts)} normalized facts.")

    train_clips, _ = load_gold_data(split_type="train")
    val_clips, _ = load_gold_data(split_type="val")
    test_clips, _ = load_gold_data(split_type="test")
    LOGGER.info(f"Splits loaded: Train {len(train_clips)}, Val {len(val_clips)}, Test {len(test_clips)} clips.")

    all_train_val_contrastive = load_contrastive_pairs(split_type="train_val")
    # Split train_val contrastive into train (first 2000) and val (remaining 684)
    train_contrastive = all_train_val_contrastive[:2000]
    val_contrastive = all_train_val_contrastive[2000:]
    test_contrastive = load_contrastive_pairs(split_type="test")
    LOGGER.info(f"Contrastive pairs: Train {len(train_contrastive)}, Val {len(val_contrastive)}, Test {len(test_contrastive)} pairs.")

    feature_cache = BenchmarkFeatureCache(device=device)
    feature_cache.extract_fact_embeddings(facts)
    feature_cache.extract_video_features([])

    rel_subgraphs_path = feature_cache.cache_dir / "landmark_subgraphs_rel.pt"
    if not rel_subgraphs_path.exists():
        raise FileNotFoundError(f"Missing {rel_subgraphs_path}! Run feature extraction first.")
    rel_subgraphs = torch.load(rel_subgraphs_path, weights_only=False)
    LOGGER.info(f"Loaded relation-aware subgraphs for {len(rel_subgraphs)} landmarks.")

    # Controlled Ablation Matrix Configurations
    experiments = [
        {
            "name": "R_VCGAT_Proposed_Full",
            "kwargs": {"d_model": 768, "num_heads": 4, "hidden_dim": 256, "dropout": 0.1,
                       "use_relations": True, "use_reliability_gate": True, "use_residual": True},
            "use_scene_hn_loss": True,
            "ckpt": CHECKPOINT_DIR / "best_paper_rvcgat_full.pt"
        },
        {
            "name": "Ablation_wo_Relations",
            "kwargs": {"d_model": 768, "num_heads": 4, "hidden_dim": 256, "dropout": 0.1,
                       "use_relations": False, "use_reliability_gate": True, "use_residual": True},
            "use_scene_hn_loss": True,
            "ckpt": CHECKPOINT_DIR / "best_paper_ablation_wo_relations.pt"
        },
        {
            "name": "Ablation_wo_ReliabilityGate",
            "kwargs": {"d_model": 768, "num_heads": 4, "hidden_dim": 256, "dropout": 0.1,
                       "use_relations": True, "use_reliability_gate": False, "use_residual": True},
            "use_scene_hn_loss": True,
            "ckpt": CHECKPOINT_DIR / "best_paper_ablation_wo_reliabilitygate.pt"
        },
        {
            "name": "Ablation_wo_Residual",
            "kwargs": {"d_model": 768, "num_heads": 4, "hidden_dim": 256, "dropout": 0.1,
                       "use_relations": True, "use_reliability_gate": True, "use_residual": False},
            "use_scene_hn_loss": True,
            "ckpt": CHECKPOINT_DIR / "best_paper_ablation_wo_residual.pt"
        },
        {
            "name": "Ablation_wo_SceneHNLoss",
            "kwargs": {"d_model": 768, "num_heads": 4, "hidden_dim": 256, "dropout": 0.1,
                       "use_relations": True, "use_reliability_gate": True, "use_residual": True},
            "use_scene_hn_loss": False,
            "ckpt": CHECKPOINT_DIR / "best_paper_ablation_wo_scenehnloss.pt"
        },
    ]

    if args.mode == "full_only":
        experiments = [e for e in experiments if e["name"] == "R_VCGAT_Proposed_Full"]
    elif args.mode == "ablation_only":
        experiments = [e for e in experiments if e["name"] != "R_VCGAT_Proposed_Full"]
    elif args.variant:
        experiments = [e for e in experiments if e["name"] == args.variant]

    # Load existing benchmark results if present to allow incremental updates
    json_path = REPORTS_DIR / "rvcgat_paper_suite_results.json"
    benchmark_results = {}
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                benchmark_results = json.load(f)
        except Exception:
            benchmark_results = {}

    for exp in experiments:
        val_m, test_m = train_and_evaluate_variant(
            config_name=exp["name"],
            model_kwargs=exp["kwargs"],
            use_scene_hn_loss=exp["use_scene_hn_loss"],
            train_clips=train_clips,
            val_clips=val_clips,
            test_clips=test_clips,
            train_contrastive=train_contrastive,
            val_contrastive=val_contrastive,
            test_contrastive=test_contrastive,
            rel_subgraphs=rel_subgraphs,
            feature_cache=feature_cache,
            facts_dict=facts_dict,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            contrastive_samples_per_epoch=args.cont_samples,
            checkpoint_path=exp["ckpt"]
        )

        benchmark_results[exp["name"]] = {
            "validation": val_m,
            "test": test_m
        }

        # Update JSON on each completed variant
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, indent=2)
        LOGGER.info(f"Updated results cache at {json_path}")

    # Save JSON results
    json_path = REPORTS_DIR / "rvcgat_paper_suite_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    LOGGER.info(f"Saved complete suite results to {json_path}")

    # Build Comparison Table with external baselines
    external_baselines = {
        "Cross-Attention Ranker (Non-Graph)": {"ndcg@5": 0.2882, "p@5": 0.2520, "hard_neg_auc": 0.6180, "contrastive_acc": 56.1, "seen_ndcg@5": 0.4196, "unseen_ndcg@5": 0.2083},
        "InternVideo2 (Fine-tuned)": {"ndcg@5": 0.2811, "p@5": 0.2463, "hard_neg_auc": 0.6284, "contrastive_acc": 58.6, "seen_ndcg@5": 0.4182, "unseen_ndcg@5": 0.1983},
        "SigLIP2 (Fine-tuned)": {"ndcg@5": 0.2654, "p@5": 0.2312, "hard_neg_auc": 0.6421, "contrastive_acc": 59.0, "seen_ndcg@5": 0.4039, "unseen_ndcg@5": 0.1820},
        "GATv2 + Video Fusion (Raw KG)": {"ndcg@5": 0.2661, "p@5": 0.2304, "hard_neg_auc": 0.6042, "contrastive_acc": 54.4, "seen_ndcg@5": 0.3402, "unseen_ndcg@5": 0.1431},
        "Graphormer + Video Fusion": {"ndcg@5": 0.1221, "p@5": 0.0988, "hard_neg_auc": 0.5287, "contrastive_acc": 52.1, "seen_ndcg@5": 0.1504, "unseen_ndcg@5": 0.1050},
        "Crisp-KG (Static Symbolic)": {"ndcg@5": 0.0331, "p@5": 0.0402, "hard_neg_auc": 0.4128, "contrastive_acc": 0.0, "seen_ndcg@5": 0.0410, "unseen_ndcg@5": 0.0283},
    }

    md_table = []
    md_table.append("# Definitive IJAR Fact Selection Benchmark & Controlled Ablation Table")
    md_table.append("Evaluated on 800 independent test clips (301 Seen Hanoi, 499 Unseen Hanoi+HCMC) over 143,124 candidate pairs and 9,040 contrastive pairs.\n")
    md_table.append("## Table 1: Main Comparison vs. Competitive External Baselines")
    md_table.append("| Model / Architecture | Overall nDCG@5 ↑ | P@5 ↑ | Hard Neg AUC ↑ | Contrastive Acc (%) ↑ | Seen nDCG@5 | Unseen nDCG@5 | Gap |")
    md_table.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for bname, bm in external_baselines.items():
        gap = bm["seen_ndcg@5"] - bm["unseen_ndcg@5"]
        md_table.append(f"| {bname} | {bm['ndcg@5']:.4f} | {bm['p@5']:.4f} | {bm['hard_neg_auc']:.4f} | {bm['contrastive_acc']:.1f}% | {bm['seen_ndcg@5']:.4f} | {bm['unseen_ndcg@5']:.4f} | +{gap:.4f} |")

    full_m = benchmark_results["R_VCGAT_Proposed_Full"]["test"]
    full_gap = full_m["seen_ndcg@5"] - full_m["unseen_ndcg@5"]
    md_table.append(f"| **R-VCGAT (Proposed Full)** | **{full_m['ndcg@5']:.4f}** | **{full_m['p@5']:.4f}** | **{full_m['hard_neg_auc']:.4f}** | **{full_m['contrastive_acc']:.1f}%** | **{full_m['seen_ndcg@5']:.4f}** | **{full_m['unseen_ndcg@5']:.4f}** | **+{full_gap:.4f}** |\n")

    md_table.append("## Table 2: Controlled Component Ablation Study")
    md_table.append("| Configuration | Overall nDCG@5 | Delta nDCG | Hard Neg AUC | Delta AUC | Contrastive Acc | Delta Cont | Unseen nDCG@5 |")
    md_table.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    base_ndcg = full_m["ndcg@5"]
    base_auc = full_m["hard_neg_auc"]
    base_cont = full_m["contrastive_acc"]

    for exp in experiments:
        name = exp["name"]
        tm = benchmark_results[name]["test"]
        d_ndcg = tm["ndcg@5"] - base_ndcg
        d_auc = tm["hard_neg_auc"] - base_auc
        d_cont = tm["contrastive_acc"] - base_cont
        sign_n = "+" if d_ndcg >= 0 else ""
        sign_a = "+" if d_auc >= 0 else ""
        sign_c = "+" if d_cont >= 0 else ""
        md_table.append(
            f"| {name} | {tm['ndcg@5']:.4f} | {sign_n}{d_ndcg:.4f} | {tm['hard_neg_auc']:.4f} | {sign_a}{d_auc:.4f} | {tm['contrastive_acc']:.1f}% | {sign_c}{d_cont:.1f}% | {tm['unseen_ndcg@5']:.4f} |"
        )

    md_content = "\n".join(md_table)
    table_path = REPORTS_DIR / "rvcgat_paper_suite_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    LOGGER.info(f"Saved markdown report to {table_path}")
    print("\n" + md_content)


if __name__ == "__main__":
    main()
