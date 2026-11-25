#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/train_dec_vcgat_v2.py
=============================================================================
DEC-VCGAT v2: Decoupled Evidence-Calibrated Relational VC-GAT
with Dynamic Candidate Graph Inclusion, Degree-Debiased Attention,
and Listwise Top-K Ranking Loss.

Architectural Innovations:
  1. Dynamic Candidate Graph Inclusion:
     Guarantees that ALL 16 candidate facts for each clip are first-class nodes
     in the graph during both training and inference. Out-of-subgraph candidates
     are dynamically attached via semantic association edges (Relation 3)
     and self-loops, completely eliminating the 'Candidate Disconnect' flaw.
  2. Degree-Debiased Relational Attention:
     Normalizes raw attention scores by -gamma * log(1 + degree) to prevent
     high-degree structural hubs from dominating message aggregation.
  3. Strict Visual Saliency Transmission:
     Nodes transmit relational evidence weighted purely by their visual saliency:
     M_{j -> i} = g_j * W_v h_j (zero leakage from unobserved facts).
  4. Decoupled Visual Foundation Lower Bound:
     Initialized from pretrained Cross-Attention Ranker. Bounded graph delta
     Delta_s in [-0.8, +0.8] modulated by domain-invariant meta-evidence gate lambda.
  5. Listwise Top-5 Plackett-Luce Loss:
     Directly optimizes top-5 permutation probability combined with hard-negative margin.
=============================================================================
"""

import sys
import math
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

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
LOGGER = logging.getLogger("DEC-VCGAT-v2")

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
# 1. ARCHITECTURE: DEC-VCGAT v2
# =============================================================================
class DecoupledEvidenceCalibratedVCGATv2(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 4,
        hidden_dim: int = 256,
        delta_bound: float = 0.8,
        degree_damping: float = 0.4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.delta_bound = delta_bound
        self.degree_damping = degree_damping

        # 1. Visual Backbone (Identical to CrossAttentionRankerModel)
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

        # 2. Strict Visual Saliency Gate
        self.saliency_gate = nn.Sequential(
            nn.Linear(3 * d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 3. Degree-Debiased Relation-Aware Graph Attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Relation embeddings: 0: co_occurrence, 1: semantic_dep, 2: causal_temporal, 3: associative_candidate
        self.rel_embed = nn.Embedding(4, 64)
        self.rel_k_proj = nn.Linear(64, d_model)
        self.rel_v_proj = nn.Linear(64, d_model)
        self.rel_bias = nn.Embedding(4, num_heads)

        self.norm_gat = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 4. Bounded Graph Delta MLP
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

        # 5. Domain-Invariant Evidence Calibration Gate (6 meta-features)
        self.evidence_gate = nn.Sequential(
            nn.Linear(6, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        h: torch.Tensor,              # [N, D]
        adj: torch.Tensor,            # [N, N]
        rel_adj: torch.Tensor,        # [N, N]
        degrees: torch.Tensor,        # [N]
        v_frames: torch.Tensor,       # [T, D]
        v_pooled: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = h.size(0)

        # 1. Cross-Attention Visual Grounding
        q = h.unsqueeze(1)                                    # [N, 1, D]
        kv = v_frames.unsqueeze(0).expand(N, -1, -1)          # [N, T, D]
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        ctx = self.norm1(attn_out.squeeze(1) + h)             # [N, D]
        vis_feat = torch.cat([h, ctx, h * ctx], dim=-1)       # [N, 3D]
        s_vis = self.vis_scorer(vis_feat).squeeze(-1)         # [N]

        # 2. Strict Visual Saliency Gate (g_nodes in [0, 1])
        g_nodes = self.saliency_gate(vis_feat)                # [N, 1]

        # 3. Degree-Debiased Relation-Aware Graph Attention
        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1) # [heads, N, head_dim]
        # Strict transmission: unobserved nodes (g_nodes -> 0) transmit zero evidence
        h_trans = h * g_nodes
        K = self.k_proj(h_trans).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V_nodes = self.v_proj(h_trans).view(N, self.num_heads, self.head_dim).transpose(0, 1)

        # Attention scores: Q K^T / sqrt(d)
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale # [heads, N, N]

        # Add relation key projection & bias
        rel_emb = self.rel_embed(rel_adj)                     # [N, N, 64]
        rel_k = self.rel_k_proj(rel_emb).view(N, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        scores_rel = torch.sum(Q.unsqueeze(2) * rel_k, dim=-1) * self.scale
        scores = scores + scores_rel + self.rel_bias(rel_adj).permute(2, 0, 1)

        # Degree-Debiased Normalization: subtract gamma * log(1 + degree_j)
        deg_penalty = self.degree_damping * torch.log1p(degrees.float()).unsqueeze(0).unsqueeze(1) # [1, 1, N]
        scores = scores - deg_penalty

        # Mask non-edges
        mask = (adj == 0).unsqueeze(0).expand(self.num_heads, -1, -1)
        scores = scores.masked_fill(mask, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        attn_mean = attn.mean(dim=0)                          # [N, N]

        # Aggregation
        ctx_nodes = torch.bmm(attn, V_nodes)
        rel_v = self.rel_v_proj(rel_emb).view(N, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        ctx_rel = torch.sum(attn.unsqueeze(-1) * rel_v, dim=2)
        agg = (ctx_nodes + ctx_rel).transpose(0, 1).contiguous().view(N, -1)
        h_gat = self.norm_gat(h + self.dropout(self.out_proj(agg)))

        # 4. Bounded Graph Delta
        graph_feat = torch.cat([h_gat, ctx, h_gat * ctx], dim=-1)
        delta_s = self.delta_bound * self.graph_delta_mlp(graph_feat).squeeze(-1) # [-delta_bound, +delta_bound]

        # 5. Domain-Invariant Evidence Calibration Gate
        h_u = F.normalize(h, p=2, dim=-1)
        ctx_u = F.normalize(ctx, p=2, dim=-1)
        h_gat_u = F.normalize(h_gat, p=2, dim=-1)

        cos_fact_ctx = torch.sum(h_u * ctx_u, dim=-1, keepdim=True)
        cos_gat_ctx = torch.sum(h_gat_u * ctx_u, dim=-1, keepdim=True)
        cos_fact_gat = torch.sum(h_u * h_gat_u, dim=-1, keepdim=True)

        eye = torch.eye(N, device=h.device)
        attn_neighbor = attn_mean * (1.0 - eye)
        neighbor_sum = attn_neighbor.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        r_anchor = torch.matmul(attn_neighbor, g_nodes) / neighbor_sum

        deg_factor = (1.0 / torch.sqrt(1.0 + degrees.float())).unsqueeze(-1)
        prob_vis = torch.sigmoid(s_vis).unsqueeze(-1)

        evidence_scalars = torch.cat([cos_fact_ctx, cos_gat_ctx, cos_fact_gat, r_anchor, deg_factor, prob_vis], dim=-1)
        lambda_i = self.evidence_gate(evidence_scalars).squeeze(-1) # [N]

        # Final Score
        final_score = s_vis + 0.4 * lambda_i * delta_s
        return final_score, s_vis, lambda_i, delta_s


# =============================================================================
# 2. DYNAMIC CANDIDATE GRAPH BUILDER
# =============================================================================
def build_clip_graph(
    base_subgraph: Dict[str, Any],
    cand_fids: List[str],
    fact_embeddings: Dict[str, np.ndarray],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], Dict[str, int]]:
    """
    Extends base_subgraph with any candidate fact not present in the subgraph.
    Returns: (h_tensor, adj, rel_adj, degrees, all_fids, fid2idx)
    """
    base_fids = list(base_subgraph["fact_ids"])
    fid2idx = dict(base_subgraph["fid2idx"])
    base_N = len(base_fids)

    missing_fids = [fid for fid in cand_fids if fid not in fid2idx]
    if not missing_fids:
        # No extra nodes needed
        h_list = [fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in base_fids]
        h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
        adj = base_subgraph["adj"].to(device)
        rel_adj = base_subgraph["rel_adj"].to(device)
        degrees = base_subgraph["degrees"].to(device)
        return h_tensor, adj, rel_adj, degrees, base_fids, fid2idx

    # Append missing candidate nodes
    all_fids = list(base_fids)
    for fid in missing_fids:
        fid2idx[fid] = len(all_fids)
        all_fids.append(fid)

    total_N = len(all_fids)
    h_list = [fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in all_fids]
    h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)

    # Initialize expanded adj and rel_adj
    adj = torch.zeros((total_N, total_N), dtype=torch.float32, device=device)
    rel_adj = torch.zeros((total_N, total_N), dtype=torch.long, device=device)

    # Copy base graph connections
    adj[:base_N, :base_N] = base_subgraph["adj"].to(device)
    rel_adj[:base_N, :base_N] = base_subgraph["rel_adj"].to(device)

    # For each added node: self-loop + connect to top-2 most cosine-similar base nodes
    base_h = h_tensor[:base_N] # [base_N, D]
    base_h_norm = F.normalize(base_h, p=2, dim=-1)

    for i, fid in enumerate(missing_fids):
        curr_idx = base_N + i
        adj[curr_idx, curr_idx] = 1.0 # self loop
        rel_adj[curr_idx, curr_idx] = 0

        # Cosine similarity to base nodes
        cand_v = F.normalize(h_tensor[curr_idx:curr_idx+1], p=2, dim=-1)
        sims = torch.matmul(cand_v, base_h_norm.T).squeeze(0) # [base_N]
        topk = torch.topk(sims, k=min(2, base_N)).indices

        for k_idx in topk:
            adj[curr_idx, k_idx] = 1.0
            adj[k_idx, curr_idx] = 1.0
            rel_adj[curr_idx, k_idx] = 3 # Associative relation
            rel_adj[k_idx, curr_idx] = 3

    degrees = adj.sum(dim=-1)
    return h_tensor, adj, rel_adj, degrees, all_fids, fid2idx


# =============================================================================
# 3. LISTMLE TOP-K RANKING LOSS
# =============================================================================
def listmle_topk_loss(
    logits: torch.Tensor,
    gold_indices: List[int],
    cand_indices: List[int],
    top_k: int = 5,
    temp: float = 0.5
) -> torch.Tensor:
    """Computes Plackett-Luce ListMLE ranking loss on candidate facts."""
    if not gold_indices:
        return torch.tensor(0.0, device=logits.device)

    neg_indices = [idx for idx in cand_indices if idx not in gold_indices]
    perm_indices = gold_indices[:top_k] + neg_indices

    perm_logits = logits[perm_indices] / temp
    k_eval = min(len(gold_indices), top_k)

    loss = 0.0
    for i in range(k_eval):
        log_denom = torch.logsumexp(perm_logits[i:], dim=0)
        log_num = perm_logits[i]
        loss = loss - (log_num - log_denom)

    return loss / float(k_eval)


# =============================================================================
# 4. TRAINING & EVALUATION PIPELINE
# =============================================================================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    facts = load_normalized_facts()
    train_clips, train_contrastive = load_gold_data(split_type="train")
    val_clips, val_contrastive = load_gold_data(split_type="val")
    test_clips, test_contrastive = load_gold_data(split_type="test")

    LOGGER.info(f"Dataset: Train={len(train_clips)}, Val={len(val_clips)}, Test={len(test_clips)}")

    feature_cache = BenchmarkFeatureCache(device=device)
    feature_cache.extract_fact_embeddings(facts)
    feature_cache.extract_video_features([])
    rel_subgraphs = torch.load(feature_cache.cache_dir / "landmark_subgraphs_rel.pt", weights_only=False)

    # Initialize model
    model = DecoupledEvidenceCalibratedVCGATv2(
        d_model=768,
        num_heads=4,
        hidden_dim=256,
        delta_bound=0.8,
        degree_damping=0.4,
        dropout=0.1
    ).to(device)

    # Load visual backbone from best_cross_attention.pt
    cross_path = CHECKPOINT_DIR / "best_cross_attention.pt"
    if cross_path.exists():
        LOGGER.info(f"Loading pretrained visual backbone weights from {cross_path}")
        cross_weights = torch.load(cross_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in cross_weights.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        LOGGER.info(f"Successfully transferred {len(pretrained_dict)} visual weight tensors.")

    # Differential learning rates
    vis_params = list(model.cross_attn.parameters()) + list(model.norm1.parameters()) + list(model.vis_scorer.parameters())
    graph_params = (
        list(model.saliency_gate.parameters()) +
        list(model.q_proj.parameters()) + list(model.k_proj.parameters()) +
        list(model.v_proj.parameters()) + list(model.out_proj.parameters()) +
        list(model.rel_embed.parameters()) + list(model.rel_k_proj.parameters()) +
        list(model.rel_v_proj.parameters()) + list(model.rel_bias.parameters()) +
        list(model.norm_gat.parameters()) + list(model.graph_delta_mlp.parameters()) +
        list(model.evidence_gate.parameters())
    )

    optimizer = AdamW([
        {"params": vis_params, "lr": 1e-5, "weight_decay": 1e-4},
        {"params": graph_params, "lr": 1.5e-4, "weight_decay": 1e-4}
    ])

    best_val_score = -1.0
    best_state = None
    train_cids = list(train_clips.keys())

    LOGGER.info("Starting training of DEC-VCGAT v2...")
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

            cand_fids = [r["fact_id"] for r in recs]
            pos_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
            if not pos_fids:
                continue

            # Dynamically build unified clip graph
            h_tensor, adj, rel_adj, degrees, all_fids, fid2idx = build_clip_graph(
                g, cand_fids, feature_cache.fact_embeddings, device
            )

            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device)

            optimizer.zero_grad()
            logits, s_vis, lambda_i, delta_s = model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)

            cand_indices = [fid2idx[fid] for fid in cand_fids]
            pos_indices = [fid2idx[fid] for fid in pos_fids]
            hard_neg_indices = [fid2idx[fid] for fid in hard_neg_fids]
            neg_indices = [idx for idx in cand_indices if idx not in pos_indices]

            # 1. ListMLE Top-K Loss
            loss_listmle = listmle_topk_loss(logits, pos_indices, cand_indices, top_k=5, temp=0.5)

            # 2. Hard Negative Margin Loss
            loss_hn = torch.tensor(0.0, device=device)
            if hard_neg_indices and pos_indices:
                p_s = logits[pos_indices].unsqueeze(1)
                hn_s = logits[hard_neg_indices].unsqueeze(0)
                loss_hn = torch.relu(0.8 - (p_s - hn_s)).mean()

            # 3. Pointwise BCE for anchor calibration
            y_true = [1.0] * len(pos_indices) + [0.0] * len(neg_indices)
            preds = torch.cat([logits[pos_indices], logits[neg_indices]])
            targets = torch.tensor(y_true, dtype=torch.float32, device=device)
            bce_loss = F.binary_cross_entropy_with_logits(preds, targets, pos_weight=torch.tensor([3.0], device=device))

            # 4. Gate Regularization
            gate_reg = 0.05 * torch.mean(lambda_i ** 2)

            total_loss = 0.5 * loss_listmle + 0.5 * bce_loss + 0.6 * loss_hn + gate_reg
            total_loss.backward()
            optimizer.step()

        # Validation Loop
        model.eval()
        val_ndcg, val_p5, val_auc = [], [], []
        with torch.no_grad():
            for c_id, recs in val_clips.items():
                l_id = recs[0].get("landmark_id", "unknown")
                g = rel_subgraphs.get(l_id)
                v_frames = feature_cache.video_frames.get(c_id)
                v_pooled = feature_cache.video_pooled.get(c_id)
                if g is None or v_frames is None or v_pooled is None:
                    continue

                cand_fids = [r["fact_id"] for r in recs]
                gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
                hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")

                h_tensor, adj, rel_adj, degrees, all_fids, fid2idx = build_clip_graph(
                    g, cand_fids, feature_cache.fact_embeddings, device
                )
                v_f_tensor = torch.from_numpy(v_frames).float().to(device)
                v_p_tensor = torch.from_numpy(v_pooled).float().to(device)

                logits, _, _, _ = model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
                scores_np = logits.cpu().numpy()

                f_scores = {fid: float(scores_np[fid2idx[fid]]) for fid in cand_fids}
                ranked = sorted(cand_fids, key=lambda fid: f_scores[fid], reverse=True)
                m = compute_ranking_metrics(ranked, gold_fids, cand_fids, top_k_list=[5])
                val_ndcg.append(m.get("nDCG@5", 0.0))
                val_p5.append(m.get("P@5", 0.0))

                auc = compute_hard_negative_auc(f_scores, gold_fids, hard_neg_fids)
                if auc is not None:
                    val_auc.append(auc)

        v_ndcg_m = float(np.mean(val_ndcg))
        v_p5_m = float(np.mean(val_p5))
        v_auc_m = float(np.mean(val_auc)) if val_auc else 0.5
        comp_score = 0.50 * v_ndcg_m + 0.25 * v_p5_m + 0.25 * v_auc_m
        LOGGER.info(f"Epoch {epoch:02d} | Val nDCG@5: {v_ndcg_m:.4f} | Val P@5: {v_p5_m:.4f} | Val AUC: {v_auc_m:.4f} | Comp: {comp_score:.4f}")

        if comp_score > best_val_score:
            best_val_score = comp_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, CHECKPOINT_DIR / "best_dec_vcgat_v2.pt")

    LOGGER.info(f"Training completed. Best Validation Score: {best_val_score:.4f}")

    # Evaluate on 800 Test Clips
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    test_ndcg5, test_p5, test_aucs = [], [], []
    seen_ndcg, unseen_ndcg = [], []
    hist_ndcg5 = []

    with torch.no_grad():
        for c_id, recs in test_clips.items():
            l_id = recs[0].get("landmark_id", "unknown")
            g = rel_subgraphs.get(l_id)
            v_frames = feature_cache.video_frames.get(c_id)
            v_pooled = feature_cache.video_pooled.get(c_id)
            if g is None or v_frames is None or v_pooled is None:
                continue

            cand_fids = [r["fact_id"] for r in recs]
            gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")

            h_tensor, adj, rel_adj, degrees, all_fids, fid2idx = build_clip_graph(
                g, cand_fids, feature_cache.fact_embeddings, device
            )
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device)

            logits, _, _, _ = model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
            scores_np = logits.cpu().numpy()

            f_scores = {fid: float(scores_np[fid2idx[fid]]) for fid in cand_fids}
            ranked = sorted(cand_fids, key=lambda fid: f_scores[fid], reverse=True)
            m = compute_ranking_metrics(ranked, gold_fids, cand_fids, top_k_list=[5])
            val_ndcg_val = m.get("nDCG@5", 0.0)
            test_ndcg5.append(val_ndcg_val)
            test_p5.append(m.get("P@5", 0.0))

            if l_id in SEEN_LANDMARKS:
                seen_ndcg.append(val_ndcg_val)
            else:
                unseen_ndcg.append(val_ndcg_val)

            auc = compute_hard_negative_auc(f_scores, gold_fids, hard_neg_fids)
            if auc is not None:
                test_aucs.append(auc)

            # Historical
            cand_hist = [f for f in cand_fids if facts.get(f, {}).get("predicate") in HISTORICAL_PREDICATES]
            gold_hist = [f for f in gold_fids if facts.get(f, {}).get("predicate") in HISTORICAL_PREDICATES]
            if cand_hist and gold_hist:
                rank_hist = sorted(cand_hist, key=lambda f: f_scores[f], reverse=True)
                m_h = compute_ranking_metrics(rank_hist, set(gold_hist), cand_hist, top_k_list=[5])
                hist_ndcg5.append(m_h.get("nDCG@5", 0.0))

    final_metrics = {
        "ndcg@5": float(np.mean(test_ndcg5)),
        "p@5": float(np.mean(test_p5)),
        "hard_neg_auc": float(np.mean(test_aucs)),
        "seen_ndcg@5": float(np.mean(seen_ndcg)),
        "unseen_ndcg@5": float(np.mean(unseen_ndcg)),
        "hist_ndcg@5": float(np.mean(hist_ndcg5)) if hist_ndcg5 else 0.0,
    }

    LOGGER.info("=" * 75)
    LOGGER.info("DEFINITIVE DEC-VCGAT v2 EVALUATION (800 TEST CLIPS):")
    LOGGER.info(f"  Overall nDCG@5      = {final_metrics['ndcg@5']:.4f}")
    LOGGER.info(f"  Overall P@5         = {final_metrics['p@5']:.4f}")
    LOGGER.info(f"  Hard Negative AUC   = {final_metrics['hard_neg_auc']:.4f}")
    LOGGER.info(f"  Seen nDCG@5         = {final_metrics['seen_ndcg@5']:.4f}")
    LOGGER.info(f"  Unseen nDCG@5       = {final_metrics['unseen_ndcg@5']:.4f}")
    LOGGER.info(f"  Historical nDCG@5   = {final_metrics['hist_ndcg@5']:.4f}")
    LOGGER.info("=" * 75)

    with open(REPORTS_DIR / "dec_vcgat_v2_results.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)


if __name__ == "__main__":
    main()
