#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/train_and_benchmark_horn_suite.py
=============================================================================
Definitive Horn-Complete Fact Selection Suite & Benchmark.

Evaluates on the verified Horn-densified Deductive Closure Graph (100% candidate coverage):
  1. SigLIP2 Zero-Shot Baseline
  2. Cross-Attention Ranker (Supervised Non-Graph Baseline)
  3. GATv2 Ranker (Standard Graph Baseline)
  4. DEC-VCGAT v2 Proposed Full Model
  5. DEC-VCGAT v2 Ablations:
     - w/o Evidence Calibration Gate (lambda = 1.0 fixed)
     - w/o Degree Debiasing (gamma = 0.0)
     - w/o Strict Visual Saliency Transmission (h_trans = h)

Evaluated across:
  - All 800 Test Clips (Overall nDCG@5, P@5, R@5)
  - Seen Landmarks (500 Hanoi Clips) vs Unseen Landmarks (300 HCMC Clips)
  - Hard Negative AUC
  - Historical Facts nDCG@5
  - Contrastive Accuracy (128,046 Challenge Pairs)
  - Paired Bootstrap Statistical Significance (1,000 resamples, 95% CI, p-value)
=============================================================================
"""

import sys
import math
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.fact_selection_benchmark.dataset_loader import (
    compute_ranking_metrics,
    compute_hard_negative_auc,
    get_fact_type,
    HISTORICAL_PREDICATES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("HornSuite")

CHECKPOINT_DIR = REPO_ROOT / "experiments" / "checkpoints_fact_selection"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = REPO_ROOT / "experiments" / "fact_selection_benchmark" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = REPO_ROOT / "experiments" / "features" / "benchmark_cache"

SEEN_LANDMARKS = {
    "hoan_kiem_lake",
    "temple_of_literature",
    "ba_dinh_square",
    "imperial_citadel_of_thang_long",
    "one_pillar_pagoda",
    "hanoi_old_quarter",
    "west_lake",
}

FACTS_HORN_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "normalized_facts_horn_complete.jsonl"
TEST_GOLD_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "scene_conditioned_gold_horn_complete.jsonl"
TRAIN_VAL_GOLD_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "scene_conditioned_gold_train_val_horn_complete.jsonl"
TEST_CONTRASTIVE_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "contrastive_pairs_horn_complete.jsonl"


# =============================================================================
# 1. MODEL ARCHITECTURES
# =============================================================================

class CrossAttentionRankerModel(nn.Module):
    """Supervised Cross-Attention Ranker (Non-Graph Multimodal Backbone)."""

    def __init__(self, d_model: int = 768, num_heads: int = 4, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
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

    def forward(self, v_frames: torch.Tensor, h_facts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = h_facts.size(0)
        q = h_facts.unsqueeze(1)
        kv = v_frames.unsqueeze(0).expand(N, -1, -1)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        ctx = self.norm1(attn_out.squeeze(1) + h_facts)
        feat = torch.cat([h_facts, ctx, h_facts * ctx], dim=-1)
        logits = self.vis_scorer(feat).squeeze(-1)
        return logits, ctx


class GATv2RankerModel(nn.Module):
    """Standard GATv2 Graph Baseline with Video Conditioning."""

    def __init__(self, d_model: int = 768, num_heads: int = 4, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.v_proj_fusion = nn.Linear(d_model, d_model)
        self.scorer = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, h: torch.Tensor, adj: torch.Tensor, v_pooled: torch.Tensor) -> torch.Tensor:
        N = h.size(0)
        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.k_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        mask = (adj == 0).unsqueeze(0).expand(self.num_heads, -1, -1)
        scores = scores.masked_fill(mask, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))

        ctx = torch.bmm(attn, V).transpose(0, 1).contiguous().view(N, self.d_model)
        h_gat = self.norm(h + self.dropout(self.out_proj(ctx)))

        v_exp = v_pooled.unsqueeze(0).expand(N, -1)
        feat = torch.cat([h_gat, v_exp, h_gat * v_exp], dim=-1)
        return self.scorer(feat).squeeze(-1)


class ModularDECV2Model(nn.Module):
    """Modular DEC-VCGAT v2 with Configurable Ablation Switches."""

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 4,
        hidden_dim: int = 256,
        delta_bound: float = 0.8,
        degree_damping: float = 0.4,       # set to 0.0 for w/o debiasing
        use_strict_saliency: bool = True,  # set to False for w/o saliency
        use_evidence_gate: bool = True,    # set to False for w/o gate
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.delta_bound = delta_bound
        self.degree_damping = degree_damping
        self.use_strict_saliency = use_strict_saliency
        self.use_evidence_gate = use_evidence_gate

        # 1. Visual Backbone
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

        # 5. Domain-Invariant Evidence Calibration Gate
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = h.size(0)

        # 1. Visual Grounding
        q = h.unsqueeze(1)
        kv = v_frames.unsqueeze(0).expand(N, -1, -1)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        ctx = self.norm1(attn_out.squeeze(1) + h)
        vis_feat = torch.cat([h, ctx, h * ctx], dim=-1)
        s_vis = self.vis_scorer(vis_feat).squeeze(-1)

        # 2. Visual Saliency
        if self.use_strict_saliency:
            g_nodes = self.saliency_gate(vis_feat)
            h_trans = h * g_nodes
        else:
            g_nodes = torch.ones((N, 1), device=h.device)
            h_trans = h

        # 3. Relational Attention (Exact O(N*D) formulation)
        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1) # [H, N, d]
        K = self.k_proj(h_trans).view(N, self.num_heads, self.head_dim).transpose(0, 1) # [H, N, d]
        V_nodes = self.v_proj(h_trans).view(N, self.num_heads, self.head_dim).transpose(0, 1) # [H, N, d]

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale # [H, N, N]

        # Fast exact relation key projection
        rel_types = torch.arange(4, device=h.device)
        rel_emb_table = self.rel_embed(rel_types) # [4, 64]
        K_rel = self.rel_k_proj(rel_emb_table).view(4, self.num_heads, self.head_dim).permute(1, 0, 2) # [H, 4, d]
        q_k_rel = torch.bmm(Q, K_rel.transpose(1, 2)) # [H, N, 4]
        scores_rel = torch.gather(q_k_rel, dim=2, index=rel_adj.unsqueeze(0).expand(self.num_heads, -1, -1)) * self.scale

        r_bias_table = self.rel_bias(rel_types) # [4, H]
        r_bias = torch.gather(
            r_bias_table.transpose(0, 1).unsqueeze(1).expand(self.num_heads, N, 4),
            dim=2,
            index=rel_adj.unsqueeze(0).expand(self.num_heads, -1, -1)
        )
        scores = scores + scores_rel + r_bias

        if self.degree_damping > 0.0:
            deg_penalty = self.degree_damping * torch.log1p(degrees.float()).unsqueeze(0).unsqueeze(1)
            scores = scores - deg_penalty

        mask = (adj == 0).unsqueeze(0).expand(self.num_heads, -1, -1)
        scores = scores.masked_fill(mask, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        attn_mean = attn.mean(dim=0)

        # Context aggregation
        ctx_nodes = torch.bmm(attn, V_nodes)
        V_rel = self.rel_v_proj(rel_emb_table).view(4, self.num_heads, self.head_dim).permute(1, 0, 2) # [H, 4, d]
        ctx_rel = torch.zeros(self.num_heads, N, self.head_dim, device=h.device)
        for r in range(4):
            mask_r = (rel_adj == r).float().unsqueeze(0)
            w_r = torch.sum(attn * mask_r, dim=-1, keepdim=True)
            ctx_rel = ctx_rel + w_r * V_rel[:, r:r+1, :]

        agg = (ctx_nodes + ctx_rel).transpose(0, 1).contiguous().view(N, -1)
        h_gat = self.norm_gat(h + self.dropout(self.out_proj(agg)))

        # 4. Bounded Graph Delta
        graph_feat = torch.cat([h_gat, ctx, h_gat * ctx], dim=-1)
        delta_s = self.delta_bound * self.graph_delta_mlp(graph_feat).squeeze(-1)

        # 5. Evidence Calibration Gate
        if self.use_evidence_gate:
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

            meta = torch.cat([cos_fact_ctx, cos_gat_ctx, cos_fact_gat, r_anchor, deg_factor, prob_vis], dim=-1)
            lambda_i = self.evidence_gate(meta).squeeze(-1)
        else:
            lambda_i = torch.ones_like(s_vis)

        final_score = s_vis + 0.4 * lambda_i * delta_s
        return final_score, s_vis, lambda_i, delta_s


# =============================================================================
# 2. LOSS FUNCTIONS
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
# 3. EVALUATION PROTOCOL
# =============================================================================

def evaluate_model_on_test(
    model: nn.Module,
    model_type: str, # "dec_v2", "gatv2", "cross_attention", "siglip2_zero"
    test_clips: Dict[str, List[Dict[str, Any]]],
    test_contrastive: List[Dict[str, Any]],
    subgraphs: Dict[str, Any],
    fact_embeddings: Dict[str, np.ndarray],
    video_frames: Dict[str, np.ndarray],
    video_pooled: Dict[str, np.ndarray],
    facts: Dict[str, Dict[str, Any]],
    device: torch.device
) -> Dict[str, Any]:
    if model is not None:
        model.eval()

    test_ndcg5, test_p5, test_r5, test_aucs = [], [], [], []
    seen_ndcg, unseen_ndcg = [], []
    hist_ndcg = []
    clip_scores = {} # cid -> {fid: score}

    with torch.no_grad():
        for c_id, recs in test_clips.items():
            l_id = recs[0].get("landmark_id", "unknown")
            g = subgraphs.get(l_id)
            if g is None:
                continue

            cand_fids = [r["fact_id"] for r in recs]
            gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")

            v_f = video_frames.get(c_id)
            v_p = video_pooled.get(c_id)

            fid2idx = g["fid2idx"]
            sub_fids = g["fact_ids"]

            if model_type == "siglip2_zero":
                if v_p is None:
                    continue
                v_p_norm = v_p / np.linalg.norm(v_p)
                f_scores = {}
                for fid in cand_fids:
                    f_emb = fact_embeddings.get(fid)
                    if f_emb is not None:
                        f_scores[fid] = float(np.dot(v_p_norm, f_emb))
                    else:
                        f_scores[fid] = -1.0

            elif model_type == "cross_attention":
                if v_f is None:
                    continue
                v_f_t = torch.from_numpy(v_f).float().to(device)
                cand_h = [fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in cand_fids]
                cand_h_t = torch.from_numpy(np.array(cand_h, dtype=np.float32)).to(device)
                logits, _ = model(v_f_t, cand_h_t)
                scores_np = logits.cpu().numpy()
                f_scores = {fid: float(scores_np[i]) for i, fid in enumerate(cand_fids)}

            elif model_type == "gatv2":
                if v_p is None:
                    continue
                h_list = [fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
                h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
                adj = g["adj"].to(device)
                v_p_t = torch.from_numpy(v_p).float().to(device)
                logits = model(h_tensor, adj, v_p_t)
                scores_np = logits.cpu().numpy()
                f_scores = {fid: float(scores_np[fid2idx[fid]]) for fid in cand_fids if fid in fid2idx}

            elif model_type == "dec_v2":
                if v_f is None:
                    continue
                h_list = [fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
                h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
                adj = g["adj"].to(device)
                rel_adj = g["rel_adj"].to(device)
                degrees = g["degrees"].to(device)
                v_f_t = torch.from_numpy(v_f).float().to(device)
                logits, _, _, _ = model(h_tensor, adj, rel_adj, degrees, v_f_t)
                scores_np = logits.cpu().numpy()
                f_scores = {fid: float(scores_np[fid2idx[fid]]) for fid in cand_fids if fid in fid2idx}

            clip_scores[c_id] = f_scores

            # Ranking metrics
            ranked = sorted(cand_fids, key=lambda fid: f_scores.get(fid, -999.0), reverse=True)
            m = compute_ranking_metrics(ranked, gold_fids, cand_fids, top_k_list=[5])
            ndcg_val = m.get("nDCG@5", 0.0)

            test_ndcg5.append(ndcg_val)
            test_p5.append(m.get("P@5", 0.0))
            test_r5.append(m.get("R@5", 0.0))

            if l_id in SEEN_LANDMARKS:
                seen_ndcg.append(ndcg_val)
            else:
                unseen_ndcg.append(ndcg_val)

            auc = compute_hard_negative_auc(f_scores, gold_fids, hard_neg_fids)
            if auc is not None:
                test_aucs.append(auc)

            # Historical facts ranking
            cand_hist = [f for f in cand_fids if facts.get(f, {}).get("predicate") in HISTORICAL_PREDICATES]
            gold_hist = [f for f in gold_fids if facts.get(f, {}).get("predicate") in HISTORICAL_PREDICATES]
            if cand_hist and gold_hist:
                rank_hist = sorted(cand_hist, key=lambda f: f_scores.get(f, -999.0), reverse=True)
                m_h = compute_ranking_metrics(rank_hist, set(gold_hist), cand_hist, top_k_list=[5])
                hist_ndcg.append(m_h.get("nDCG@5", 0.0))

    # Contrastive Challenge Accuracy
    cont_correct = 0
    cont_total = 0
    for p in test_contrastive:
        c_a = p["clip_A"]["clip_id"]
        c_b = p["clip_B"]["clip_id"]
        fid = p["fact_id"]
        if c_a in clip_scores and c_b in clip_scores:
            s_a = clip_scores[c_a].get(fid)
            s_b = clip_scores[c_b].get(fid)
            if s_a is not None and s_b is not None:
                cont_total += 1
                if s_a > s_b:
                    cont_correct += 1

    cont_acc = (cont_correct / cont_total * 100.0) if cont_total > 0 else 50.0

    return {
        "ndcg5": float(np.mean(test_ndcg5)),
        "p5": float(np.mean(test_p5)),
        "r5": float(np.mean(test_r5)),
        "auc": float(np.mean(test_aucs)) if test_aucs else 0.5,
        "seen_ndcg5": float(np.mean(seen_ndcg)),
        "unseen_ndcg5": float(np.mean(unseen_ndcg)),
        "hist_ndcg5": float(np.mean(hist_ndcg)) if hist_ndcg else 0.0,
        "contrastive_acc": float(cont_acc),
        "per_clip_ndcg5": test_ndcg5,
    }


# =============================================================================
# 4. TRAINING HARNESS
# =============================================================================

def train_cross_attention(
    train_clips: Dict[str, List[Dict[str, Any]]],
    val_clips: Dict[str, List[Dict[str, Any]]],
    subgraphs: Dict[str, Any],
    fact_embeddings: Dict[str, np.ndarray],
    video_frames: Dict[str, np.ndarray],
    device: torch.device,
    epochs: int = 15,
) -> nn.Module:
    model = CrossAttentionRankerModel(d_model=768, num_heads=4, hidden_dim=256, dropout=0.1).to(device)
    ckpt_path = CHECKPOINT_DIR / "best_cross_attention_horn.pt"
    if ckpt_path.exists():
        LOGGER.info(f"Loading existing Cross-Attention checkpoint from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        return model

    LOGGER.info("=== Training Cross-Attention Ranker ===")
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_score = -1.0
    best_state = None
    train_cids = list(train_clips.keys())
    patience = 4
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(train_cids)

        for c_id in train_cids:
            recs = train_clips[c_id]
            v_f = video_frames.get(c_id)
            if v_f is None:
                continue

            cand_fids = [r["fact_id"] for r in recs]
            pos_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
            if not pos_fids:
                continue

            v_f_t = torch.from_numpy(v_f).float().to(device)
            cand_h = [fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in cand_fids]
            cand_h_t = torch.from_numpy(np.array(cand_h, dtype=np.float32)).to(device)

            optimizer.zero_grad()
            logits, _ = model(v_f_t, cand_h_t)

            fid2idx = {fid: i for i, fid in enumerate(cand_fids)}
            pos_idx = [fid2idx[fid] for fid in pos_fids]
            hard_neg_idx = [fid2idx[fid] for fid in hard_neg_fids if fid in fid2idx]
            cand_idx = list(range(len(cand_fids)))
            neg_idx = [i for i in cand_idx if i not in pos_idx]

            loss_listmle = listmle_topk_loss(logits, pos_idx, cand_idx, top_k=5, temp=0.5)

            loss_hn = torch.tensor(0.0, device=device)
            if hard_neg_idx and pos_idx:
                p_s = logits[pos_idx].unsqueeze(1)
                hn_s = logits[hard_neg_idx].unsqueeze(0)
                loss_hn = torch.relu(0.8 - (p_s - hn_s)).mean()

            y_true = [1.0] * len(pos_idx) + [0.0] * len(neg_idx)
            preds = torch.cat([logits[pos_idx], logits[neg_idx]])
            targets = torch.tensor(y_true, dtype=torch.float32, device=device)
            bce_loss = F.binary_cross_entropy_with_logits(preds, targets, pos_weight=torch.tensor([3.0], device=device))

            total_loss = 0.5 * loss_listmle + 0.5 * bce_loss + 0.6 * loss_hn
            total_loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_ndcg = []
        with torch.no_grad():
            for c_id, recs in val_clips.items():
                v_f = video_frames.get(c_id)
                if v_f is None:
                    continue
                cand_fids = [r["fact_id"] for r in recs]
                gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
                v_f_t = torch.from_numpy(v_f).float().to(device)
                cand_h = [fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in cand_fids]
                cand_h_t = torch.from_numpy(np.array(cand_h, dtype=np.float32)).to(device)
                logits, _ = model(v_f_t, cand_h_t)
                scores = logits.cpu().numpy()
                ranked = sorted(cand_fids, key=lambda fid: scores[cand_fids.index(fid)], reverse=True)
                m = compute_ranking_metrics(ranked, gold_fids, cand_fids, top_k_list=[5])
                val_ndcg.append(m.get("nDCG@5", 0.0))

        mean_ndcg = float(np.mean(val_ndcg)) if val_ndcg else 0.0
        LOGGER.info(f"[Cross-Attention] Epoch {epoch:02d} | Val nDCG@5: {mean_ndcg:.4f}")
        if mean_ndcg > best_score:
            best_score = mean_ndcg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                LOGGER.info(f"[{variant_name}] Early stopping triggered at epoch {epoch:02d} (patience={patience})")
                break

    ckpt_path = CHECKPOINT_DIR / "best_cross_attention_horn.pt"
    torch.save(best_state, ckpt_path)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def train_gatv2(
    train_clips: Dict[str, List[Dict[str, Any]]],
    val_clips: Dict[str, List[Dict[str, Any]]],
    subgraphs: Dict[str, Any],
    fact_embeddings: Dict[str, np.ndarray],
    video_pooled: Dict[str, np.ndarray],
    device: torch.device,
    epochs: int = 15,
) -> nn.Module:
    model = GATv2RankerModel(d_model=768, num_heads=4, hidden_dim=256, dropout=0.1).to(device)
    ckpt_path = CHECKPOINT_DIR / "best_gatv2_horn.pt"
    if ckpt_path.exists():
        LOGGER.info(f"Loading existing GATv2 checkpoint from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        return model

    LOGGER.info("=== Training GATv2 Ranker ===")
    optimizer = AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)

    best_score = -1.0
    best_state = None
    train_cids = list(train_clips.keys())
    patience = 4
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(train_cids)

        for c_id in train_cids:
            recs = train_clips[c_id]
            l_id = recs[0].get("landmark_id", "unknown")
            g = subgraphs.get(l_id)
            v_p = video_pooled.get(c_id)
            if g is None or v_p is None:
                continue

            cand_fids = [r["fact_id"] for r in recs]
            pos_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
            if not pos_fids:
                continue

            sub_fids = g["fact_ids"]
            fid2idx = g["fid2idx"]
            h_list = [fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
            h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
            adj = g["adj"].to(device)
            v_p_t = torch.from_numpy(v_p).float().to(device)

            optimizer.zero_grad()
            logits = model(h_tensor, adj, v_p_t)

            pos_idx = [fid2idx[fid] for fid in pos_fids if fid in fid2idx]
            cand_idx = [fid2idx[fid] for fid in cand_fids if fid in fid2idx]
            hard_neg_idx = [fid2idx[fid] for fid in hard_neg_fids if fid in fid2idx]
            neg_idx = [i for i in cand_idx if i not in pos_idx]

            loss_listmle = listmle_topk_loss(logits, pos_idx, cand_idx, top_k=5, temp=0.5)

            loss_hn = torch.tensor(0.0, device=device)
            if hard_neg_idx and pos_idx:
                p_s = logits[pos_idx].unsqueeze(1)
                hn_s = logits[hard_neg_idx].unsqueeze(0)
                loss_hn = torch.relu(0.8 - (p_s - hn_s)).mean()

            y_true = [1.0] * len(pos_idx) + [0.0] * len(neg_idx)
            preds = torch.cat([logits[pos_idx], logits[neg_idx]])
            targets = torch.tensor(y_true, dtype=torch.float32, device=device)
            bce_loss = F.binary_cross_entropy_with_logits(preds, targets, pos_weight=torch.tensor([3.0], device=device))

            total_loss = 0.5 * loss_listmle + 0.5 * bce_loss + 0.6 * loss_hn
            total_loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_ndcg = []
        with torch.no_grad():
            for c_id, recs in val_clips.items():
                l_id = recs[0].get("landmark_id", "unknown")
                g = subgraphs.get(l_id)
                v_p = video_pooled.get(c_id)
                if g is None or v_p is None:
                    continue
                cand_fids = [r["fact_id"] for r in recs]
                gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
                sub_fids = g["fact_ids"]
                fid2idx = g["fid2idx"]
                h_list = [fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
                h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
                adj = g["adj"].to(device)
                v_p_t = torch.from_numpy(v_p).float().to(device)
                logits = model(h_tensor, adj, v_p_t)
                scores = logits.cpu().numpy()
                ranked = sorted(cand_fids, key=lambda fid: scores[fid2idx[fid]], reverse=True)
                m = compute_ranking_metrics(ranked, gold_fids, cand_fids, top_k_list=[5])
                val_ndcg.append(m.get("nDCG@5", 0.0))

        mean_ndcg = float(np.mean(val_ndcg)) if val_ndcg else 0.0
        LOGGER.info(f"[GATv2] Epoch {epoch:02d} | Val nDCG@5: {mean_ndcg:.4f}")
        if mean_ndcg > best_score:
            best_score = mean_ndcg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                LOGGER.info(f"[{variant_name}] Early stopping triggered at epoch {epoch:02d} (patience={patience})")
                break

    ckpt_path = CHECKPOINT_DIR / "best_gatv2_horn.pt"
    torch.save(best_state, ckpt_path)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def train_modular_dec_v2(
    variant_name: str,
    kwargs: Dict[str, Any],
    train_clips: Dict[str, List[Dict[str, Any]]],
    val_clips: Dict[str, List[Dict[str, Any]]],
    subgraphs: Dict[str, Any],
    fact_embeddings: Dict[str, np.ndarray],
    video_frames: Dict[str, np.ndarray],
    device: torch.device,
    cross_attn_state: Optional[Dict[str, Any]] = None,
    epochs: int = 15,
) -> nn.Module:
    LOGGER.info(f"=== Training {variant_name} ===")
    model = ModularDECV2Model(**kwargs).to(device)

    # Initialize visual backbone from cross attention
    if cross_attn_state is not None:
        model_dict = model.state_dict()
        transferred = {k: v for k, v in cross_attn_state.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(transferred)
        model.load_state_dict(model_dict)
        LOGGER.info(f"Transferred {len(transferred)} weights from Cross-Attention backbone.")

    vis_params = list(model.cross_attn.parameters()) + list(model.norm1.parameters()) + list(model.vis_scorer.parameters())
    graph_params = [p for p in model.parameters() if not any(p is vp for vp in vis_params)]

    optimizer = AdamW([
        {"params": vis_params, "lr": 1e-5, "weight_decay": 1e-4},
        {"params": graph_params, "lr": 1.5e-4, "weight_decay": 1e-4}
    ])

    best_score = -1.0
    best_state = None
    train_cids = list(train_clips.keys())
    patience = 4
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(train_cids)

        for c_id in train_cids:
            recs = train_clips[c_id]
            l_id = recs[0].get("landmark_id", "unknown")
            g = subgraphs.get(l_id)
            v_f = video_frames.get(c_id)
            if g is None or v_f is None:
                continue

            cand_fids = [r["fact_id"] for r in recs]
            pos_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
            if not pos_fids:
                continue

            sub_fids = g["fact_ids"]
            fid2idx = g["fid2idx"]
            h_list = [fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
            h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
            adj = g["adj"].to(device)
            rel_adj = g["rel_adj"].to(device)
            degrees = g["degrees"].to(device)
            v_f_t = torch.from_numpy(v_f).float().to(device)

            optimizer.zero_grad()
            logits, s_vis, lambda_i, delta_s = model(h_tensor, adj, rel_adj, degrees, v_f_t)

            pos_idx = [fid2idx[fid] for fid in pos_fids if fid in fid2idx]
            cand_idx = [fid2idx[fid] for fid in cand_fids if fid in fid2idx]
            hard_neg_idx = [fid2idx[fid] for fid in hard_neg_fids if fid in fid2idx]
            neg_idx = [i for i in cand_idx if i not in pos_idx]

            loss_listmle = listmle_topk_loss(logits, pos_idx, cand_idx, top_k=5, temp=0.5)

            loss_hn = torch.tensor(0.0, device=device)
            if hard_neg_idx and pos_idx:
                p_s = logits[pos_idx].unsqueeze(1)
                hn_s = logits[hard_neg_idx].unsqueeze(0)
                loss_hn = torch.relu(0.8 - (p_s - hn_s)).mean()

            y_true = [1.0] * len(pos_idx) + [0.0] * len(neg_idx)
            preds = torch.cat([logits[pos_idx], logits[neg_idx]])
            targets = torch.tensor(y_true, dtype=torch.float32, device=device)
            bce_loss = F.binary_cross_entropy_with_logits(preds, targets, pos_weight=torch.tensor([3.0], device=device))

            total_loss = 0.5 * loss_listmle + 0.5 * bce_loss + 0.6 * loss_hn
            total_loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_ndcg = []
        with torch.no_grad():
            for c_id, recs in val_clips.items():
                l_id = recs[0].get("landmark_id", "unknown")
                g = subgraphs.get(l_id)
                v_f = video_frames.get(c_id)
                if g is None or v_f is None:
                    continue
                cand_fids = [r["fact_id"] for r in recs]
                gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
                sub_fids = g["fact_ids"]
                fid2idx = g["fid2idx"]
                h_list = [fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
                h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
                adj = g["adj"].to(device)
                rel_adj = g["rel_adj"].to(device)
                degrees = g["degrees"].to(device)
                v_f_t = torch.from_numpy(v_f).float().to(device)
                logits, _, _, _ = model(h_tensor, adj, rel_adj, degrees, v_f_t)
                scores = logits.cpu().numpy()
                ranked = sorted(cand_fids, key=lambda fid: scores[fid2idx[fid]], reverse=True)
                m = compute_ranking_metrics(ranked, gold_fids, cand_fids, top_k_list=[5])
                val_ndcg.append(m.get("nDCG@5", 0.0))

        mean_ndcg = float(np.mean(val_ndcg)) if val_ndcg else 0.0
        LOGGER.info(f"[{variant_name}] Epoch {epoch:02d} | Val nDCG@5: {mean_ndcg:.4f}")
        if mean_ndcg > best_score:
            best_score = mean_ndcg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                LOGGER.info(f"[{variant_name}] Early stopping triggered at epoch {epoch:02d} (patience={patience})")
                break

    ckpt_path = CHECKPOINT_DIR / f"best_{variant_name.lower()}_horn.pt"
    torch.save(best_state, ckpt_path)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


# =============================================================================
# 5. STATISTICAL SIGNIFICANCE (PAIRED BOOTSTRAP)
# =============================================================================

def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    n_resamples: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """Tests if scores_a is significantly greater than scores_b."""
    rng = np.random.default_rng(seed)
    diffs = np.array(scores_a) - np.array(scores_b)
    obs_diff = float(np.mean(diffs))
    N = len(diffs)

    boot_diffs = []
    for _ in range(n_resamples):
        indices = rng.integers(0, N, size=N)
        boot_diffs.append(float(np.mean(diffs[indices])))

    boot_diffs = np.array(boot_diffs)
    p_value = float(np.mean(boot_diffs <= 0.0))
    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))

    return {
        "observed_diff": obs_diff,
        "ci_95": [ci_low, ci_high],
        "p_value": p_value,
        "significant_p05": bool(p_value < 0.05),
        "significant_p01": bool(p_value < 0.01),
    }


# =============================================================================
# 6. MAIN ORCHESTRATOR
# =============================================================================

def main():
    device = torch.device("cpu")
    torch.set_num_threads(24)
    LOGGER.info(f"Initializing Horn Complete Benchmark Suite on Device: {device} (24 CPU threads)")

    # 1. Load Data
    LOGGER.info("Loading Horn normalized facts...")
    facts = {}
    with open(FACTS_HORN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                facts[r["fact_id"]] = r
    LOGGER.info(f"Loaded {len(facts)} Horn facts.")

    LOGGER.info("Loading precomputed feature embeddings...")
    arr_t = np.load(CACHE_DIR / "fact_embeddings_siglip2_horn.npy")
    with open(CACHE_DIR / "fact_embeddings_manifest_horn.json", "r", encoding="utf-8") as f:
        fids_t = json.load(f)
    fact_embeddings = {fid: arr_t[i] for i, fid in enumerate(fids_t)}

    arr_vf = np.load(CACHE_DIR / "video_frames_siglip2.npy")
    arr_vp = np.load(CACHE_DIR / "video_pooled_siglip2.npy")
    with open(CACHE_DIR / "video_manifest.json", "r", encoding="utf-8") as f:
        cids_v = json.load(f)
    video_frames = {cid: arr_vf[i] for i, cid in enumerate(cids_v)}
    video_pooled = {cid: arr_vp[i] for i, cid in enumerate(cids_v)}

    LOGGER.info("Loading Horn complete subgraphs...")
    subgraphs = torch.load(CACHE_DIR / "landmark_subgraphs_horn_complete.pt", weights_only=False)
    LOGGER.info(f"Loaded {len(subgraphs)} subgraphs.")

    LOGGER.info("Loading train, val, and test splits...")
    train_clips = defaultdict(list)
    val_clips = defaultdict(list)
    with open(TRAIN_VAL_GOLD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("split") == "train":
                    train_clips[rec["clip_id"]].append(rec)
                elif rec.get("split") == "val":
                    val_clips[rec["clip_id"]].append(rec)

    test_clips = defaultdict(list)
    with open(TEST_GOLD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                test_clips[rec["clip_id"]].append(rec)

    LOGGER.info(f"Loaded Splits: Train {len(train_clips)} clips, Val {len(val_clips)} clips, Test {len(test_clips)} clips.")

    test_contrastive = []
    if TEST_CONTRASTIVE_PATH.exists():
        with open(TEST_CONTRASTIVE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    test_contrastive.append(json.loads(line))
    LOGGER.info(f"Loaded {len(test_contrastive)} test contrastive pairs.")

    results = {}

    # ---------------------------------------------------------
    # Baseline 1: SigLIP2 Zero-Shot
    # ---------------------------------------------------------
    LOGGER.info("\n--- Evaluating Baseline: SigLIP2 Zero-Shot ---")
    res_siglip2 = evaluate_model_on_test(
        model=None,
        model_type="siglip2_zero",
        test_clips=test_clips,
        test_contrastive=test_contrastive,
        subgraphs=subgraphs,
        fact_embeddings=fact_embeddings,
        video_frames=video_frames,
        video_pooled=video_pooled,
        facts=facts,
        device=device
    )
    results["SigLIP2_ZeroShot"] = res_siglip2
    LOGGER.info(f"SigLIP2 Zero-Shot | nDCG@5: {res_siglip2['ndcg5']:.4f} | P@5: {res_siglip2['p5']:.4f} | R@5: {res_siglip2['r5']:.4f} | AUC: {res_siglip2['auc']:.4f} | Cont: {res_siglip2['contrastive_acc']:.1f}%")

    # ---------------------------------------------------------
    # Baseline 2: Cross-Attention Ranker
    # ---------------------------------------------------------
    model_cross = train_cross_attention(
        train_clips=train_clips,
        val_clips=val_clips,
        subgraphs=subgraphs,
        fact_embeddings=fact_embeddings,
        video_frames=video_frames,
        device=device,
        epochs=15
    )
    res_cross = evaluate_model_on_test(
        model=model_cross,
        model_type="cross_attention",
        test_clips=test_clips,
        test_contrastive=test_contrastive,
        subgraphs=subgraphs,
        fact_embeddings=fact_embeddings,
        video_frames=video_frames,
        video_pooled=video_pooled,
        facts=facts,
        device=device
    )
    results["CrossAttention_Ranker"] = res_cross
    LOGGER.info(f"Cross-Attention | nDCG@5: {res_cross['ndcg5']:.4f} | P@5: {res_cross['p5']:.4f} | R@5: {res_cross['r5']:.4f} | AUC: {res_cross['auc']:.4f} | Cont: {res_cross['contrastive_acc']:.1f}%")

    cross_state = {k: v.cpu().clone() for k, v in model_cross.state_dict().items()}

    # ---------------------------------------------------------
    # Baseline 3: GATv2 Ranker
    # ---------------------------------------------------------
    model_gatv2 = train_gatv2(
        train_clips=train_clips,
        val_clips=val_clips,
        subgraphs=subgraphs,
        fact_embeddings=fact_embeddings,
        video_pooled=video_pooled,
        device=device,
        epochs=15
    )
    res_gatv2 = evaluate_model_on_test(
        model=model_gatv2,
        model_type="gatv2",
        test_clips=test_clips,
        test_contrastive=test_contrastive,
        subgraphs=subgraphs,
        fact_embeddings=fact_embeddings,
        video_frames=video_frames,
        video_pooled=video_pooled,
        facts=facts,
        device=device
    )
    results["GATv2"] = res_gatv2
    LOGGER.info(f"GATv2 | nDCG@5: {res_gatv2['ndcg5']:.4f} | P@5: {res_gatv2['p5']:.4f} | R@5: {res_gatv2['r5']:.4f} | AUC: {res_gatv2['auc']:.4f} | Cont: {res_gatv2['contrastive_acc']:.1f}%")

    # ---------------------------------------------------------
    # Proposed Full: DEC-VCGAT v2 Full
    # ---------------------------------------------------------
    model_dec_full = train_modular_dec_v2(
        variant_name="DEC_VCGAT_v2_Full",
        kwargs={
            "d_model": 768, "num_heads": 4, "hidden_dim": 256,
            "delta_bound": 0.8, "degree_damping": 0.4,
            "use_strict_saliency": True, "use_evidence_gate": True,
            "dropout": 0.1
        },
        train_clips=train_clips,
        val_clips=val_clips,
        subgraphs=subgraphs,
        fact_embeddings=fact_embeddings,
        video_frames=video_frames,
        device=device,
        cross_attn_state=cross_state,
        epochs=15
    )
    res_dec_full = evaluate_model_on_test(
        model=model_dec_full,
        model_type="dec_v2",
        test_clips=test_clips,
        test_contrastive=test_contrastive,
        subgraphs=subgraphs,
        fact_embeddings=fact_embeddings,
        video_frames=video_frames,
        video_pooled=video_pooled,
        facts=facts,
        device=device
    )
    results["DEC_VCGAT_v2_Full"] = res_dec_full
    LOGGER.info(f"DEC-VCGAT v2 Full | nDCG@5: {res_dec_full['ndcg5']:.4f} | P@5: {res_dec_full['p5']:.4f} | R@5: {res_dec_full['r5']:.4f} | AUC: {res_dec_full['auc']:.4f} | Cont: {res_dec_full['contrastive_acc']:.1f}%")

    # ---------------------------------------------------------
    # Ablation 1: w/o Evidence Calibration Gate (lambda = 1.0)
    # ---------------------------------------------------------
    model_ablation_gate = train_modular_dec_v2(
        variant_name="DEC_v2_wo_Gate",
        kwargs={
            "d_model": 768, "num_heads": 4, "hidden_dim": 256,
            "delta_bound": 0.8, "degree_damping": 0.4,
            "use_strict_saliency": True, "use_evidence_gate": False,
            "dropout": 0.1
        },
        train_clips=train_clips,
        val_clips=val_clips,
        subgraphs=subgraphs,
        fact_embeddings=fact_embeddings,
        video_frames=video_frames,
        device=device,
        cross_attn_state=cross_state,
        epochs=15
    )
    res_ablation_gate = evaluate_model_on_test(
        model=model_ablation_gate,
        model_type="dec_v2",
        test_clips=test_clips,
        test_contrastive=test_contrastive,
        subgraphs=subgraphs,
        fact_embeddings=fact_embeddings,
        video_frames=video_frames,
        video_pooled=video_pooled,
        facts=facts,
        device=device
    )
    results["DEC_v2_wo_Gate"] = res_ablation_gate

    # ---------------------------------------------------------
    # Ablation 2: w/o Degree Debiasing (gamma = 0.0)
    # ---------------------------------------------------------
    model_ablation_debias = train_modular_dec_v2(
        variant_name="DEC_v2_wo_Debias",
        kwargs={
            "d_model": 768, "num_heads": 4, "hidden_dim": 256,
            "delta_bound": 0.8, "degree_damping": 0.0,
            "use_strict_saliency": True, "use_evidence_gate": True,
            "dropout": 0.1
        },
        train_clips=train_clips,
        val_clips=val_clips,
        subgraphs=subgraphs,
        fact_embeddings=fact_embeddings,
        video_frames=video_frames,
        device=device,
        cross_attn_state=cross_state,
        epochs=15
    )
    res_ablation_debias = evaluate_model_on_test(
        model=model_ablation_debias,
        model_type="dec_v2",
        test_clips=test_clips,
        test_contrastive=test_contrastive,
        subgraphs=subgraphs,
        fact_embeddings=fact_embeddings,
        video_frames=video_frames,
        video_pooled=video_pooled,
        facts=facts,
        device=device
    )
    results["DEC_v2_wo_Debias"] = res_ablation_debias

    # ---------------------------------------------------------
    # Ablation 3: w/o Strict Visual Saliency Transmission
    # ---------------------------------------------------------
    model_ablation_saliency = train_modular_dec_v2(
        variant_name="DEC_v2_wo_Saliency",
        kwargs={
            "d_model": 768, "num_heads": 4, "hidden_dim": 256,
            "delta_bound": 0.8, "degree_damping": 0.4,
            "use_strict_saliency": False, "use_evidence_gate": True,
            "dropout": 0.1
        },
        train_clips=train_clips,
        val_clips=val_clips,
        subgraphs=subgraphs,
        fact_embeddings=fact_embeddings,
        video_frames=video_frames,
        device=device,
        cross_attn_state=cross_state,
        epochs=15
    )
    res_ablation_saliency = evaluate_model_on_test(
        model=model_ablation_saliency,
        model_type="dec_v2",
        test_clips=test_clips,
        test_contrastive=test_contrastive,
        subgraphs=subgraphs,
        fact_embeddings=fact_embeddings,
        video_frames=video_frames,
        video_pooled=video_pooled,
        facts=facts,
        device=device
    )
    results["DEC_v2_wo_Saliency"] = res_ablation_saliency

    # ---------------------------------------------------------
    # Statistical Significance Testing (Paired Bootstrap)
    # ---------------------------------------------------------
    LOGGER.info("\n--- Computing Paired Bootstrap Statistical Significance (1,000 resamples) ---")
    sig_vs_cross = paired_bootstrap_test(
        res_dec_full["per_clip_ndcg5"],
        res_cross["per_clip_ndcg5"],
        n_resamples=1000
    )
    sig_vs_gatv2 = paired_bootstrap_test(
        res_dec_full["per_clip_ndcg5"],
        res_gatv2["per_clip_ndcg5"],
        n_resamples=1000
    )

    significance_results = {
        "DEC_v2_vs_CrossAttention": sig_vs_cross,
        "DEC_v2_vs_GATv2": sig_vs_gatv2,
    }

    LOGGER.info(f"DEC-v2 vs Cross-Attention: Delta={sig_vs_cross['observed_diff']:+.4f}, 95% CI={sig_vs_cross['ci_95']}, p={sig_vs_cross['p_value']:.4f}")
    LOGGER.info(f"DEC-v2 vs GATv2: Delta={sig_vs_gatv2['observed_diff']:+.4f}, 95% CI={sig_vs_gatv2['ci_95']}, p={sig_vs_gatv2['p_value']:.4f}")

    # Strip per-clip arrays for cleaner JSON export
    cleaned_results = {}
    for k, v in results.items():
        v_clean = dict(v)
        v_clean.pop("per_clip_ndcg5", None)
        cleaned_results[k] = v_clean

    output_payload = {
        "benchmark": "Horn_Complete_Fact_Selection_800Test",
        "total_test_clips": len(test_clips),
        "total_train_clips": len(train_clips),
        "total_val_clips": len(val_clips),
        "horn_facts_total": len(facts),
        "subgraphs_count": len(subgraphs),
        "models": cleaned_results,
        "statistical_significance": significance_results,
    }

    report_path = REPORTS_DIR / "horn_complete_benchmark_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    LOGGER.info(f"\nSuccessfully completed Horn Complete Benchmark! Results saved -> {report_path}")


if __name__ == "__main__":
    main()
