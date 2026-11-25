#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/train_dcse_suite_and_ablation.py
=============================================================================
Definitive Controlled Experiment Suite:
  1. GATv2 + DCSE: Standard GATv2 Graph Baseline equipped with DCSE candidate inclusion.
  2. DEC-VCGAT v2 w/o Evidence Gate (lambda = 1.0 fixed)
  3. DEC-VCGAT v2 w/o Degree Debiasing (gamma = 0.0)
  4. DEC-VCGAT v2 w/o Strict Visual Saliency Transmission (h_trans = h unconditioned)
  5. DEC-VCGAT v2 Full (Reference: 0.3543)

All models evaluated on:
  - Full 800 Test Clips
  - Precision@5, Recall@5, Hard Negative AUC
  - Paired Bootstrap Statistical Significance vs Baseline
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
    compute_ranking_metrics,
    compute_hard_negative_auc,
    HISTORICAL_PREDICATES,
)
from experiments.fact_selection_benchmark.feature_cache import BenchmarkFeatureCache
from experiments.fact_selection_benchmark.train_dec_vcgat_v2 import (
    build_clip_graph,
    listmle_topk_loss,
    SEEN_LANDMARKS,
)
from experiments.fact_selection_benchmark.models.gatv2_ranker import GATv2VideoFusionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("DCSE-Ablation-Suite")

CHECKPOINT_DIR = REPO_ROOT / "experiments" / "checkpoints_fact_selection"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = REPO_ROOT / "experiments" / "fact_selection_benchmark" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. MODULAR DEC-VCGAT v2 FOR ABLATION
# =============================================================================
class ModularDECV2Model(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 4,
        hidden_dim: int = 256,
        delta_bound: float = 0.8,
        degree_damping: float = 0.4,       # Ablation: set to 0.0 for w/o debiasing
        use_strict_saliency: bool = True,  # Ablation: set to False for w/o saliency
        use_evidence_gate: bool = True,    # Ablation: set to False for w/o gate
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

        # 2. Saliency Gate
        self.saliency_gate = nn.Sequential(
            nn.Linear(3 * d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 3. Relational Attention
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

        # 4. Graph Delta MLP
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

        # 5. Evidence Gate
        if self.use_evidence_gate:
            self.evidence_gate = nn.Sequential(
                nn.Linear(6, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        rel_adj: torch.Tensor,
        degrees: torch.Tensor,
        v_frames: torch.Tensor,
        v_pooled: Optional[torch.Tensor] = None
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
        g_nodes = self.saliency_gate(vis_feat)

        # 3. Transmission: Strict vs Unconditioned
        if self.use_strict_saliency:
            h_trans = h * g_nodes
        else:
            h_trans = h

        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.k_proj(h_trans).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V_nodes = self.v_proj(h_trans).view(N, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        rel_emb = self.rel_embed(rel_adj)
        rel_k = self.rel_k_proj(rel_emb).view(N, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        scores_rel = torch.sum(Q.unsqueeze(2) * rel_k, dim=-1) * self.scale
        scores = scores + scores_rel + self.rel_bias(rel_adj).permute(2, 0, 1)

        # 4. Degree Debiasing
        if self.degree_damping > 0.0:
            deg_penalty = self.degree_damping * torch.log1p(degrees.float()).unsqueeze(0).unsqueeze(1)
            scores = scores - deg_penalty

        mask = (adj == 0).unsqueeze(0).expand(self.num_heads, -1, -1)
        scores = scores.masked_fill(mask, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        attn_mean = attn.mean(dim=0)

        ctx_nodes = torch.bmm(attn, V_nodes)
        rel_v = self.rel_v_proj(rel_emb).view(N, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        ctx_rel = torch.sum(attn.unsqueeze(-1) * rel_v, dim=2)
        agg = (ctx_nodes + ctx_rel).transpose(0, 1).contiguous().view(N, -1)
        h_gat = self.norm_gat(h + self.dropout(self.out_proj(agg)))

        # 5. Delta Score
        graph_feat = torch.cat([h_gat, ctx, h_gat * ctx], dim=-1)
        delta_s = self.delta_bound * self.graph_delta_mlp(graph_feat).squeeze(-1)

        # 6. Evidence Gate
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

            evidence_scalars = torch.cat([cos_fact_ctx, cos_gat_ctx, cos_fact_gat, r_anchor, deg_factor, prob_vis], dim=-1)
            lambda_i = self.evidence_gate(evidence_scalars).squeeze(-1)
        else:
            lambda_i = torch.ones(N, device=h.device)

        final_score = s_vis + 0.4 * lambda_i * delta_s
        return final_score, s_vis, lambda_i, delta_s


# =============================================================================
# 2. TRAINING FUNCTION FOR DEC-V2 VARIANTS
# =============================================================================
def train_dec_variant(
    variant_name: str,
    model: nn.Module,
    epochs: int,
    train_clips: Dict[str, Any],
    val_clips: Dict[str, Any],
    test_clips: Dict[str, Any],
    feature_cache: BenchmarkFeatureCache,
    rel_subgraphs: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    LOGGER.info(f"=== Starting Training for {variant_name} ===")

    # Initialize visual backbone from best_cross_attention.pt
    cross_path = CHECKPOINT_DIR / "best_cross_attention.pt"
    if cross_path.exists():
        cross_weights = torch.load(cross_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in cross_weights.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        LOGGER.info(f"Transferred {len(pretrained_dict)} visual weight tensors from Cross-Attention.")

    vis_params = list(model.cross_attn.parameters()) + list(model.norm1.parameters()) + list(model.vis_scorer.parameters())
    graph_params = [p for p in model.parameters() if not any(p is vp for vp in vis_params)]

    optimizer = AdamW([
        {"params": vis_params, "lr": 1e-5, "weight_decay": 1e-4},
        {"params": graph_params, "lr": 1.5e-4, "weight_decay": 1e-4}
    ])

    best_val_score = -1.0
    best_state = None
    train_cids = list(train_clips.keys())

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(train_cids)
        epoch_loss = 0.0
        n_steps = 0

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

            # DCSE Graph Construction
            h_tensor, adj, rel_adj, degrees, all_fids, fid2idx = build_clip_graph(
                g, cand_fids, feature_cache.fact_embeddings, device
            )
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device)

            optimizer.zero_grad()
            logits, _, lambda_i, _ = model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)

            cand_indices = [fid2idx[fid] for fid in cand_fids]
            pos_indices = [fid2idx[fid] for fid in pos_fids]
            hard_neg_indices = [fid2idx[fid] for fid in hard_neg_fids if fid in fid2idx]
            neg_indices = [idx for idx in cand_indices if idx not in pos_indices]

            loss_listmle = listmle_topk_loss(logits, pos_indices, cand_indices, top_k=5, temp=0.5)

            loss_hn = torch.tensor(0.0, device=device)
            if hard_neg_indices and pos_indices:
                p_s = logits[pos_indices].unsqueeze(1)
                hn_s = logits[hard_neg_indices].unsqueeze(0)
                loss_hn = torch.relu(0.8 - (p_s - hn_s)).mean()

            y_true = [1.0] * len(pos_indices) + [0.0] * len(neg_indices)
            preds = torch.cat([logits[pos_indices], logits[neg_indices]])
            targets = torch.tensor(y_true, dtype=torch.float32, device=device)
            bce_loss = F.binary_cross_entropy_with_logits(preds, targets, pos_weight=torch.tensor([3.0], device=device))

            gate_reg = 0.05 * torch.mean(lambda_i ** 2)

            total_loss = 0.5 * loss_listmle + 0.5 * bce_loss + 0.6 * loss_hn + gate_reg
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            n_steps += 1

        # Validation
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

        LOGGER.info(f"[{variant_name}] Epoch {epoch:02d} | Val nDCG@5: {v_ndcg_m:.4f} | Val P@5: {v_p5_m:.4f} | Val AUC: {v_auc_m:.4f} | Comp: {comp_score:.4f}")

        if comp_score > best_val_score:
            best_val_score = comp_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, CHECKPOINT_DIR / f"best_{variant_name.lower()}.pt")

    # Final Test on 800 clips
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    test_ndcg5, test_p5, test_r5, test_aucs = [], [], [], []
    seen_ndcg, unseen_ndcg = [], []
    hist_ndcg = []
    facts = load_normalized_facts()

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
            test_r5.append(m.get("R@5", 0.0))

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
                hist_ndcg.append(m_h.get("nDCG@5", 0.0))

    results = {
        "variant": variant_name,
        "ndcg5": float(np.mean(test_ndcg5)),
        "p5": float(np.mean(test_p5)),
        "r5": float(np.mean(test_r5)),
        "auc": float(np.mean(test_aucs)) if test_aucs else 0.5,
        "seen_ndcg5": float(np.mean(seen_ndcg)),
        "unseen_ndcg5": float(np.mean(unseen_ndcg)),
        "hist_ndcg5": float(np.mean(hist_ndcg)) if hist_ndcg else 0.0,
    }
    LOGGER.info(f"[{variant_name}] TEST: nDCG@5={results['ndcg5']:.4f} | P@5={results['p5']:.4f} | AUC={results['auc']:.4f} | Seen={results['seen_ndcg5']:.4f} | Unseen={results['unseen_ndcg5']:.4f}")
    return results


# =============================================================================
# 3. TRAINING FUNCTION FOR GATv2 + DCSE
# =============================================================================
def train_gatv2_dcse(
    epochs: int,
    train_clips: Dict[str, Any],
    val_clips: Dict[str, Any],
    test_clips: Dict[str, Any],
    feature_cache: BenchmarkFeatureCache,
    rel_subgraphs: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    variant_name = "GATv2_DCSE"
    LOGGER.info(f"=== Starting Training for {variant_name} ===")

    model = GATv2VideoFusionModel(d_model=768, num_heads=4, hidden_dim=256, dropout=0.1).to(device)
    optimizer = AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)

    best_val_score = -1.0
    best_state = None
    train_cids = list(train_clips.keys())

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(train_cids)
        epoch_loss = 0.0
        n_steps = 0

        for c_id in train_cids:
            recs = train_clips[c_id]
            l_id = recs[0].get("landmark_id", "unknown")
            g = rel_subgraphs.get(l_id)
            v_pooled = feature_cache.video_pooled.get(c_id)
            if g is None or v_pooled is None:
                continue

            cand_fids = [r["fact_id"] for r in recs]
            pos_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
            if not pos_fids:
                continue

            # Build DCSE graph
            h_tensor, adj, _, _, _, fid2idx = build_clip_graph(
                g, cand_fids, feature_cache.fact_embeddings, device
            )
            edge_index = (adj > 0).nonzero().t() # [2, E]
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device)

            optimizer.zero_grad()
            logits = model(h_tensor, edge_index, v_p_tensor)

            cand_indices = [fid2idx[fid] for fid in cand_fids]
            pos_indices = [fid2idx[fid] for fid in pos_fids]
            hard_neg_indices = [fid2idx[fid] for fid in hard_neg_fids if fid in fid2idx]
            neg_indices = [idx for idx in cand_indices if idx not in pos_indices]

            loss_listmle = listmle_topk_loss(logits, pos_indices, cand_indices, top_k=5, temp=0.5)

            loss_hn = torch.tensor(0.0, device=device)
            if hard_neg_indices and pos_indices:
                p_s = logits[pos_indices].unsqueeze(1)
                hn_s = logits[hard_neg_indices].unsqueeze(0)
                loss_hn = torch.relu(0.8 - (p_s - hn_s)).mean()

            y_true = [1.0] * len(pos_indices) + [0.0] * len(neg_indices)
            preds = torch.cat([logits[pos_indices], logits[neg_indices]])
            targets = torch.tensor(y_true, dtype=torch.float32, device=device)
            bce_loss = F.binary_cross_entropy_with_logits(preds, targets, pos_weight=torch.tensor([3.0], device=device))

            total_loss = 0.5 * loss_listmle + 0.5 * bce_loss + 0.6 * loss_hn
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            n_steps += 1

        # Validation
        model.eval()
        val_ndcg, val_p5, val_auc = [], [], []
        with torch.no_grad():
            for c_id, recs in val_clips.items():
                l_id = recs[0].get("landmark_id", "unknown")
                g = rel_subgraphs.get(l_id)
                v_pooled = feature_cache.video_pooled.get(c_id)
                if g is None or v_pooled is None:
                    continue

                cand_fids = [r["fact_id"] for r in recs]
                gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
                hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")

                h_tensor, adj, _, _, _, fid2idx = build_clip_graph(
                    g, cand_fids, feature_cache.fact_embeddings, device
                )
                edge_index = (adj > 0).nonzero().t()
                v_p_tensor = torch.from_numpy(v_pooled).float().to(device)

                logits = model(h_tensor, edge_index, v_p_tensor)
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

        LOGGER.info(f"[{variant_name}] Epoch {epoch:02d} | Val nDCG@5: {v_ndcg_m:.4f} | Val P@5: {v_p5_m:.4f} | Val AUC: {v_auc_m:.4f} | Comp: {comp_score:.4f}")

        if comp_score > best_val_score:
            best_val_score = comp_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, CHECKPOINT_DIR / "best_gatv2_dcse.pt")

    # Evaluate on 800 test clips
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    test_ndcg5, test_p5, test_r5, test_aucs = [], [], [], []
    seen_ndcg, unseen_ndcg = [], []
    hist_ndcg = []
    facts = load_normalized_facts()

    with torch.no_grad():
        for c_id, recs in test_clips.items():
            l_id = recs[0].get("landmark_id", "unknown")
            g = rel_subgraphs.get(l_id)
            v_pooled = feature_cache.video_pooled.get(c_id)
            if g is None or v_pooled is None:
                continue

            cand_fids = [r["fact_id"] for r in recs]
            gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")

            h_tensor, adj, _, _, _, fid2idx = build_clip_graph(
                g, cand_fids, feature_cache.fact_embeddings, device
            )
            edge_index = (adj > 0).nonzero().t()
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device)

            logits = model(h_tensor, edge_index, v_p_tensor)
            scores_np = logits.cpu().numpy()

            f_scores = {fid: float(scores_np[fid2idx[fid]]) for fid in cand_fids}
            ranked = sorted(cand_fids, key=lambda fid: f_scores[fid], reverse=True)
            m = compute_ranking_metrics(ranked, gold_fids, cand_fids, top_k_list=[5])
            val_ndcg_val = m.get("nDCG@5", 0.0)

            test_ndcg5.append(val_ndcg_val)
            test_p5.append(m.get("P@5", 0.0))
            test_r5.append(m.get("R@5", 0.0))

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
                hist_ndcg.append(m_h.get("nDCG@5", 0.0))

    results = {
        "variant": variant_name,
        "ndcg5": float(np.mean(test_ndcg5)),
        "p5": float(np.mean(test_p5)),
        "r5": float(np.mean(test_r5)),
        "auc": float(np.mean(test_aucs)) if test_aucs else 0.5,
        "seen_ndcg5": float(np.mean(seen_ndcg)),
        "unseen_ndcg5": float(np.mean(unseen_ndcg)),
        "hist_ndcg5": float(np.mean(hist_ndcg)) if hist_ndcg else 0.0,
    }
    LOGGER.info(f"[{variant_name}] TEST: nDCG@5={results['ndcg5']:.4f} | P@5={results['p5']:.4f} | AUC={results['auc']:.4f} | Seen={results['seen_ndcg5']:.4f} | Unseen={results['unseen_ndcg5']:.4f}")
    return results


# =============================================================================
# 4. MAIN PIPELINE: TRAIN ALL 4 VARIANTS & COMPOSE TABLE
# =============================================================================
def main():
    device = torch.device("cpu")
    torch.set_num_threads(24) # Utilize 24 of the 32 Ryzen threads
    LOGGER.info(f"Using device: {device} with 24 CPU threads")

    facts = load_normalized_facts()
    train_clips, _ = load_gold_data(split_type="train")
    val_clips, _ = load_gold_data(split_type="val")
    test_clips, _ = load_gold_data(split_type="test")

    feature_cache = BenchmarkFeatureCache(device=device)
    feature_cache.extract_fact_embeddings(facts)
    feature_cache.extract_video_features([])
    rel_subgraphs = torch.load(feature_cache.cache_dir / "landmark_subgraphs_rel.pt", weights_only=False)

    all_suite_results = {}

    # 1. GATv2 + DCSE (Baseline with same candidate inclusion)
    res_gatv2_dcse = train_gatv2_dcse(
        epochs=10,
        train_clips=train_clips,
        val_clips=val_clips,
        test_clips=test_clips,
        feature_cache=feature_cache,
        rel_subgraphs=rel_subgraphs,
        device=device
    )
    all_suite_results["GATv2_DCSE"] = res_gatv2_dcse

    # 2. DEC-VCGAT v2 w/o Evidence Gate (lambda = 1.0 fixed)
    model_no_gate = ModularDECV2Model(
        d_model=768,
        degree_damping=0.4,
        use_strict_saliency=True,
        use_evidence_gate=False,
    ).to(device)
    res_no_gate = train_dec_variant(
        variant_name="Ablation_wo_EvidenceGate",
        model=model_no_gate,
        epochs=10,
        train_clips=train_clips,
        val_clips=val_clips,
        test_clips=test_clips,
        feature_cache=feature_cache,
        rel_subgraphs=rel_subgraphs,
        device=device
    )
    all_suite_results["Ablation_wo_EvidenceGate"] = res_no_gate

    # 3. DEC-VCGAT v2 w/o Degree Debiasing (gamma = 0.0)
    model_no_debias = ModularDECV2Model(
        d_model=768,
        degree_damping=0.0,
        use_strict_saliency=True,
        use_evidence_gate=True,
    ).to(device)
    res_no_debias = train_dec_variant(
        variant_name="Ablation_wo_DegreeDebiasing",
        model=model_no_debias,
        epochs=10,
        train_clips=train_clips,
        val_clips=val_clips,
        test_clips=test_clips,
        feature_cache=feature_cache,
        rel_subgraphs=rel_subgraphs,
        device=device
    )
    all_suite_results["Ablation_wo_DegreeDebiasing"] = res_no_debias

    # 4. DEC-VCGAT v2 w/o Strict Visual Saliency Transmission (unconditioned h_trans = h)
    model_no_saliency = ModularDECV2Model(
        d_model=768,
        degree_damping=0.4,
        use_strict_saliency=False,
        use_evidence_gate=True,
    ).to(device)
    res_no_saliency = train_dec_variant(
        variant_name="Ablation_wo_VisualSaliency",
        model=model_no_saliency,
        epochs=10,
        train_clips=train_clips,
        val_clips=val_clips,
        test_clips=test_clips,
        feature_cache=feature_cache,
        rel_subgraphs=rel_subgraphs,
        device=device
    )
    all_suite_results["Ablation_wo_VisualSaliency"] = res_no_saliency

    # 5. Reference: DEC-VCGAT v2 Full
    all_suite_results["DEC_VCGAT_v2_Full"] = {
        "variant": "DEC_VCGAT_v2_Full",
        "ndcg5": 0.3543,
        "p5": 0.2963,
        "r5": 0.1623,
        "auc": 0.6838,
        "seen_ndcg5": 0.5049,
        "unseen_ndcg5": 0.2957,
        "hist_ndcg5": 0.2425
    }

    # Save to json
    out_path = REPORTS_DIR / "dcse_graph_suite_and_ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_suite_results, f, indent=2)

    LOGGER.info("=" * 80)
    LOGGER.info("COMPLETE DCSE GRAPH SUITE & ABLATION RESULTS (800 TEST CLIPS):")
    LOGGER.info(f"{'Model / Ablation Variant':<40} | {'nDCG@5':<8} | {'P@5':<8} | {'Hard Neg AUC':<12} | {'Seen':<8} | {'Unseen':<8}")
    LOGGER.info("-" * 88)
    for k, v in all_suite_results.items():
        LOGGER.info(f"{v['variant']:<40} | {v['ndcg5']:.4f}   | {v['p5']:.4f}   | {v['auc']:.4f}       | {v['seen_ndcg5']:.4f}   | {v['unseen_ndcg5']:.4f}")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
