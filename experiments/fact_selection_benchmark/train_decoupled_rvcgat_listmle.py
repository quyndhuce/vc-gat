#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/train_decoupled_rvcgat_listmle.py
=============================================================================
Decoupled Visual Grounding + Relational Evidence Branch with ListMLE Top-5 Loss.

Addresses the two fundamental scientific questions for IJAR:
  1. Hypothesis Verification: Does decoupling visual facts (90.4% gold) from
     historical facts (5.7% gold) prevent graph smoothing from hurting visual ranking?
  2. Evidence-Chain Grounding: Does an explicit visual-anchor-to-history propagation
     branch significantly improve Historical nDCG@5 and Unseen nDCG@5?
  3. Listwise Optimization: Does ListMLE Top-5 ranking loss overcome pointwise BCE
     limitations by directly optimizing the Top-5 permutation likelihood?
=============================================================================
"""

import sys
import math
import json
import logging
from pathlib import Path
from collections import defaultdict
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
LOGGER = logging.getLogger("Decoupled-ListMLE")

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
# 1. DECOUPLED ARCHITECTURE: PERCEPTUAL GROUNDING + EVIDENCE-CHAIN HISTORY
# =============================================================================
class DecoupledRVCGATModel(nn.Module):
    """
    Decoupled Two-Branch Architecture:
      - Branch 1 (Perceptual Grounding): Dedicated to Visual Facts (90.4% gold).
        Cross-attention between fact query and 8 temporal video frames.
        Produces s_vis and visual anchor confidence c_v in (0, 1).
      - Branch 2 (Relational Evidence Branch): Dedicated to Historical & Non-Visual Facts.
        Instead of blending node embeddings, historical facts gather evidence exclusively
        from active visual anchors via bridge edges:
        s_hist = MLP_hist([h_hist, sum_anchor(c_v * h_v * W_rel), max_anchor(c_v)])
    """
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # ---------------------------------------------------------------------
        # Branch 1: Visual Perceptual Grounding (Cross-Attention)
        # ---------------------------------------------------------------------
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(d_model)
        
        # Visual Scorer
        self.vis_scorer = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Visual Anchor Confidence Head: c_v in (0, 1)
        self.anchor_confidence = nn.Sequential(
            nn.Linear(3 * d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # ---------------------------------------------------------------------
        # Branch 2: Relational Evidence Branch (Non-Visual / Historical Facts)
        # ---------------------------------------------------------------------
        self.rel_embed = nn.Embedding(4, 64)
        self.rel_proj = nn.Linear(64, d_model)

        # Anchor-to-History Evidence Attention
        self.anchor_proj = nn.Linear(d_model, d_model)
        self.hist_proj = nn.Linear(d_model, d_model)

        # Historical Evidence Scorer
        self.hist_scorer = nn.Sequential(
            nn.Linear(3 * d_model + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Learnable gate for prior vs evidence
        self.hist_prior_weight = nn.Parameter(torch.tensor([0.2]))
        self.hist_evidence_weight = nn.Parameter(torch.tensor([0.8]))

    def forward(
        self,
        h: torch.Tensor,                # [N, D] all candidate fact embeddings in subgraph
        adj_mask: torch.Tensor,         # [N, N] binary adjacency
        rel_adj: torch.Tensor,          # [N, N] relation types (0: self, 1: vis_vis, 2: bridge, 3: sem_sem)
        v_frames: torch.Tensor,         # [8, D] temporal video frames
        is_visual_mask: torch.Tensor    # [N] boolean mask (True if predicate is visual)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes decoupled scores:
          - Visual facts: pure cross-attention perceptual score
          - Historical facts: evidence-chain score grounded on active visual anchors
        Returns: (final_scores [N], s_vis [N], s_hist_evidence [N])
        """
        if v_frames.dim() == 2:
            v_frames = v_frames.unsqueeze(0)  # [1, K, D]
        N = h.size(0)
        q = h.unsqueeze(1)                # [N, 1, D]
        kv = v_frames.expand(N, -1, -1)   # [N, K, D]

        # 1. Branch 1: Perceptual Grounding for all nodes
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        ctx = self.norm_cross(attn_out.squeeze(1) + h)  # [N, D]
        vis_feat = torch.cat([h, ctx, h * ctx], dim=-1)  # [N, 3D]

        s_vis = self.vis_scorer(vis_feat).squeeze(-1)            # [N]
        c_anchor = self.anchor_confidence(vis_feat).squeeze(-1)  # [N] in (0, 1)

        # 2. Branch 2: Relational Evidence Propagation (Visual Anchors -> History)
        # Visual anchor mask: node must be visual AND have high confidence
        is_anchor = is_visual_mask.float() * c_anchor  # [N] in [0, 1]

        # Candidate bridge edges: from visual anchors to non-visual facts
        # rel_adj == 2 is 'vis_sem_bridge', rel_adj == 1 is 'vis_vis', rel_adj == 3 is 'sem_sem'
        bridge_mask = ((rel_adj == 2) | (rel_adj == 1) | (rel_adj == 3)) * adj_mask  # [N, N]

        # Anchor features weighted by anchor confidence
        anchor_feat = self.anchor_proj(h) * is_anchor.unsqueeze(-1)  # [N, D]
        hist_queries = self.hist_proj(h)                             # [N, D]

        # Compute evidence attention scores: Q_hist * K_anchor^T
        scores_ev = torch.matmul(hist_queries, anchor_feat.transpose(0, 1)) / math.sqrt(self.d_model) # [N, N]
        scores_ev = scores_ev.masked_fill(bridge_mask == 0, -1e9)
        attn_ev = F.softmax(scores_ev, dim=-1)  # [N, N]
        attn_ev = torch.nan_to_num(attn_ev, nan=0.0)

        # Aggregate evidence context
        rel_emb = self.rel_embed(rel_adj)  # [N, N, 64]
        rel_vec = self.rel_proj(rel_emb)   # [N, N, D]
        
        # Weighted anchor features + relation vector
        anchor_val = (anchor_feat.unsqueeze(0) + rel_vec)  # [N, N, D]
        hist_ev_context = torch.sum(attn_ev.unsqueeze(-1) * anchor_val, dim=1)  # [N, D]

        # Max anchor support and mean anchor support
        max_anchor_c = torch.max(is_anchor.unsqueeze(0) * bridge_mask.float(), dim=-1)[0].unsqueeze(-1)  # [N, 1]
        mean_anchor_c = (torch.sum(is_anchor.unsqueeze(0) * bridge_mask.float(), dim=-1) / bridge_mask.float().sum(dim=-1).clamp_min(1.0)).unsqueeze(-1)

        # Historical Evidence Score
        hist_input = torch.cat([h, ctx, hist_ev_context, max_anchor_c, mean_anchor_c], dim=-1)
        s_hist_evidence = self.hist_scorer(hist_input).squeeze(-1)  # [N]

        # 3. Decoupled Combination
        # For visual facts: use pure perceptual score (prevent graph blurring)
        # For historical facts: use prior visual score + grounded evidence
        final_scores = torch.where(
            is_visual_mask,
            s_vis,
            self.hist_prior_weight * s_vis + self.hist_evidence_weight * s_hist_evidence
        )

        return final_scores, s_vis, s_hist_evidence


# =============================================================================
# 2. LISTMLE TOP-K RANKING LOSS & HYBRID MULTITASK OBJECTIVE
# =============================================================================
def listmle_topk_loss(
    logits: torch.Tensor,
    gold_indices: List[int],
    cand_indices: List[int],
    top_k: int = 5,
    temp: float = 0.5
) -> torch.Tensor:
    """
    Computes ListMLE Top-K ranking loss (Plackett-Luce likelihood) for ordered ranking.
    Places positive gold facts at top positions 1..min(K, |gold|).
    """
    if not gold_indices:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Permutation: gold facts first, followed by negative facts
    neg_indices = [idx for idx in cand_indices if idx not in gold_indices]
    perm_indices = gold_indices[:top_k] + neg_indices

    perm_logits = logits[perm_indices] / temp
    k_eval = min(len(gold_indices), top_k)

    loss = 0.0
    for i in range(k_eval):
        # Log of denominator: logsumexp from position i to end
        log_denom = torch.logsumexp(perm_logits[i:], dim=0)
        log_num = perm_logits[i]
        loss = loss - (log_num - log_denom)

    return loss / float(k_eval)


# =============================================================================
# 3. EVALUATION FUNCTION (BREAKDOWN BY METRIC AND PREDICATE)
# =============================================================================
def evaluate_decoupled_model(
    model: nn.Module,
    clip_records: Dict[str, List[Dict[str, Any]]],
    contrastive_pairs: List[Dict[str, Any]],
    rel_subgraphs: Dict[str, Dict[str, Any]],
    feature_cache: BenchmarkFeatureCache,
    facts_dict: Dict[str, Dict[str, Any]],
    device: torch.device
) -> Dict[str, float]:
    """Evaluates the decoupled model over test clips with full historical breakdown."""
    model.eval()
    flat_scores = {}
    ndcg5, p5, aucs = [], [], []
    seen_ndcg, unseen_ndcg = [], []
    vis_ndcg5, hist_ndcg5 = [], []

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

            # Build visual mask
            is_vis = [facts_dict.get(fid, {}).get("predicate", "").lower() in VISUAL_PREDICATES for fid in sub_fids]
            is_vis_tensor = torch.tensor(is_vis, dtype=torch.bool, device=device)

            logits, _, _ = model(h_tensor, adj, rel_adj, v_f_tensor, is_vis_tensor)
            scores_np = logits.cpu().numpy()

            idx_map = g["fid2idx"]
            f_scores = {}
            gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
            cand_fids = [r["fact_id"] for r in recs]

            for fid in cand_fids:
                s_val = float(scores_np[idx_map[fid]]) if fid in idx_map else 0.0
                f_scores[fid] = s_val
                flat_scores[(c_id, fid)] = s_val

            ranked_fids = sorted(cand_fids, key=lambda fid: f_scores[fid], reverse=True)
            m = compute_ranking_metrics(ranked_fids, gold_fids, cand_fids, top_k_list=[5])
            c_ndcg = m.get("nDCG@5", 0.0)
            ndcg5.append(c_ndcg)
            p5.append(m.get("P@5", 0.0))

            auc = compute_hard_negative_auc(f_scores, gold_fids, hard_neg_fids)
            if auc is not None:
                aucs.append(auc)

            if l_id in SEEN_LANDMARKS:
                seen_ndcg.append(c_ndcg)
            else:
                unseen_ndcg.append(c_ndcg)

            # Visual vs Historical breakdown
            cand_vis = [f for f in cand_fids if facts_dict.get(f, {}).get("predicate") in VISUAL_PREDICATES]
            gold_vis = [f for f in gold_fids if facts_dict.get(f, {}).get("predicate") in VISUAL_PREDICATES]
            if cand_vis and gold_vis:
                rank_vis = sorted(cand_vis, key=lambda f: f_scores[f], reverse=True)
                m_v = compute_ranking_metrics(rank_vis, set(gold_vis), cand_vis, top_k_list=[5])
                vis_ndcg5.append(m_v.get("nDCG@5", 0.0))

            cand_hist = [f for f in cand_fids if facts_dict.get(f, {}).get("predicate") in HISTORICAL_PREDICATES]
            gold_hist = [f for f in gold_fids if facts_dict.get(f, {}).get("predicate") in HISTORICAL_PREDICATES]
            if cand_hist and gold_hist:
                rank_hist = sorted(cand_hist, key=lambda f: f_scores[f], reverse=True)
                m_h = compute_ranking_metrics(rank_hist, set(gold_hist), cand_hist, top_k_list=[5])
                hist_ndcg5.append(m_h.get("nDCG@5", 0.0))

    cont_acc = compute_contrastive_accuracy(flat_scores, contrastive_pairs)

    return {
        "ndcg@5": float(np.mean(ndcg5)),
        "p@5": float(np.mean(p5)),
        "hard_neg_auc": float(np.mean(aucs)) if aucs else 0.0,
        "contrastive_acc": float(cont_acc),
        "seen_ndcg@5": float(np.mean(seen_ndcg)),
        "unseen_ndcg@5": float(np.mean(unseen_ndcg)),
        "vis_ndcg@5": float(np.mean(vis_ndcg5)) if vis_ndcg5 else 0.0,
        "hist_ndcg@5": float(np.mean(hist_ndcg5)) if hist_ndcg5 else 0.0,
    }


# =============================================================================
# 4. TRAINING ENGINE WITH LISTMLE + SCENE HARD-NEGATIVE MARGIN
# =============================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Initializing Decoupled R-VCGAT with ListMLE Loss on Device: {device}")

    facts = load_normalized_facts()
    facts_dict = facts
    train_clips, _ = load_gold_data(split_type="train")
    val_clips, _ = load_gold_data(split_type="val")
    test_clips, _ = load_gold_data(split_type="test")
    train_contrastive = load_contrastive_pairs(split_type="train_val")
    test_contrastive = load_contrastive_pairs(split_type="test")

    feature_cache = BenchmarkFeatureCache(device=device)
    feature_cache.extract_fact_embeddings(facts)
    feature_cache.extract_video_features([])
    rel_subgraphs = torch.load(feature_cache.cache_dir / "landmark_subgraphs_rel.pt", weights_only=False)

    model = DecoupledRVCGATModel(d_model=768).to(device)
    optimizer = AdamW(model.parameters(), lr=1.0e-4, weight_decay=1e-4)

    train_cids = [c for c in train_clips.keys() if any(r.get("is_scene_gold") for r in train_clips[c])]

    best_val_score = -1.0
    best_state = None
    best_val_metrics = {}

    LOGGER.info("Starting training loop (15 Epochs with ListMLE + Scene Hard-Neg Margin)...")
    for epoch in range(1, 16):
        model.train()
        np.random.shuffle(train_cids)

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

            is_vis = [facts_dict.get(fid, {}).get("predicate", "").lower() in VISUAL_PREDICATES for fid in sub_fids]
            is_vis_tensor = torch.tensor(is_vis, dtype=torch.bool, device=device)

            optimizer.zero_grad()
            logits, s_vis, s_hist = model(h_tensor, adj, rel_adj, v_f_tensor, is_vis_tensor)
            idx_map = g["fid2idx"]

            pos_idx = [idx_map[fid] for fid in pos_fids if fid in idx_map]
            hard_neg_idx = [idx_map[fid] for fid in hard_neg_fids if fid in idx_map]
            cand_idx = [idx_map[r["fact_id"]] for r in recs if r["fact_id"] in idx_map]

            if not pos_idx or not cand_idx:
                continue

            # 1. ListMLE Top-K Ranking Loss
            loss_listmle = listmle_topk_loss(logits, pos_idx, cand_idx, top_k=5, temp=0.5)

            # 2. Scene Hard-Negative Triplet Margin (margin 0.8)
            p_scores = logits[pos_idx].unsqueeze(1)
            loss_hn = torch.tensor(0.0, device=device)
            if hard_neg_idx:
                hn_scores = logits[hard_neg_idx].unsqueeze(0)
                loss_hn = torch.relu(0.8 - (p_scores - hn_scores)).mean()

            # 3. Auxiliary BCE Loss on candidates for calibration
            y_true = [1.0] * len(pos_idx) + [0.0] * len(hard_neg_idx)
            if hard_neg_idx:
                bce_preds = torch.cat([logits[pos_idx], logits[hard_neg_idx]])
                bce_targets = torch.tensor(y_true, dtype=torch.float32, device=device)
                loss_bce = F.binary_cross_entropy_with_logits(bce_preds, bce_targets, pos_weight=torch.tensor([3.0], device=device))
            else:
                loss_bce = torch.tensor(0.0, device=device)

            total_loss = loss_listmle + 0.8 * loss_hn + 0.3 * loss_bce
            total_loss.backward()
            optimizer.step()

        # Validation Evaluation
        val_m = evaluate_decoupled_model(model, val_clips, train_contrastive, rel_subgraphs, feature_cache, facts_dict, device)
        val_score = 0.50 * val_m["ndcg@5"] + 0.30 * val_m["hard_neg_auc"] + 0.20 * val_m["hist_ndcg@5"]

        LOGGER.info(
            f"Epoch {epoch:02d} | Val nDCG@5: {val_m['ndcg@5']:.4f} | Hist nDCG@5: {val_m['hist_ndcg@5']:.4f} | "
            f"Val AUC: {val_m['hard_neg_auc']:.4f} | ValScore: {val_score:.4f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_metrics = val_m
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            LOGGER.info(f"  >>> Best Checkpoint at Epoch {epoch:02d} (ValScore: {val_score:.4f})")

    # Save Checkpoint
    ckpt_path = CHECKPOINT_DIR / "best_decoupled_rvcgat_listmle.pt"
    torch.save(best_state, ckpt_path)
    LOGGER.info(f"Saved best checkpoint to {ckpt_path}")

    # Final Test Evaluation on 800 Clips
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_m = evaluate_decoupled_model(model, test_clips, test_contrastive, rel_subgraphs, feature_cache, facts_dict, device)

    LOGGER.info("===========================================================================")
    LOGGER.info("FINAL TEST RESULTS (DECOUPLED R-VCGAT + LISTMLE LOSS):")
    LOGGER.info(f"  Overall nDCG@5      = {test_m['ndcg@5']:.4f}")
    LOGGER.info(f"  Overall P@5         = {test_m['p@5']:.4f}")
    LOGGER.info(f"  Hard Negative AUC   = {test_m['hard_neg_auc']:.4f}")
    LOGGER.info(f"  Contrastive Acc     = {test_m['contrastive_acc']:.1f}%")
    LOGGER.info(f"  Seen nDCG@5         = {test_m['seen_ndcg@5']:.4f}")
    LOGGER.info(f"  Unseen nDCG@5       = {test_m['unseen_ndcg@5']:.4f}")
    LOGGER.info(f"  Visual nDCG@5       = {test_m['vis_ndcg@5']:.4f}")
    LOGGER.info(f"  Historical nDCG@5   = {test_m['hist_ndcg@5']:.4f}")
    LOGGER.info("===========================================================================")

    # Comparison Table with Baseline Cross-Attention
    baseline_metrics = {
        "ndcg@5": 0.2882,
        "p@5": 0.2520,
        "hard_neg_auc": 0.6180,
        "contrastive_acc": 56.1,
        "seen_ndcg@5": 0.4196,
        "unseen_ndcg@5": 0.2083,
        "vis_ndcg@5": 0.3512,
        "hist_ndcg@5": 0.2245
    }

    report = {
        "baseline_cross_attention": baseline_metrics,
        "decoupled_rvcgat_listmle": test_m,
        "deltas": {
            "delta_ndcg5": test_m["ndcg@5"] - baseline_metrics["ndcg@5"],
            "delta_p5": test_m["p@5"] - baseline_metrics["p@5"],
            "delta_auc": test_m["hard_neg_auc"] - baseline_metrics["hard_neg_auc"],
            "delta_unseen": test_m["unseen_ndcg@5"] - baseline_metrics["unseen_ndcg@5"],
            "delta_historical": test_m["hist_ndcg@5"] - baseline_metrics["hist_ndcg@5"]
        }
    }

    json_path = REPORTS_DIR / "decoupled_rvcgat_listmle_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    LOGGER.info(f"Saved results JSON to {json_path}")

    # Build Markdown
    md = []
    md.append("# Decoupled Visual Grounding + Relational Evidence Branch with ListMLE Loss")
    md.append("Evaluated on 800 independent Test clips against Cross-Attention Baseline.\n")
    md.append("| Metric | Cross-Attention (Baseline) | Decoupled R-VCGAT + ListMLE | Delta (Δ) | Relative Improvement |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for k, name in [
        ("ndcg@5", "Overall nDCG@5"),
        ("p@5", "Precision@5 (P@5)"),
        ("hard_neg_auc", "Hard Negative AUC"),
        ("contrastive_acc", "Contrastive Accuracy (%)"),
        ("seen_ndcg@5", "Seen Landmarks nDCG@5"),
        ("unseen_ndcg@5", "Unseen Landmarks nDCG@5"),
        ("vis_ndcg@5", "Visual Facts nDCG@5"),
        ("hist_ndcg@5", "Historical Facts nDCG@5 (Target)"),
    ]:
        b = baseline_metrics[k]
        m = test_m[k]
        d = m - b
        pct = (d / b * 100.0) if b != 0 else 0.0
        unit = "%" if "acc" in k else ""
        md.append(f"| **{name}** | {b:.4f}{unit} | **{m:.4f}{unit}** | **{d:+.4f}** | **{pct:+.2f}%** |")

    md_content = "\n".join(md)
    table_path = REPORTS_DIR / "decoupled_rvcgat_listmle_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    LOGGER.info(f"Saved Markdown table to {table_path}")
    print("\n" + md_content)


if __name__ == "__main__":
    main()
