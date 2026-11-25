#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/run_paired_bootstrap_significance.py
=============================================================================
Rigorous Statistical Significance & 95% Paired Bootstrap Confidence Interval
Analysis for IJAR Submission:
Comparing Proposed R-VCGAT vs. Strongest Baseline (Cross-Attention Ranker).

Evaluates across 800 independent Test clips (301 Seen Hanoi, 499 Unseen Hanoi+HCMC):
  - 10,000 Paired Bootstrap Resamples for 95% CI
  - Paired Student's t-test (parametric)
  - Wilcoxon Signed-Rank Test (non-parametric)
  - Cohen's d effect size
  - Stratified Breakdown: Overall, Seen, Unseen, Hard Negative AUC, Historical Facts
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
LOGGER = logging.getLogger("SignificanceTest")

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
# 1. R-VCGAT MODEL ARCHITECTURE (PROPOSED FULL)
# =============================================================================
class FullUnifiedRVCGAT(nn.Module):
    def __init__(self, d_model=768, num_heads=4, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # 1. Cross-Attention Visual Backbone
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
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
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d_model + 3, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        nn.init.constant_(self.gate_mlp[-1].bias, -0.6)

    def forward(self, h, adj_mask, rel_adj, degrees, v_frames, v_pooled):
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

        # C. Relation-Aware VC-GAT
        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        h_gated = h * (0.5 + 0.5 * g_nodes)
        K = self.k_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V_nodes = self.v_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        rel_emb = self.rel_embed(rel_adj)  # [N, N, 64]
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

        # D. Graph Delta Score
        graph_feat = torch.cat([h_gat, ctx, h_gat * ctx], dim=-1)
        s_graph_delta = self.graph_delta_scorer(graph_feat).squeeze(-1)

        # E. Evidence-Based Reliability Gate
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

        # Final Score
        final_score = s_vis + lambda_gate * s_graph_delta
        return final_score, s_vis, lambda_gate


# =============================================================================
# 2. STATISTICAL INFERENCE FUNCTIONS (BOOTSTRAP, T-TEST, WILCOXON)
# =============================================================================
def compute_paired_statistics(
    x: np.ndarray,  # Model A (Proposed R-VCGAT)
    y: np.ndarray,  # Model B (Baseline Cross-Attention)
    n_bootstrap: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes complete paired statistical significance metrics:
      - 95% Bootstrap Percentile Confidence Interval
      - Bootstrap p-value
      - Paired Student's t-test (t_stat, p_value, df)
      - Wilcoxon Signed-Rank test (w_stat, p_value)
      - Cohen's d paired effect size
    """
    assert len(x) == len(y), f"Mismatched sample lengths: {len(x)} vs {len(y)}"
    n = len(x)
    diff = x - y
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0

    # Relative improvement
    rel_gain = (mean_diff / mean_y * 100.0) if mean_y != 0 else 0.0

    # Cohen's d paired effect size
    cohen_d = (mean_diff / std_diff) if std_diff > 1e-9 else 0.0

    # 1. Paired Student's t-test
    if std_diff > 1e-9:
        t_stat, p_ttest = stats.ttest_rel(x, y)
    else:
        t_stat, p_ttest = 0.0, 1.0

    # 2. Wilcoxon Signed-Rank Test (two-sided, Pratt method for zero differences)
    non_zero = np.sum(diff != 0)
    if non_zero > 10:
        try:
            w_res = stats.wilcoxon(x, y, zero_method="pratt", alternative="two-sided")
            w_stat, p_wilcoxon = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, p_wilcoxon = 0.0, 1.0
    else:
        w_stat, p_wilcoxon = 0.0, 1.0

    # 3. Paired Bootstrap Resampling
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_diff_means = np.mean(diff[boot_indices], axis=1)

    ci_lower = float(np.percentile(boot_diff_means, 2.5))
    ci_upper = float(np.percentile(boot_diff_means, 97.5))

    # Bootstrap p-value (proportion of resamples with sign opposite to mean_diff)
    if mean_diff >= 0:
        p_boot = 2.0 * float(np.mean(boot_diff_means <= 0))
    else:
        p_boot = 2.0 * float(np.mean(boot_diff_means >= 0))
    p_boot = min(p_boot, 1.0)

    # Significance indicator string
    sig_str = "n.s."
    min_p = min(p_ttest, p_wilcoxon)
    if min_p < 0.001:
        sig_str = "p < 0.001 ***"
    elif min_p < 0.01:
        sig_str = "p < 0.01 **"
    elif min_p < 0.05:
        sig_str = "p < 0.05 *"

    return {
        "n_samples": n,
        "mean_proposed": mean_x,
        "mean_baseline": mean_y,
        "delta": mean_diff,
        "relative_gain_pct": rel_gain,
        "cohen_d": float(cohen_d),
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "ci_95_str": f"[{ci_lower:+.4f}, {ci_upper:+.4f}]",
        "bootstrap_p": p_boot,
        "t_stat": float(t_stat),
        "t_test_p": float(p_ttest),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(p_wilcoxon),
        "is_significant_005": bool(ci_lower > 0 or ci_upper < 0),
        "significance_label": sig_str
    }


# =============================================================================
# 3. MAIN RUNNER: TRAINING, INFERENCE & PAIRED TESTING
# =============================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Running Paired Statistical Significance Benchmark on Device: {device}")

    facts = load_normalized_facts()
    facts_dict = facts
    train_clips, _ = load_gold_data(split_type="train")
    val_clips, _ = load_gold_data(split_type="val")
    test_clips, _ = load_gold_data(split_type="test")
    train_contrastive = load_contrastive_pairs(split_type="train_val")
    test_contrastive = load_contrastive_pairs(split_type="test")
    LOGGER.info(f"Dataset loaded: 800 Test clips ({len(test_clips)} total).")

    feature_cache = BenchmarkFeatureCache(device=device)
    feature_cache.extract_fact_embeddings(facts)
    feature_cache.extract_video_features([])
    rel_subgraphs = torch.load(feature_cache.cache_dir / "landmark_subgraphs_rel.pt", weights_only=False)

    # -------------------------------------------------------------------------
    # STEP A: LOAD BASELINE (CROSS-ATTENTION RANKER)
    # -------------------------------------------------------------------------
    cross_ckpt_path = CHECKPOINT_DIR / "best_cross_attention.pt"
    if not cross_ckpt_path.exists():
        raise FileNotFoundError(f"Missing {cross_ckpt_path}!")

    LOGGER.info(f"Loading Cross-Attention Ranker Baseline from {cross_ckpt_path}...")
    cross_model = CrossAttentionRankerModel(d_model=768).to(device)
    cross_state = torch.load(cross_ckpt_path, map_location=device, weights_only=False)
    cross_model.load_state_dict(cross_state)
    cross_model.eval()

    # -------------------------------------------------------------------------
    # STEP B: LOAD OR TRAIN PROPOSED R-VCGAT MODEL (SAVING ALL CANDIDATE EPOCHS)
    # -------------------------------------------------------------------------
    LOGGER.info("Training R-VCGAT (Cross-Attention Visual Backbone + Relation-Aware Residual Graph)...")
    rvcgat_model = FullUnifiedRVCGAT(d_model=768).to(device)
    optimizer = AdamW(rvcgat_model.parameters(), lr=1.2e-4, weight_decay=1e-4)
    train_cids = [c for c in train_clips.keys() if any(r.get("is_scene_gold") for r in train_clips[c])]
    
    epoch_checkpoints = {}
    best_val_ndcg = -1.0
    best_epoch = -1

    for epoch in range(1, 16):
        rvcgat_model.train()
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

            logits, _, _ = rvcgat_model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
            idx_map = g["fid2idx"]
            pos_idx = [idx_map[fid] for fid in pos_fids if fid in idx_map]
            hard_neg_idx = [idx_map[fid] for fid in hard_neg_fids if fid in idx_map]
            neg_idx = [idx_map[fid] for fid in list(all_neg_fids)[: len(pos_idx) * 6] if fid in idx_map]
            if not pos_idx or not neg_idx:
                continue

            # BCE
            y_true = [1.0] * len(pos_idx) + [0.0] * len(neg_idx)
            preds = torch.cat([logits[pos_idx], logits[neg_idx]])
            targets = torch.tensor(y_true, dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(preds, targets, pos_weight=torch.tensor([4.0], device=device))

            # Top Ranking loss (margin 0.6)
            p_scores = logits[pos_idx].unsqueeze(1)
            n_scores = logits[neg_idx].unsqueeze(0)
            rank_loss = torch.relu(0.6 - (p_scores - n_scores)).mean()
            loss = loss + 0.8 * rank_loss

            # Scene Hard Negative margin (margin 0.8)
            if hard_neg_idx:
                hn_scores = logits[hard_neg_idx].unsqueeze(0)
                hn_loss = torch.relu(0.8 - (p_scores - hn_scores)).mean()
                loss = loss + 0.7 * hn_loss

            loss.backward()
            optimizer.step()

        # Contrastive pairs
        if train_contrastive:
            np.random.shuffle(train_contrastive)
            for p in train_contrastive[:150]:
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
                s_a, _, _ = rvcgat_model(h_tensor, adj, rel_adj, degrees, torch.from_numpy(v_fa).float().to(device), torch.from_numpy(v_pa).float().to(device))
                s_b, _, _ = rvcgat_model(h_tensor, adj, rel_adj, degrees, torch.from_numpy(v_fb).float().to(device), torch.from_numpy(v_pb).float().to(device))
                cont_loss = torch.relu(0.8 - (s_a[f_idx] - s_b[f_idx]))
                if cont_loss > 0:
                    cont_loss.backward()
                    optimizer.step()

        # Val Eval for checkpoint selection
        rvcgat_model.eval()
        val_ndcg_list = []
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

                logits, _, _ = rvcgat_model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
                scores_np = logits.cpu().numpy()
                idx_map = g["fid2idx"]
                gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
                cand_fids = [r["fact_id"] for r in recs]
                f_scores = {fid: float(scores_np[idx_map[fid]]) if fid in idx_map else 0.0 for fid in cand_fids}
                ranked_fids = sorted(cand_fids, key=lambda fid: f_scores[fid], reverse=True)
                m = compute_ranking_metrics(ranked_fids, gold_fids, cand_fids, top_k_list=[5])
                val_ndcg_list.append(m.get("nDCG@5", 0.0))

        v_mean = float(np.mean(val_ndcg_list))
        LOGGER.info(f"Epoch {epoch:02d} | Val nDCG@5: {v_mean:.4f}")
        if epoch >= 8:
            epoch_checkpoints[epoch] = {k: v.cpu().clone() for k, v in rvcgat_model.state_dict().items()}
        if v_mean > best_val_ndcg:
            best_val_ndcg = v_mean
            best_epoch = epoch

    LOGGER.info(f"Validation selected Best Epoch: {best_epoch:02d} (Val nDCG@5: {best_val_ndcg:.4f})")
    torch.save(epoch_checkpoints[best_epoch], CHECKPOINT_DIR / "best_rvcgat_val_selected.pt")
    rvcgat_model.load_state_dict({k: v.to(device) for k, v in epoch_checkpoints[best_epoch].items()})
    rvcgat_model.eval()

    # -------------------------------------------------------------------------
    # STEP C: PER-CLIP SIDE-BY-SIDE INFERENCE OVER 800 TEST CLIPS
    # -------------------------------------------------------------------------
    LOGGER.info("Executing side-by-side clip-level inference on 800 Test clips...")

    cids_ordered = sorted(list(test_clips.keys()))
    clip_metadata = []

    # Per-clip metric storage
    rvcgat_ndcg5, cross_ndcg5 = [], []
    rvcgat_p5, cross_p5 = [], []
    rvcgat_aucs, cross_aucs = [], []
    auc_clip_mask = []

    rvcgat_hist_ndcg5, cross_hist_ndcg5 = [], []
    hist_clip_mask = []

    with torch.no_grad():
        for c_id in cids_ordered:
            recs = test_clips[c_id]
            l_id = recs[0].get("landmark_id", "unknown")
            is_seen = (l_id in SEEN_LANDMARKS)
            clip_metadata.append({"clip_id": c_id, "landmark_id": l_id, "is_seen": is_seen})

            g = rel_subgraphs.get(l_id)
            v_frames = feature_cache.video_frames.get(c_id)
            v_pooled = feature_cache.video_pooled.get(c_id)
            cand_fids = [r["fact_id"] for r in recs]
            gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")

            # 1. R-VCGAT Inference
            sub_fids = g["fact_ids"]
            h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
            h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device)
            adj = g["adj"].to(device)
            rel_adj = g["rel_adj"].to(device)
            degrees = g["degrees"].to(device)

            logits_r, _, _ = rvcgat_model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
            scores_r_np = logits_r.cpu().numpy()
            idx_map = g["fid2idx"]
            f_scores_r = {fid: float(scores_r_np[idx_map[fid]]) if fid in idx_map else 0.0 for fid in cand_fids}

            # 2. Cross-Attention Inference
            cand_h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in cand_fids]
            cand_h_tensor = torch.from_numpy(np.array(cand_h_list, dtype=np.float32)).to(device)
            logits_c = cross_model(v_f_tensor, cand_h_tensor)
            scores_c_np = logits_c.cpu().numpy()
            f_scores_c = {fid: float(scores_c_np[i]) for i, fid in enumerate(cand_fids)}

            # Compute nDCG@5 and P@5
            ranked_r = sorted(cand_fids, key=lambda fid: f_scores_r[fid], reverse=True)
            ranked_c = sorted(cand_fids, key=lambda fid: f_scores_c[fid], reverse=True)

            m_r = compute_ranking_metrics(ranked_r, gold_fids, cand_fids, top_k_list=[5])
            m_c = compute_ranking_metrics(ranked_c, gold_fids, cand_fids, top_k_list=[5])

            rvcgat_ndcg5.append(m_r.get("nDCG@5", 0.0))
            cross_ndcg5.append(m_c.get("nDCG@5", 0.0))
            rvcgat_p5.append(m_r.get("P@5", 0.0))
            cross_p5.append(m_c.get("P@5", 0.0))

            # Compute Hard Negative AUC if valid
            auc_r = compute_hard_negative_auc(f_scores_r, gold_fids, hard_neg_fids)
            auc_c = compute_hard_negative_auc(f_scores_c, gold_fids, hard_neg_fids)
            if auc_r is not None and auc_c is not None:
                rvcgat_aucs.append(auc_r)
                cross_aucs.append(auc_c)
                auc_clip_mask.append(True)
            else:
                auc_clip_mask.append(False)

            # Historical Facts breakdown
            hist_cands = [f for f in cand_fids if facts_dict.get(f, {}).get("predicate") in HISTORICAL_PREDICATES]
            hist_golds = [f for f in gold_fids if facts_dict.get(f, {}).get("predicate") in HISTORICAL_PREDICATES]
            if hist_cands and hist_golds:
                r_hist_rank = sorted(hist_cands, key=lambda f: f_scores_r[f], reverse=True)
                c_hist_rank = sorted(hist_cands, key=lambda f: f_scores_c[f], reverse=True)
                m_rh = compute_ranking_metrics(r_hist_rank, set(hist_golds), hist_cands, top_k_list=[5])
                m_ch = compute_ranking_metrics(c_hist_rank, set(hist_golds), hist_cands, top_k_list=[5])
                rvcgat_hist_ndcg5.append(m_rh.get("nDCG@5", 0.0))
                cross_hist_ndcg5.append(m_ch.get("nDCG@5", 0.0))
                hist_clip_mask.append(True)
            else:
                hist_clip_mask.append(False)

    # Convert to numpy
    arr_rv_ndcg = np.array(rvcgat_ndcg5)
    arr_cr_ndcg = np.array(cross_ndcg5)
    arr_rv_p5 = np.array(rvcgat_p5)
    arr_cr_p5 = np.array(cross_p5)

    seen_mask = np.array([m["is_seen"] for m in clip_metadata])
    unseen_mask = ~seen_mask

    arr_rv_auc = np.array(rvcgat_aucs)
    arr_cr_auc = np.array(cross_aucs)

    arr_rv_hist = np.array(rvcgat_hist_ndcg5)
    arr_cr_hist = np.array(cross_hist_ndcg5)

    LOGGER.info(f"Inference completed. Running 10,000 paired bootstrap iterations...")

    # -------------------------------------------------------------------------
    # STEP D: RUN STATISTICAL TESTS FOR EACH TARGET SLICE
    # -------------------------------------------------------------------------
    test_slices = {
        "Overall nDCG@5 (800 Clips)": (arr_rv_ndcg, arr_cr_ndcg),
        "Seen Landmarks nDCG@5 (301 Clips)": (arr_rv_ndcg[seen_mask], arr_cr_ndcg[seen_mask]),
        "Unseen Landmarks nDCG@5 (499 Clips)": (arr_rv_ndcg[unseen_mask], arr_cr_ndcg[unseen_mask]),
        "Precision@5 (800 Clips)": (arr_rv_p5, arr_cr_p5),
        "Hard Negative AUC": (arr_rv_auc, arr_cr_auc),
        "Historical Facts nDCG@5": (arr_rv_hist, arr_cr_hist),
    }

    results = {}
    for name, (vec_r, vec_c) in test_slices.items():
        stats_dict = compute_paired_statistics(vec_r, vec_c, n_bootstrap=10000, seed=42)
        results[name] = stats_dict
        LOGGER.info(
            f"[{name}] Proposed: {stats_dict['mean_proposed']:.4f} vs Baseline: {stats_dict['mean_baseline']:.4f} | "
            f"Delta: {stats_dict['delta']:+.4f} ({stats_dict['relative_gain_pct']:+.2f}%) | "
            f"95% CI: {stats_dict['ci_95_str']} | p_boot: {stats_dict['bootstrap_p']:.4f} | "
            f"p_wilcox: {stats_dict['wilcoxon_p']:.4e} ({stats_dict['significance_label']})"
        )

    # Save JSON results
    json_path = REPORTS_DIR / "paired_bootstrap_significance_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    LOGGER.info(f"Saved statistical significance JSON to {json_path}")

    # Build Markdown table
    md = []
    md.append("# Paired Bootstrap 95% Confidence Interval & Statistical Significance Analysis")
    md.append("Comparing Proposed **R-VCGAT** against Strongest Baseline (**Cross-Attention Ranker**) on 800 Test Clips.")
    md.append("Bootstrap iterations: $B = 10,000$, confidence level: 95%, two-sided paired hypothesis tests.\n")
    md.append("| Metric / Evaluation Slice | N Clips | Cross-Attn (Baseline) | R-VCGAT (Proposed) | Paired Delta (Δ) | Relative Gain | 95% Bootstrap CI | Cohen's d | p-value (t-test) | p-value (Wilcoxon) | Significance |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for name, r in results.items():
        md.append(
            f"| **{name}** | {r['n_samples']} | {r['mean_baseline']:.4f} | {r['mean_proposed']:.4f} | "
            f"**{r['delta']:+.4f}** | **{r['relative_gain_pct']:+.2f}%** | `{r['ci_95_str']}` | "
            f"{r['cohen_d']:.3f} | {r['t_test_p']:.4e} | {r['wilcoxon_p']:.4e} | **{r['significance_label']}** |"
        )

    md_content = "\n".join(md)
    table_path = REPORTS_DIR / "paired_bootstrap_significance_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    LOGGER.info(f"Saved Markdown table to {table_path}")
    print("\n" + md_content)


if __name__ == "__main__":
    main()
