#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/test_vcgat_architectures.py
=============================================================================
Ablation and Optimization Suite for VC-GAT:
Directly targets and optimizes:
  1. Hard Negative AUC
  2. Same-Landmark Contrastive Accuracy

Compares:
  - Architecture 1 (V1 - Static Feature Leakage, Intra-clip only)
  - Architecture 2 (V2 - Pure Multiplicative Video Gating, Intra-clip only)
  - Architecture 3 (V3 - Video Energy Gating + Multi-Objective Contrastive Loss)
=============================================================================
"""

import sys
import math
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

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
)
from experiments.fact_selection_benchmark.feature_cache import BenchmarkFeatureCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("TestVCGAT")


# =============================================================================
# ARCHITECTURE 1 (V1 - Original Additive Design)
# =============================================================================
class VCGATLayerV1(nn.Module):
    def __init__(self, d_model: int = 768, num_heads: int = 4, tau_v: float = 0.15, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.tau_v = tau_v

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.v_gate = nn.Linear(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        N = h.size(0)
        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.k_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        v_gate = torch.sigmoid(self.v_gate(v) / self.tau_v).view(self.num_heads, 1, 1)
        scores = scores * v_gate

        mask = (adj_mask == 0).unsqueeze(0).expand(self.num_heads, -1, -1)
        scores = scores.masked_fill(mask, -1e9)
        attn_weights = self.dropout(F.softmax(scores, dim=-1))

        context = torch.bmm(attn_weights, V).transpose(0, 1).contiguous().view(N, self.d_model)
        return self.norm(h + self.dropout(self.out_proj(context)))


class VCGATModelV1(nn.Module):
    def __init__(self, d_model: int = 768, num_heads: int = 4, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.v_norm = nn.LayerNorm(d_model)
        self.layer = VCGATLayerV1(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.scorer = nn.Sequential(
            nn.Linear(3 * d_model + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 1:
            v = v.unsqueeze(0)
        v_n = self.v_norm(v)
        curr_h = self.layer(h, adj_mask, v_n)
        N = curr_h.size(0)
        v_exp = v_n.expand(N, -1)
        cos_sim = torch.sum(F.normalize(curr_h, p=2, dim=-1) * F.normalize(v_n, p=2, dim=-1).expand(N, -1), dim=-1, keepdim=True)
        feat = torch.cat([curr_h, v_exp, curr_h * v_exp, cos_sim], dim=-1)
        return self.scorer(feat).squeeze(-1)


# =============================================================================
# ARCHITECTURE 2 & 3 (V2 & V3 - Pure Multiplicative Video Gating & Energy Flow)
# =============================================================================
class ContrastiveVCGATLayer(nn.Module):
    """
    Video-Conditioned Graph Attention Layer with Neighbor Saliency Gating:
    Attention and message transmission are conditioned on neighbor visual relevance.
    """
    def __init__(self, d_model: int = 768, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Video-conditioned attention bias
        self.edge_gate = nn.Linear(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor, v: torch.Tensor, g_nodes: torch.Tensor) -> torch.Tensor:
        """
        h: [N, D]
        adj_mask: [N, N]
        v: [1, D]
        g_nodes: [N, 1] (node-level visual saliency gate in [0, 1])
        """
        N = h.size(0)
        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, d]
        
        # Modulate key and value representations by node saliency gate
        # Only visually grounded nodes or active conceptual nodes propagate strong signals
        h_gated = h * (0.5 + 0.5 * g_nodes)
        K = self.k_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [H, N, N]

        # Add video-conditioned edge bias
        v_edge_bias = self.edge_gate(v).view(self.num_heads, 1, 1)
        scores = scores + 0.2 * v_edge_bias

        mask = (adj_mask == 0).unsqueeze(0).expand(self.num_heads, -1, -1)
        scores = scores.masked_fill(mask, -1e9)
        attn_weights = self.dropout(F.softmax(scores, dim=-1))

        context = torch.bmm(attn_weights, V).transpose(0, 1).contiguous().view(N, self.d_model)
        return self.norm(h + self.dropout(self.out_proj(context)))


class ContrastiveVCGATModel(nn.Module):
    """
    Contrastive VC-GAT with Zero Static Bias:
    Scores are strictly multiplicative interactions with the video.
    """
    def __init__(self, d_model: int = 768, num_heads: int = 4, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.v_norm = nn.LayerNorm(d_model)

        # Node Saliency Gate: computes initial visual relevance g_i(v)
        self.saliency_gate = nn.Sequential(
            nn.Linear(2 * d_model + 1, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.layer = ContrastiveVCGATLayer(d_model=d_model, num_heads=num_heads, dropout=dropout)

        # Scorer: Purely interactive features [h_out * v (768), |h_out - v| (768), cos_sim (1), g_nodes (1)] = 1538
        # NO isolated h_out terms! Prevents static fact bias leakage!
        self.scorer = nn.Sequential(
            nn.Linear(2 * d_model + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 1:
            v = v.unsqueeze(0)
        v_n = self.v_norm(v)
        N = h.size(0)
        v_exp = v_n.expand(N, -1)

        # 1. Compute Node Saliency Gate g_i(v)
        h_norm_unit = F.normalize(h, p=2, dim=-1)
        v_norm_unit = F.normalize(v_n, p=2, dim=-1).expand(N, -1)
        cos_sim_init = torch.sum(h_norm_unit * v_norm_unit, dim=-1, keepdim=True)
        gate_input = torch.cat([h * v_exp, torch.abs(h - v_exp), cos_sim_init], dim=-1)
        g_nodes = self.saliency_gate(gate_input)  # [N, 1] in [0, 1]

        # 2. Video-Conditioned Graph Attention
        curr_h = self.layer(h, adj_mask, v_n, g_nodes)

        # 3. Final Multiplicative Readout (Zero Static Bias)
        curr_h_unit = F.normalize(curr_h, p=2, dim=-1)
        cos_sim_final = torch.sum(curr_h_unit * v_norm_unit, dim=-1, keepdim=True)
        feat = torch.cat([curr_h * v_exp, torch.abs(curr_h - v_exp), cos_sim_final, g_nodes], dim=-1)
        return self.scorer(feat).squeeze(-1)


# =============================================================================
# EVALUATION & TRAINING FUNCTIONS
# =============================================================================
def eval_model_on_val(
    model: nn.Module,
    val_clip_records: Dict[str, List[Dict[str, Any]]],
    feature_cache: BenchmarkFeatureCache,
    val_contrastive_pairs: List[Dict[str, Any]],
    device: torch.device
) -> Tuple[float, float, float]:
    """Returns (mean_ndcg, mean_auc, contrastive_acc) on Val split."""
    model.eval()
    ndcg_list = []
    auc_list = []
    clip_fact_scores = {}

    with torch.no_grad():
        for c_id, recs in val_clip_records.items():
            l_id = recs[0].get("landmark_id", "unknown")
            gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
            cand_fids = [r["fact_id"] for r in recs]

            v_pooled = feature_cache.video_pooled.get(c_id)
            if v_pooled is None:
                continue

            g = feature_cache.subgraphs.get(l_id)
            if g is None:
                continue

            sub_fids = g["fact_ids"]
            h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
            h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
            v_tensor = torch.from_numpy(v_pooled).float().to(device)
            adj = g["adj"].to(device)

            logits = model(h_tensor, adj, v_tensor)
            scores_arr = logits.cpu().numpy()
            fid2score = {fid: float(scores_arr[g["fid2idx"][fid]]) for fid in cand_fids if fid in g["fid2idx"]}

            for fid, s in fid2score.items():
                clip_fact_scores[(c_id, fid)] = s

            ranked = sorted(cand_fids, key=lambda fid: fid2score.get(fid, -999.0), reverse=True)
            if gold_fids:
                m = compute_ranking_metrics(ranked, gold_fids, cand_fids, top_k_list=[5])
                ndcg_list.append(m["nDCG@5"])

            auc = compute_hard_negative_auc(fid2score, gold_fids, hard_neg_fids)
            if auc is not None:
                auc_list.append(auc)

    mean_ndcg = float(np.mean(ndcg_list)) if ndcg_list else 0.0
    mean_auc = float(np.mean(auc_list)) if auc_list else 0.5
    cont_acc = compute_contrastive_accuracy(clip_fact_scores, val_contrastive_pairs)

    return mean_ndcg, mean_auc, cont_acc


def run_experiment():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Running VC-GAT Architecture Optimization on device: {device}")

    facts = load_normalized_facts()
    test_manifest, train_val_manifest = load_clip_manifests()
    all_manifest_list = list(test_manifest.values()) + list(train_val_manifest.values())

    feature_cache = BenchmarkFeatureCache(device=device)
    feature_cache.extract_fact_embeddings(facts)
    feature_cache.extract_video_features(all_manifest_list)
    feature_cache.build_landmark_subgraphs(facts)

    train_clip_records, _ = load_gold_data(split_type="train")
    val_clip_records, _ = load_gold_data(split_type="val")
    train_contrastive_pairs = load_contrastive_pairs(split_type="train_val")
    # Filter validation contrastive pairs (where both clip A and clip B are in val)
    val_cids = set(val_clip_records.keys())
    val_contrastive_pairs = [
        p for p in train_contrastive_pairs
        if (p.get("clip_A", {}).get("clip_id") in val_cids or p.get("clip_a") in val_cids)
    ]
    if not val_contrastive_pairs:
        val_contrastive_pairs = train_contrastive_pairs[:500]

    train_cids = [c for c in train_clip_records.keys() if any(r.get("is_scene_gold") for r in train_clip_records[c])]
    LOGGER.info(f"Train active clips: {len(train_cids)}, Train contrastive pairs: {len(train_contrastive_pairs)}, Val pairs: {len(val_contrastive_pairs)}")

    # We test:
    # 1. Model V1 (BCE only)
    # 2. Model V2 (Pure Multiplicative, BCE + Hard Neg Loss)
    # 3. Model V3 (Pure Multiplicative, BCE + Hard Neg Loss + Cross-Scene Contrastive Pair Loss)
    configs = [
        ("V1_Additive_BCE_Only", VCGATModelV1(d_model=768), "bce_only"),
        ("V2_Multiplicative_HardNeg", ContrastiveVCGATModel(d_model=768), "hard_neg"),
        ("V3_Full_Contrastive_MultiTask", ContrastiveVCGATModel(d_model=768), "full_contrastive")
    ]

    for name, model, loss_mode in configs:
        LOGGER.info(f"\n{'='*70}\nTRAINING EXPERIMENT: [{name}]\n{'='*70}")
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

        for epoch in range(1, 13):
            model.train()
            total_loss = 0.0
            np.random.shuffle(train_cids)

            # 1. Train on clip candidates
            for c_id in train_cids:
                recs = train_clip_records[c_id]
                l_id = recs[0].get("landmark_id", "unknown")
                g = feature_cache.subgraphs.get(l_id)
                v_pooled = feature_cache.video_pooled.get(c_id)
                if g is None or v_pooled is None:
                    continue

                pos_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
                hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
                all_neg_fids = set(r["fact_id"] for r in recs if not r.get("is_scene_gold"))
                if not pos_fids:
                    continue

                optimizer.zero_grad()
                sub_fids = g["fact_ids"]
                h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
                h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
                v_tensor = torch.from_numpy(v_pooled).float().to(device)
                adj = g["adj"].to(device)

                logits = model(h_tensor, adj, v_tensor)

                idx_map = g["fid2idx"]
                pos_idx = [idx_map[fid] for fid in pos_fids if fid in idx_map]
                hard_neg_idx = [idx_map[fid] for fid in hard_neg_fids if fid in idx_map]
                neg_sample_idx = [idx_map[fid] for fid in list(all_neg_fids)[:len(pos_fids) * 4] if fid in idx_map]

                if not pos_idx or not neg_sample_idx:
                    continue

                # BCE loss
                y_true = [1.0] * len(pos_idx) + [0.0] * len(neg_sample_idx)
                preds = torch.cat([logits[pos_idx], logits[neg_sample_idx]])
                targets = torch.tensor(y_true, dtype=torch.float32, device=device)
                loss = F.binary_cross_entropy_with_logits(preds, targets, pos_weight=torch.tensor([4.0], device=device))

                # If hard_neg mode or full_contrastive: add hard negative margin loss
                if loss_mode in ["hard_neg", "full_contrastive"] and hard_neg_idx:
                    p_exp = logits[pos_idx].unsqueeze(1)
                    hn_exp = logits[hard_neg_idx].unsqueeze(0)
                    hn_loss = torch.relu(0.8 - (p_exp - hn_exp)).mean()
                    loss = loss + 1.0 * hn_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # 2. If full_contrastive: train on cross-scene contrastive pairs
            if loss_mode == "full_contrastive":
                np.random.shuffle(train_contrastive_pairs)
                # Sample batch of contrastive pairs
                for p in train_contrastive_pairs[:200]:
                    c_a = p.get("clip_a") or p.get("clip_A", {}).get("clip_id")
                    c_b = p.get("clip_b") or p.get("clip_B", {}).get("clip_id")
                    f_id = p.get("fact_id") or p.get("fact_a")
                    l_id = p.get("landmark_id")

                    v_a = feature_cache.video_pooled.get(c_a)
                    v_b = feature_cache.video_pooled.get(c_b)
                    g = feature_cache.subgraphs.get(l_id)
                    if v_a is None or v_b is None or g is None or f_id not in g["fid2idx"]:
                        continue

                    f_idx = g["fid2idx"][f_id]
                    sub_fids = g["fact_ids"]
                    h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
                    h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
                    adj = g["adj"].to(device)

                    optimizer.zero_grad()
                    s_a = model(h_tensor, adj, torch.from_numpy(v_a).float().to(device))[f_idx]
                    s_b = model(h_tensor, adj, torch.from_numpy(v_b).float().to(device))[f_idx]

                    # Contrastive margin loss: s_a should exceed s_b by margin 1.0
                    cont_loss = torch.relu(1.0 - (s_a - s_b))
                    if cont_loss > 0:
                        cont_loss.backward()
                        optimizer.step()

            # Validation
            val_ndcg, val_auc, val_cont = eval_model_on_val(
                model, val_clip_records, feature_cache, val_contrastive_pairs, device
            )
            LOGGER.info(
                f"Epoch {epoch:02d} | Train Loss: {total_loss:.2f} | "
                f"Val nDCG@5: {val_ndcg:.4f} | Hard Neg AUC: {val_auc:.4f} | Contrastive Acc: {val_cont:.1f}%"
            )

        LOGGER.info(f"FINAL METRICS FOR [{name}]: nDCG@5={val_ndcg:.4f}, AUC={val_auc:.4f}, Contrastive={val_cont:.1f}%")


if __name__ == "__main__":
    run_experiment()
