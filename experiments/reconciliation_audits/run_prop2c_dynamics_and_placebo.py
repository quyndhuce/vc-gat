#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_prop2c_dynamics_and_placebo.py
=============================================================================
Implementation of Proposition 2c'' (Structural Prior at Initialization & Placebos):

1. Training Trajectory (t=0, Epoch 1, 3, 5, 15):
   - Measure Spearman rho and TV between u_L(t) and pi_gamma
   - Measure content logit spread M*(t)
   - Show that correlation is high at initialization and drops as M*(t) grows.

2. Gradient Concentration at Step 1 / Epoch 1:
   - Node embedding gradient norms g_i = ||nabla_{h_i} L||_2
   - Compare gamma=0 vs gamma=0.40 on:
     * Top-1% hub gradient share
     * Gini coefficient of gradient norms

3. Placebo Experiments (3 seeds each: 42, 43, 44):
   - Branch A: Proposed gamma = 0.40 (True Degrees)
   - Branch B (Placebo 1): Permuted Degrees (randomly shuffled degrees per clip graph)
   - Branch C (Placebo 2): Temperature / Logit Contraction (scale attention logits by 0.70)
   - Branch D: Baseline gamma = 0.0 (No Debiasing)
   Measure Overall nDCG@5, P@5, Hard-Negative AUC on Hanoi-500 test clips with
   parent-video cluster bootstrap 95% CIs.
=============================================================================
"""

import sys
import json
import math
import copy
from pathlib import Path
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.fact_selection_benchmark.train_dec_vcgat_v2 import DecoupledEvidenceCalibratedVCGATv2, build_clip_graph
from experiments.fact_selection_benchmark.dataset_loader import (
    load_gold_data,
    load_normalized_facts,
    compute_ranking_metrics,
    compute_hard_negative_auc,
)
from experiments.fact_selection_benchmark.feature_cache import BenchmarkFeatureCache

SEED = 42
B_RESAMPLES = 10000
SEEN_LANDMARKS = {
    "hoan_kiem_lake", "temple_of_literature", "ba_dinh_square",
    "imperial_citadel_of_thang_long", "one_pillar_pagoda", "hanoi_old_quarter", "west_lake"
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def compute_theoretical_pi(adj, degrees, gamma):
    d = np.array(degrees, dtype=np.float64)
    a = (1.0 + d) ** (-gamma)
    W = np.array(adj, dtype=np.float64)
    S = W @ a
    S_safe = np.where(S > 1e-12, S, 1.0)
    A = (W * a[None, :]) / S_safe[:, None]
    isolated = (S <= 1e-12)
    A[isolated, :] = 0.0
    A[isolated, isolated] = 1.0
    numerator = a * S
    Z = np.sum(numerator)
    pi = numerator / Z if Z > 1e-12 else np.ones(len(d)) / len(d)
    return pi, A


def compute_gini(x):
    x = np.asarray(x, dtype=np.float64)
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x)))


def cluster_bootstrap_mean_ci(clip_records, cluster_key, metric_key, B=B_RESAMPLES, ci=0.95):
    clusters = defaultdict(list)
    for rec in clip_records:
        clusters[rec[cluster_key]].append(rec[metric_key])
    cluster_ids = list(clusters.keys())
    K = len(cluster_ids)
    
    boot_estimates = []
    rng = np.random.RandomState(SEED)
    
    for _ in range(B):
        sampled_cids = rng.choice(cluster_ids, size=K, replace=True)
        sampled_vals = []
        for cid in sampled_cids:
            sampled_vals.extend(clusters[cid])
        boot_estimates.append(np.mean(sampled_vals))
        
    boot_estimates = np.array(boot_estimates)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_estimates, alpha * 100))
    high = float(np.percentile(boot_estimates, (1.0 - alpha) * 100))
    se = float(np.std(boot_estimates))
    return float(np.mean(boot_estimates)), low, high, se


# =============================================================================
# PART 1: TRAINING TRAJECTORY & GRADIENT CONCENTRATION
# =============================================================================

def run_trajectory_and_gradients(train_clips, test_clips, subgraphs, feature_cache):
    print("\n" + "=" * 75)
    print("PART 1: TRAINING TRAJECTORY (t=0 to t=15) & INITIAL GRADIENT CONCENTRATION")
    print("=" * 75)
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # 1. Gradient Concentration Comparison at Step 1: gamma=0 vs gamma=0.40
    def measure_initial_gradients(gamma_val):
        model = DecoupledEvidenceCalibratedVCGATv2(d_model=768, degree_damping=gamma_val).to(device)
        cross_path = REPO_ROOT / "experiments" / "checkpoints_fact_selection" / "best_cross_attention.pt"
        if cross_path.exists():
            cross_weights = torch.load(cross_path, map_location=device)
            m_dict = model.state_dict()
            p_dict = {k: v for k, v in cross_weights.items() if k in m_dict and v.shape == m_dict[k].shape}
            m_dict.update(p_dict)
            model.load_state_dict(m_dict)
            
        model.train()
        train_cids = [cid for cid in train_clips if not cid.startswith("hcmc_")]
        
        all_grad_norms = []
        all_is_hub = []
        
        for c_id in train_cids[:40]: # sample first 40 clips
            recs = train_clips[c_id]
            l_id = recs[0].get("landmark_id", "unknown")
            g = subgraphs.get(l_id)
            v_frames = feature_cache.video_frames.get(c_id)
            v_pooled = feature_cache.video_pooled.get(c_id)
            if g is None or v_frames is None:
                continue
                
            cand_fids = [r["fact_id"] for r in recs]
            pos_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            if not pos_fids:
                continue
                
            h_tensor, adj, rel_adj, degrees, all_fids, fid2idx = build_clip_graph(
                g, cand_fids, feature_cache.fact_embeddings, device
            )
            h_tensor = h_tensor.clone().detach().requires_grad_(True)
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device) if v_pooled is not None else None
            
            logits, _, _, _ = model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
            
            cand_idx = [fid2idx[f] for f in cand_fids]
            pos_idx = [fid2idx[f] for f in pos_fids]
            neg_idx = [idx for idx in cand_idx if idx not in pos_idx]
            perm_idx = pos_idx[:5] + neg_idx
            perm_logits = logits[perm_idx] / 1.0
            loss = 0.0
            k_eval = min(len(pos_idx), 5)
            for i in range(k_eval):
                log_denom = torch.logsumexp(perm_logits[i:], dim=0)
                loss = loss - (perm_logits[i] - log_denom)
            loss = loss / float(k_eval)
            
            loss.backward()
            
            grads = h_tensor.grad.detach().cpu().numpy() # [N, D]
            gnorms = np.linalg.norm(grads, axis=-1)       # [N]
            
            deg_np = degrees.cpu().numpy()
            k_top = max(1, int(math.ceil(0.01 * len(deg_np))))
            hub_thresh = np.sort(deg_np)[::-1][k_top - 1]
            is_hub = (deg_np >= hub_thresh)
            
            all_grad_norms.extend(gnorms.tolist())
            all_is_hub.extend(is_hub.tolist())
            
        all_grad_norms = np.array(all_grad_norms)
        all_is_hub = np.array(all_is_hub)
        
        tot_grad = np.sum(all_grad_norms)
        hub_grad = np.sum(all_grad_norms[all_is_hub])
        hub_share = (hub_grad / tot_grad) if tot_grad > 0 else 0.0
        gini = compute_gini(all_grad_norms)
        
        return hub_share, gini

    hub_share_0, gini_0 = measure_initial_gradients(gamma_val=0.0)
    hub_share_4, gini_4 = measure_initial_gradients(gamma_val=0.40)
    
    print("\nInitial Gradient Concentration (Step 1):")
    print(f"  gamma=0.00: Top-1% Hub Gradient Share = {hub_share_0*100:.2f}% | Gini = {gini_0:.4f}")
    print(f"  gamma=0.40: Top-1% Hub Gradient Share = {hub_share_4*100:.2f}% | Gini = {gini_4:.4f}")
    print(f"  Gradient Suppression Ratio on Hubs: {hub_share_0 / max(hub_share_4, 1e-6):.2f}x reduction!")
    
    # 2. Training Trajectory Trace
    print("\nTracing Trajectory across Epochs (t=0, 1, 3, 5, 15)...")
    model = DecoupledEvidenceCalibratedVCGATv2(d_model=768, degree_damping=0.40).to(device)
    cross_path = REPO_ROOT / "experiments" / "checkpoints_fact_selection" / "best_cross_attention.pt"
    if cross_path.exists():
        cross_weights = torch.load(cross_path, map_location=device)
        m_dict = model.state_dict()
        p_dict = {k: v for k, v in cross_weights.items() if k in m_dict and v.shape == m_dict[k].shape}
        m_dict.update(p_dict)
        model.load_state_dict(m_dict)
        
    vis_params = list(model.cross_attn.parameters()) + list(model.norm1.parameters()) + list(model.vis_scorer.parameters())
    graph_params = (
        list(model.saliency_gate.parameters()) + list(model.q_proj.parameters()) +
        list(model.k_proj.parameters()) + list(model.v_proj.parameters()) +
        list(model.out_proj.parameters()) + list(model.rel_embed.parameters()) +
        list(model.rel_k_proj.parameters()) + list(model.rel_v_proj.parameters()) +
        list(model.rel_bias.parameters()) + list(model.norm_gat.parameters()) +
        list(model.graph_delta_mlp.parameters()) + list(model.evidence_gate.parameters())
    )
    optimizer = AdamW([{"params": vis_params, "lr": 1e-5}, {"params": graph_params, "lr": 1.5e-4}])
    
    hanoi_test_cids = sorted([cid for cid in test_clips if not cid.startswith("hcmc_")])
    train_cids = sorted([cid for cid in train_clips if not cid.startswith("hcmc_")])
    
    def evaluate_trajectory_snapshot(epoch_label):
        model.eval()
        sp_list, tv_list, m_star_list = [], [], []
        with torch.no_grad():
            for cid in hanoi_test_cids[:100]: # sample 100 test clips
                recs = test_clips[cid]
                lid = recs[0].get("landmark_id", "unknown")
                g = subgraphs.get(lid)
                v_frames = feature_cache.video_frames.get(cid)
                if g is None or v_frames is None:
                    continue
                cand_fids = [r["fact_id"] for r in recs]
                v_f_tensor = torch.from_numpy(v_frames).float().to(device)
                h_tensor, adj, rel_adj, degrees, all_fids, fid2idx = build_clip_graph(
                    g, cand_fids, feature_cache.fact_embeddings, device
                )
                N = h_tensor.size(0)
                
                q = h_tensor.unsqueeze(1)
                kv = v_f_tensor.unsqueeze(0).expand(N, -1, -1)
                attn_out, _ = model.cross_attn(query=q, key=kv, value=kv)
                ctx = model.norm1(attn_out.squeeze(1) + h_tensor)
                vis_feat = torch.cat([h_tensor, ctx, h_tensor * ctx], dim=-1)
                g_nodes = model.saliency_gate(vis_feat)
                
                Q = model.q_proj(h_tensor).view(N, model.num_heads, model.head_dim).transpose(0, 1)
                h_trans = h_tensor * g_nodes
                K = model.k_proj(h_trans).view(N, model.num_heads, model.head_dim).transpose(0, 1)
                scores = torch.bmm(Q, K.transpose(1, 2)) * model.scale
                rel_emb = model.rel_embed(rel_adj)
                rel_k = model.rel_k_proj(rel_emb).view(N, N, model.num_heads, model.head_dim).permute(2, 0, 1, 3)
                scores_rel = torch.sum(Q.unsqueeze(2) * rel_k, dim=-1) * model.scale
                scores = scores + scores_rel + model.rel_bias(rel_adj).permute(2, 0, 1)
                
                content_mean = scores.mean(dim=0).cpu().numpy()
                deg_penalty = model.degree_damping * torch.log1p(degrees.float()).unsqueeze(0).unsqueeze(1)
                scores_damped = scores - deg_penalty
                mask = (adj == 0).unsqueeze(0).expand(model.num_heads, -1, -1)
                scores_damped = scores_damped.masked_fill(mask, -1e9)
                attn = F.softmax(scores_damped, dim=-1).mean(dim=0).cpu().numpy()
                
                adj_np = adj.cpu().numpy()
                diff = np.abs(content_mean[adj_np > 0])
                m_star_list.append(float(np.max(diff)) if len(diff) > 0 else 0.0)
                
                A_4 = np.linalg.matrix_power(attn, 4)
                u_4 = np.mean(A_4, axis=0)
                
                pi_04, _ = compute_theoretical_pi(adj_np, degrees.cpu().numpy(), gamma=0.40)
                sp, _ = stats.spearmanr(u_4, pi_04)
                if not np.isnan(sp):
                    sp_list.append(float(sp))
                tv = 0.5 * np.sum(np.abs(u_4 - pi_04))
                tv_list.append(float(tv))
                
        mean_sp = float(np.mean(sp_list))
        mean_tv = float(np.mean(tv_list))
        mean_m = float(np.mean(m_star_list))
        print(f"  Epoch {epoch_label:>2s}: Spearman rho(u_4, pi_0.4) = {mean_sp:+.4f} | TV = {mean_tv:.4f} | M* = {mean_m:.3f}")
        return {"epoch": epoch_label, "spearman": mean_sp, "tv": mean_tv, "m_star": mean_m}

    trajectory_records = []
    trajectory_records.append(evaluate_trajectory_snapshot("0"))
    
    eval_epochs = {1, 3, 5, 15}
    for ep in range(1, 16):
        model.train()
        np.random.shuffle(train_cids)
        for c_id in train_cids:
            recs = train_clips[c_id]
            l_id = recs[0].get("landmark_id", "unknown")
            g = subgraphs.get(l_id)
            v_frames = feature_cache.video_frames.get(c_id)
            v_pooled = feature_cache.video_pooled.get(c_id)
            if g is None or v_frames is None:
                continue
            cand_fids = [r["fact_id"] for r in recs]
            pos_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            if not pos_fids:
                continue
            h_tensor, adj, rel_adj, degrees, all_fids, fid2idx = build_clip_graph(
                g, cand_fids, feature_cache.fact_embeddings, device
            )
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device) if v_pooled is not None else None
            
            optimizer.zero_grad()
            logits, _, _, _ = model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
            
            cand_idx = [fid2idx[f] for f in cand_fids]
            pos_idx = [fid2idx[f] for f in pos_fids]
            neg_idx = [idx for idx in cand_idx if idx not in pos_idx]
            perm_idx = pos_idx[:5] + neg_idx
            perm_logits = logits[perm_idx] / 1.0
            loss = 0.0
            k_eval = min(len(pos_idx), 5)
            for i in range(k_eval):
                log_denom = torch.logsumexp(perm_logits[i:], dim=0)
                loss = loss - (perm_logits[i] - log_denom)
            loss = loss / float(k_eval)
            loss.backward()
            optimizer.step()
            
        if ep in eval_epochs:
            trajectory_records.append(evaluate_trajectory_snapshot(str(ep)))
            
    return {
        "gradient_concentration": {
            "gamma_0_hub_share": hub_share_0,
            "gamma_0_gini": gini_0,
            "gamma_4_hub_share": hub_share_4,
            "gamma_4_gini": gini_4
        },
        "trajectory": trajectory_records
    }


# =============================================================================
# PART 2: PLACEBO EXPERIMENTS (TRUE DEGREES vs PERMUTED DEGREES vs TEMPERATURE)
# =============================================================================

class ModularDECV2ModelPlacebo(DecoupledEvidenceCalibratedVCGATv2):
    def __init__(self, d_model=768, degree_damping=0.4, temp_scale=1.0):
        super().__init__(d_model=d_model, degree_damping=degree_damping)
        self.temp_scale = temp_scale

    def forward(self, h, adj, rel_adj, degrees, v_frames, v_pooled=None):
        N = h.size(0)
        q = h.unsqueeze(1)
        kv = v_frames.unsqueeze(0).expand(N, -1, -1)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        ctx = self.norm1(attn_out.squeeze(1) + h)
        vis_feat = torch.cat([h, ctx, h * ctx], dim=-1)
        s_vis = self.vis_scorer(vis_feat).squeeze(-1)
        g_nodes = self.saliency_gate(vis_feat)

        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        h_trans = h * g_nodes
        K = self.k_proj(h_trans).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V_nodes = self.v_proj(h_trans).view(N, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.bmm(Q, K.transpose(1, 2)) * (self.scale * self.temp_scale)
        rel_emb = self.rel_embed(rel_adj)
        rel_k = self.rel_k_proj(rel_emb).view(N, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        scores_rel = torch.sum(Q.unsqueeze(2) * rel_k, dim=-1) * (self.scale * self.temp_scale)
        scores = scores + scores_rel + self.rel_bias(rel_adj).permute(2, 0, 1)

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

        graph_feat = torch.cat([h_gat, ctx, h_gat * ctx], dim=-1)
        delta_s = self.delta_bound * self.graph_delta_mlp(graph_feat).squeeze(-1)

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

        final_score = s_vis + 0.4 * lambda_i * delta_s
        return final_score, s_vis, lambda_i, delta_s


def train_and_eval_branch(branch_name, damping_mode, seed, train_clips, test_clips, subgraphs, feature_cache, clip_to_video):
    torch.manual_seed(seed)
    np.random.seed(seed)

    temp_factor = 0.70 if damping_mode == 'uniform_temp' else 1.0
    deg_damp = 0.40 if damping_mode in ['proposed', 'permuted'] else 0.0

    model = ModularDECV2ModelPlacebo(
        d_model=768,
        degree_damping=deg_damp,
        temp_scale=temp_factor
    ).to(device)
    
    cross_path = REPO_ROOT / "experiments" / "checkpoints_fact_selection" / "best_cross_attention.pt"
    if cross_path.exists():
        cross_weights = torch.load(cross_path, map_location=device)
        m_dict = model.state_dict()
        p_dict = {k: v for k, v in cross_weights.items() if k in m_dict and v.shape == m_dict[k].shape}
        m_dict.update(p_dict)
        model.load_state_dict(m_dict)
        
    vis_params = list(model.cross_attn.parameters()) + list(model.norm1.parameters()) + list(model.vis_scorer.parameters())
    graph_params = (
        list(model.saliency_gate.parameters()) + list(model.q_proj.parameters()) +
        list(model.k_proj.parameters()) + list(model.v_proj.parameters()) +
        list(model.out_proj.parameters()) + list(model.rel_embed.parameters()) +
        list(model.rel_k_proj.parameters()) + list(model.rel_v_proj.parameters()) +
        list(model.rel_bias.parameters()) + list(model.norm_gat.parameters()) +
        list(model.graph_delta_mlp.parameters()) + list(model.evidence_gate.parameters())
    )
    optimizer = AdamW([{"params": vis_params, "lr": 1e-5}, {"params": graph_params, "lr": 1.5e-4}])
    
    train_cids = sorted([cid for cid in train_clips if not cid.startswith("hcmc_")])
    
    for ep in range(1, 16):
        model.train()
        np.random.shuffle(train_cids)
        for c_id in train_cids:
            recs = train_clips[c_id]
            l_id = recs[0].get("landmark_id", "unknown")
            g = subgraphs.get(l_id)
            v_frames = feature_cache.video_frames.get(c_id)
            v_pooled = feature_cache.video_pooled.get(c_id)
            if g is None or v_frames is None:
                continue
            cand_fids = [r["fact_id"] for r in recs]
            pos_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            if not pos_fids:
                continue
            h_tensor, adj, rel_adj, degrees, all_fids, fid2idx = build_clip_graph(
                g, cand_fids, feature_cache.fact_embeddings, device
            )
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device) if v_pooled is not None else None
            
            # If permuted, shuffle degrees of the clip graph
            if damping_mode == 'permuted':
                perm_idx = torch.randperm(len(degrees), device=device)
                use_degs = degrees[perm_idx]
            else:
                use_degs = degrees
                
            optimizer.zero_grad()
            logits, _, _, _ = model(h_tensor, adj, rel_adj, use_degs, v_f_tensor, v_p_tensor)
            
            cand_idx = [fid2idx[f] for f in cand_fids]
            pos_idx = [fid2idx[f] for f in pos_fids]
            neg_idx = [idx for idx in cand_idx if idx not in pos_idx]
            perm_idx = pos_idx[:5] + neg_idx
            perm_logits = logits[perm_idx] / 1.0
            loss = 0.0
            k_eval = min(len(pos_idx), 5)
            for i in range(k_eval):
                log_denom = torch.logsumexp(perm_logits[i:], dim=0)
                loss = loss - (perm_logits[i] - log_denom)
            loss = loss / float(k_eval)
            loss.backward()
            optimizer.step()

    # Evaluate on Hanoi-500 test clips
    model.eval()
    hanoi_test_cids = sorted([cid for cid in test_clips if not cid.startswith("hcmc_")])
    clip_records = []
    
    with torch.no_grad():
        for cid in hanoi_test_cids:
            recs = test_clips[cid]
            lid = recs[0].get("landmark_id", "unknown")
            vid = clip_to_video.get(cid, "unknown")
            g = subgraphs.get(lid)
            v_frames = feature_cache.video_frames.get(cid)
            v_pooled = feature_cache.video_pooled.get(cid)
            if g is None or v_frames is None:
                continue
            cand_fids = [r["fact_id"] for r in recs]
            gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
            
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device) if v_pooled is not None else None
            
            h_tensor, adj, rel_adj, degrees, all_fids, fid2idx = build_clip_graph(
                g, cand_fids, feature_cache.fact_embeddings, device
            )
            
            if damping_mode == 'permuted':
                perm_idx = torch.randperm(len(degrees), device=device)
                use_degs = degrees[perm_idx]
            else:
                use_degs = degrees
                
            logits, _, _, _ = model(h_tensor, adj, rel_adj, use_degs, v_f_tensor, v_p_tensor)
            scores_np = logits.cpu().numpy()
            f_scores = {fid: float(scores_np[fid2idx[fid]]) for fid in cand_fids}
            ranked = sorted(cand_fids, key=lambda f: f_scores[f], reverse=True)
            
            m = compute_ranking_metrics(ranked, gold_fids, cand_fids, top_k_list=[5])
            auc = compute_hard_negative_auc(f_scores, gold_fids, hard_neg_fids)
            
            clip_records.append({
                "clip_id": cid,
                "video_id": vid,
                "ndcg5": m["nDCG@5"],
                "p5": m["P@5"],
                "auc": auc if auc is not None else np.nan
            })
            
    mean_ndcg, low_ndcg, high_ndcg, se_ndcg = cluster_bootstrap_mean_ci(clip_records, "video_id", "ndcg5")
    mean_auc, low_auc, high_auc, se_auc = cluster_bootstrap_mean_ci([r for r in clip_records if not np.isnan(r["auc"])], "video_id", "auc")
    
    print(f"  [{branch_name} | Seed {seed}] nDCG@5: {mean_ndcg:.4f} [{low_ndcg:.4f}, {high_ndcg:.4f}] | AUC: {mean_auc:.4f} [{low_auc:.4f}, {high_auc:.4f}]")
    return {
        "branch": branch_name,
        "seed": seed,
        "ndcg5": mean_ndcg,
        "ndcg5_ci": [low_ndcg, high_ndcg],
        "auc": mean_auc,
        "auc_ci": [low_auc, high_auc]
    }


def run_placebo_suite(train_clips, test_clips, subgraphs, feature_cache, clip_to_video):
    print("\n" + "=" * 75)
    print("PART 2: PLACEBO SUITE & MULTI-SEED RIGOR (SEEDS 42, 43, 44)")
    print("=" * 75)
    
    branches = [
        ("Branch A: Proposed (gamma=0.40, True Degrees)", "proposed"),
        ("Branch B: Placebo 1 (Permuted Degrees)", "permuted"),
        ("Branch C: Placebo 2 (Uniform Temperature Scale 0.70)", "uniform_temp"),
        ("Branch D: Baseline (gamma=0.00, No Debias)", "no_debias")
    ]
    
    seeds = [42, 43, 44]
    suite_results = defaultdict(list)
    
    for branch_name, mode in branches:
        print(f"\n--- Running {branch_name} across 3 seeds ---")
        for s in seeds:
            res = train_and_eval_branch(branch_name, mode, s, train_clips, test_clips, subgraphs, feature_cache, clip_to_video)
            suite_results[branch_name].append(res)
            
    print("\n" + "=" * 75)
    print("PLACEBO & ABLATION SUMMARY ACROSS 3 SEEDS (Mean +/- Std):")
    print("=" * 75)
    summary = {}
    for branch_name, res_list in suite_results.items():
        ndcgs = [r["ndcg5"] for r in res_list]
        aucs = [r["auc"] for r in res_list]
        mean_n, std_n = float(np.mean(ndcgs)), float(np.std(ndcgs))
        mean_a, std_a = float(np.mean(aucs)), float(np.std(aucs))
        summary[branch_name] = {
            "ndcg5_mean": mean_n, "ndcg5_std": std_n,
            "auc_mean": mean_a, "auc_std": std_a,
            "runs": res_list
        }
        print(f"  {branch_name:<48}: nDCG@5 = {mean_n:.4f} +/- {std_n:.4f} | AUC = {mean_a:.4f} +/- {std_a:.4f}")
        
    return summary


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main():
    print("=" * 75)
    print("STARTING PROPOSITION 2c'' DYNAMICS & PLACEBO EXPERIMENTS")
    print("=" * 75)
    
    facts = load_normalized_facts()
    train_clips, _ = load_gold_data(split_type="train")
    test_clips, _ = load_gold_data(split_type="test")
    
    manifest_path = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "evaluation_clips_manifest.json"
    manifest_items = json.loads(manifest_path.read_text())
    clip_to_video = {item["clip_id"]: item.get("source_video_id") or item.get("video_id") for item in manifest_items}
    
    feature_cache = BenchmarkFeatureCache(device=device)
    feature_cache.extract_fact_embeddings(facts)
    feature_cache.extract_video_features([])
    subgraphs = torch.load(feature_cache.cache_dir / "landmark_subgraphs_rel.pt", weights_only=False)
    
    # 1. Trajectory and Initial Gradients
    p1_results = run_trajectory_and_gradients(train_clips, test_clips, subgraphs, feature_cache)
    
    # 2. Placebo experiments across 3 seeds
    p2_results = run_placebo_suite(train_clips, test_clips, subgraphs, feature_cache, clip_to_video)
    
    output_data = {
        "trajectory_and_gradients": p1_results,
        "placebo_suite": p2_results
    }
    
    out_file = REPO_ROOT / "results_prop2c_dynamics_and_placebo.json"
    out_file.write_text(json.dumps(output_data, indent=2))
    print(f"\nAll Prop 2c'' results exported to: {out_file}")


if __name__ == "__main__":
    main()
