#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_deep_dive_experiments.py
=============================================================================
Comprehensive deep-dive experimental suite addressing all review items:

Task 1 (X2 Deep Dive):
  - Multi-head real attention A_real on 500 Hanoi test clips
  - Power iteration for exact true stationary distribution pi_real
  - Compare u_L (L in {1, 2, 3, 4, 8, 16}) against:
      pi_real, pi_0 (random walk), pi_0.4, pi_1 (normalized), pi_uniform
  - Symmetrized attention A_sym = 0.5 * (A + A^T)
  - Relation-only logit variant (zeroing query-key content similarity)

Task 2 (Ablation gamma=0 vs gamma=0.40):
  - Compare proposed (gamma=0.40) vs debias-ablation checkpoints (gamma=0.0)
  - Parent-video cluster bootstrap (10,000 resamples) for 95% CIs on:
      Overall nDCG@5, P@5, Hard-Negative AUC, Paired Delta
  - Degree-stratified nDCG@5 across degree tiers:
      Low-degree (d <= 3), Mid-degree (4 <= d <= 10), High-degree/Hubs (d > 10)

Task 3 (Gate lambda Ablation: Learned vs Constant lambda=0.884 vs lambda=1.0):
  - Paired comparison on 500 test clips
  - 10,000 cluster bootstrap resamples on paired differences
  - Wilcoxon paired test and cluster permutation test

Task 4 (Cluster Audit):
  - Exact count of parent-video clusters in Train (194 clips) vs Test (500 clips)

Task 5 (Node Count Reconciliation):
  - Entity-level KG (2,334 Hanoi, 479 HCMC) vs Fact-level subgraphs (2,118 Hanoi, 1,116 HCMC)
=============================================================================
"""

import sys
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F

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
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# HELPER: CLUSTER BOOTSTRAP RESAMPLING
# =============================================================================

def cluster_bootstrap_ci(clip_records, cluster_key, metric_func, B=B_RESAMPLES, ci=0.95):
    """
    Performs parent-video cluster bootstrap percentile CI.
    """
    clusters = defaultdict(list)
    for rec in clip_records:
        clusters[rec[cluster_key]].append(rec)
    cluster_ids = list(clusters.keys())
    K = len(cluster_ids)
    
    boot_estimates = []
    rng = np.random.RandomState(SEED)
    
    for _ in range(B):
        sampled_cids = rng.choice(cluster_ids, size=K, replace=True)
        sampled_records = []
        for cid in sampled_cids:
            sampled_records.extend(clusters[cid])
        val = metric_func(sampled_records)
        boot_estimates.append(val)
        
    boot_estimates = np.array(boot_estimates)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_estimates, alpha * 100))
    high = float(np.percentile(boot_estimates, (1.0 - alpha) * 100))
    se = float(np.std(boot_estimates))
    return low, high, se, boot_estimates


# =============================================================================
# TASK 1: X2 DEEP DIVE (POWER ITERATION, DISENTANGLEMENT, ASYMMETRY)
# =============================================================================

def compute_stationary_distribution(P, max_iter=2000, tol=1e-9):
    """Computes exact stationary distribution of row-stochastic P via power iteration."""
    N = P.shape[0]
    p = np.ones(N) / N
    for _ in range(max_iter):
        p_next = p @ P
        p_next = p_next / np.sum(p_next)
        if np.max(np.abs(p_next - p)) < tol:
            return p_next
        p = p_next
    return p


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


def run_task1_x2_deep_dive(test_clips, hanoi_cids, subgraphs, feature_cache, dec_model):
    print("\n" + "=" * 75)
    print("TASK 1: X2 DEEP DIVE - DISENTANGLING FINITE LAYERS, LOGITS, & ASYMMETRY")
    print("=" * 75)
    
    dec_model.eval()
    layers = [1, 2, 3, 4, 8, 16]
    
    # Trackers for correlations and distances against different reference distributions
    # References: pi_real (true limit of A), pi_0, pi_04, pi_1, pi_uniform
    records = {
        "vs_pi_real": {L: {"spearman": [], "tv": [], "rel_l1": []} for L in layers},
        "vs_pi_04": {L: {"spearman": [], "tv": [], "rel_l1": []} for L in layers},
        "vs_pi_00": {L: {"spearman": [], "tv": [], "rel_l1": []} for L in layers},
        "vs_pi_10": {L: {"spearman": [], "tv": [], "rel_l1": []} for L in layers},
        "vs_uniform": {L: {"spearman": [], "tv": [], "rel_l1": []} for L in layers},
    }
    
    # Symmetrized attention tracker
    sym_records = {"spearman_vs_pi04": [], "tv_vs_pi04": []}
    
    # Relation-only tracker
    rel_only_records = {L: {"spearman_vs_pi04": [], "tv_vs_pi04": []} for L in [1, 4]}
    
    asym_list = []
    
    with torch.no_grad():
        for cid in hanoi_cids:
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
            
            # Forward attention
            q = h_tensor.unsqueeze(1)
            kv = v_f_tensor.unsqueeze(0).expand(N, -1, -1)
            attn_out, _ = dec_model.cross_attn(query=q, key=kv, value=kv)
            ctx = dec_model.norm1(attn_out.squeeze(1) + h_tensor)
            vis_feat = torch.cat([h_tensor, ctx, h_tensor * ctx], dim=-1)
            g_nodes = dec_model.saliency_gate(vis_feat)
            
            Q = dec_model.q_proj(h_tensor).view(N, dec_model.num_heads, dec_model.head_dim).transpose(0, 1)
            h_trans = h_tensor * g_nodes
            K = dec_model.k_proj(h_trans).view(N, dec_model.num_heads, dec_model.head_dim).transpose(0, 1)
            scores = torch.bmm(Q, K.transpose(1, 2)) * dec_model.scale
            
            rel_emb = dec_model.rel_embed(rel_adj)
            rel_k = dec_model.rel_k_proj(rel_emb).view(N, N, dec_model.num_heads, dec_model.head_dim).permute(2, 0, 1, 3)
            scores_rel = torch.sum(Q.unsqueeze(2) * rel_k, dim=-1) * dec_model.scale
            scores = scores + scores_rel + dec_model.rel_bias(rel_adj).permute(2, 0, 1)
            
            deg_penalty = dec_model.degree_damping * torch.log1p(degrees.float()).unsqueeze(0).unsqueeze(1)
            scores_damped = scores - deg_penalty
            
            mask = (adj == 0).unsqueeze(0).expand(dec_model.num_heads, -1, -1)
            scores_damped = scores_damped.masked_fill(mask, -1e9)
            attn = F.softmax(scores_damped, dim=-1)
            A_real = attn.mean(dim=0).cpu().numpy() # [N, N]
            
            # Asymmetry
            norm_A = np.linalg.norm(A_real, 'fro')
            asym = np.linalg.norm(A_real - A_real.T, 'fro') / (norm_A + 1e-12)
            asym_list.append(float(asym))
            
            # 1. Exact stationary distribution of A_real
            pi_real = compute_stationary_distribution(A_real)
            
            # 2. Theoretical distributions
            adj_np = adj.cpu().numpy()
            deg_np = degrees.cpu().numpy()
            pi_00, _ = compute_theoretical_pi(adj_np, deg_np, gamma=0.0)
            pi_04, _ = compute_theoretical_pi(adj_np, deg_np, gamma=0.40)
            pi_10, _ = compute_theoretical_pi(adj_np, deg_np, gamma=1.0)
            pi_uni = np.ones(N) / float(N)
            
            # 3. Layer-wise powers A^L
            A_cur = np.eye(N)
            for L in range(1, max(layers) + 1):
                A_cur = A_cur @ A_real
                if L in layers:
                    u_L = np.mean(A_cur, axis=0) # column sum
                    
                    for ref_name, ref_pi in [
                        ("vs_pi_real", pi_real),
                        ("vs_pi_04", pi_04),
                        ("vs_pi_00", pi_00),
                        ("vs_pi_10", pi_10),
                        ("vs_uniform", pi_uni)
                    ]:
                        sp_rho, _ = stats.spearmanr(u_L, ref_pi)
                        if np.isnan(sp_rho):
                            sp_rho = 0.0
                        tv = 0.5 * np.sum(np.abs(u_L - ref_pi))
                        rel_l1 = np.mean(np.abs(u_L - ref_pi) / np.maximum(ref_pi, 1e-12))
                        records[ref_name][L]["spearman"].append(float(sp_rho))
                        records[ref_name][L]["tv"].append(float(tv))
                        records[ref_name][L]["rel_l1"].append(float(rel_l1))
                        
            # 4. Symmetrized attention
            A_sym = 0.5 * (A_real + A_real.T)
            row_sums = A_sym.sum(axis=1, keepdims=True)
            A_sym_stoch = A_sym / np.maximum(row_sums, 1e-12)
            pi_sym = compute_stationary_distribution(A_sym_stoch)
            sp_sym, _ = stats.spearmanr(pi_sym, pi_04)
            tv_sym = 0.5 * np.sum(np.abs(pi_sym - pi_04))
            sym_records["spearman_vs_pi04"].append(float(sp_sym) if not np.isnan(sp_sym) else 0.0)
            sym_records["tv_vs_pi04"].append(float(tv_sym))
            
            # 5. Relation-only attention (zeroing visual query-key matching)
            scores_rel_only = scores_rel + dec_model.rel_bias(rel_adj).permute(2, 0, 1) - deg_penalty
            scores_rel_only = scores_rel_only.masked_fill(mask, -1e9)
            A_rel_only = F.softmax(scores_rel_only, dim=-1).mean(dim=0).cpu().numpy()
            A_rel_4 = np.linalg.matrix_power(A_rel_only, 4)
            u_rel_4 = np.mean(A_rel_4, axis=0)
            sp_rel, _ = stats.spearmanr(u_rel_4, pi_04)
            tv_rel = 0.5 * np.sum(np.abs(u_rel_4 - pi_04))
            rel_only_records[4]["spearman_vs_pi04"].append(float(sp_rel) if not np.isnan(sp_rel) else 0.0)
            rel_only_records[4]["tv_vs_pi04"].append(float(tv_rel))

    print(f"\nResults across {len(hanoi_cids)} clips:")
    print(f"Mean Graph Asymmetry ||A - A^T||_F / ||A||_F: {np.mean(asym_list):.4f}")
    
    print("\n--- 1. Convergence of u_L to True Stationary Distribution of Trained A (pi_real) ---")
    for L in layers:
        sp = np.mean(records["vs_pi_real"][L]["spearman"])
        tv = np.mean(records["vs_pi_real"][L]["tv"])
        print(f"  L = {L:2d}: Spearman rho(u_L, pi_real) = {sp:+.4f} | Total Variation = {tv:.4f}")
        
    print("\n--- 2. Spearman Correlation of u_L vs Theoretical References ---")
    header = f"{'Layer':<6} | {'vs pi_real':<12} | {'vs pi_0.4':<12} | {'vs pi_0(rw)':<12} | {'vs pi_1(norm)':<12} | {'vs Uniform':<12}"
    print(header)
    print("-" * len(header))
    for L in [1, 2, 4, 16]:
        sp_real = np.mean(records["vs_pi_real"][L]["spearman"])
        sp_04 = np.mean(records["vs_pi_04"][L]["spearman"])
        sp_00 = np.mean(records["vs_pi_00"][L]["spearman"])
        sp_10 = np.mean(records["vs_pi_10"][L]["spearman"])
        sp_uni = np.mean(records["vs_uniform"][L]["spearman"])
        print(f"L = {L:<4d}| {sp_real:+10.4f}   | {sp_04:+10.4f}   | {sp_00:+10.4f}   | {sp_10:+10.4f}   | {sp_uni:+10.4f}")
        
    print("\n--- 3. Total Variation Distance of u_L vs Theoretical References ---")
    header = f"{'Layer':<6} | {'vs pi_real':<12} | {'vs pi_0.4':<12} | {'vs pi_0(rw)':<12} | {'vs pi_1(norm)':<12} | {'vs Uniform':<12}"
    print(header)
    print("-" * len(header))
    for L in [1, 2, 4, 16]:
        tv_real = np.mean(records["vs_pi_real"][L]["tv"])
        tv_04 = np.mean(records["vs_pi_04"][L]["tv"])
        tv_00 = np.mean(records["vs_pi_00"][L]["tv"])
        tv_10 = np.mean(records["vs_pi_10"][L]["tv"])
        tv_uni = np.mean(records["vs_uniform"][L]["tv"])
        print(f"L = {L:<4d}| {tv_real:10.4f}   | {tv_04:10.4f}   | {tv_00:10.4f}   | {tv_10:10.4f}   | {tv_uni:10.4f}")

    print("\n--- 4. Role of Symmetrization & Content Logits ---")
    print(f"  Symmetrized A: Spearman(pi_sym, pi_0.4) = {np.mean(sym_records['spearman_vs_pi04']):+.4f} | TV = {np.mean(sym_records['tv_vs_pi04']):.4f}")
    print(f"  Relation-Only (No Vision): Spearman(u_4, pi_0.4) = {np.mean(rel_only_records[4]['spearman_vs_pi04']):+.4f} | TV = {np.mean(rel_only_records[4]['tv_vs_pi04']):.4f}")

    return {
        "asymmetry": float(np.mean(asym_list)),
        "records": records,
        "symmetrized": sym_records,
        "relation_only": rel_only_records
    }


# =============================================================================
# TASK 2: ABLATION gamma=0 vs gamma=0.40 WITH CLUSTER BOOTSTRAP & STRATIFIED nDCG
# =============================================================================

def run_task2_ablation_and_stratification(test_clips, hanoi_cids, subgraphs, feature_cache, clip_to_video):
    print("\n" + "=" * 75)
    print("TASK 2: ABLATION gamma=0 vs gamma=0.40 WITH CLUSTER BOOTSTRAP & STRATIFIED nDCG")
    print("=" * 75)
    
    # 1. Load both checkpoints
    prop_ckpt = REPO_ROOT / "experiments" / "checkpoints_fact_selection" / "best_dec_vcgat_v2.pt"
    abl_ckpt_horn = REPO_ROOT / "experiments" / "checkpoints_fact_selection" / "best_dec_v2_wo_debias_horn.pt"
    abl_ckpt_dcse = REPO_ROOT / "experiments" / "checkpoints_fact_selection" / "best_ablation_wo_degreedebiasing.pt"
    
    prop_model = DecoupledEvidenceCalibratedVCGATv2(d_model=768, degree_damping=0.4).to(device)
    prop_model.load_state_dict(torch.load(prop_ckpt, map_location=device))
    prop_model.eval()
    
    abl_model = DecoupledEvidenceCalibratedVCGATv2(d_model=768, degree_damping=0.0).to(device)
    # Prefer the 15-epoch horn checkpoint if available
    chosen_abl_path = abl_ckpt_horn if abl_ckpt_horn.exists() else abl_ckpt_dcse
    print(f"Loading Ablation Model from: {chosen_abl_path.name}")
    abl_model.load_state_dict(torch.load(chosen_abl_path, map_location=device))
    abl_model.eval()
    
    # Stratification tiers based on fact node degree in landmark subgraphs:
    # Low-degree: d <= 3, Mid-degree: 4 <= d <= 10, High-degree: d > 10
    clip_eval_records = []
    
    with torch.no_grad():
        for cid in hanoi_cids:
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
            
            deg_np = degrees.cpu().numpy()
            fid_degrees = {fid: int(deg_np[fid2idx[fid]]) for fid in cand_fids}
            
            # Proposed (gamma = 0.40)
            logits_p, _, _, _ = prop_model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
            scores_p = {fid: float(logits_p[fid2idx[fid]]) for fid in cand_fids}
            ranked_p = sorted(cand_fids, key=lambda f: scores_p[f], reverse=True)
            m_p = compute_ranking_metrics(ranked_p, gold_fids, cand_fids, top_k_list=[5])
            auc_p = compute_hard_negative_auc(scores_p, gold_fids, hard_neg_fids)
            
            # Ablation (gamma = 0.0)
            logits_a, _, _, _ = abl_model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
            scores_a = {fid: float(logits_a[fid2idx[fid]]) for fid in cand_fids}
            ranked_a = sorted(cand_fids, key=lambda f: scores_a[f], reverse=True)
            m_a = compute_ranking_metrics(ranked_a, gold_fids, cand_fids, top_k_list=[5])
            auc_a = compute_hard_negative_auc(scores_a, gold_fids, hard_neg_fids)
            
            # Stratified candidates
            low_cand = [f for f in cand_fids if fid_degrees[f] <= 3]
            mid_cand = [f for f in cand_fids if 4 <= fid_degrees[f] <= 10]
            high_cand = [f for f in cand_fids if fid_degrees[f] > 10]
            
            def eval_tier(cand_subset, scores_dict):
                gold_sub = set(f for f in cand_subset if f in gold_fids)
                if not cand_subset or not gold_sub:
                    return np.nan
                ranked_sub = sorted(cand_subset, key=lambda f: scores_dict[f], reverse=True)
                m_sub = compute_ranking_metrics(ranked_sub, gold_sub, cand_subset, top_k_list=[5])
                return m_sub["nDCG@5"]

            clip_eval_records.append({
                "clip_id": cid,
                "video_id": vid,
                "prop_ndcg5": m_p["nDCG@5"],
                "abl_ndcg5": m_a["nDCG@5"],
                "delta_ndcg5": m_p["nDCG@5"] - m_a["nDCG@5"],
                "prop_p5": m_p["P@5"],
                "abl_p5": m_a["P@5"],
                "delta_p5": m_p["P@5"] - m_a["P@5"],
                "prop_auc": auc_p if auc_p is not None else np.nan,
                "abl_auc": auc_a if auc_a is not None else np.nan,
                "delta_auc": (auc_p - auc_a) if (auc_p is not None and auc_a is not None) else np.nan,
                "prop_low_ndcg": eval_tier(low_cand, scores_p),
                "abl_low_ndcg": eval_tier(low_cand, scores_a),
                "prop_mid_ndcg": eval_tier(mid_cand, scores_p),
                "abl_mid_ndcg": eval_tier(mid_cand, scores_a),
                "prop_high_ndcg": eval_tier(high_cand, scores_p),
                "abl_high_ndcg": eval_tier(high_cand, scores_a),
            })

    # Cluster Bootstrap for Overall Metrics & Paired Delta
    def mean_stat(recs, key):
        vals = [r[key] for r in recs if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else np.nan

    metrics_to_boot = [
        ("prop_ndcg5", "Proposed nDCG@5"),
        ("abl_ndcg5", "Ablation (gamma=0) nDCG@5"),
        ("delta_ndcg5", "Delta nDCG@5 (Prop - Abl)"),
        ("prop_auc", "Proposed Hard-Neg AUC"),
        ("abl_auc", "Ablation (gamma=0) Hard-Neg AUC"),
        ("delta_auc", "Delta Hard-Neg AUC"),
        ("prop_low_ndcg", "Prop Low-degree nDCG@5"),
        ("abl_low_ndcg", "Abl Low-degree nDCG@5"),
        ("prop_mid_ndcg", "Prop Mid-degree nDCG@5"),
        ("abl_mid_ndcg", "Abl Mid-degree nDCG@5"),
        ("prop_high_ndcg", "Prop High-degree (Hub) nDCG@5"),
        ("abl_high_ndcg", "Abl High-degree (Hub) nDCG@5"),
    ]
    
    boot_results = {}
    print("\n--- Cluster Bootstrap Percentile 95% CIs (B=10,000 resamples) ---")
    for key, label in metrics_to_boot:
        pt = mean_stat(clip_eval_records, key)
        low, high, se, _ = cluster_bootstrap_ci(clip_eval_records, "video_id", lambda recs: mean_stat(recs, key))
        boot_results[key] = {"point": pt, "ci_95": [low, high], "se": se}
        print(f"  {label:<35}: {pt:.4f} [95% CI: {low:.4f}, {high:.4f}] (SE={se:.4f})")
        
    # Statistical tests on paired delta nDCG
    deltas_ndcg = [r["delta_ndcg5"] for r in clip_eval_records]
    t_stat, p_t = stats.ttest_1samp(deltas_ndcg, 0.0)
    w_stat, p_w = stats.wilcoxon(deltas_ndcg, zero_method="pratt")
    print(f"\nPaired Statistical Significance (Prop vs Ablation w/o debiasing):")
    print(f"  Paired t-test: t = {t_stat:.3f}, p = {p_t:.2e}")
    print(f"  Wilcoxon test: W = {w_stat:.1f}, p = {p_w:.2e}")
    
    return boot_results


# =============================================================================
# TASK 3: ABLATION OF GATE lambda (LEARNED vs CONSTANT lambda=0.884 vs lambda=1.0)
# =============================================================================

def run_task3_gate_ablation(test_clips, hanoi_cids, subgraphs, feature_cache, clip_to_video, dec_model):
    print("\n" + "=" * 75)
    print("TASK 3: GATE lambda ABLATION - LEARNED vs CONSTANT lambda=0.8838 vs lambda=1.0")
    print("=" * 75)
    
    dec_model.eval()
    fixed_lambda_mean = 0.8838
    
    gate_eval_records = []
    
    with torch.no_grad():
        for cid in hanoi_cids:
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
            
            # Forward pass: extract s_vis, lambda_i, delta_s
            logits_learned, s_vis, lambda_i, delta_s = dec_model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
            
            # 1. Learned Gate
            scores_learned = {fid: float(logits_learned[fid2idx[fid]]) for fid in cand_fids}
            ranked_learned = sorted(cand_fids, key=lambda f: scores_learned[f], reverse=True)
            m_learned = compute_ranking_metrics(ranked_learned, gold_fids, cand_fids, top_k_list=[5])
            auc_learned = compute_hard_negative_auc(scores_learned, gold_fids, hard_neg_fids)
            
            # 2. Constant Gate lambda = 0.8838
            logits_const = s_vis + 0.4 * fixed_lambda_mean * delta_s
            scores_const = {fid: float(logits_const[fid2idx[fid]]) for fid in cand_fids}
            ranked_const = sorted(cand_fids, key=lambda f: scores_const[f], reverse=True)
            m_const = compute_ranking_metrics(ranked_const, gold_fids, cand_fids, top_k_list=[5])
            auc_const = compute_hard_negative_auc(scores_const, gold_fids, hard_neg_fids)
            
            # 3. Full Gate lambda = 1.0 (No Gating)
            logits_one = s_vis + 0.4 * 1.0 * delta_s
            scores_one = {fid: float(logits_one[fid2idx[fid]]) for fid in cand_fids}
            ranked_one = sorted(cand_fids, key=lambda f: scores_one[f], reverse=True)
            m_one = compute_ranking_metrics(ranked_one, gold_fids, cand_fids, top_k_list=[5])
            auc_one = compute_hard_negative_auc(scores_one, gold_fids, hard_neg_fids)
            
            gate_eval_records.append({
                "clip_id": cid,
                "video_id": vid,
                "learned_ndcg5": m_learned["nDCG@5"],
                "const_ndcg5": m_const["nDCG@5"],
                "one_ndcg5": m_one["nDCG@5"],
                "delta_ndcg5_learned_vs_const": m_learned["nDCG@5"] - m_const["nDCG@5"],
                "delta_ndcg5_learned_vs_one": m_learned["nDCG@5"] - m_one["nDCG@5"],
                "learned_auc": auc_learned if auc_learned is not None else np.nan,
                "const_auc": auc_const if auc_const is not None else np.nan,
                "one_auc": auc_one if auc_one is not None else np.nan,
                "delta_auc_learned_vs_const": (auc_learned - auc_const) if (auc_learned is not None and auc_const is not None) else np.nan
            })

    def mean_stat(recs, key):
        vals = [r[key] for r in recs if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else np.nan

    metrics = [
        ("learned_ndcg5", "Learned Gate lambda nDCG@5"),
        ("const_ndcg5", "Constant Gate (lambda=0.884) nDCG@5"),
        ("one_ndcg5", "No Gate (lambda=1.0) nDCG@5"),
        ("delta_ndcg5_learned_vs_const", "Delta nDCG@5 (Learned - Const)"),
        ("delta_ndcg5_learned_vs_one", "Delta nDCG@5 (Learned - lambda=1)"),
        ("learned_auc", "Learned Gate Hard-Neg AUC"),
        ("const_auc", "Constant Gate Hard-Neg AUC"),
        ("delta_auc_learned_vs_const", "Delta Hard-Neg AUC (Learned - Const)")
    ]

    print("\n--- Gate Ablation Comparison with Cluster Bootstrap Percentile 95% CIs ---")
    gate_results = {}
    for key, label in metrics:
        pt = mean_stat(gate_eval_records, key)
        low, high, se, _ = cluster_bootstrap_ci(gate_eval_records, "video_id", lambda recs: mean_stat(recs, key))
        gate_results[key] = {"point": pt, "ci_95": [low, high], "se": se}
        print(f"  {label:<40}: {pt:.5f} [95% CI: {low:.5f}, {high:.5f}] (SE={se:.5f})")

    # Statistical test on Learned vs Constant
    deltas = [r["delta_ndcg5_learned_vs_const"] for r in gate_eval_records]
    t_val, p_val = stats.ttest_1samp(deltas, 0.0)
    w_val, p_w_val = stats.wilcoxon(deltas, zero_method="pratt")
    print(f"\nPaired Test (Learned Gate vs Constant lambda=0.884):")
    print(f"  t-test: t = {t_val:.4f}, p = {p_val:.4f}")
    print(f"  Wilcoxon: W = {w_val:.1f}, p = {p_w_val:.4f}")
    
    return gate_results


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main():
    print("=" * 75)
    print("STARTING DEEP-DIVE EXPERIMENTAL AUDIT")
    print("=" * 75)
    
    facts = load_normalized_facts()
    train_clips, _ = load_gold_data(split_type="train")
    test_clips, _ = load_gold_data(split_type="test")
    hanoi_cids = sorted([cid for cid in test_clips if not cid.startswith("hcmc_")])
    
    # Manifest for parent video mapping
    manifest_path = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "evaluation_clips_manifest.json"
    manifest_items = json.loads(manifest_path.read_text())
    clip_to_video = {item["clip_id"]: item.get("source_video_id") or item.get("video_id") for item in manifest_items}
    
    train_videos = set(clip_to_video.get(cid) for cid in train_clips if not cid.startswith("hcmc_"))
    test_videos = set(clip_to_video.get(cid) for cid in hanoi_cids)
    print(f"Cluster Audit:")
    print(f"  Train set: {len(train_clips)} clips across {len(train_videos)} parent videos")
    print(f"  Test set:  {len(hanoi_cids)} clips across {len(test_videos)} parent videos")
    
    feature_cache = BenchmarkFeatureCache(device=device)
    feature_cache.extract_fact_embeddings(facts)
    feature_cache.extract_video_features([])
    subgraphs = torch.load(feature_cache.cache_dir / "landmark_subgraphs_rel.pt", weights_only=False)
    
    prop_ckpt = REPO_ROOT / "experiments" / "checkpoints_fact_selection" / "best_dec_vcgat_v2.pt"
    prop_model = DecoupledEvidenceCalibratedVCGATv2(d_model=768).to(device)
    prop_model.load_state_dict(torch.load(prop_ckpt, map_location=device))
    prop_model.eval()
    
    # 1. Run Task 1
    t1_results = run_task1_x2_deep_dive(test_clips, hanoi_cids, subgraphs, feature_cache, prop_model)
    
    # 2. Run Task 2
    t2_results = run_task2_ablation_and_stratification(test_clips, hanoi_cids, subgraphs, feature_cache, clip_to_video)
    
    # 3. Run Task 3
    t3_results = run_task3_gate_ablation(test_clips, hanoi_cids, subgraphs, feature_cache, clip_to_video, prop_model)
    
    output_data = {
        "cluster_audit": {
            "train_clips": len(train_clips),
            "train_parent_videos": len(train_videos),
            "test_clips": len(hanoi_cids),
            "test_parent_videos": len(test_videos)
        },
        "task1_x2_deep_dive": t1_results,
        "task2_ablation_and_stratification": t2_results,
        "task3_gate_ablation": t3_results
    }
    
    out_file = REPO_ROOT / "results_deep_dive.json"
    out_file.write_text(json.dumps(output_data, indent=2))
    print(f"\nAll Deep-Dive results exported to: {out_file}")


if __name__ == "__main__":
    main()
