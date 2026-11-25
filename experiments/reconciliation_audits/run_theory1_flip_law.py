#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_theory1_flip_law.py
=============================================================================
Implementation of Theory 1: First-Order Perturbation Law of Pairwise Flips

1. Validation Estimation (92 Hanoi Val Clips):
   - For all pairs (i, j) with y_i > y_j (gold vs negative):
     g = s_vis,i - s_vis,j
     d_0 = lambda_i * Delta_i - lambda_j * Delta_j
   - Density estimation for f_+(0) and f_-(0) using KDE and bin boundary.
   - Expectation of one-sided perturbations: E[d_0^+] and E[d_0^-].
   - Direct independence test: corr(g, d_0) and corr(|g|, |d_0|).

2. First-Order & Full Integral Predictions across 8 alpha values:
   - Linear approximation:
       P_pred(F) = alpha * [f_+(0) * E[d_0^-] + f_-(0) * E[d_0^+]]
       a_pred = f_-(0) * E[d_0^+] / [f_+(0) * E[d_0^-] + f_-(0) * E[d_0^+]]
   - Full integral formula (accounting for saturation under d_0 perp g):
       P_pred_full(F) = int f_+(x) P(alpha d_0 < -x) dx + int f_-(x) P(alpha d_0 > x) dx
       a_pred_full = int f_-(x) P(alpha d_0 > x) dx / P_pred_full(F)

3. Empirical Evaluation on Hanoi-500 Test Clips (60 Parent Videos):
   - Empirical P_test(F) and a_test(alpha) with parent-video cluster bootstrap 95% CIs.
   - Comparison and relative prediction error.
   - Rank-5 boundary pair analysis (explaining why nDCG@5 stays flat).
=============================================================================
"""

import sys
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
import scipy.stats as stats
from scipy.stats import gaussian_kde
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.fact_selection_benchmark.train_dec_vcgat_v2 import DecoupledEvidenceCalibratedVCGATv2, build_clip_graph
from experiments.fact_selection_benchmark.dataset_loader import (
    load_gold_data,
    load_normalized_facts,
    compute_ranking_metrics,
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


def fast_cluster_bootstrap_stats(video_ids, g_arr, d0_arr, alpha, B=B_RESAMPLES, ci=0.95):
    d_arr = alpha * d0_arr
    g_comb = g_arr + d_arr
    
    harmful = (g_arr > 0) & (g_comb < 0)
    correct = (g_arr < 0) & (g_comb > 0)
    flips = harmful | correct
    
    unique_vids, inverse_vids = np.unique(video_ids, return_inverse=True)
    K = len(unique_vids)
    
    N_v = np.bincount(inverse_vids, minlength=K)
    F_v = np.bincount(inverse_vids, weights=flips.astype(float), minlength=K)
    C_v = np.bincount(inverse_vids, weights=correct.astype(float), minlength=K)
    
    tot_pairs = np.sum(N_v)
    tot_flips = np.sum(F_v)
    tot_correct = np.sum(C_v)
    pf_point = float(tot_flips / tot_pairs) if tot_pairs > 0 else 0.0
    a_point = float(tot_correct / tot_flips) if tot_flips > 0 else 0.50
    
    rng = np.random.RandomState(SEED)
    sample_indices = rng.choice(K, size=(B, K), replace=True)
    
    boot_F = np.take(F_v, sample_indices).sum(axis=1)
    boot_N = np.take(N_v, sample_indices).sum(axis=1)
    boot_C = np.take(C_v, sample_indices).sum(axis=1)
    
    boot_pf = np.where(boot_N > 0, boot_F / boot_N, 0.0)
    boot_a = np.where(boot_F > 0, boot_C / boot_F, 0.50)
    
    alpha_ci = (1.0 - ci) / 2.0
    low_pf = float(np.percentile(boot_pf, alpha_ci * 100))
    high_pf = float(np.percentile(boot_pf, (1.0 - alpha_ci) * 100))
    low_a = float(np.percentile(boot_a, alpha_ci * 100))
    high_a = float(np.percentile(boot_a, (1.0 - alpha_ci) * 100))
    
    return pf_point, [low_pf, high_pf], a_point, [low_a, high_a]


def extract_clip_pair_data(clips_dict, cids, subgraphs, feature_cache, model, clip_to_video=None):
    model.eval()
    all_pairs = []
    clip_data_list = []
    
    with torch.no_grad():
        for cid in cids:
            recs = clips_dict[cid]
            lid = recs[0].get("landmark_id", "unknown")
            vid = clip_to_video.get(cid, "unknown") if clip_to_video else "val"
            g = subgraphs.get(lid)
            v_frames = feature_cache.video_frames.get(cid)
            v_pooled = feature_cache.video_pooled.get(cid)
            if g is None or v_frames is None:
                continue
                
            cand_fids = [r["fact_id"] for r in recs]
            gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            if not gold_fids or len(gold_fids) == len(cand_fids):
                continue
                
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device) if v_pooled is not None else None
            
            h_tensor, adj, rel_adj, degrees, all_fids, fid2idx = build_clip_graph(
                g, cand_fids, feature_cache.fact_embeddings, device
            )
            
            _, s_vis, lambda_i, delta_s = model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
            
            cand_idx = [fid2idx[f] for f in cand_fids]
            s_vis_np = s_vis[cand_idx].cpu().numpy()
            lam_np = lambda_i[cand_idx].cpu().numpy()
            delta_np = delta_s[cand_idx].cpu().numpy()
            deg_np = degrees[cand_idx].cpu().numpy()
            
            # Rank candidates by visual score to identify rank-5 boundary pairs
            vis_order = np.argsort(s_vis_np)[::-1]
            vis_rank = {cand_fids[idx]: r + 1 for r, idx in enumerate(vis_order)}
            
            K = len(cand_fids)
            clip_pairs = []
            for i in range(K):
                if cand_fids[i] not in gold_fids:
                    continue
                for j in range(K):
                    if cand_fids[j] in gold_fids:
                        continue
                    g_val = float(s_vis_np[i] - s_vis_np[j])
                    d0_val = float(lam_np[i] * delta_np[i] - lam_np[j] * delta_np[j])
                    
                    pair_rec = {
                        "clip_id": cid,
                        "video_id": vid,
                        "landmark_id": lid,
                        "is_seen": (lid in SEEN_LANDMARKS),
                        "gold_fid": cand_fids[i],
                        "neg_fid": cand_fids[j],
                        "gold_deg": float(deg_np[i]),
                        "neg_deg": float(deg_np[j]),
                        "g": g_val,
                        "d0": d0_val,
                        "is_pos_g": (g_val > 0),
                        "abs_g": abs(g_val),
                        "rank_i": vis_rank[cand_fids[i]],
                        "rank_j": vis_rank[cand_fids[j]],
                        "is_rank5_boundary": (vis_rank[cand_fids[i]] in [4, 5, 6] or vis_rank[cand_fids[j]] in [4, 5, 6])
                    }
                    all_pairs.append(pair_rec)
                    clip_pairs.append(pair_rec)
                    
            clip_data_list.append({
                "clip_id": cid,
                "video_id": vid,
                "pairs": clip_pairs
            })
            
    return all_pairs, clip_data_list


def estimate_boundary_densities(val_pairs):
    """Estimates f_+(0), f_-(0), and expectations of d0."""
    g_pos = np.array([p["g"] for p in val_pairs if p["g"] > 0])
    g_neg = np.array([-p["g"] for p in val_pairs if p["g"] < 0]) # positive magnitudes of negative gaps
    d0_vals = np.array([p["d0"] for p in val_pairs])
    
    # 1. Check Independence
    g_all = np.array([p["g"] for p in val_pairs])
    corr_g_d0, p_corr1 = stats.pearsonr(g_all, d0_vals)
    corr_abs, p_corr2 = stats.pearsonr(np.abs(g_all), np.abs(d0_vals))
    sp_corr, p_sp = stats.spearmanr(g_all, d0_vals)
    
    # 2. Density Estimation near 0
    # Method A: Small-bin empirical density f(0) ~ P(0 <= X <= eps) / eps
    eps = 0.10
    f_plus_bin = float(np.mean((g_pos >= 0) & (g_pos <= eps)) / eps)
    f_minus_bin = float(np.mean((g_neg >= 0) & (g_neg <= eps)) / eps)
    
    # Method B: Gaussian KDE
    kde_pos = gaussian_kde(g_pos, bw_method='silverman')
    kde_neg = gaussian_kde(g_neg, bw_method='silverman')
    f_plus_kde = float(kde_pos(0.0)[0])
    f_minus_kde = float(kde_neg(0.0)[0])
    
    # Combine / use robust estimate
    f_plus_0 = 0.5 * (f_plus_bin + f_plus_kde)
    f_minus_0 = 0.5 * (f_minus_bin + f_minus_kde)
    
    # 3. Expectations of d0
    d0_pos = np.maximum(d0_vals, 0.0)
    d0_neg = np.maximum(-d0_vals, 0.0)
    e_d0_pos = float(np.mean(d0_pos))
    e_d0_neg = float(np.mean(d0_neg))
    
    # Branch-specific expectations
    d0_when_g_pos = np.array([p["d0"] for p in val_pairs if p["g"] > 0])
    d0_when_g_neg = np.array([p["d0"] for p in val_pairs if p["g"] < 0])
    e_d0_neg_given_g_pos = float(np.mean(np.maximum(-d0_when_g_pos, 0.0)))
    e_d0_pos_given_g_neg = float(np.mean(np.maximum(d0_when_g_neg, 0.0)))
    
    return {
        "n_val_pairs": len(val_pairs),
        "f_plus_0": f_plus_0,
        "f_plus_bin": f_plus_bin,
        "f_plus_kde": f_plus_kde,
        "f_minus_0": f_minus_0,
        "f_minus_bin": f_minus_bin,
        "f_minus_kde": f_minus_kde,
        "e_d0_pos": e_d0_pos,
        "e_d0_neg": e_d0_neg,
        "e_d0_neg_given_g_pos": e_d0_neg_given_g_pos,
        "e_d0_pos_given_g_neg": e_d0_pos_given_g_neg,
        "independence_test": {
            "pearson_corr_g_d0": float(corr_g_d0),
            "p_val_corr1": float(p_corr1),
            "pearson_corr_abs": float(corr_abs),
            "spearman_corr": float(sp_corr),
            "p_val_spearman": float(p_sp)
        },
        "g_pos_sample": g_pos,
        "g_neg_sample": g_neg,
        "d0_vals": d0_vals
    }


def main():
    print("=" * 75)
    print("STARTING THEORY 1 EXPERIMENT: FIRST-ORDER FLIP PERTURBATION LAW")
    print("=" * 75)
    
    facts = load_normalized_facts()
    val_clips, _ = load_gold_data(split_type="val")
    test_clips, _ = load_gold_data(split_type="test")
    hanoi_val_cids = sorted([cid for cid in val_clips if not cid.startswith("hcmc_")])
    hanoi_test_cids = sorted([cid for cid in test_clips if not cid.startswith("hcmc_")])
    
    manifest_path = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "evaluation_clips_manifest.json"
    manifest_items = json.loads(manifest_path.read_text())
    clip_to_video = {item["clip_id"]: item.get("source_video_id") or item.get("video_id") for item in manifest_items}
    
    feature_cache = BenchmarkFeatureCache(device=device)
    feature_cache.extract_fact_embeddings(facts)
    feature_cache.extract_video_features([])
    subgraphs = torch.load(feature_cache.cache_dir / "landmark_subgraphs_rel.pt", weights_only=False)
    
    ckpt_path = REPO_ROOT / "experiments" / "checkpoints_fact_selection" / "best_dec_vcgat_v2.pt"
    dec_model = DecoupledEvidenceCalibratedVCGATv2(d_model=768).to(device)
    dec_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    dec_model.eval()
    
    # -------------------------------------------------------------------------
    # STEP 1: VALIDATION ESTIMATION
    # -------------------------------------------------------------------------
    print("\n--- Step 1: Estimating Boundary Densities on Validation Set ---")
    val_pairs, _ = extract_clip_pair_data(val_clips, hanoi_val_cids, subgraphs, feature_cache, dec_model)
    est = estimate_boundary_densities(val_pairs)
    
    print(f"Validation Pairs Evaluated: {est['n_val_pairs']:,}")
    print(f"Independence Check: corr(g, d0) = {est['independence_test']['pearson_corr_g_d0']:+.4f} (p = {est['independence_test']['p_val_corr1']:.3f})")
    print(f"  Spearman rho(g, d0) = {est['independence_test']['spearman_corr']:+.4f} (p = {est['independence_test']['p_val_spearman']:.3f})")
    print(f"  corr(|g|, |d0|)     = {est['independence_test']['pearson_corr_abs']:+.4f}")
    print(f"Boundary Densities:")
    print(f"  f_+(0) (Density of positive visual margins near 0): {est['f_plus_0']:.4f}")
    print(f"  f_-(0) (Density of negative visual margins near 0): {est['f_minus_0']:.4f}")
    print(f"Perturbation Expectations:")
    print(f"  E[d0^+] (Mean positive graph boost)               : {est['e_d0_pos']:.4f}")
    print(f"  E[d0^-] (Mean negative graph penalty)             : {est['e_d0_neg']:.4f}")
    
    f_plus = est["f_plus_0"]
    f_minus = est["f_minus_0"]
    e_pos = est["e_d0_pos"]
    e_neg = est["e_d0_neg"]
    
    # Predicted linear slope: C_flip = f_+(0) * E[d_0^-] + f_-(0) * E[d_0^+]
    c_flip_pred = f_plus * e_neg + f_minus * e_pos
    a_pred_linear = (f_minus * e_pos) / c_flip_pred
    print(f"\nFirst-Order Predictions (as alpha -> 0):")
    print(f"  Predicted linear rate P(F) / alpha = {c_flip_pred:.4f}")
    print(f"  Predicted asymptotic flip accuracy a_pred = {a_pred_linear:.4f}")

    # -------------------------------------------------------------------------
    # STEP 2: TEST PREDICTIONS VS EMPIRICAL MEASUREMENTS (500 TEST CLIPS)
    # -------------------------------------------------------------------------
    print("\n--- Step 2: Testing Predictions on 500 Hanoi Test Clips ---")
    test_pairs, test_clip_data = extract_clip_pair_data(test_clips, hanoi_test_cids, subgraphs, feature_cache, dec_model, clip_to_video)
    print(f"Total Test Candidate Pairs: {len(test_pairs):,} across 60 parent videos")
    
    # Pre-extract numpy arrays for ultra-fast vectorized operations
    test_vids = np.array([p["video_id"] for p in test_pairs])
    test_g = np.array([p["g"] for p in test_pairs])
    test_d0 = np.array([p["d0"] for p in test_pairs])
    
    g_pos_val = est["g_pos_sample"]
    g_neg_val = est["g_neg_sample"]
    d0_val = est["d0_vals"]
    
    # Precompute validation KDEs once outside the loop
    kde_p = gaussian_kde(g_pos_val, bw_method='silverman')
    kde_n = gaussian_kde(g_neg_val, bw_method='silverman')
    x_grid = np.linspace(0.001, 2.5, 500)
    dx = x_grid[1] - x_grid[0]
    f_p_grid = kde_p(x_grid)
    f_n_grid = kde_n(x_grid)
    
    alpha_vals = [0.05, 0.10, 0.20, 0.40, 0.80, 1.00, 1.50, 2.00]
    comparison_table = []
    
    for alpha in alpha_vals:
        # 1. Theoretical Predictions from Validation Parameters
        p_pred_lin = alpha * c_flip_pred
        a_pred_lin = a_pred_linear
        
        # (b) Full integral under independence using validation empirical distributions
        d_scaled = alpha * d0_val
        p_d_neg = np.array([np.mean(d_scaled < -x) for x in x_grid])
        p_d_pos = np.array([np.mean(d_scaled > x) for x in x_grid])
        
        int_pos = float(np.sum(f_p_grid * p_d_neg) * dx)
        int_neg = float(np.sum(f_n_grid * p_d_pos) * dx)
        p_pred_full = int_pos + int_neg
        a_pred_full = int_neg / p_pred_full if p_pred_full > 0 else 0.50
        
        # 2. Ultra-Fast Vectorized Empirical Measurement on Test Clips with Cluster Bootstrap
        p_emp_test, ci_pf, a_emp_test, ci_a = fast_cluster_bootstrap_stats(test_vids, test_g, test_d0, alpha)
        low_pf, high_pf = ci_pf
        low_a, high_a = ci_a
        
        # Prediction Relative Errors
        rel_err_p = abs(p_pred_full - p_emp_test) / p_emp_test if p_emp_test > 0 else 0.0
        rel_err_a = abs(a_pred_full - a_emp_test) / a_emp_test if a_emp_test > 0 else 0.0
        
        row = {
            "alpha": alpha,
            "p_pred_linear": p_pred_lin,
            "p_pred_full": p_pred_full,
            "p_emp_test": p_emp_test,
            "p_emp_test_ci": [low_pf, high_pf],
            "rel_err_p": rel_err_p,
            "a_pred_linear": a_pred_lin,
            "a_pred_full": a_pred_full,
            "a_emp_test": a_emp_test,
            "a_emp_test_ci": [low_a, high_a],
            "rel_err_a": rel_err_a
        }
        comparison_table.append(row)
        print(f"alpha={alpha:.2f} | P(F) Pred={p_pred_full*100:4.2f}% vs Emp={p_emp_test*100:4.2f}% [{low_pf*100:4.2f}%, {high_pf*100:4.2f}%] (Err={rel_err_p*100:4.1f}%) | "
              f"a Pred={a_pred_full:.3f} vs Emp={a_emp_test:.3f} [{low_a:.3f}, {high_a:.3f}] (Err={rel_err_a*100:4.1f}%)", flush=True)

    # -------------------------------------------------------------------------
    # STEP 3: EXPLAINING FLAT nDCG@5 VIA RANK-5 BOUNDARY PAIRS
    # -------------------------------------------------------------------------
    print("\n--- Step 3: Rank-5 Boundary Pair Analysis (Why nDCG@5 is Flat) ---")
    alpha_04 = 0.40
    
    total_test_pairs = len(test_pairs)
    rank5_pairs = [p for p in test_pairs if p["is_rank5_boundary"]]
    non_rank5_pairs = [p for p in test_pairs if not p["is_rank5_boundary"]]
    
    def analyze_subpop(sub_pairs, label):
        if not sub_pairs:
            return {"pairs": 0, "flips": 0, "correct": 0, "harmful": 0, "accuracy": 0.50, "p_flip": 0.0}
        sub_g = np.array([p["g"] for p in sub_pairs])
        sub_d0 = np.array([p["d0"] for p in sub_pairs])
        sub_d = alpha_04 * sub_d0
        sub_comb = sub_g + sub_d
        correct = int(np.sum((sub_g < 0) & (sub_comb > 0)))
        harmful = int(np.sum((sub_g > 0) & (sub_comb < 0)))
        flips = correct + harmful
        p_f = flips / float(len(sub_pairs))
        acc = correct / float(flips) if flips > 0 else 0.50
        print(f"  {label:<30}: Pairs={len(sub_pairs):6d} ({len(sub_pairs)/total_test_pairs*100:4.1f}%) | "
              f"Flips={flips:4d} | Correct={correct:4d} | Harmful={harmful:4d} | a = {acc:.4f} | P(F) = {p_f*100:.2f}%", flush=True)
        return {"pairs": len(sub_pairs), "flips": flips, "correct": correct, "harmful": harmful, "accuracy": acc, "p_flip": p_f}

    res_all = analyze_subpop(test_pairs, "All Pairs")
    res_bnd = analyze_subpop(rank5_pairs, "Rank-5 Boundary Pairs")
    res_non = analyze_subpop(non_rank5_pairs, "Internal (Non-Boundary) Pairs")

    # -------------------------------------------------------------------------
    # STEP 4: STRATIFIED A PRIORI PREDICTIONS (SEEN/UNSEEN & DEGREE TIERS)
    # -------------------------------------------------------------------------
    print("\n--- Step 4: Stratified Validation-to-Test Predictions ---")
    strata = [
        ("Seen Landmarks", lambda p: p["is_seen"]),
        ("Unseen Landmarks", lambda p: not p["is_seen"]),
        ("High-degree Hubs (d > 10)", lambda p: p["gold_deg"] > 10),
        ("Mid-degree Facts (4 <= d <= 10)", lambda p: 4 <= p["gold_deg"] <= 10),
        ("Low-degree Leaves (d <= 3)", lambda p: p["gold_deg"] <= 3),
    ]

    stratified_results = []
    for stratum_name, filter_fn in strata:
        val_sub = [p for p in val_pairs if filter_fn(p)]
        test_sub = [p for p in test_pairs if filter_fn(p)]
        
        if len(val_sub) < 50 or len(test_sub) < 50:
            print(f"Skipping {stratum_name} due to insufficient samples.")
            continue
            
        est_sub = estimate_boundary_densities(val_sub)
        f_p_sub = est_sub["f_plus_0"]
        f_m_sub = est_sub["f_minus_0"]
        e_p_sub = est_sub["e_d0_pos"]
        e_n_sub = est_sub["e_d0_neg"]
        
        c_flip_sub = f_p_sub * e_n_sub + f_m_sub * e_p_sub
        a_pred_sub = (f_m_sub * e_p_sub) / c_flip_sub if c_flip_sub > 0 else 0.50
        p_pred_sub = alpha_04 * c_flip_sub
        
        # Empirical test measurements at alpha=0.40 using fast cluster bootstrap
        sub_vids = np.array([p["video_id"] for p in test_sub])
        sub_g = np.array([p["g"] for p in test_sub])
        sub_d0 = np.array([p["d0"] for p in test_sub])
        p_emp_sub, ci_p_sub, a_emp_sub, ci_a_sub = fast_cluster_bootstrap_stats(sub_vids, sub_g, sub_d0, alpha_04)
        low_p_sub, high_p_sub = ci_p_sub
        low_a_sub, high_a_sub = ci_a_sub
        
        rel_err_a = abs(a_pred_sub - a_emp_sub) / a_emp_sub if a_emp_sub > 0 else 0.0
        rel_err_p = abs(p_pred_sub - p_emp_sub) / p_emp_sub if p_emp_sub > 0 else 0.0
        
        row_strat = {
            "stratum": stratum_name,
            "val_pairs": len(val_sub),
            "test_pairs": len(test_sub),
            "f_plus_0": f_p_sub,
            "f_minus_0": f_m_sub,
            "e_d0_pos": e_p_sub,
            "e_d0_neg": e_n_sub,
            "p_pred": p_pred_sub,
            "p_emp": p_emp_sub,
            "p_emp_ci": [low_p_sub, high_p_sub],
            "rel_err_p": rel_err_p,
            "a_pred": a_pred_sub,
            "a_emp": a_emp_sub,
            "a_emp_ci": [low_a_sub, high_a_sub],
            "rel_err_a": rel_err_a
        }
        stratified_results.append(row_strat)
        print(f"  {stratum_name:<30}: a_pred={a_pred_sub:.3f} vs a_emp={a_emp_sub:.3f} [{low_a_sub:.3f}, {high_a_sub:.3f}] (Err={rel_err_a*100:4.1f}%) | "
              f"P(F) pred={p_pred_sub*100:4.2f}% vs emp={p_emp_sub*100:4.2f}% (Err={rel_err_p*100:4.1f}%)")

    # Export results
    output_data = {
        "validation_parameters": {
            "f_plus_0": est["f_plus_0"],
            "f_minus_0": est["f_minus_0"],
            "e_d0_pos": est["e_d0_pos"],
            "e_d0_neg": est["e_d0_neg"],
            "independence_test": est["independence_test"]
        },
        "comparison_table": comparison_table,
        "boundary_analysis": {
            "all": res_all,
            "boundary_rank5": res_bnd,
            "non_boundary": res_non
        },
        "stratified_analysis": stratified_results
    }
    
    out_file = REPO_ROOT / "results_theory1_flip_law.json"
    out_file.write_text(json.dumps(output_data, indent=2))
    print(f"\nAll Theory 1 results exported to: {out_file}")


if __name__ == "__main__":
    main()
