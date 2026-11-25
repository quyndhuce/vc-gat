#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_prop2b_flip_decomposition.py
=============================================================================
Rigorous empirical implementation of Proposition 2b'' (Exact Pairwise Flip Decomposition):

For every candidate pair (i, j) with y_i > y_j (gold fact vs negative fact):
  gap_vis = s_vis,i - s_vis,j
  gap_comb = s_i - s_j = gap_vis + (tilde_Delta_i - tilde_Delta_j)
  where tilde_Delta_i = alpha * lambda_i * delta_s,i

Flip event F: sign(gap_comb) != sign(gap_vis)
Correct flip: gap_vis <= 0 and gap_comb > 0  (graph rescues visual error)
Harmful flip: gap_vis > 0  and gap_comb <= 0 (graph corrupts visual correct order)

Flip accuracy: a = P(correct flip | F)
Net change in pairwise ranking: Delta R_pair = P(F) * (1 - 2a)
Theoretical envelope: P(F) <= P(|gap_vis| <= eps_i + eps_j) <= P(|gap_vis| <= 2 * alpha * delta_max * bar_lambda)

Experiments:
  1. Alpha sweep in {0.05, 0.10, 0.20, 0.40, 0.80, 1.00, 1.50, 2.00}:
     Measure empirical P(F) vs Theoretical Upper Bound P(|gap_vis| <= eps_i + eps_j).
     Check bound tightness: P(F) / Bound.
  2. Measure flip accuracy 'a' with cluster bootstrap 95% CIs across 60 parent videos.
     Hypothesis test: H0: a = 0.50 (t-test, Wilcoxon).
  3. Stratification of a:
     (a) By visual margin |gap_vis|: [0, 0.05), [0.05, 0.10), [0.10, 0.20), [0.20, 0.40), [0.40, +inf)
     (b) By seen vs unseen landmarks
     (c) By fact node degrees in KG (low vs high degree)
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


def cluster_bootstrap_ci(clip_records, cluster_key, metric_func, B=B_RESAMPLES, ci=0.95):
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
    return low, high, se


def main():
    print("=" * 75)
    print("STARTING EXPERIMENT: PROPOSITION 2b'' (EXACT PAIRWISE FLIP DECOMPOSITION)")
    print("=" * 75)
    
    facts = load_normalized_facts()
    test_clips, _ = load_gold_data(split_type="test")
    hanoi_cids = sorted([cid for cid in test_clips if not cid.startswith("hcmc_")])
    
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
    
    # Extract all candidate pairs across all 500 clips
    # For each clip, extract candidate predictions
    clip_predictions = []
    
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
            
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device) if v_pooled is not None else None
            
            h_tensor, adj, rel_adj, degrees, all_fids, fid2idx = build_clip_graph(
                g, cand_fids, feature_cache.fact_embeddings, device
            )
            
            _, s_vis, lambda_i, delta_s = dec_model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
            
            cand_indices = [fid2idx[f] for f in cand_fids]
            s_vis_np = s_vis[cand_indices].cpu().numpy()
            lam_np = lambda_i[cand_indices].cpu().numpy()
            delta_np = delta_s[cand_indices].cpu().numpy()
            deg_np = degrees[cand_indices].cpu().numpy()
            
            clip_predictions.append({
                "clip_id": cid,
                "video_id": vid,
                "landmark_id": lid,
                "is_seen": (lid in SEEN_LANDMARKS),
                "cand_fids": cand_fids,
                "gold_fids": gold_fids,
                "s_vis": s_vis_np,
                "lambda": lam_np,
                "delta_s": delta_np,
                "degrees": deg_np
            })

    print(f"Loaded evaluation data for {len(clip_predictions)} clips.")
    
    # -------------------------------------------------------------------------
    # PART 1: ALPHA SWEEP - P(F) VS THEORETICAL BOUND
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("PART 1: ALPHA SWEEP - EMPIRICAL FLIP PROBABILITY P(F) VS BOUND")
    print("=" * 75)
    
    alpha_vals = [0.05, 0.10, 0.20, 0.40, 0.80, 1.00, 1.50, 2.00]
    sweep_results = []
    
    for alpha in alpha_vals:
        total_pairs = 0
        total_flips = 0
        correct_flips = 0
        harmful_flips = 0
        bound_violations = 0
        bound_satisfied_pairs = 0 # pairs satisfying |gap_vis| <= eps_i + eps_j
        
        for cdata in clip_predictions:
            s_vis = cdata["s_vis"]
            lam = cdata["lambda"]
            delta = cdata["delta_s"]
            gold = cdata["gold_fids"]
            fids = cdata["cand_fids"]
            
            tilde_delta = alpha * lam * delta # [K]
            eps = alpha * lam * np.abs(delta) # [K]
            
            K = len(fids)
            for i in range(K):
                if fids[i] not in gold:
                    continue
                for j in range(K):
                    if fids[j] in gold:
                        continue
                    # Pair (i, j) with y_i = 1 > y_j = 0
                    total_pairs += 1
                    gap_vis = s_vis[i] - s_vis[j]
                    gap_comb = gap_vis + (tilde_delta[i] - tilde_delta[j])
                    
                    is_flip = (np.sign(gap_comb) != np.sign(gap_vis)) and (gap_vis != 0)
                    
                    # Envelope check: is flip possible only when |gap_vis| <= eps_i + eps_j?
                    max_perturbation = eps[i] + eps[j]
                    if abs(gap_vis) <= max_perturbation:
                        bound_satisfied_pairs += 1
                    elif is_flip:
                        bound_violations += 1
                        
                    if is_flip:
                        total_flips += 1
                        if gap_vis < 0 and gap_comb > 0:
                            correct_flips += 1
                        elif gap_vis > 0 and gap_comb < 0:
                            harmful_flips += 1

        p_flip = total_flips / float(total_pairs) if total_pairs > 0 else 0.0
        p_bound = bound_satisfied_pairs / float(total_pairs) if total_pairs > 0 else 0.0
        a = correct_flips / float(total_flips) if total_flips > 0 else 0.50
        delta_r = p_flip * (1.0 - 2.0 * a)
        
        rec = {
            "alpha": alpha,
            "alpha_delta_max": alpha * 0.80,
            "total_pairs": total_pairs,
            "total_flips": total_flips,
            "p_flip": p_flip,
            "p_bound": p_bound,
            "tightness": p_flip / p_bound if p_bound > 0 else 0.0,
            "correct_flips": correct_flips,
            "harmful_flips": harmful_flips,
            "flip_accuracy_a": a,
            "delta_r_pair": delta_r,
            "bound_violations": bound_violations
        }
        sweep_results.append(rec)
        print(f"alpha={alpha:.2f} (scale={alpha*0.80:.3f}) | P(F): {p_flip*100:5.2f}% | Bound: {p_bound*100:5.2f}% | "
              f"Tightness: {rec['tightness']*100:4.1f}% | a: {a:.4f} | Correct: {correct_flips:4d} | Harmful: {harmful_flips:4d} | "
              f"Delta R_pair: {delta_r:+.6f} | Violations: {bound_violations}")

    # -------------------------------------------------------------------------
    # PART 2: AT DEFAULT ALPHA=0.40 - DETAILED CI AND HYPOTHESIS TEST H0: a = 0.5
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("PART 2: FLIP ACCURACY a AT DEFAULT alpha=0.40 (CLUSTER BOOTSTRAP 95% CI)")
    print("=" * 75)
    
    alpha_default = 0.40
    per_clip_stats = []
    
    for cdata in clip_predictions:
        s_vis = cdata["s_vis"]
        lam = cdata["lambda"]
        delta = cdata["delta_s"]
        gold = cdata["gold_fids"]
        fids = cdata["cand_fids"]
        
        tilde_delta = alpha_default * lam * delta
        
        c_pairs = 0
        c_flips = 0
        c_correct = 0
        c_harmful = 0
        
        K = len(fids)
        for i in range(K):
            if fids[i] not in gold:
                continue
            for j in range(K):
                if fids[j] in gold:
                    continue
                c_pairs += 1
                gap_vis = s_vis[i] - s_vis[j]
                gap_comb = gap_vis + (tilde_delta[i] - tilde_delta[j])
                if np.sign(gap_comb) != np.sign(gap_vis) and (gap_vis != 0):
                    c_flips += 1
                    if gap_vis < 0 and gap_comb > 0:
                        c_correct += 1
                    elif gap_vis > 0 and gap_comb < 0:
                        c_harmful += 1
                        
        per_clip_stats.append({
            "clip_id": cdata["clip_id"],
            "video_id": cdata["video_id"],
            "landmark_id": cdata["landmark_id"],
            "is_seen": cdata["is_seen"],
            "pairs": c_pairs,
            "flips": c_flips,
            "correct": c_correct,
            "harmful": c_harmful,
            "a": (c_correct / c_flips) if c_flips > 0 else np.nan,
            "delta_acc": ((c_correct - c_harmful) / c_pairs) if c_pairs > 0 else 0.0
        })

    def calc_global_a(recs):
        tot_c = sum(r["correct"] for r in recs)
        tot_f = sum(r["flips"] for r in recs)
        return (tot_c / float(tot_f)) if tot_f > 0 else 0.50

    def calc_global_p_flip(recs):
        tot_f = sum(r["flips"] for r in recs)
        tot_p = sum(r["pairs"] for r in recs)
        return (tot_f / float(tot_p)) if tot_p > 0 else 0.0

    def calc_global_delta_r(recs):
        tot_c = sum(r["correct"] for r in recs)
        tot_h = sum(r["harmful"] for r in recs)
        tot_p = sum(r["pairs"] for r in recs)
        return ((tot_h - tot_c) / float(tot_p)) if tot_p > 0 else 0.0

    pt_a = calc_global_a(per_clip_stats)
    low_a, high_a, se_a = cluster_bootstrap_ci(per_clip_stats, "video_id", calc_global_a)
    
    pt_pf = calc_global_p_flip(per_clip_stats)
    low_pf, high_pf, se_pf = cluster_bootstrap_ci(per_clip_stats, "video_id", calc_global_p_flip)
    
    pt_dr = calc_global_delta_r(per_clip_stats)
    low_dr, high_dr, se_dr = cluster_bootstrap_ci(per_clip_stats, "video_id", calc_global_delta_r)
    
    # Hypothesis test H0: a = 0.50
    # On clips with at least one flip:
    clip_a_vals = [r["a"] for r in per_clip_stats if not np.isnan(r["a"])]
    t_stat, p_val_t = stats.ttest_1samp(clip_a_vals, 0.50)
    w_stat, p_val_w = stats.wilcoxon([x - 0.50 for x in clip_a_vals], zero_method="pratt")
    
    print(f"Overall Flip Accuracy a : {pt_a:.4f} [95% CI: {low_a:.4f}, {high_a:.4f}] (SE={se_a:.4f})")
    print(f"Overall Flip Probability: {pt_pf*100:.2f}% [95% CI: {low_pf*100:.2f}%, {high_pf*100:.2f}%]")
    print(f"Net Pairwise Risk Change: {pt_dr:+.6f} [95% CI: {low_dr:+.6f}, {high_dr:+.6f}]")
    print(f"\nTest of H0: a = 0.50 (No Net Directional Skill):")
    print(f"  Paired t-test vs 0.50 : t = {t_stat:+.4f}, p = {p_val_t:.4f}")
    print(f"  Wilcoxon test vs 0.50 : W = {w_stat:.1f}, p = {p_val_w:.4f}")
    print(f"  Null hypothesis a = 0.50 cannot be rejected: {p_val_t > 0.05}")

    # -------------------------------------------------------------------------
    # PART 3: STRATIFIED FLIP ACCURACY a
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("PART 3: STRATIFICATION OF FLIP ACCURACY a")
    print("=" * 75)
    
    # (a) By Visual Margin |gap_vis|
    margin_bins = [
        (0.00, 0.05, "[0.00, 0.05)"),
        (0.05, 0.10, "[0.05, 0.10)"),
        (0.10, 0.20, "[0.10, 0.20)"),
        (0.20, 0.40, "[0.20, 0.40)"),
        (0.40, 999.0, "[0.40, +inf)")
    ]
    
    print("\n--- 3a. Stratified by Visual Margin |gap_vis| ---")
    margin_results = []
    for low_m, high_m, label in margin_bins:
        tot_f, tot_c, tot_h, tot_p = 0, 0, 0, 0
        for cdata in clip_predictions:
            s_vis = cdata["s_vis"]
            lam = cdata["lambda"]
            delta = cdata["delta_s"]
            gold = cdata["gold_fids"]
            fids = cdata["cand_fids"]
            tilde_delta = alpha_default * lam * delta
            K = len(fids)
            for i in range(K):
                if fids[i] not in gold:
                    continue
                for j in range(K):
                    if fids[j] in gold:
                        continue
                    tot_p += 1
                    gap_vis = s_vis[i] - s_vis[j]
                    abs_gap = abs(gap_vis)
                    if low_m <= abs_gap < high_m:
                        gap_comb = gap_vis + (tilde_delta[i] - tilde_delta[j])
                        if np.sign(gap_comb) != np.sign(gap_vis) and (gap_vis != 0):
                            tot_f += 1
                            if gap_vis < 0 and gap_comb > 0:
                                tot_c += 1
                            elif gap_vis > 0 and gap_comb < 0:
                                tot_h += 1
                                
        acc_a = (tot_c / float(tot_f)) if tot_f > 0 else np.nan
        rec_m = {
            "margin_bin": label,
            "flips": tot_f,
            "correct": tot_c,
            "harmful": tot_h,
            "accuracy_a": acc_a
        }
        margin_results.append(rec_m)
        print(f"Margin {label:<16}: Flips={tot_f:4d} | Correct={tot_c:4d} | Harmful={tot_h:4d} | a = {acc_a:.4f}" if not np.isnan(acc_a) else f"Margin {label:<16}: Flips=   0 | a = N/A")

    # (b) By Seen vs Unseen Landmarks
    print("\n--- 3b. Stratified by Landmark Group ---")
    for group_name, is_s in [("Seen Landmarks", True), ("Unseen Landmarks", False)]:
        sub_recs = [r for r in per_clip_stats if r["is_seen"] == is_s]
        tot_c = sum(r["correct"] for r in sub_recs)
        tot_h = sum(r["harmful"] for r in sub_recs)
        tot_f = sum(r["flips"] for r in sub_recs)
        acc_grp = (tot_c / float(tot_f)) if tot_f > 0 else np.nan
        print(f"{group_name:<20}: Flips={tot_f:4d} | Correct={tot_c:4d} | Harmful={tot_h:4d} | a = {acc_grp:.4f}")

    # (c) By Fact Node Degree in KG
    print("\n--- 3c. Stratified by KG Fact Node Degree ---")
    deg_tiers = [
        ("Low-degree (d <= 3)", lambda d: d <= 3),
        ("Mid-degree (4 <= d <= 10)", lambda d: 4 <= d <= 10),
        ("High-degree/Hubs (d > 10)", lambda d: d > 10)
    ]
    
    deg_results = []
    for deg_label, deg_fn in deg_tiers:
        tot_f, tot_c, tot_h = 0, 0, 0
        for cdata in clip_predictions:
            s_vis = cdata["s_vis"]
            lam = cdata["lambda"]
            delta = cdata["delta_s"]
            gold = cdata["gold_fids"]
            fids = cdata["cand_fids"]
            degs = cdata["degrees"]
            tilde_delta = alpha_default * lam * delta
            K = len(fids)
            for i in range(K):
                if fids[i] not in gold or not deg_fn(degs[i]):
                    continue
                for j in range(K):
                    if fids[j] in gold:
                        continue
                    gap_vis = s_vis[i] - s_vis[j]
                    gap_comb = gap_vis + (tilde_delta[i] - tilde_delta[j])
                    if np.sign(gap_comb) != np.sign(gap_vis) and (gap_vis != 0):
                        tot_f += 1
                        if gap_vis < 0 and gap_comb > 0:
                            tot_c += 1
                        elif gap_vis > 0 and gap_comb < 0:
                            tot_h += 1
        acc_d = (tot_c / float(tot_f)) if tot_f > 0 else np.nan
        print(f"{deg_label:<28}: Flips={tot_f:4d} | Correct={tot_c:4d} | Harmful={tot_h:4d} | a = {acc_d:.4f}")

    # Export results
    output_data = {
        "alpha_sweep": sweep_results,
        "default_alpha_stats": {
            "p_flip": pt_pf,
            "p_flip_ci": [low_pf, high_pf],
            "flip_accuracy_a": pt_a,
            "flip_accuracy_ci": [low_a, high_a],
            "delta_r_pair": pt_dr,
            "delta_r_pair_ci": [low_dr, high_dr],
            "h0_p_value_ttest": float(p_val_t),
            "h0_p_value_wilcoxon": float(p_val_w)
        },
        "margin_stratification": margin_results
    }
    
    out_file = REPO_ROOT / "results_prop2b_flip_decomposition.json"
    out_file.write_text(json.dumps(output_data, indent=2))
    print(f"\nAll Flip Decomposition results exported to: {out_file}")


if __name__ == "__main__":
    main()
