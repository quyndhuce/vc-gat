#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_theory2_conformal_admission.py
=============================================================================
Implementation of Theory 2:
Score-Invariant Validity, Score-Dependent Efficiency (Conformal Risk Control on Fact Admission)

Extends Proposition 6 (conformal risk control) to fact admission.
Guarantees E[ell(hat_theta_s)] <= alpha_risk for ANY monotonic scoring function:
  s in {s_VCGAT, s_vis, s_degree, s_random}

Evaluates:
  1. Score-Invariant Validity: Empirical risk <= alpha across 1,000 splits.
     Tested under:
       - Protocol A: Parent-Video Grouped Cluster Splits (30 calib / 30 test videos)
       - Protocol B: Random Clip Splits (250 calib / 250 test clips)
  2. Score-Dependent Efficiency: Coverage (grounded fact recall) comparison at
     alpha in {0.05, 0.10, 0.20}.
  3. Paired Differences: Delta Coverage = Coverage(s_VCGAT) - Coverage(s_vis)
     with 95% Cluster CIs and paired hypothesis testing.
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
)
from experiments.fact_selection_benchmark.feature_cache import BenchmarkFeatureCache

SEED = 42
N_SPLITS = 1000
ALPHA_RISKS = [0.05, 0.10, 0.20]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def theta_hat_fast(scores: np.ndarray, y: np.ndarray, alpha: float) -> float:
    """
    Exact O(M log M) Conformal Threshold Algorithm using sorted negative error quantile.
    Achieves 100% exact numerical equivalence to the discrete infimum search without
    allocating massive N x N broadcasting matrices.
    Loss: ell_k(theta) = 1[s_k > theta and y_k == 0]
    Condition: (n / (n+1)) * (k / n) + 1 / (n+1) <= alpha <=> k <= (n+1)*alpha - 1
    """
    n = len(scores)
    if n == 0:
        return 1.0
    k_max = int(np.floor((n + 1.0) * alpha - 1.0))
    if k_max < 0:
        return 1.0
    neg_scores = np.sort(scores[y == 0])
    m = len(neg_scores)
    if m <= k_max:
        return 0.0
    idx = m - 1 - k_max
    return float(neg_scores[idx])


def extract_fact_admission_dataset(clips_dict, cids, subgraphs, feature_cache, model, clip_to_video, alpha_val=0.40):
    model.eval()
    records = []
    
    with torch.no_grad():
        for cid in cids:
            recs = clips_dict[cid]
            lid = recs[0].get("landmark_id", "unknown")
            vid = clip_to_video.get(cid, "unknown")
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
            
            s_vcgat_np = s_vis_np + alpha_val * lam_np * delta_np
            
            for i, fid in enumerate(cand_fids):
                records.append({
                    "clip_id": cid,
                    "video_id": vid,
                    "landmark_id": lid,
                    "fact_id": fid,
                    "y_grounded": 1 if (fid in gold_fids) else 0,
                    "s_vis": float(s_vis_np[i]),
                    "s_vcgat": float(s_vcgat_np[i]),
                    "s_degree": float(deg_np[i]),
                })
                
    return records


def normalize_scores_01(arr: np.ndarray) -> np.ndarray:
    min_v, max_v = np.min(arr), np.max(arr)
    if max_v - min_v > 1e-8:
        return (arr - min_v) / (max_v - min_v)
    return np.zeros_like(arr)


def run_conformal_experiment(fact_records):
    print("=" * 75)
    print("RUNNING THEORY 2: CONFORMAL RISK CONTROL ON FACT ADMISSION")
    print(f"Total Candidate Facts: {len(fact_records):,}")
    n_gold = sum(r["y_grounded"] for r in fact_records)
    print(f"Total Grounded Facts: {n_gold:,} ({n_gold / len(fact_records) * 100:.1f}%)")
    
    unique_videos = sorted(list(set(r["video_id"] for r in fact_records)))
    unique_clips = sorted(list(set(r["clip_id"] for r in fact_records)))
    print(f"Parent Videos (Clusters): {len(unique_videos)}, Unique Clips: {len(unique_clips)}")
    
    # Pre-extract arrays
    y_all = np.array([r["y_grounded"] for r in fact_records], dtype=int)
    s_vis_raw = np.array([r["s_vis"] for r in fact_records])
    s_vcgat_raw = np.array([r["s_vcgat"] for r in fact_records])
    s_deg_raw = np.array([r["s_degree"] for r in fact_records])
    
    # Scale to [0, 1] for conformal consistency
    s_vis = normalize_scores_01(s_vis_raw)
    s_vcgat = normalize_scores_01(s_vcgat_raw)
    s_degree = normalize_scores_01(s_deg_raw)
    
    rng = np.random.RandomState(SEED)
    s_random = rng.uniform(0.0, 1.0, size=len(fact_records))
    
    score_dict = {
        "s_VCGAT": s_vcgat,
        "s_vis": s_vis,
        "s_degree (Placebo 1)": s_degree,
        "s_random (Placebo 2)": s_random,
    }
    
    # Pre-group indices by video and by clip
    video_to_indices = defaultdict(list)
    clip_to_indices = defaultdict(list)
    for idx, r in enumerate(fact_records):
        video_to_indices[r["video_id"]].append(idx)
        clip_to_indices[r["clip_id"]].append(idx)
        
    protocols = [
        ("Protocol A: Parent-Video Grouped Cluster Splits (60 clusters)", "video", unique_videos, video_to_indices),
        ("Protocol B: Random Clip Splits (500 clips)", "clip", unique_clips, clip_to_indices)
    ]
    
    full_protocol_results = {}
    
    for proto_name, proto_key, group_items, group_dict in protocols:
        print("\n" + "-" * 75)
        print(f"Executing: {proto_name} (R = {N_SPLITS} splits)")
        print("-" * 75)
        
        K = len(group_items)
        K_cal = K // 2
        
        # Store results: proto_results[score_name][alpha] = list of dicts
        sim_data = {
            s_name: {
                a: {"risks": [], "coverages": [], "theta_hats": [], "total_admitted": []}
                for a in ALPHA_RISKS
            }
            for s_name in score_dict
        }
        
        # Also store paired differences: delta_cov = cov(s_VCGAT) - cov(s_vis)
        paired_delta_cov = {a: [] for a in ALPHA_RISKS}
        
        rng_split = np.random.RandomState(SEED)
        
        for rep in range(N_SPLITS):
            perm = rng_split.permutation(K)
            cal_groups = [group_items[i] for i in perm[:K_cal]]
            test_groups = [group_items[i] for i in perm[K_cal:]]
            
            cal_idx = []
            for g in cal_groups:
                cal_idx.extend(group_dict[g])
            test_idx = []
            for g in test_groups:
                test_idx.extend(group_dict[g])
                
            cal_idx = np.array(cal_idx)
            test_idx = np.array(test_idx)
            
            y_cal = y_all[cal_idx]
            y_test = y_all[test_idx]
            n_gold_test = np.sum(y_test == 1)
            
            cov_vcgat_this_split = {}
            cov_vis_this_split = {}
            
            for s_name, s_arr in score_dict.items():
                s_cal = s_arr[cal_idx]
                s_test = s_arr[test_idx]
                
                for a in ALPHA_RISKS:
                    th = theta_hat_fast(s_cal, y_cal, a)
                    admitted = (s_test > th)
                    
                    # Risk = E[1[admit and ungrounded]]
                    emp_risk = float(np.mean(admitted & (y_test == 0)))
                    
                    # Coverage = Recall of grounded facts: 1[admit and grounded] / Total Grounded
                    emp_cov = float(np.sum(admitted & (y_test == 1)) / n_gold_test) if n_gold_test > 0 else 0.0
                    tot_adm_rate = float(np.mean(admitted))
                    
                    sim_data[s_name][a]["risks"].append(emp_risk)
                    sim_data[s_name][a]["coverages"].append(emp_cov)
                    sim_data[s_name][a]["theta_hats"].append(th)
                    sim_data[s_name][a]["total_admitted"].append(tot_adm_rate)
                    
                    if s_name == "s_VCGAT":
                        cov_vcgat_this_split[a] = emp_cov
                    elif s_name == "s_vis":
                        cov_vis_this_split[a] = emp_cov
                        
            for a in ALPHA_RISKS:
                paired_delta_cov[a].append(cov_vcgat_this_split[a] - cov_vis_this_split[a])

        # Summary for this protocol
        summary_table = []
        for a in ALPHA_RISKS:
            print(f"\n--- Target Risk Budget alpha = {a:.2f} ---")
            for s_name in score_dict:
                d = sim_data[s_name][a]
                mean_r = float(np.mean(d["risks"])) * 100.0
                r_ci = (float(np.percentile(d["risks"], 2.5)) * 100.0, float(np.percentile(d["risks"], 97.5)) * 100.0)
                mean_cov = float(np.mean(d["coverages"])) * 100.0
                cov_ci = (float(np.percentile(d["coverages"], 2.5)) * 100.0, float(np.percentile(d["coverages"], 97.5)) * 100.0)
                mean_th = float(np.mean(d["theta_hats"]))
                
                valid = (mean_r <= (a * 100.0 + 1e-4))
                status = "VALID" if valid else "SLIGHT INFLATION"
                
                print(f"  {s_name:<22}: Empirical Risk={mean_r:5.2f}% [{r_ci[0]:5.2f}%, {r_ci[1]:5.2f}%] ({status}) | "
                      f"Coverage (Recall)={mean_cov:5.2f}% [{cov_ci[0]:5.2f}%, {cov_ci[1]:5.2f}%] | Mean theta={mean_th:.3f}", flush=True)
                
                summary_table.append({
                    "protocol": proto_name,
                    "alpha_target": a,
                    "score_name": s_name,
                    "mean_risk_pct": mean_r,
                    "risk_ci_pct": list(r_ci),
                    "is_valid": valid,
                    "mean_coverage_pct": mean_cov,
                    "coverage_ci_pct": list(cov_ci),
                    "mean_theta": mean_th
                })
                
            # Paired comparison
            diffs = np.array(paired_delta_cov[a]) * 100.0
            mean_diff = float(np.mean(diffs))
            ci_diff = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
            t_stat, p_val = stats.ttest_1samp(diffs, 0.0)
            print(f"  --> PAIRED DELTA (s_VCGAT - s_vis): {mean_diff:+.3f}% [95% CI: {ci_diff[0]:+.3f}%, {ci_diff[1]:+.3f}%] (t={t_stat:.2f}, p={p_val:.4f})", flush=True)
            
        full_protocol_results[proto_key] = {
            "name": proto_name,
            "summary": summary_table,
            "paired_deltas": {
                a: {
                    "mean_diff_pct": float(np.mean(paired_delta_cov[a])) * 100.0,
                    "ci_diff_pct": [float(np.percentile(paired_delta_cov[a], 2.5)) * 100.0, float(np.percentile(paired_delta_cov[a], 97.5)) * 100.0],
                    "p_val": float(stats.ttest_1samp(paired_delta_cov[a], 0.0)[1])
                }
                for a in ALPHA_RISKS
            }
        }
        
    out_file = REPO_ROOT / "results_theory2_conformal_admission.json"
    out_file.write_text(json.dumps(full_protocol_results, indent=2))
    print(f"\nAll Theory 2 results successfully exported to: {out_file}")
    return full_protocol_results


def main():
    facts = load_normalized_facts()
    test_clips, _ = load_gold_data(split_type="test")
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
    
    print("\n--- Extracting Hanoi-500 Candidate Fact Dataset ---")
    fact_records = extract_fact_admission_dataset(
        test_clips, hanoi_test_cids, subgraphs, feature_cache, dec_model, clip_to_video, alpha_val=0.40
    )
    
    run_conformal_experiment(fact_records)


if __name__ == "__main__":
    main()
