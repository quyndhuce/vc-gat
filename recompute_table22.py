#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recompute_table22.py
=============================================================================
Independent verification and recomputation of Table 22 (Fact Selection Grounding)
on Hanoi-500 test clips using verified checkpoints:
- Baseline: Cross-Attention Ranker (best_cross_attention.pt)
- Proposed: VC-GAT DEC-v2 (best_dec_vcgat_v2.pt)

Computes:
1. Per-clip metrics across all 6 evaluation slices.
2. Paired point estimates.
3. Clip-level percentile bootstrap 95% CIs (B=10,000, seed=42).
4. Parent-video cluster bootstrap percentile 95% CIs (B=10,000, seed=42).
5. Exact p-values (t-test, Wilcoxon, cluster permutation) and Holm-Bonferroni correction.
6. Effect sizes (Cohen's d).
7. Exports raw metrics, bootstrap replicates (.npz), and verified results (.json).
=============================================================================
"""

import sys
import json
import csv
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
import scipy.stats as stats
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

SEED = 42
B_RESAMPLES = 10000
SEEN_LANDMARKS = {
    "hoan_kiem_lake", "temple_of_literature", "ba_dinh_square",
    "imperial_citadel_of_thang_long", "one_pillar_pagoda", "hanoi_old_quarter", "west_lake"
}

csv_path = REPO_ROOT / "table22_per_clip_metrics.csv"
per_clip_records = []

if csv_path.exists() and (not HAS_TORCH or "--from-cache" in sys.argv):
    print(f"Loading precomputed per-clip metrics from {csv_path.name}...")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            per_clip_records.append({
                "clip_id": r["clip_id"],
                "source_video_id": r["source_video_id"],
                "landmark_id": r["landmark_id"],
                "is_seen": r["is_seen"].lower() in ("true", "1"),
                "has_hard_neg": r["has_hard_neg"].lower() in ("true", "1"),
                "has_hist_gold": r["has_hist_gold"].lower() in ("true", "1"),
                "baseline_ndcg5": float(r["baseline_ndcg5"]) if r["baseline_ndcg5"] not in ("", "nan") else np.nan,
                "vcgat_ndcg5": float(r["vcgat_ndcg5"]) if r["vcgat_ndcg5"] not in ("", "nan") else np.nan,
                "baseline_p5": float(r["baseline_p5"]) if r["baseline_p5"] not in ("", "nan") else np.nan,
                "vcgat_p5": float(r["vcgat_p5"]) if r["vcgat_p5"] not in ("", "nan") else np.nan,
                "baseline_auc": float(r["baseline_auc"]) if r["baseline_auc"] not in ("", "nan") else np.nan,
                "vcgat_auc": float(r["vcgat_auc"]) if r["vcgat_auc"] not in ("", "nan") else np.nan,
                "baseline_hist_ndcg5": float(r["baseline_hist_ndcg5"]) if r["baseline_hist_ndcg5"] not in ("", "nan") else np.nan,
                "vcgat_hist_ndcg5": float(r["vcgat_hist_ndcg5"]) if r["vcgat_hist_ndcg5"] not in ("", "nan") else np.nan,
            })
    print(f"Loaded {len(per_clip_records)} cached clip records.")
elif HAS_TORCH:
    from experiments.fact_selection_benchmark.train_dec_vcgat_v2 import DecoupledEvidenceCalibratedVCGATv2, build_clip_graph
    from experiments.fact_selection_benchmark.models.cross_attention_ranker import CrossAttentionRankerModel
    from experiments.fact_selection_benchmark.dataset_loader import (
        load_gold_data,
        load_normalized_facts,
        compute_ranking_metrics,
        compute_hard_negative_auc,
    )
    from experiments.fact_selection_benchmark.feature_cache import BenchmarkFeatureCache

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device for inference: {device}")

    # Load Manifest for source_video_id mapping
    manifest_path = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "evaluation_clips_manifest.json"
    manifest_items = json.loads(manifest_path.read_text())
    clip_to_source_video = {item["clip_id"]: item.get("source_video_id") or item.get("video_id") for item in manifest_items}

    # Load Facts and Test Clips
    facts = load_normalized_facts()
    test_clips, _ = load_gold_data(split_type="test")
    hanoi_cids = sorted([cid for cid in test_clips if not cid.startswith("hcmc_")])
    print(f"Total Hanoi test clips: {len(hanoi_cids)}")

    # Feature cache and graphs
    feature_cache = BenchmarkFeatureCache(device=device)
    feature_cache.extract_fact_embeddings(facts)
    feature_cache.extract_video_features([])
    rel_subgraphs = torch.load(feature_cache.cache_dir / "landmark_subgraphs_rel.pt", weights_only=False)

    # Checkpoint paths
    cross_ckpt = REPO_ROOT / "experiments" / "checkpoints_fact_selection" / "best_cross_attention.pt"
    dec_ckpt = REPO_ROOT / "experiments" / "checkpoints_fact_selection" / "best_dec_vcgat_v2.pt"

    cross_model = CrossAttentionRankerModel(d_model=768).to(device)
    cross_model.load_state_dict(torch.load(cross_ckpt, map_location=device))
    cross_model.eval()

    dec_model = DecoupledEvidenceCalibratedVCGATv2(d_model=768).to(device)
    dec_model.load_state_dict(torch.load(dec_ckpt, map_location=device))
    dec_model.eval()

    with torch.no_grad():
        for cid in hanoi_cids:
            recs = test_clips[cid]
            lid = recs[0].get("landmark_id", "unknown")
            vid = clip_to_source_video.get(cid, "unknown_video")
            is_seen = (lid in SEEN_LANDMARKS)

            g = rel_subgraphs.get(lid)
            v_frames = feature_cache.video_frames.get(cid)
            v_pooled = feature_cache.video_pooled.get(cid)
            if g is None or v_frames is None:
                continue

            cand_fids = [r["fact_id"] for r in recs]
            gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("label") == "HARD_NEGATIVE")

            hist_gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold") and facts.get(r["fact_id"], {}).get("fact_type") == "historical")
            hist_cand_fids = [fid for fid in cand_fids if facts.get(fid, {}).get("fact_type") == "historical"]
            has_hist = len(hist_gold_fids) > 0

            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device) if v_pooled is not None else None

            # 1. Baseline Cross-Attention
            scores_c = cross_model(cand_fids, feature_cache.fact_embeddings, v_f_tensor, v_p_tensor)
            f_scores_c = {fid: float(scores_c[i]) for i, fid in enumerate(cand_fids)}
            ranked_c = sorted(cand_fids, key=lambda fid: f_scores_c[fid], reverse=True)

            m_c = compute_ranking_metrics(ranked_c, gold_fids, cand_fids, top_k_list=[5])
            auc_c = compute_hard_negative_auc(f_scores_c, gold_fids, hard_neg_fids)
            hist_ndcg_c = None
            if has_hist:
                ranked_h_c = [fid for fid in ranked_c if fid in hist_cand_fids]
                m_h_c = compute_ranking_metrics(ranked_h_c, hist_gold_fids, hist_cand_fids, top_k_list=[5])
                hist_ndcg_c = m_h_c["nDCG@5"]

            # 2. Proposed VC-GAT DEC-v2
            h_tensor, adj, rel_adj, degrees, all_fids, fid2idx = build_clip_graph(
                g, cand_fids, feature_cache.fact_embeddings, device
            )
            logits_d, s_vis, lambda_i, delta_s = dec_model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
            scores_d = logits_d.cpu().numpy()
            f_scores_d = {fid: float(scores_d[fid2idx[fid]]) for fid in cand_fids}
            ranked_d = sorted(cand_fids, key=lambda fid: f_scores_d[fid], reverse=True)

            m_d = compute_ranking_metrics(ranked_d, gold_fids, cand_fids, top_k_list=[5])
            auc_d = compute_hard_negative_auc(f_scores_d, gold_fids, hard_neg_fids)
            hist_ndcg_d = None
            if has_hist:
                ranked_h_d = [fid for fid in ranked_d if fid in hist_cand_fids]
                m_h_d = compute_ranking_metrics(ranked_h_d, hist_gold_fids, hist_cand_fids, top_k_list=[5])
                hist_ndcg_d = m_h_d["nDCG@5"]

            has_hn = (auc_c is not None and auc_d is not None)

            per_clip_records.append({
                "clip_id": cid,
                "source_video_id": vid,
                "landmark_id": lid,
                "is_seen": is_seen,
                "has_hard_neg": has_hn,
                "has_hist_gold": has_hist,
                "baseline_ndcg5": m_c["nDCG@5"],
                "vcgat_ndcg5": m_d["nDCG@5"],
                "baseline_p5": m_c["P@5"],
                "vcgat_p5": m_d["P@5"],
                "baseline_auc": auc_c if has_hn else np.nan,
                "vcgat_auc": auc_d if has_hn else np.nan,
                "baseline_hist_ndcg5": hist_ndcg_c if has_hist else np.nan,
                "vcgat_hist_ndcg5": hist_ndcg_d if has_hist else np.nan
            })

    print(f"Processed per-clip records for {len(per_clip_records)} clips.")

    # Export per-clip metrics CSV
    fieldnames = list(per_clip_records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_clip_records:
            writer.writerow(row)
    print(f"Exported per-clip metrics: {csv_path}")
else:
    raise RuntimeError("PyTorch is required for live inference when table22_per_clip_metrics.csv is not found.")

# =============================================================================
# STATISTICAL EVALUATION & BOOTSTRAP (Task 2)
# =============================================================================

slices = {
    "Overall nDCG@5": {
        "clips": per_clip_records,
        "base_key": "baseline_ndcg5",
        "vcgat_key": "vcgat_ndcg5",
        "estimand": "Difference in clip-level nDCG@5 across all Hanoi test clips"
    },
    "Precision@5": {
        "clips": per_clip_records,
        "base_key": "baseline_p5",
        "vcgat_key": "vcgat_p5",
        "estimand": "Difference in clip-level Precision@5 across all Hanoi test clips"
    },
    "Hard Negative AUC": {
        "clips": [r for r in per_clip_records if r["has_hard_neg"]],
        "base_key": "baseline_auc",
        "vcgat_key": "vcgat_auc",
        "estimand": "Difference in Hard-Negative ROC-AUC on clips with >=1 hard negative fact"
    },
    "Seen Landmarks nDCG@5": {
        "clips": [r for r in per_clip_records if r["is_seen"]],
        "base_key": "baseline_ndcg5",
        "vcgat_key": "vcgat_ndcg5",
        "estimand": "Difference in nDCG@5 on clips belonging to 7 Seen landmarks"
    },
    "Unseen Landmarks nDCG@5": {
        "clips": [r for r in per_clip_records if not r["is_seen"]],
        "base_key": "baseline_ndcg5",
        "vcgat_key": "vcgat_ndcg5",
        "estimand": "Difference in nDCG@5 on clips belonging to Unseen landmarks"
    },
    "Historical Facts nDCG@5": {
        "clips": [r for r in per_clip_records if r["has_hist_gold"]],
        "base_key": "baseline_hist_ndcg5",
        "vcgat_key": "vcgat_hist_ndcg5",
        "estimand": "Difference in nDCG@5 on clips with historical/commemorative non-visual gold facts"
    }
}

verified_results = {}
bootstrap_replicates_dict = {}

rng = np.random.default_rng(SEED)
raw_p_values = []

for slice_name, sinfo in slices.items():
    records = sinfo["clips"]
    bkey = sinfo["base_key"]
    vkey = sinfo["vcgat_key"]

    y_base = np.array([r[bkey] for r in records], dtype=np.float64)
    y_vcgat = np.array([r[vkey] for r in records], dtype=np.float64)
    diff = y_vcgat - y_base
    n_clips = len(diff)

    # Video cluster grouping
    video_to_indices = defaultdict(list)
    for idx, r in enumerate(records):
        video_to_indices[r["source_video_id"]].append(idx)
    clusters = sorted(list(video_to_indices.keys()))
    n_clusters = len(clusters)

    mean_base = float(np.mean(y_base))
    mean_vcgat = float(np.mean(y_vcgat))
    point_delta = float(np.mean(diff))

    # Standard paired tests
    t_stat, p_t = stats.ttest_rel(y_vcgat, y_base)
    w_res = stats.wilcoxon(y_vcgat, y_base, zero_method="pratt")
    p_w = float(w_res.pvalue)
    cohen_d = float(point_delta / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0

    # 1. Clip-level Percentile Bootstrap (B = 10,000)
    clip_boot_idx = rng.integers(0, n_clips, size=(B_RESAMPLES, n_clips))
    clip_reps = np.mean(diff[clip_boot_idx], axis=1)
    clip_ci_low = float(np.percentile(clip_reps, 2.5))
    clip_ci_high = float(np.percentile(clip_reps, 97.5))

    # 2. Cluster-aware Percentile Bootstrap (B = 10,000)
    cluster_reps = np.empty(B_RESAMPLES, dtype=np.float64)
    cluster_boot_idx = rng.integers(0, n_clusters, size=(B_RESAMPLES, n_clusters))

    for b in range(B_RESAMPLES):
        sampled_clusters = [clusters[c_idx] for c_idx in cluster_boot_idx[b]]
        sampled_clip_indices = []
        for c in sampled_clusters:
            sampled_clip_indices.extend(video_to_indices[c])
        cluster_reps[b] = np.mean(diff[sampled_clip_indices])

    clust_ci_low = float(np.percentile(cluster_reps, 2.5))
    clust_ci_high = float(np.percentile(cluster_reps, 97.5))

    # Cluster hypothesis test (null centered at zero)
    cluster_reps_centered = cluster_reps - np.mean(cluster_reps)
    p_cluster = float((np.sum(np.abs(cluster_reps_centered) >= np.abs(point_delta)) + 1) / (B_RESAMPLES + 1))

    raw_p_values.append(p_cluster)

    bootstrap_replicates_dict[f"{slice_name}__clip"] = clip_reps
    bootstrap_replicates_dict[f"{slice_name}__cluster"] = cluster_reps

    verified_results[slice_name] = {
        "slice_name": slice_name,
        "estimand": sinfo["estimand"],
        "n_clips": n_clips,
        "n_parent_video_clusters": n_clusters,
        "baseline_mean": round(mean_base, 4),
        "vcgat_mean": round(mean_vcgat, 4),
        "point_delta": round(point_delta, 4),
        "clip_bootstrap_ci_95": [round(clip_ci_low, 4), round(clip_ci_high, 4)],
        "clip_bootstrap_ci_str": f"[{clip_ci_low:+.4f}, {clip_ci_high:+.4f}]",
        "cluster_bootstrap_ci_95": [round(clust_ci_low, 4), round(clust_ci_high, 4)],
        "cluster_bootstrap_ci_str": f"[{clust_ci_low:+.4f}, {clust_ci_high:+.4f}]",
        "cohen_d": round(cohen_d, 3),
        "cohen_d_definition": "mean(paired_diff) / std(paired_diff, ddof=1)",
        "p_value_t_test": float(p_t),
        "p_value_wilcoxon": p_w,
        "p_value_cluster_bootstrap_raw": p_cluster,
        "random_seed": SEED,
        "b_resamples": B_RESAMPLES
    }

# Holm-Bonferroni correction on cluster p-values
sorted_indices = np.argsort(raw_p_values)
m_hypotheses = len(raw_p_values)
adj_p_values = [0.0] * m_hypotheses

current_max = 0.0
for rank, idx in enumerate(sorted_indices):
    mult = m_hypotheses - rank
    raw_p = raw_p_values[idx]
    adj_p = min(1.0, raw_p * mult)
    adj_p = max(adj_p, current_max)
    current_max = adj_p
    adj_p_values[idx] = adj_p

slice_keys = list(slices.keys())
for i, skey in enumerate(slice_keys):
    verified_results[skey]["p_value_cluster_bootstrap_holm_adj"] = float(adj_p_values[i])
    sig = "***" if adj_p_values[i] < 0.001 else ("**" if adj_p_values[i] < 0.01 else ("*" if adj_p_values[i] < 0.05 else "n.s."))
    verified_results[skey]["significance_label"] = sig

# Save NPZ replicates
npz_path = REPO_ROOT / "table22_bootstrap_replicates.npz"
np.savez_compressed(npz_path, **bootstrap_replicates_dict)
print(f"Exported bootstrap replicates: {npz_path}")

# Save verified results JSON
json_path = REPO_ROOT / "table22_verified_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(verified_results, f, indent=2)
print(f"Exported verified results: {json_path}")

print("="*105)
print(f"{'Slice':<26} {'N':<5} {'Clust':<6} {'Base':<8} {'VC-GAT':<8} {'Delta':<8} {'Clip 95% CI':<20} {'Cluster 95% CI':<20} {'Adj p':<8}")
print("="*105)
for sname, v in verified_results.items():
    print(f"{sname:<26} {v['n_clips']:<5} {v['n_parent_video_clusters']:<6} {v['baseline_mean']:<8.4f} {v['vcgat_mean']:<8.4f} {v['point_delta']:<+8.4f} {v['clip_bootstrap_ci_str']:<20} {v['cluster_bootstrap_ci_str']:<20} {v['p_value_cluster_bootstrap_holm_adj']:<8.4e}")
print("="*105)
