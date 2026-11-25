#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_ranking_certificates.py
=============================================================================
Computes exact empirical coverage and verification metrics for Proposition 2b
Ranking Preservation Certificates on Hanoi-500 test clips using DEC-v2:
- All-pair certificate coverage
- Adjacent-pair certificate coverage
- Top-5 boundary certificate coverage (candidate 5 vs 6)
- Full Top-5 set certificate coverage:
    min_{i in T_5}(s_vis,i - eps_i) > max_{j not in T_5}(s_vis,j + eps_j)
- Empirical visual-to-graph Top-5 agreement
- Certified pair violations
- Certified Top-5 set violations
- Exports ranking_candidate_predictions.csv and ranking_certificate_results.json
=============================================================================
"""

import sys
import json
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.fact_selection_benchmark.train_dec_vcgat_v2 import DecoupledEvidenceCalibratedVCGATv2, build_clip_graph
from experiments.fact_selection_benchmark.dataset_loader import load_gold_data, load_normalized_facts
from experiments.fact_selection_benchmark.feature_cache import BenchmarkFeatureCache

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

facts = load_normalized_facts()
test_clips, _ = load_gold_data(split_type="test")
hanoi_cids = sorted([cid for cid in test_clips if not cid.startswith("hcmc_")])
print(f"Total Hanoi test clips for certificate evaluation: {len(hanoi_cids)}")

feature_cache = BenchmarkFeatureCache(device=device)
feature_cache.extract_fact_embeddings(facts)
feature_cache.extract_video_features([])
rel_subgraphs = torch.load(feature_cache.cache_dir / "landmark_subgraphs_rel.pt", weights_only=False)

dec_ckpt = REPO_ROOT / "experiments" / "checkpoints_fact_selection" / "best_dec_vcgat_v2.pt"
dec_model = DecoupledEvidenceCalibratedVCGATv2(d_model=768).to(device)
dec_model.load_state_dict(torch.load(dec_ckpt, map_location=device))
dec_model.eval()

prediction_rows = []

# Certificate accumulators
total_all_pairs = 0
certified_all_pairs = 0
certified_all_pair_violations = 0

total_adj_pairs = 0
certified_adj_pairs = 0
certified_adj_pair_violations = 0

total_boundary_clips = 0
certified_boundary_clips = 0
certified_boundary_violations = 0

total_set_clips = 0
certified_set_clips = 0
certified_set_violations = 0

empirical_top5_match_clips = 0

with torch.no_grad():
    for cid in hanoi_cids:
        recs = test_clips[cid]
        lid = recs[0].get("landmark_id", "unknown")
        g = rel_subgraphs.get(lid)
        v_frames = feature_cache.video_frames.get(cid)
        v_pooled = feature_cache.video_pooled.get(cid)
        if g is None or v_frames is None:
            continue

        cand_fids = [r["fact_id"] for r in recs]
        v_f_tensor = torch.from_numpy(v_frames).float().to(device)
        v_p_tensor = torch.from_numpy(v_pooled).float().to(device) if v_pooled is not None else None

        h_tensor, adj, rel_adj, degrees, all_fids, fid2idx = build_clip_graph(
            g, cand_fids, feature_cache.fact_embeddings, device
        )
        logits_d, s_vis, lambda_i, delta_s = dec_model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)

        final_scores = logits_d.cpu().numpy()
        vis_scores = s_vis.cpu().numpy()
        gates = lambda_i.cpu().numpy()
        residuals = delta_s.cpu().numpy()

        # Extract per-candidate scalars
        cands_data = []
        for fid in cand_fids:
            idx = fid2idx[fid]
            sv = float(vis_scores[idx])
            lam = float(gates[idx])
            res = float(residuals[idx])
            fs = float(final_scores[idx])
            eps = 0.32 * lam
            cands_data.append({
                "clip_id": cid,
                "candidate_id": fid,
                "visual_score": sv,
                "reliability_gate": lam,
                "graph_residual": res,
                "final_score": fs,
                "epsilon": eps
            })

        # Sort to assign visual ranks (1 to M) and final ranks (1 to M)
        cands_by_vis = sorted(cands_data, key=lambda x: x["visual_score"], reverse=True)
        for r_idx, c in enumerate(cands_by_vis, start=1):
            c["visual_rank"] = r_idx

        cands_by_final = sorted(cands_data, key=lambda x: x["final_score"], reverse=True)
        for r_idx, c in enumerate(cands_by_final, start=1):
            c["final_rank"] = r_idx

        for c in cands_data:
            prediction_rows.append(c)

        # -------------------------------------------------------------
        # Evaluate Proposition 2b Certificates for this clip
        # -------------------------------------------------------------
        M = len(cands_by_vis)
        top5_vis_ids = set(c["candidate_id"] for c in cands_by_vis[:5])
        top5_final_ids = set(c["candidate_id"] for c in cands_by_final[:5])

        if top5_vis_ids == top5_final_ids:
            empirical_top5_match_clips += 1

        # 1. All-pair certificate evaluation
        for i in range(M):
            for j in range(i + 1, M):
                total_all_pairs += 1
                ci = cands_by_vis[i]
                cj = cands_by_vis[j]
                # ci has higher visual score than cj
                margin = ci["visual_score"] - cj["visual_score"]
                cert_thresh = ci["epsilon"] + cj["epsilon"]
                if margin > cert_thresh:
                    certified_all_pairs += 1
                    # Check violation in final score
                    if ci["final_score"] <= cj["final_score"]:
                        certified_all_pair_violations += 1

        # 2. Adjacent-pair certificate evaluation
        for i in range(M - 1):
            total_adj_pairs += 1
            ci = cands_by_vis[i]
            c_next = cands_by_vis[i + 1]
            margin = ci["visual_score"] - c_next["visual_score"]
            cert_thresh = ci["epsilon"] + c_next["epsilon"]
            if margin > cert_thresh:
                certified_adj_pairs += 1
                if ci["final_score"] <= c_next["final_score"]:
                    certified_adj_pair_violations += 1

        # 3. Top-5 Boundary certificate evaluation (Candidate 5 vs Candidate 6)
        if M >= 6:
            total_boundary_clips += 1
            c5 = cands_by_vis[4]  # 0-indexed rank 5
            c6 = cands_by_vis[5]  # 0-indexed rank 6
            boundary_margin = c5["visual_score"] - c6["visual_score"]
            boundary_thresh = c5["epsilon"] + c6["epsilon"]
            if boundary_margin > boundary_thresh:
                certified_boundary_clips += 1
                if c5["final_score"] <= c6["final_score"]:
                    certified_boundary_violations += 1

        # 4. Full Top-5 set certificate evaluation:
        # min_{i in T_5} (s_vis,i - eps_i) > max_{j not in T_5} (s_vis,j + eps_j)
        if M >= 6:
            total_set_clips += 1
            min_lower_top5 = min(c["visual_score"] - c["epsilon"] for c in cands_by_vis[:5])
            max_upper_outside = max(c["visual_score"] + c["epsilon"] for c in cands_by_vis[5:])
            if min_lower_top5 > max_upper_outside:
                certified_set_clips += 1
                if top5_vis_ids != top5_final_ids:
                    certified_set_violations += 1

# Export candidate predictions CSV
pred_csv_path = REPO_ROOT / "ranking_candidate_predictions.csv"
fieldnames = [
    "clip_id", "candidate_id", "visual_score", "reliability_gate",
    "graph_residual", "final_score", "epsilon", "visual_rank", "final_rank"
]
with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in prediction_rows:
        writer.writerow(row)
print(f"Exported prediction-level rows ({len(prediction_rows)} rows): {pred_csv_path}")

# Compile Certificate Results JSON
all_pair_coverage_pct = (certified_all_pairs / total_all_pairs * 100) if total_all_pairs > 0 else 0.0
adj_pair_coverage_pct = (certified_adj_pairs / total_adj_pairs * 100) if total_adj_pairs > 0 else 0.0
boundary_coverage_pct = (certified_boundary_clips / total_boundary_clips * 100) if total_boundary_clips > 0 else 0.0
set_coverage_pct = (certified_set_clips / total_set_clips * 100) if total_set_clips > 0 else 0.0
top5_agreement_pct = (empirical_top5_match_clips / len(hanoi_cids) * 100) if len(hanoi_cids) > 0 else 0.0

certificate_results = {
    "evaluation_scope": "Hanoi-500 test clips",
    "total_clips_evaluated": len(hanoi_cids),
    "total_candidates_evaluated": len(prediction_rows),
    "all_pairs": {
        "description": "Pairwise condition: s_vis,i - s_vis,j > eps_i + eps_j",
        "numerator_certified": certified_all_pairs,
        "denominator_total": total_all_pairs,
        "coverage_percentage": round(all_pair_coverage_pct, 2),
        "certified_violations": certified_all_pair_violations,
        "violation_rate": 0.0 if certified_all_pairs == 0 else (certified_all_pair_violations / certified_all_pairs)
    },
    "adjacent_pairs": {
        "description": "Adjacent visual ranking condition: s_vis,i - s_vis,i+1 > eps_i + eps_i+1",
        "numerator_certified": certified_adj_pairs,
        "denominator_total": total_adj_pairs,
        "coverage_percentage": round(adj_pair_coverage_pct, 2),
        "certified_violations": certified_adj_pair_violations,
        "violation_rate": 0.0 if certified_adj_pairs == 0 else (certified_adj_pair_violations / certified_adj_pairs)
    },
    "top5_boundary": {
        "description": "Boundary separation condition: s_vis,5 - s_vis,6 > eps_5 + eps_6",
        "numerator_certified": certified_boundary_clips,
        "denominator_total": total_boundary_clips,
        "coverage_percentage": round(boundary_coverage_pct, 2),
        "certified_violations": certified_boundary_violations,
        "violation_rate": 0.0 if certified_boundary_clips == 0 else (certified_boundary_violations / certified_boundary_clips)
    },
    "full_top5_set": {
        "description": "Full set preservation condition: min_{i in T_5}(s_vis,i - eps_i) > max_{j not in T_5}(s_vis,j + eps_j)",
        "numerator_certified": certified_set_clips,
        "denominator_total": total_set_clips,
        "coverage_percentage": round(set_coverage_pct, 2),
        "certified_violations": certified_set_violations,
        "violation_rate": 0.0 if certified_set_clips == 0 else (certified_set_violations / certified_set_clips)
    },
    "empirical_visual_to_graph_top5_agreement": {
        "description": "Exact match between Visual Top-5 set and Composite VC-GAT Top-5 set",
        "numerator_matched": empirical_top5_match_clips,
        "denominator_total": len(hanoi_cids),
        "agreement_percentage": round(top5_agreement_pct, 2)
    }
}

cert_json_path = REPO_ROOT / "ranking_certificate_results.json"
with open(cert_json_path, "w", encoding="utf-8") as f:
    json.dump(certificate_results, f, indent=2)
print(f"Exported certificate results: {cert_json_path}")

print("="*80)
print("RANKING PRESERVATION CERTIFICATE SUMMARY (PROPOSITION 2b)")
print("="*80)
print(f"All-Pair Certificate Coverage:      {certified_all_pairs} / {total_all_pairs} ({all_pair_coverage_pct:.2f}%) | Violations: {certified_all_pair_violations}")
print(f"Adjacent-Pair Certificate Coverage:  {certified_adj_pairs} / {total_adj_pairs} ({adj_pair_coverage_pct:.2f}%) | Violations: {certified_adj_pair_violations}")
print(f"Top-5 Boundary Coverage (K=5 vs 6): {certified_boundary_clips} / {total_boundary_clips} ({boundary_coverage_pct:.2f}%) | Violations: {certified_boundary_violations}")
print(f"Full Top-5 Set Certificate:         {certified_set_clips} / {total_set_clips} ({set_coverage_pct:.2f}%) | Violations: {certified_set_violations}")
print(f"Empirical Top-5 Set Agreement:      {empirical_top5_match_clips} / {len(hanoi_cids)} ({top5_agreement_pct:.2f}%)")
print("="*80)

