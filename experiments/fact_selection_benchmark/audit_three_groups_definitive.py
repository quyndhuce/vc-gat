#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/audit_three_groups_definitive.py
=============================================================================
Definitive 3-Group Decoupling Audit on 800 Test Clips:
  - Group 1 (195 clips): Complete Subgraph (0 missing candidates, 0 missing gold).
  - Group 2 (517 clips): Missing Negatives only (>=1 missing candidate, 0 missing gold).
  - Group 3 (88 clips):  Missing Gold facts (>=1 missing gold fact).
  - Combined (712 clips): Groups 1 + 2 (No missing gold facts anywhere).
  - Overall (800 clips).

Evaluates across:
  1. Cross-Attention Ranker (Non-Graph Baseline - pure visual, no subgraph needed)
  2. Unified R-VCGAT (Fair Fallback to s_vis)
  3. Calibrated R-VCGAT (CR-VCGAT - Fair Fallback to s_vis)
  4. Proposed DEC-VCGAT v2 (Dynamic Candidate Graph Inclusion DCSE + ListMLE)
=============================================================================
"""

import sys
sys.path.insert(0, "/home/nguyendinhquy/TransMTL_OntoKG-main")
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import numpy as np

from experiments.fact_selection_benchmark.train_dec_vcgat_v2 import DecoupledEvidenceCalibratedVCGATv2, build_clip_graph
from experiments.fact_selection_benchmark.train_calibrated_rvcgat import CalibratedResidualVCGAT
from experiments.fact_selection_benchmark.train_rvcgat_study import ProposedRVCGATModel
from experiments.fact_selection_benchmark.models.cross_attention_ranker import CrossAttentionRankerModel
from experiments.fact_selection_benchmark.dataset_loader import (
    load_gold_data, load_normalized_facts, compute_ranking_metrics, compute_hard_negative_auc
)
from experiments.fact_selection_benchmark.feature_cache import BenchmarkFeatureCache

device = torch.device("cpu")
print(f"Using device: {device}")

facts = load_normalized_facts()
test_clips, _ = load_gold_data(split_type="test")
feature_cache = BenchmarkFeatureCache(device=device)
feature_cache.extract_fact_embeddings(facts)
feature_cache.extract_video_features([])
rel_subgraphs = torch.load(feature_cache.cache_dir / "landmark_subgraphs_rel.pt", weights_only=False)

# 1. Load Models
# Model 1: Cross-Attention
cross_model = CrossAttentionRankerModel(d_model=768).to(device)
cross_model.load_state_dict(torch.load("/home/nguyendinhquy/TransMTL_OntoKG-main/experiments/checkpoints_fact_selection/best_cross_attention.pt", map_location=device))
cross_model.eval()

# Model 2: Unified R-VCGAT (Fair Fallback)
unified_model = ProposedRVCGATModel(d_model=768, use_relations=True, use_reliability_gate=True, use_residual=True).to(device)
unified_model.load_state_dict(torch.load("/home/nguyendinhquy/TransMTL_OntoKG-main/experiments/checkpoints_fact_selection/best_r_vcgat_proposed_full.pt", map_location=device))
unified_model.eval()

# Model 3: Calibrated R-VCGAT (CR-VCGAT)
cr_model = CalibratedResidualVCGAT(d_model=768, delta_bound=0.8).to(device)
cr_model.load_state_dict(torch.load("/home/nguyendinhquy/TransMTL_OntoKG-main/experiments/checkpoints_fact_selection/best_calibrated_rvcgat.pt", map_location=device))
cr_model.eval()

# Model 4: DEC-VCGAT v2
dec_model = DecoupledEvidenceCalibratedVCGATv2(d_model=768).to(device)
dec_model.load_state_dict(torch.load("/home/nguyendinhquy/TransMTL_OntoKG-main/experiments/checkpoints_fact_selection/best_dec_vcgat_v2.pt", map_location=device))
dec_model.eval()

# 2. Partition Clips into 3 Groups
g1_cids = [] # Complete Subgraph (0 missing candidates)
g2_cids = [] # Missing Negatives only (>= 1 missing candidate, 0 missing gold)
g3_cids = [] # Missing Gold (>= 1 missing gold)

cids = sorted(list(test_clips.keys()))
for cid in cids:
    recs = test_clips[cid]
    lid = recs[0].get("landmark_id", "unknown")
    g = rel_subgraphs.get(lid)
    f2i = g["fid2idx"]
    has_miss_cand = False
    has_miss_gold = False
    for r in recs:
        fid = r["fact_id"]
        is_gold = r.get("is_scene_gold", False)
        if fid not in f2i:
            has_miss_cand = True
            if is_gold:
                has_miss_gold = True
    if not has_miss_cand:
        g1_cids.append(cid)
    elif not has_miss_gold:
        g2_cids.append(cid)
    else:
        g3_cids.append(cid)

print(f"Group 1 (Complete): {len(g1_cids)} | Group 2 (Missing Neg): {len(g2_cids)} | Group 3 (Missing Gold): {len(g3_cids)} | Total: {len(cids)}")

# 3. Evaluation Function
models = [
    ("Cross-Attention Ranker (Pure Visual, No Graph)", "cross"),
    ("Unified R-VCGAT (Fair s_vis Fallback)", "unified_fair"),
    ("Calibrated R-VCGAT (CR-VCGAT - Fair Fallback)", "cr_vcgat"),
    ("DEC-VCGAT v2 (Dynamic Graph + ListMLE)", "dec_v2")
]

results = {m[1]: {cid: {} for cid in cids} for m in models}

with torch.no_grad():
    for cid in cids:
        recs = test_clips[cid]
        lid = recs[0].get("landmark_id", "unknown")
        g = rel_subgraphs.get(lid)
        vf = feature_cache.video_frames.get(cid)
        vp = feature_cache.video_pooled.get(cid)
        if g is None or vf is None:
            continue

        cand_fids = [r["fact_id"] for r in recs]
        gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
        hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")

        # Candidate tensor
        cand_h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in cand_fids]
        cand_h_tensor = torch.from_numpy(np.array(cand_h_list, dtype=np.float32)).to(device)
        v_f_tensor = torch.from_numpy(vf).float().to(device)
        v_p_tensor = torch.from_numpy(vp).float().to(device) if vp is not None else None

        # 1. Cross-Attention
        logits_c = cross_model(v_f_tensor, cand_h_tensor).cpu().numpy()
        scores_c = {fid: float(logits_c[i]) for i, fid in enumerate(cand_fids)}

        # 2. Unified R-VCGAT with Fair Fallback to s_vis
        # Subgraph forward
        sub_fids = g["fact_ids"]
        h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
        h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
        adj = g["adj"].to(device)
        rel_adj = g["rel_adj"].to(device)
        degrees = g["degrees"].to(device)
        logits_u, _, _ = unified_model(h_tensor, adj, rel_adj, degrees, v_p_tensor)
        u_graph_scores = logits_u.cpu().numpy()
        idx_map = g["fid2idx"]

        # Fair visual base score for all candidates
        # s_vis for cand_h_tensor:
        v_n = unified_model.v_norm(v_p_tensor.unsqueeze(0) if v_p_tensor.dim() == 1 else v_p_tensor)
        N_c = len(cand_fids)
        v_exp_c = v_n.expand(N_c, -1)
        h_u_c = F.normalize(cand_h_tensor, p=2, dim=-1)
        v_u_c = F.normalize(v_n, p=2, dim=-1).expand(N_c, -1)
        cos_vis_c = torch.sum(h_u_c * v_u_c, dim=-1, keepdim=True)
        vis_feat_c = torch.cat([cand_h_tensor * v_exp_c, torch.abs(cand_h_tensor - v_exp_c), cos_vis_c], dim=-1)
        u_vis_cand = unified_model.vis_scorer(vis_feat_c).squeeze(-1).cpu().numpy()

        scores_u = {
            fid: float(u_graph_scores[idx_map[fid]]) if fid in idx_map else float(u_vis_cand[i])
            for i, fid in enumerate(cand_fids)
        }

        # 3. CR-VCGAT with Fair Fallback
        # cand visual features
        q_cand = cand_h_tensor.unsqueeze(1)
        kv_cand = v_f_tensor.unsqueeze(0).expand(N_c, -1, -1)
        attn_cand, _ = cr_model.cross_attn(query=q_cand, key=kv_cand, value=kv_cand)
        ctx_cand = cr_model.norm1(attn_cand.squeeze(1) + cand_h_tensor)
        vis_feat_cand = torch.cat([cand_h_tensor, ctx_cand, cand_h_tensor * ctx_cand], dim=-1)
        cr_vis_cand = cr_model.vis_scorer(vis_feat_cand).squeeze(-1).cpu().numpy()

        logits_cr, s_vis_cr, lambda_cr, delta_cr = cr_model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
        cr_graph_scores = (s_vis_cr + 0.4 * lambda_cr * delta_cr).cpu().numpy()
        scores_cr = {
            fid: float(cr_graph_scores[idx_map[fid]]) if fid in idx_map else float(cr_vis_cand[i])
            for i, fid in enumerate(cand_fids)
        }

        # 4. DEC-VCGAT v2 with Dynamic Candidate Graph
        h_t_v2, adj_v2, rel_adj_v2, deg_v2, all_fids_v2, fid2idx_v2 = build_clip_graph(
            g, cand_fids, feature_cache.fact_embeddings, device
        )
        logits_d, _, _, _ = dec_model(h_t_v2, adj_v2, rel_adj_v2, deg_v2, v_f_tensor, v_p_tensor)
        scores_d_np = logits_d.cpu().numpy()
        scores_d = {fid: float(scores_d_np[fid2idx_v2[fid]]) for fid in cand_fids}

        # Compute metrics for each model
        for m_key, s_dict in [("cross", scores_c), ("unified_fair", scores_u), ("cr_vcgat", scores_cr), ("dec_v2", scores_d)]:
            rk = sorted(cand_fids, key=lambda f: s_dict[f], reverse=True)
            m = compute_ranking_metrics(rk, gold_fids, cand_fids, top_k_list=[5])
            auc = compute_hard_negative_auc(s_dict, gold_fids, hard_neg_fids)
            results[m_key][cid] = {
                "ndcg5": m.get("nDCG@5", 0.0),
                "p5": m.get("P@5", 0.0),
                "r5": m.get("R@5", 0.0),
                "auc": auc
            }

# 4. Summary Across Groups
groups = [
    ("Group 1: Complete Subgraph (0% missing)", g1_cids),
    ("Group 2: Missing Negatives Only (0% missing gold)", g2_cids),
    ("Group 3: Missing Gold Facts (coverage issue)", g3_cids),
    ("Combined Group 1 + 2 (No Missing Gold)", g1_cids + g2_cids),
    ("Overall (Full Test Set)", cids)
]

print("\n" + "=" * 90)
print("DEFINITIVE 3-GROUP DECOUPLING AUDIT RESULTS")
print("=" * 90)

for g_name, g_list in groups:
    print(f"\n--- {g_name} [N = {len(g_list)} clips] ---")
    print(f"{'Model':<45} | {'nDCG@5':<8} | {'P@5':<8} | {'R@5':<8} | {'Hard Neg AUC':<12}")
    print("-" * 88)
    for model_name, m_key in models:
        m_dict = results[m_key]
        ndcg_vals = [m_dict[c]["ndcg5"] for c in g_list if c in m_dict]
        p5_vals = [m_dict[c]["p5"] for c in g_list if c in m_dict]
        r5_vals = [m_dict[c]["r5"] for c in g_list if c in m_dict]
        auc_vals = [m_dict[c]["auc"] for c in g_list if c in m_dict and m_dict[c]["auc"] is not None]

        mean_ndcg = np.mean(ndcg_vals) if ndcg_vals else 0.0
        mean_p5 = np.mean(p5_vals) if p5_vals else 0.0
        mean_r5 = np.mean(r5_vals) if r5_vals else 0.0
        mean_auc = np.mean(auc_vals) if auc_vals else 0.0

        print(f"{model_name:<45} | {mean_ndcg:.4f}   | {mean_p5:.4f}   | {mean_r5:.4f}   | {mean_auc:.4f}")

# Save results to json
out_summary = {}
for g_name, g_list in groups:
    out_summary[g_name] = {}
    for model_name, m_key in models:
        m_dict = results[m_key]
        ndcg_vals = [m_dict[c]["ndcg5"] for c in g_list if c in m_dict]
        p5_vals = [m_dict[c]["p5"] for c in g_list if c in m_dict]
        r5_vals = [m_dict[c]["r5"] for c in g_list if c in m_dict]
        auc_vals = [m_dict[c]["auc"] for c in g_list if c in m_dict and m_dict[c]["auc"] is not None]
        out_summary[g_name][m_key] = {
            "ndcg5": float(np.mean(ndcg_vals)),
            "p5": float(np.mean(p5_vals)),
            "r5": float(np.mean(r5_vals)),
            "auc": float(np.mean(auc_vals)) if auc_vals else None
        }

with open("/home/nguyendinhquy/TransMTL_OntoKG-main/experiments/fact_selection_benchmark/reports/three_group_decoupling_audit.json", "w") as f:
    json.dump(out_summary, f, indent=2)

print("\nAudit successfully saved to three_group_decoupling_audit.json")
