#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/home/nguyendinhquy/TransMTL_OntoKG-main")
import torch
import numpy as np
from pathlib import Path
from experiments.fact_selection_benchmark.train_dec_vcgat_v2 import DecoupledEvidenceCalibratedVCGATv2, build_clip_graph
from experiments.fact_selection_benchmark.train_calibrated_rvcgat import CalibratedResidualVCGAT
from experiments.fact_selection_benchmark.models.cross_attention_ranker import CrossAttentionRankerModel
from experiments.fact_selection_benchmark.dataset_loader import load_gold_data, load_normalized_facts, compute_ranking_metrics, compute_hard_negative_auc
from experiments.fact_selection_benchmark.feature_cache import BenchmarkFeatureCache

device = torch.device("cuda:0")
facts = load_normalized_facts()
test_clips, _ = load_gold_data(split_type="test")
feature_cache = BenchmarkFeatureCache(device=device)
feature_cache.extract_fact_embeddings(facts)
feature_cache.extract_video_features([])
rel_subgraphs = torch.load(feature_cache.cache_dir / "landmark_subgraphs_rel.pt", weights_only=False)

cross_model = CrossAttentionRankerModel(d_model=768).to(device)
cross_model.load_state_dict(torch.load("/home/nguyendinhquy/TransMTL_OntoKG-main/experiments/checkpoints_fact_selection/best_cross_attention.pt", map_location=device))
cross_model.eval()

cr_model = CalibratedResidualVCGAT(d_model=768, delta_bound=0.8).to(device)
cr_model.load_state_dict(torch.load("/home/nguyendinhquy/TransMTL_OntoKG-main/experiments/checkpoints_fact_selection/best_calibrated_rvcgat.pt", map_location=device))
cr_model.eval()

dec_model = DecoupledEvidenceCalibratedVCGATv2(d_model=768).to(device)
dec_model.load_state_dict(torch.load("/home/nguyendinhquy/TransMTL_OntoKG-main/experiments/checkpoints_fact_selection/best_dec_vcgat_v2.pt", map_location=device))
dec_model.eval()

SEEN = {"hoan_kiem_lake", "temple_of_literature", "ba_dinh_square", "imperial_citadel_of_thang_long", "one_pillar_pagoda", "hanoi_old_quarter", "west_lake"}

res_cross = {"all": [], "seen": [], "unseen": [], "complete_sub": [], "missing_sub": []}
res_cr_fallback = {"all": [], "seen": [], "unseen": [], "complete_sub": [], "missing_sub": []}
res_dec_v2 = {"all": [], "seen": [], "unseen": [], "complete_sub": [], "missing_sub": []}

cids = sorted(list(test_clips.keys()))

with torch.no_grad():
    for c_id in cids:
        recs = test_clips[c_id]
        l_id = recs[0].get("landmark_id", "unknown")
        is_seen = (l_id in SEEN)
        g = rel_subgraphs.get(l_id)
        v_frames = feature_cache.video_frames.get(c_id)
        v_pooled = feature_cache.video_pooled.get(c_id)
        if g is None or v_frames is None:
            continue

        cand_fids = [r["fact_id"] for r in recs]
        gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
        is_complete = all(fid in g["fid2idx"] for fid in cand_fids)

        # 1. Cross Attention
        cand_h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in cand_fids]
        cand_h_tensor = torch.from_numpy(np.array(cand_h_list, dtype=np.float32)).to(device)
        v_f_tensor = torch.from_numpy(v_frames).float().to(device)
        logits_c = cross_model(v_f_tensor, cand_h_tensor).cpu().numpy()
        f_scores_c = {fid: float(logits_c[i]) for i, fid in enumerate(cand_fids)}
        ranked_c = sorted(cand_fids, key=lambda fid: f_scores_c[fid], reverse=True)
        m_c = compute_ranking_metrics(ranked_c, gold_fids, cand_fids, top_k_list=[5])["nDCG@5"]

        # 2. CR-VCGAT with passive fallback
        q_cand = cand_h_tensor.unsqueeze(1)
        kv_cand = v_f_tensor.unsqueeze(0).expand(len(cand_fids), -1, -1)
        attn_cand, _ = cr_model.cross_attn(query=q_cand, key=kv_cand, value=kv_cand)
        ctx_cand = cr_model.norm1(attn_cand.squeeze(1) + cand_h_tensor)
        vis_feat_cand = torch.cat([cand_h_tensor, ctx_cand, cand_h_tensor * ctx_cand], dim=-1)
        base_vis_dict = {fid: float(cr_model.vis_scorer(vis_feat_cand).squeeze(-1)[i]) for i, fid in enumerate(cand_fids)}

        sub_fids = g["fact_ids"]
        h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
        h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
        adj = g["adj"].to(device)
        rel_adj = g["rel_adj"].to(device)
        degrees = g["degrees"].to(device)
        v_p_tensor = torch.from_numpy(v_pooled).float().to(device) if v_pooled is not None else None
        logits_cr, s_vis, lambda_i, delta_s = cr_model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
        graph_scores = (s_vis + 0.4 * lambda_i * delta_s).cpu().numpy()
        idx_map = g["fid2idx"]
        f_scores_cr = {fid: float(graph_scores[idx_map[fid]]) if fid in idx_map else base_vis_dict[fid] for fid in cand_fids}
        ranked_cr = sorted(cand_fids, key=lambda fid: f_scores_cr[fid], reverse=True)
        m_cr = compute_ranking_metrics(ranked_cr, gold_fids, cand_fids, top_k_list=[5])["nDCG@5"]

        # 3. DEC-VCGAT v2 with Dynamic Graph Inclusion
        h_t_v2, adj_v2, rel_adj_v2, deg_v2, all_fids_v2, fid2idx_v2 = build_clip_graph(
            g, cand_fids, feature_cache.fact_embeddings, device
        )
        logits_d, _, _, _ = dec_model(h_t_v2, adj_v2, rel_adj_v2, deg_v2, v_f_tensor, v_p_tensor)
        scores_d = logits_d.cpu().numpy()
        f_scores_d = {fid: float(scores_d[fid2idx_v2[fid]]) for fid in cand_fids}
        ranked_d = sorted(cand_fids, key=lambda fid: f_scores_d[fid], reverse=True)
        m_d = compute_ranking_metrics(ranked_d, gold_fids, cand_fids, top_k_list=[5])["nDCG@5"]

        # Append
        for (m, d) in [(m_c, res_cross), (m_cr, res_cr_fallback), (m_d, res_dec_v2)]:
            d["all"].append(m)
            if is_seen:
                d["seen"].append(m)
            else:
                d["unseen"].append(m)
            if is_complete:
                d["complete_sub"].append(m)
            else:
                d["missing_sub"].append(m)

print("=== EXACT DECOUPLING AUDIT ACROSS ALL SLICES (nDCG@5) ===")
models = [
    ("1. Cross-Attention Ranker (Non-Graph Baseline)", res_cross),
    ("2. CR-VCGAT (Passive Fallback to s_vis)", res_cr_fallback),
    ("3. DEC-VCGAT v2 (Dynamic Candidate Graph + ListMLE)", res_dec_v2)
]
for name, d in models:
    m_all = np.mean(d['all'])
    m_seen = np.mean(d['seen'])
    m_unseen = np.mean(d['unseen'])
    m_comp = np.mean(d['complete_sub'])
    m_miss = np.mean(d['missing_sub'])
    print(f"{name}:")
    print(f"   Overall (800 clips)          : {m_all:.4f}")
    print(f"   Seen Landmarks (224 clips)   : {m_seen:.4f}")
    print(f"   Unseen Landmarks (576 clips) : {m_unseen:.4f}")
    print(f"   Complete Subgraph (195 clips): {m_comp:.4f}")
    print(f"   Missing Subgraph (605 clips) : {m_miss:.4f}")
