#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/home/nguyendinhquy/TransMTL_OntoKG-main")
import torch
import numpy as np
import scipy.stats as stats
from pathlib import Path
from experiments.fact_selection_benchmark.train_dec_vcgat_v2 import DecoupledEvidenceCalibratedVCGATv2, build_clip_graph
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

dec_model = DecoupledEvidenceCalibratedVCGATv2(d_model=768).to(device)
dec_model.load_state_dict(torch.load("/home/nguyendinhquy/TransMTL_OntoKG-main/experiments/checkpoints_fact_selection/best_dec_vcgat_v2.pt", map_location=device))
dec_model.eval()

SEEN = {"hoan_kiem_lake", "temple_of_literature", "ba_dinh_square", "imperial_citadel_of_thang_long", "one_pillar_pagoda", "hanoi_old_quarter", "west_lake"}

cids = sorted(list(test_clips.keys()))
x_ndcg, y_ndcg = [], []
x_p5, y_p5 = [], []
x_auc, y_auc = [], []
seen_mask = []

with torch.no_grad():
    for c_id in cids:
        recs = test_clips[c_id]
        l_id = recs[0].get("landmark_id", "unknown")
        seen_mask.append(l_id in SEEN)
        g = rel_subgraphs.get(l_id)
        v_frames = feature_cache.video_frames.get(c_id)
        v_pooled = feature_cache.video_pooled.get(c_id)
        if g is None or v_frames is None:
            continue

        cand_fids = [r["fact_id"] for r in recs]
        gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
        hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")

        # Cross Attention
        cand_h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in cand_fids]
        cand_h_tensor = torch.from_numpy(np.array(cand_h_list, dtype=np.float32)).to(device)
        v_f_tensor = torch.from_numpy(v_frames).float().to(device)
        logits_c = cross_model(v_f_tensor, cand_h_tensor).cpu().numpy()
        f_scores_c = {fid: float(logits_c[i]) for i, fid in enumerate(cand_fids)}
        ranked_c = sorted(cand_fids, key=lambda fid: f_scores_c[fid], reverse=True)
        m_c = compute_ranking_metrics(ranked_c, gold_fids, cand_fids, top_k_list=[5])
        y_ndcg.append(m_c.get("nDCG@5", 0.0))
        y_p5.append(m_c.get("P@5", 0.0))
        auc_c = compute_hard_negative_auc(f_scores_c, gold_fids, hard_neg_fids)

        # DEC-VCGAT v2
        h_tensor, adj, rel_adj, degrees, all_fids, fid2idx = build_clip_graph(
            g, cand_fids, feature_cache.fact_embeddings, device
        )
        v_p_tensor = torch.from_numpy(v_pooled).float().to(device) if v_pooled is not None else None
        logits_d, _, _, _ = dec_model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
        scores_d = logits_d.cpu().numpy()
        f_scores_d = {fid: float(scores_d[fid2idx[fid]]) for fid in cand_fids}
        ranked_d = sorted(cand_fids, key=lambda fid: f_scores_d[fid], reverse=True)
        m_d = compute_ranking_metrics(ranked_d, gold_fids, cand_fids, top_k_list=[5])
        x_ndcg.append(m_d.get("nDCG@5", 0.0))
        x_p5.append(m_d.get("P@5", 0.0))
        auc_d = compute_hard_negative_auc(f_scores_d, gold_fids, hard_neg_fids)

        if auc_c is not None and auc_d is not None:
            x_auc.append(auc_d)
            y_auc.append(auc_c)

vec_x_ndcg = np.array(x_ndcg)
vec_y_ndcg = np.array(y_ndcg)
vec_x_p5 = np.array(x_p5)
vec_y_p5 = np.array(y_p5)
vec_x_auc = np.array(x_auc)
vec_y_auc = np.array(y_auc)
s_mask = np.array(seen_mask)
u_mask = ~s_mask

def run_boot(vx, vy, name):
    diff = vx - vy
    n = len(diff)
    rng = np.random.default_rng(42)
    boots = np.mean(diff[rng.integers(0, n, size=(10000, n))], axis=1)
    ci = (np.percentile(boots, 2.5), np.percentile(boots, 97.5))
    t_stat, p_t = stats.ttest_rel(vx, vy)
    w_res = stats.wilcoxon(vx, vy, zero_method="pratt")
    cohen_d = np.mean(diff) / np.std(diff, ddof=1)
    sig = "p < 0.001 ***" if min(p_t, w_res.pvalue) < 0.001 else ("p < 0.01 **" if min(p_t, w_res.pvalue) < 0.01 else ("p < 0.05 *" if min(p_t, w_res.pvalue) < 0.05 else "n.s."))
    print(f"[{name}] Baseline={np.mean(vy):.4f} | Proposed={np.mean(vx):.4f} | Delta={np.mean(diff):+.4f} | 95% CI=[{ci[0]:+.4f}, {ci[1]:+.4f}] | d={cohen_d:.3f} | p_t={p_t:.4e} | p_w={w_res.pvalue:.4e} ({sig})")

print("=== DEFINITIVE STATISTICAL SIGNIFICANCE FOR DEC-VCGAT v2 (10,000 BOOTSTRAP RESAMPLES) ===")
run_boot(vec_x_ndcg, vec_y_ndcg, "Overall nDCG@5 (800 Clips)")
run_boot(vec_x_p5, vec_y_p5, "Precision@5 (800 Clips)")
run_boot(vec_x_ndcg[s_mask], vec_y_ndcg[s_mask], "Seen Landmarks nDCG@5 (301 Clips)")
run_boot(vec_x_ndcg[u_mask], vec_y_ndcg[u_mask], "Unseen Landmarks nDCG@5 (499 Clips)")
run_boot(vec_x_auc, vec_y_auc, "Hard Negative AUC (676 Clips)")
