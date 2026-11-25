#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/evaluate_landmark_grounding_retrieval_and_calibration.py
=============================================================================
Rigorous Landmark Grounding Benchmark:
Compares Visual Retrieval & Grounding Baselines on 1,119 Heritage Test Videos.
Metrics:
- R@1 (%)  : Top-1 Recall / Accuracy
- R@5 (%)  : Top-5 Recall
- MRR      : Mean Reciprocal Rank (1/rank)
- ECE (%)  : Expected Calibration Error (M=10 bins)
=============================================================================
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
data_dir = repo_root / "data/unified_coco_dataset"
feat_dir = repo_root / "experiments/features"
out_dir = repo_root / "experiments/outputs"

# Load Annotations
with open(data_dir / "annotations/train_coco.json", "r", encoding="utf-8") as f:
    train_coco = json.load(f)
with open(data_dir / "annotations/test_coco.json", "r", encoding="utf-8") as f:
    test_coco = json.load(f)

train_anns = train_coco["annotations"]
test_anns = test_coco["annotations"]
N_te = len(test_anns)
N_tr = len(train_anns)

PARENT_CLASSES_30 = [
    'Ba Dinh Square', 'Bat Trang Pottery Village', 'Botanical Garden', 'Dien Hong Park',
    'Dong Da Mound', 'Ha Noi Station', 'Hanoi Cuisine', 'Hanoi Museum', 'Hanoi Old Quarter',
    'Hoan Kiem Lake', 'Huu Tiep Lake', 'Imperial Citadel of Thang Long', 'Lake Ba Mau',
    'Lake Thien Quang', 'Le Nin Park', 'Ngu Xa Traditional Bronze Casting Workshop',
    'One Pillar Pagoda', 'Temple of Literature', 'Thang Long Water Puppetry Theatre',
    'The Government Guest House', 'Thong Nhat park', 'Thu Le Park', 'Truc Bach Lake',
    'Van Phuc Silk Village', 'Vietnam Museum of Ethnology', 'Vietnam National Museum of History',
    'Vietnam National Puppetry Theatre', "Vietnam Women's Museum", 'West Lake', 'Yen So park'
]
parent2idx = {p: i for i, p in enumerate(PARENT_CLASSES_30)}
gt_parents = [a["parent_landmark"] for a in test_anns]
gt_indices = np.array([parent2idx[p] for p in gt_parents])
y_gallery = np.array([parent2idx[a["parent_landmark"]] for a in train_anns])

def compute_ece(probs: np.ndarray, gt_idx: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across n_bins."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == gt_idx)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(gt_idx)
    
    for i in range(n_bins):
        b_low, b_high = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > b_low) & (confidences <= b_high) if i > 0 else (confidences >= b_low) & (confidences <= b_high)
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / N) * np.abs(bin_acc - bin_conf)
            
    return float(ece * 100.0)

def evaluate_predictions_from_probs(probs: np.ndarray, gt_idx: np.ndarray):
    """Computes R@1, R@5, MRR, ECE from (N, 30) class probability matrix."""
    N = len(gt_idx)
    ranks = []
    
    for i in range(N):
        p_row = probs[i]
        gt = gt_idx[i]
        # Sort classes descending
        ranked_classes = np.argsort(-p_row)
        rank = np.where(ranked_classes == gt)[0][0] + 1
        ranks.append(rank)
        
    ranks = np.array(ranks)
    r1 = float(np.mean(ranks == 1) * 100.0)
    r5 = float(np.mean(ranks <= 5) * 100.0)
    mrr = float(np.mean(1.0 / ranks))
    ece = compute_ece(probs, gt_idx)
    
    return {"R@1": r1, "R@5": r5, "MRR": mrr, "ECE": ece}

def evaluate_1nn_from_features(q_feat: np.ndarray, g_feat: np.ndarray, gt_idx: np.ndarray, y_g: np.ndarray, tau: float = 0.07):
    """Computes R@1, R@5, MRR, ECE for 1-NN Cosine Instance Retrieval."""
    # Normalize
    q_norm = q_feat / (np.linalg.norm(q_feat, axis=1, keepdims=True) + 1e-8)
    g_norm = g_feat / (np.linalg.norm(g_feat, axis=1, keepdims=True) + 1e-8)
    
    # Cosine sim matrix (N_q, N_gallery)
    sim = np.dot(q_norm, g_norm.T)
    N_q = len(gt_idx)
    
    # Class-level scores: S(q, c) = max_{j in gallery, y_j == c} sim(q, j)
    class_scores = np.full((N_q, 30), -1e9, dtype=np.float32)
    for c in range(30):
        c_mask = (y_g == c)
        if np.any(c_mask):
            class_scores[:, c] = np.max(sim[:, c_mask], axis=1)
            
    # Softmax to compute probability distribution for ECE
    exp_s = np.exp((class_scores - np.max(class_scores, axis=1, keepdims=True)) / tau)
    probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
    
    ranks = []
    for i in range(N_q):
        # In instance retrieval, rank is determined by the first appearance of each class
        sim_row = sim[i]
        ranked_gallery_idx = np.argsort(-sim_row)
        seen_parents = []
        for g_i in ranked_gallery_idx:
            c = y_g[g_i]
            if c not in seen_parents:
                seen_parents.append(c)
            if len(seen_parents) == 30:
                break
        gt = gt_idx[i]
        rank = seen_parents.index(gt) + 1 if gt in seen_parents else 30
        ranks.append(rank)
        
    ranks = np.array(ranks)
    r1 = float(np.mean(ranks == 1) * 100.0)
    r5 = float(np.mean(ranks <= 5) * 100.0)
    mrr = float(np.mean(1.0 / ranks))
    ece = compute_ece(probs, gt_idx)
    
    return {"R@1": r1, "R@5": r5, "MRR": mrr, "ECE": ece}

# =============================================================================
# 1. EVALUATE ALL 1-NN RETRIEVAL BASELINES
# =============================================================================
results = {}

print("Evaluating 1-NN Image Retrieval Baselines...")

# B1: DINOv2
te_dino = np.load(feat_dir / "test_dinov2.npy")
tr_dino = np.load(feat_dir / "train_dinov2.npy")
results["DINOv2 Base (Cosine 1-NN)"] = evaluate_1nn_from_features(te_dino, tr_dino, gt_indices, y_gallery)

# B2: SigLIP 2 (1-frame)
te_sig1 = np.load(feat_dir / "test_siglip_1f.npy")
tr_sig1 = np.load(feat_dir / "train_siglip_1f.npy")
results["SigLIP 2 (1-frame 1-NN)"] = evaluate_1nn_from_features(te_sig1, tr_sig1, gt_indices, y_gallery)

# B3: SigLIP 2 (6-frame Mean-Pool)
te_sig6 = np.load(feat_dir / "test_siglip_6f.npy")
tr_sig6 = np.load(feat_dir / "train_siglip_6f.npy")
results["SigLIP 2 (6-frame Pool 1-NN)"] = evaluate_1nn_from_features(te_sig6, tr_sig6, gt_indices, y_gallery)

# B5: Whole-frame Concept 285-d
te_c_w = np.load(feat_dir / "test_concepts_whole.npy")
tr_c_w = np.load(feat_dir / "train_concepts_whole.npy")
results["Concept Whole-Frame (285-d 1-NN)"] = evaluate_1nn_from_features(te_c_w, tr_c_w, gt_indices, y_gallery)

# B6: BBox Concept 285-d
te_c_b = np.load(feat_dir / "test_bbox_noisy_or_285d.npy")
tr_c_b = np.load(feat_dir / "train_bbox_noisy_or_285d.npy")
results["Concept BBox Max-Pool (285-d 1-NN)"] = evaluate_1nn_from_features(te_c_b, tr_c_b, gt_indices, y_gallery)

# B10: Fused SigLIP 6f + BBox Concept
with open(out_dir / "results_baselines_1nn.json", "r") as f:
    nn_json = json.load(f)
w_tuned = nn_json.get("B10_ScoreFusion_SigLIP_BBoxConcept", {}).get("tuned_w", 0.35)

# Normalized fusion of SigLIP 6f + BBox Concept
te_s_norm = te_sig6 / (np.linalg.norm(te_sig6, axis=1, keepdims=True) + 1e-8)
tr_s_norm = tr_sig6 / (np.linalg.norm(tr_sig6, axis=1, keepdims=True) + 1e-8)
sim_s = np.dot(te_s_norm, tr_s_norm.T)

te_b_norm = te_c_b / (np.linalg.norm(te_c_b, axis=1, keepdims=True) + 1e-8)
tr_b_norm = tr_c_b / (np.linalg.norm(tr_c_b, axis=1, keepdims=True) + 1e-8)
sim_b = np.dot(te_b_norm, tr_b_norm.T)

sim_fused = (1.0 - w_tuned) * sim_s + w_tuned * sim_b
class_scores_fused = np.full((N_te, 30), -1e9, dtype=np.float32)
for c in range(30):
    c_mask = (y_gallery == c)
    if np.any(c_mask):
        class_scores_fused[:, c] = np.max(sim_fused[:, c_mask], axis=1)
exp_sf = np.exp((class_scores_fused - np.max(class_scores_fused, axis=1, keepdims=True)) / 0.07)
probs_fused = exp_sf / np.sum(exp_sf, axis=1, keepdims=True)

ranks_fused = []
for i in range(N_te):
    sim_row = sim_fused[i]
    ranked_gallery_idx = np.argsort(-sim_row)
    seen_parents = []
    for g_i in ranked_gallery_idx:
        c = y_gallery[g_i]
        if c not in seen_parents: seen_parents.append(c)
        if len(seen_parents) == 30: break
    gt = gt_indices[i]
    rank = seen_parents.index(gt) + 1 if gt in seen_parents else 30
    ranks_fused.append(rank)
ranks_fused = np.array(ranks_fused)
results["Score Fusion (SigLIP 6f + BBox Concept)"] = {
    "R@1": float(np.mean(ranks_fused == 1) * 100.0),
    "R@5": float(np.mean(ranks_fused <= 5) * 100.0),
    "MRR": float(np.mean(1.0 / ranks_fused)),
    "ECE": compute_ece(probs_fused, gt_indices)
}

# =============================================================================
# 2. EVALUATE SUPERVISED NEURAL NETS & PROPOSED ENSEMBLE
# =============================================================================
print("Evaluating Supervised Neural Grounding Models...")
with open(out_dir / "results_baselines_mlp.json", "r") as f:
    mlp_json = json.load(f)

mlp_names = {
    "C1_DINOv2_MLP": "DINOv2 (768-d) + Supervised MLP",
    "C2_SigLIP6f_MLP": "SigLIP 2 (6-frame) + Supervised MLP",
    "C3_WholeConcept_MLP": "Concept Whole-Frame + MLP",
    "C4_BBoxConcept_MLP": "Concept BBox Max-Pool + MLP",
    "C5_MultiTask_MLP": "MultiTask MLP (1,053-d)",
    "C6_LateFusion_SigLIP_Concept_MLP": "Late Fusion (SigLIP 6f + Concepts) MLP",
    "C7_Full_Proposed_System": "⭐ HỆ THỐNG ĐỀ XUẤT (SuperEnsemble + Geo-Priors)"
}

for k, name in mlp_names.items():
    if k in mlp_json and "avg_probs" in mlp_json[k]:
        probs = np.array(mlp_json[k]["avg_probs"])
        results[name] = evaluate_predictions_from_probs(probs, gt_indices)

# =============================================================================
# 3. PRINT FORMATTED TABLE & SAVE RESULTS
# =============================================================================
print("\n" + "═" * 90)
print(f"  🏆 BẢNG ĐÁNH GIÁ ĐỘC LẬP: LANDMARK GROUNDING (TRUY XUẤT & ĐỊNH DANH TỪ VIDEO)")
print("  Đánh giá trên 1,119 video test scenes thuộc 30 địa danh di sản Hà Nội")
print("═" * 90)
print(f"{'Method / Model Architecture':<46} | {'R@1 ↑ (%)':<10} | {'R@5 ↑ (%)':<10} | {'MRR ↑':<8} | {'ECE ↓ (%)':<10}")
print("─" * 90)

for m_name, m_dict in results.items():
    r1_str = f"{m_dict['R@1']:6.2f}%"
    r5_str = f"{m_dict['R@5']:6.2f}%"
    mrr_str = f"{m_dict['MRR']:.4f}"
    ece_str = f"{m_dict['ECE']:6.2f}%"
    print(f"{m_name:<46} | {r1_str:<10} | {r5_str:<10} | {mrr_str:<8} | {ece_str:<10}")

print("═" * 90)

# Save JSON
out_file = out_dir / "landmark_grounding_r1_r5_mrr_ece_benchmark.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✓ Saved complete landmark grounding benchmark to {out_file}")

