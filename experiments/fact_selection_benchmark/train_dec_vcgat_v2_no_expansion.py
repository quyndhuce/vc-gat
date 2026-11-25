#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/train_dec_vcgat_v2_no_expansion.py
=============================================================================
Ablation Experiment: DEC-VCGAT v2 WITHOUT Dynamic Candidate Graph Expansion.
Strictly per user specification:
  - Keep identical DEC-v2 architecture (Decoupled Visual Grounding, Strict Visual
    Saliency Gate, Degree-Debiased Relational Attention, Bounded Delta, Domain-Invariant Gate).
  - DO NOT add candidate facts to the landmark subgraph.
  - DO NOT create cosine Top-2 edges.
  - Candidates outside the subgraph are scored strictly by s_vis (fair visual baseline,
    NO artificial penalty).
  - Candidates inside the subgraph receive full graph-adjusted score s_vis + 0.4 * lambda * delta_s.
  - Retrain from scratch with identical training data (702 train clips), validation data (149 val),
    loss (ListMLE + BCE + Hard-Neg Margin + Gate Reg), optimizer, and 15 epochs.
  - Evaluate on all 800 test clips against the 0.2878 baseline.
=============================================================================
"""

import sys
import math
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.fact_selection_benchmark.dataset_loader import (
    load_normalized_facts,
    load_gold_data,
    compute_ranking_metrics,
    compute_hard_negative_auc,
    HISTORICAL_PREDICATES,
)
from experiments.fact_selection_benchmark.feature_cache import BenchmarkFeatureCache
from experiments.fact_selection_benchmark.train_dec_vcgat_v2 import DecoupledEvidenceCalibratedVCGATv2, listmle_topk_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("DEC-v2-NoExpansion")

CHECKPOINT_DIR = REPO_ROOT / "experiments" / "checkpoints_fact_selection"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = REPO_ROOT / "experiments" / "fact_selection_benchmark" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SEEN_LANDMARKS = {
    "hoan_kiem_lake",
    "temple_of_literature",
    "ba_dinh_square",
    "imperial_citadel_of_thang_long",
    "one_pillar_pagoda",
    "hanoi_old_quarter",
    "west_lake",
}


def score_clip_candidates_no_expansion(
    model: DecoupledEvidenceCalibratedVCGATv2,
    g: Dict[str, Any],
    cand_fids: List[str],
    cand_h_tensor: torch.Tensor,
    feature_cache: BenchmarkFeatureCache,
    v_f_tensor: torch.Tensor,
    v_p_tensor: Optional[torch.Tensor],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scores candidates without expanding the graph:
      - Subgraph nodes receive full DEC-VCGAT v2 score: s_vis + 0.4 * lambda * delta_s
      - Out-of-subgraph nodes receive s_vis directly via model visual scorer (NO penalty)
    Returns:
      cand_scores: torch.Tensor of shape [len(cand_fids)]
      lambda_g: torch.Tensor of lambda values for subgraph nodes
    """
    sub_fids = g["fact_ids"]
    idx_map = g["fid2idx"]

    # 1. Forward pass on the pure landmark subgraph
    h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
    h_sub_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
    adj = g["adj"].to(device)
    rel_adj = g["rel_adj"].to(device)
    degrees = g["degrees"].to(device)

    logits_g, s_vis_g, lambda_g, delta_g = model(h_sub_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)

    # 2. Check if any candidate is outside the subgraph
    missing_indices = [i for i, fid in enumerate(cand_fids) if fid not in idx_map]

    s_vis_missing = None
    if missing_indices:
        # Compute s_vis directly using model visual backbone for out-of-subgraph candidates
        missing_h = cand_h_tensor[missing_indices] # [M, D]
        M = missing_h.size(0)
        q = missing_h.unsqueeze(1)
        kv = v_f_tensor.unsqueeze(0).expand(M, -1, -1)
        attn_out, _ = model.cross_attn(query=q, key=kv, value=kv)
        ctx = model.norm1(attn_out.squeeze(1) + missing_h)
        vis_feat = torch.cat([missing_h, ctx, missing_h * ctx], dim=-1)
        s_vis_missing = model.vis_scorer(vis_feat).squeeze(-1) # [M]

    # 3. Assemble cand_scores
    cand_score_list = []
    miss_ptr = 0
    for i, fid in enumerate(cand_fids):
        if fid in idx_map:
            cand_score_list.append(logits_g[idx_map[fid]])
        else:
            cand_score_list.append(s_vis_missing[miss_ptr])
            miss_ptr += 1

    cand_scores = torch.stack(cand_score_list) # [16]
    return cand_scores, lambda_g


def main():
    device = torch.device("cpu")
    torch.set_num_threads(16)
    LOGGER.info(f"Using device: {device} with 16 CPU threads")

    facts = load_normalized_facts()
    train_clips, train_contrastive = load_gold_data(split_type="train")
    val_clips, val_contrastive = load_gold_data(split_type="val")
    test_clips, test_contrastive = load_gold_data(split_type="test")

    LOGGER.info(f"Dataset: Train={len(train_clips)}, Val={len(val_clips)}, Test={len(test_clips)}")

    feature_cache = BenchmarkFeatureCache(device=device)
    feature_cache.extract_fact_embeddings(facts)
    feature_cache.extract_video_features([])
    rel_subgraphs = torch.load(feature_cache.cache_dir / "landmark_subgraphs_rel.pt", weights_only=False)

    # Initialize model with identical architecture
    model = DecoupledEvidenceCalibratedVCGATv2(
        d_model=768,
        num_heads=4,
        hidden_dim=256,
        delta_bound=0.8,
        degree_damping=0.4,
        dropout=0.1
    ).to(device)

    # Initialize visual backbone from best_cross_attention.pt
    cross_path = CHECKPOINT_DIR / "best_cross_attention.pt"
    if cross_path.exists():
        LOGGER.info(f"Loading pretrained visual backbone weights from {cross_path}")
        cross_weights = torch.load(cross_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in cross_weights.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        LOGGER.info(f"Successfully transferred {len(pretrained_dict)} visual weight tensors.")

    # Differential learning rates
    vis_params = list(model.cross_attn.parameters()) + list(model.norm1.parameters()) + list(model.vis_scorer.parameters())
    graph_params = (
        list(model.saliency_gate.parameters()) +
        list(model.q_proj.parameters()) + list(model.k_proj.parameters()) +
        list(model.v_proj.parameters()) + list(model.out_proj.parameters()) +
        list(model.rel_embed.parameters()) + list(model.rel_k_proj.parameters()) +
        list(model.rel_v_proj.parameters()) + list(model.rel_bias.parameters()) +
        list(model.norm_gat.parameters()) + list(model.graph_delta_mlp.parameters()) +
        list(model.evidence_gate.parameters())
    )

    optimizer = AdamW([
        {"params": vis_params, "lr": 1e-5, "weight_decay": 1e-4},
        {"params": graph_params, "lr": 1.5e-4, "weight_decay": 1e-4}
    ])

    best_val_score = -1.0
    best_state = None
    train_cids = list(train_clips.keys())

    LOGGER.info("Starting training of DEC-VCGAT v2 (NO GRAPH EXPANSION ABLATION)...")
    for epoch in range(1, 16):
        model.train()
        np.random.shuffle(train_cids)

        epoch_loss = 0.0
        n_steps = 0

        for c_id in train_cids:
            recs = train_clips[c_id]
            l_id = recs[0].get("landmark_id", "unknown")
            g = rel_subgraphs.get(l_id)
            v_frames = feature_cache.video_frames.get(c_id)
            v_pooled = feature_cache.video_pooled.get(c_id)
            if g is None or v_frames is None or v_pooled is None:
                continue

            cand_fids = [r["fact_id"] for r in recs]
            pos_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
            if not pos_fids:
                continue

            cand_h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in cand_fids]
            cand_h_tensor = torch.from_numpy(np.array(cand_h_list, dtype=np.float32)).to(device)
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device)

            optimizer.zero_grad()
            cand_scores, lambda_g = score_clip_candidates_no_expansion(
                model=model,
                g=g,
                cand_fids=cand_fids,
                cand_h_tensor=cand_h_tensor,
                feature_cache=feature_cache,
                v_f_tensor=v_f_tensor,
                v_p_tensor=v_p_tensor,
                device=device
            )

            cand_fid2idx = {fid: i for i, fid in enumerate(cand_fids)}
            pos_indices = [cand_fid2idx[fid] for fid in pos_fids]
            hard_neg_indices = [cand_fid2idx[fid] for fid in hard_neg_fids if fid in cand_fid2idx]
            neg_indices = [i for i in range(len(cand_fids)) if i not in pos_indices]

            # 1. ListMLE Loss
            loss_listmle = listmle_topk_loss(cand_scores, pos_indices, list(range(len(cand_fids))), top_k=5, temp=0.5)

            # 2. Hard Negative Margin Loss
            loss_hn = torch.tensor(0.0, device=device)
            if hard_neg_indices and pos_indices:
                p_s = cand_scores[pos_indices].unsqueeze(1)
                hn_s = cand_scores[hard_neg_indices].unsqueeze(0)
                loss_hn = torch.relu(0.8 - (p_s - hn_s)).mean()

            # 3. BCE Loss
            y_true = [1.0] * len(pos_indices) + [0.0] * len(neg_indices)
            preds = torch.cat([cand_scores[pos_indices], cand_scores[neg_indices]])
            targets = torch.tensor(y_true, dtype=torch.float32, device=device)
            bce_loss = F.binary_cross_entropy_with_logits(preds, targets, pos_weight=torch.tensor([3.0], device=device))

            # 4. Gate Regularization
            gate_reg = 0.05 * torch.mean(lambda_g ** 2)

            total_loss = 0.5 * loss_listmle + 0.5 * bce_loss + 0.6 * loss_hn + gate_reg
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            n_steps += 1

        # Validation Loop
        model.eval()
        val_ndcg, val_p5, val_auc = [], [], []
        with torch.no_grad():
            for c_id, recs in val_clips.items():
                l_id = recs[0].get("landmark_id", "unknown")
                g = rel_subgraphs.get(l_id)
                v_frames = feature_cache.video_frames.get(c_id)
                v_pooled = feature_cache.video_pooled.get(c_id)
                if g is None or v_frames is None or v_pooled is None:
                    continue

                cand_fids = [r["fact_id"] for r in recs]
                gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
                hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")

                cand_h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in cand_fids]
                cand_h_tensor = torch.from_numpy(np.array(cand_h_list, dtype=np.float32)).to(device)
                v_f_tensor = torch.from_numpy(v_frames).float().to(device)
                v_p_tensor = torch.from_numpy(v_pooled).float().to(device)

                cand_scores, _ = score_clip_candidates_no_expansion(
                    model=model,
                    g=g,
                    cand_fids=cand_fids,
                    cand_h_tensor=cand_h_tensor,
                    feature_cache=feature_cache,
                    v_f_tensor=v_f_tensor,
                    v_p_tensor=v_p_tensor,
                    device=device
                )
                scores_np = cand_scores.cpu().numpy()
                f_scores = {fid: float(scores_np[i]) for i, fid in enumerate(cand_fids)}

                ranked = sorted(cand_fids, key=lambda fid: f_scores[fid], reverse=True)
                m = compute_ranking_metrics(ranked, gold_fids, cand_fids, top_k_list=[5])
                val_ndcg.append(m.get("nDCG@5", 0.0))
                val_p5.append(m.get("P@5", 0.0))

                auc = compute_hard_negative_auc(f_scores, gold_fids, hard_neg_fids)
                if auc is not None:
                    val_auc.append(auc)

        v_ndcg_m = float(np.mean(val_ndcg))
        v_p5_m = float(np.mean(val_p5))
        v_auc_m = float(np.mean(val_auc)) if val_auc else 0.5
        comp_score = 0.50 * v_ndcg_m + 0.25 * v_p5_m + 0.25 * v_auc_m
        LOGGER.info(f"Epoch {epoch:02d} | Loss: {epoch_loss/max(1, n_steps):.4f} | Val nDCG@5: {v_ndcg_m:.4f} | Val P@5: {v_p5_m:.4f} | Val AUC: {v_auc_m:.4f} | Comp: {comp_score:.4f}")

        if comp_score > best_val_score:
            best_val_score = comp_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, CHECKPOINT_DIR / "best_dec_vcgat_v2_no_expansion.pt")

    LOGGER.info(f"Training completed. Best Validation Score: {best_val_score:.4f}")

    # =========================================================================
    # 800 TEST CLIPS EVALUATION
    # =========================================================================
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    test_ndcg5, test_p5, test_r5, test_aucs = [], [], [], []
    seen_ndcg, unseen_ndcg = [], []
    hist_ndcg5 = []

    # 3-Group tracking
    g1_cids, g2_cids, g3_cids = [], [], []
    group_metrics = {"g1": [], "g2": [], "g3": []}

    for cid, recs in test_clips.items():
        lid = recs[0].get("landmark_id", "unknown")
        g = rel_subgraphs.get(lid)
        f2i = g["fid2idx"]
        has_miss_cand = any(r["fact_id"] not in f2i for r in recs)
        has_miss_gold = any(r["fact_id"] not in f2i and r.get("is_scene_gold") for r in recs)
        if not has_miss_cand:
            g1_cids.append(cid)
        elif not has_miss_gold:
            g2_cids.append(cid)
        else:
            g3_cids.append(cid)

    g1_set = set(g1_cids)
    g2_set = set(g2_cids)
    g3_set = set(g3_cids)

    with torch.no_grad():
        for c_id, recs in test_clips.items():
            l_id = recs[0].get("landmark_id", "unknown")
            g = rel_subgraphs.get(l_id)
            v_frames = feature_cache.video_frames.get(c_id)
            v_pooled = feature_cache.video_pooled.get(c_id)
            if g is None or v_frames is None or v_pooled is None:
                continue

            cand_fids = [r["fact_id"] for r in recs]
            gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")

            cand_h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in cand_fids]
            cand_h_tensor = torch.from_numpy(np.array(cand_h_list, dtype=np.float32)).to(device)
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device)

            cand_scores, _ = score_clip_candidates_no_expansion(
                model=model,
                g=g,
                cand_fids=cand_fids,
                cand_h_tensor=cand_h_tensor,
                feature_cache=feature_cache,
                v_f_tensor=v_f_tensor,
                v_p_tensor=v_p_tensor,
                device=device
            )
            scores_np = cand_scores.cpu().numpy()
            f_scores = {fid: float(scores_np[i]) for i, fid in enumerate(cand_fids)}

            ranked = sorted(cand_fids, key=lambda fid: f_scores[fid], reverse=True)
            m = compute_ranking_metrics(ranked, gold_fids, cand_fids, top_k_list=[5])
            val_ndcg_val = m.get("nDCG@5", 0.0)
            val_p5_val = m.get("P@5", 0.0)
            val_r5_val = m.get("R@5", 0.0)

            test_ndcg5.append(val_ndcg_val)
            test_p5.append(val_p5_val)
            test_r5.append(val_r5_val)

            if l_id in SEEN_LANDMARKS:
                seen_ndcg.append(val_ndcg_val)
            else:
                unseen_ndcg.append(val_ndcg_val)

            auc = compute_hard_negative_auc(f_scores, gold_fids, hard_neg_fids)
            if auc is not None:
                test_aucs.append(auc)

            # Historical
            cand_hist = [f for f in cand_fids if facts.get(f, {}).get("predicate") in HISTORICAL_PREDICATES]
            gold_hist = [f for f in gold_fids if facts.get(f, {}).get("predicate") in HISTORICAL_PREDICATES]
            if cand_hist and gold_hist:
                rank_hist = sorted(cand_hist, key=lambda f: f_scores[f], reverse=True)
                m_h = compute_ranking_metrics(rank_hist, set(gold_hist), cand_hist, top_k_list=[5])
                hist_ndcg5.append(m_h.get("nDCG@5", 0.0))

            # Groups
            if c_id in g1_set:
                group_metrics["g1"].append(val_ndcg_val)
            elif c_id in g2_set:
                group_metrics["g2"].append(val_ndcg_val)
            elif c_id in g3_set:
                group_metrics["g3"].append(val_ndcg_val)

    final_metrics = {
        "overall_ndcg@5": float(np.mean(test_ndcg5)),
        "precision@5": float(np.mean(test_p5)),
        "recall@5": float(np.mean(test_r5)),
        "hard_neg_auc": float(np.mean(test_aucs)) if test_aucs else 0.5,
        "seen_ndcg@5": float(np.mean(seen_ndcg)),
        "unseen_ndcg@5": float(np.mean(unseen_ndcg)),
        "hist_ndcg@5": float(np.mean(hist_ndcg5)) if hist_ndcg5 else 0.0,
        "g1_complete_ndcg@5": float(np.mean(group_metrics["g1"])),
        "g2_neg_only_ndcg@5": float(np.mean(group_metrics["g2"])),
        "g3_gold_miss_ndcg@5": float(np.mean(group_metrics["g3"])),
        "g1_g2_clean_ndcg@5": float(np.mean(group_metrics["g1"] + group_metrics["g2"])),
    }

    LOGGER.info("=" * 80)
    LOGGER.info("DEC-VCGAT v2 (NO GRAPH EXPANSION ABLATION) TEST RESULTS (800 CLIPS):")
    LOGGER.info(f"  Overall nDCG@5                = {final_metrics['overall_ndcg@5']:.4f}")
    LOGGER.info(f"  Precision@5                   = {final_metrics['precision@5']:.4f}")
    LOGGER.info(f"  Recall@5                      = {final_metrics['recall@5']:.4f}")
    LOGGER.info(f"  Hard Negative AUC             = {final_metrics['hard_neg_auc']:.4f}")
    LOGGER.info(f"  Seen nDCG@5 (224 clips)       = {final_metrics['seen_ndcg@5']:.4f}")
    LOGGER.info(f"  Unseen nDCG@5 (576 clips)     = {final_metrics['unseen_ndcg@5']:.4f}")
    LOGGER.info(f"  Historical nDCG@5             = {final_metrics['hist_ndcg@5']:.4f}")
    LOGGER.info(f"  Group 1 (195 Complete clips)  = {final_metrics['g1_complete_ndcg@5']:.4f}")
    LOGGER.info(f"  Group 2 (517 Neg-only clips)  = {final_metrics['g2_neg_only_ndcg@5']:.4f}")
    LOGGER.info(f"  Group 3 (88 Gold-miss clips)  = {final_metrics['g3_gold_miss_ndcg@5']:.4f}")
    LOGGER.info(f"  Group 1+2 (712 Clean clips)   = {final_metrics['g1_g2_clean_ndcg@5']:.4f}")
    LOGGER.info("=" * 80)

    out_file = REPORTS_DIR / "dec_vcgat_v2_no_expansion_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    LOGGER.info(f"Results successfully saved to {out_file}")


if __name__ == "__main__":
    main()
