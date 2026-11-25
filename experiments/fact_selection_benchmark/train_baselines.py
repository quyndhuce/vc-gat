#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/train_baselines.py
=============================================================================
Unified Training Harness for Supervised Fact Selection Models:
  1. SigLIP2-FT (Supervised Ranking Head)
  2. InternVideo2-FT (Supervised Temporal Head)
  3. Cross-Attention Ranker
  4. GATv2 + Video Fusion
  5. Graphormer + Video Fusion
  6. VC-GAT Gold-Supervised (Multi-Objective Contrastive & Hard-Negative Aware)

Training Rules & Protocols:
  - Identical Splits: Train (702 clips, 33,789 pairs), Val (149 clips, 16,016 pairs)
  - Identical Candidates: Normalized 3,234 facts pool
  - Multi-Objective Loss: Weighted BCE + Same-Scene Hard Negative Margin + Cross-Scene Contrastive Loss
  - Model Selection: Combined validation nDCG@5, Hard Negative AUC, and Contrastive Accuracy
=============================================================================
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.fact_selection_benchmark.dataset_loader import (
    load_normalized_facts,
    load_clip_manifests,
    load_gold_data,
    load_contrastive_pairs,
    compute_ranking_metrics,
    compute_hard_negative_auc,
    compute_contrastive_accuracy,
)
from experiments.fact_selection_benchmark.feature_cache import BenchmarkFeatureCache
from experiments.fact_selection_benchmark.models.siglip2_ranker import SigLIP2RankingHead
from experiments.fact_selection_benchmark.models.internvideo2_ranker import InternVideo2RankingHead
from experiments.fact_selection_benchmark.models.cross_attention_ranker import CrossAttentionRankerModel
from experiments.fact_selection_benchmark.models.gatv2_ranker import GATv2VideoFusionModel
from experiments.fact_selection_benchmark.models.graphormer_ranker import GraphormerVideoFusionModel
from experiments.fact_selection_benchmark.models.vcgat_gold import VCGATGoldModel

CHECKPOINT_DIR = REPO_ROOT / "experiments" / "checkpoints_fact_selection"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("TrainBaselines")


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_single_model(
    model_name: str,
    model: nn.Module,
    train_clip_records: Dict[str, List[Dict[str, Any]]],
    val_clip_records: Dict[str, List[Dict[str, Any]]],
    train_contrastive_pairs: List[Dict[str, Any]],
    val_contrastive_pairs: List[Dict[str, Any]],
    feature_cache: BenchmarkFeatureCache,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 6
) -> nn.Module:
    """Trains a model with combined BCE, Hard-Negative margin, and Cross-Scene Contrastive loss."""
    LOGGER.info(f"\n{'='*70}\nTRAINING MODEL: [{model_name}]\n{'='*70}")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_score = -1.0
    best_weights = None
    no_improve = 0

    train_cids = list(train_clip_records.keys())
    val_cids = list(val_clip_records.keys())

    active_train_cids = [c for c in train_cids if any(r.get("is_scene_gold") for r in train_clip_records[c])]
    active_val_cids = [c for c in val_cids if any(r.get("is_scene_gold") for r in val_clip_records[c])]

    LOGGER.info(f"Active training clips: {len(active_train_cids)}, Active val clips: {len(active_val_cids)}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        np.random.shuffle(active_train_cids)

        for c_id in active_train_cids:
            recs = train_clip_records[c_id]
            l_id = recs[0].get("landmark_id", "unknown")

            v_pooled = feature_cache.video_pooled.get(c_id)
            v_frames = feature_cache.video_frames.get(c_id)
            if v_pooled is None:
                continue

            g = feature_cache.subgraphs.get(l_id)
            if model_name in ["gatv2", "graphormer", "vcgat_gold"] and g is None:
                continue

            pos_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
            all_neg_fids = set(r["fact_id"] for r in recs if not r.get("is_scene_gold"))
            if not pos_fids:
                continue

            optimizer.zero_grad()

            if model_name in ["gatv2", "graphormer", "vcgat_gold"]:
                sub_fids = g["fact_ids"]
                h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
                h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
                v_tensor = torch.from_numpy(v_pooled).float().to(device)

                if model_name == "gatv2":
                    edge_index = g["edge_index"].to(device)
                    logits = model(h_tensor, edge_index, v_tensor)
                elif model_name == "graphormer":
                    sp = g["shortest_path"].to(device)
                    deg = g["degrees"].to(device)
                    logits = model(h_tensor, sp, deg, v_tensor)
                else:  # vcgat_gold
                    adj = g["adj"].to(device)
                    logits = model(h_tensor, adj, v_tensor)

                idx_map = g["fid2idx"]
                pos_idx = [idx_map[fid] for fid in pos_fids if fid in idx_map]
                hard_neg_idx = [idx_map[fid] for fid in hard_neg_fids if fid in idx_map]
                neg_idx = [idx_map[fid] for fid in list(all_neg_fids)[:len(pos_fids) * 4] if fid in idx_map]

                if not pos_idx or not neg_idx:
                    continue

                y_true = [1.0] * len(pos_idx) + [0.0] * len(neg_idx)
                preds = torch.cat([logits[pos_idx], logits[neg_idx]])
                targets = torch.tensor(y_true, dtype=torch.float32, device=device)
                bce_loss = F.binary_cross_entropy_with_logits(preds, targets, pos_weight=torch.tensor([4.0], device=device))

                # Hard negative margin loss for VC-GAT
                if model_name == "vcgat_gold" and hard_neg_idx:
                    p_exp = logits[pos_idx].unsqueeze(1)
                    hn_exp = logits[hard_neg_idx].unsqueeze(0)
                    hn_loss = torch.relu(0.8 - (p_exp - hn_exp)).mean()
                    loss = bce_loss + 1.0 * hn_loss
                else:
                    pos_logits = logits[pos_idx]
                    neg_logits = logits[neg_idx]
                    p_exp = pos_logits.unsqueeze(1)
                    n_exp = neg_logits.unsqueeze(0)
                    rank_loss = torch.relu(0.5 - (p_exp - n_exp)).mean()
                    loss = bce_loss + 0.5 * rank_loss

            elif model_name == "siglip2_ft":
                cand_fids = list(pos_fids) + list(all_neg_fids)[:len(pos_fids) * 4]
                t_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in cand_fids]
                t_tensor = torch.from_numpy(np.array(t_list, dtype=np.float32)).to(device)
                v_tensor = torch.from_numpy(v_pooled).float().to(device)
                logits = model(v_tensor, t_tensor)

                targets = torch.tensor([1.0 if fid in pos_fids else 0.0 for fid in cand_fids], dtype=torch.float32, device=device)
                loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=torch.tensor([4.0], device=device))

            elif model_name in ["internvideo2_ft", "cross_attention"]:
                cand_fids = list(pos_fids) + list(all_neg_fids)[:len(pos_fids) * 4]
                t_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in cand_fids]
                t_tensor = torch.from_numpy(np.array(t_list, dtype=np.float32)).to(device)
                v_frames_tensor = torch.from_numpy(v_frames).float().to(device)
                logits = model(v_frames_tensor, t_tensor)

                targets = torch.tensor([1.0 if fid in pos_fids else 0.0 for fid in cand_fids], dtype=torch.float32, device=device)
                loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=torch.tensor([4.0], device=device))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Multi-task contrastive pairs step for VC-GAT
        if model_name == "vcgat_gold" and train_contrastive_pairs:
            np.random.shuffle(train_contrastive_pairs)
            for p in train_contrastive_pairs[:500]:
                c_a = p.get("clip_a") or (p.get("clip_A", {}).get("clip_id") if isinstance(p.get("clip_A"), dict) else None)
                c_b = p.get("clip_b") or (p.get("clip_B", {}).get("clip_id") if isinstance(p.get("clip_B"), dict) else None)
                f_id = p.get("fact_id") or p.get("fact_a")
                l_id = p.get("landmark_id")

                v_a = feature_cache.video_pooled.get(c_a)
                v_b = feature_cache.video_pooled.get(c_b)
                g = feature_cache.subgraphs.get(l_id)
                if v_a is None or v_b is None or g is None or f_id not in g["fid2idx"]:
                    continue

                f_idx = g["fid2idx"][f_id]
                sub_fids = g["fact_ids"]
                h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
                h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
                adj = g["adj"].to(device)

                optimizer.zero_grad()
                s_a = model(h_tensor, adj, torch.from_numpy(v_a).float().to(device))[f_idx]
                s_b = model(h_tensor, adj, torch.from_numpy(v_b).float().to(device))[f_idx]

                cont_loss = torch.relu(1.0 - (s_a - s_b))
                if cont_loss > 0:
                    cont_loss.backward()
                    optimizer.step()

        # Validation evaluation
        model.eval()
        val_ndcg_list = []
        val_auc_list = []
        clip_fact_scores = {}

        with torch.no_grad():
            for c_id in active_val_cids:
                recs = val_clip_records[c_id]
                l_id = recs[0].get("landmark_id", "unknown")
                gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
                hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
                cand_fids = [r["fact_id"] for r in recs]

                v_pooled = feature_cache.video_pooled.get(c_id)
                v_frames = feature_cache.video_frames.get(c_id)
                if v_pooled is None:
                    continue

                if model_name in ["gatv2", "graphormer", "vcgat_gold"]:
                    g = feature_cache.subgraphs.get(l_id)
                    if g is None:
                        continue
                    sub_fids = g["fact_ids"]
                    h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
                    h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
                    v_tensor = torch.from_numpy(v_pooled).float().to(device)

                    if model_name == "gatv2":
                        logits = model(h_tensor, g["edge_index"].to(device), v_tensor)
                    elif model_name == "graphormer":
                        logits = model(h_tensor, g["shortest_path"].to(device), g["degrees"].to(device), v_tensor)
                    else:
                        logits = model(h_tensor, g["adj"].to(device), v_tensor)

                    fid2score = {fid: float(logits[g["fid2idx"][fid]]) for fid in cand_fids if fid in g["fid2idx"]}

                elif model_name == "siglip2_ft":
                    t_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in cand_fids]
                    t_tensor = torch.from_numpy(np.array(t_list, dtype=np.float32)).to(device)
                    v_tensor = torch.from_numpy(v_pooled).float().to(device)
                    logits = model(v_tensor, t_tensor)
                    fid2score = {fid: float(logits[i]) for i, fid in enumerate(cand_fids)}

                elif model_name in ["internvideo2_ft", "cross_attention"]:
                    t_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in cand_fids]
                    t_tensor = torch.from_numpy(np.array(t_list, dtype=np.float32)).to(device)
                    v_frames_tensor = torch.from_numpy(v_frames).float().to(device)
                    logits = model(v_frames_tensor, t_tensor)
                    fid2score = {fid: float(logits[i]) for i, fid in enumerate(cand_fids)}

                for fid, s in fid2score.items():
                    clip_fact_scores[(c_id, fid)] = s

                ranked_fids = sorted(cand_fids, key=lambda fid: fid2score.get(fid, -999.0), reverse=True)
                m = compute_ranking_metrics(ranked_fids, gold_fids, cand_fids, top_k_list=[5])
                val_ndcg_list.append(m["nDCG@5"])

                auc = compute_hard_negative_auc(fid2score, gold_fids, hard_neg_fids)
                if auc is not None:
                    val_auc_list.append(auc)

        mean_val_ndcg = float(np.mean(val_ndcg_list)) if val_ndcg_list else 0.0
        mean_val_auc = float(np.mean(val_auc_list)) if val_auc_list else 0.5
        val_cont_acc = compute_contrastive_accuracy(clip_fact_scores, val_contrastive_pairs)

        # Composite score
        composite_score = mean_val_ndcg + 0.3 * mean_val_auc + 0.2 * (val_cont_acc / 100.0)

        LOGGER.info(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_loss:.2f} | "
            f"Val nDCG@5: {mean_val_ndcg:.4f} | Hard Neg AUC: {mean_val_auc:.4f} | "
            f"Contrastive Acc: {val_cont_acc:.1f}% | Score: {composite_score:.4f}"
        )

        if composite_score > best_score:
            best_score = composite_score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            LOGGER.info(f"  --> [*] New best model (Score: {best_score:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                LOGGER.info(f"Early stopping at epoch {epoch}.")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
        save_path = CHECKPOINT_DIR / f"best_{model_name}.pt"
        torch.save(best_weights, save_path)
        LOGGER.info(f"Saved best model [{model_name}] -> {save_path}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "siglip2_ft", "internvideo2_ft", "cross_attention", "gatv2", "graphormer", "vcgat_gold"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = get_device()
    LOGGER.info(f"Training on device: {device}")

    facts = load_normalized_facts()
    test_manifest, train_val_manifest = load_clip_manifests()
    all_manifest_list = list(test_manifest.values()) + list(train_val_manifest.values())

    feature_cache = BenchmarkFeatureCache(device=device)
    feature_cache.extract_fact_embeddings(facts)
    feature_cache.extract_video_features(all_manifest_list)
    feature_cache.build_landmark_subgraphs(facts)

    train_clip_records, _ = load_gold_data(split_type="train")
    val_clip_records, _ = load_gold_data(split_type="val")
    all_contrastive_pairs = load_contrastive_pairs(split_type="train_val")
    train_cids = set(train_clip_records.keys())
    val_cids = set(val_clip_records.keys())

    train_contrastive_pairs = [
        p for p in all_contrastive_pairs
        if (p.get("clip_a") in train_cids or p.get("clip_A", {}).get("clip_id") in train_cids)
        and (p.get("clip_b") in train_cids or p.get("clip_B", {}).get("clip_id") in train_cids)
    ]
    val_contrastive_pairs = [
        p for p in all_contrastive_pairs
        if (p.get("clip_a") in val_cids or p.get("clip_A", {}).get("clip_id") in val_cids)
        and (p.get("clip_b") in val_cids or p.get("clip_B", {}).get("clip_id") in val_cids)
    ]
    LOGGER.info(f"Loaded {len(train_contrastive_pairs)} intra-train and {len(val_contrastive_pairs)} intra-val contrastive pairs.")

    models_to_train = [
        ("siglip2_ft", SigLIP2RankingHead(d_model=768)),
        ("internvideo2_ft", InternVideo2RankingHead(d_model=768)),
        ("cross_attention", CrossAttentionRankerModel(d_model=768)),
        ("gatv2", GATv2VideoFusionModel(d_model=768)),
        ("graphormer", GraphormerVideoFusionModel(d_model=768)),
        ("vcgat_gold", VCGATGoldModel(d_model=768))
    ]

    if args.model != "all":
        models_to_train = [m for m in models_to_train if m[0] == args.model]

    for m_name, m_inst in models_to_train:
        train_single_model(
            model_name=m_name,
            model=m_inst,
            train_clip_records=train_clip_records,
            val_clip_records=val_clip_records,
            train_contrastive_pairs=train_contrastive_pairs,
            val_contrastive_pairs=val_contrastive_pairs,
            feature_cache=feature_cache,
            device=device,
            epochs=args.epochs,
            lr=args.lr
        )

    LOGGER.info("\nALL REQUESTED SUPERVISED MODELS SUCCESSFULLY TRAINED!")


if __name__ == "__main__":
    main()
