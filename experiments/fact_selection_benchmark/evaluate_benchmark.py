#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/evaluate_benchmark.py
=============================================================================
Evaluation Suite for Layer 1 — Fact Selection Benchmark.

Evaluates 7 Baselines + 1 Proposed Model:
  B1. Crisp-KG (Static Symbolic Ranking)
  B2. SigLIP2 (Zero-shot & Fine-tuned)
  B3. InternVideo2 (Zero-shot & Fine-tuned)
  B4. LLaVA-NeXT-8B (Multimodal LLM Ranker Baseline)
  B5. Cross-Attention Ranker (Supervised Token Interaction)
  B6. GATv2 + Video Fusion (Dynamic Graph Attention)
  B7. Graphormer + Video Fusion (Graph Transformer)
  Ours. VC-GAT Gold-Supervised (Proposed Video-Conditioned Graph Attention)

Experiments Covered:
  - E1. Full-Pool Fact Ranking (P@5, R@5, nDCG@5, nDCG@16, MAP)
  - E2. Hard Negative Discrimination (Hard Negative AUC)
  - E3. Same-Landmark Contrastive (Bidirectional Contrastive Accuracy)
  - E4. Fact-Type Evaluation (Visual / Historical / Inferred)
  - E5. Cross-Landmark Generalization (Hanoi Test 500 / HCMC OOD 300)
=============================================================================
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
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
from experiments.fact_selection_benchmark.models.crisp_kg import CrispKGRanker
from experiments.fact_selection_benchmark.models.siglip2_ranker import SigLIP2Ranker, SigLIP2RankingHead
from experiments.fact_selection_benchmark.models.internvideo2_ranker import InternVideo2Ranker, InternVideo2RankingHead
from experiments.fact_selection_benchmark.models.llava_next_ranker import LLaVANextRanker
from experiments.fact_selection_benchmark.models.cross_attention_ranker import CrossAttentionRanker, CrossAttentionRankerModel
from experiments.fact_selection_benchmark.models.gatv2_ranker import GATv2Ranker, GATv2VideoFusionModel
from experiments.fact_selection_benchmark.models.graphormer_ranker import GraphormerRanker, GraphormerVideoFusionModel
from experiments.fact_selection_benchmark.models.vcgat_gold import VCGATGoldRanker, VCGATGoldModel

CHECKPOINT_DIR = REPO_ROOT / "experiments" / "checkpoints_fact_selection"
REPORTS_DIR = REPO_ROOT / "experiments" / "fact_selection_benchmark" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("EvaluateBenchmark")


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model_checkpoints(device: torch.device):
    """Loads trained weights into model instances if available."""
    models_dict = {}

    # B2: SigLIP2-FT
    siglip2_head = SigLIP2RankingHead(d_model=768)
    ckpt = CHECKPOINT_DIR / "best_siglip2_ft.pt"
    if ckpt.exists():
        siglip2_head.load_state_dict(torch.load(ckpt, map_location=device))
        LOGGER.info("Loaded SigLIP2-FT checkpoint.")
    models_dict["siglip2_head"] = siglip2_head

    # B3: InternVideo2-FT
    iv2_head = InternVideo2RankingHead(d_model=768)
    ckpt = CHECKPOINT_DIR / "best_internvideo2_ft.pt"
    if ckpt.exists():
        iv2_head.load_state_dict(torch.load(ckpt, map_location=device))
        LOGGER.info("Loaded InternVideo2-FT checkpoint.")
    models_dict["iv2_head"] = iv2_head

    # B5: Cross-Attention
    cross_attn = CrossAttentionRankerModel(d_model=768)
    ckpt = CHECKPOINT_DIR / "best_cross_attention.pt"
    if ckpt.exists():
        cross_attn.load_state_dict(torch.load(ckpt, map_location=device))
        LOGGER.info("Loaded Cross-Attention checkpoint.")
    models_dict["cross_attn"] = cross_attn

    # B6: GATv2
    gatv2_mod = GATv2VideoFusionModel(d_model=768)
    ckpt = CHECKPOINT_DIR / "best_gatv2.pt"
    if ckpt.exists():
        gatv2_mod.load_state_dict(torch.load(ckpt, map_location=device))
        LOGGER.info("Loaded GATv2 checkpoint.")
    models_dict["gatv2_mod"] = gatv2_mod

    # B7: Graphormer
    graphormer_mod = GraphormerVideoFusionModel(d_model=768)
    ckpt = CHECKPOINT_DIR / "best_graphormer.pt"
    if ckpt.exists():
        graphormer_mod.load_state_dict(torch.load(ckpt, map_location=device))
        LOGGER.info("Loaded Graphormer checkpoint.")
    models_dict["graphormer_mod"] = graphormer_mod

    # Ours: VC-GAT Gold
    vcgat_mod = VCGATGoldModel(d_model=768)
    ckpt = CHECKPOINT_DIR / "best_vcgat_gold.pt"
    if ckpt.exists():
        vcgat_mod.load_state_dict(torch.load(ckpt, map_location=device))
        LOGGER.info("Loaded VC-GAT Gold checkpoint.")
    models_dict["vcgat_mod"] = vcgat_mod

    return models_dict


def evaluate_all(mock: bool = False):
    device = get_device()
    LOGGER.info(f"Running Fact Selection Benchmark on device: {device} (Mock={mock})")

    # Load data
    facts = load_normalized_facts()
    test_manifest, train_val_manifest = load_clip_manifests()
    all_manifest_list = list(test_manifest.values()) + list(train_val_manifest.values())
    test_clip_records, _ = load_gold_data(split_type="test")
    contrastive_pairs = load_contrastive_pairs(split_type="test")

    LOGGER.info(f"Loaded {len(test_clip_records)} test clips and {len(contrastive_pairs)} contrastive pairs.")

    # Feature cache
    feature_cache = BenchmarkFeatureCache(device=device)
    if not mock:
        feature_cache.extract_fact_embeddings(facts)
        feature_cache.extract_video_features(all_manifest_list)
        feature_cache.build_landmark_subgraphs(facts)

    models_dict = load_model_checkpoints(device)

    # Initialize model wrappers
    rankers = {
        "B1_Crisp_KG": (CrispKGRanker(facts), "crisp_kg"),
        "B2_SigLIP2_ZeroShot": (SigLIP2Ranker(feature_cache.video_pooled, feature_cache.fact_embeddings, device=device), "siglip2_zs"),
        "B2_SigLIP2_FineTuned": (SigLIP2Ranker(feature_cache.video_pooled, feature_cache.fact_embeddings, head_model=models_dict["siglip2_head"], device=device), "siglip2_ft"),
        "B3_InternVideo2_ZeroShot": (InternVideo2Ranker(feature_cache.video_frames, feature_cache.fact_embeddings, device=device), "iv2_zs"),
        "B3_InternVideo2_FineTuned": (InternVideo2Ranker(feature_cache.video_frames, feature_cache.fact_embeddings, head_model=models_dict["iv2_head"], device=device), "iv2_ft"),
        "B4_LLaVA_NeXT_8B": (LLaVANextRanker(facts, feature_cache.video_pooled, feature_cache.fact_embeddings, device=device), "llava_next"),
        "B5_Cross_Attention": (CrossAttentionRanker(feature_cache.video_frames, feature_cache.fact_embeddings, model=models_dict["cross_attn"], device=device), "cross_attn"),
        "B6_GATv2_Video": (GATv2Ranker(feature_cache.video_pooled, feature_cache.fact_embeddings, feature_cache.subgraphs, model=models_dict["gatv2_mod"], device=device), "gatv2"),
        "B7_Graphormer_Video": (GraphormerRanker(feature_cache.video_pooled, feature_cache.fact_embeddings, feature_cache.subgraphs, model=models_dict["graphormer_mod"], device=device), "graphormer"),
        "Ours_VCGAT_Gold": (VCGATGoldRanker(feature_cache.video_pooled, feature_cache.fact_embeddings, feature_cache.subgraphs, model=models_dict["vcgat_mod"], device=device), "vcgat")
    }

    # Data structures for results
    # results[model_name][experiment_slice] = metrics
    all_results = {}

    for model_key, (ranker_inst, model_type) in rankers.items():
        LOGGER.info(f"Evaluating model: [{model_key}]...")

        # Store clip scores: (clip_id, fact_id) -> score
        clip_fact_scores = {}

        # Slices
        all_metrics_list = []
        hanoi_metrics_list = []
        hcmc_metrics_list = []
        visual_metrics_list = []
        hist_metrics_list = []
        infer_metrics_list = []
        hard_neg_auc_list = []

        for c_id, recs in tqdm(test_clip_records.items(), desc=f"Eval {model_key}", leave=False):
            l_id = recs[0].get("landmark_id", "unknown")
            region = recs[0].get("region", "")
            cand_fids = [r["fact_id"] for r in recs]
            gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")

            # Score candidates
            if model_type == "crisp_kg":
                scores = ranker_inst.score_clip_facts(c_id, cand_fids)
            elif model_type == "siglip2_zs":
                scores = ranker_inst.score_clip_facts(c_id, cand_fids, mode="zero_shot")
            elif model_type == "siglip2_ft":
                scores = ranker_inst.score_clip_facts(c_id, cand_fids, mode="fine_tuned")
            elif model_type == "iv2_zs":
                scores = ranker_inst.score_clip_facts(c_id, cand_fids, mode="zero_shot")
            elif model_type == "iv2_ft":
                scores = ranker_inst.score_clip_facts(c_id, cand_fids, mode="fine_tuned")
            elif model_type == "llava_next":
                scores = ranker_inst.score_clip_facts(c_id, cand_fids)
            elif model_type == "cross_attn":
                scores = ranker_inst.score_clip_facts(c_id, cand_fids)
            elif model_type == "gatv2":
                scores = ranker_inst.score_clip_facts(c_id, l_id, cand_fids)
            elif model_type == "graphormer":
                scores = ranker_inst.score_clip_facts(c_id, l_id, cand_fids)
            elif model_type == "vcgat":
                scores = ranker_inst.score_clip_facts(c_id, l_id, cand_fids)
            else:
                scores = {fid: 0.0 for fid in cand_fids}

            for fid, s in scores.items():
                clip_fact_scores[(c_id, fid)] = s

            # Sort by score descending
            ranked_fids = sorted(cand_fids, key=lambda fid: scores.get(fid, -999.0), reverse=True)
            clip_m = compute_ranking_metrics(ranked_fids, gold_fids, cand_fids, top_k_list=[3, 5, 10, 16])
            all_metrics_list.append(clip_m)

            if region == "Hanoi":
                hanoi_metrics_list.append(clip_m)
            elif region == "HCMC":
                hcmc_metrics_list.append(clip_m)

            # Hard negative AUC
            auc = compute_hard_negative_auc(scores, gold_fids, hard_neg_fids)
            if auc is not None:
                hard_neg_auc_list.append(auc)

            # Fact-type breakdowns
            # Visual facts
            vis_gold = set(r["fact_id"] for r in recs if r.get("is_scene_gold") and r.get("fact_type") == "visual")
            vis_cand = [r["fact_id"] for r in recs if r.get("fact_type") == "visual"]
            if vis_gold:
                vis_ranked = [fid for fid in ranked_fids if fid in vis_cand]
                visual_metrics_list.append(compute_ranking_metrics(vis_ranked, vis_gold, vis_cand, top_k_list=[5]))

            # Historical facts
            hist_gold = set(r["fact_id"] for r in recs if r.get("is_scene_gold") and r.get("fact_type") == "historical")
            hist_cand = [r["fact_id"] for r in recs if r.get("fact_type") == "historical"]
            if hist_gold:
                hist_ranked = [fid for fid in ranked_fids if fid in hist_cand]
                hist_metrics_list.append(compute_ranking_metrics(hist_ranked, hist_gold, hist_cand, top_k_list=[5]))

            # Inferred facts
            inf_gold = set(r["fact_id"] for r in recs if r.get("is_scene_gold") and r.get("fact_type") == "inferred")
            inf_cand = [r["fact_id"] for r in recs if r.get("fact_type") == "inferred"]
            if inf_gold:
                inf_ranked = [fid for fid in ranked_fids if fid in inf_cand]
                infer_metrics_list.append(compute_ranking_metrics(inf_ranked, inf_gold, inf_cand, top_k_list=[5]))

        # Contrastive accuracy
        contrastive_acc = compute_contrastive_accuracy(clip_fact_scores, contrastive_pairs)

        # Helper to average metrics dicts
        def avg_m(m_list):
            if not m_list:
                return {k: 0.0 for k in ["P@5", "R@5", "nDCG@5", "nDCG@16", "AP"]}
            res = {}
            for k in m_list[0].keys():
                res[k] = float(np.mean([m[k] for m in m_list]))
            return res

        e1_full = avg_m(all_metrics_list)
        e5_hanoi = avg_m(hanoi_metrics_list)
        e5_hcmc = avg_m(hcmc_metrics_list)
        e4_vis = avg_m(visual_metrics_list)
        e4_hist = avg_m(hist_metrics_list)
        e4_infer = avg_m(infer_metrics_list)
        mean_auc = float(np.mean(hard_neg_auc_list)) if hard_neg_auc_list else 0.5

        all_results[model_key] = {
            "E1_Full_Pool": {
                "P@5": round(e1_full.get("P@5", 0.0), 3),
                "R@5": round(e1_full.get("R@5", 0.0), 3),
                "nDCG@5": round(e1_full.get("nDCG@5", 0.0), 3),
                "nDCG@16": round(e1_full.get("nDCG@16", 0.0), 3),
                "MAP": round(e1_full.get("AP", 0.0), 3),
            },
            "E2_Hard_Negative_AUC": round(mean_auc, 3),
            "E3_Contrastive_Acc": round(contrastive_acc, 1),
            "E4_Fact_Types": {
                "Visual_nDCG@5": round(e4_vis.get("nDCG@5", 0.0), 3),
                "Historical_nDCG@5": round(e4_hist.get("nDCG@5", 0.0), 3),
                "Inferred_nDCG@5": round(e4_infer.get("nDCG@5", 0.0), 3),
            },
            "E5_Generalization": {
                "Hanoi_nDCG@5": round(e5_hanoi.get("nDCG@5", 0.0), 3),
                "HCMC_OOD_nDCG@5": round(e5_hcmc.get("nDCG@5", 0.0), 3),
            }
        }

    # Save JSON results
    json_path = REPORTS_DIR / "fact_selection_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    LOGGER.info(f"Saved benchmark results -> {json_path}")

    # Generate Publication Markdown Table
    md_path = REPORTS_DIR / "fact_selection_benchmark_table.md"
    generate_markdown_report(all_results, md_path)
    LOGGER.info(f"Saved publication table -> {md_path}")


def generate_markdown_report(results: Dict[str, Any], output_path: Path):
    lines = [
        "# Layer 1 — Fact Selection Comprehensive Benchmark",
        "",
        "Evaluated on 800 independent test clips (500 Hanoi In-Domain + 300 HCMC OOD) over 143,124 clip-fact candidate pairs.",
        "",
        "## Table 1: Main Fact Selection Performance (E1, E2, E3)",
        "",
        "| Model ID | Method / Architecture | Paradigm | P@5 | R@5 | nDCG@5 | nDCG@16 | MAP | Hard Neg AUC | Contrastive Acc (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    method_meta = {
        "B1_Crisp_KG": ("Crisp-KG", "Static Symbolic KG"),
        "B2_SigLIP2_ZeroShot": ("SigLIP2 (Zero-shot)", "Vision-Lang Dual Enc"),
        "B2_SigLIP2_FineTuned": ("SigLIP2 (Fine-tuned)", "Vision-Lang Dual Enc"),
        "B3_InternVideo2_ZeroShot": ("InternVideo2 (Zero-shot)", "Video-Lang Foundation"),
        "B3_InternVideo2_FineTuned": ("InternVideo2 (Fine-tuned)", "Video-Lang Foundation"),
        "B4_LLaVA_NeXT_8B": ("LLaVA-NeXT-8B", "Multimodal LLM"),
        "B5_Cross_Attention": ("Cross-Attention Ranker", "Supervised Non-Graph"),
        "B6_GATv2_Video": ("GATv2 + Video Fusion", "Dynamic GNN"),
        "B7_Graphormer_Video": ("Graphormer + Video Fusion", "Graph Transformer"),
        "Ours_VCGAT_Gold": ("**VC-GAT Gold-Supervised**", "**Video-Conditioned GAT**")
    }

    for key, (name, paradigm) in method_meta.items():
        if key not in results:
            continue
        r = results[key]
        e1 = r["E1_Full_Pool"]
        auc = r["E2_Hard_Negative_AUC"]
        c_acc = r["E3_Contrastive_Acc"]
        lines.append(
            f"| {key[:2]} | {name} | {paradigm} | {e1['P@5']:.3f} | {e1['R@5']:.3f} | **{e1['nDCG@5']:.3f}** | {e1['nDCG@16']:.3f} | {e1['MAP']:.3f} | {auc:.3f} | **{c_acc:.1f}%** |"
        )

    lines.extend([
        "",
        "*Note on LLaVA-NeXT-8B: Multimodal LLM (Meta-Llama-3-8B-Instruct + CLIP-ViT-L-336px) evaluated independently without shared weights or annotation prompts.",
        "",
        "---",
        "",
        "## Table 2: Fact-Type & Generalization Breakdown (E4, E5)",
        "",
        "| Model ID | Method | Visual nDCG@5 | Historical nDCG@5 | Inferred nDCG@5 | Hanoi Test (500) | HCMC OOD (300) | Generalization Drop |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    )

    for key, (name, _) in method_meta.items():
        if key not in results:
            continue
        r = results[key]
        e4 = r["E4_Fact_Types"]
        e5 = r["E5_Generalization"]
        hanoi_s = e5["Hanoi_nDCG@5"]
        hcmc_s = e5["HCMC_OOD_nDCG@5"]
        drop = round(hanoi_s - hcmc_s, 3)
        lines.append(
            f"| {key[:2]} | {name} | {e4['Visual_nDCG@5']:.3f} | {e4['Historical_nDCG@5']:.3f} | {e4['Inferred_nDCG@5']:.3f} | {hanoi_s:.3f} | {hcmc_s:.3f} | {drop:+.3f} |"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    evaluate_all(mock=args.mock)
