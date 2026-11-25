#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/dataset_loader.py
=============================================================================
Unified Dataset Loader and Evaluation Metrics for Fact Selection Benchmark.

Covers:
  - 800 Test clips (500 Hanoi In-Domain, 300 HCMC OOD)
  - 851 Train+Val clips (702 Train, 149 Val)
  - 3,234 Normalized Candidate Facts
  - Gold Standard Labels (scene_conditioned_gold.jsonl & train_val)
  - Contrastive Challenge Pairs (intra-landmark cross-scene)
  - Fact-Type Categorization (Visual, Historical, Inferred)
  - Standard Evaluation Metrics: P@K, R@K, nDCG@K, MAP, Hard-Negative AUC, Contrastive Acc
=============================================================================
"""

import os
import json
import gzip
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Paths to canonical datasets
FACTS_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "normalized_facts.jsonl"
TEST_MANIFEST_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "evaluation_clips_manifest.json"
TRAIN_VAL_MANIFEST_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "train_val_clips_manifest.json"
TEST_GOLD_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "scene_conditioned_gold.jsonl"
TRAIN_VAL_GOLD_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "scene_conditioned_gold_train_val.jsonl"
TEST_CONTRASTIVE_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "contrastive_pairs.jsonl"
TRAIN_VAL_CONTRASTIVE_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "contrastive_pairs_train_val.jsonl"

# Fact Predicate Categories
VISUAL_PREDICATES: Set[str] = {
    "has_part", "part_of", "has_component", "made_of", "has_material",
    "has_color", "has_shape", "has_roof_type", "has_roof_shape",
    "has_column_type", "has_window_type", "has_door_type", "has_ornament",
    "depicts_motif", "has_number_of_levels", "has_number_of_columns",
    "has_surface_texture", "has_pattern", "has_inscription", "has_decoration",
    "located_inside", "located_in_front_of", "located_behind", "adjacent_to",
    "faces", "surrounded_by", "connected_to", "above", "below", "has_entrance",
    "overlooks", "near", "on", "over", "leading_to", "structure", "appearance",
    "contains", "features"
}

HISTORICAL_PREDICATES: Set[str] = {
    "located_in", "founded_by", "founded_in", "built_by", "constructed_in",
    "symbol_of", "renamed_to", "named_after", "built_in", "dated_to", "used_as",
    "associated_dynasty", "dynasty", "history", "emperor", "established_by",
    "historical_period", "origin", "reign", "country", "province", "postal_code",
    "ruled_by", "commissioned_by", "has_physical_condition", "has_restoration_feature"
}

INFERRED_PREDICATES: Set[str] = {
    "belongs_to_dynasty", "architectural_style", "has_style", "worships",
    "dedicated_to", "associated_with", "commemorates", "features_style"
}


def get_fact_type(predicate: str) -> str:
    """Classifies predicate into visual, historical, or inferred."""
    pred = predicate.lower().strip()
    if pred in VISUAL_PREDICATES:
        return "visual"
    elif pred in INFERRED_PREDICATES:
        return "inferred"
    elif pred in HISTORICAL_PREDICATES:
        return "historical"
    # Fallback heuristic
    if any(k in pred for k in ["color", "shape", "material", "roof", "column", "wall", "gate", "door", "window", "made", "part", "component"]):
        return "visual"
    if any(k in pred for k in ["style", "worship", "dedicate"]):
        return "inferred"
    return "historical"


def _open_data_file(filepath: Path):
    """Helper to transparently open plain text or gzipped (.gz) jsonl files."""
    p_gz = Path(str(filepath) + ".gz")
    if not filepath.exists() and p_gz.exists():
        return gzip.open(p_gz, "rt", encoding="utf-8")
    return open(filepath, "r", encoding="utf-8")


def load_normalized_facts() -> Dict[str, Dict[str, Any]]:
    """Loads all normalized candidate facts indexed by fact_id."""
    facts = {}
    with _open_data_file(FACTS_PATH) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item["fact_type"] = get_fact_type(item.get("predicate", ""))
                facts[item["fact_id"]] = item
    return facts


def load_clip_manifests() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Loads test manifests and train_val manifests indexed by clip_id."""
    test_clips = {}
    with open(TEST_MANIFEST_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
        for it in items:
            test_clips[it["clip_id"]] = it

    train_val_clips = {}
    with open(TRAIN_VAL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
        for it in items:
            train_val_clips[it["clip_id"]] = it

    return test_clips, train_val_clips


def load_gold_data(split_type: str = "test") -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Loads gold records grouped by clip_id and all raw records.
    split_type: 'test', 'train', 'val', or 'train_val'
    """
    if split_type == "test":
        target_path = TEST_GOLD_PATH
    else:
        target_path = TRAIN_VAL_GOLD_PATH

    clip_records = defaultdict(list)
    all_records = []
    with _open_data_file(Path(target_path)) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["fact_type"] = get_fact_type(rec.get("predicate", ""))
            
            # Filter by split if requested
            rec_split = rec.get("split", "")
            if split_type == "train" and rec_split != "train":
                continue
            if split_type == "val" and rec_split != "val":
                continue
                
            clip_records[rec["clip_id"]].append(rec)
            all_records.append(rec)

    return clip_records, all_records


def load_contrastive_pairs(split_type: str = "test") -> List[Dict[str, Any]]:
    """Loads cross-scene intra-landmark contrastive pairs."""
    path = TEST_CONTRASTIVE_PATH if split_type == "test" else TRAIN_VAL_CONTRASTIVE_PATH
    pairs = []
    p = Path(path)
    if p.exists() or Path(str(p) + ".gz").exists():
        with _open_data_file(p) as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
    return pairs


# =============================================================================
# METRIC EVALUATION FUNCTIONS
# =============================================================================

def compute_ranking_metrics(
    ranked_fact_ids: List[str],
    gold_fact_ids: Set[str],
    all_candidate_fact_ids: List[str],
    top_k_list: List[int] = [3, 5, 10, 16]
) -> Dict[str, float]:
    """
    Computes precision, recall, nDCG, and average precision (AP) for a single clip.
    """
    metrics = {}
    if not gold_fact_ids:
        for k in top_k_list:
            metrics[f"P@{k}"] = 0.0
            metrics[f"R@{k}"] = 0.0
            metrics[f"nDCG@{k}"] = 0.0
        metrics["AP"] = 0.0
        return metrics

    # Precision & Recall @ K
    for k in top_k_list:
        top_k = ranked_fact_ids[:k]
        hits = sum(1 for fid in top_k if fid in gold_fact_ids)
        metrics[f"P@{k}"] = hits / k
        metrics[f"R@{k}"] = hits / len(gold_fact_ids)

    # nDCG @ K (binary relevance)
    for k in top_k_list:
        dcg = 0.0
        for i, fid in enumerate(ranked_fact_ids[:k]):
            rel = 1.0 if fid in gold_fact_ids else 0.0
            if rel > 0:
                dcg += rel / math.log2(i + 2)
        
        # Ideal DCG
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold_fact_ids))))
        metrics[f"nDCG@{k}"] = (dcg / idcg) if idcg > 0 else 0.0

    # Average Precision (AP) for MAP
    num_hits = 0
    sum_prec = 0.0
    for i, fid in enumerate(ranked_fact_ids):
        if fid in gold_fact_ids:
            num_hits += 1
            sum_prec += num_hits / (i + 1)
    metrics["AP"] = (sum_prec / len(gold_fact_ids)) if len(gold_fact_ids) > 0 else 0.0

    return metrics


def compute_hard_negative_auc(
    fact_scores: Dict[str, float],
    gold_fact_ids: Set[str],
    hard_negative_ids: Set[str]
) -> Optional[float]:
    """
    Computes ROC-AUC distinguishing positive gold facts from same-landmark hard negatives.
    Returns None if either class is empty.
    """
    y_true = []
    y_score = []
    for fid in gold_fact_ids:
        if fid in fact_scores:
            y_true.append(1)
            y_score.append(fact_scores[fid])
    for fid in hard_negative_ids:
        if fid in fact_scores:
            y_true.append(0)
            y_score.append(fact_scores[fid])

    if len(set(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def compute_contrastive_accuracy(
    score_fn_or_dict: Any,  # Dict[(clip_id, fact_id), float] or callable (clip_id, fact_id) -> float
    contrastive_pairs: List[Dict[str, Any]]
) -> float:
    """
    Computes Bidirectional Contrastive Accuracy across intra-landmark pairs.
    Pair: fact f is positive for clip A, negative for clip B.
    Success condition: score(clip_A, f) > score(clip_B, f).
    """
    if not contrastive_pairs:
        return 0.0

    correct = 0
    total = 0
    for p in contrastive_pairs:
        # Support both flat keys and nested clip_A/clip_B dicts
        c_a = p.get("clip_a") or (p.get("clip_A", {}).get("clip_id") if isinstance(p.get("clip_A"), dict) else None)
        c_b = p.get("clip_b") or (p.get("clip_B", {}).get("clip_id") if isinstance(p.get("clip_B"), dict) else None)
        f_a = p.get("fact_id") or p.get("fact_a")
        f_b = p.get("fact_b")

        if not c_a or not c_b:
            continue

        # Evaluate fact_a: positive for A, negative for B
        if f_a:
            if isinstance(score_fn_or_dict, dict):
                if (c_a, f_a) in score_fn_or_dict and (c_b, f_a) in score_fn_or_dict:
                    s_a = score_fn_or_dict[(c_a, f_a)]
                    s_b = score_fn_or_dict[(c_b, f_a)]
                    if s_a > s_b:
                        correct += 1
                    total += 1
            else:
                s_a = score_fn_or_dict(c_a, f_a)
                s_b = score_fn_or_dict(c_b, f_a)
                if s_a > s_b:
                    correct += 1
                total += 1

        # Evaluate fact_b: positive for B, negative for A
        if f_b:
            if isinstance(score_fn_or_dict, dict):
                if (c_b, f_b) in score_fn_or_dict and (c_a, f_b) in score_fn_or_dict:
                    s_b = score_fn_or_dict[(c_b, f_b)]
                    s_a = score_fn_or_dict[(c_a, f_b)]
                    if s_b > s_a:
                        correct += 1
                    total += 1
            else:
                s_b = score_fn_or_dict(c_b, f_b)
                s_a = score_fn_or_dict(c_a, f_b)
                if s_b > s_a:
                    correct += 1
                total += 1

    return (correct / total * 100.0) if total > 0 else 0.0
