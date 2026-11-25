#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Blind Pairwise Dataset for Head-to-Head Caption Evaluation
===================================================================
Tạo bộ dữ liệu so sánh đối đầu theo cặp mù (Blind Pairwise Evaluation):
So sánh mô hình đề xuất ('ours') với 5 baselines & ablations:
1. vlm_direct: VLM trực tiếp (Qwen3-VL-8B)
2. vlm_rag: VLM + RAG (Qwen3-VL-8B + SigLIP2)
3. frozen_base: Base LLM đóng băng không fine-tune
4. ablation_no_threshold: Bỏ ngưỡng (Luôn khẳng định)
5. ablation_no_anchor: Bỏ neo (Greedy mu_F)

Đặc điểm:
- Ẩn danh hoàn toàn (Model A vs Model B) xáo trộn ngẫu nhiên 50/50 để chống thiên vị vị trí.
- Mỗi cặp có 3 câu hỏi:
  1. Q1: Caption nào đúng với cảnh hơn? (Scene Consistency)
  2. Q2: Caption nào đáng tin hơn? (Factual Reliability / No Hallucination)
  3. Q3: Caption nào muốn nghe hơn? (Audio/Narrative Engagement)
- Xuất 2 file:
  1. `pairwise_blind_for_evaluation.json`: Dành cho người chấm độc lập / external LLM.
  2. `pairwise_blind_keys.json`: Bảng giải mã bí mật để tính Win/Tie/Loss.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("CreateBlindPairwise")

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_BENCHMARK_PATH = REPO_ROOT / "experiments" / "outputs" / "benchmark_all_models_250_captions.json"
OUTPUT_BLIND_PATH = REPO_ROOT / "experiments" / "outputs" / "pairwise_blind_for_evaluation.json"
OUTPUT_KEYS_PATH = REPO_ROOT / "experiments" / "outputs" / "pairwise_blind_keys.json"

COMPARED_BASELINES = [
    ("vlm_direct", "caption_vlm_direct", "VLM trực tiếp (Qwen3-VL-8B)"),
    ("vlm_rag", "caption_vlm_rag", "VLM + RAG (Qwen3-VL-8B + SigLIP2)"),
    ("frozen_base", "caption_frozen_base", "LLM đóng băng (Base Qwen2.5-3B)"),
    ("ablation_no_threshold", "caption_ablation_no_threshold", "Bỏ ngưỡng (Always Affirm)"),
    ("ablation_no_anchor", "caption_ablation_no_anchor", "Bỏ neo (Greedy mu_F)"),
]


def create_blind_pairwise_dataset(seed: int = 42):
    random.seed(seed)
    
    if not INPUT_BENCHMARK_PATH.exists():
        LOGGER.error("Benchmark file not found: %s", INPUT_BENCHMARK_PATH)
        return

    with open(INPUT_BENCHMARK_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    LOGGER.info("Loaded %d test samples from: %s", len(samples), INPUT_BENCHMARK_PATH)

    blind_records = []
    key_records = {}

    pair_counter = 0

    for sample_idx, sample in enumerate(samples):
        ours_caption = sample.get("caption_ours", "").strip()
        video_file = sample.get("file_name", "")
        gt_parent = sample.get("groundtruth_parent", "")
        gt_sub = sample.get("groundtruth_sub", "")
        certainty = sample.get("certainty_level", "HIGH")

        for baseline_key, cap_field, baseline_name in COMPARED_BASELINES:
            baseline_caption = sample.get(cap_field, "").strip()
            pair_counter += 1
            pair_id = f"PAIR_{pair_counter:04d}"

            # Random 50/50 assignment to A or B
            is_ours_model_a = random.random() < 0.5
            if is_ours_model_a:
                caption_a = ours_caption
                caption_b = baseline_caption
                model_a_id = "ours"
                model_b_id = baseline_key
            else:
                caption_a = baseline_caption
                caption_b = ours_caption
                model_a_id = baseline_key
                model_b_id = "ours"

            # Blind record for evaluator
            blind_item = {
                "pair_id": pair_id,
                "sample_index": sample_idx,
                "video_file": video_file,
                "groundtruth_context": {
                    "parent_landmark": gt_parent,
                    "sub_landmark": gt_sub,
                    "certainty_level": certainty,
                },
                "caption_a": caption_a,
                "caption_b": caption_b,
                "evaluation": {
                    "q1_more_scene_consistent": "",  # Choices: "Model A", "Model B", "Tie"
                    "q2_more_factual_reliable": "",  # Choices: "Model A", "Model B", "Tie"
                    "q3_more_engaging_to_listen": "", # Choices: "Model A", "Model B", "Tie"
                    "brief_rationale": "",
                }
            }
            blind_records.append(blind_item)

            # Secret key record
            key_records[pair_id] = {
                "pair_id": pair_id,
                "sample_index": sample_idx,
                "video_file": video_file,
                "baseline_comparison": baseline_key,
                "baseline_name": baseline_name,
                "model_a_identity": model_a_id,
                "model_b_identity": model_b_id,
                "is_ours_model_a": is_ours_model_a,
            }

    # Save blind evaluation dataset
    OUTPUT_BLIND_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_BLIND_PATH, "w", encoding="utf-8") as f:
        json.dump(blind_records, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved %d blind pairwise evaluation items to: %s", len(blind_records), OUTPUT_BLIND_PATH)

    # Save secret keys
    with open(OUTPUT_KEYS_PATH, "w", encoding="utf-8") as f:
        json.dump(key_records, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved secret mapping keys to: %s", OUTPUT_KEYS_PATH)


if __name__ == "__main__":
    create_blind_pairwise_dataset()
