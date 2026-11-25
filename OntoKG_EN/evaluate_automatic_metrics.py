#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Automatic Benchmark Metrics Across All Models
======================================================
Tính toán 3 chỉ số tự động khách quan (không cần trích xuất phức tạp):
1. Tỷ lệ nêu tên sai (Wrong Landmark Rate %):
   Kiểm tra caption chứa tên landmark nào trong danh sách 30-31 di tích, so với nhãn Ground Truth.
2. Tỷ lệ im lặng (Silence Rate %):
   Bao nhiêu % caption không nêu bất kỳ tên landmark nào (an toàn / kiềm chế khi thị giác mơ hồ).
3. Tỷ lệ vi phạm luật 3 mức (3-Tier Rule Violation Rate %):
   - Mức HIGH: ảo giác thực thể ngoài prompt.
   - Mức MEDIUM: nêu tên candidate sub-landmark.
   - Mức LOW: chứa tên riêng.
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("AutoMetrics")

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_INPUT_PATH = REPO_ROOT / "experiments" / "outputs" / "benchmark_all_models_250_captions.json"
OUTPUT_REPORT_PATH = REPO_ROOT / "experiments" / "outputs" / "automatic_metrics_benchmark_report.json"

# Taxonomy of 30 parent landmarks and distinctive aliases
LANDMARK_ALIASES = {
    "Ba Dinh Square": ["ba dinh square", "ba dinh", "ho chi minh mausoleum", "lang bac", "presidential palace"],
    "Bat Trang Pottery Village": ["bat trang", "bat trang pottery"],
    "Botanical Garden": ["botanical garden", "bach thao"],
    "Dien Hong Park": ["dien hong", "con coc"],
    "Dong Da Mound": ["dong da mound", "go dong da", "dong da"],
    "Ha Noi Station": ["ha noi station", "hanoi station", "ga ha noi"],
    "Hanoi Cuisine": ["hanoi cuisine", "am thuc ha noi"],
    "Hanoi Museum": ["hanoi museum", "bao tang ha noi"],
    "Hanoi Old Quarter": ["hanoi old quarter", "old quarter", "pho co", "hang bac", "hang ma", "hang gai", "ta hien", "o quan chuong"],
    "Hoan Kiem Lake": ["hoan kiem lake", "hoan kiem", "ho guom", "sword lake", "ngoc son", "the huc", "turtle tower", "thap rua"],
    "Huu Tiep Lake": ["huu tiep", "b52 lake", "b-52"],
    "Imperial Citadel of Thang Long": ["imperial citadel of thang long", "thang long imperial citadel", "imperial citadel", "thang long citadel", "hoang thanh thang long", "doan mon", "kinh thien", "hau lau", "bac mon"],
    "Lake Ba Mau": ["lake ba mau", "ba mau lake", "ba mau"],
    "Lake Thien Quang": ["lake thien quang", "thien quang lake", "thien quang", "lien thuy"],
    "Le Nin Park": ["le nin park", "lenin park", "le nin", "lenin"],
    "Ngu Xa Traditional Bronze Casting Workshop": ["ngu xa"],
    "One Pillar Pagoda": ["one pillar pagoda", "chua mot cot", "dien huu"],
    "Temple of Literature": ["temple of literature", "van mieu", "quoc tu giam", "khue van", "thien quang tinh", "dai thanh", "thai hoc"],
    "Thang Long Water Puppetry Theatre": ["thang long water puppetry", "water puppetry theatre", "water puppet theatre", "roi nuoc thang long"],
    "The Government Guest House": ["government guest house", "nha khach chinh phu", "bac bo phu"],
    "Thong Nhat park": ["thong nhat park", "thong nhat"],
    "Thu Le Park": ["thu le park", "thu le", "hanoi zoo"],
    "Truc Bach Lake": ["truc bach lake", "truc bach", "chau long"],
    "Van Phuc Silk Village": ["van phuc silk", "van phuc", "silk village"],
    "Vietnam Museum of Ethnology": ["museum of ethnology", "dan toc hoc"],
    "Vietnam National Museum of History": ["national museum of history", "lich su quoc gia"],
    "Vietnam National Puppetry Theatre": ["national puppetry theatre", "mua roi viet nam"],
    "Vietnam Women's Museum": ["women's museum", "phu nu viet nam"],
    "West Lake": ["west lake", "ho tay", "tran quoc", "phu tay ho", "thanh nien causeway"],
    "Yen So park": ["yen so park", "yen so"],
}

# General proper nouns check for LOW certainty violation
LOW_FORBIDDEN_PROPER_NOUNS = [
    "hanoi", "ha noi", "vietnam", "vietnamese", "ba dinh", "thang long", "van mieu",
    "khue van", "hoan kiem", "west lake", "thien quang", "thong nhat", "tran quoc",
    "bat trang", "van phuc", "le nin", "dong da", "thu le", "truc bach", "dien hong",
    "ngu xa", "french", "chinese", "cham", "ly", "tran", "nguyen", "temple of literature",
    "ho chi minh", "unesco", "hue", "saigon", "da nang"
]


def detect_mentioned_landmarks(caption: str) -> list:
    cap_lower = caption.lower()
    mentioned = set()
    for parent_name, aliases in LANDMARK_ALIASES.items():
        for alias in aliases:
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, cap_lower):
                mentioned.add(parent_name)
                break
    return list(mentioned)


def check_3tier_violation(tier: str, caption: str, groundtruth_parent: str, candidate_places: list, prompt_text: str) -> Tuple[bool, str]:
    cap_lower = caption.lower()
    
    if tier == "LOW":
        # Rule: Must not name any proper noun, site, city, dynasty
        for term in LOW_FORBIDDEN_PROPER_NOUNS:
            if re.search(r"\b" + re.escape(term) + r"\b", cap_lower):
                return True, f"Named proper noun: {term}"
        return False, "Obeyed"
        
    elif tier == "MEDIUM":
        # Rule: Must NOT name candidate specific sub-landmarks
        for cand in candidate_places:
            c_clean = cand.lower().strip()
            if len(c_clean) > 3 and c_clean != groundtruth_parent.lower() and c_clean not in ("gateway", "pavilion", "courtyard"):
                if re.search(r"\b" + re.escape(c_clean) + r"\b", cap_lower):
                    return True, f"Named candidate sub-place: {cand}"
        return False, "Obeyed"
        
    else:  # HIGH
        # Rule: Must not hallucinate entities completely absent from prompt
        # Check if caption introduces obvious foreign landmarks or unprompted claims
        for parent_name, aliases in LANDMARK_ALIASES.items():
            if parent_name != groundtruth_parent:
                for alias in aliases:
                    if re.search(r"\b" + re.escape(alias) + r"\b", cap_lower) and alias not in prompt_text.lower():
                        return True, f"Introduced ungrounded foreign landmark: {alias}"
        return False, "Obeyed"


def main():
    LOGGER.info("Loading benchmark results from: %s", BENCHMARK_INPUT_PATH)
    with open(BENCHMARK_INPUT_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    models = [
        ("ours", "Mô hình đề xuất (Qwen2.5-3B LoRA + OntoKG + Uncertainty Gate)", "caption_ours"),
        ("frozen_base", "Ablation: LLM đóng băng (Base 3B không fine-tune)", "caption_frozen_base"),
        ("ablation_no_threshold", "Ablation: Bỏ ngưỡng (Luôn khẳng định / Always HIGH)", "caption_ablation_no_threshold"),
        ("ablation_no_anchor", "Ablation: Bỏ neo (Greedy mu_F, không lọc candidate)", "caption_ablation_no_anchor"),
        ("vlm_direct", "Baseline ngoài: VLM trực tiếp (Qwen3-VL-8B, video only)", "caption_vlm_direct"),
        ("vlm_rag", "Baseline ngoài: VLM + RAG (Qwen3-VL-8B + SigLIP2 keyframe RAG)", "caption_vlm_rag"),
    ]

    LOGGER.info("Evaluating automatic metrics across %d models on %d samples...", len(models), len(samples))

    results = {}

    for model_key, model_name, cap_field in models:
        total_samples = len(samples)
        wrong_landmark_count = 0
        silence_count = 0
        violation_count = 0

        tier_violations = defaultdict(int)
        tier_totals = defaultdict(int)

        for s in samples:
            cap = s.get(cap_field, "")
            gt_parent = s.get("groundtruth_parent", "")
            cert = s.get("certainty_level", "HIGH")
            cands = s.get("candidate_places", [])
            prompt_txt = s.get("user_prompt", "")

            tier_totals[cert] += 1

            # 1. Detect mentioned landmarks
            mentioned = detect_mentioned_landmarks(cap)

            if len(mentioned) == 0:
                silence_count += 1
            else:
                # Check if ANY mentioned landmark is wrong
                is_wrong = any(m != gt_parent for m in mentioned)
                if is_wrong:
                    wrong_landmark_count += 1

            # 2. Check 3-tier violation
            violated, reason = check_3tier_violation(cert, cap, gt_parent, cands, prompt_txt)
            if violated:
                violation_count += 1
                tier_violations[cert] += 1

        wrong_rate = round(wrong_landmark_count / total_samples * 100, 2)
        silence_rate = round(silence_count / total_samples * 100, 2)
        violation_rate = round(violation_count / total_samples * 100, 2)

        results[model_key] = {
            "model_name": model_name,
            "total_samples": total_samples,
            "wrong_landmark_count": wrong_landmark_count,
            "wrong_landmark_rate_pct": wrong_rate,
            "silence_count": silence_count,
            "silence_rate_pct": silence_rate,
            "violation_count": violation_count,
            "violation_rate_pct": violation_rate,
            "violations_by_tier": {
                t: f"{tier_violations[t]}/{tier_totals[t]} ({tier_violations[t]/tier_totals[t]*100:.1f}%)"
                for t in ["HIGH", "MEDIUM", "LOW"] if tier_totals[t] > 0
            }
        }

        LOGGER.info(
            "[%-22s] Wrong Landmark: %3d (%5.2f%%) | Silence: %3d (%5.2f%%) | Violation: %3d (%5.2f%%)",
            model_key, wrong_landmark_count, wrong_rate, silence_count, silence_rate, violation_count, violation_rate
        )

    # Save report
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved automatic metrics report to: %s", OUTPUT_REPORT_PATH)


if __name__ == "__main__":
    main()
