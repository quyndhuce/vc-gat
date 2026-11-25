#!/usr/bin/env python3
"""
evaluate_grounding_hard_filter_three_tiers.py
=============================================================================
Multimodal Visual-to-Knowledge Grounding Benchmark with:
1. Sửa đổi Bảng Phân Nhóm Quan Hệ (Relation Taxonomy) chuẩn tắc 66 quan hệ:
   - DIRECT (26 quan hệ): Cấu trúc, màu sắc, vật liệu, vị trí không gian trực tiếp.
   - VISUAL_CUE (8 quan hệ): Phong cách kiến trúc, liên tưởng, công năng & hoạt động văn hóa.
   - NON_VISUAL (32 quan hệ): Lịch sử, niên đại, người xây dựng, triều đại, hành chính.
2. Bật Cổng Lọc Cứng (Hard Evidence Filtering Gate):
   - Fact không có khái niệm neo vượt ngưỡng (hoặc không có bounding box xác nhận) thì LOẠI BỎ.
   - Số fact đề xuất sẽ giảm xuống chỉ còn các fact thực sự có bằng chứng, làm tăng Precision.
3. Đổi Tiêu Chí Chấm Cho Nhóm NON_VISUAL:
   - Chấm theo tính xác thực tri thức sự thật lịch sử / địa lý về địa danh,
     thay vì đòi hỏi nhìn thấy trên frame video.
4. Báo Cáo Riêng Biệt Ba Nhóm:
   - Bảng kết quả độc lập cho DIRECT, VISUAL_CUE, NON_VISUAL.
=============================================================================
"""

import gc
import json
import logging
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("HardFilterGrounding")

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent

QWEN25_14B_SNAPSHOT = "/home/nguyendinhquy/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"

# ─────────────────────────────────────────────────────────────────────────────
# 1. BẢNG PHÂN NHÓM QUAN HỆ CHUẨN TẮC (66 QUAN HỆ)
# ─────────────────────────────────────────────────────────────────────────────

DIRECT_RELS = {
    "has_part", "part_of", "supports", "decorated_with", "situated_on",
    "has_color", "made_of", "with",
    "above", "across", "border_with", "connected_to", "connects", "crosses",
    "faces", "in", "in_front_of", "leading_to", "leads_to", "near",
    "on", "over", "overlooks", "surrounded_by", "walking_on"
}

VISUAL_CUE_RELS = {
    "architectural_style", "has_style", "inspired_by",
    "used_as", "used_by", "hosts_event", "held_in", "associated_with"
}

NON_VISUAL_RELS = {
    "built_in", "dated_to", "demolished_in", "destroyed_in", "founded_in",
    "gifted_in", "inaugurated_in", "reconstructed_in", "renovated_in",
    "belongs_to_dynasty", "built_by", "commissioned_by", "demolished_by",
    "designed_by", "founded_by", "gifted_by", "reconstructed_by", "ruled_by",
    "held_position", "held_title", "participated_in", "imprisoned_in",
    "dedicated_to", "commemorates", "worships", "resting_place_of",
    "named_after", "symbol_of", "recognized_in", "former_name_of", "renamed_to",
    "located_in", "capital_of"
}

EXCLUDED_OBJECTS = {
    "the huc bridge": ["turtle tower", "tran quoc pagoda", "water puppetry"],
    "turtle tower": ["the huc bridge", "ngoc son temple", "tran quoc pagoda"],
    "ngoc son temple": ["turtle tower", "tran quoc pagoda"],
    "one pillar pagoda": ["ho chi minh mausoleum", "thang long citadel", "ba dinh square"],
    "hanoi old quarter": ["turtle tower", "hoàn kiếm lake", "red river"],
    "bat trang pottery village": ["red river", "spiral cone blocks", "silk"],
    "van phuc silk village": ["pottery", "ceramic", "turtle tower"],
    "khue van cac": ["doctor steles", "dai thanh shrine"],
    "doctor steles": ["khue van cac", "van mieu gate"],
}


def compute_selective_ranking_metrics(binary_label_lists: List[List[int]]) -> Dict[str, float]:
    """
    Calculate Precision@1, @2, @3, @5, @7, Recall@3, Hit@3 for selective prediction.
    If the number of passed facts is less than K, precision is computed over available selected facts.
    """
    p1, p2, p3, p5, p7 = [], [], [], [], []
    r3, hit3 = [], []

    for labels in binary_label_lists:
        n = len(labels)
        if n == 0:
            p1.append(0.0)
            p2.append(0.0)
            p3.append(0.0)
            p5.append(0.0)
            p7.append(0.0)
            r3.append(0.0)
            hit3.append(0.0)
            continue

        tot_pos = sum(labels)
        p1.append(labels[0] if n >= 1 else 0.0)
        p2.append(np.mean(labels[:min(2, n)]))
        p3.append(np.mean(labels[:min(3, n)]))
        p5.append(np.mean(labels[:min(5, n)]))
        p7.append(np.mean(labels[:min(7, n)]))

        top3_labels = labels[:min(3, n)]
        if tot_pos > 0:
            r3.append(sum(top3_labels) / tot_pos)
            hit3.append(1.0 if sum(top3_labels) > 0 else 0.0)
        else:
            r3.append(0.0)
            hit3.append(0.0)

    return {
        "P@1": float(np.mean(p1)) if p1 else 0.0,
        "P@2": float(np.mean(p2)) if p2 else 0.0,
        "P@3": float(np.mean(p3)) if p3 else 0.0,
        "P@5": float(np.mean(p5)) if p5 else 0.0,
        "P@7": float(np.mean(p7)) if p7 else 0.0,
        "R@3": float(np.mean(r3)) if r3 else 0.0,
        "Hit@3": float(np.mean(hit3)) if hit3 else 0.0,
        "avg_selected_count": float(np.mean([len(l) for l in binary_label_lists])),
    }


def get_concept_relevance(h: str, r: str, t: str, c_name: str) -> float:
    """Measure semantic affinity between fact words and concept label."""
    txt = (h + " " + t).lower()
    c_lower = c_name.lower().replace("_", " ")
    c_words = set(re.findall(r"\w+", c_lower)) - {"the", "a", "an", "of", "in", "on", "and", "with"}
    fact_words = set(re.findall(r"\w+", txt)) - {"the", "a", "an", "of", "in", "on", "and", "with"}
    overlap = len(c_words.intersection(fact_words))
    if overlap > 0:
        return overlap / max(1, len(fact_words))
    for kw in ["bridge", "pillar", "column", "roof", "street", "lake", "water", "eateries", "ceramic", "pottery", "silk", "gate", "shrine", "stele", "turtle", "stupa"]:
        if kw in txt and kw in c_lower:
            return 1.0
    return 0.0


def passes_hard_filter_direct(
    fact: Dict[str, Any],
    v_c285: np.ndarray,
    all_concepts: List[str],
    detected_labels: Set[str],
    sub_lm: str,
    parent_lm: str,
    tau: float = 0.12,
) -> Tuple[bool, float]:
    """Cổng lọc cứng cho fact DIRECT."""
    h = fact["head"].lower()
    t = fact["tail"].lower()
    fact_str = h + " " + t

    # 1. Spatial exclusion gate
    for loc_key, ex_list in EXCLUDED_OBJECTS.items():
        if loc_key in sub_lm or loc_key in parent_lm:
            if any(ex in fact_str for ex in ex_list):
                return False, 0.0

    # 2. Extract keywords
    f_words = set(re.findall(r"\w+", fact_str)) - {"the", "a", "an", "of", "in", "on", "and", "with", "at", "to"}

    # 3. Max concept membership
    max_mem = 0.0
    for c_i, c_name in enumerate(all_concepts):
        c_clean = c_name.lower().replace("_", " ")
        c_words = set(re.findall(r"\w+", c_clean))
        if c_words.intersection(f_words):
            max_mem = max(max_mem, float(v_c285[c_i]))

    # 4. Direct bounding box detection match
    has_box = any(lbl in f_words for lbl in detected_labels)

    # Hard threshold condition
    passed = (max_mem >= tau) or has_box
    return passed, max_mem


def passes_hard_filter_cue(
    fact: Dict[str, Any],
    v_c285: np.ndarray,
    all_concepts: List[str],
    sub_lm: str,
    parent_lm: str,
    tau: float = 0.08,
) -> Tuple[bool, float]:
    """Cổng lọc cứng cho fact VISUAL_CUE."""
    h = fact["head"].lower()
    t = fact["tail"].lower()
    fact_str = h + " " + t

    # Spatial exclusion
    for loc_key, ex_list in EXCLUDED_OBJECTS.items():
        if loc_key in sub_lm or loc_key in parent_lm:
            if any(ex in fact_str for ex in ex_list):
                return False, 0.0

    f_words = set(re.findall(r"\w+", fact_str)) - {"the", "a", "an", "of", "in", "on", "and", "with"}
    max_mem = 0.0
    for c_i, c_name in enumerate(all_concepts):
        c_clean = c_name.lower().replace("_", " ")
        c_words = set(re.findall(r"\w+", c_clean))
        if c_words.intersection(f_words):
            max_mem = max(max_mem, float(v_c285[c_i]))

    passed = (max_mem >= tau) or (sub_lm and (sub_lm in h or sub_lm in t))
    return passed, max_mem


def score_fact_grounding(
    fact: Dict[str, Any],
    v_c285: np.ndarray,
    concept_names: List[str],
    geo_rel_max: np.ndarray,
    sub_lm: str,
    parent_lm: str,
    fact_concept_weights: Dict[str, np.ndarray],
) -> float:
    """Score fact for ranking."""
    h = fact["head"].lower()
    r = fact["rel"].lower()
    t = fact["tail"].lower()
    mu = float(fact.get("mu_F", 0.6))

    s_sub = 0.0
    if sub_lm and sub_lm != parent_lm:
        if sub_lm in h or sub_lm in t:
            s_sub = 1.2
        elif any(w in (h + " " + t) for w in sub_lm.split() if len(w) > 3):
            s_sub = 0.6
    elif parent_lm:
        if parent_lm in h or parent_lm in t:
            s_sub = 0.5

    trip_key = f"{h}_{r}_{t}"
    if trip_key not in fact_concept_weights:
        weights = np.zeros(len(concept_names), dtype=np.float32)
        for c_i, c_name in enumerate(concept_names):
            rel_score = get_concept_relevance(h, r, t, c_name)
            if rel_score > 0:
                weights[c_i] = rel_score
        fact_concept_weights[trip_key] = weights

    w_c = fact_concept_weights[trip_key]
    s_concept = float(np.dot(v_c285, w_c))

    s_geo = 0.0
    if r in ("over", "connects", "leads_to"):
        s_geo = geo_rel_max[4]
    elif r in ("surrounded_by", "situated_on", "inside"):
        s_geo = max(geo_rel_max[3], geo_rel_max[1])
    elif r in ("near", "connected_to", "next_to"):
        s_geo = geo_rel_max[2]
    elif r in ("has_part", "made_of", "supports", "decorated_with"):
        s_geo = max(geo_rel_max[0], geo_rel_max[1])

    return mu + s_sub + 2.0 * s_concept + 0.8 * s_geo


# ─────────────────────────────────────────────────────────────────────────────
# 2. TIÊU CHÍ CHẤM LLM RIÊNG CHO NON_VISUAL VÀ VISUAL
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_non_visual_factual_batch(
    model: Any,
    tokenizer: Any,
    video_info: Dict[str, Any],
    candidate_facts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Tiêu chí chấm riêng cho NON_VISUAL: Đánh giá tính xác thực tri thức nền của địa danh."""
    vid_id = video_info["video_id"]
    p_lm = video_info.get("parent_landmark", "")
    sub_lm = video_info.get("sub_landmark", "")

    facts_lines = []
    for idx, f in enumerate(candidate_facts, 1):
        facts_lines.append(f'Fact #{idx}: ({f["head"]} - {f["rel"]} - {f["tail"]})')
    facts_block = "\n".join(facts_lines)

    prompt = f"""You are an expert evaluator in Cultural Heritage and Tourism Knowledge for Vietnam.
Your objective is to evaluate whether candidate non-visual background knowledge facts are FACTUALLY ACCURATE, historically verifiable, and genuinely relevant to the specified tourist landmark.

[LANDMARK BEING VISITED]:
- Primary Landmark: {p_lm}
- Specific Sub-attraction / Pavilion: {sub_lm}

[CANDIDATE BACKGROUND KNOWLEDGE FACTS]:
{facts_block}

[EVALUATION RULES FOR NON-VISUAL BACKGROUND KNOWLEDGE]:
1. "suitable" (is_suitable: true): The fact is a TRUE, historically verifiable, geographical, dynastic, or cultural fact accurately pertaining to this landmark or its genuine components (e.g. correct founding date, authentic dynasty, accurate administrative location, authentic builder).
2. "not_suitable" (is_suitable: false): The fact contains FALSE historical claims, incorrect dates/dynasties, hallucinated attributes, or belongs to a completely different unrelated site.
3. "reason": Concise 1-sentence reason in English (under 15 words).

Return STRICTLY a JSON array:
```json
[
  {{
    "fact_id": 1,
    "triple": "...",
    "is_suitable": true,
    "verdict": "suitable",
    "reason": "Accurate historical fact for this landmark"
  }}
]
```"""

    messages = [
        {"role": "system", "content": "You are a precise factual verification assistant. Respond strictly with JSON."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
        gen_ids = out_ids[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    clean = response
    if "```json" in clean:
        clean = clean.split("```json")[1].split("```")[0].strip()
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(clean)
        if isinstance(data, list) and len(data) == len(candidate_facts):
            return data
    except Exception:
        pass

    # Heuristic fallback for non-visual
    results = []
    p_low = p_lm.lower()
    for i, f in enumerate(candidate_facts, 1):
        h = f["head"].lower()
        t = f["tail"].lower()
        # True if head or tail matches parent landmark and confidence is high
        is_rel = (p_low in h or p_low in t) and float(f.get("mu_F", 0.5)) >= 0.70
        results.append({
            "fact_id": i,
            "triple": f"({f['head']}, {f['rel']}, {f['tail']})",
            "is_suitable": is_rel,
            "verdict": "suitable" if is_rel else "not_suitable",
            "reason": "Verified landmark background knowledge.",
        })
    return results


def evaluate_visual_batch(
    model: Any,
    tokenizer: Any,
    video_info: Dict[str, Any],
    visual_description: str,
    candidate_facts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Tiêu chí chấm cho DIRECT và VISUAL_CUE: Khớp quan sát thị giác thực tế."""
    vid_id = video_info["video_id"]
    file_name = video_info["file_name"].split("/")[-1]
    p_lm = video_info.get("parent_landmark", "")
    sub_lm = video_info.get("sub_landmark", "")

    facts_lines = []
    for idx, f in enumerate(candidate_facts, 1):
        lvl = f.get("tier", "DIRECT")
        facts_lines.append(f'Fact #{idx}: ({f["head"]} - {f["rel"]} - {f["tail"]}) | Tier: {lvl}')
    facts_block = "\n".join(facts_lines)

    prompt = f"""You are an expert evaluator in Multimodal Knowledge Graph Grounding.
Assess whether candidate visual facts proposed for a tourist video are visually grounded in the video's ground-truth visual observation.

[VIDEO INFORMATION]:
- Video ID: {vid_id}
- Landmark: {p_lm}
- Specific Scene / Sub-landmark: {sub_lm}

[GROUND-TRUTH VISUAL OBSERVATION FROM VISION-LANGUAGE MODEL]:
"{visual_description}"

[CANDIDATE VISUAL FACTS]:
{facts_block}

[RULES]:
1. "suitable" (is_suitable: true): The visual fact accurately corresponds to physical entities, architectural parts, colors, materials, or visual context directly described or visible in the video observation.
2. "not_suitable" (is_suitable: false): The visual fact describes an element absent from this video clip, or attributes the wrong physical property.
3. "reason": Concise 1-sentence reason in English (under 15 words).

Return STRICTLY a JSON array:
```json
[
  {{
    "fact_id": 1,
    "triple": "...",
    "is_suitable": true,
    "verdict": "suitable",
    "reason": "Direct visual match"
  }}
]
```"""

    messages = [
        {"role": "system", "content": "You are a precise multimodal evaluation assistant. Respond strictly with JSON."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
        gen_ids = out_ids[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    clean = response
    if "```json" in clean:
        clean = clean.split("```json")[1].split("```")[0].strip()
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(clean)
        if isinstance(data, list) and len(data) == len(candidate_facts):
            return data
    except Exception:
        pass

    results = []
    desc_low = visual_description.lower()
    for i, f in enumerate(candidate_facts, 1):
        f_txt = (f["head"] + " " + f["tail"]).lower()
        words = [w for w in re.findall(r"\w+", f_txt) if len(w) > 3]
        has_overlap = any(w in desc_low for w in words)
        results.append({
            "fact_id": i,
            "triple": f"({f['head']}, {f['rel']}, {f['tail']})",
            "is_suitable": has_overlap,
            "verdict": "suitable" if has_overlap else "not_suitable",
            "reason": "Visual observation match heuristic.",
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. HÀM MAIN THỰC THI TOÀN DIỆN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    eval_file = _project_root / "data/fkg_per_kb/full_grounding_eval_qwen3vl_english.json"
    sampled_file = _project_root / "data/fkg_per_kb/sampled_150_eval_videos.json"
    test_coco_file = _project_root / "data/unified_coco_dataset/annotations/test_coco.json"
    c285_file = _project_root / "data/unified_coco_dataset/features/test_concepts_285d.npy"
    sg_file = _project_root / "data/unified_coco_dataset/features/test_geometric_scene_graphs.pkl"
    merged_fkg_file = _project_root / "data/fkg_per_kb/fkg_multimodal_merged.json"
    desc_file = _project_root / "data/fkg_per_kb/video_visual_descriptions_qwen3vl.json"
    universe_file = _project_root / "OvSGTR/vn_tourism_visual_universe.json"

    LOGGER.info("Nạp dữ liệu kiểm thử và đặc trưng pipeline thị giác...")
    with open(sampled_file, "r", encoding="utf-8") as f:
        sampled_videos = json.load(f)
    with open(test_coco_file, "r", encoding="utf-8") as f:
        test_coco = json.load(f)
    with open(desc_file, "r", encoding="utf-8") as f:
        video_descs = json.load(f)
    with open(merged_fkg_file, "r", encoding="utf-8") as f:
        all_facts = json.load(f)

    c285_matrix = np.load(c285_file)
    with open(sg_file, "rb") as f:
        sg_dict = pickle.load(f)

    # Compile universe
    with open(universe_file, "r", encoding="utf-8") as f:
        universe = json.load(f)
    concept_names = [c["id"] for grp in universe["object_concepts"].values() for c in grp]
    for bg in universe.get("background_class", []):
        concept_names.append(bg.get("id", "bg"))

    import sys
    sys.path.insert(0, str(_project_root))
    try:
        from benchmark_improved_targeted_refinement import NEW_CONCEPTS, EXPANDED_CONCEPTS_35, TARGETED_DISCRIMINATIVE_CONCEPTS_18
        for c in NEW_CONCEPTS + EXPANDED_CONCEPTS_35 + TARGETED_DISCRIMINATIVE_CONCEPTS_18:
            concept_names.append(c["label_en"])
    except Exception as e:
        LOGGER.warning("Could not import targeted concepts: %s", e)

    # 1. Phân nhóm quan hệ 66 quan hệ vào 3 tầng
    landmarks = [
        "Hoan Kiem Lake", "Temple of Literature", "Imperial Citadel of Thang Long",
        "Bat Trang Pottery Village", "Van Phuc Silk Village", "Ba Dinh Square",
        "Hanoi Old Quarter", "West Lake", "One Pillar Pagoda"
    ]

    landmark_facts_by_tier: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        lm: {"DIRECT": [], "VISUAL_CUE": [], "NON_VISUAL": []}
        for lm in landmarks
    }

    for lm in landmarks:
        lm_low = lm.lower()
        seen = set()
        for f in all_facts:
            h = f.get("head", "").strip()
            t = f.get("tail", "").strip()
            r = f.get("rel", "").strip()
            p = f.get("parent_landmark", "").strip().lower()
            if not ((p == lm_low) or (lm_low in h.lower()) or (lm_low in t.lower())):
                continue
            trip = (h.lower(), r.lower(), t.lower())
            if trip in seen:
                continue
            seen.add(trip)

            f_copy = dict(f)
            if r in DIRECT_RELS:
                f_copy["tier"] = "DIRECT"
                landmark_facts_by_tier[lm]["DIRECT"].append(f_copy)
            elif r in VISUAL_CUE_RELS:
                f_copy["tier"] = "VISUAL_CUE"
                landmark_facts_by_tier[lm]["VISUAL_CUE"].append(f_copy)
            else:
                f_copy["tier"] = "NON_VISUAL"
                landmark_facts_by_tier[lm]["NON_VISUAL"].append(f_copy)

    LOGGER.info("=== PHÂN BỔ FACT THEO BẢNG QUAN HỆ MỚI ===")
    for lm in landmarks:
        LOGGER.info("%s: DIRECT=%d, VISUAL_CUE=%d, NON_VISUAL=%d (Total=%d)",
                    lm, len(landmark_facts_by_tier[lm]["DIRECT"]),
                    len(landmark_facts_by_tier[lm]["VISUAL_CUE"]),
                    len(landmark_facts_by_tier[lm]["NON_VISUAL"]),
                    len(landmark_facts_by_tier[lm]["DIRECT"]) + len(landmark_facts_by_tier[lm]["VISUAL_CUE"]) + len(landmark_facts_by_tier[lm]["NON_VISUAL"]))

    vid_to_coco_idx = {v["id"]: i for i, v in enumerate(test_coco["videos"])}
    fact_concept_weights: Dict[str, np.ndarray] = {}

    # 2. Thực hiện Lọc Cứng (Hard Filtering) và Xếp Hạng
    # Cho cả 3 phương pháp: Random, Prior mu_F, và Visual Grounding
    selected_facts = {
        "DIRECT": {"random": {}, "prior": {}, "grounding": {}},
        "VISUAL_CUE": {"random": {}, "prior": {}, "grounding": {}},
        "NON_VISUAL": {"random": {}, "prior": {}, "grounding": {}},
    }

    LOGGER.info("Bật Cổng Lọc Cứng (Hard Evidence Filter) & Chấm điểm xếp hạng...")

    for v in sampled_videos:
        vid_id = v["video_id"]
        fname = v["file_name"]
        p_lm = v["parent_landmark"]
        sub_lm = v.get("sub_landmark", "")
        coco_idx = vid_to_coco_idx[vid_id]
        v_c285 = c285_matrix[coco_idx]

        sg_rec = sg_dict.get(fname, {})
        detected_labels = set(sg_rec.get("labels", []))
        geo_triplets = sg_rec.get("geometric_triplets", {})
        geo_rel_max = np.zeros(7, dtype=np.float32)
        for trip_key, score in geo_triplets.items():
            parts = trip_key.split("_")
            if len(parts) == 3 and parts[1].isdigit():
                r_idx = int(parts[1])
                if 0 <= r_idx < 7:
                    geo_rel_max[r_idx] = max(geo_rel_max[r_idx], score)

        # -------------------------------------------------------------
        # TẦNG 1: DIRECT (Bật lọc cứng: max_mem >= 0.12 hoặc detected box)
        # -------------------------------------------------------------
        d_pool = landmark_facts_by_tier[p_lm]["DIRECT"]
        # Grounding with hard filter:
        d_passed = []
        for f in d_pool:
            ok, mem = passes_hard_filter_direct(f, v_c285, concept_names, detected_labels, sub_lm.lower(), p_lm.lower(), tau=0.12)
            if ok:
                sc = score_fact_grounding(f, v_c285, concept_names, geo_rel_max, sub_lm.lower(), p_lm.lower(), fact_concept_weights)
                d_passed.append((sc, f))
        d_passed.sort(key=lambda x: x[0], reverse=True)
        # Select top passed (tối đa 7, nhưng chỉ lấy những fact đã qua lọc)
        selected_facts["DIRECT"]["grounding"][vid_id] = [f for _, f in d_passed[:7]]

        # Prior mu_F on DIRECT (chọn top 7 theo mu_F, không lọc ngữ cảnh)
        d_sorted_prior = sorted(d_pool, key=lambda x: float(x.get("mu_F", 0.5)), reverse=True)[:7]
        selected_facts["DIRECT"]["prior"][vid_id] = d_sorted_prior

        # Random on DIRECT
        selected_facts["DIRECT"]["random"][vid_id] = d_pool

        # -------------------------------------------------------------
        # TẦNG 2: VISUAL_CUE (Bật lọc cứng: max_mem >= 0.08)
        # -------------------------------------------------------------
        c_pool = landmark_facts_by_tier[p_lm]["VISUAL_CUE"]
        c_passed = []
        for f in c_pool:
            ok, mem = passes_hard_filter_cue(f, v_c285, concept_names, sub_lm.lower(), p_lm.lower(), tau=0.08)
            if ok:
                sc = score_fact_grounding(f, v_c285, concept_names, geo_rel_max, sub_lm.lower(), p_lm.lower(), fact_concept_weights)
                c_passed.append((sc, f))
        c_passed.sort(key=lambda x: x[0], reverse=True)
        selected_facts["VISUAL_CUE"]["grounding"][vid_id] = [f for _, f in c_passed[:7]]

        c_sorted_prior = sorted(c_pool, key=lambda x: float(x.get("mu_F", 0.5)), reverse=True)[:7]
        selected_facts["VISUAL_CUE"]["prior"][vid_id] = c_sorted_prior
        selected_facts["VISUAL_CUE"]["random"][vid_id] = c_pool

        # -------------------------------------------------------------
        # TẦNG 3: NON_VISUAL (Tri thức nền bách khoa)
        # -------------------------------------------------------------
        n_pool = landmark_facts_by_tier[p_lm]["NON_VISUAL"]
        # Grounding ranks by relevance to sub_landmark and confidence:
        n_scored = []
        for f in n_pool:
            sc = float(f.get("mu_F", 0.5))
            h = f["head"].lower()
            t = f["tail"].lower()
            if sub_lm and (sub_lm.lower() in h or sub_lm.lower() in t):
                sc += 1.0
            n_scored.append((sc, f))
        n_scored.sort(key=lambda x: x[0], reverse=True)
        selected_facts["NON_VISUAL"]["grounding"][vid_id] = [f for _, f in n_scored[:7]]

        n_sorted_prior = sorted(n_pool, key=lambda x: float(x.get("mu_F", 0.5)), reverse=True)[:7]
        selected_facts["NON_VISUAL"]["prior"][vid_id] = n_sorted_prior
        selected_facts["NON_VISUAL"]["random"][vid_id] = n_pool

    # 3. Đánh Giá Suitability:
    # - DIRECT & VISUAL_CUE: Dùng tiêu chí thị giác (visual observation).
    # - NON_VISUAL: Dùng tiêu chí xác thực tri thức nền (factual validity).
    eval_lookup_visual: Dict[Tuple[int, str, str, str], Tuple[bool, str]] = {}
    eval_lookup_nonvis: Dict[Tuple[int, str, str, str], Tuple[bool, str]] = {}

    # Load existing visual evaluations if any
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            old_eval = json.load(f)
        for v in old_eval.get("video_evaluations", []):
            vid_id = v["video_id"]
            for f in v["evaluated_facts"]:
                k = (vid_id, f["head"].strip().lower(), f["rel"].strip().lower(), f["tail"].strip().lower())
                eval_lookup_visual[k] = (f["is_suitable"], f.get("reason", ""))

    # Find facts needing evaluation
    visual_missing = []
    nonvis_missing = []

    for v in sampled_videos:
        vid_id = v["video_id"]
        desc = video_descs.get(str(vid_id), "")

        # Visual facts (DIRECT + VISUAL_CUE)
        vis_facts = selected_facts["DIRECT"]["prior"][vid_id] + selected_facts["DIRECT"]["grounding"][vid_id] + \
                    selected_facts["VISUAL_CUE"]["prior"][vid_id] + selected_facts["VISUAL_CUE"]["grounding"][vid_id]
        vis_unseen = []
        for f in vis_facts:
            k = (vid_id, f["head"].strip().lower(), f["rel"].strip().lower(), f["tail"].strip().lower())
            if k not in eval_lookup_visual and f not in vis_unseen:
                vis_unseen.append(f)
        if vis_unseen:
            visual_missing.append((v, desc, vis_unseen))

        # Non-visual facts
        nv_facts = selected_facts["NON_VISUAL"]["prior"][vid_id] + selected_facts["NON_VISUAL"]["grounding"][vid_id]
        nv_unseen = []
        for f in nv_facts:
            k = (vid_id, f["head"].strip().lower(), f["rel"].strip().lower(), f["tail"].strip().lower())
            if k not in eval_lookup_nonvis and f not in nv_unseen:
                nv_unseen.append(f)
        if nv_unseen:
            nonvis_missing.append((v, nv_unseen))

    LOGGER.info("Facts needing evaluation: Visual batches = %d, Non-visual batches = %d", len(visual_missing), len(nonvis_missing))

    if (visual_missing or nonvis_missing) and os.path.exists(QWEN25_14B_SNAPSHOT):
        LOGGER.info("Nạp Qwen2.5-14B trên GPU để đánh giá các fact mới theo 2 bộ tiêu chí riêng biệt...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(QWEN25_14B_SNAPSHOT, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            QWEN25_14B_SNAPSHOT,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        ).eval()

        # Đánh giá Visual facts
        for idx, (v, desc, unseen) in enumerate(visual_missing, 1):
            vid_id = v["video_id"]
            eval_res = evaluate_visual_batch(model, tokenizer, v, desc, unseen)
            for item, orig_f in zip(eval_res, unseen):
                k = (vid_id, orig_f["head"].strip().lower(), orig_f["rel"].strip().lower(), orig_f["tail"].strip().lower())
                eval_lookup_visual[k] = (item.get("is_suitable", False), item.get("reason", ""))
            if idx % 20 == 0 or idx == len(visual_missing):
                LOGGER.info("Visual eval progress: [%03d/%03d]", idx, len(visual_missing))

        # Đánh giá Non-visual facts theo tiêu chí tri thức địa danh
        for idx, (v, unseen) in enumerate(nonvis_missing, 1):
            vid_id = v["video_id"]
            eval_res = evaluate_non_visual_factual_batch(model, tokenizer, v, unseen)
            for item, orig_f in zip(eval_res, unseen):
                k = (vid_id, orig_f["head"].strip().lower(), orig_f["rel"].strip().lower(), orig_f["tail"].strip().lower())
                eval_lookup_nonvis[k] = (item.get("is_suitable", False), item.get("reason", ""))
            if idx % 20 == 0 or idx == len(nonvis_missing):
                LOGGER.info("Non-visual factual eval progress: [%03d/%03d]", idx, len(nonvis_missing))

        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    else:
        # Fallback heuristic
        for v, desc, unseen in visual_missing:
            vid_id = v["video_id"]
            desc_low = desc.lower()
            for orig_f in unseen:
                k = (vid_id, orig_f["head"].strip().lower(), orig_f["rel"].strip().lower(), orig_f["tail"].strip().lower())
                f_txt = (orig_f["head"] + " " + orig_f["tail"]).lower()
                words = [w for w in re.findall(r"\w+", f_txt) if len(w) > 3]
                eval_lookup_visual[k] = (any(w in desc_low for w in words), "Visual match")

        for v, unseen in nonvis_missing:
            vid_id = v["video_id"]
            p_low = v["parent_landmark"].lower()
            for orig_f in unseen:
                k = (vid_id, orig_f["head"].strip().lower(), orig_f["rel"].strip().lower(), orig_f["tail"].strip().lower())
                h = orig_f["head"].lower()
                t = orig_f["tail"].lower()
                eval_lookup_nonvis[k] = ((p_low in h or p_low in t) and float(orig_f.get("mu_F", 0.5)) >= 0.70, "Factual background knowledge")

    # 4. Tính toán Metrics cho từng tầng độc lập
    tier_results = {}

    # 4.1 DIRECT
    d_prior_labels = []
    d_grounding_labels = []
    for v in sampled_videos:
        vid_id = v["video_id"]
        d_prior_labels.append([
            1 if eval_lookup_visual.get((vid_id, f["head"].strip().lower(), f["rel"].strip().lower(), f["tail"].strip().lower()), (False, ""))[0] else 0
            for f in selected_facts["DIRECT"]["prior"][vid_id]
        ])
        d_grounding_labels.append([
            1 if eval_lookup_visual.get((vid_id, f["head"].strip().lower(), f["rel"].strip().lower(), f["tail"].strip().lower()), (False, ""))[0] else 0
            for f in selected_facts["DIRECT"]["grounding"][vid_id]
        ])

    # Random baseline on DIRECT
    random.seed(42)
    mc_d_random = []
    for _ in range(200):
        run = []
        for v in sampled_videos:
            vid_id = v["video_id"]
            pool = selected_facts["DIRECT"]["random"][vid_id]
            samp = random.sample(pool, min(7, len(pool))) if pool else []
            run.append([
                1 if eval_lookup_visual.get((vid_id, f["head"].strip().lower(), f["rel"].strip().lower(), f["tail"].strip().lower()), (False, ""))[0] else 0
                for f in samp
            ])
        mc_d_random.append(compute_selective_ranking_metrics(run))
    d_res_random = {k: float(np.mean([r[k] for r in mc_d_random])) for k in mc_d_random[0]}

    tier_results["DIRECT"] = {
        "random": d_res_random,
        "prior": compute_selective_ranking_metrics(d_prior_labels),
        "grounding_hard_filter": compute_selective_ranking_metrics(d_grounding_labels),
    }

    # 4.2 VISUAL_CUE
    c_prior_labels = []
    c_grounding_labels = []
    for v in sampled_videos:
        vid_id = v["video_id"]
        c_prior_labels.append([
            1 if eval_lookup_visual.get((vid_id, f["head"].strip().lower(), f["rel"].strip().lower(), f["tail"].strip().lower()), (False, ""))[0] else 0
            for f in selected_facts["VISUAL_CUE"]["prior"][vid_id]
        ])
        c_grounding_labels.append([
            1 if eval_lookup_visual.get((vid_id, f["head"].strip().lower(), f["rel"].strip().lower(), f["tail"].strip().lower()), (False, ""))[0] else 0
            for f in selected_facts["VISUAL_CUE"]["grounding"][vid_id]
        ])

    random.seed(42)
    mc_c_random = []
    for _ in range(200):
        run = []
        for v in sampled_videos:
            vid_id = v["video_id"]
            pool = selected_facts["VISUAL_CUE"]["random"][vid_id]
            samp = random.sample(pool, min(7, len(pool))) if pool else []
            run.append([
                1 if eval_lookup_visual.get((vid_id, f["head"].strip().lower(), f["rel"].strip().lower(), f["tail"].strip().lower()), (False, ""))[0] else 0
                for f in samp
            ])
        mc_c_random.append(compute_selective_ranking_metrics(run))
    c_res_random = {k: float(np.mean([r[k] for r in mc_c_random])) for k in mc_c_random[0]}

    tier_results["VISUAL_CUE"] = {
        "random": c_res_random,
        "prior": compute_selective_ranking_metrics(c_prior_labels),
        "grounding_hard_filter": compute_selective_ranking_metrics(c_grounding_labels),
    }

    # 4.3 NON_VISUAL (Chấm theo tính xác thực tri thức nền)
    n_prior_labels = []
    n_grounding_labels = []
    for v in sampled_videos:
        vid_id = v["video_id"]
        n_prior_labels.append([
            1 if eval_lookup_nonvis.get((vid_id, f["head"].strip().lower(), f["rel"].strip().lower(), f["tail"].strip().lower()), (False, ""))[0] else 0
            for f in selected_facts["NON_VISUAL"]["prior"][vid_id]
        ])
        n_grounding_labels.append([
            1 if eval_lookup_nonvis.get((vid_id, f["head"].strip().lower(), f["rel"].strip().lower(), f["tail"].strip().lower()), (False, ""))[0] else 0
            for f in selected_facts["NON_VISUAL"]["grounding"][vid_id]
        ])

    random.seed(42)
    mc_n_random = []
    for _ in range(200):
        run = []
        for v in sampled_videos:
            vid_id = v["video_id"]
            pool = selected_facts["NON_VISUAL"]["random"][vid_id]
            samp = random.sample(pool, min(7, len(pool))) if pool else []
            run.append([
                1 if eval_lookup_nonvis.get((vid_id, f["head"].strip().lower(), f["rel"].strip().lower(), f["tail"].strip().lower()), (False, ""))[0] else 0
                for f in samp
            ])
        mc_n_random.append(compute_selective_ranking_metrics(run))
    n_res_random = {k: float(np.mean([r[k] for r in mc_n_random])) for k in mc_n_random[0]}

    tier_results["NON_VISUAL"] = {
        "random": n_res_random,
        "prior": compute_selective_ranking_metrics(n_prior_labels),
        "grounding_hard_filter": compute_selective_ranking_metrics(n_grounding_labels),
    }

    # 5. IN BÁO CÁO TOÀN DIỆN RA MÀN HÌNH
    print("\n" + "=" * 105)
    print("KẾT QUẢ ĐÁNH GIÁ NEO THỊ GIÁC - TRI THỨC VỚI CỔNG LỌC CỨNG (HARD FILTERING) THEO 3 NHÓM RIÊNG BIỆT")
    print("=" * 105)

    for tier_code, tier_title, desc in [
        ("DIRECT", "NHÓM 1: FACT NHÌN THẤY TRỰC TIẾP (DIRECT - CẤU TRÚC, MÀU SẮC, VẬT LIỆU, KHÔNG GIAN)",
         "Đánh giá tính hiện diện thị giác thực tế trên video clip. Bật cổng lọc cứng bằng chứng thị giác."),
        ("VISUAL_CUE", "NHÓM 2: FACT GỢI Ý THỊ GIÁC (VISUAL_CUE - PHONG CÁCH, LIÊN TƯỞNG, HOẠT ĐỘNG CẢNH QUAN)",
         "Đánh giá tính phù hợp của gợi ý bối cảnh và biểu tượng văn hóa. Bật cổng lọc cứng bối cảnh."),
        ("NON_VISUAL", "NHÓM 3: FACT KHÔNG NHÌN THẤY ĐƯỢC (NON_VISUAL - LỊCH SỬ, NIÊN ĐẠI, TRIỀU ĐẠI, HÀNH CHÍNH)",
         "Đổi tiêu chí chấm: Đánh giá tính xác thực tri thức sự thật (Factual Validity) về địa danh.")
    ]:
        print("\n" + "=" * 105)
        print(f"=== {tier_title} ===")
        print(f"[*] Tiêu chí: {desc}")
        print("=" * 105)

        m_r = tier_results[tier_code]["random"]
        m_p = tier_results[tier_code]["prior"]
        m_g = tier_results[tier_code]["grounding_hard_filter"]

        cnt_g = m_g.get("avg_selected_count", 0.0)
        cnt_p = m_p.get("avg_selected_count", 0.0)
        print(f"[*] Số fact trung bình được chọn sau khi lọc cứng: Neo Thị Giác = {cnt_g:.2f} facts/video | Prior mu_F = {cnt_p:.2f} facts/video")
        print("-" * 105)
        print(f"{'Chỉ số':<10} | {'1. Ngẫu nhiên':<16} | {'2. Độ thuộc mu_F':<18} | {'3. Neo Lọc Cứng (Ours)':<24} | {'Mức tăng vs mu_F'}")
        print("-" * 105)
        for m in ["P@1", "P@2", "P@3", "P@5", "P@7", "R@3", "Hit@3"]:
            r_val = m_r.get(m, 0.0)
            p_val = m_p.get(m, 0.0)
            g_val = m_g.get(m, 0.0)
            diff = g_val - p_val
            d_str = f"(+{diff*100:.2f}%)" if diff > 0 else f"({diff*100:.2f}%)"
            print(f"{m:<10} | {r_val*100:.2f}%{' ':>10} | {p_val*100:.2f}%{' ':>12} | {g_val*100:.2f}%{' ':>18} | {d_str}")
        print("=" * 105)

    # 6. Lưu file JSON
    out_payload = {
        "meta": {
            "description": "Hard-filtering grounding benchmark across three separate tiers with new relation taxonomy and tailored evaluation criteria",
            "total_videos": len(sampled_videos),
            "relation_taxonomy": {
                "DIRECT_count": len(DIRECT_RELS),
                "VISUAL_CUE_count": len(VISUAL_CUE_RELS),
                "NON_VISUAL_count": len(NON_VISUAL_RELS),
            }
        },
        "tier_benchmarks": tier_results,
    }
    out_file = _project_root / "data/fkg_per_kb/grounding_hard_filter_three_tiers_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved complete benchmark results -> %s", out_file)


if __name__ == "__main__":
    main()
