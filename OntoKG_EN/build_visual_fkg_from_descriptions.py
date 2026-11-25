#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal Fuzzy Knowledge Graph Builder from Visual Descriptions & Video Evidence
===================================================================================
Xây dựng Đồ thị Tri thức Mờ Đa phương thức (Multimodal FKG) từ mô tả thị giác:
  1. Chỉ sử dụng trường `visual_descriptions` (overview, architecture, surroundings)
     từ file `data/hanoi_landmarks_visual_descriptions.json`.
  2. Gán độ tin cậy tiên nghiệm cho dữ liệu văn bản: mu_text = 0.60.
  3. Trích xuất thực thể & quan hệ thị giác bằng mô hình GLiNER-Relex.
  4. Neo ngữ nghĩa vào vũ trụ khái niệm thị giác (Visual Universe / OvSGTR 207 concepts + 78 expanded).
  5. Đối soát bằng chứng thị giác từ video thực tế (grounded_segments_cache / unified_coco_dataset):
     - Tính điểm quan sát thực tế S_video từ các video segment của địa danh.
     - Hợp nhất độ tin cậy bằng T-conorm (Algebraic Sum):
         mu_fused = 1 - (1 - mu_text) * (1 - S_video)
     - Phân tầng neo thị giác: DIRECT (>= 0.50), VISUAL_CUE (0.20 - 0.50), NON_VISUAL (< 0.20).
  6. Hợp nhất vào Đồ thị tri thức toàn cục (data/fkg_per_kb/fkg_grounded.json) thành
     Đồ thị hoàn chỉnh kết hợp cả Tri thức văn bản (Lịch sử, Hành chính) và Thị giác (Kiến trúc, Vật liệu, Bối cảnh).

Usage:
  # Chế độ thử nghiệm 2-3 địa danh:
  python3 OntoKG_EN/build_visual_fkg_from_descriptions.py --pilot --device cuda

  # Chế độ toàn bộ 30 địa danh:
  python3 OntoKG_EN/build_visual_fkg_from_descriptions.py --full --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Thêm root vào sys.path
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("VisualFKGBuilder")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cấu hình Vũ trụ Khái niệm & Ánh xạ từ khóa Thị giác (Concept Keyword Map)
# ─────────────────────────────────────────────────────────────────────────────

# Bảng ánh xạ danh từ riêng địa danh sang Key trong cache video OvSGTR
LANDMARK_TO_CACHE_KEY: Dict[str, str] = {
    "One Pillar Pagoda": "One Pillar Pagoda",
    "Hoan Kiem Lake": "Hoan Kiem Lake",
    "Temple of Literature": "Van Mieu Quoc Tu Giam",
    "Imperial Citadel of Thang Long": "HoangThanhThangLong",
    "West Lake": "West Lake",
    "Ba Dinh Square": "Ho Chi Minh's Mausoleum",
    "Hanoi Old Quarter": "The Old Quarter",
    "Vietnam National Museum of History": "Vietnam National Museum of History",
    "Bat Trang Pottery Village": "Bat Trang Pottery Village",
    "Van Phuc Silk Village": "Van Phuc Silk Village",
}

# Ánh xạ từ khóa tiếng Anh trong văn bản thị giác sang các concept ID của OvSGTR (207 concepts)
TEXT_TO_CONCEPT_MAP: Dict[str, List[str]] = {
    # Kiến trúc tôn giáo & cổ điển
    "pagoda": ["pagoda", "temple_den"],
    "shrine": ["temple_den", "pagoda"],
    "wooden shrine": ["temple_den", "pagoda"],
    "temple": ["temple_den", "pagoda"],
    "communal house": ["communal_house"],
    "stupa": ["stupa"],
    "bell tower": ["bell_tower_asian"],
    "three arched passages": ["three_entrance_gate"],
    "traditional gate": ["three_entrance_gate"],
    "gate": ["three_entrance_gate"],
    "van mieu gate": ["three_entrance_gate"],
    "ink slab gate": ["three_entrance_gate"],
    "pavilion": ["wooden_pavilion", "stele_house"],
    "two-storey pavilion": ["wooden_pavilion", "stele_house"],
    "khue van cac": ["wooden_pavilion", "stele_house"],
    "constellation of literature pavilion": ["wooden_pavilion", "stele_house"],
    "tower": ["stupa", "cham_tower", "tower_pagoda"],
    "turtle tower": ["tower_pagoda", "turtle_statue"],
    "pen tower": ["stupa", "cham_tower"],
    "hoa phong tower": ["stupa", "cham_tower"],
    "citadel": ["cham_tower", "three_entrance_gate"],
    "mausoleum": ["colonial_building", "government_building"],
    "presidential palace": ["colonial_building"],

    # Chi tiết kiến trúc & cấu kiện
    "stone pillar": ["stone_pillar"],
    "single stone pillar": ["stone_pillar"],
    "square stone columns": ["stone_pillar"],
    "pillars": ["stone_pillar", "wooden_pillar"],
    "wooden pillars": ["wooden_pillar"],
    "wooden brackets": ["wooden_pillar"],
    "red lacquered wooden columns": ["wooden_pillar"],
    "columns": ["stone_pillar", "wooden_pillar"],
    "roof": ["curved_tiled_roof"],
    "tiled roof": ["curved_tiled_roof"],
    "tiled roofs": ["curved_tiled_roof"],
    "curved tiled roof": ["curved_tiled_roof"],
    "curved tiled roofs": ["curved_tiled_roof"],
    "wide curved tiled roof": ["curved_tiled_roof"],
    "roof ridge": ["roof_dragon_ornament", "curved_tiled_roof"],
    "ridge of the roof": ["roof_dragon_ornament", "curved_tiled_roof"],
    "dragon figures": ["roof_dragon_ornament", "dragon_statue"],
    "dragons": ["roof_dragon_ornament", "dragon_statue"],
    "dragon": ["roof_dragon_ornament", "dragon_statue"],
    "round windows": ["moon_gate", "round_window"],
    "arched openings": ["three_entrance_gate"],
    "staircase": ["dragon_carved_stairs"],
    "stone staircase": ["dragon_carved_stairs"],
    "narrow stone staircase": ["dragon_carved_stairs"],
    "stone paths": ["dragon_carved_stairs"],
    "stone platform": ["stone_pillar"],
    "stepped stone platform": ["stone_pillar"],
    "railing": ["balcony_railing"],
    "stone railing": ["balcony_railing"],
    "red railings": ["balcony_railing"],
    "low brick railing": ["balcony_railing"],
    "brick walls": ["brick_wall"],
    "courtyard": ["temple_courtyard", "stele_house"],
    "courtyards": ["temple_courtyard", "stele_house"],
    "walled courtyards": ["temple_courtyard", "stele_house"],
    "five walled courtyards": ["temple_courtyard", "stele_house"],
    "gilded horizontal boards": ["parallel_sentence_board"],
    "horizontal boards": ["parallel_sentence_board"],

    # Bia đá & rùa
    "stone stelae": ["stele_house", "turtle_statue"],
    "stone stelae of doctors": ["stele_house", "turtle_statue"],
    "stelae": ["stele_house"],
    "stone turtles": ["turtle_statue", "stele_house"],
    "turtle": ["turtle_statue"],
    "statue": ["statue"],
    "le thai to statue": ["statue"],

    # Cảnh quan tự nhiên & mặt nước
    "bridge": ["red_wooden_bridge", "bamboo_bridge"],
    "wooden bridge": ["red_wooden_bridge"],
    "curved wooden bridge": ["red_wooden_bridge"],
    "the huc bridge": ["red_wooden_bridge"],
    "huc bridge": ["red_wooden_bridge"],
    "lake": ["lake", "lake_pond"],
    "hoan kiem lake": ["lake"],
    "pond": ["lake_pond", "lotus_pond"],
    "lotus pond": ["lotus_pond"],
    "square lotus pond": ["lotus_pond"],
    "square pond": ["lake_pond", "lotus_pond"],
    "well": ["lake_pond"],
    "well of heavenly clarity": ["lake_pond"],
    "water": ["lake", "lake_pond"],
    "lotus flower": ["lotus_motif", "lotus_pond"],
    "water lilies": ["lotus_pond"],
    "island": ["lake"],
    "small island": ["lake"],
    "jade island": ["lake"],
    "trees": ["ancient_tree", "banyan_tree"],
    "old trees": ["ancient_tree", "banyan_tree"],
    "willow trees": ["ancient_tree"],
    "banyan tree": ["banyan_tree", "ancient_tree"],
    "old banyan trees": ["banyan_tree", "ancient_tree"],
    "bodhi tree": ["ancient_tree"],
    "lawn": ["flower_garden"],
    "lawns": ["flower_garden"],
    "green lawn": ["flower_garden"],
    "garden": ["flower_garden"],

    # Hoạt động & Con người
    "visitors": ["tourist_group", "photo_taking"],
    "graduating students": ["tourist_group", "traditional_costume"],
    "honor guards": ["guard_soldier", "white_uniform"],
    "people": ["tourist_group"],
    "walking paths": ["walking_street"],
    "streets": ["ancient_street", "walking_street"],
    "flagpole": ["flag_tower", "national_flag"],
    "national flag": ["national_flag"],

    # Làng nghề truyền thống
    "pottery wheel": ["pottery_wheel_clay_sculpting", "ceramic_vase_display"],
    "clay": ["pottery_wheel_clay_sculpting"],
    "pottery": ["ceramic_vase_display"],
    "ceramic kiln": ["pottery_kiln"],
    "loom": ["traditional_silk_loom", "silk_fabric_bolts"],
    "silk loom": ["traditional_silk_loom"],
    "silk": ["silk_fabric_bolts"],
    "bronze casting": ["bronze_casting_workshop"],
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Chuẩn hóa & Ánh xạ thực thể sang URI
# ─────────────────────────────────────────────────────────────────────────────

def normalize_entity_key(text: str) -> str:
    """Chuẩn hóa xâu ký tự để làm key băm."""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s-]", "", t)
    t = re.sub(r"\s+", "_", t)
    return t

def resolve_entity_uri(
    mention: str,
    landmark_name: str,
    landmark_kb_id: str,
    registry_uri_map: Dict[str, str],
) -> Tuple[str, str]:
    """
    Phân giải mention sang canonical URI theo chuẩn TransMTL:
      - Nếu là chính landmark: gán URI của KB node
      - Nếu có trong shared_node_registry: dùng URI đó (Wikidata hoặc TransMTL KB)
      - Nếu là sub-component: neo scoped URI theo pattern: http://transmtl.vn/kb/{lm_id}#{comp_id}
    """
    m_clean = mention.strip()
    m_norm = normalize_entity_key(m_clean)
    lm_norm = normalize_entity_key(landmark_name)

    # 1. Trùng chính địa danh
    if m_norm == lm_norm or m_norm in lm_norm or lm_norm in m_norm:
        if landmark_kb_id in registry_uri_map:
            return registry_uri_map[landmark_kb_id], landmark_name
        return f"http://transmtl.vn/kb/{landmark_kb_id}", landmark_name

    # 2. Tra cứu trong shared_node_registry
    if m_norm in registry_uri_map:
        return registry_uri_map[m_norm], m_clean

    # 3. Phân cấp thành phần kiến trúc thuộc về Landmark
    scoped_uri = f"http://transmtl.vn/kb/{landmark_kb_id}#{m_norm}"
    return scoped_uri, m_clean


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hàm đối soát bằng chứng thị giác từ Video (Video Grounding Engine)
# ─────────────────────────────────────────────────────────────────────────────

def match_text_to_concepts(text: str) -> List[str]:
    """Tìm tất cả concept IDs phù hợp với cụm từ văn bản."""
    t_lower = text.lower().strip()
    matched = set()

    # Exact or substring match in text-to-concept map
    for kw, cids in TEXT_TO_CONCEPT_MAP.items():
        if kw in t_lower or (len(kw) > 4 and t_lower in kw):
            matched.update(cids)

    return list(matched)


def compute_video_evidence_for_fact(
    head: str,
    rel: str,
    tail: str,
    landmark_name: str,
    cache_data: Dict[str, Any],
    engine_concept_id_to_idx: Dict[str, int],
    lm_max_scores: Dict[str, np.ndarray],
    unified_coco_support: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Tính điểm quan sát thị giác thực tế S_video cho fact (head, rel, tail).
    Sử dụng cả cache grounded_segments và tập đặc trưng 285-d của video.
    """
    ADMIN_RELATIONS = {"located_in"}
    ADMIN_KEYWORDS = {
        "district", "city", "province", "country", "center", "hanoi", "ha noi",
        "dong da", "ba dinh", "old quarter", "western shore", "eastern shore"
    }

    t_lower = tail.lower().strip()
    # 1. Lọc fact hành chính / địa lý vĩ mô -> Không thể chứng minh bằng video
    if rel in ADMIN_RELATIONS and any(kw in t_lower for kw in ADMIN_KEYWORDS):
        return {
            "score": 0.0,
            "confirmed": False,
            "best_concept": "none",
            "matched_concepts": [],
            "grounding_level": "NON_VISUAL",
            "evidence_details": f"Administrative/geographic knowledge fact ({tail}), preserved with base text confidence 0.60",
        }

    head_concepts = match_text_to_concepts(head)
    tail_concepts = match_text_to_concepts(tail)
    all_matched_concepts = list(set(head_concepts + tail_concepts))

    # Nếu không khớp với khái niệm thị giác nào -> NON_VISUAL
    if not all_matched_concepts:
        return {
            "score": 0.0,
            "confirmed": False,
            "best_concept": "none",
            "matched_concepts": [],
            "grounding_level": "NON_VISUAL",
            "evidence_details": "No physical visual concept matched (abstract/symbolic)",
        }

    # Lấy vector điểm quan sát của địa danh này từ cache
    scores_vec = lm_max_scores.get(landmark_name)

    def get_score(cid: str) -> float:
        if scores_vec is not None and cid in engine_concept_id_to_idx:
            return float(scores_vec[engine_concept_id_to_idx[cid]])
        if unified_coco_support and landmark_name in unified_coco_support:
            return float(unified_coco_support[landmark_name].get(cid, 0.0))
        return 0.0

    concept_scores = [(cid, get_score(cid)) for cid in all_matched_concepts]
    concept_scores.sort(key=lambda x: x[1], reverse=True)

    head_sc = [get_score(c) for c in head_concepts] if head_concepts else [0.0]
    tail_sc = [get_score(c) for c in tail_concepts] if tail_concepts else [0.0]
    max_h = max(head_sc)
    max_t = max(tail_sc)

    # 2. Xử lý theo bản chất ngữ nghĩa của quan hệ
    SPATIAL_RELATIONS = {
        "situated_on", "in_front_of", "faces", "surrounded_by", "connects",
        "connected_to", "leads_to", "near", "supports"
    }

    if head_concepts and tail_concepts and rel in SPATIAL_RELATIONS:
        # Quan hệ không gian giữa 2 thực thể thị giác -> Yêu cầu cả 2 cùng xuất hiện trong video
        max_score = float(np.sqrt(max_h * max_t))
        best_concept = f"{head_concepts[0]}+{tail_concepts[0]}"
    elif rel == "inspired_by":
        # Cảm hứng biểu tượng văn hóa -> Tối đa chỉ là VISUAL_CUE
        max_score = min(0.45, float(max(max_h, max_t)))
        best_concept = concept_scores[0][0] if concept_scores else "symbolic"
    else:
        # Quan hệ thuộc tính, vật liệu, thành phần cấu trúc (has_part, made_of, has_color, decorated_with)
        max_score = concept_scores[0][1] if concept_scores else 0.0
        best_concept = concept_scores[0][0] if concept_scores else "attribute"

    # Phân loại tầng neo & ngưỡng xác thực
    if max_score >= 0.50:
        level = "DIRECT"
        confirmed = True
    elif max_score >= 0.20:
        level = "VISUAL_CUE"
        confirmed = False
    else:
        level = "NON_VISUAL"
        confirmed = False

    return {
        "score": round(max_score, 4),
        "confirmed": confirmed,
        "best_concept": best_concept,
        "matched_concepts": [{"concept": c, "score": round(s, 4)} for c, s in concept_scores],
        "grounding_level": level,
        "evidence_details": f"Video observation score={max_score:.3f} (concept='{best_concept}', level={level})",
    }


def fuse_fuzzy_confidence(mu_text: float, s_video: float, grounding_level: str) -> float:
    """
    Hợp nhất độ tin cậy mờ giữa Văn bản (mu_text = 0.60) và Video (S_video):
      - DIRECT (S_video >= 0.50): Hợp nhất T-conorm (Algebraic sum)
          mu_fused = 1 - (1 - mu_text) * (1 - S_video)
          Ví dụ: mu_text=0.6, S_video=0.95 -> mu_fused = 1 - 0.4*0.05 = 0.980
      - VISUAL_CUE (0.20 <= S_video < 0.50): Gia tăng nhẹ có kiểm soát
          mu_fused = 1 - (1 - mu_text) * (1 - S_video * 0.5)
      - NON_VISUAL (S_video < 0.20): Giữ nguyên độ tin cậy tiên nghiệm văn bản (0.60)
    """
    if grounding_level == "DIRECT":
        fused = 1.0 - (1.0 - mu_text) * (1.0 - s_video)
    elif grounding_level == "VISUAL_CUE":
        fused = 1.0 - (1.0 - mu_text) * (1.0 - s_video * 0.5)
    else:
        fused = mu_text

    return round(float(fused), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Trích xuất Thực thể & Quan hệ bằng GLiNER-Relex
# ─────────────────────────────────────────────────────────────────────────────

GLINER_ENTITY_LABELS = [
    "landmark", "architectural element", "building material", "color",
    "architectural style", "water body", "tree", "monument", "statue",
    "gate", "bridge", "tower", "pavilion", "courtyard", "shrine",
    "temple", "pagoda", "person", "location", "district"
]

GLINER_RELATION_LABELS = [
    "located in", "has part", "made of", "situated on", "connects",
    "connected to", "surrounded by", "in front of", "faces", "near",
    "decorated with", "inspired by", "supports", "leads to", "stands on",
    "built on", "has color", "has style"
]

# Chuẩn hóa tên quan hệ sang chuẩn Ontology TransMTL
RELATION_NORMALIZE = {
    "located in": "located_in",
    "has part": "has_part",
    "made of": "made_of",
    "situated on": "situated_on",
    "stands on": "situated_on",
    "built on": "situated_on",
    "connects": "connects",
    "connected to": "connected_to",
    "leads to": "leads_to",
    "surrounded by": "surrounded_by",
    "in front of": "in_front_of",
    "faces": "faces",
    "near": "near",
    "decorated with": "decorated_with",
    "inspired by": "inspired_by",
    "supports": "supports",
    "has color": "has_color",
    "has style": "has_style",
}


def extract_triples_with_gliner(
    text: str,
    gliner_model: Any,
    section_name: str,
    threshold: float = 0.35,
    relation_threshold: float = 0.30,
) -> List[Dict[str, Any]]:
    """Trích xuất triples từ một đoạn văn bản bằng GLiNER-Relex."""
    if not text.strip():
        return []

    try:
        entities, relations = gliner_model.inference(
            texts=[text],
            labels=GLINER_ENTITY_LABELS,
            relations=GLINER_RELATION_LABELS,
            threshold=threshold,
            relation_threshold=relation_threshold,
            return_relations=True,
        )
    except Exception as e:
        LOGGER.warning("Lỗi GLiNER inference trên đoạn text: %s", e)
        return []

    raw_triples = []
    for r in relations[0]:
        h_text = r["head"]["text"].strip()
        t_text = r["tail"]["text"].strip()
        rel_raw = r["relation"].strip().lower()
        score = float(r["score"])

        # Lọc bỏ self-loop hoặc mention quá ngắn
        if h_text.lower() == t_text.lower() or len(h_text) < 2 or len(t_text) < 2:
            continue

        rel_canon = RELATION_NORMALIZE.get(rel_raw, rel_raw.replace(" ", "_"))
        raw_triples.append({
            "head": h_text,
            "rel": rel_canon,
            "tail": t_text,
            "score": score,
            "head_label": r["head"].get("label", ""),
            "tail_label": r["tail"].get("label", ""),
            "section": section_name,
        })

    return raw_triples


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pipeline Xây dựng Đồ thị Thị giác Đầy đủ (Main Builder)
# ─────────────────────────────────────────────────────────────────────────────

def build_visual_fkg(
    landmarks_file: Path,
    registry_file: Path,
    cache_file: Path,
    coco_feat_dir: Path,
    target_landmarks: Optional[List[str]] = None,
    mu_text: float = 0.60,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """
    Xây dựng danh sách các fact thị giác mờ có bằng chứng video.
    """
    LOGGER.info("=" * 80)
    LOGGER.info("KHỞI CHẠY XÂY DỰNG MULTIMODAL FUZZY KNOWLEDGE GRAPH")
    LOGGER.info("  Độ tin cậy tiên nghiệm văn bản: mu_text = %.2f", mu_text)
    LOGGER.info("  Nguồn dữ liệu: Chỉ sử dụng trường 'visual_descriptions'")
    LOGGER.info("=" * 80)

    # 1. Load GLiNER-Relex
    LOGGER.info("[1/5] Nạp mô hình GLiNER-Relex (knowledgator/gliner-relex-large-v1.0)...")
    from gliner import GLiNER
    model = GLiNER.from_pretrained("knowledgator/gliner-relex-large-v1.0")
    if device == "cuda":
        model = model.to("cuda")
    LOGGER.info("  ✓ Đã nạp GLiNER-Relex thành công trên thiết bị %s", device)

    # 2. Load Registry & Cache
    LOGGER.info("[2/5] Nạp Shared Node Registry và Grounded Video Cache...")
    with open(registry_file, "r", encoding="utf-8") as f:
        registry = json.load(f)
    registry_uri_map = registry.get("kb_id_to_uri", {})

    with open(cache_file, "rb") as f:
        cache_data = pickle.load(f)
    engine_concepts = cache_data["engine_concepts"]
    engine_concept_id_to_idx = {c["id"]: i for i, c in enumerate(engine_concepts)}

    # Tính trước vector max membership cho các địa danh từ video cache
    support_segs = cache_data["support_segments"]
    test_vids = cache_data["test_videos"]

    lm_max_scores = {}
    for lm_name, cache_key in LANDMARK_TO_CACHE_KEY.items():
        all_mems = []
        if cache_key in support_segs:
            for s in support_segs[cache_key]:
                all_mems.append(s["concept_membership"])
        if cache_key in test_vids:
            for v in test_vids[cache_key]:
                for s in v["segments"]:
                    all_mems.append(s["concept_membership"])
        if all_mems:
            lm_max_scores[lm_name] = np.max(np.array(all_mems), axis=0)

    # Load COCO features nếu có (hỗ trợ các địa danh bổ sung)
    unified_coco_support = {}
    train_concepts_path = coco_feat_dir / "train_concepts_285d.npy"
    train_coco_ann_path = _project_root / "data/unified_coco_dataset/annotations/train_coco.json"
    if train_concepts_path.exists() and train_coco_ann_path.exists():
        try:
            feats_285 = np.load(train_concepts_path)
            with open(train_coco_ann_path) as f:
                coco_ann = json.load(f)
            # Gom trung bình hoặc max theo parent_landmark
            lm_video_indices = defaultdict(list)
            for idx, ann in enumerate(coco_ann["annotations"]):
                lm_video_indices[ann["parent_landmark"]].append(idx)
            for lm_k, idxs in lm_video_indices.items():
                if idxs:
                    sub_f = feats_285[idxs, :207] # 207 concept đầu tương ứng engine concepts
                    unified_coco_support[lm_k] = {
                        engine_concepts[ci]["id"]: float(np.max(sub_f[:, ci]))
                        for ci in range(min(207, sub_f.shape[1]))
                    }
            LOGGER.info("  ✓ Đã nạp bổ sung đặc trưng video COCO cho %d địa danh", len(unified_coco_support))
        except Exception as e:
            LOGGER.warning("Không nạp được COCO 285d features: %s", e)

    # 3. Đọc dữ liệu visual_descriptions
    LOGGER.info("[3/5] Đọc tập dữ liệu hanoi_landmarks_visual_descriptions.json...")
    with open(landmarks_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    all_landmarks = raw_data["landmarks"]

    if target_landmarks:
        landmarks_to_process = [lm for lm in all_landmarks if lm["name_en"] in target_landmarks]
    else:
        landmarks_to_process = all_landmarks

    LOGGER.info("  ✓ Tổng số địa danh cần xử lý: %d", len(landmarks_to_process))

    # 4. Trích xuất Fact & Đối soát Video
    LOGGER.info("[4/5] Thực hiện trích xuất fact & đối soát bằng chứng thị giác...")
    all_fused_facts: List[Dict[str, Any]] = []

    for lm_idx, lm in enumerate(landmarks_to_process, 1):
        lm_name = lm["name_en"]
        lm_id = lm["id"]
        # Xác định kb_id cho địa danh
        kb_id_cand = normalize_entity_key(lm_name)
        # Tra cứu chuẩn kb_id nếu có
        for k in registry_uri_map:
            if k == kb_id_cand or k in kb_id_cand or kb_id_cand in k:
                kb_id_cand = k
                break

        vis_desc = lm.get("visual_descriptions", {})
        LOGGER.info(
            "[%02d/%02d] Xử lý: %s (ID: %d, Sections: %s)",
            lm_idx, len(landmarks_to_process), lm_name, lm_id, list(vis_desc.keys())
        )

        seen_triples = set()

        for sec_name, sec_text in vis_desc.items():
            if not sec_text.strip():
                continue

            extracted = extract_triples_with_gliner(sec_text, model, sec_name)

            for item in extracted:
                h = item["head"]
                r = item["rel"]
                t = item["tail"]
                trip_key = (h.lower(), r, t.lower())
                if trip_key in seen_triples:
                    continue
                seen_triples.add(trip_key)

                # Phân giải URI
                h_uri, h_clean = resolve_entity_uri(h, lm_name, kb_id_cand, registry_uri_map)
                t_uri, t_clean = resolve_entity_uri(t, lm_name, kb_id_cand, registry_uri_map)

                # Đối soát Video Evidence
                video_ev = compute_video_evidence_for_fact(
                    head=h_clean,
                    rel=r,
                    tail=t_clean,
                    landmark_name=lm_name,
                    cache_data=cache_data,
                    engine_concept_id_to_idx=engine_concept_id_to_idx,
                    lm_max_scores=lm_max_scores,
                    unified_coco_support=unified_coco_support,
                )

                s_video = video_ev["score"]
                grounding_lvl = video_ev["grounding_level"]

                # Hợp nhất Fuzzy Confidence
                mu_fused = fuse_fuzzy_confidence(mu_text, s_video, grounding_lvl)

                fact_record = {
                    "head": h_clean,
                    "head_uri": h_uri,
                    "rel": r,
                    "tail": t_clean,
                    "tail_uri": t_uri,
                    "mu_text": mu_text,
                    "mu_F": mu_fused,
                    "grounding_level": grounding_lvl,
                    "parent_landmark": lm_name,
                    "visual_anchors": [mc["concept"] if isinstance(mc, dict) else str(mc) for mc in video_ev.get("matched_concepts", [])],
                    "video_evidence": video_ev,
                    "source_section": sec_name,
                    "nlp_confidence": round(item["score"], 4),
                }
                all_fused_facts.append(fact_record)

    LOGGER.info("[5/5] Hoàn tất trích xuất & neo bằng chứng thị giác.")
    LOGGER.info("  ✓ Tổng số fact thị giác thu được: %d", len(all_fused_facts))
    levels_count = Counter(f["grounding_level"] for f in all_fused_facts)
    LOGGER.info("  ✓ Phân phối tầng neo: %s", dict(levels_count))
    return all_fused_facts


def merge_with_global_fkg(
    visual_facts: List[Dict[str, Any]],
    global_fkg_path: Path,
    output_merged_path: Path,
) -> Dict[str, Any]:
    """
    Hợp nhất tập fact thị giác mới vào Global FKG hiện hữu (fkg_grounded.json).
    """
    LOGGER.info("=" * 80)
    LOGGER.info("HỢP NHẤT VÀO ĐỒ THỊ TRI THỨC TOÀN CỤC (GLOBAL FKG)")
    LOGGER.info("=" * 80)

    with open(global_fkg_path, "r", encoding="utf-8") as f:
        global_facts = json.load(f)
    LOGGER.info("  ✓ Đã nạp Global FKG gốc: %d facts", len(global_facts))

    # Xây dựng bảng tra cứu fact gốc: (head_norm, rel, tail_norm)
    existing_map = {}
    for i, f in enumerate(global_facts):
        k = (f["head"].strip().lower(), f["rel"].strip().lower(), f["tail"].strip().lower())
        existing_map[k] = i

    merged_facts = list(global_facts)
    new_facts_added = 0
    facts_updated = 0

    for vf in visual_facts:
        k = (vf["head"].strip().lower(), vf["rel"].strip().lower(), vf["tail"].strip().lower())
        if k in existing_map:
            # Fact đã tồn tại -> Cập nhật nâng cấp độ tin cậy bằng T-conorm nếu có video chứng minh
            orig_idx = existing_map[k]
            orig_fact = merged_facts[orig_idx]
            old_mu = float(orig_fact.get("mu_F", 0.5))
            new_mu = float(vf["mu_F"])

            # T-conorm fusion
            fused_mu = round(1.0 - (1.0 - old_mu) * (1.0 - new_mu), 4)
            orig_fact["mu_F"] = fused_mu

            # Nếu fact mới là DIRECT thì nâng cấp tầng neo
            if vf["grounding_level"] == "DIRECT":
                orig_fact["grounding_level"] = "DIRECT"
            orig_fact["video_evidence"] = vf["video_evidence"]
            facts_updated += 1
        else:
            # Fact mới hoàn toàn -> Thêm vào đồ thị
            merged_facts.append(vf)
            new_facts_added += 1

    LOGGER.info(
        "  ✓ Hợp nhất hoàn tất: %d facts cũ, %d facts mới thêm, %d facts được cập nhật",
        len(global_facts), new_facts_added, facts_updated
    )
    LOGGER.info("  ✓ Tổng quy mô Đồ thị Tri thức Đa phương thức mới: %d facts", len(merged_facts))

    # Ghi file kết quả
    with open(output_merged_path, "w", encoding="utf-8") as f:
        json.dump(merged_facts, f, ensure_ascii=False, indent=2)
    LOGGER.info("  ✓ Đã lưu Đồ thị Hợp nhất -> %s", output_merged_path)

    summary = {
        "original_facts_count": len(global_facts),
        "visual_facts_extracted": len(visual_facts),
        "new_facts_added": new_facts_added,
        "facts_updated_with_video": facts_updated,
        "total_merged_facts": len(merged_facts),
        "grounding_distribution": dict(Counter(f.get("grounding_level", "UNKNOWN") for f in merged_facts)),
    }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build Multimodal FKG from Visual Descriptions and Video Evidence")
    parser.add_argument("--pilot", action="store_true", help="Chạy thử nghiệm trên 3 địa danh tiêu biểu trước khi chạy toàn bộ")
    parser.add_argument("--full", action="store_true", help="Chạy trên toàn bộ 30 địa danh")
    parser.add_argument("--mu_text", type=float, default=0.60, help="Độ tin cậy tiên nghiệm của văn bản (mặc định: 0.60)")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị tính toán (cuda hoặc cpu)")
    parser.add_argument("--output_dir", type=str, default=str(_project_root / "data/fkg_per_kb"), help="Thư mục xuất file")
    args = parser.parse_args()

    landmarks_file = _project_root / "data/hanoi_landmarks_visual_descriptions.json"
    registry_file = _project_root / "data/fkg_per_kb/shared_node_registry.json"
    cache_file = _project_root / "OvSGTR/fkg/output/grounded_segments_cache.pkl"
    coco_feat_dir = _project_root / "data/unified_coco_dataset/features"
    global_fkg_path = _project_root / "data/fkg_per_kb/fkg_grounded.json"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pilot_targets = ["One Pillar Pagoda", "Hoan Kiem Lake", "Temple of Literature"]

    if args.pilot:
        LOGGER.info("=== CHẾ ĐỘ THỬ NGHIỆM PILOT TRÊN 3 ĐỊA DANH TIÊU BIỂU ===")
        visual_facts = build_visual_fkg(
            landmarks_file=landmarks_file,
            registry_file=registry_file,
            cache_file=cache_file,
            coco_feat_dir=coco_feat_dir,
            target_landmarks=pilot_targets,
            mu_text=args.mu_text,
            device=args.device,
        )

        pilot_out_file = out_dir / "pilot_visual_fkg_3_landmarks.json"
        with open(pilot_out_file, "w", encoding="utf-8") as f:
            json.dump(visual_facts, f, ensure_ascii=False, indent=2)
        LOGGER.info("Đã lưu kết quả thử nghiệm pilot -> %s", pilot_out_file)

        # In bảng kết quả chi tiết từng fact
        print("\n" + "=" * 110)
        print("BẢNG KIỂM TRA CHI TIẾT CÁC FACT THỊ GIÁC ĐƯỢC XÁC THỰC BẰNG BẰNG CHỨNG VIDEO (PILOT 3 LANDMARKS)")
        print("=" * 110)
        print(f"{'Landmark':<22} | {'Head':<20} | {'Relation':<14} | {'Tail':<22} | {'mu_text':<7} | {'S_video':<7} | {'mu_fused':<8} | {'Level':<10}")
        print("-" * 110)
        for f in visual_facts:
            s_vid = f["video_evidence"]["score"]
            print(
                f"{f['parent_landmark']:<22} | "
                f"{f['head'][:20]:<20} | "
                f"{f['rel'][:14]:<14} | "
                f"{f['tail'][:22]:<22} | "
                f"{f['mu_text']:<7.2f} | "
                f"{s_vid:<7.3f} | "
                f"{f['mu_F']:<8.4f} | "
                f"{f['grounding_level']:<10}"
            )
        print("=" * 110)

        # In thống kê theo tầng neo
        direct_facts = [f for f in visual_facts if f["grounding_level"] == "DIRECT"]
        cue_facts = [f for f in visual_facts if f["grounding_level"] == "VISUAL_CUE"]
        non_vis_facts = [f for f in visual_facts if f["grounding_level"] == "NON_VISUAL"]

        print(f"\n📊 TỔNG KẾT PILOT:")
        print(f"  • Tổng số facts trích xuất: {len(visual_facts)}")
        print(f"  • DIRECT (Video xác thực mạnh >= 0.50) : {len(direct_facts)} ({len(direct_facts)/len(visual_facts)*100:.1f}%) -> mu_fused: {np.mean([f['mu_F'] for f in direct_facts]):.3f}")
        print(f"  • VISUAL_CUE (Video quan sát một phần) : {len(cue_facts)} ({len(cue_facts)/len(visual_facts)*100:.1f}%) -> mu_fused: {np.mean([f['mu_F'] for f in cue_facts]):.3f}")
        print(f"  • NON_VISUAL (Khái niệm hành chính/vắng mặt) : {len(non_vis_facts)} ({len(non_vis_facts)/len(visual_facts)*100:.1f}%) -> mu_fused: {np.mean([f['mu_F'] for f in non_vis_facts]):.3f}")
        print("\n=> Hãy kiểm tra kỹ các fact trên trước khi chạy toàn bộ 30 địa danh!")

    elif args.full:
        LOGGER.info("=== CHẾ ĐỘ CHẠY TOÀN BỘ 30 ĐỊA DANH ===")
        visual_facts = build_visual_fkg(
            landmarks_file=landmarks_file,
            registry_file=registry_file,
            cache_file=cache_file,
            coco_feat_dir=coco_feat_dir,
            target_landmarks=None,
            mu_text=args.mu_text,
            device=args.device,
        )

        full_vis_file = out_dir / "fkg_visual_descriptions.json"
        with open(full_vis_file, "w", encoding="utf-8") as f:
            json.dump(visual_facts, f, ensure_ascii=False, indent=2)
        LOGGER.info("Đã lưu toàn bộ facts thị giác -> %s", full_vis_file)

        # Hợp nhất với Global FKG
        merged_file = out_dir / "fkg_multimodal_merged.json"
        merge_summary = merge_with_global_fkg(visual_facts, global_fkg_path, merged_file)
        print("\n" + "=" * 80)
        print("TỔNG KẾT HỢP NHẤT TOÀN CỤC:")
        print(json.dumps(merge_summary, indent=2, ensure_ascii=False))
        print("=" * 80)

    else:
        print("Vui lòng chỉ định --pilot (để thử nghiệm 2-3 địa danh) hoặc --full (để chạy toàn bộ 30 địa danh)!")


if __name__ == "__main__":
    main()
