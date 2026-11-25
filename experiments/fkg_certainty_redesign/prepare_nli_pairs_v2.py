#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_nli_pairs_v2.py
=============================================================================
Stage 2A (V2): Context-Enriched NLI Pair Generation with Evidence Relevance Gate.

Key Improvements:
  1. Ontology Schema Gate: Flags non-admissible triples (dummy headers, stopwords, food-city part_of).
  2. Landmark Context Enrichment: Prepends 'At {landmark_name}: ' to passage chunks to preserve
     entity binding for pronouns ("It") and section lists.
  3. Evidence Relevance Gate: Only admits secondary BM25 passages if they mention Subject or Object.
  4. Deduplication of passages with Jaccard overlap > 0.65.
=============================================================================
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("PrepareNLIPairsV2")

REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")
DICT_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "fgold_facts_dictionary.json"
BM25_PKL_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "bm25_indices.pkl"
OUTPUT_PAIRS_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "nli_inference_tasks_v2.json"

STOPWORDS = {"the", "a", "an", "and", "or", "in", "on", "at", "it", "this", "that", "there", ".", ",", "-", ";"}
DUMMY_HEADERS = {
    "characteristic physical features", "visual components", "physical features",
    "named places", "sections", "parts", "exterior", "inside", "performance", "design"
}
FOOD_ENTITIES = {
    "pho", "noodles", "rice noodles", "beef pho", "bun cha", "cha ca", "banh cuon",
    "broth", "fresh herbs", "street-side stools", "bowls of pho", "bun cha grills",
    "traditional southern vietnamese street food"
}

LANDMARK_NAMES = {
    "ba_dinh_square": "Ba Dinh Square",
    "ba_thien_hau_temple": "Ba Thien Hau Temple",
    "bat_trang_pottery_village": "Bat Trang Pottery Village",
    "ben_thanh_market": "Ben Thanh Market",
    "bitexco_financial_tower": "Bitexco Financial Tower",
    "bui_vien_walking_street": "Bui Vien Walking Street",
    "central_post_office": "Saigon Central Post Office",
    "cu_chi_tunnels": "Cu Chi Tunnels",
    "dam_sen_cultural_park": "Dam Sen Cultural Park",
    "dong_khoi_street": "Dong Khoi Street",
    "hanoi_cuisine": "Hanoi",
    "hanoi_old_quarter": "Hanoi Old Quarter",
    "hanoi_opera_house": "Hanoi Opera House",
    "ho_chi_minh_city_museum": "Ho Chi Minh City Museum",
    "hoan_kiem_lake": "Hoan Kiem Lake",
    "imperial_citadel_of_thang_long": "Imperial Citadel of Thang Long",
    "independence_palace": "Independence Palace",
    "landmark_81": "Landmark 81",
    "le_nin_park": "Le Nin Park",
    "nguyen_hue_walking_street": "Nguyen Hue Walking Street",
    "notre_dame_cathedral": "Notre Dame Cathedral",
    "one_pillar_pagoda": "One Pillar Pagoda",
    "quang_truong_lam_vien": "Lam Vien Square",
    "saigon_central_post_office": "Saigon Central Post Office",
    "saigon_notre_dame_basilica": "Notre Dame Cathedral Basilica of Saigon",
    "saigon_opera_house": "Saigon Opera House",
    "saigon_zoo_and_botanical_gardens": "Saigon Zoo and Botanical Gardens",
    "st_josephs_cathedral": "St. Joseph's Cathedral",
    "suoi_tien_theme_park": "Suoi Tien Theme Park",
    "ta_hien_beer_street": "Ta Hien Beer Street",
    "temple_of_literature": "Temple of Literature",
    "thang_long_water_puppetry_theatre": "Thang Long Water Puppetry Theatre",
    "thu_le_park": "Thu Le Park",
    "van_phuc_silk_village": "Van Phuc Silk Village",
    "vietnam_military_history_museum": "Vietnam Military History Museum",
    "vietnam_museum_of_ethnology": "Vietnam Museum of Ethnology",
    "vietnam_national_museum_of_history": "Vietnam National Museum of History",
    "vietnam_womens_museum": "Vietnam Women's Museum",
    "west_lake": "West Lake",
    "yen_so_park": "Yen So Park"
}


def clean_tok(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


def jaccard_similarity(s1: str, s2: str) -> float:
    t1 = set(clean_tok(s1))
    t2 = set(clean_tok(s2))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def is_admissible_triple(s: str, p: str, o: str) -> bool:
    s_c = s.strip().lower()
    o_c = o.strip().lower()
    p_c = p.strip().lower()

    if s_c in STOPWORDS or o_c in STOPWORDS or len(s_c) <= 1 or len(o_c) <= 1:
        return False
    if s_c in DUMMY_HEADERS or o_c in DUMMY_HEADERS:
        return False
    if s_c in FOOD_ENTITIES and p_c in {"part_of", "located_in"} and o_c in {"hanoi", "ha noi", "ho chi minh city", "saigon", "vietnam"}:
        return False
    macro_locs = {"hanoi", "ha noi", "vietnam", "ho chi minh city", "saigon", "cholon"}
    if s_c in macro_locs and p_c in {"part_of", "has_component", "located_inside"}:
        return False
    return True


def verbalize(s: str, p: str, o: str) -> str:
    s = s.strip()
    o = o.strip()
    p_clean = p.replace("_", " ")
    templates = {
        "located_in": f"{s} is located in {o}.",
        "located_inside": f"{s} is located inside {o}.",
        "houses": f"{s} houses {o}.",
        "part_of": f"{s} is part of {o}.",
        "has_part": f"{s} has part {o}.",
        "has_component": f"{s} has component {o}.",
        "dedicated_to": f"{s} is dedicated to {o}.",
        "architectural_style": f"{s} has an architectural style of {o}.",
        "designed_by": f"{s} was designed by {o}.",
        "built_by": f"{s} was built by {o}.",
        "renamed_to": f"{s} was renamed to {o}.",
        "former_name_of": f"{s} is a former name of {o}.",
        "named_after": f"{s} was named after {o}.",
        "belongs_to_dynasty": f"{s} belongs to the {o}.",
        "adjacent_to": f"{s} is adjacent to {o}.",
        "connected_to": f"{s} is connected to {o}.",
        "surrounded_by": f"{s} is surrounded by {o}.",
        "near": f"{s} is near {o}.",
        "border_with": f"{s} borders with {o}.",
        "commemorates": f"{s} commemorates {o}.",
        "founded_by": f"{s} was founded by {o}.",
        "worships": f"{s} worships {o}.",
        "on": f"{s} is situated on {o}.",
        "above": f"{s} is above {o}.",
        "below": f"{s} is below {o}.",
    }
    return templates.get(p, f"{s} {p_clean} {o}.")


def enrich_premise_with_landmark_context(premise_text: str, landmark_id: str) -> str:
    lm_name = LANDMARK_NAMES.get(landmark_id, landmark_id.replace("_", " ").title())
    if lm_name.lower() in premise_text.lower():
        return premise_text
    return f"At {lm_name}: {premise_text}"


def prepare_pairs_v2():
    LOGGER.info("Loading BM25 indices...")
    with open(BM25_PKL_PATH, "rb") as f:
        bm25_indices = pickle.load(f)

    LOGGER.info("Loading facts dictionary...")
    with open(DICT_PATH, "r", encoding="utf-8") as f:
        facts_dict = json.load(f)["facts"]

    LOGGER.info(f"Loaded {len(facts_dict):,} facts.")

    all_tasks: List[Dict[str, Any]] = []
    task_id_counter = 0
    admissible_count = 0

    for fid, fact in facts_dict.items():
        s = fact.get("subject", "").strip()
        p = fact.get("predicate", "").strip()
        o = fact.get("object", "").strip()
        lmid = fact.get("landmark_id", "")
        hypo = verbalize(s, p, o)

        # Check admissibility
        is_adm = is_admissible_triple(s, p, o)
        if not is_adm:
            # Triples failing schema admissibility are flagged and not scored by NLI
            continue

        admissible_count += 1
        ev_list: List[Dict[str, Any]] = []

        # 1. Original sentence if present
        orig_ev = fact.get("evidence", {})
        orig_sent = (orig_ev.get("original_sentence") or "").strip()
        if len(orig_sent) > 10:
            enriched_orig = enrich_premise_with_landmark_context(orig_sent, lmid)
            ev_list.append({
                "source_type": "extraction_origin",
                "source_doc": orig_ev.get("source_document", "unknown"),
                "text": enriched_orig,
                "raw_text": orig_sent,
                "reliability": 0.85,
            })

        # 2. BM25 retrieved passages with Evidence Relevance Gate
        s_l = s.lower()
        o_l = o.lower()
        q_toks = clean_tok(f"{s} {p} {o}")

        if q_toks:
            bm25, passages = bm25_indices.get(lmid, bm25_indices.get("__GLOBAL__", (None, [])))
            if bm25 and passages:
                scores = bm25.get_scores(q_toks)
                top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:5]
                for idx in top_indices:
                    score = float(scores[idx])
                    if score < 1.0:
                        continue
                    p_entry = passages[idx]
                    p_text = p_entry["text"]

                    # Evidence Relevance Gate: Must mention subject or object!
                    p_text_lower = p_text.lower()
                    if s_l not in p_text_lower and o_l not in p_text_lower:
                        continue

                    # Check deduplication against already added passages
                    is_dup = False
                    for existing in ev_list:
                        if jaccard_similarity(p_text, existing["raw_text"]) > 0.65:
                            is_dup = True
                            break
                    if not is_dup:
                        rel = 0.90 if p_entry["source_type"] == "expanded_nlp_corpus" else 0.80
                        if score < 2.5:
                            rel *= 0.90
                        enriched_p = enrich_premise_with_landmark_context(p_text, lmid)
                        ev_list.append({
                            "source_type": p_entry["source_type"],
                            "source_doc": p_entry["source_file"],
                            "text": enriched_p,
                            "raw_text": p_text,
                            "reliability": round(rel, 4),
                            "bm25_score": round(score, 3),
                        })
                    if len(ev_list) >= 4:
                        break

        # Register tasks
        for ev_idx, ev in enumerate(ev_list):
            task_entry = {
                "task_id": f"task_{task_id_counter:07d}",
                "fact_id": fid,
                "evidence_idx": ev_idx,
                "premise": ev["text"],
                "hypothesis": hypo,
                "source_type": ev["source_type"],
                "source_doc": ev["source_doc"],
                "reliability": ev["reliability"],
                "landmark_id": lmid,
                "provenance_type": fact.get("provenance_type", "Unknown"),
            }
            all_tasks.append(task_entry)
            task_id_counter += 1

    LOGGER.info(f"Admissible facts: {admissible_count:,} / {len(facts_dict):,} (Filtered: {len(facts_dict)-admissible_count:,}).")
    LOGGER.info(f"Generated {len(all_tasks):,} NLI inference pairs.")
    avg_ev = len(all_tasks) / admissible_count if admissible_count > 0 else 0
    LOGGER.info(f"Average evidence passages per admissible fact: {avg_ev:.2f}")

    with open(OUTPUT_PAIRS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_tasks, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Saved V2 NLI inference tasks to {OUTPUT_PAIRS_PATH}")


if __name__ == "__main__":
    prepare_pairs_v2()
