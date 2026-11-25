#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_c0_c2_tasks.py
=============================================================================
Generates NLI task suites for C0 (Raw Premise) and C2 (Hypothesis-Leaked Premise),
adopting the user's explicit metadata schema:

Schema:
  - passage_id: task_id
  - document_id: doc_{landmark_id}
  - document_subject: {landmark_title}
  - document_subject_source: "verified_metadata"
  - raw_text: raw uncontextualized chunk
  - contextualized_text: Document-grounded context (C1)

Configurations:
  - C0: raw_text (no prefix)
  - C1: contextualized_text (pre-computed in nli_inference_tasks_v2.json)
  - C2: hypothesis-leaked context ("At {hypothesis_subject}: {raw_text}")
=============================================================================
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("PrepareC0C2Tasks")

REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")
DICT_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "fgold_facts_dictionary.json"
TASKS_V2_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "nli_inference_tasks_v2.json"

OUTPUT_C0_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "nli_tasks_c0_raw.json"
OUTPUT_C2_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "nli_tasks_c2_hypo.json"
OUTPUT_CORPUS_SCHEMA_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "corpus_metadata_enriched.json"

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


def build_c0_and_c2():
    LOGGER.info(f"Loading tasks from {TASKS_V2_PATH}...")
    with open(TASKS_V2_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    LOGGER.info(f"Loading facts dictionary from {DICT_PATH}...")
    with open(DICT_PATH, "r", encoding="utf-8") as f:
        facts_dict = json.load(f)["facts"]

    c0_tasks = []
    c2_tasks = []
    metadata_entries = []

    for t in tasks:
        fid = t["fact_id"]
        fact = facts_dict.get(fid, {})
        s = (fact.get("subject") or "").strip()
        lmid = t.get("landmark_id", "")
        lm_name = LANDMARK_NAMES.get(lmid, lmid.replace("_", " ").title())

        p_orig = t["premise"]
        prefix = f"At {lm_name}: "
        if p_orig.startswith(prefix):
            raw_text = p_orig[len(prefix):]
        else:
            raw_text = p_orig

        # C0: Raw passage
        t_c0 = dict(t)
        t_c0["premise"] = raw_text
        c0_tasks.append(t_c0)

        # C2: Hypothesis Subject Prepend
        t_c2 = dict(t)
        if s and s.lower() not in raw_text.lower():
            t_c2["premise"] = f"At {s}: {raw_text}"
        else:
            t_c2["premise"] = raw_text
        c2_tasks.append(t_c2)

        # Corpus metadata entry conforming to user schema
        metadata_entries.append({
            "passage_id": t["task_id"],
            "fact_id": fid,
            "document_id": f"doc_{lmid}",
            "document_subject": lm_name,
            "document_subject_source": "verified_metadata",
            "raw_text": raw_text,
            "contextualized_text": p_orig,
            "hypothesis_injected_text": t_c2["premise"]
        })

    LOGGER.info(f"Generated {len(c0_tasks):,} tasks for C0 (Raw).")
    LOGGER.info(f"Generated {len(c2_tasks):,} tasks for C2 (Hypothesis Subject).")

    with open(OUTPUT_C0_PATH, "w", encoding="utf-8") as f:
        json.dump(c0_tasks, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Saved C0 tasks to {OUTPUT_C0_PATH}")

    with open(OUTPUT_C2_PATH, "w", encoding="utf-8") as f:
        json.dump(c2_tasks, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Saved C2 tasks to {OUTPUT_C2_PATH}")

    # Save a sample of corpus schema metadata (first 1000)
    with open(OUTPUT_CORPUS_SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_entries[:1000], f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Saved sample corpus metadata schema to {OUTPUT_CORPUS_SCHEMA_PATH}")


if __name__ == "__main__":
    build_c0_and_c2()
