#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_sft_subsets.py
=============================================================================
Generates supervised training subsets for SFT from Train/Val splits.

Integrates two complementary fact streams:
  1. Visual Grounding Facts: concepts detected during landmark visual grounding
     (scene focus, prominent visible entities, architectural components, spatial relations).
     Certainty = 1.0 (directly grounded in video frames).
  2. Cultural Knowledge Facts: from F_gold(V_i) and Horn closure
     - F_i^+ = F_gold(V_i) ∩ F_ref(TRUE) -> ASSERT or HEDGE
     - F_i^- = F_gold(V_i) ∩ F_ref(FALSE) -> OMIT

Generates 3 subset types:
  1. Clean Subsets: Visual Facts + TRUE Knowledge Facts
  2. Mixed Subsets: Visual Facts + TRUE + FALSE Knowledge Facts
  3. Counter-Bias Subsets: Visual Facts + TRUE (low mu_F) + FALSE (high mu_F)

Strict split isolation: ONLY uses Train and Val splits. Zero test data exposure.
=============================================================================
"""

import json
import random
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("PrepareSFTSubsets")

REPO_ROOT = Path("/home/nguyendinhquy/TransMTL_OntoKG-main")
if not REPO_ROOT.exists():
    REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")

OUTPUT_DIR = REPO_ROOT / "scene_conditioned_gold" / "outputs"
MASTER_JSON = OUTPUT_DIR / "scene_conditioned_fgold_horn_complete.json"
INVENTORY_TV = OUTPUT_DIR / "scene_inventory_train_val.jsonl"
OUT_SFT_SUBSETS = REPO_ROOT / "experiments" / "fact_to_caption" / "sft_subsets_train_val.json"


def load_inventories(path: Path) -> Dict[str, Any]:
    invs = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    invs[item["clip_id"]] = item
    return invs


def extract_visual_grounding_facts(inv: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Converts concepts detected in landmark visual grounding into explicit facts."""
    facts = []
    cid = inv.get("clip_id", "clip")
    
    # 1. Scene focus fact
    focus = inv.get("scene_focus")
    if focus:
        clean_focus = focus.strip().rstrip(".")
        facts.append({
            "fact_id": f"vis_focus_{cid}",
            "fact_type": "VISUAL_GROUNDING",
            "category": "SCENE_FOCUS",
            "fact_text": f"Visual observation: Scene focuses on {clean_focus}.",
            "mu_F": 1.0,
            "ref_status": "TRUE",
            "policy_label": "ASSERT"
        })
        
    # 2. Prominent visible entities
    vis_ents = inv.get("visible_entities", [])
    for idx, e in enumerate(vis_ents[:5]):
        ename = e.get("entity_name")
        etype = e.get("entity_type", "element")
        if ename:
            facts.append({
                "fact_id": f"vis_ent_{idx}_{cid}",
                "fact_type": "VISUAL_GROUNDING",
                "category": "VISIBLE_ENTITY",
                "fact_text": f"Visual observation: The scene visibly captures {ename} ({etype}).",
                "mu_F": 1.0,
                "ref_status": "TRUE",
                "policy_label": "ASSERT"
            })

    # 3. Spatial relations
    spatial_rels = inv.get("spatial_relations", [])
    for idx, r in enumerate(spatial_rels[:3]):
        subj = r.get("subject")
        rel = r.get("relation")
        obj = r.get("object")
        if subj and rel and obj:
            facts.append({
                "fact_id": f"vis_spatial_{idx}_{cid}",
                "fact_type": "VISUAL_GROUNDING",
                "category": "SPATIAL_RELATION",
                "fact_text": f"Visual relation: {subj} is positioned {rel} {obj}.",
                "mu_F": 0.95,
                "ref_status": "TRUE",
                "policy_label": "ASSERT"
            })

    return facts


def main():
    LOGGER.info("=" * 70)
    LOGGER.info("PREPARING SUPERVISED SFT SUBSETS WITH VISUAL GROUNDING FACTS")
    LOGGER.info("=" * 70)

    random.seed(42)

    with open(MASTER_JSON, "r", encoding="utf-8") as f:
        master = json.load(f)

    inventories = load_inventories(INVENTORY_TV)
    LOGGER.info(f"Loaded {len(inventories)} scene inventories.")

    subsets_by_split = {"train": [], "val": []}
    stats = defaultdict(lambda: {"clean": 0, "mixed": 0, "counter_bias": 0, "total": 0})

    for split in ["train", "val"]:
        clips = master[split]
        for cid, cdata in clips.items():
            lm_id = cdata["landmark_id"]
            lm_name = cdata["landmark_name"]
            inv = inventories.get(cid, {})
            
            # Extract visual grounding facts for this clip
            vis_facts = extract_visual_grounding_facts(inv)
            scene_context = [e.get("entity_name", "") for e in inv.get("visible_entities", []) if e.get("entity_name")]

            all_gold = cdata["gold_facts"]
            f_plus = [gf for gf in all_gold if gf.get("review", {}).get("result") == "đúng"]
            f_minus = [gf for gf in all_gold if gf.get("review", {}).get("result") == "sai"]

            if not f_plus:
                continue

            # 1. Generate Clean Subsets (3 subsets per clip)
            n_clean = min(3, len(f_plus))
            for i in range(n_clean):
                k = min(len(f_plus), random.randint(2, 4))
                sample_kg = random.sample(f_plus, k)
                
                # Sample 1-2 visual grounding facts
                sample_vis = random.sample(vis_facts, min(len(vis_facts), 2)) if vis_facts else []

                facts_payload = []
                # Visual facts
                for vf in sample_vis:
                    facts_payload.append(vf.copy())

                # Knowledge facts
                for f in sample_kg:
                    facts_payload.append({
                        "fact_id": f["fact_id"],
                        "fact_type": "KNOWLEDGE_FACT",
                        "fact_text": f["fact_text"],
                        "mu_F": round(random.uniform(0.75, 0.98), 2),
                        "ref_status": "TRUE",
                        "policy_label": "ASSERT"
                    })

                subsets_by_split[split].append({
                    "sample_id": f"{cid}_clean_{i+1}",
                    "clip_id": cid,
                    "split": split,
                    "landmark_name": lm_name,
                    "landmark_id": lm_id,
                    "scene_context": scene_context[:8],
                    "subset_type": "CLEAN",
                    "input_facts": facts_payload,
                    "target_assert_ids": [f["fact_id"] for f in facts_payload],
                    "target_omit_ids": []
                })
                stats[split]["clean"] += 1

            # 2. Generate Mixed Subsets (2 subsets per clip if f_minus available)
            if f_minus:
                n_mixed = 2
                for i in range(n_mixed):
                    k_pos = min(len(f_plus), random.randint(2, 3))
                    k_neg = min(len(f_minus), random.randint(1, 2))
                    sample_pos = random.sample(f_plus, k_pos)
                    sample_neg = random.sample(f_minus, k_neg)
                    sample_vis = random.sample(vis_facts, min(len(vis_facts), 2)) if vis_facts else []

                    facts_payload = []
                    assert_ids = []
                    omit_ids = []

                    for vf in sample_vis:
                        facts_payload.append(vf.copy())
                        assert_ids.append(vf["fact_id"])

                    for f in sample_pos:
                        facts_payload.append({
                            "fact_id": f["fact_id"],
                            "fact_type": "KNOWLEDGE_FACT",
                            "fact_text": f["fact_text"],
                            "mu_F": round(random.uniform(0.70, 0.95), 2),
                            "ref_status": "TRUE",
                            "policy_label": "ASSERT"
                        })
                        assert_ids.append(f["fact_id"])

                    for f in sample_neg:
                        facts_payload.append({
                            "fact_id": f["fact_id"],
                            "fact_type": "KNOWLEDGE_FACT",
                            "fact_text": f["fact_text"],
                            "mu_F": round(random.uniform(0.50, 0.85), 2),
                            "ref_status": "FALSE",
                            "policy_label": "OMIT"
                        })
                        omit_ids.append(f["fact_id"])

                    random.shuffle(facts_payload)

                    subsets_by_split[split].append({
                        "sample_id": f"{cid}_mixed_{i+1}",
                        "clip_id": cid,
                        "split": split,
                        "landmark_name": lm_name,
                        "landmark_id": lm_id,
                        "scene_context": scene_context[:8],
                        "subset_type": "MIXED",
                        "input_facts": facts_payload,
                        "target_assert_ids": assert_ids,
                        "target_omit_ids": omit_ids
                    })
                    stats[split]["mixed"] += 1

            # 3. Generate Counter-Bias Subsets (1 subset per clip)
            k_pos = min(len(f_plus), random.randint(2, 3))
            sample_pos = random.sample(f_plus, k_pos)
            sample_vis = random.sample(vis_facts, min(len(vis_facts), 1)) if vis_facts else []

            facts_payload = []
            assert_ids = []
            omit_ids = []

            for vf in sample_vis:
                facts_payload.append(vf.copy())
                assert_ids.append(vf["fact_id"])

            for idx, f in enumerate(sample_pos):
                if idx == 0:
                    # Low certainty TRUE fact -> HEDGE
                    facts_payload.append({
                        "fact_id": f["fact_id"],
                        "fact_type": "KNOWLEDGE_FACT",
                        "fact_text": f["fact_text"],
                        "mu_F": round(random.uniform(0.35, 0.48), 2),
                        "ref_status": "TRUE",
                        "policy_label": "HEDGE"
                    })
                else:
                    facts_payload.append({
                        "fact_id": f["fact_id"],
                        "fact_type": "KNOWLEDGE_FACT",
                        "fact_text": f["fact_text"],
                        "mu_F": round(random.uniform(0.80, 0.95), 2),
                        "ref_status": "TRUE",
                        "policy_label": "ASSERT"
                    })
                assert_ids.append(f["fact_id"])

            if f_minus:
                # High certainty FALSE fact -> OMIT
                f_bad = random.choice(f_minus)
                facts_payload.append({
                    "fact_id": f_bad["fact_id"],
                    "fact_type": "KNOWLEDGE_FACT",
                    "fact_text": f_bad["fact_text"],
                    "mu_F": round(random.uniform(0.88, 0.97), 2),
                    "ref_status": "FALSE",
                    "policy_label": "OMIT"
                })
                omit_ids.append(f_bad["fact_id"])

            random.shuffle(facts_payload)

            subsets_by_split[split].append({
                "sample_id": f"{cid}_counter_bias_1",
                "clip_id": cid,
                "split": split,
                "landmark_name": lm_name,
                "landmark_id": lm_id,
                "scene_context": scene_context[:8],
                "subset_type": "COUNTER_BIAS",
                "input_facts": facts_payload,
                "target_assert_ids": assert_ids,
                "target_omit_ids": omit_ids
            })
            stats[split]["counter_bias"] += 1

    total_train = len(subsets_by_split["train"])
    total_val = len(subsets_by_split["val"])
    LOGGER.info(f"Generated SFT Subsets with Visual Grounding:")
    LOGGER.info(f"  Train: {total_train:,} samples (Clean: {stats['train']['clean']}, Mixed: {stats['train']['mixed']}, Counter-bias: {stats['train']['counter_bias']})")
    LOGGER.info(f"  Val:   {total_val:,} samples (Clean: {stats['val']['clean']}, Mixed: {stats['val']['mixed']}, Counter-bias: {stats['val']['counter_bias']})")

    OUT_SFT_SUBSETS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SFT_SUBSETS, "w", encoding="utf-8") as f:
        json.dump(subsets_by_split, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Saved SFT subsets to {OUT_SFT_SUBSETS} ({OUT_SFT_SUBSETS.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
