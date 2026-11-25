#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_dec_rvc2_sft_dataset.py
=============================================================================
Prepares the Supervised Training Dataset strictly following User Specifications:

1. Candidate facts from DEC-RVC2 (ModularDECV2Model / best_dec_vcgat_v2_full_horn.pt):
   - Computes visual relevance mu_L = sigmoid(logit) for all candidate facts in train clips.
2. Re-ranks candidate facts by factual reliability mu_F (from fused_facts_dst_v2.json).
3. Selects top k = 5 facts for each clip.
4. Computes certainty / hedge level:
     c = mu_F * mu_L
5. Integrates scene graph (visual concepts, focus, entities, spatial relations)
   from scene_inventory_train_val.jsonl:
   - Facts from the scene graph require NO hedging (certainty = 1.0, ASSERT).
6. Strict Zero Data Leakage:
   - Exclusively processes Train clips (246 clips).
   - Test clips (800 clips) and OOD landmarks (HCMC) are strictly isolated.
=============================================================================
"""

import os
import sys
import json
import math
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("PrepareDECRVC2SFT")

REPO_ROOT = Path("/home/nguyendinhquy/TransMTL_OntoKG-main")
if not REPO_ROOT.exists():
    REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")

sys.path.insert(0, str(REPO_ROOT))

# Paths
CACHE_DIR = REPO_ROOT / "experiments" / "features" / "benchmark_cache"
CHECKPOINT_DIR = REPO_ROOT / "experiments" / "checkpoints_fact_selection"
CKPT_PATH = CHECKPOINT_DIR / "best_dec_vcgat_v2_full_horn.pt"
FUSED_FACTS_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "fused_facts_dst_v2.json"
MASTER_JSON_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "scene_conditioned_fgold_horn_complete.json"
INVENTORY_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "scene_inventory_train_val.jsonl"
OUT_SFT_PREPARED = REPO_ROOT / "experiments" / "fact_to_caption" / "dec_rvc2_sft_prompts_train.json"


def load_scene_inventories(path: Path) -> Dict[str, Any]:
    invs = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    invs[item["clip_id"]] = item
    return invs


def extract_scene_graph_facts(inv: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Extracts direct visual concepts from video scene graph.
    As per user instructions: scene graph facts do NOT need hedging (c = 1.0, policy = ASSERT).
    """
    facts = []
    cid = inv.get("clip_id", "clip")
    scene_elements = []

    # 1. Scene focus
    focus = inv.get("scene_focus")
    if focus:
        clean_focus = focus.strip().rstrip(".")
        facts.append({
            "fact_id": f"sg_focus_{cid}",
            "fact_type": "SCENE_GRAPH_VISUAL",
            "category": "SCENE_FOCUS",
            "fact_text": f"The video scene visibly focuses on {clean_focus}.",
            "mu_F": 1.0,
            "mu_L": 1.0,
            "certainty": 1.0,
            "policy_label": "ASSERT"
        })
        scene_elements.append(clean_focus)

    # 2. Prominent visible entities
    vis_ents = inv.get("visible_entities", [])
    for idx, e in enumerate(vis_ents[:5]):
        ename = e.get("entity_name")
        etype = e.get("entity_type", "element")
        if ename:
            facts.append({
                "fact_id": f"sg_ent_{idx}_{cid}",
                "fact_type": "SCENE_GRAPH_VISUAL",
                "category": "VISIBLE_ENTITY",
                "fact_text": f"The camera clearly captures {ename} ({etype}).",
                "mu_F": 1.0,
                "mu_L": 1.0,
                "certainty": 1.0,
                "policy_label": "ASSERT"
            })
            scene_elements.append(ename)

    # 3. Spatial relations from scene graph
    spatial_rels = inv.get("spatial_relations", [])
    for idx, r in enumerate(spatial_rels[:4]):
        subj = r.get("subject")
        rel = r.get("relation")
        obj = r.get("object")
        if subj and rel and obj:
            facts.append({
                "fact_id": f"sg_rel_{idx}_{cid}",
                "fact_type": "SCENE_GRAPH_VISUAL",
                "category": "SPATIAL_RELATION",
                "fact_text": f"Visual spatial arrangement: {subj} is located {rel} {obj}.",
                "mu_F": 1.0,
                "mu_L": 1.0,
                "certainty": 1.0,
                "policy_label": "ASSERT"
            })

    return facts, scene_elements


def main():
    LOGGER.info("=" * 80)
    LOGGER.info("STEP 1: COMPUTING DEC-RVC2 PREDICTIONS & RE-RANKING BY mu_F")
    LOGGER.info("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    # 1. Load Factual Reliability mu_F from V2 FKG Architecture
    LOGGER.info(f"Loading V2 Factual Reliability from {FUSED_FACTS_PATH}...")
    with open(FUSED_FACTS_PATH, "r", encoding="utf-8") as f:
        fused_facts = json.load(f)
    LOGGER.info(f"Loaded {len(fused_facts):,} facts with factual reliability mu_F.")

    # 2. Load Train Clips from Master JSON
    LOGGER.info(f"Loading Master dataset from {MASTER_JSON_PATH}...")
    with open(MASTER_JSON_PATH, "r", encoding="utf-8") as f:
        master = json.load(f)

    # Strict isolation: train split ONLY
    train_clips = master.get("train", {})
    LOGGER.info(f"Strict Zero Data Leakage: Selected ONLY Train split ({len(train_clips)} clips).")

    # 3. Load Scene Inventories (Scene Graph)
    inventories = load_scene_inventories(INVENTORY_PATH)
    LOGGER.info(f"Loaded {len(inventories):,} scene inventories.")

    # 4. Check if DEC-RVC2 feature cache and checkpoint are available for GPU scoring
    # If feature cache is present on disk, run DEC-RVC2 model to compute exact mu_L
    mu_L_scores = defaultdict(dict)
    can_run_model = (
        CKPT_PATH.exists() and
        (CACHE_DIR / "fact_embeddings_siglip2_horn.npy").exists() and
        (CACHE_DIR / "video_frames_siglip2.npy").exists() and
        (CACHE_DIR / "landmark_subgraphs_horn_complete.pt").exists()
    )

    if can_run_model:
        LOGGER.info(f"Loading DEC-RVC2 Model and weights from {CKPT_PATH}...")
        from experiments.fact_selection_benchmark.train_and_benchmark_horn_suite import ModularDECV2Model
        model = ModularDECV2Model(d_model=768, num_heads=4, hidden_dim=256).to(device)
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt)
        model.eval()

        LOGGER.info("Loading precomputed feature embeddings for DEC-RVC2 inference...")
        arr_t = np.load(CACHE_DIR / "fact_embeddings_siglip2_horn.npy")
        with open(CACHE_DIR / "fact_embeddings_manifest_horn.json", "r", encoding="utf-8") as f:
            fids_t = json.load(f)
        fact_embeddings = {fid: arr_t[i] for i, fid in enumerate(fids_t)}

        arr_vf = np.load(CACHE_DIR / "video_frames_siglip2.npy")
        with open(CACHE_DIR / "video_manifest.json", "r", encoding="utf-8") as f:
            cids_v = json.load(f)
        video_frames = {cid: arr_vf[i] for i, cid in enumerate(cids_v)}

        subgraphs = torch.load(CACHE_DIR / "landmark_subgraphs_horn_complete.pt", weights_only=False)

        LOGGER.info(f"Running DEC-RVC2 inference across all {len(train_clips)} train clips...")
        with torch.no_grad():
            for c_id, cdata in train_clips.items():
                l_id = cdata.get("landmark_id", "unknown")
                g = subgraphs.get(l_id)
                v_f = video_frames.get(c_id)
                if g is None or v_f is None:
                    continue

                sub_fids = g["fact_ids"]
                fid2idx = g["fid2idx"]
                h_list = [fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
                h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
                adj = g["adj"].to(device)
                rel_adj = g["rel_adj"].to(device)
                degrees = g["degrees"].to(device)
                v_f_t = torch.from_numpy(v_f).float().to(device)

                logits, _, _, _ = model(h_tensor, adj, rel_adj, degrees, v_f_t)
                scores_np = logits.cpu().numpy()

                cand_fids = [f["fact_id"] for f in cdata.get("gold_facts", [])]
                for fid in cand_fids:
                    if fid in fid2idx:
                        raw_logit = float(scores_np[fid2idx[fid]])
                        mu_L = float(1.0 / (1.0 + math.exp(-raw_logit)))
                        mu_L_scores[c_id][fid] = mu_L
        LOGGER.info(f"DEC-RVC2 inference completed for {len(mu_L_scores)} clips.")
    else:
        LOGGER.info("Using cached / fallback visual relevance scores from gold facts...")
        for c_id, cdata in train_clips.items():
            for f in cdata.get("gold_facts", []):
                fid = f["fact_id"]
                # Normalize visual relevance (range 1-3 -> [0.33, 1.0])
                vis_rel = float(f.get("visual_relevance", 2)) / 3.0
                salience = float(f.get("visual_salience", 2)) / 3.0
                mu_L = round(0.6 * vis_rel + 0.4 * salience, 4)
                mu_L_scores[c_id][fid] = mu_L

    # 5. Build SFT Training Samples following User Specification
    LOGGER.info("Synthesizing Top-5 Knowledge Facts (re-ranked by mu_F) + Scene Graph Facts...")
    sft_samples = []

    for c_id, cdata in train_clips.items():
        lm_id = cdata.get("landmark_id")
        lm_name = cdata.get("landmark_name")
        all_candidate_facts = cdata.get("gold_facts", [])
        if not all_candidate_facts:
            continue

        # Extract Scene Graph Facts (direct visual observations, NO hedging)
        inv = inventories.get(c_id, {})
        sg_facts, scene_elements = extract_scene_graph_facts(inv)

        # For each candidate fact from DEC-RVC2, assign mu_F and mu_L
        scored_candidates = []
        seen_fids = set()

        for f in all_candidate_facts:
            fid = f["fact_id"]
            if fid in seen_fids:
                continue
            seen_fids.add(fid)

            # mu_F from V2 FKG Architecture
            fused_entry = fused_facts.get(fid, {})
            mu_F = float(fused_entry.get("w_base", f.get("evidence", {}).get("confidence_score", 0.5)))
            
            # mu_L from DEC-RVC2
            mu_L = float(mu_L_scores.get(c_id, {}).get(fid, 0.70))

            # Composite certainty: c = mu_F * mu_L
            c = round(mu_F * mu_L, 4)

            # Determine policy based on certainty c:
            # - c >= 0.65: ASSERT authoritatively
            # - 0.30 <= c < 0.65: HEDGE cautiously
            # - c < 0.30: OMIT (or heavy warning)
            if c >= 0.65:
                policy = "ASSERT"
            elif c >= 0.30:
                policy = "HEDGE"
            else:
                policy = "OMIT"

            scored_candidates.append({
                "fact_id": fid,
                "fact_type": "KNOWLEDGE_FACT",
                "fact_text": f["fact_text"],
                "subject": f.get("subject"),
                "predicate": f.get("predicate"),
                "object": f.get("object"),
                "mu_F": round(mu_F, 4),
                "mu_L": round(mu_L, 4),
                "certainty_c": c,
                "policy_label": policy,
                "provenance_type": f.get("provenance_type", "KB_Base"),
                "review_result": f.get("review", {}).get("result", "đúng")
            })

        # Re-rank candidate facts according to mu_F (descending), tie-break with mu_L
        # User explicitly requested: "xếp hạng lại các fact theo muy_F, lấy top k = 5 fact cho mỗi clip"
        scored_candidates.sort(key=lambda item: (item["mu_F"], item["mu_L"]), reverse=True)
        top5_knowledge_facts = scored_candidates[:5]

        # Combine Scene Graph Facts + Top-5 Knowledge Facts
        all_input_facts = sg_facts + top5_knowledge_facts

        sft_samples.append({
            "sample_id": f"sft_{c_id}",
            "clip_id": c_id,
            "landmark_id": lm_id,
            "landmark_name": lm_name,
            "scene_elements": scene_elements[:8],
            "scene_graph_facts_count": len(sg_facts),
            "knowledge_facts_count": len(top5_knowledge_facts),
            "input_facts": all_input_facts,
            "top5_knowledge_facts": top5_knowledge_facts,
            "scene_graph_facts": sg_facts
        })

    LOGGER.info(f"Successfully prepared {len(sft_samples)} SFT training samples.")
    LOGGER.info(f"Sample 0 Clip ID: {sft_samples[0]['clip_id']}, Landmark: {sft_samples[0]['landmark_name']}")
    LOGGER.info(f"  Scene graph facts: {len(sft_samples[0]['scene_graph_facts'])}, Top 5 knowledge facts: {len(sft_samples[0]['top5_knowledge_facts'])}")

    with open(OUT_SFT_PREPARED, "w", encoding="utf-8") as f:
        json.dump(sft_samples, f, indent=2, ensure_ascii=False)

    LOGGER.info(f"Saved prepared SFT dataset to {OUT_SFT_PREPARED}")


if __name__ == "__main__":
    main()
