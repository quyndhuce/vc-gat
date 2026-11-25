#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare 250 Test Prompts with Real Subgraphs & Epistemic Uncertainty
===================================================================
Trích xuất 250 mẫu từ tập test (1.119 video) cân bằng 3 mức (HIGH, MEDIUM, LOW),
được trang bị Subgraph thị giác thực tế và Subgraph tri thức lan truyền trọng số.
"""

import argparse
import json
import logging
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("Prepare250TestPrompts")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
UNIFIED_COCO_DIR = DATA_DIR / "unified_coco_dataset"
ANNOTATIONS_DIR = UNIFIED_COCO_DIR / "annotations"
EXP_FEAT_DIR = REPO_ROOT / "experiments" / "features"
FKG_MERGED_PATH = DATA_DIR / "fkg_per_kb" / "fkg_multimodal_merged.json"
OUTPUT_TEST_PATH = UNIFIED_COCO_DIR / "test_250_prompts.json"

SYSTEM_PROMPT = """You are a tourism narrator for Vietnamese heritage sites. You describe
what a camera is currently showing, one caption at a time, paced for dense video narration.

STRICT RULES:
1. Use ONLY the information given in the OBSERVED and KNOWLEDGE blocks.
   Never add facts from your own memory, even if you are confident.
2. Never state a name, date, number, material or person that does not
   appear in the blocks below.
3. Obey the CERTAINTY level exactly. It controls which names you may use.
4. Strictly obey the required sentence count and word count for the video segment duration so narration aligns with video playback.
5. Natural spoken narration style, no bullet points, no meta commentary.
6. Do not mention confidence scores, the knowledge graph, or this prompt."""


def clean_concept_name(cid: str) -> str:
    raw = cid.strip()
    if raw.startswith("bg_"):
        raw = raw[3:]
    return raw.replace("_", " ")


def verbalize_fkg_triple(head: str, rel: str, tail: str) -> str:
    r = rel.lower().strip()
    if r == "located_in":
        return f"{head} is located in {tail}"
    elif r == "renamed_to":
        return f"{head} was formerly known as {tail}"
    elif r in ("founded_in", "built_in"):
        return f"{head} was constructed in {tail}"
    elif r in ("founded_by", "built_by"):
        return f"{head} was founded by {tail}"
    elif r == "made_of":
        return f"{head} is built with {tail}"
    elif r in ("has_part", "includes"):
        return f"{head} features {tail}"
    elif r in ("part_of", "belongs_to"):
        return f"{head} is part of {tail}"
    elif r in ("surrounded_by", "bordered_by"):
        return f"{head} is surrounded by {tail}"
    elif r == "architectural_style":
        return f"{head} exhibits {tail} architecture"
    elif r == "has_color":
        return f"{head} is colored in {tail}"
    elif r in ("across", "spanning_across"):
        return f"{head} spans across {tail}"
    elif r in ("near", "next_to", "adjacent_to"):
        return f"{head} is situated near {tail}"
    else:
        return f"{head} {rel.replace('_', ' ')} {tail}"


def get_pacing_constraints(duration: float) -> Tuple[str, str]:
    if duration < 5.0:
        pacing_desc = "exactly 1 sentence (15 to 25 words)"
        pacing_rule = "Write exactly 1 sentence (15 to 25 words) for real-time dense video narration."
    elif duration < 10.0:
        pacing_desc = "1 to 2 sentences (25 to 40 words)"
        pacing_rule = "Write 1 to 2 sentences (25 to 40 words) for real-time dense video narration."
    else:
        pacing_desc = "2 to 3 sentences (40 to 60 words)"
        pacing_rule = "Write 2 to 3 sentences (40 to 60 words) for real-time dense video narration."
    return pacing_desc, pacing_rule


class LandmarkClassifier(nn.Module):
    def __init__(self, in_dim: int = 1053, num_classes: int = 30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def extract_visual_subgraph(
    sg_entry: Dict[str, Any],
    c285_vec: np.ndarray,
    concept_names: List[str],
    relation_names: List[str]
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    sparse_trips = sg_entry.get("sparse_triplets", {})
    sorted_trips = sorted(sparse_trips.items(), key=lambda x: x[1], reverse=True)
    distinct_trips = []
    seen = set()
    for trip_k, sc in sorted_trips:
        parts = trip_k.split("_")
        if len(parts) == 3:
            c1_i, r_i, c2_i = int(parts[0]), int(parts[1]), int(parts[2])
            if c1_i != c2_i and sc >= 0.12:
                c1_name = clean_concept_name(concept_names[c1_i])
                r_name = relation_names[r_i].replace("_", " ")
                c2_name = clean_concept_name(concept_names[c2_i])
                if (c1_name, r_name, c2_name) not in seen:
                    distinct_trips.append({
                        "head": c1_name,
                        "rel": r_name,
                        "tail": c2_name,
                        "score": round(float(sc), 2)
                    })
                    seen.add((c1_name, r_name, c2_name))
                    if len(distinct_trips) >= 3:
                        break

    top_indices = np.argsort(-c285_vec)
    visual_concepts = []
    for ci in top_indices:
        score = float(c285_vec[ci])
        if score >= 0.05:
            c_name = clean_concept_name(concept_names[ci])
            visual_concepts.append({
                "concept": c_name,
                "score": round(score, 2)
            })
            if len(visual_concepts) >= 4:
                break

    observed_lines = []
    for dt in distinct_trips:
        observed_lines.append(f"- ({dt['head']}, {dt['rel']}, {dt['tail']}) ({dt['score']:.2f})")
    for vc in visual_concepts:
        observed_lines.append(f"- {vc['concept']} ({vc['score']:.2f})")

    return observed_lines, distinct_trips, visual_concepts


def extract_knowledge_subgraph(
    parent_landmark: str,
    sub_landmark: str,
    p_conf: float,
    certainty_level: str,
    fkg_by_parent: Dict[str, List[Dict[str, Any]]],
    c285_vec: np.ndarray,
    concept_names: List[str],
    max_facts: int = 8
) -> Tuple[List[str], List[Dict[str, Any]]]:
    if certainty_level == "LOW" or p_conf < 0.40:
        return [], []

    cand_facts = fkg_by_parent.get(parent_landmark, [])
    scored_facts = []
    rel_counts = Counter()

    for f in cand_facts:
        h = f.get("head", "")
        r = f.get("rel", "")
        t = f.get("tail", "")
        mu_f = float(f.get("mu_F", 0.8))
        lvl = f.get("grounding_level", "NON_VISUAL")
        anchors = f.get("visual_anchors", [])

        if certainty_level == "MEDIUM":
            if sub_landmark and sub_landmark != parent_landmark and sub_landmark.lower() in h.lower():
                continue

        if lvl in ("DIRECT", "VISUAL_CUE") and anchors:
            anchor_scores = [float(c285_vec[concept_names.index(a)]) for a in anchors if a in concept_names]
            s_vis = max(anchor_scores) if anchor_scores else 0.1
            tau = 0.12 if lvl == "DIRECT" else 0.08
            if s_vis < tau:
                continue
        else:
            s_vis = 1.0

        mu_active = round(mu_f * s_vis * p_conf, 3)
        if mu_active >= 0.20:
            if r in ("located_in", "capital_of") and t.lower() in ("vietnam", "hanoi", "ha noi"):
                if rel_counts["admin"] >= 1:
                    continue
                rel_counts["admin"] += 1
            elif r == "surrounded_by":
                if rel_counts["surrounded_by"] >= 2:
                    continue
                rel_counts["surrounded_by"] += 1
            elif r == "border_with":
                if rel_counts["border_with"] >= 1:
                    continue
                rel_counts["border_with"] += 1

            boost = 0.25 if (certainty_level == "HIGH" and sub_landmark and sub_landmark.lower() in (h + " " + t).lower()) else 0.0
            scored_facts.append({
                "head": h,
                "rel": r,
                "tail": t,
                "mu_active": mu_active,
                "ranking_score": mu_active + boost,
                "grounding_level": lvl
            })

    scored_facts.sort(key=lambda x: x["ranking_score"], reverse=True)
    selected_facts = scored_facts[:max_facts]

    knowledge_lines = []
    for sf in selected_facts:
        v_fact = verbalize_fkg_triple(sf['head'], sf['rel'], sf['tail'])
        knowledge_lines.append(f"- {v_fact}")

    return knowledge_lines, selected_facts


def build_user_prompt(
    certainty_level: str,
    place_header: str,
    parent_landmark: str,
    candidate_places: List[str],
    observed_lines: List[str],
    knowledge_lines: List[str],
    duration: float = 5.0
) -> str:
    pacing_desc, pacing_rule = get_pacing_constraints(duration)
    obs_block = "\n".join(observed_lines)

    if certainty_level == "HIGH":
        know_block = "\n".join(knowledge_lines) if knowledge_lines else f"- {parent_landmark} is located in Hanoi"
        return f"""CERTAINTY: HIGH

PLACE: {place_header}
VIDEO DURATION: {duration:.1f} seconds (LENGTH REQUIREMENT: {pacing_desc})

OBSERVED (visual concepts detected, with membership degree):
{obs_block}

KNOWLEDGE (verified facts about this place):
{know_block}

RULES FOR THIS LEVEL:
- Explicitly name the place from PLACE at the beginning of the caption.
- State established knowledge facts about this place.
- Combine the visual observations with the verified knowledge facts.
- {pacing_rule}

Write the caption."""

    elif certainty_level == "MEDIUM":
        cand_str = "\n".join([f"- {c}" for c in candidate_places[:2]])
        know_block = "\n".join(knowledge_lines) if knowledge_lines else f"- {parent_landmark} is located in Hanoi"
        return f"""CERTAINTY: MEDIUM

PARENT PLACE: {parent_landmark}, Hanoi
CANDIDATE SPECIFIC PLACES (uncertain, do NOT name any of them):
{cand_str}
VIDEO DURATION: {duration:.1f} seconds (LENGTH REQUIREMENT: {pacing_desc})

OBSERVED (visual concepts detected, with membership degree):
{obs_block}

KNOWLEDGE (facts shared by all candidates, safe to use):
{know_block}

RULES FOR THIS LEVEL:
- You MUST NOT name any candidate specific place.
- You MAY name the parent place only.
- Describe what is visible, then place it in the parent site.
- Do not use words like "probably", "likely", "appears to be" with a specific name. Simply omit the specific name.
- {pacing_rule}

Write the caption."""

    else:  # LOW
        return f"""CERTAINTY: LOW

PLACE: unknown
VIDEO DURATION: {duration:.1f} seconds (LENGTH REQUIREMENT: {pacing_desc})

OBSERVED (visual concepts detected, with membership degree):
{obs_block}

KNOWLEDGE: none available at this certainty level.

RULES FOR THIS LEVEL:
- You MUST NOT name any place, person, dynasty, year or site.
- Describe only what the observed concepts show.
- No historical or cultural claims of any kind.
- Generic architectural vocabulary is allowed (pavilion, gateway, courtyard) as long as it names no specific site.
- {pacing_rule}

Write the caption."""


def main():
    parser = argparse.ArgumentParser(description="Prepare 250 Test Prompts for Evaluation")
    parser.add_argument("--num_samples", type=int, default=250, help="Total test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    LOGGER.info("=" * 80)
    LOGGER.info("PREPARING 250 TEST SAMPLES WITH REAL SUBGRAPHS FOR EVALUATION")
    LOGGER.info("=" * 80)

    # 1. Load Annotations & Taxonomy
    with open(ANNOTATIONS_DIR / "train_coco.json", "r", encoding="utf-8") as f:
        tc_tr = json.load(f)
    train_vids = tc_tr["videos"]
    train_anns = tc_tr["annotations"]

    with open(ANNOTATIONS_DIR / "test_coco.json", "r", encoding="utf-8") as f:
        tc_te = json.load(f)
    test_vids = tc_te["videos"]
    test_anns = tc_te["annotations"]

    with open(ANNOTATIONS_DIR / "taxonomy_categories.json", "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    parent_to_subs = defaultdict(list)
    for sc in taxonomy.get("sub_categories", []):
        p_name = sc.get("supercategory", "")
        s_name = sc.get("name", "")
        if p_name and s_name and s_name != p_name and s_name not in parent_to_subs[p_name]:
            parent_to_subs[p_name].append(s_name)

    # 2. Load Visual Features & Graphs
    LOGGER.info("Loading train/test multimodal features and scene graphs...")
    tr_sig = np.load(EXP_FEAT_DIR / "train_siglip_6f.npy")
    tr_cw = np.load(EXP_FEAT_DIR / "train_concepts_whole.npy")
    X_tr = np.concatenate([tr_sig, tr_cw], axis=-1)

    te_sig = np.load(EXP_FEAT_DIR / "test_siglip_6f.npy")
    te_cw = np.load(EXP_FEAT_DIR / "test_concepts_whole.npy")
    X_te = np.concatenate([te_sig, te_cw], axis=-1)

    with open(EXP_FEAT_DIR / "test_scene_graphs.pkl", "rb") as f:
        test_scene_graphs = pickle.load(f)

    # Load 285 concept universe
    with open(REPO_ROOT / "OvSGTR/vn_tourism_visual_universe.json", "r", encoding="utf-8") as f:
        u_data = json.load(f)
    concept_names: List[str] = [b["id"] for b in u_data.get("background_class", [])]
    for grp, items in u_data.get("object_concepts", {}).items():
        for it in items:
            concept_names.append(it["id"])

    sys.path.insert(0, str(REPO_ROOT))
    from benchmark_augmented_concepts import NEW_CONCEPTS
    from expanded_concept_universe_267 import EXPANDED_CONCEPTS_35
    from targeted_discriminative_concepts import TARGETED_DISCRIMINATIVE_CONCEPTS_18
    for c in NEW_CONCEPTS: concept_names.append(c["id"])
    for c in EXPANDED_CONCEPTS_35: concept_names.append(c["id"])
    for c in TARGETED_DISCRIMINATIVE_CONCEPTS_18: concept_names.append(c["id"])
    assert len(concept_names) == 285

    from OvSGTR.fkg.relations import RELATION_NAMES

    # 3. Load & Index Global FKG
    with open(FKG_MERGED_PATH, "r", encoding="utf-8") as f:
        all_fkg = json.load(f)

    parent_list = sorted(list(set(a["parent_landmark"] for a in train_anns)))
    fkg_by_parent = defaultdict(list)
    for p in parent_list:
        p_norm = p.lower().replace(" ", "_")
        matched = [
            f for f in all_fkg
            if (f.get("parent_landmark") == p)
            or (p_norm in f.get("head_uri", "").lower())
            or (p.lower() in f.get("head", "").lower())
            or (p_norm in f.get("tail_uri", "").lower())
        ]
        fkg_by_parent[p] = matched

    # 4. Train Landmark Classifier & Predict Calibrated Probabilities on Test
    LOGGER.info("Training classifier on train set and predicting calibrated test probabilities...")
    p2idx = {p: i for i, p in enumerate(parent_list)}
    idx2p = {i: p for p, i in p2idx.items()}
    y_tr = np.array([p2idx[a["parent_landmark"]] for a in train_anns])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_clf = LandmarkClassifier(in_dim=1053, num_classes=len(parent_list)).to(device)
    opt = torch.optim.AdamW(model_clf.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.04)

    ds_tr = torch.utils.data.TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    ld_tr = torch.utils.data.DataLoader(ds_tr, batch_size=64, shuffle=True)

    model_clf.train()
    for ep in range(35):
        for bx, by in ld_tr:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(model_clf(bx), by)
            loss.backward()
            opt.step()

    model_clf.eval()
    with torch.no_grad():
        test_logits = model_clf(torch.tensor(X_te, dtype=torch.float32).to(device))
        test_probs = F.softmax(test_logits, dim=-1).cpu().numpy()

    # 5. Categorize Test Pool by Certainty
    test_pool = []
    real_high_pool = []
    real_med_pool = []
    real_low_pool = []

    for idx, (v_info, a_info) in enumerate(zip(test_vids, test_anns)):
        v_id = v_info["id"]
        fname = v_info["file_name"]
        seg_id = v_info.get("segment_id", Path(fname).stem)
        gt_p = a_info.get("parent_landmark", "")
        gt_s = a_info.get("sub_landmark", "")
        dur = float(v_info.get("duration", 8.0))
        if dur <= 0.0: dur = 8.0

        prob_vec = test_probs[idx]
        pred_p_idx = int(np.argmax(prob_vec))
        p_conf = float(prob_vec[pred_p_idx])
        pred_p = idx2p[pred_p_idx]

        sg_entry = test_scene_graphs.get(v_id, {})
        c285_vec = te_cw[idx]
        obs_lines, vis_trips, vis_concs = extract_visual_subgraph(sg_entry, c285_vec, concept_names, RELATION_NAMES)

        item = {
            "test_idx": idx,
            "video_id": v_id,
            "segment_id": seg_id,
            "file_name": fname,
            "duration": round(dur, 2),
            "groundtruth_parent": gt_p,
            "groundtruth_sub": gt_s,
            "predicted_parent": pred_p,
            "confidence_score": round(p_conf, 4),
            "c285_vec": c285_vec,
            "observed_lines": obs_lines,
            "visual_subgraph": {
                "triplets": vis_trips,
                "concepts": vis_concs
            }
        }
        test_pool.append(item)

        if p_conf >= 0.70:
            real_high_pool.append(item)
        elif p_conf >= 0.40:
            real_med_pool.append(item)
        else:
            real_low_pool.append(item)

    LOGGER.info("Test Pool Counts (Total %d): HIGH=%d, MEDIUM=%d, LOW=%d",
                len(test_pool), len(real_high_pool), len(real_med_pool), len(real_low_pool))

    # Balanced 250 test samples: 90 HIGH, 80 MEDIUM, 80 LOW
    target_high = 90
    target_med = 80
    target_low = 80

    random.shuffle(real_high_pool)
    random.shuffle(real_med_pool)
    random.shuffle(real_low_pool)

    # Stratified selection for HIGH
    high_by_parent = defaultdict(list)
    for it in real_high_pool:
        high_by_parent[it["groundtruth_parent"]].append(it)

    selected_high = []
    for p in sorted(high_by_parent.keys()):
        if high_by_parent[p]:
            selected_high.append(high_by_parent[p].pop())
            if len(selected_high) >= target_high:
                break
    rem_high = [it for it in real_high_pool if it not in selected_high]
    while len(selected_high) < target_high and rem_high:
        selected_high.append(rem_high.pop())

    # MEDIUM items
    selected_med = real_med_pool[:target_med]
    needed_med = target_med - len(selected_med)
    donor_med = rem_high[:needed_med] if needed_med > 0 else []
    rem_high = rem_high[needed_med:]

    # LOW items
    selected_low = real_low_pool[:target_low]
    needed_low = target_low - len(selected_low)
    donor_low = rem_high[:needed_low] if needed_low > 0 else []

    LOGGER.info("Constructing 250 Test Prompts: %d HIGH | %d real MED + %d sim MED | %d real LOW + %d sim LOW",
                len(selected_high), len(selected_med), len(donor_med), len(selected_low), len(donor_low))

    final_test_records = []

    # A. HIGH (90 samples)
    for it in selected_high:
        gt_p = it["groundtruth_parent"]
        gt_s = it["groundtruth_sub"]
        p_conf = it["confidence_score"]
        c285_vec = it["c285_vec"]
        dur = it["duration"]

        place_header = f"{gt_s} (part of {gt_p}, Hanoi)" if (gt_s and gt_s != gt_p) else f"{gt_p}, Hanoi"
        know_lines, know_facts = extract_knowledge_subgraph(
            gt_p, gt_s, p_conf, "HIGH", fkg_by_parent, c285_vec, concept_names
        )
        prompt = build_user_prompt("HIGH", place_header, gt_p, [], it["observed_lines"], know_lines, dur)

        final_test_records.append({
            "test_idx": it["test_idx"],
            "video_id": it["video_id"],
            "segment_id": it["segment_id"],
            "file_name": it["file_name"],
            "duration": dur,
            "groundtruth_parent": gt_p,
            "groundtruth_sub": gt_s,
            "predicted_parent": it["predicted_parent"],
            "confidence_score": p_conf,
            "certainty_level": "HIGH",
            "source_type": "real",
            "place_header": place_header,
            "candidate_places": [],
            "visual_subgraph": it["visual_subgraph"],
            "knowledge_subgraph": know_facts,
            "observed_block": "\n".join(it["observed_lines"]),
            "knowledge_block": "\n".join(know_lines),
            "user_prompt": prompt,
        })

    # B. MEDIUM Real
    for it in selected_med:
        gt_p = it["groundtruth_parent"]
        gt_s = it["groundtruth_sub"]
        p_conf = it["confidence_score"]
        c285_vec = it["c285_vec"]
        dur = it["duration"]

        cand_list = list(parent_to_subs.get(gt_p, []))
        if gt_s and gt_s in cand_list: cand_list.remove(gt_s)
        other_cand = cand_list[0] if cand_list else f"{gt_p} pavilion"
        cand_places = [gt_s if (gt_s and gt_s != gt_p) else f"{gt_p} gateway", other_cand]

        know_lines, know_facts = extract_knowledge_subgraph(
            gt_p, gt_s, p_conf, "MEDIUM", fkg_by_parent, c285_vec, concept_names
        )
        prompt = build_user_prompt("MEDIUM", "", gt_p, cand_places, it["observed_lines"], know_lines, dur)

        final_test_records.append({
            "test_idx": it["test_idx"],
            "video_id": it["video_id"],
            "segment_id": it["segment_id"],
            "file_name": it["file_name"],
            "duration": dur,
            "groundtruth_parent": gt_p,
            "groundtruth_sub": gt_s,
            "predicted_parent": it["predicted_parent"],
            "confidence_score": p_conf,
            "certainty_level": "MEDIUM",
            "source_type": "real",
            "place_header": "",
            "candidate_places": cand_places,
            "visual_subgraph": it["visual_subgraph"],
            "knowledge_subgraph": know_facts,
            "observed_block": "\n".join(it["observed_lines"]),
            "knowledge_block": "\n".join(know_lines),
            "user_prompt": prompt,
        })

    # C. MEDIUM Simulated
    for it in donor_med:
        gt_p = it["groundtruth_parent"]
        gt_s = it["groundtruth_sub"]
        sim_conf = round(random.uniform(0.45, 0.65), 4)
        c285_vec = it["c285_vec"]
        dur = it["duration"]

        cand_list = list(parent_to_subs.get(gt_p, []))
        if gt_s and gt_s in cand_list: cand_list.remove(gt_s)
        other_cand = cand_list[0] if cand_list else f"{gt_p} pavilion"
        cand_places = [gt_s if (gt_s and gt_s != gt_p) else f"{gt_p} gateway", other_cand]

        know_lines, know_facts = extract_knowledge_subgraph(
            gt_p, gt_s, sim_conf, "MEDIUM", fkg_by_parent, c285_vec, concept_names
        )
        prompt = build_user_prompt("MEDIUM", "", gt_p, cand_places, it["observed_lines"], know_lines, dur)

        final_test_records.append({
            "test_idx": it["test_idx"],
            "video_id": it["video_id"],
            "segment_id": it["segment_id"],
            "file_name": it["file_name"],
            "duration": dur,
            "groundtruth_parent": gt_p,
            "groundtruth_sub": gt_s,
            "predicted_parent": it["predicted_parent"],
            "confidence_score": sim_conf,
            "certainty_level": "MEDIUM",
            "source_type": "epistemic_simulation",
            "place_header": "",
            "candidate_places": cand_places,
            "visual_subgraph": it["visual_subgraph"],
            "knowledge_subgraph": know_facts,
            "observed_block": "\n".join(it["observed_lines"]),
            "knowledge_block": "\n".join(know_lines),
            "user_prompt": prompt,
        })

    # D. LOW Real
    for it in selected_low:
        dur = it["duration"]
        prompt = build_user_prompt("LOW", "", "", [], it["observed_lines"], [], dur)
        final_test_records.append({
            "test_idx": it["test_idx"],
            "video_id": it["video_id"],
            "segment_id": it["segment_id"],
            "file_name": it["file_name"],
            "duration": dur,
            "groundtruth_parent": it["groundtruth_parent"],
            "groundtruth_sub": it["groundtruth_sub"],
            "predicted_parent": it["predicted_parent"],
            "confidence_score": it["confidence_score"],
            "certainty_level": "LOW",
            "source_type": "real",
            "place_header": "",
            "candidate_places": [],
            "visual_subgraph": it["visual_subgraph"],
            "knowledge_subgraph": [],
            "observed_block": "\n".join(it["observed_lines"]),
            "knowledge_block": "",
            "user_prompt": prompt,
        })

    # E. LOW Simulated
    for it in donor_low:
        dur = it["duration"]
        sim_conf = round(random.uniform(0.12, 0.38), 4)
        prompt = build_user_prompt("LOW", "", "", [], it["observed_lines"], [], dur)
        final_test_records.append({
            "test_idx": it["test_idx"],
            "video_id": it["video_id"],
            "segment_id": it["segment_id"],
            "file_name": it["file_name"],
            "duration": dur,
            "groundtruth_parent": it["groundtruth_parent"],
            "groundtruth_sub": it["groundtruth_sub"],
            "predicted_parent": it["predicted_parent"],
            "confidence_score": sim_conf,
            "certainty_level": "LOW",
            "source_type": "epistemic_simulation",
            "place_header": "",
            "candidate_places": [],
            "visual_subgraph": it["visual_subgraph"],
            "knowledge_subgraph": [],
            "observed_block": "\n".join(it["observed_lines"]),
            "knowledge_block": "",
            "user_prompt": prompt,
        })

    assert len(final_test_records) == args.num_samples
    LOGGER.info("Total test prompts prepared: %d", len(final_test_records))
    t_counts = Counter(r["certainty_level"] for r in final_test_records)
    for t, c in sorted(t_counts.items()):
        LOGGER.info("  %s: %d (%.1f%%)", t, c, c / len(final_test_records) * 100)

    with open(OUTPUT_TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(final_test_records, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved 250 test prompts to %s", OUTPUT_TEST_PATH)


if __name__ == "__main__":
    main()
