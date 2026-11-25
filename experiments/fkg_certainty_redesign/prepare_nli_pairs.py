#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_nli_pairs.py
=============================================================================
Stage 2A: Prepare (Premise, Hypothesis) Pairs for NLI Verification.

For all 16,347 unique facts:
  1. Verbalize triple (s, p, o) -> Hypothesis text h(f)
  2. Ingest original extraction sentence E_0 (if valid)
  3. Query BM25Okapi index for landmark-specific passages (top K=3 with score > 1.0)
  4. Deduplicate passages with lexical overlap > 0.65
  5. Export structured NLI inference tasks for GPU cross-encoder scoring.
=============================================================================
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("PrepareNLIPairs")

REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")
DICT_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "fgold_facts_dictionary.json"
BM25_PKL_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "bm25_indices.pkl"
OUTPUT_PAIRS_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "nli_inference_tasks.json"


def clean_tok(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric tokens."""
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


def jaccard_similarity(s1: str, s2: str) -> float:
    t1 = set(clean_tok(s1))
    t2 = set(clean_tok(s2))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


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


def prepare_pairs():
    LOGGER.info("Loading BM25 indices...")
    with open(BM25_PKL_PATH, "rb") as f:
        bm25_indices = pickle.load(f)

    LOGGER.info("Loading facts dictionary...")
    with open(DICT_PATH, "r", encoding="utf-8") as f:
        facts_dict = json.load(f)["facts"]

    LOGGER.info(f"Loaded {len(facts_dict):,} facts.")

    all_tasks: List[Dict[str, Any]] = []
    task_id_counter = 0

    for fid, fact in facts_dict.items():
        s = fact.get("subject", "").strip()
        p = fact.get("predicate", "").strip()
        o = fact.get("object", "").strip()
        lmid = fact.get("landmark_id", "")
        hypo = verbalize(s, p, o)

        ev_list: List[Dict[str, Any]] = []

        # 1. Original sentence if present
        orig_ev = fact.get("evidence", {})
        orig_sent = (orig_ev.get("original_sentence") or "").strip()
        if len(orig_sent) > 10:
            ev_list.append({
                "source_type": "extraction_origin",
                "source_doc": orig_ev.get("source_document", "unknown"),
                "text": orig_sent,
                "reliability": 0.85,
            })

        # 2. BM25 retrieved passages
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

                    # Check deduplication against already added passages
                    is_dup = False
                    for existing in ev_list:
                        if jaccard_similarity(p_text, existing["text"]) > 0.65:
                            is_dup = True
                            break
                    if not is_dup:
                        rel = 0.90 if p_entry["source_type"] == "expanded_nlp_corpus" else 0.80
                        if score < 2.5:
                            rel *= 0.90
                        ev_list.append({
                            "source_type": p_entry["source_type"],
                            "source_doc": p_entry["source_file"],
                            "text": p_text,
                            "reliability": round(rel, 4),
                            "bm25_score": round(score, 3),
                        })
                    if len(ev_list) >= 4:  # At most 4 distinct passages per fact
                        break

        # If no evidence at all, fall back to global BM25
        if not ev_list and q_toks:
            g_bm25, g_passages = bm25_indices["__GLOBAL__"]
            scores = g_bm25.get_scores(q_toks)
            top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:2]
            for idx in top_indices:
                if scores[idx] > 1.5:
                    p_entry = g_passages[idx]
                    ev_list.append({
                        "source_type": "global_bm25_fallback",
                        "source_doc": p_entry["source_file"],
                        "text": p_entry["text"],
                        "reliability": 0.75,
                        "bm25_score": round(float(scores[idx]), 3),
                    })

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

    LOGGER.info(f"Generated {len(all_tasks):,} NLI inference pairs across {len(facts_dict):,} facts.")
    avg_ev = len(all_tasks) / len(facts_dict)
    LOGGER.info(f"Average evidence passages per fact: {avg_ev:.2f}")

    with open(OUTPUT_PAIRS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_tasks, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Saved NLI inference tasks to {OUTPUT_PAIRS_PATH}")


if __name__ == "__main__":
    prepare_pairs()
