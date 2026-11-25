#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_bm25_corpus.py
=============================================================================
Stage 1: Document Chunking and BM25 Passage Index Construction.

Processes documentary evidence across all 35 landmarks:
  1. data/sources/VCGAT_35_Landmarks_NLP_Factual_Expanded/*.txt
  2. data/fkg_per_kb/<kb_id>/docs/*_clean.txt

Produces:
  - corpus_passages.json (Structured chunked passages with metadata)
  - bm25_indices.pkl (Pre-built BM25Okapi indices partitioned by landmark)
=============================================================================
"""

import json
import logging
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("BuildBM25Corpus")

REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")
EXP_NLP_DIR = REPO_ROOT / "data" / "sources" / "VCGAT_35_Landmarks_NLP_Factual_Expanded"
PER_KB_DIR = REPO_ROOT / "data" / "fkg_per_kb"
OUTPUT_DIR = REPO_ROOT / "experiments" / "fkg_certainty_redesign"


def clean_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


def split_into_passages(text: str, window_size: int = 2, step: int = 1) -> List[str]:
    """
    Splits text into overlapping passages of window_size sentences.
    Preserves context, conditions, and negation cues.
    """
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]
    if not sentences:
        return []
    if len(sentences) <= window_size:
        return [" ".join(sentences)]

    passages = []
    seen = set()
    for i in range(0, len(sentences), step):
        chunk = sentences[i : i + window_size]
        if not chunk:
            continue
        passage_str = " ".join(chunk).strip()
        if passage_str and passage_str not in seen:
            seen.add(passage_str)
            passages.append(passage_str)
        if i + window_size >= len(sentences):
            break
    return passages


def build_corpus_and_indices():
    LOGGER.info("Starting BM25 passage indexing for all 35 landmarks...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    passages_by_lm: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_passages: List[Dict[str, Any]] = []
    seen_hashes = set()

    passage_counter = 0

    # 1. Ingest VCGAT_35_Landmarks_NLP_Factual_Expanded
    if EXP_NLP_DIR.exists():
        for txt_file in sorted(EXP_NLP_DIR.glob("*.txt")):
            lm_id = txt_file.stem[3:] if txt_file.stem[:2].isdigit() and txt_file.stem[2] == "_" else txt_file.stem
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            chunks = split_into_passages(content, window_size=2, step=1)
            for chunk in chunks:
                chash = f"{lm_id}|{chunk.lower()}"
                if chash not in seen_hashes:
                    seen_hashes.add(chash)
                    p_entry = {
                        "passage_id": f"pass_{passage_counter:06d}",
                        "landmark_id": lm_id,
                        "source_file": txt_file.name,
                        "source_type": "expanded_nlp_corpus",
                        "text": chunk,
                        "tokens": clean_tokenize(chunk),
                    }
                    passages_by_lm[lm_id].append(p_entry)
                    all_passages.append(p_entry)
                    passage_counter += 1

    LOGGER.info(f"Ingested {len(all_passages)} passages from Expanded NLP Corpus.")

    # 2. Ingest per-KB doc clean files
    if PER_KB_DIR.exists():
        per_kb_count = 0
        for kb_dir in sorted(PER_KB_DIR.iterdir()):
            if not kb_dir.is_dir():
                continue
            lm_id = kb_dir.name
            docs_dir = kb_dir / "docs"
            if not docs_dir.exists():
                continue
            for doc_file in sorted(docs_dir.glob("*_clean.txt")):
                with open(doc_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                chunks = split_into_passages(content, window_size=2, step=1)
                for chunk in chunks:
                    chash = f"{lm_id}|{chunk.lower()}"
                    if chash not in seen_hashes:
                        seen_hashes.add(chash)
                        p_entry = {
                            "passage_id": f"pass_{passage_counter:06d}",
                            "landmark_id": lm_id,
                            "source_file": f"{lm_id}/docs/{doc_file.name}",
                            "source_type": "per_kb_clean_doc",
                            "text": chunk,
                            "tokens": clean_tokenize(chunk),
                        }
                        passages_by_lm[lm_id].append(p_entry)
                        all_passages.append(p_entry)
                        passage_counter += 1
                        per_kb_count += 1
        LOGGER.info(f"Ingested {per_kb_count} additional passages from per-KB clean docs.")

    LOGGER.info(f"Total indexed passages: {len(all_passages):,} across {len(passages_by_lm)} landmarks.")

    # 3. Build BM25 Okapi indices
    LOGGER.info("Building BM25Okapi index structures...")
    bm25_indices: Dict[str, Tuple[BM25Okapi, List[Dict[str, Any]]]] = {}

    for lm_id, p_list in passages_by_lm.items():
        tokenized_corpus = [p["tokens"] for p in p_list]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_indices[lm_id] = (bm25, p_list)

    # Global BM25 index across all landmarks
    global_tokenized = [p["tokens"] for p in all_passages]
    global_bm25 = BM25Okapi(global_tokenized)
    bm25_indices["__GLOBAL__"] = (global_bm25, all_passages)

    # 4. Save to files
    passages_json_path = OUTPUT_DIR / "corpus_passages.json"
    with open(passages_json_path, "w", encoding="utf-8") as f:
        json.dump(all_passages, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Saved passages to {passages_json_path}")

    index_pkl_path = OUTPUT_DIR / "bm25_indices.pkl"
    with open(index_pkl_path, "wb") as f:
        pickle.dump(bm25_indices, f)
    LOGGER.info(f"Saved BM25 indices to {index_pkl_path}")

    return all_passages, bm25_indices


if __name__ == "__main__":
    build_corpus_and_indices()
