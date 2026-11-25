#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/build_horn_subgraphs_cache.py
=============================================================================
Builds Multimodal Feature Cache & Landmark Subgraphs for the Horn-Complete Graph.

1. SigLIP2 Text Embeddings:
   Encodes all 16,347 Horn-complete normalized facts -> fact_embeddings_siglip2_horn.npy
2. Horn-Complete Subgraphs for 35 Landmarks:
   Constructs topologically pruned, relation-typed subgraphs for each landmark:
   - Type 0: Self-loop
   - Type 1: Visual-Visual (spatial / component)
   - Type 2: Visual-to-Semantic Bridge
   - Type 3: Semantic-to-Semantic Context
   Saves to: landmark_subgraphs_horn_complete.pt
=============================================================================
"""

import os
import sys
import json
import logging
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.fact_selection_benchmark.dataset_loader import get_fact_type

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("BuildHornCache")

CACHE_DIR = REPO_ROOT / "experiments" / "features" / "benchmark_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FACTS_HORN_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "normalized_facts_horn_complete.jsonl"
POOL_HORN_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "landmark_candidates_pool_horn_complete.json"

DEFAULT_SIGLIP2_MODEL = "google/siglip2-base-patch16-224"
COMMON_CONTAINERS = {
    "ha noi", "hanoi", "ho chi minh", "ho chi minh city", "hcmc",
    "tp ho chi minh", "vietnam", "viet nam", "vietnamese"
}


def extract_horn_fact_embeddings(device: torch.device):
    emb_path = CACHE_DIR / "fact_embeddings_siglip2_horn.npy"
    manifest_path = CACHE_DIR / "fact_embeddings_manifest_horn.json"

    if emb_path.exists() and manifest_path.exists():
        LOGGER.info(f"Horn fact embeddings already exist at {emb_path}. Skipping.")
        return

    LOGGER.info(f"Loading {FACTS_HORN_PATH}...")
    fact_ids = []
    fact_texts = []
    with open(FACTS_HORN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                fact_ids.append(item["fact_id"])
                fact_texts.append(item.get("fact_text", ""))

    LOGGER.info(f"Loaded {len(fact_ids)} facts. Loading SigLIP2 on {device}...")
    processor = AutoProcessor.from_pretrained(DEFAULT_SIGLIP2_MODEL)
    model = AutoModel.from_pretrained(DEFAULT_SIGLIP2_MODEL, torch_dtype=torch.float16).to(device).eval()

    all_embs = []
    batch_size = 128
    with torch.no_grad():
        for i in tqdm(range(0, len(fact_texts), batch_size), desc="Extracting text embeddings"):
            batch_texts = fact_texts[i : i + batch_size]
            inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
            out = model.get_text_features(**inputs)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                t_emb = out.pooler_output
            else:
                t_emb = out[0] if isinstance(out, (list, tuple)) else out
            t_emb = F.normalize(t_emb.float(), p=2, dim=-1).cpu().numpy().astype(np.float32)
            all_embs.append(t_emb)

    arr = np.vstack(all_embs)
    np.save(emb_path, arr)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(fact_ids, f)

    LOGGER.info(f"Saved {len(fact_ids)} Horn fact embeddings -> {emb_path} ({arr.shape})")


def build_horn_subgraphs():
    subgraphs_path = CACHE_DIR / "landmark_subgraphs_horn_complete.pt"
    if subgraphs_path.exists():
        LOGGER.info(f"Subgraphs already exist at {subgraphs_path}. Skipping.")
        return

    LOGGER.info("Loading Horn normalized facts...")
    facts = {}
    with open(FACTS_HORN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                facts[r["fact_id"]] = r

    with open(POOL_HORN_PATH, "r", encoding="utf-8") as f:
        pool = json.load(f)

    LOGGER.info(f"Building relation subgraphs for {len(pool)} landmarks...")
    rel_subgraphs = {}

    for l_id, fids in pool.items():
        N = len(fids)
        fid2idx = {fid: i for i, fid in enumerate(fids)}
        l_tokens = set(l_id.replace("_", " ").split())

        # Map entities to facts within this landmark's candidate pool
        entity_to_facts = defaultdict(set)
        for i, fid in enumerate(fids):
            f = facts[fid]
            s = str(f.get("subject", "")).lower().strip()
            o = str(f.get("object", "")).lower().strip()
            if s: entity_to_facts[s].add(i)
            if o: entity_to_facts[o].add(i)

        # Adjacency matrix initialization (self-loops)
        adj = np.eye(N, dtype=np.float32)

        for ent, idxs in entity_to_facts.items():
            ent_tokens = set(ent.split())
            is_container = ent in COMMON_CONTAINERS
            is_landmark_root = (len(ent_tokens.intersection(l_tokens)) >= min(2, len(l_tokens))) or (ent == l_id.replace("_", " "))
            is_frequency_hub = len(idxs) > max(3, int(0.15 * N))

            # Prune generic structural hubs to maintain localized semantics
            if not (is_container or is_landmark_root or is_frequency_hub):
                if len(idxs) > 1:
                    for u in idxs:
                        for v in idxs:
                            if u != v:
                                adj[u, v] = 1.0

        rows, cols = np.where(adj > 0)
        edge_index = np.vstack([rows, cols])
        num_edges = int(adj.sum() - N)
        avg_deg = num_edges / N if N > 0 else 0

        # Shortest path distance matrix (clipped at 10)
        dist_matrix = np.full((N, N), 512, dtype=np.int64)
        for i in range(N):
            dist_matrix[i, i] = 0
        for r, c in zip(rows, cols):
            dist_matrix[r, c] = 1

        for k in range(N):
            dist_matrix = np.minimum(dist_matrix, dist_matrix[:, [k]] + dist_matrix[[k], :])
        dist_matrix = np.clip(dist_matrix, 0, 10)
        degrees = np.sum(adj, axis=1).astype(np.int64)

        # Relation types assignment:
        # Type 0: Self-loop
        # Type 1: Visual-Visual
        # Type 2: Visual-to-Semantic Bridge
        # Type 3: Semantic-to-Semantic Context
        E = edge_index.shape[1]
        edge_types = torch.zeros(E, dtype=torch.long)
        rel_adj = torch.zeros((N, N), dtype=torch.long)

        for k in range(E):
            u = int(edge_index[0, k])
            v = int(edge_index[1, k])
            t_u = get_fact_type(facts[fids[u]].get("predicate", ""))
            t_v = get_fact_type(facts[fids[v]].get("predicate", ""))

            if u == v:
                r_type = 0
            elif t_u == "visual" and t_v == "visual":
                r_type = 1
            elif (t_u == "visual" and t_v != "visual") or (t_v == "visual" and t_u != "visual"):
                r_type = 2
            else:
                r_type = 3

            edge_types[k] = r_type
            rel_adj[u, v] = r_type

        rel_subgraphs[l_id] = {
            "landmark_id": l_id,
            "fact_ids": fids,
            "fid2idx": fid2idx,
            "adj": torch.from_numpy(adj).float(),
            "edge_index": torch.from_numpy(edge_index).long(),
            "dist_matrix": torch.from_numpy(dist_matrix).long(),
            "degrees": torch.from_numpy(degrees).long(),
            "avg_degree": float(avg_deg),
            "num_nodes": N,
            "num_edges": num_edges,
            "edge_types": edge_types,
            "rel_adj": rel_adj,
        }
        LOGGER.info(f"Landmark {l_id:32s} | Nodes: {N:4d} | Edges: {num_edges:5d} | AvgDeg: {avg_deg:4.1f}")

    torch.save(rel_subgraphs, subgraphs_path)
    LOGGER.info(f"Saved {len(rel_subgraphs)} Horn complete subgraphs -> {subgraphs_path} ({os.path.getsize(subgraphs_path)/(1024*1024):.1f} MB)")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using computing device: {device}")
    extract_horn_fact_embeddings(device)
    build_horn_subgraphs()


if __name__ == "__main__":
    main()
