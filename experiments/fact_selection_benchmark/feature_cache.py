#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/feature_cache.py
=============================================================================
Unified Multimodal Feature Extractor & Subgraph Cache for Fact Selection Benchmark.

Features Extracted & Cached:
  1. Video Representations (All 1,651 Clips: 800 Test, 851 Train+Val):
     - Frame-level embeddings [K=8, D=768] (for Cross-Attention Ranker)
     - Temporal mean-pooled video embedding [D=768] (for Dual Encoders, GATv2, Graphormer, VC-GAT)
  2. Fact Text Representations (All 3,234 Normalized Facts):
     - Text claim embeddings [D=768] via SigLIP2 text encoder
  3. Landmark Subgraph Topologies:
     - Sparse Adjacency Matrices A (shared entity & semantic relations)
     - Shortest Path Distance Matrices D (for Graphormer spatial encodings)
     - Node Degree Encodings (for Graphormer centrality encodings)
=============================================================================
"""

import os
import sys
import json
import logging
import cv2
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = REPO_ROOT / "experiments" / "features" / "benchmark_cache"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("FeatureCache")

DEFAULT_SIGLIP2_MODEL = "google/siglip2-base-patch16-224"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def sample_video_frames(video_path: str, num_frames: int = 8, target_size: int = 224) -> Optional[np.ndarray]:
    """
    Extracts num_frames uniformly spaced temporal frames from video.
    Returns RGB numpy array of shape [num_frames, target_size, target_size, 3] uint8.
    """
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            # Replicate last frame or black frame
            if frames:
                frame = frames[-1].copy()
            else:
                frame = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((target_size, target_size, 3), dtype=np.uint8))

    return np.array(frames, dtype=np.uint8)


def extract_tensor(out) -> torch.Tensor:
    """Extracts raw tensor from HF ModelOutput object."""
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state[:, 0, :]
    return out[0]


class BenchmarkFeatureCache:
    """Manages precomputed multimodal video features, text embeddings, and graph topologies."""
    
    def __init__(self, cache_dir: Path = FEATURES_DIR, device: Optional[torch.device] = None):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or get_device()
        self.model = None
        self.processor = None
        
        # In-memory caches
        self.video_pooled: Dict[str, np.ndarray] = {}       # clip_id -> [768]
        self.video_frames: Dict[str, np.ndarray] = {}       # clip_id -> [8, 768]
        self.fact_embeddings: Dict[str, np.ndarray] = {}    # fact_id -> [768]
        self.subgraphs: Dict[str, Dict[str, Any]] = {}      # landmark_id -> graph data

    def init_model(self, model_name: str = DEFAULT_SIGLIP2_MODEL):
        if self.model is None:
            LOGGER.info(f"Loading feature extractor {model_name} onto {self.device}...")
            self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(model_name)
            LOGGER.info("Feature extractor loaded successfully.")

    def extract_fact_embeddings(self, facts: Dict[str, Dict[str, Any]], force_recompute: bool = False) -> Dict[str, np.ndarray]:
        """Encodes all fact claims using SigLIP2 text encoder."""
        cache_path = self.cache_dir / "fact_embeddings_siglip2.npy"
        manifest_path = self.cache_dir / "fact_embeddings_manifest.json"

        if not force_recompute and cache_path.exists() and manifest_path.exists():
            LOGGER.info(f"Loading cached fact embeddings from {cache_path}...")
            arr = np.load(cache_path)
            with open(manifest_path, "r", encoding="utf-8") as f:
                fids = json.load(f)
            self.fact_embeddings = {fid: arr[i] for i, fid in enumerate(fids)}
            return self.fact_embeddings

        self.init_model()
        LOGGER.info(f"Extracting embeddings for {len(facts)} candidate facts...")
        fact_ids = list(facts.keys())
        fact_texts = [facts[fid].get("fact_text", "") for fid in fact_ids]

        all_embs = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(fact_texts), batch_size):
                b_texts = fact_texts[i : i + batch_size]
                inputs = self.processor(text=b_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
                out = self.model.get_text_features(**inputs)
                t_emb = extract_tensor(out)
                t_emb = F.normalize(t_emb, p=2, dim=-1).cpu().numpy()
                all_embs.append(t_emb)

        arr = np.vstack(all_embs)
        np.save(cache_path, arr)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(fact_ids, f)

        self.fact_embeddings = {fid: arr[i] for i, fid in enumerate(fact_ids)}
        LOGGER.info(f"Saved {len(self.fact_embeddings)} fact embeddings -> {cache_path}")
        return self.fact_embeddings

    def extract_video_features(
        self,
        clip_manifests: List[Dict[str, Any]],
        batch_size: int = 16,
        force_recompute: bool = False
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Extracts both frame-level [8, 768] and pooled [768] video features for clips.
        """
        pooled_path = self.cache_dir / "video_pooled_siglip2.npy"
        frames_path = self.cache_dir / "video_frames_siglip2.npy"
        manifest_path = self.cache_dir / "video_manifest.json"

        if not force_recompute and pooled_path.exists() and frames_path.exists() and manifest_path.exists():
            LOGGER.info(f"Loading cached video features from {pooled_path}...")
            arr_p = np.load(pooled_path)
            arr_f = np.load(frames_path)
            with open(manifest_path, "r", encoding="utf-8") as f:
                cids = json.load(f)
            self.video_pooled = {cid: arr_p[i] for i, cid in enumerate(cids)}
            self.video_frames = {cid: arr_f[i] for i, cid in enumerate(cids)}
            return self.video_pooled, self.video_frames

        # Deduplicate manifests by clip_id
        unique_manifests = {}
        for m in clip_manifests:
            if m["clip_id"] not in unique_manifests:
                unique_manifests[m["clip_id"]] = m
        unique_list = list(unique_manifests.values())

        self.init_model()
        LOGGER.info(f"Extracting features for {len(unique_list)} unique video clips...")
        clip_ids = []
        pooled_list = []
        frames_list = []

        for clip_meta in tqdm(unique_list, desc="Encoding videos"):
            cid = clip_meta["clip_id"]
            v_path = clip_meta["video_path"]
            frames_np = sample_video_frames(v_path, num_frames=8, target_size=224)
            if frames_np is None:
                # Fallback zero representation
                frame_embs = np.zeros((8, 768), dtype=np.float32)
                pooled_emb = np.zeros(768, dtype=np.float32)
            else:
                with torch.no_grad():
                    inputs = self.processor(images=list(frames_np), return_tensors="pt").to(self.device)
                    v_out = self.model.get_image_features(**inputs)
                    v_tensor = extract_tensor(v_out)
                    v_tensor = F.normalize(v_tensor, p=2, dim=-1) # [8, 768]
                    frame_embs = v_tensor.cpu().numpy().astype(np.float32)
                    pooled_t = F.normalize(v_tensor.mean(dim=0, keepdim=True), p=2, dim=-1)
                    pooled_emb = pooled_t.squeeze(0).cpu().numpy().astype(np.float32)

            clip_ids.append(cid)
            pooled_list.append(pooled_emb)
            frames_list.append(frame_embs)

        arr_p = np.array(pooled_list, dtype=np.float32)
        arr_f = np.array(frames_list, dtype=np.float32)

        np.save(pooled_path, arr_p)
        np.save(frames_path, arr_f)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(clip_ids, f)

        self.video_pooled = {cid: arr_p[i] for i, cid in enumerate(clip_ids)}
        self.video_frames = {cid: arr_f[i] for i, cid in enumerate(clip_ids)}
        LOGGER.info(f"Saved {len(clip_ids)} video features -> {pooled_path}")
        return self.video_pooled, self.video_frames

    def build_landmark_subgraphs(self, facts: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Builds semantic entity-overlap graph topology for each landmark.
        For each landmark L:
          - Nodes: facts belonging to landmark L
          - Adjacency A: A_ij = 1 if facts share subject, object, or inverse relation
          - Shortest path distance matrix D (for Graphormer)
          - Degree vector deg (for centrality encoding)
        """
        cache_path = self.cache_dir / "landmark_subgraphs.pt"
        if cache_path.exists():
            LOGGER.info(f"Loading landmark subgraphs from {cache_path}...")
            self.subgraphs = torch.load(cache_path, weights_only=False)
            return self.subgraphs

        LOGGER.info("Building landmark subgraphs topology...")
        by_landmark = defaultdict(list)
        for fid, f in facts.items():
            l_id = f.get("landmark_id", "unknown")
            by_landmark[l_id].append(fid)

        subgraphs = {}
        for l_id, fids in by_landmark.items():
            N = len(fids)
            fid2idx = {fid: i for i, fid in enumerate(fids)}
            
            # Adjacency matrix
            adj = np.eye(N, dtype=np.float32)
            edge_index_list = []
            
            # Map entities to fact indices
            entity_to_facts = defaultdict(set)
            for i, fid in enumerate(fids):
                f = facts[fid]
                s = str(f.get("subject", "")).lower().strip()
                o = str(f.get("object", "")).lower().strip()
                if s: entity_to_facts[s].add(i)
                if o: entity_to_facts[o].add(i)

            for ent, idxs in entity_to_facts.items():
                if len(idxs) > 1:
                    idx_list = list(idxs)
                    for u in idx_list:
                        for v in idx_list:
                            if u != v:
                                adj[u, v] = 1.0

            # Compute edge_index for PyG / GATv2
            rows, cols = np.where(adj > 0)
            edge_index = np.vstack([rows, cols])

            # Compute shortest path distance matrix via Floyd-Warshall / BFS
            dist_matrix = np.full((N, N), 512, dtype=np.int64)
            for i in range(N):
                dist_matrix[i, i] = 0
            for r, c in zip(rows, cols):
                dist_matrix[r, c] = 1

            for k in range(N):
                dist_matrix = np.minimum(dist_matrix, dist_matrix[:, [k]] + dist_matrix[[k], :])
            # Cap max distance at 10 for graph transformer spatial encoding
            dist_matrix = np.clip(dist_matrix, 0, 10)

            # Node degrees
            degrees = np.sum(adj, axis=1).astype(np.int64)

            subgraphs[l_id] = {
                "fact_ids": fids,
                "fid2idx": fid2idx,
                "num_nodes": N,
                "adj": torch.from_numpy(adj).float(),
                "edge_index": torch.from_numpy(edge_index).long(),
                "shortest_path": torch.from_numpy(dist_matrix).long(),
                "degrees": torch.from_numpy(degrees).long()
            }

        torch.save(subgraphs, cache_path)
        self.subgraphs = subgraphs
        LOGGER.info(f"Saved {len(subgraphs)} landmark subgraphs -> {cache_path}")
        return self.subgraphs
