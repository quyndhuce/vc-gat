"""
FTransE: Fuzzy Knowledge Graph Embedding & Deductive Subgraph Reasoning Engine.
Implements Weighted TransE with Fuzzy Membership Weights (μ_F, Bel) from the FKG.

Key Capabilities:
  1. Train KGE with Fuzzy Loss Weighting on GPU.
  2. Deductive Link Prediction (Suy luận tri thức còn thiếu h + r ≈ t).
  3. Semantic Similarity / Entity Clustering (KNN in latent embedding space).
  4. Embedding-guided Subgraph Pruning & Extraction for GraphRAG.
"""

import json
import logging
import math
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("FTransE")


# ══════════════════════════════════════════════════════════
# 1. DATASET & VOCABULARY
# ══════════════════════════════════════════════════════════

class FKGDataset(Dataset):
    """Dataset of (head_id, relation_id, tail_id, fuzzy_weight)."""

    def __init__(
        self,
        triples: List[Tuple[int, int, int, float]],
        num_entities: int,
        num_neg: int = 5,
    ):
        self.triples = triples
        self.num_entities = num_entities
        self.num_neg = num_neg

        # Precompute existing edges for filtered negative sampling
        self.existing_edges: Set[Tuple[int, int, int]] = {
            (h, r, t) for h, r, t, _ in triples
        }

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int):
        h, r, t, w = self.triples[idx]

        # Generate negative samples
        neg_tails = []
        for _ in range(self.num_neg):
            while True:
                corrupt_t = random.randint(0, self.num_entities - 1)
                if (h, r, corrupt_t) not in self.existing_edges or corrupt_t == t:
                    neg_tails.append(corrupt_t)
                    break

        return (
            torch.tensor(h, dtype=torch.long),
            torch.tensor(r, dtype=torch.long),
            torch.tensor(t, dtype=torch.long),
            torch.tensor(w, dtype=torch.float),
            torch.tensor(neg_tails, dtype=torch.long),
        )


# ══════════════════════════════════════════════════════════
# 2. FTRANSE MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════

class FTransE(nn.Module):
    """Fuzzy TransE with Softplus / Margin BCE Loss weighted by μ_F."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 128,
        margin: float = 6.0,
        p_norm: int = 2,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.p_norm = p_norm

        self.ent_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.rel_embeddings = nn.Embedding(num_relations, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        # Normalize initial relation embeddings
        self.rel_embeddings.weight.data = F.normalize(
            self.rel_embeddings.weight.data, p=2, dim=-1
        )

    def normalize_entities(self):
        """Constrain entity embeddings to unit sphere ||e||_2 <= 1."""
        with torch.no_grad():
            self.ent_embeddings.weight.data = F.normalize(
                self.ent_embeddings.weight.data, p=2, dim=-1
            )

    def distance(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Translational distance: ||h + r - t||_p"""
        diff = h + r - t
        if self.p_norm == 1:
            return torch.sum(torch.abs(diff), dim=-1)
        else:
            return torch.norm(diff, p=2, dim=-1)

    def forward(
        self,
        h_idx: torch.Tensor,
        r_idx: torch.Tensor,
        t_idx: torch.Tensor,
        weights: torch.Tensor,
        neg_t_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Fuzzy-weighted translation loss.

        Loss = weight * [ log(1 + exp(d_pos - margin)) + sum(log(1 + exp(margin - d_neg))) / k ]
        """
        h = self.ent_embeddings(h_idx)          # [B, D]
        r = self.rel_embeddings(r_idx)          # [B, D]
        t = self.ent_embeddings(t_idx)          # [B, D]

        # Positive distance
        pos_dist = self.distance(h, r, t)        # [B]

        # Negative distance
        B, K = neg_t_idx.shape
        h_exp = h.unsqueeze(1).expand(-1, K, -1) # [B, K, D]
        r_exp = r.unsqueeze(1).expand(-1, K, -1) # [B, K, D]
        neg_t = self.ent_embeddings(neg_t_idx)   # [B, K, D]
        neg_dist = self.distance(h_exp, r_exp, neg_t) # [B, K]

        # Softplus loss with margin
        pos_loss = F.softplus(pos_dist - self.margin)
        neg_loss = F.softplus(self.margin - neg_dist).mean(dim=-1)

        # Weighted by fuzzy membership μ_F
        loss = (weights * (pos_loss + neg_loss)).mean()
        return loss

    @torch.no_grad()
    def predict_plausibility(
        self, h_idx: torch.Tensor, r_idx: torch.Tensor, t_idx: torch.Tensor
    ) -> torch.Tensor:
        """Compute plausibility score in range (0, 1): sigmoid(margin - distance)."""
        h = self.ent_embeddings(h_idx)
        r = self.rel_embeddings(r_idx)
        t = self.ent_embeddings(t_idx)
        dist = self.distance(h, r, t)
        return torch.sigmoid(self.margin - dist)


# ══════════════════════════════════════════════════════════
# 3. KGE PIPELINE & REASONING ENGINE
# ══════════════════════════════════════════════════════════

class FTransEEngine:
    """End-to-end KGE trainer, link predictor & deductive subgraph extractor."""

    def __init__(
        self,
        embedding_dim: int = 128,
        margin: float = 6.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.device = torch.device(device)

        self.entity_to_id: Dict[str, int] = {}
        self.id_to_entity: Dict[int, str] = {}
        self.relation_to_id: Dict[str, int] = {}
        self.id_to_relation: Dict[int, str] = {}
        self.surface_map: Dict[str, str] = {}

        self.model: Optional[FTransE] = None
        self.training_triples: List[Tuple[int, int, int, float]] = []
        self.existing_facts: Set[Tuple[str, str, str]] = set()

    def build_vocab_and_triples(
        self,
        json_path: Path = Path("data/fkg_per_kb/fkg_global_merged.json"),
        registry_path: Path = Path("data/fkg_per_kb/shared_node_registry.json"),
    ):
        LOGGER.info("Loading FKG data from %s...", json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        # Build surface map
        uri_info = registry.get("uri_info", {})
        for uri, info in uri_info.items():
            self.surface_map[uri] = info.get("canonical_surface", uri)

        raw_triples: List[Tuple[str, str, str, float]] = []

        # 1. Single-valued winners
        for g in data.get("single_valued_groups", []):
            h_uri = g.get("head_uri")
            rel = g.get("relation")
            w_uri = g.get("winner_uri")
            bel = float(g.get("winner_bel", 0.8))
            if h_uri and rel and w_uri:
                raw_triples.append((h_uri, rel, w_uri, max(0.2, min(1.0, bel))))
                self.existing_facts.add((h_uri, rel, w_uri))
                if g.get("head_surface"):
                    self.surface_map[h_uri] = g["head_surface"]
                if g.get("winner_surface"):
                    self.surface_map[w_uri] = g["winner_surface"]

        # 2. Multi-valued facts
        for m in data.get("multi_valued_facts", []):
            h_uri = m.get("head_uri")
            rel = m.get("relation")
            t_uri = m.get("tail_uri")
            mu = float(m.get("mu_F", 0.5))
            if h_uri and rel and t_uri:
                raw_triples.append((h_uri, rel, t_uri, max(0.2, min(1.0, mu))))
                self.existing_facts.add((h_uri, rel, t_uri))
                if m.get("head_surface"):
                    self.surface_map[h_uri] = m["head_surface"]
                if m.get("tail_surface"):
                    self.surface_map[t_uri] = m["tail_surface"]

        LOGGER.info("Loaded %d raw triples with fuzzy weights.", len(raw_triples))

        # Build Vocabularies
        entities = sorted(list(set(h for h, _, _, _ in raw_triples) | set(t for _, _, t, _ in raw_triples)))
        relations = sorted(list(set(r for _, r, _, _ in raw_triples)))

        self.entity_to_id = {e: i for i, e in enumerate(entities)}
        self.id_to_entity = {i: e for i, e in enumerate(entities)}
        self.relation_to_id = {r: i for i, r in enumerate(relations)}
        self.id_to_relation = {i: r for i, r in enumerate(relations)}

        LOGGER.info(
            "Vocab built: %d unique entities, %d unique relations",
            len(self.entity_to_id), len(self.relation_to_id)
        )

        # Convert to ID tuples
        self.training_triples = [
            (self.entity_to_id[h], self.relation_to_id[r], self.entity_to_id[t], w)
            for h, r, t, w in raw_triples
        ]

    def train(
        self,
        epochs: int = 250,
        batch_size: int = 512,
        lr: float = 0.003,
        num_neg: int = 5,
    ):
        """Train FTransE on GPU with Fuzzy-weighted softplus loss."""
        if not self.training_triples:
            raise ValueError("No training triples loaded!")

        self.model = FTransE(
            num_entities=len(self.entity_to_id),
            num_relations=len(self.relation_to_id),
            embedding_dim=self.embedding_dim,
            margin=self.margin,
        ).to(self.device)

        dataset = FKGDataset(
            triples=self.training_triples,
            num_entities=len(self.entity_to_id),
            num_neg=num_neg,
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        LOGGER.info(
            "Training FTransE: %d epochs | batch=%d | lr=%.4f | device=%s",
            epochs, batch_size, lr, self.device
        )

        t0 = time.time()
        self.model.train()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for h, r, t, w, neg_t in dataloader:
                h = h.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)
                w = w.to(self.device)
                neg_t = neg_t.to(self.device)

                optimizer.zero_grad()
                loss = self.model(h, r, t, w, neg_t)
                loss.backward()
                optimizer.step()

                # Project entity embeddings back to unit sphere
                self.model.normalize_entities()
                total_loss += loss.item() * len(h)

            scheduler.step()
            avg_loss = total_loss / len(self.training_triples)

            if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
                LOGGER.info(
                    "  Epoch [%3d/%3d] — Fuzzy Loss: %.5f (lr=%.5f)",
                    epoch, epochs, avg_loss, scheduler.get_last_lr()[0]
                )

        LOGGER.info("✓ FTransE Training finished in %.2fs", time.time() - t0)

    def save(self, output_dir: Path = Path("data/fkg_per_kb")):
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output_dir / "ftranse_model.pt")
        vocab = {
            "entity_to_id": self.entity_to_id,
            "relation_to_id": self.relation_to_id,
            "surface_map": self.surface_map,
            "embedding_dim": self.embedding_dim,
            "margin": self.margin,
        }
        with open(output_dir / "ftranse_vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        LOGGER.info("Saved FTransE model & vocab to %s", output_dir)

    def load(self, model_dir: Path = Path("data/fkg_per_kb")):
        with open(model_dir / "ftranse_vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.entity_to_id = vocab["entity_to_id"]
        self.id_to_entity = {int(v): k for k, v in self.entity_to_id.items()}
        self.relation_to_id = vocab["relation_to_id"]
        self.id_to_relation = {int(v): k for k, v in self.relation_to_id.items()}
        self.surface_map = vocab["surface_map"]
        self.embedding_dim = vocab.get("embedding_dim", 128)
        self.margin = vocab.get("margin", 6.0)

        self.model = FTransE(
            num_entities=len(self.entity_to_id),
            num_relations=len(self.relation_to_id),
            embedding_dim=self.embedding_dim,
            margin=self.margin,
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_dir / "ftranse_model.pt", map_location=self.device))
        self.model.eval()

    # ══════════════════════════════════════════════════════
    # 4. INFERENCE: LINK PREDICTION & DEDUCTIVE REASONING
    # ══════════════════════════════════════════════════════

    @torch.no_grad()
    def infer_missing_facts(
        self,
        landmark_uri: str,
        relations_to_test: Optional[List[str]] = None,
        top_k: int = 5,
        min_plausibility: float = 0.50,
    ) -> List[Dict[str, Any]]:
        """Deduce missing facts for a landmark via translation reasoning: h + r ≈ t.

        Filters out facts that already exist in the ground truth graph.
        """
        if landmark_uri not in self.entity_to_id:
            return []

        self.model.eval()
        h_idx = self.entity_to_id[landmark_uri]
        h_vec = self.model.ent_embeddings.weight[h_idx]  # [D]

        if not relations_to_test:
            # High-value ontological relations to deduce
            relations_to_test = [
                "belongs_to_dynasty", "during_era", "architectural_style",
                "located_in", "near", "worships", "associated_with",
                "part_of", "has_part", "symbol_of"
            ]

        inferred = []
        all_ent_vecs = self.model.ent_embeddings.weight  # [N, D]

        for rel_name in relations_to_test:
            if rel_name not in self.relation_to_id:
                continue

            r_idx = self.relation_to_id[rel_name]
            r_vec = self.model.rel_embeddings.weight[r_idx]  # [D]

            # Translation vector target: target = h + r
            target_vec = h_vec + r_vec  # [D]

            # Distances to all entities
            dists = torch.norm(all_ent_vecs - target_vec.unsqueeze(0), p=2, dim=-1)  # [N]
            plausibility = torch.sigmoid(self.margin - dists)  # [N]

            # Sort descending
            scores, top_indices = torch.topk(plausibility, k=min(top_k + 20, len(all_ent_vecs)))

            for score, cand_idx in zip(scores.tolist(), top_indices.tolist()):
                cand_uri = self.id_to_entity[cand_idx]
                if cand_uri == landmark_uri:
                    continue

                # Check if this fact ALREADY existed in the training graph
                is_known = (landmark_uri, rel_name, cand_uri) in self.existing_facts
                if is_known:
                    continue  # We want NEW DEDUCED FACTS!

                if score < min_plausibility:
                    continue

                inferred.append({
                    "head_uri": landmark_uri,
                    "head_surface": self.surface_map.get(landmark_uri, landmark_uri),
                    "relation": rel_name,
                    "tail_uri": cand_uri,
                    "tail_surface": self.surface_map.get(cand_uri, cand_uri),
                    "plausibility": round(score, 4),
                    "deduction_type": "inferred_novel_fact",
                })

                if len([x for x in inferred if x["relation"] == rel_name]) >= top_k:
                    break

        return inferred

    @torch.no_grad()
    def find_nearest_entities(
        self,
        landmark_uri: str,
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """Find semantically closest entities in latent embedding space via cosine similarity."""
        if landmark_uri not in self.entity_to_id:
            return []

        self.model.eval()
        h_idx = self.entity_to_id[landmark_uri]
        h_vec = self.model.ent_embeddings.weight[h_idx].unsqueeze(0)  # [1, D]

        all_vecs = self.model.ent_embeddings.weight  # [N, D]
        cos_sim = F.cosine_similarity(h_vec, all_vecs, dim=-1)  # [N]

        scores, indices = torch.topk(cos_sim, k=top_k + 1)
        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            uri = self.id_to_entity[idx]
            if uri == landmark_uri:
                continue
            results.append({
                "uri": uri,
                "surface": self.surface_map.get(uri, uri),
                "cosine_sim": round(score, 4),
            })
            if len(results) >= top_k:
                break

        return results

    @torch.no_grad()
    def extract_scored_subgraph(
        self,
        landmark_uri: str,
        max_edges: int = 15,
    ) -> Dict[str, Any]:
        """Extract and score 1-hop and 2-hop subgraph edges using FTransE energy scores."""
        if landmark_uri not in self.entity_to_id:
            return {}

        self.model.eval()
        h_idx = self.entity_to_id[landmark_uri]

        # Collect 1-hop edges
        candidate_edges = []
        for (h, r, t) in self.existing_facts:
            if h == landmark_uri:
                h_i = self.entity_to_id[h]
                r_i = self.relation_to_id[r]
                t_i = self.entity_to_id[t]
                pl = self.model.predict_plausibility(
                    torch.tensor([h_i], device=self.device),
                    torch.tensor([r_i], device=self.device),
                    torch.tensor([t_i], device=self.device),
                ).item()
                candidate_edges.append({
                    "head": self.surface_map.get(h, h),
                    "relation": r,
                    "tail": self.surface_map.get(t, t),
                    "score": round(pl, 4),
                    "direction": "outgoing",
                })
            elif t == landmark_uri:
                h_i = self.entity_to_id[h]
                r_i = self.relation_to_id[r]
                t_i = self.entity_to_id[t]
                pl = self.model.predict_plausibility(
                    torch.tensor([h_i], device=self.device),
                    torch.tensor([r_i], device=self.device),
                    torch.tensor([t_i], device=self.device),
                ).item()
                candidate_edges.append({
                    "head": self.surface_map.get(h, h),
                    "relation": r,
                    "tail": self.surface_map.get(t, t),
                    "score": round(pl, 4),
                    "direction": "incoming",
                })

        candidate_edges.sort(key=lambda x: x["score"], reverse=True)
        return {
            "landmark": self.surface_map.get(landmark_uri, landmark_uri),
            "subgraph_edges": candidate_edges[:max_edges],
        }


# ══════════════════════════════════════════════════════════
# 5. CLI TEST RUNNER
# ══════════════════════════════════════════════════════════

def main():
    engine = FTransEEngine(embedding_dim=128, margin=6.0)
    engine.build_vocab_and_triples()
    engine.train(epochs=250, batch_size=512, lr=0.005)
    engine.save()

    print("\n" + "=" * 80)
    print("DEMO: KNOWLEDGE GRAPH EMBEDDING & DEDUCTIVE REASONING (FTransE)")
    print("=" * 80)

    # Resolve target landmarks
    test_landmarks = [
        ("http://transmtl.vn/kb/one_pillar_pagoda", "One Pillar Pagoda"),
        ("http://transmtl.vn/kb/temple_of_literature", "Temple of Literature"),
        ("http://transmtl.vn/kb/tran_quoc_pagoda", "Tran Quoc Pagoda"),
    ]

    for uri, name in test_landmarks:
        print(f"\n{'─'*70}")
        print(f"🏛️ LANDMARK: {name}")
        print(f"{'─'*70}")

        # 1. Deductive Inferences (Link Prediction)
        inferred = engine.infer_missing_facts(uri, top_k=3, min_plausibility=0.55)
        print("\n  🔍 [TRI THỨC MỚI ĐƯỢC SUY LUẬN (INFERRED / NOVEL FACTS)]:")
        if inferred:
            for inf in inferred:
                print(f"    ⭐ ({inf['head_surface']}) ──[{inf['relation']}]──► {inf['tail_surface']}  (Độ tin cậy KGE: {inf['plausibility']:.4f})")
        else:
            print("    (Không có quan hệ suy luận đạt ngưỡng tin cậy)")

        # 2. Semantic Cluster / Nearest Neighbors in Vector Space
        neighbors = engine.find_nearest_entities(uri, top_k=5)
        print("\n  🌐 [THỰC THỂ TƯƠNG ĐỒNG NHẤT TRONG KHÔNG GIAN VECTOR (KNN)]:")
        for n in neighbors:
            print(f"    • {n['surface']} (Cosine Similarity: {n['cosine_sim']:.4f})")

        # 3. Embedding-Pruned Subgraph
        subgraph = engine.extract_scored_subgraph(uri, max_edges=6)
        print("\n  🎯 [SUBGRAPH TINH TÚY ĐÃ ĐƯỢC ĐÁNH GIÁ ĐỘ TƯƠNG THÍCH BỞI KGE]:")
        for e in subgraph.get("subgraph_edges", []):
            print(f"    • {e['head']} ──[{e['relation']}]──► {e['tail']}  (Score: {e['score']:.4f})")


if __name__ == "__main__":
    main()
