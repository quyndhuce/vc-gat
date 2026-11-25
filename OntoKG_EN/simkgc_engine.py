# -*- coding: utf-8 -*-
"""
OntoKG_EN/simkgc_engine.py
=============================================================================
Multimodal Fuzzy SimKGC: Contrastive Pre-trained Language Model KGE Engine
=============================================================================
Optimized Feature-Based Contrastive Learning:
- Pre-trained Language Model (all-MiniLM-L6-v2, 384-d)
- Contrastive Projection Heads (Linear -> LayerNorm -> GELU -> Linear)
- Fuzzy-Weighted InfoNCE Loss with In-Batch Negatives
- Fast Precomputed Embedding Inference with Type-Masking
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("SimKGC")


def clean_relation_name(r: str) -> str:
    return r.replace("_", " ").strip()


class ContrastiveProjectionNet(nn.Module):
    """Deep Non-Linear Projection Heads for Contrastive Alignment."""
    def __init__(self, in_dim: int = 384, proj_dim: int = 256, temperature: float = 0.05):
        super().__init__()
        self.hr_proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim, proj_dim)
        )
        self.t_proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim, proj_dim)
        )
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(temperature))))
        self.proj_dim = proj_dim

    @property
    def temperature(self):
        return torch.exp(self.log_tau).clamp(min=0.01, max=0.5)

    def forward(self, h_hr, h_t, weights):
        e_hr = F.normalize(self.hr_proj(h_hr), p=2, dim=-1)
        e_t = F.normalize(self.t_proj(h_t), p=2, dim=-1)

        sim_matrix = torch.matmul(e_hr, e_t.t()) / self.temperature
        batch_size = sim_matrix.size(0)
        labels = torch.arange(batch_size, device=sim_matrix.device)

        log_probs = F.log_softmax(sim_matrix, dim=-1)
        pos_log_probs = log_probs[torch.arange(batch_size), labels]

        loss = -(weights * pos_log_probs).sum() / torch.clamp(weights.sum(), min=1e-5)
        return loss, sim_matrix

class SimKGCEngine:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        proj_dim: int = 256,
        temperature: float = 0.05,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.hidden_dim = self.encoder.config.hidden_size
        self.proj_net = ContrastiveProjectionNet(in_dim=self.hidden_dim, proj_dim=proj_dim, temperature=temperature).to(self.device)
        self.proj_dim = proj_dim

        self.triples: List[Dict[str, Any]] = []
        self.entities: Dict[str, str] = {}
        self.relations: Set[str] = set()
        self.known_facts: Set[Tuple[str, str, str]] = set()

        self.all_entity_uris: List[str] = []
        self.cached_tail_embs: Optional[torch.Tensor] = None

    def _encode_texts(self, texts: List[str], batch_size: int = 128, max_len: int = 40) -> torch.Tensor:
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            b_txt = texts[i : i + batch_size]
            enc = self.tokenizer(b_txt, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.encoder(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).expand(out[0].size()).float()
                pooled = torch.sum(out[0] * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            all_vecs.append(pooled.cpu())
        return torch.cat(all_vecs, dim=0)

    def load_fkg(self, json_path: Path, registry_path: Path):
        LOGGER.info("Loading FKG data from %s...", json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        surface_map = {}
        for uri, info in registry.get("uri_info", {}).items():
            surface_map[uri] = info.get("canonical_surface", uri)

        raw = []
        for g in data.get("single_valued_groups", []):
            h, r, w = g.get("head_uri"), g.get("relation"), g.get("winner_uri")
            bel = float(g.get("winner_bel", 0.8))
            if h and r and w:
                hs = g.get("head_surface") or surface_map.get(h, h)
                ws = g.get("winner_surface") or surface_map.get(w, w)
                surface_map[h] = hs
                surface_map[w] = ws
                raw.append({
                    "head_uri": h, "head_text": hs,
                    "relation_raw": r, "relation_text": clean_relation_name(r),
                    "tail_uri": w, "tail_text": ws,
                    "weight": max(0.1, min(1.0, bel))
                })
                self.known_facts.add((h, r, w))

        for m in data.get("multi_valued_facts", []):
            h, r, t = m.get("head_uri"), m.get("relation"), m.get("tail_uri")
            mu = float(m.get("mu_F", 0.5))
            if h and r and t:
                hs = m.get("head_surface") or surface_map.get(h, h)
                ts = m.get("tail_surface") or surface_map.get(t, t)
                surface_map[h] = hs
                surface_map[t] = ts
                raw.append({
                    "head_uri": h, "head_text": hs,
                    "relation_raw": r, "relation_text": clean_relation_name(r),
                    "tail_uri": t, "tail_text": ts,
                    "weight": max(0.1, min(1.0, mu))
                })
                self.known_facts.add((h, r, t))

        dedup = {}
        for item in raw:
            k = (item["head_uri"], item["relation_raw"], item["tail_uri"])
            if k not in dedup or item["weight"] > dedup[k]["weight"]:
                dedup[k] = item

        self.triples = list(dedup.values())
        self.entities = surface_map
        self.relations = set(item["relation_raw"] for item in self.triples)
        LOGGER.info("Loaded %d unique triples across %d entities and %d relations.", len(self.triples), len(self.entities), len(self.relations))

    def train(self, epochs: int = 15, batch_size: int = 128, lr: float = 1e-3):
        random.seed(42)
        torch.manual_seed(42)

        random.shuffle(self.triples)
        split_idx = int(len(self.triples) * 0.90)
        train_data = self.triples[:split_idx]
        test_data = self.triples[split_idx:]

        LOGGER.info("Pre-encoding train and test text representations with %s...", self.model_name)
        t0 = time.time()
        
        train_hr_texts = [f"landmark: {it['head_text']} | relation: {it['relation_text']}" for it in train_data]
        train_t_texts = [f"entity: {it['tail_text']}" for it in train_data]
        train_weights = torch.tensor([it["weight"] for it in train_data], dtype=torch.float32)

        feat_hr = self._encode_texts(train_hr_texts)
        feat_t = self._encode_texts(train_t_texts)
        LOGGER.info("Finished pre-encoding in %.2fs.", time.time() - t0)

        dataset = TensorDataset(feat_hr, feat_t, train_weights)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.AdamW(self.proj_net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        LOGGER.info("Training Contrastive Projection Net for %d epochs...", epochs)
        t_train = time.time()
        self.proj_net.train()

        for ep in range(1, epochs + 1):
            tot_loss = 0.0
            n_b = 0
            for b_hr, b_t, b_w in loader:
                b_hr = b_hr.to(self.device)
                b_t = b_t.to(self.device)
                b_w = b_w.to(self.device)

                optimizer.zero_grad()
                loss, _ = self.proj_net(b_hr, b_t, b_w)
                loss.backward()
                optimizer.step()

                tot_loss += loss.item()
                n_b += 1

            scheduler.step()
            if ep % 3 == 0 or ep == 1 or ep == epochs:
                LOGGER.info("  Epoch [%2d/%2d] — InfoNCE Loss: %.4f | Temp: %.4f", ep, epochs, tot_loss / n_b, self.proj_net.temperature.item())

        LOGGER.info("✓ Contrastive Training finished in %.2fs.", time.time() - t_train)

        # Cache all entity tail embeddings
        self.precompute_all_tails()
        return test_data

    @torch.no_grad()
    def precompute_all_tails(self):
        LOGGER.info("Precomputing all entity tail embeddings...")
        self.proj_net.eval()
        self.all_entity_uris = list(self.entities.keys())
        all_tail_texts = [f"entity: {self.entities[u]}" for u in self.all_entity_uris]

        raw_vecs = self._encode_texts(all_tail_texts, batch_size=256).to(self.device)
        self.cached_tail_embs = F.normalize(self.proj_net.t_proj(raw_vecs), p=2, dim=-1)
        LOGGER.info("Cached %d normalized tail vectors.", len(self.cached_tail_embs))

    @torch.no_grad()
    def predict_links(
        self,
        head_text: str,
        relation_raw: str,
        visual_cue: str = "",
        top_k: int = 5,
        allowed_tail_keywords: Optional[List[str]] = None,
        forbidden_tail_keywords: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        self.proj_net.eval()
        r_text = clean_relation_name(relation_raw)
        if visual_cue:
            hr_prompt = f"landmark: {head_text} | relation: {r_text} | visual context: {visual_cue}"
        else:
            hr_prompt = f"landmark: {head_text} | relation: {r_text}"

        raw_hr = self._encode_texts([hr_prompt]).to(self.device)
        e_hr = F.normalize(self.proj_net.hr_proj(raw_hr), p=2, dim=-1)

        scores = torch.matmul(e_hr, self.cached_tail_embs.t()).squeeze(0)

        if allowed_tail_keywords or forbidden_tail_keywords:
            mask = torch.ones_like(scores, dtype=torch.bool)
            for idx, uri in enumerate(self.all_entity_uris):
                surf = self.entities.get(uri, "").lower()
                if allowed_tail_keywords and not any(kw in surf for kw in allowed_tail_keywords):
                    mask[idx] = False
                if forbidden_tail_keywords and any(kw in surf for kw in forbidden_tail_keywords):
                    mask[idx] = False
            scores[~mask] = -999.0

        top_scores, top_indices = torch.topk(scores, k=min(top_k + 15, len(self.all_entity_uris)))

        results = []
        for sc, idx in zip(top_scores.tolist(), top_indices.tolist()):
            if sc < -500.0: continue
            cand_uri = self.all_entity_uris[idx]
            cand_surf = self.entities[cand_uri]
            if cand_surf.lower() == head_text.lower():
                continue
            results.append({
                "head": head_text,
                "relation": relation_raw,
                "tail": cand_surf,
                "tail_uri": cand_uri,
                "score": round(float(sc), 4),
            })
            if len(results) >= top_k:
                break
        return results

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.proj_net.state_dict(), path / "simkgc_proj_net.pt")
        with open(path / "simkgc_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "model_name": self.model_name,
                "proj_dim": self.proj_dim,
                "entities": self.entities,
                "relations": list(self.relations),
            }, f, indent=2, ensure_ascii=False)
        LOGGER.info("Saved SimKGC model & meta to %s", path)

    def load(self, path: Path):
        meta_file = path / "simkgc_meta.json"
        weights_file = path / "simkgc_proj_net.pt"
        if not meta_file.exists() or not weights_file.exists():
            raise FileNotFoundError(f"Missing SimKGC files in {path}")
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.entities = meta["entities"]
        self.relations = set(meta["relations"])
        self.proj_net.load_state_dict(torch.load(weights_file, map_location=self.device))
        self.proj_net.eval()
        LOGGER.info("Loaded SimKGC model weights and %d entities from %s", len(self.entities), path)


MultimodalFuzzySimKGC = SimKGCEngine
