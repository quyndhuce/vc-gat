# module3_entity_linking_en.py
"""
OntoKG_EN - Module 3: International Linked Data (LOD) Entity Linking for English.

Pipeline:
  Input : Module 2 output (ner_entities, concept_mentions, article_id, topics)
  Output: Linked Data entities with Wikidata URIs (http://www.wikidata.org/entity/Qxxx)
          or internal mint URIs (http://transmtl.vn/ent/...)
          Ontology concept URIs (http://transmtl.vn/onto/...)
          Time URIs (http://transmtl.vn/time/...)

Multi-tier Entity Resolution Architecture:
  Tầng 0: Manual alias dictionary normalization (aliases_en.json)
  Tầng 1: Surface / Alias match in EntityRegistry (O(1) lookup)
  Tầng 2: Wikidata API (wbsearchentities) with local JSON cache, rate limiting,
          and English type-hint disambiguation
          + Ingestion of multilingual aliases (en, vi) into Registry
  Tầng 3: Semantic embedding similarity with MPNet (sentence-transformers/all-mpnet-base-v2)
          Vectorized cosine similarity (threshold >= 0.88)
  Tầng 4: Internal URI minting for unlinked entities (http://transmtl.vn/ent/<md5>)
          and concepts (http://transmtl.vn/onto/<Concept>)

Zero NLTK, zero Vietnamese packages. Pure Python, Hugging Face transformers, and PyTorch.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import pickle
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except Exception:
    torch = None
    AutoModel = None
    AutoTokenizer = None

# ──────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────
LOGGER = logging.getLogger("module3_entity_linking_en")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)

# ──────────────────────────────────────────────────────────
# Linked Data Namespaces (W3C / LOD Standards)
# ──────────────────────────────────────────────────────────
BASE_ENT = "http://transmtl.vn/ent"
BASE_ONTO = "http://transmtl.vn/onto"
BASE_ARTICLE = "http://transmtl.vn/article"
BASE_TOPIC = "http://transmtl.vn/onto/topic"
BASE_TIME = "http://transmtl.vn/time"
WD_URI = "http://www.wikidata.org/entity"

WHITESPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")

# Labels eligible for Wikidata resolution
WIKIDATA_ENABLED_LABELS = {"LOC", "PER", "ORG", "EVENT"}

# Labels eligible for semantic embedding matching (proper-noun entities)
EMB_MATCH_LABELS = {"LOC", "PER", "ORG", "EVENT"}

# English semantic type hints for Wikidata description disambiguation
WIKIDATA_TYPE_HINTS: Dict[str, List[str]] = {
    "LOC": [
        "city", "province", "country", "citadel", "fortress", "palace", "tower",
        "gate", "place", "region", "district", "commune", "town", "village",
        "island", "bridge", "lake", "river", "mountain", "complex", "monument",
        "historic site", "archaeological site", "structure", "architecture",
        "flagpole", "capital", "municipality", "state", "colony", "ward",
        "territory", "vietnam", "hanoi", "asia", "dynasty", "kingdom", "empire",
        "administrative", "prefecture", "county", "seat", "site", "park"
    ],
    "PER": [
        "human", "person", "emperor", "king", "ruler", "dynasty", "politician",
        "general", "military commander", "leader", "official", "architect",
        "engineer", "monarch", "sovereign", "queen", "writer", "historical figure",
        "commander", "statesman", "philosopher", "founder", "prince", "lord",
        "scholar", "poet", "artist"
    ],
    "ORG": [
        "organization", "military", "army", "armed forces", "company", "dynasty",
        "ministry", "agency", "institution", "association", "division", "corps",
        "force", "authority", "empire", "federation", "regiment", "group",
        "party", "government", "navy", "air force", "corps"
    ],
    "EVENT": [
        "coup", "war", "battle", "event", "revolution", "rebellion", "festival",
        "treaty", "incident", "conflict", "operation", "ceremony", "campaign",
        "revolt", "conference", "uprising", "period"
    ],
    "TIME": [
        "year", "century", "decade", "time period", "calendar year", "era", "epoch"
    ]
}


# ──────────────────────────────────────────────────────────
# Normalization Utilities
# ──────────────────────────────────────────────────────────
def normalize_text(value: Any) -> str:
    """Clean whitespaces, HTML entities, and control characters."""
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFC", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\xa0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def normalize_key(text: str) -> str:
    """Produce clean alphanumeric lowercase key for lookup."""
    text = normalize_text(text).lower()
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def make_uri(surface: str, label: str = "LOC") -> str:
    """Generate canonical Linked Data URI for unlinked entities."""
    key = normalize_key(surface)
    if label == "TIME":
        clean_time = re.sub(r"[^\w\-]+", "_", key)
        return f"{BASE_TIME}/{clean_time}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
    return f"{BASE_ENT}/{h}"


def load_alias_map(path: Optional[str]) -> Dict[str, str]:
    """
    Load alias dictionary from JSON file.
    Supports both:
      - {"Canonical Name": ["alias1", "alias2"]}
      - {"alias": "Canonical Name"}
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        LOGGER.debug("Alias file %s does not exist; skipping manual alias map.", path)
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        LOGGER.warning("Error reading alias file %s: %s", path, e)
        return {}

    amap: Dict[str, str] = {}
    for key, val in raw.items():
        if isinstance(val, list):
            canonical = str(key)
            amap[normalize_key(canonical)] = canonical
            for alias in val:
                if alias:
                    amap[normalize_key(alias)] = canonical
        elif val:
            amap[normalize_key(key)] = str(val)
    return amap


# ──────────────────────────────────────────────────────────
# Wikidata Linker with Persistent File Cache
# ──────────────────────────────────────────────────────────
class WikidataLinkerEN:
    """
    Wikidata API Client adhering to Linked Open Data (LOD) standards:
      - Local JSON persistent cache (avoids repeated network calls)
      - Rate limiting (default 0.2s between calls)
      - Domain disambiguation via English description semantic matching
      - Multilingual alias extraction (wbgetentities)
      - Graceful network error / offline fallback
    """

    WIKIDATA_API = "https://www.wikidata.org/w/api.php"
    REQUEST_INTERVAL = 0.2
    HEADERS = {
        "User-Agent": "TransMTL-OntoKG-EN/1.0 (English cultural LOD research; mailto:anonymous@example.com)"
    }

    def __init__(self, cache_path: str = "./data/wikidata_cache_en.json"):
        self.cache_path = Path(cache_path)
        self._cache: Dict[str, Any] = {}
        self._last_request = 0.0
        self._load_cache()

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
                LOGGER.info("Loaded Wikidata cache (%d entries) from %s", len(self._cache), self.cache_path)
            except Exception as e:
                LOGGER.warning("Could not read Wikidata cache: %s", e)

    def save_cache(self):
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            LOGGER.debug("Saved Wikidata cache (%d entries)", len(self._cache))
        except Exception as e:
            LOGGER.warning("Failed to save Wikidata cache: %s", e)

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.REQUEST_INTERVAL:
            time.sleep(self.REQUEST_INTERVAL - elapsed)
        self._last_request = time.time()

    def _search(self, query: str, lang: str = "en", limit: int = 10) -> List[Dict[str, Any]]:
        self._rate_limit()
        params = {
            "action": "wbsearchentities",
            "search": query,
            "language": lang,
            "type": "item",
            "format": "json",
            "limit": limit,
        }
        try:
            resp = requests.get(
                self.WIKIDATA_API, params=params, headers=self.HEADERS, timeout=10
            )
            resp.raise_for_status()
            return resp.json().get("search", [])
        except Exception as e:
            LOGGER.debug("Wikidata API search failed for '%s': %s", query, e)
            return []

    def _type_matches(self, description: str, label: str) -> bool:
        if not description:
            return False
        desc_lower = description.lower()
        hints = WIKIDATA_TYPE_HINTS.get(label, [])
        return any(h in desc_lower for h in hints)

    def lookup(self, surface: str, label: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Query Wikidata for entity surface and label.
        Returns: (wikidata_id, description) or (None, None)
        Results are cached to file.
        """
        if label not in WIKIDATA_ENABLED_LABELS:
            return None, None

        cache_key = f"{label}::{normalize_key(surface)}"
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if isinstance(entry, dict):
                return entry.get("id"), entry.get("description")
            return entry, None

        results = self._search(surface, lang="en", limit=10)
        wikidata_id = None
        matched_desc = None

        # 1. First pass: exact type match on description
        for item in results:
            desc = item.get("description", "")
            if self._type_matches(desc, label):
                wikidata_id = item["id"]
                matched_desc = desc
                break

        # 2. Second pass: for LOC if no desc, accept if label matches closely
        if not wikidata_id and results:
            for item in results:
                desc = item.get("description", "")
                lbl = item.get("label", "").lower()
                if label == "LOC" and (not desc or "vietnam" in desc.lower() or "hanoi" in desc.lower()):
                    if normalize_key(surface) in normalize_key(lbl):
                        wikidata_id = item["id"]
                        matched_desc = desc or "location"
                        break

        # Cache both positive and negative results
        if wikidata_id:
            self._cache[cache_key] = {"id": wikidata_id, "description": matched_desc}
        else:
            self._cache[cache_key] = None

        return wikidata_id, matched_desc

    def get_aliases(self, wikidata_id: str, langs: Tuple[str, ...] = ("en", "vi")) -> List[str]:
        """
        Fetch entity label and aliases in English and Vietnamese from Wikidata.
        Used to register all surface variants into Registry at Tầng 1.
        """
        if not wikidata_id:
            return []
        cache_key = f"ALIASES::{wikidata_id}"
        if cache_key in self._cache:
            return self._cache[cache_key] or []

        self._rate_limit()
        params = {
            "action": "wbgetentities",
            "ids": wikidata_id,
            "props": "labels|aliases",
            "languages": "|".join(langs),
            "format": "json",
        }
        surfaces: List[str] = []
        try:
            resp = requests.get(
                self.WIKIDATA_API, params=params, headers=self.HEADERS, timeout=10
            )
            resp.raise_for_status()
            ent = resp.json().get("entities", {}).get(wikidata_id, {})
            for lang in langs:
                lbl = ent.get("labels", {}).get(lang, {}).get("value")
                if lbl:
                    surfaces.append(lbl)
                for al in ent.get("aliases", {}).get(lang, []):
                    val = al.get("value")
                    if val:
                        surfaces.append(val)
        except Exception as e:
            LOGGER.debug("Wikidata aliases retrieval failed for %s: %s", wikidata_id, e)

        seen: Set[str] = set()
        clean_aliases: List[str] = []
        for s in surfaces:
            s_norm = normalize_text(s)
            k = normalize_key(s_norm)
            if s_norm and k not in seen:
                seen.add(k)
                clean_aliases.append(s_norm)

        self._cache[cache_key] = clean_aliases
        return clean_aliases


# ──────────────────────────────────────────────────────────
# MPNet Semantic Embedding Encoder
# ──────────────────────────────────────────────────────────
class EmbeddingEncoderEN:
    """
    Computes 768-dimensional sentence embeddings using:
    sentence-transformers/all-mpnet-base-v2 via pure Hugging Face transformers.
    Mean pooling over token embeddings with attention mask.
    L2 normalized for fast BLAS dot-product cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._cache: Dict[str, np.ndarray] = {}

        if device is None:
            if torch and torch.cuda.is_available():
                self.device = "cuda"
            elif torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if torch is None or AutoTokenizer is None or AutoModel is None:
            LOGGER.warning("PyTorch/transformers not available — embedding encoder disabled.")
            return

        try:
            LOGGER.info("Loading MPNet embedding model: %s on %s...", model_name, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            if self.device != "cpu":
                self.model = self.model.to(self.device)
            LOGGER.info("✓ Loaded MPNet embedding model: %s", model_name)
        except Exception as e:
            LOGGER.warning("Failed to load MPNet model %s: %s", model_name, e)

    @property
    def available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def encode(self, text: str) -> Optional[np.ndarray]:
        """Encode text into a unit-normalized 768-dim float32 vector."""
        if not self.available:
            return None
        text_clean = normalize_text(text)
        if not text_clean:
            return None
        key = normalize_key(text_clean)
        if key in self._cache:
            return self._cache[key]

        try:
            inputs = self.tokenizer(
                text_clean,
                return_tensors="pt",
                truncation=True,
                max_length=64,
                padding=True,
            )
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = self.model(**inputs)

            mask = inputs["attention_mask"].unsqueeze(-1).float()
            sum_emb = (out.last_hidden_state * mask).sum(dim=1)
            mean_emb = sum_emb / mask.sum(dim=1).clamp(min=1)
            arr = mean_emb[0].cpu().numpy().astype(np.float32)

            norm = np.linalg.norm(arr)
            if norm > 1e-9:
                arr = arr / norm

            self._cache[key] = arr
            return arr
        except Exception as e:
            LOGGER.debug("Encode failed for '%s': %s", text_clean, e)
            return None

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# ──────────────────────────────────────────────────────────
# Entity Registry with Vectorized BLAS Search
# ──────────────────────────────────────────────────────────
class EntityRegistryEN:
    """
    Central repository for all discovered entities in the Knowledge Graph.
    Features:
      - O(1) hash table lookup by normalized surface form (Tầng 1)
      - Fast BLAS matrix multiplication for vector similarity (Tầng 3)
      - Grouping by entity type (LOC, PER, ORG, EVENT) to prevent cross-type collision
      - Persistence to disk via pickle
    """

    def __init__(self):
        # norm_key → uri
        self.surface_to_uri: Dict[str, str] = {}
        # uri → metadata
        self.uri_info: Dict[str, Dict[str, Any]] = {}
        # uri → numpy embedding vector
        self.embeddings: Dict[str, np.ndarray] = {}

        # Vectorized embedding index
        self._emb_buf: Optional[np.ndarray] = None  # (cap, 768) float32
        self._emb_cap: int = 0
        self._emb_n: int = 0
        self._emb_uris: List[str] = []
        self._label_idx: Dict[str, List[int]] = {}

    def get_uri_by_surface(self, norm_key: str) -> Optional[str]:
        return self.surface_to_uri.get(norm_key)

    def _emb_add(self, uri: str, label: str, emb: np.ndarray):
        v = np.asarray(emb, dtype=np.float32)
        n = np.linalg.norm(v)
        if n < 1e-9:
            return
        v = v / n
        dim = v.shape[0]

        if self._emb_buf is None:
            self._emb_cap = 1024
            self._emb_buf = np.zeros((self._emb_cap, dim), dtype=np.float32)

        if self._emb_n >= self._emb_cap:
            self._emb_cap *= 2
            new_buf = np.zeros((self._emb_cap, dim), dtype=np.float32)
            new_buf[: self._emb_n] = self._emb_buf[: self._emb_n]
            self._emb_buf = new_buf

        row = self._emb_n
        self._emb_buf[row] = v
        self._emb_uris.append(uri)
        self._label_idx.setdefault(label, []).append(row)
        self._emb_n += 1

    def get_uri_by_embedding(
        self, emb: np.ndarray, threshold: float = 0.88, label: Optional[str] = None
    ) -> Optional[str]:
        """Vectorized cosine similarity search with label restriction."""
        if self._emb_buf is None or self._emb_n == 0:
            return None

        q = np.asarray(emb, dtype=np.float32)
        nq = np.linalg.norm(q)
        if nq < 1e-9:
            return None
        q = q / nq

        if label is not None:
            rows = self._label_idx.get(label)
            if not rows:
                return None
            sub = self._emb_buf[rows]
            sims = sub @ q
            best_idx = int(np.argmax(sims))
            return self._emb_uris[rows[best_idx]] if sims[best_idx] >= threshold else None

        mat = self._emb_buf[: self._emb_n]
        sims = mat @ q
        best_idx = int(np.argmax(sims))
        return self._emb_uris[best_idx] if sims[best_idx] >= threshold else None

    def register(
        self,
        norm_key: str,
        uri: str,
        surface: str,
        label: str,
        uri_source: str,
        wikidata_id: Optional[str] = None,
        wikidata_desc: Optional[str] = None,
        emb: Optional[np.ndarray] = None,
    ):
        """Register a new entity occurrence or update statistics for an existing entity."""
        if norm_key:
            self.surface_to_uri[norm_key] = uri

        if uri not in self.uri_info:
            self.uri_info[uri] = {
                "uri": uri,
                "label": label,
                "canonical_surface": surface,
                "surface_forms": set(),
                "wikidata_id": wikidata_id,
                "wikidata_description": wikidata_desc,
                "uri_source": uri_source,
                "count": 0,
            }

        info = self.uri_info[uri]
        if surface:
            info["surface_forms"].add(surface)
        info["count"] += 1
        if wikidata_id and not info.get("wikidata_id"):
            info["wikidata_id"] = wikidata_id
        if wikidata_desc and not info.get("wikidata_description"):
            info["wikidata_description"] = wikidata_desc

        if emb is not None and uri not in self.embeddings:
            self.embeddings[uri] = emb
            if label in EMB_MATCH_LABELS:
                self._emb_add(uri, label, emb)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for uri, info in self.uri_info.items():
            serializable[uri] = {**info, "surface_forms": list(info["surface_forms"])}
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "uri_info": serializable,
                    "surface_to_uri": self.surface_to_uri,
                    "embeddings": self.embeddings,
                },
                f,
            )
        LOGGER.info("Entity Registry saved: %d entities → %s", len(self.uri_info), path)

    @classmethod
    def load(cls, path: str) -> "EntityRegistryEN":
        reg = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        reg.surface_to_uri = data.get("surface_to_uri", {})
        reg.embeddings = data.get("embeddings", {})
        for uri, info in data.get("uri_info", {}).items():
            info["surface_forms"] = set(info.get("surface_forms", []))
            reg.uri_info[uri] = info
        LOGGER.info("Entity Registry loaded: %d entities from %s", len(reg.uri_info), path)
        return reg


# ──────────────────────────────────────────────────────────
# Main Module 3 Entity Linker (English Linked Data)
# ──────────────────────────────────────────────────────────
class Module3EntityLinkerEN:
    """
    Module 3 Entity Linker for English Knowledge Graph Construction.
    Features:
      - Multi-tier entity resolution adhering to LOD standards
      - Wikidata linking with persistent local cache
      - Automatic multilingual alias ingestion
      - MPNet semantic embedding similarity
      - Concept phrase linking to ontology URIs
      - Standard temporal URI generation
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,
        wikidata_cache_path: str = "./data/wikidata_cache_en.json",
        similarity_threshold: float = 0.88,
        use_wikidata: bool = True,
        use_embedding_matching: bool = True,
        alias_dict_path: Optional[str] = None,
        ingest_wikidata_aliases: bool = True,
    ):
        self.registry = EntityRegistryEN()
        self.wikidata = WikidataLinkerEN(cache_path=wikidata_cache_path) if use_wikidata else None
        self.encoder = EmbeddingEncoderEN(model_name=embedding_model, device=device)
        self.sim_threshold = similarity_threshold
        self.use_wikidata = use_wikidata
        self.use_embedding = use_embedding_matching and self.encoder.available

        # Load manual domain alias dictionary
        if alias_dict_path is None:
            default_alias_file = Path(__file__).with_name("aliases_en.json")
            if default_alias_file.exists():
                alias_dict_path = str(default_alias_file)

        self.alias_dict_path = alias_dict_path
        self.alias_map = load_alias_map(alias_dict_path)
        if self.alias_map:
            LOGGER.info("Loaded %d manual alias mappings from %s", len(self.alias_map), alias_dict_path)

        self.ingest_wd_aliases = ingest_wikidata_aliases
        self._aliases_ingested: Set[str] = set()

    def save_alias_cache(self, path: Optional[str] = None):
        """Persist all discovered aliases and canonical mappings to aliases_en.json."""
        target_path = Path(path) if path else (Path(self.alias_dict_path) if self.alias_dict_path else Path(__file__).with_name("aliases_en.json"))
        
        current_dict: Dict[str, List[str]] = {}
        if target_path.exists():
            try:
                raw = json.loads(target_path.read_text(encoding="utf-8"))
                for k, v in raw.items():
                    if isinstance(v, list):
                        current_dict[k] = list(v)
                    elif v:
                        current_dict[k] = [str(v)]
            except Exception as e:
                LOGGER.warning("Could not read existing alias file for update: %s", e)

        # Merge newly discovered aliases from registry
        for uri, info in self.registry.uri_info.items():
            canonical = info.get("canonical_surface")
            if not canonical:
                continue
            
            aliases_for_uri = [
                s for s in info.get("surface_forms", set())
                if s and s.strip().lower() != canonical.strip().lower()
            ]
            
            if aliases_for_uri:
                if canonical not in current_dict:
                    current_dict[canonical] = []
                for al in aliases_for_uri:
                    if al not in current_dict[canonical]:
                        current_dict[canonical].append(al)

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(
                json.dumps(current_dict, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            LOGGER.info("Saved updated alias cache (%d canonical entities) to %s", len(current_dict), target_path)
        except Exception as e:
            LOGGER.warning("Failed to save alias cache: %s", e)

    def _ingest_wikidata_aliases(self, uri: str, wikidata_id: str, label: str):
        """Register all English & Vietnamese aliases for a Q-ID to point to the same URI."""
        if not (self.ingest_wd_aliases and self.use_wikidata and self.wikidata):
            return
        if uri in self._aliases_ingested:
            return
        self._aliases_ingested.add(uri)

        aliases = self.wikidata.get_aliases(wikidata_id, langs=("en", "vi"))
        for alias in aliases:
            k = normalize_key(alias)
            if k and k not in self.registry.surface_to_uri:
                self.registry.register(
                    norm_key=k,
                    uri=uri,
                    surface=alias,
                    label=label,
                    uri_source="wikidata_alias",
                    wikidata_id=wikidata_id,
                    emb=None,
                )

    def _link_one(
        self,
        surface: str,
        label: str,
        needs_review_labels: Set[str] = frozenset({"PER", "ORG", "LOC"}),
    ) -> Dict[str, Any]:
        """Resolve a single entity surface and label through the multi-tier pipeline."""
        original_surface = surface

        # ── Tầng 0: Manual alias dictionary normalization ──
        if self.alias_map:
            canonical = self.alias_map.get(normalize_key(surface))
            if canonical:
                surface = canonical

        norm_key = normalize_key(surface)
        emb = self.encoder.encode(surface) if (self.use_embedding and label in EMB_MATCH_LABELS) else None

        # ── Tầng 1: Surface / Alias match in Registry (O(1)) ──
        uri = self.registry.get_uri_by_surface(norm_key)
        if uri:
            info = self.registry.uri_info[uri]
            self.registry.register(
                norm_key=norm_key,
                uri=uri,
                surface=surface,
                label=label,
                uri_source=info["uri_source"],
                wikidata_id=info.get("wikidata_id"),
                wikidata_desc=info.get("wikidata_description"),
                emb=emb,
            )
            return {
                "uri": uri,
                "uri_source": info["uri_source"],
                "wikidata_id": info.get("wikidata_id"),
                "wikidata_description": info.get("wikidata_description"),
                "needs_review": False,
                "embedding": emb.tolist() if emb is not None else None,
            }

        # ── Tầng 2: Wikidata lookup (LOC, PER, ORG, EVENT) ──
        wikidata_id, wikidata_desc = None, None
        if self.use_wikidata and self.wikidata and label in WIKIDATA_ENABLED_LABELS:
            wikidata_id, wikidata_desc = self.wikidata.lookup(surface, label)

        if wikidata_id:
            uri = f"{WD_URI}/{wikidata_id}"
            self.registry.register(
                norm_key=norm_key,
                uri=uri,
                surface=surface,
                label=label,
                uri_source="wikidata",
                wikidata_id=wikidata_id,
                wikidata_desc=wikidata_desc,
                emb=emb,
            )
            # Ingest multilingual aliases for future O(1) resolution
            self._ingest_wikidata_aliases(uri, wikidata_id, label)
            return {
                "uri": uri,
                "uri_source": "wikidata",
                "wikidata_id": wikidata_id,
                "wikidata_description": wikidata_desc,
                "needs_review": False,
                "embedding": emb.tolist() if emb is not None else None,
            }

        # ── Tầng 3: Semantic embedding similarity via MPNet ──
        if emb is not None and self.use_embedding and label in EMB_MATCH_LABELS:
            matched_uri = self.registry.get_uri_by_embedding(
                emb, threshold=self.sim_threshold, label=label
            )
            if matched_uri:
                info = self.registry.uri_info[matched_uri]
                self.registry.register(
                    norm_key=norm_key,
                    uri=matched_uri,
                    surface=surface,
                    label=label,
                    uri_source=info["uri_source"],
                    wikidata_id=info.get("wikidata_id"),
                    wikidata_desc=info.get("wikidata_description"),
                    emb=emb,
                )
                return {
                    "uri": matched_uri,
                    "uri_source": "embedding_match",
                    "wikidata_id": info.get("wikidata_id"),
                    "wikidata_description": info.get("wikidata_description"),
                    "needs_review": False,
                    "embedding": emb.tolist() if emb is not None else None,
                }

        # ── Tầng 4: Internal URI minting ──
        uri = make_uri(surface, label=label)
        needs_review = (label in needs_review_labels)
        self.registry.register(
            norm_key=norm_key,
            uri=uri,
            surface=surface,
            label=label,
            uri_source="new",
            wikidata_id=None,
            wikidata_desc=None,
            emb=emb,
        )
        return {
            "uri": uri,
            "uri_source": "new",
            "wikidata_id": None,
            "wikidata_description": None,
            "needs_review": needs_review,
            "embedding": emb.tolist() if emb is not None else None,
        }

    def _link_concept(self, concept_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Link a domain concept mention to an ontology URI.
        e.g. onto:AncientCitadel -> http://transmtl.vn/onto/AncientCitadel
        """
        concept = concept_dict.get("concept", "")
        linked_id = concept_dict.get("linked_ontology_id")
        action = concept_dict.get("action", "new")

        if linked_id and action == "link":
            clean_class = linked_id.replace("onto:", "").replace("onto/", "").strip()
            uri = f"{BASE_ONTO}/{clean_class}"
            uri_source = "ontology"
        else:
            # Capitalize words into CamelCase concept identifier
            clean_name = "".join(word.capitalize() for word in re.findall(r"\w+", concept))
            uri = f"{BASE_ONTO}/{clean_name}"
            uri_source = "new_concept"

        return {
            "uri": uri,
            "uri_source": uri_source,
            "wikidata_id": None,
            "wikidata_description": None,
            "needs_review": (action != "link"),
        }

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single article record from Module 2:
          - Resolve ner_entities to Linked Data URIs
          - Resolve concept_mentions to ontology URIs
          - Mint canonical article_uri and topic_uris
        """
        article_id = record.get("article_id", "art_unknown")
        topic_list = record.get("topic_list", [])

        # 1. Link Named Entities
        linked_entities: List[Dict[str, Any]] = []
        for ent in record.get("ner_entities", []):
            link_info = self._link_one(ent["surface"], ent["label"])
            linked_entities.append({**ent, **link_info})

        # 2. Link Concepts
        linked_concepts: List[Dict[str, Any]] = []
        for con in record.get("concept_mentions", []):
            link_info = self._link_concept(con)
            linked_concepts.append({**con, **link_info})

        # 3. Mint metadata URIs
        article_uri = f"{BASE_ARTICLE}/{article_id}"
        topic_uris = [f"{BASE_TOPIC}/{t}" for t in topic_list]

        # 4. Assemble output record
        out = dict(record)
        out["ner_entities"] = linked_entities
        out["concept_mentions"] = linked_concepts
        out["article_uri"] = article_uri
        out["topic_uris"] = topic_uris

        return out

    def link_all(
        self,
        input_path: str,
        output_path: str,
        registry_pkl: str,
        wikidata_cache_json: str,
        save_every: int = 200,
    ):
        """
        Process an input file (.json or .jsonl) and write linked records.
        Periodically checkpoints registry and cache.
        """
        in_p = Path(input_path)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if not in_p.exists():
            raise FileNotFoundError(f"Input file not found: {in_p}")

        LOGGER.info("Starting Entity Linking on %s...", in_p)
        is_jsonl = in_p.suffix == ".jsonl"

        records: List[Dict[str, Any]] = []
        if is_jsonl:
            with in_p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        else:
            raw_content = in_p.read_text(encoding="utf-8").strip()
            data = json.loads(raw_content)
            records = data if isinstance(data, list) else [data]

        linked_records: List[Dict[str, Any]] = []
        errors = 0

        for i, rec in enumerate(records, start=1):
            try:
                linked = self.process_record(rec)
                linked_records.append(linked)
            except Exception as e:
                LOGGER.exception("Error processing record %d: %s", i, e)
                errors += 1

            if i % save_every == 0:
                self.registry.save(registry_pkl)
                if self.wikidata:
                    self.wikidata.save_cache()
                LOGGER.info("Progress: %d/%d processed (%d errors)", i, len(records), errors)

        # Final save
        if is_jsonl:
            with out_p.open("w", encoding="utf-8") as f:
                for r in linked_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            with out_p.open("w", encoding="utf-8") as f:
                if len(linked_records) == 1 and not isinstance(data, list):
                    json.dump(linked_records[0], f, indent=2, ensure_ascii=False)
                else:
                    json.dump(linked_records, f, indent=2, ensure_ascii=False)

        self.registry.save(registry_pkl)
        if self.wikidata:
            self.wikidata.save_cache()

        LOGGER.info("✓ Finished Entity Linking: %d processed, %d errors → %s", len(linked_records), errors, out_p)


# ──────────────────────────────────────────────────────────
# Standalone CLI Entrypoint
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    current_dir = Path(__file__).resolve().parent
    default_input = current_dir / "test_output.json"
    default_output = current_dir / "test_linked_output.json"
    default_registry = current_dir / "entity_registry_en.pkl"
    default_cache = current_dir / "wikidata_cache_en.json"

    input_file = os.environ.get("OKG_INPUT_FILE", str(default_input))
    output_file = os.environ.get("OKG_OUTPUT_FILE", str(default_output))
    registry_file = os.environ.get("OKG_REGISTRY_FILE", str(default_registry))
    cache_file = os.environ.get("OKG_CACHE_FILE", str(default_cache))

    linker = Module3EntityLinkerEN(
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        wikidata_cache_path=cache_file,
        similarity_threshold=0.88,
        use_wikidata=True,
        use_embedding_matching=True,
        ingest_wikidata_aliases=True,
    )

    linker.link_all(
        input_path=input_file,
        output_path=output_file,
        registry_pkl=registry_file,
        wikidata_cache_json=cache_file,
    )
