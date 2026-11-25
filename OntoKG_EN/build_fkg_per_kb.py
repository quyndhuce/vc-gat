# build_fkg_per_kb.py
"""
Build Landmark-Centric Fuzzy Knowledge Graphs (FKG) per KB-ID from Multi-Document Sources.
========================================================================================

Architecture & Theoretical Foundations:
  1. Wikipedia Link Cleaner: Strips markdown links, wikitext links, HTML tags,
     and citation brackets from Wikipedia/web documents, preserving anchor entity text.
  2. Shared Node Registry with Landmark Hierarchy & Anaphora Resolution:
     - Indexes all 233 KB nodes and maps their parent-child hierarchy.
     - Maps English & Vietnamese aliases, generates compound variants.
     - Resolves Wikipedia URLs to Wikidata IDs (Q-identifiers).
     - Context-Aware Landmark Anaphora: Resolves generic references ("the complex",
       "the site", "the relic", "the section") to the enclosing Landmark/Parent Landmark.
     - Architectural Meronym Anchoring: Meronyms and sub-components (courtyard, wall,
       gate, sanctuary, pavilion) are bound to the Landmark (e.g. "Temple of Literature - Courtyard")
       with scoped URIs and automatic hierarchical `has_part` edges.
  3. Strict Ontological Validation: Enforces constraints (e.g., architectural elements
     cannot be `part_of` macro-geographic entities like countries or provinces).
  4. Landmark-Centric Graph Filtering: Ensures all triples in a KB's graph strictly revolve
     around the primary Landmark (0-hop direct facts and 1-hop high-value context attributes),
     pruning disconnected peripheral noise.
  5. Per-KB FKG Fusion (Module 5): Combines all document graphs for a kb_id using
     hybrid Dempster-Shafer (single-valued) and product t-conorm (multi-valued) evidence
     fusion with copy detection and entity unification.
  6. Global Structured Index: Writes clean structured outputs under `data/fkg_per_kb/`
     enabling cross-KB global graph synthesis.

Execution:
  python3 OntoKG_EN/build_fkg_per_kb.py --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ── Make root importable ───────────────────────────────────
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

import numpy as np

# ── Try importing rdflib ───────────────────────────────────
try:
    from rdflib import Graph, URIRef, Literal, Namespace, BNode
    from rdflib.namespace import RDF, RDFS, XSD
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False

# ── Try importing torch & GLiNER ───────────────────────────
try:
    import torch
    from gliner import GLiNER
    HAS_TORCH_GLINER = True
except ImportError:
    HAS_TORCH_GLINER = False

# ── Module 1, 4, 5 Imports ────────────────────────────────
from OntoKG_EN.module1_preprocess_en import split_sentences_en, normalize_text_field
from OntoKG_EN.module4_relation_extraction_en import (
    ONTOLOGY_SCHEMA,
    PATTERN_TO_CANONICAL,
    ALL_GLINER_RELATION_LABELS,
    validate_triple_en,
)
from OntoKG_EN.module5_fkg_builder_en import (
    FKGBuilder,
    Evidence,
    FactCandidate,
    FactGroup,
    FuzzyFact,
    MetadataFact,
    t_conorm_combine,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("build_fkg_per_kb")


# ══════════════════════════════════════════════════════════
# 0. VOCABULARIES FOR ANCHORING & LANDMARK CENTRICITY
# ══════════════════════════════════════════════════════════

ARCHITECTURAL_MERONYMS: Set[str] = {
    "courtyard", "sân", "gạch vồ wall", "wall", "tường", "gate", "cổng",
    "sanctuary", "điện", "bái đường", "hậu cung", "hall", "pavilion", "tower",
    "tháp", "shrine", "well", "giếng", "pond", "ao", "lake", "hồ", "garden",
    "vườn", "pathway", "con đường", "belfry", "gác chuông", "stele", "bia",
    "altar", "bàn thờ", "corridor", "hành lang", "roof", "mái", "column",
    "pillar", "cột", "door", "cửa", "railing", "lan can", "lawn", "bãi cỏ",
    "ancient trees", "stone tortoises", "curved tile roofs", "brick path",
    "brick paths", "stelae", "doctor stelae", "belfrey"
}

LANDMARK_ANAPHORA_WORDS: Set[str] = {
    "complex", "the complex", "this complex", "the site", "site", "this site",
    "the relic", "relic", "this relic", "the monument", "monument", "this monument",
    "the sanctuary", "sanctuary", "the compound", "compound",
    "quần thể", "khu di tích", "di tích", "di tích này", "khu này"
}

SECTION_ANAPHORA_WORDS: Set[str] = {
    "section", "the section", "this section", "the area", "area", "this area",
    "khu", "khu vực", "khu này"
}

HIGH_VALUE_CONTEXT_RELS: Set[str] = {
    "belongs_to_dynasty",
    "during_era",
    "reigned_from",
    "reigned_until",
    "capital_of",
    "located_in",
    "held_position",
    "born_in",
    "died_in",
    "ruled_by",
    "honored_as",
}


# ══════════════════════════════════════════════════════════
# 1. WIKIPEDIA LINK & TEXT CLEANER
# ══════════════════════════════════════════════════════════

def clean_wikipedia_document(raw_text: str) -> Tuple[str, Dict[str, str]]:
    """Clean Wikipedia / Web document content while extracting metadata.

    Strips:
      - Leading headers ('Source URL: ...', 'Article Title: ...')
      - Markdown links: [Anchor Text](http...) -> Anchor Text
      - Wikitext links: [[Target|Anchor Text]] -> Anchor Text, [[Target]] -> Target
      - HTML links & tags: <a href="...">Anchor</a> -> Anchor, <[^>]+> -> ''
      - Citation brackets: [1], [23], [citation needed], [edit] -> ''
      - Bare URLs: https?://... -> ''
      - Excessive whitespace and empty lines.

    Preserves the anchor text as the entity surface mention!
    """
    if not raw_text:
        return "", {}

    meta = {}
    lines = raw_text.splitlines()
    body_lines = []

    # 1. Extract metadata header if present
    for line in lines:
        sline = line.strip()
        if sline.startswith("Source URL:"):
            meta["source_url"] = sline[len("Source URL:"):].strip()
        elif sline.startswith("Article Title:"):
            meta["article_title"] = sline[len("Article Title:"):].strip()
        else:
            body_lines.append(line)

    text = "\n".join(body_lines)

    # 2. Markdown links: [anchor](url) -> anchor
    text = re.sub(r"\[([^\]\n]+)\]\([^\)\n]+\)", r"\1", text)

    # 3. Wikitext links: [[target|anchor]] -> anchor, [[target]] -> target
    def _clean_wikilink(m: re.Match) -> str:
        content = m.group(1)
        if "|" in content:
            parts = content.split("|", 1)
            return parts[1].strip()
        return content.strip()

    text = re.sub(r"\[\[([^\]\n]+)\]\]", _clean_wikilink, text)

    # 4. HTML links: <a href="...">anchor</a> -> anchor
    text = re.sub(r"<a\b[^>]*>(.*?)</a>", r"\1", text, flags=re.IGNORECASE | re.DOTALL)

    # 5. Other HTML tags: <b>, </i>, <p>, etc. -> ''
    text = re.sub(r"<[^>]+>", " ", text)

    # 6. Wikipedia citations & section edit markers: [1], [12], [citation needed], [edit]
    text = re.sub(r"\[(?:\d+|citation needed|edit|note \d+)\]", "", text, flags=re.IGNORECASE)

    # 7. Bare URLs: https://... or http://...
    text = re.sub(r"https?://\S+", "", text)

    # 8. Clean leftover empty brackets or artifacts: [], [ ], etc.
    text = re.sub(r"\[\s*\]", "", text)

    # 9. Normalize whitespace and paragraph boundaries
    cleaned_lines = []
    for line in text.splitlines():
        cl = re.sub(r"[ \t]+", " ", line).strip()
        if cl:
            cleaned_lines.append(cl)

    cleaned_text = "\n\n".join(cleaned_lines)
    return cleaned_text, meta


# ══════════════════════════════════════════════════════════
# 2. SHARED NODE REGISTRY WITH ANCHORING & HIERARCHY
# ══════════════════════════════════════════════════════════

def _resolve_wikipedia_url_to_wikidata(
    url: str,
    cache: Dict[str, Optional[str]],
    allow_network: bool = True,
) -> Optional[str]:
    """Query Wikipedia API to get the exact Wikidata item ID (e.g. Q1858)."""
    if not url or "wikipedia.org" not in url:
        return None

    if url in cache:
        return cache[url]

    if not allow_network:
        return None

    try:
        parsed = urllib.parse.urlparse(url)
        lang = "en"
        if "vi.wikipedia.org" in parsed.netloc:
            lang = "vi"

        slug = parsed.path.split("/")[-1]
        title = urllib.parse.unquote(slug).replace("_", " ")

        api_url = (
            f"https://{lang}.wikipedia.org/w/api.php?action=query&prop=pageprops"
            f"&ppprop=wikibase_item&titles={urllib.parse.quote(title)}&format=json"
        )
        req = urllib.request.Request(api_url, headers={"User-Agent": "TransMTL-FKG-PerKB/1.0"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.load(resp)
            pages = data.get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                if pid != "-1":
                    qid = pdata.get("pageprops", {}).get("wikibase_item")
                    if qid:
                        cache[url] = qid
                        return qid
    except Exception as e:
        LOGGER.debug("Wikipedia API query failed for %s: %s", url, e)

    cache[url] = None
    return None


class SharedNodeRegistry:
    """Shared entity registry across all KB nodes and documents.

    Maps:
      - kb_id -> canonical_uri
      - surface forms (names, aliases, normalized strings) -> canonical_uri
      - uri -> metadata (canonical_name, name_vi, wikidata_id, type)
      - parent/child landmark hierarchy
      - landmark anaphora resolution
      - architectural meronym anchoring
    """

    def __init__(self, cache_path: Optional[str] = None):
        self.cache_path = Path(cache_path) if cache_path else None
        self.uri_info: Dict[str, Dict[str, Any]] = {}
        self.surface_to_uri: Dict[str, str] = {}
        self.kb_id_to_uri: Dict[str, str] = {}
        self.mongo_id_to_kb_id: Dict[str, str] = {}
        self.kb_id_to_parent_kb_id: Dict[str, str] = {}
        self.kb_id_to_children_kb_ids: Dict[str, List[str]] = defaultdict(list)
        self.wikidata_cache: Dict[str, Optional[str]] = {}

        if self.cache_path and self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.wikidata_cache = json.load(f)
                LOGGER.info("Loaded %d cached Wikidata entries", len(self.wikidata_cache))
            except Exception as e:
                LOGGER.warning("Could not load wikidata cache: %s", e)

    def _save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.wikidata_cache, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _normalize_surface(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[\s_–—\-]+", " ", s)
        return s

    def build_from_kb_export(
        self,
        kb_export_path: str,
        aliases_path: Optional[str] = None,
        allow_network: bool = True,
    ) -> None:
        """Build shared registry from nodes_flat in kb_full_export JSON and aliases."""
        with open(kb_export_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes_flat = data.get("nodes_flat", [])
        LOGGER.info("Building SharedNodeRegistry from %d KB nodes...", len(nodes_flat))

        # 1. Build mongo ID to kb_id map
        for node in nodes_flat:
            nid = node.get("id") or node.get("_id")
            kbid = node.get("kb_id", "").strip()
            if nid and kbid:
                self.mongo_id_to_kb_id[str(nid)] = kbid

        # 2. Build parent-child relationships
        for node in nodes_flat:
            kbid = node.get("kb_id", "").strip()
            pid = node.get("parent_id")
            if kbid and pid and str(pid) in self.mongo_id_to_kb_id:
                pkbid = self.mongo_id_to_kb_id[str(pid)]
                self.kb_id_to_parent_kb_id[kbid] = pkbid
                self.kb_id_to_children_kb_ids[pkbid].append(kbid)

        # Load aliases dictionary if available
        external_aliases: Dict[str, List[str]] = {}
        if aliases_path and Path(aliases_path).exists():
            with open(aliases_path, "r", encoding="utf-8") as f:
                external_aliases = json.load(f)
            LOGGER.info("Loaded %d external alias groups", len(external_aliases))

        wiki_count = 0
        for node in nodes_flat:
            kb_id = node.get("kb_id", "").strip()
            if not kb_id:
                continue

            name_en = node.get("name", "").strip()
            name_vi = node.get("name_vi", "").strip()
            ent_type = node.get("type", "object")

            # Check if any document has a Wikipedia URL
            wiki_url = None
            wikidata_id = None
            for doc in node.get("documents", []):
                u = doc.get("url", "")
                if u and "wikipedia.org" in u:
                    wiki_url = u
                    break

            if wiki_url:
                wikidata_id = _resolve_wikipedia_url_to_wikidata(
                    wiki_url, self.wikidata_cache, allow_network=allow_network
                )
                if wikidata_id:
                    wiki_count += 1

            # Determine Canonical URI
            if wikidata_id:
                canonical_uri = f"http://www.wikidata.org/entity/{wikidata_id}"
                uri_source = "wikidata"
            else:
                canonical_uri = f"http://transmtl.vn/kb/{kb_id}"
                uri_source = "kb_node"

            self.kb_id_to_uri[kb_id] = canonical_uri

            # Collect surface forms
            surface_forms: Set[str] = set()
            if name_en:
                surface_forms.add(name_en)
            if name_vi:
                surface_forms.add(name_vi)

            # Add external aliases
            for alias in external_aliases.get(name_en, []):
                surface_forms.add(alias)
            for alias in external_aliases.get(name_vi, []):
                surface_forms.add(alias)

            # 3. Auto-generate compound and trimmed aliases for cultural landmarks
            auto_aliases: Set[str] = set()
            for s in [name_en, name_vi]:
                if not s:
                    continue
                # Trim 'section'
                if re.search(r"\bsection\b", s, re.IGNORECASE):
                    auto_aliases.add(re.sub(r"\bsection\b", "", s, flags=re.IGNORECASE).strip())
                # Trim 'gate'
                if re.search(r"\bgate\b", s, re.IGNORECASE):
                    auto_aliases.add(re.sub(r"\bgate\b", "", s, flags=re.IGNORECASE).strip())
                # Trim 'courtyard'
                if re.search(r"\bcourtyard\b", s, re.IGNORECASE):
                    auto_aliases.add(re.sub(r"\bcourtyard\b", "", s, flags=re.IGNORECASE).strip())
                # Vietnamese prefixes
                if s.startswith("Khu "):
                    auto_aliases.add(s[len("Khu "):].strip())
                if s.startswith("Cổng "):
                    auto_aliases.add(s[len("Cổng "):].strip())
                if s.startswith("Sân "):
                    auto_aliases.add(s[len("Sân "):].strip())
                if s.startswith("Đền "):
                    auto_aliases.add(s[len("Đền "):].strip())
                if s.startswith("Chùa "):
                    auto_aliases.add(s[len("Chùa "):].strip())

            # Specific high-value historical compound names
            if kb_id == "temple_of_literature":
                auto_aliases.update({
                    "Temple of Literature - Quoc Tu Giam",
                    "Temple of Literature - Quốc Tử Giám",
                    "Temple of Literature - Quoc Tu Giam complex",
                    "Temple of Literature - Quốc Tử Giám complex",
                    "Temple of Literature complex",
                    "Văn Miếu – Quốc Tử Giám",
                    "Văn Miếu - Quốc Tử Giám",
                    "Văn Miếu - Quốc Tử Giám complex",
                    "Quần thể Văn Miếu - Quốc Tử Giám",
                    "Quốc Tử Giám",
                    "Văn Miếu",
                })
            elif kb_id == "nhap_dao_section":
                auto_aliases.update({
                    "Nhập Đạo",
                    "Nhap Dao",
                    "Khu Nhập Đạo",
                    "Nhap Dao courtyard",
                })
            elif kb_id == "thanh_dat_section":
                auto_aliases.update({
                    "Thành Đạt",
                    "Thanh Dat",
                    "Khu Thành Đạt",
                })
            elif kb_id == "van_mieu_gate":
                auto_aliases.update({
                    "Văn Miếu Môn",
                    "Van Mieu Mon",
                    "Cổng Văn Miếu",
                })

            for a in auto_aliases:
                if a and len(a) >= 3:
                    surface_forms.add(a)

            # Store in uri_info
            if canonical_uri not in self.uri_info:
                self.uri_info[canonical_uri] = {
                    "uri":               canonical_uri,
                    "kb_id":             kb_id,
                    "canonical_surface": name_en or name_vi or kb_id,
                    "name_en":           name_en,
                    "name_vi":           name_vi,
                    "label":             ent_type.upper(),
                    "wikidata_id":       wikidata_id,
                    "wikipedia_url":     wiki_url,
                    "uri_source":        uri_source,
                    "surface_forms":     sorted(surface_forms),
                }
            else:
                self.uri_info[canonical_uri]["surface_forms"] = sorted(
                    set(self.uri_info[canonical_uri]["surface_forms"]) | surface_forms
                )

            # Register in surface_to_uri
            for sf in surface_forms:
                norm = self._normalize_surface(sf)
                if norm:
                    self.surface_to_uri[norm] = canonical_uri

            # Also register kb_id
            self.surface_to_uri[self._normalize_surface(kb_id)] = canonical_uri

        self._save_cache()
        LOGGER.info(
            "SharedNodeRegistry built: %d unique URIs (%d mapped to Wikidata), %d surface forms, %d hierarchy links",
            len(self.uri_info), wiki_count, len(self.surface_to_uri), len(self.kb_id_to_parent_kb_id),
        )

    def resolve(
        self,
        surface: str,
        context_kb_id: Optional[str] = None,
        ent_type: Optional[str] = None,
        is_head: bool = True,
    ) -> Tuple[str, str, Optional[str], str]:
        """Resolve a surface mention with Landmark-Anchoring & Contextual Anaphora.

        Resolution Priority:
          1. Exact match in surface_to_uri
          2. Contextual Landmark Type Anaphora (e.g. 'the pagoda', 'citadel', 'site' -> parent/current landmark)
          3. Contextual Section Anaphora ('section', 'the section', 'area' -> current section if applicable)
          4. Substring / Infix Matching against Known Landmark Nodes
          5. Architectural Meronym Anchoring (courtyard, wall, gate -> anchored to context landmark, is_head=True only)
          6. Fallback: Deterministic Hash URI
        """
        norm = self._normalize_surface(surface)
        if not norm:
            return ("http://transmtl.vn/ent/unknown", surface, None, "empty")

        norm_no_the = re.sub(r"^the\s+", "", norm).strip()

        # 1. Exact match in registry (try full and without leading 'the')
        for k in [norm, norm_no_the]:
            if k in self.surface_to_uri:
                uri = self.surface_to_uri[k]
                info = self.uri_info.get(uri, {})
                return (
                    uri,
                    info.get("canonical_surface", surface),
                    info.get("wikidata_id"),
                    info.get("uri_source", "registry"),
                )

        # 2. Contextual Landmark Type Anaphora (e.g. "pagoda", "temple", "citadel", "complex", "site")
        # In a document about a specific landmark, bare mentions of its type refer to the landmark itself!
        CONTEXTUAL_TYPE_WORDS = {
            "pagoda", "temple", "citadel", "lake", "village", "craft village", "church", "cathedral",
            "monument", "sanctuary", "structure", "compound", "complex", "site", "relic", "park",
            "theater", "theatre", "palace", "market",
            "chùa", "đền", "thành", "hồ", "làng", "nhà thờ", "di tích", "quần thể", "công viên"
        }
        if context_kb_id and (norm in LANDMARK_ANAPHORA_WORDS or norm_no_the in CONTEXTUAL_TYPE_WORDS or norm in CONTEXTUAL_TYPE_WORDS):
            parent_kb = self.kb_id_to_parent_kb_id.get(context_kb_id)
            target_kb = parent_kb if parent_kb else context_kb_id
            target_uri = self.kb_id_to_uri.get(target_kb, f"http://transmtl.vn/kb/{target_kb}")
            info = self.uri_info.get(target_uri, {})
            return (
                target_uri,
                info.get("canonical_surface", target_kb),
                info.get("wikidata_id"),
                "context_landmark_anaphora",
            )

        # 3. Contextual Section Anaphora (e.g. "section", "the section", "area")
        if context_kb_id and (norm in SECTION_ANAPHORA_WORDS or norm_no_the in SECTION_ANAPHORA_WORDS):
            ctx_uri = self.kb_id_to_uri.get(context_kb_id, f"http://transmtl.vn/kb/{context_kb_id}")
            info = self.uri_info.get(ctx_uri, {})
            return (
                ctx_uri,
                info.get("canonical_surface", context_kb_id),
                info.get("wikidata_id"),
                "context_section_anaphora",
            )

        # 4. Substring / Infix Matching against Known Registered Landmark Names
        best_match_uri = None
        best_match_len = 0
        for sf, uri in self.surface_to_uri.items():
            if len(sf) >= 6 and (sf in norm or sf in norm_no_the):
                if sf not in LANDMARK_ANAPHORA_WORDS and sf not in SECTION_ANAPHORA_WORDS:
                    if len(sf) > best_match_len:
                        best_match_len = len(sf)
                        best_match_uri = uri

        if best_match_uri:
            info = self.uri_info.get(best_match_uri, {})
            return (
                best_match_uri,
                info.get("canonical_surface", surface),
                info.get("wikidata_id"),
                "substring_match",
            )

        # 5. Architectural Meronym Anchoring (Sub-Component Binding)
        # CRUCIAL CONSTRAINT: Only anchor if is_head=True!
        # When anchored, use CLEAN NATURAL SURFACE (no landmark prefix in surface name!)
        FORBIDDEN_ENTITY_TYPES = {
            "person", "religious figure", "dynasty", "historical state or colony",
            "country", "city", "province", "year", "date", "date range", "century",
            "historical event", "war or battle", "title or rank", "government or regime",
            "building material", "era name",
        }
        NON_MERONYM_BLACKLIST = {
            "westward", "eastward", "northward", "southward", "taoism", "buddhism", "confucianism",
            "wood", "stone", "brick", "gold", "silver", "bronze", "marble", "clay", "timber",
            "landscape", "sky", "water", "air", "earth", "fire", "nature"
        }
        is_blacklisted = (
            any(b in norm for b in NON_MERONYM_BLACKLIST)
            or (ent_type in FORBIDDEN_ENTITY_TYPES)
        )
        has_meronym_word = False
        if not is_blacklisted:
            for m in ARCHITECTURAL_MERONYMS:
                if re.search(r"\b" + re.escape(m) + r"\b", norm, re.IGNORECASE):
                    has_meronym_word = True
                    break

        is_meronym = (
            is_head
            and not is_blacklisted
            and (
                ent_type in {"architectural element", "structure"}
                or has_meronym_word
            )
        )
        if is_meronym and context_kb_id:
            ctx_uri = self.kb_id_to_uri.get(context_kb_id, f"http://transmtl.vn/kb/{context_kb_id}")
            clean_part = re.sub(r"[^\w\-]+", "_", norm).strip("_")
            scoped_uri = f"{ctx_uri}#{clean_part}"
            # Surface is pure natural surface: e.g. "dragon motif", "column", "lotus flower"
            scoped_surface = surface.strip()
            return (scoped_uri, scoped_surface, None, "anchored_meronym")

        # 6. Fallback: mint deterministic URI
        h = hashlib.md5(norm.encode("utf-8")).hexdigest()[:8]
        minted_uri = f"http://transmtl.vn/ent/{h}"
        return (minted_uri, surface, None, "minted")

    def get_canonical_surface(self, kb_id: str) -> str:
        uri = self.kb_id_to_uri.get(kb_id)
        if uri and uri in self.uri_info:
            return self.uri_info[uri].get("canonical_surface", kb_id)
        return kb_id

    def get_wikidata_id(self, kb_id: str) -> Optional[str]:
        uri = self.kb_id_to_uri.get(kb_id)
        if uri and uri in self.uri_info:
            return self.uri_info[uri].get("wikidata_id")
        return None

    def get_kb_uri(self, kb_id: str) -> str:
        return self.kb_id_to_uri.get(kb_id, f"http://transmtl.vn/kb/{kb_id}")

    def save(self, json_path: Path, pkl_path: Path) -> None:
        """Save registry to both JSON and Pickle formats."""
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "uri_info":                self.uri_info,
                "surface_to_uri":          self.surface_to_uri,
                "kb_id_to_uri":            self.kb_id_to_uri,
                "mongo_id_to_kb_id":       self.mongo_id_to_kb_id,
                "kb_id_to_parent_kb_id":   self.kb_id_to_parent_kb_id,
                "kb_id_to_children_kb_ids": dict(self.kb_id_to_children_kb_ids),
            }, f, ensure_ascii=False, indent=2)

        with open(pkl_path, "wb") as f:
            pickle.dump({
                "uri_info":                self.uri_info,
                "surface_to_uri":          self.surface_to_uri,
                "kb_id_to_uri":            self.kb_id_to_uri,
                "mongo_id_to_kb_id":       self.mongo_id_to_kb_id,
                "kb_id_to_parent_kb_id":   self.kb_id_to_parent_kb_id,
                "kb_id_to_children_kb_ids": dict(self.kb_id_to_children_kb_ids),
            }, f)
        LOGGER.info("Saved SharedNodeRegistry -> %s and %s", json_path, pkl_path)


# ══════════════════════════════════════════════════════════
# 3. GLINER EXTRACTOR & MERONYM LINKER (GPU ACCELERATED)
# ══════════════════════════════════════════════════════════

GLINER_ENTITY_LABELS = [
    "location", "person", "organization", "dynasty", "city", "province",
    "country", "ancient citadel or fortress", "tower", "gate", "bridge",
    "year", "date range", "century", "historical state or colony",
    "historical event", "war or battle", "architectural style",
    "building material", "era name", "historical place name",
    "title or rank", "government or regime", "architectural element",
    "religious figure", "official position", "lunar date",
    "temple", "pagoda", "lake", "building", "mountain", "monument",
    "roof", "column", "pillar", "window", "door", "ornament", "motif",
    "inscription", "facade", "courtyard", "structure", "relic", "statue"
]

class DocumentGraphExtractor:
    """Extracts entities and relations from cleaned document text using GLiNER on GPU."""

    def __init__(
        self,
        model_name: str = "knowledgator/gliner-relex-large-v1.0",
        device: str = "cuda",
        window_size: int = 2,
        threshold: float = 0.25,
    ):
        self.device = device
        self.window_size = window_size
        self.threshold = threshold
        self.model = None

        if HAS_TORCH_GLINER:
            LOGGER.info("Loading GLiNER-Relex model '%s' on %s (threshold=%.2f)...", model_name, device, threshold)
            t0 = time.time()
            self.model = GLiNER.from_pretrained(model_name).to(device)
            LOGGER.info("✓ Model loaded on %s in %.2fs", device, time.time() - t0)
        else:
            LOGGER.warning("GLiNER or PyTorch not available!")

    def extract_document_triples(
        self,
        cleaned_text: str,
        article_id: str,
        kb_id: str,
        registry: SharedNodeRegistry,
    ) -> List[Dict[str, Any]]:
        """Run sentence sliding window and extract validated triples with meronym binding."""
        if not self.model or not cleaned_text:
            return []

        sentences = split_sentences_en(cleaned_text)
        if not sentences:
            return []

        # 1. Build Multi-Sentence Sliding Windows
        windows: List[Tuple[int, str]] = []
        n_sent = len(sentences)
        k = self.window_size

        for start_idx in range(0, n_sent, 1):
            end_idx = min(start_idx + k, n_sent)
            chunk = sentences[start_idx:end_idx]
            w_text = " ".join(normalize_text_field(s) for s in chunk if s.strip())
            if w_text:
                windows.append((start_idx, w_text))
            if end_idx >= n_sent:
                break

        if not windows:
            return []

        window_texts = [w[1] for w in windows]
        window_indices = [w[0] for w in windows]

        # 2. Run GLiNER inference in batches on GPU
        try:
            _, batch_relations = self.model.inference(
                texts=window_texts,
                labels=GLINER_ENTITY_LABELS,
                relations=ALL_GLINER_RELATION_LABELS,
                flat_ner=True,
                threshold=self.threshold,
            )
        except Exception as e:
            LOGGER.warning("Inference failed for doc %s: %s", article_id, e)
            return []

        # 3. Process extracted triples
        extracted_triples: List[Dict[str, Any]] = []

        host_uri = registry.get_kb_uri(kb_id)
        host_surface = registry.get_canonical_surface(kb_id)
        host_qid = registry.get_wikidata_id(kb_id)

        for w_idx, rel_list in enumerate(batch_relations):
            sent_start = window_indices[w_idx]
            w_text = window_texts[w_idx]

            for rel_dict in rel_list:
                head_dict = rel_dict.get("head", {})
                tail_dict = rel_dict.get("tail", {})
                raw_rel = rel_dict.get("relation", "")
                conf = float(rel_dict.get("score", 0.5))

                head_text = head_dict.get("text", "").strip()
                tail_text = tail_dict.get("text", "").strip()
                head_type = head_dict.get("type", "entity")
                tail_type = tail_dict.get("type", "entity")

                if not head_text or not tail_text or head_text.lower() == tail_text.lower():
                    continue

                # Map to canonical schema relation
                canonical_rel = PATTERN_TO_CANONICAL.get(raw_rel.lower(), raw_rel.lower().replace(" ", "_"))

                # Validate against ontology constraints
                is_valid, reason = validate_triple_en(
                    {"surface": head_text, "label": head_type},
                    canonical_rel,
                    {"surface": tail_text, "label": tail_type},
                    confidence=conf,
                    min_conf_override=self.threshold,
                )
                if not is_valid:
                    continue


                # Resolve Head and Tail with Shared Node Registry & Landmark Context
                # is_head=True for head (allows architectural meronym scoping)
                # is_head=False for tail (strictly preserves natural canonical surface)
                h_uri, h_canon, h_qid, h_src = registry.resolve(
                    head_text, context_kb_id=kb_id, ent_type=head_type, is_head=True
                )
                t_uri, t_canon, t_qid, t_src = registry.resolve(
                    tail_text, context_kb_id=kb_id, ent_type=tail_type, is_head=False
                )

                # Skip self-loops after URI resolution
                if h_uri == t_uri:
                    continue

                triple_record = {
                    "article_id":    article_id,
                    "kb_id":         kb_id,
                    "window_idx":    sent_start,
                    "evidence":      w_text,
                    "head_surface":  h_canon,
                    "head_uri":      h_uri,
                    "head_wikidata": h_qid,
                    "head_type":     head_type,
                    "raw_relation":  raw_rel,
                    "relation":      canonical_rel,
                    "tail_surface":  t_canon,
                    "tail_uri":      t_uri,
                    "tail_wikidata": t_qid,
                    "tail_type":     tail_type,
                    "confidence":    round(conf, 4),
                    "method":        "gliner_relex",
                }
                extracted_triples.append(triple_record)

                # If head was an anchored meronym, automatically emit (Landmark, has_part, head)
                # Tail surface MUST BE clean natural surface (e.g. "dragon motif"), NEVER prefixed!
                if h_src == "anchored_meronym" and h_uri != host_uri:
                    extracted_triples.append({
                        "article_id":    article_id,
                        "kb_id":         kb_id,
                        "window_idx":    sent_start,
                        "evidence":      w_text,
                        "head_surface":  host_surface,
                        "head_uri":      host_uri,
                        "head_wikidata": host_qid,
                        "head_type":     "built_heritage",
                        "raw_relation":  "has part",
                        "relation":      "has_part",
                        "tail_surface":  head_text.strip(),
                        "tail_uri":      h_uri,
                        "tail_wikidata": None,
                        "tail_type":     head_type,
                        "confidence":    0.95,
                        "method":        "landmark_meronym_anchor",
                    })

        return extracted_triples


# ══════════════════════════════════════════════════════════
# 4. LANDMARK-CENTRIC GRAPH CONNECTIVITY FILTER
# ══════════════════════════════════════════════════════════

def filter_landmark_centric_triples(
    triples: List[Dict[str, Any]],
    kb_id: str,
    registry: SharedNodeRegistry,
) -> List[Dict[str, Any]]:
    """Filter triples so the Knowledge Graph strictly revolves around the central Landmark.

    Tiers:
      1. Direct Landmark Facts: Head or Tail is in the Core Landmark Set
         (the landmark, its parent landmark, its child sub-sections, or anchored meronyms).
      2. Direct Context: Head or Tail is a direct 1-hop neighbor of the Core Landmark Set
         AND relation is an ontological attribute (dynasty, era, reign, province, city).
      3. Disconnected / irrelevant peripheral triples are pruned.
    """
    if not triples:
        return []

    # 1. Define Core Landmark Set
    core_uris: Set[str] = set()
    ctx_uri = registry.get_kb_uri(kb_id)
    if ctx_uri:
        core_uris.add(ctx_uri)

    parent_kb = registry.kb_id_to_parent_kb_id.get(kb_id)
    if parent_kb:
        p_uri = registry.get_kb_uri(parent_kb)
        if p_uri:
            core_uris.add(p_uri)

    for ch_kb in registry.kb_id_to_children_kb_ids.get(kb_id, []):
        ch_uri = registry.get_kb_uri(ch_kb)
        if ch_uri:
            core_uris.add(ch_uri)

    # Add all anchored meronyms (URIs with # belonging to this landmark or its parent)
    for t in triples:
        h_u = t.get("head_uri", "")
        t_u = t.get("tail_uri", "")
        if h_u.startswith(f"{ctx_uri}#"):
            core_uris.add(h_u)
        if t_u.startswith(f"{ctx_uri}#"):
            core_uris.add(t_u)
        if parent_kb:
            p_prefix = f"{registry.get_kb_uri(parent_kb)}#"
            if h_u.startswith(p_prefix):
                core_uris.add(h_u)
            if t_u.startswith(p_prefix):
                core_uris.add(t_u)

    # 2. Find Direct 1-hop Neighbors of Core Set
    direct_neighbors: Set[str] = set()
    for t in triples:
        h, rel, tail = t.get("head_uri", ""), t.get("relation", ""), t.get("tail_uri", "")
        if h in core_uris and tail not in core_uris:
            direct_neighbors.add(tail)
        elif tail in core_uris and h not in core_uris:
            direct_neighbors.add(h)

    # 3. Retain only Tier 1 & Tier 2 Triples
    filtered: List[Dict[str, Any]] = []
    seen_dedup: Set[Tuple[str, str, str, int]] = set()

    for t in triples:
        h = t.get("head_uri", "")
        rel = t.get("relation", "")
        tail = t.get("tail_uri", "")
        w_idx = t.get("window_idx", -1)

        if not h or not tail or h == tail:
            continue

        # Tier 1: Direct fact of Core Landmark Set
        is_tier1 = (h in core_uris) or (tail in core_uris)

        # Tier 2: Direct context of a core neighbor with high-value relation
        is_tier2 = (
            (h in direct_neighbors or tail in direct_neighbors)
            and rel in HIGH_VALUE_CONTEXT_RELS
        )

        if is_tier1 or is_tier2:
            key = (h, rel, tail, w_idx)
            if key not in seen_dedup:
                seen_dedup.add(key)
                filtered.append(t)

    return filtered


# ══════════════════════════════════════════════════════════
# 5. PER-DOCUMENT & PER-KB GRAPH EXPORTER
# ══════════════════════════════════════════════════════════

def export_document_ttl(
    triples: List[Dict[str, Any]],
    doc_id: str,
    kb_id: str,
    output_path: Path,
) -> None:
    """Export single document triples to RDF Turtle format."""
    if not HAS_RDFLIB:
        return

    g = Graph()
    ONTO = Namespace("http://transmtl.vn/onto/")
    FKG = Namespace("http://transmtl.vn/fkg/")
    g.bind("onto", ONTO)
    g.bind("fkg", FKG)

    for t in triples:
        h_ref = URIRef(t["head_uri"])
        t_ref = URIRef(t["tail_uri"])
        r_name = re.sub(r"[^\w\-]", "_", t["relation"].strip()) or "relation"
        r_ref = ONTO[r_name]

        # Direct triple
        g.add((h_ref, r_ref, t_ref))

        # Reified statement with confidence and document provenance
        stmt = BNode()
        g.add((stmt, RDF.type, FKG.DocumentStatement))
        g.add((stmt, RDF.subject, h_ref))
        g.add((stmt, RDF.predicate, r_ref))
        g.add((stmt, RDF.object, t_ref))
        g.add((stmt, FKG.confidence, Literal(t["confidence"], datatype=XSD.decimal)))
        g.add((stmt, FKG.docId, Literal(doc_id)))
        g.add((stmt, FKG.kbId, Literal(kb_id)))

    g.serialize(destination=str(output_path), format="turtle")


def process_all_kbs(
    kb_export_path: str,
    output_base_dir: str,
    aliases_path: Optional[str] = None,
    device: str = "cuda",
    limit_kbs: Optional[int] = None,
    resume: bool = True,
    target_kb_ids: Optional[Set[str]] = None,
    gliner_threshold: float = 0.25,
) -> Dict[str, Any]:
    """Master pipeline:
    1. Builds SharedNodeRegistry with hierarchy, aliases & anaphora resolution
    2. Processes each document per KB (cleaning Wikipedia links, extracting & filtering triples)
    3. Fuses KB-level FKG using Module 5 (Dempster + t-conorm)
    4. Writes global index
    """
    out_dir = Path(output_base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Build Shared Node Registry ──────────────────
    cache_path = str(_this_dir / "wikidata_cache_en.json")
    registry = SharedNodeRegistry(cache_path=cache_path)
    registry.build_from_kb_export(kb_export_path, aliases_path)

    registry_json = out_dir / "shared_node_registry.json"
    registry_pkl = out_dir / "shared_node_registry.pkl"
    registry.save(registry_json, registry_pkl)

    # ── Step 2: Initialize Extractor & Builder ───────────────
    extractor = DocumentGraphExtractor(device=device, threshold=gliner_threshold)

    # ── Step 3: Iterate over KBs ────────────────────────────
    with open(kb_export_path, "r", encoding="utf-8") as f:
        export_data = json.load(f)

    nodes_flat = export_data.get("nodes_flat", [])
    if target_kb_ids:
        nodes_flat = [n for n in nodes_flat if n.get("kb_id") in target_kb_ids]
    elif limit_kbs:
        nodes_flat = nodes_flat[:limit_kbs]

    LOGGER.info("Starting processing for %d KBs on %s...", len(nodes_flat), device)

    kb_index: Dict[str, Any] = {
        "export_info": export_data.get("export_info", {}),
        "total_kbs": len(nodes_flat),
        "kbs": {},
    }

    total_docs_processed = 0
    total_triples_extracted = 0

    t_start = time.time()

    for idx, node in enumerate(nodes_flat, 1):
        kb_id = node.get("kb_id", f"kb_{idx}")
        name = node.get("name", kb_id)
        docs = node.get("documents", [])

        kb_folder = out_dir / kb_id
        kb_docs_folder = kb_folder / "docs"
        kb_fkg_facts_path = kb_folder / "fkg_facts.json"

        # Check resume
        if resume and kb_fkg_facts_path.exists():
            LOGGER.info("[%d/%d] Skipping already processed KB: %s", idx, len(nodes_flat), kb_id)
            continue

        LOGGER.info(
            "[%d/%d] Processing KB: %s ('%s') — %d documents",
            idx, len(nodes_flat), kb_id, name, len(docs),
        )

        kb_docs_folder.mkdir(parents=True, exist_ok=True)

        kb_triples: List[Dict[str, Any]] = []
        kb_doc_meta: List[Dict[str, Any]] = []

        # Process each document
        for d_idx, doc in enumerate(docs, 1):
            doc_id = doc.get("doc_id", f"{kb_id}_doc{d_idx}")
            raw_content = doc.get("content", "")
            source = doc.get("source", "unknown")
            url = doc.get("url", "")

            # 1. Clean Wikipedia / web content
            clean_text, meta = clean_wikipedia_document(raw_content)

            # Save clean text
            clean_txt_path = kb_docs_folder / f"{doc_id}_clean.txt"
            clean_txt_path.write_text(clean_text, encoding="utf-8")

            # 2. Extract Triples with Meronym Binding
            raw_triples = extractor.extract_document_triples(
                cleaned_text=clean_text,
                article_id=doc_id,
                kb_id=kb_id,
                registry=registry,
            )

            # 3. Apply Landmark-Centric Filtering
            triples = filter_landmark_centric_triples(
                triples=raw_triples,
                kb_id=kb_id,
                registry=registry,
            )

            # Save document triples
            doc_triples_path = kb_docs_folder / f"{doc_id}_triples.json"
            doc_triples_path.write_text(
                json.dumps(triples, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # Save document Turtle graph
            doc_ttl_path = kb_docs_folder / f"{doc_id}_graph.ttl"
            export_document_ttl(triples, doc_id, kb_id, doc_ttl_path)

            kb_triples.extend(triples)
            total_docs_processed += 1
            total_triples_extracted += len(triples)

            kb_doc_meta.append({
                "doc_id": doc_id,
                "source": source,
                "url": url,
                "n_triples": len(triples),
            })

        # Save KB metadata
        kb_meta_path = kb_folder / "kb_meta.json"
        kb_meta_path.write_text(
            json.dumps({
                "kb_id": kb_id,
                "name": name,
                "name_vi": node.get("name_vi", ""),
                "type": node.get("type", "object"),
                "n_docs": len(docs),
                "n_triples": len(kb_triples),
                "documents": kb_doc_meta,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 4. Fuse KB-level FKG using Module 5 engine
        if kb_triples:
            fkg_builder = FKGBuilder(
                conflict_K_threshold=0.50,
                copy_sim_threshold=0.85,
            )
            triple_files = [str(p) for p in kb_docs_folder.glob("*_triples.json")]
            fkg_builder.build(
                triple_files=triple_files,
                linked_files=[],
                registry_path=str(registry_pkl),
                output_dir=str(kb_folder),
            )

            LOGGER.info(
                "  ✓ KB %s fused: %d SV groups, %d MV facts, %d nodes",
                kb_id, len(fkg_builder.fact_groups),
                len(fkg_builder.fuzzy_facts), len(fkg_builder.entity_index),
            )
        else:
            LOGGER.info("  Notice: KB %s has 0 extracted triples", kb_id)

        # Record in global index
        kb_index["kbs"][kb_id] = {
            "name": name,
            "name_vi": node.get("name_vi", ""),
            "n_docs": len(docs),
            "n_triples": len(kb_triples),
            "kb_folder": str(kb_folder),
            "fkg_facts": str(kb_fkg_facts_path) if kb_fkg_facts_path.exists() else None,
        }

    # Save Master Global Index
    index_path = out_dir / "kb_index.json"
    kb_index["total_docs_processed"] = total_docs_processed
    kb_index["total_triples_extracted"] = total_triples_extracted
    kb_index["elapsed_seconds"] = round(time.time() - t_start, 2)

    index_path.write_text(
        json.dumps(kb_index, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    LOGGER.info(
        "╔════════════════════════════════════════════════════════════════════╗\n"
        "║ ALL KBs PROCESSED SUCCESSFULLY!                                   ║\n"
        "║ Total KBs: %-4d | Total Docs: %-4d | Total Triples: %-6d        ║\n"
        "║ Output Dir: %-52s  ║\n"
        "╚════════════════════════════════════════════════════════════════════╝",
        len(nodes_flat), total_docs_processed, total_triples_extracted, str(out_dir)
    )

    return kb_index


# ══════════════════════════════════════════════════════════
# 6. CLI RUNNER
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build Landmark-Centric FKG per KB-ID from multi-document sources")
    parser.add_argument(
        "--kb_export",
        default="data/sources/kb_full_export_2026-10-01.json",
        help="Path to full KB export JSON file",
    )
    parser.add_argument(
        "--output_dir",
        default="data/fkg_per_kb",
        help="Directory to save per-KB FKG outputs",
    )
    parser.add_argument(
        "--aliases",
        default="OntoKG_EN/aliases_en.json",
        help="Path to external aliases JSON",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run neural models on ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--limit_kbs",
        type=int,
        default=None,
        help="Limit number of KBs to process (for debugging/testing)",
    )
    parser.add_argument(
        "--kb_ids",
        default=None,
        help="Comma-separated list of specific kb_ids to process (e.g. 'nhap_dao_section,temple_of_literature')",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not skip already processed KBs",
    )
    parser.add_argument(
        "--gliner_threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for GLiNER relation extraction (default: 0.25)",
    )

    args = parser.parse_args()

    target_kb_ids = None
    if args.kb_ids:
        target_kb_ids = {k.strip() for k in args.kb_ids.split(",") if k.strip()}

    process_all_kbs(
        kb_export_path=args.kb_export,
        output_base_dir=args.output_dir,
        aliases_path=args.aliases,
        device=args.device,
        limit_kbs=args.limit_kbs,
        resume=not args.no_resume,
        target_kb_ids=target_kb_ids,
        gliner_threshold=args.gliner_threshold,
    )


if __name__ == "__main__":
    main()

