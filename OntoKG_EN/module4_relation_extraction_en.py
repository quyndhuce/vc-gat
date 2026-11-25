# module4_relation_extraction_en.py
"""
OntoKG_EN - Module 4: English Relation Extraction with GLiNER-Relex & Comprehensive Ontology Validation.

Pipeline:
  Input : Module 3 output (test_linked_output.json / module3_entity_linked.jsonl)
          where entities have resolved URIs, Wikidata IDs, and fine-grained labels.
  Output: 1. module4_triples_en.jsonl (all extracted triples: head_uri, relation, tail_uri)
          2. module4_enriched_en.json (original record enriched with 'triples' field)

Core Innovations:
  1. Multi-Sentence Sliding Window:
     Combines 2-3 consecutive sentences per window to capture cross-sentence facts
     and long-range semantic dependencies using GLiNER's Transformer Self-Attention.
  2. Post-GLiNER Entity Linking & Deduplication:
     Maps predicted head/tail mentions to canonical Linked Data URIs (Module 3 Registry),
     automatically merging textual synonyms and eliminating self-loops (head_uri == tail_uri).
  3. Comprehensive Ontology Validation:
     Enforces strict Domain & Range constraints across 50+ cultural, historical, architectural,
     and geographical relations to prevent false positives and semantic hallucinations.
  4. Dual Extraction:
     A. Metadata Triples (W3C LOD rdf:type, belongsTo, hasEntity, hasConcept)
     B. Semantic Relation Triples (GLiNER-Relex + Ontology Validation)
"""

from __future__ import annotations

import html
import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import torch
    from gliner import GLiNER
except Exception:
    torch = None
    GLiNER = None

# ──────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────
LOGGER = logging.getLogger("module4_relation_extraction_en")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)

# ──────────────────────────────────────────────────────────
# W3C Linked Data Namespaces
# ──────────────────────────────────────────────────────────
ONTO_URI = "http://transmtl.vn/onto"
ARTICLE_URI = "http://transmtl.vn/article"
TOPIC_URI = "http://transmtl.vn/onto/topic"
RDF_TYPE = "rdf:type"

WHITESPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFC", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\xa0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def normalize_key(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    return WHITESPACE_RE.sub(" ", text).strip()


# ──────────────────────────────────────────────────────────
# Ontology Class Mapping
# ──────────────────────────────────────────────────────────
LABEL_TO_CLASS_URI: Dict[str, str] = {
    "PER": f"{ONTO_URI}/Person",
    "ORG": f"{ONTO_URI}/Organization",
    "LOC": f"{ONTO_URI}/Location",
    "TIME": f"{ONTO_URI}/TimeRef",
    "EVENT": f"{ONTO_URI}/Event",
    "HERITAGE": f"{ONTO_URI}/Heritage",
    "ARTIFACT": f"{ONTO_URI}/Artifact",
    "STYLE": f"{ONTO_URI}/Style",
    "PRODUCT": f"{ONTO_URI}/Product",
    "ATTR": f"{ONTO_URI}/Attribute",
    "MISC": f"{ONTO_URI}/Concept",
    "CONCEPT": f"{ONTO_URI}/Concept",
    "dynasty": f"{ONTO_URI}/Dynasty",
}


# ──────────────────────────────────────────────────────────
# Dataclass Triple
# ──────────────────────────────────────────────────────────
@dataclass
class Triple:
    head_uri: str
    relation: str
    tail_uri: str
    head_surface: str
    tail_surface: str
    confidence: float
    method: str  # "metadata" | "gliner_relex"
    article_id: str
    window_idx: int = -1
    raw_relation: str = ""
    evidence: str = ""
    head_span: Optional[List[int]] = None
    tail_span: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────
# Canonical Modular Domain & Range Sets
# ──────────────────────────────────────────────────────────
ARCHITECTURAL_ELEMENT_TYPES: Set[str] = {
    "roof", "column", "pillar", "window", "door", "ornament", "motif",
    "inscription", "facade", "courtyard", "structure", "relic", "statue",
    "architectural element", "gate", "wall", "pathway", "steps", "platform",
    "tower", "sanctuary", "lawn", "pond", "garden", "lake", "well", "belfry",
    "stele", "stelae", "altar", "corridor", "railing", "balcony", "dome",
    "arch", "forecourt", "flagpole", "building", "monument", "hall"
}

BUILT_HERITAGE_DOMAIN: Set[str] = {
    "LOC", "landmark", "ancient citadel or fortress", "citadel", "fortress",
    "temple or pagoda", "pagoda", "temple", "lake", "mountain", "communal house or shrine",
    "church or cathedral", "cathedral", "church", "mausoleum or tomb", "monument or memorial",
    "monument", "museum", "palace or hall", "palace", "square or plaza", "park or garden",
    "craft village", "craft_village", "bridge", "tower", "gate", "architectural element",
    "building", "statue", "relic", "artwork or artifact", "structure", "historical monument",
    "built_heritage", "object", "entity"
} | ARCHITECTURAL_ELEMENT_TYPES

SPATIAL_ADMIN_DOMAIN: Set[str] = {
    "LOC", "country", "province", "city", "district", "ward",
    "street address", "street name", "historical place name", "historical state or colony",
    "historical administrative unit", "location", "gpe"
}

PERSON_FIGURE_DOMAIN: Set[str] = {
    "PER", "person", "historical figure", "religious figure", "monarch", "dynasty",
    "ethnic group", "people"
}

TEMPORAL_DOMAIN: Set[str] = {
    "TIME", "year", "date", "century", "historical period", "date range",
    "lunar date", "era name", "month", "time period", "relative time"
}

EVENT_DOMAIN: Set[str] = {
    "EVENT", "historical event", "traditional festival", "named conference or exhibition",
    "named cultural event", "war or battle", "ceremony or ritual", "festival or celebration",
    "festival or event", "event"
}

HERITAGE_DOMAIN: Set[str] = {
    "HERITAGE", "heritage designation", "law or legal document", "heritage"
}

ARTIFACT_DOMAIN: Set[str] = {
    "ARTIFACT", "artwork or artifact", "statue", "artwork", "artifact", "relic"
}

STYLE_DOMAIN: Set[str] = {
    "STYLE", "architectural style", "building material", "style"
}

PRODUCT_DOMAIN: Set[str] = {
    "PRODUCT", "dish or food", "traditional craft or handicraft", "craft or product", "product"
}

CONCEPT_DOMAIN: Set[str] = {
    "CONCEPT", "MISC", "religion or belief", "art form", "plant or species", "tree"
}

ATTR_DOMAIN: Set[str] = {
    "ATTR", "color", "measurement", "title or rank", "official position"
}

ORG_DOMAIN: Set[str] = {
    "ORG", "organization", "government or regime", "military", "facility"
}

VISUAL_OBJECT_DOMAIN: Set[str] = (
    BUILT_HERITAGE_DOMAIN
    | SPATIAL_ADMIN_DOMAIN
    | PERSON_FIGURE_DOMAIN
    | ARTIFACT_DOMAIN
    | CONCEPT_DOMAIN
    | {
        "bridge", "temple", "temple or pagoda", "lake", "water", "tree",
        "railing", "person", "gate", "roof", "building", "mountain", "river",
        "statue", "artwork or artifact", "LOC", "PER", "CONCEPT", "Flora",
        "plant or species", "structure", "landmark", "wall", "courtyard",
        "pathway", "steps", "platform", "tower", "sanctuary", "palaces",
        "temples", "shrines", "mountains", "ricefields"
    }
)


# ──────────────────────────────────────────────────────────
# Comprehensive Ontology Validation Schema (50+ Relations)
# ──────────────────────────────────────────────────────────
ONTOLOGY_SCHEMA: Dict[str, Dict[str, Any]] = {
    # ── Group 1: Spatial & Geography ──
    "located_in": {
        "patterns": ["located in", "situated in", "lies in", "found in", "placed in"],
        "domain": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN | ORG_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "range": SPATIAL_ADMIN_DOMAIN | BUILT_HERITAGE_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "min_conf": 0.35,
    },
    "part_of": {
        "patterns": ["part of", "component of", "within the complex of", "entrance to", "section of"],
        "domain": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN | ORG_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "range": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN | ORG_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "min_conf": 0.30,
    },
    "has_part": {
        "patterns": ["has part", "consists of", "includes", "comprises", "has section", "contains section"],
        "domain": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN | ORG_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "range": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN | ORG_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "min_conf": 0.30,
    },
    "situated_on": {
        "patterns": ["situated on", "stands on", "located on", "perched on", "built on"],
        "domain": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "range": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN | VISUAL_OBJECT_DOMAIN | {"LOC", "island", "mountain", "hill", "lake", "river", "bank", "coast", "gate", "landmark"},
        "min_conf": 0.30,
    },
    "near": {
        "patterns": ["near", "adjacent to", "close to", "neighboring", "next to", "nearby"],
        "domain": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "range": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN | VISUAL_OBJECT_DOMAIN | {"LOC", "river", "lake", "mountain"},
        "min_conf": 0.30,
    },
    "surrounded_by": {
        "patterns": ["surrounded by", "encircled by", "bordered by", "protected by"],
        "domain": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "range": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN | VISUAL_OBJECT_DOMAIN | {"river", "lake", "canal", "wall", "mountain", "trench", "garden", "LOC"},
        "min_conf": 0.30,
    },
    "crosses": {
        "patterns": ["crosses", "spans", "crossing"],
        "domain": BUILT_HERITAGE_DOMAIN | {"bridge", "LOC"},
        "range": SPATIAL_ADMIN_DOMAIN | {"river", "lake", "canal", "bay", "sea", "LOC"},
        "min_conf": 0.35,
    },
    "connects": {
        "patterns": ["connects", "links", "connecting", "leads to", "leading to"],
        "domain": BUILT_HERITAGE_DOMAIN | {"bridge", "gate", "LOC"},
        "range": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN,
        "min_conf": 0.35,
    },
    "capital_of": {
        "patterns": ["capital of", "administrative capital of"],
        "domain": {"city", "historical place name", "province"},
        "range": {"country", "historical state or colony", "dynasty"},
        "min_conf": 0.35,
        "single_valued": True,
    },
    "border_with": {
        "patterns": ["border with", "borders", "borders with"],
        "domain": SPATIAL_ADMIN_DOMAIN,
        "range": SPATIAL_ADMIN_DOMAIN,
        "min_conf": 0.40,
    },
    "faces": {
        "patterns": ["faces", "facing", "fronts", "fronting"],
        "domain": BUILT_HERITAGE_DOMAIN,
        "range": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN,
        "min_conf": 0.35,
    },
    "overlooks": {
        "patterns": ["overlooks", "overlooking"],
        "domain": BUILT_HERITAGE_DOMAIN,
        "range": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN,
        "min_conf": 0.35,
    },

    # ── Group 2: History, Chronology & Construction ──
    "built_in": {
        "patterns": ["built in", "constructed in", "erected in"],
        "domain": BUILT_HERITAGE_DOMAIN,
        "range": TEMPORAL_DOMAIN,
        "min_conf": 0.40,
        "single_valued": True,
    },
    "founded_in": {
        "patterns": ["founded in", "established in"],
        "domain": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN | ORG_DOMAIN,
        "range": TEMPORAL_DOMAIN,
        "min_conf": 0.35,
        "single_valued": True,
    },
    "inaugurated_in": {
        "patterns": ["inaugurated in", "opened in", "formally inaugurated on"],
        "domain": BUILT_HERITAGE_DOMAIN | ORG_DOMAIN,
        "range": TEMPORAL_DOMAIN,
        "min_conf": 0.35,
        "single_valued": True,
    },
    "renovated_in": {
        "patterns": ["renovated in", "restored in", "repaired in"],
        "domain": BUILT_HERITAGE_DOMAIN,
        "range": TEMPORAL_DOMAIN | EVENT_DOMAIN,
        "min_conf": 0.35,
    },
    "reconstructed_in": {
        "patterns": ["reconstructed in", "rebuilt in"],
        "domain": BUILT_HERITAGE_DOMAIN,
        "range": TEMPORAL_DOMAIN | EVENT_DOMAIN,
        "min_conf": 0.35,
    },
    "destroyed_in": {
        "patterns": ["destroyed in", "damaged in", "burnt down in"],
        "domain": BUILT_HERITAGE_DOMAIN,
        "range": TEMPORAL_DOMAIN | EVENT_DOMAIN,
        "min_conf": 0.35,
    },
    "demolished_in": {
        "patterns": ["demolished in", "torn down in"],
        "domain": BUILT_HERITAGE_DOMAIN,
        "range": TEMPORAL_DOMAIN | EVENT_DOMAIN,
        "min_conf": 0.35,
    },
    "dated_to": {
        "patterns": [
            "dated to", "dating to", "dates back to", "dates to", "dates from",
            "dating from", "originating in", "originated in", "crafted in"
        ],
        "domain": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN | CONCEPT_DOMAIN | ATTR_DOMAIN,
        "range": TEMPORAL_DOMAIN,
        "min_conf": 0.30,
    },
    "belongs_to_dynasty": {
        "patterns": ["belongs to dynasty", "under dynasty", "dating back to the dynasty of"],
        "domain": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN | PERSON_FIGURE_DOMAIN | EVENT_DOMAIN | ARTIFACT_DOMAIN,
        "range": {"dynasty"} | PERSON_FIGURE_DOMAIN,
        "min_conf": 0.35,
    },
    "during_era": {
        "patterns": ["during the era of", "in the era of", "during the reign of", "in the year of", "dating back to the era of"],
        "domain": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN | PERSON_FIGURE_DOMAIN | EVENT_DOMAIN | ARTIFACT_DOMAIN,
        "range": TEMPORAL_DOMAIN,
        "min_conf": 0.35,
    },
    "former_name_of": {
        "patterns": ["former name of", "previous name of", "ancient name of", "earlier name of", "historical name of"],
        "domain": SPATIAL_ADMIN_DOMAIN | BUILT_HERITAGE_DOMAIN,
        "range": SPATIAL_ADMIN_DOMAIN | BUILT_HERITAGE_DOMAIN,
        "min_conf": 0.35,
    },
    "renamed_to": {
        "patterns": ["renamed to", "changed name to"],
        "domain": SPATIAL_ADMIN_DOMAIN | BUILT_HERITAGE_DOMAIN,
        "range": SPATIAL_ADMIN_DOMAIN | BUILT_HERITAGE_DOMAIN,
        "min_conf": 0.35,
    },
    "ruled_by": {
        "patterns": ["ruled by", "controlled by", "governed by"],
        "domain": SPATIAL_ADMIN_DOMAIN,
        "range": PERSON_FIGURE_DOMAIN | ORG_DOMAIN,
        "min_conf": 0.35,
    },
    "ceded_to": {
        "patterns": ["ceded to", "surrendered to"],
        "domain": SPATIAL_ADMIN_DOMAIN,
        "range": SPATIAL_ADMIN_DOMAIN | ORG_DOMAIN | PERSON_FIGURE_DOMAIN,
        "min_conf": 0.35,
    },

    # ── Group 3: Agency, People, Worship & Authorship ──
    "built_by": {
        "patterns": ["built by", "erected by", "constructed by"],
        "domain": BUILT_HERITAGE_DOMAIN,
        "range": PERSON_FIGURE_DOMAIN | ORG_DOMAIN,
        "min_conf": 0.35,
    },
    "founded_by": {
        "patterns": ["founded by", "established by"],
        "domain": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN | ORG_DOMAIN,
        "range": PERSON_FIGURE_DOMAIN | ORG_DOMAIN,
        "min_conf": 0.35,
    },
    "designed_by": {
        "patterns": ["designed by", "architect of", "engineered by"],
        "domain": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN,
        "range": PERSON_FIGURE_DOMAIN | ORG_DOMAIN | {"architect", "engineer"},
        "min_conf": 0.35,
    },
    "commissioned_by": {
        "patterns": ["commissioned by", "ordered by"],
        "domain": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN,
        "range": PERSON_FIGURE_DOMAIN | ORG_DOMAIN,
        "min_conf": 0.35,
    },
    "reconstructed_by": {
        "patterns": ["reconstructed by", "remodeled by", "rebuilt by"],
        "domain": BUILT_HERITAGE_DOMAIN,
        "range": PERSON_FIGURE_DOMAIN | ORG_DOMAIN,
        "min_conf": 0.35,
    },
    "demolished_by": {
        "patterns": ["demolished by", "torn down by"],
        "domain": BUILT_HERITAGE_DOMAIN,
        "range": PERSON_FIGURE_DOMAIN | ORG_DOMAIN,
        "min_conf": 0.35,
    },
    "named_after": {
        "patterns": ["named after", "named in honor of"],
        "domain": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN,
        "range": PERSON_FIGURE_DOMAIN | EVENT_DOMAIN | BUILT_HERITAGE_DOMAIN,
        "min_conf": 0.35,
    },
    "dedicated_to": {
        "patterns": [
            "dedicated to", "dedicated in honor of", "built in memory of",
            "built in honor of", "devoted to", "erected in honor of", "consecrated to"
        ],
        "domain": BUILT_HERITAGE_DOMAIN | EVENT_DOMAIN,
        "range": PERSON_FIGURE_DOMAIN | EVENT_DOMAIN | {"deity", "god or goddess"},
        "min_conf": 0.35,
    },
    "worships": {
        "patterns": [
            "worships", "worshiped in", "worshipped in", "worship", "reveres",
            "venerates", "pays homage to", "dedicated to the worship of"
        ],
        "domain": BUILT_HERITAGE_DOMAIN | PERSON_FIGURE_DOMAIN | {"ethnic group"},
        "range": PERSON_FIGURE_DOMAIN | {"deity", "god or goddess"},
        "min_conf": 0.30,
    },
    "resting_place_of": {
        "patterns": ["resting place of", "serves as the resting place of", "burial place of", "tomb of", "mausoleum of"],
        "domain": BUILT_HERITAGE_DOMAIN | {"mausoleum or tomb", "monument or memorial"},
        "range": PERSON_FIGURE_DOMAIN,
        "min_conf": 0.35,
    },
    "commemorates": {
        "patterns": ["commemorates", "honors the memory of"],
        "domain": BUILT_HERITAGE_DOMAIN | EVENT_DOMAIN | ARTIFACT_DOMAIN,
        "range": PERSON_FIGURE_DOMAIN | EVENT_DOMAIN,
        "min_conf": 0.35,
    },
    "gifted_by": {
        "patterns": [
            "gifted by", "presented by", "donated by", "gift from", "given by",
            "bestowed by", "conferred by", "offered by"
        ],
        "domain": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN | CONCEPT_DOMAIN,
        "range": PERSON_FIGURE_DOMAIN | SPATIAL_ADMIN_DOMAIN | ORG_DOMAIN,
        "min_conf": 0.30,
    },
    "gifted_in": {
        "patterns": ["gifted in", "presented in", "donated in", "bestowed in"],
        "domain": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN | CONCEPT_DOMAIN,
        "range": TEMPORAL_DOMAIN,
        "min_conf": 0.30,
        "single_valued": True,
    },
    "defeated": {
        "patterns": ["defeated", "triumphed over", "vanquished"],
        "domain": PERSON_FIGURE_DOMAIN | ORG_DOMAIN,
        "range": PERSON_FIGURE_DOMAIN | ORG_DOMAIN | SPATIAL_ADMIN_DOMAIN,
        "min_conf": 0.35,
    },
    "defeated_in": {
        "patterns": ["defeated in"],
        "domain": PERSON_FIGURE_DOMAIN | ORG_DOMAIN,
        "range": TEMPORAL_DOMAIN | EVENT_DOMAIN,
        "min_conf": 0.35,
    },
    "used_by": {
        "patterns": ["used by", "occupied by"],
        "domain": BUILT_HERITAGE_DOMAIN,
        "range": PERSON_FIGURE_DOMAIN | ORG_DOMAIN,
        "min_conf": 0.35,
    },
    "imprisoned_in": {
        "patterns": ["imprisoned in", "held in"],
        "domain": PERSON_FIGURE_DOMAIN,
        "range": BUILT_HERITAGE_DOMAIN,
        "min_conf": 0.35,
    },
    "captured_during": {
        "patterns": ["captured during", "captured in"],
        "domain": PERSON_FIGURE_DOMAIN,
        "range": EVENT_DOMAIN,
        "min_conf": 0.35,
    },
    "successor_of": {
        "patterns": ["successor of", "succeeded by"],
        "domain": PERSON_FIGURE_DOMAIN,
        "range": PERSON_FIGURE_DOMAIN,
        "min_conf": 0.35,
    },
    "held_title": {
        "patterns": ["held title", "titled as", "conferred the title of", "honored as"],
        "domain": PERSON_FIGURE_DOMAIN,
        "range": ATTR_DOMAIN | CONCEPT_DOMAIN,
        "min_conf": 0.35,
    },
    "held_position": {
        "patterns": ["held position", "served as", "appointed as", "position of"],
        "domain": PERSON_FIGURE_DOMAIN,
        "range": ATTR_DOMAIN | CONCEPT_DOMAIN,
        "min_conf": 0.35,
    },

    # ── Group 4: Architecture, Materials & Physical Properties ──
    "architectural_style": {
        "patterns": ["architectural style", "style of", "built in the style of"],
        "domain": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN,
        "range": STYLE_DOMAIN | CONCEPT_DOMAIN,
        "min_conf": 0.35,
    },
    "followed_plan_of": {
        "patterns": ["followed plan of", "modeled after the plan of"],
        "domain": BUILT_HERITAGE_DOMAIN,
        "range": BUILT_HERITAGE_DOMAIN,
        "min_conf": 0.35,
    },
    "made_of": {
        "patterns": ["made of", "built with", "constructed with"],
        "domain": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN,
        "range": STYLE_DOMAIN | CONCEPT_DOMAIN,
        "min_conf": 0.35,
    },
    "has_color": {
        "patterns": ["has color", "painted in", "painted color"],
        "domain": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN | STYLE_DOMAIN,
        "range": ATTR_DOMAIN,
        "min_conf": 0.35,
    },
    "has_height": {
        "patterns": ["has height", "height of", "tall", "high"],
        "domain": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN,
        "range": ATTR_DOMAIN,
        "min_conf": 0.35,
        "single_valued": True,
    },
    "has_width": {
        "patterns": ["has width", "width of", "wide"],
        "domain": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN,
        "range": ATTR_DOMAIN,
        "min_conf": 0.35,
        "single_valued": True,
    },
    "has_length": {
        "patterns": ["has length", "length of", "long"],
        "domain": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN,
        "range": ATTR_DOMAIN,
        "min_conf": 0.35,
        "single_valued": True,
    },
    "has_area": {
        "patterns": ["has area", "covering an area of", "area of"],
        "domain": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN,
        "range": ATTR_DOMAIN,
        "min_conf": 0.35,
        "single_valued": True,
    },
    "inspired_by": {
        "patterns": ["inspired by", "modeled after", "influenced by"],
        "domain": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN | STYLE_DOMAIN,
        "range": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN | STYLE_DOMAIN,
        "min_conf": 0.35,
    },
    "houses": {
        "patterns": ["houses", "preserves", "displays", "contains"],
        "domain": BUILT_HERITAGE_DOMAIN,
        "range": ARTIFACT_DOMAIN | CONCEPT_DOMAIN,
        "min_conf": 0.35,
    },
    "used_as": {
        "patterns": ["used as", "served as"],
        "domain": BUILT_HERITAGE_DOMAIN,
        "range": CONCEPT_DOMAIN | BUILT_HERITAGE_DOMAIN,
        "min_conf": 0.35,
    },

    # ── Group 5: Heritage, Culture, Symbolism & Events ──
    "recognized_as": {
        "patterns": ["recognized as", "inscribed as", "designated as", "registered as"],
        "domain": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN | EVENT_DOMAIN | ORG_DOMAIN,
        "range": HERITAGE_DOMAIN | CONCEPT_DOMAIN,
        "min_conf": 0.35,
    },
    "recognized_in": {
        "patterns": ["recognized in", "designated in", "registered in"],
        "domain": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN | EVENT_DOMAIN | HERITAGE_DOMAIN | ORG_DOMAIN,
        "range": TEMPORAL_DOMAIN,
        "min_conf": 0.35,
        "single_valued": True,
    },
    "symbol_of": {
        "patterns": ["symbol of", "emblem of"],
        "domain": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN,
        "range": SPATIAL_ADMIN_DOMAIN | BUILT_HERITAGE_DOMAIN,
        "min_conf": 0.35,
    },
    "known_for": {
        "patterns": ["known for", "renowned for", "famous for"],
        "domain": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN,
        "range": PRODUCT_DOMAIN | CONCEPT_DOMAIN | HERITAGE_DOMAIN,
        "min_conf": 0.30,
    },
    "hosts_event": {
        "patterns": ["hosts event", "venue for"],
        "domain": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN,
        "range": EVENT_DOMAIN,
        "min_conf": 0.35,
    },
    "held_in": {
        "patterns": ["held in", "takes place in"],
        "domain": EVENT_DOMAIN,
        "range": TEMPORAL_DOMAIN | SPATIAL_ADMIN_DOMAIN | BUILT_HERITAGE_DOMAIN,
        "min_conf": 0.35,
    },
    "participated_in": {
        "patterns": ["participated in", "took part in", "fought in"],
        "domain": PERSON_FIGURE_DOMAIN | ORG_DOMAIN,
        "range": EVENT_DOMAIN,
        "min_conf": 0.35,
    },
    "famous_dish_of": {
        "patterns": ["famous dish of", "specialty of"],
        "domain": PRODUCT_DOMAIN,
        "range": SPATIAL_ADMIN_DOMAIN,
        "min_conf": 0.35,
    },
    "craft_of": {
        "patterns": ["craft of", "product of"],
        "domain": PRODUCT_DOMAIN,
        "range": SPATIAL_ADMIN_DOMAIN,
        "min_conf": 0.35,
    },
    "associated_with": {
        "patterns": ["associated with", "linked to"],
        "domain": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN | PERSON_FIGURE_DOMAIN | EVENT_DOMAIN | ARTIFACT_DOMAIN,
        "range": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN | PERSON_FIGURE_DOMAIN | EVENT_DOMAIN | ARTIFACT_DOMAIN,
        "min_conf": 0.25,
    },

    # ── Group 6: Visual & Spatial Scene Relations (Multimodal Scene Graph) ──
    "over": {
        "patterns": ["over", "spans over", "crossing over", "hangs over", "positioned over"],
        "domain": VISUAL_OBJECT_DOMAIN,
        "range": VISUAL_OBJECT_DOMAIN,
        "min_conf": 0.30,
    },
    "across": {
        "patterns": ["across", "spans across", "stretches across", "built across"],
        "domain": VISUAL_OBJECT_DOMAIN,
        "range": VISUAL_OBJECT_DOMAIN,
        "min_conf": 0.30,
    },
    "leading_to": {
        "patterns": ["leading to", "leads to", "leading towards", "leads towards"],
        "domain": VISUAL_OBJECT_DOMAIN,
        "range": VISUAL_OBJECT_DOMAIN,
        "min_conf": 0.30,
    },
    "on": {
        "patterns": ["on", "stands on", "located on", "resting on", "perched on", "situated on", "at the top of"],
        "domain": VISUAL_OBJECT_DOMAIN,
        "range": VISUAL_OBJECT_DOMAIN,
        "min_conf": 0.30,
    },
    "in": {
        "patterns": ["in", "inside", "within", "situated inside"],
        "domain": VISUAL_OBJECT_DOMAIN,
        "range": VISUAL_OBJECT_DOMAIN,
        "min_conf": 0.30,
    },
    "walking_on": {
        "patterns": ["walking on", "walks on", "strolling on", "crossing on foot"],
        "domain": PERSON_FIGURE_DOMAIN | {"person", "PER"},
        "range": VISUAL_OBJECT_DOMAIN,
        "min_conf": 0.30,
    },
    "with": {
        "patterns": ["with", "accompanied by", "having", "featuring"],
        "domain": VISUAL_OBJECT_DOMAIN,
        "range": VISUAL_OBJECT_DOMAIN,
        "min_conf": 0.30,
    },
    "above": {
        "patterns": ["above", "towering above", "elevated above", "higher than"],
        "domain": VISUAL_OBJECT_DOMAIN,
        "range": VISUAL_OBJECT_DOMAIN,
        "min_conf": 0.30,
    },

    # ── Group: Architectural Components & Morphology (29 Expanded Onto Relations) ──
    "has_component": {
        "patterns": ["has component", "has structural component", "features component", "equipped with", "has section", "consists of", "has architectural component"],
        "domain": BUILT_HERITAGE_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "range": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN | VISUAL_OBJECT_DOMAIN | {"column", "pillar", "roof", "gate", "door", "window", "stair", "steps", "balcony", "dome", "arch", "tower", "courtyard", "stele", "stelae", "altar", "facade", "forecourt", "lawn", "flagpole"},
        "min_conf": 0.30,
    },
    "has_roof_type": {
        "patterns": ["has roof type", "roof type", "roof style", "roofed with", "curved tiled roof", "tiled roof"],
        "domain": BUILT_HERITAGE_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "range": STYLE_DOMAIN | CONCEPT_DOMAIN | {"curved roof", "tiled roof", "gable roof", "hip roof", "yin-yang roof", "flat roof", "vaulted roof", "curved tiled roof"},
        "min_conf": 0.30,
    },
    "has_roof_shape": {
        "patterns": ["has roof shape", "roof shape", "curved roof", "multi-tiered roof", "pyramidal roof", "dome-shaped roof", "barrel-vaulted ceiling"],
        "domain": BUILT_HERITAGE_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "range": STYLE_DOMAIN | CONCEPT_DOMAIN | ATTR_DOMAIN | {"curved", "curved upwards", "tiered", "pyramidal", "dome", "flat", "barrel-vaulted"},
        "min_conf": 0.30,
    },
    "has_column_type": {
        "patterns": ["has column type", "column type", "column style", "supported by columns", "pillar type", "pillar style", "wrought-iron pillars"],
        "domain": BUILT_HERITAGE_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "range": STYLE_DOMAIN | CONCEPT_DOMAIN | {"stone column", "wooden pillar", "square stone columns", "round column", "iron pillar", "wrought-iron pillars", "pilasters"},
        "min_conf": 0.30,
    },
    "has_window_type": {
        "patterns": ["has window type", "window type", "window style", "arched window", "arched windows", "louvered window"],
        "domain": BUILT_HERITAGE_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "range": STYLE_DOMAIN | CONCEPT_DOMAIN | {"arched window", "arched windows", "louvered window", "circular window", "french window", "stained glass"},
        "min_conf": 0.30,
    },
    "has_door_type": {
        "patterns": ["has door type", "door type", "door style", "wooden door", "carved door", "iron gate", "folding door"],
        "domain": BUILT_HERITAGE_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "range": STYLE_DOMAIN | CONCEPT_DOMAIN | {"wooden door", "carved door", "iron gate", "folding door", "portal"},
        "min_conf": 0.30,
    },
    "has_ornament": {
        "patterns": ["has ornament", "ornamented with", "adorned with ornament", "embellished with", "features ornament", "ornamental relief"],
        "domain": BUILT_HERITAGE_DOMAIN | VISUAL_OBJECT_DOMAIN | ARTIFACT_DOMAIN,
        "range": STYLE_DOMAIN | CONCEPT_DOMAIN | ARTIFACT_DOMAIN | {"relief", "carving", "figurine", "plaque", "sculpture", "ceramic mosaic", "porcelain figurines", "wood carvings"},
        "min_conf": 0.30,
    },
    "depicts_motif": {
        "patterns": ["depicts motif", "depicting", "features motif of", "motif of", "carved with the motif of", "depicts"],
        "domain": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "range": CONCEPT_DOMAIN | ARTIFACT_DOMAIN | {"dragon", "phoenix", "lotus", "turtle", "cloud", "four sacred beasts", "sacred animals", "dragon motif"},
        "min_conf": 0.30,
    },
    "has_shape": {
        "patterns": ["has shape", "shaped as", "in the shape of", "form of", "cylindrical form", "rectangular form", "pyramidal form"],
        "domain": BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "range": ATTR_DOMAIN | CONCEPT_DOMAIN | {"cylindrical", "rectangular", "octagonal", "square", "pyramidal", "linear", "circular", "pyramid"},
        "min_conf": 0.30,
    },
    "has_number_of_levels": {
        "patterns": ["has number of levels", "number of floors", "stories", "tiered levels", "two stories", "three stories", "levels"],
        "domain": BUILT_HERITAGE_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "range": TEMPORAL_DOMAIN | ATTR_DOMAIN | {"1", "2", "3", "4", "5", "two-story", "three-tiered", "multi-story", "levels"},
        "min_conf": 0.30,
    },
    "has_number_of_columns": {
        "patterns": ["has number of columns", "number of pillars", "supported by single pillar", "supported by columns", "single pillar"],
        "domain": BUILT_HERITAGE_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "range": ATTR_DOMAIN | CONCEPT_DOMAIN | {"single pillar", "one column", "four columns", "eight columns", "columns"},
        "min_conf": 0.30,
    },

    # ── Group: Spatial Relations & Spatial Layout ──
    "located_inside": {
        "patterns": ["located inside", "situated inside", "stands inside", "inside", "within the interior of"],
        "domain": VISUAL_OBJECT_DOMAIN | BUILT_HERITAGE_DOMAIN | ARTIFACT_DOMAIN,
        "range": BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "min_conf": 0.30,
    },
    "located_in_front_of": {
        "patterns": ["located in front of", "in front of", "stands in front of", "forecourt of", "fronting"],
        "domain": VISUAL_OBJECT_DOMAIN | BUILT_HERITAGE_DOMAIN,
        "range": VISUAL_OBJECT_DOMAIN | BUILT_HERITAGE_DOMAIN,
        "min_conf": 0.30,
    },
    "located_behind": {
        "patterns": ["located behind", "behind", "stands behind", "at the rear of", "situated behind"],
        "domain": VISUAL_OBJECT_DOMAIN | BUILT_HERITAGE_DOMAIN,
        "range": VISUAL_OBJECT_DOMAIN | BUILT_HERITAGE_DOMAIN,
        "min_conf": 0.30,
    },
    "adjacent_to": {
        "patterns": ["adjacent to", "flanked by", "directly next to", "adjoining", "next to"],
        "domain": VISUAL_OBJECT_DOMAIN | BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN,
        "range": VISUAL_OBJECT_DOMAIN | BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN,
        "min_conf": 0.30,
    },
    "connected_to": {
        "patterns": ["connected to", "linked to", "joins with", "connected with", "bridges"],
        "domain": VISUAL_OBJECT_DOMAIN | BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN,
        "range": VISUAL_OBJECT_DOMAIN | BUILT_HERITAGE_DOMAIN | SPATIAL_ADMIN_DOMAIN,
        "min_conf": 0.30,
    },
    "below": {
        "patterns": ["below", "beneath", "underneath", "under", "situated below"],
        "domain": VISUAL_OBJECT_DOMAIN,
        "range": VISUAL_OBJECT_DOMAIN,
        "min_conf": 0.30,
    },
    "has_entrance": {
        "patterns": ["has entrance", "entrance via", "accessible through", "entered through", "entry at", "main entrance at"],
        "domain": BUILT_HERITAGE_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "range": BUILT_HERITAGE_DOMAIN | VISUAL_OBJECT_DOMAIN | {"gate", "door", "portico", "courtyard"},
        "min_conf": 0.30,
    },

    # ── Group: Materials, Surface Textures & Physical State ──
    "has_surface_texture": {
        "patterns": ["has surface texture", "surface texture", "rough texture", "smooth finish", "stone cladding", "weathered surface", "masonry finish"],
        "domain": BUILT_HERITAGE_DOMAIN | VISUAL_OBJECT_DOMAIN | ARTIFACT_DOMAIN,
        "range": ATTR_DOMAIN | CONCEPT_DOMAIN | {"stone cladding", "rough", "smooth", "weathered", "glazed", "lacquered", "polished", "masonry"},
        "min_conf": 0.30,
    },
    "has_pattern": {
        "patterns": ["has pattern", "patterned with", "decorative pattern", "geometric pattern", "floral pattern", "geometric bands"],
        "domain": BUILT_HERITAGE_DOMAIN | VISUAL_OBJECT_DOMAIN | ARTIFACT_DOMAIN,
        "range": STYLE_DOMAIN | CONCEPT_DOMAIN | {"geometric pattern", "floral pattern", "spiral pattern", "checkerboard", "tiled pattern", "geometric bands"},
        "min_conf": 0.30,
    },
    "has_inscription": {
        "patterns": ["has inscription", "inscribed with", "bearing inscription", "inscribed plaque", "stelae inscription", "calligraphy inscription", "inscribed with inventors"],
        "domain": BUILT_HERITAGE_DOMAIN | VISUAL_OBJECT_DOMAIN | ARTIFACT_DOMAIN,
        "range": CONCEPT_DOMAIN | ARTIFACT_DOMAIN | {"inscription", "stelae", "calligraphy", "plaque", "epitaph"},
        "min_conf": 0.30,
    },
    "has_decoration": {
        "patterns": ["has decoration", "decorated with", "adorned with", "garnished with", "embellished with"],
        "domain": BUILT_HERITAGE_DOMAIN | VISUAL_OBJECT_DOMAIN | ARTIFACT_DOMAIN,
        "range": STYLE_DOMAIN | CONCEPT_DOMAIN | ARTIFACT_DOMAIN | {"Shiwan porcelain", "ceramic relief", "wood carvings", "gilded lacquer", "frescoes", "decorations"},
        "min_conf": 0.30,
    },
    "has_physical_condition": {
        "patterns": ["has physical condition", "physical condition", "preserved condition", "ancient", "moss-covered", "dilapidated", "well-preserved", "weather-beaten"],
        "domain": BUILT_HERITAGE_DOMAIN | VISUAL_OBJECT_DOMAIN | ARTIFACT_DOMAIN,
        "range": ATTR_DOMAIN | CONCEPT_DOMAIN | {"well-preserved", "preserved", "ancient", "moss-covered", "weathered", "intact", "relic"},
        "min_conf": 0.30,
    },
    "has_restoration_feature": {
        "patterns": ["has restoration feature", "restoration feature", "restored in", "renovated with", "reconstructed feature", "renovations and expansions"],
        "domain": BUILT_HERITAGE_DOMAIN | VISUAL_OBJECT_DOMAIN,
        "range": TEMPORAL_DOMAIN | CONCEPT_DOMAIN | {"renovation", "reconstruction", "reinforcement", "restoration", "modern restoration"},
        "min_conf": 0.30,
    },
}

# ──────────────────────────────────────────────────────────
# Inverse, Symmetric & Alias Standardization Tables
# ──────────────────────────────────────────────────────────
INVERSE_RELATIONS: Dict[str, str] = {
    "has_component": "part_of",
    "part_of": "has_component",
    "has_part": "part_of",
    "located_inside": "houses",
    "houses": "located_inside",
    "located_in_front_of": "located_behind",
    "located_behind": "located_in_front_of",
    "above": "below",
    "below": "above",
}

SYMMETRIC_RELATIONS: Set[str] = {
    "adjacent_to",
    "connected_to",
    "border_with",
}

RELATION_ALIASES: Dict[str, str] = {
    "has_structural_component": "has_component",
    "spatial_adjacent_to": "adjacent_to",
    "inside": "located_inside",
    "next_to": "adjacent_to",
    "in_front_of": "located_in_front_of",
    "behind": "located_behind",
    "under": "below",
    "component_of": "part_of",
}

# Compile pattern-to-canonical mapping
PATTERN_TO_CANONICAL: Dict[str, str] = {}
ALL_GLINER_RELATION_LABELS: List[str] = []

for can_rel, info in ONTOLOGY_SCHEMA.items():
    for pat in info["patterns"]:
        PATTERN_TO_CANONICAL[pat.lower()] = can_rel
    primary_pat = info["patterns"][0]
    if primary_pat not in ALL_GLINER_RELATION_LABELS:
        ALL_GLINER_RELATION_LABELS.append(primary_pat)

for alias, can in RELATION_ALIASES.items():
    PATTERN_TO_CANONICAL[alias.lower()] = can
    PATTERN_TO_CANONICAL[alias.lower().replace("_", " ")] = can

# Add high-yield natural synonyms that differ substantially in lexical vocabulary
_EXTRA_GLINER_RELATIONS = [
    "worships", "worshiped in", "dedicated to", "dating to", "dated to",
    "gifted by", "presented by", "donated by", "contains",
    "during the reign of", "previous name of", "stands on", "honored as",
    "resting place of", "serves as the resting place of", "inspired by",
    "has width", "has height", "has length", "has area", "made of",
    "over", "across", "leading to", "on", "near", "in", "part of",
    "surrounded by", "walking on", "with", "above"
]
for p in _EXTRA_GLINER_RELATIONS:
    if p not in ALL_GLINER_RELATION_LABELS:
        ALL_GLINER_RELATION_LABELS.append(p)




# ──────────────────────────────────────────────────────────
# Ontology Validation Function
# ──────────────────────────────────────────────────────────
def validate_triple_en(
    head_ent: Dict[str, Any],
    canonical_relation: str,
    tail_ent: Dict[str, Any],
    confidence: float = 1.0,
    min_conf_override: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Validate (Head, Relation, Tail) against the Comprehensive Ontology Schema.
    Returns: (is_valid: bool, reason: str)
    """
    canonical_relation = RELATION_ALIASES.get(canonical_relation, canonical_relation)
    schema = ONTOLOGY_SCHEMA.get(canonical_relation)
    if not schema:
        return False, f"Unknown relation: {canonical_relation}"

    # Enforce per-relation minimum confidence from schema (or override if provided)
    schema_min = min_conf_override if min_conf_override is not None else schema.get("min_conf", 0.35)
    if confidence < schema_min:
        return False, f"Confidence {confidence:.3f} below schema min_conf {schema_min} for {canonical_relation}"

    head_uri = head_ent.get("uri", "")
    tail_uri = tail_ent.get("uri", "")

    # Anti-reflexivity: eliminate self-loops
    if head_uri and tail_uri and head_uri == tail_uri:
        if canonical_relation == "renamed_to":
            head_s = head_ent.get("surface", "").strip().lower()
            tail_s = tail_ent.get("surface", "").strip().lower()
            if head_s == tail_s:
                return False, f"Self-loop: identical surface name for renamed_to ({head_ent.get('surface')})"
        else:
            return False, f"Self-loop: head_uri == tail_uri ({head_uri})"

    # Collect all types for Head
    head_types: Set[str] = set()
    if head_ent.get("label"):
        head_types.add(head_ent["label"])
    if head_ent.get("fine_grained_label"):
        head_types.add(head_ent["fine_grained_label"])

    if head_types.intersection(ARCHITECTURAL_ELEMENT_TYPES):
        head_types.update({"architectural element", "structure", "built_heritage", "object", "entity"})

    # Collect all types for Tail
    tail_types: Set[str] = set()
    if tail_ent.get("label"):
        tail_types.add(tail_ent["label"])
    if tail_ent.get("fine_grained_label"):
        tail_types.add(tail_ent["fine_grained_label"])

    if tail_types.intersection(ARCHITECTURAL_ELEMENT_TYPES):
        tail_types.update({"architectural element", "structure", "built_heritage", "object", "entity"})

    # ── Explicit Semantic Rules ──
    head_s = head_ent.get("surface", "").strip().lower()
    tail_s = tail_ent.get("surface", "").strip().lower()

    # Rule 1: belongs_to_dynasty cannot be between 2 dynasties, and tail CANNOT be an era name
    if canonical_relation == "belongs_to_dynasty":
        if "dynasty" in head_types or "dynasty" in head_s:
            return False, f"Rule violation: Head '{head_ent.get('surface')}' is a dynasty; a dynasty cannot belong to another dynasty."
        if ("dynasty" in head_types or "dynasty" in head_s) and ("dynasty" in tail_types or "dynasty" in tail_s):
            return False, f"Rule violation: belongs_to_dynasty cannot be between two dynasties ({head_ent.get('surface')} -> {tail_ent.get('surface')})."
        if "era name" in tail_types or any(kw in tail_s for kw in {"niên hiệu", "thần vũ", "long thụy", "thiệu bảo"}):
            return False, f"Rule violation: Tail '{tail_ent.get('surface')}' is an era name, not a dynasty. Use during_era instead."
        if not ({"dynasty"}.intersection(tail_types) or "dynasty" in tail_s):
            return False, f"Rule violation: Tail '{tail_ent.get('surface')}' must be a dynasty for belongs_to_dynasty."

    # Rule 2: capital_of domain cannot be a physical structure / fortress / architectural element
    if canonical_relation == "capital_of":
        structure_types = {
            "ancient citadel or fortress", "tower", "gate", "bridge", "statue",
            "palace", "temple or pagoda", "architectural element", "museum",
            "square or plaza", "landmark", "mausoleum or tomb", "building", "structure"
        }
        if structure_types.intersection(head_types):
            return False, f"Rule violation: Structure '{head_ent.get('surface')}' cannot be capital_of (must be a city/province)."
        if structure_types.intersection(tail_types):
            return False, f"Rule violation: Tail '{tail_ent.get('surface')}' is a structure/museum, cannot be target of capital_of (must be a country/state)."

    # Rule 3: ceded_to head must be a territory or historical place: cannot be a dynasty or person
    if canonical_relation == "ceded_to":
        if "dynasty" in head_types or "person" in head_types or "dynasty" in head_s:
            return False, f"Rule violation: Head '{head_ent.get('surface')}' is a dynasty/person; only territories/cities can be ceded_to."

    # Rule 4: symbol_of tail must be an administrative region/city/country: cannot be an architectural structure
    if canonical_relation == "symbol_of":
        if "architectural element" in tail_types or any(t in tail_types for t in {"ancient citadel or fortress", "tower", "gate", "bridge", "statue"}):
            return False, f"Rule violation: Tail '{tail_ent.get('surface')}' is an architectural structure; symbol_of must point to a city/province/country."

    # Rule 5: ruled_by head must not be a Can Chi year, era name, or architectural element/structure/palace
    if canonical_relation == "ruled_by":
        forbidden_structures = {"architectural element", "ancient citadel or fortress", "palace", "temple or pagoda", "tower", "gate", "bridge", "statue", "museum"}
        if forbidden_structures.intersection(head_types):
            return False, f"Rule violation: Head '{head_ent.get('surface')}' is a physical structure/citadel/palace, cannot be ruled_by."
        if "era name" in head_types:
            return False, f"Rule violation: Head '{head_ent.get('surface')}' is an era name, cannot be ruled_by."
        _canchi_stems = {
            "giáp", "ất", "bính", "đinh", "mậu", "kỷ", "canh", "tân", "nhâm", "quý",
        }
        _canchi_branches = {
            "tý", "sửu", "dần", "mão", "thìn", "tỵ", "ngọ", "mùi",
            "thân", "dậu", "tuất", "hợi",
        }
        head_words = head_s.split()
        if len(head_words) == 2:
            w0, w1 = head_words
            if w0 in _canchi_stems and w1 in _canchi_branches:
                return False, (
                    f"Rule 5: Head '{head_ent.get('surface')}' is a Sexagenary cycle year name "
                    f"(Can Chi), not a country/territory. Cannot be ruled_by."
                )
        if len(head_s) <= 12 and not any(geo in head_types for geo in {
            "country", "historical state or colony", "city", "province", "government or regime",
        }):
            head_has_geo_keyword = any(kw in head_s for kw in {
                "vietnam", "việt", "china", "trung", "japan", "nhật", "laos", "lào",
                "cambodia", "miên", "france", "pháp", "indochina", "đông dương",
                "hanoi", "hà nội", "huế", "hue", "saigon", "sài gòn",
            })
            if not head_has_geo_keyword and head_types == {"LOC"}:
                return False, (
                    f"Rule 5: Head '{head_ent.get('surface')}' typed only as LOC with no "
                    f"country/city indicators — likely a misclassified era/year name."
                )

    # Rule 6: used_by head must be a specific building/fortress, not a broad historical place name
    if canonical_relation == "used_by":
        if "historical place name" in head_types and not any(t in head_types for t in {"ancient citadel or fortress", "landmark", "architectural element", "tower", "gate"}):
            return False, f"Rule violation: Head '{head_ent.get('surface')}' is a broad historical place name, not a specific building/fortress."

    # Rule 7: dedicated_to / worships tail must be a religious figure, deity, or venerated historical figure
    if canonical_relation == "dedicated_to":
        if any(t in tail_types for t in {"TIME", "year", "century", "city", "province", "country", "ancient citadel or fortress"}):
            return False, f"Rule violation: Tail '{tail_ent.get('surface')}' is a time or place; dedicated_to must point to a religious/historical figure, deity, or event."

    if canonical_relation == "worships":
        if any(t in tail_types for t in {"TIME", "year", "century", "city", "province", "country", "ancient citadel or fortress"}):
            return False, f"Rule violation: Tail '{tail_ent.get('surface')}' is a time or place; worships must point to a religious figure, deity, or ancestor."

    # Rule 8: dated_to tail must be a temporal entity
    if canonical_relation == "dated_to":
        valid_time_types = {"TIME", "year", "date", "century", "historical period", "date range"}
        if not tail_types.intersection(valid_time_types):
            return False, f"Rule violation: Tail '{tail_ent.get('surface')}' must be a time/date/century for dated_to."

    # Rule 9: gifted_by tail must not be purely a time expression
    if canonical_relation == "gifted_by":
        if any(t in tail_types for t in {"TIME", "year", "century", "date range"}):
            return False, f"Rule violation: Tail '{tail_ent.get('surface')}' is a time; gifted_by must point to a person, country, or organization. Use gifted_in for dates."

    # Rule 10: part_of semantic validity (prevent architectural parts being part_of a country/city)
    if canonical_relation == "part_of":
        arch_part_keywords = {
            "courtyard", "wall", "gate", "pathway", "sanctuary", "pavilion", "section",
            "sân", "tường", "cổng", "hall", "well", "pond", "lake", "garden", "belfry",
            "stele", "altar", "corridor", "roof", "railing", "column", "pillar", "door"
        }
        is_arch_part = (
            "architectural element" in head_types
            or any(kw in head_s for kw in arch_part_keywords)
        )
        geo_admin_types = {"country", "province", "city", "historical state or colony"}
        is_macro_geo = (
            bool(tail_types.intersection(geo_admin_types))
            or tail_s in {"vietnam", "việt nam", "hanoi", "hà nội", "china", "france"}
        )
        if is_arch_part and is_macro_geo:
            return False, (
                f"Rule 10: Architectural part '{head_ent.get('surface')}' cannot be part_of "
                f"macro geographic region '{tail_ent.get('surface')}'. Use located_in on the parent landmark."
            )

    # Check Domain constraint (Head)
    valid_domains = schema["domain"]
    if valid_domains is not None:
        if not head_types.intersection(valid_domains):
            return False, f"Domain mismatch for {canonical_relation}: head has {head_types}, required {valid_domains}"

    # Check Range constraint (Tail)
    valid_ranges = schema["range"]
    if valid_ranges is not None:
        if not tail_types.intersection(valid_ranges):
            return False, f"Range mismatch for {canonical_relation}: tail has {tail_types}, required {valid_ranges}"

    return True, "Valid"


# ──────────────────────────────────────────────────────────
# Module 4 Relation Extractor (English GLiNER-Relex + Window)
# ──────────────────────────────────────────────────────────
class Module4RelationExtractorEN:
    """
    Module 4 Relation Extractor for English Knowledge Graph Construction.
    Features:
      - Multi-Sentence Sliding Window for cross-sentence entity relations
      - GLiNER-Relex zero-shot relation extraction (knowledgator/gliner-relex-large-v1.0)
      - Post-GLiNER Entity Linking and deduplication using Module 3 Registry
      - Elimination of self-loops and duplicate triples
      - Comprehensive Ontology Validation (50+ relations)
    """

    DEFAULT_MODEL = "knowledgator/gliner-relex-large-v1.0"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        window_size: int = 2,
        window_stride: int = 1,
        min_confidence: float = 0.35,
        batch_size: int = 1,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.window_size = window_size
        self.window_stride = window_stride
        self.min_confidence = min_confidence
        self.batch_size = batch_size
        self.model = None

        if GLiNER is None:
            LOGGER.warning("GLiNER library not installed. pip install gliner")
            return

        try:
            LOGGER.info("Loading GLiNER-Relex model: %s ...", model_name)
            self.model = GLiNER.from_pretrained(model_name)
            if device:
                self.model = self.model.to(device)
            elif torch and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            elif torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Note: GLiNER-Relex has known logit truncation issues on MPS with 50+ relation labels; CPU is fully stable and fast
                self.model = self.model.to("cpu")
            LOGGER.info("✓ Loaded GLiNER-Relex model successfully.")
        except Exception as e:
            LOGGER.warning("Failed to load GLiNER-Relex model: %s", e)

    @property
    def available(self) -> bool:
        return self.model is not None

    def _resolve_mention_to_entity(
        self,
        mention_text: str,
        entities: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Map a predicted relation span (e.g. 'Imperial Citadel', 'Gia Long')
        to the best matching entity from Module 3.
        """
        if not mention_text:
            return None
        norm_m = normalize_key(mention_text)

        # 1. Exact normalized match
        for ent in entities:
            if normalize_key(ent.get("surface", "")) == norm_m:
                return ent

        # 2a. Mention contains entity: pick the longest matching entity (e.g. "Long Bien Bridge" over "Bridge")
        contained_matches = [
            ent for ent in entities
            if normalize_key(ent.get("surface", "")) in norm_m
        ]
        if contained_matches:
            return max(contained_matches, key=lambda e: len(normalize_key(e.get("surface", ""))))

        # 2b. Entity contains mention: pick the shortest/closest enclosing entity (avoids overshooting to sub-elements)
        enclosing_matches = [
            ent for ent in entities
            if norm_m in normalize_key(ent.get("surface", ""))
        ]
        if enclosing_matches:
            return min(enclosing_matches, key=lambda e: len(normalize_key(e.get("surface", ""))))

        # 3. Fallback: match via word overlap
        words_m = set(norm_m.split())
        for ent in entities:
            surf_words = set(normalize_key(ent.get("surface", "")).split())
            if words_m and surf_words and (words_m.issubset(surf_words) or surf_words.issubset(words_m)):
                return ent

        return None

    # ── A. Metadata Triples ─────────────────────────────────
    def _make_metadata_triples(
        self,
        record: Dict[str, Any],
        article_id: str,
        article_uri: str,
    ) -> List[Triple]:
        triples: List[Triple] = []

        # (article, rdf:type, Article)
        triples.append(
            Triple(
                head_uri=article_uri,
                relation=RDF_TYPE,
                tail_uri=f"{ONTO_URI}/Article",
                head_surface=article_id,
                tail_surface="Article",
                confidence=1.0,
                method="metadata",
                article_id=article_id,
            )
        )

        # (article, belongsTo, topic)
        for topic_uri in record.get("topic_uris", []):
            triples.append(
                Triple(
                    head_uri=article_uri,
                    relation="belongsTo",
                    tail_uri=topic_uri,
                    head_surface=article_id,
                    tail_surface=topic_uri.split("/")[-1],
                    confidence=1.0,
                    method="metadata",
                    article_id=article_id,
                )
            )

        # (entity, rdf:type, OntologyClass) & (article, hasEntity, entity)
        for ent in record.get("ner_entities", []):
            uri = ent.get("uri")
            if not uri:
                continue
            label = ent.get("label", "MISC")
            fine_label = ent.get("fine_grained_label", "")
            surface = ent.get("surface", "")
            if fine_label == "dynasty":
                class_uri = f"{ONTO_URI}/Dynasty"
            else:
                class_uri = LABEL_TO_CLASS_URI.get(label, f"{ONTO_URI}/Entity")

            triples.append(
                Triple(
                    head_uri=uri,
                    relation=RDF_TYPE,
                    tail_uri=class_uri,
                    head_surface=surface,
                    tail_surface=label,
                    confidence=1.0,
                    method="metadata",
                    article_id=article_id,
                )
            )

            triples.append(
                Triple(
                    head_uri=article_uri,
                    relation="hasEntity",
                    tail_uri=uri,
                    head_surface=article_id,
                    tail_surface=surface,
                    confidence=1.0,
                    method="metadata",
                    article_id=article_id,
                )
            )

        # (concept, rdf:type, Concept) & (article, hasConcept, concept)
        for con in record.get("concept_mentions", []):
            uri = con.get("uri")
            if not uri:
                continue
            c_text = con.get("concept", "")

            triples.append(
                Triple(
                    head_uri=uri,
                    relation=RDF_TYPE,
                    tail_uri=f"{ONTO_URI}/Concept",
                    head_surface=c_text,
                    tail_surface="Concept",
                    confidence=1.0,
                    method="metadata",
                    article_id=article_id,
                )
            )

            triples.append(
                Triple(
                    head_uri=article_uri,
                    relation="hasConcept",
                    tail_uri=uri,
                    head_surface=article_id,
                    tail_surface=c_text,
                    confidence=1.0,
                    method="metadata",
                    article_id=article_id,
                )
            )

        return triples

    # ── B. Semantic Relation Triples (GLiNER + Window) ──────
    def _extract_relation_triples(
        self,
        record: Dict[str, Any],
        article_id: str,
    ) -> List[Triple]:
        if not self.available:
            LOGGER.warning("GLiNER model not available — skipping neural relation extraction.")
            return []

        sentences = record.get("full_text_sentences", [])
        if not sentences:
            return []

        known_entities = record.get("ner_entities", [])
        if not known_entities:
            return []

        # 1. Build Multi-Sentence Sliding Windows
        windows: List[Tuple[int, str]] = []
        n_sent = len(sentences)
        k = self.window_size
        stride = self.window_stride

        for start_idx in range(0, n_sent, stride):
            end_idx = min(start_idx + k, n_sent)
            chunk_sents = sentences[start_idx:end_idx]
            w_text = " ".join(normalize_text(s) for s in chunk_sents if s.strip())
            if w_text:
                windows.append((start_idx, w_text))
            if end_idx >= n_sent:
                break

        if not windows:
            return []

        window_texts = [w[1] for w in windows]
        window_indices = [w[0] for w in windows]

        # Entity labels for GLiNER
        gliner_entity_labels = [
            "location", "person", "organization", "dynasty", "city", "province",
            "country", "ancient citadel or fortress", "tower", "gate", "bridge",
            "year", "date range", "century", "historical state or colony",
            "historical event", "war or battle", "architectural style",
            "building material", "era name", "historical place name",
            "title or rank", "government or regime", "architectural element",
            "religious figure", "official position", "lunar date",
            "statue", "artwork or artifact", "relic",
            "temple", "lake", "water", "tree", "railing", "roof", "building", "mountain"
        ]

        # 2. Run GLiNER-Relex Inference in batches
        if torch and hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        try:
            _, batch_relations = self.model.inference(
                texts=window_texts,
                labels=gliner_entity_labels,
                relations=ALL_GLINER_RELATION_LABELS,
                threshold=0.40,
                relation_threshold=self.min_confidence,
                batch_size=self.batch_size,
                return_relations=True,
                flat_ner=False,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                LOGGER.warning("MPS out of memory in GLiNER inference; falling back to CPU...")
                self.model = self.model.to("cpu")
                _, batch_relations = self.model.inference(
                    texts=window_texts,
                    labels=gliner_entity_labels,
                    relations=ALL_GLINER_RELATION_LABELS,
                    threshold=0.40,
                    relation_threshold=self.min_confidence,
                    batch_size=1,
                    return_relations=True,
                    flat_ner=False,
                )
            else:
                LOGGER.exception("GLiNER inference error on article %s: %s", article_id, e)
                return []
        except Exception as e:
            LOGGER.exception("GLiNER inference error on article %s: %s", article_id, e)
            return []

        # 3. Post-GLiNER Resolution, Deduplication & Ontology Validation
        candidate_triples: List[Triple] = []

        for w_idx, rels in enumerate(batch_relations):
            sent_start = window_indices[w_idx]
            w_text = window_texts[w_idx]
            end_idx = min(sent_start + self.window_size, n_sent)
            chunk_sents = sentences[sent_start:end_idx]

            for r in rels:
                head_dict = r.get("head", {})
                tail_dict = r.get("tail", {})
                head_mention = head_dict.get("text", "")
                rel_str = r.get("relation", "").strip().lower()
                tail_mention = tail_dict.get("text", "")
                conf = float(r.get("score", 0.0))

                head_span = [int(head_dict["start"]), int(head_dict["end"])] if "start" in head_dict and "end" in head_dict else None
                tail_span = [int(tail_dict["start"]), int(tail_dict["end"])] if "start" in tail_dict and "end" in tail_dict else None

                # Determine direct evidence sentence from window sentences
                evidence_sent = ""
                hm_lower = head_mention.lower()
                tm_lower = tail_mention.lower()
                for sent in chunk_sents:
                    s_lower = sent.lower()
                    if hm_lower in s_lower and tm_lower in s_lower:
                        evidence_sent = sent.strip()
                        break
                if not evidence_sent:
                    # If cross-sentence within window, use the multi-sentence window text
                    evidence_sent = w_text.strip()

                canonical_rel = PATTERN_TO_CANONICAL.get(rel_str, rel_str.replace(" ", "_"))

                # Resolve head and tail against known Linked Data entities from Module 3
                head_ent = self._resolve_mention_to_entity(head_mention, known_entities)
                tail_ent = self._resolve_mention_to_entity(tail_mention, known_entities)

                if not head_ent or not tail_ent:
                    continue

                head_uri = head_ent.get("uri", "")
                tail_uri = tail_ent.get("uri", "")

                if not head_uri or not tail_uri:
                    continue

                # Ontology Validation
                is_valid, reason = validate_triple_en(head_ent, canonical_rel, tail_ent, confidence=conf)
                if not is_valid:
                    LOGGER.debug("Rejected triple (%s, %s, %s): %s", head_mention, canonical_rel, tail_mention, reason)
                    continue

                candidate_triples.append(
                    Triple(
                        head_uri=head_uri,
                        relation=canonical_rel,
                        tail_uri=tail_uri,
                        head_surface=head_ent.get("surface", head_mention),
                        tail_surface=tail_ent.get("surface", tail_mention),
                        confidence=round(conf, 4),
                        method="gliner_relex",
                        article_id=article_id,
                        window_idx=sent_start,
                        raw_relation=rel_str,
                        evidence=evidence_sent,
                        head_span=head_span,
                        tail_span=tail_span,
                    )
                )

        # 4. Deduplicate (keep the highest confidence triple for each (head_uri, relation, tail_uri))
        unique_triples: Dict[Tuple[str, str, str], Triple] = {}
        for t in candidate_triples:
            key = (t.head_uri, t.relation, t.tail_uri)
            if key not in unique_triples or t.confidence > unique_triples[key].confidence:
                unique_triples[key] = t

        # Rule 4: Remove contradictory successor_of pairs — if both (A→B) and (B→A) exist,
        # keep only the one with higher confidence
        successor_keys = [k for k in unique_triples if k[1] == "successor_of"]
        to_remove: Set[Tuple[str, str, str]] = set()
        for k in successor_keys:
            reverse_k = (k[2], "successor_of", k[0])
            if reverse_k in unique_triples and k not in to_remove and reverse_k not in to_remove:
                if unique_triples[k].confidence >= unique_triples[reverse_k].confidence:
                    to_remove.add(reverse_k)
                else:
                    to_remove.add(k)
        for k in to_remove:
            del unique_triples[k]

        # Rule 5: Functional & Single-valued Relation Conflict Resolution (W3C OWL FunctionalProperty)
        # Keeps only the highest-confidence tail for single-valued relations
        sv_relations = {r for r, info in ONTOLOGY_SCHEMA.items() if info.get("single_valued")}
        sv_best: Dict[Tuple[str, str], Tuple[str, float]] = {}
        for (h, r, t), triple_obj in unique_triples.items():
            if r in sv_relations:
                if (h, r) not in sv_best or triple_obj.confidence > sv_best[(h, r)][1]:
                    sv_best[(h, r)] = (t, triple_obj.confidence)

        # Spatial consistency: a physical structure cannot be situated in multiple disjoint cities
        city_uris = {
            e.get("uri") for e in known_entities
            if e.get("fine_grained_label") == "city" or e.get("label") == "city"
        }
        loc_city_best: Dict[str, Tuple[str, float]] = {}
        for (h, r, t), triple_obj in unique_triples.items():
            if r == "located_in" and t in city_uris:
                if h not in loc_city_best or triple_obj.confidence > loc_city_best[h][1]:
                    loc_city_best[h] = (t, triple_obj.confidence)

        to_prune: Set[Tuple[str, str, str]] = set()
        for (h, r, t), triple_obj in unique_triples.items():
            if r in sv_relations and (h, r) in sv_best:
                if t != sv_best[(h, r)][0]:
                    to_prune.add((h, r, t))
            if r == "located_in" and t in city_uris and h in loc_city_best:
                if t != loc_city_best[h][0]:
                    to_prune.add((h, r, t))

        for k in to_prune:
            if k in unique_triples:
                del unique_triples[k]

        return list(unique_triples.values())

    # ── C. Record & Batch Processing ────────────────────────
    def process_record(self, record: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Triple]]:
        article_id = record.get("article_id", "art_unknown")
        article_uri = record.get("article_uri", f"{ARTICLE_URI}/{article_id}")

        metadata_triples = self._make_metadata_triples(record, article_id, article_uri)
        relation_triples = self._extract_relation_triples(record, article_id)

        all_triples = metadata_triples + relation_triples

        enriched = dict(record)
        enriched["triples"] = [t.to_dict() for t in all_triples]

        stats = enriched.setdefault("stats", {})
        stats["n_metadata_triples"] = len(metadata_triples)
        stats["n_relation_triples"] = len(relation_triples)
        stats["n_total_triples"] = len(all_triples)

        return enriched, all_triples

    def extract_all(
        self,
        input_path: str,
        output_triples_path: str,
        output_enriched_path: str,
    ):
        in_p = Path(input_path)
        trip_p = Path(output_triples_path)
        enr_p = Path(output_enriched_path)
        trip_p.parent.mkdir(parents=True, exist_ok=True)
        enr_p.parent.mkdir(parents=True, exist_ok=True)

        if not in_p.exists():
            raise FileNotFoundError(f"Input file not found: {in_p}")

        LOGGER.info("Starting English Relation Extraction on %s...", in_p)
        is_jsonl = in_p.suffix == ".jsonl"

        records: List[Dict[str, Any]] = []
        if is_jsonl:
            with in_p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        else:
            raw = in_p.read_text(encoding="utf-8").strip()
            data = json.loads(raw)
            records = data if isinstance(data, list) else [data]

        all_extracted_triples: List[Triple] = []
        enriched_records: List[Dict[str, Any]] = []
        total_meta = 0
        total_rel = 0

        for i, rec in enumerate(records, start=1):
            try:
                enr, triples = self.process_record(rec)
                enriched_records.append(enr)
                all_extracted_triples.extend(triples)

                meta_count = enr["stats"].get("n_metadata_triples", 0)
                rel_count = enr["stats"].get("n_relation_triples", 0)
                total_meta += meta_count
                total_rel += rel_count

                LOGGER.info(
                    "Article %d/%d (%s): +%d metadata, +%d semantic relation triples",
                    i, len(records), rec.get("article_id", ""), meta_count, rel_count
                )
            except Exception as e:
                LOGGER.exception("Error processing article %d: %s", i, e)

        # Write output
        if is_jsonl:
            with enr_p.open("w", encoding="utf-8") as fe, trip_p.open("w", encoding="utf-8") as ft:
                for r in enriched_records:
                    fe.write(json.dumps(r, ensure_ascii=False) + "\n")
                for t in all_extracted_triples:
                    ft.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
        else:
            with enr_p.open("w", encoding="utf-8") as fe, trip_p.open("w", encoding="utf-8") as ft:
                if len(enriched_records) == 1 and not isinstance(data, list):
                    json.dump(enriched_records[0], fe, indent=2, ensure_ascii=False)
                else:
                    json.dump(enriched_records, fe, indent=2, ensure_ascii=False)

                json.dump([t.to_dict() for t in all_extracted_triples], ft, indent=2, ensure_ascii=False)

        LOGGER.info(
            "✓ Finished Relation Extraction: %d articles | %d metadata, %d semantic relation triples → %s",
            len(enriched_records), total_meta, total_rel, trip_p
        )


# ──────────────────────────────────────────────────────────
# Standalone CLI Entrypoint
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    current_dir = Path(__file__).resolve().parent
    default_input = current_dir / "test_linked_output.json"
    default_triples = current_dir / "test_triples_en.json"
    default_enriched = current_dir / "test_enriched_en.json"

    input_file = os.environ.get("OKG_INPUT_FILE", str(default_input))
    triples_file = os.environ.get("OKG_TRIPLES_FILE", str(default_triples))
    enriched_file = os.environ.get("OKG_ENRICHED_FILE", str(default_enriched))

    extractor = Module4RelationExtractorEN(
        model_name="knowledgator/gliner-relex-large-v1.0",
        window_size=2,
        window_stride=1,
        min_confidence=0.35,
        batch_size=4,
    )

    extractor.extract_all(
        input_path=input_file,
        output_triples_path=triples_file,
        output_enriched_path=enriched_file,
    )
