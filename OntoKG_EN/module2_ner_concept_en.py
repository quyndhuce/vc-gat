from __future__ import annotations

import ast
import html
import json
import logging
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# Optional PyTorch & SentenceTransformers for semantic concept linking
try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = None
    AutoTokenizer = None

# GLiNER zero-shot entity extraction
try:
    from gliner import GLiNER
except Exception:
    GLiNER = None

# spaCy for syntactic dependency parsing & noun chunk extraction
try:
    import spacy
    _SPACY_NLP = spacy.load("en_core_web_sm")
except Exception:
    _SPACY_NLP = None


# =========================================================
# Logging
# =========================================================
LOGGER = logging.getLogger("module2_ner_concept_en")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    LOGGER.addHandler(stream_handler)


# =========================================================
# Regexes & String Helpers
# =========================================================
CONTROL_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
PUNCT_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$")
WHITESPACE_RE = re.compile(r"\s+")
LEADING_CONNECTOR_RE = re.compile(r"^(?:the|a|an|of|at|in|to|from|by|for|with|on|under|over|into|onto|upon)\s+", re.IGNORECASE)
TRAILING_CONNECTOR_RE = re.compile(r"\s+(?:and|or|of|in|to|for|with|by|at|from|the|a|an)$", re.IGNORECASE)
POSSESSIVE_RE = re.compile(r"[’\']s$", re.IGNORECASE)

# Dedicated English temporal patterns (years, date ranges, centuries, relative time)
DATE_RANGE_RE = re.compile(r"\b(\d{4}\s*[–—\-]\s*\d{4})\b")
CENTURY_RE = re.compile(r"\b(\d{1,2}(?:st|nd|rd|th)\s+century)\b", re.IGNORECASE)
RELATIVE_TIME_RE = re.compile(r"\b((?:early|mid|late)-\d{4})\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")


def expand_span_with_noun_chunk(
    text: str,
    start_char: int,
    end_char: int,
    nlp_model: Any = None,
) -> Tuple[int, int, str]:
    """
    Syntactic Noun Phrase Boundary Expansion (Rule-free, generalizable):
    Given a predicted mention span [start_char, end_char] within a sentence,
    uses spaCy dependency parsing and noun chunk tree to expand the span to include
    all associated syntactic modifiers (compounds, proper nouns, numerals).
    
    Example:
      text = "The One Pillar Pagoda formally belongs to..."
      input span: "Pagoda" (start=15, end=21)
      output: (4, 21, "One Pillar Pagoda")
    """
    if not nlp_model or not text or start_char < 0 or end_char <= start_char:
        return start_char, end_char, text[start_char:end_char]

    try:
        doc = nlp_model(text)
    except Exception:
        return start_char, end_char, text[start_char:end_char]

    for chunk in doc.noun_chunks:
        if chunk.start_char <= start_char and end_char <= chunk.end_char:
            chunk_tokens = list(chunk)
            target_idx = None
            for idx, tok in enumerate(chunk_tokens):
                if tok.idx <= start_char < tok.idx + len(tok.text):
                    target_idx = idx
                    break
            if target_idx is None:
                continue

            # Expand backward while preceding token is a syntactic nominal modifier:
            # compound noun, flat name, numeral modifier, or proper noun
            # (Excludes verb participles like 'Flanking', 'Including', 'Surrounding')
            left_idx = target_idx
            while left_idx > 0:
                prev_tok = chunk_tokens[left_idx - 1]
                if (
                    prev_tok.pos_ not in ("VERB", "AUX")
                    and prev_tok.tag_ not in ("VBG", "VBN", "VB", "VBD", "VBZ", "VBP")
                    and (
                        prev_tok.dep_ in ("compound", "flat", "nummod")
                        or prev_tok.pos_ in ("PROPN", "NUM")
                        or prev_tok.tag_ in ("NNP", "NNPS", "CD")
                    )
                ):
                    left_idx -= 1
                else:
                    break

            new_start = chunk_tokens[left_idx].idx
            new_end = max(end_char, chunk.end_char)
            return new_start, new_end, text[new_start:new_end].strip()

    return start_char, end_char, text[start_char:end_char]


# =========================================================
# 10 Top-Level Ontology Categories
# (LANDMARK, GEO and all spatial/address labels grouped under parent LOC;
# HERITAGE is strictly separated from LOC)
# =========================================================
NER_LABEL_MAP = {
    "LOC": "Location",                 # Unified parent for all spatial entities: admin, address, landmarks, citadels, geography, temples, museums
    "PER": "Person",                    # People, historical figures, dynasties, ethnic groups
    "ORG": "Organization",              # Institutions, committees, centers, agencies
    "TIME": "TimeRef",                  # Temporal: year, date, century, historical period
    "EVENT": "Event",                   # Named conferences/exhibitions, traditional festivals, events, battles
    "HERITAGE": "Heritage",             # Heritage designations & laws/legal documents (strictly separated from LOC)
    "ARTIFACT": "Artifact",             # Artworks, statues, archaeological artifacts
    "STYLE": "Style",                   # Architectural styles, building materials
    "PRODUCT": "Product",               # Traditional specialty food, crafts, products
    "ATTR": "Attribute",                # Colors, measurements
}

VALID_NER_LABELS = set(NER_LABEL_MAP.keys())


# =========================================================
# Fine-Grained Entity Labels (Domain-Tailored for Hanoi Tourism,
# Heritage, History & Administration)
# =========================================================
GLINER_ENTITY_LABELS_DEFAULT: List[str] = [
    # --- 1. Administrative, Urban & Historical Locations (all vote LOC) ---
    "country",
    "province",
    "city",
    "district",                     # User requested (e.g. Ba Dinh District, Hoan Kiem)
    "ward",                         # User requested (e.g. Quan Thanh Ward, Hang Gai)
    "street address",               # User requested (e.g. No. 12 Nguyen Tri Phuong Street)
    "historical place name",        # User requested (e.g. Thang Long, Dong Do, Co Loa)
    "historical state or colony",   # User requested (e.g. Đại Việt, French Indochina, Indochina Federation)
    "historical administrative unit", # User requested (e.g. Hanoi province, administrative divisions)
    # --- 2. Built Landmarks & Architectural Heritage (vote LOC) ---
    "landmark",
    "ancient citadel or fortress",  # User requested (e.g. Imperial Citadel of Thang Long, Co Loa)
    "temple or pagoda",             # Hanoi Tourism (e.g. Tran Quoc Pagoda, Ngoc Son Temple, One Pillar)
    "temple",                       # Visual entity (Đền, chùa, miếu, điện thờ)
    "building",                     # Visual entity (Tòa nhà, công trình)
    "communal house or shrine",     # Hanoi Heritage (e.g. Dinh Bat Trang, Dinh Kim Ngan)
    "museum",                       # Hanoi Tourism (e.g. Vietnam National Museum of History)
    "monument or memorial",         # Hanoi Tourism (e.g. Ho Chi Minh Mausoleum, Ly Thai To Monument)
    "mausoleum or tomb",            # User requested (e.g. Ho Chi Minh Mausoleum, Lenin's Mausoleum)
    "craft village",                # Hanoi Tourism (e.g. Bat Trang ceramic village, Van Phuc silk village)
    "bridge",                       # Visual entity (e.g. Long Bien Bridge, The Huc Bridge)
    "tower",                        # (e.g. Turtle Tower, Flag Tower of Hanoi)
    "gate",                         # Visual entity (e.g. O Quan Chuong Gate, Doan Mon Gate)
    "roof",                         # Visual entity (Mái)
    "railing",                      # Visual entity (Lan can, tay vịn)
    "architectural element",        # User requested (e.g. Flag Tower, North Gate, Đoan Môn, Kinh Thiên, Khuê Văn Các)
    # --- 3. Natural Geography & Scenic Landscapes (vote LOC) ---
    "lake",                         # Visual entity (e.g. Hoan Kiem Lake, West Lake, Truc Bach Lake)
    "water",                        # Visual entity (Mặt nước nói chung)
    "river",                        # (e.g. Red River, To Lich River, Duong River)
    "mountain",                     # (e.g. Ba Vi Mountain, Soc Son Mountain)
    "tree",                         # Visual entity (Cây, tán cây)
    "cave",                         # (e.g. Huong Tich Cave)
    "island",
    "canal",
    "bay",
    "beach",
    # --- 4. People, Historical Figures & Society ---
    "person",
    "historical figure",            # Hanoi History (e.g. King Ly Thai To, King An Duong Vuong, Gia Long, Minh Mang)
    "religious figure",             # User requested (e.g. Confucius, Buddha, Trần Hưng Đạo, Thánh Gióng)
    "title or rank",                # User requested (e.g. Duke of Zhou, Marquis, Hầu, Công)
    "official position",            # User requested (e.g. Tổng đốc, Thượng thư, Governor-General)
    "dynasty",                      # (e.g. Ly Dynasty, Nguyen Dynasty, Qing Dynasty)
    "ethnic group",
    # --- 5. Organizations & Institutions ---
    "organization",                 # (e.g. Hanoi People's Committee, UNESCO, Imperial Japanese Army)
    "government or regime",         # User requested (e.g. French Indochina, French protectorate, Nguyễn court)
    # --- 6. Temporal References (Years, Ranges, Centuries, Periods) ---
    "year",                         # (e.g. 1802, 1805, 1831, 1873, 1888, 1945)
    "date",                         # (e.g. March 1945)
    "century",                      # (e.g. 18th century, 21st century)
    "date range",                   # (e.g. 1885–1954)
    "time period",
    "historical period",
    "era name",                     # User requested (e.g. Thần Vũ, Long Thụy Thái Bình, Thiệu Bảo)
    "lunar date",                   # User requested (e.g. 8th lunar month, tháng Giêng âm lịch)
    # --- 7. Cultural & Historical Events (Named Events & Festivals) ---
    "historical event",             # User requested (e.g. administrative reform, coup d'état, extensive reconstruction)
    "traditional festival",         # Hanoi Culture (e.g. Giong Festival, Huong Pagoda Festival)
    "named conference or exhibition", # Named conferences & exhibitions with proper titles
    "named cultural event",         # Named cultural events with proper titles
    "war or battle",
    # --- 8. Heritage Designations & Legal Documents (Strictly HERITAGE, separated from LOC) ---
    "heritage designation",         # (e.g. UNESCO World Cultural Heritage site)
    "law or legal document",        # User requested (e.g. Law on Cultural Heritage)
    # --- 9. Artworks, Statues & Relics ---
    "artwork or artifact",          # (e.g. archaeological materials, artifacts)
    "statue",                       # (e.g. Statue of King Ly Thai To, statue of Buddha)
    # --- 10. Architecture, Materials & Style ---
    "architectural style",
    "building material",
    # --- 11. Traditional Cuisine & Handicrafts ---
    "dish or food",                 # (e.g. Pho Hanoi, Bun cha, Cha ca La Vong)
    "traditional craft or handicraft", # (e.g. Bat Trang ceramics, Van Phuc silk)
    "craft or product",
    # --- 12. Attributes & Measurements ---
    "color",
    "measurement",
    # --- 13. Religion, Beliefs, Arts & Nature ---
    "religion or belief",           # User requested (e.g. Buddhism, Confucianism, Taoism)
    "art form",                     # User requested (e.g. ca trù, quan họ, múa rối nước)
    "plant or species",             # User requested (e.g. banyan tree, lotus)
]


# =========================================================
# Mapping from Fine-Grained Labels to (Ontology Code, Display Name)
# Note: LANDMARK, GEO, and all location types map to parent code LOC
# HERITAGE is strictly separated from LOC
# =========================================================
LABEL_MAPPING: Dict[str, Tuple[str, str]] = {
    # --- LOC (Administrative, Address & Historical Locations) ---
    "country": ("LOC", "Country"),
    "province": ("LOC", "Province"),
    "city": ("LOC", "City"),
    "district": ("LOC", "District"),
    "ward": ("LOC", "Ward"),
    "street address": ("LOC", "Street Address"),
    "street name": ("LOC", "Street Name"),
    "historical place name": ("LOC", "Historical Place Name"),
    "historical state or colony": ("LOC", "Historical State or Colony"),
    "historical administrative unit": ("LOC", "Historical Administrative Unit"),
    "location": ("LOC", "Location"),
    "loc": ("LOC", "Location"),
    "gpe": ("LOC", "Location"),

    # --- LOC (Built Landmarks, Fortresses, Temples, Pagodas, Museums, Visual Entities) ---
    "landmark": ("LOC", "Landmark"),
    "ancient citadel or fortress": ("LOC", "Ancient Citadel or Fortress"),
    "temple or pagoda": ("LOC", "Temple or Pagoda"),
    "temple": ("LOC", "Temple"),
    "building": ("LOC", "Building"),
    "communal house or shrine": ("LOC", "Communal House or Shrine"),
    "craft village": ("LOC", "Craft Village"),
    "museum": ("LOC", "Museum"),
    "monument or memorial": ("LOC", "Monument or Memorial"),
    "mausoleum or tomb": ("LOC", "Mausoleum or Tomb"),
    "bridge": ("LOC", "Bridge"),
    "tower": ("LOC", "Tower"),
    "gate": ("LOC", "Gate"),
    "roof": ("LOC", "Roof"),
    "railing": ("LOC", "Railing"),

    # --- LOC (Natural Geography, Scenic Features & Water) ---
    "lake": ("LOC", "Lake"),
    "water": ("LOC", "Water"),
    "river": ("LOC", "River"),
    "canal": ("LOC", "Canal"),
    "mountain": ("LOC", "Mountain"),
    "tree": ("CONCEPT", "Tree"),
    "cave": ("LOC", "Cave"),
    "island": ("LOC", "Island"),
    "bay": ("LOC", "Bay"),
    "beach": ("LOC", "Beach"),

    # --- PER (People, Historical Figures, Dynasties) ---
    "person": ("PER", "Person"),
    "historical figure": ("PER", "Historical Figure"),
    "dynasty": ("PER", "Dynasty"),
    "ethnic group": ("PER", "Ethnic Group"),
    "people": ("PER", "Person"),
    "per": ("PER", "Person"),

    # --- ORG (Organizations & Committees) ---
    "organization": ("ORG", "Organization"),
    "org": ("ORG", "Organization"),
    "facility": ("ORG", "Organization"),

    # --- TIME (Temporal References) ---
    "year": ("TIME", "Year"),
    "date": ("TIME", "Date"),
    "century": ("TIME", "Century"),
    "date range": ("TIME", "Date Range"),
    "time period": ("TIME", "Time Period"),
    "relative time": ("TIME", "Relative Time"),
    "historical period": ("TIME", "Historical Period"),
    "time": ("TIME", "Time"),

    # --- EVENT (Festivals, Named Exhibitions & Conferences, Battles, Historical Events) ---
    "historical event": ("EVENT", "Historical Event"),
    "traditional festival": ("EVENT", "Traditional Festival"),
    "named conference or exhibition": ("EVENT", "Named Conference or Exhibition"),
    "named cultural event": ("EVENT", "Named Cultural Event"),
    "war or battle": ("EVENT", "War or Battle"),
    "festival or event": ("EVENT", "Festival or Event"),
    "event": ("EVENT", "Event"),

    # --- HERITAGE (Strictly separated from LOC: Designations & Laws) ---
    "heritage designation": ("HERITAGE", "Heritage Designation"),
    "law or legal document": ("HERITAGE", "Law or Legal Document"),
    "heritage": ("HERITAGE", "Heritage"),

    # --- ARTIFACT (Tangible Relics, Statues, Artworks) ---
    "artwork or artifact": ("ARTIFACT", "Artwork or Artifact"),
    "statue": ("ARTIFACT", "Statue"),
    "artwork": ("ARTIFACT", "Artwork"),
    "artifact": ("ARTIFACT", "Artifact"),

    # --- STYLE (Architectural Styles & Materials) ---
    "architectural style": ("STYLE", "Architectural Style"),
    "building material": ("STYLE", "Building Material"),
    "style": ("STYLE", "Style"),

    # --- PRODUCT (Cuisine, Traditional Crafts & Products) ---
    "dish or food": ("PRODUCT", "Dish or Food"),
    "traditional craft or handicraft": ("PRODUCT", "Traditional Craft"),
    "craft or product": ("PRODUCT", "Craft or Product"),
    "product": ("PRODUCT", "Product"),

    # --- ATTR (Attributes & Measurements) ---
    "color": ("ATTR", "Color"),
    "measurement": ("ATTR", "Measurement"),
    "title or rank": ("ATTR", "Title or Rank"),
    "official position": ("ATTR", "Official Position"),

    # --- New Domain-Tailored Types ---
    "era name": ("TIME", "Era Name"),
    "lunar date": ("TIME", "Lunar Date"),
    "architectural element": ("LOC", "Architectural Element"),
    "religious figure": ("PER", "Religious Figure"),
    "government or regime": ("ORG", "Government or Regime"),
    "religion or belief": ("CONCEPT", "Religion or Belief"),
    "art form": ("CONCEPT", "Art Form"),
    "plant or species": ("CONCEPT", "Plant or Species"),
}



# Grammatical function words / closed-class stopwords for concept boundaries
# (Direct English equivalent of Vietnamese STOPWORDS_CONCEPT from code gốc)
STOPWORDS_CONCEPT = {
    "of", "and", "or", "the", "a", "an", "in", "on", "at", "to", "for", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "they", "their", "them", "this", "that", "these", "those",
    "under", "within", "between", "through", "about", "into", "onto", "upon",
    "he", "she", "his", "her", "we", "our", "us", "you", "your",
}


# =========================================================
# Utils
# =========================================================
def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFC", text)
    text = CONTROL_RE.sub("", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("\xa0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def normalize_key(text: str) -> str:
    text = normalize_text(text).lower()
    text = PUNCT_EDGE_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def dedupe_preserve(items: Sequence[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        x = normalize_text(x)
        if not x:
            continue
        key = normalize_key(x)
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def parse_list_like(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [normalize_text(x) for x in value if normalize_text(x)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [normalize_text(x) for x in parsed if normalize_text(x)]
        except Exception:
            pass
        return [normalize_text(x) for x in re.split(r"\s*[,;|]\s*", text) if normalize_text(x)]
    return [normalize_text(value)]


def pos_is_nounlike(pos_tag: str) -> bool:
    pos_tag = normalize_text(pos_tag)
    return pos_tag.startswith("NN")


def jaccard_similarity(text_a: str, text_b: str) -> float:
    a = set(normalize_key(text_a).split())
    b = set(normalize_key(text_b).split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# =========================================================
# Dataclasses
# =========================================================
@dataclass
class NERMentionEN:
    surface: str
    norm_surface: str
    label: str                     # Top-level ontology code: LOC, ORG, PER, HERITAGE, etc.
    label_name: str                # Human-readable fine-grained category name
    confidence: float
    source: str
    sentence_index: int
    fine_grained_label: str = ""   # Exact GLiNER predicted label
    votes: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class ConceptMentionEN:
    concept: str
    norm_concept: str
    score: float
    source: str
    linked_ontology_id: Optional[str]
    similarity: Optional[float]
    action: str  # "link", "review", "new"

    def to_dict(self):
        return asdict(self)


# =========================================================
# Main Extractor Class
# =========================================================
class Module2NerConceptExtractorEN:
    """
    Module 2: Named Entity Recognition & Concept Extraction for English.
    Uses GLiNER (knowledgator/gliner-relex-large-v1.0) as the sole NER engine
    with domain-specific zero-shot entity labels, parent LOC voting for all spatial
    entities, strict HERITAGE separation, and post-processing common-noun filtering.
    """

    def __init__(
        self,
        ontology_concepts: Optional[List[Dict[str, Any]]] = None,
        use_gliner: bool = True,
        gliner_model_name: str = "knowledgator/gliner-relex-large-v1.0",
        gliner_threshold: float = 0.50,
        gliner_entity_labels: Optional[List[str]] = None,
        concept_embedding_model_name: Optional[str] = "sentence-transformers/all-mpnet-base-v2",
        concept_link_threshold: float = 0.85,
        concept_review_threshold: float = 0.65,
        concept_min_score: float = 0.45,
        max_concept_tokens: int = 5,
        max_concepts: int = 25,
    ):
        self.ontology_concepts = ontology_concepts or []
        self.concept_link_threshold = concept_link_threshold
        self.concept_review_threshold = concept_review_threshold
        self.concept_min_score = concept_min_score
        self.max_concept_tokens = max_concept_tokens
        self.max_concepts = max_concepts

        # Device selection
        if torch is not None:
            if torch.cuda.is_available():
                self._torch_device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._torch_device = torch.device("mps")
            else:
                self._torch_device = torch.device("cpu")
        else:
            self._torch_device = None

        # 1. GLiNER NER Model (Sole NER engine)
        self.use_gliner = use_gliner and (GLiNER is not None)
        self.gliner_model_name = gliner_model_name
        self.gliner_threshold = gliner_threshold
        self.gliner_entity_labels = gliner_entity_labels or GLINER_ENTITY_LABELS_DEFAULT
        self.gliner_model = None

        if self.use_gliner:
            try:
                LOGGER.info("Loading GLiNER model: %s ...", gliner_model_name)
                self.gliner_model = GLiNER.from_pretrained(gliner_model_name)
                if self._torch_device is not None:
                    self.gliner_model.to(self._torch_device)
                LOGGER.info("GLiNER model loaded successfully.")
            except Exception as e:
                self.gliner_model = None
                LOGGER.warning("Could not load GLiNER model (%s): %s", gliner_model_name, e)

        # 2. Concept Embedding Model for Semantic Linking (Optional)
        self.concept_embedding_model_name = concept_embedding_model_name
        self.sbert_model = None
        self.embed_tokenizer = None
        self.embed_model = None
        self._embed_cache: Dict[str, Any] = {}
        self._ontology_embed_cache: Dict[str, Any] = {}

        if concept_embedding_model_name:
            if SentenceTransformer is not None:
                try:
                    self.sbert_model = SentenceTransformer(concept_embedding_model_name)
                    if self._torch_device is not None:
                        self.sbert_model.to(self._torch_device)
                    LOGGER.info("Loaded SBERT model: %s", concept_embedding_model_name)
                except Exception as e:
                    self.sbert_model = None
                    LOGGER.info("SentenceTransformer not loaded (%s), using lexical linking.", e)

            if self.sbert_model is None and AutoTokenizer is not None and AutoModel is not None:
                try:
                    self.embed_tokenizer = AutoTokenizer.from_pretrained(concept_embedding_model_name)
                    self.embed_model = AutoModel.from_pretrained(concept_embedding_model_name)
                    self.embed_model.eval()
                    if self._torch_device is not None:
                        self.embed_model.to(self._torch_device)
                    LOGGER.info("Loaded HF AutoModel: %s", concept_embedding_model_name)
                except Exception:
                    self.embed_tokenizer = None
                    self.embed_model = None

    # =========================================================
    # Embedding Helpers
    # =========================================================
    def _embed_text(self, text: str) -> Optional[Any]:
        text = normalize_text(text)
        if not text:
            return None
        key = normalize_key(text)
        if key in self._embed_cache:
            return self._embed_cache[key]

        if self.sbert_model is not None:
            try:
                emb = self.sbert_model.encode(text, convert_to_tensor=True)
                self._embed_cache[key] = emb
                return emb
            except Exception:
                pass

        if self.embed_model is not None and self.embed_tokenizer is not None and torch is not None:
            try:
                inputs = self.embed_tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=128, padding=True
                )
                if self._torch_device is not None:
                    inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.embed_model(**inputs)
                last_hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).type_as(last_hidden)
                summed = (last_hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                emb = (summed / denom)[0].detach().cpu()
                self._embed_cache[key] = emb
                return emb
            except Exception:
                pass

        return None

    def _cosine_similarity(self, vec_a: Any, vec_b: Any) -> float:
        if vec_a is None or vec_b is None:
            return 0.0
        if torch is not None and isinstance(vec_a, torch.Tensor) and isinstance(vec_b, torch.Tensor):
            try:
                return float(F.cosine_similarity(vec_a.view(1, -1), vec_b.view(1, -1)).item())
            except Exception:
                return 0.0
        return 0.0

    # =========================================================
    # GLiNER NER Inference with Sub-Span Containment Resolution
    # =========================================================
    def _extract_gliner_ner(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        if self.gliner_model is None or not texts:
            return [[] for _ in texts]

        if torch and hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        raw = None
        try:
            raw = self.gliner_model.inference(
                texts=texts,
                labels=self.gliner_entity_labels,
                threshold=self.gliner_threshold,
                batch_size=2,
                flat_ner=False,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "mps" in str(e).lower():
                LOGGER.warning("MPS out of memory in GLiNER NER; falling back to CPU...")
                self.gliner_model = self.gliner_model.to("cpu")
                try:
                    raw = self.gliner_model.inference(
                        texts=texts,
                        labels=self.gliner_entity_labels,
                        threshold=self.gliner_threshold,
                        batch_size=1,
                        flat_ner=False,
                    )
                except Exception as e2:
                    LOGGER.warning("GLiNER NER CPU retry error: %s", e2)
                    return [[] for _ in texts]
            else:
                LOGGER.warning("GLiNER NER inference error: %s", e)
                return [[] for _ in texts]
        except Exception as e:
            LOGGER.warning("GLiNER NER inference error: %s", e)
            return [[] for _ in texts]

        try:

            if isinstance(raw, tuple):
                entities = raw[0]
            else:
                entities = raw

            if entities and isinstance(entities[0], list) and entities[0] and isinstance(entities[0][0], list):
                entities = [inner[0] if inner else [] for inner in entities]

            # Sentence-level sub-span resolution for maximum precision
            resolved_all = []
            for s_idx, s_ents in enumerate(entities):
                sent_text = texts[s_idx] if s_idx < len(texts) else ""
                filtered = []
                for e in s_ents:
                    sa = e.get("start", 0)
                    ea = e.get("end", 0)

                    # Syntactic Noun Chunk Expansion based on Dependency & POS tags
                    if sent_text and _SPACY_NLP is not None:
                        sa, ea, e_text = expand_span_with_noun_chunk(sent_text, sa, ea, _SPACY_NLP)
                    else:
                        e_text = e.get("text", "")

                    surface = PUNCT_EDGE_RE.sub("", e_text).strip()
                    surface = LEADING_CONNECTOR_RE.sub("", surface).strip()
                    surface = TRAILING_CONNECTOR_RE.sub("", surface).strip()
                    surface = POSSESSIVE_RE.sub("", surface).strip()
                    surface = PUNCT_EDGE_RE.sub("", surface).strip()
                    raw_label = e.get("label", "").lower().strip()
                    onto_code, label_name = LABEL_MAPPING.get(raw_label, (raw_label.upper(), raw_label.title()))

                    if not surface or self._is_false_positive(surface, onto_code, raw_label):
                        continue

                    e_copy = dict(e)
                    e_copy["text"] = surface
                    e_copy["start"] = sa
                    e_copy["end"] = ea
                    filtered.append(e_copy)

                suppressed = set()
                n = len(filtered)
                for i in range(n):
                    a = filtered[i]
                    sa, ea = a.get("start", 0), a.get("end", 0)
                    la = ea - sa
                    ta = a["text"]

                    for j in range(n):
                        if i == j:
                            continue
                        b = filtered[j]
                        sb, eb = b.get("start", 0), b.get("end", 0)
                        lb = eb - sb

                        # Check if span A is strictly contained inside span B
                        if sb <= sa and ea <= eb and la < lb:
                            lb_label = b.get("label", "")
                            la_label = a.get("label", "")

                            # Rule 0: Inside date range / time period, suppress nested single years
                            if lb_label in ("date range", "time period") and la_label in ("year", "date"):
                                suppressed.add(i)
                                break
                            # Rule 1: Inside 'street address', suppress internal number/name fragments
                            if lb_label == "street address":
                                suppressed.add(i)
                                break
                            # Rule 2: Same or compatible category (e.g. ward inside ward, district inside district)
                            if la_label == lb_label:
                                suppressed.add(i)
                                break
                            # Rule 3: Single token fragment inside a multi-token proper name
                            if " " not in ta.strip():
                                suppressed.add(i)
                                break
                            # Rule 4: Generic boundary word (e.g. "district", "ward", "relic", "citadel", "center")
                            if ta.lower() in self._GENERIC_FRAGMENT_WORDS:
                                suppressed.add(i)
                                break

                resolved_all.append([filtered[i] for i in range(n) if i not in suppressed])

            return resolved_all
        except Exception as e:
            LOGGER.warning("GLiNER NER inference error: %s", e)
            return [[] for _ in texts]

    # =========================================================
    # Temporal Entity Extraction (Years, Ranges, Centuries, Relative Time)
    # =========================================================
    def _extract_temporal_entities(self, sentences: List[str]) -> List[NERMentionEN]:
        """
        Extracts comprehensive English temporal entities:
        - Date ranges (e.g. 1885–1954, 1885-1954)
        - Centuries (e.g. 18th century, 21st century)
        - Relative time (e.g. mid-1945, early-19th century)
        - Standalone 4-digit years (e.g. 1802, 1805, 1831, 1873, 1888, 1945)
        """
        mentions: List[NERMentionEN] = []
        for idx, sent in enumerate(sentences):
            temp_spans: List[Tuple[int, int, str, str, str]] = []

            # 1. Date ranges (e.g. 1885–1954)
            for m in DATE_RANGE_RE.finditer(sent):
                temp_spans.append((m.start(), m.end(), m.group(1), "date range", "Date Range"))

            # 2. Century (e.g. 18th century, 21st century)
            for m in CENTURY_RE.finditer(sent):
                temp_spans.append((m.start(), m.end(), m.group(1), "century", "Century"))

            # 3. Relative time (e.g. mid-1945)
            for m in RELATIVE_TIME_RE.finditer(sent):
                temp_spans.append((m.start(), m.end(), m.group(1), "relative time", "Relative Time"))

            # 4. Standalone 4-digit years (outside any date range or relative time)
            for m in YEAR_RE.finditer(sent):
                s_pos, e_pos = m.start(), m.end()
                if any(ts <= s_pos and e_pos <= te for ts, te, _, _, _ in temp_spans):
                    continue
                temp_spans.append((s_pos, e_pos, m.group(1), "year", "Year"))

            for _, _, span_text, fg_label, display_label in temp_spans:
                cleaned = PUNCT_EDGE_RE.sub("", span_text).strip()
                if cleaned and not self._is_false_positive(cleaned, "TIME", fg_label):
                    mentions.append(
                        NERMentionEN(
                            surface=cleaned,
                            norm_surface=normalize_key(cleaned),
                            label="TIME",
                            label_name=display_label,
                            confidence=0.98 if fg_label == "year" else 0.95,
                            source="temporal_extractor",
                            sentence_index=idx,
                            fine_grained_label=fg_label,
                            votes={"TIME": 0.98 if fg_label == "year" else 0.95},
                        )
                    )
        return mentions

    # =========================================================
    # False-Positive & Noise Filtering (Hậu kỳ)
    # =========================================================
    _FP_SURFACE_BLACKLIST = {
        "a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "for",
        "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
        "it", "its", "they", "their", "them", "this", "that", "these", "those",
        "he", "she", "his", "her", "we", "our", "us", "you", "your",
        "which", "who", "whom", "whose", "what", "where", "when", "how",
        "not", "no", "nor", "but", "if", "so", "than", "too", "very",
        "can", "could", "will", "would", "shall", "should", "may", "might",
        "do", "does", "did", "has", "have", "had",
        "under", "within", "between", "through", "about", "into",
        # Adverbs, prepositions & false positive phrases
        "until", "after", "before", "during", "since", "while", "officially",
        "after which", "this day", "period", "become", "existed", "destructive",
        "systematic", "administrative", "day",
    }

    _COMMON_VERB_NOUN_FALSE_POSITIVES = {
        "builds", "build", "organizes", "organize", "implements", "implement",
        "coordinates", "coordinate", "manages", "manage", "preserves", "preserve",
        "promotes", "promote", "ensures", "ensure", "develops", "develop",
        "protects", "protect", "prevents", "prevent", "enables", "enable",
        "archives", "archive", "supports", "support", "introduces", "introduce",
        "related", "especially", "actively", "appropriately",
    }

    _GENERIC_FRAGMENT_WORDS = {
        # Administrative, area & complex
        "district", "ward", "site", "sites", "center", "relic", "relics",
        "heritage", "tangible", "intangible", "programs", "systems",
        "operations", "documentation", "professional", "management",
        "assets", "area", "central area", "city", "province", "landmark",
        "federation", "quarter", "old quarter", "complex",
        # Time & history
        "century", "centuries", "year", "years", "period", "21st", "18th", "17th", "19th",
        # Tourism architectural structures (Hanoi & Vietnam domain)
        "temple", "temples", "pagoda", "pagodas", "shrine", "shrines",
        "church", "churches", "cathedral", "cathedrals", "communal house",
        "citadel", "citadels", "fortress", "palace", "palaces", "pavilion", "pavilions",
        "gate", "gates", "tower", "towers", "wall", "walls", "bridge", "bridges",
        "monument", "monuments", "mausoleum", "mausoleums", "statue", "statues",
        "museum", "museums", "theatre", "theater", "opera house", "opera",
        "prison", "prisons", "market", "markets", "square", "squares",
        "academy", "university", "school", "hospital",
        # Geography & Water bodies (Hanoi lakes, rivers, islands)
        "lake", "lakes", "river", "rivers", "islet", "islets", "island", "islands",
        "pond", "ponds", "well", "wells", "bank", "banks", "shore", "shores", "dyke", "dykes",
    }

    @staticmethod
    def _is_false_positive(surface: str, label: str, fine_grained: str = "") -> bool:
        s = PUNCT_EDGE_RE.sub("", surface).strip()
        sl = s.lower()
        is_time = (label == "TIME") or (fine_grained in (
            "year", "date", "century", "date range", "time period", "historical period", "relative time"
        )) or (label in (
            "year", "date", "century", "date range", "time period", "historical period", "relative time"
        ))

        # 1. Punctuation only, empty, or too short
        if not s or len(s) <= 1 or not any(c.isalnum() for c in s):
            return True
        # 2. Subword token artefacts
        if s.startswith("##"):
            return True
        # 3. Pure digits: allow 4-digit years (e.g. 1802, 1805, 1831, 1873, 1888, 1945) for TIME
        if re.match(r"^\d+$", s):
            if is_time and len(s) == 4:
                return False
            return True
        # 4. Stopwords / pronouns / grammatical function words
        if sl in Module2NerConceptExtractorEN._FP_SURFACE_BLACKLIST:
            return True
        # 5. Prepositional phrases starting with 'to '
        if sl.startswith("to ") and len(sl.split()) <= 2:
            return True
        # 6. Verbal false-positives
        if sl in Module2NerConceptExtractorEN._COMMON_VERB_NOUN_FALSE_POSITIVES or sl in (
            "studied", "established", "opened", "closed", "founded", "built", "demolished"
        ):
            return True
        # 7. Post-processing: Filter all-lowercase common nouns for Named Entities (LOC, ORG, PER, EVENT, HERITAGE, ARTIFACT)
        # TIME, ATTR, and Visual Entities (bridge, temple, lake, water, tree, railing, gate, roof, building) are allowed to be lowercase
        _VISUAL_OBJECT_CLASSES = {
            "bridge", "temple", "temple or pagoda", "lake", "water", "tree",
            "railing", "person", "gate", "roof", "building", "mountain", "river",
            "temples", "bridges", "trees", "gates", "roofs", "buildings", "lakes"
        }
        if s.islower() and not is_time and label not in ("ATTR",) and fine_grained not in _VISUAL_OBJECT_CLASSES and sl not in _VISUAL_OBJECT_CLASSES:
            return True

        # 7b. Specific filtering for 'title or rank' to prevent common noun pollution
        if fine_grained == "title or rank":
            _non_title_words = {
                "temple", "temples", "studied", "established", "here", "other",
                "times", "capacity", "members", "elite", "doctors", "century",
                "centuries", "year", "years", "pavilions", "halls", "statues",
                "stelae", "ceremonies", "sessions", "exams", "gifts", "decorations",
                "styles", "relics", "restorations", "campaigns", "parts",
            }
            if sl in _non_title_words:
                return True
            if s.islower() and sl not in {
                "duke", "marquis", "count", "baron", "prince", "princes",
                "emperor", "emperors", "king", "kings", "queen", "queens",
                "lord", "lords", "lady", "ladies", "general", "generals",
                "marshal", "marshals", "minister", "ministers", "scholar",
                "scholars", "philosopher", "philosophers", "mandarin", "mandarins",
                "hầu", "công", "bá", "tử", "nam", "vương", "thái tử", "hoàng đế"
            }:
                return True

        # 8. Filter standalone generic common noun "map"
        if sl == "map":
            return True
        # 9. Very short all-lowercase function words
        if len(s) <= 3 and s.islower() and sl not in ("war", "bay", "sea", "law"):
            return True

        return False

    _INTRODUCTORY_WORDS = {
        "flanking", "surrounding", "bordering", "facing", "overlooking",
        "alongside", "behind", "near", "amidst", "within", "throughout"
    }

    def _merge_ner_mentions(self, mentions: List[NERMentionEN]) -> List[Dict[str, Any]]:
        cleaned_mentions = []
        for m in mentions:
            surf = m.surface.strip()
            words = surf.split()
            # Strip introductory participles/prepositions from surface
            if len(words) > 1 and words[0].lower() in self._INTRODUCTORY_WORDS:
                surf = " ".join(words[1:]).strip()
            # Strip trailing unmatched bracket/parenthesis artifact e.g. "60 m ...), Hanoi Flag"
            if "), " in surf:
                surf = surf.split("), ")[0].strip()

            if surf != m.surface:
                m = NERMentionEN(
                    surface=surf,
                    norm_surface=normalize_key(surf),
                    label=m.label,
                    label_name=m.label_name,
                    fine_grained_label=m.fine_grained_label,
                    confidence=m.confidence,
                    source=m.source,
                    sentence_index=m.sentence_index,
                    votes=m.votes,
                )
            cleaned_mentions.append(m)

        valid_mentions = [
            m for m in cleaned_mentions
            if not self._is_false_positive(m.surface, m.label, m.fine_grained_label)
        ]

        grouped = defaultdict(list)
        for m in valid_mentions:
            grouped[m.norm_surface].append(m)

        final_mentions = []
        for norm_surface, group in grouped.items():
            label_scores = defaultdict(float)
            for m in group:
                label_scores[m.label] += m.confidence

            # Voting for the best parent category (LOC, ORG, HERITAGE, etc.)
            best_label = max(label_scores.items(), key=lambda x: x[1])[0]

            best_mention = sorted(
                group,
                key=lambda x: (x.confidence, len(x.surface)),
                reverse=True,
            )[0]

            # Syntactic Head Noun Semantic Enforcement (Rightmost noun determines category in English compounds):
            # A building/structure named after a person (e.g. Ho Chi Minh Mausoleum, Lenin's Mausoleum) is a LOC, not a PER.
            last_token = best_mention.surface.lower().split()[-1].rstrip(".,;:!?\"')")
            if last_token in {
                "mausoleum", "tomb", "monument", "memorial", "pagoda", "temple",
                "shrine", "citadel", "fortress", "palace", "tower", "bridge",
                "gate", "square", "plaza", "cathedral", "church", "museum",
                "hall", "pavilion", "sanctuary", "building", "complex", "quarter",
                "street", "road", "avenue", "lane", "alley", "district", "ward",
                "city", "province", "village", "lake", "river", "canal", "mountain",
                "hill", "island", "cave", "bay", "portico", "colonnade", "facade",
                "façade", "platform", "dome", "pillar", "arch", "gallery", "wall",
                "courtyard", "terrace", "roof", "aisle"
            }:
                best_label = "LOC"
            elif last_token in {
                "death", "birth", "ceremony", "festival", "battle", "war", "campaign",
                "conference", "exhibition", "celebration", "reform", "coup", "coronation",
                "anniversary", "revolution", "rebellion", "uprising", "siege"
            }:
                best_label = "EVENT"
            elif last_token in {
                "declaration", "decree", "treaty", "law", "act", "constitution",
                "convention", "accord", "proclamation", "order", "edict", "designation"
            }:
                best_label = "HERITAGE"
            elif last_token in {
                "statue", "stela", "stele", "relic", "bell", "drum", "urn", "throne",
                "sword", "crown", "seal", "painting", "sculpture", "plaque", "canon", "cannon"
            }:
                best_label = "ARTIFACT"
            elif last_token in {"style", "order"} and any(
                w in best_mention.surface.lower() for w in ("architectural", "gothic", "baroque", "colonial", "traditional")
            ):
                best_label = "STYLE"

            avg_conf = sum(x.confidence for x in group) / len(group)
            sources = sorted(set(x.source for x in group))

            final_mentions.append(
                {
                    "surface": best_mention.surface,
                    "norm_surface": norm_surface,
                    "label": best_label,
                    "label_name": best_mention.label_name,
                    "fine_grained_label": best_mention.fine_grained_label,
                    "confidence": round(avg_conf, 4),
                    "sources": sources,
                    "votes": dict(label_scores),
                }
            )

        # Resolve sub-token fragments when full entity is present:
        surfaces_lower = [m["surface"].lower() for m in final_mentions]
        filtered_mentions = []

        for m in final_mentions:
            sl = m["surface"].lower()
            s_raw = m["surface"]

            # If it's a generic word and appears as a strict substring of a longer entity
            if sl in self._GENERIC_FRAGMENT_WORDS:
                if any(sl in other and sl != other for other in surfaces_lower):
                    continue
            core_words = [w for w in sl.split() if w not in {"the", "a", "an", "with", "at", "in", "on", "of", "to", "from", "by", "for"}]
            if len(core_words) == 1 and core_words[0] in self._GENERIC_FRAGMENT_WORDS:
                if any(core_words[0] in other and core_words[0] != other for other in surfaces_lower):
                    continue

            # If it's a short Vietnamese syllable fragment (e.g. "Co", "Loa", "Long", "Trấn", "Quốc", "Thăng")
            if s_raw in ("Co", "Loa", "Long", "Trấn", "Quốc", "Bắc", "Khai", "Thăng", "Hoài", "Đức", "Tây", "Kim", "Ngư", "Miếu", "Giám"):
                if any(s_raw in other and s_raw != other for other in [x["surface"] for x in final_mentions]):
                    continue

            # If it's a single word entity strictly contained in a multi-token entity of matching or broader type
            if " " not in s_raw and any(
                re.search(rf"\b{re.escape(sl)}\b", other) and sl != other
                for other in surfaces_lower
            ):
                if sl in self._GENERIC_FRAGMENT_WORDS or sl in (
                    "citadel", "complex", "wall", "gate", "tower", "city", "province", "federation", "21st", "18th", "century"
                ):
                    continue

            filtered_mentions.append(m)

        # Rule: If a 4-digit year appears as an endpoint of an extracted date range (e.g. 1885 in 1885–1954),
        # suppress the individual year so only the date range (Khoảng thời gian) is kept
        range_years = set()
        for m in filtered_mentions:
            if m.get("fine_grained_label") in ("date range", "time period") or "–" in m["surface"] or "-" in m["surface"]:
                for yr in YEAR_RE.findall(m["surface"]):
                    range_years.add(yr)

        filtered_mentions = [
            m for m in filtered_mentions
            if not (m.get("fine_grained_label") == "year" and m["surface"] in range_years)
        ]

        filtered_mentions.sort(
            key=lambda x: (x["confidence"], len(x["surface"])),
            reverse=True,
        )

        return filtered_mentions



    # =========================================================
    # Candidate Phrase Extraction (Exact Mirror of Code Gốc)
    # =========================================================
    def _extract_candidate_phrases(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extracts candidate domain concepts using:
        1. cleaned_keywords from Module 1
        2. POS noun phrases from full_text_pos (must end with a NOUN, no standalone adjectives)
        3. Title n-grams (clean segments split on delimiters)
        """
        title = normalize_text(record.get("title", ""))
        cleaned_keywords = parse_list_like(record.get("cleaned_keywords", []))
        full_pos = record.get("full_text_pos", [])
        ner_entities = record.get("ner_entities", [])

        entity_norms = {
            normalize_key(e.get("surface", ""))
            for e in ner_entities
            if isinstance(e, dict) and e.get("surface")
        }

        candidates: List[Dict[str, Any]] = []

        # 1. cleaned_keywords
        for kw in cleaned_keywords:
            norm_kw = normalize_key(kw)
            if not norm_kw:
                continue
            if norm_kw in entity_norms:
                continue
            candidates.append(
                {
                    "phrase": kw,
                    "source": "cleaned_keywords",
                    "evidence": "cleaned_keywords",
                }
            )

        # 2. POS Noun Phrases from full_text_pos
        if isinstance(full_pos, list) and full_pos:
            current: List[Tuple[str, str]] = []

            def flush_np():
                nonlocal current
                if not current:
                    return

                # A noun phrase MUST end with a NOUN (NN, NNS, NNP, NNPS).
                if not current[-1][1].startswith("NN"):
                    current = []
                    return

                toks = [normalize_text(x[0]) for x in current if normalize_text(x[0])]
                if 1 <= len(toks) <= self.max_concept_tokens:
                    phrase = normalize_text(" ".join(toks))
                    norm_phrase = normalize_key(phrase)
                    if norm_phrase and norm_phrase not in entity_norms:
                        candidates.append(
                            {
                                "phrase": phrase,
                                "source": "pos_np",
                                "evidence": "full_text_pos",
                            }
                        )
                current = []

            for item in full_pos:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue

                tok = normalize_text(item[0])
                pos = normalize_text(item[1])

                if not tok:
                    continue

                if tok.lower() in STOPWORDS_CONCEPT:
                    flush_np()
                    continue

                # Noun or Adjective pre-modifier
                if pos_is_nounlike(pos):
                    current.append((tok, pos))
                    if len(current) >= self.max_concept_tokens:
                        flush_np()
                elif pos.startswith("JJ") and not current:
                    current.append((tok, pos))
                elif pos.startswith("JJ") and current:
                    current.append((tok, pos))
                else:
                    flush_np()

            flush_np()

        # 3. Title N-grams (Clean segments split on delimiters)
        title_segments = re.split(r"[\s–—\-:;,|/]+", title)
        title_tokens = [x for x in title_segments if x and x.lower() not in STOPWORDS_CONCEPT]

        for n in range(2, min(5, len(title_tokens)) + 1):
            for i in range(0, len(title_tokens) - n + 1):
                sub_tokens = title_tokens[i: i + n]
                if sub_tokens[0].lower() in STOPWORDS_CONCEPT or sub_tokens[-1].lower() in STOPWORDS_CONCEPT:
                    continue
                phrase = " ".join(sub_tokens)
                norm_phrase = normalize_key(phrase)
                if norm_phrase and norm_phrase not in entity_norms:
                    candidates.append(
                        {
                            "phrase": phrase,
                            "source": "title_ngram",
                            "evidence": "title",
                        }
                    )

        # Deduplicate preserving order
        seen = set()
        deduped = []
        for c in candidates:
            k = normalize_key(c["phrase"])
            if k and k not in seen:
                seen.add(k)
                deduped.append(c)

        return deduped

    # =========================================================
    # Concept Scoring (Exact Formula from Code Gốc)
    # =========================================================
    def _score_concept(self, phrase: str, record: Dict[str, Any]) -> float:
        phrase_l = phrase.lower()
        full_text = normalize_text(record.get("full_text", "")).lower()
        title = normalize_text(record.get("title", "")).lower()
        summary = normalize_text(record.get("summary", "")).lower()
        cleaned_keywords = [normalize_text(x).lower() for x in parse_list_like(record.get("cleaned_keywords", []))]

        count = full_text.count(phrase_l)
        tf_score = min(count / 3.0, 1.0)

        title_boost = 1.0 if phrase_l in title else 0.0
        summary_boost = 0.5 if phrase_l in summary else 0.0
        keyword_boost = 1.0 if phrase_l in cleaned_keywords else 0.0
        length_boost = min(len(phrase.split()) / 4.0, 1.0)

        score = (
            0.30 * tf_score
            + 0.20 * length_boost
            + 0.20 * title_boost
            + 0.10 * summary_boost
            + 0.20 * keyword_boost
        )

        return round(max(0.0, min(score, 1.0)), 4)

    # =========================================================
    # Concept Linking (Exact Mirror of Code Gốc)
    # =========================================================
    def _link_concept(self, phrase: str) -> Dict[str, Any]:
        phrase_emb = self._embed_text(phrase)
        if phrase_emb is not None:
            best_label = None
            best_sim = 0.0

            for item in self.ontology_concepts:
                label = normalize_text(item.get("label", item.get("name", "")))
                aliases = item.get("aliases", []) or item.get("alias", []) or []
                all_names = [label] + parse_list_like(aliases)

                for name in all_names:
                    name = normalize_text(name)
                    if not name:
                        continue
                    key = normalize_key(name)
                    if key in self._ontology_embed_cache:
                        name_emb = self._ontology_embed_cache[key]
                    else:
                        name_emb = self._embed_text(name)
                        if name_emb is not None:
                            self._ontology_embed_cache[key] = name_emb

                    sim = self._cosine_similarity(phrase_emb, name_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_label = item.get("id", label)

            if best_sim >= self.concept_link_threshold:
                action = "link"
            elif best_sim >= self.concept_review_threshold:
                action = "review"
            else:
                action = "new"

            return {
                "linked_concept": best_label,
                "similarity": round(best_sim, 4),
                "action": action,
            }

        # Fallback lexical overlap (Jaccard similarity)
        phrase_norm = normalize_key(phrase)
        best_label = None
        best_sim = 0.0

        for item in self.ontology_concepts:
            label = normalize_text(item.get("label", item.get("name", "")))
            aliases = item.get("aliases", []) or item.get("alias", []) or []
            all_names = [label] + parse_list_like(aliases)

            for name in all_names:
                sim = jaccard_similarity(phrase_norm, name)
                if sim > best_sim:
                    best_sim = sim
                    best_label = item.get("id", label)

        if best_sim >= self.concept_link_threshold:
            action = "link"
        elif best_sim >= self.concept_review_threshold:
            action = "review"
        else:
            action = "new"

        return {
            "linked_concept": best_label,
            "similarity": round(best_sim, 4),
            "action": action,
        }

    def _extract_concepts(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates = self._extract_candidate_phrases(record)
        out = []

        for c in candidates:
            phrase = c["phrase"]
            score = self._score_concept(phrase, record)
            if score < self.concept_min_score:
                continue

            link_info = self._link_concept(phrase)
            out.append(
                ConceptMentionEN(
                    concept=phrase,
                    norm_concept=normalize_key(phrase),
                    score=score,
                    source=c["source"],
                    linked_ontology_id=link_info["linked_concept"],
                    similarity=link_info["similarity"],
                    action=link_info["action"],
                ).to_dict()
            )

        out.sort(key=lambda x: x["score"], reverse=True)
        return out[: self.max_concepts]

    # =========================================================
    # Main Pipeline Execution
    # =========================================================
    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        full_text = record.get("full_text", "")
        sentences = record.get("full_text_sentences", [])
        if not sentences and full_text:
            sentences = [full_text]

        all_mentions: List[NERMentionEN] = []

        # 1. GLiNER NER (Zero-shot with domain-specific fine-grained labels)
        if self.use_gliner and self.gliner_model is not None:
            gliner_entities = self._extract_gliner_ner(sentences)
            for idx, sent_ents in enumerate(gliner_entities):
                for ent in sent_ents:
                    surface = normalize_text(ent.get("text", ""))
                    raw_label = ent.get("label", "").lower().strip()
                    score = float(ent.get("score", 0.90))

                    if not surface:
                        continue

                    # Clean leading determiners/connectors, trailing connectors, and edge punctuation
                    cleaned_surface = LEADING_CONNECTOR_RE.sub("", surface).strip()
                    cleaned_surface = TRAILING_CONNECTOR_RE.sub("", cleaned_surface).strip()
                    cleaned_surface = POSSESSIVE_RE.sub("", cleaned_surface).strip()
                    cleaned_surface = PUNCT_EDGE_RE.sub("", cleaned_surface).strip()
                    if not cleaned_surface:
                        continue

                    # Map fine-grained label to unified ontology class and display name
                    if raw_label in LABEL_MAPPING:
                        onto_code, label_name = LABEL_MAPPING[raw_label]
                    else:
                        onto_code = raw_label.upper()
                        label_name = raw_label.title()

                    if onto_code in VALID_NER_LABELS:
                        all_mentions.append(
                            NERMentionEN(
                                surface=cleaned_surface,
                                norm_surface=normalize_key(cleaned_surface),
                                label=onto_code,
                                label_name=label_name,
                                confidence=score,
                                source="gliner",
                                sentence_index=idx,
                                fine_grained_label=raw_label,
                                votes={onto_code: score},
                            )
                        )

        # 2. Comprehensive English Temporal Extraction (Years, Ranges, Centuries, Relative Time)
        temporal_mentions = self._extract_temporal_entities(sentences)
        all_mentions.extend(temporal_mentions)

        # Merge mentions & filter false positives
        ner_entities = self._merge_ner_mentions(all_mentions)

        # 2. Concept Extraction
        enriched_record = dict(record)
        enriched_record["ner_entities"] = ner_entities
        concept_mentions = self._extract_concepts(enriched_record)

        return {
            "article_id": record.get("article_id", ""),
            "title": record.get("title", ""),
            "summary": record.get("summary", ""),
            "content": record.get("content", ""),
            "publish_time": record.get("publish_time"),
            "topic_list": record.get("topic_list", []),
            "cleaned_keywords": parse_list_like(record.get("cleaned_keywords", [])),
            "full_text": full_text,
            "full_text_sentences": sentences,
            "full_text_pos": record.get("full_text_pos", []),
            "ner_entities": ner_entities,
            "concept_mentions": concept_mentions,
            "entity_surface_forms": dedupe_preserve([e["surface"] for e in ner_entities]),
            "concept_surface_forms": dedupe_preserve([c["concept"] for c in concept_mentions]),
            "stats": {
                "n_entities": len(ner_entities),
                "n_concepts": len(concept_mentions),
            },
        }

    def process_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.process_record(r) for r in records]

    def process_jsonl(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> int:
        in_p = Path(input_path)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        records = []
        with in_p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        results = self.process_records(records)
        with out_p.open("w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        LOGGER.info("Saved %d articles to %s", len(results), out_p)
        return len(results)
