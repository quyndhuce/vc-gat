from __future__ import annotations

import ast
import html
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

# =========================================================
# Logging
# =========================================================
LOGGER = logging.getLogger("module1_preprocess_en")
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
WHITESPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
QUOTE_RE = re.compile(r"^[\'\"“”‘’]+|[\'\"“”‘’]+$")
BRACKET_RE = re.compile(r"^[\[\(\{]\s*|\s*[\]\)\}]$")

REQUIRED_COLUMNS = [
    "title",
    "summary",
    "content",
    "publish_time",
    "topic",
    "cleaned_keywords",
]

OPTIONAL_COLUMNS = [
    "article_id",
]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def normalize_text_field(value: Any, preserve_newlines: bool = False) -> str:
    """Normalize text field: HTML unescape, Unicode NFC, remove control chars & HTML tags."""
    if _is_missing(value):
        return ""
    text = str(value)
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = CONTROL_RE.sub("", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if preserve_newlines:
        lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
        return "\n".join(lines).strip()
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_keyword_item(value: Any) -> str:
    text = normalize_text_field(value)
    if not text:
        return ""
    text = QUOTE_RE.sub("", text)
    text = BRACKET_RE.sub("", text)
    text = text.strip(" ,;:|/\\")
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_topic_item(value: Any) -> str:
    text = normalize_text_field(value)
    if not text:
        return ""
    text = QUOTE_RE.sub("", text).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    return WHITESPACE_RE.sub("_", text)


def parse_list_like(value: Any) -> List[str]:
    if _is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed = None
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            items = list(parsed)
        elif parsed is not None and not isinstance(parsed, (dict, bytes)):
            items = [parsed]
        else:
            items = re.split(r"\s*[,;|]\s*", text)
    else:
        items = [value]

    cleaned = []
    for item in items:
        s = normalize_text_field(item)
        if s:
            cleaned.append(s)
    return cleaned


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        if item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


# =========================================================
# Pure Python English NLP Functions
# (Zero external NLTK or Transformer dependency for preprocessing)
# =========================================================

# English abbreviations that must not trigger false sentence splitting
_EN_ABBREVIATIONS = {
    # Titles & honorifics
    "mr", "mrs", "ms", "dr", "prof", "rev", "gen", "col", "capt", "lt", "sgt", "hon", "pres",
    # Academic & Latin
    "etc", "eg", "e.g", "ie", "i.e", "vs", "viz", "cf", "al", "ibid", "ca",
    # Administrative & Commercial
    "inc", "corp", "co", "ltd", "dept", "div", "no", "nos", "vol", "vols", "fig", "figs", "approx", "est",
    # Addresses & Geography
    "st", "ave", "blvd", "rd", "ct", "pl", "sq", "ln", "apt",
    # Months
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    # Acronyms & others
    "u.s", "u.k", "e.u", "u.n", "d.c", "a.m", "p.m", "b.c", "a.d",
}

_ABBREV_PATTERN = re.compile(r"\b([A-Za-z]\.|[A-Za-z]{2,6}\.)")


def split_sentences_en(text: str) -> List[str]:
    """
    Robust, rule-based English sentence segmentation without NLTK.
    Correctly protects abbreviations (e.g., 'No. 12 Nguyen Tri Phuong Street',
    'Dr. Smith', 'U.S.', 'etc.') and numbers from premature splitting.
    """
    text = normalize_text_field(text, preserve_newlines=True)
    if not text:
        return []

    # Clean Wikipedia reference citations like [3], [12], etc.
    text = re.sub(r"\[\d+\]", "", text)

    # Preserve multi-line paragraph boundaries
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    results: List[str] = []

    for para in paragraphs:
        # Step 1: Protect known abbreviations and single-letter initials
        def _protect_abbrev(match: re.Match) -> str:
            w = match.group(0).rstrip(".").lower()
            if w in _EN_ABBREVIATIONS or len(w) == 1:
                return match.group(0).replace(".", "___DOT___")
            return match.group(0)

        protected = _ABBREV_PATTERN.sub(_protect_abbrev, para)

        # Protect decimal numbers (e.g., 3.14 -> 3___DOT___14)
        protected = re.sub(r"(?<=\d)\.(?=\d)", "___DOT___", protected)

        # Step 2: Split on sentence termination punctuation (. ! ? …) followed by optional bracket/quote & whitespace
        raw_sents = re.split(r"(?<=[.!?…])\s+(?=[A-Z0-9“\"\'\(\[])|(?<=[.!?…][”\"\'\]\)])\s+(?=[A-Z0-9“\"\'\(\[])", protected)

        for sent in raw_sents:
            cleaned = sent.replace("___DOT___", ".").strip()
            if cleaned:
                results.append(cleaned)

    return results if results else [text]


def tokenize_en(text: str) -> List[str]:
    """
    Clean English word tokenization without NLTK.
    Extracts words, numbers, contractions/possessives, dashes, and punctuation.
    """
    text = normalize_text_field(text)
    if not text:
        return []

    token_pattern = re.compile(
        r"[A-Za-z0-9]+(?:[’\'][A-Za-z0-9]+)?|[–—\-]|[^\w\s]"
    )
    tokens = token_pattern.findall(text)
    return [t for t in tokens if t.strip()]


# =========================================================
# Rule & Lexicon-Based English POS Tagger (Penn Treebank Tagset)
# =========================================================
_DETERMINERS = {
    "the", "a", "an", "this", "that", "these", "those",
    "each", "every", "all", "both", "some", "any", "no",
    "neither", "either", "another", "such",
}

_PREPOSITIONS = {
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about",
    "as", "into", "through", "after", "over", "between", "out", "against",
    "during", "without", "before", "under", "around", "among", "within",
    "upon", "toward", "towards", "behind", "across", "along",
}

_CONJUNCTIONS = {"and", "but", "or", "nor", "for", "yet", "so"}

_PRONOUNS_PERSONAL = {
    "i", "me", "myself", "you", "yourselves", "yourself",
    "he", "him", "himself", "she", "her", "herself", "it", "itself",
    "we", "us", "ourselves", "they", "them", "themselves",
    "who", "whom", "whose", "which", "what",
}

_PRONOUNS_POSSESSIVE = {"my", "your", "his", "her", "its", "our", "their"}

_AUX_VERBS = {
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "done", "doing",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
}

_COMMON_VERBS = {
    "manage", "manages", "managed", "preserve", "preserves", "preserved",
    "promote", "promotes", "promoted", "ensure", "ensures", "ensured",
    "develop", "develops", "developed", "implement", "implements", "implemented",
    "formulate", "formulates", "formulated", "organize", "organizes", "organized",
    "encourage", "encourages", "encouraged", "protect", "protects", "protected",
    "prevent", "prevents", "prevented", "coordinate", "coordinates", "coordinated",
    "conduct", "conducts", "conducted", "collect", "collects", "collected",
    "identify", "identifies", "identified", "enable", "enables", "enabled",
    "build", "builds", "built", "archive", "archives", "archived",
    "support", "supports", "supported", "introduce", "introduces", "introduced",
    "relate", "relates", "related", "include", "includes", "included",
}

_ADJ_SUFFIXES = (
    "able", "ible", "al", "ful", "ic", "ical", "ive", "less",
    "ous", "ious", "ish", "ary", "ory", "thy",
)

_NOUN_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ance", "ence", "ity", "ism",
    "ist", "ship", "hood", "ure", "ery", "dom", "age",
)

_COMMON_ADJECTIVES = {
    "responsible", "worthy", "distinguished", "central", "tangible", "intangible",
    "cultural", "historical", "scientific", "technological", "international",
    "relevant", "archaeological", "appropriate", "master", "public", "private",
    "national", "regional", "local", "traditional", "original", "important",
}


def pos_tag_en(text: str) -> List[Tuple[str, str]]:
    """
    Model-free, high-precision English POS tagger based on Penn Treebank tags.
    Correctly distinguishes nouns (NN, NNS, NNP, NNPS), adjectives (JJ),
    verbs (VB*), prepositions (IN), determiners (DT), pronouns (PRP/PRP$), adverbs (RB), etc.
    """
    tokens = tokenize_en(text)
    if not tokens:
        return []

    tagged: List[Tuple[str, str]] = []
    n = len(tokens)

    for i, tok in enumerate(tokens):
        tl = tok.lower()
        prev_tok = tokens[i - 1].lower() if i > 0 else ""

        # 1. Punctuation
        if not re.search(r"\w", tok):
            tagged.append((tok, tok))
            continue

        # 2. Closed-class grammatical categories
        if tl in _DETERMINERS:
            tagged.append((tok, "DT"))
        elif tl in _PREPOSITIONS:
            tagged.append((tok, "IN"))
        elif tl in _CONJUNCTIONS:
            tagged.append((tok, "CC"))
        elif tl in _PRONOUNS_PERSONAL:
            tagged.append((tok, "PRP"))
        elif tl in _PRONOUNS_POSSESSIVE:
            tagged.append((tok, "PRP$"))
        elif tl in _AUX_VERBS:
            if tl in ("is", "has", "does"):
                tagged.append((tok, "VBZ"))
            elif tl in ("was", "were", "had", "did"):
                tagged.append((tok, "VBD"))
            elif tl in ("are", "have", "do"):
                tagged.append((tok, "VBP"))
            elif tl in ("can", "could", "will", "would", "shall", "should", "may", "might", "must"):
                tagged.append((tok, "MD"))
            else:
                tagged.append((tok, "VB"))

        # 3. Numeric
        elif re.match(r"^\d+(?:st|nd|rd|th)?$", tok):
            tagged.append((tok, "CD"))

        # 4. Contextual 3rd person singular verb after personal subject pronoun ("It coordinates", "It builds", etc.)
        elif prev_tok in ("it", "he", "she") and (tl.endswith("s") or tl in _COMMON_VERBS):
            tagged.append((tok, "VBZ"))

        # 5. Adverbs ending in -ly (e.g. "especially", "actively", "appropriately")
        elif tl.endswith("ly") and len(tl) > 3:
            tagged.append((tok, "RB"))

        # 6. Common verbs & verbal derivations
        elif tl in _COMMON_VERBS:
            if tl.endswith("s") and not tl.endswith("ss"):
                tagged.append((tok, "VBZ"))
            elif tl.endswith("ed"):
                tagged.append((tok, "VBN"))
            else:
                tagged.append((tok, "VB"))

        # 7. Participles / Verbs
        elif tl.endswith("ing") and len(tl) > 4:
            tagged.append((tok, "VBG"))
        elif tl.endswith("ed") and len(tl) > 4:
            tagged.append((tok, "VBN"))

        # 8. Adjectives (morphology + common adjective lexicon)
        elif tl in _COMMON_ADJECTIVES or tl.endswith(_ADJ_SUFFIXES):
            tagged.append((tok, "JJ"))

        # 9. Proper Nouns (capitalized, non-initial or preceded by capitalized)
        elif tok[0].isupper() and (i > 0 or n > 1 and tl not in ("the", "a", "an", "it", "this", "that")):
            if tl.endswith("s") and not tl.endswith("ss"):
                tagged.append((tok, "NNPS"))
            else:
                tagged.append((tok, "NNP"))

        # 10. Nouns (noun suffixes, plurals, or default)
        elif tl.endswith(_NOUN_SUFFIXES):
            tagged.append((tok, "NN"))
        elif tl.endswith("s") and not tl.endswith("ss"):
            tagged.append((tok, "NNS"))
        else:
            tagged.append((tok, "NN"))

    return tagged


def parse_publish_time(value: Any) -> Optional[str]:
    if _is_missing(value):
        return None
    text = normalize_text_field(value)
    if not text:
        return None
    candidates = [
        "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%B %d, %Y"
    ]
    for fmt in candidates:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.isoformat(sep=" ")
        except Exception:
            continue
    return text


def parse_cleaned_keywords(value: Any) -> List[str]:
    items = parse_list_like(value)
    normalized = []
    seen = set()
    for kw in items:
        item = normalize_keyword_item(kw)
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            normalized.append(item)
    return normalized


# =========================================================
# Data Model
# =========================================================
@dataclass
class ProcessedArticleEN:
    article_id: str
    title_raw: str
    summary_raw: str
    content_raw: str
    publish_time_raw: Any
    topic_raw: Any
    cleaned_keywords_raw: Any

    title: str = ""
    summary: str = ""
    content: str = ""
    full_text: str = ""
    publish_time: Optional[str] = None

    topic_list: List[str] = field(default_factory=list)
    cleaned_keywords: List[str] = field(default_factory=list)

    title_sentences: List[str] = field(default_factory=list)
    summary_sentences: List[str] = field(default_factory=list)
    content_sentences: List[str] = field(default_factory=list)
    full_text_sentences: List[str] = field(default_factory=list)

    title_tokens: List[str] = field(default_factory=list)
    summary_tokens: List[str] = field(default_factory=list)
    content_tokens: List[str] = field(default_factory=list)
    full_text_tokens: List[str] = field(default_factory=list)

    full_text_pos: List[Tuple[str, str]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================
# FastCoref Resolver (English Anaphora & Coreference)
# =========================================================
class FastCorefResolverEN:
    """Coreference Resolver using FCoref (fastcoref) with syntactic head/canonical selection."""
    PRONOUNS_AND_ANAPHORA = {
        "it", "its", "it.", "it,", "they", "them", "their", "he", "him", "his", "she", "her",
        "the tower", "the gate", "the structure", "the citadel", "the monument",
        "this tower", "this gate", "this structure", "the city", "this city",
        "the temple", "this temple", "the pagoda", "this pagoda", "the original pagoda",
        "the complex", "this complex", "the shrine", "this shrine"
    }

    # ── Long-text chunking parameters ──────────────────────
    # FCoref (RoBERTa-based) processes documents by segmenting internally,
    # but coreference quality degrades when segments are far apart.  We
    # split long texts into overlapping *semantic chunks* so that every
    # pronoun has its antecedent within the same chunk.
    #
    # MAX_CHUNK_WORDS : approximate words per chunk (RoBERTa ≈ 1.3 tokens/word)
    # OVERLAP_SENTS   : sentences of backward context carried into the next chunk
    _MAX_CHUNK_WORDS: int = 450           # ~585 subword tokens, optimal for 512-token local attention
    _OVERLAP_SENTS:   int = 5             # ~5 sentences of backward context carried into the next chunk
    _TOKENS_PER_WORD: float = 1.3         # English average for RoBERTa tokeniser

    _RELATIVE_PRONOUNS = {"who", "whom", "whose", "which", "that", "where", "when"}

    def __init__(
        self,
        device: str = "cpu",
        max_chunk_words: int = 450,
        overlap_sents: int = 5,
    ):
        self.device = device
        self.max_chunk_words = max_chunk_words
        self.overlap_sents = overlap_sents
        self.model = None
        try:
            from fastcoref import FCoref
            LOGGER.info("Loading FastCoref FCoref model on %s...", device)
            self.model = FCoref(device=device)
            LOGGER.info("✓ FastCoref model loaded successfully.")
        except Exception as e:
            LOGGER.warning("Could not load fastcoref: %s", e)

    @property
    def available(self) -> bool:
        return self.model is not None

    @classmethod
    def clean_antecedent_span(cls, span_text: str) -> str:
        """
        Rút gọn một mention dài về canonical / head noun:
        1. Bỏ chú thích trong ngoặc đơn
        2. Cắt bỏ relative clause (who, which, that...) và mệnh đề bổ ngữ sau dấu phẩy
        3. Cắt bỏ mạo từ ở đầu (the, a, an)
        """
        text = span_text.strip()
        # Bỏ chú thích trong ngoặc đơn e.g. "Diên Hựu tự (chữ Hán: 延祐寺)" -> "Diên Hựu tự"
        text = re.sub(r"\s*\([^)]*\)", "", text).strip()

        # Cắt tại dấu phẩy nếu phía sau là relative clause hoặc mệnh đề giải thích
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) > 1:
            first_part = parts[0]
            second_words = parts[1].lower().split()
            if second_words and second_words[0] in cls._RELATIVE_PRONOUNS:
                text = first_part
            elif any(w in cls._RELATIVE_PRONOUNS for w in text.lower().split()):
                text = first_part

        # Bỏ mạo từ đầu câu
        if text.lower().startswith("the "):
            text = text[4:].strip()
        elif text.lower().startswith("a "):
            text = text[2:].strip()
        elif text.lower().startswith("an "):
            text = text[3:].strip()

        return text.strip()

    @classmethod
    def select_best_canonical_mention(cls, str_cluster: List[str]) -> str:
        """
        Chọn mention tối ưu nhất trong cluster:
        - Bỏ đại từ và danh từ chỉ định anaphora
        - Ưu tiên mention có Proper Nouns (viết hoa)
        - Chọn mention ngắn gọn, súc tích nhất (loại bỏ mệnh đề dài)
        """
        candidates = []
        for t in str_cluster:
            raw = t.strip()
            if raw.lower() in cls.PRONOUNS_AND_ANAPHORA or len(raw) <= 2:
                continue
            cleaned = cls.clean_antecedent_span(raw)
            if cleaned and cleaned.lower() not in cls.PRONOUNS_AND_ANAPHORA:
                candidates.append(cleaned)

        if not candidates:
            return cls.clean_antecedent_span(str_cluster[0])

        def score(c: str) -> Tuple[bool, int, int]:
            words = c.split()
            has_proper = any(w[0].isupper() for w in words if w)
            return (not has_proper, len(words), len(c))

        candidates.sort(key=score)
        return candidates[0]

    # ── Core logic: extract replacements from FCoref predictions ──

    def _get_replacements(self, text: str) -> List[Tuple[int, int, str]]:
        """Run FCoref on *text* and return ``(start, end, replacement)`` tuples.

        This is the core anaphora-detection logic, separated from the
        text-mutation step so it can be reused by ``_resolve_chunked``.
        """
        preds = self.model.predict(texts=[text])
        pred = preds[0]
        raw_clusters = pred.get_clusters(as_strings=False)
        str_clusters = pred.get_clusters(as_strings=True)

        replacements: List[Tuple[int, int, str]] = []
        for spans, texts_cluster in zip(raw_clusters, str_clusters):
            best_head = self.select_best_canonical_mention(texts_cluster)
            if not best_head or len(best_head) < 3:
                continue

            # Chỉ thay thế khi best_head là thực thể hợp lệ (có Proper Noun viết hoa)
            words = best_head.split()
            has_capital = any(w[0].isupper() for w in words if w)
            if not has_capital:
                continue

            for (start, end), span_txt in zip(spans, texts_cluster):
                span_lower = span_txt.lower().strip()
                # Thay thế nếu là đại từ hoặc danh từ chỉ định anaphora
                is_anaphora = (
                    span_lower in self.PRONOUNS_AND_ANAPHORA
                    or (span_lower.startswith(("the ", "this ")) and span_lower != best_head.lower())
                )
                # Tránh thay thế nhầm chính bản thân tên đầy đủ
                if is_anaphora and span_lower != best_head.lower():
                    if span_txt.strip() != best_head:
                        replacements.append((start, end, best_head))

        return replacements

    @staticmethod
    def _apply_replacements(
        text: str,
        replacements: List[Tuple[int, int, str]],
    ) -> str:
        """Apply character-level replacements in reverse order."""
        if not replacements:
            return text
        replacements = sorted(replacements, key=lambda x: x[0], reverse=True)
        chars = list(text)
        for start, end, rep in replacements:
            chars[start:end] = list(rep)
        return "".join(chars)

    # ── Sentence-boundary detection with char offsets ──────

    @staticmethod
    def _find_sentence_spans(text: str) -> List[Tuple[int, int]]:
        """Return ``[(start, end), ...]`` character offsets of each sentence.

        Uses the same abbreviation-aware rules as ``split_sentences_en``
        but preserves original character positions for offset mapping.
        """
        sentences = split_sentences_en(text)
        spans: List[Tuple[int, int]] = []
        search_from = 0
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            idx = text.find(sent, search_from)
            if idx < 0:
                # Fuzzy fallback: sentence might have been slightly
                # normalised by split_sentences_en.  Try the first 40 chars.
                prefix = sent[:40]
                idx = text.find(prefix, search_from)
            if idx >= 0:
                spans.append((idx, idx + len(sent)))
                search_from = idx + len(sent)
            else:
                # Last resort: approximate position
                spans.append((search_from, search_from + len(sent)))
                search_from += len(sent)
        if not spans and text.strip():
            spans = [(0, len(text))]
        return spans

    # ── Chunked resolution for long texts ─────────────────

    def _resolve_chunked(self, text: str, title: Optional[str] = None) -> str:
        """Resolve coreference in long text using overlapping chunks.

        Algorithm
        ─────────
        1. Split *text* into sentences with character offsets.
        2. Group sentences into chunks of ≤ ``_MAX_CHUNK_WORDS`` words,
           with ``_OVERLAP_SENTS`` sentences of backward context.
        3. **Forward-chain**: process chunks left-to-right.  After
           resolving chunk *k*, the resolved text is spliced back into
           the working copy so that chunk *k+1*'s overlap region already
           contains entity names rather than pronouns.
        4. From each chunk, only accept replacements that fall in the
           *owned* (non-overlap) region to avoid double-replacing.

        Why forward-chaining?
        ─────────────────────
        A naïve independent-chunk approach misses antecedents resolved in
        earlier chunks.  Example:

            Chunk 1 resolves "It" → "Hoa Lư" in sentence 8.
            Chunk 2 overlap includes sentence 8 → FCoref sees "Hoa Lư"
            in the overlap and can link pronouns in sentences 12–20.

        The overlap sentences act as a **context bridge**, carrying resolved
        entity names forward.
        """
        sent_spans = self._find_sentence_spans(text)
        n_sents = len(sent_spans)

        if n_sents <= 1:
            return self._apply_replacements(text, self._get_replacements(text))

        # ── Determine how many sentences fit in one chunk ──────────
        word_counts = [
            len(text[s:e].split()) for s, e in sent_spans
        ]
        chunk_size_sents = n_sents          # default: all fit
        max_chunk = getattr(self, "max_chunk_words", self._MAX_CHUNK_WORDS)
        overlap_cfg = getattr(self, "overlap_sents", self._OVERLAP_SENTS)

        # ── Determine chunk boundaries based on word counts ───────
        chunks: List[Tuple[int, int, int]] = []
        pos = 0
        while pos < n_sents:
            # Accumulate sentences up to max_chunk words
            accum_words = 0
            end = pos
            while end < n_sents:
                wc = len(text[sent_spans[end][0]:sent_spans[end][1]].split())
                if accum_words + wc > max_chunk and end > pos:
                    break
                accum_words += wc
                end += 1

            # Ensure at least 1 sentence per chunk
            end = max(end, pos + 1)
            chunk_sents = end - pos
            overlap = min(overlap_cfg, max(1, chunk_sents // 3))

            owned_start = pos if not chunks else min(pos + overlap, end)
            chunks.append((pos, end, owned_start))
            if end >= n_sents:
                break
            pos = max(pos + 1, end - overlap)

        LOGGER.info(
            "Chunked coref: %d sentences → %d chunks (target_words=%d, overlap_sents=%d)",
            n_sents, len(chunks), max_chunk, overlap_cfg,
        )

        # ── Determine document lead anchor (e.g. Title or primary subject) ──
        lead_anchor = ""
        if title and len(title.strip()) >= 3:
            lead_anchor = self.clean_antecedent_span(title)
        if not lead_anchor and n_sents > 0:
            first_sent = text[sent_spans[0][0]:sent_spans[0][1]]
            # Extract first capitalised proper noun phrase as fallback anchor
            caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", first_sent)
            if caps and len(caps[0]) >= 3 and caps[0].lower() not in self.PRONOUNS_AND_ANAPHORA:
                lead_anchor = caps[0]

        # ── Forward-chain: process each chunk, splice resolved text ──
        working = text

        for chunk_idx, (s_start, s_end, owned_s_start) in enumerate(chunks):
            current_spans = self._find_sentence_spans(working)

            if len(current_spans) < s_end:
                s_end = len(current_spans)
                owned_s_start = min(owned_s_start, s_end)
            if s_start >= len(current_spans):
                break

            chunk_char_start = current_spans[s_start][0]
            chunk_char_end   = current_spans[s_end - 1][1]
            chunk_text       = working[chunk_char_start:chunk_char_end]

            owned_char_start = (
                current_spans[owned_s_start][0] if owned_s_start < len(current_spans)
                else chunk_char_end
            )

            # Context Anchoring: if lead_anchor is absent from this chunk,
            # prepend it as a synthetic sentence anchor to preserve global antecedent
            prefix = ""
            prefix_len = 0
            if lead_anchor and lead_anchor.lower() not in chunk_text.lower():
                prefix = f"{lead_anchor}. "
                prefix_len = len(prefix)

            eval_text = prefix + chunk_text

            try:
                raw_replacements = self._get_replacements(eval_text)
            except Exception as exc:
                LOGGER.warning(
                    "FCoref failed on chunk %d/%d: %s — skipping chunk",
                    chunk_idx + 1, len(chunks), exc,
                )
                continue

            # Un-shift prefix offsets & filter to owned region
            owned_replacements: List[Tuple[int, int, str]] = []
            for (r_start, r_end, rep) in raw_replacements:
                if r_start < prefix_len:
                    # Ignore any mutation inside synthetic anchor
                    continue
                c_start = r_start - prefix_len
                c_end   = r_end - prefix_len

                global_start = c_start + chunk_char_start
                global_end   = c_end + chunk_char_start

                if global_start >= owned_char_start:
                    owned_replacements.append((global_start, global_end, rep))

            if owned_replacements:
                working = self._apply_replacements(working, owned_replacements)

        return working

    # ── Public API ────────────────────────────────────────

    def resolve(self, text: str, title: Optional[str] = None) -> str:
        """Resolve anaphoric pronouns and demonstratives to canonical entity names.

        For short texts (≤ ``max_chunk_words`` words), processes in a single
        FCoref call.  For long texts (e.g. 50+ sentences), uses overlapping-chunk
        forward-chaining with lead-anchor context preservation.

        Parameters
        ----------
        text : str
            The input text (typically ``full_text`` or ``content``).
        title : Optional[str]
            Document title or primary subject, used as global antecedent anchor.

        Returns
        -------
        str
            Text with pronouns/demonstratives replaced by entity names.
        """
        if not self.available or not text or len(text.strip()) < 10:
            return text

        max_chunk = getattr(self, "max_chunk_words", self._MAX_CHUNK_WORDS)
        n_words = len(text.split())
        if n_words <= max_chunk:
            try:
                # Single pass: if title is available and not in text, anchor it
                eval_text = text
                prefix_len = 0
                if title and len(title.strip()) >= 3 and title.lower() not in text.lower():
                    clean_title = self.clean_antecedent_span(title)
                    prefix = f"{clean_title}. "
                    prefix_len = len(prefix)
                    eval_text = prefix + text

                reps = self._get_replacements(eval_text)
                if prefix_len > 0:
                    shifted_reps = [
                        (s - prefix_len, e - prefix_len, r)
                        for s, e, r in reps if s >= prefix_len
                    ]
                    return self._apply_replacements(text, shifted_reps)
                return self._apply_replacements(text, reps)
            except Exception as e:
                LOGGER.warning("FastCoref resolution failed: %s; returning original text.", e)
                return text

        # Long text — chunked resolution with forward-chaining and lead anchor
        LOGGER.info(
            "Long text detected (%d words > %d): engaging chunked coref resolution.",
            n_words, max_chunk,
        )
        try:
            return self._resolve_chunked(text, title=title)
        except Exception as e:
            LOGGER.warning("Chunked coref resolution failed: %s; returning original text.", e)
            return text


# =========================================================
# Module 1 Preprocessor Class
# =========================================================
class Module1PreprocessorEN:
    """
    Module 1: English Preprocessing Pipeline.
    Cleans and structures raw articles, generating sentences, tokens, POS tags,
    with optional FastCoref coreference resolution.
    """

    def __init__(
        self,
        use_pos: bool = True,
        use_coref: bool = False,
        coref_device: str = "cpu",
        include_summary_in_full_text: bool = True,
    ):
        self.use_pos = use_pos
        self.use_coref = use_coref
        self.include_summary_in_full_text = include_summary_in_full_text
        self.coref = FastCorefResolverEN(device=coref_device) if use_coref else None

    def _combine_full_text(self, title: str, summary: str, content: str) -> str:
        parts = [title]
        if self.include_summary_in_full_text and summary:
            parts.append(summary)
        if content:
            parts.append(content)
        return "\n\n".join([p for p in parts if p]).strip()

    def preprocess_row(self, row: Dict[str, Any], article_id: Optional[str] = None) -> ProcessedArticleEN:
        title = normalize_text_field(row.get("title", ""))
        summary = normalize_text_field(row.get("summary", ""))
        content = normalize_text_field(row.get("content", ""), preserve_newlines=True)
        publish_time = parse_publish_time(row.get("publish_time"))

        topic_raw = row.get("topic", "")
        cleaned_keywords_raw = row.get("cleaned_keywords", "")

        if article_id is None:
            article_id = str(row.get("article_id", "")).strip() or ""

        topic_list = parse_list_like(topic_raw)
        topic_list = [normalize_topic_item(t) for t in topic_list if normalize_topic_item(t)]
        topic_list = dedupe_preserve_order(topic_list)

        cleaned_keywords = parse_cleaned_keywords(cleaned_keywords_raw)

        full_text = self._combine_full_text(title, summary, content)

        # Coreference resolution if enabled
        if self.use_coref and self.coref and self.coref.available:
            full_text = self.coref.resolve(full_text, title=title)
            content = self.coref.resolve(content, title=title)

        title_sentences = split_sentences_en(title)
        summary_sentences = split_sentences_en(summary)
        content_sentences = split_sentences_en(content)
        full_text_sentences = split_sentences_en(full_text)

        title_tokens = tokenize_en(title)
        summary_tokens = tokenize_en(summary)
        content_tokens = tokenize_en(content)
        full_text_tokens = tokenize_en(full_text)

        # POS tagging on full_text used by Module 2
        full_text_pos = pos_tag_en(full_text) if self.use_pos else []

        stats = {
            "n_title_sentences": len(title_sentences),
            "n_summary_sentences": len(summary_sentences),
            "n_content_sentences": len(content_sentences),
            "n_full_text_sentences": len(full_text_sentences),
            "n_title_tokens": len(title_tokens),
            "n_summary_tokens": len(summary_tokens),
            "n_content_tokens": len(content_tokens),
            "n_full_text_tokens": len(full_text_tokens),
            "n_topics": len(topic_list),
            "n_keywords": len(cleaned_keywords),
        }

        return ProcessedArticleEN(
            article_id=article_id,
            title_raw=str(row.get("title", "")),
            summary_raw=str(row.get("summary", "")),
            content_raw=str(row.get("content", "")),
            publish_time_raw=row.get("publish_time"),
            topic_raw=topic_raw,
            cleaned_keywords_raw=cleaned_keywords_raw,
            title=title,
            summary=summary,
            content=content,
            full_text=full_text,
            publish_time=publish_time,
            topic_list=topic_list,
            cleaned_keywords=cleaned_keywords,
            title_sentences=title_sentences,
            summary_sentences=summary_sentences,
            content_sentences=content_sentences,
            full_text_sentences=full_text_sentences,
            title_tokens=title_tokens,
            summary_tokens=summary_tokens,
            content_tokens=content_tokens,
            full_text_tokens=full_text_tokens,
            full_text_pos=full_text_pos,
            stats=stats,
        )

    def process_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outputs = []
        for idx, rec in enumerate(records):
            aid = rec.get("article_id") or f"art_{idx+1:04d}"
            processed = self.preprocess_row(rec, article_id=aid)
            outputs.append(processed.to_dict())
        return outputs

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

        processed = self.process_records(records)
        with out_p.open("w", encoding="utf-8") as f:
            for item in processed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        LOGGER.info("Processed %d articles: %s -> %s", len(processed), in_p, out_p)
        return len(processed)
