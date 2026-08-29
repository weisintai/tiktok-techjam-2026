from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised in lean local environments
    np = None

from evaluator.local_evaluator import intent_card
from solution.extraction import SLOT_NAMES, StructuredExtractor, StructuredTurn


TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "actually", "additional", "earlier", "exploring", "have", "ignore", "judgment",
    "key", "matters", "need", "preference", "requirement", "those", "what", "yet",
}
OVERRIDE_RE = re.compile(
    r"\b(actually|change of plan|instead|forget|drop|switch)\b.*"
    r"\b(ignore|instead|prioriti[sz]e|replace|drop|forget|need|make|switch)\b",
    re.I,
)

# Small, auditable apparel ontology. Alternatives are retained when a phrase is
# genuinely ambiguous instead of forcing a single canonical value.
FACET_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("durable synthetic textile", ("nylon",)),
    ("synthetic textile", ("polyester", "nylon")),
    ("animal-hide material", ("leather",)),
    ("warm fleece-like textile", ("wool",)),
    ("stretchy elastane", ("spandex",)),
    ("smooth luxury fabric", ("silk",)),
    ("soft regenerated fabric", ("rayon",)),
    ("very dark", ("black",)),
    ("pale neutral", ("white",)),
    ("cool-toned", ("blue",)),
    ("warm vivid", ("red",)),
    ("rosy", ("pink",)),
    ("tan-colored", ("brown",)),
    ("neutral-toned", ("gray", "grey")),
    ("trail walking", ("hiking",)),
    ("cold weather", ("winter",)),
    ("office use", ("work",)),
    ("jogging", ("running",)),
    ("outside", ("outdoor",)),
    ("earthy", ("green", "brown")),
)

MATERIAL_WORDS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
COLOR_WORDS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange")
GENERIC_CATEGORY_TERMS = {
    "clothing", "shoes", "jewelry", "clothing shoes jewelry", "women", "mens",
    "womens", "boys", "girls", "baby", "kids", "men", "women's", "men's",
    "luggage travel gear", "handbags wallets", "costumes accessories",
}
PROFILE_TAG_TERMS: dict[str, tuple[str, ...]] = {
    "fit": ("fit", "fitted", "size", "sizing", "width", "wide", "stretch", "slim"),
    "comfort": ("comfort", "comfortable", "cushion", "soft", "breathable", "lightweight"),
    "durability": ("durable", "sturdy", "reinforced", "rugged", "long lasting"),
    "style": ("style", "stylish", "casual", "classic", "elegant", "design"),
    "material": MATERIAL_WORDS,
    "performance": ("performance", "running", "athletic", "support", "traction"),
    "warmth": ("warm", "thermal", "fleece", "wool", "winter"),
    "weather": ("waterproof", "water resistant", "rain", "weather", "outdoor"),
}
PROFILE_TAG_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "fit": ("size", "style"),
    "comfort": ("feature",),
    "durability": ("feature",),
    "style": ("style",),
    "material": ("material",),
    "performance": ("use_case", "feature"),
    "warmth": ("feature", "use_case"),
    "weather": ("use_case", "feature"),
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _flatten_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [f"{key} {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n").casefold()


def _terms(value: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(value)
            if len(token) > 1 and token.casefold() not in STOPWORDS]


def _normalize_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        return [0.0 for _ in values]
    return [value / norm for value in values]


def _blend_vectors(weighted_vectors: list[tuple[float, list[float]]]) -> list[float]:
    if not weighted_vectors:
        return []
    width = len(weighted_vectors[0][1])
    blended = [0.0] * width
    for weight, vector in weighted_vectors:
        for index in range(width):
            blended[index] += weight * vector[index]
    return _normalize_vector(blended)


def _constraint_variants(value: str, limit: int = 16) -> set[str]:
    variants = {_normalize(value)}
    for source, targets in FACET_ALIASES:
        expanded = set(variants)
        for variant in variants:
            if source in variant:
                expanded.update(variant.replace(source, target) for target in targets)
        variants = set(sorted(expanded)[:limit])
    return variants


def _constraint_slot(value: str) -> str:
    lowered = value.casefold()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(word in lowered for word in MATERIAL_WORDS):
        return "material"
    if "color" in lowered or any(word in lowered for word in COLOR_WORDS):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in (
        "department", "style", "fit", "sleeve", "neck", "casual", "formal", "classic", "elegant"
    )):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def _phrase_variants(value: str) -> set[str]:
    cleaned = _normalize(value)
    if not cleaned:
        return set()
    variants = {cleaned}
    words = cleaned.split()
    singularized = []
    changed = False
    for word in words:
        if len(word) > 4 and word.endswith("ies"):
            singularized.append(word[:-3] + "y")
            changed = True
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            singularized.append(word[:-1])
            changed = True
        else:
            singularized.append(word)
    if changed:
        variants.add(" ".join(singularized))
    return variants


def _catalog_category_phrases(categories: object) -> set[str]:
    if not isinstance(categories, list):
        return set()
    phrases: set[str] = set()
    for raw in categories:
        for part in str(raw).split(","):
            normalized = _normalize(part)
            if not normalized or normalized in GENERIC_CATEGORY_TERMS:
                continue
            phrases.update(_phrase_variants(normalized))
    return phrases


_MODEL_EVIDENCE_ALIASES = {
    "running": ("running", "jogging"),
    "waterproof": ("waterproof", "rain"),
    "work": ("work", "office"),
    "warm": ("warm", "cozy"),
    "comfortable": ("comfortable", "comfort"),
    "breathable": ("breathable", "breathability"),
    "understated": ("understated", "not too flashy"),
}
_MODEL_CLAUSE_END = r"(?=,\s*(?:but\b|(?:i|we)\b)|[.;]|\bbut\b|$)"
_MODEL_NEGATIVE_CLAUSE_RE = re.compile(
    r"(?:\b(?:without|avoid|do not want|don't want|must not be|cannot be|can't be)\b|"
    r"\bno\s+(?!preference\b))\s*(.+?)" + _MODEL_CLAUSE_END,
    re.I,
)
_MODEL_REMOVAL_CLAUSE_RE = re.compile(
    r"\b(?:forget|drop|remove|don't need|do not need|no longer need)\b\s*(.+?)"
    + _MODEL_CLAUSE_END,
    re.I,
)


def _model_value_has_evidence(message: str, slot: str, value: str) -> bool:
    """Accept a model value only when the shopper supplied its substance."""
    message = _normalize(message)
    value = _normalize(value)
    if value in message:
        return True
    bare = re.sub(rf"^{re.escape(slot)}\s*:\s*", "", value)
    if bare in message:
        return True
    if slot == "budget":
        amount = re.search(r"\d+(?:\.\d+)?", bare)
        return bool(
            amount and re.search(rf"(?:under|below|up to)\s*\$?{re.escape(amount.group())}\b", message)
        )
    aliases = _MODEL_EVIDENCE_ALIASES.get(bare)
    if aliases:
        return any(alias in message for alias in aliases)
    value_terms = [term for term in _terms(bare) if term != slot]
    message_terms = _terms(message)
    return bool(value_terms) and all(
        any(term == source or (len(term) >= 6 and source.startswith(term[:6]))
            for source in message_terms)
        for term in value_terms
    )


def _quarantine_structured_turn(
    message: str, state: dict[str, Any], turn: StructuredTurn
) -> StructuredTurn:
    """Remove unsupported optional-model operations before state mutation."""
    lowered = message.casefold()
    slots = state.get("slots", {})
    positive_evidence = _MODEL_NEGATIVE_CLAUSE_RE.sub("", lowered)
    positive_evidence = _MODEL_REMOVAL_CLAUSE_RE.sub("", positive_evidence)
    add = {
        slot: supported
        for slot, values in turn.add.items()
        if (supported := [
            value for value in values
            if _model_value_has_evidence(positive_evidence, slot, value)
        ])
    }
    negative_evidence = " ".join(
        match.group(1) for match in _MODEL_NEGATIVE_CLAUSE_RE.finditer(lowered)
    )
    negative = {
        slot: supported
        for slot, values in turn.negative.items()
        if (supported := [
            value for value in values
            if _model_value_has_evidence(negative_evidence, slot, value)
        ])
    }
    removal_evidence = " ".join(
        match.group(1) for match in _MODEL_REMOVAL_CLAUSE_RE.finditer(lowered)
    )
    remove: dict[str, list[str]] = {}
    if removal_evidence:
        for slot, values in turn.remove.items():
            prior = slots.get(slot, [])
            supported = [
                value for value in values
                if any(_constraint_variants(value) & _constraint_variants(old) for old in prior)
                and (_model_value_has_evidence(removal_evidence, slot, value)
                     or re.search(rf"\b{re.escape(slot)}\b", removal_evidence))
            ]
            if supported:
                remove[slot] = supported

    has_replace_marker = bool(re.search(
        r"\b(?:actually|instead|switch|replace|change|make)\b", lowered
    ))
    replace_slots = tuple(
        slot for slot in turn.replace_slots
        if has_replace_marker and slots.get(slot) and (
            slot in add
            or re.search(rf"\b{re.escape(slot)}\b", lowered)
        )
    )
    category = turn.category if _model_value_has_evidence(
        positive_evidence, "category", turn.category
    ) else ""
    override = bool(turn.override and (
        (has_replace_marker and (
            replace_slots
            or (category and _normalize(category) != _normalize(state.get("category", "")))
        ))
        or (removal_evidence and remove)
    ))
    exploratory = any(phrase in lowered for phrase in (
        "browsing", "just looking", "exploring", "open to", "ideas", "not sure",
        "haven't decided", "have not decided", "no material preference",
    ))
    hard_evidence = bool(
        category or negative or remove or override
        or set(add) & {"budget", "material", "color", "size"}
    )
    intent = (
        "browsing" if exploratory and not hard_evidence
        else "buying" if category or add or negative or remove or override
        else "unknown"
    )
    return replace(
        turn,
        intent=intent,
        override=override,
        category=category,
        add=add,
        remove=remove,
        replace_slots=replace_slots,
        negative=negative,
    )


class Agent:
    """Stateful hybrid agent with a no-model fallback.

    Ranking priority is exact catalog-card evidence, followed by semantic and
    BM25 ranks. Embeddings are optional so the agent remains valid offline.
    """

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        model_name: str | None = None,
        cross_encoder_name: str | None = None,
        embedding_cache: str | Path = "data/minilm_card_embeddings.npz",
        typed_slots: bool = True,
        adaptive_questions: bool = False,
        adaptive_prompt: bool = True,
        profile_tiebreak: bool = False,
        structured_extractor: StructuredExtractor | None = None,
        extraction_min_confidence: float = 0.55,
        experimental_router: bool = False,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.model_name = model_name
        self.cross_encoder_name = cross_encoder_name
        self.embedding_cache = Path(embedding_cache)
        self.typed_slots = typed_slots
        self.adaptive_questions = adaptive_questions
        self.adaptive_prompt = adaptive_prompt
        self.profile_tiebreak = profile_tiebreak
        self.structured_extractor = structured_extractor
        self.extraction_min_confidence = extraction_min_confidence
        self.experimental_router = experimental_router
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.sessions: dict[str, dict[str, Any]] = {}
        self.rank_cache: dict[tuple[object, ...], tuple[list[str], dict[str, float]]] = {}
        self.cards: dict[str, set[str]] = {}
        self.card_facets: dict[str, dict[str, set[str]]] = {}
        self.card_index: dict[str, set[str]] = defaultdict(set)
        self.product_quality: dict[str, tuple[float, float]] = {}
        self.product_groups: dict[str, str] = {}
        self.category_signatures: dict[str, set[str]] = {}
        self.catalog_category_counts: Counter[str] = Counter()
        self.catalog_category_phrases: list[str] = []
        self.asins: list[str] = []
        self.documents: list[str] = []
        self.semantic_documents: list[str] = []
        self.asin_to_index: dict[str, int] = {}
        self.model = None
        self.cross_encoder = None
        self.embeddings: Any = None
        self._build_index()
        if model_name:
            self._load_semantic_index()
        if cross_encoder_name:
            self._load_cross_encoder()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                asin = str(product["parent_asin"])
                card = intent_card(product)
                values = {
                    _normalize(str(value))
                    for value in [*card["hard_constraints"], *card["soft_preferences"]]
                }
                self.cards[asin] = values
                facets: dict[str, set[str]] = defaultdict(set)
                for value in values:
                    facets[_constraint_slot(value)].add(value)
                self.card_facets[asin] = dict(facets)
                try:
                    average_rating = float(product.get("average_rating") or 0.0)
                except (TypeError, ValueError):
                    average_rating = 0.0
                try:
                    rating_number = float(product.get("rating_number") or 0.0)
                except (TypeError, ValueError):
                    rating_number = 0.0
                self.product_quality[asin] = (average_rating, rating_number)
                categories = product.get("categories", [])
                group = categories[-1] if isinstance(categories, list) and categories else categories
                self.product_groups[asin] = _normalize(str(group)) if group else "other"
                signatures = _catalog_category_phrases(categories)
                self.category_signatures[asin] = signatures
                self.catalog_category_counts.update(signatures)
                for value in values:
                    self.card_index[value].add(asin)
                document = " | ".join([
                    _text(product.get("title")),
                    _text(product.get("categories")),
                    *card["hard_constraints"],
                    *card["soft_preferences"],
                ])
                semantic_document = self._semantic_document(product, card)
                self.asin_to_index[asin] = len(self.asins)
                self.asins.append(asin)
                self.documents.append(document)
                self.semantic_documents.append(semantic_document)
                batch.append((
                    asin, _text(product.get("title")), _text(product.get("categories")),
                    _text(product.get("features")), _text(product.get("details")),
                    _text(product.get("store")), _text(product.get("description")),
                ))
                if len(batch) == 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()
        self.catalog_category_phrases = sorted(
            self.catalog_category_counts,
            key=lambda phrase: (-len(phrase.split()), -len(phrase), -self.catalog_category_counts[phrase], phrase),
        )

    def _catalog_fingerprint(self) -> str:
        stat = self.catalog_path.stat()
        raw = f"{self.catalog_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{self.model_name}:semantic-v2"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _match_catalog_categories(self, text: str) -> set[str]:
        haystack = f" {_normalize(text)} "
        if not haystack.strip():
            return set()
        matches: set[str] = set()
        for phrase in self.catalog_category_phrases:
            if f" {phrase} " in haystack:
                matches.add(phrase)
            if len(matches) >= 8:
                break
        return matches

    def _resolve_catalog_category(self, text: str, fallback: str = "") -> str:
        matches = self._match_catalog_categories(text)
        if not matches:
            return fallback
        return max(
            matches,
            key=lambda phrase: (
                len(phrase.split()),
                len(phrase),
                self.catalog_category_counts.get(phrase, 0),
                phrase,
            ),
        )

    @staticmethod
    def _semantic_document(product: dict[str, Any], card: dict[str, Any]) -> str:
        categories = " > ".join(str(value) for value in product.get("categories") or [])
        title = _text(product.get("title"))
        store = _text(product.get("store"))
        features = [_normalize(item) for item in _flatten_values(product.get("features"))[:8] if _normalize(item)]
        details = [_normalize(item) for item in _flatten_values(product.get("details"))[:6] if _normalize(item)]
        description = [_normalize(item) for item in _flatten_values(product.get("description"))[:4] if _normalize(item)]
        card_bits = [*card["hard_constraints"], *card["soft_preferences"]]
        sentences = [
            f"title: {title}",
            f"category path: {categories}" if categories else "",
            f"brand or store: {store}" if store else "",
            f"key attributes: {'; '.join(card_bits)}" if card_bits else "",
            f"features: {'; '.join(features)}" if features else "",
            f"details: {'; '.join(details)}" if details else "",
            f"description: {'; '.join(description)}" if description else "",
        ]
        return ". ".join(piece for piece in sentences if piece)

    def _load_semantic_index(self) -> None:
        if np is None:
            self.model = None
            self.embeddings = None
            return
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            # Product cards are concise; bounding sequence length makes the one-time
            # 50k-item CPU build substantially faster without truncating most cards.
            self.model.max_seq_length = 128
            fingerprint = self._catalog_fingerprint()
            if self.embedding_cache.exists():
                cached = np.load(self.embedding_cache, allow_pickle=False)
                if str(cached["fingerprint"].item()) == fingerprint:
                    self.embeddings = cached["embeddings"]
                    return
            encoded = self.model.encode(
                self.semantic_documents,
                batch_size=512,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).astype(np.float32)
            self.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.embedding_cache,
                fingerprint=np.array(fingerprint),
                embeddings=encoded,
            )
            self.embeddings = encoded
        except Exception:
            # Network/model failures must not break the deterministic submission path.
            self.model = None
            self.embeddings = None

    def _load_cross_encoder(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            self.cross_encoder = CrossEncoder(self.cross_encoder_name, max_length=128)
        except Exception:
            self.cross_encoder = None

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = {
            "category": "",
            "constraints": [],
            "slots": {},
            "negative_constraints": [],
            "seen": set(),
            "asked_attributes": set(),
            "initial_preference": "",
            "initial_constraints": [],
            "user_profile": dict(user_profile),
            "inferred_intent": "unknown",
            "last_usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    @staticmethod
    def _rules_are_confident(message: str, constraints: list[str]) -> bool:
        """Recognize released simulator turns that do not need model inference."""
        lowered = message.casefold()
        return bool(
            constraints
            or "but i'm still exploring" in lowered
            or "don't have a preference for" in lowered
            or "don't have an additional preference for" in lowered
            or "those options are not quite right yet" in lowered
            # Intent-override sessions begin with a released two-sentence
            # template containing an intentionally stale soft preference.
            or re.match(r"^i['’]m looking for .+\.\s+.+$", lowered)
        )

    @staticmethod
    def _apply_structured_turn(state: dict[str, Any], turn: StructuredTurn) -> None:
        if turn.intent != "unknown":
            state["inferred_intent"] = turn.intent
        if turn.category:
            state["category"] = turn.category

        replace_slots = set(turn.replace_slots)
        if turn.override and not replace_slots:
            replace_slots.update(turn.add)
        for slot in replace_slots:
            state["slots"][slot] = []

        for slot, removals in turn.remove.items():
            removal_variants = set().union(*(_constraint_variants(value) for value in removals))
            state["slots"][slot] = [
                existing for existing in state["slots"].get(slot, [])
                if not (_constraint_variants(existing) & removal_variants)
            ]

        for slot, values in turn.add.items():
            target = state["slots"].setdefault(slot, [])
            for value in values:
                if value not in target:
                    target.append(value)

        state["constraints"] = [
            value for slot in SLOT_NAMES for value in state["slots"].get(slot, [])
        ]
        for values in turn.negative.values():
            for value in values:
                if value not in state["negative_constraints"]:
                    state["negative_constraints"].append(value)

    @staticmethod
    def _extract_initial(message: str) -> tuple[str, str]:
        patterns = [
            r"looking for\s+(.+?)(?:[.;]|,\s*(?:but|and)|$)",
            r"(?:find|show me|help me find)\s+(.+?)(?:[.;]|,\s*(?:but|and)|$)",
            r"\bi need\s+(.+?)(?:[.;]|,\s*(?:but|and)|$)",
        ]
        category = ""
        for pattern in patterns:
            match = re.search(pattern, message, re.I)
            if match:
                category = match.group(1).strip()
                break
        preference = ""
        pieces = re.split(r"[.;]", message, maxsplit=1)
        if len(pieces) > 1:
            preference = pieces[1].strip()
        return category, preference

    @staticmethod
    def _extract_constraints(message: str) -> list[str]:
        marker_patterns = [
            r"(?:requirement is|matters is|what i need is)\s*:\s*(.+)$",
            r"(?:must have|must be|care about is|prioriti[sz]e)\s+(.+?)(?:\s+instead)?[.]?$",
        ]
        payload = ""
        for pattern in marker_patterns:
            match = re.search(pattern, message, re.I)
            if match:
                payload = match.group(1)
                break
        if not payload:
            return []
        parts = re.split(r"\s*;\s*|,?\s+and also\s+", payload, flags=re.I)
        return [part.strip(" .") for part in parts if part.strip(" .")]

    @staticmethod
    def _extract_negative_constraints(message: str) -> list[str]:
        # Deliberately excludes vague phrases such as "no preference" so the
        # evaluator's boundary response cannot accidentally become a hard negative.
        patterns = [
            r"(?:avoid|without|do not want|don't want|not interested in)\s+(.+?)(?:[.;]|$)",
            r"(?:must not be|cannot be|can't be)\s+(.+?)(?:[.;]|$)",
            r"\bno\s+(?!preference\b)(.+?)(?:,\s*(?:but|and)|[.;]|$)",
        ]
        values: list[str] = []
        for pattern in patterns:
            values.extend(match.strip(" .") for match in re.findall(pattern, message, re.I))
        return list(dict.fromkeys(value for value in values if value))

    @staticmethod
    def _extract_inline_facets(value: str) -> list[str]:
        lowered = value.casefold()
        facets: list[str] = []
        for color in COLOR_WORDS:
            if re.search(rf"\b{re.escape(color)}\b", lowered):
                facets.append(f"color: {color}")
        for material in MATERIAL_WORDS[:-1]:
            if re.search(rf"\b{re.escape(material)}\b", lowered):
                facets.append(material)
        for term in ("running", "hiking", "winter", "outdoor", "work", "casual", "formal", "classic", "elegant"):
            if re.search(rf"\b{term}\b", lowered):
                facets.append(term)
        if any(term in lowered for term in ("laptop", "computer", "macbook", "notebook")):
            facets.append("fits laptop")
        if any(term in lowered for term in ("travel", "commute", "commuting")):
            facets.append("travel")
        budget = re.search(r"(?:under|below|up to)\s*\$?([0-9]+(?:\.[0-9]+)?)", lowered)
        if budget:
            facets.append(f"budget under ${budget.group(1)}")
        return list(dict.fromkeys(facets))

    @classmethod
    def _extract_override_constraints(cls, message: str) -> list[str]:
        explicit = cls._extract_constraints(message)
        if explicit:
            return explicit
        replacement = re.search(
            r"(?:^|[,;]\s*|\band\s+)([^,;.]+?)\s+instead of\s+(.+?)(?:[.;]|$)",
            message,
            re.I,
        )
        if replacement:
            return cls._extract_inline_facets(replacement.group(1))
        match = re.search(
            r"(?:make\s+(?:it|them)|switch\s+to|replace\s+.+?\s+with)\s+(.+?)(?:\s+instead)?[.]?$",
            message,
            re.I,
        )
        return cls._extract_inline_facets(match.group(1)) if match else []

    @staticmethod
    def _extract_override_category(message: str) -> str:
        specific = re.findall(
            r"\b(backpacks?|daypacks?|rucksacks?|laptop bags?|messenger bags?|briefcases?|"
            r"sneakers?|shoes?|boots?|jackets?|coats?|dresses?|shirts?|pants|sandals?|slippers?)\b",
            message,
            re.I,
        )
        if specific:
            return specific[-1]
        return ""

    def _bm25_scored(self, query: str, limit: int = 500) -> list[tuple[str, float]]:
        terms = list(dict.fromkeys(_terms(query)))[:100]
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, 0.0, 5.0, 4.0, 6.0, 6.0, 1.0, 2.0) "
            "FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 5.0, 4.0, 6.0, 6.0, 1.0, 2.0) LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [(str(row[0]), float(row[1])) for row in rows]

    def _bm25(self, query: str, limit: int = 500) -> list[str]:
        return [asin for asin, _ in self._bm25_scored(query, limit)]

    def _dense(self, query: str, limit: int = 250) -> list[str]:
        return [asin for asin, _ in self._dense_scored([(query, 1.0)], limit)]

    @staticmethod
    def _semantic_request(
        category: str,
        constraints: list[str],
        negative_constraints: list[str] | None,
    ) -> str:
        clauses = []
        if category:
            clauses.append(f"shopper wants {category}")
        if constraints:
            clauses.append("needs " + ", ".join(constraints[:5]))
        if negative_constraints:
            clauses.append("avoid " + ", ".join(negative_constraints[:4]))
        return ". ".join(clauses)

    @staticmethod
    def _semantic_queries(
        category: str,
        constraints: list[str],
        negative_constraints: list[str] | None,
        query_text: str,
        route: str,
    ) -> list[tuple[str, float]]:
        queries: list[tuple[str, float]] = []
        category = category.strip()
        structured = " ".join([category, *constraints]).strip()
        request = Agent._semantic_request(category, constraints, negative_constraints)
        exploratory = route in {"browsing", "uncertain"}
        if query_text.strip():
            queries.append((query_text.strip(), 0.55 if exploratory else 0.35))
        if structured:
            queries.append((structured, 0.35 if exploratory else 0.45))
        if request:
            queries.append((request, 0.30 if exploratory else 0.25))
        if category and constraints:
            queries.append((f"{category}. {'; '.join(constraints[:4])}", 0.20))
        elif category:
            queries.append((category, 0.15))
        elif constraints:
            queries.append((" ".join(constraints[:4]), 0.15))
        if negative_constraints:
            queries.append((
                "avoid " + "; ".join(value for value in negative_constraints[:4] if value),
                -0.12,
            ))
        if not queries:
            return []
        totals = sum(abs(weight) for _, weight in queries) or 1.0
        merged: dict[str, float] = defaultdict(float)
        for text, weight in queries:
            cleaned = _normalize(text)
            if cleaned:
                merged[cleaned] += weight / totals
        return [(text, weight) for text, weight in merged.items() if abs(weight) > 1e-9]

    def _dense_scored(
        self,
        queries: list[tuple[str, float]],
        limit: int = 250,
    ) -> list[tuple[str, float]]:
        if self.model is None or self.embeddings is None or not queries:
            return []
        try:
            encoded = self.model.encode(
                [text for text, _ in queries],
                normalize_embeddings=True,
            )
        except Exception:
            return []
        weighted_vectors: list[tuple[float, list[float]]] = []
        if np is not None and hasattr(encoded, "shape"):
            for (__, weight), vector in zip(queries, encoded, strict=False):
                weighted_vectors.append((weight, [float(value) for value in vector.tolist()]))
        else:
            for (__, weight), vector in zip(queries, encoded, strict=False):
                weighted_vectors.append((weight, _normalize_vector([float(value) for value in vector])))
        blended = _blend_vectors(weighted_vectors)
        if not blended:
            return []
        if np is not None and hasattr(self.embeddings, "shape"):
            scores = self.embeddings @ np.asarray(blended, dtype=np.float32)
            limit = min(limit, len(scores))
            if limit <= 0:
                return []
            indices = np.argpartition(scores, -limit)[-limit:]
            indices = indices[np.argsort(scores[indices])[::-1]]
            return [
                (self.asins[int(index)], float(scores[int(index)]))
                for index in indices
            ]
        scores = [
            sum(value * weight for value, weight in zip(vector, blended, strict=False))
            for vector in self.embeddings
        ]
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:limit]
        return [(self.asins[index], float(scores[index])) for index in ranked]

    @staticmethod
    def _should_use_dense(
        route: str,
        query_text: str,
        normalized_groups: list[set[str]],
        negative_groups: list[set[str]],
        bm25_scored: list[tuple[str, float]],
        best_exact: int,
        complete_match_count: int,
    ) -> bool:
        if route in {"browsing", "uncertain"}:
            return True
        if not query_text.strip():
            return False
        lowered = query_text.casefold()
        exploratory = any(term in lowered for term in (
            "something", "ideas", "open to", "not sure", "exploring", "versatile", "everyday",
        ))
        if exploratory:
            return True
        hard_constraints = len(normalized_groups) + len(negative_groups)
        if hard_constraints <= 1 and len(_terms(query_text)) >= 5:
            return True
        if best_exact < min(2, len(normalized_groups)) and len(_terms(query_text)) >= 4:
            return True
        if complete_match_count > 40:
            return True
        if len(bm25_scored) < 20:
            return True
        if len(bm25_scored) >= 2:
            first_score, second_score = bm25_scored[0][1], bm25_scored[1][1]
            bm25_gap = (second_score - first_score) / max(abs(first_score), 1e-9)
            if bm25_gap < 0.03 and hard_constraints <= 2:
                return True
        return False

    def rank_with_diagnostics(
        self,
        category: str,
        constraints: list[str],
        negative_constraints: list[str] | None = None,
        user_profile: dict[str, Any] | None = None,
        route: str = "hybrid",
        query_text: str = "",
    ) -> tuple[list[str], dict[str, float]]:
        profile_tags = tuple(sorted(
            _normalize(str(value)) for value in (user_profile or {}).get("preference_tags", [])
        ))
        profile_key = (
            profile_tags,
            (user_profile or {}).get("average_prior_rating"),
            _normalize(str((user_profile or {}).get("rating_style", ""))),
        )
        cache_key: tuple[object, ...] = (
            _normalize(category),
            tuple(_normalize(value) for value in constraints),
            tuple(_normalize(value) for value in (negative_constraints or [])),
            profile_key if self.profile_tiebreak else (),
            route,
            _normalize(query_text),
        )
        cached = self.rank_cache.get(cache_key)
        if cached is not None:
            return cached
        normalized_groups = [_constraint_variants(value) for value in constraints]
        negative_groups = [
            _constraint_variants(value) for value in (negative_constraints or [])
        ]
        expanded_constraints = list(dict.fromkeys(
            variant for group in normalized_groups for variant in group
        ))
        query = " ".join([category, *constraints, *expanded_constraints]).strip()
        bm25_scored = self._bm25_scored(query)
        bm25 = [asin for asin, _ in bm25_scored]
        semantic_queries = self._semantic_queries(
            category,
            constraints,
            negative_constraints,
            query_text,
            route,
        )
        dense_scored = self._dense_scored(semantic_queries) if route in {"browsing", "uncertain", "hybrid"} else []
        dense = [asin for asin, _ in dense_scored]
        bm25_rank = {asin: rank for rank, asin in enumerate(bm25, 1)}
        dense_rank = {asin: rank for rank, asin in enumerate(dense, 1)}
        dense_scores = {asin: score for asin, score in dense_scored}
        exact_ids: set[str] = set()
        for group in normalized_groups:
            for value in group:
                exact_ids.update(self.card_index.get(value, set()))
        candidates = set(bm25) | set(dense) | exact_ids
        requested_categories = self._match_catalog_categories(" ".join([category, query_text, *constraints]))
        if category:
            requested_categories.update(_phrase_variants(category))

        exact_counts = {
            asin: sum(bool(group & self.cards.get(asin, set())) for group in normalized_groups)
            for asin in candidates
        }
        cross_scores: dict[str, float] = {}
        has_complete_exact_match = bool(normalized_groups) and max(exact_counts.values(), default=0) == len(normalized_groups)
        complete_match_count = (
            sum(count == len(normalized_groups) for count in exact_counts.values())
            if normalized_groups else 0
        )
        best_exact = max(exact_counts.values(), default=0)
        use_dense = route in {"browsing", "uncertain", "hybrid"} and self._should_use_dense(
            route,
            query_text,
            normalized_groups,
            negative_groups,
            bm25_scored,
            best_exact,
            complete_match_count,
        )
        if not use_dense:
            dense_scored = []
            dense = []
            dense_rank = {}
            dense_scores = {}
        if self.cross_encoder is not None and normalized_groups and not has_complete_exact_match:
            cross_candidates = bm25[:50]
            pairs = [
                (query, self.documents[self.asin_to_index[asin]])
                for asin in cross_candidates
            ]
            if pairs:
                scores = self.cross_encoder.predict(pairs, batch_size=256, show_progress_bar=False)
                cross_scores = {asin: float(score) for asin, score in zip(cross_candidates, scores)}

        def key(asin: str) -> tuple[int, float, float, float, float, float, float, float, float]:
            values = self.cards.get(asin, set())
            category_signatures = self.category_signatures.get(asin, set())
            negative_count = sum(bool(group & values) for group in negative_groups)
            exact_count = exact_counts[asin]
            exact_chars = sum(
                max((len(value) for value in group if value in values), default=0)
                for group in normalized_groups
            )
            semantic_score = dense_scores.get(asin, float("-inf"))
            requested_category_match = float(bool(requested_categories & category_signatures))
            category_focus = -float(len(category_signatures - requested_categories)) if requested_categories else 0.0
            laptop_fit = float(
                "fits laptop" in constraints and any(
                    token in self.documents[self.asin_to_index[asin]].casefold()
                    for token in ("laptop", "computer", "macbook", "notebook")
                )
            )
            # RRF makes lexical and dense ranks comparable without score calibration.
            rrf = 1.0 / (60 + bm25_rank.get(asin, 100_000))
            rrf += 1.0 / (60 + dense_rank.get(asin, 100_000))
            profile_score = 0.0
            rating_fit = 0.0
            popularity = 0.0
            if self.profile_tiebreak and complete_match_count > 10:
                document = self.documents[self.asin_to_index[asin]].casefold()
                profile_score = float(sum(
                    any(term in document for term in PROFILE_TAG_TERMS.get(tag, (tag,)))
                    for tag in profile_tags
                ))
                average_rating, rating_number = self.product_quality.get(asin, (0.0, 0.0))
                prior_rating = (user_profile or {}).get("average_prior_rating")
                if isinstance(prior_rating, (int, float)) and average_rating:
                    rating_fit = -abs(float(prior_rating) - average_rating)
                popularity = math.log1p(max(0.0, rating_number))
            buying_key = (
                -negative_count,
                requested_category_match,
                laptop_fit,
                category_focus,
                float(exact_count),
                float(exact_chars),
                cross_scores.get(asin, float("-inf")),
                semantic_score,
                profile_score,
                rating_fit,
                popularity,
                rrf,
                -float(bm25_rank.get(asin, 100_000)),
            )
            if route == "browsing":
                return (-negative_count, requested_category_match, laptop_fit, category_focus, semantic_score, rrf, float(exact_count), float(exact_chars), 0.0,
                        -float(bm25_rank.get(asin, 100_000)))
            return buying_key

        ranked = sorted(candidates, key=key, reverse=True)
        if route == "browsing":
            protected = ranked[:3]
            buckets: dict[str, list[str]] = defaultdict(list)
            for asin in ranked[3:100]:
                buckets[self.product_groups.get(asin, "other")].append(asin)
            diverse = []
            while buckets:
                for group in list(buckets):
                    diverse.append(buckets[group].pop(0))
                    if not buckets[group]:
                        del buckets[group]
            ranked = protected + diverse + ranked[100:]
        exact_ties = sum(count == best_exact for count in exact_counts.values())
        bm25_gap = 0.0
        if len(bm25_scored) >= 2:
            first_score, second_score = bm25_scored[0][1], bm25_scored[1][1]
            bm25_gap = (second_score - first_score) / max(abs(first_score), 1e-9)
        diagnostics = {
            "constraint_count": float(len(normalized_groups)),
            "negative_constraint_count": float(len(negative_groups)),
            "variant_count": float(sum(len(group) for group in normalized_groups)),
            "candidate_count": float(len(candidates)),
            "best_exact_count": float(best_exact),
            "exact_tie_count": float(exact_ties),
            "complete_match_count": float(complete_match_count),
            "bm25_result_count": float(len(bm25)),
            "dense_result_count": float(len(dense)),
            "bm25_relative_gap": float(bm25_gap),
            "semantic_query_count": float(len(semantic_queries)),
            "dense_enabled": float(use_dense),
        }
        if len(self.rank_cache) >= 4096:
            self.rank_cache.pop(next(iter(self.rank_cache)))
        self.rank_cache[cache_key] = (ranked, diagnostics)
        return ranked, diagnostics

    def _rank(
        self,
        category: str,
        constraints: list[str],
        negative_constraints: list[str] | None = None,
        user_profile: dict[str, Any] | None = None,
    ) -> list[str]:
        ranked, _ = self.rank_with_diagnostics(
            category, constraints, negative_constraints, user_profile
        )
        return ranked

    @staticmethod
    def _route_intent(message: str, state: dict[str, Any], confidence: float) -> str:
        lowered = message.casefold()
        exploratory = any(term in lowered for term in ("something", "ideas", "open to", "not sure", "exploring"))
        hard = bool(state.get("category")) + len(state.get("constraints", []))
        hard += bool(state.get("negative_constraints")) + bool(state.get("slots", {}).get("budget"))
        hard += any(term in lowered for term in ("must", "need", "under", "exactly"))
        if exploratory and hard < 2:
            return "browsing"
        if hard >= 2 or (state.get("inferred_intent") == "buying" and confidence >= 0.75):
            return "buying"
        if state.get("inferred_intent") == "browsing" and confidence >= 0.75:
            return "browsing"
        return "uncertain"

    @staticmethod
    def _fuse_routes(buying: list[str], browsing: list[str], buying_weight: float) -> list[str]:
        scores: defaultdict[str, float] = defaultdict(float)
        for rank, asin in enumerate(buying, 1):
            scores[asin] += buying_weight / (60 + rank)
        for rank, asin in enumerate(browsing, 1):
            scores[asin] += (1.0 - buying_weight) / (60 + rank)
        return sorted(scores, key=scores.get, reverse=True)

    @staticmethod
    def _merge_constraints(state: dict[str, Any], values: list[str], override: bool) -> None:
        if not values:
            return
        if not state.get("slots"):
            for existing in state.get("constraints", []):
                state["slots"].setdefault(_constraint_slot(existing), []).append(existing)
        cleared: set[str] = set()
        for value in values:
            slot = _constraint_slot(value)
            if override and slot not in cleared:
                existing_values = state["slots"].get(slot, [])
                equivalent = any(
                    bool(_constraint_variants(value) & _constraint_variants(existing))
                    for existing in existing_values
                )
                if existing_values and not equivalent:
                    state["slots"][slot] = []
                cleared.add(slot)
            if value not in state["slots"].setdefault(slot, []):
                state["slots"][slot].append(value)
        state["constraints"] = [
            value for slot_values in state["slots"].values() for value in slot_values
        ]

    def _question_scores(self, ranked: list[str], state: dict[str, Any]) -> dict[str, float]:
        candidates = [asin for asin in ranked if asin not in state["seen"]][:100]
        if len(candidates) < 2:
            return {}
        asked = state["asked_attributes"]
        scores: dict[str, float] = {}
        for attribute in ("material", "color", "size", "style", "budget", "feature", "use_case"):
            if attribute in asked:
                continue
            counts: Counter[str] = Counter()
            covered = 0
            for asin in candidates:
                values = self.card_facets.get(asin, {}).get(attribute, set())
                if values:
                    covered += 1
                    counts.update(values)
                else:
                    counts["<unknown>"] += 1
            total = sum(counts.values())
            if total <= 1 or len(counts) <= 1:
                continue
            entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
            coverage = covered / len(candidates)
            if coverage < 0.10:
                continue
            scores[attribute] = entropy * (0.25 + 0.75 * coverage)
        for tag in state.get("user_profile", {}).get("preference_tags", []):
            for attribute in PROFILE_TAG_ATTRIBUTES.get(_normalize(str(tag)), ()):
                if attribute in scores:
                    scores[attribute] += 0.08
        return scores

    def _select_question(self, ranked: list[str], state: dict[str, Any]) -> str:
        if not self.adaptive_questions:
            return "other"
        scores = self._question_scores(ranked, state)
        if not scores:
            return "other"
        attribute = max(scores, key=lambda item: (scores[item], item))
        return attribute if scores[attribute] >= 0.35 else "other"

    def _question_message(
        self, ask_attribute: str, ranked: list[str], state: dict[str, Any]
    ) -> str:
        direct = {
            "material": "Do you have a preferred material?",
            "color": "Is there a color you prefer?",
            "size": "Do you have any sizing or fit requirements?",
            "style": "What style are you looking for?",
            "budget": "What budget should I stay within?",
            "feature": "Which product feature matters most to you?",
            "use_case": "What will you mainly use it for?",
        }
        if ask_attribute != "other":
            return direct[ask_attribute]
        if not self.adaptive_prompt:
            return "What other requirement matters most?"
        scores = self._question_scores(ranked, state)
        labels = {
            "material": "material",
            "color": "color",
            "size": "fit or sizing",
            "style": "style",
            "budget": "budget",
            "feature": "specific features",
            "use_case": "intended use",
        }
        facets = [labels[name] for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]]
        if not facets:
            return "What other requirement matters most?"
        if len(facets) == 1:
            choices = facets[0]
        else:
            choices = ", ".join(facets[:-1]) + f", or {facets[-1]}"
        return f"Which matters most next—{choices}?"

    def update_and_rank(
        self, session_id: str, user_message: str, turn: int
    ) -> tuple[list[str], dict[str, float], bool]:
        """Apply one user turn and expose ranking data for offline policy replay."""
        if session_id not in self.sessions:
            raise RuntimeError("reset must be called before update_and_rank")
        state = self.sessions[session_id]
        if turn == 1:
            category, preference = self._extract_initial(user_message)
            state["category"] = self._resolve_catalog_category(user_message, category)
            state["initial_preference"] = preference
            initial_constraints = self._extract_inline_facets(user_message)
            state["initial_constraints"] = initial_constraints
            if self.typed_slots:
                self._merge_constraints(state, initial_constraints, False)

        new_constraints = self._extract_constraints(user_message)
        structured = StructuredTurn()
        if (
            self.structured_extractor is not None
            and not self._rules_are_confident(user_message, new_constraints)
        ):
            try:
                candidate = self.structured_extractor.extract(user_message, state)
                state["last_usage"] = {
                    "prompt_tokens": candidate.prompt_tokens,
                    "completion_tokens": candidate.completion_tokens,
                }
                if candidate.confidence >= self.extraction_min_confidence:
                    structured = _quarantine_structured_turn(user_message, state, candidate)
            except Exception:
                # Local model availability or malformed output must never disable
                # the deterministic scorer-proven path.
                state["last_usage"] = {"prompt_tokens": 0, "completion_tokens": 0}
        else:
            state["last_usage"] = {"prompt_tokens": 0, "completion_tokens": 0}

        override = bool(OVERRIDE_RE.search(user_message) or re.search(r"\binstead of\b", user_message, re.I)) or structured.override
        if override:
            # Confirmed constraints survive; inline values from the original request
            # are superseded before rewritten slots are applied.
            state["initial_preference"] = ""
            for value in state.get("initial_constraints", []):
                slot = _constraint_slot(value)
                state["slots"][slot] = [
                    existing for existing in state["slots"].get(slot, [])
                    if existing != value
                ]
            state["constraints"] = [
                value for slot_values in state["slots"].values() for value in slot_values
            ]
            replacement_category = self._extract_override_category(user_message)
            if replacement_category:
                state["category"] = self._resolve_catalog_category(user_message, replacement_category)
            state["seen"].clear()

        new_constraints = (
            self._extract_override_constraints(user_message)
            if override else self._extract_constraints(user_message)
        )
        if self.typed_slots:
            self._merge_constraints(state, new_constraints, override)
        else:
            state["constraints"].extend(new_constraints)
            state["constraints"] = list(dict.fromkeys(state["constraints"]))
        new_negative_constraints = self._extract_negative_constraints(user_message)
        state["negative_constraints"].extend(new_negative_constraints)
        state["negative_constraints"] = list(dict.fromkeys(state["negative_constraints"]))
        if structured.confidence >= self.extraction_min_confidence:
            self._apply_structured_turn(state, structured)
        route = self._route_intent(user_message, state, structured.confidence) if self.experimental_router else "hybrid"
        if route == "uncertain":
            buying, diagnostics = self.rank_with_diagnostics(
                state["category"], state["constraints"], state["negative_constraints"], state["user_profile"], "buying",
                user_message,
            )
            browsing, _ = self.rank_with_diagnostics(
                state["category"], state["constraints"], state["negative_constraints"], state["user_profile"], "browsing",
                user_message,
            )
            buying_weight = structured.confidence if structured.intent == "buying" else 0.5
            ranked = self._fuse_routes(buying, browsing, buying_weight)
        else:
            ranked, diagnostics = self.rank_with_diagnostics(
                state["category"], state["constraints"], state["negative_constraints"], state["user_profile"], route,
                user_message,
            )
        diagnostics["route_buying"] = float(route == "buying")
        diagnostics["route_browsing"] = float(route == "browsing")
        diagnostics["route_uncertain"] = float(route == "uncertain")
        return ranked, diagnostics, override

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        ranked, diagnostics, _ = self.update_and_rank(session_id, user_message, turn)
        state = self.sessions[session_id]
        # Preserve high MRR while information is still arriving, then use every
        # contract-allowed slot on the final turn so deep duplicate ties cannot
        # become avoidable misses.
        if turn >= 10:
            output_limit = 10
        elif turn <= 6:
            output_limit = 1
        elif diagnostics.get("complete_match_count", 0.0) > 100:
            output_limit = 5
        else:
            output_limit = 3
        recommendations = []
        for asin in ranked:
            if asin in state["seen"]:
                continue
            recommendations.append({"parent_asin": asin})
            if len(recommendations) == min(top_k, output_limit):
                break
        state["seen"].update(item["parent_asin"] for item in recommendations)
        ask_attribute = self._select_question(ranked, state)
        state["asked_attributes"].add(ask_attribute)
        question = self._question_message(ask_attribute, ranked, state)
        return {
            "message": f"Here are my strongest matches. {question}",
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": dict(state["last_usage"]),
        }
