from __future__ import annotations

import re
from dataclasses import dataclass, field


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

PRODUCT_TYPE_PATTERNS = (
    ("running_shoes", r"\b(?:running|jogging|athletic)\s+(?:shoes?|sneakers?|trainers?)\b"),
    ("hiking_shoes", r"\b(?:hiking|trail)\s+(?:shoes?|boots?|sneakers?)\b"),
    ("shoes", r"\b(?:shoes?|sneakers?|trainers?|footwear)\b"),
    ("boots", r"\bboots?\b"),
    ("sandals", r"\b(?:sandals?|flip[ -]?flops?)\b"),
    ("socks", r"\bsocks?\b"),
    ("jackets", r"\b(?:jackets?|coats?|parkas?)\b"),
    ("shirts", r"\b(?:shirts?|t[ -]?shirts?|tees?|blouses?|tops?)\b"),
    ("pants", r"\b(?:pants?|trousers?|joggers?|leggings?)\b"),
    ("shorts", r"\bshorts?\b"),
    ("dresses", r"\b(?:dresses?|gowns?)\b"),
    ("skirts", r"\bskirts?\b"),
    ("bras", r"\bbras?\b"),
    ("hats", r"\b(?:hats?|caps?|beanies?|headbands?)\b"),
    ("watches", r"\bwatches?\b"),
    ("jewelry", r"\b(?:jewelry|necklaces?|bracelets?|earrings?|rings?)\b"),
    ("bags", r"\b(?:bags?|handbags?|backpacks?|purses?)\b"),
)

AUDIENCE_PATTERNS = {
    "women": r"\b(?:women|women's|womens|female|ladies|lady)\b",
    "men": r"\b(?:men|men's|mens|male|gentlemen)\b",
    "girls": r"\b(?:girls|girl's)\b",
    "boys": r"\b(?:boys|boy's)\b",
    "kids": r"\b(?:kids|children|child|youth|toddler)\b",
    "unisex": r"\bunisex\b",
}

SLOT_VOCABULARIES = {
    "material": {
        "cotton", "polyester", "nylon", "leather", "wool", "spandex",
        "silk", "rayon", "denim", "satin", "fleece", "rubber", "suede",
        "synthetic", "mesh",
    },
    "color": {
        "black", "white", "blue", "red", "pink", "green", "brown",
        "gray", "grey", "purple", "yellow", "orange", "gold", "silver",
        "navy", "beige",
    },
    "size": {
        "small", "medium", "large", "xl", "xxl", "plus", "petite",
        "wide", "narrow", "regular", "tall", "short",
    },
    "style": {
        "casual", "formal", "classic", "modern", "vintage", "slim",
        "loose", "relaxed", "hooded", "crew", "athletic", "comfortable",
    },
    "use_case": {
        "running", "jogging", "hiking", "gym", "work", "winter", "outdoor",
        "trail", "wedding", "party", "travel", "beach", "sports", "training",
        "school", "office", "sleep", "rain",
    },
}

FEATURE_TERMS = {
    "breathable", "compression", "cushioned", "durable", "insulated",
    "lightweight", "non-slip", "nonslip", "padded", "quick-dry", "stretchy",
    "supportive", "water-resistant", "waterproof", "windproof",
}


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def extract_product_types(text: str) -> set[str]:
    lowered = text.lower()
    found = {name for name, pattern in PRODUCT_TYPE_PATTERNS if re.search(pattern, lowered)}
    if "running_shoes" in found or "hiking_shoes" in found:
        found.add("shoes")
    return found


def extract_audiences(text: str) -> set[str]:
    lowered = text.lower()
    return {name for name, pattern in AUDIENCE_PATTERNS.items() if re.search(pattern, lowered)}


def _extract_phrases(text: str) -> tuple[list[str], list[str]]:
    required: list[str] = []
    preferred: list[str] = []
    for pattern in (
        r"\ba key requirement is:\s*(.+?)(?:\.|$)",
        r"\bwhat i need is:\s*(.+?)(?:\.|$)",
        r"\b(?:must|needs? to|has to)\s+(?:be|have|include)?\s*(.+?)(?:\.|,|$)",
    ):
        required.extend(match.strip(" -;,") for match in re.findall(pattern, text, re.I))
    for pattern in (
        r"\bwhat matters is:\s*(.+?)(?:\.|$)",
        r"\bi (?:prefer|would like)\s+(.+?)(?:\.|$)",
    ):
        for match in re.findall(pattern, text, re.I):
            preferred.extend(part.strip(" -;,") for part in match.split(";") if part.strip(" -;,"))
    return required, preferred


def _extract_exclusions(text: str) -> set[str]:
    exclusions: set[str] = set()
    for match in re.findall(r"\b(?:no|without|avoid|exclude)\s+([a-z0-9 -]+?)(?:[.,;]|$)", text, re.I):
        exclusions.update(_tokens(match))
    return exclusions


@dataclass
class IntentDelta:
    operation: str = "accumulate"
    mode: str | None = None
    product_types: set[str] = field(default_factory=set)
    audiences: set[str] = field(default_factory=set)
    slots: dict[str, set[str]] = field(default_factory=dict)
    required_phrases: list[str] = field(default_factory=list)
    preferred_phrases: list[str] = field(default_factory=list)
    excluded_terms: set[str] = field(default_factory=set)
    budget_max: float | None = None


def parse_intent(text: str) -> IntentDelta:
    lowered = text.lower()
    operation = "replace" if re.search(r"\b(?:actually|instead|ignore|forget|rather)\b", lowered) else "accumulate"
    if "still exploring" in lowered:
        mode = "browsing"
    elif "key requirement" in lowered or re.search(r"\b(?:must|need)\b", lowered):
        mode = "buying"
    else:
        mode = None

    terms = _tokens(text)
    exclusions = _extract_exclusions(text)
    slots = {
        name: (terms & vocabulary) - exclusions
        for name, vocabulary in SLOT_VOCABULARIES.items()
        if (terms & vocabulary) - exclusions
    }
    features = terms & FEATURE_TERMS
    if features:
        slots["feature"] = features
    budget_match = re.search(r"(?:\$|under|below|up to|max(?:imum)?(?: of)?)\s*\$?\s*(\d+(?:\.\d+)?)", lowered)
    required, preferred = _extract_phrases(text)
    return IntentDelta(
        operation=operation,
        mode=mode,
        product_types=extract_product_types(text),
        audiences=extract_audiences(text),
        slots=slots,
        required_phrases=required,
        preferred_phrases=preferred,
        excluded_terms=exclusions,
        budget_max=float(budget_match.group(1)) if budget_match else None,
    )
