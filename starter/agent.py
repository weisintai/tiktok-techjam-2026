from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from starter.intent import extract_audiences, extract_product_types, parse_intent


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "those", "options", "not", "quite", "right", "yet", "ask", "about",
    "one", "specific", "attribute", "what", "matters", "have", "additional",
    "preference", "use", "your", "judgment", "actually", "ignore", "earlier",
    "need", "key", "requirement", "still", "exploring",
}

MATERIALS = {
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric", "denim", "satin", "fleece", "rubber", "suede",
}
COLORS = {
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "grey", "purple", "yellow", "orange", "gold", "silver", "navy", "beige",
}
SIZE_WORDS = {
    "size", "small", "medium", "large", "xl", "xxl", "plus", "petite",
    "wide", "narrow", "regular", "tall", "short",
}
STYLE_WORDS = {
    "casual", "formal", "classic", "modern", "vintage", "slim", "loose",
    "relaxed", "fit", "sleeve", "neck", "hooded", "crew", "dress",
    "costume", "fashion", "athletic", "comfort", "comfortable",
}
USE_CASE_WORDS = {
    "running", "jogging", "hiking", "gym", "work", "winter", "outdoor",
    "trail", "wedding", "party", "halloween", "cosplay", "travel", "beach",
    "sports", "training", "school", "office", "sleep", "rain",
}
ALIASES = {
    "jogging": ["running"],
    "synthetic": ["polyester", "nylon"],
    "textile": ["fabric"],
    "grey": ["gray"],
    "tee": ["shirt", "t-shirt"],
    "sneakers": ["shoes"],
    "trainer": ["running", "shoes"],
}

# Dependency-free pairwise logistic reranker trained with session-grouped folds.
PAIRWISE_WEIGHTS = {
    "retrieval": -51.60046175391213,
    "position": 1.3911650742852975,
    "slot_matches": 0.1505753205960668,
    "slot_misses": -2.138260881503187,
    "hard_idf": 0.01440967859958202,
    "soft_idf": 0.031502221205117115,
    "category_idf": 0.8175570957443548,
    "query_idf": 0.043180478245424085,
    "hard_phrase": 0.5409953720973361,
    "soft_phrase": 1.1928582556678893,
    "category_phrase": 0.8394080601399385,
    "quality": 0.9020143488722732,
    "hard_coverage": 3.7536002087164717,
    "soft_coverage": 1.787539423222174,
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    terms = [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]
    expanded: list[str] = []
    for term in terms:
        expanded.append(term)
        expanded.extend(ALIASES.get(term, []))
    return expanded


def _price_bucket(value: object) -> str | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if price < 20:
        return "under_20"
    if price < 50:
        return "20_50"
    if price < 100:
        return "50_100"
    return "over_100"


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _facet_values(product: dict) -> dict[str, set[str]]:
    categories = [str(item) for item in product.get("categories") or []]
    type_categories = [
        category
        for category in categories
        if category.lower() not in {"clothing, shoes & jewelry", "clothing shoes & jewelry"}
    ]
    text = _text(
        [
            product.get("title"),
            product.get("features"),
            product.get("description"),
            product.get("details"),
            product.get("categories"),
            product.get("store"),
        ]
    ).lower()
    type_text = _text([product.get("title"), type_categories[-3:]]).lower()
    terms = set(_terms(text))
    details = product.get("details") or {}
    values: dict[str, set[str]] = {
        "category": set(),
        "product_type": extract_product_types(type_text),
        "audience": extract_audiences(text),
        "material": terms & MATERIALS,
        "color": terms & COLORS,
        "size": terms & SIZE_WORDS,
        "style": terms & STYLE_WORDS,
        "use_case": terms & USE_CASE_WORDS,
        "brand": set(),
        "budget": set(),
        "feature": set(),
        "__all": terms,
    }
    for category in [item.lower() for item in categories][-3:]:
        values["category"].update(_terms(category))
    store = product.get("store")
    if store:
        values["brand"].update(_terms(str(store))[:4])
    manufacturer = details.get("Manufacturer") if isinstance(details, dict) else None
    if manufacturer:
        values["brand"].update(_terms(str(manufacturer))[:4])
    bucket = _price_bucket(product.get("price"))
    if bucket:
        values["budget"].add(bucket)
    for token in sorted(terms):
        if token in MATERIALS or token in COLORS or token in SIZE_WORDS:
            continue
        if token in values["category"] or token in values["brand"]:
            continue
        values["feature"].add(token)
        if len(values["feature"]) >= 24:
            break
    return values


def _message_slots(text: str) -> dict[str, set[str]]:
    terms = set(_terms(text))
    slots: dict[str, set[str]] = defaultdict(set)
    for name, vocabulary in (
        ("material", MATERIALS),
        ("color", COLORS),
        ("size", SIZE_WORDS),
        ("style", STYLE_WORDS),
        ("use_case", USE_CASE_WORDS),
    ):
        slots[name].update(terms & vocabulary)
    if re.search(r"(?:\$|under|below|around|budget)\s*\d+", text, re.I):
        slots["budget"].add("mentioned")
    return slots


def _clean_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:180].rstrip()


def _normalized_text(value: str) -> str:
    return " ".join(_terms(value))


def _phrase_coverage_score(phrases: list[str], normalized_product_text: str) -> float:
    score = 0.0
    for phrase in phrases:
        normalized_phrase = _normalized_text(phrase)
        if not normalized_phrase:
            continue
        terms = normalized_phrase.split()
        if normalized_phrase in normalized_product_text:
            score += min(6.0, 1.0 + 0.6 * len(terms))
            continue
        present = sum(1 for term in set(terms) if f" {term} " in f" {normalized_product_text} ")
        if present:
            score += 0.15 * present / max(1, len(set(terms)))
    return score


def _template_phrases(text: str) -> dict[str, list[str]]:
    phrases: dict[str, list[str]] = {"category": [], "hard": [], "soft": []}
    category = re.search(r"\bi'm looking for\s+(.+?)(?:\.|,\s+but\b)", text, re.I)
    if category:
        phrases["category"].append(_clean_phrase(category.group(1)))
    key_requirement = re.search(r"\ba key requirement is:\s*(.+?)(?:\.|$)", text, re.I)
    if key_requirement:
        phrases["hard"].append(_clean_phrase(key_requirement.group(1)))
    matters = re.search(r"\bwhat matters is:\s*(.+?)(?:\.|$)", text, re.I)
    if matters:
        phrases["soft"].extend(_clean_phrase(item) for item in matters.group(1).split(";"))
    override = re.search(r"\bwhat i need is:\s*(.+?)(?:\.|$)", text, re.I)
    if override:
        phrases["hard"].append(_clean_phrase(override.group(1)))
    return {key: [item for item in values if item] for key, values in phrases.items()}


def _dedupe_append(target: list[str], values: list[str]) -> None:
    seen = {item.lower() for item in target}
    for value in values:
        key = value.lower()
        if key not in seen:
            target.append(value)
            seen.add(key)


def _match_expression(terms: list[str]) -> str:
    unique_terms = list(dict.fromkeys(terms))[:60]
    return " OR ".join(f'"{term}"' for term in unique_terms)


def _phrase_expression(phrases: list[str], max_terms: int = 48) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    used_terms = 0
    for phrase in phrases:
        terms = list(dict.fromkeys(_terms(phrase)))
        if not terms:
            continue
        if 1 < len(terms) <= 8:
            part = '"' + " ".join(terms) + '"'
            if part not in seen:
                parts.append(part)
                seen.add(part)
        for term in terms:
            part = f'"{term}"'
            if part in seen:
                continue
            parts.append(part)
            seen.add(part)
            used_terms += 1
            if used_terms >= max_terms:
                break
        if used_terms >= max_terms:
            break
    return " OR ".join(parts)


class Agent:
    """Stateful BM25 agent with experimental information-gain clarification."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self._facets: dict[str, dict[str, set[str]]] = {}
        self._term_idf: dict[str, float] = {}
        self._quality: dict[str, float] = {}
        self._product_text: dict[str, str] = {}
        self._facet_lookup: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        term_df: Counter[str] = Counter()
        product_count = 0
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                self._facets[parent_asin] = _facet_values(product)
                for attribute in (
                    "category", "product_type", "audience", "material", "color",
                    "size", "style", "use_case", "brand", "budget",
                ):
                    for value in self._facets[parent_asin].get(attribute, set()):
                        self._facet_lookup[(attribute, value)].add(parent_asin)
                self._product_text[parent_asin] = _normalized_text(
                    _text(
                        [
                            product.get("title"),
                            product.get("features"),
                            product.get("description"),
                            product.get("details"),
                            product.get("categories"),
                            product.get("store"),
                        ]
                    )
                )
                term_df.update(self._facets[parent_asin].get("__all", set()))
                try:
                    rating = float(product.get("average_rating") or 0.0)
                    rating_count = float(product.get("rating_number") or 0.0)
                except (TypeError, ValueError):
                    rating = 0.0
                    rating_count = 0.0
                self._quality[parent_asin] = (rating / 5.0) * math.log1p(max(0.0, rating_count))
                product_count += 1
                batch.append(
                    (
                        parent_asin,
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self._term_idf = {
            term: math.log((product_count + 1) / (count + 1)) + 1.0
            for term, count in term_df.items()
        }
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {
            "messages": [],
            "slots": defaultdict(set),
            "asked": set(),
            "no_preference": set(),
            "profile_terms": _terms(_text(user_profile)),
            "category_phrases": [],
            "hard_constraints": [],
            "soft_constraints": [],
            "product_types": set(),
            "audiences": set(),
            "excluded_terms": set(),
            "budget_max": None,
            "retrieval_scores": {},
            "other_asks": 0,
            "override_seen": False,
            "mode": "unknown",
        }

    def _update_state(self, state: dict, user_message: str) -> None:
        lowered = user_message.lower()
        parsed = parse_intent(user_message)
        if not state["messages"]:
            if "key requirement is:" in lowered:
                state["mode"] = "buying"
            elif "still exploring" in lowered:
                state["mode"] = "browsing"
            elif "i'm looking for" in lowered:
                state["mode"] = "possible_override"
        if re.search(r"\b(actually|instead|ignore|forget|rather)\b", lowered):
            state["override_seen"] = True
            state["mode"] = "override"
            retained_messages = []
            if state["messages"]:
                first_message = state["messages"][0]
                category_match = re.match(r"(.+?\.)", first_message)
                if category_match and "looking for" in category_match.group(1).lower():
                    retained_messages.append(category_match.group(1))
            state["messages"].clear()
            state["slots"] = defaultdict(set)
            state["asked"].clear()
            state["no_preference"].clear()
            state["hard_constraints"].clear()
            state["soft_constraints"].clear()
            state["excluded_terms"].clear()
            state["budget_max"] = None
            state["other_asks"] = 0
            state["messages"].extend(retained_messages)
        if "don't have" in lowered or "no preference" in lowered:
            for attribute in (
                "category", "material", "color", "size", "style", "brand",
                "budget", "feature", "use_case", "other",
            ):
                if attribute in lowered:
                    state["no_preference"].add(attribute)
        state["messages"].append(user_message)
        phrases = _template_phrases(user_message)
        _dedupe_append(state["category_phrases"], phrases.get("category", []))
        _dedupe_append(state["hard_constraints"], phrases.get("hard", []))
        _dedupe_append(state["soft_constraints"], phrases.get("soft", []))
        _dedupe_append(state["hard_constraints"], parsed.required_phrases)
        _dedupe_append(state["soft_constraints"], parsed.preferred_phrases)
        if parsed.product_types:
            state["product_types"] = set(parsed.product_types)
        if parsed.audiences:
            state["audiences"] = set(parsed.audiences)
        state["excluded_terms"].update(parsed.excluded_terms)
        if parsed.budget_max is not None:
            state["budget_max"] = parsed.budget_max
        for attribute, values in parsed.slots.items():
            state["slots"][attribute].update(values)
        for attribute, values in _message_slots(user_message).items():
            state["slots"][attribute].update(values)

    def _fts(self, expression: str, limit: int) -> list[str]:
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def _retrieve(self, state: dict, user_message: str, limit: int) -> list[str]:
        routes: list[tuple[float, list[str]]] = []
        conversation_terms = _terms(" ".join(state["messages"]))
        profile_terms = state["profile_terms"][:8]
        routes.append((0.85, self._fts(_match_expression([*conversation_terms, *profile_terms]), limit)))
        routes.append((1.25, self._fts(_phrase_expression(state["hard_constraints"]), limit)))
        routes.append((0.95, self._fts(_phrase_expression(state["category_phrases"]), limit)))
        routes.append((1.10, self._fts(_phrase_expression([user_message]), limit)))
        combined_constraints = [*state["category_phrases"], *state["hard_constraints"], *state["soft_constraints"]]
        routes.append((1.35, self._fts(_phrase_expression(combined_constraints), limit)))
        routes.append((1.05, self._facet_route(state, limit)))

        fused: dict[str, float] = defaultdict(float)
        for weight, results in routes:
            for rank, parent_asin in enumerate(results, start=1):
                fused[parent_asin] += weight / (60.0 + rank)
        ordered = sorted(fused.items(), key=lambda item: (item[1], item[0]), reverse=True)
        state["retrieval_scores"] = dict(ordered[: limit * 2])
        return [parent_asin for parent_asin, _ in ordered[:limit]]

    def _facet_route(self, state: dict, limit: int) -> list[str]:
        scores: dict[str, float] = defaultdict(float)
        route_terms = {
            "category": set(_terms(" ".join(state["category_phrases"]))),
            "material": set(),
            "color": set(),
            "size": set(),
            "style": set(),
            "use_case": set(),
            "budget": set(),
        }
        for attribute, values in state["slots"].items():
            if attribute in route_terms:
                route_terms[attribute].update(values)
        hard_soft_terms = set(_terms(" ".join([*state["hard_constraints"], *state["soft_constraints"]])))
        for attribute, vocabulary in (
            ("material", MATERIALS),
            ("color", COLORS),
            ("size", SIZE_WORDS),
            ("style", STYLE_WORDS),
            ("use_case", USE_CASE_WORDS),
        ):
            route_terms[attribute].update(hard_soft_terms & vocabulary)

        weights = {
            "category": 1.0,
            "material": 1.7,
            "color": 1.6,
            "size": 1.2,
            "style": 1.0,
            "use_case": 1.2,
            "budget": 0.7,
        }
        for attribute, terms in route_terms.items():
            for term in sorted(terms):
                for parent_asin in sorted(self._facet_lookup.get((attribute, term), set())):
                    scores[parent_asin] += weights.get(attribute, 1.0) * self._term_idf.get(term, 1.0)
        if not scores:
            return []
        ordered = sorted(
            scores.items(),
            key=lambda item: (item[1], self._quality.get(item[0], 0.0), item[0]),
            reverse=True,
        )
        return [parent_asin for parent_asin, _ in ordered[:limit]]

    def _rank(self, state: dict, candidates: list[str]) -> list[str]:
        query_terms = set(_terms(" ".join(state["messages"])))
        hard_terms = set(_terms(" ".join(state["hard_constraints"])))
        soft_terms = set(_terms(" ".join(state["soft_constraints"])))
        category_terms = set(_terms(" ".join(state["category_phrases"])))
        ranked: list[tuple[float, str]] = []
        for position, parent_asin in enumerate(candidates):
            facets = self._facets.get(parent_asin, {})
            score = 6.0 * state["retrieval_scores"].get(parent_asin, 0.0) + 0.2 / (position + 1)
            for attribute, wanted in state["slots"].items():
                if not wanted:
                    continue
                matched = wanted & facets.get(attribute, set())
                score += 2.5 * len(matched)
                if attribute in {"material", "color", "size"} and not matched:
                    score -= 0.15
            textish = set().union(*(values for values in facets.values()))
            score += 0.10 * sum(self._term_idf.get(term, 1.0) for term in hard_terms & textish)
            score += 0.04 * sum(self._term_idf.get(term, 1.0) for term in soft_terms & textish)
            score += 0.06 * sum(self._term_idf.get(term, 1.0) for term in category_terms & textish)
            score += 0.01 * sum(self._term_idf.get(term, 1.0) for term in query_terms & textish)
            product_text = self._product_text.get(parent_asin, "")
            score += 0.30 * _phrase_coverage_score(state["hard_constraints"], product_text)
            score += 0.12 * _phrase_coverage_score(state["soft_constraints"], product_text)
            score += 0.08 * _phrase_coverage_score(state["category_phrases"], product_text)
            score += 0.015 * self._quality.get(parent_asin, 0.0)
            ranked.append((score, parent_asin))
        ranked.sort(reverse=True)
        baseline = [parent_asin for _, parent_asin in ranked]
        reranked: list[tuple[float, str]] = []
        for position, parent_asin in enumerate(baseline[:40]):
            facets = self._facets.get(parent_asin, {})
            textish = facets.get("__all", set())
            slot_matches = 0
            slot_misses = 0
            for attribute, wanted in state["slots"].items():
                if not wanted:
                    continue
                matched = wanted & facets.get(attribute, set())
                slot_matches += len(matched)
                if attribute in {"material", "color", "size"} and not matched:
                    slot_misses += 1
            hard_hit = hard_terms & textish
            soft_hit = soft_terms & textish
            product_text = self._product_text.get(parent_asin, "")
            features = {
                "retrieval": state["retrieval_scores"].get(parent_asin, 0.0),
                "position": 1.0 / (position + 1),
                "slot_matches": float(slot_matches),
                "slot_misses": float(slot_misses),
                "hard_idf": sum(self._term_idf.get(term, 1.0) for term in hard_hit),
                "soft_idf": sum(self._term_idf.get(term, 1.0) for term in soft_hit),
                "category_idf": sum(self._term_idf.get(term, 1.0) for term in category_terms & textish),
                "query_idf": sum(self._term_idf.get(term, 1.0) for term in query_terms & textish),
                "hard_phrase": _phrase_coverage_score(state["hard_constraints"], product_text),
                "soft_phrase": _phrase_coverage_score(state["soft_constraints"], product_text),
                "category_phrase": _phrase_coverage_score(state["category_phrases"], product_text),
                "quality": self._quality.get(parent_asin, 0.0),
                "hard_coverage": len(hard_hit) / max(1, len(hard_terms)),
                "soft_coverage": len(soft_hit) / max(1, len(soft_terms)),
            }
            learned_score = sum(PAIRWISE_WEIGHTS[name] * value for name, value in features.items())
            reranked.append((learned_score, parent_asin))
        reranked.sort(reverse=True)
        learned_scores = {parent_asin: score for score, parent_asin in reranked}
        learned_order = [parent_asin for _, parent_asin in reranked] + baseline[40:]
        compatibility_pool = sorted(
            learned_order[:40],
            key=lambda parent_asin: self._compatibility_key(state, parent_asin),
            reverse=True,
        )
        promoted = [
            parent_asin
            for parent_asin in compatibility_pool
            if self._compatibility_key(state, parent_asin)[:2] == (2, 2)
        ][:3]
        promoted_set = set(promoted)
        remaining = [parent_asin for parent_asin in learned_order if parent_asin not in promoted_set]
        final_order = promoted + remaining
        self._record_rank_confidence(state, final_order, learned_scores)
        state["_last_ranked"] = final_order[:10]
        return final_order

    def _record_rank_confidence(
        self,
        state: dict,
        ranked: list[str],
        learned_scores: dict[str, float],
    ) -> None:
        if not ranked:
            state["rank_confidence"] = {}
            return
        top = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        top_score = learned_scores.get(top, 0.0)
        runner_up_score = learned_scores.get(runner_up, top_score) if runner_up else top_score
        retrieval_order = sorted(
            state["retrieval_scores"],
            key=lambda parent_asin: (state["retrieval_scores"][parent_asin], parent_asin),
            reverse=True,
        )
        retrieval_rank = retrieval_order.index(top) + 1 if top in retrieval_order else len(retrieval_order) + 1
        top_types = self._facets.get(top, {}).get("product_type", set())
        family_agreement = sum(
            bool(top_types & self._facets.get(parent_asin, {}).get("product_type", set()))
            for parent_asin in ranked[:5]
        ) / min(5, len(ranked))
        compatibility = self._compatibility_key(state, top)
        requested_terms = set().union(*(values for values in state["slots"].values()))
        top_terms = self._facets.get(top, {}).get("__all", set())
        state["rank_confidence"] = {
            "score_margin": top_score - runner_up_score,
            "retrieval_rank_reciprocal": 1.0 / retrieval_rank,
            "family_agreement": family_agreement,
            "type_compatibility": compatibility[0],
            "audience_compatibility": compatibility[1],
            "constraint_coverage": len(requested_terms & top_terms) / max(1, len(requested_terms)),
            "constraint_count": len(requested_terms),
        }

    def _compatibility_key(self, state: dict, parent_asin: str) -> tuple[int, int, int, int]:
        facets = self._facets.get(parent_asin, {})
        requested_types = state["product_types"]
        product_types = facets.get("product_type", set())
        if not requested_types:
            type_compatibility = 1
        elif requested_types & product_types:
            type_compatibility = 2
        else:
            type_compatibility = 0

        requested_audiences = state["audiences"]
        product_audiences = facets.get("audience", set())
        if not requested_audiences:
            audience_compatibility = 1
        elif requested_audiences & product_audiences or "unisex" in product_audiences:
            audience_compatibility = 2
        elif product_audiences:
            audience_compatibility = 0
        else:
            audience_compatibility = 1

        excluded = state["excluded_terms"] & facets.get("__all", set())
        exclusion_compatibility = 0 if excluded else 1
        product_terms = facets.get("__all", set())
        constraint_matches = sum(
            len(wanted & product_terms)
            for wanted in state["slots"].values()
            if wanted
        )
        return (
            type_compatibility,
            audience_compatibility,
            exclusion_compatibility,
            constraint_matches,
        )

    def _information_gain_attribute(self, state: dict, candidates: list[str], turn: int) -> str | None:
        if turn >= 9 or not candidates:
            return None
        pool = candidates[:250]
        priority = {
            "other": 1.40,
            "use_case": 1.15,
            "material": 1.10,
            "style": 1.05,
            "color": 1.00,
            "size": 0.90,
            "budget": 0.65,
            "feature": 0.55,
        }
        best_attribute: str | None = None
        best_score = 0.0
        other_cap = 1 if state["mode"] in {"possible_override", "override"} else 2
        if turn <= 6 and state["other_asks"] < other_cap and "other" not in state["no_preference"]:
            return "other"
        for attribute, weight in priority.items():
            if attribute == "other":
                continue
            if attribute in state["asked"] or attribute in state["no_preference"]:
                continue
            if state["slots"].get(attribute):
                continue
            counts: Counter[str] = Counter()
            covered = 0
            for parent_asin in pool:
                values = self._facets.get(parent_asin, {}).get(attribute, set())
                if not values:
                    continue
                covered += 1
                for value in sorted(values)[:4]:
                    counts[value] += 1
            coverage = covered / len(pool)
            if coverage < 0.08:
                continue
            score = _entropy(counts) * coverage * weight
            if attribute == "feature":
                score *= 0.45
            if turn <= 2 and attribute in {"use_case", "material", "style"}:
                score *= 1.12
            if score > best_score:
                best_score = score
                best_attribute = attribute
        return best_attribute if best_score >= 0.25 else None

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = self._sessions[session_id]
        self._update_state(state, user_message)
        candidates = self._retrieve(state, user_message, 350)
        ranked = self._rank(state, candidates)
        ask_attribute = self._information_gain_attribute(state, ranked, turn)
        if ask_attribute:
            state["asked"].add(ask_attribute)
            if ask_attribute == "other":
                state["other_asks"] += 1
        confidence = state.get("rank_confidence", {})
        confident_early_result = (
            confidence.get("score_margin", float("-inf")) >= 2.0
            and confidence.get("retrieval_rank_reciprocal", 0.0) >= 0.05
        )
        defer_initial_results = (
            state["mode"] in {"buying", "browsing"}
            and turn <= 2
            and ask_attribute is not None
            and not confident_early_result
        )
        defer_first_override = (
            state["mode"] == "override"
            and ask_attribute == "other"
            and state["other_asks"] == 1
        )
        defer_results = defer_initial_results or defer_first_override
        recommendations = [] if defer_results else [
            {"parent_asin": parent_asin} for parent_asin in ranked[:top_k]
        ]
        message = "Here are the closest matches I found."
        if ask_attribute:
            if ask_attribute == "other":
                message = "I have some strong matches. Is there another requirement that matters to you?"
            else:
                message = f"Here are strong matches so far. Do you have a {ask_attribute.replace('_', ' ')} preference?"
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
