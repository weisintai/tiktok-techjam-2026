from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from evaluator.local_evaluator import (
    MAX_TURNS,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.agent import Agent, _phrase_coverage_score, _terms


FEATURES = (
    "retrieval", "position", "slot_matches", "slot_misses", "hard_idf",
    "soft_idf", "category_idf", "query_idf", "hard_phrase", "soft_phrase",
    "category_phrase", "quality", "hard_coverage", "soft_coverage",
)


def candidate_features(agent: Agent, state: dict, candidates: list[str]) -> dict[str, list[float]]:
    query_terms = set(_terms(" ".join(state["messages"])))
    hard_terms = set(_terms(" ".join(state["hard_constraints"])))
    soft_terms = set(_terms(" ".join(state["soft_constraints"])))
    category_terms = set(_terms(" ".join(state["category_phrases"])))
    rows: dict[str, list[float]] = {}
    for position, parent_asin in enumerate(candidates):
        facets = agent._facets[parent_asin]
        text_terms = facets["__all"]
        slot_matches = 0
        slot_misses = 0
        for attribute, wanted in state["slots"].items():
            if not wanted:
                continue
            matched = wanted & facets.get(attribute, set())
            slot_matches += len(matched)
            if attribute in {"material", "color", "size"} and not matched:
                slot_misses += 1
        product_text = agent._product_text[parent_asin]
        hard_hit = hard_terms & text_terms
        soft_hit = soft_terms & text_terms
        rows[parent_asin] = [
            state["retrieval_scores"].get(parent_asin, 0.0),
            1.0 / (position + 1),
            float(slot_matches),
            float(slot_misses),
            sum(agent._term_idf.get(term, 1.0) for term in hard_hit),
            sum(agent._term_idf.get(term, 1.0) for term in soft_hit),
            sum(agent._term_idf.get(term, 1.0) for term in category_terms & text_terms),
            sum(agent._term_idf.get(term, 1.0) for term in query_terms & text_terms),
            _phrase_coverage_score(state["hard_constraints"], product_text),
            _phrase_coverage_score(state["soft_constraints"], product_text),
            _phrase_coverage_score(state["category_phrases"], product_text),
            agent._quality.get(parent_asin, 0.0),
            len(hard_hit) / max(1, len(hard_terms)),
            len(soft_hit) / max(1, len(soft_terms)),
        ]
    return rows


def collect_records(agent: Agent, samples: list[dict], categories: dict, products: dict) -> list[dict]:
    records: list[dict] = []
    for sample in samples:
        session_id = sample["sample_id"]
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)
        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, message, turn, 10)
            state = agent._sessions[session_id]
            candidates = list(state["retrieval_scores"])[:350]
            ranked = agent._rank(state, candidates)
            if override_applied:
                features = candidate_features(agent, state, ranked[:40])
                records.append({
                    "session": session_id,
                    "scenario": sample["scenario_type"],
                    "turn": turn,
                    "target": target,
                    "candidates": ranked[:40],
                    "features": features,
                })
                if target in ranked[:10]:
                    break
            if turn == MAX_TURNS:
                break
            override = behavior.get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                message, boundary_used = customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )
    return records


def pair_differences(records: list[dict], sessions: set[str]) -> list[list[float]]:
    pairs: list[list[float]] = []
    for record in records:
        if record["session"] not in sessions or record["target"] not in record["features"]:
            continue
        positive = record["features"][record["target"]]
        negatives = [asin for asin in record["candidates"][:20] if asin != record["target"]]
        for asin in negatives:
            pairs.append([p - n for p, n in zip(positive, record["features"][asin])])
    return pairs


def fit(pairs: list[list[float]], epochs: int = 120) -> tuple[list[float], list[float]]:
    scales = []
    for column in range(len(FEATURES)):
        mean_square = sum(row[column] ** 2 for row in pairs) / max(1, len(pairs))
        scales.append(max(1e-6, math.sqrt(mean_square)))
    normalized = [[value / scales[i] for i, value in enumerate(row)] for row in pairs]
    weights = [0.0] * len(FEATURES)
    rng = random.Random(2026)
    for epoch in range(epochs):
        rng.shuffle(normalized)
        rate = 0.08 / math.sqrt(epoch + 1)
        for row in normalized:
            margin = max(-30.0, min(30.0, sum(w * x for w, x in zip(weights, row))))
            gradient = 1.0 / (1.0 + math.exp(margin))
            for i, value in enumerate(row):
                weights[i] += rate * (gradient * value - 0.0005 * weights[i])
    return [weight / scale for weight, scale in zip(weights, scales)], scales


def score_records(records: list[dict], sessions: set[str], weights: list[float]) -> float:
    by_session: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["session"] in sessions:
            by_session[record["session"]].append(record)
    reciprocal_ranks = []
    for session_records in by_session.values():
        reciprocal_rank = 0.0
        for record in sorted(session_records, key=lambda item: item["turn"]):
            ordered = sorted(
                record["candidates"],
                key=lambda asin: (sum(w * x for w, x in zip(weights, record["features"][asin])), asin),
                reverse=True,
            )
            if record["target"] in ordered[:10]:
                reciprocal_rank = 1.0 / (ordered.index(record["target"]) + 1)
                break
        reciprocal_ranks.append(reciprocal_rank)
    return sum(reciprocal_ranks) / max(1, len(reciprocal_ranks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results/results_pairwise_reranker.json")
    args = parser.parse_args()
    samples = load_jsonl(args.dataset)
    _, categories, products = catalog_index(args.catalog)
    records = collect_records(Agent(args.catalog), samples, categories, products)
    session_ids = sorted({record["session"] for record in records})
    folds = [set(session_ids[index::5]) for index in range(5)]
    baseline_weights = [6.0, 0.2, 2.5, -0.15, 0.10, 0.04, 0.06, 0.01, 0.30, 0.12, 0.08, 0.015, 0.0, 0.0]
    fold_results = []
    for fold_index, validation in enumerate(folds):
        training = set(session_ids) - validation
        weights, _ = fit(pair_differences(records, training))
        fold_results.append({
            "fold": fold_index,
            "validation_sessions": len(validation),
            "baseline_mrr": score_records(records, validation, baseline_weights),
            "mrr": score_records(records, validation, weights),
        })
    weights, scales = fit(pair_differences(records, set(session_ids)))
    result = {
        "features": list(FEATURES),
        "weights": weights,
        "scales": scales,
        "folds": fold_results,
        "baseline_mrr": score_records(records, set(session_ids), baseline_weights),
        "mean_validation_mrr": sum(item["mrr"] for item in fold_results) / len(fold_results),
        "training_mrr": score_records(records, set(session_ids), weights),
        "record_count": len(records),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
