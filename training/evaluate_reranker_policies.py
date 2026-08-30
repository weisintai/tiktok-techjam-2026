from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from evaluator.local_evaluator import load_jsonl
from solution.agent import Agent
from training.train_learned_reranker import collect, load_products


def policy_metrics(model: object, queries: list[tuple[str, list[str], list[list[float]]]]) -> dict:
    policies: dict[str, list[float]] = {
        "baseline": [],
        "learned_full": [],
        "learned_top5": [],
        "learned_top10": [],
        "exact_tier_learned": [],
        **{f"blend_{weight:.2f}": [] for weight in (0.05, 0.10, 0.20, 0.30, 0.40)},
    }
    for target, candidates, features in queries:
        if target not in candidates:
            for values in policies.values():
                values.append(0.0)
            continue
        probabilities = model.predict_proba(features)[:, 1]
        baseline_rank = {asin: rank for rank, asin in enumerate(candidates, 1)}

        def record(name: str, ordered: list[str]) -> None:
            policies[name].append(1.0 / (ordered.index(target) + 1))

        record("baseline", candidates)
        learned = [
            asin for _, _, asin in sorted(
                zip(probabilities, range(len(candidates)), candidates),
                key=lambda item: (item[0], -item[1]),
                reverse=True,
            )
        ]
        record("learned_full", learned)
        for size in (5, 10):
            head = [asin for asin in learned if baseline_rank[asin] <= size]
            ordered = head + candidates[size:]
            record(f"learned_top{size}", ordered)
        tiered = [
            asin for _, _, _, asin in sorted(
                (
                    -features[index][2],
                    -float(probabilities[index]),
                    index,
                    asin,
                )
                for index, asin in enumerate(candidates)
            )
        ]
        record("exact_tier_learned", tiered)
        for weight in (0.05, 0.10, 0.20, 0.30, 0.40):
            scores = [
                (1.0 - weight) / (index + 1) + weight * float(probability)
                for index, probability in enumerate(probabilities)
            ]
            blended = [
                asin for _, _, asin in sorted(
                    zip(scores, range(len(candidates)), candidates),
                    key=lambda item: (item[0], -item[1]),
                    reverse=True,
                )
            ]
            record(f"blend_{weight:.2f}", blended)
    return {
        name: round(float(np.mean(values)), 6) if values else 0.0
        for name, values in policies.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare conservative learned reranking policies")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-set", default="data/public_set.jsonl")
    parser.add_argument("--model", default="artifacts/models/catalog_reranker.joblib")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--products", type=int, default=300)
    parser.add_argument("--output")
    args = parser.parse_args()

    catalog = load_products(args.catalog)
    quarantine = {
        str(row["ground_truth"]["parent_asin"]) for row in load_jsonl(args.public_set)
    }
    agent = Agent(args.catalog)
    _, _, queries = collect(agent, catalog, args.split, quarantine, args.products, 12, 0)
    model = joblib.load(args.model)["model"]
    report = {
        "split": args.split,
        "products": args.products,
        "queries": len(queries),
        "mrr": policy_metrics(model, queries),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
