from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate
from solution.agent import Agent
from training.evaluate_split import load_samples


VARIANTS = {
    "baseline": {},
    "field_reranker": {"field_reranker": True},
    "trigram_retrieval": {"trigram_retrieval": True},
    "confidence_topk": {"confidence_topk": True},
    "combined": {
        "field_reranker": True,
        "trigram_retrieval": True,
        "confidence_topk": True,
    },
}


def summary(result: dict) -> dict:
    values = {
        key: result[key]
        for key in ("hit_rate_at_10", "mrr", "mttc")
    }
    efficiency = max(0.0, min(1.0, (11.0 - float(values["mttc"])) / 10.0))
    values["technical_score"] = round(
        0.5 * values["hit_rate_at_10"] + 0.3 * values["mrr"] + 0.2 * efficiency,
        6,
    )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare gated ranking experiments")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS))
    parser.add_argument("--output")
    parser.add_argument("--learned-reranker")
    parser.add_argument(
        "--learned-reranker-scopes",
        nargs="+",
        choices=("off", "freeform", "all"),
        default=("freeform",),
    )
    args = parser.parse_args()

    samples = load_samples(Path(args.dataset), args.limit)
    identifiers, categories, products = catalog_index(args.catalog)
    results = {}
    for name in args.variants:
        evaluated = evaluate(
            Agent(args.catalog, **VARIANTS[name]),
            samples,
            identifiers,
            categories,
            products,
        )
        results[name] = summary(evaluated)
        print(name, json.dumps(results[name], sort_keys=True))
    if args.learned_reranker:
        for scope in args.learned_reranker_scopes:
            name = f"learned_reranker_{scope}"
            evaluated = evaluate(
                Agent(
                    args.catalog,
                    learned_reranker_path=args.learned_reranker,
                    learned_reranker_scope=scope,
                ),
                samples,
                identifiers,
                categories,
                products,
            )
            results[name] = summary(evaluated)
            print(name, json.dumps(results[name], sort_keys=True))
    report = {"dataset": args.dataset, "sessions": len(samples), "results": results}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
