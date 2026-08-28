from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate
from solution.agent import Agent


def load_samples(path: Path, limit: int | None) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    kwargs = {"mode": "rt", "encoding": "utf-8"} if path.suffix == ".gz" else {"mode": "r", "encoding": "utf-8"}
    samples: list[dict] = []
    with opener(path, **kwargs) as handle:
        for line in handle:
            if line.strip():
                samples.append(json.loads(line))
            if limit is not None and len(samples) >= limit:
                break
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an ASIN-separated synthetic split")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="training/generated/validation.jsonl.gz")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--output", default="training/validation_results.json")
    args = parser.parse_args()

    samples = load_samples(Path(args.dataset), args.limit)
    identifiers, categories, products = catalog_index(args.catalog)
    result = evaluate(Agent(args.catalog), samples, identifiers, categories, products)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
