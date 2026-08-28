from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, TextIO

from evaluator.local_evaluator import classify_constraint, intent_card


SCENARIOS = ("buying", "browsing", "intent_override", "boundary")


def split_for_asin(parent_asin: str, seed: int = 20260828) -> str:
    """Assign an ASIN to a stable 80/10/10 split without Python hash randomness."""
    digest = hashlib.sha256(f"{seed}\0{parent_asin}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def deterministic_choice(parent_asin: str, values: tuple[str, ...], salt: str) -> str:
    digest = hashlib.sha256(f"{salt}\0{parent_asin}".encode()).digest()
    return values[int.from_bytes(digest[:4], "big") % len(values)]


def user_profile(product: dict) -> dict:
    parent_asin = str(product["parent_asin"])
    card = intent_card(product)
    constraints = [*card["hard_constraints"], *card["soft_preferences"]]
    tags = list(dict.fromkeys(classify_constraint(str(value)) for value in constraints))[:3]
    try:
        rating = float(product.get("average_rating", 4.0))
    except (TypeError, ValueError):
        rating = 4.0
    return {
        "preference_tags": tags or ["feature"],
        "average_prior_rating": round(rating, 2),
        "purchase_frequency": deterministic_choice(
            parent_asin, ("occasional", "regular", "frequent"), "frequency"
        ),
        "rating_style": "selective" if rating < 4.0 else "positive",
        "summary": "Synthetic catalog-derived profile; weak prior only.",
    }


def generated_samples(product: dict) -> Iterable[dict]:
    parent_asin = str(product["parent_asin"])
    profile = user_profile(product)
    for scenario in SCENARIOS:
        yield {
            "sample_id": f"synthetic_{parent_asin}_{scenario}",
            "scenario_type": scenario,
            "category_bucket": "clothing_shoes_jewelry",
            "difficulty_bucket": deterministic_choice(
                parent_asin, ("easy", "medium", "hard"), f"difficulty:{scenario}"
            ),
            "user_profile": profile,
            "ground_truth": {"parent_asin": parent_asin},
        }


def _open_output(path: Path, compress: bool) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        return gzip.open(path, "wt", encoding="utf-8", newline="\n")
    return path.open("w", encoding="utf-8", newline="\n")


def generate(
    catalog_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 20260828,
    limit_products: int | None = None,
    compress: bool = True,
    quarantine_asins: set[str] | None = None,
) -> dict:
    catalog_path = Path(catalog_path)
    output_dir = Path(output_dir)
    suffix = ".jsonl.gz" if compress else ".jsonl"
    split_names = ("train", "validation", "test", "quarantine")
    paths = {name: output_dir / f"{name}{suffix}" for name in split_names}
    handles = {name: _open_output(path, compress) for name, path in paths.items()}
    product_counts: Counter[str] = Counter()
    session_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    seen_asins: set[str] = set()
    try:
        with catalog_path.open(encoding="utf-8") as catalog:
            for line in catalog:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                if parent_asin in seen_asins:
                    continue
                seen_asins.add(parent_asin)
                split = (
                    "quarantine"
                    if parent_asin in (quarantine_asins or set())
                    else split_for_asin(parent_asin, seed)
                )
                product_counts[split] += 1
                for sample in generated_samples(product):
                    handles[split].write(json.dumps(sample, separators=(",", ":")) + "\n")
                    session_counts[split] += 1
                    scenario_counts[sample["scenario_type"]] += 1
                if limit_products is not None and len(seen_asins) >= limit_products:
                    break
    finally:
        for handle in handles.values():
            handle.close()

    metadata = {
        "schema_version": 1,
        "catalog": str(catalog_path),
        "seed": seed,
        "split_policy": "stable SHA256 ASIN split: train 80%, validation 10%, test 10%",
        "products": dict(sorted(product_counts.items())),
        "sessions": dict(sorted(session_counts.items())),
        "scenarios": dict(sorted(scenario_counts.items())),
        "files": {name: str(path) for name, path in paths.items()},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leakage-safe catalog sessions")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output-dir", default="training/generated")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--limit-products", type=int)
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument(
        "--quarantine-dataset",
        default="data/public_set.jsonl",
        help="JSONL targets excluded from train/validation/test; empty string disables",
    )
    args = parser.parse_args()
    quarantine_asins: set[str] = set()
    if args.quarantine_dataset:
        with Path(args.quarantine_dataset).open(encoding="utf-8") as handle:
            quarantine_asins = {
                str(json.loads(line)["ground_truth"]["parent_asin"])
                for line in handle if line.strip()
            }
    metadata = generate(
        args.catalog,
        args.output_dir,
        seed=args.seed,
        limit_products=args.limit_products,
        compress=not args.no_compress,
        quarantine_asins=quarantine_asins,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
