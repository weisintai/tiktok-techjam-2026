from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evaluator.local_evaluator import load_jsonl
from training.generate_sessions import user_profile


SCENARIOS = ("buying", "browsing", "intent_override", "boundary", "reference_feedback")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare disjoint packets for blind human writers")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-set", default="data/public_set.jsonl")
    parser.add_argument("--output-dir", default="training/blind_packets")
    parser.add_argument("--writers", type=int, default=3)
    parser.add_argument("--sessions-per-writer", type=int, default=20)
    args = parser.parse_args()
    if args.writers < 2 or args.sessions_per_writer < 1:
        parser.error("use at least two writers and one session per writer")

    quarantined = {
        str(row["ground_truth"]["parent_asin"])
        for row in load_jsonl(args.public_set)
    }
    products = []
    seen = set()
    with Path(args.catalog).open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            asin = str(product["parent_asin"])
            if asin in quarantined or asin in seen:
                continue
            seen.add(asin)
            digest = hashlib.sha256(f"blind-v1\0{asin}".encode()).hexdigest()
            products.append((digest, product))
    products.sort(key=lambda item: item[0])
    required = args.writers * args.sessions_per_writer
    if len(products) < required:
        raise RuntimeError("not enough non-public products")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for writer in range(args.writers):
        path = output_dir / f"writer_{writer + 1:02d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for local_index in range(args.sessions_per_writer):
                global_index = writer * args.sessions_per_writer + local_index
                product = products[global_index][1]
                categories = product.get("categories", [])
                row = {
                    "case_id": f"blind_w{writer + 1:02d}_{local_index + 1:03d}",
                    "writer_id": f"writer_{writer + 1:02d}",
                    "split": "development" if local_index < args.sessions_per_writer // 2 else "test",
                    "scenario_type": SCENARIOS[global_index % len(SCENARIOS)],
                    "target_asin": str(product["parent_asin"]),
                    "product_brief": {
                        "title": product.get("title", ""),
                        "categories": categories[-3:] if isinstance(categories, list) else categories,
                        "features": (product.get("features") or [])[:6],
                        "description": (product.get("description") or [])[:2],
                    },
                    "user_profile": user_profile(product),
                    "turns": [],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "writers": args.writers,
        "sessions_per_writer": args.sessions_per_writer,
        "sessions": required,
        "output_dir": str(output_dir),
        "instruction": "Each writer fills turns with 1-10 natural shopper messages without viewing agent code.",
    }, indent=2))


if __name__ == "__main__":
    main()
