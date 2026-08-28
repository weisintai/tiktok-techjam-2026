from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from starter.agent import Agent


DEFAULT_PROFILE = {
    "preference_tags": ["comfort", "durability"],
    "purchase_frequency": "occasional",
    "rating_style": "usually positive",
    "summary": "Prefers comfortable and durable products.",
}


def load_product_summaries(catalog_path: str) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            parent_asin = str(product["parent_asin"])
            categories = product.get("categories") or []
            summaries[parent_asin] = {
                "title": str(product.get("title") or "Untitled product"),
                "price": product.get("price"),
                "category": str(categories[-1]) if categories else "Uncategorized",
            }
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat manually with the shopping agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    agent = Agent(args.catalog)
    products = load_product_summaries(args.catalog)
    session_id = f"manual-{uuid.uuid4().hex}"
    agent.reset(session_id, DEFAULT_PROFILE)
    turn = 1

    print("Shopping agent ready. Enter a customer message.")
    print("Commands: /reset starts a new session; /quit exits.")
    while True:
        try:
            message = input(f"\nCustomer [{turn}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message == "/quit":
            break
        if message == "/reset":
            session_id = f"manual-{uuid.uuid4().hex}"
            agent.reset(session_id, DEFAULT_PROFILE)
            turn = 1
            print("Started a new session.")
            continue

        response = agent.respond(session_id, message, turn=turn, top_k=args.top_k)
        print(f"Agent: {response['message']}")
        print(f"Ask attribute: {response.get('ask_attribute')}")
        recommendations = response.get("recommendations") or []
        if recommendations:
            print("Recommendations:")
            for rank, recommendation in enumerate(recommendations, start=1):
                parent_asin = recommendation["parent_asin"]
                product = products.get(parent_asin, {})
                price = product.get("price")
                price_label = f"${price}" if price not in (None, "") else "price unavailable"
                print(f"  {rank:>2}. {product.get('title', 'Unknown product')}")
                print(f"      {parent_asin} | {price_label} | {product.get('category', 'Uncategorized')}")
        else:
            print("Recommendations: deferred pending clarification")
        turn += 1
        if turn > 10:
            print("Session reached the 10-turn competition limit. Use /reset to continue.")


if __name__ == "__main__":
    main()
