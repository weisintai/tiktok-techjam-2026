from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solution.agent import Agent  # noqa: E402


def load_products(catalog_path: Path) -> dict[str, dict[str, Any]]:
    products: dict[str, dict[str, Any]] = {}
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            products[str(item["parent_asin"])] = item
    return products


CATALOG_PATH = ROOT / "data" / "catalog.jsonl"
AGENT = Agent(CATALOG_PATH)
PRODUCTS = load_products(CATALOG_PATH)


def product_payload(asin: str, rank: int) -> dict[str, Any]:
    product = PRODUCTS.get(asin, {})
    categories = product.get("categories") or []
    return {
        "asin": asin,
        "rank": rank,
        "title": product.get("title") or asin,
        "category": " > ".join(str(value) for value in categories[-3:]) or "Catalog item",
        "price": product.get("price"),
        "rating": product.get("average_rating"),
        "ratingCount": product.get("rating_number"),
        "store": product.get("store") or "Amazon catalog",
    }


def handle(payload: dict[str, Any]) -> dict[str, Any]:
    command = payload.get("command")
    session_id = str(payload.get("sessionId") or "demo")
    if command == "reset":
        AGENT.reset(session_id, payload.get("userProfile") or {})
        return {"ok": True, "sessionId": session_id}
    if command != "respond":
        return {"ok": False, "error": f"unknown command: {command}"}

    if session_id not in AGENT.sessions:
        AGENT.reset(session_id, payload.get("userProfile") or {})
    turn = max(1, int(payload.get("turn") or 1))
    response = AGENT.respond(session_id, str(payload.get("message") or ""), turn, 10)
    state = AGENT.sessions[session_id]
    recommendations = [
        product_payload(str(item["parent_asin"]), index + 1)
        for index, item in enumerate(response.get("recommendations") or [])
        if isinstance(item, dict) and item.get("parent_asin")
    ]
    return {
        "ok": True,
        "sessionId": session_id,
        "turn": turn,
        "message": response.get("message", ""),
        "askAttribute": response.get("ask_attribute"),
        "recommendations": recommendations,
        "state": {
            "category": state.get("category", ""),
            "constraints": state.get("constraints", []),
            "slots": state.get("slots", {}),
            "negativeConstraints": state.get("negative_constraints", []),
            "intent": state.get("inferred_intent", "unknown"),
            "softQueries": state.get("soft_queries", []),
            "seenCount": len(state.get("seen", [])),
        },
        "usage": response.get("usage", {}),
    }


for raw in sys.stdin:
    try:
        request = json.loads(raw)
        request_id = request.get("id")
        result = handle(request.get("payload") or {})
        print(json.dumps({"id": request_id, "result": result}), flush=True)
    except Exception as exc:
        print(
            json.dumps({"id": None, "error": f"{type(exc).__name__}: {exc}"}),
            flush=True,
        )
