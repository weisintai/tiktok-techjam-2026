from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from solution.agent import Agent


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "demo"


@dataclass
class DemoSession:
    agent_key: str
    sample_id: str | None
    session_id: str
    target_asin: str | None
    turn: int = 0


def _product_payload(product: dict[str, Any] | None, *, asin: str | None = None) -> dict[str, Any] | None:
    if product is None:
        return None
    categories = [str(value) for value in product.get("categories") or []]
    return {
        "parent_asin": asin or str(product.get("parent_asin", "")),
        "title": str(product.get("title") or "Unknown product"),
        "store": str(product.get("store") or ""),
        "categories": categories,
        "price": product.get("price"),
        "features": [str(value) for value in (product.get("features") or [])[:4]],
    }


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


class DemoApp:
    def __init__(self, catalog_path: Path, dataset_path: Path) -> None:
        self.catalog_path = catalog_path
        self.dataset_path = dataset_path
        self.catalog_ids, self.categories, self.products = catalog_index(catalog_path)
        self.samples = load_jsonl(dataset_path)
        self.samples_by_id = {str(sample["sample_id"]): sample for sample in self.samples}
        self.sessions: dict[str, DemoSession] = {}
        self.agents: dict[str, Agent] = {"baseline": Agent(catalog_path)}
        self.lock = threading.Lock()

    def close(self) -> None:
        for agent in self.agents.values():
            agent.close()

    def _get_agent(self, mode: str) -> tuple[Agent, bool]:
        if mode not in {"baseline", "dense"}:
            mode = "baseline"
        if mode == "baseline":
            return self.agents["baseline"], False
        agent = self.agents.get("dense")
        if agent is None:
            agent = Agent(
                self.catalog_path,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            )
            self.agents["dense"] = agent
        dense_loaded = bool(agent.model is not None and agent.embeddings is not None)
        return agent, dense_loaded

    def bootstrap(self) -> dict[str, Any]:
        sample_cards = []
        for sample in self.samples:
            sample_id = str(sample["sample_id"])
            target_asin = str(sample["ground_truth"]["parent_asin"])
            intent_card, behavior = materialize_hidden_fields(sample, self.products)
            starter = initial_message(
                sample | {"intent_card": intent_card, "behavior": behavior},
                coarse_category(self.categories.get(target_asin, [])),
                set(),
            )
            sample_cards.append({
                "sample_id": sample_id,
                "scenario_type": str(sample["scenario_type"]),
                "difficulty_bucket": str(sample.get("difficulty_bucket", "")),
                "starter_prompt": starter,
                "profile_summary": str(sample.get("user_profile", {}).get("summary", "")),
                "target": _product_payload(self.products.get(target_asin), asin=target_asin),
            })
        return {
            "catalog_path": str(self.catalog_path),
            "dataset_path": str(self.dataset_path),
            "sample_count": len(sample_cards),
            "samples": sample_cards,
        }

    def create_session(self, sample_id: str | None, mode: str) -> dict[str, Any]:
        agent, dense_loaded = self._get_agent(mode)
        sample = self.samples_by_id.get(sample_id or "")
        target_asin = None
        starter_prompt = ""
        sample_payload = None
        if sample is not None:
            target_asin = str(sample["ground_truth"]["parent_asin"])
            intent_card, behavior = materialize_hidden_fields(sample, self.products)
            starter_prompt = initial_message(
                sample | {"intent_card": intent_card, "behavior": behavior},
                coarse_category(self.categories.get(target_asin, [])),
                set(),
            )
            sample_payload = {
                "sample_id": str(sample["sample_id"]),
                "scenario_type": str(sample["scenario_type"]),
                "difficulty_bucket": str(sample.get("difficulty_bucket", "")),
                "profile_summary": str(sample.get("user_profile", {}).get("summary", "")),
                "target": _product_payload(self.products.get(target_asin), asin=target_asin),
            }
        session_token = uuid.uuid4().hex
        session_id = f"demo_{session_token}"
        agent.reset(session_id, {} if sample is None else sample["user_profile"])
        with self.lock:
            self.sessions[session_token] = DemoSession(
                agent_key=mode if mode in {"baseline", "dense"} else "baseline",
                sample_id=str(sample["sample_id"]) if sample is not None else None,
                session_id=session_id,
                target_asin=target_asin,
            )
        return {
            "session_token": session_token,
            "sample": sample_payload,
            "starter_prompt": starter_prompt,
            "dense_loaded": dense_loaded,
            "mode": mode,
        }

    @staticmethod
    def _output_limit(turn: int, diagnostics: dict[str, float]) -> int:
        if turn >= 10:
            return 10
        if turn <= 6:
            return 1
        if diagnostics.get("complete_match_count", 0.0) > 100:
            return 5
        return 3

    def respond(self, session_token: str, prompt: str, top_k: int = 10) -> dict[str, Any]:
        with self.lock:
            session = self.sessions.get(session_token)
        if session is None:
            raise KeyError("Unknown session")
        agent, dense_loaded = self._get_agent(session.agent_key)
        session.turn += 1
        ranked, diagnostics, __ = agent.update_and_rank(session.session_id, prompt, session.turn)
        state = agent.sessions[session.session_id]
        output_limit = min(top_k, self._output_limit(session.turn, diagnostics))
        visible_ranked = [asin for asin in ranked if asin not in state["seen"]]
        returned_asins = visible_ranked[:output_limit]
        state["seen"].update(returned_asins)
        ask_attribute = agent._select_question(ranked, state)
        state["asked_attributes"].add(ask_attribute)
        target_rank_all = None
        target_rank_visible = None
        if session.target_asin:
            if session.target_asin in ranked:
                target_rank_all = ranked.index(session.target_asin) + 1
            if session.target_asin in visible_ranked:
                target_rank_visible = visible_ranked.index(session.target_asin) + 1
        target = _product_payload(
            self.products.get(session.target_asin) if session.target_asin else None,
            asin=session.target_asin,
        )
        return {
            "turn": session.turn,
            "mode": session.agent_key,
            "dense_loaded": dense_loaded,
            "message": agent._question_message(ask_attribute, ranked, state),
            "ask_attribute": ask_attribute,
            "returned_count": len(returned_asins),
            "returned": [
                self._decorate_product(
                    asin,
                    state["category"],
                    state["constraints"],
                    state["negative_constraints"],
                    rank=index + 1,
                    returned=True,
                    target_asin=session.target_asin,
                )
                for index, asin in enumerate(returned_asins)
            ],
            "top_10_preview": [
                self._decorate_product(
                    asin,
                    state["category"],
                    state["constraints"],
                    state["negative_constraints"],
                    rank=index + 1,
                    returned=asin in returned_asins,
                    target_asin=session.target_asin,
                )
                for index, asin in enumerate(visible_ranked[:10])
            ],
            "target": target,
            "target_rank_all": target_rank_all,
            "target_rank_visible": target_rank_visible,
            "target_in_returned": bool(session.target_asin and session.target_asin in returned_asins),
            "state": {
                "category": state["category"],
                "constraints": list(state["constraints"]),
                "negative_constraints": list(state["negative_constraints"]),
                "inferred_intent": state["inferred_intent"],
            },
            "diagnostics": diagnostics,
        }

    def _decorate_product(
        self,
        asin: str,
        category: str,
        constraints: list[str],
        negative_constraints: list[str],
        *,
        rank: int,
        returned: bool,
        target_asin: str | None,
    ) -> dict[str, Any]:
        product = self.products.get(asin)
        payload = _product_payload(product, asin=asin) or {"parent_asin": asin, "title": asin, "categories": [], "store": "", "price": None, "features": []}
        haystack = _normalize_text(" ".join([
            payload["title"],
            " ".join(payload["categories"]),
            " ".join(payload["features"]),
            payload["store"],
        ]))
        reasons: list[str] = []
        if category and any(term in haystack for term in _normalize_text(category).split()):
            reasons.append(f"Matches category intent: {category}")
        for value in constraints[:4]:
            needle = _normalize_text(value.replace("color:", "").replace("budget under", "").strip())
            if needle and all(term in haystack for term in needle.split()[:3]):
                reasons.append(f"Aligned with: {value}")
        for value in negative_constraints[:2]:
            needle = _normalize_text(value)
            if needle and any(term in haystack for term in needle.split()[:3]):
                reasons.append(f"Potential conflict: {value}")
        if not reasons and payload["features"]:
            reasons.append(f"Top feature hint: {payload['features'][0]}")
        payload["rank"] = rank
        payload["returned"] = returned
        payload["is_target"] = bool(target_asin and asin == target_asin)
        payload["reasons"] = reasons[:3]
        return payload


class DemoRequestHandler(BaseHTTPRequestHandler):
    app: DemoApp

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/bootstrap":
            self._write_json(self.app.bootstrap())
            return
        if self.path == "/" or self.path == "/index.html":
            self._serve_file(STATIC_DIR / "index.html")
            return
        if self.path.startswith("/"):
            candidate = STATIC_DIR / self.path.lstrip("/")
            if candidate.is_file() and candidate.resolve().is_relative_to(STATIC_DIR.resolve()):
                self._serve_file(candidate)
                return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        payload = self._read_json()
        if self.path == "/api/session":
            result = self.app.create_session(payload.get("sample_id"), str(payload.get("mode", "baseline")))
            self._write_json(result)
            return
        if self.path == "/api/respond":
            try:
                result = self.app.respond(
                    str(payload.get("session_token", "")),
                    str(payload.get("prompt", "")),
                    int(payload.get("top_k", 10)),
                )
            except KeyError:
                self.send_error(HTTPStatus.NOT_FOUND, "Unknown session")
                return
            self._write_json(result)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {}
        return payload if isinstance(payload, dict) else {}

    def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _serve_file(self, path: Path) -> None:
        mime, __ = mimetypes.guess_type(path.name)
        raw = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime or 'text/plain'}; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local demo UI for the shopping copilot")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = DemoApp(Path(args.catalog), Path(args.dataset))
    DemoRequestHandler.app = app
    server = ThreadingHTTPServer((args.host, args.port), DemoRequestHandler)
    try:
        print(f"Demo running at http://{args.host}:{args.port}")
        server.serve_forever()
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
