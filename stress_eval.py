from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from solution.agent import Agent


SYNONYMS = {
    "polyester": "synthetic textile",
    "nylon": "durable synthetic textile",
    "leather": "animal-hide material",
    "wool": "warm fleece-like textile",
    "spandex": "stretchy elastane",
    "silk": "smooth luxury fabric",
    "rayon": "soft regenerated fabric",
    "black": "very dark",
    "white": "pale neutral",
    "blue": "cool-toned",
    "red": "warm vivid",
    "pink": "rosy",
    "green": "earthy",
    "brown": "tan-colored",
    "gray": "neutral-toned",
    "grey": "neutral-toned",
    "running": "jogging",
    "hiking": "trail walking",
    "winter": "cold weather",
    "outdoor": "outside",
    "work": "office use",
}


def semantic_paraphrase(value: str) -> str:
    for source, target in SYNONYMS.items():
        value = re.sub(rf"\b{re.escape(source)}\b", target, value, flags=re.I)
    value = re.sub(
        r"budget around \$([0-9.]+)", r"costing roughly \1 dollars", value, flags=re.I
    )
    return value


def transform_message(message: str) -> str:
    match = re.fullmatch(
        r"I'm looking for (.+?)\. A key requirement is: (.+)\.", message, re.S
    )
    if match:
        return f"Help me find {match.group(1)}; it must have {semantic_paraphrase(match.group(2))}."

    match = re.fullmatch(r"For that, what matters is: (.+)\.", message, re.S)
    if match:
        parts = [semantic_paraphrase(part.strip()) for part in match.group(1).split(";")]
        return "Another thing I care about is " + " and also ".join(parts) + "."

    match = re.fullmatch(
        r"Actually, ignore my earlier preference\. What I need is: (.+)\.", message, re.S
    )
    if match:
        return (
            "Change of plan—drop what I said before and prioritize "
            f"{semantic_paraphrase(match.group(1))} instead."
        )

    match = re.fullmatch(r"I'm looking for (.+), but I'm still exploring\.", message, re.S)
    if match:
        return f"Help me find {match.group(1)}, though I haven't settled on the details."

    return semantic_paraphrase(message)


class InputTransformAgent:
    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return self.agent.respond(session_id, transform_message(user_message), turn, top_k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="stress_results.json")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--browsing-dense", action="store_true")
    parser.add_argument("--cross-encoder", action="store_true")
    parser.add_argument("--adaptive-questions", action="store_true")
    parser.add_argument("--reference-feedback", action="store_true")
    parser.add_argument("--profile-tiebreak", action="store_true")
    parser.add_argument("--experimental-router", action="store_true")
    parser.add_argument("--field-reranker", action="store_true")
    parser.add_argument("--trigram-retrieval", action="store_true")
    parser.add_argument("--confidence-topk", action="store_true")
    parser.add_argument("--learned-reranker")
    parser.add_argument(
        "--learned-reranker-scope", choices=("off", "freeform", "all"), default="freeform"
    )
    parser.add_argument(
        "--learned-reranker-policy", choices=("full", "exact_tier", "blend"), default="full"
    )
    parser.add_argument("--learned-reranker-weight", type=float, default=0.4)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    ids, categories, products = catalog_index(args.catalog)
    base = Agent(
        args.catalog,
        model_name="sentence-transformers/all-MiniLM-L6-v2" if args.dense or args.browsing_dense else None,
        cross_encoder_name="cross-encoder/ms-marco-MiniLM-L6-v2" if args.cross_encoder else None,
        adaptive_questions=args.adaptive_questions,
        profile_tiebreak=args.profile_tiebreak,
        experimental_router=args.experimental_router or args.browsing_dense,
        dense_routes=("browsing",) if args.browsing_dense else ("browsing", "uncertain", "hybrid"),
        reference_feedback=args.reference_feedback,
        field_reranker=args.field_reranker,
        trigram_retrieval=args.trigram_retrieval,
        confidence_topk=args.confidence_topk,
        learned_reranker_path=args.learned_reranker,
        learned_reranker_scope=args.learned_reranker_scope,
        learned_reranker_policy=args.learned_reranker_policy,
        learned_reranker_weight=args.learned_reranker_weight,
    )
    result = evaluate(InputTransformAgent(base), samples, ids, categories, products)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
