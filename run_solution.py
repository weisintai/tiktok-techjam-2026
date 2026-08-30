from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from solution.agent import Agent
from solution.extraction import LlamaCppExtractor, TimeoutExtractor, TransformersLocalExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="solution_results.json")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument(
        "--browsing-dense",
        action="store_true",
        help="Route intent explicitly and use dense candidates only for browsing",
    )
    parser.add_argument("--cross-encoder", action="store_true")
    parser.add_argument("--adaptive-questions", action="store_true")
    parser.add_argument("--reference-feedback", action="store_true")
    parser.add_argument("--profile-tiebreak", action="store_true")
    parser.add_argument("--experimental-router", action="store_true")
    parser.add_argument(
        "--no-override-retain-hard",
        dest="override_retain_hard",
        action="store_false",
        help="Discard the superseded preference instead of re-admitting it after an override",
    )
    parser.add_argument(
        "--popularity-tiebreak",
        action="store_true",
        help="Break ties inside metadata-identical blocks by catalog review volume",
    )
    parser.add_argument("--popularity-gate", type=int, default=10)
    parser.add_argument("--popularity-min-turn", type=int, default=3)
    parser.add_argument(
        "--recombine-constraints",
        action="store_true",
        help="Rejoin '; '-split fragments that form a single catalog value",
    )
    parser.add_argument("--learned-reranker", help="Optional joblib top-50 reranker artifact")
    parser.add_argument(
        "--learned-reranker-scope",
        choices=("off", "freeform", "all"),
        default="freeform",
    )
    parser.add_argument(
        "--learned-reranker-policy", choices=("full", "exact_tier", "blend"), default="full"
    )
    parser.add_argument("--learned-reranker-weight", type=float, default=0.4)
    parser.add_argument(
        "--extractor-model",
        help="Local Hugging Face causal model path/name, e.g. Qwen/Qwen3-0.6B",
    )
    parser.add_argument("--extractor-gguf", help="Local GGUF model for llama.cpp extraction")
    parser.add_argument("--extraction-min-confidence", type=float, default=0.55)
    parser.add_argument("--extraction-timeout", type=float, default=3.0)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    ids, categories, products = catalog_index(args.catalog)
    if args.extractor_gguf and args.extractor_model:
        parser.error("choose only one of --extractor-gguf and --extractor-model")
    extractor = None
    if args.extractor_gguf:
        extractor = LlamaCppExtractor(args.extractor_gguf)
    elif args.extractor_model:
        extractor = TransformersLocalExtractor(args.extractor_model)
    if extractor is not None:
        extractor = TimeoutExtractor(extractor, args.extraction_timeout)
    agent = Agent(
        args.catalog,
        model_name="sentence-transformers/all-MiniLM-L6-v2" if args.dense or args.browsing_dense else None,
        cross_encoder_name="cross-encoder/ms-marco-MiniLM-L6-v2" if args.cross_encoder else None,
        adaptive_questions=args.adaptive_questions,
        profile_tiebreak=args.profile_tiebreak,
        structured_extractor=extractor,
        extraction_min_confidence=args.extraction_min_confidence,
        experimental_router=args.experimental_router or args.browsing_dense,
        dense_routes=("browsing",) if args.browsing_dense else ("browsing", "uncertain", "hybrid"),
        reference_feedback=args.reference_feedback,
        override_retain_hard=args.override_retain_hard,
        popularity_tiebreak=args.popularity_tiebreak,
        popularity_gate=args.popularity_gate,
        popularity_min_turn=args.popularity_min_turn,
        recombine_constraints=args.recombine_constraints,
        learned_reranker_path=args.learned_reranker,
        learned_reranker_scope=args.learned_reranker_scope,
        learned_reranker_policy=args.learned_reranker_policy,
        learned_reranker_weight=args.learned_reranker_weight,
    )
    result = evaluate(agent, samples, ids, categories, products)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
