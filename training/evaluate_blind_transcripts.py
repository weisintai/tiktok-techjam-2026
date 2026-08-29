from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evaluator.local_evaluator import catalog_index, load_jsonl, metric_summary, normalize_recommendations
from solution.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Score independently written fixed transcripts")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--split", choices=("development", "test"))
    parser.add_argument("--reference-feedback", action="store_true")
    parser.add_argument("--adaptive-questions", action="store_true")
    parser.add_argument("--field-reranker", action="store_true")
    parser.add_argument("--trigram-retrieval", action="store_true")
    parser.add_argument("--confidence-topk", action="store_true")
    parser.add_argument("--learned-reranker")
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = [row for path in args.cases for row in load_jsonl(path)]
    if args.split:
        rows = [row for row in rows if row.get("split") == args.split]
    catalog_ids, _, _ = catalog_index(args.catalog)
    agent = Agent(
        args.catalog,
        reference_feedback=args.reference_feedback,
        adaptive_questions=args.adaptive_questions,
        field_reranker=args.field_reranker,
        trigram_retrieval=args.trigram_retrieval,
        confidence_topk=args.confidence_topk,
        learned_reranker_path=args.learned_reranker,
    )
    sessions = []
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        turns = row.get("turns")
        if not isinstance(turns, list) or not 1 <= len(turns) <= 10 or not all(
            isinstance(turn, str) and turn.strip() for turn in turns
        ):
            raise ValueError(f"{row.get('case_id')}: turns must contain 1-10 messages")
        session_id = str(row["case_id"])
        target = str(row["target_asin"])
        agent.reset(session_id, row.get("user_profile", {}))
        hit_turn = best_rank = None
        for turn_number, message in enumerate(turns, 1):
            response = agent.respond(session_id, message, turn_number, 10)
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if target in ranked:
                hit_turn = turn_number
                best_rank = ranked.index(target) + 1
                break
        session = {
            "sample_id": row["case_id"],
            "scenario_type": row.get("scenario_type", "unknown"),
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        }
        sessions.append(session)
        grouped[session["scenario_type"]].append(session)

    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    result = {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(
            0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency,
            6,
        ),
        "scenario_metrics": {
            name: metric_summary(items) for name, items in sorted(grouped.items())
        },
        "sessions": sessions,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
