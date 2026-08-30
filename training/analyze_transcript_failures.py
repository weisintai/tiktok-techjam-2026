from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluator.local_evaluator import load_jsonl
from solution.agent import Agent, _constraint_variants, _normalize, _terms


def output_limit(turn: int, complete_match_count: float) -> int:
    if turn >= 10:
        return 10
    if turn <= 6:
        return 1
    return 5 if complete_match_count > 100 else 3


def analyze_case(agent: Agent, row: dict) -> dict:
    session_id = str(row["case_id"])
    target = str(row["target_asin"])
    agent.reset(session_id, row.get("user_profile", {}))
    best_full_rank = None
    best_bm25_rank = None
    best_overlap = 0.0
    hit = False
    cutoff_blocked = False
    duplicate_tie = False
    traces = []
    for turn, message in enumerate(row["turns"], 1):
        ranked, diagnostics, _ = agent.update_and_rank(session_id, message, turn)
        state = agent.sessions[session_id]
        soft_query = " ".join(state.get("soft_queries", []))
        groups = [_constraint_variants(value) for value in state["constraints"]]
        expanded = [variant for group in groups for variant in group]
        query = " ".join([state["category"], *state["constraints"], *expanded, soft_query]).strip()
        bm25 = [asin for asin, _ in agent._bm25_scored(query)]
        full_rank = ranked.index(target) + 1 if target in ranked else None
        bm25_rank = bm25.index(target) + 1 if target in bm25 else None
        if full_rank is not None:
            best_full_rank = min(best_full_rank or full_rank, full_rank)
        if bm25_rank is not None:
            best_bm25_rank = min(best_bm25_rank or bm25_rank, bm25_rank)
        query_terms = set(_terms(query))
        target_terms = set(_terms(agent.documents[agent.asin_to_index[target]]))
        overlap = len(query_terms & target_terms) / max(1, len(query_terms))
        best_overlap = max(best_overlap, overlap)
        limit = output_limit(turn, diagnostics.get("complete_match_count", 0.0))
        available = [asin for asin in ranked if asin not in state["seen"]]
        output = available[:limit]
        if target in output:
            hit = True
        elif full_rank is not None and target not in state["seen"]:
            cutoff_blocked = True
            leader = available[0] if available else None
            if leader:
                duplicate_tie = (
                    agent.cards.get(leader, set()) == agent.cards.get(target, set())
                    or _normalize(agent.documents[agent.asin_to_index[leader]])
                    == _normalize(agent.documents[agent.asin_to_index[target]])
                )
        state["seen"].update(output)
        state["last_recommendations"] = output
        traces.append({
            "turn": turn,
            "query": query,
            "full_rank": full_rank,
            "bm25_rank": bm25_rank,
            "output_limit": limit,
            "query_target_token_overlap": round(overlap, 4),
        })
        if hit:
            break

    if hit:
        failure = "hit"
    elif best_full_rank is not None and duplicate_tie:
        failure = "near_duplicate_tie"
    elif best_full_rank is not None and cutoff_blocked:
        failure = "below_output_cutoff"
    elif best_bm25_rank is None and best_overlap < 0.15:
        failure = "insufficient_lexical_evidence"
    elif best_bm25_rank is None:
        failure = "bm25_candidate_miss"
    else:
        failure = "fusion_or_candidate_miss"
    return {
        "case_id": row["case_id"],
        "writer_id": row.get("writer_id"),
        "split": row.get("split"),
        "scenario_type": row.get("scenario_type"),
        "failure": failure,
        "best_full_rank": best_full_rank,
        "best_bm25_rank": best_bm25_rank,
        "best_query_target_token_overlap": round(best_overlap, 4),
        "traces": traces,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--split", choices=("development", "test"))
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = [row for path in args.cases for row in load_jsonl(path)]
    if args.split:
        rows = [row for row in rows if row.get("split") == args.split]
    agent = Agent(args.catalog)
    diagnostics = [analyze_case(agent, row) for row in rows]
    failures = Counter(item["failure"] for item in diagnostics)
    rank_bands: Counter[str] = Counter()
    by_scenario: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_writer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for item in diagnostics:
        by_scenario[str(item["scenario_type"])][item["failure"]] += 1
        by_writer[str(item["writer_id"])][item["failure"]] += 1
        rank = item["best_full_rank"]
        if rank is None:
            rank_bands["not_in_fused_pool"] += 1
        elif rank <= 3:
            rank_bands["rank_1_3"] += 1
        elif rank <= 10:
            rank_bands["rank_4_10"] += 1
        elif rank <= 100:
            rank_bands["rank_11_100"] += 1
        else:
            rank_bands["rank_over_100"] += 1
    result = {
        "cases": len(diagnostics),
        "failure_counts": dict(sorted(failures.items())),
        "failure_rates": {
            name: round(count / max(1, len(diagnostics)), 4)
            for name, count in sorted(failures.items())
        },
        "best_fused_rank_bands": dict(sorted(rank_bands.items())),
        "by_scenario": {name: dict(sorted(counts.items())) for name, counts in sorted(by_scenario.items())},
        "by_writer": {name: dict(sorted(counts.items())) for name, counts in sorted(by_writer.items())},
        "diagnostics": diagnostics,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
