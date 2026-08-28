from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluator.local_evaluator import (
    Agent,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)


def action_utility(rank: int | None, turn: int) -> float:
    return 0.5 * (rank is not None) + 0.3 * (0.0 if rank is None else 1.0 / rank) + 0.02 * (11 - turn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay sessions to analyze recommendation timing")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="experiments/stopping_replay.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    _, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)
    rows: list[dict] = []
    oracle_turns: Counter[tuple[str, int]] = Counter()

    for sample in samples:
        session_id = sample["sample_id"]
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)
        session_rows: list[dict] = []

        for turn in range(1, 11):
            response = agent.respond(session_id, message, turn, 10)
            state = agent._sessions[session_id]
            ranked = state.get("_last_ranked", [])
            rank = ranked.index(target) + 1 if target in ranked and override_applied else None
            session_rows.append({
                "sample_id": sample["sample_id"],
                "scenario": sample["scenario_type"],
                "turn": turn,
                "rank": rank,
                "utility": action_utility(rank, turn),
                **state.get("rank_confidence", {}),
            })
            if turn == 10:
                break
            override = behavior.get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                message, boundary_used = customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )

        best = max(session_rows, key=lambda row: (row["utility"], -row["turn"]))
        oracle_turns[(sample["scenario_type"], best["turn"])] += 1
        best_future = float("-inf")
        for row in reversed(session_rows):
            row["recommend_now"] = row["utility"] >= best_future
            best_future = max(best_future, row["utility"])
        rows.extend(session_rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
    print("rows", len(rows))
    print("oracle turns", dict(sorted(oracle_turns.items())))


if __name__ == "__main__":
    main()
