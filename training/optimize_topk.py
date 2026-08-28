from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

from evaluator.local_evaluator import (
    MAX_TURNS,
    behavior_for,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    intent_card,
    materialize_hidden_fields,
)
from solution.agent import Agent
from training.evaluate_split import load_samples


Policy = Callable[[int, dict[str, float]], int]
OFFICIAL_SCENARIO_WEIGHTS = {
    "buying": 0.40,
    "browsing": 0.40,
    "intent_override": 0.15,
    "boundary": 0.05,
}


def trace_sample(
    agent: Agent,
    sample: dict,
    categories: dict[str, list[str]],
    products: dict[str, dict],
    rank_limit: int = 100,
) -> dict:
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    session_id = f"trace_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)
    turns: list[dict] = []
    for turn in range(1, MAX_TURNS + 1):
        ranked, diagnostics, reset_seen = agent.update_and_rank(session_id, message, turn)
        turns.append({
            "turn": turn,
            "eligible": override_applied,
            "reset_seen": reset_seen,
            "ranked": ranked[:rank_limit],
            "diagnostics": diagnostics,
        })
        if turn == MAX_TURNS:
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
                effective, "other", disclosed, boundary_used
            )
    return {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "target": target,
        "turns": turns,
    }


def build_traces(
    catalog_path: Path,
    dataset_path: Path,
    output_path: Path,
    limit: int,
    rank_limit: int = 100,
) -> list[dict]:
    samples = load_samples(dataset_path, limit)
    _, categories, products = catalog_index(catalog_path)
    agent = Agent(catalog_path)
    traces = [
        trace_sample(agent, sample, categories, products, rank_limit)
        for sample in samples
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, separators=(",", ":")) + "\n")
    return traces


def load_traces(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def replay(trace: dict, policy: Policy) -> dict:
    target = trace["target"]
    seen: set[str] = set()
    for turn_data in trace["turns"]:
        if turn_data["reset_seen"]:
            seen.clear()
        candidates = [asin for asin in turn_data["ranked"] if asin not in seen]
        k = max(1, min(10, int(policy(turn_data["turn"], turn_data["diagnostics"]))))
        recommendations = candidates[:k]
        if turn_data["eligible"] and target in recommendations:
            return {
                "hit": True,
                "first_hit_turn": turn_data["turn"],
                "best_rank": recommendations.index(target) + 1,
            }
        seen.update(recommendations)
    return {"hit": False, "first_hit_turn": None, "best_rank": None}


def score(
    traces: list[dict],
    policy: Policy,
    scenario_weights: dict[str, float] | None = None,
) -> dict:
    outcomes = [(trace, replay(trace, policy)) for trace in traces]
    count = len(outcomes)
    scenario_counts = Counter(trace["scenario_type"] for trace, _ in outcomes)
    if scenario_weights:
        weights = [
            scenario_weights[trace["scenario_type"]] / scenario_counts[trace["scenario_type"]]
            for trace, _ in outcomes
        ]
    else:
        weights = [1.0 / count] * count
    hit_rate = sum(weight * outcome["hit"] for weight, (_, outcome) in zip(weights, outcomes))
    mrr = sum(
        weight * (0.0 if outcome["best_rank"] is None else 1.0 / outcome["best_rank"])
        for weight, (_, outcome) in zip(weights, outcomes)
    )
    mttc = sum(
        weight * (outcome["first_hit_turn"] if outcome["first_hit_turn"] is not None else 11)
        for weight, (_, outcome) in zip(weights, outcomes)
    )
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    technical = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency
    return {
        "sample_count": count,
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
        "technical_score": round(technical, 6),
    }


def schedule_policy(schedule: tuple[int, ...]) -> Policy:
    return lambda turn, diagnostics: schedule[turn - 1]


def gated_policy(
    schedule: tuple[int, ...], start_turn: int, tie_threshold: int, wide_k: int
) -> Policy:
    def choose(turn: int, diagnostics: dict[str, float]) -> int:
        base = schedule[turn - 1]
        if (
            turn >= start_turn
            and diagnostics.get("complete_match_count", 0.0) > tie_threshold
        ):
            return max(base, wide_k)
        return base
    return choose


def candidate_schedules() -> Iterable[tuple[str, tuple[int, ...], Policy]]:
    # Piecewise non-decreasing schedules are expressive enough for this dialog
    # contract and avoid overfitting ten independent turn parameters.
    for early, middle, late, final in itertools.product(
        (1, 3), (1, 3, 5, 10), (3, 5, 10), (3, 5, 10)
    ):
        if not (early <= middle <= late <= final):
            continue
        schedule = (early, early, early, middle, middle, middle, late, late, late, final)
        name = f"static_{early}_{middle}_{late}_{final}"
        yield name, schedule, schedule_policy(schedule)
        for start_turn, threshold, wide_k in itertools.product(
            (3, 4, 5, 7), (1, 3, 10, 30, 100), (5, 10)
        ):
            if wide_k <= middle:
                continue
            gated_name = f"gate_{early}_{middle}_{late}_{final}_t{start_turn}_n{threshold}_k{wide_k}"
            yield gated_name, schedule, gated_policy(schedule, start_turn, threshold, wide_k)


def optimize(
    traces: list[dict],
    top_n: int = 20,
    scenario_weights: dict[str, float] | None = None,
) -> list[dict]:
    rows = []
    for name, schedule, policy in candidate_schedules():
        rows.append({
            "name": name,
            "schedule": list(schedule),
            **score(traces, policy, scenario_weights),
        })
    return sorted(
        rows,
        key=lambda row: (
            row["technical_score"], row["mrr"], row["hit_rate_at_10"], -sum(row["schedule"])
        ),
        reverse=True,
    )[:top_n]


def policy_from_name(name: str, schedule: tuple[int, ...]) -> Policy:
    if name.startswith("static_"):
        return schedule_policy(schedule)
    parts = name.split("_")
    start_turn = int(parts[-3][1:])
    threshold = int(parts[-2][1:])
    wide_k = int(parts[-1][1:])
    return gated_policy(schedule, start_turn, threshold, wide_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize Top-K policy by counterfactual replay")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="training/generated/validation.jsonl.gz")
    parser.add_argument("--traces", default="training/validation_traces.jsonl.gz")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--rank-limit", type=int, default=100)
    parser.add_argument("--output", default="training/topk_optimization.json")
    parser.add_argument("--rebuild-traces", action="store_true")
    parser.add_argument(
        "--selected-report",
        help="Evaluate the frozen leader from another report instead of optimizing this split",
    )
    args = parser.parse_args()

    trace_path = Path(args.traces)
    if args.rebuild_traces or not trace_path.exists():
        traces = build_traces(
            Path(args.catalog), Path(args.dataset), trace_path, args.limit, args.rank_limit
        )
    else:
        traces = load_traces(trace_path)[:args.limit]
    baseline_schedule = (1, 1, 1, 3, 3, 3, 3, 3, 3, 10)
    baseline = score(
        traces, schedule_policy(baseline_schedule), OFFICIAL_SCENARIO_WEIGHTS
    )
    if args.selected_report:
        selected = json.loads(Path(args.selected_report).read_text(encoding="utf-8"))["leaders"][0]
        policy = policy_from_name(selected["name"], tuple(selected["schedule"]))
        report = {
            "trace_count": len(traces),
            "baseline": baseline,
            "selected_policy": selected,
            "selected_score": score(traces, policy, OFFICIAL_SCENARIO_WEIGHTS),
            "scenario_weights": OFFICIAL_SCENARIO_WEIGHTS,
        }
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    leaders = optimize(traces, scenario_weights=OFFICIAL_SCENARIO_WEIGHTS)
    report = {
        "trace_count": len(traces),
        "scenario_weights": OFFICIAL_SCENARIO_WEIGHTS,
        "baseline": baseline,
        "leaders": leaders,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
