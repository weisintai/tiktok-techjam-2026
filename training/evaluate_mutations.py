from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, metric_summary
from solution.agent import Agent
from training.evaluate_split import load_samples


FAMILY = "independent_constraint_order"


def reorder_constraints(message: str) -> str:
    marker = "For that, what matters is: "
    if marker not in message:
        return message
    head, payload = message.split(marker, 1)
    suffix = "." if payload.endswith(".") else ""
    clauses = [clause.strip() for clause in payload.removesuffix(suffix).split(";")]
    if len(clauses) < 2 or any(not clause for clause in clauses):
        return message
    return head + marker + "; ".join(reversed(clauses)) + suffix


def _parsed_state(state: dict) -> dict:
    return {
        "category": str(state.get("category", "")).casefold(),
        "intent": state.get("inferred_intent", "unknown"),
        "slots": {
            slot: sorted(str(value).casefold() for value in values)
            for slot, values in sorted(state.get("slots", {}).items())
            if values
        },
        "negative_constraints": sorted(
            str(value).casefold() for value in state.get("negative_constraints", [])
        ),
    }


def _response_signature(response: dict) -> tuple:
    recommendations = response.get("recommendations") or []
    asins = tuple(
        str(item.get("parent_asin", "")) if isinstance(item, dict) else str(item)
        for item in recommendations
    )
    return response.get("ask_attribute"), asins


class MutationAgent:
    def __init__(self, catalog: str | Path, sample_ids: list[str]) -> None:
        self.baseline = Agent(catalog)
        self.mutated = Agent(catalog)
        self.sample_ids = iter(sample_ids)
        self.traces: dict[str, dict] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        sample_id = next(self.sample_ids)
        self.traces[session_id] = {
            "sample_id": sample_id,
            "mutated_turns": [],
            "first_state_divergence_turn": None,
            "first_output_divergence_turn": None,
        }
        self.baseline.reset(session_id, user_profile)
        self.mutated.reset(session_id, user_profile)

    def respond(self, session_id: str, message: str, turn: int, top_k: int) -> dict:
        mutated_message = reorder_constraints(message)
        baseline_response = self.baseline.respond(session_id, message, turn, top_k)
        mutated_response = self.mutated.respond(session_id, mutated_message, turn, top_k)
        trace = self.traces[session_id]
        if mutated_message != message:
            trace["mutated_turns"].append({
                "turn": turn,
                "original": message,
                "mutated": mutated_message,
            })
            if (
                trace["first_state_divergence_turn"] is None
                and _parsed_state(self.baseline.sessions[session_id])
                != _parsed_state(self.mutated.sessions[session_id])
            ):
                trace["first_state_divergence_turn"] = turn
        if (
            trace["first_output_divergence_turn"] is None
            and _response_signature(baseline_response) != _response_signature(mutated_response)
        ):
            trace["first_output_divergence_turn"] = turn
        return mutated_response


def _technical_score(summary: dict) -> float:
    if not summary["sample_count"]:
        return 0.0
    efficiency = max(0.0, min(1.0, (11.0 - float(summary["mttc"])) / 10.0))
    return round(0.5 * summary["hit_rate_at_10"] + 0.3 * summary["mrr"] + 0.2 * efficiency, 6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate state-equivalent simulator mutations")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="training/adversarial_mutation_results.json")
    args = parser.parse_args()

    samples = load_samples(Path(args.dataset), args.limit)
    identifiers, categories, products = catalog_index(args.catalog)
    baseline = evaluate(Agent(args.catalog), samples, identifiers, categories, products)
    mutation_agent = MutationAgent(args.catalog, [sample["sample_id"] for sample in samples])
    mutated = evaluate(mutation_agent, samples, identifiers, categories, products)

    baseline_sessions = {session["sample_id"]: session for session in baseline["sessions"]}
    mutated_sessions = {session["sample_id"]: session for session in mutated["sessions"]}
    traces = {trace["sample_id"]: trace for trace in mutation_agent.traces.values()}
    cases = []
    for sample in samples:
        sample_id = sample["sample_id"]
        trace = traces[sample_id]
        if not trace["mutated_turns"]:
            continue
        baseline_session = baseline_sessions[sample_id]
        mutated_session = mutated_sessions[sample_id]
        divergent_turns = [
            turn for turn in (
                trace["first_state_divergence_turn"], trace["first_output_divergence_turn"]
            ) if turn is not None
        ]
        cases.append({
            "sample_id": sample_id,
            "scenario_type": sample["scenario_type"],
            "mutation_family": FAMILY,
            "state_equivalent": trace["first_state_divergence_turn"] is None,
            "first_divergent_turn": min(divergent_turns, default=None),
            **trace,
            "baseline_outcome": {
                key: baseline_session[key] for key in ("hit", "first_hit_turn", "best_rank")
            },
            "mutated_outcome": {
                key: mutated_session[key] for key in ("hit", "first_hit_turn", "best_rank")
            },
            "reciprocal_rank_delta": round(
                mutated_session["reciprocal_rank"] - baseline_session["reciprocal_rank"], 6
            ),
        })

    eligible_ids = {case["sample_id"] for case in cases if case["state_equivalent"]}
    eligible_baseline = [session for session in baseline["sessions"] if session["sample_id"] in eligible_ids]
    eligible_mutated = [session for session in mutated["sessions"] if session["sample_id"] in eligible_ids]
    baseline_summary = metric_summary(eligible_baseline)
    mutated_summary = metric_summary(eligible_mutated)
    report = {
        "mutation_family": FAMILY,
        "evaluated_sessions": len(samples),
        "mutated_sessions": len(cases),
        "state_equivalent_sessions": len(eligible_ids),
        "rejected_state_divergences": len(cases) - len(eligible_ids),
        "eligible_baseline": {**baseline_summary, "technical_score": _technical_score(baseline_summary)},
        "eligible_mutated": {**mutated_summary, "technical_score": _technical_score(mutated_summary)},
        "technical_score_delta": round(
            _technical_score(mutated_summary) - _technical_score(baseline_summary), 6
        ),
        "cases": cases,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
