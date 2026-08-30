from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import (
    MAX_TURNS,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from solution.agent import Agent, _constraint_variants, _terms


DEFAULT_SAMPLES = {"public_0020", "public_0096", "public_0144", "public_0174"}
BASE_WEIGHTS = (0.0, 5.0, 4.0, 6.0, 6.0, 1.0, 2.0)
VARIANTS = {
    "title_minus_10pct": (0.0, 4.5, 4.0, 6.0, 6.0, 1.0, 2.0),
    "title_plus_10pct": (0.0, 5.5, 4.0, 6.0, 6.0, 1.0, 2.0),
    "category_minus_10pct": (0.0, 5.0, 3.6, 6.0, 6.0, 1.0, 2.0),
    "category_plus_10pct": (0.0, 5.0, 4.4, 6.0, 6.0, 1.0, 2.0),
}


def _weighted_bm25(agent: Agent, query: str, weights: tuple[float, ...]) -> list[str]:
    terms = list(dict.fromkeys(_terms(query)))[:100]
    if not terms:
        return []
    expression = " OR ".join(f'"{term}"' for term in terms)
    arguments = ", ".join(str(weight) for weight in weights)
    rows = agent.connection.execute(
        f"SELECT parent_asin FROM products WHERE products MATCH ? "
        f"ORDER BY bm25(products, {arguments}) LIMIT 500",
        (expression,),
    ).fetchall()
    return [str(row[0]) for row in rows]


def _jaccard(left: list[str], right: list[str], k: int = 10) -> float:
    a, b = set(left[:k]), set(right[:k])
    return len(a & b) / len(a | b) if a or b else 1.0


def _rerank(
    agent: Agent,
    baseline: list[str],
    bm25: list[str],
    constraints: list[str],
    negative_constraints: list[str],
) -> list[str]:
    groups = [_constraint_variants(value) for value in constraints]
    negative_groups = [_constraint_variants(value) for value in negative_constraints]
    ranks = {asin: rank for rank, asin in enumerate(bm25, 1)}

    def key(asin: str) -> tuple[int, int, int, int]:
        card = agent.cards.get(asin, set())
        return (
            -sum(bool(group & card) for group in negative_groups),
            sum(bool(group & card) for group in groups),
            sum(max((len(value) for value in group if value in card), default=0) for group in groups),
            -ranks.get(asin, 100_000),
        )

    # Stable sorting preserves production order when a perturbation adds no evidence.
    return sorted(baseline, key=key, reverse=True)


def _turn_row(agent: Agent, message: str, target: str, turn: int) -> dict:
    baseline, diagnostics, _ = agent.update_and_rank("stability", message, turn)
    state = agent.sessions["stability"]
    groups = [_constraint_variants(value) for value in state["constraints"]]
    expanded = list(dict.fromkeys(value for group in groups for value in group))
    query = " ".join([state["category"], *state["constraints"], *expanded]).strip()
    target_rank = baseline.index(target) + 1 if target in baseline else None
    rows = []
    for name, weights in VARIANTS.items():
        perturbed = _rerank(
            agent,
            baseline,
            _weighted_bm25(agent, query, weights),
            state["constraints"],
            state["negative_constraints"],
        )
        rows.append({
            "variant": name,
            "leader_survived": not baseline or perturbed[0] == baseline[0],
            "top10_jaccard": round(_jaccard(baseline, perturbed), 6),
            "target_rank": perturbed.index(target) + 1 if target in perturbed else None,
            "target_rank_delta": None if target_rank is None else perturbed.index(target) + 1 - target_rank,
            "top10": perturbed[:10],
        })
    return {
        "turn": turn,
        "user_message": message,
        "category": state["category"],
        "constraints": state["constraints"],
        "negative_constraints": state["negative_constraints"],
        "candidate_count": int(diagnostics["candidate_count"]),
        "exact_tie_count": int(diagnostics["exact_tie_count"]),
        "baseline_leader": baseline[0] if baseline else None,
        "baseline_target_rank": target_rank,
        "baseline_top10": baseline[:10],
        "variants": rows,
    }


def trace_session(agent: Agent, sample: dict, categories: dict[str, list[str]], products: dict[str, dict]) -> dict:
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)
    agent.reset("stability", sample["user_profile"])
    turns = []
    for turn in range(1, MAX_TURNS + 1):
        turns.append(_turn_row(agent, message, target, turn))
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
            message, boundary_used = customer_reply(effective, "other", disclosed, boundary_used)
    return {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "target": target,
        "turns": turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only BM25 weight stability diagnostic")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="training/bm25_stability_report.json")
    parser.add_argument("--sample", action="append", dest="samples")
    args = parser.parse_args()
    assert _jaccard(["a", "b"], ["b", "c"], 2) == 1 / 3
    selected = set(args.samples or DEFAULT_SAMPLES)
    samples = [sample for sample in load_jsonl(args.dataset) if sample["sample_id"] in selected]
    _, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)
    sessions = [trace_session(agent, sample, categories, products) for sample in samples]
    variant_rows = [variant for session in sessions for turn in session["turns"] for variant in turn["variants"]]
    summary = {
        name: {
            "observations": len(rows),
            "leader_survival_rate": round(sum(row["leader_survived"] for row in rows) / len(rows), 6),
            "mean_top10_jaccard": round(sum(row["top10_jaccard"] for row in rows) / len(rows), 6),
            "minimum_top10_jaccard": min(row["top10_jaccard"] for row in rows),
            "target_rank_change_rate": round(
                sum(row["target_rank_delta"] not in (None, 0) for row in rows) / len(rows), 6
            ),
            "mean_absolute_target_rank_delta": round(
                sum(abs(row["target_rank_delta"] or 0) for row in rows) / len(rows), 6
            ),
        }
        for name in VARIANTS
        if (rows := [row for row in variant_rows if row["variant"] == name])
    }
    report = {
        "method": "Freeze the production candidate pool, then rerank it after changing one FTS5 field weight by +/-10%.",
        "production_ranking_changed": False,
        "weights": {"baseline": BASE_WEIGHTS, **VARIANTS},
        "summary": summary,
        "sessions": sessions,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
