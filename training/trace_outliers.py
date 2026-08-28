from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import (
    MAX_TURNS,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from solution.agent import Agent, _constraint_variants, _normalize, _terms


DEFAULT_SAMPLES = {"public_0020", "public_0096", "public_0144", "public_0174"}


def _rank_evidence(
    agent: Agent,
    asin: str,
    groups: list[set[str]],
    negative_groups: list[set[str]],
    bm25_ranks: dict[str, int],
    category: str,
    title: str,
) -> dict[str, Any]:
    card = agent.cards.get(asin, set())
    category_terms = set(_terms(category))
    return {
        "negative_matches": sum(bool(group & card) for group in negative_groups),
        "exact_card_matches": sum(bool(group & card) for group in groups),
        "exact_card_characters": sum(
            max((len(value) for value in group if value in card), default=0)
            for group in groups
        ),
        "bm25_rank": bm25_ranks.get(asin),
        "title_category_term_matches": len(category_terms & set(_terms(title))),
    }


def _outrank_reason(competitor: dict[str, Any], target: dict[str, Any]) -> str:
    if competitor["negative_matches"] < target["negative_matches"]:
        return "fewer negative-constraint matches"
    if competitor["exact_card_matches"] > target["exact_card_matches"]:
        return "more constraint groups match its exact catalog card"
    if competitor["exact_card_characters"] > target["exact_card_characters"]:
        return "more exact-card evidence after matching the same number of constraints"
    competitor_bm25 = competitor["bm25_rank"] or 100_000
    target_bm25 = target["bm25_rank"] or 100_000
    if competitor_bm25 < target_bm25:
        return "higher BM25 rank after the exact-card tie"
    return "all production tie-break signals are equal"


def _output_limit(turn: int, complete_match_count: float) -> int:
    if turn >= 10:
        return 10
    if turn <= 6:
        return 1
    return 5 if complete_match_count > 100 else 3


def trace_sample(agent: Agent, sample: dict, categories: dict[str, list[str]], products: dict[str, dict]) -> dict:
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": card, "behavior": behavior}
    session_id = f"trace_{sample['sample_id']}"
    agent.reset(session_id, sample["user_profile"])
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)
    turns = []

    for turn in range(1, MAX_TURNS + 1):
        ranked, diagnostics, override = agent.update_and_rank(session_id, message, turn)
        state = agent.sessions[session_id]
        groups = [_constraint_variants(value) for value in state["constraints"]]
        negative_groups = [_constraint_variants(value) for value in state["negative_constraints"]]
        expanded = list(dict.fromkeys(value for group in groups for value in group))
        query = " ".join([state["category"], *state["constraints"], *expanded]).strip()
        bm25 = agent._bm25(query)
        bm25_ranks = {asin: rank for rank, asin in enumerate(bm25, 1)}
        target_rank = ranked.index(target) + 1 if target in ranked else None
        target_evidence = _rank_evidence(
            agent, target, groups, negative_groups, bm25_ranks,
            state["category"], str(products[target].get("title", "")),
        )
        limit = min(10, _output_limit(turn, diagnostics["complete_match_count"]))
        target_removed_as_seen = target in state["seen"]
        recommendations = [asin for asin in ranked if asin not in state["seen"]][:limit]
        state["seen"].update(recommendations)
        ask_attribute = agent._select_question(ranked, state)
        state["asked_attributes"].add(ask_attribute)

        competitors = []
        for asin in (candidate for candidate in ranked if candidate != target):
            evidence = _rank_evidence(
                agent, asin, groups, negative_groups, bm25_ranks,
                state["category"], str(products[asin].get("title", "")),
            )
            competitors.append({
                "parent_asin": asin,
                "title": products[asin].get("title"),
                "internal_rank": ranked.index(asin) + 1,
                "already_shown": asin in state["seen"] and asin not in recommendations,
                "evidence": evidence,
                "why_it_outranked_target": _outrank_reason(evidence, target_evidence),
            })
            if len(competitors) == 5:
                break

        output_rank = recommendations.index(target) + 1 if target in recommendations else None
        turns.append({
            "turn": turn,
            "user_message": message,
            "extracted_category": state["category"],
            "extracted_slots": state["slots"],
            "negative_constraints": state["negative_constraints"],
            "override_applied_this_turn": override,
            "target_bm25_rank": target_evidence["bm25_rank"],
            "target_exact_card_match_count": target_evidence["exact_card_matches"],
            "target_final_internal_rank": target_rank,
            "target_removed_as_already_shown": target_removed_as_seen,
            "target_output_rank": output_rank,
            "candidate_count": int(diagnostics["candidate_count"]),
            "exact_tie_count": int(diagnostics["exact_tie_count"]),
            "complete_match_count": int(diagnostics["complete_match_count"]),
            "recommendations": recommendations,
            "ask_attribute": ask_attribute,
            "target_evidence": target_evidence,
            "top_competitors": competitors,
        })

        if override_applied and output_rank is not None:
            break
        if turn == MAX_TURNS:
            break
        override_config = behavior.get("override") or {}
        if not override_applied and turn + 1 == int(override_config.get("turn", 3)):
            override_applied = True
            new_value = str(override_config.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            message = str(override_config.get("message", "Actually, please ignore my earlier preference."))
        else:
            message, boundary_used = customer_reply(effective_sample, ask_attribute, disclosed, boundary_used)

    return {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "target": target,
        "target_title": products[target].get("title"),
        "turns": turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace scorer-relevant ranks for selected public sessions")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="training/outlier_trace_report.json")
    parser.add_argument("--sample", action="append", dest="samples")
    args = parser.parse_args()
    selected = set(args.samples or DEFAULT_SAMPLES)
    samples = [sample for sample in load_jsonl(args.dataset) if sample["sample_id"] in selected]
    identifiers, categories, products = catalog_index(args.catalog)
    del identifiers
    agent = Agent(args.catalog)
    report = {
        "samples": [trace_sample(agent, sample, categories, products) for sample in samples],
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        sample["sample_id"]: {
            "turns": len(sample["turns"]),
            "final_target_rank": sample["turns"][-1]["target_final_internal_rank"],
            "final_output_rank": sample["turns"][-1]["target_output_rank"],
        }
        for sample in report["samples"]
    }, indent=2))


if __name__ == "__main__":
    main()
