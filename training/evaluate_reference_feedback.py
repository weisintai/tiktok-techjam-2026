from __future__ import annotations

import argparse
import json
from pathlib import Path

from solution.agent import Agent


SLOT_LABELS = {
    "material": "material",
    "color": "color",
    "size": "fit",
    "style": "style",
    "feature": "feature",
    "use_case": "use",
}


def evaluate(agent: Agent, limit: int) -> dict[str, float | int]:
    cases = exact = leaked_slots = similarity_cases = similarity_exact = 0
    asins = [asin for asin in agent.asins if agent.card_facets.get(asin)]
    for offset, asin in enumerate(asins):
        distractors = [value for value in asins[offset + 1:offset + 3] if value != asin]
        if len(distractors) < 2:
            continue
        position = offset % 3
        shown = distractors[:]
        shown.insert(position, asin)
        state = {"last_recommendations": shown}
        ordinal = (("first", "1st"), ("second", "2nd"), ("third", "3rd"))[position]
        for slot, values in agent.card_facets[asin].items():
            label = SLOT_LABELS.get(slot)
            if not label or not values:
                continue
            messages = (
                f"I prefer the same {label} as the {ordinal[0]} one.",
                f"Match the {label} of the {ordinal[1]} option.",
                f"I like the {ordinal[0]} product's {label}.",
            )
            turn, soft_query = agent._reference_feedback(messages[cases % len(messages)], state)
            cases += 1
            if turn.add == {slot: sorted(values)[:4]} and not soft_query:
                exact += 1
            leaked_slots += len(set(turn.add) - {slot})
        similarity_messages = (
            f"Show me something more like the {ordinal[0]} one.",
            f"Find products similar to the {ordinal[1]} option.",
            f"Move closer to the {ordinal[0]} result.",
        )
        turn, soft_query = agent._reference_feedback(
            similarity_messages[similarity_cases % len(similarity_messages)], state
        )
        similarity_cases += 1
        expected = " ".join(sorted(agent.cards.get(asin, set())))
        if not turn.add and soft_query == expected:
            similarity_exact += 1
        if cases >= limit:
            break
    return {
        "facet_reference_cases": cases,
        "facet_exact_match": round(exact / max(1, cases), 4),
        "leaked_slot_count": leaked_slots,
        "similarity_cases": similarity_cases,
        "similarity_exact_match": round(similarity_exact / max(1, similarity_cases), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate(Agent(args.catalog), args.limit)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
