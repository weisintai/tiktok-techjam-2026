from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from solution.agent import _quarantine_structured_turn
from solution.extraction import StructuredTurn
from training.evaluate_extraction import apply_turn, evaluate_prediction, load_cases, state_atoms


def state_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    payload = getattr(prediction, "delta", {})
    if not isinstance(payload, dict):
        return 0.0
    state = json.loads(example.current_state)
    turn = _quarantine_structured_turn(
        example.shopper_message,
        state,
        StructuredTurn.from_payload(payload),
    )
    outcome = evaluate_prediction({
        "message": example.shopper_message,
        "state": state,
        "expected": example.delta,
    }, turn)
    gold = state_atoms(apply_turn(state, outcome["expected"]))
    predicted = state_atoms(apply_turn(state, turn))
    precision = len(gold & predicted) / max(1, len(predicted))
    recall = len(gold & predicted) / max(1, len(gold))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    unsafe = sum(
        atom.startswith(("category=", "add:", "negative:"))
        for atom in outcome["delta_extra"]
    )
    return max(0.0, f1 - 0.15 * unsafe)


def render_prompt(base_prompt: str, cases: list[dict[str, Any]]) -> str:
    base = base_prompt.split("Examples:", 1)[0].rstrip()
    examples = []
    for case in cases:
        examples.append(
            "CURRENT_STATE: " + json.dumps(case.get("state", {}), ensure_ascii=False) +
            "\nSHOPPER_MESSAGE: " + case["message"] +
            "\nJSON: " + json.dumps(case["expected"], ensure_ascii=False, separators=(",", ":"))
        )
    return base + "\n\nExamples:\n" + "\n\n".join(examples) + "\n\nJSON only. /no_think\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8123/v1")
    parser.add_argument("--model", default="qwen-local")
    parser.add_argument("--cases", default="training/freeform_extraction_cases.jsonl")
    parser.add_argument("--base-prompt", default="training/extraction_prompt_v2.txt")
    parser.add_argument("--output", default="training/extraction_prompt_dspy.txt")
    parser.add_argument("--report", default="training/extraction_prompt_dspy.json")
    parser.add_argument("--train-limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    import dspy

    class ExtractTurn(dspy.Signature):
        """Extract only the latest shopper-message changes as one typed shopping delta.

        Never copy unchanged current state. Capture every explicit fact. Use add for
        new positive facts, negative for exclusions, remove for retractions, and
        replace_slots for changed values. Category changes, removals and replacements
        set override=true. Exploratory or category-uncertain requests are browsing.
        """

        current_state: str = dspy.InputField()
        shopper_message: str = dspy.InputField()
        delta: dict = dspy.OutputField(
            desc="JSON with intent, optional category/add/negative/remove/replace_slots/override, and confidence"
        )

    all_cases = load_cases(args.cases)
    train_cases = [case for case in all_cases if case["split"] == "train"]
    random.Random(args.seed).shuffle(train_cases)
    train_cases = train_cases[:args.train_limit]
    examples = [
        dspy.Example(
            current_state=json.dumps(case.get("state", {}), ensure_ascii=False),
            shopper_message=case["message"] + " /no_think",
            delta=case["expected"],
        ).with_inputs("current_state", "shopper_message")
        for case in train_cases
    ]

    lm = dspy.LM(
        f"openai/{args.model}",
        api_base=args.endpoint,
        api_key="local",
        temperature=0,
        max_tokens=256,
        cache=False,
    )
    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
    optimizer = dspy.BootstrapFewShot(
        metric=state_metric,
        metric_threshold=0.75,
        max_bootstrapped_demos=4,
        max_labeled_demos=0,
        max_rounds=1,
        max_errors=20,
    )
    optimized = optimizer.compile(dspy.Predict(ExtractTurn), trainset=examples)
    demos = optimized.predictors()[0].demos
    selected: list[dict[str, Any]] = []
    for demo in demos:
        message = str(demo.shopper_message).removesuffix(" /no_think")
        match = next((case for case in train_cases if case["message"] == message), None)
        if match and match not in selected:
            selected.append(match)
    if not selected:
        raise RuntimeError("DSPy did not select any demonstrations")

    prompt = render_prompt(Path(args.base_prompt).read_text(encoding="utf-8"), selected)
    Path(args.output).write_text(prompt, encoding="utf-8")
    Path(args.report).write_text(json.dumps({
        "optimizer": "BootstrapFewShot",
        "metric_threshold": 0.75,
        "train_cases_considered": len(train_cases),
        "selected_case_ids": [case["case_id"] for case in selected],
        "output": args.output,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_case_ids": [case["case_id"] for case in selected],
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
