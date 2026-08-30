from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import re
import resource
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from solution.agent import Agent, OVERRIDE_RE, _constraint_slot, _quarantine_structured_turn
from solution.extraction import (
    LlamaCppExtractor,
    StructuredTurn,
    TransformersLocalExtractor,
    extract_deterministic_turn,
)


SPLITS = {"train", "development", "test"}


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    identifiers: set[str] = set()
    for index, case in enumerate(cases, 1):
        identifier = str(case.get("case_id", f"case_{index:04d}"))
        if identifier in identifiers:
            raise ValueError(f"duplicate case_id: {identifier}")
        identifiers.add(identifier)
        split = case.get("split", "test")
        if split not in SPLITS:
            raise ValueError(f"invalid split for {identifier}: {split}")
        if not isinstance(case.get("message"), str) or not isinstance(case.get("expected"), dict):
            raise ValueError(f"invalid case payload: {identifier}")
        case.setdefault("case_id", identifier)
        case.setdefault("split", split)
    return cases


def rule_extract(message: str, state: dict[str, Any]) -> StructuredTurn:
    return extract_deterministic_turn(message, state)


def legacy_rule_extract(message: str, state: dict[str, Any]) -> StructuredTurn:
    """Reproduce the rule extractor shipped on main before the fallback."""
    category, _ = Agent._extract_initial(message)
    constraints = Agent._extract_constraints(message)
    add: dict[str, list[str]] = {}
    for value in constraints:
        add.setdefault(_constraint_slot(value), []).append(value)
    negative: dict[str, list[str]] = {}
    for value in Agent._extract_negative_constraints(message):
        negative.setdefault(_constraint_slot(value), []).append(value)
    override = bool(OVERRIDE_RE.search(message))
    return StructuredTurn(
        intent="buying" if category or constraints else "unknown",
        override=override,
        category=category,
        add=add,
        negative=negative,
        confidence=1.0 if category or constraints or negative else 0.0,
    )


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9$]+", " ", value.casefold()).strip()


def atoms(turn: StructuredTurn) -> set[str]:
    result = {f"intent={turn.intent}"}
    if turn.override:
        result.add("override")
    if turn.category:
        result.add(f"category={normalized(turn.category)}")
    for field_name in ("add", "remove", "negative"):
        values_by_slot = getattr(turn, field_name)
        for slot, values in values_by_slot.items():
            result.update(f"{field_name}:{slot}={normalized(value)}" for value in values)
    result.update(f"replace={slot}" for slot in turn.replace_slots)
    result.update(f"no_preference={slot}" for slot in turn.no_preference)
    result.update(f"unresolved={slot}" for slot in turn.unresolved)
    if turn.show_options_first:
        result.add("show_options_first")
    return result


def apply_turn(state: dict[str, Any], turn: StructuredTurn) -> dict[str, Any]:
    applied = copy.deepcopy(state)
    applied.setdefault("category", "")
    applied.setdefault("slots", {})
    applied.setdefault("constraints", [
        value for values in applied["slots"].values() for value in values
    ])
    applied.setdefault("negative_constraints", [])
    applied.setdefault("inferred_intent", "unknown")
    Agent._apply_structured_turn(applied, turn)
    return applied


def state_atoms(state: dict[str, Any]) -> set[str]:
    result = {f"intent={state.get('inferred_intent', 'unknown')}"}
    category = normalized(str(state.get("category", "")))
    if category:
        result.add(f"category={category}")
    for slot, values in state.get("slots", {}).items():
        result.update(f"slot:{slot}={normalized(str(value))}" for value in values)
    result.update(
        f"negative:{_constraint_slot(str(value))}={normalized(str(value))}"
        for value in state.get("negative_constraints", [])
    )
    result.update(f"no_preference={slot}" for slot in state.get("no_preference", set()))
    result.update(f"unresolved={slot}" for slot in state.get("unresolved", set()))
    if state.get("show_options_first", False):
        result.add("show_options_first")
    return result


def evaluate_prediction(case: dict[str, Any], predicted: StructuredTurn) -> dict[str, Any]:
    expected = StructuredTurn.from_payload({**case["expected"], "confidence": 1.0})
    gold_delta, predicted_delta = atoms(expected), atoms(predicted)
    gold_state = state_atoms(apply_turn(case.get("state", {}), expected))
    predicted_state = state_atoms(apply_turn(case.get("state", {}), predicted))
    state_missing = gold_state - predicted_state
    state_extra = predicted_state - gold_state
    prior_state = state_atoms(apply_turn(case.get("state", {}), StructuredTurn()))
    preserved_constraints = {
        atom
        for atom in prior_state & gold_state
        if atom.startswith(("slot:", "negative:"))
    }
    preserved_correct = len(preserved_constraints & predicted_state)
    return {
        "expected": expected,
        "gold_delta": gold_delta,
        "predicted_delta": predicted_delta,
        "delta_missing": gold_delta - predicted_delta,
        "delta_extra": predicted_delta - gold_delta,
        "state_missing": state_missing,
        "state_extra": state_extra,
        "preserved_correct": preserved_correct,
        "preserved_total": len(preserved_constraints),
    }


def _metrics(outcomes: list[dict[str, Any]]) -> dict[str, float | int]:
    true_positive = sum(
        len(state_atoms(item["gold_state"]) & state_atoms(item["predicted_state"]))
        for item in outcomes
    )
    false_positive = sum(len(item["state_extra"]) for item in outcomes)
    false_negative = sum(len(item["state_missing"]) for item in outcomes)
    exact = sum(not item["state_missing"] and not item["state_extra"] for item in outcomes)
    false_additions = sum(
        sum(atom.startswith(("category=", "add:", "negative:")) for atom in item["delta_extra"])
        for item in outcomes
    )
    predicted_constraints = sum(
        sum(atom.startswith(("category=", "add:", "negative:")) for atom in item["predicted_delta"])
        for item in outcomes
    )
    preserved_correct = sum(item["preserved_correct"] for item in outcomes)
    preserved_total = sum(item["preserved_total"] for item in outcomes)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "cases": len(outcomes),
        "exact_state_match": round(exact / max(1, len(outcomes)), 4),
        "state_micro_precision": round(precision, 4),
        "state_micro_recall": round(recall, 4),
        "state_micro_f1": round(2 * precision * recall / max(1e-12, precision + recall), 4),
        "false_addition_rate": round(false_additions / max(1, predicted_constraints), 4),
        "sibling_preservation": round(preserved_correct / max(1, preserved_total), 4),
    }


def _outcome(case: dict[str, Any], predicted: StructuredTurn) -> dict[str, Any]:
    result = evaluate_prediction(case, predicted)
    result["gold_state"] = apply_turn(case.get("state", {}), result["expected"])
    result["predicted_state"] = apply_turn(case.get("state", {}), predicted)
    return result


def score(cases: list[dict[str, Any]], extractor: Any, *, show_failures: bool = False) -> dict[str, Any]:
    true_positive = false_positive = false_negative = exact = 0
    outcomes: list[dict[str, Any]] = []
    for case in cases:
        expected = StructuredTurn.from_payload({**case["expected"], "confidence": 1.0})
        predicted = extractor.extract(case["message"], case.get("state", {}))
        gold_atoms, predicted_atoms = atoms(expected), atoms(predicted)
        true_positive += len(gold_atoms & predicted_atoms)
        false_positive += len(predicted_atoms - gold_atoms)
        false_negative += len(gold_atoms - predicted_atoms)
        exact += gold_atoms == predicted_atoms
        outcome = _outcome(case, predicted)
        outcomes.append(outcome)
        if show_failures and (outcome["state_missing"] or outcome["state_extra"]):
            print(json.dumps({
                "case_id": case["case_id"],
                "message": case["message"],
                "missing": sorted(outcome["state_missing"]),
                "extra": sorted(outcome["state_extra"]),
                "confidence": predicted.confidence,
            }, ensure_ascii=False))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    result = {
        "cases": len(cases),
        "exact_match": round(exact / max(1, len(cases)), 4),
        "micro_precision": round(precision, 4),
        "micro_recall": round(recall, 4),
        "micro_f1": round(2 * precision * recall / max(1e-12, precision + recall), 4),
    }
    result.update(_metrics(outcomes))
    result["by_split"] = {
        split: _metrics([outcome for case, outcome in zip(cases, outcomes) if case["split"] == split])
        for split in sorted(SPLITS)
        if any(case["split"] == split for case in cases)
    }
    return result


def _peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(value / (1024 * 1024 if platform.system() == "Darwin" else 1024), 2)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def benchmark(
    cases: list[dict[str, Any]], extractor: Any, *, show_failures: bool = False
) -> dict[str, Any]:
    baseline_rss = _peak_rss_mb()
    started = time.perf_counter()
    extractor._load()
    load_seconds = time.perf_counter() - started
    true_positive = false_positive = false_negative = exact = 0
    raw_true_positive = raw_false_positive = raw_false_negative = raw_exact = 0
    latencies: list[float] = []
    diagnostics = []
    outcomes: list[dict[str, Any]] = []
    raw_outcomes: list[dict[str, Any]] = []
    prompt_tokens = completion_tokens = 0
    for index, case in enumerate(cases, 1):
        expected = StructuredTurn.from_payload({**case["expected"], "confidence": 1.0})
        started = time.perf_counter()
        raw_predicted = extractor.extract(case["message"], case.get("state", {}))
        predicted = _quarantine_structured_turn(
            case["message"], case.get("state", {}), raw_predicted
        )
        latency = time.perf_counter() - started
        latencies.append(latency)
        prompt_tokens += raw_predicted.prompt_tokens
        completion_tokens += raw_predicted.completion_tokens
        gold_atoms, predicted_atoms = atoms(expected), atoms(predicted)
        raw_atoms = atoms(raw_predicted)
        missing = sorted(gold_atoms - predicted_atoms)
        extra = sorted(predicted_atoms - gold_atoms)
        true_positive += len(gold_atoms & predicted_atoms)
        false_positive += len(extra)
        false_negative += len(missing)
        exact += not missing and not extra
        raw_true_positive += len(gold_atoms & raw_atoms)
        raw_false_positive += len(raw_atoms - gold_atoms)
        raw_false_negative += len(gold_atoms - raw_atoms)
        raw_exact += gold_atoms == raw_atoms
        outcome = _outcome(case, predicted)
        outcomes.append(outcome)
        raw_outcomes.append(_outcome(case, raw_predicted))
        item = {
            "case": index,
            "case_id": case["case_id"],
            "split": case["split"],
            "message": case["message"],
            "exact_match": not missing and not extra,
            "missing": missing,
            "extra": extra,
            "latency_seconds": round(latency, 4),
            "prompt_tokens": predicted.prompt_tokens,
            "completion_tokens": predicted.completion_tokens,
            "prompt_only_predicted": asdict(raw_predicted),
            "predicted": asdict(predicted),
            "state_missing": sorted(outcome["state_missing"]),
            "state_extra": sorted(outcome["state_extra"]),
        }
        diagnostics.append(item)
        if show_failures and (missing or extra):
            print(json.dumps(item, ensure_ascii=False))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    metrics = {
        "cases": len(cases),
        "exact_match": round(exact / max(1, len(cases)), 4),
        "micro_precision": round(precision, 4),
        "micro_recall": round(recall, 4),
        "micro_f1": round(2 * precision * recall / max(1e-12, precision + recall), 4),
    }
    metrics.update(_metrics(outcomes))
    raw_precision = raw_true_positive / max(1, raw_true_positive + raw_false_positive)
    raw_recall = raw_true_positive / max(1, raw_true_positive + raw_false_negative)
    prompt_only_metrics = {
        "cases": len(cases),
        "exact_match": round(raw_exact / max(1, len(cases)), 4),
        "micro_precision": round(raw_precision, 4),
        "micro_recall": round(raw_recall, 4),
        "micro_f1": round(
            2 * raw_precision * raw_recall / max(1e-12, raw_precision + raw_recall), 4
        ),
    }
    prompt_only_metrics.update(_metrics(raw_outcomes))
    return {
        "metrics": metrics,
        "prompt_only_metrics": prompt_only_metrics,
        "performance": {
            "model_loading_seconds": round(load_seconds, 4),
            "median_extraction_seconds": round(statistics.median(latencies), 4),
            "p95_extraction_seconds": round(_percentile(latencies, 0.95), 4),
            "peak_process_rss_mb": _peak_rss_mb(),
            "peak_process_rss_increase_mb": round(_peak_rss_mb() - baseline_rss, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "diagnostics": diagnostics,
    }


class RuleExtractor:
    def extract(self, message: str, state: dict[str, Any]) -> StructuredTurn:
        return rule_extract(message, state)


class LegacyRuleExtractor:
    def extract(self, message: str, state: dict[str, Any]) -> StructuredTurn:
        return legacy_rule_extract(message, state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="training/freeform_extraction_cases.jsonl")
    parser.add_argument("--model", help="Local Hugging Face causal model path/name")
    parser.add_argument("--gguf", action="append", help="Local GGUF model path; repeat to compare")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", action="append", choices=sorted(SPLITS))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt-file", help="Use an alternate extraction system prompt")
    parser.add_argument(
        "--extractor",
        choices=("deterministic", "legacy", "both"),
        default="deterministic",
        help="Select the offline rule path; legacy reproduces main's original extractor",
    )
    parser.add_argument("--show-failures", action="store_true")
    parser.add_argument("--output", help="Write the complete benchmark report as JSON")
    args = parser.parse_args()
    cases = load_cases(args.cases)
    if args.split:
        cases = [case for case in cases if case["split"] in args.split]
    if args.limit:
        cases = cases[:args.limit]
    system_prompt = (
        Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else None
    )
    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpus": os.cpu_count(),
        },
    }
    if args.extractor in {"deterministic", "both"}:
        results["rules"] = score(cases, RuleExtractor(), show_failures=args.show_failures)
    if args.extractor in {"legacy", "both"}:
        results["legacy_rules"] = score(
            cases, LegacyRuleExtractor(), show_failures=args.show_failures
        )
    if args.model:
        results[args.model] = benchmark(
            cases,
            TransformersLocalExtractor(
                args.model,
                max_new_tokens=args.max_new_tokens,
                **({"system_prompt": system_prompt} if system_prompt else {}),
            ),
            show_failures=args.show_failures,
        )
    for gguf in args.gguf or []:
        model_result = benchmark(
            cases,
            LlamaCppExtractor(
                gguf,
                max_new_tokens=args.max_new_tokens,
                **({"system_prompt": system_prompt} if system_prompt else {}),
            ),
            show_failures=args.show_failures,
        )
        quantization = re.search(r"(Q\d[^./]*)", Path(gguf).name, re.I)
        model_result["runtime"] = {
            "backend": "llama.cpp",
            "device": "Metal GPU with CPU orchestration" if platform.system() == "Darwin" else "GPU/CPU auto",
            "quantization": quantization.group(1) if quantization else "unknown",
            "model_bytes": Path(gguf).stat().st_size,
        }
        results[gguf] = model_result
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
