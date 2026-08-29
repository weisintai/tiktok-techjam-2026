from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODELS = ("qwen3:1.7b", "gemma3:1b", "llama3.2:1b")
TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "turns": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "items": {"type": "string", "minLength": 2},
        }
    },
    "required": ["turns"],
    "additionalProperties": False,
}


def prompt_for(row: dict[str, Any]) -> str:
    scenario = row["scenario_type"]
    guidance = {
        "buying": "Express a concrete purchase need and reveal constraints naturally over turns.",
        "browsing": "Start open-ended and gradually reveal what would suit the situation.",
        "intent_override": "Start with a plausible different preference, then clearly change it later.",
        "boundary": "Be initially vague or indifferent, then provide useful detail after guidance.",
        "reference_feedback": "Discuss options naturally; include feedback such as first, second, or third option in a later turn.",
    }[scenario]
    return f"""You are an independent shopper writing a backend evaluation transcript.
Write 2 to 4 concise natural customer messages that could lead a shopping assistant to the target product.
{guidance}
Do not mention the ASIN, evaluation, target, labels, parser, or product title verbatim.
Do not copy metadata sentences. Paraphrase relevant needs as a normal shopper.
Messages must be clean English but should vary sentence structure and may include soft preferences,
negation, corrections, uncertainty, or no-preference statements when natural.

Scenario: {scenario}
Product brief: {json.dumps(row['product_brief'], ensure_ascii=False)}

Return JSON with only a turns array."""


def generate(model: str, row: dict[str, Any], endpoint: str) -> list[str]:
    payload = {
        "model": model,
        "prompt": prompt_for(row),
        "stream": False,
        "format": TURN_SCHEMA,
        "keep_alive": "2m",
        "options": {
            "temperature": 0.7,
            "seed": int(row["case_id"].split("_")[-1]),
            "num_predict": 1024,
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        outer = json.load(response)
    parsed = json.loads(outer["response"])
    turns = parsed.get("turns")
    if not isinstance(turns, list) or not 1 <= len(turns) <= 10:
        raise ValueError("model returned an invalid turns array")
    cleaned = [str(turn).strip() for turn in turns]
    if not all(cleaned):
        raise ValueError("model returned an empty turn")
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill blind packets with isolated local models")
    parser.add_argument("--input-dir", default="training/blind_packets")
    parser.add_argument("--output-dir", default="training/model_blind_packets")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/generate")
    args = parser.parse_args()

    input_paths = sorted(Path(args.input_dir).glob("writer_*.jsonl"))
    if len(input_paths) != len(args.models):
        parser.error("provide exactly one model for each writer packet")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total = sum(
        len([line for line in path.read_text(encoding="utf-8").splitlines() if line])
        for path in input_paths
    )
    completed_count = 0
    for path, model in zip(input_paths, args.models):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        output = output_dir / path.name
        existing = {
            row["case_id"]: row
            for row in (
                json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
            )
        } if output.exists() else {}
        for row in rows:
            if row["case_id"] in existing:
                completed_count += 1
                continue
            error: Exception | None = None
            for _ in range(3):
                try:
                    row["turns"] = generate(model, row, args.endpoint)
                    error = None
                    break
                except (ValueError, json.JSONDecodeError) as exc:
                    error = exc
            if error:
                raise error
            row["writer_id"] = f"model:{model}"
            existing[row["case_id"]] = row
            ordered = [existing[key] for key in sorted(existing)]
            output.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered),
                encoding="utf-8",
            )
            completed_count += 1
            print(f"[{completed_count}/{total}] {row['case_id']} <- {model}", flush=True)


if __name__ == "__main__":
    main()
