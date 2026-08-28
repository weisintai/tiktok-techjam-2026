from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from typing import Any, Protocol


SLOT_NAMES = ("material", "color", "size", "style", "budget", "feature", "use_case")
INTENTS = {"buying", "browsing", "unknown"}


def _clean(value: object, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", str(value)).strip(" -;,.\t\n")[:limit]


def _slot_map(value: object) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, list[str]] = {}
    for slot in SLOT_NAMES:
        raw_values = value.get(slot, [])
        if isinstance(raw_values, str):
            raw_values = [raw_values]
        if not isinstance(raw_values, list):
            continue
        cleaned = list(dict.fromkeys(_clean(item) for item in raw_values if _clean(item)))
        if cleaned:
            result[slot] = cleaned[:8]
    return result


@dataclass(frozen=True)
class StructuredTurn:
    intent: str = "unknown"
    override: bool = False
    category: str = ""
    add: dict[str, list[str]] = field(default_factory=dict)
    remove: dict[str, list[str]] = field(default_factory=dict)
    replace_slots: tuple[str, ...] = ()
    negative: dict[str, list[str]] = field(default_factory=dict)
    confidence: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> "StructuredTurn":
        if not isinstance(payload, dict):
            return cls(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        intent = str(payload.get("intent", "unknown")).casefold()
        if intent not in INTENTS:
            intent = "unknown"
        try:
            confidence = min(1.0, max(0.0, float(payload.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        replace_slots = tuple(
            dict.fromkeys(
                str(slot).casefold() for slot in payload.get("replace_slots", [])
                if str(slot).casefold() in SLOT_NAMES
            )
        ) if isinstance(payload.get("replace_slots", []), list) else ()
        return cls(
            intent=intent,
            override=bool(payload.get("override", False)),
            category=_clean(payload.get("category", ""), 120),
            add=_slot_map(payload.get("add")),
            remove=_slot_map(payload.get("remove")),
            replace_slots=replace_slots,
            negative=_slot_map(payload.get("negative")),
            confidence=confidence,
            prompt_tokens=max(0, int(prompt_tokens)),
            completion_tokens=max(0, int(completion_tokens)),
        )


class StructuredExtractor(Protocol):
    def extract(self, message: str, state: dict[str, Any]) -> StructuredTurn:
        ...


class TimeoutExtractor:
    """Discard late extraction results without letting them mutate agent state."""

    def __init__(self, extractor: StructuredExtractor, timeout: float) -> None:
        self.extractor = extractor
        self.timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="extractor")

    def extract(self, message: str, state: dict[str, Any]) -> StructuredTurn:
        future = self._executor.submit(self.extractor.extract, message, state)
        try:
            return future.result(timeout=self.timeout)
        except TimeoutError:
            future.cancel()
            return StructuredTurn()


EXTRACTION_SYSTEM_PROMPT = """Return one JSON shopping-turn delta. Use only facts in
SHOPPER_MESSAGE; CURRENT_STATE is read-only context, so never copy unchanged values.
Allowed slots: material, color, size, style, budget, feature, use_case.

Use add for new positive constraints, negative for explicit exclusions, remove for a
retracted old constraint, and replace_slots for a changed slot. A change, retraction,
or category switch sets override=true. "Keep everything else" outputs nothing else.
"No preference" is not negative. Output category only when this message states a new
one. Browsing means exploratory or category-uncertain; concrete categories and hard
requirements mean buying. Never invent a constraint. Use confidence 0.95 when clear.

Canonicalize colors as "color: blue", budgets as "budget under $80", jogging as
"running", rain protection as "waterproof", office as "work", and cozy as "warm".

Examples:
MESSAGE: No leather.
JSON: {"intent":"buying","negative":{"material":["leather"]},"confidence":0.95}
STATE material=leather; MESSAGE: Forget leather.
JSON: {"intent":"buying","override":true,"remove":{"material":["leather"]},"confidence":0.95}
STATE color=black, material=cotton; MESSAGE: Blue instead of black. Keep everything else.
JSON: {"intent":"buying","override":true,"add":{"color":["color: blue"]},"replace_slots":["color"],"confidence":0.95}
MESSAGE: I have no preference; show me some everyday ideas.
JSON: {"intent":"browsing","add":{"use_case":["everyday"]},"confidence":0.95}

JSON only. /no_think"""

SLOT_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        slot: {"type": "array", "items": {"type": "string"}, "maxItems": 8}
        for slot in SLOT_NAMES
    },
    "additionalProperties": False,
}
EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"enum": ["buying", "browsing", "unknown"]},
        "override": {"type": "boolean"},
        "category": {"type": "string"},
        "add": SLOT_OBJECT_SCHEMA,
        "remove": SLOT_OBJECT_SCHEMA,
        "replace_slots": {
            "type": "array",
            "items": {"enum": list(SLOT_NAMES)},
            "uniqueItems": True,
        },
        "negative": SLOT_OBJECT_SCHEMA,
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["intent", "confidence"],
    "additionalProperties": False,
}


def _first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first balanced JSON object from possibly fenced model output."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start:index + 1])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, dict) else None
    return None


class TransformersLocalExtractor:
    """Lazy local Hugging Face causal-LM extractor.

    Model files must already be present locally for offline judging. The model is
    loaded only when the rule parser encounters a message it cannot confidently
    structure, so the official deterministic path pays no model latency.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_new_tokens: int = 256,
        system_prompt: str = EXTRACTION_SYSTEM_PROMPT,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            dtype="auto",
        )
        self._model.eval()
        if torch.cuda.is_available():
            self._model.to("cuda")

    def extract(self, message: str, state: dict[str, Any]) -> StructuredTurn:
        self._load()
        import torch

        compact_state = {
            "category": state.get("category", ""),
            "slots": state.get("slots", {}),
            "negative_constraints": state.get("negative_constraints", []),
            "inferred_intent": state.get("inferred_intent", "unknown"),
        }
        user_prompt = (
            "CURRENT_STATE:\n" + json.dumps(compact_state, ensure_ascii=False) +
            "\nSHOPPER_MESSAGE:\n" + message
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tokenizer = self._tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            prompt = self.system_prompt + "\n\n" + user_prompt + "\nJSON:"
        device = next(self._model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        payload = _first_json_object(text)
        return StructuredTurn.from_payload(
            payload,
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            completion_tokens=int(generated.shape[0]),
        )


class LlamaCppExtractor:
    """Quantized GGUF extractor with grammar-constrained JSON output."""

    def __init__(
        self,
        model_path: str,
        *,
        max_new_tokens: int = 256,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        n_gpu_layers: int = -1,
        system_prompt: str = EXTRACTION_SYSTEM_PROMPT,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.n_ctx = n_ctx
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cpus = os.cpu_count() or 4
        self.n_threads = n_threads or max(1, available_cpus - 1)
        self.n_gpu_layers = n_gpu_layers
        self.system_prompt = system_prompt
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from llama_cpp import Llama

        self._model = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )

    def extract(self, message: str, state: dict[str, Any]) -> StructuredTurn:
        self._load()
        compact_state = {
            "category": state.get("category", ""),
            "slots": state.get("slots", {}),
            "negative_constraints": state.get("negative_constraints", []),
            "inferred_intent": state.get("inferred_intent", "unknown"),
        }
        user_prompt = (
            "CURRENT_STATE:\n" + json.dumps(compact_state, ensure_ascii=False) +
            "\nSHOPPER_MESSAGE:\n" + message + "\n/no_think"
        )
        response = self._model.create_chat_completion(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object", "schema": EXTRACTION_JSON_SCHEMA},
            temperature=0.0,
            seed=0,
            max_tokens=self.max_new_tokens,
        )
        choice = response.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage = response.get("usage", {})
        return StructuredTurn.from_payload(
            _first_json_object(str(content)),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )
