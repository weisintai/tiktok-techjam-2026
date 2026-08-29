# Training and validation

`generate_sessions.py` streams 200,000 deterministic sessions from the frozen
50,000-product catalog. All four scenarios for an ASIN stay in one split. The
200 public target ASINs are written to a separate quarantine file so public
evaluation cannot leak into policy training.

`train_action_policy.py` creates rank-confidence features and fits a small
histogram gradient-boosted classifier. It is intentionally not loaded by the
production agent: its held-out performance is strong at distinguishing rank 1
from a miss, but weak in the rank 2–10 region where Top-K decisions matter.

`evaluate_split.py` runs the official conversation harness over a bounded,
ASIN-separated synthetic slice. This is a useful regression test, not a fully
independent estimate: both its intent cards and customer simulator come from
the released evaluator.

`optimize_topk.py` stores catalog-ranking traces and replays many Top-K policies
without rerunning retrieval. It optimizes TechnicalScore on validation, freezes
one leader, and can evaluate that exact policy once against the test split via
`--selected-report`. The chosen ambiguity-gated policy improved validation from
`0.937625` to `0.943475` and test from `0.936450` to `0.938188`.
All optimizer metrics now weight scenarios using the private evaluation mix:
40% Buying, 40% Browsing, 15% Intent Override and 5% Boundary.

Generated JSONL files and model binaries are ignored because they are fully
reproducible and too large for the submission snapshot.

`trace_outliers.py` reproduces the scorer-visible conversation and the agent's
internal rank evidence for selected sessions without modifying either one:

```bash
.venv/bin/python -m training.trace_outliers
```

Two read-only robustness harnesses probe generalization without changing the
official evaluator or production ranker:

```bash
.venv/bin/python -m training.evaluate_mutations
.venv/bin/python -m training.bm25_stability
```

The clause-order mutator accepted 189 state-equivalent public mutations and
observed no recommendation or score change. BM25 title/category weight
perturbations preserved every leader across 40 outlier turns, so rank stability
remains diagnostic evidence rather than a Top-K feature.

## Free-form extraction benchmark

The seed corpus contains 200 turns covering natural category requests,
browsing language, negation, accumulation, slot replacement, constraint
removal, category switches and boundary responses. Paraphrase families stay
within frozen 60/20/20 train, development and test splits.

The evaluator reports raw-delta F1 and the state produced after applying each
delta, including false additions and preservation of unrelated sibling slots.
The current `writer_id=codex_seed` cases establish the workflow; supplement or
replace the frozen test slice with independently written team examples before
using it as a generalization claim.

Compare the rule parser with a local structured extractor using:

```bash
.venv/bin/python -m training.evaluate_extraction
.venv/bin/python -m training.evaluate_extraction --split test
.venv/bin/python -m training.evaluate_extraction --model /path/to/Qwen3-0.6B
.venv/bin/python -m training.evaluate_extraction --gguf /path/to/Qwen3-0.6B-Q8_0.gguf
```

Recorded baselines:

- `freeform_rule_baseline.json`: all 200 seed cases, rule-only raw-delta F1
  `0.0583` and applied-state F1 `0.5689`.
- `freeform_extraction_test_baseline.json`: historical result from before the
  duplicate hardcoded grounding parser was removed; do not compare it with the
  current extractor as a promotion result.
- `freeform_extraction_development_v2.json`: the best post-consolidation
  development candidate, with applied-state F1 `0.6822` and false-addition rate
  `0.5283`; it was rejected for runtime promotion.

### Independent free-form corpus

`independent_freeform_cases.jsonl` adds 60 messages written after the current
extractor was implemented. It covers indirect preferences, no-preference
statements, replacements, removals, exclusions, category switches, unresolved
constraints, browse-first requests, conflicts, references, and deliberately
unknown vocabulary. The 20 `development` cases may be inspected while changing
the extractor. The 40 `test` cases are frozen and must be evaluated without
`--show-failures`; do not inspect or tune against individual test failures.

The frozen file SHA256 is recorded in `independent_freeform_cases.sha256`.
This is a stronger temporal holdout than the seed corpus, but it still has one
author and therefore is not evidence of multi-user language generalization.

Main-versus-hybrid extraction A/B using the same frozen cases and scorer:

| Extractor | Split | Cases | Exact state | State precision | State recall | State F1 | False additions |
|---|---|---:|---:|---:|---:|---:|---:|
| Main legacy rules | Development | 20 | 0.0500 | 0.5926 | 0.4211 | 0.4923 | 0.9000 |
| Hybrid fallback | Development | 20 | 0.9000 | 0.9740 | 0.9868 | 0.9804 | 0.1081 |
| Main legacy rules | Frozen test | 40 | 0.1000 | 0.6091 | 0.4589 | 0.5234 | 0.9500 |
| Hybrid fallback | Frozen test | 40 | 0.3750 | 0.8403 | 0.8288 | 0.8345 | 0.2466 |

The frozen test was rerun without failure diagnostics only after the development
candidate was selected. Its state F1 remains below the seed test (`0.8782`) and
is retained as the honest robustness result. The `+0.3111` frozen-test
state-F1 improvement over legacy rules measures intent extraction only;
it is not an official end-to-end TechnicalScore delta. `--extractor legacy`
reproduces the original main rule extractor for future A/B runs.

```bash
.venv/bin/python -m training.evaluate_extraction \
  --cases training/independent_freeform_cases.jsonl --split development
.venv/bin/python -m training.evaluate_extraction \
  --cases training/independent_freeform_cases.jsonl --split test --extractor both \
  --output artifacts/evaluations/independent_freeform_test_baseline.json
shasum -a 256 -c training/independent_freeform_cases.sha256
```

DSPy prompt generation is an offline experiment. OpenAI may propose prompt
candidates, but the shipped Qwen model must execute and score them locally.
Neither DSPy, OpenAI credentials, nor Qwen weights are required for the official
deterministic evaluator path. Further optimization is deferred unless arbitrary
free-form private input is confirmed.

## Blind human transcript evaluation

Generate disjoint non-public target assignments for at least three writers:

```bash
.venv/bin/python -m training.prepare_blind_sessions \
  --writers 3 --sessions-per-writer 20
```

Writers fill each record's `turns` with 1-10 natural shopper messages without
viewing agent code, parser vocabulary, or another writer's messages. Freeze and
checksum completed files before developers inspect the development half. Never
inspect test transcripts while selecting a candidate.

Score the completed files with the competition metric formula:

```bash
.venv/bin/python -m training.evaluate_blind_transcripts \
  --cases training/blind_packets/writer_*.jsonl --split development
.venv/bin/python -m training.evaluate_blind_transcripts \
  --cases training/blind_packets/writer_*.jsonl --split test \
  --reference-feedback --adaptive-questions
```

Fixed transcripts do not react to the agent's exact question wording, so this
is a robustness comparison rather than a replacement for the official dynamic
simulator.

For a reproducible model-authored stress corpus, first generate assignments and
then isolate one local Ollama model per packet:

```bash
.venv/bin/python -m training.prepare_blind_sessions --writers 3 --sessions-per-writer 20
.venv/bin/python -m training.generate_model_blind_sessions
shasum -a 256 -c training/model_blind_packets.sha256
```

The checked-in model corpus uses Qwen3 1.7B, Gemma3 1B and Llama 3.2 1B. It is
synthetic evidence and must not be described as human-blind evaluation.
