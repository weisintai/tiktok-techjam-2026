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

DSPy prompt generation is an offline experiment. OpenAI may propose prompt
candidates, but the shipped Qwen model must execute and score them locally.
Neither DSPy, OpenAI credentials, nor Qwen weights are required for the official
deterministic evaluator path. Further optimization is deferred unless arbitrary
free-form private input is confirmed.
