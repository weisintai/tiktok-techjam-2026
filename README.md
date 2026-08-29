# TechJam Shopping Copilot

An offline conversational shopping agent that turns multi-turn dialogue into
typed constraints, preserves intent changes, and retrieves the exact purchased
Amazon product in an average of `2.97` turns on the released evaluation set.

The scorer-proven path is deterministic: it uses structured session state,
catalog-derived exact cards, and weighted SQLite FTS5 BM25. The official
evaluator drives a frozen, structured simulator protocol, so this path requires
no network, API key, model weight, or token spend. A small local model remains
an optional demonstration of unfamiliar free-form language rather than a
dependency of the submitted agent.

## Results

| Evaluation | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| Official public set, 200 sessions | `0.995` | `0.985167` | `2.97` | **`0.95365`** |
| Deterministic paraphrase stress set | `0.995` | `0.978667` | `2.975` | **`0.95160`** |
| Unseen-ASIN synthetic validation, 400 sessions | `0.990` | `0.977583` | `3.24` | `0.943475` |
| Untouched synthetic test, 400 sessions | `0.980` | `0.970792` | `3.1525` | `0.938188` |

The released BM25 starter achieved Hit@10 `0.125`, MRR `0.068034`, and MTTC
`9.81`. This agent reaches Hit@10 `0.995`, MRR `0.985167`, and MTTC `2.97`
without changing the official evaluator, catalog, labels, or protocol.

## How it works

```text
Shopper message
  ├─ released simulator wording ──> deterministic typed parser
  └─ unfamiliar free-form text ──> optional local model
                                      │
                                      └─ timeout + confidence + evidence gate
                                                       │
                                                       v
Typed session state: category, positive slots, negative slots, overrides
                              │
                              v
Exact catalog-card matches + weighted in-memory SQLite FTS5 BM25
                              │
                              v
Constraint-first reranking + unseen filtering + ambiguity-aware Top-K
                              │
                              v
Recommendations and a candidate-informed clarification question
```

The session state accumulates confirmed information while replacing only the
slot named by an override. “No leather” becomes a negative material constraint,
“forget leather” removes a prior material constraint, and “blue instead of
black” rewrites colour without erasing unrelated requirements.

The default ranker prioritizes satisfied hard constraints before lexical rank.
It returns Top-1 while confidence is high, widens under large exact-card ties,
and uses all ten allowed positions on the final turn. Already shown products
are filtered so each turn explores new candidates.

An explicit Buying/Browsing/Uncertain router is available behind
`--experimental-router`. Buying uses exact constraints and BM25; Browsing adds
dense retrieval and diversity; Uncertain fuses both routes. It remains off by
default because its public score was `0.95355`, slightly below the proven
`0.95365` path.

## Quick start

Requirements: Python `3.10+` (`3.13` is used in the verified setup), `uv`, and
the official frozen catalog.

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r solution/requirements.txt
```

Download `catalog.jsonl.gz` from the
[official participant-kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit),
decompress it, and place the 50,000-row file at:

```text
data/catalog.jsonl
```

Run the public evaluation:

```bash
.venv/bin/python run_solution.py --output results.json
```

Expected TechnicalScore: `0.95365` with zero reported model tokens.

## Reproduce the verified checks

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_solution.py --output results.json
.venv/bin/python stress_eval.py --output stress_results.json
.venv/bin/python -m training.evaluate_extraction
```

Expected results:

- Tests: `29/29`
- Public TechnicalScore: `0.95365`
- Stress TechnicalScore: `0.95160`
- Rule-only free-form extraction micro-F1: `0.0583` across 200 seed cases
- Model calls on released simulator templates: `0`

The submission ZIP was also verified from a fresh temporary directory with
only `data/catalog.jsonl` added; it reproduced the same tests and scores.

## Optional experiments

The deterministic default only requires NumPy. Install heavier research
dependencies explicitly:

```bash
uv pip install --python .venv/bin/python -r solution/requirements-experiments.txt
```

This enables dense retrieval, the experimental intent router, cross-encoder
diagnostics, and the scikit-learn training scripts. These paths may download
model weights and are not required for official scoring.

For the optional local structured extractor:

```bash
uv pip install --python .venv/bin/python -r solution/requirements-llm.txt
.venv/bin/python run_solution.py \
  --extractor-gguf /absolute/path/to/Qwen3-0.6B-Q8_0.gguf \
  --extraction-timeout 3
```

On an 18 GB Apple M3 Pro using Metal, Qwen3-0.6B Q8_0 loaded in `0.54s` and
used about `1.04 GB` additional peak process RSS. Its extraction benchmark is
single-author research data, and no tested prompt met the promotion gate after
the duplicate hardcoded grounding parser was removed. DSPy and OpenAI are used
only offline to generate prompt candidates; Qwen executes and scores those
candidates locally, and neither service is needed for official evaluation.

## Engineering evidence

We kept experiments only when they passed public, stress, and ASIN-separated
synthetic gates.

| Experiment | Outcome | Decision |
|---|---|---|
| Typed state, exact cards, aliases, negative constraints | Public `0.95365`; stress `0.95160` | Accepted |
| Ambiguity-gated Top-K policy | Improved validation and test TechnicalScore | Accepted |
| Clause-order metamorphic test | 189/189 state-equivalent; score delta `0.0` | Accepted as regression harness |
| Global dense retrieval and cross-encoder reranking | Lower end-to-end score | Rejected |
| Dense Buying/Browsing router | Public `0.95355` | Experimental flag only |
| Profile, title-specificity, popularity tie-breaks | Reduced public and/or stress score | Rejected |
| Override output widening | Improved MTTC but reduced combined score | Rejected |
| Free-form-gated top-100 reranker | Public/stress unchanged; model dev improved; frozen model test neutral | Experimental flag only |
| Trigram recall and confidence Top-K | Reduced public/stress MRR or smoke TechnicalScore | Rejected |
| Catalog-trained top-50 reranker | Public/stress unchanged; product-disjoint and frozen free-form tests improved | Validated optional candidate |

Detailed diagnostics and ablation reports live in `training/`; raw evaluator
outputs are retained under `artifacts/evaluations/` and are not needed at
runtime.

## Limitations

- Some intent cards describe hundreds of metadata-identical products without
  disclosing the title phrase that distinguishes the purchased item. Any
  tie-breaker then encodes a prior rather than evidence from the conversation.
- The released simulator is deterministic. The stress and metamorphic harnesses
  test paraphrases and clause order, but they are not a substitute for a large
  independently authored conversation set.
- Dense retrieval and the explicit router are implemented but remain
  experimental because they did not beat the deterministic ranker end to end.
- The optional local extractor has not met its safety and accuracy promotion
  gates. The 200-case corpus is single-author seed data and must not be
  presented as independent human validation.

The remaining competition work should improve candidate recall and target rank
on unseen products: trace the dense-router regression, calibrate selective
BM25/vector fusion, and test reranking and Top-K changes on ASIN-separated
validation. More prompt optimization is only worthwhile for a post-hackathon
human-facing product or if organizers confirm free-form hidden messages.

## Repository layout

```text
solution/agent.py                 production agent and retrieval pipeline
solution/extraction.py            optional local structured extraction
run_solution.py                   official public evaluation entry point
stress_eval.py                    deterministic paraphrase stress harness
training/                         synthetic splits, traces, ablations and reports
artifacts/evaluations/            archived raw experiment outputs
evaluator/                        unmodified official evaluator
data/public_set.jsonl             official 200-session development set
tests/                            parser, state, evaluator and training checks
docs/                             challenge contract and team handoff
```

## Technology and cost

- Python, SQLite FTS5, NumPy, and `uv`
- Optional: sentence-transformers, scikit-learn, llama.cpp, Qwen3 GGUF
- Dataset: frozen 50,000-product Amazon Reviews 2023
  `Clothing_Shoes_and_Jewelry` catalog supplied by TechJam
- Scorer-proven path: fully offline, no external API, no credentials, zero model
  tokens, and zero marginal inference cost

## Team

Team names and contribution ownership are intentionally left for the team to
confirm before Devpost submission. The proposed allocation and demo checklist
are in [`docs/team_handoff.md`](docs/team_handoff.md).

## Attribution

See [`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md) for dataset attribution and
[`docs/competition_specification.md`](docs/competition_specification.md) for the
official evaluation contract.
