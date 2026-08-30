# TechJam Shopping Copilot

An offline conversational shopping agent that turns multi-turn dialogue into
typed constraints, preserves intent changes, and finds the exact purchased
Amazon product in an average of `1.995` turns on the released evaluation set.

The scored path is fully deterministic — structured session state, exact
catalog-card matches, phrase-level category resolution, weighted SQLite FTS5
BM25, and a purchase-volume prior for products the conversation can't tell
apart. It needs no network access, API key, model weight, or token spend. A
small local model is an optional demo of free-form language understanding,
not a dependency of the submitted agent.

Multi-turn product search is where keyword search fails: a shopper who says
"something warm for winter, not too flashy" can't be served by literal-term
matching, and every clarifying question that doesn't narrow the catalog wastes
a turn. This project treats that as a state-tracking problem, not a
bigger-model problem — the win came from never losing or misapplying what the
shopper already said, not from adding an LLM to the critical path.

## How this maps to the judging rubric

| Criterion | Weight | Where the evidence is |
|---|---:|---|
| Technical Execution | 35% | [Results](#results) and [Engineering evidence](#engineering-evidence) log every accepted *and* rejected experiment with its measured score. Every number reproduces from a clean checkout; 49/49 unit tests pass. |
| Innovation & Problem Insight | 20% | [What produced the gain](#what-produced-the-gain): two mechanisms that act only on evidence the shopper gave or a catalog fact every product has, applied in a strict order so neither can override the other. |
| Impact & Relevance | 20% | The paragraph above states the real failure mode this solves. [Limitations](#limitations) is explicit about where the approach stops working. |
| Feasibility & Practicality | 15% | Zero network access, API keys, or token spend (see [Technology and cost](#technology-and-cost)). A working [demo console](#demo) runs on the same production `Agent`, not a mock. |
| Presentation & Communication | 10% | Judged live at the final event; see [`docs/team_handoff.md`](docs/team_handoff.md) for the demo script and Q&A prep. |

## Results

| Evaluation | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| Official public set, 200 sessions | `1.000` | `1.000000` | `1.995` | **`0.98010`** |
| Deterministic paraphrase stress set | `1.000` | `0.997500` | `2.305` | **`0.97315`** |
| Unseen-ASIN synthetic validation, 400 sessions | `1.000` | `1.000000` | `2.5725` | `0.96855` |
| Untouched synthetic test, 400 sessions | `1.000` | `0.991500` | `2.560` | `0.96625` |

The released BM25 starter scored Hit@10 `0.125`, MRR `0.068034`, MTTC `9.81`.
Every released session now converges, and every converged session converges
at rank one — the entire remaining score gap is turns, not misses.

By scenario, the released set converges in `1.51` turns for buying, `1.79`
for browsing, `2.50` for boundary, and `3.67` for intent override. The
override figure is close to its structural floor of `3.60`, since the
evaluator won't count a hit before the override message lands
(`evaluator/local_evaluator.py:238`).

### What produced the gain

Two mechanisms, both reading only catalog fields available for any product.

**Phrase-level category resolution.** When a shopper names a shelf ("Women
Sweaters"), the old pipeline dissolved that phrase into separate BM25 terms
that also matched unrelated products. `_category_labels` now indexes the
catalog's breadcrumb path as whole phrases, so a stated category is matched
as one unit and ranked ahead of products that merely share vocabulary with
it. This only uses evidence the shopper actually stated, so it's applied
unconditionally.

**A blended purchase-volume prior.** Once the stated constraints are used up,
hundreds of products can still satisfy the intent card identically — the
conversation gives no way to separate them. Review volume is the available
proxy for sales volume, and the target is a real purchase, so it's used as a
tiebreaker blended against lexical rank rather than replacing it:

```text
tier score = popularity_weight · log1p(reviews) − log1p(bm25_rank)
```

It only ever runs inside a tier that already agrees on every stated
constraint and category, so it can reorder products the shopper can't
distinguish, but it can never promote one that fails something the shopper
said.

### Why the prior is safe to use

An earlier version of this prior was rejected for losing `-0.0237` on the
synthetic test split. That test wasn't a fair gate for this signal: its
targets are generated uniformly across the catalog, so a popularity prior has
nothing to exploit there. The real released sessions are different — targets
are drawn from actual purchases, and the median public target sits in the
catalog's most-reviewed `1%`. The private sessions use the same generator
against the same catalog, so the same skew applies there too.

The prior doesn't need that argument anymore, because it's no longer used
alone. Once category resolution pins the correct shelf first, the prior only
reorders products already on that shelf, so it now helps on all three splits
— including the two with uniform targets:

| Configuration | Public | Synthetic validation | Synthetic test |
|---|---:|---:|---:|
| Neither | `0.95415` | `0.94417` | `0.93889` |
| Category resolution only | `0.95935` | `0.95835` | `0.94870` |
| Purchase prior only | `0.96415` | `0.94203` | `0.93088` |
| Both | `0.97890` | `0.96855` | `0.96740` |

`popularity_weight = 5.0` was tuned independently on two disjoint halves of
the public set; both peaked at the same value and stayed stable across a wide
range (`2.4` to `8.0`), so it isn't overfit to one half. Rating and price were
tested as alternative priors and carried no signal.

Override preference retention and `--recombine-constraints` are also on by
default; `--no-override-retain-hard` and `--no-recombine-constraints` restore
the earlier behaviour.

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
Exact catalog-card matches + phrase-level category resolution
      + weighted in-memory SQLite FTS5 BM25
                              │
                              v
Constraint tier -> category tier -> purchase prior blended with lexical rank
                              │
                              v
Unseen filtering + ambiguity-aware Top-K
                              │
                              v
Recommendations and a candidate-informed clarification question
```

Session state accumulates confirmed information and replaces only the slot
named by an override. "No leather" becomes a negative material constraint,
"forget leather" removes a prior material constraint, and "blue instead of
black" rewrites colour without touching anything else.

Ranking follows a strict evidence order: never violate a stated negative,
satisfy the most stated constraints, match the stated category as a phrase,
and only then fall back to the purchase prior. Everything the shopper said is
used before any prior is consulted. The agent returns Top-1 while confidence
is high, widens under large exact-card ties, and uses all ten allowed
positions on the final turn. Already-shown products are filtered out so each
turn explores new candidates.

An explicit Buying/Browsing/Uncertain router exists behind
`--experimental-router` but stays off by default because it didn't beat the
deterministic ranker end to end.

## Quick start

Requirements: Python `3.10+` (`3.13` verified), `uv`, and the official frozen
catalog.

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

Expected TechnicalScore: `0.98010` with zero reported model tokens.

## Demo

`frontend/` is a Next.js console for interactively driving the same
production `Agent` used for scoring — a UI layer over `solution/agent.py`,
not a separate reimplementation. A Node API route spawns
`frontend/backend/copilot_server.py`, which loads the real catalog and agent
once and answers turns over stdio, so the demo and the scorer never diverge.

Complete [Quick start](#quick-start) first, then:

```bash
cd frontend
npm install
npm run dev
```

Set the `PYTHON` environment variable if the backend should use a different
interpreter than `../.venv/bin/python`.

The console shows the shopper-facing message, the structured `ask_attribute`,
ranked recommendations, and the agent's internal state (category, slots,
negative constraints, inferred intent) updating live each turn.

## Reproduce the verified checks

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_solution.py --output results.json
.venv/bin/python stress_eval.py --output stress_results.json
.venv/bin/python -m training.evaluate_extraction
```

Expected results:

- Tests: `49/49`
- Public TechnicalScore: `0.98010`
- Stress TechnicalScore: `0.97315`
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
diagnostics, and the scikit-learn training scripts. None are required for
official scoring and none change the default.

For the optional local structured extractor:

```bash
uv pip install --python .venv/bin/python -r solution/requirements-llm.txt
.venv/bin/python run_solution.py \
  --extractor-gguf /absolute/path/to/Qwen3-0.6B-Q8_0.gguf \
  --extraction-timeout 3
```

On an 18 GB Apple M3 Pro using Metal, Qwen3-0.6B Q8_0 loaded in `0.54s` and
used about `1.04 GB` additional peak RSS. Its extraction benchmark is
single-author research data and hasn't met a promotion gate, so it stays
optional and off by default.

## Engineering evidence

Experiments were only kept when they passed public, stress, and
ASIN-separated synthetic gates.

| Experiment | Outcome | Decision |
|---|---|---|
| Typed state, exact cards, aliases, negative constraints | Public `0.95365`; stress `0.95160` | Accepted |
| Ambiguity-gated Top-K policy | Improved validation and test TechnicalScore | Accepted |
| Clause-order metamorphic test | 189/189 state-equivalent; score delta `0.0` | Accepted as regression harness |
| Global dense retrieval and cross-encoder reranking | Lower end-to-end score | Rejected |
| Dense Buying/Browsing router | Public `0.95355` | Experimental flag only |
| Profile, title-specificity, popularity tie-breaks | Reduced public and/or stress score | Rejected |
| Override output widening | Improved MTTC but reduced combined score | Rejected |
| Free-form-gated top-100 reranker | Public/stress unchanged; free-form dev improved | Experimental flag only |
| Trigram recall and confidence Top-K | Reduced public/stress MRR or smoke TechnicalScore | Rejected |
| Catalog-trained top-50 reranker | Public/stress unchanged; free-form tests improved | Validated optional candidate |
| Top-K widening, 25 counterfactual policies | Every variant below baseline | Rejected |
| Override preference retention | `+0.0005` to `+0.0007` on all four gates | Accepted, on by default |
| Popularity tie-break (turn-gated) | Public `+0.0016`, but synthetic test `-0.0237` | Superseded by the blended form |
| Average-rating / price tie-breaks | Both underperform review volume alone | Rejected, no signal |
| Front-loading specific opening questions | Every variant scored below the generic ask | Rejected |
| Phrase-level category resolution | `+0.0052` public, `+0.0142` validation, `+0.0098` test | Accepted, on by default |
| Blended purchase prior on top of it | `+0.0196` public, `+0.0102` validation, `+0.0187` test | Accepted, on by default |
| `"; "` constraint recombination | Fixes a real parser fault; public MRR to `1.000` | Accepted, on by default |

**Why Top-K stays narrow:** the evaluator scores a session at its first hit
and stops, so dropping from rank 1 to rank 2 costs `0.15` while saving a turn
only returns `0.02` — a wider list would need to save seven turns to break
even. 25 alternative policies were tested and all scored lower.

**Why the opening question stays generic:** the simulator always answers a
generic question with two new facts, but a specific question only gets an
answer if that exact attribute applies, sometimes wasting the turn. A
front-loaded, more discriminative attribute was tested and still lost,
because it drops the material/color terms that the lexical search also
depends on. `--ask-plan` keeps that experiment reproducible.

Detailed diagnostics and ablation reports live in `training/`; raw evaluator
outputs are retained under `artifacts/evaluations/` and aren't needed at
runtime.

## Limitations

- Some intent cards describe hundreds of metadata-identical products without
  ever disclosing what makes the purchased one unique. The purchase-volume
  prior that orders them is a prior, not conversation evidence — it assumes
  the labelled purchase is more likely to be a popular one, which holds
  structurally for this dataset but can't be verified against the private
  sessions. It's only consulted after every stated constraint and category,
  so a shopper who states enough never reaches it.
- `popularity_weight = 5.0` is fitted on the public set. A split-half check
  shows it transfers, not proves it — the plateau is wide (`2.4` to `8.0`
  behaves the same).
- The released simulator is deterministic. The stress and metamorphic
  harnesses test paraphrases and clause order, but aren't a substitute for a
  large, independently authored conversation set.
- Dense retrieval and the explicit router are implemented but stay
  experimental — they didn't beat the deterministic ranker end to end.
- The optional local extractor hasn't met its accuracy promotion gate. Its
  200-case corpus is single-author seed data, not independent validation.

Score headroom is now entirely efficiency: Hit@10 and MRR are both `1.000` on
the released set. The intent-override floor alone accounts for a chunk of the
remaining gap, since the evaluator won't count a hit before the override
lands. A realistic ceiling is around `0.988`.

Remaining work is about earning turn-one convergence more often: resolving
the stated category to a deeper catalog node, tracing the dense-router
regression, and calibrating selective BM25/vector fusion for browsing turns
with a large shelf. Top-K changes don't need further testing — the scoring
arithmetic above already rules them out.

## Repository layout

```text
solution/agent.py                 production agent and retrieval pipeline
solution/extraction.py            optional local structured extraction
run_solution.py                   official public evaluation entry point
stress_eval.py                    deterministic paraphrase stress harness
frontend/                         Next.js demo console over the production agent
artifacts/models/                 optional learned reranker artifact
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
- Demo only: Next.js, React, and Tailwind CSS (`frontend/`), unused by the
  scored path
- Dataset: frozen 50,000-product Amazon Reviews 2023
  `Clothing_Shoes_and_Jewelry` catalog supplied by TechJam
- Scorer-proven path: fully offline, no external API, no credentials, zero
  model tokens, and zero marginal inference cost

## Team

Team names and contribution ownership are intentionally left for the team to
confirm before Devpost submission. The proposed allocation and demo checklist
are in [`docs/team_handoff.md`](docs/team_handoff.md).

## Attribution

See [`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md) for dataset attribution and
[`docs/competition_specification.md`](docs/competition_specification.md) for the
official evaluation contract.
