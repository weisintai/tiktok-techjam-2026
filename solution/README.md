# Hybrid Shopping Agent

This agent combines three in-memory retrieval signals:

1. exact matches against normalized catalog-derived product cards;
2. weighted SQLite FTS5 BM25;
3. optional MiniLM semantic retrieval with reciprocal-rank fusion.

It keeps confirmed constraints in structured session state, selectively handles
intent overrides, excludes already shown items, asks the open-ended `other`
attribute, and uses a score-optimized output schedule: Top-1 through turn 6,
then Top-3, widening to Top-5 only for more than 100 complete catalog ties, and
using all ten allowed slots on the final turn. Explicit negative constraints are
retained separately and demote conflicting products.

Constraints are stored in typed slots. A genuine override rewrites only the
conflicting slot and preserves independent confirmed requirements. Equivalent
paraphrases confirm an existing slot instead of erasing sibling constraints.
Natural overrides such as `Actually, make them casual white sneakers` also
rewrite the stale category and inline facets.

Messages outside the established simulator fast path use an evidence-grounded
deterministic delta extractor. It handles explicit additions, replacements,
removals, exclusions, no-preference slots, unresolved budget mentions, and
show-options-first instructions. The fallback updates state through the same
`StructuredTurn` boundary and adds no model call or dependency.

The natural clarification prompt is selected from candidate-facet entropy and
weak aggregate-profile priors. The structured contract deliberately remains
`ask_attribute="other"`: direct attribute selection reduced the official score,
whereas a broad multi-facet prompt lets the customer reveal the most useful
remaining information. Profile signals never override explicit session slots.

## Run

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r solution/requirements.txt
.venv/bin/python run_solution.py
```

The normalized exact-card/BM25 agent is the recommended default. It has the
best untouched development score and nearly preserves it under the included
paraphrase stress test. A confidence-gated cross-encoder remains available as
an ablation:

```bash
uv pip install --python .venv/bin/python -r solution/requirements-experiments.txt
.venv/bin/python run_solution.py --cross-encoder
```

`--dense` enables the experimental full-catalog MiniLM route and builds
`data/minilm_card_embeddings.npz` on first use. Neither neural route improved
the end-to-end stress score in the current experiments, so both are retained
for reproducibility rather than recommended for submission.

## Optional local structured extraction

The deterministic parser remains the zero-latency path for released simulator
messages and common free-form operations. For unfamiliar shopper language, an
optional local causal model can propose typed slots, exclusions, intent and
slot-level replacements:

```bash
uv pip install --python .venv/bin/python -r solution/requirements-llm.txt
.venv/bin/python run_solution.py --extractor-model /path/to/Qwen3-0.6B
.venv/bin/python run_solution.py --extractor-gguf /path/to/Qwen3-0.6B-Q8_0.gguf
```

The GGUF/llama.cpp path is recommended for local execution. The optional backend
is included in `requirements-llm.txt`; keep it in the project `.venv` rather than
installing it into the global Python environment.

Model files must be downloaded or bundled before offline judging. Output is
schema-validated and confidence-gated; model loading, inference, malformed JSON
or low-confidence extraction automatically falls back to the deterministic
path. Optional-model operations are also evidence-gated at the shared state
boundary, so unsupported additions, exclusions, removals, replacements and
intent labels cannot mutate confirmed state. Explicit session constraints
remain authoritative.

## Robustness stress test

```bash
.venv/bin/python stress_eval.py
.venv/bin/python stress_eval.py --cross-encoder
```

The stress harness paraphrases common materials, colors, use cases, price
phrases, clarification templates, and override language before the agent sees
them. It is intentionally harder than the released deterministic simulator.

## Leakage-safe synthetic evaluation

```bash
.venv/bin/python -m training.generate_sessions
.venv/bin/python -m training.evaluate_split --limit 400
.venv/bin/python -m training.train_action_policy --train-products 400 --validation-products 150
.venv/bin/python -m training.optimize_topk --rebuild-traces --limit 400
```

The generator creates four scenario types for every catalog product, assigns
whole ASINs to stable 80/10/10 splits, and quarantines every public target from
all training and tuning splits. The current learned rank-bucket policy remains
an experiment: it predicts rank-1 and miss states well but is not reliable
enough on ranks 2–10 to replace the fixed output policy.

`optimize_topk.py` instead records each ranking trace once and counterfactually
replays hundreds of static and ambiguity-gated policies. The selected policy
improved TechnicalScore on both the quarantined validation split and the
untouched synthetic test split before being promoted to the agent.
Policy selection is weighted to the official scenario mix: 40% Buying, 40%
Browsing, 15% Intent Override and 5% Boundary. Public and paraphrase-stress
scores remain mandatory promotion gates because synthetic optimization alone
selected one threshold that did not generalize.

## Catalog-derived extraction lexicon

During the existing catalog-index build, the agent counts leaf categories and
typed intent-card facets. The free-form fallback receives a lexicon made only
from short phrases appearing in at least three catalog products. Generic
taxonomy nodes and known metadata boilerplate are excluded, while longer
matches suppress nested generic matches.

On the frozen 50,000-product catalog this produces 535 leaf-category phrases
and 1,145 facet phrases. It expands coverage for catalog terms such as
`rash guard shirts`, `loafers and slip ons`, and specific closure
types without changing the read-only catalog or adding a runtime dependency.
Conversational operations remain explicit deterministic rules; catalog text
supplies product vocabulary rather than intent labels.

Unrecognized wording is retained for at most three fallback turns as soft query
text. It participates in BM25 and optional dense retrieval but never becomes an
exact facet, exclusion, or state mutation. Intent overrides clear this residual
text. This lets open-vocabulary terms remain searchable while typed constraints
keep their higher ranking priority.

## Offline robustness diagnostics

```bash
.venv/bin/python -m training.evaluate_mutations
.venv/bin/python -m training.bm25_stability
```

The mutation harness only scores clause reorderings after confirming equivalent
parsed state. The stability tool perturbs BM25 field weights over a frozen
candidate pool and reports leader survival and Top-K overlap; neither tool can
change production recommendations.
