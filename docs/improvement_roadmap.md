# Shopping Copilot Improvement Roadmap

## Executive decision

The current deterministic agent remains the submission baseline. It scores
`0.95365` on the public set and `0.95160` on the paraphrase stress set, runs
offline, and has passed every existing promotion gate. New work should improve
hidden-set robustness and challenge alignment without replacing that path until
it wins on independent evaluation.

The next useful investment is retrieval and ranking on unseen products. The
official protocol uses frozen scenario-driven simulator sessions, and the
deterministic parser already handles the released templates with zero model
calls. The private set changes users and target products; the supplied contract
does not state that it introduces arbitrary human-written turns. Free-form
prompt optimization is therefore paused as an optional product experiment,
while dense routing and ranking can directly change Hit@10 and MRR.

## Phase 1 progress — 28 August 2026

- Expanded the extraction corpus from 20 to 200 seed turns with frozen
  `120/40/40` train, development and test splits.
- Kept every paraphrase family inside one split and added behaviour tags,
  stable case identifiers and explicit single-author provenance.
- Added applied-state F1, exact-state accuracy, false-addition rate, sibling
  preservation and per-split reporting to the existing evaluator.
- Recorded a rule-only baseline across all 200 cases and a grounded Qwen3-0.6B
  baseline on the 40-case frozen test split.
- Added regression tests for corpus size, split isolation and equivalent state
  transitions.

The original grounded Qwen test result was recorded before the duplicate
hardcoded grounding parser was removed, so it is historical rather than a
current promotion result. After consolidation, the best development prompt
reached applied-state F1 `0.6822` with false-addition rate `0.5283`, which is
well below the promotion gate. The workflow remains reproducible, but further
prompt work is deferred unless independent user-language evaluation becomes a
requirement.

## Current baseline

| Evaluation | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| Public, 200 sessions | `0.995` | `0.985167` | `2.97` | `0.95365` |
| Paraphrase stress | `0.995` | `0.978667` | `2.975` | `0.95160` |
| Unseen-ASIN synthetic validation | `0.990` | `0.977583` | `3.24` | `0.943475` |
| Unseen-ASIN synthetic test | `0.980` | `0.970792` | `3.1525` | `0.938188` |

The current production path is:

```text
Shopper message
    |
    v
Deterministic parser
    | unfamiliar wording only
    +------------------------> optional local structured extractor
                                   |
                                   v
                         confidence + evidence gate
                                   |
    +------------------------------+
    v
Typed session state
    |
    v
Exact catalog evidence + weighted SQLite FTS5 BM25
    |
    v
Constraint-first ranking + unseen filtering
    |
    v
Ambiguity-aware Top-K + structured clarification
```

The following components exist as experiments but are not promoted:

- The Buying/Browsing router and dense MiniLM route score `0.95355`, slightly
  below the deterministic default because Browsing MTTC increased by one total
  turn across the 80 public browsing sessions.
- Prompt-only Qwen extraction reaches roughly `0.44-0.47` micro-F1 on 20 cases.
  The grounded path reaches `1.0`, but the grounding rules and benchmark are too
  small to establish generalization.
- Cross-encoder, profile, popularity, specificity, and direct clarification
  experiments reduced public or stress performance.

## What success means

Work is successful when it does at least one of these without breaking the
baseline:

1. It improves performance on independently written free-form conversations.
2. It improves retrieval recall or ranking on ASIN-separated validation and
   test data.
3. It gives credible evidence for a challenge requirement while preserving the
   scorer-proven fallback.
4. It improves reproducibility, latency, or failure safety in a way judges can
   verify.

Public score alone is not sufficient evidence. Repeatedly tuning against the
same 200 labelled sessions would make the number look better while increasing
the risk of failure on the hidden 800 sessions.

## Priority 0: Protect the baseline

Before changing behaviour, preserve a reproducible reference run.

### Required actions

- Keep the no-flag `run_solution.py` path unchanged until an experiment passes
  all promotion gates.
- Record the Python version, dependency lock or exact constraints, model flags,
  catalog checksum, output metrics, latency, and peak memory for each promoted
  candidate.
- Keep optional model files, generated embeddings, credentials, and private
  data outside the submission archive.
- Run the complete test, public, stress, and synthetic checks before promoting
  any behavioural change.
- Save failed experiments as short ablation reports instead of leaving dormant
  production branches enabled by default.

### Promotion gate

An experiment can replace the default only when it:

- passes all unit and contract tests;
- does not reduce public TechnicalScore below `0.95365`;
- does not reduce stress TechnicalScore below `0.95160`;
- does not reduce the frozen unseen-ASIN validation and test scores;
- stays inside the official timeout and memory limits on the demo machine; and
- retains a deterministic offline fallback when it depends on model inference.

If a feature improves challenge coverage but misses one metric gate, keep it as
an honest demoable experiment rather than forcing it into production.

## Priority 1: Make semantic retrieval selective

### Problem

The dense route is already implemented. Running it across browsing sessions
adds semantic recall and diversity, but the current router slightly worsens
MTTC. The regression is small enough to investigate, but adding another vector
database, embedding model, or retriever would duplicate existing machinery
before the activation policy is understood.

### Current routes

```text
Buying
  -> exact catalog evidence + BM25 + constraint-first ranking

Browsing
  -> BM25 + MiniLM + reciprocal-rank fusion + product-group diversity

Uncertain
  -> run both routes and fuse their rankings
```

### 1. Add route-level diagnostics

For every evaluation turn, record:

- inferred route and the evidence used to choose it;
- target rank in BM25, dense, fused, and final lists;
- Recall@10, Recall@50, and Recall@100 for each retrieval source;
- number of exact matches and complete-match ties;
- BM25 leading-score gap;
- candidate count before and after fusion;
- whether diversity moved the target up or down; and
- retrieval and total turn latency.

The first investigation should identify the exact browsing session responsible
for the additional public turn and determine whether the target was hurt by
dense fusion, group diversity, route classification, or unseen filtering.

### 2. Test a small activation matrix

Keep the experiment matrix narrow:

| Variant | Behaviour | Question answered |
|---|---|---|
| A | Current deterministic default | Reference |
| B | Router, lexical retrieval only | Does routing itself change behaviour? |
| C | Dense only for low-evidence browsing turns | Does selective semantics help? |
| D | Dense fusion without diversity | Is fusion or diversification causing loss? |
| E | Preserve lexical leaders, diversify the tail | Can diversity improve coverage without lowering MRR? |

A low-evidence gate can use diagnostics already produced by the ranker, such as
few exact matches, a large unresolved tie, or weak lexical separation. Avoid a
new classifier until these observable signals fail; the evaluation data is too
small to justify another learned routing model.

### 3. Tune only meaningful parameters

The existing route exposes the decisions that matter:

- BM25 and dense retrieval depths;
- reciprocal-rank fusion weights;
- Buying/Browsing/Uncertain thresholds;
- the number of lexical leaders protected from diversity; and
- the number of candidates passed to any expensive reranker.

Use a coarse grid on ASIN-separated validation, freeze one candidate, then run
it once on synthetic test, stress, and public promotion gates. Do not optimize
against the four known public outliers; the error analysis shows their target
products are often indistinguishable from large groups using the disclosed
evidence.

### Exit criteria

Promote semantic routing only if it:

- preserves Buying and Intent Override metrics;
- improves Browsing candidate recall or MRR on independent data;
- meets every baseline promotion gate; and
- has acceptable index-build time, peak memory, and P95 turn latency offline.

If it improves semantic examples but misses the TechnicalScore gate, keep the
flag and use it in the demo to show that the requested route was implemented,
measured, and rejected from the default for a concrete reason.

## Priority 2: Improve ranking only where evidence exists

### Problem

The remaining public errors are dominated by metadata-identical products. In
one buying miss, 530 products satisfy the best disclosed evidence and the
shopper never reveals the title phrase that identifies the purchase. No
reranker can infer the exact target from absent evidence without introducing a
prior that may hurt other sessions.

### Useful ranking work

Focus on cases where retrieval has meaningful differentiating evidence:

- verify that explicit negative constraints dominate all positive scores;
- parse and compare numeric budgets when catalog prices are present;
- preserve exact slot satisfaction before semantic similarity;
- use semantic reranking only when there is no complete exact match and the
  shopper supplied descriptive free-form language; and
- measure target rank before and after every reranking stage.

If an LLM or cross-encoder is tested, restrict it to a small top candidate set,
such as 10-20 products. Give it the typed state and compact catalog cards, not
the full conversation and not hundreds of documents. Its output should be a
ranking over supplied identifiers, and invalid or invented identifiers must be
discarded.

### What not to optimize

Do not add title specificity, popularity, rating, or profile priors solely to
solve known public ties. Existing ablations show that these priors move a few
targets upward while reducing overall MRR and TechnicalScore.

### Exit criteria

Promote a reranker only when it improves MRR on independent free-form or
ASIN-separated data, preserves Hit@10 and MTTC, and stays within the latency
budget. A reranker that merely changes the ordering of evidence-free ties does
not qualify.

## Priority 3: Strengthen evaluation and failure analysis

### Evaluation layers

Use different datasets for different questions:

| Dataset | Purpose | Must not be used for |
|---|---|---|
| Unit and mutation cases | Parser/state invariants | Ranking claims |
| Free-form train | Prompt compilation | Final reporting |
| Free-form development | Threshold and prompt selection | Final reporting |
| Frozen free-form test | Generalization report | Further tuning |
| ASIN-separated synthetic validation | Retrieval and Top-K selection | Final confirmation |
| ASIN-separated synthetic test | One-time confirmation | Iterative tuning |
| Paraphrase stress | Regression gate | Prompt optimization |
| Public 200 sessions | Organizer-facing promotion gate | Repeated outlier tuning |

### Error taxonomy

Every miss or delayed conversion should be assigned one primary cause:

1. **Understanding error:** the typed state differs from the shopper's message.
2. **State error:** accumulation, removal, or replacement is applied incorrectly.
3. **Retrieval miss:** the target is absent from the candidate pool.
4. **Ranking loss:** the target is retrieved but ordered too low despite useful
   differentiating evidence.
5. **Evidence-free tie:** the disclosed intent cannot distinguish the target.
6. **Question-policy loss:** the selected attribute delays useful disclosure.
7. **Output-policy loss:** the target is ranked but omitted by the current Top-K
   schedule or unseen filtering.
8. **Operational failure:** timeout, invalid output, model load failure, or
   excess resource use.

This taxonomy prevents work on the wrong layer. A retrieval model cannot fix a
state override bug, and a reranker cannot reliably solve an evidence-free tie.

### Minimum experiment report

Every experiment should record:

```json
{
  "hypothesis": "Selective dense retrieval improves vague browsing recall",
  "change": "Enable dense fusion only when no exact constraint matches",
  "datasets": ["synthetic_validation", "freeform_development"],
  "baseline": {},
  "candidate": {},
  "latency": {"median_ms": 0, "p95_ms": 0},
  "memory": {"peak_rss_mb": 0},
  "decision": "promote | reject | retain_experimental",
  "reason": ""
}
```

Keep the report small and reproducible. A new experiment framework is
unnecessary because the repository already has evaluators, trace tools, and
JSON ablation artifacts.
