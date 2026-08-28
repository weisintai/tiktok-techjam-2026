# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator --output results/public_evaluation.json
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
Evaluator outputs are stored in the `results/` directory.

For an interactive free-form conversation, run:

```bash
PYTHONPATH=. python3 scripts/manual_chat.py
```

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Results

The current offline agent combines systematic typed intent parsing, stateful
tracking, multi-route FTS and facet retrieval, information-gain clarification, a
pairwise top-40 reranker, bounded product compatibility matching, and
conversion-aware recommendation gating.

| Evaluation | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| Official public set, 200 sessions | **0.995** | **0.905875** | **2.805** | **0.933163** |

Scenario breakdown from the same run:

| Scenario | Sessions | Hit@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Boundary | 10 | 1.0000 | 0.574167 | 3.30 |
| Browsing | 80 | 1.0000 | 0.922292 | 2.4875 |
| Buying | 80 | 0.9875 | 0.928542 | 2.3875 |
| Intent override | 30 | 1.0000 | 0.912222 | 4.60 |

The released BM25 starter achieved Hit@10 `0.125`, MRR `0.068034`, and MTTC
`9.81`. This branch reaches Hit@10 `0.995`, MRR `0.905875`, and MTTC `2.805`
without changing the official evaluator, catalog, labels, or protocol. These
figures were reproduced from commit `5897124` on branch `shwe-experiment` with
`PYTHONHASHSEED=1`. The output is stored locally at
`results/shwe_experiment_public.json`. The stress and synthetic datasets are not
present in this workspace, so no stress or synthetic scores are claimed. The
default evaluator pipeline makes zero external model calls.

See `docs/how_it_works.md` for an end-to-end explanation of the runtime pipeline.
See `docs/project_reference.md` for the consolidated Track 4 brief, architecture,
rubric mapping, deliverables, and caveats. See `docs/experiment_log.md` for every
tested hypothesis, including rejected experiments and historical checkpoints.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/how_it_works.md              end-to-end implementation guide
results/                          generated evaluation outputs
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
