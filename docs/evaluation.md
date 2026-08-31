# Evaluation methodology and results

The released public evaluation, team-created stress tests and synthetic splits
answer different questions. Results are kept separate to avoid presenting
team-created data as organizer-provided evidence.

## Results

| Evaluation | Sessions | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Released public set | 200 | `1.000` | `1.000000` | `1.995` | **`0.98010`** |
| Deterministic paraphrase stress test | 200 | `1.000` | `0.997500` | `2.305` | **`0.97315`** |
| Product-disjoint synthetic validation | 400 | `1.000` | `1.000000` | `2.5725` | `0.96855` |
| Product-disjoint synthetic test | 400 | `1.000` | `0.991500` | `2.560` | `0.96625` |

The released BM25 starter scored Hit@10 `0.125`, MRR `0.068034` and MTTC
`9.81` on the same public set.

## What each evaluation establishes

- **Released public set:** compatibility with the official evaluator and the
  primary reported score. All 200 targets are in the top ten and rank first at
  conversion.
- **Paraphrase stress test:** deterministic rewrites of released conversation
  patterns test sensitivity to equivalent wording. It is team-created and not
  a substitute for unseen human dialogue.
- **Product-disjoint synthetic splits:** whole target ASINs are separated by a
  stable hash, and all public targets are quarantined. Because the generator
  uses released intent cards and simulator logic, these are regression sets,
  not independent estimates of private performance.
- **Independent free-form extraction:** a frozen 40-case, single-author test
  measures typed state extraction. The deterministic hybrid fallback reaches
  state F1 `0.8345`; this is not an end-to-end TechnicalScore.
- **Model-authored transcripts:** three local models produced frozen stress
  conversations. These are synthetic robustness probes and are not described
  as human evaluation.

## Leakage and integrity controls

- Public target ASINs are excluded from learned-policy and reranker training.
- Synthetic train, validation and test assignments are product-disjoint.
- The independent free-form test file is checksummed and evaluated without
  failure inspection during candidate selection.
- The default result uses no hardcoded sample IDs, target-specific ASIN rules,
  catalog mutations or evaluator modifications.
- Optional learned artifacts receive only runtime-observable features.

## Reproduce the primary checks

Place the supplied catalog at `data/catalog.jsonl`, install the default
requirements, then run:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_solution.py --output results.json
.venv/bin/python stress_eval.py --output stress_results.json
```

Expected primary results are public TechnicalScore `0.98010` and stress
TechnicalScore `0.97315`. See [`training/README.md`](../training/README.md) for
the synthetic, extraction, mutation and reranking commands.

## Important limitations

- The released conversations are scripted rather than collected from users.
- The popularity weight was tuned on the public set, although stability and
  product-disjoint checks reduce the risk of a single-point overfit.
- Some products are indistinguishable from catalog evidence revealed in the
  conversation; popularity is a best-effort prior in those ties.
- The deterministic free-form extractor is less reliable than the released
  simulator path, so free-form extraction results are reported separately.
- Optional neural components did not improve the default end-to-end score and
  remain experimental.
