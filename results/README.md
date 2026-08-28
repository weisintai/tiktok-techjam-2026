# Evaluation Results

This directory contains generated evaluator and experiment outputs. JSON result
files are intentionally ignored because they contain reproducible per-session
records and create substantial repository noise.

Generate the current public evaluation with:

```bash
PYTHONHASHSEED=1 PYTHONPATH=. python3 -m evaluator.local_evaluator \
  --output results/public_evaluation.json
```

Generate pairwise-reranker diagnostics with:

```bash
PYTHONHASHSEED=1 PYTHONPATH=. python3 experiments/train_pairwise_reranker.py
```

The benchmark summary and provenance notes are maintained in
`docs/experiment_log.md` and the root `README.md`.
