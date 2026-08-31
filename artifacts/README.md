# Experiment artifacts

The only checked-in runtime artifact is:

- `models/catalog_reranker.joblib`: optional scikit-learn top-50 reranker for
  unfamiliar free-form requests. The default scored pipeline does not load it.

Generated evaluation outputs belong under `artifacts/evaluations/`. That
directory is intentionally ignored because the reports can be large and are
reproducible from the maintained scripts in `training/`.

Compact checked-in results and ablations live in `training/`; see
[`training/README.md`](../training/README.md) and
[`docs/evaluation.md`](../docs/evaluation.md) for the current index and
reproduction commands.
