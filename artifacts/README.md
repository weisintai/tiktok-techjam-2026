# Experiment artifacts

`evaluations/` contains raw per-session outputs retained for auditability. They
are not loaded by the agent and are not required to reproduce the final score.
`legacy/` preserves superseded experiment drivers and the older snapshot note;
use the maintained scripts in `training/` for current evaluation.

The canonical, compact conclusions live in `training/`:

- `router_ablation_results.json`
- `retrieval_error_analysis.json`
- `specificity_tiebreak_ablation.json`
- `override_widening_ablation.json`
- `popularity_tiebreak_ablation.json`
- `adversarial_mutation_results.json`
- `bm25_stability_report.json`
- `local_extraction_report.json`

Accepted baselines are `evaluations/local_baseline.json` and
`evaluations/stress_baseline.json`. Other files preserve rejected experiment
outputs so reported trade-offs remain verifiable.
