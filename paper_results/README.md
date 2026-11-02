# Numerical paper results

These CSV/JSON files support the synthetic-model and hardware replay comparisons. The physical nine-motion predictions are under [hardware_prediction/manuscript/data/](../hardware_prediction/manuscript/data/).

- [Synthetic results](synthetic_webots/): compare with `python tools/reproduce.py synthetic`.
- [Hardware replay results](rl_application/): compare with `python tools/reproduce.py replay`.
- [Reviewer guide](../docs/REVIEWER_GUIDE.md): expected values and interpretation.

All commands run from the repository root. Read the recorded evaluation definitions before pooling episodes, seeds, or physical repetitions.
