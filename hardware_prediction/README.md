# Physical power and energy prediction

The current test data are **nine motions x five executions = 45 recordings**. Start with the [repeated-evaluation README](repeated_evaluation_20261109/README.md) and the [reviewer guide](../docs/REVIEWER_GUIDE.md).

From the repository root, `python tools/reproduce.py hardware` verifies the original 116/23 split and frozen models, reconstructs all 45 raw executions, and checks the current metrics. `coverage`, `windows`, and `figures` also use this current set by default.

| Location | Role |
|---|---|
| [repeated_evaluation_20261109/](repeated_evaluation_20261109/) | Current 45-recording raw archive, metrics, portable scripts and analysis |
| [hardware_experiment/regression_results/ridge_20260915_143619/](hardware_experiment/regression_results/ridge_20260915_143619/) | Unchanged 116 training/23 validation inputs, split and frozen R0-R3 models |
| [Training raw data](hardware_experiment/session_v3_uniform_10hz/live_batches/) / [validation raw data](hardware_experiment/session_v3_validation_10hz/live_batches/) | Original acquisition streams |
| [manuscript/data/](manuscript/data/) | Historical single-execution results and original subset identities; not the current primary test set |
| [manuscript/scripts/](manuscript/scripts/) | Original analysis implementations retained for provenance |
| [hardware_experiment/](hardware_experiment/) | Original acquisition, calibration and feature code |
| [external/](external/) | Supplier library and retained notices |

English reading documents explain the current workflow. Hash-protected original records can retain Japanese strings; do not change their bytes or line endings. The [update note](../docs/UPDATE_45_TRIALS.md) explains the distinction between current and historical results.
