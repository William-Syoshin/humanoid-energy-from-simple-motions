# Physical power and energy prediction

Start with the [reviewer guide](../docs/REVIEWER_GUIDE.md). From the repository root, `python tools/reproduce.py hardware` verifies the 116/23 data split, refits R0-R3, and reconstructs the nine test recordings.

| Location | Role |
|---|---|
| [hardware_experiment/regression_results/ridge_20260915_143619/](hardware_experiment/regression_results/ridge_20260915_143619/) | Exact aligned fitting inputs, split, frozen models, and validation results |
| [Training raw data](hardware_experiment/session_v3_uniform_10hz/live_batches/) / [validation raw data](hardware_experiment/session_v3_validation_10hz/live_batches/) | Original acquisition streams |
| [manuscript/data/](manuscript/data/) | Test selection, saved predictions, and retrospective analyses |
| [manuscript/scripts/](manuscript/scripts/) | Original analysis implementations |
| [hardware_experiment/](hardware_experiment/) | Original acquisition, calibration, and feature code |
| [external/](external/) | Supplier library and retained notices |

[English trial metadata](../docs/english_metadata/primary_test_trials.csv) translates labels without replacing the original CSV used by the analyses. [Archive notes](../docs/ARCHIVE_NOTES.md) explain historic fields and supplier-code comments. For the integration-window analysis, read [the English results](manuscript/frontiers_20261023/analysis/integration_windows_20261101/RESULTS.txt).
