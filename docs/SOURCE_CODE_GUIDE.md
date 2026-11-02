# Source-code guide

Use [tools/reproduce.py](../tools/reproduce.py) for analysis. It resolves paths relative to this repository and does not open serial ports or launch the simulator. Historical acquisition/build scripts are retained for transparency; they are not interchangeable with the portable entry point.

| Area | Source | What it does |
|---|---|---|
| Physical power models | [analysis.py](../hardware_prediction/hardware_experiment/analysis.py) | Builds R0–R3 features, fits ridge regression, predicts power and computes metrics. |
| Generalization/experience audit | [transfer_audit.py](../hardware_prediction/manuscript/scripts/transfer_audit.py) | Reconstructs held-out motions and runs feature/subset comparisons. |
| Supporting simulation design | [protocol.py](../synthetic_webots/src/protocol.py) | Defines the controlled synthetic experiment and trial catalogue. |
| Supporting simulation analysis | [analyze.py](../synthetic_webots/analysis/analyze.py) | Fits M1–M5 and summarizes stored observations. |
| PPO training/evaluation | [frozen source](../motion_learning/logs/distance_goal_20261011/source/) | Original Webots environment, gait generator, training/evaluation and configuration. |
| Transfer/export | [transfer directory](../motion_learning/hardware_transfer_distance/) | Reconstructs 50 Hz commanded positions, validates exports, logs and analyzes hardware replay. |
| Actual hardware acquisition | [returned experiment](../motion_learning/logs/real_results_20261011_import/KHR3HV_Real_Results_Seeds10-15_20261011/experiment/) | Code and configurations retained with the actual 24 recordings. |
| Numerical paper panels | [tools](../tools/) | Recreates numerical figures from the retained data. |

The original robot-control dependency is [Rcb4Lib](../hardware_prediction/external/Rcb4Lib/README_EN.md). The original Japanese comments, supplier notices and frozen report strings are preserved rather than modifying code checked by experiment manifests. English experiment guides explain each operational path. Analysis users need not read those Japanese strings or run any acquisition code.

The code includes model fitting, evaluation, Webots simulation and saved-command hardware replay. It does not include an independently validated live PPO controller for the physical robot. A saved command sequence is not online policy feedback.
