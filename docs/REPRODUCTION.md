# Reproduction guide

All documented `tools/reproduce.py` actions are offline. They read only this repository and create a new `generated/` subdirectory. Start with `verify`, then `hardware`. Use `all` for the complete audit.

| Action | Check performed |
|---|---|
| verify | SHA-256 of the prepared repository payload |
| hardware | Frozen-input hashes; train/validation disjointness; refit R0–R3 with validation selection; compare coefficients; reconstruct nine held-out recordings from device streams; reproduce energy and waveform metrics |
| coverage | Recompute feature comparisons, random experience subsets, and coverage-constrained subsets for all 20 seeds; compare selected subsets and metrics with saved results |
| windows | Reintegrate frozen predictions over 2/5/10 s windows and full recordings; independent integration consistency checks and saved-summary comparison |
| rl | Verify frozen source, final policy hashes and configurations; recompute features, OOD flags and power integrals for all 240 evaluations; match paired comparisons and Holm-adjusted tests |
| replay | Check the returned measurement archive, plans/configs/commands, policy-export identities and acknowledgments; reintegrate all 24 real recordings and compare published summaries |
| synthetic | Verify all 1,114 raw hashes; refit M1–M5 and all reported subset comparisons; compare saved summaries and mechanical checks |
| figures | Rebuild numerical panels corresponding to paper Figs. 4, 5, 6, 7 and 10 |

Tested analysis environment: Python 3.11, with exact package versions in `requirements-analysis.txt`. Validation reports distinguish executed checks from merely included source. A fresh Windows/macOS/Linux installation has not been certified by this packaging run. PyTorch/PPO is not required for offline energy analysis.

## Generating new simulator data

The frozen PPO implementation and its original dependency file are under `motion_learning/logs/distance_goal_20261011/source/`. Webots R2025a supplies the `controller` Python module; do not install an unrelated PyPI package with that name. `rl/train.py`, `rl/evaluate.py`, `env/gait.py`, configs and motor parameters are retained. Original experiment runners are collection-time macOS scripts with local executable paths and sleep-inhibition calls. Port those environment settings and use a new output directory for new runs; do not point a runner at the archived experiment directory. Statistical reanalysis is reproducible here without rerunning PPO. Simulator/platform variation may affect a fresh rollout.

The synthetic collection code is in `synthetic_webots/src/runner.py`, `controllers/benchmark/`, `worlds/`, and `plugins/physics/contact_logger/`. For a fresh run, install Webots R2025a, configure the Python/runtime and Webots executable paths, rebuild the C contact plugin for the target platform, and collect into a new run name. The stored `full_v2` observations are authoritative for the paper. No simulator was launched as part of repository preparation.

## Hardware acquisition

Original acquisition code and firmware are included for method transparency. Offline tools never import a robot-control entry point. Servo mapping, calibration, timing and local ports in saved configs describe the recorded experiments; they are not a universal ready-to-run configuration for another robot. The package is not a newly validated Windows robot-deployment release. No physical repetition or new hardware validation was performed in preparing it.

## Figure sources

The current ten paper figures are supplied as vector PDFs. The portable plotting script reproduces numeric data panels, not exact photographer crops, schematic arrangement or platform-specific typography. Historical builder scripts are included for provenance; use the portable tools rather than executing old absolute-path builders directly. No PDF authoring or journal template is required for reanalysis.
