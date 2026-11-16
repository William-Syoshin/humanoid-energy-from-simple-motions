# Reproduction guide

All documented `tools/reproduce.py` actions are offline. They read only this repository and create a new `generated/` subdirectory. Start with `verify`, then `hardware`. Use `all` for the complete audit.

| Action | Check performed |
|---|---|
| verify | SHA-256 of the prepared repository payload |
| hardware | Frozen-input hashes; train/validation disjointness; refit R0–R3 with validation selection; compare coefficients; reconstruct 45 held-out executions (nine motions x five repetitions) from device streams; reproduce energy and waveform metrics |
| coverage | Recompute feature comparisons, random experience subsets, and coverage-constrained subsets for all 20 seeds; compare selected subsets and metrics with saved results |
| windows | Reintegrate frozen predictions over 2/5/10 s windows and full recordings; independent integration consistency checks and saved-summary comparison |
| rl | Verify frozen source, final policy hashes and configurations; recompute features, OOD flags and power integrals for all 240 evaluations; match paired comparisons and Holm-adjusted tests |
| replay | Check the returned measurement archive, plans/configs/commands, policy-export identities and acknowledgments; reintegrate all 24 real recordings and compare published summaries |
| synthetic | Reproduce revision_02: verify 1,114 raw hashes and 1,093 reused records; refit M1–M5/subsets; compare saved metrics, all 21 sustained mechanical checks and supplementary command-window results |
| synthetic_original | Reproduce historical full_v2 (9/21 endpoint successes); this is not the adopted revised result |
| figures | Rebuild numerical panels corresponding to current paper Figs. 4, 5, 6 and 7 |

Tested analysis environment: Python 3.11, with exact package versions in `requirements-analysis.txt`. Validation reports distinguish executed checks from merely included source. A fresh Windows/macOS/Linux installation has not been certified by this packaging run. PyTorch/PPO is not required for offline energy analysis.

## Generating new simulator data

The frozen PPO implementation and its original dependency file are under `motion_learning/logs/distance_goal_20261011/source/`. Webots R2025a supplies the `controller` Python module; do not install an unrelated PyPI package with that name. `rl/train.py`, `rl/evaluate.py`, `env/gait.py`, configs and motor parameters are retained. Original experiment runners are collection-time macOS scripts with local executable paths and sleep-inhibition calls. Port those environment settings and use a new output directory for new runs; do not point a runner at the archived experiment directory. Statistical reanalysis is reproducible here without rerunning PPO. Simulator/platform variation may affect a fresh rollout.

The adopted frozen collection code is in `synthetic_webots/runs/revision_02/source/`. The stored `revision_02` observations are authoritative for the revised supporting experiment. See its [README](../synthetic_webots/runs/revision_02/README.md) for the optional fresh-run adapter and platform-specific plugin requirements. No simulator was launched as part of this publication update; the preparation mode and offline analysis were checked.

## Hardware acquisition

Original acquisition code and firmware are included for method transparency. Offline tools never import a robot-control entry point. Servo mapping, calibration, timing and local ports in saved configs describe the recorded experiments; they are not a universal ready-to-run configuration for another robot. The package is not a newly validated Windows robot-deployment release. The current archive contains 45 physical recordings provided by the researcher. Packaging and offline checks did not collect additional measurements.

## Figure sources

The current eleven paper figures are supplied as vector PDFs. The portable plotting script reproduces numeric data panels, not exact photographer crops, schematic arrangement or platform-specific typography. Historical builder scripts are included for provenance; use the portable tools rather than executing old absolute-path builders directly. No PDF authoring or journal template is required for reanalysis.

## Repeated-evaluation update

All default physical prediction actions use [the current 45-recording dataset](../hardware_prediction/repeated_evaluation_20261109/README.md). The original nine single-execution records remain historical. Commands can also be invoked using an absolute path to `tools/reproduce.py` from another working directory. Data and script paths are resolved relative to the repository. Helvetica is used for regenerated plots when installed; DejaVu Sans is the portable fallback. The supplied manuscript PDFs retain their original vector artwork and fonts.
