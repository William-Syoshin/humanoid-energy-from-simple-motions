# Reviewer guide

This guide follows the manuscript's evidence: physical prediction, supporting simulation, then motion learning and physical command replay. All commands below run from the repository root. Install [the analysis requirements](../requirements-analysis.txt) first, as described in the [README](../README.md).

## 1. Check the central prediction result

```sh
python tools/reproduce.py verify
python tools/reproduce.py hardware
```

The second command uses 116 training and 23 validation recordings, checks the frozen-model coefficients, reconstructs the nine test traces from recorded device streams, and compares the saved metrics. Its report must contain `passed: true` and `held_out_motions: 9`.

| Model | Mean absolute relative energy error |
|---|---:|
| R0 | 10.52% |
| R1 | 5.44% |
| R2 | 3.55% |
| R3 | 2.59% |

These are equally weighted errors over nine complete movements, not pooled sample errors. Model selection uses validation power RMSE, which selects R2. R3's lower test error does not retrospectively change this selection.

- [Train/validation split](../hardware_prediction/hardware_experiment/regression_results/ridge_20260915_143619/trials.csv)
- [Frozen feature/model metadata](../hardware_prediction/hardware_experiment/regression_results/ridge_20260915_143619/models/frozen.json)
- [Nine test recordings](../hardware_prediction/manuscript/data/primary_test_trials.csv) and [English metadata view](english_metadata/primary_test_trials.csv)
- [Figure 4: per-motion energy errors](../paper_figures/fig04_six_page_hardware_trial_errors.pdf)
- [Figure 5: power and cumulative residual](../paper_figures/fig05_six_page_hardware_traces.pdf)

One physical recording is available per test motion. In Supine get-up, total energy error is small even though predicted power contains large spikes; over- and underprediction can cancel during integration.

## 2. Check which information and experiences help

```sh
python tools/reproduce.py coverage
python tools/reproduce.py windows
```

`coverage` repeats the fixed data-subset selections and body-feature comparisons. `windows` repeats the 2/5/10-second energy integrations. Both are retrospective analyses of existing test recordings, not independent prospective validation.

- [Feature and coverage results](../hardware_prediction/manuscript/data/transfer_audit_20260928/)
- [Short-window results in English](../hardware_prediction/manuscript/frontiers_20261023/analysis/integration_windows_20261101/RESULTS.txt)
- [Figure 6: training-experience coverage](../paper_figures/fig06_six_page_experience_coverage.pdf)

## 3. Check the supporting synthetic experiment

```sh
python tools/reproduce.py synthetic
```

This verifies all 1,114 raw observations and refits M1-M5. Expected energy errors are 14.974%, 12.284%, 2.204%, 0.342%, and 2.344%. The target is a prescribed synthetic power reference with known terms. It does not validate a physical power law. All 21 test trials are included regardless of mechanical success; standing controls are separate.

- [Design and splits](../synthetic_webots/runs/full_v2/design.json)
- [Saved model summary](../paper_results/synthetic_webots/model_summary.csv)
- [Synthetic experiment directory](../synthetic_webots/README.md)

## 4. Check motion learning and real measurements

```sh
python tools/reproduce.py rl
python tools/reproduce.py replay
```

`rl` checks 24 policies and 240 saved evaluations, all predictor integrals, OOD flags, and paired statistics. Each policy has 100,352 training steps. Six training seeds, rather than the ten evaluations per policy, are the independent units. R2 is the common energy scale across conditions; three comparisons with Task-only use Holm correction. A significant energy comparison is distinct from meeting the combined performance/applicability criterion, which was not met.

`replay` verifies the returned measurement archive and reintegrates all 24 physical recordings. Expected condition means are 103.91 J (Task-only), 83.54 J (R1), 82.73 J (R2), and 82.14 J (R3). Each policy was physically replayed once. The saved commands were executed by the position controller; PPO did not make new hardware decisions. Distance and arrival time were not measured on hardware.

- [Simulation design](../motion_learning/logs/distance_goal_20261011/design.json)
- [Simulation outcomes and limitations](../motion_learning/docs/distance_goal_20261011_RESULTS.md)
- [Hardware recording guide](../motion_learning/logs/real_results_20261011_import/KHR3HV_Real_Results_Seeds10-15_20261011/README.md)
- [Measured replay summaries](../paper_results/rl_application/summary.csv)
- [Figure 10: predicted and measured replay energy](../paper_figures/fig10_six_page_rl_replay_energy.pdf)

The archived initial 12- and 72-trial plans are historical planning documents. The paper uses the **24 actual single-replay recordings**, not those proposed counts. The old returned README's mention of MuJoCo is a historical naming error: the replay analysis checks each command sequence against the retained Webots export. The original record is preserved and the clarification is documented in [archive notes](ARCHIVE_NOTES.md).

## 5. Inspect provenance and what was actually rerun

[verification/reproduction_results.json](../verification/reproduction_results.json) records the original seven successful offline actions. [verification/repository_review.json](../verification/repository_review.json) records the current English/documentation review and executed follow-up checks. No fresh robot trials, simulator rollouts, or PPO training are implied by these audits.

English documentation is the reading path. Byte-exact raw records, frozen source, and vendor notices may retain Japanese text; translated labels and guides are linked in [archive notes](ARCHIVE_NOTES.md). Do not rename raw files or normalize their line endings: original manifests check their bytes.
