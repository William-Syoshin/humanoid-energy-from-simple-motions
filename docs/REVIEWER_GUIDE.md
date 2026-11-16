# Reviewer guide

This guide follows the manuscript's evidence: physical prediction, supporting simulation, then motion learning and physical command replay. All commands below run from the repository root. Install [the analysis requirements](../requirements-analysis.txt) first, as described in the [README](../README.md).

## 1. Check the central prediction result

```sh
python tools/reproduce.py verify
python tools/reproduce.py hardware
```

The hardware command verifies 116 training and 23 validation inputs, checks the frozen-model coefficients, reconstructs all 45 test traces from recorded device streams, and compares the saved metrics. Its report must contain `passed: true`, `physical_recordings: 45`, `held_out_motions: 9`, and `repeats_per_motion: 5`.

| Model | Mean absolute relative energy error |
|---|---:|
| R0 | 9.41% |
| R1 | 8.67% |
| R2 | 4.06% |
| R3 | 3.14% |

Each motion is measured five times. Errors are averaged first within motions, then equally across nine motion means. Model selection still uses the original validation power RMSE, which selects R2; lower R3 test energy error does not retrospectively change that selection. None of the test data are used to retrain the primary models.

- [Train/validation split](../hardware_prediction/hardware_experiment/regression_results/ridge_20260915_143619/trials.csv)
- [Frozen model metadata](../hardware_prediction/hardware_experiment/regression_results/ridge_20260915_143619/models/frozen.json)
- [45 raw recording identifiers](../hardware_prediction/repeated_evaluation_20261109/source/export_manifest.json)
- [Current per-execution metrics](../hardware_prediction/repeated_evaluation_20261109/verified_data/per_trial_verified.csv)
- [Figure and table map](FIGURE_MAP.md)

Figure 4 error bars describe within-motion repeatability (n=5); Figure 5 describes variation across nine motion means. Figure 6 contains favorable illustrative Clap/Bow executions chosen by the lowest R3 power RMSE among five, with this selection disclosed in the caption. Small integrated error alone does not imply accurate instantaneous peaks.

## 2. Check which information and experiences help

```sh
python tools/reproduce.py coverage
python tools/reproduce.py windows
```

`coverage` retains the original training-subset identities and validation-only hyperparameter selection, but evaluates the exploratory models on all 45 current recordings. `windows` repeats the 2/5/10-second integrations on all 45 traces. These are retrospective analyses, not independent prospective validation.

- [Current analysis tables](../hardware_prediction/repeated_evaluation_20261109/analysis/)
- [Definitions, waveform selection and window weighting](../hardware_prediction/repeated_evaluation_20261109/README.md)

## 3. Check the supporting synthetic experiment

```sh
python tools/reproduce.py synthetic
```

This verifies all 1,114 raw observations and refits M1-M5. Expected revised energy errors are 14.551%, 11.014%, 2.299%, 0.234%, and 1.626%. The sustained mechanical audit passes 21/21 fixed-condition tests; original and intermediate outcomes are preserved. The target is a prescribed synthetic power reference with known terms. It does not validate a physical power law. All 21 test trials are included regardless of mechanical success; standing controls are separate.

- [Design and splits](../synthetic_webots/runs/revision_02/design.json)
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
- [Figure 11: predicted and measured replay energy](../paper_figures/fig11_six_page_rl_replay_energy.pdf)

The archived initial 12- and 72-trial plans are historical planning documents. The paper uses the **24 actual single-replay recordings**, not those proposed counts. The old returned README's mention of MuJoCo is a historical naming error: the replay analysis checks each command sequence against the retained Webots export. The original record is preserved and the clarification is documented in [archive notes](ARCHIVE_NOTES.md).

## 5. Inspect provenance and what was actually rerun

[verification/reproduction_results.json](../verification/reproduction_results.json) records the original seven successful offline actions. [verification/repository_review.json](../verification/repository_review.json) records the earlier English/documentation review. The latest local update is documented in [the update record](UPDATE_45_TRIALS.md) and its linked verification report. No fresh robot trials, simulator rollouts, or PPO training are implied by these audits.

English documentation is the reading path. Byte-exact raw records, frozen source, and vendor notices may retain Japanese text; translated labels and guides are linked in [archive notes](ARCHIVE_NOTES.md). Do not rename raw files or normalize their line endings: original manifests check their bytes.
