# Detailed methods and implementation map

This guide records the acquisition, preprocessing, and analysis settings behind the manuscript. It complements the short calibration/preprocessing appendix and keeps implementation details next to the code that implements them. The frozen data and trained models are unchanged.

For installation and executable offline checks, use [REPRODUCTION.md](REPRODUCTION.md). For the separate synthetic-power study, use [SIMULATION_REFERENCE.md](SIMULATION_REFERENCE.md). Paths below are relative to this repository; historical absolute paths inside acquisition records describe the collection computer.

## 1. Find the inputs for each analysis

| Analysis | Authoritative inputs | Offline command from repository root |
|---|---|---|
| Main physical R0-R3 comparison | [116/23 split](../hardware_prediction/hardware_experiment/regression_results/ridge_20260915_143619/trials.csv), [frozen model record](../hardware_prediction/hardware_experiment/regression_results/ridge_20260915_143619/models/frozen.json), [45 test recordings](../hardware_prediction/repeated_evaluation_20261109/README.md) | `python tools/reproduce.py hardware` |
| Body-context and experience-coverage comparisons | Original training/validation inputs; [recorded subset identities](../hardware_prediction/repeated_evaluation_20261109/analysis/coverage_subsets.json); same 45 tests | `python tools/reproduce.py coverage` |
| Short-interval energy errors | Frozen predictions for the 45 test executions | `python tools/reproduce.py windows` |
| Motion-learning and hardware-replay analyses | [PPO experiment](../motion_learning/README.md), archived final policies, simulated evaluations, and 24 hardware recordings | `python tools/reproduce.py rl` and `python tools/reproduce.py replay` |

These commands write new outputs under `generated/`. They do not collect new physical data or launch learning. The hardware action refits the original models as an identity check, then evaluates the frozen predictors. It starts from archived aligned training inputs; it does not claim to rerun the complete acquisition pipeline from every original sensor stream. The 45 held-out recordings are reconstructed from their recorded streams.

## 2. Physical data collection

### Signals and calibration

The KHR-3HV has 17 position-controlled joints and an RCB-4 controller. Supply voltage and a single supply-side current measurement provide the aggregate target power, `P = V I`. Joint commands, measured positions, and accelerometer channels supply model inputs. There are no measured joint torques or contact forces in this physical dataset.

Command and measured-position conversions use the same trim-corrected reference and joint signs. One controller unit corresponds to `135/4000` degrees. The actual zero positions, signs, limits, and calibration values are stored in each acquisition's `config.json`, rather than assumed from the generic template in [core.py](../hardware_prediction/hardware_experiment/core.py).

The acceleration-based inclination estimate uses upright and approximately 15-degree calibration recordings. In [orientation.py](../hardware_prediction/hardware_experiment/orientation.py), `estimate_accelerometer` defines

```text
forward_g = (ad3_zero - ad3) / ad3_per_g
right_g   = (ad4 - ad4_zero) / ad4_per_g
tilt      = asin(min(1, sqrt(forward_g**2 + right_g**2)))
```

The inclination range is 0-90 degrees. The two channels cannot distinguish upright from inverted configurations, and acceleration affects the estimate during motion. Support categories are assigned by the protocol or operator annotations; they are not measured contact states.

### Training and validation recipes

[PHYSICAL_TRIAL_SETTINGS.csv](PHYSICAL_TRIAL_SETTINGS.csv) maps all 139 fitting/validation trials to their exact stored recipe and configuration. It contains only selected metadata from the immutable acquisition records, including the recipe hash. For movement recipes, the `frames` array contains the exact commanded trajectory; use it for reconstruction rather than synthesizing a new waveform from the nominal settings below. Static holds use `center_rad` and an empty `frames` array.

| Level | Experience | Training | Validation | Nominal training settings |
|---|---|---:|---:|---|
| L0 | Upright, squat, and supine holds | 6 | 0 | Two holds per posture |
| L1 | One moving joint | 51 | 12 | Each of 17 joints at 0.25, 0.50, 0.75 Hz; 15-degree amplitude, except 10 degrees for hip roll |
| L2 | Two moving joints | 30 | 6 | Ten joint combinations; 0.50 Hz, 10-degree amplitude; relative phases 0, 90, 180 degrees |
| L3 | Postural context | 15 | 3 | Five joints, moved separately in upright, squat, and supine postures; 0.25 Hz, 10-degree amplitude |
| L4 | Support context | 8 | 2 | Four joints, moved separately with foot or hand support; 0.25 Hz, 10-degree amplitude |
| L5 | Four-joint primitives | 6 | 0 | Arm or leg combinations, three phase patterns; 0.25 Hz, 10-degree amplitude |

Joint offsets and postures avoid self-collision during acquisition. Hip-roll trials use the supine posture; hip-pitch, knee, and ankle trials use side-lying postures. The validation recipes have their own amplitudes/frequencies/phases, recorded in the linked recipe files. For example, validation recipe `V_L1_LSD_RHP_A7F0375` uses a nominal 7-degree amplitude and 0.375 Hz; the nominal training table should not be applied to every validation trajectory.

Each trial contributes an eight-second scoring interval with 81 samples at 10 Hz: 9,396 training and 1,863 validation samples. Setup and repositioning are excluded. An entire trial belongs to only one split, so temporally adjacent samples from an execution are not divided between training and validation.

### Held-out whole-body tests

There are nine motion types with five executions each. All are excluded from fitting and validation selection. Work (walking), Clap, Bow, Cheer, and Kick use the assigned `feet` label; all four crawling conditions use `ground`. These labels remain constant during an execution, including any actual contact changes.

The recorded evaluation interval includes approximately two seconds of settling. Crawling also includes its recorded return-to-start sequence. The interval boundaries and actual playback timing are retained in the [45-recording export](../hardware_prediction/repeated_evaluation_20261109/source/export_manifest.json); the test set is not reconstructed from the historical single-execution dataset. The original coefficients and selected regularization strengths remain fixed.

## 3. Signal alignment and preprocessing

The implementation is [processing.py](../hardware_prediction/hardware_experiment/processing.py); [analysis.py](../hardware_prediction/hardware_experiment/analysis.py) constructs the fitted features from the aligned files. Per-acquisition configurations and quality records preserve the timing/calibration inputs.

- Measured joint and body signals are interpolated onto the common 0.1-second grid. Commands are matched to the latest acknowledged command at each grid time.
- Supply power is computed from calibrated raw voltage and current, then averaged over the temporal cell surrounding each grid point. The end cells have half-widths. These widths equal the trapezoidal weights, preserving the measured power integral over the scoring interval. The separately interpolated `voltage_V` and `current_A` display channels are not multiplied again when an explicit `power_W` column is present.
- Joint position is smoothed with a centered five-sample mean. The first and last samples are repeated for padding. Velocity is `numpy.gradient` of the smoothed position with respect to the recorded grid time.
- Tracking error is commanded minus measured position, in radians. The five-sample smoothing applies to velocity estimation, not to the tracking-error difference.
- The centered filter and differentiation use future position samples. The motion-learning application buffers three 0.1-second samples before finalizing its energy cost, as described in Section 6 below.

## 4. Physical predictors and validation selection

For every joint, R1 includes absolute velocity, squared velocity, absolute tracking error, and squared tracking error, plus an intercept. R2 adds absolute products of errors for 20 predefined pairs. These are the 12 adjacent edges within the two five-joint leg chains and two three-joint arm chains, plus eight bilateral counterparts. Head yaw has local features but no pair. The exact ordering is `EDGES` in [analysis.py](../hardware_prediction/hardware_experiment/analysis.py).

R3 adds `h = 1 - cos(tilt)`, `h**2`, and one-hot support columns. The archived implementation has 95 columns: the 94 features shown in the manuscript plus the retained `object` support column. That column is zero in these fitting inputs and does not affect predictions. `other` is the omitted reference category in the stored encoding. These implementation details do not imply that an unseen support category has a calibrated effect.

Each feature is divided by its root-mean-square value over the training samples, with a lower bound of `1e-6`. The intercept's scale is 1 and its coefficient receives no ridge penalty. For each model, alpha is selected from `{0.001, 0.1, 10, 100}` by the lowest power RMSE across the pooled validation samples. Coefficients and scaling are fitted using training data only; there is no refit using combined training and validation data.

| Predictor | Stored dimensions | Selected alpha | Validation power RMSE (W) |
|---|---:|---:|---:|
| R0 | 1 | 0.001; intercept is unpenalized | 0.726170 |
| R1 | 69 | 0.001 | 0.258946 |
| R2 | 89 | 0.001 | 0.251581 |
| R3 | 95 | 10 | 0.262912 |

R0 predicts the training-mean power, approximately 8.709 W. Physical prediction evaluation uses total power directly: it neither subtracts a baseline nor clips negative predictions. This differs from the positive-power cost used during motion learning and from the separately specified synthetic experiment. All four physical predictors are evaluated; the stored `chosen: R2` field records the lowest validation RMSE and is used for the common predictor in the motion-learning analysis, not to discard the R1/R3 test comparisons.

Measured and predicted energy use trapezoidal integration over identical samples. Energy error is `100 * abs(predicted_J - measured_J) / measured_J`; power RMSE is computed within each execution. Results average the five executions within each motion and then the nine motion means equally. With five executions for every motion this equals the average over all 45 execution metrics. Sample standard deviations use `ddof=1`: within-motion SD uses five values, whereas between-motion SD uses the nine motion means. Neither is a confidence interval.

The waveform examples for Clap and Bow use the repetition with the smallest R3 power RMSE among each motion's five executions (repetition 3 for both). The aggregate results include all 45 executions. These example choices do not select or refit the models.

## 5. Secondary physical analyses

These are exploratory comparisons on the same held-out recordings. They do not constitute independent confirmation or identify an optimal minimum training set. The portable implementation is [analyze_repeats.py](../hardware_prediction/repeated_evaluation_20261109/scripts/analyze_repeats.py).

### Body context

R2, R2 plus inclination, R2 plus support, and full R3 are compared at the common `alpha = 10`, using all 116 training trials. This removes the regularization-strength difference from the comparison of included inputs. The main R2 result instead uses its validation-selected alpha of 0.001. The common-alpha R2 error is therefore not expected to equal the primary R2 result. The full alpha-grid records are in [body_context_summary.csv](../hardware_prediction/repeated_evaluation_20261109/analysis/body_context_summary.csv).

### Training-experience coverage

Each strategy uses 20 subset seeds. Random sampling uses budgets of 12, 24, 48, 80, and 116 complete training trials. Coverage-based sampling uses budgets of 20, 24, 48, 80, and 116. It first draws one single-joint trial for each of the 17 joints and one static hold for each of the three postures, then fills the remaining budget without replacement from the other training trials. Thus its minimum budget is 20, not 12.

The original subset identities are retained in [coverage_subsets.json](../hardware_prediction/repeated_evaluation_20261109/analysis/coverage_subsets.json). For each subset and model, feature scales are recomputed and alpha is selected using the same 23 validation trials. Test power is never used to select subsets or fit coefficients. Each resulting model is scored on all 45 recordings with equal motion weighting. The figure uses median error and interquartile range over the 20 subsets at each budget. These bands represent sensitivity to the training subset, not repeatability over physical trials. [coverage_audit.py](../hardware_prediction/manuscript/scripts/coverage_audit.py) records the original coverage-selection rule; the current portable analysis uses the saved identities rather than rerunning a new selection.

### Short integration intervals

The same frozen predictions are integrated over complete, nonoverlapping 2-, 5-, and 10-second windows starting at the beginning of each evaluation interval. Incomplete final windows are omitted. Linear interpolation provides samples exactly at window boundaries before trapezoidal integration. Errors are averaged over windows within an execution, over the five executions within a motion, and finally over the nine motions.

The primary analysis contains 628, 240, and 110 windows, covering 1,256, 1,200, and 1,100 seconds, respectively, of the approximately 1,285.3 seconds recorded. The excluded tails differ by window length. A secondary overlapping analysis uses a 0.1-second stride. Windows are segments of the same 45 recordings, not extra independent experiments. See [window_summary.csv](../hardware_prediction/repeated_evaluation_20261109/analysis/window_summary.csv) and [summary.json](../hardware_prediction/repeated_evaluation_20261109/analysis/summary.json).

## 6. Predictor use during motion learning

The fixed physical predictors R1-R3 provide costs to PPO; the synthetic M1-M5 models are not used. [power.py](../motion_learning/logs/distance_goal_20261011/source/energy/power.py) defines the feature construction and buffer. The assigned support input is always `feet` in this application. This is a constant categorical input describing the protocol, not a constraint that both feet remain fixed or continuously in contact.

Prediction coefficients and feature scales remain fixed while PPO updates the policy and value networks. Predicted power is clipped below at zero only for the learning energy cost, preventing negative predictions from being rewarded as energy production. The 0.3-second buffer accommodates the offline velocity calculation. Buffered costs are included at termination; with discount factor 1, this delay does not discount the accumulated energy term.

The simulation uses 4 ms physics, 20 ms joint targets, and 100 ms policy actions. PPO adjusts five parameters of a fixed gait generator. The four conditions use Task-only with lambda 0 and R1-R3 with lambda 0.3. Six training seeds per condition yield 24 policies, each evaluated in ten simulated episodes. The [distance protocol](../motion_learning/docs/DISTANCE_GOAL_EXPERIMENT.md) and [frozen design](../motion_learning/logs/distance_goal_20261011/design.json) contain the complete reward, task, termination, input-range, and PPO settings.

Hardware execution replays saved joint commands from evaluation seed 20000 once per policy; it does not run PPO online. The measured replay interval excludes the pre/post holds and restoration movement. The source and data for that analysis are in the [hardware replay archive](../motion_learning/logs/real_results_20261011_import/KHR3HV_Real_Results_Seeds10-15_20261011/README.md). Physical displacement was not measured, so replay-energy comparisons alone do not establish savings for matched physical travel distances.
