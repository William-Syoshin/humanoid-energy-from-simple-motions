# Supporting simulation: reference, settings, and model definitions

This document specifies the controlled synthetic-power experiment used in the manuscript. Its models M1-M5 are independent of the physical R0-R3 predictors. The target is an explicitly constructed reference, not a calibrated electrical model of the robot. This lets the analysis test recovery of known relationships between local joint, joint-pair, body, and contact information and power.

## Reproduce the recorded experiment

Install the analysis dependencies and run, from the repository root:

```sh
python tools/reproduce.py synthetic
```

The command verifies the recorded `revision_02` inputs and reproduces the fits and evaluations without launching Webots. For the output files and optional fresh-simulation procedure, see the [revision_02 guide](../synthetic_webots/runs/revision_02/README.md). The current experiment is `revision_02`; `full_v2` is a historical record.

| Definition | Authoritative file |
|---|---|
| Complete trial specifications and frozen reference coefficients | [design.json](../synthetic_webots/runs/revision_02/design.json) |
| Local/pair/body/contact features and reference equation | [protocol.py](../synthetic_webots/runs/revision_02/source/src/protocol.py) |
| Joint names, motor mapping, and signs | [joint_mapping.json](../synthetic_webots/runs/revision_02/source/src/joint_mapping.json) |
| Motor settings, reset, sampling, and observation definitions | [benchmark.py](../synthetic_webots/runs/revision_02/source/controllers/benchmark/benchmark.py) |
| World and robot definition | [benchmark.wbt](../synthetic_webots/runs/revision_02/source/worlds/benchmark.wbt) |
| Contact-force accumulation and friction | [contact_logger.c](../synthetic_webots/runs/revision_02/source/plugins/physics/contact_logger/contact_logger.c) |
| Fitting, model selection, and evaluation | [analyze.py](../synthetic_webots/runs/revision_02/source/analysis/analyze.py) |
| Shortened command-window definition | [command_window_diagnostic.py](../synthetic_webots/runs/revision_02/analysis_scripts/command_window_diagnostic.py) |

## Simulation and trajectory settings

Webots R2025a simulates a 1.7 kg floating-base KHR-3HV with 17 actuated joints. Physics runs at 2 ms, and commands/observations update every 10 ms (100 Hz). Motors use position-control PID gains `(10, 0, 0)`, a velocity limit of 7.48 rad/s, and available torque limited to 1.373 N m. Self-collision is disabled. The instrumented robot-floor and robot-object contacts use friction coefficients 1.0 and 0.4, respectively; the interaction box weighs 0.1 kg. The contact plugin uses soft ERP 0.2 and soft CFM `1e-5`.

The 1,000 four-second training trials contain 40 static, 160 single-joint, and 200 trials each of pair, posture, contact-context, and multijoint excitation. Pair trials use neighboring or bilateral joints. Posture/contact trials excite two joints; multijoint trials excite 4, 8, 12, or 17 joints. The validation split has 90 separate four-second trials: 4 static, 14 single-joint, and 18 of each remaining group.

Training/validation sinusoidal recipes draw amplitude from `{0.2, 0.35}` rad, frequency from `{0.3, 0.6, 1.0}` Hz, offset from `{0, 0.15}` rad, and phase increments from `{0, pi/2, pi}`. Their commands are clipped to `[-1.4, 1.4]` rad. The exact draws and joint selections are in `design.json`. Contexts include standing, crouching, lying, wide/asymmetric stances, attempted hand support, and object contact. Initial support need not persist during a trajectory. The 40 free-space training diagnostics and four corresponding validation trials use zero gravity; all other contexts use 9.81 m/s².

Each reset is followed by one second of settling, excluded from the recording. A nominal four-second trial contains 400 samples from 0.01 to 4.00 s; the trapezoidal integration interval is consequently 3.99 s. All splits use complete trajectories.

The evaluation has seven motions, each with joint-perturbation seeds 101, 202, and 303 (independent normal perturbations with SD 0.001 rad at initialization):

| Motion | Recorded duration, including terminal holds |
|---|---:|
| Get up | 45 s |
| Lie down | 9.2 s |
| Walk | 12 s |
| Walk to stop | 6 s |
| Push a box | 14 s |
| Turn 90 degrees | 45 s |
| Turn 180 degrees | 60 s |

All 21 are included in the prediction evaluation. Three standing controls are separate because standing is represented in training. Walking, stopping, and pushing use a fixed gait generator; get-up/lie-down use stored keyframes and smooth interpolation; turns use the frozen heading-feedback controller. The evaluation recipes in `design.json` select the appropriate controller or keyframes, so the older fallback commands in `protocol.py` alone do not recreate the revised test motions. Native actuator limits are approximately ±2.356 rad; some get-up/lie-down commands exceed the training cap. Turns may also translate. These recordings test prediction under the specified fixed controllers and perturbations, not controller generalization to novel conditions. [The revision record](../synthetic_webots/runs/revision_02/README.md) preserves controller provenance and mechanical checks.

## Synthetic reference equation

For each sample, let `tau_i` be measured simulator motor torque feedback and `v_i` be joint velocity. Velocity is the difference between successive 100 Hz positions divided by 0.01 s, with the pre-recording position used for the first difference. Define

```text
L = sum_i tau_i**2
h = 1 - e_z.T @ R @ e_z
```

Here `R` is the body orientation matrix in world coordinates and `e_z` is the vertical unit vector. The synthetic power is

```text
P_star = P0
       + sum_i (a_i * abs(tau_i * v_i) + b_i * tau_i**2
                + c_i * abs(v_i) + d_i)
       + sum_(i,j in G) k_ij * abs(tau_i * tau_j)
       + g(h, F, L) + r(t)

g = 1.8*h**2 + 0.35*h*L
  + 0.025*Fn + 0.035*Ft + 0.075*Fnf + 0.05*Fo
  + 0.45*single*L

r = 0.025 * (sum_i abs(tau_i*v_i))**1.3 + 0.06*T
```

The 20 pairs in `G` consist of 12 adjacent limb-joint pairs and eight bilateral counterparts. `PAIRS` in `protocol.py` and `joint_mapping.json` specify their order and names. Contact quantities are accumulated from simulated contact reactions:

- `Fn`: sum of normal-force magnitudes at instrumented robot-floor and robot-object contacts.
- `Ft`: corresponding sum of tangential-force magnitudes.
- `Fnf`: sum of force magnitudes for floor contacts with robot body parts other than the feet.
- `Fo`: sum of force magnitudes at robot-object contacts.
- `single`: 1 when exactly one foot's total normal load exceeds 0.05 N, otherwise 0. The separate `both` feature is 1 when both feet exceed this threshold.

Each observation uses the most recent available contact packet from the 2 ms physics stepping, not a 10 ms average. Contact counts are logged but do not enter the reference or these four force summaries.

The load-history state obeys `dT/dt = (L - T)/25`, starts at zero, and is advanced by forward Euler with a 0.01 s step using the previous sample's load. It is a synthetic history variable, not measured temperature. The reference is designed so that local/pair/body/contact features capture its dominant components while the nonlinear/history residual is not explicitly included as an input to M1-M4.

The baseline is `P0 = 2.2 W`. Coefficients are drawn once using `numpy.random.default_rng(20260913)` and stored with the design:

| Coefficient | Range | Count |
|---|---|---:|
| `a_i` | 1.4-2.2 | 17 |
| `b_i` | 1.0-2.0 | 17 |
| `c_i` | 0.035-0.08 | 17 |
| `d_i` | 0.09-0.16 | 17 |
| `k_ij` | 0.15-0.65 | 20 |

All quantities use compatible SI units to yield watts. Independent Gaussian label noise with SD 0.08 W is added to training and validation using each trial's seed plus 800000. Test reference values are noise-free. The coefficient arrays and noise procedure specify the artificial target; they are not parameters identified from real electrical measurements.

## Models and fitting

| Model | Input additions | Total number of features | Regressor |
|---|---|---:|---|
| M1 | Intercept; each joint's `abs(tau*v)`, `tau**2`, and `abs(v)` | 52 | Ridge |
| M2 | M1 plus 20 `abs(tau_i*tau_j)` products | 72 | Ridge |
| M3 | M2 plus `h`, `h**2`, `h*L`, center-of-mass height, and squared base angular-speed magnitude | 77 | Ridge |
| M4 | M3 plus `Fn`, `Ft`, `Fnf`, `Fo`, `both`, `single`, and `single*L` | 84 | Ridge |
| M5 | Same inputs as M4 | 84 | Histogram-based gradient boosting |

M1-M4 divide each feature by its training RMS, floored at `1e-6`. The intercept is neither scaled nor penalized. Alpha is selected from `{1e-5, 0.001, 0.1, 10, 100}` using pooled validation power RMSE. Training uses every fourth recorded sample (25 Hz). Validation and evaluation use 100 Hz. All evaluated model predictions, including validation predictions for selection, are clipped below at zero.

M5 uses `HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15, learning_rate=0.08, early_stopping=False, random_state=20260913)`. It uses the same downsampled training inputs and full-rate test inputs as the other models, with the same nonnegative prediction clipping. It changes the regression method without adding information.

Absolute relative energy error uses trapezoidal integration over each full recorded interval and weights the 21 trial errors equally. Pooled power RMSE concatenates all recorded power errors, so longer trials receive greater weight. This differs from the motion-balanced physical summaries. Error bars over the 21 energy errors use the sample standard deviation, not a confidence interval.

M4 includes the dominant terms used to define the target. The nonlinear/history residual contributes approximately 0.592% of total evaluation energy. Consequently its low error demonstrates recovery of the specified relationship; it does not establish that ridge regression is generally superior or that the synthetic equation is a physical electrical law. The no-pair comparison refits M4 without its 20 pair columns under the same selection procedure. Saved outputs are [model_summary.csv](../synthetic_webots/runs/revision_02/analysis/model_summary.csv) and [experience_comparisons.csv](../synthetic_webots/runs/revision_02/analysis/experience_comparisons.csv).

## Sensitivity to terminal holds

The full recorded intervals are the primary evaluation. A separate diagnostic ends each interval one second after the final adjacent-sample joint-command change greater than `1e-6` rad in any joint, capped at the original endpoint. If no change is detected the full interval is retained. When the cutoff lies between samples, already-computed power series are linearly interpolated at the endpoint. Features, reference power, and predictions are calculated using the complete history before shortening the integration interval; there is no refitting or model/controller selection.

The shortened interval contains commanded movement and a fixed tail; it is not an independently measured active-motion period. M3 and M4 mean energy errors are approximately 2.108% and 0.159% in this diagnostic, while pooled power RMSE increases for all models. Complete results and the recorded rule are under [command_window/](../synthetic_webots/runs/revision_02/command_window/).
