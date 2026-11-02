# Physical ridge-regression validation results

**Model selected by validation power RMSE: R2.** This is the historical fitting-stage report, not the nine-motion test summary.

Training uses 116 trials and validation uses 23, each with 81 samples. Trials have equal weight. The target is `power_W`. Coefficients and RMS feature scales are estimated from training only; regularization is chosen from 0.001, 0.1, 10, and 100. Training and validation are not pooled for a final refit.

| Model | Alpha | Validation power RMSE (W) | Mean validation energy error (%) |
|---|---:|---:|---:|
| R0 | 0.001 | 0.7262 | 7.875 |
| R1 | 0.001 | 0.2589 | 2.474 |
| R2 | 0.001 | 0.2516 | 2.117 |
| R3 | 10 | 0.2629 | 2.234 |

R0 is constant power. R1 uses absolute and squared angular velocities and tracking errors for 17 joints. R2 adds error products for selected body-linked joint pairs. R3 adds accelerometer-derived inclination and support categories. Joint positions enter through errors and velocities; voltage, current, and power are not predictor inputs.

CSV `dq`/`e` and analysis features were checked across 139 trials. The centered five-point smoother is an offline estimator. Accelerometer inclination is not a guarantee of dynamic orientation accuracy. Recorded support labels and pending mechanical-success fields are retained, not automatically converted to success.

At the time of this fitting-stage record, independent whole-body testing was still planned; the release now contains that nine-motion test separately. Effects of supply, temperature, and recording time were not isolated. For portable reproduction use `python tools/reproduce.py hardware` from the repository root; the retained `run_ridge_regression.py` is the original session-specific script.
