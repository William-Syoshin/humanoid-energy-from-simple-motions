# Simulation results supplied with the transfer package

The archived experiment is `distance_goal_20261011`: four conditions, six training seeds, 100,352 steps per policy, ten evaluations per policy (240 total). It uses Webots, a 20 cm arrival target, 40 s deadline and gamma = 1. These are not the older fixed-20-second policies.

All 240 simulations arrived. Mean common-R2 predicted energies were 106.975 J (Task-only), 82.203 J (R1), 81.316 J (R2), and 79.989 J (R3). All six seed differences favored each energy-aware condition; the three Holm-adjusted p-values were 0.046875. Nevertheless, the combined primary criterion **failed**, because some episodes exceeded the 10% OOD limit. Mean arrival times were 12.167, 9.483, 9.417, and 9.383 s, respectively.

Shorter duration accounts for much of the predicted energy difference. Predictions do not establish physical energy savings. R1–R3 share training data; their agreement is not independent physical validation. Statistical comparisons were with Task-only, not superiority tests between R1–R3.

See the [complete English simulation report](../../docs/distance_goal_20261011_RESULTS.md), including all three predictor scales, OOD counts, timing, GUI checks and limitations. Physical replay evidence is reported separately.
