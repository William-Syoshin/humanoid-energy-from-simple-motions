# Distance-goal results: Task-only, R1, R2, and R3

**All 240 simulated evaluations reached 20 cm.** Each energy-aware condition had lower predicted total energy than Task-only in all six training seeds. On the common R2 scale, mean energy decreased by approximately 23–25%.

**The combined primary criterion was not met**, because some episodes exceeded the prespecified 10% OOD limit. Statistical differences in predicted energy do not establish robust arrival within the predictor's applicability range or physical energy savings.

Training began on 11 October 2026 at 00:36 JST and training/evaluation ended at 05:07 (about 4 h 31 min). Automated audit/aggregation finished at 05:17, followed by GUI checks. All 24 policies completed 100,352 steps. There were six training seeds and ten deterministic evaluations per policy. Policies stop on arrival at the first 10 Hz evaluation boundary, with a 40 s deadline and no speed-tracking reward.

## Outcomes

Values are mean ± sample SD across **six training seeds**; ten evaluations are averaged within each seed. Energy/time values are conditional on successful arrival. Here all episodes succeeded; failed low-energy episodes would not count as energy-efficient success.

| Condition | Success fraction | Time (s) | R1 energy (J) | R2 energy (J) | R3 energy (J) |
|---|---:|---:|---:|---:|---:|
| Task-only | 1.000 ± 0.000 | 12.167 ± 1.439 | 108.271 ± 11.242 | 106.975 ± 11.059 | 105.202 ± 11.084 |
| R1 | 1.000 ± 0.000 | 9.483 ± 0.214 | 83.856 ± 2.405 | 82.203 ± 3.361 | 80.437 ± 3.269 |
| R2 | 1.000 ± 0.000 | 9.417 ± 0.117 | 83.500 ± 1.869 | 81.316 ± 2.587 | 79.543 ± 2.474 |
| R3 | 1.000 ± 0.000 | 9.383 ± 0.075 | 82.775 ± 1.077 | 79.989 ± 1.684 | 78.270 ± 1.587 |

| Comparison with Task-only | Paired R2 difference (J) | Holm p, three comparisons | Seeds with lower energy | Combined criterion |
|---|---:|---:|---:|---|
| R1 | -24.771 | 0.04688 | 6/6 | Not met |
| R2 | -25.659 | 0.04688 | 6/6 | Not met |
| R3 | -26.986 | 0.04688 | 6/6 | Not met |

The exact adjusted p is 0.046875. The primary tests compare each condition with Task-only, not R1 with R2 with R3. Rankings of sample means do not establish superiority among the three models. These are fixed-budget PPO outcomes, not proof that Task-only reached its optimal policy. All predictors share physical training data; agreement between them is not independent hardware validation.

## Why the combined criterion failed

All success and paired-success rates were 100%. There were no falls, escapes, timeouts or negative predictions. Maximum overshoot was 2.52 mm, below 5 mm. The failure was the **per-episode OOD criterion**, not a condition-average criterion. Thresholds were not relaxed after seeing results.

| Condition | Mean OOD ± seed SD | Maximum OOD | Training seeds above 10% | Episodes above 10% |
|---|---:|---:|---|---:|
| Task-only | 11.70 ± 3.21% | 16.15% | 11, 14, 15 | 30/60 |
| R1 | 5.79 ± 3.83% | 12.11% | 13 | 10/60 |
| R2 | 5.45 ± 3.51% | 10.94% | 14 | 10/60 |
| R3 | 6.23 ± 2.14% | 10.22% | 11 | 10/60 |

OOD flags extrapolation; it neither proves that every flagged prediction is wrong nor makes low average OOD a validation of accuracy. The other combined requirements are specified in the [protocol](DISTANCE_GOAL_EXPERIMENT.md).

## Total energy versus mean power

| Condition | R2 energy reduction, ratio of condition means | Mean of episode E/T across seeds |
|---|---:|---:|
| Task-only | Reference | 8.812 W |
| R1 | 23.16% | 8.667 W |
| R2 | 23.99% | 8.636 W |
| R3 | 25.23% | 8.526 W |

A 23–25% decrease in predicted **energy** is not a 23–25% decrease in **power**. Mean predicted power differs by about 1.6–3.2%, while mean arrival time decreases from 12.17 s to about 9.4 s. Shorter arrival time accounts for much of the energy difference. The experiment does not compare electrical power at a common walking speed.

## Audit and GUI comparison

The original audit checked all 24 policies and 240 episodes, frozen source, conditions, training steps, terminal states and reintegrated predictions. Exported Task-only and R2 models for the preselected training seed 10 were reevaluated in the GUI with evaluation seed 20000, saved configurations and frozen source. All episode fields matched the saved results exactly except the policy-file path. Export hashes, embedded 100,352-step counters and all 240 CSV rows were checked.

- [Analysis](../logs/distance_goal_20261011/analysis.json)
- [All episodes](../logs/distance_goal_20261011/all_episodes.csv)
- [Seed summaries](../logs/distance_goal_20261011/seed_summary.csv)
- [Frozen design](../logs/distance_goal_20261011/design.json)
- [Original offline audit](../logs/distance_goal_20261011/audit.json)

Run `python tools/reproduce.py rl` from the repository root for offline reanalysis. This packaging check does not repeat GUI evaluations or train new policies.

This report concerns simulation. The **later** 24 hardware command replays are described in the [motion-learning guide](../README.md). Physical displacement and arrival time were not measured. Do not interpret this simulation report as physical validation or substitute the old fixed-duration Windows package for the variable-duration replay.
