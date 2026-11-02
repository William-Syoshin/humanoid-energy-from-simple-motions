# Current manuscript figure map

| Figure | Role | Data/source and reproduction |
|---|---|---|
| 1 | Method overview: simple experiences, R1–R3, whole-body prediction | Final author diagram in `paper_figures/fig01_*`; feature definitions in frozen hardware analysis and config |
| 2 | Hardware and measurement arrangement | Final author figure in `paper_figures/fig02_*`; hardware acquisition code/configs |
| 3 | Acquisition postures | Final photographic catalog of upright, squat, supine, side-lying and hand-support postures; training/validation catalogue |
| 4 | Energy errors for nine motions | `hardware` reconstructs each value; `figures` rebuilds the nine panels |
| 5 | Crawl and supine-get-up power/residual traces | Raw test recordings and `*_predictions.csv`; `hardware` and `figures` |
| 6 | Training-experience coverage | `coverage` recomputes fixed subsets; `figures` rebuilds numeric comparisons |
| 7 | Synthetic M1–M5 comparison | `synthetic` refits from all raw trials; `figures` rebuilds the bars |
| 8 | Simulation scenes | Final scene images from the Webots supporting experiment |
| 9 | Energy-aware learning / saved-command replay workflow | Final author diagram; frozen RL source, exported commands and recording protocol |
| 10 | Simulated and measured replay energy | `rl` audits simulation; `replay` audits real measurements; `figures` plots mean ± sample SD across six seeds |

The unaccepted expanded Figure 10 preview is not included as a paper figure. Duration and mean power remain available in the numerical summaries for interpreting energy. Additional regenerated plots are analysis artifacts, not silently substituted manuscript figures.

Table 1 (training/validation catalogue) corresponds to `hardware_prediction/hardware_experiment/regression_results/ridge_20260915_143619/trials.csv` and acquisition metadata. Table 2 (common-alpha feature comparison) is computed from `feature_regularization_metrics.csv` at alpha = 10 and recomputed by `coverage`. Table 3 (PPO setup) corresponds to the fixed `motion_learning/logs/distance_goal_20261011/design.json` configuration. The repository does not contain a substitute manuscript or renumbered paper layout.
