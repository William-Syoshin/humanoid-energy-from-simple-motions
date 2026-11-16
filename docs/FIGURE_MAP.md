# Current manuscript figure map (2026-11-10)

This map describes the eleven-figure manuscript snapshot of 2026-11-10. It does not describe the later twelve-figure Overleaf layout. Physical results use all 45 recordings. The two simulation figures have been updated to revision_02; the later Bow/Kick posture-error illustration is not bundled.

| Figure | Role | Current source / reproduction |
|---|---|---|
| 1 | Simple experiences and R1-R3 prediction overview | [PDF](../paper_figures/fig01_fig1_concept.pdf); frozen model definitions |
| 2 | Robot, series shunt and measurement/control signals | [PDF](../paper_figures/fig02_fig2_hardware.pdf); acquisition code/configs |
| 3 | Five acquisition postures | [PDF](../paper_figures/fig03_fig3_catalog.pdf); training/validation catalogue |
| 4 | Nine motions, mean error and within-motion SD over five executions | [PDF](../paper_figures/fig04_six_page_hardware_trial_errors.pdf); `hardware`, `figures` |
| 5 | Overall energy error and SD across nine motion means | [PDF](../paper_figures/fig05_hardware_energy_error_mean_sd.pdf); `hardware`, `figures` |
| 6 | Clap/Bow measured/predicted power and cumulative residual | [PDF](../paper_figures/fig06_six_page_hardware_traces.pdf); repeat 3 of each, minimum R3 RMSE; `hardware`, `figures` |
| 7 | Random vs coverage-based training subsets, R1/R2/R3 | [PDF](../paper_figures/fig07_experience_coverage_45_executions.pdf); current 45-execution evaluation, `coverage`, `figures` |
| 8 | M1-M5 synthetic-reference error and SD across 21 test executions | [PDF](../paper_figures/fig08_models.pdf); separate synthetic data, `synthetic` |
| 9 | Five-frame simulation motion sequences | [PDF](../paper_figures/fig09_sim_webots_scenes.pdf); saved supporting-simulation observations |
| 10 | PPO training and saved-command replay workflow | [PDF](../paper_figures/fig10_rl_learning_workflow_compact.pdf); frozen RL source and export/replay protocol |
| 11 | Predicted and measured learned-command replay energy | [PDF](../paper_figures/fig11_six_page_rl_replay_energy.pdf); six seeds per condition, `rl` and `replay` |

Table 1 is the unchanged 116/23 training/validation catalogue. Table 2 is the common-alpha context comparison in [body_context_summary.csv](../hardware_prediction/repeated_evaluation_20261109/analysis/body_context_summary.csv), evaluated on 45 recordings. Table 3 is the unchanged PPO setup in the distance-goal design.

`figures` rebuilds current numeric panels 4-7 without replacing publication artwork. Synthetic and replay data are verified by their dedicated actions. Author-created schematics, photographs and simulation scenes are supplied as the exact current PDF assets. Historical artwork and build scripts are explicitly separated; they are not current manuscript defaults.
