# Data and analysis scope

## Physical prediction

`hardware_prediction/hardware_experiment/regression_results/ridge_20260915_143619/trials.csv` is the authoritative 116/23 train/validation split. Its `aligned/` files are the exact 10 Hz fitting inputs. `models/frozen.json` records feature schema, model hashes, chosen hyperparameters, joint ordering, and per-input hashes. Acquisition raw files are retained under the corresponding `live_batches/*/raw/` folders. The portable fitting check starts from the aligned training inputs; it does not claim to re-execute every acquisition/alignment step from the original device streams.

The current held-out set is [repeated_evaluation_20261109](../hardware_prediction/repeated_evaluation_20261109/README.md): nine motion types, five executions each, with 45 raw captures listed in its export manifest. It contains Bow instead of the historical Supine get-up motion. Original nine-recording results under `manuscript/data/` and `session_v3_test_10hz/raw/` remain historical and are not pooled with the new set. Acquisition metadata and original archive hashes remain unchanged.

R1 has 69 features: intercept and joint speed/tracking-error terms. R2 adds 20 absolute tracking-error products (89). R3 adds inclination and assigned support context (95). The original R0-R3 parameters are unchanged. Train/validation/test recordings are never pooled for model selection. Power traces use the original 10 Hz alignment, five-sample centered position smoothing, differentiation and support labels. Whole-movement energy uses trapezoidal integration.

Five execution errors are averaged per motion, then nine motion means are weighted equally. Figure 4 SD is within-motion (n=5); Figure 5 SD is between-motion (n=9). Neither is a confidence interval. The two waveform examples are the minimum-R3-RMSE repetitions within Clap and Bow; all 45 executions contribute to reported aggregate results.

Feature/context and 20-seed experience-coverage analyses are retrospective. They retain original subset identities and select hyperparameters on the original validation recordings, then score all 45 current executions. Integration windows of 2, 5 and 10 seconds are retrospective: complete nonoverlapping windows are primary, incomplete tails are excluded, and errors are averaged within execution, within motion, then across motions. Overlapping 0.1 s-stride windows are secondary; they do not constitute additional independent repeats.

## Synthetic supporting experiment

`synthetic_webots/runs/revision_02/design.json` defines the adopted 1,114 trials; the original `full_v2` remains historical. The 1,093 training/validation/control recordings are reused byte-for-byte; the 21 tests were newly simulated after command revisions and meet sustained mechanical criteria. Known evaluation seeds were reused. The test split contains 21 nonstatic command executions, including all mechanical failures; three standing controls are separate. Webots R2025a supplies motion/contact observations. The electrical reference is synthetic and contains terms directly available to the feature hierarchy. Physical R0–R3 and synthetic M1–M5 are independently fitted models.

Collection history includes a reconstruction of an earlier simulation design. The two CSV catalogs in `reference/` were inputs to that design and are retained because `src/protocol.py` uses them. The obsolete comparison manuscript and its figure-provenance file are omitted; their absence is explicitly recorded in the portable synthetic audit. No older simulation result is substituted for the Webots results in this package. The original source snapshot and historical metadata remain distinguishable from the portable analysis adapter.

## Motion learning and physical replay

The archived distance experiment is `motion_learning/logs/distance_goal_20261011/`. Policies stop on reaching 0.2 m, with a 40 s deadline and gamma = 1. This is the earlier distance experiment used in the paper, not the subsequent foot-clearance study. It does not guarantee lifted-foot walking.

All 24 final models and 240 evaluation traces are retained, with saved configurations, original completion/audit records and source hashes. Intermediate checkpoints, TensorBoard files, GUI movies and temporary simulator instances are excluded. Independent statistical units are six training seeds; three Task-only comparisons use Holm correction. OOD thresholds and failed performance gates are not altered.

`motion_learning/logs/real_results_20261011_import/` contains the original returned 24-recording archive in extracted form, including command acknowledgments, configs, plans and hashes. The analysis verifies identity against the exported commands and simulated seed-20000 episodes, then recomputes measured energy from voltage times current over the replay interval. The original fixed order, commissioning flag, provisional joint mapping and one replay per policy are retained. No physical distance or arrival-time measurement is inferred from the simulated value.

## Deliberate exclusions

Old drafts, Frontiers templates/logos and branded manuscript PDFs; advisor correspondence; Desktop recovery files; full video/audio collections; the removed four-condition diagnostic study; older 20-second models; later foot-lift experiments; unrelated exploratory robot programs; Python environments and compiled firmware are not part of this upload folder. Originals remain in the research archive. Existing archive paths in metadata are historical strings, not runtime dependencies of `tools/reproduce.py`.
