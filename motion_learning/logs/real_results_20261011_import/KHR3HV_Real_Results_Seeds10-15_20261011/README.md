# Returned hardware measurements — 11 October 2026

This English guide describes the **24 recorded trials used in the paper**. The original Japanese records remain byte-for-byte intact for `SHA256_MANIFEST.json`. English companions are additional documentation, not altered acquisition records.

Open [RESULTS_EN.html](RESULTS_EN.html) for the English results table, or [RESULTS.json](RESULTS.json) for the unified 24-recording list. From the repository root, `python tools/reproduce.py replay` verifies this archive, checks the exported commands against the saved Webots source and reintegrates the measurements. Alternatively, run `python verify_results.py` here (standard library only). Neither command connects to a robot.

## Design and provenance

Training seeds 10–15 × R1/R2/R3/Baseline × one physical replay = **24 recordings**. Baseline means Task-only, with lambda = 0; the other conditions use lambda = 0.3. These names identify the motion-learning reward, not the accuracy of a predictor fitted to the measured replay power.

Saved joint-position commands are replayed at 50 Hz and voltage/current sampled at about 10 Hz. The order is fixed: seeds 10 through 15, with R1, R3, R2 and Baseline within each seed. Six seeds are six learned policies per condition, not six repetitions of one policy. The 12/72-trial plans retained in `experiment/` were earlier proposals.

**Historical naming correction:** the original Japanese README calls the command source MuJoCo. The retained source and command-identity check establish **Webots** provenance. The original wording is preserved as a record, not adopted as the current description.

## Contents

- `RESULTS.json`: integrated energy, mean power, duration and relative raw paths for all 24 recordings.
- `RESULTS.html`: original result page; use this guide and the repository summaries for English interpretation.
- `verify_results.py`, `SHA256_MANIFEST.json`: original integrity and reintegration check.
- `experiment/commissioning_raw/`: actual 24 recordings.
- `experiment/calibration/`: separate start-pose/position checks and return operations.
- `experiment/data/`: 24 command sequences and manifests.
- `experiment/provenance/`: configurations and material supplied with the transfer package.
- `experiment/*.py`, `hardware*.json`, `plan*.json`: code, settings and plans actually retained from acquisition.
- `experiment/original_code/`: earlier code versions; `archive/`: pre-measurement previews, not these final measurements.

Within a raw trial, `power.csv` contains PC/device times, voltage, current and clipping flags. `commands.csv` records transmissions, acknowledgments and delays; the commanded angles are in `motion.json`. `initial_feedback.csv` measures joint positions only before the run, not continuously during motion. `config.json` is the actual trial configuration; `meta.json`/`trial.json` retain conditions, intervals and identities. Use the unified `RESULTS.json` where early seed-10 summaries lack seed columns. Operator confirmations record only statements actually supplied.

Historic absolute Windows paths are retained as metadata. Use `raw_directory` in `RESULTS.json` for portable relative paths.

## Measurement calculation

For every sample, power is `voltage_V * current_A`. Integrate over `score_start` to `score_end` on the PC monotonic clock, interpolating linearly at interval boundaries and using the trapezoidal rule. Mean power is energy divided by duration. Initial/final logging and return-to-start operations are excluded; idle power is not subtracted.

Checks cover time monotonicity, interval coverage, clipping and sample gaps. Existing voltage/current calibration was used; no new calibration or firmware change is claimed. The receive-time correction is zero and the configured timing uncertainty is 0.101 s.

## Unmeasured quantities and limitations

Physical displacement and first-arrival time were **not measured**. A `target_distance_m` value of 0.2 refers to simulation, not evidence of physical arrival. Durations differ, so replay energy cannot establish energy efficiency over a common distance. Falls, contacts and manual support were not automatically classified; completed communication does not establish task success.

The operator confirmed forward motion for R1, seed 10. Other recordings have no additional individual success confirmation. `mapping_verified=false` and `commissioning=true` are retained. Six leg-pitch directions and start poses were checked, but an exact all-axis simulator-to-servo mapping was not established. Continuous hardware joint-angle and IMU feedback were not recorded during these replays.

Monitoring limits changed during commissioning; each trial's configuration and code history are retained. The command sequences were not clipped, slowed or reshaped. All 24 acquisitions are marked complete and their transmission counts, power records and integrals were checked. This does not manufacture missing physical outcomes.

The archived initial handoff documentation and `VALIDATION.json` describe an earlier preparation stage. The present guide, final measurements and offline checks describe the evidence used in the manuscript. Original Japanese confirmation statements remain original records; see the repository's archive notes for an English reading guide.
