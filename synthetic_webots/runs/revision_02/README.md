# Revised supporting simulation: revision_02

This is the supporting synthetic-power experiment adopted in the manuscript. It is separate from R0-R3 physical prediction and the PPO walking application.

## Reproduce the recorded results

From the repository root, with `requirements-analysis.txt` installed:

```sh
python tools/reproduce.py synthetic
```

No simulator is launched. This refits M1-M5 on the original training/validation recordings, compares all saved results, verifies all 1,114 raw hashes, recomputes the sustained mechanical criteria for all 21 tests, and checks the supplementary command-window analysis. Outputs go to a fresh `generated/` directory.

| Model | Mean energy error (%) | Pooled power RMSE (W) |
|---|---:|---:|
| M1 | 14.5507 | 1.73388 |
| M2 | 11.0137 | 1.27899 |
| M3 | 2.2994 | 1.19728 |
| M4 | 0.2338 | 0.05253 |
| M5 | 1.6265 | 1.41029 |

## What changed and what did not

Get-up, lie-down and turning commands were revised using mechanical development trials. The new formal evaluation contains seven actions (walk, stop, push a box, get up, lie down, turn 90 degrees, turn 180 degrees), each with the original seeds 101, 202 and 303. All 21 pass the fixed sustained criteria. The original result (9/21) remains under `../full_v2/`. Summary records of the intermediate revision (20/21, including its failed turn) remain under `historical_revision_01/`.

The 1,000 training, 90 validation and three separate control recordings are reused byte-for-byte, not newly collected. All 21 test recordings are newly simulated. The robot, physics settings, synthetic power target, feature definitions and predictor fitting/selection procedure remain unchanged. Ridge model coefficients match the original fits. Joint commands remain inside native motor limits, although some exceed the original arbitrary command cap. The known evaluation seeds were reused; 21/21 is a fixed-condition result, not a new blind-test success estimate.

## Files

- `design.json`, `evaluation_order.json`: frozen trials and persistent-session order.
- `raw/`, `status/`: all 1,114 recordings and their original checksums/specifications.
- `source/`: byte-preserved controller, world, physics-plugin source and model-fitting code.
- `analysis_scripts/`, `auxiliary_hashes.json`: frozen mechanical checks and supporting scripts.
- `motion_thresholds.json`, `revision_report/motion_audit.json`: sustained mechanical criteria and outcomes. `analysis/mechanical_checks.csv` contains only the earlier endpoint criteria.
- `reuse_manifest.json`: identity of the 1,093 reused recordings.
- `analysis/`: M1-M5 parameters, predictions' numerical summaries, subsets and model comparisons.
- `command_window/`: supplementary scoring through one second after the final significant command change; full recordings remain primary.
- `finalization.json`: historical completion record, including separate GUI checks. GUI videos are not bundled; two of four GUI demonstrations did not numerically match the formal trajectories despite passing mechanical checks.

Absolute paths in frozen metadata and old launch scripts describe the original collection environment. Use the portable offline entry point above. Development-search archives and GUI videos remain in the original research archive; they are not part of this code/data update.

## Run fresh Webots trials (optional, macOS)

The archived environment used Webots R2025a, 2 ms physics, 100 Hz command/recording, and a 1.7 kg, 17-actuator robot. Prepare a new working copy without altering the frozen source:

```sh
python tools/run_revised_simulation.py --out generated/revised_webots_preparation --prepare-only
```

Inspect the copied world and plugin. The preserved `.dylib` is platform-specific. To rebuild on macOS, set `WEBOTS_HOME` to the Webots application bundle and run `make` in the copied `source/plugins/physics/contact_logger` directory. The Makefile uses `$WEBOTS_HOME/Contents/resources/Makefile.include`. Other platforms require the corresponding plugin build/launch setup; they have not been validated here.

For a new run using the archived macOS plugin and an installed compatible Webots:

```sh
python tools/run_revised_simulation.py --out generated/revised_webots_new_run --webots /Applications/Webots.app/Contents/MacOS/webots --gui
```

Use a new output path each time. The launcher runs all 21 tests in their original order in one persistent session and writes new recordings separately. The adapter's preparation mode is checked; no new simulator run is claimed for this publication update. Platform/contact dynamics can change trajectories, so compare fresh results rather than assuming bitwise identity.

The reference power is synthetic. M4 directly includes its dominant terms. The low prediction error is not validation of real electrical consumption, and these mechanical checks do not establish hardware transfer or non-sliding walking.
