# Windows handoff

Read [README.md](README.md) before acquisition. The repository's reviewer commands are offline and do not require a robot.

1. Install the recorded Python dependencies with `setup_windows.ps1`; list local serial ports.
2. Generate a local hardware configuration. Establish and verify servo mapping, signs, limits, disabled mixing, power calibration and timing. Do not reuse another robot's flags as evidence.
3. Inspect the selected command file, start pose and trial plan. The original pilot/all-seed plans proposed 12/72 recordings; the paper uses 24 actual single replays.
4. Perform dry runs without `--execute`; these do not open ports. Actual replay requires the explicit execution option and verified setup. Final targets remain held; there is no automatic distance stop or motor power-off.
5. Save every attempt in a new raw directory. Measure distance and first-arrival time independently if making an equal-distance claim. The archived paper trials did not measure these quantities.
6. Run the analysis with a common simulation evaluator (R2) and retain failures, timing uncertainty and incomplete records. See the returned archive's English README for the completed measurements.

The original `VALIDATION.json` records software/export checks at handoff time, not a complete hardware validation. New acquisition requires the actual machine's calibration and operator supervision.
