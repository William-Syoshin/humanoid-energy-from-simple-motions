# Variable-duration command replay on Windows

This is an English guide to the original transfer package. It documents the prepared software and original plans, not a newly certified hardware deployment. The later paper data consist of **24 actual recordings**, one for each of six seeds and four conditions. The initial 12- and 72-trial plans below are historical proposals, not the completed experiment.

## What is replayed

The package exports 24 motions: Task-only, R1, R2 and R3 for training seeds 10–15, with evaluation seed 20000. Task-only uses lambda = 0 and the others use 0.3. Each contains 17 joint-position commands at 50 Hz reconstructed from the frozen gait generator and saved policy actions, including its delays. These are commanded positions, not interpolated measured joint angles. The simulation source is Webots.

The hardware controller replays saved commands. PPO does not make new hardware decisions. The servo position loop remains active; there is no automatic displacement measurement or stop at 20 cm. The final target is held: this package does not include an automatic return to neutral or motor power-off. Seed-10 durations are 12.1 s (Task-only), 9.4 s (R1), and 9.5 s (R2 and R3).

## Files and plans

- `data/motions/`: 24 exported command sequences and source identities.
- `plan_pilot.json`: initially proposed 12 trials (four conditions, three repeats, seed 10), randomized within repeat.
- `plan_all_seeds.json`: initially proposed 72 trials (four conditions, six seeds, three repeats).
- `experiment.py`, `capture.py`, `common.py`, `analyze.py`, `export.py`: execution, logging, checks and analysis.
- `VALIDATION.json`: original export and dry-run verification; it records that Windows/hardware had not been tested **at package creation**.

The returned archive preserves the settings actually used later, including provisional mapping and commissioning mode. Its current English README, not an initial plan, defines the 24 recordings used in the paper.

## Environment and calibration

From this directory on the intended Windows machine:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
.\.venv\Scripts\python.exe experiment.py config --out hardware.json
.\.venv\Scripts\python.exe -m serial.tools.list_ports
```

Do not treat saved COM ports or joint mapping as a universal configuration. Establish the 17 servo IDs, SIO lines, zero positions, directions, angle units, limits and maximum speeds for the actual robot. The mapping is `servo = zero + sign * simulation_radians * units_per_radian`. Do not assume that simulator and servo signs agree. Confirm mixing is disabled and verify measurement calibration and timing before setting the corresponding verification flags.

The robot and power device use separate ports; close any competing GUI connection. The legacy receive path may repeat old measurements. Use a fresh-device mode only with firmware verified to provide fresh samples. Do not assume zero latency. Confirm the 50 Hz command timing and one-frame servo command support. The executor rejects limit violations rather than silently clipping commands; missed deadlines or acknowledgment errors stop the sequence rather than sending a late burst.

## Execution behavior

A run without `--execute` is a dry run: no serial ports are opened and no measurement output is created. For example, after selecting an actual trial ID from the plan:

```powershell
.\.venv\Scripts\python.exe experiment.py run --trial $trial --out "raw\$trial"
```

Actual execution requires the verified configuration and the explicit `--execute` option. Check `--help` for configuration/plan arguments before acquiring new data. This README does not authorize unattended hardware execution.

Initial measured positions are compared with `initial_pose_rad`; excessive initial error is rejected. The program does not automatically align the start pose. Command acknowledgments are not proof of joint tracking, and continuous measured joint angles are not recorded during replay. There is no automatic fall monitor. Ctrl+C stops the program but is not equivalent to motor power-off.

Power logging includes a two-second initial hold and one second after replay, but scored energy uses only the replay interval. Keep failed or interrupted records. Record floor/support, battery voltage, temperature, rest periods and changes in setup. Do not overwrite earlier trials or modify command timing/amplitude to improve a failed result without labeling it a new experiment.

## Distance and interpretation

For a new equal-distance experiment, independently measure displacement and first arrival at 20 cm, with synchronized timing and uncertainty. Use a calibrated body reference, not foot movement or inferred average speed. An unknown arrival time remains unknown. Annotate success, shortfall, fall, deviation, operator stop or other failure; do not assign successful arrival from command completion. Post-stop inertia is not automatically part of a scored arrival.

The actual 24 paper recordings have **no physical displacement or first-arrival measurement**. Their replay-energy differences cannot establish equal-distance locomotion efficiency.

## Analysis

```powershell
.\.venv\Scripts\python.exe experiment.py analyze --raw raw --out figures_R2 --sim-evaluator R2
```

Measured power is voltage times current; energy is trapezoidal integration over interpolated replay boundaries, without idle-power subtraction. Use a common R2 simulation evaluator across training conditions. Goal-energy analysis requires independently measured goal times, with failures retained separately. Missing records must remain marked incomplete.

The planned three physical repeats per policy would quantify within-policy variation, not three independent learned policies. With multiple training seeds, average physical repeats within seed before comparing seeds. The completed paper archive has one physical replay per policy, so repeatability was not estimated. Exported analyses include trial summaries, seed means, plotting summaries and paired seed effects; statistical significance is not inferred automatically from a visual gap.

The original package passed nine software tests and four-condition dry runs on macOS. Those are software/export checks, not new Windows or physical validation. Mock/synthetic logs are marked as such and must not be presented as physical measurements. For sampling considerations, see [Lakens (2022)](https://doi.org/10.1525/collabra.33267).

Keep vendor notices and original frozen files unchanged. Review the repository-level offline reproduction guide before running acquisition code; reviewers need only the offline tools.
