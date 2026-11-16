# Distance-goal experiment protocol — 11 October 2026

This is the English version of the original simulation protocol. It compares predicted energy until reaching 20 cm, using newly trained policies rather than the earlier 20-second models. Later physical recordings are documented separately in the [motion-learning guide](../README.md).

## Design

- Four conditions: Task-only (lambda = 0), and R1, R2, R3 (lambda = 0.3). Each seed has one common Task-only baseline.
- Training seeds 10–15: 24 policies, each requested to train for 100,000 steps. PPO's 512-step rollout blocks result in **100,352 executed steps**.
- Ten deterministic evaluations per policy, seeds 20000–20009: **240 evaluations**.
- Three concurrent jobs on CPU, one PyTorch thread per job. The fixed physical power predictors and gait generator are reused; PPO policies start from scratch.
- Success: world-forward displacement of at least 0.20 m from the starting position, evaluated at the first 100 ms boundary. Overshoot is recorded.
- Deadline: 40 s. Falling, direction/lateral escape, and timeout are failures. The lateral limit is 0.05 m; the heading-error limit is 0.35 rad.
- There is no target-speed or speed-tracking reward. The observation includes normalized remaining distance and remaining time.
- The action consists of five gait parameters, not 17 independently generated joint commands.

## Reward

```text
r = 100 * capped_forward_progress_m + 200 * goal
    - 200 * fall_or_escape - 200 * timeout
    - 0.05 * dt - 0.5 * tilt_squared * dt
    - 0.5 * heading_squared * dt - 2 * lateral_squared * dt
    - 0.005 * mean_joint_velocity_squared * dt
    - 0.03 * mean_action_change_squared - 0.5 * OOD_duration
    - lambda * predicted_positive_energy_increment_J
```

Progress is the change in a position potential capped at the target distance. Positive survival/upright rewards are removed to avoid rewarding prolonged episodes. The time penalty is 0.05 per second, not a speed target.

The fixed physical power model is evaluated at 10 Hz. Buffered/delayed contributions are included when the episode terminates. The energy penalty uses nonnegative predictions; raw predictions and negative-prediction rates are also retained. Gamma is 1.0 so that delaying arrival does not discount the energy penalty. The observed 40 s finite horizon ends in a true failure terminal state.

## Evaluation and criteria

Report success, falls, timeouts, arrival time, overshoot, total energy under **each** of R1–R3, and the fraction of time outside the training distribution (OOD). Retain every evaluation, including failures. A failed low-energy episode is not a successful low-energy arrival.

The primary energy scale is the **same R2 predictor for all policies**. R1/R3 predictions provide sensitivity analyses. For pairs that both succeed with the same training/evaluation seed, form a mean difference within each training seed. The independent units are the six training seeds. Use a one-sided paired sign-flip test, Holm correction for the three comparisons with Task-only, and saved bootstrap intervals.

The prespecified combined criterion requires: at least 90% success and paired-success fraction in every seed; OOD no greater than 10% in the evaluated episodes of both policies; no negative predictions; overshoot within 5 mm in successful pairs; lower R2 energy in all six seeds; Holm-adjusted p < 0.05; and lower mean energy under R1/R3. A failed criterion must remain reported as failed. Conditional comparisons and small-sample limitations remain relevant.

## Records and reruns

The [archived experiment](../logs/distance_goal_20261011/) contains the frozen design, source, final policies and evaluation traces. The [results report](distance_goal_20261011_RESULTS.md) explains the outcomes. The portable offline check is `python tools/reproduce.py rl`, run from the repository root.

Original collection used a macOS Python environment and a runner lock. Do not launch a duplicate runner or overwrite partial training. Original status/error records and frozen `source/scripts/run_distance_experiment.py` document collection; original resume logic rejects automatic replacement of partial training. A fresh experiment needs a new output directory and platform-specific executable settings. See the [reproduction guide](../../docs/REPRODUCTION.md).

The preflight had 30 passing software tests and a short four-policy experiment (512 steps each, eight evaluations) reaching 20 cm in about 16 s. These short runs are not the reported training results.

## Hardware scope

At planning time, the older Windows package replayed a fixed 20 s sequence and required changes for variable-duration commands and measurement windows. Simulation alone does not demonstrate physical energy savings. The later variable-duration transfer and actual 24 physical recordings are retained separately; see [motion learning and replay](../README.md). This distance protocol does not require verified foot clearance.
