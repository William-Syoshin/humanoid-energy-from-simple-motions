"""Independent mechanical audit for the revised synthetic-power experiment.

The power target, prediction errors, and model identity are deliberately unused.
Input is an NPZ path or a mapping of arrays. Thresholds are fixed below; callers
cannot pass per-trial overrides. All endpoint gates apply to recorded states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Any

import numpy as np


THRESHOLDS = {
    "version": "motion_audit_v1_20261111",
    "terminal_window_s": 1.0,
    "upright_height_m": 0.28,
    "upright_tilt_deg": 20.0,
    "mean_horizontal_speed_limit_m_s": 0.02,
    "mean_linear_speed_limit_m_s": 0.02,
    "mean_angular_speed_limit_rad_s": 0.1,
    "foot_normal_load_threshold_N": 0.05,
    "both_feet_supported_min_time_fraction": 0.95,
    "lying_height_limit_m": 0.18,
    "lying_h_min": 0.7,
    "turn_error_limit_deg": 10.0,
    "turn_max_tilt_deg": 45.0,
    "turn_direction": "positive_yaw",
    "walk_forward_min_m": 0.05,
    "stop_forward_min_m": 0.02,
    "push_object_displacement_min_m": 0.05,
    "legacy_upright_height_min_m": 0.20,
    "legacy_upright_h_max": 0.3,
    "legacy_turn_shortfall_rad": 0.2,
    "original_command_bound_rad_diagnostic_only": 1.4,
    "orientation_consistency_tolerance": 1e-5,
}


def _finite(name: str, value: Any, shape=None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: non-finite values")
    return array


def _terminal(t: np.ndarray, arrays: dict[str, np.ndarray]):
    """Clip to exactly the final second, interpolating only the left boundary."""
    duration = THRESHOLDS["terminal_window_s"]
    complete = bool(t[-1] - t[0] >= duration - 1e-9)
    start = max(t[0], t[-1] - duration)
    selected = t >= start - 1e-10
    tw = t[selected].copy()
    out = {key: values[selected].copy() for key, values in arrays.items()}
    if tw[0] > start + 1e-10:
        right = int(np.searchsorted(t, start)); left = right - 1
        fraction = (start - t[left]) / (t[right] - t[left])
        tw = np.r_[start, tw]
        for key, values in arrays.items():
            first = values[left] + fraction * (values[right] - values[left])
            out[key] = np.concatenate((np.asarray(first)[None], out[key]), axis=0)
    return tw, out, complete


def _time_mean(values: np.ndarray, t: np.ndarray) -> float:
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(values, t) / (t[-1] - t[0]))


def audit_motion(raw: str | Path | Mapping[str, Any], action: str | None = None) -> dict:
    source = None
    if isinstance(raw, (str, Path)):
        source = Path(raw).resolve()
        with np.load(source, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
    else:
        data = dict(raw)
    spec = json.loads(str(data["spec_json"])) if "spec_json" in data else {}
    if action is not None and spec.get("action") not in (None, action):
        raise ValueError("Explicit action conflicts with the recorded spec_json")
    action = action or spec.get("action")
    actions = {"walk", "stop", "push_box", "get_up", "lie_down", "turn_90", "turn_180"}
    if action not in actions:
        raise ValueError(f"Unsupported or unspecified action: {action!r}")
    t = _finite("t", data["t"])
    if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("t must contain at least two strictly increasing samples")
    n = len(t)
    position = _finite("position", data["position"], (n, 3))
    saved_h = _finite("h", data["h"], (n,))
    saved_yaw = _finite("yaw", data["yaw"], (n,))
    contact = _finite("contact", data["contact"])
    if contact.ndim != 2 or contact.shape[0] != n or contact.shape[1] < 6:
        raise ValueError("contact must have shape (N, >=6), with left/right loads in columns 4/5")
    origin = _finite("origin", data.get("origin", position[0]), (3,))
    sources = {"initial_state": "first recorded sample after settling"}
    orientation_checks = {}
    if "root_orientation" in data:
        rotation = _finite("root_orientation", data["root_orientation"], (n, 3, 3))
        gram = rotation.transpose(0, 2, 1) @ rotation
        orientation_checks = {
            "max_orthogonality_error": float(np.max(abs(gram - np.eye(3)))),
            "max_determinant_error": float(np.max(abs(np.linalg.det(rotation) - 1))),
            "max_h_difference": float(np.max(abs(saved_h - (1 - rotation[:, 2, 2])))),
        }
        if max(orientation_checks.values()) > THRESHOLDS["orientation_consistency_tolerance"]:
            raise ValueError(f"Inconsistent root orientation: {orientation_checks}")
        h = 1 - rotation[:, 2, 2]
        yaw = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
        sources["orientation"] = "root_orientation"
    else:
        h, yaw = saved_h, saved_yaw
        sources["orientation"] = "stored h and yaw (legacy recording)"
    if np.any((h < -1e-7) | (h > 2 + 1e-7)):
        raise ValueError("h lies outside its geometric range [0, 2]")
    tilt = np.degrees(np.arccos(np.clip(1 - h, -1, 1)))
    if "root_velocity" in data:
        velocity = _finite("root_velocity", data["root_velocity"], (n, 6))
        linear, angular = velocity[:, :3], velocity[:, 3:]
        sources["velocity"] = "root_velocity from Webots"
    else:
        linear = np.gradient(position, t, axis=0, edge_order=1)
        angular = _finite("root_omega", data["root_omega"], (n, 3))
        sources["velocity"] = "linear: finite differences of position; angular: root_omega"
    horizontal_speed = np.linalg.norm(linear[:, :2], axis=1)
    linear_speed = np.linalg.norm(linear, axis=1)
    angular_speed = np.linalg.norm(angular, axis=1)
    both = (contact[:, 4] > THRESHOLDS["foot_normal_load_threshold_N"]) & (
        contact[:, 5] > THRESHOLDS["foot_normal_load_threshold_N"])
    tw, end, complete = _terminal(t, dict(height=position[:, 2], h=h, tilt=tilt,
        horizontal_speed=horizontal_speed, linear_speed=linear_speed,
        angular_speed=angular_speed, both=both.astype(float)))
    # Support fraction integrates the 100 Hz Boolean support indicator. This is
    # a sampled approximation, not proof of uninterrupted contact between samples.
    support_fraction = _time_mean(end["both"], tw)
    mean_horizontal = _time_mean(end["horizontal_speed"], tw)
    mean_linear = _time_mean(end["linear_speed"], tw)
    mean_angular = _time_mean(end["angular_speed"], tw)
    forward = float(position[-1, 0] - origin[0])
    yaw_change = float(np.unwrap(yaw)[-1] - np.unwrap(yaw)[0])
    legacy_yaw_change = float(np.unwrap(saved_yaw)[-1] - np.unwrap(saved_yaw)[0])
    target = math.pi / 2 if action == "turn_90" else math.pi if action == "turn_180" else None
    object_distance = None
    if "object_position" in data:
        obj = _finite("object_position", data["object_position"], (n, 3))
        obj0 = _finite("object_origin", data.get("object_origin", obj[0]), (3,))
        object_distance = float(np.linalg.norm(obj[-1] - obj0))
    initial_upright = bool(position[0, 2] >= THRESHOLDS["upright_height_m"] and
                           tilt[0] <= THRESHOLDS["upright_tilt_deg"] + 1e-9)
    initial_lying = bool(position[0, 2] < THRESHOLDS["lying_height_limit_m"] and
                        h[0] > THRESHOLDS["lying_h_min"])
    gates = {
        "full_terminal_window": complete,
        "terminal_upright_height": bool(np.all(end["height"] >= THRESHOLDS["upright_height_m"])),
        "terminal_upright_tilt": bool(np.all(end["tilt"] <= THRESHOLDS["upright_tilt_deg"] + 1e-9)),
        "terminal_horizontal_speed": bool(mean_horizontal < THRESHOLDS["mean_horizontal_speed_limit_m_s"]),
        "terminal_linear_speed": bool(mean_linear < THRESHOLDS["mean_linear_speed_limit_m_s"]),
        "terminal_angular_speed": bool(mean_angular < THRESHOLDS["mean_angular_speed_limit_rad_s"]),
        "terminal_both_feet_support": bool(support_fraction >= THRESHOLDS["both_feet_supported_min_time_fraction"] - 1e-12),
        "initial_upright": initial_upright,
        "initial_lying": initial_lying,
        "terminal_lying_height": bool(np.all(end["height"] < THRESHOLDS["lying_height_limit_m"])),
        "terminal_lying_h": bool(np.all(end["h"] > THRESHOLDS["lying_h_min"])),
        "walk_forward": forward > THRESHOLDS["walk_forward_min_m"],
        "stop_forward": forward > THRESHOLDS["stop_forward_min_m"],
        "push_object_displacement": bool(object_distance is not None and object_distance > THRESHOLDS["push_object_displacement_min_m"]),
        "turn_positive_direction": yaw_change > 0,
        "turn_target_error": bool(target is not None and abs(yaw_change - target) <= math.radians(THRESHOLDS["turn_error_limit_deg"]) + 1e-12),
        "turn_upright_throughout": bool(np.all(tilt <= THRESHOLDS["turn_max_tilt_deg"] + 1e-9)),
    }
    upright_required = ["full_terminal_window", "terminal_upright_height", "terminal_upright_tilt"]
    stable_required = upright_required + ["terminal_horizontal_speed", "terminal_angular_speed", "terminal_both_feet_support"]
    required = {
        "walk": upright_required + ["walk_forward"],
        "push_box": upright_required + ["push_object_displacement"],
        "stop": stable_required + ["stop_forward"],
        "get_up": stable_required + ["initial_lying"],
        "lie_down": ["full_terminal_window", "initial_upright", "terminal_lying_height", "terminal_lying_h", "terminal_linear_speed", "terminal_angular_speed"],
        "turn_90": stable_required + ["initial_upright", "turn_positive_direction", "turn_target_error", "turn_upright_throughout"],
        "turn_180": stable_required + ["initial_upright", "turn_positive_direction", "turn_target_error", "turn_upright_throughout"],
    }[action]
    legacy_gates = {
        "endpoint_upright": bool(position[-1, 2] > THRESHOLDS["legacy_upright_height_min_m"] and saved_h[-1] < THRESHOLDS["legacy_upright_h_max"]),
        "endpoint_lying": bool(position[-1, 2] < THRESHOLDS["lying_height_limit_m"] and saved_h[-1] > THRESHOLDS["lying_h_min"]),
        "walk_forward": gates["walk_forward"], "stop_forward": gates["stop_forward"],
        "push_object_displacement": gates["push_object_displacement"],
        "turn_absolute_angle": bool(target is not None and abs(legacy_yaw_change) >= target - THRESHOLDS["legacy_turn_shortfall_rad"]),
    }
    legacy_required = {
        "walk": ["endpoint_upright", "walk_forward"],
        "stop": ["endpoint_upright", "stop_forward"],
        "push_box": ["endpoint_upright", "push_object_displacement"],
        "get_up": ["endpoint_upright"], "lie_down": ["endpoint_lying"],
        "turn_90": ["endpoint_upright", "turn_absolute_angle"],
        "turn_180": ["endpoint_upright", "turn_absolute_angle"],
    }[action]
    diagnostics = {"qcmd_present": "qcmd" in data}
    if "qcmd" in data:
        qcmd = _finite("qcmd", data["qcmd"], (n, 17))
        diagnostics.update(max_abs_qcmd_rad=float(np.max(abs(qcmd))),
            samples_outside_original_command_bound=int(np.count_nonzero(np.any(abs(qcmd) > THRESHOLDS["original_command_bound_rad_diagnostic_only"] + 1e-10, axis=1))))
    return {
        "action": action, "spec": spec, "source": str(source) if source else None,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
        "thresholds": dict(THRESHOLDS), "signal_sources": sources,
        "legacy_endpoint_pass": all(legacy_gates[key] for key in legacy_required),
        "legacy_required_gates": legacy_required, "legacy_gates": legacy_gates,
        "revised_motion_pass": all(gates[key] for key in required),
        "revised_required_gates": required, "revised_gates": gates,
        "failed_revised_gates": [key for key in required if not gates[key]],
        "metrics": {
            "samples": n, "recorded_duration_s": float(t[-1] - t[0]),
            "terminal_window_start_s": float(tw[0]), "terminal_window_end_s": float(tw[-1]),
            "initial_height_m": float(position[0, 2]), "initial_tilt_deg": float(tilt[0]),
            "final_height_m": float(position[-1, 2]), "final_tilt_deg": float(tilt[-1]),
            "terminal_min_height_m": float(end["height"].min()),
            "terminal_max_height_m": float(end["height"].max()),
            "terminal_max_tilt_deg": float(end["tilt"].max()),
            "terminal_min_h": float(end["h"].min()), "whole_trial_max_tilt_deg": float(tilt.max()),
            "terminal_mean_horizontal_speed_m_s": mean_horizontal,
            "terminal_mean_linear_speed_m_s": mean_linear,
            "terminal_mean_angular_speed_rad_s": mean_angular,
            "terminal_both_feet_supported_time_fraction": support_fraction,
            "forward_m": forward, "object_displacement_m": object_distance,
            "signed_unwrapped_yaw_change_rad": yaw_change,
            "signed_unwrapped_yaw_change_deg": math.degrees(yaw_change),
            "turn_target_deg": math.degrees(target) if target is not None else None,
            "turn_error_deg": math.degrees(yaw_change - target) if target is not None else None,
        },
        "orientation_checks": orientation_checks, "command_diagnostics": diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, nargs="+")
    parser.add_argument("--action", help="Override action only when absent from spec_json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thresholds-out", type=Path)
    args = parser.parse_args()
    reports = [audit_motion(path, args.action) for path in args.raw]
    payload = reports[0] if len(reports) == 1 else reports
    text = json.dumps(payload, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(text + "\n")
    else:
        print(text)
    if args.thresholds_out:
        args.thresholds_out.parent.mkdir(parents=True, exist_ok=True)
        with args.thresholds_out.open("x") as handle:
            json.dump(THRESHOLDS, handle, indent=2, allow_nan=False)
            handle.write("\n")


if __name__ == "__main__":
    main()
