"""Read-only KHR-3HV and power-monitor preflight; sends no servo targets."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import serial

from common import NAMES, parse_power, read, save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("hardware.json"))
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = read(args.config)
    motion = read(args.motion)
    target = motion["initial_pose_rad"]
    sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
    from Rcb4BaseLib import Rcb4BaseLib

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "servo_targets_sent": False,
        "robot": {"ok": False, "joints": []},
        "power": {"ok": False, "samples": []},
    }

    rcb = Rcb4BaseLib()
    try:
        opened = bool(rcb.open(cfg["robot_port"], cfg["robot_baud"], 0.1))
        acknowledged = bool(opened and rcb.checkAcknowledge())
        report["robot"].update(opened=opened, acknowledged=acknowledged)
        if acknowledged:
            for joint, desired in zip(cfg["joints"], target):
                ok, position = rcb.getSinglePos(joint["id"], joint["sio"])
                angle = ((position - joint["zero_pos"]) / joint["units_per_rad"]
                         * joint["sign"])
                error = angle - desired
                report["robot"]["joints"].append({
                    "name": joint["name"], "ok": bool(ok),
                    "position": int(position), "q_rad": float(angle),
                    "target_rad": float(desired), "error_rad": float(error),
                    "within_initial_tolerance": bool(ok and abs(error) <= cfg["max_initial_error_rad"]),
                })
            report["robot"]["ok"] = all(
                row["within_initial_tolerance"] for row in report["robot"]["joints"]
            )
    except Exception as exc:
        report["robot"]["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        rcb.close()

    port = None
    try:
        port = serial.Serial(cfg["power_port"], cfg["power_baud"], timeout=0.25)
        # Opening this CH340 resets the legacy ESP32; allow its full boot time.
        time.sleep(2.0)
        port.reset_input_buffer()
        port.write(b"START\n")
        port.flush()
        deadline = time.perf_counter() + 3.0
        while len(report["power"]["samples"]) < 5 and time.perf_counter() < deadline:
            line = port.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            row = parse_power(line)
            if row is not None:
                report["power"]["samples"].append(row)
        report["power"]["ok"] = len(report["power"]["samples"]) >= 2
    except Exception as exc:
        report["power"]["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if port is not None:
            try:
                port.write(b"STOP\n")
                port.flush()
            finally:
                port.close()

    report["joint_order"] = NAMES
    save(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
