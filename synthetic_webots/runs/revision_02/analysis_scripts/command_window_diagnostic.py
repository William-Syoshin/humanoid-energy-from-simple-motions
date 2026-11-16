"""Supplementary command-window scoring; never fit or select a model/controller.

Usage: python command_window_diagnostic.py RUN --out NEW_DIRECTORY_OUTSIDE_RUN
Full recordings remain the primary evaluation. The shortened interval contains
joint-command changes and is not a measurement of the robot's active period.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib
import json
import sys

import numpy as np

RULES = {
    "version": "command_window_v1",
    "command_change_threshold_rad": 1e-6,
    "threshold_comparison": "strictly greater than",
    "tail_after_last_change_s": 1.0,
    "change_time": "time of the latter sample in an adjacent command pair",
    "start": "first recorded sample",
    "end": "min(last recorded time, last command-change time + 1.0 s)",
    "no_command_change": "retain the complete recorded interval",
    "endpoint_between_samples": "linear interpolation of already computed power series",
    "primary": "full recorded interval",
    "supplementary": "shortened window containing joint-command changes",
    "supplementary_ja": "関節指令変化を含む短縮区間",
    "history": "compute features, oracle, and predictions on full raw history before slicing",
    "energy_aggregation": "equal-weight mean of trial absolute relative energy errors",
    "power_aggregation": "sample-pooled RMSE; includes interpolated endpoint if needed",
    "prohibited_use": "not for controller, timing, model, hyperparameter, or seed selection",
    "refitting": "prohibited; use the supplied frozen model files",
}


def command_window(t, qcmd):
    """Return fixed interval boundaries and command-change diagnostics."""
    t = np.asarray(t, dtype=float)
    qcmd = np.asarray(qcmd, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("t must contain at least two finite, increasing sample times")
    if qcmd.shape != (len(t), 17) or not np.isfinite(qcmd).all():
        raise ValueError("qcmd must be finite with shape (N, 17)")
    changed = np.flatnonzero(np.max(np.abs(np.diff(qcmd, axis=0)), axis=1)
                             > RULES["command_change_threshold_rad"]) + 1
    last = float(t[changed[-1]]) if len(changed) else None
    end = min(float(t[-1]), last + RULES["tail_after_last_change_s"]) if last is not None else float(t[-1])
    return dict(start_s=float(t[0]), end_s=end, duration_s=end-float(t[0]),
                full_end_s=float(t[-1]), full_duration_s=float(t[-1]-t[0]),
                changed_pairs=len(changed), last_command_change_s=last,
                shortened=bool(end < t[-1]))


def truncate_series(t, series, end):
    """Slice fully computed 1-D series, inserting an exact endpoint if needed."""
    t = np.asarray(t, dtype=float)
    series = [np.asarray(y, dtype=float) for y in series]
    if not t[0] < end <= t[-1] or any(y.shape != t.shape for y in series):
        raise ValueError("Invalid window endpoint or power-series shape")
    keep = t <= end
    tw = t[keep]
    values = [y[keep] for y in series]
    if tw[-1] < end:
        tw = np.r_[tw, end]
        values = [np.r_[v, np.interp(end, t, y)] for y, v in zip(series, values)]
    return tw, values


def power_metrics(t, truth, prediction):
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.shape != np.shape(t) or prediction.shape != truth.shape:
        raise ValueError("Power series must match the window sample times")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite power")
    actual = float(np.trapz(truth, t))
    estimated = float(np.trapz(prediction, t))
    if actual <= 0:
        raise ValueError("Relative energy error requires positive reference energy")
    sse = float(np.sum((prediction-truth)**2))
    return dict(reference_energy_J=actual, predicted_energy_J=estimated,
                absolute_relative_energy_error_pct=100*abs(estimated-actual)/actual,
                squared_power_error_sum=sse, power_samples=len(truth),
                power_RMSE_W=float(np.sqrt(sse/len(truth))))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    if out == run or run in out.parents:
        raise ValueError("Output must be a new directory outside the input run")
    if out.exists():
        raise FileExistsError(out)
    design_path = run/"design.json"
    design = json.loads(design_path.read_text())
    tests = [s for s in design["trials"] if s["split"] == "test"]
    if not tests or len({s["id"] for s in tests}) != len(tests):
        raise ValueError("Missing tests or duplicate trial identifiers")
    tracked = {str(design_path): digest(design_path)}
    for relative, expected in design["source_hashes"].items():
        path = run/"source"/relative
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Frozen source hash mismatch: {relative}")
        tracked[str(path)] = actual
    src = run/"source/src"
    # Importing frozen modules must not create __pycache__ inside the input run.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(src))
    protocol = importlib.import_module("protocol")
    if Path(protocol.__file__).resolve() != (src/"protocol.py").resolve():
        raise RuntimeError("The frozen protocol module was not loaded")
    import joblib
    models = {}
    for name in ["M1", "M2", "M3", "M4"]:
        path = run/"analysis/models"/(name+".npz")
        tracked[str(path)] = digest(path)
        with np.load(path, allow_pickle=False) as z:
            models[name] = {k: z[k].copy() for k in ["cols", "scale", "beta", "alpha"]}
    path = run/"analysis/models/M5.joblib"
    tracked[str(path)] = digest(path)
    tree = joblib.load(path)
    rows, windows = [], []
    for spec in tests:
        key = "test_"+spec["id"]
        path = run/"raw"/(key+".npz")
        status_path = run/"status"/(key+".json")
        tracked[str(path)] = digest(path)
        tracked[str(status_path)] = digest(status_path)
        status = json.loads(status_path.read_text())
        if tracked[str(path)] != status["sha256"]:
            raise ValueError(f"Raw/status hash mismatch: {key}")
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        if json.loads(str(data["spec_json"])) != spec:
            raise ValueError(f"Raw/design specification mismatch: {key}")
        window = command_window(data["t"], data["qcmd"])
        windows.append(dict(trial=key, action=spec["action"], seed=spec["seed"], **window))
        # Full-history evaluation is essential for the oracle's thermal state.
        x = protocol.features(data)[3]
        truth, _ = protocol.oracle(data)
        predictions = {name: np.maximum(0, (x[:, m["cols"]]/m["scale"])@m["beta"])
                       for name, m in models.items()}
        predictions["M5"] = np.maximum(0, tree.predict(x))
        for name, prediction in predictions.items():
            for label, end in [("primary_full", window["full_end_s"]),
                               ("supplementary_command_window", window["end_s"])]:
                t, (y, pred) = truncate_series(data["t"], [truth, prediction], end)
                rows.append(dict(trial=key, action=spec["action"], seed=spec["seed"],
                                 model=name, window=label, start_s=float(t[0]),
                                 end_s=float(t[-1]), duration_s=float(t[-1]-t[0]),
                                 **power_metrics(t, y, pred)))
    primary_path = run/"analysis/per_trial_metrics.csv"
    tracked[str(primary_path)] = digest(primary_path)
    with primary_path.open() as f:
        primary_rows = list(csv.DictReader(f))
    primary = {(r["model"], "test_"+r["trial"]): r for r in primary_rows}
    if len(primary) != len(primary_rows) or len(primary) != 5*len(tests):
        raise ValueError("Existing primary scores have missing or duplicate trials")
    metric_columns = {"reference_energy_J": "true_energy_J", "predicted_energy_J": "pred_energy_J",
                      "absolute_relative_energy_error_pct": "energy_error_pct", "power_RMSE_W": "power_RMSE_W"}
    for row in (r for r in rows if r["window"] == "primary_full"):
        old_row = primary[(row["model"], row["trial"])]
        if any(not np.isclose(row[new], float(old_row[old]), rtol=1e-10, atol=1e-10)
               for new, old in metric_columns.items()):
            raise ValueError(f"Full-window diagnostic does not reproduce primary scores: {row['model']} {row['trial']}")
    summary = []
    for name in ["M1", "M2", "M3", "M4", "M5"]:
        for label in ["primary_full", "supplementary_command_window"]:
            selected = [r for r in rows if r["model"] == name and r["window"] == label]
            assert len(selected) == len(tests)
            summary.append(dict(model=name, window=label, trials=len(selected),
                mean_window_duration_s=float(np.mean([r["duration_s"] for r in selected])),
                total_window_duration_s=float(sum(r["duration_s"] for r in selected)),
                mean_absolute_relative_energy_error_pct=float(np.mean([r["absolute_relative_energy_error_pct"] for r in selected])),
                pooled_power_RMSE_W=float(np.sqrt(sum(r["squared_power_error_sum"] for r in selected)/sum(r["power_samples"] for r in selected)))))
    changed = [p for p, h in tracked.items() if digest(p) != h]
    if changed:
        raise RuntimeError(f"Inputs changed during diagnostic: {changed}")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"rules.json", RULES)
    write_json(out/"input_hashes.json", tracked)
    write_json(out/"window_boundaries.json", windows)
    write_csv(out/"window_boundaries.csv", windows)
    write_csv(out/"per_trial_metrics.csv", rows)
    write_csv(out/"model_summary.csv", summary)
    write_json(out/"diagnostic.json", dict(rules=RULES, input_run=str(run),
        inputs_unchanged=True, refitted=False, full_window_matches_existing_primary=True,
        trials=len(tests), windows=windows,
        model_summary=summary, script_sha256=digest(__file__),
        interpretation="Full interval remains primary. Shortened command windows are supplemental; do not use these values to select motions or models."))
    print(out)


if __name__ == "__main__":
    main()
