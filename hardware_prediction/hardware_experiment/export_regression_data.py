"""Collect processed trial CSVs into one regression-ready directory.

The original raw/processed hierarchy remains authoritative. Exported CSVs are
byte-for-byte copies with stable, human-readable names and a provenance manifest.
"""
import csv
import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
POSTURE_NAMES = {
    "upright": "stand",
    "squat": "squat",
    "supine": "supine",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_shape(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        return sum(1 for _ in reader), len(header)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, default=ROOT / "session_v2_proposed")
    args = parser.parse_args()
    session = args.session.resolve()
    source_root = session / "live_batches"
    destination_root = session / "regression_data" / "aligned"
    manifest_path = session / "regression_data" / "manifest.csv"
    candidates = []
    for aligned in source_root.glob("*/raw/*/processed/aligned.csv"):
        trial = aligned.parent.parent
        recipe_path = trial / "recipe.json"
        qc_path = aligned.parent / "qc.json"
        if not recipe_path.is_file() or not qc_path.is_file():
            continue
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if qc.get("usable") is not True:
            continue
        candidates.append((recipe["level"], recipe["posture"], recipe["trial_id"], aligned, recipe, qc, qc_path))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    destination_root.mkdir(parents=True, exist_ok=True)
    # This directory is a reproducible view of live, usable trials. Remove
    # stale exports when a pilot batch is excluded or a trial is replaced.
    for stale in destination_root.glob("*.csv"):
        stale.unlink()
    counters = defaultdict(int)
    rows = []
    for level, posture, trial_id, source, recipe, qc, qc_path in candidates:
        joint_name = None
        condition_name = None
        parts = recipe.get("recipe_group", "").split(":")
        if level in ("L1", "L3"):
            if len(parts) >= 3:
                joint_name = parts[2].lower().replace(" ", "_")
            if level == "L3" and len(parts) >= 5:
                condition_name = "_".join(parts[3:5]).lower().replace(" ", "_").replace(".", "p")
        elif level == "L2":
            if len(parts) >= 3:
                joint_name = parts[2].lower().replace(",", "_").replace(" ", "_")
            if len(parts) >= 6:
                condition_name = "_".join(parts[3:6]).lower().replace(" ", "_").replace(".", "p")
        elif level == "L4":
            if len(parts) >= 3:
                joint_name = parts[2].lower().replace(" ", "_")
            if len(parts) >= 5:
                condition_name = "_".join([parts[1], *parts[3:5]]).lower().replace(" ", "_").replace(".", "p")
        elif level == "L5":
            if len(parts) >= 3:
                joint_name = parts[2].lower().replace(",", "_").replace(" ", "_")
            if len(parts) >= 6:
                condition_name = "_".join(parts[3:6]).lower().replace(" ", "_").replace(".", "p")
        key = (level.lower(), posture, joint_name, condition_name)
        counters[key] += 1
        posture_name = POSTURE_NAMES.get(posture, posture.lower().replace(" ", "_"))
        joint_part = f"_{joint_name}" if joint_name else ""
        condition_part = f"_{condition_name}" if condition_name else ""
        filename = f"{level.lower()}_{posture_name}{joint_part}{condition_part}_{counters[key]:04d}.csv"
        destination = destination_root / filename
        shutil.copy2(source, destination)
        row_count, column_count = csv_shape(destination)
        rows.append({
            "file": filename,
            "trial_id": trial_id,
            "level": level,
            "posture": posture,
            "support": recipe["support"],
            "split": recipe["split"],
            "rows": row_count,
            "columns": column_count,
            "analysis_hz": qc.get("analysis_hz"),
            "trial_weight": 1.0,
            "sample_weight_per_row": 1.0 / row_count,
            "energy_J": qc["energy_J"],
            "processed_usable": int(qc["usable"] is True),
            "mechanical_success": "pending" if qc.get("mechanical_success") is None else int(bool(qc["mechanical_success"])),
            "sha256": digest(destination),
            "source_aligned": str(source.resolve()),
            "source_qc": str(qc_path.resolve()),
        })

    fields = ["file", "trial_id", "level", "posture", "support", "split", "rows", "columns",
              "analysis_hz", "trial_weight", "sample_weight_per_row",
              "energy_J", "processed_usable", "mechanical_success", "sha256", "source_aligned", "source_qc"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} files to {destination_root}")


if __name__ == "__main__":
    main()
