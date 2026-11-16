"""Audit saved turn trials and describe command activity, without power metrics."""
from pathlib import Path
import json
import sys
import hashlib

import numpy as np

from motion_audit import audit_motion


root = Path(__file__).resolve().parents[1]
out = root / 'development' / sys.argv[1]
rows = []
audits = []
for path in sorted((out / 'raw').glob('*.npz')):
    audit = audit_motion(path)
    audits.append(audit)
    with np.load(path, allow_pickle=False) as d:
        t = d['t']
        q = d['qcmd']
        yaw = np.rad2deg(np.unwrap(d['yaw']))
        yaw -= yaw[0]
        # This describes intervals differing from the final standing reference;
        # it is not a measured mechanical stopping time or an energy interval.
        active = np.max(np.abs(q - q[-1]), axis=1) > 1e-5
        edges = np.diff(np.r_[False, active, False].astype(int))
        starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
        segments = []
        for start, stop in zip(starts, stops):
            settled = min(len(t)-1, stop + 140)
            segments.append(dict(start_s=float(t[start]), end_s=float(t[stop-1]),
                before_yaw_deg=float(yaw[max(0, start-1)]),
                yaw_1p4s_after_end_deg=float(yaw[settled])))
        xy = np.linalg.norm(d['position'][:, :2] - d['origin'][:2], axis=1)
        rows.append(dict(raw=str(path), seed=audit['spec']['seed'],
            action=audit['action'], passed=audit['revised_motion_pass'],
            failed_gates=audit['failed_revised_gates'],
            final_yaw_deg=audit['metrics']['signed_unwrapped_yaw_change_deg'],
            max_xy_displacement_m=float(xy.max()),
            max_abs_qcmd_rad=float(np.abs(q).max()),
            last_nonstanding_command_s=float(t[np.where(active)[0][-1]]) if active.any() else 0.,
            inferred_correction_bursts=max(0, len(segments)-1), command_segments=segments))
summary = dict(trials=len(rows), passed=sum(r['passed'] for r in rows),
    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.glob('turn_revision*.py')}, rows=rows)
for action in ('turn_90', 'turn_180'):
    subset = [r for r in rows if r['action'] == action]
    if subset:
        summary[action] = dict(n=len(subset), passed=sum(r['passed'] for r in subset),
            final_yaw_range_deg=[min(r['final_yaw_deg'] for r in subset), max(r['final_yaw_deg'] for r in subset)],
            last_command_range_s=[min(r['last_nonstanding_command_s'] for r in subset), max(r['last_nonstanding_command_s'] for r in subset)],
            max_xy_displacement_m=max(r['max_xy_displacement_m'] for r in subset),
            max_correction_bursts=max(r['inferred_correction_bursts'] for r in subset))
(out / 'independent_motion_audit.json').write_text(json.dumps(audits, indent=2))
(out / 'validation_report.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='rows'}, indent=2))
