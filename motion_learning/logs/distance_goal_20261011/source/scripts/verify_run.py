"""Verify recorded simulator observations independently of the control loop."""
import argparse
import csv
import json
import math
from pathlib import Path


def verify(directory):
    meta = json.loads((directory/'metadata.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    config = meta['config']
    names = [j['name'] for j in meta['model']['joints']]
    rows = list(csv.DictReader((directory/'sensors.csv').open()))
    assert len(names) == 17
    assert len(meta['all_devices']) == 37  # 17 motors, 17 joint sensors, 3 torso sensors.
    assert summary['status'] == 'passed', summary
    assert len(rows) == summary['samples'] == math.ceil(config['duration_s'] / (config['control_timestep_ms']/1000))
    previous = None
    for raw in rows:
        row = {k: float(v) for k, v in raw.items()}
        assert all(math.isfinite(v) for v in row.values()), 'Non-finite measurement'
        if previous:
            dt = row['time_s'] - previous['time_s']
            assert math.isclose(dt, config['control_timestep_ms']/1000, abs_tol=1e-10)
            for n in names:
                expected = (row['q_rad/'+n] - previous['q_rad/'+n]) / dt
                assert math.isclose(row['dq_rad_s/'+n], expected, abs_tol=1e-9)
        previous = row
    if config['mode'] == 'sweep':
        name = config['sweep']['joint']
        measured = [float(r['q_rad/'+name]) for r in rows]
        commands = [float(r['target_rad/'+name]) for r in rows]
        # Reject a disconnected or permanently zero position sensor.
        assert max(measured)-min(measured) > config['sweep']['amplitude_rad'], name
        mean_q, mean_t = sum(measured)/len(rows), sum(commands)/len(rows)
        covariance = sum((q-mean_q)*(t-mean_t) for q,t in zip(measured,commands))
        variance = math.sqrt(sum((q-mean_q)**2 for q in measured)*sum((t-mean_t)**2 for t in commands))
        assert covariance/variance > 0.8, 'Measured motion does not correlate with commanded motion'
    return {k: summary[k] for k in ('status', 'samples', 'max_tilt_rad', 'max_tracking_error_rad')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))
