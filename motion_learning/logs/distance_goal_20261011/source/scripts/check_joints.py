"""Run independent small sine commands for all 17 joints and check sensor response."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from verify_run import verify

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grounded', action='store_true', help='Use gravity and floor contact instead of unloaded diagnostics')
    parser.add_argument('--headless', action='store_true', help='Linux xvfb execution')
    args = parser.parse_args()
    config = ROOT/'config/joint_test.json'
    joints = json.loads(config.read_text())['joints']
    world = ROOT/'worlds'/('khr_training.wbt' if args.grounded else 'khr_joint_test.wbt')
    log_parent = ROOT/'logs'
    log_parent.mkdir(exist_ok=True)
    results = []
    for joint in joints:
        before = set(log_parent.iterdir())
        command = [sys.executable, str(ROOT/'scripts/launch.py'), '--batch', '--world', str(world),
                   '--config', str(config), '--joint', joint]
        if args.headless:
            command.append('--headless')
        run = subprocess.run(command, capture_output=True, text=True)
        created = set(log_parent.iterdir()) - before
        # Single-environment diagnostic: do not run another launcher concurrently.
        directory = next(iter(created))/'run' if len(created) == 1 else None
        try:
            if directory is None:
                raise RuntimeError('Cannot uniquely locate run; do not run diagnostics concurrently')
            metrics = verify(directory)
            if run.returncode:
                raise RuntimeError(run.stdout + run.stderr)
            result = dict(joint=joint, log=str(directory.relative_to(ROOT)), **metrics)
        except Exception as exc:
            result = dict(joint=joint, log=str(directory), status='failed', error=str(exc))
        results.append(result)
        print(json.dumps(result), flush=True)
    path = log_parent/('joint-suite-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.json')
    path.write_text(json.dumps(results, indent=2)+'\n')
    print(f'Results: {path}')
    return int(any(r['status'] != 'passed' for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
