"""Run the authorized four-condition seed12-15 block, stopping on any failure."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from common import read, save, rows, sha, validate_config, integrate_power
from capture import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    bundle = root / 'data'
    plan_path = root / 'plan_commissioning_seed12_15.json'
    plan = read(plan_path)
    previous = root / 'commissioning_raw/Baseline_seed11_01'
    prepared = []
    assert plan['training_seeds'] == [12,13,14,15] and len(plan['trials']) == 16
    assert sha(bundle / 'manifest.json') == plan['manifest_sha256']
    for trial in plan['trials']:
        label = 'Baseline' if trial['condition'] == 'Task-only' else trial['condition']
        config = root / f'hardware.commissioning_{label}_seed{trial["training_seed"]}.json'
        motion = bundle / trial['motion']
        assert sha(motion) == trial['motion_sha256']
        validate_config(read(config), read(motion), commissioning=True)
        out = root / 'commissioning_raw' / trial['trial_id']
        reset = root / 'calibration' / ('before_' + trial['trial_id'])
        if out.exists() or reset.exists():
            raise FileExistsError('Existing attempt; do not automatically replace or repeat: ' + str(out))
        prepared.append((trial, label, config, motion, out, reset))
    if not args.execute:
        print('All 16 motions and configurations validated; no serial ports opened.')
        return
    results = []
    status_path = root / 'commissioning_raw/seed12_15_block_status.json'
    status = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  completed=[], distance_m=None, automatic_fall_detection=False)
    try:
        for trial, label, config, motion, out, reset in prepared:
            status['current_trial'] = trial['trial_id']
            save(status_path, status)
            print(f'{trial["trial_id"]}: restoring shared initial pose over 8 seconds', flush=True)
            with (root / 'calibration' / (reset.name + '_console.txt')).open('x', encoding='utf-8') as log:
                subprocess.run([sys.executable, str(root / 'restore_trial_initial.py'),
                                '--config', str(config), '--motion', str(motion),
                                '--previous-raw', str(previous), '--out', str(reset), '--execute'],
                               cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
            if read(reset / 'meta.json')['status'] != 'complete':
                raise RuntimeError('Posture reset did not complete')
            print(f'{trial["trial_id"]}: capture starting', flush=True)
            run(bundle, plan_path, trial['trial_id'], config, out, execute=True, commissioning=True)
            meta = read(out / 'meta.json')
            commands = rows(out / 'commands.csv')
            if meta['status'] != 'complete' or meta['reader_errors']:
                raise RuntimeError('Capture failed')
            if len(commands) != len(read(motion)['commands']) or not all(int(c['ok']) for c in commands):
                raise RuntimeError('Command log incomplete')
            power = rows(out / 'power.csv')
            result = integrate_power(power, meta['score_start'], meta['score_end'], read(config))
            result.update(condition=label, training_seed=trial['training_seed'], trial_id=trial['trial_id'],
                          command_count=len(commands), power_sample_count=len(power),
                          distance_m=None, physical_outcome='not_individually_reported',
                          commissioning=True, mapping_verified=False, source_dir=str(out))
            save(out / 'measurement_summary.json', result)
            results.append(result)
            status['completed'].append(trial['trial_id'])
            save(root / 'commissioning_raw/seed12_15_measurements.json', dict(results=results,
                 note='One replay per condition. Distance unmeasured. Return transitions excluded.'))
            print(f'{trial["trial_id"]}: complete, {result["energy_J"]:.3f} J, {result["mean_power_W"]:.3f} W', flush=True)
            previous = out
        status['status'] = 'complete'
    except BaseException as exc:
        status.update(status='stopped', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        status['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(status_path, status)


if __name__ == '__main__':
    main()
