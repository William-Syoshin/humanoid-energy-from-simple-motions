"""Prospective 2 x 2 arm coordination / posture experiment.

Planning and evaluation never open a serial port. Acquisition is one trial at a
time, requires an explicit posture confirmation, and uses the existing recorder.
"""
import argparse
import copy
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
from core import read, save, rows, csv_save, digest, compile_recipe

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'session_v3_uniform_10hz/live_batches/L5_upright_upper_four_joint_20260915/raw/L5_UPPER_P000_U10'
MODELS = ROOT / 'regression_results/ridge_20260915_143619/models'
DEFAULT = ROOT / 'session_v4_controlled_10hz'
ARMS = ((10, 12), (13, 15))
CONDITIONS = {'A': ('upright', 'feet', True), 'B': ('upright', 'feet', False),
              'C': ('supine', 'ground', True), 'D': ('supine', 'ground', False)}


def make_recipe(template, condition, repetition):
    """Identical per-joint raised-cosine excursions, translated in time only."""
    posture, support, simultaneous = CONDITIONS[condition]
    r = copy.deepcopy(template)
    for key in ('focused_feedback_joint', 'focused_feedback_joints'):
        r.pop(key, None)
    r.update(trial_id=f'DIAG_{condition}_{repetition:02d}', split='test', level='controlled',
             approved=True, kind='frames', duration_s=10.0, pre_s=2.0, post_s=2.0,
             settle_s=0.0, posture=posture, support=support, support_intervals=[],
             excitation=[], recipe_group=f'controlled:{condition}',
             success_gate='No arm contact or loss of support; both arms complete excursion and return to starting posture.',
             notes='Support is operator-assigned. +10 degree excursion uses prior joint mapping; physical clearance must be checked.',
             condition=condition, repetition=repetition)
    # Alternate early/late simultaneous movement and left/right sequential lead.
    variant = (repetition - 1) % 2
    starts = [1 + 4 * variant] * 2 if simultaneous else ([1, 5] if variant == 0 else [5, 1])
    center = np.asarray(r['center_rad'])
    frames = []
    for t in np.linspace(0, 10, 101):
        q = center.copy()
        for indices, start in zip(ARMS, starts):
            u = (t - start) / 4
            pulse = (1 - math.cos(2 * math.pi * u)) / 2 if 0 < u < 1 else 0
            for j in indices:
                q[j] += math.radians(10) * pulse
        frames.append({'t_s': float(t), 'q_rad': q.tolist()})
    r['frames'] = frames
    r['timing_variant'] = variant
    return r


def plan(root, repetitions=6, seed=20260927):
    if repetitions < 2 or repetitions % 2:
        raise ValueError('Use an even number of repetitions (at least 2) for temporal balance')
    cfg, template = read(SOURCE / 'config.json'), read(SOURCE / 'recipe.json')
    if cfg['command_hz'] != 10 or cfg['analysis_hz'] != 10:
        raise ValueError('Expected established 10 Hz configuration')
    recipes = [make_recipe(template, c, n) for n in range(1, repetitions + 1) for c in CONDITIONS]
    from protocol import validate_primary, freeze_design
    for r in recipes:
        validate_primary(cfg, r, True)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'recipes').mkdir()
    (root / 'preview').mkdir()
    shutil.copyfile(SOURCE / 'config.json', root / 'config.json')
    shutil.copytree(MODELS, root / 'models')
    rng = random.Random(seed)
    schedule = []
    for n in range(1, repetitions + 1):
        # Alternate posture order across repeats. Repositioning is always manual.
        groups = [('A', 'B'), ('C', 'D')]
        if n % 2 == 0:
            groups.reverse()
        for group in groups:
            order = list(group)
            rng.shuffle(order)
            for c in order:
                r = next(x for x in recipes if x['condition'] == c and x['repetition'] == n)
                tid = r['trial_id']
                save(root / 'recipes' / f'{tid}.json', r)
                csv_save(root / 'preview' / f'{tid}.csv', compile_recipe(cfg, r))
                schedule.append(dict(order=len(schedule) + 1, trial_id=tid, split='test',
                                     condition=c, repetition=n, posture=r['posture'], support=r['support'],
                                     recipe=f'recipes/{tid}.json'))
    csv_save(root / 'trials.csv', schedule)
    save(root / 'protocol.json', dict(seed=seed, repetitions=repetitions, amplitude_deg=10,
         excursion_s=4, scoring_s=10, source=str(SOURCE), source_config_sha256=digest(SOURCE / 'config.json'),
         primary_metric='trial power RMSE (W)', secondary_metric='absolute relative energy error (%)',
         interpretation='Frozen historical models; support-label audit remains pending. No causal attribution to added features alone.',
         comparisons=['R1 minus R2 RMSE: A vs B and C vs D', 'R2 minus R3 RMSE: A vs C and B vs D'],
         inclusion='All processed attempts retained; success and repeat attempts reported separately. First attempted trial is the primary analysis unit.'))
    freeze_design(root, root / 'design.json')
    frozen = {str(p.relative_to(root)): digest(p) for p in root.rglob('*') if p.is_file()}
    save(root / 'integrity.json', frozen)
    return schedule


def verify(root):
    for rel, sha in read(root / 'integrity.json').items():
        if digest(root / rel) != sha:
            raise ValueError('Plan/model changed: ' + rel)


def run(root, trial_id, posture, clear, execute=False):
    verify(root)
    selected = [r for r in rows(root / 'trials.csv') if r['trial_id'] == trial_id]
    if len(selected) != 1:
        raise ValueError('Unknown trial ID')
    r = read(root / selected[0]['recipe'])
    preview = compile_recipe(read(root / 'config.json'), r)
    if not execute:
        print(f'DRY RUN {trial_id}: {r["posture"]}, {len(preview)} commands; no hardware connection.')
        return
    if posture != r['posture'] or not clear:
        raise ValueError('Confirm current posture and arm clearance; do not infer posture from previous trials')
    # Enforce declared order. An incomplete attempt requires inspection, not auto retry.
    for entry in rows(root / 'trials.csv'):
        if entry['trial_id'] == trial_id:
            break
        if not (root / 'raw' / entry['trial_id'] / 'observation.json').is_file():
            raise ValueError('Observe preceding trial first: ' + entry['trial_id'])
    out = root / 'raw' / trial_id
    if out.exists():
        raise ValueError('Attempt already exists; retained unchanged. Create a new declared session for repeats.')
    from acquire import record
    record(root / 'config.json', root / selected[0]['recipe'], out,
           execute=True, purpose='primary', design_path=root / 'design.json')
    print('Recorded. Inspect motion and register success/failure with observe before continuing.')


def observe(root, trial_id, success, note):
    verify(root)
    if trial_id not in {r['trial_id'] for r in rows(root / 'trials.csv')}:
        raise ValueError('Unknown trial')
    raw = root / 'raw' / trial_id
    meta = read(raw / 'meta.json')
    target = raw / 'observation.json'
    if target.exists():
        raise ValueError('Observation already exists; do not overwrite')
    if success and meta['status'] != 'complete':
        raise ValueError('Incomplete acquisition cannot be marked successful')
    save(target, dict(success=bool(success), note=note))
    if not (raw / 'processed/qc.json').is_file():
        print('Observation retained; no usable processed data available.')
        return
    qc = read(raw / 'processed/qc.json')
    if not qc.get('usable') or qc['aligned_sha256'] != digest(raw / 'processed/aligned.csv'):
        raise ValueError('Processed data failed integrity check')
    aligned = root / 'regression_data/aligned'
    aligned.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw / 'processed/aligned.csv', aligned / f'{trial_id}.csv')
    shutil.copyfile(raw / 'processed/qc.json', aligned / f'{trial_id}.qc.json')


def evaluate(root, output):
    verify(root)
    import analysis as a
    f = read(root / 'models/frozen.json')
    if f['analysis_sha256'] != digest(ROOT / 'analysis.py') or f['model_sha256'] != digest(root / 'models/frozen_models.npz'):
        raise ValueError('Frozen feature code or coefficients do not match')
    params = dict(np.load(root / 'models/frozen_models.npz', allow_pickle=False))
    results, inventory = [], []
    for r in rows(root / 'trials.csv'):
        tid = r['trial_id']; raw = root / 'raw' / tid
        ob = read(raw / 'observation.json') if (raw / 'observation.json').exists() else None
        p = root / 'regression_data/aligned' / f'{tid}.csv'
        inventory.append(dict(trial_id=tid, attempted=raw.exists(), observed=ob is not None, available=p.exists(),
                              success=ob['success'] if ob else 'unknown'))
        if ob is None or not p.exists():
            continue
        meta = dict(trial_id=tid, file=p.name, task=r['condition'], success=str(int(ob['success'])))
        d = a.read_trial(meta, p.parent, f['schema'])
        for model in a.MODELS:
            prediction = a.predict(d, model, (params[model + '_scale'], params[model + '_coef']))
            m = a.metrics(d, model, prediction)
            m.update(condition=r['condition'], repetition=int(r['repetition']),
                     absolute_residual_J=float(a.integrate(abs(prediction - d['p']), d['t'])))
            results.append(m)
    output.mkdir(parents=True, exist_ok=False)
    csv_save(output / 'attempt_inventory.csv', inventory)
    if results:
        csv_save(output / 'per_trial_metrics.csv', results)
        improvements = []
        for r in rows(root / 'trials.csv'):
            rr = {x['model']: x for x in results if x['trial_id'] == r['trial_id']}
            if not rr:
                continue
            improvements.append(dict(trial_id=r['trial_id'], condition=r['condition'], repetition=r['repetition'],
                  success=rr['R1']['success'], R1_minus_R2_RMSE_W=rr['R1']['power_RMSE_W']-rr['R2']['power_RMSE_W'],
                  R2_minus_R3_RMSE_W=rr['R2']['power_RMSE_W']-rr['R3']['power_RMSE_W']))
        csv_save(output / 'paired_model_improvements.csv', improvements)
        contrasts = []
        for rep in sorted({x['repetition'] for x in improvements}):
            by_condition = {x['condition']: x for x in improvements if x['repetition'] == rep}
            for first, second, metric in [('A','B','R1_minus_R2_RMSE_W'), ('C','D','R1_minus_R2_RMSE_W'),
                                           ('C','A','R2_minus_R3_RMSE_W'), ('D','B','R2_minus_R3_RMSE_W')]:
                if first in by_condition and second in by_condition:
                    x, y = by_condition[first], by_condition[second]
                    contrasts.append(dict(repetition=rep, contrast=f'{first}-{second}', metric=metric,
                         difference_W=x[metric]-y[metric], both_successful=x['success']==y['success']=='1'))
        if contrasts:
            csv_save(output / 'within_repetition_contrasts.csv', contrasts)
    save(output / 'interpretation.json', dict(complete=all(r['available'] and r['observed'] for r in inventory),
         processed_trials=len(results)//4, frozen_models_sha256=digest(root / 'models/frozen_models.npz'),
         note='Descriptive evaluation of historical frozen models, including failed motions with usable measurements. Support-label audit pending; R3 adds context AND refits coefficients with different regularization. Posture and support are confounded by design.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, default=DEFAULT)
    sub = p.add_subparsers(dest='mode', required=True)
    q = sub.add_parser('plan'); q.add_argument('--repetitions', type=int, default=6)
    sub.add_parser('status')
    q = sub.add_parser('run'); q.add_argument('trial'); q.add_argument('--posture', choices=['upright', 'supine'])
    q.add_argument('--clear', action='store_true'); q.add_argument('--execute', action='store_true')
    q = sub.add_parser('observe'); q.add_argument('trial'); q.add_argument('--success', type=int, choices=[0, 1], required=True)
    q.add_argument('--note', required=True)
    q = sub.add_parser('evaluate'); q.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); root = args.session.resolve()
    if args.mode == 'plan':
        print(f'Created {len(plan(root, args.repetitions))} trials in {root}')
    elif args.mode == 'run': run(root, args.trial, args.posture, args.clear, args.execute)
    elif args.mode == 'observe': observe(root, args.trial, args.success, args.note)
    elif args.mode == 'evaluate': evaluate(root, args.out)
    else:
        verify(root)
        for r in rows(root / 'trials.csv'):
            raw = root / 'raw' / r['trial_id']
            state = 'observed' if (raw/'observation.json').exists() else ('inspect attempt' if raw.exists() else 'pending')
            print(r['order'], r['trial_id'], r['posture'], state)


if __name__ == '__main__':
    main()
