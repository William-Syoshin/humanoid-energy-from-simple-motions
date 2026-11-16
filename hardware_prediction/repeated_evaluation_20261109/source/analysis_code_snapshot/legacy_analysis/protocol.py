"""Shared acquisition/processing contract. No hardware I/O.

Missing evidence blocks primary acquisition, never becomes a guessed value.
Diagnostics are retained but cannot be registered as identification data.
"""
from pathlib import Path
import numpy as np
from core import NAMES, SUPPORTS, finite, read, rows, digest, save, compile_recipe


def evidence(item, label):
    if not isinstance(item, dict) or item.get('verified') is not True:
        raise ValueError(label + ': verified evidence required')
    path = Path(item.get('evidence_path') or '')
    if not path.is_file() or item.get('evidence_sha256') != digest(path):
        raise ValueError(label + ': missing or changed evidence file')


def preflight(cfg, recipe, execute):
    errors = []
    def check(fn):
        try: fn()
        except (ValueError, KeyError, TypeError) as exc: errors.append(str(exc))
    def require(ok, message):
        if not ok: raise ValueError(message)
    check(lambda: require(execute, 'Primary trials must log actual acknowledged commands; use --execute'))
    check(lambda: require(cfg.get('power_timestamp_mode','fresh_device') in
                          ('fresh_device','receive_time_user_accepted'),
                          'Declare supported power timestamp mode'))
    if cfg.get('power_timestamp_mode','fresh_device') == 'fresh_device':
        check(lambda: require(cfg.get('required_power_firmware')=='KHR3HV_POWER_V2',
                              'Require fresh-update timestamp firmware'))
    check(lambda: compile_recipe(cfg, recipe))
    check(lambda: require(bool(recipe.get('trial_id')) and bool(recipe.get('success_gate')),
                          'Trial ID and predeclared mechanical success gate required'))
    check(lambda: require(bool(recipe.get('posture')) and recipe.get('support') in SUPPORTS,
                          'Independent posture and valid support category required'))
    check(lambda: require(recipe.get('split') in ('train','validation','test','qc'), 'Declare split in recipe'))
    check(lambda: require(bool(recipe.get('recipe_group')), 'Declare recipe group before acquisition'))
    check(lambda: require(finite(cfg.get('max_initial_error_rad')) and cfg['max_initial_error_rad'] > 0,
                          'Verified initial-position tolerance required'))
    check(lambda: evidence(cfg.get('power_calibration'), 'Power calibration'))
    check(lambda: require(all(finite(cfg['power_calibration'].get(k)) for k in
                              ('voltage_gain','voltage_offset','current_gain','current_offset')),
                          'Set numeric power calibration coefficients'))
    check(lambda: evidence(cfg.get('timing'), 'Timing pilot'))
    check(lambda: evidence(cfg.get('throughput'), 'Throughput pilot'))
    def schema_check():
        schema = cfg['analysis_schema']
        require(schema['feedback'] == cfg['feedback'], 'Feedback schema must match logger')
        require(schema['support'] is True, 'R3 support branch must be recorded for every trial')
        require(schema['tilt'] in ('none','accelerometer','fusion'),
                'Use none, validated accelerometer, or validated fusion for tilt')
        if schema['tilt'] == 'accelerometer':
            evidence(cfg['tilt_calibration'], 'Static accelerometer tilt calibration')
            from orientation import estimate_accelerometer
            estimate_accelerometer([],cfg['tilt_calibration'],cfg['max_gap_s'])
        if schema['tilt'] == 'fusion':
            evidence(cfg['orientation'], 'Orientation validation')
            require(recipe.get('initial_gravity') is not None, 'Per-trial initial gravity required')
            from orientation import estimate
            estimate([],dict(cfg['orientation'],initial_gravity=recipe['initial_gravity']),cfg['max_gap_s'])
        require(finite(cfg['analysis_hz']) and cfg['analysis_hz'] > 0, 'Declare common analysis rate')
        require(finite(cfg['throughput'].get('validated_analysis_hz')), 'Set pilot-validated analysis rate')
        require(cfg['analysis_hz'] <= cfg['throughput']['validated_analysis_hz'], 'Analysis rate exceeds pilot validation')
        require(cfg['throughput']['feedback'] == schema['feedback'], 'Throughput pilot used different feedback branch')
        require(cfg['throughput']['command_hz'] == cfg['command_hz'] and cfg['throughput']['ad_hz'] == cfg['ad_hz'],
                'Throughput pilot used different scheduling rates')
        focused = recipe.get('focused_feedback_joints')
        if focused is None and recipe.get('focused_feedback_joint') is not None:
            focused = [recipe['focused_feedback_joint']]
        if focused:
            require(isinstance(focused, list) and len(focused) <= 2 and len(set(focused)) == len(focused)
                    and all(isinstance(j, int) and 0 <= j < 17 for j in focused),
                    'Focused feedback supports one or two unique joint indices')
            require(finite(cfg.get('focused_feedback_hz')) and cfg['focused_feedback_hz'] > 0,
                    'Focused feedback trial requires focused_feedback_hz')
            require(cfg['throughput'].get('focused_feedback') is True,
                    'Throughput pilot did not validate focused feedback')
            require(cfg['throughput'].get('focused_feedback_hz') == cfg['focused_feedback_hz'],
                    'Focused feedback rate differs from throughput pilot')
        require(cfg['timing']['method'] == 'calibrated_receive', 'Timing method must be calibrated_receive')
        for key in ('receive_latency_s','uncertainty_s','max_receive_jitter_s'):
            require(finite(cfg['timing'].get(key)) and cfg['timing'][key] >= 0, 'Missing timing '+key)
        require(finite(cfg.get('max_alignment_uncertainty_s')), 'Declare allowed timing uncertainty')
        require(cfg['timing']['uncertainty_s'] <= cfg['max_alignment_uncertainty_s'], 'Timing uncertainty exceeds protocol')
        require(finite(cfg['max_energy_difference_fraction']) and 0 < cfg['max_energy_difference_fraction'] < 1,
                'Declare energy-grid discrepancy tolerance')
    check(schema_check)
    return {'ready': not errors, 'errors': errors,
            'purpose': 'primary', 'trial_id': recipe.get('trial_id')}


def validate_primary(cfg, recipe, execute):
    result = preflight(cfg, recipe, execute)
    if not result['ready']: raise ValueError('Primary preflight failed:\n- ' + '\n- '.join(result['errors']))


def stream_stats(data, column):
    if len(data) < 2: return {'samples':len(data), 'valid_time':False}
    t = np.array([float(r[column]) for r in data]); dt = np.diff(t)
    valid = bool(np.isfinite(t).all() and np.all(dt > 0))
    return {'samples':len(data), 'valid_time':valid,
            'rate_hz':float((len(t)-1)/(t[-1]-t[0])) if valid else None,
            'max_gap_s':float(dt.max()) if valid else None}


def audit_raw(raw):
    raw = Path(raw); meta = read(raw/'meta.json'); cfg = read(raw/'config.json'); recipe = read(raw/'recipe.json')
    reasons = []
    if meta.get('status') != 'complete': reasons.append('acquisition_incomplete')
    if meta.get('purpose') != 'primary': reasons.append('diagnostic_or_legacy_capture')
    commands = rows(raw/'commands.csv')
    if not commands: reasons.append('no_recorded_joint_commands')
    elif any(r['ok'] != '1' for r in commands): reasons.append('command_failure')
    if recipe.get('support') not in SUPPORTS: reasons.append('invalid_support_category')
    if not cfg.get('analysis_schema'): reasons.append('analysis_schema_not_frozen')
    if (cfg.get('power_timestamp_mode','fresh_device')=='fresh_device' and
            meta.get('power_timestamp_source')!='fresh_update_observed_ms'):
        reasons.append('power_freshness_and_timestamp_unverified')
    for key in ('power_calibration','timing','throughput'):
        try: evidence(cfg.get(key), key)
        except ValueError: reasons.append(key+'_evidence_missing')
    power = rows(raw/'power.csv'); ad = rows(raw/'ad.csv')
    if any(r.get('ok') != '1' for r in ad): reasons.append('ad_read_failure')
    feedback = rows(raw/'feedback.csv')
    feedback_stats = {}
    if cfg.get('feedback'):
        for joint in range(17):
            channel = [r for r in feedback if r.get('joint') == str(joint) and r.get('ok') == '1']
            feedback_stats[str(joint)] = stream_stats(channel,'pc_s')
            if not channel: reasons.append(f'missing_feedback_joint_{joint}')
        if any(r.get('ok') != '1' for r in feedback): reasons.append('feedback_read_failure')
    return {'raw':str(raw.resolve()), 'trial_id':recipe.get('trial_id'),
            'acquisition_complete':meta.get('status') == 'complete',
            'primary_candidate':not reasons, 'analysis_ready':False,
            'reasons':reasons, 'power':stream_stats(power,'device_s'),
            'ad':stream_stats([r for r in ad if r['ok']=='1'],'pc_s'),
            'feedback':feedback_stats,
            'note':'Analysis-ready requires successful prepare and registration; raw completion alone is insufficient.',
            'hashes':{p.name:digest(p) for p in raw.iterdir() if p.is_file()}}


def audit_session(session, output):
    reports = [audit_raw(p) for p in sorted((Path(session)/'raw').glob('L0_*')) if (p/'meta.json').exists()]
    save(output, {'trials':reports, 'raw_trial_count':len(reports),
                  'primary_candidates':sum(r['primary_candidate'] for r in reports)})
    from core import csv_save
    manifest=rows(Path(session)/'trials.csv')
    for row in manifest:
        matches=[r for r in reports if r['trial_id']==row['trial_id']]
        row['raw_attempt']=''
        if matches:
            row.update(attempted='1',usable='0',success='0',raw_attempt=matches[0]['raw'],
                       exclusion_reason=';'.join(matches[0]['reasons']) or 'pending_prepare_and_mechanical_review')
    csv_save(Path(output).with_suffix('.manifest.csv'),manifest)


def freeze_design(session, output):
    root=Path(session); cfg=read(root/'config.json'); manifest=rows(root/'trials.csv')
    included={}; signatures={'train':set(),'validation':set()}
    import json
    for row in manifest:
        if row.get('exclusion_reason','').startswith('excluded_before_freeze:'): continue
        recipe_path=root/row['recipe']; recipe=read(recipe_path)
        validate_primary(cfg,recipe,True)
        if recipe['trial_id']!=row['trial_id'] or recipe['split']!=row['split']:
            raise ValueError('Recipe and manifest identity mismatch')
        included[recipe['trial_id']]=digest(recipe_path)
        signature=json.dumps(dict(trajectory=compile_recipe(cfg,recipe),
                                  support=recipe['support'],posture=recipe['posture'],
                                  support_intervals=recipe.get('support_intervals',[])),sort_keys=True)
        if recipe['split'] in signatures:signatures[recipe['split']].add(signature)
    if not included: raise ValueError('No approved recipes')
    if signatures['train'] & signatures['validation']: raise ValueError('Train/validation duplicate recipe')
    save(output,dict(config_sha256=digest(root/'config.json'),manifest_sha256=digest(root/'trials.csv'),
                     recipes=included,code={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
                     excluded=[r for r in manifest if r['trial_id'] not in included]))


def verify_design(path, config_path, recipe_path):
    if not path: raise ValueError('Primary acquisition requires --design from freeze')
    frozen=read(path); recipe=read(recipe_path)
    if frozen['config_sha256']!=digest(config_path) or frozen['recipes'].get(recipe['trial_id'])!=digest(recipe_path):
        raise ValueError('Config or recipe differs from frozen design')
    for name,value in frozen['code'].items():
        if digest(Path(__file__).parent/name)!=value: raise ValueError('Code changed after design freeze: '+name)
    return frozen
