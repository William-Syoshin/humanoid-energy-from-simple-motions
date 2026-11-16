"""Experiment preparation and processing; no hardware imports or I/O on import."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

NAMES = [f'{s}_{j}' for s in ('left', 'right') for j in
         ('hip_roll', 'hip_pitch', 'knee_pitch', 'ankle_pitch', 'ankle_roll')]
NAMES += [f'{s}_{j}' for s in ('left', 'right') for j in
          ('shoulder_pitch', 'shoulder_roll', 'elbow_pitch')] + ['head_yaw']
SUPPORTS = ['feet', 'ground', 'fixture', 'object', 'other']

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)

def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def csv_save(path, data):
    if not data:
        raise ValueError('No rows to save')
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)

def digest(path):
    path = Path(path)
    for attempt in range(6):
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.2)

def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

def configuration():
    ids = [6, 7, 8, 9, 10, 6, 7, 8, 9, 10, 1, 2, 4, 1, 2, 4, 0]
    return {'rcb_library': 'C:/Users/BIRL-/OneDrive/Desktop/khr3hv_exploration/Rcb4Lib',
            'robot_port': 'COM3', 'robot_baud': 1250000, 'power_port': 'COM4',
            'power_baud': 115200, 'command_hz': 10, 'ad_hz': 10,
            'feedback': False, 'feedback_hz': 1, 'mixing_off_verified': False,
            'mapping_verified': False, 'max_initial_error_rad': None,
            'max_current_A': None, 'min_voltage_V': None,
            'power_stale_s': 1.0, 'max_gap_s': 0.25,
            'required_power_firmware': 'KHR3HV_POWER_V2',
            'analysis_hz': 10, 'max_energy_difference_fraction': 0.01,
            'focused_feedback_hz': 10,
            'velocity_smoothing_window_s': 0.4,
            'max_alignment_uncertainty_s': None,
            'analysis_schema': {'feedback': False, 'tilt': 'none', 'support': True},
            'timing': {'verified': False, 'method': 'calibrated_receive',
                       'receive_latency_s': None, 'uncertainty_s': None,
                       'max_receive_jitter_s': None, 'evidence_path': None, 'evidence_sha256': None},
            'throughput': {'verified': False, 'validated_analysis_hz': None,
                          'feedback': False, 'command_hz': 10, 'ad_hz': 10,
                          'evidence_path': None, 'evidence_sha256': None},
            'power_calibration': {'verified': False, 'voltage_gain': None,
                'voltage_offset': None, 'current_gain': None, 'current_offset': None},
            'alignment': {'scale': None, 'offset_s': None, 'residual_s': None,
                          'method': None},
            'orientation': {'verified': False, 'accel_zero': None, 'accel_per_g': None,
                'gyro_zero': None, 'gyro_rad_s_per_count': None,
                'accel_body_xy_matrix': None, 'gyro_body_xy_matrix': None,
                'initial_gravity': None, 'correction_rate_s': None,
                'accel_norm_tolerance': None},
            'tilt_calibration': {'verified': False, 'ad3_zero': None,
                'ad4_zero': None, 'ad3_per_g': None, 'ad4_per_g': None,
                'note': 'Quasi-static gravity inclination only; validate dynamically before R3.'},
            'joints': [{'name': n, 'id': i, 'sio': 2 if n.startswith('right') else 1,
                'zero_pos': 7500, 'sign': None, 'min_rad': None, 'max_rad': None,
                'max_speed_rad_s': None} for n, i in zip(NAMES, ids)]}

def initialize(path):
    path = Path(path)
    path.mkdir(exist_ok=False)
    for folder in ('raw', 'aligned', 'calibration', 'videos', 'recipes'):
        (path / folder).mkdir()
    save(path / 'config.json', configuration())
    from design import planned_recipes
    manifest = []
    for recipe in planned_recipes():
        tid=recipe['trial_id']
        manifest.append(dict(trial_id=tid,file=f'{tid}.csv',split=recipe['split'],
            task=recipe['level'],level=recipe['level'],attempted=0,usable=0,success=0,
            recipe=f'recipes/{tid}.json',exclusion_reason='pending'))
        save(path/'recipes'/f'{tid}.json',recipe)
    csv_save(path / 'trials.csv', manifest)

def compile_recipe(config, recipe):
    if recipe.get('approved') is not True:
        raise ValueError('Recipe is not approved')
    if not config['mapping_verified'] or not config['mixing_off_verified']:
        raise ValueError('Verify hardware mapping and mixing settings')
    if [j['name'] for j in config['joints']] != NAMES:
        raise ValueError('Joint column order mismatch')
    for j in config['joints']:
        if j['sign'] not in (-1,1) or not all(finite(j[k]) for k in
                ('zero_pos','min_rad','max_rad','max_speed_rad_s')):
            raise ValueError(f'Incomplete joint calibration: {j["name"]}')
        if j['min_rad'] >= j['max_rad'] or j['max_speed_rad_s'] <= 0:
            raise ValueError('Invalid joint limits')
    hz, duration = config['command_hz'], recipe['duration_s']
    if not finite(hz) or not 0 < hz <= 50 or not finite(duration) or duration <= 0:
        raise ValueError('Invalid command rate or duration')
    if recipe['support'] is not None and recipe['support'] not in SUPPORTS:
        raise ValueError('Unknown support category')
    for k in ('pre_s','post_s','settle_s'):
        if not finite(recipe[k]) or recipe[k] < 0:
            raise ValueError('Invalid timing')
    t = np.linspace(0, duration, math.ceil(duration*hz)+1)
    center = np.array(recipe['center_rad'], dtype=float)
    if center.shape != (17,) or not np.isfinite(center).all():
        raise ValueError('All 17 center angles are required')
    q = np.tile(center, (len(t),1))
    if recipe['kind'] == 'sine':
        ramp = min(1.0, duration/2)
        envelope = np.sin(np.minimum(np.minimum(t, duration-t)/ramp,1)*np.pi/2)**2
        seen = set()
        for e in recipe['excitation']:
            index = NAMES.index(e['joint'])
            if index in seen: raise ValueError('Duplicate excitation joint')
            seen.add(index)
            a,f,p = (e[k] for k in ('amplitude_rad','frequency_Hz','phase_rad'))
            if not all(finite(v) for v in (a,f,p)) or a < 0 or f <= 0 or f >= hz/2:
                raise ValueError('Invalid excitation')
            q[:,index] += envelope*a*np.sin(2*np.pi*f*t+p)
    elif recipe['kind'] == 'frames':
        frames = recipe['frames']
        ft = np.array([f['t_s'] for f in frames], dtype=float)
        fq = np.array([f['q_rad'] for f in frames], dtype=float)
        if len(ft)<2 or fq.shape != (len(ft),17) or not np.isfinite(fq).all() or not np.isfinite(ft).all():
            raise ValueError('Complete finite frames required')
        if ft[0]!=0 or ft[-1]!=duration or np.any(np.diff(ft)<=0) or not np.allclose(fq[0], center):
            raise ValueError('Frames must cover duration and start at center')
        for i,j in enumerate(config['joints']):
            if np.max(abs(np.diff(fq[:,i])/np.diff(ft))) > j['max_speed_rad_s']:
                raise ValueError('Frame speed violation')
            if np.any(fq[:,i]<j['min_rad']) or np.any(fq[:,i]>j['max_rad']):
                raise ValueError('Frame range violation')
        for i in range(17): q[:,i] = np.interp(t,ft,fq[:,i])
    elif recipe['kind'] != 'hold':
        raise ValueError('Supported kinds: hold, sine, frames')
    for i,j in enumerate(config['joints']):
        pos = j['zero_pos'] + j['sign']*q[:,i]*4000/math.radians(135)
        if np.any(q[:,i]<j['min_rad']) or np.any(q[:,i]>j['max_rad']) or np.any(pos<3500) or np.any(pos>11500):
            raise ValueError(f'Range violation: {j["name"]}')
        if np.max(abs(np.diff(q[:,i])/np.diff(t))) > j['max_speed_rad_s']:
            raise ValueError(f'Speed violation: {j["name"]}')
    return [dict(t_s=float(ti), **{f'cmd_{i}':float(v) for i,v in enumerate(qi)}) for ti,qi in zip(t,q)]

def linear_calibration(path, out):
    data = rows(path)
    x = np.array([float(r['sensor']) for r in data])
    y = np.array([float(r['reference']) for r in data])
    if len(x)<3 or len(set(x))<2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('At least 3 finite points at >=2 levels required')
    gain,offset = np.linalg.lstsq(np.c_[x,np.ones(len(x))], y,rcond=None)[0]
    save(out, dict(gain=float(gain),offset=float(offset),
        residual_rmse=float(np.sqrt(np.mean((gain*x+offset-y)**2))),
        source_sha256=digest(path), verified=False))

def interpolate(t, x, target, max_gap):
    t,x,target = np.asarray(t),np.asarray(x),np.asarray(target)
    if len(t)<2 or not np.isfinite(t).all() or not np.isfinite(x).all() or np.any(np.diff(t)<=0):
        raise ValueError('Invalid or missing samples')
    if target[0]<t[0] or target[-1]>t[-1]: raise ValueError('No extrapolation allowed')
    k = np.clip(np.searchsorted(t,target,side='right'),1,len(t)-1)
    if np.any(t[k]-t[k-1]>max_gap): raise ValueError('Dropout exceeds configured bound')
    return np.interp(target,t,x)

def prepare(raw, config_path, out):
    """Compatibility entry point; all processing uses the primary contract."""
    from processing import prepare as validated_prepare
    return validated_prepare(raw, config_path, out)

def sensor_summary(raw, out):
    data = [r for r in rows(Path(raw)/'ad.csv') if r['ok']=='1']
    if not data: raise ValueError('No successful AD reads')
    report={}
    for k in range(1,5):
        values=np.array([int(r[f'ad{k}']) for r in data])
        if k<=2: values=np.where(values>=32768,values-65536,values)
        report[f'ad{k}']={'min':float(values.min()),'max':float(values.max()),
            'mean':float(values.mean()),'std':float(values.std()),
            'interpretation':'signed RAM value, physical gain unverified' if k<=2 else 'unsigned RAM value'}
    t=np.array([float(r['pc_s']) for r in data])
    report.update(samples=len(data),rate_Hz=float((len(t)-1)/(t[-1]-t[0])) if len(t)>1 else None)
    save(out,report)

def register(session, raw, processed, success, reason, out_manifest, manifest_path=None):
    """Produce a new manifest revision; preserve previous attempts and source files."""
    import shutil
    session,raw,processed=map(Path,(session,raw,processed))
    meta,recipe,qc=read(raw/'meta.json'),read(raw/'recipe.json'),read(processed/'qc.json')
    if meta['status']!='complete': raise ValueError('Cannot register incomplete measurement')
    if qc.get('usable') is not True or qc.get('aligned_sha256')!=digest(processed/'aligned.csv'):
        raise ValueError('Missing QC approval or changed aligned data')
    if any(not (raw/name).is_file() or digest(raw/name)!=h for name,h in qc['sources'].items()):
        raise ValueError('Processed data does not match raw attempt')
    manifest=rows(manifest_path or session/'trials.csv')
    matches=[r for r in manifest if r['trial_id']==recipe['trial_id']]
    if len(matches)!=1: raise ValueError('Unknown trial ID')
    row=matches[0]
    if row['usable']=='1': raise ValueError('Already registered; preserve attempt')
    if row.get('raw_attempt') and Path(row['raw_attempt']).resolve()!=raw.resolve():
        raise ValueError('Different raw attempt already associated with this ID; allocate a new attempt ID')
    if row['split']!=recipe['split']: raise ValueError('Recipe split differs from manifest')
    if success not in (0,1): raise ValueError('Mechanical outcome must be 0 or 1')
    target=session/'aligned'/row['file']
    if target.resolve().parent != (session/'aligned').resolve(): raise ValueError('Invalid destination')
    if target.exists() or Path(out_manifest).exists(): raise ValueError('Destination already exists')
    row.update(attempted='1',usable='1',success=str(int(success)),exclusion_reason=reason,
               raw_attempt=str(raw.resolve()))
    for r in manifest: r.setdefault('raw_attempt','')
    shutil.copyfile(processed/'aligned.csv',target)
    save(target.with_suffix('.qc.json'),qc)
    csv_save(out_manifest,manifest)

def repeatability(paths, out):
    energies=[read(Path(p)/'qc.json')['energy_J'] for p in paths]
    if len(energies)<3 or not all(finite(e) and e>0 for e in energies):
        raise ValueError('At least three positive-energy QC trials required')
    mean=float(np.mean(energies)); sd=float(np.std(energies,ddof=1))
    save(out,dict(energies_J=energies,mean_J=mean,std_J=sd,cv_pct=100*sd/mean,
                  investigate=sd/mean>.05, note='Compare only identical recipes and scoring windows'))
