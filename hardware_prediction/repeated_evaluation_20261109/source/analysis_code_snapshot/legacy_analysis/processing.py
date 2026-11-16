"""Validated primary processing: per-attempt timing, area-preserving power grid."""
from pathlib import Path
import math
import numpy as np
from core import read, rows, save, csv_save, digest, interpolate, SUPPORTS
from protocol import validate_primary, stream_stats

integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def smooth_time_window(measured, analysis_hz, window_s):
    """Centered moving average with a sampling-rate independent duration."""
    samples = max(1, int(round(window_s * analysis_hz)))
    if samples % 2 == 0:
        samples += 1
    radius = samples // 2
    padded = np.pad(measured, ((radius, radius), (0, 0)), mode='edge')
    return sum(padded[k:k+len(measured)] for k in range(samples)) / samples, samples


def power_grid(t, power, target, max_gap):
    """Average raw P=VI over each node's temporal cell (box integration).

    Cell widths equal trapezoidal weights on a uniform grid, so the integral
    is preserved, including exact scoring boundaries. VI display channels
    remain separate from this explicitly averaged power target.
    """
    interpolate(t, power, target, max_gap)
    boundaries = np.r_[target[0], (target[:-1]+target[1:])/2, target[-1]]
    result = []
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        tt = np.r_[a,t[(t>a)&(t<b)],b]
        pp = interpolate(t,power,tt,max_gap)
        result.append(float(integrate(pp,tt)/(b-a)))
    return np.array(result)


def prepare(raw, config_path, out):
    raw, out = Path(raw), Path(out)
    cfg, meta, recipe = read(config_path), read(raw/'meta.json'), read(raw/'recipe.json')
    if digest(config_path) != digest(raw/'config.json'):
        raise ValueError('Use immutable acquisition config; revised calibration requires an explicit new provenance workflow')
    validate_primary(cfg,recipe,meta.get('execute',False))
    if meta.get('purpose') != 'primary' or meta.get('status') != 'complete':
        raise ValueError('Only complete primary acquisitions can be prepared')
    if cfg.get('power_timestamp_mode','fresh_device')=='fresh_device' and (
            meta.get('power_firmware')!=cfg.get('required_power_firmware') or
            meta.get('power_timestamp_source')!='fresh_update_observed_ms'):
        raise ValueError('Unverified firmware timestamp/freshness semantics')
    frozen=read(raw/'design.json')
    if frozen.get('config_sha256')!=digest(raw/'config.json') or frozen.get('recipes',{}).get(recipe['trial_id'])!=digest(raw/'recipe.json'):
        raise ValueError('Raw configuration or recipe differs from saved frozen design')
    p = rows(raw/'power.csv')
    device = np.array([float(r['device_s']) for r in p]); received = np.array([float(r['pc_s']) for r in p])
    if not stream_stats(p,'device_s')['valid_time'] or not stream_stats(p,'pc_s')['valid_time']:
        raise ValueError('Power clock reset, duplicate or nonmonotonic timestamps')
    timing = cfg['timing']
    # Offset is measured anew for every connection. Only independently calibrated
    # transport latency is subtracted; this never aligns to prediction or test power.
    offsets = received-device
    offset = float(np.median(offsets)-timing['receive_latency_s'])
    jitter = float(np.max(abs(offsets-np.median(offsets))))
    if jitter > timing['max_receive_jitter_s']: raise ValueError('Receive timing exceeds pilot jitter bound')
    t = device+offset
    cal = cfg['power_calibration']
    voltage = np.array([float(r['voltage']) for r in p])*cal['voltage_gain']+cal['voltage_offset']
    current = np.array([float(r['current']) for r in p])*cal['current_gain']+cal['current_offset']
    if not np.isfinite(voltage).all() or not np.isfinite(current).all() or np.any(voltage<=0):
        raise ValueError('Invalid calibrated power channels')
    a,b = meta['score_start'],meta['score_end']
    if not math.isfinite(a) or not math.isfinite(b) or b <= a: raise ValueError('Invalid scoring window')
    target = np.linspace(a,b,int(math.ceil((b-a)*cfg['analysis_hz']))+1)
    if len(target)<7: raise ValueError('Too few analysis points')
    v=interpolate(t,voltage,target,cfg['max_gap_s']); i=interpolate(t,current,target,cfg['max_gap_s'])
    rawp = voltage*current
    pp=power_grid(t,rawp,target,cfg['max_gap_s'])
    clipped=np.r_[a,t[(t>a)&(t<b)],b]
    energy=float(integrate(interpolate(t,rawp,clipped,cfg['max_gap_s']),clipped))
    aligned_energy=float(integrate(pp,target))
    difference=abs(aligned_energy-energy)/max(abs(energy),1e-12)
    if difference>cfg['max_energy_difference_fraction']: raise ValueError('Energy changed during resampling')
    commands=rows(raw/'commands.csv')
    if not commands or any(r['ok']!='1' for r in commands): raise ValueError('Missing or failed commands')
    ct=np.array([float(r['response_s']) for r in commands])
    cq=np.array([[float(r[f'cmd_{j}']) for j in range(17)] for r in commands])
    if not np.isfinite(cq).all() or not np.isfinite(ct).all() or np.any(np.diff(ct)<=0) or ct[0]>a:
        raise ValueError('Commands do not cover scoring start')
    qcmd=cq[np.searchsorted(ct,target,side='right')-1]
    data=[dict(t_s=float(tt-a),voltage_V=float(vv),current_A=float(ii),power_W=float(pw),
               **{f'cmd_{j}':float(q[j]) for j in range(17)})
          for tt,vv,ii,pw,q in zip(target,v,i,pp,qcmd)]
    schema=cfg['analysis_schema']
    feedback_stats={}
    velocity_window_s = cfg.get('velocity_smoothing_window_s', 0.4)
    velocity_window_samples = None
    if schema['feedback']:
        feedback=rows(raw/'feedback.csv')
        for j in range(17):
            channel=[r for r in feedback if int(r['joint'])==j and r['ok']=='1']
            q=interpolate([float(r['pc_s']) for r in channel],[float(r['q_rad']) for r in channel],target,cfg['max_gap_s'])
            feedback_stats[str(j)]=stream_stats(channel,'pc_s')
            for row,value in zip(data,q): row[f'q_{j}']=float(value)
        # Persist the exact kinematics used by R1-R3 so the model inputs are
        # inspectable and reproducible without silently differentiating again.
        measured=np.array([[row[f'q_{j}'] for j in range(17)] for row in data])
        if not math.isfinite(velocity_window_s) or velocity_window_s <= 0:
            raise ValueError('Set positive velocity_smoothing_window_s')
        smooth, velocity_window_samples = smooth_time_window(
            measured, cfg['analysis_hz'], velocity_window_s)
        velocity=np.gradient(smooth,target,axis=0)
        tracking_error=qcmd-measured
        for k,row in enumerate(data):
            for j in range(17):
                row[f'dq_{j}']=float(velocity[k,j])
                row[f'e_{j}']=float(tracking_error[k,j])
    if schema['tilt']=='accelerometer':
        from orientation import estimate_accelerometer
        ad=[r for r in rows(raw/'ad.csv') if r['ok']=='1']
        orientation=estimate_accelerometer(ad,cfg['tilt_calibration'],cfg['max_gap_s'])
        for key in ('tilt_rad','pitch_rad','roll_rad','accel_forward_g','accel_right_g'):
            values=interpolate([r['pc_s'] for r in orientation],[r[key] for r in orientation],target,cfg['max_gap_s'])
            for row,value in zip(data,values): row[key]=float(value)
    elif schema['tilt']=='fusion':
        from orientation import estimate
        calibration=dict(cfg['orientation'],initial_gravity=recipe['initial_gravity'])
        ad=[r for r in rows(raw/'ad.csv') if r['ok']=='1']
        orientation=estimate(ad,calibration,cfg['max_gap_s'])
        tilt=interpolate([r['pc_s'] for r in orientation],[r['tilt_rad'] for r in orientation],target,cfg['max_gap_s'])
        for row,value in zip(data,tilt): row['tilt_rad']=float(value)
    intervals=recipe.get('support_intervals',[])
    if intervals:
        end=0.0
        for interval in intervals:
            if interval['start_s']!=end or interval['end_s']<=end or interval['support'] not in SUPPORTS:
                raise ValueError('Support intervals must be sorted and cover the whole window without gaps/overlap')
            end=interval['end_s']
        if not np.isclose(end,b-a,atol=1e-6,rtol=0): raise ValueError('Support intervals do not match scoring duration')
    for row in data:
        row['support']=recipe['support']
        for interval in intervals:
            if interval['start_s']<=row['t_s']<=interval['end_s']: row['support']=interval['support']
    out.mkdir(parents=True,exist_ok=False)
    csv_save(out/'aligned.csv',data)
    save(out/'qc.json',dict(usable=True,energy_J=energy,aligned_energy_J=aligned_energy,
         energy_difference_fraction=difference,duration_s=b-a,analysis_hz=float(1/(target[1]-target[0])),
         power=stream_stats(p,'device_s'),feedback=feedback_stats,analysis_schema=schema,
         joint_velocity_definition=(f'dq_j is np.gradient of a centered {velocity_window_s:g} s '
             f'({velocity_window_samples} sample) edge-padded moving average of q_j') if schema['feedback'] else None,
         tracking_error_definition='e_j = cmd_j - q_j on the aligned grid',
         clock_offset_s=offset,receive_jitter_s=jitter,alignment_uncertainty_s=timing['uncertainty_s'],
         power_definition='Native calibrated VI, averaged over grid temporal cells; power_W is authoritative',
         mechanical_success=None,aligned_sha256=digest(out/'aligned.csv'),
         sources={p.name:digest(p) for p in raw.iterdir() if p.is_file()},
         processing_config_sha256=digest(config_path)))
