"""Portable motion and measurement contracts. No hardware access on import."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

NAMES = [f'{side}_{j}' for side in ('left','right') for j in
         ('hip_roll','hip_pitch','knee_pitch','ankle_pitch','ankle_roll')]
NAMES += [f'{side}_{j}' for side in ('left','right') for j in
          ('shoulder_pitch','shoulder_roll','elbow_pitch')] + ['head_yaw']

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def save(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def rows(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def finite(v):
    return isinstance(v, (int,float)) and not isinstance(v,bool) and math.isfinite(v)

def motion_arrays(m):
    if m['joint_names'] != NAMES or m['schema'] != 'khr-transfer-v1':
        raise ValueError('Motion coordinate schema mismatch')
    t = np.array([x['t_s'] for x in m['commands']],float)
    q = np.array([x['q_rad'] for x in m['commands']],float)
    initial = np.array(m['initial_pose_rad'],float)
    duration=m['duration_s'];dt=m['control_dt_s']
    if (len(t)<2 or q.shape!=(len(t),17) or initial.shape!=(17,) or
        not all(np.isfinite(a).all() for a in [t,q,initial]) or not finite(duration) or
        not finite(dt) or dt<=0 or duration<=0 or abs(t[0])>1e-9 or
        not np.allclose(np.diff(t),dt,rtol=0,atol=1e-8) or abs(t[-1]+dt-duration)>1e-8):
        raise ValueError('Invalid motion timing/angles')
    return t,q,initial

def validate_config(cfg,m,commissioning=False):
    t,q,initial=motion_arrays(m)
    if commissioning:
        scope=cfg.get('commissioning',{})
        if scope.get('operator_requested') is not True or scope.get('motion_id')!=m.get('motion_id'):
            raise ValueError('Commissioning requires explicit operator request for this motion')
        duration_limit=scope.get('max_duration_s',10.)
        angle_limit=scope.get('max_abs_angle_deg',45.)
        if not finite(duration_limit) or not 0<duration_limit<=13 or not finite(angle_limit) or not 0<angle_limit<=46:
            raise ValueError('Commissioning configured bounds exceed 13 seconds / 46 degrees')
        if m['duration_s']>duration_limit or np.max(abs(np.r_[initial[None,:],q]))>np.deg2rad(angle_limit):
            raise ValueError('Commissioning duration/angle limit exceeded')
        if np.max(abs(np.diff(np.r_[initial[None,:],q],axis=0))/m['control_dt_s'])>.65:
            raise ValueError('Commissioning speed limit exceeded')
    if (cfg.get('mapping_verified') is not True and not commissioning) or cfg.get('mixing_off_verified') is not True:
        raise ValueError('Verify actual simulator-to-servo mapping and disable robot mixing')
    if [j['name'] for j in cfg['joints']]!=NAMES: raise ValueError('Joint order mismatch')
    if len({(j['id'],j['sio']) for j in cfg['joints']})!=17: raise ValueError('Duplicate servo channel')
    positions=[];initial_pos=[]
    for k,j in enumerate(cfg['joints']):
        if j['sign'] not in (-1,1) or not all(finite(j.get(x)) for x in
                ['zero_pos','units_per_rad','min_rad','max_rad','max_speed_rad_s']):
            raise ValueError('Incomplete calibration: '+j['name'])
        if not (isinstance(j['id'],int) and 0<=j['id']<=31 and j['sio'] in (1,2)):
            raise ValueError('Invalid servo address')
        if j['units_per_rad']<=0 or j['max_speed_rad_s']<=0 or j['min_rad']>=j['max_rad']:
            raise ValueError('Invalid joint bounds')
        values=np.r_[initial[k],q[:,k]]
        if values.min()<j['min_rad'] or values.max()>j['max_rad']:
            raise ValueError('Joint range violation: '+j['name'])
        speeds=abs(np.diff(values))/m['control_dt_s']
        if speeds.max()>j['max_speed_rad_s']: raise ValueError('Joint speed violation: '+j['name'])
        p=np.rint(j['zero_pos']+j['sign']*values*j['units_per_rad']).astype(int)
        if p.min()<3500 or p.max()>11500: raise ValueError('Servo range violation')
        initial_pos.append(int(p[0]));positions.append(p[1:])
    for key in ['max_current_A','min_voltage_V','power_stale_s','max_initial_error_rad',
                'max_command_lateness_s','max_power_gap_s']:
        if not finite(cfg.get(key)) or cfg[key]<=0: raise ValueError('Set positive '+key)
    if cfg['max_command_lateness_s']>=m['control_dt_s']: raise ValueError('Lateness must be less than command interval')
    if cfg['robot_port']==cfg['power_port']: raise ValueError('Robot and power ports must differ')
    cal=cfg['power_calibration'];timing=cfg['timing']
    if cal.get('verified') is not True or not all(finite(cal.get(k)) for k in
        ['voltage_gain','voltage_offset','current_gain','current_offset']):
        raise ValueError('Power calibration required')
    if cal['voltage_gain']<=0 or cal['current_gain']==0: raise ValueError('Invalid power calibration')
    if timing.get('verified') is not True or not all(finite(timing.get(k)) and timing[k]>=0 for k in ['receive_latency_s','uncertainty_s']):
        raise ValueError('Measured receive latency and timing uncertainty required')
    if cfg.get('power_timestamp_contract') not in ('fresh_device','legacy_receive'):
        raise ValueError('Select power timestamp contract')
    if not isinstance(cfg['servo_frame'],int) or not 1<=cfg['servo_frame']<=255: raise ValueError('Invalid servo frame')
    return np.array(positions).T.tolist(),initial_pos

def parse_power(line):
    line=line.strip()
    if not line or line.startswith('#') or line.lower().startswith('timestamp_ms'): return None
    fields=line.split(',')
    if len(fields) not in (3,4): raise ValueError('Malformed power line: '+line)
    ts,v,i=map(float,fields[:3])
    if not all(math.isfinite(x) for x in [ts,v,i]) or ts<0: raise ValueError('Nonfinite power sample')
    if len(fields)==4 and fields[3] not in ('0','1'): raise ValueError('Invalid clipping flag')
    return dict(device_s=ts/1000,voltage_raw=v,current_raw=i,clip=int(fields[3]) if len(fields)==4 else int(abs(i)>=9.5))

def integrate_power(samples,start,end,cfg):
    if not finite(start) or not finite(end) or end<=start: raise ValueError('Invalid measurement window')
    if len(samples)<2: raise ValueError('Missing power samples')
    t=np.array([float(x['pc_s']) for x in samples])-cfg['timing']['receive_latency_s']
    device=np.array([float(x['device_s']) for x in samples]);v=np.array([float(x['voltage_V']) for x in samples]);i=np.array([float(x['current_A']) for x in samples])
    if not all(np.isfinite(a).all() for a in [t,device,v,i]) or np.any(np.diff(t)<=0) or np.any(np.diff(device)<=0):
        raise ValueError('Nonmonotonic/reset/missing power clock')
    if start<t[0] or end>t[-1]: raise ValueError('Power samples do not bracket measurement window')
    keep=(t>=start)&(t<=end);a=max(0,np.searchsorted(t,start)-1);b=min(len(t),np.searchsorted(t,end)+1)
    if np.max(np.diff(t[a:b]))>cfg['max_power_gap_s'] or np.max(np.diff(device[a:b]))>cfg['max_power_gap_s']: raise ValueError('Power sampling gap')
    if any(int(x['clip']) for x in samples[a:b]): raise ValueError('Clipped power samples')
    power=v*i;grid=np.r_[start,t[keep],end];grid=np.unique(grid)
    value=float(np.trapezoid(np.interp(grid,t,power),grid))
    return dict(energy_J=value,mean_power_W=value/(end-start),duration_s=end-start,
                timing_uncertainty_s=cfg['timing']['uncertainty_s'],negative_power_samples=int((power[a:b]<0).sum()))
