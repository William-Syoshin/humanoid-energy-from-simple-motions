"""Configuration and provenance shared by training and evaluation."""
import copy
import hashlib
import json
import math
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path or ROOT / 'rl/config.yaml')
    cfg = yaml.safe_load(path.read_text())
    validate_config(cfg)
    return copy.deepcopy(cfg)


def validate_config(cfg):
    if cfg['condition'] not in ('task_only', 'R1', 'R2', 'R3'):
        raise ValueError('condition must be task_only, R1, R2 or R3')
    s = cfg['simulation']
    for name in ('physics_ms','control_ms','action_ms'):
        if not isinstance(s[name], int) or s[name] <= 0:
            raise ValueError(name+' must be a positive integer')
    if s['control_ms'] % s['physics_ms'] or s['action_ms'] % s['control_ms']:
        raise ValueError('Timesteps must be integer multiples')
    if s['action_ms'] != 100:
        raise ValueError('Frozen hardware predictors require a 100 ms analysis grid')
    if not math.isfinite(cfg['energy']['weight']) or cfg['energy']['weight'] < 0:
        raise ValueError('Energy weight must be nonnegative')
    if not 0 < s['motor_torque_fraction'] <= 1 or not 0 < s['motor_velocity_rad_s'] <= 7.48:
        raise ValueError('Servo setting exceeds the pinned model limits')
    for key in ('episode_s','settle_s','startup_timeout_s','socket_timeout_s'):
        if not math.isfinite(s[key]) or s[key] <= 0:
            raise ValueError(key+' must be positive and finite')
    for key, tick in [('episode_s', s['action_ms']/1000), ('settle_s',s['control_ms']/1000)]:
        if not math.isclose(s[key]/tick,round(s[key]/tick),abs_tol=1e-8):
            raise ValueError(key+' must be an integer multiple of its timestep')
    for key in ('initial_yaw_std_rad','sensor_joint_std_rad','sensor_accel_std_m_s2','friction'):
        if not math.isfinite(s[key]) or s[key] < 0: raise ValueError(key+' must be finite and nonnegative')
    if not isinstance(s['command_latency_steps'], int) or s['command_latency_steps'] < 0:
        raise ValueError('command_latency_steps must be a nonnegative integer')
    if len(s['pid']) != 3 or any(not math.isfinite(x) or x < 0 for x in s['pid']):
        raise ValueError('PID must be three finite nonnegative numbers')
    g = cfg['gait']
    if any(len(g[k]) != 5 for k in ('nominal','lower','upper','slew_per_s')):
        raise ValueError('Gait requires exactly five parameters')
    for lo,x,hi,slew in zip(g['lower'],g['nominal'],g['upper'],g['slew_per_s']):
        if not all(math.isfinite(v) for v in (lo,x,hi,slew)) or not lo <= x <= hi or lo >= hi or slew <= 0:
            raise ValueError('Invalid gait range/nominal/slew')
    if g['lower'][0] <= 0 or g['ramp_s'] <= 0 or not 0 < g['joint_limit_rad'] <= 2.356:
        raise ValueError('Invalid gait period/ramp/joint bound')
    tr = cfg['training']
    for key in ('total_timesteps','n_steps','batch_size','n_epochs','torch_threads','model_save_interval'):
        if not isinstance(tr[key], int) or tr[key] <= 0: raise ValueError(key+' must be a positive integer')
    if tr['batch_size'] < 2 or tr['n_steps'] % tr['batch_size']:
        raise ValueError('batch_size must be >=2 and divide n_steps for this single-env experiment')
    if not 0 < tr['gamma'] <= 1 or not 0 < tr['gae_lambda'] <= 1:
        raise ValueError('Invalid discount/GAE value')
    if not math.isfinite(tr['learning_rate']) or tr['learning_rate'] <= 0:
        raise ValueError('learning_rate must be positive')
    if cfg['task']['target_speed_m_s'] <= 0 or cfg['task']['speed_tolerance_m_s'] <= 0:
        raise ValueError('Speed target/tolerance must be positive')
    mode = cfg['task'].get('mode', 'speed')
    if mode not in ('speed', 'distance'):
        raise ValueError('Unknown task mode')
    if mode == 'distance':
        target = cfg['task']['target_distance_m']
        if not math.isfinite(target) or target <= 0:
            raise ValueError('Target distance must be positive and finite')
        if tr['gamma'] != 1.0:
            raise ValueError('Distance experiment uses undiscounted episodic energy (gamma=1)')
        for key in ('alive', 'speed', 'upright'):
            if cfg['reward'][key] != 0:
                raise ValueError('Distance experiment must not reward speed or surviving longer')
        for key in ('goal', 'timeout', 'fall', 'progress', 'time'):
            if not math.isfinite(cfg['reward'][key]) or cfg['reward'][key] < 0:
                raise ValueError('Distance reward weights must be finite and nonnegative')



def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def provenance():
    paths = ['rl/config.yaml', 'protos/Khr3hv17RL.proto',
             'worlds/khr_walk.wbt', 'config/joint_mapping.json',
             'energy/assets/frozen_models.npz', 'energy/assets/frozen.json',
             'energy/assets/training_bounds.npz']
    for directory in ('rl','env','energy','backends','controllers/khr_rl_controller'):
        paths.extend(str(p.relative_to(ROOT)) for p in (ROOT/directory).glob('*.py'))
    return {p: sha256(ROOT / p) for p in paths}

