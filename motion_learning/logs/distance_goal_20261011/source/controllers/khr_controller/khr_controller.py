"""Phase 1: finite duration position control with complete sensor logging."""
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from controller import Supervisor
from backends.webots_backend import WebotsBackend


def run(robot, output):
    config_path = Path(os.environ.get('KHR_CONFIG', ROOT / 'config/phase1.json'))
    config = json.loads(config_path.read_text())
    for key, variable, convert in [('mode', 'KHR_MODE', str), ('duration_s', 'KHR_DURATION', float)]:
        if variable in os.environ:
            config[key] = convert(os.environ[variable])
    if 'KHR_JOINT' in os.environ:
        config['sweep']['joint'] = os.environ['KHR_JOINT']
    if config['mode'] not in ('hold', 'sweep') or not math.isfinite(config['duration_s']) or config['duration_s'] <= 0:
        raise ValueError('Invalid test mode/duration')
    manifest = json.loads((ROOT / 'config/model_manifest.json').read_text())
    backend = WebotsBackend(robot, config, manifest)
    sweep = config['sweep']
    if config['mode'] == 'sweep':
        j = config['joints'][sweep['joint']]
        a, period = sweep['amplitude_rad'], sweep['period_s']
        if not math.isfinite(a) or a <= 0 or not math.isfinite(period) or period <= 0:
            raise ValueError('Invalid sine wave amplitude/period')
        if j['neutral_rad'] - a < j['safe_min_rad'] or j['neutral_rad'] + a > j['safe_max_rad']:
            raise ValueError('Sweep exceeds configured position range')
        if a * 2 * math.pi / period > config['motor_velocity_rad_s']:
            raise ValueError('Sweep exceeds configured motor velocity')
    metadata = dict(config=config, model=manifest, device_inventory=backend.inventory,
                    all_devices=[{'name': n, 'node_type': d.getNodeType()} for n, d in backend.devices.items()],
                    python=sys.version, platform=platform.platform(),
                    webots_version=os.environ.get('KHR_WEBOTS_VERSION', 'direct GUI launch; see application version'),
                    world_path=robot.getWorldPath(),
                    world_sha256=hashlib.sha256(Path(robot.getWorldPath()).read_bytes()).hexdigest(),
                    proto_sha256=hashlib.sha256((ROOT/'protos/Khr3hv17.proto').read_bytes()).hexdigest())
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Validated {len(backend.names)} motors and position sensors; imu/gyro/accelerometer enabled.', flush=True)
    fields = ['time_s', 'roll_rad', 'pitch_rad', 'yaw_rad', 'gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s',
              'accel_x_m_s2', 'accel_y_m_s2', 'accel_z_m_s2', 'height_m']
    fields += [f'{prefix}/{n}' for prefix in ('q_rad', 'dq_rad_s', 'target_rad') for n in backend.names]
    count, fallen, max_tilt, max_error = 0, False, 0.0, 0.0
    start = robot.getTime()
    first = last = None
    mins = {n: math.inf for n in backend.names}
    maxs = {n: -math.inf for n in backend.names}
    with (output/'sensors.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        steps = math.ceil(config['duration_s'] / backend.dt)
        for i in range(steps):
            targets = {n: config['joints'][n]['neutral_rad'] for n in backend.names}
            if config['mode'] == 'sweep':
                targets[sweep['joint']] += sweep['amplitude_rad'] * math.sin(2 * math.pi * i * backend.dt / sweep['period_s'])
            backend.set_targets(targets)
            state = backend.step()
            if state is None:
                raise RuntimeError('Simulation stopped before test duration completed')
            first = first or state
            last = state
            row = dict(zip(fields[:11], [state.time_s, *state.rpy_rad, *state.gyro_rad_s,
                                        *state.acceleration_m_s2, state.height_m]))
            for n in backend.names:
                q = state.joint_position_rad[n]
                row.update({f'q_rad/{n}': q, f'dq_rad_s/{n}': state.joint_velocity_rad_s[n],
                            f'target_rad/{n}': backend.applied[n]})
                mins[n], maxs[n] = min(mins[n], q), max(maxs[n], q)
                max_error = max(max_error, abs(q - backend.applied[n]))
            writer.writerow(row)
            count += 1
            tilt = max(abs(state.rpy_rad[0]), abs(state.rpy_rad[1]))
            max_tilt = max(max_tilt, tilt)
            fallen = tilt > config['fall']['max_tilt_rad'] or state.height_m < config['fall']['min_height_m']
            if i % max(1, round(1 / backend.dt)) == 0:
                print(f't={state.time_s:.3f}s rpy={tuple(round(x,4) for x in state.rpy_rad)} height={state.height_m:.4f}m', flush=True)
            if fallen:
                break
    if os.environ.get('KHR_SNAPSHOT') == '1':
        robot.exportImage(str(output/'final.png'), 95)
    summary = dict(status='fallen' if fallen else 'passed', samples=count,
                   elapsed_simulation_s=robot.getTime()-start, max_tilt_rad=max_tilt,
                   max_tracking_error_rad=max_error,
                   joint_motion_range_rad={n: maxs[n]-mins[n] for n in backend.names},
                   first_state=asdict(first), last_state=asdict(last))
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('first_state','last_state','joint_motion_range_rad')}), flush=True)
    return 2 if fallen else 0


if __name__ == '__main__':
    robot = Supervisor()
    output = Path(os.environ.get('KHR_LOG_DIR', ROOT/'logs/manual'))
    # Refuse to overwrite a previous experiment, including from GUI restarts.
    if output.exists():
        from datetime import datetime
        output = output.with_name(output.name + '-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True)
    try:
        result = run(robot, output)
    except Exception as exc:
        traceback.print_exc()
        (output/'summary.json').write_text(json.dumps({'status':'error','error':str(exc)}, indent=2)+'\n')
        result = 1
    print(f'Logs: {output}', flush=True)
    if os.environ.get('KHR_QUIT') == '1':
        robot.simulationQuit(result)
        robot.step(int(robot.getBasicTimeStep()))
    else:
        robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
        while robot.step(int(robot.getBasicTimeStep())) != -1:
            pass
