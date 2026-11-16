"""Synchronous simulator worker. Standard library only; no ML in Webots."""
from collections import deque
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback
from controller import Supervisor, Node

ROOT = Path(os.environ.get('KHR_RL_ROOT', Path(__file__).resolve().parents[2]))


def main():
    robot = Supervisor()
    if 'KHR_RL_CONFIG' not in os.environ:
        print('Launch this world using .venv/bin/python rl/evaluate.py --baseline --gui', flush=True)
        robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
        return
    cfg = json.loads(Path(os.environ['KHR_RL_CONFIG']).read_text())
    sim = cfg['simulation']
    mapping = json.loads((ROOT / 'config/joint_mapping.json').read_text())['joints']
    names = [j['motor'] for j in mapping]
    signs = [j['webots_sign'] for j in mapping]
    motors = [robot.getDevice(n) for n in names]
    sensors = [robot.getDevice(n + '_sensor') for n in names]
    for m, s in zip(motors, sensors):
        if m is None or m.getNodeType() != Node.ROTATIONAL_MOTOR or s is None:
            raise ValueError('Missing motor or position sensor')
        if m.getPositionSensor().getName() != s.getName():
            raise ValueError('Sensor attachment mismatch')
        s.enable(sim['control_ms'])
    imu, gyro, accel = [robot.getDevice(n) for n in ('imu', 'gyro', 'accelerometer')]
    for device in (imu, gyro, accel):
        device.enable(sim['control_ms'])
    body = robot.getSelf()
    joints = [body.getFromProtoDef('KHR3HV_' + n) for n in names]
    if not all(joints):
        raise RuntimeError('Cannot access internal joint nodes for reset')
    feet = [body.getFromProtoDef('KHR3HV_'+n).getField('endPoint').getSFNode()
            for n in ('LeftFoot','RightFoot')]
    body.enableContactPointsTracking(sim['control_ms'], True)
    body.saveState('initial')
    rng = random.Random(0)
    pending = deque()
    applied = [0.] * 17
    video_guides_added = False

    def configure():
        for m in motors:
            m.setVelocity(sim['motor_velocity_rad_s'])
            m.setAvailableTorque(m.getMaxTorque() * sim['motor_torque_fraction'])
            m.setControlPID(*sim['pid'])

    def targets(q):
        nonlocal applied
        if len(q) != 17 or not all(math.isfinite(x) and abs(x) <= cfg['gait']['joint_limit_rad'] for x in q):
            raise ValueError('Invalid target vector')
        pending.append(list(q))
        applied = pending.popleft()
        for m, x, sign in zip(motors, applied, signs):
            raw = x * sign
            if not m.getMinPosition() <= raw <= m.getMaxPosition():
                raise ValueError('Target exceeds model bounds')
            m.setPosition(raw)

    def state():
        q = [s.getValue() * sign + rng.gauss(0, sim['sensor_joint_std_rad']) for s, sign in zip(sensors, signs)]
        acc = [x + rng.gauss(0, sim['sensor_accel_std_m_s2']) for x in accel.getValues()]
        value = dict(time=robot.getTime(), q=q, cmd=applied, rpy=list(imu.getRollPitchYaw()),
                     gyro=list(gyro.getValues()), accel=acc, position=list(body.getPosition()),
                     velocity=list(body.getVelocity()))
        value['foot_positions'] = [x for foot in feet for x in foot.getPosition()]
        contacts = body.getContactPoints(True)
        # PROTO-internal getId() can return -1. Preserve simulator contact IDs
        # and points rather than guessing a per-foot association.
        value['body_contact_count'] = len(contacts)
        value['contact_point_node_ids'] = [p.node_id for p in contacts]
        value['contact_points_world'] = [x for p in contacts for x in p.point]
        if not all(math.isfinite(x) for v in value.values() for x in (v if isinstance(v, list) else [v])):
            raise RuntimeError('Nonfinite simulation state')
        return value

    def advance():
        if robot.step(sim['control_ms']) == -1:
            raise RuntimeError('Webots stopped')

    configure()
    conn = socket.create_connection(('127.0.0.1', int(os.environ['KHR_RL_PORT'])), timeout=30)
    # Training updates can take time; the parent owns timeout/cleanup.
    conn.settimeout(None)
    stream = conn.makefile('rwb')
    def send(value):
        stream.write((json.dumps(value, allow_nan=False) + '\n').encode())
        stream.flush()
    send(dict(token=os.environ['KHR_RL_TOKEN'], motors=names, signs=signs,
              limits=[[m.getMinPosition(), m.getMaxPosition()] for m in motors], python=sys.version,
              contact_log_version=3))
    try:
        for raw in stream:
            request = json.loads(raw)
            command = request['command']
            if command == 'close':
                break
            if command == 'reset':
                rng.seed(request['seed'])
                body.loadState('initial')
                configure()
                q = request['targets']
                for joint, x, sign in zip(joints, q, signs):
                    joint.setJointPosition(x * sign)
                body.getField('rotation').setSFRotation([0, 0, 1, request['yaw']])
                body.resetPhysics()
                pending.clear()
                pending.extend(list(q) for _ in range(sim['command_latency_steps']))
                for _ in range(round(sim['settle_s'] * 1000 / sim['control_ms'])):
                    targets(q)
                    advance()
                send({'state': state()})
            elif command == 'step':
                states = []
                for q in request['targets']:
                    targets(q)
                    advance()
                    states.append(state())
                send({'states': states})
            elif command == 'screenshot':
                robot.exportImage(request['path'], 95)
                send({'ok': True})
            elif command == 'movie_start':
                if not robot.movieIsReady():
                    raise RuntimeError('A movie recording is already active')
                if request.get('guides') and not video_guides_added:
                    # Visual-only lines: no Solid, boundingObject or physics.
                    children = robot.getRoot().getField('children')
                    for axis in (0, 1):
                        for i in range(-10, 21):
                            position = [0., 0., .002]
                            position[axis] = i * .1
                            size = [.0015, 3., .0005] if axis == 0 else [3., .0015, .0005]
                            children.importMFNodeFromString(-1, 'Pose { translation %s children [ Shape { '
                                'castShadows FALSE appearance PBRAppearance { baseColor .42 .46 .49 '
                                'roughness 1 metalness 0 } geometry Box { size %s } } ] }' % (
                                    ' '.join(map(str, position)), ' '.join(map(str, size))))
                    video_guides_added = True
                    for i in range(children.getCount()):
                        node = children.getMFNode(i)
                        if node.getType() == Node.VIEWPOINT:
                            node.getField('fieldOfView').setSFFloat(.62)
                robot.movieStartRecording(request['path'], request.get('width', 640),
                    request.get('height', 480), 0, 90, request.get('acceleration', 4), False)
                send({'ok': True})
            elif command == 'movie_stop':
                before = robot.getTime()
                robot.movieStopRecording()
                robot.step(0)  # Flush the request without changing the physics clock.
                deadline = time.monotonic() + 120
                while not robot.movieIsReady() and time.monotonic() < deadline:
                    robot.step(0)
                    time.sleep(.02)
                if not robot.movieIsReady() or robot.movieFailed():
                    raise RuntimeError('Movie encoding failed or timed out')
                send({'ok': True, 'simulation_time_before': before, 'simulation_time_after': robot.getTime()})
            else:
                raise ValueError('Unknown command: ' + command)
    except Exception as exc:
        traceback.print_exc()
        try:
            send({'error': str(exc)})
        except OSError:
            pass
    finally:
        stream.close()
        conn.close()
        robot.simulationQuit(0)
        robot.step(sim['control_ms'])


if __name__ == '__main__':
    main()
