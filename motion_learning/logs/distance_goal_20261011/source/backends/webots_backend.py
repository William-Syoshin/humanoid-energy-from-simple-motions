"""Small Webots adapter: inspect devices, command positions, sample SI state."""
from collections import deque
import math
import random
from .base import RobotState


class WebotsBackend:
    def __init__(self, robot, config, manifest):
        from controller import Node
        self.robot, self.config = robot, config
        self.names = [j['name'] for j in manifest['joints']]
        if set(config['joints']) != set(self.names):
            raise ValueError('Configuration must specify every manifest joint exactly once')
        basic = robot.getBasicTimeStep()
        self.dt_ms = config['control_timestep_ms']
        if basic != config['basic_timestep_ms'] or self.dt_ms < basic or self.dt_ms % basic:
            raise ValueError('World timestep differs from config or control timestep is not a multiple')
        self.dt = self.dt_ms / 1000
        self.devices = {robot.getDeviceByIndex(i).getName(): robot.getDeviceByIndex(i)
                        for i in range(robot.getNumberOfDevices())}
        def device(name, kind):
            d = self.devices.get(name)
            if d is None or d.getNodeType() != kind:
                raise ValueError(f'Missing/wrong device: {name}; found {list(self.devices)}')
            return d
        self.motors, self.sensors, self.inventory = {}, {}, []
        velocity = config['motor_velocity_rad_s']
        fraction = config['motor_torque_fraction']
        if not 0 < fraction <= 1 or not math.isfinite(velocity) or velocity <= 0:
            raise ValueError('Invalid motor speed/torque setting')
        for spec in manifest['joints']:
            name = spec['name']
            motor = device(name, Node.ROTATIONAL_MOTOR)
            sensor = device(spec['sensor'], Node.POSITION_SENSOR)
            if motor.getPositionSensor().getName() != sensor.getName():
                raise ValueError(f'Position sensor is not attached to {name}')
            lo, hi = motor.getMinPosition(), motor.getMaxPosition()
            for actual, expected in [(lo, spec['model_min_rad']), (hi, spec['model_max_rad']),
                                     (motor.getMaxVelocity(), spec['model_max_velocity_rad_s']),
                                     (motor.getMaxTorque(), spec['model_max_torque_nm'])]:
                if not math.isclose(actual, expected, abs_tol=1e-6):
                    raise ValueError(f'Runtime model differs from pinned manifest: {name}')
            limits = config['joints'][name]
            low, neutral, high = (limits[k] for k in ('safe_min_rad', 'neutral_rad', 'safe_max_rad'))
            if not all(math.isfinite(x) for x in (low, neutral, high)) or not lo <= low <= neutral <= high <= hi or low == high:
                raise ValueError(f'Invalid safe/neutral range: {name}')
            if velocity > motor.getMaxVelocity():
                raise ValueError(f'Configured velocity exceeds model limit: {name}')
            motor.setVelocity(velocity)
            motor.setAvailableTorque(motor.getMaxTorque() * fraction)
            sensor.enable(self.dt_ms)
            self.motors[name], self.sensors[name] = motor, sensor
            self.inventory.append(dict(spec, runtime_min_rad=lo, runtime_max_rad=hi))
        names = config['sensors']
        self.imu = device(names['imu'], Node.INERTIAL_UNIT)
        self.gyro = device(names['gyro'], Node.GYRO)
        self.accel = device(names['accelerometer'], Node.ACCELEROMETER)
        for d in (self.imu, self.gyro, self.accel):
            d.enable(self.dt_ms)
        self.node = robot.getSelf()
        self.rng = random.Random(config['seed'])
        self.previous = None
        self.previous_time = None
        latency = config['latency_steps']
        if not isinstance(latency, int) or latency < 0:
            raise ValueError('latency_steps must be a nonnegative integer')
        noise = config['sensor_noise']
        if any(not math.isfinite(v) or v < 0 for v in noise.values()):
            raise ValueError('Noise standard deviations must be finite and nonnegative')
        neutral = {n: config['joints'][n]['neutral_rad'] for n in self.names}
        self.pending = deque(dict(neutral) for _ in range(latency))
        self.applied = neutral
        # Hold a defined position during the first sensor sampling interval.
        for n, m in self.motors.items():
            m.setPosition(neutral[n])

    def set_targets(self, targets):
        """Reject invalid commands instead of silently clipping researcher mistakes."""
        if set(targets) != set(self.names):
            raise ValueError('Command must include all joints')
        for n, value in targets.items():
            limits = self.config['joints'][n]
            if not math.isfinite(value) or not limits['safe_min_rad'] <= value <= limits['safe_max_rad']:
                raise ValueError(f'Target out of configured range: {n}={value}')
        self.pending.append(dict(targets))
        self.applied = self.pending.popleft()
        for name, value in self.applied.items():
            self.motors[name].setPosition(value)

    def step(self):
        if self.robot.step(self.dt_ms) == -1:
            return None
        noise = self.config['sensor_noise']
        def noisy(x, key):
            sigma = noise[key]
            return x + self.rng.gauss(0, sigma) if sigma else x
        q = {n: noisy(s.getValue(), 'joint_std_rad') for n, s in self.sensors.items()}
        now = self.robot.getTime()
        dq = {n: (q[n] - self.previous[n]) / (now - self.previous_time)
              if self.previous is not None else 0.0 for n in self.names}
        rpy = tuple(noisy(x, 'orientation_std_rad') for x in self.imu.getRollPitchYaw())
        gyro = tuple(noisy(x, 'gyro_std_rad_s') for x in self.gyro.getValues())
        accel = tuple(self.accel.getValues())
        height = self.node.getPosition()[2]
        if not all(math.isfinite(x) for x in [*q.values(), *dq.values(), *rpy, *gyro, *accel, height]):
            raise RuntimeError('Non-finite sensor sample after simulation step')
        self.previous, self.previous_time = q, now
        return RobotState(now, q, dq, rpy, gyro, accel, height)
