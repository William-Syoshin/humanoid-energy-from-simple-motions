"""Gymnasium walking task with interchangeable simulation and fixed power models."""
from collections import deque
import copy
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from backends.socket_backend import WebotsProcess
from energy.power import FrozenPower
from env.gait import Gait
from rl.common import load_config, validate_config


class KHR3HVEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 50}

    def __init__(self, config=None, render_mode=None, backend=None):
        super().__init__()
        self.config = copy.deepcopy(config if config is not None else load_config())
        validate_config(self.config)
        if render_mode not in (None, 'human'):
            raise ValueError('Only human rendering is supported')
        self.render_mode = render_mode
        if render_mode == 'human':
            self.config['simulation']['gui'] = True
        self.dt = self.config['simulation']['action_ms']/1000
        if not math.isclose(self.dt, 0.1):
            raise ValueError('Frozen model evaluation requires 10 Hz')
        self.gait = Gait(self.config['gait'])
        self.power = FrozenPower(self.config['energy']['support'])
        self.action_space = spaces.Box(-1., 1., shape=(5,), dtype=np.float32)
        # Seven q/cmd pairs expose the filter history, plus motion and gait state.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(263,), dtype=np.float32)
        self.backend = backend
        self.history = deque(maxlen=7)
        self.finished = True
        self.last_episode = None

    def _observation(self):
        s = self.state
        # Fixed scales only: task-only and energy-aware use identical observations.
        data = np.concatenate([np.asarray(self.history).ravel(), s['rpy'], s['gyro'],
                               np.asarray(s['velocity'][:3])*10,
                               [s['position'][2], s['position'][1]-self.origin[1]],
                               [math.sin(self.gait.phase), math.cos(self.gait.phase)],
                               (self.gait.parameters-self.gait.lower)/(self.gait.upper-self.gait.lower),
                               self.previous_action,
                               [self.steps*self.dt/self.config['simulation']['episode_s']],
                               [max(0., self.config['task']['target_distance_m'] - (s['position'][0]-self.origin[0])) / self.config['task']['target_distance_m']
                                if self.config['task'].get('mode') == 'distance' else self.config['task']['target_speed_m_s']]])
        if data.shape != self.observation_space.shape or not np.isfinite(data).all():
            raise ValueError(f'Invalid observation {data.shape}')
        return data.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.backend is None:
            self.backend = WebotsProcess(self.config)
        self.gait.reset()
        self.power.reset()
        self.steps = 0
        disturbance_seed = int(self.np_random.integers(0, 2**31-1))
        yaw = float(self.np_random.normal(0, self.config['simulation']['initial_yaw_std_rad']))
        self.state = self.backend.reset(disturbance_seed, self.gait.neutral(), yaw)
        if self.state['position'][2] < self.config['task']['fall_height_m'] or max(abs(x) for x in self.state['rpy'][:2]) > self.config['task']['fall_tilt_rad']:
            raise RuntimeError('Robot fell during reset/settling; baseline configuration needs repair')
        self.origin = np.array(self.state['position'])
        self.start_time = self.state['time']
        self.previous_action = self.gait.nominal_action()
        self.history.clear()
        for _ in range(7):
            self.history.append(self.state['q'] + self.state['cmd'])
        self.power.push(self.state)
        self.trace = [dict(copy.deepcopy(self.state), gait_parameters=self.gait.parameters.tolist(),
                           gait_phase_rad=self.gait.phase, action=self.previous_action.tolist())]
        self.reward_totals = {}
        self.finished = False
        return self._observation(), {'disturbance_seed': disturbance_seed, 'world_random_seed': 0}

    def step(self, action):
        if self.finished:
            raise RuntimeError('Call reset before stepping a finished episode')
        action = np.asarray(action, dtype=float)
        if action.shape != (5,) or not np.isfinite(action).all() or np.any(abs(action) > 1.000001):
            raise ValueError('Action must contain five finite values in [-1,1]')
        action = np.clip(action, -1, 1)
        sim, task, weights = self.config['simulation'], self.config['task'], self.config['reward']
        previous = self.state
        count = sim['action_ms']//sim['control_ms']
        commands = [self.gait.advance(action, sim['control_ms']/1000).tolist() for _ in range(count)]
        states = self.backend.step(commands)
        self.state = states[-1]
        self.steps += 1
        s = self.state
        elapsed = s['time'] - previous['time']
        if not math.isclose(elapsed, self.dt, abs_tol=1e-6):
            raise RuntimeError('Simulator/analysis clocks diverged')
        self.history.append(s['q'] + s['cmd'])
        self.trace.append(dict(copy.deepcopy(s), gait_parameters=self.gait.parameters.tolist(),
                               gait_phase_rad=self.gait.phase, action=action.tolist()))
        tilt = max(max(abs(x) for x in v['rpy'][:2]) for v in states)
        fallen = min(v['position'][2] for v in states) < task['fall_height_m'] or tilt > task['fall_tilt_rad']
        heading = abs(s['rpy'][2])
        lateral = s['position'][1]-self.origin[1]
        escaped = heading > task['heading_limit_rad'] or abs(lateral) > task['lateral_limit_m']
        distance_task = task.get('mode') == 'distance'
        distance = s['position'][0]-self.origin[0]
        reached = bool(distance_task and distance >= task['target_distance_m'] and not (fallen or escaped))
        deadline = self.steps*self.dt >= sim['episode_s']-1e-8
        # Finite-horizon failure is a true terminal state, not a bootstrap-able time limit.
        terminated = bool(fallen or escaped or (distance_task and (reached or deadline)))
        truncated = bool(deadline and not terminated)
        self.finished = terminated or truncated
        raw_j, positive_j, ood_s, negative_s = self.power.push(s, terminal=self.finished)
        vx = (s['position'][0]-previous['position'][0])/self.dt
        dq = (np.asarray(s['q'])-previous['q'])/self.dt
        terms = {
            'alive': weights['alive']*self.dt,
            'speed': weights['speed']*math.exp(-((vx-task['target_speed_m_s'])/task['speed_tolerance_m_s'])**2)*self.dt,
            'progress': weights['progress']*(s['position'][0]-previous['position'][0]),
            'upright': weights['upright']*math.cos(s['rpy'][0])*math.cos(s['rpy'][1])*self.dt,
            'tilt': -weights['tilt']*sum(x*x for x in s['rpy'][:2])*self.dt,
            'heading': -weights['heading']*heading**2*self.dt,
            'lateral': -weights['lateral']*lateral**2*self.dt,
            'joint_velocity': -weights['joint_velocity']*float(np.mean(dq*dq))*self.dt,
            'action_change': -weights['action_change']*float(np.mean((action-self.previous_action)**2)),
            'ood': -weights['ood']*ood_s,
            'fall': -weights['fall'] if terminated else 0.,
            'energy': 0.,
        }
        if distance_task:
            # Capped potential difference cannot pay repeatedly for crossing the goal.
            before = min(task['target_distance_m'], previous['position'][0]-self.origin[0])
            after = min(task['target_distance_m'], distance)
            terms.update(alive=0., speed=0., upright=0.,
                         progress=weights['progress']*(after-before),
                         time=-weights['time']*self.dt,
                         goal=weights['goal'] if reached else 0.,
                         fall=-weights['fall'] if fallen or escaped else 0.,
                         timeout=-weights['timeout'] if deadline and not reached and not (fallen or escaped) else 0.)
        condition = self.config['condition']
        if condition != 'task_only':
            # The raw frozen prediction is always logged. Guard only the reward
            # against treating negative, nonphysical estimates as a bonus.
            terms['energy'] = -self.config['energy']['weight']*positive_j[condition]
        self.previous_action = action.copy()
        for k, v in terms.items():
            self.reward_totals[k] = self.reward_totals.get(k, 0.) + v
        info = {'reward_terms': terms, 'power_energy_increment_J': raw_j, 'forward_speed_m_s': vx,
                'gait_parameters': self.gait.parameters.copy(), 'fallen': fallen}
        if self.finished:
            duration = self.steps*self.dt
            distance = s['position'][0]-self.origin[0]
            p = self.power.summary()
            success = reached if distance_task else bool(truncated and distance/duration >= task['target_speed_m_s']*self.config['evaluation']['minimum_speed_fraction'])
            summary = dict(condition=condition, duration_s=duration, distance_m=distance,
                           mean_speed_m_s=distance/duration, fallen=bool(fallen), escaped=bool(escaped),
                           completed_horizon=truncated, task_success=success,
                           heading_error_rad=heading, lateral_error_m=abs(lateral),
                           mean_abs_heading_rad=float(np.mean([abs(t['rpy'][2]) for t in self.trace])),
                           **p, reward_terms=self.reward_totals.copy(),
                           energy_per_m_J={m: e/distance if success and distance > 1e-6 else None for m,e in p['predicted_energy_J'].items()})
            if distance_task:
                summary.update(task_mode='distance', target_distance_m=task['target_distance_m'],
                               reached_goal=reached, timed_out=bool(deadline and not reached and not (fallen or escaped)),
                               time_to_goal_s=duration if reached else None,
                               goal_overshoot_m=max(0., distance-task['target_distance_m']) if reached else None,
                               termination_reason='goal' if reached else 'fall' if fallen else 'escape' if escaped else 'timeout')
            summary['in_distribution_success'] = success and p['ood_rate'] <= self.config['evaluation']['maximum_ood_rate']
            q_trace = np.array([t['q'] for t in self.trace])
            cmd_trace = np.array([t['cmd'] for t in self.trace])
            summary['joint_velocity_rms_rad_s'] = float(np.sqrt(np.mean((np.diff(q_trace,axis=0)/self.dt)**2)))
            summary['tracking_error_rms_rad'] = float(np.sqrt(np.mean((cmd_trace-q_trace)**2)))
            summary['tilt_rms_rad'] = float(np.sqrt(np.mean(np.array([t['rpy'][:2] for t in self.trace])**2)))
            summary['mean_gait_parameters'] = np.mean([t['gait_parameters'] for t in self.trace],axis=0).tolist()
            self.last_episode = summary
            info['summary'] = summary
        return self._observation(), float(sum(terms.values())), terminated, truncated, info

    def render(self):
        return None  # Webots owns the human GUI.

    def close(self):
        if self.backend is not None:
            self.backend.close()
            self.backend = None
