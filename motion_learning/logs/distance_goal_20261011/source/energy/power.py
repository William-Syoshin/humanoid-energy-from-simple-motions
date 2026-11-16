"""Exact hardware features with three-sample delayed online evaluation.

At 10 Hz the centered five-position average followed by centered differentiation
requires q[t+3]. Mature samples are emitted once; terminal edge-padding flushes
the rest. Summed rewards integrate exactly the same trace as offline evaluation.
"""
import json
from pathlib import Path
import numpy as np
from rl.common import ROOT, sha256


def feature_matrix(q, cmd, tilt, edges, support='feet', dt=0.1, times=None):
    q, cmd = np.asarray(q, dtype=float), np.asarray(cmd, dtype=float)
    if q.ndim != 2 or q.shape[1] != 17 or q.shape != cmd.shape or len(q) < 2:
        raise ValueError('Need at least two samples of all 17 joints')
    padded = np.pad(q, ((2, 2), (0, 0)), mode='edge')
    smooth = sum(padded[k:k + len(q)] for k in range(5)) / 5
    vel = np.gradient(smooth, np.arange(len(q)) * dt if times is None else times, axis=0)
    err = cmd - q
    h = 1 - np.cos(np.asarray(tilt))
    categories = ['feet', 'ground', 'fixture', 'object', 'other']
    if support not in categories:
        raise ValueError('Unknown protocol support label')
    ctx = np.tile([float(support == s) for s in categories[:-1]], (len(q), 1))
    return np.column_stack([np.ones(len(q)), abs(vel), vel**2, abs(err), err**2,
                            np.column_stack([abs(err[:, i] * err[:, j]) for i, j in edges]), h, h*h, ctx])


class FrozenPower:
    def __init__(self, support='feet'):
        assets = ROOT / 'energy/assets'
        self.meta = json.loads((assets / 'provenance.json').read_text())
        for name, key in [('frozen_models.npz','model_sha256'), ('frozen.json','frozen_json_sha256'),
                          ('training_bounds.npz','bounds_sha256')]:
            if sha256(assets / name) != self.meta[key]:
                raise ValueError('Frozen energy asset changed: ' + name)
        self.frozen = json.loads((assets / 'frozen.json').read_text())
        self.params = dict(np.load(assets / 'frozen_models.npz', allow_pickle=False))
        self.bounds = dict(np.load(assets / 'training_bounds.npz', allow_pickle=False))
        self.edges = self.meta['edges']
        self.support = support
        self.dt = self.frozen['schema']['analysis_dt_s']
        self.mapping = json.loads((ROOT / 'config/joint_mapping.json').read_text())['joints']
        if [m['name'] for m in self.mapping] != self.frozen['joint_order']:
            raise ValueError('Joint order differs from frozen predictor')
        self.sign = np.array([m['power_sign'] for m in self.mapping])
        self.offset = np.array([m['power_offset_rad'] for m in self.mapping])
        self.reset()

    def reset(self):
        self.q, self.cmd, self.tilt = [], [], []
        self.next_index = 0
        self.previous_power = None
        self.total = {m: 0. for m in ('R1','R2','R3')}
        self.positive_total = dict(self.total)
        self.ood_samples = self.scored_samples = self.negative_samples = 0
        self.predictions = []
        self.ood_components = {'position': 0, 'command': 0, 'velocity': 0, 'error': 0, 'pairs': 0, 'context': 0}
        self.previous_ood = self.previous_negative = None

    def push(self, state, terminal=False):
        self.q.append(np.asarray(state['q']) * self.sign + self.offset)
        self.cmd.append(np.asarray(state['cmd']) * self.sign + self.offset)
        # Emulate the hardware two-horizontal-channel accelerometer estimator,
        # rather than substituting perfect InertialUnit orientation.
        a = np.asarray(state['accel'])
        self.tilt.append(float(np.arcsin(min(1., np.linalg.norm(a[:2]) / 9.81))))
        delta = {m: 0. for m in self.total}
        positive = dict(delta)
        if len(self.q) < 2:
            return delta, positive, 0., 0.
        last = len(self.q) - 1 if terminal else len(self.q) - 4
        if last < self.next_index:
            return delta, positive, 0., 0.
        x = feature_matrix(self.q, self.cmd, self.tilt, self.edges, self.support, self.dt)
        ood_count = negative_count = 0
        ood_duration = negative_duration = 0.
        for i in range(self.next_index, last + 1):
            power = {m: float((x[i, :n] / self.params[m+'_scale']) @ self.params[m+'_coef'])
                     for m, n in [('R1',69),('R2',89),('R3',95)]}
            lo, hi = self.bounds['feature_min'], self.bounds['feature_max']
            bad = (x[i] < lo-1e-8) | (x[i] > hi+1e-8)
            components = dict(position=bool(np.any(self.q[i] < self.bounds['q_min']-1e-8) or np.any(self.q[i] > self.bounds['q_max']+1e-8)),
                              command=bool(np.any(self.cmd[i] < self.bounds['cmd_min']-1e-8) or np.any(self.cmd[i] > self.bounds['cmd_max']+1e-8)),
                              velocity=bool(np.any(bad[1:35])), error=bool(np.any(bad[35:69])),
                              pairs=bool(np.any(bad[69:89])), context=bool(np.any(bad[89:95])))
            ood = any(components.values())
            for key, value in components.items():
                self.ood_components[key] += int(value)
            negative = any(v < 0 for v in power.values())
            self.predictions.append(dict(sample=i, time_s=i*self.dt, **power, ood=ood, negative=negative))
            if self.previous_power is not None:
                for m in power:
                    delta[m] += (self.previous_power[m] + power[m]) * self.dt / 2
                    positive[m] += (max(0., self.previous_power[m]) + max(0., power[m])) * self.dt / 2
                ood_duration += (int(self.previous_ood)+int(ood))*self.dt/2
                negative_duration += (int(self.previous_negative)+int(negative))*self.dt/2
            self.previous_power = power
            self.previous_ood, self.previous_negative = ood, negative
            ood_count += int(ood)
            negative_count += int(negative)
            self.scored_samples += 1
        self.next_index = last + 1
        self.ood_samples += ood_count
        self.negative_samples += negative_count
        for m in delta:
            self.total[m] += delta[m]
            self.positive_total[m] += positive[m]
        return delta, positive, ood_duration, negative_duration

    def summary(self):
        # Rate uses samples on the fixed grid; duration-weighted rate is
        # separately calculated with trapezoidal endpoint weights.
        flags = np.array([p['ood'] for p in self.predictions], dtype=float)
        neg = np.array([p['negative'] for p in self.predictions], dtype=float)
        duration = max(0, len(flags)-1)*self.dt
        return {'predicted_energy_J': self.total.copy(), 'positive_energy_J': self.positive_total.copy(),
                'ood_rate': float(np.clip(np.trapezoid(flags, dx=self.dt)/duration, 0, 1)) if duration else 0.,
                'ood_component_sample_rates': {k: v/max(1,self.scored_samples) for k,v in self.ood_components.items()},
                'negative_power_rate': float(np.trapezoid(neg, dx=self.dt)/duration) if duration else 0.}
