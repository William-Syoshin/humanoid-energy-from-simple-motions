"""Bounded five-parameter gait; both policies use the identical generator.

Angles are anatomical simulator coordinates: both legs pitch about +Y and roll
about +X. Device signs are applied only at the Webots boundary.
"""
import math
import numpy as np


class Gait:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lower = np.asarray(cfg['lower'], dtype=float)
        self.upper = np.asarray(cfg['upper'], dtype=float)
        self.reset()

    def reset(self):
        self.parameters = np.asarray(self.cfg['nominal'], dtype=float).copy()
        self.phase = 0.
        self.elapsed = 0.

    def neutral(self):
        return self.pose(0., 0.)

    def pose(self, phase, amplitude):
        period, stride, lift, sway, lean = self.parameters
        q = np.zeros(17)
        l1, l2 = self.cfg['thigh_m'], self.cfg['shin_m']
        roll = amplitude * sway * math.sin(phase + self.cfg['roll_phase_rad'])
        for side, start in [(0, 0), (1, 5)]:
            cycle = ((phase + side*math.pi) % (2*math.pi)) / (2*math.pi)
            swing = cycle < 0.5
            u = cycle*2 if swing else (cycle-0.5)*2
            # Smooth swing advance; stance foot travels backwards under body.
            x = stride * (-0.5 + (u - math.sin(2*math.pi*u)/(2*math.pi))) if swing else stride*(0.5-u)
            z = lift * math.sin(math.pi*u)**2 if swing else 0.
            x *= amplitude
            d = self.cfg['leg_height_m'] - z*amplitude
            r = math.hypot(x, d)
            if not abs(l1-l2) < r < l1+l2:
                raise ValueError('Unreachable foot trajectory; tighten gait bounds')
            knee = math.pi - math.acos(np.clip((l1*l1+l2*l2-r*r)/(2*l1*l2), -1, 1))
            hip = math.atan2(-x, d) - math.acos(np.clip((l1*l1+r*r-l2*l2)/(2*l1*r), -1, 1))
            ankle = -hip-knee
            q[start:start+5] = [roll, hip-lean, knee, ankle, -roll]
        q[10] = self.cfg['arm_swing_rad']*amplitude*math.sin(phase)
        q[13] = -q[10]
        if np.any(abs(q) > self.cfg['joint_limit_rad']):
            raise ValueError('Gait exceeds joint command limits')
        return q

    def advance(self, action, dt):
        target = self.lower + (np.asarray(action)+1)*0.5*(self.upper-self.lower)
        change = np.asarray(self.cfg['slew_per_s'])*dt
        self.parameters += np.clip(target-self.parameters, -change, change)
        self.phase = (self.phase + 2*math.pi*dt/self.parameters[0]) % (2*math.pi)
        self.elapsed += dt
        ramp = min(1., self.elapsed/self.cfg['ramp_s'])
        return self.pose(self.phase, ramp*ramp*(3-2*ramp))

    def nominal_action(self):
        return 2*(np.asarray(self.cfg['nominal'])-self.lower)/(self.upper-self.lower)-1
