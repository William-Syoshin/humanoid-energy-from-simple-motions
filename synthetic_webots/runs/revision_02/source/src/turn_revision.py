"""Joint-command-only turning candidates for the revised synthetic benchmark.

Development parameters change reference motions only. No world fields, forces,
motor gains, friction, or mechanical parameters are accessed here.
"""
import math
import numpy as np
from protocol import pose, CFG

_previous = None
_yaw = 0.
_exit = None
_exit_q = None
_cycle_index = None
_cycle_amplitude = 1.

def reset(spec):
    global _previous, _yaw, _exit, _exit_q, _cycle_index, _cycle_amplitude
    _previous = None
    _yaw = 0.
    _exit = None
    _exit_q = None
    _cycle_index = None
    _cycle_amplitude = 1.

def smooth(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3.-2.*x)

def rocking(t, p):
    period = p.get('period', 2.0)
    phase = 2.*math.pi*t/period
    ramp = smooth(t/p.get('ramp', 1.5))
    q = pose('standing').copy()
    height = p.get('height', CFG['leg_height_m'])
    l1, l2 = CFG['thigh_m'], CFG['shin_m']
    roll = ramp*p.get('roll', .18)*math.sin(phase)
    pc = ramp*p.get('pitch_cycle', .20)*math.cos(phase+p.get('pitch_phase', 0.))
    for i, side in ((0,1.), (5,-1.)):
        swing = max(0., -side*math.sin(phase+p.get('lift_phase', 0.)))
        lift = ramp*p.get('lift', .008)*max(0.,(swing-.6)/.4)
        d = height-lift
        x = ramp*p.get('stride', 0.)*side*.5*math.cos(phase)
        dist = np.clip(math.hypot(x,d), abs(l1-l2)+1e-5,l1+l2-1e-5)
        knee = math.pi-math.acos(np.clip((l1*l1+l2*l2-dist*dist)/(2*l1*l2),-1,1))
        hip = math.atan2(-x,d)-math.acos(np.clip((l1*l1+dist*dist-l2*l2)/(2*l1*dist),-1,1))
        ankle = -hip-knee
        q[i:i+5] = [roll,hip-pc-p.get('lean',0.),knee,ankle+p.get('ankle_pc',0.)*pc,-roll]
    return q

def vendor(t, p):
    """HLO020-inspired three-phase angle changes, not factory-equivalent replay.

    Factory servo-parameter instructions are deliberately not applied. Reference
    changes are expressed around the existing simulator standing posture.
    """
    q = pose('standing').copy()
    frames = []
    direction = p.get('direction', 1.)
    for phase in ('A','B','C'):
        x=np.zeros(17)
        x[[0,5]] = (.05236 if phase != 'C' else -.05236)*p.get('hip_roll_scale',1.)
        x[5] *= p.get('right_roll_sign',1.)
        if phase != 'C':
            x[[1,6]] = [direction*.15708,-direction*.15708]
            x[[3,8]] = [-direction*.15708,direction*.15708]
        x[[4,9]] = (0. if phase=='C' else (.05236 if phase=='A' else .1309))*p.get('foot_roll_scale',1.)
        x[9] *= p.get('right_roll_sign',1.)
        frames.append(x)
    speed=p.get('time_scale',1.)
    durations=np.array([.04,.16,.20])*speed
    u=t%float(durations.sum())
    j=int(np.searchsorted(np.cumsum(durations),u,side='right'))
    before=0. if j==0 else float(durations[:j].sum())
    f=(u-before)/durations[j]
    # Original frame commands specify a transition to each pose.
    delta=(1-f)*frames[(j-1)%3]+f*frames[j]
    q += smooth(t/p.get('ramp',1.5))*p.get('amplitude',1.)*delta
    return q

def command(spec, t, state):
    global _previous, _yaw, _exit, _exit_q, _cycle_index, _cycle_amplitude
    p = spec.get('turn_parameters', {})
    R = state['orientation']
    raw = math.atan2(R[1,0],R[0,0])
    if t <= 0 or _previous is None:
        _previous, _yaw = raw, 0.
    else:
        _yaw += math.atan2(math.sin(raw-_previous),math.cos(raw-_previous))
        _previous = raw
    wanted = math.radians(p.get('target_deg', 90. if spec['action']=='turn_90' else 180.))
    sign = 1. if wanted >= 0. else -1.
    effective = dict(p)
    if p.get('approach_deg', 0.) > 0.:
        remaining = math.degrees(sign*(wanted-_yaw))
        fraction = float(np.clip(remaining/p['approach_deg'], 0., 1.))
        amplitude = p.get('amplitude',1.)*(p.get('minimum_amplitude',.5)+(1.-p.get('minimum_amplitude',.5))*fraction)
        index = int((t+1e-7)/(.4*p.get('time_scale',1.)))
        if _cycle_index != index:
            _cycle_amplitude, _cycle_index = amplitude, index
        effective['amplitude'] = _cycle_amplitude if p.get('cycle_amplitude',False) else amplitude
    q = vendor(t, effective) if p.get('mode')=='vendor' else rocking(t, effective)
    if _exit is None and t > p.get('ramp',1.5)+1.:
        cycle=.4*p.get('time_scale',1.)
        phase=t%cycle
        gate=not p.get('stop_at_cycle_end',False) or min(phase,cycle-phase)<.001
        if (gate and sign*_yaw >= abs(wanted)-math.radians(p.get('brake_lead_deg',3.))) or t >= spec['duration']-p.get('settle_s',2.5):
            _exit, _exit_q = t, q.copy()
    if _exit is not None:
        u = smooth((t-_exit)/p.get('brake_s',1.5))
        q = (1-u)*_exit_q+u*pose('standing')
    return q
