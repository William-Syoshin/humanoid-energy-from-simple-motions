"""Heading-controlled joint commands with stationary terminal corrections.

No physical model, torque setting, root pose, or force is modified. The coarse
reference remains the recorded revision_01 controller. Correction pulses start
and end in the same standing reference and are followed by a settling interval.
"""
import math
from collections import deque
import numpy as np
import turn_revision as coarse
from protocol import pose

_previous = None
_heading = 0.
_stage = 'coarse'
_stage_start = 0.
_pulse_end_q = None
_pulse_parameters = None
_pulse_duration = 0.
_correction_count = 0
_pulse_stop_time = None
_pulse_stop_q = None
_pulse_sign = 1.
_history = deque(maxlen=20)

def reset(spec):
    global _previous, _heading, _stage, _stage_start, _pulse_end_q, _pulse_parameters, _pulse_duration, _correction_count, _pulse_stop_time, _pulse_stop_q, _pulse_sign
    coarse.reset(spec)
    _previous=None
    _heading=0.
    _stage='coarse'
    _stage_start=0.
    _pulse_end_q=None
    _pulse_parameters=None
    _pulse_duration=0.
    _correction_count=0
    _pulse_stop_time=None
    _pulse_stop_q=None
    _pulse_sign=1.
    _history.clear()

def pulse(local, p, duration):
    if local < duration:
        return coarse.vendor(local,p)
    end=coarse.vendor(duration,p)
    u=coarse.smooth((local-duration)/p.get('brake_s',.5))
    return (1-u)*end+u*pose('standing')

def command(spec,t,state):
    global _previous, _heading, _stage, _stage_start, _pulse_parameters, _pulse_duration, _correction_count, _pulse_stop_time, _pulse_stop_q, _pulse_sign
    p=spec.get('turn_parameters',{})
    R=state['orientation']
    raw=math.atan2(R[1,0],R[0,0])
    if t<=0 or _previous is None:
        _previous=raw
        _heading=0.
        _history.clear()
    else:
        _heading+=math.atan2(math.sin(raw-_previous),math.cos(raw-_previous))
        _previous=raw
    _history.append(_heading)
    if p.get('pulse_probe',False):
        return pulse(t,p,p.get('pulse_duration',.4))
    wanted=math.radians(p.get('target_deg',90 if spec['action']=='turn_90' else 180))
    if _stage=='coarse':
        q=coarse.command(dict(spec,duration=p.get('coarse_duration',spec['duration'])),t,state)
        if coarse._exit is not None and t>=coarse._exit+p.get('brake_s',.5)+p.get('rest_s',1.5):
            _stage='check'
        else:
            return q
    if _stage=='check':
        error=math.degrees(wanted-float(np.median(_history)))
        if abs(error)<=p.get('settled_tolerance_deg',5.) or _correction_count>=p.get('max_corrections',3) or t>=spec['duration']-p.get('final_reserve_s',5.):
            _stage='done'
        else:
            amplitude=p.get('correction_amplitude',.7) if error>0 else p.get('negative_correction_amplitude',p.get('correction_amplitude',.7))
            _pulse_parameters=dict(p,direction=(-1. if error>0 else 1.),amplitude=amplitude,ramp=p.get('correction_ramp',.05))
            _pulse_duration=p.get('positive_correction_duration',2.4) if error>0 else p.get('negative_correction_duration',.8)
            _pulse_sign=1. if error>0 else -1.
            _pulse_stop_time=None
            _pulse_stop_q=None
            _correction_count+=1
            _stage_start=t
            _stage='pulse'
    if _stage=='pulse':
        local=t-_stage_start
        q=coarse.vendor(local,_pulse_parameters)
        if _pulse_stop_time is None:
            cycle=.4*p.get('time_scale',1.)
            phase=local%cycle
            boundary=min(phase,cycle-phase)<.001
            remaining=math.degrees(_pulse_sign*(wanted-_heading))
            reached=(local>=p.get('correction_min_s',.4)-1e-9 and boundary and
                     remaining<=p.get('correction_brake_lead_deg',3.))
            if reached or local>=_pulse_duration-1e-9:
                _pulse_stop_time=local
                _pulse_stop_q=q.copy()
        if _pulse_stop_time is not None:
            u=coarse.smooth((local-_pulse_stop_time)/p.get('brake_s',.5))
            q=(1-u)*_pulse_stop_q+u*pose('standing')
            if local>=_pulse_stop_time+p.get('brake_s',.5)+p.get('rest_s',1.5):
                _stage='check'
        return q
    return pose('standing')
