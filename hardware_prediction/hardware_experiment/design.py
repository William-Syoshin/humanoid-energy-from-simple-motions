"""Explicit proposed 120/24/21 design; unknown physical settings stay null.

This generates reviewable recipes, not mechanically approved motions.
"""
import math
from core import NAMES


def planned_recipes():
    recipes=[]
    def add(split,level,posture,support,joints=(),scale=.5,freq=.25,phase=0,kind='sine'):
        tid=f'T{len(recipes)+1:03d}'
        recipes.append(dict(trial_id=tid,split=split,level=level,approved=False,kind=kind,
            duration_s=8,pre_s=2,post_s=2,settle_s=0,support=support,posture=posture,
            recipe_group=f'{level}:{posture}:'+','.join(joints),center_rad=[None]*17,
            excitation=[dict(joint=j,amplitude_rad=None,amplitude_fraction=scale,
                             frequency_Hz=freq,phase_rad=phase if k else 0)
                        for k,j in enumerate(joints)],frames=[],support_intervals=[],
            initial_gravity=None,success_gate=None,
            notes='Proposed design; populate measured centers/amplitudes and mechanical gate before approval.'))
    for posture,support in [('upright','feet'),('squat','feet'),('supine','ground')]:
        for _ in range(2):add('train','L0',posture,support,kind='hold')
    for joint in NAMES:
        for scale,freq in [(.5,.25),(1,.25),(.5,.5)]:add('train','L1','supported_reference','fixture',[joint],scale,freq)
    pairs=[]
    for side in ('left','right'):
        for a,b in [('hip_pitch','knee_pitch'),('knee_pitch','ankle_pitch'),('shoulder_pitch','elbow_pitch')]:
            pairs.append((side+'_'+a,side+'_'+b))
    pairs += [('left_'+j,'right_'+j) for j in ('hip_pitch','knee_pitch','shoulder_pitch','ankle_pitch')]
    for pair in pairs:
        for phase in (0,math.pi/2,math.pi):add('train','L2','supported_reference','fixture',pair,phase=phase)
    local=['left_hip_pitch','right_hip_pitch','left_knee_pitch','left_shoulder_pitch','right_shoulder_pitch']
    for posture,support in [('upright','feet'),('squat','feet'),('supine','ground')]:
        for joint in local:add('train','L3',posture,support,[joint])
    for posture,support in [('upright','feet'),('prone_hand_loading','ground'),('light_object_contact','object')]:
        for joint in local[:4]:add('train','L4',posture,support,[joint])
    groups=[['left_hip_pitch','left_knee_pitch','right_hip_pitch','right_knee_pitch'],
            ['left_shoulder_pitch','left_elbow_pitch','right_shoulder_pitch','right_elbow_pitch']]
    for group in groups:
        for phase in (0,math.pi/2,math.pi):add('train','L5','supported_reference','fixture',group,phase=phase)
    for joint in NAMES[:12]:add('validation','L1','supported_reference','fixture',[joint],.75,.375)
    for pair in pairs[:6]:add('validation','L2','supported_reference','fixture',pair,.75,.375,math.pi/4)
    for posture,support in [('upright','feet'),('squat','feet'),('supine','ground')]:
        add('validation','L3',posture,support,[local[0]],.75,.375)
    for posture,support in [('upright','feet'),('prone_hand_loading','ground'),('light_object_contact','object')]:
        add('validation','L4',posture,support,[local[1]],.75,.375)
    for task in ('walk','turn90','turn180','walk_stop','get_up','lie_down','push'):
        for _ in range(3):
            add('test',task,'pending_task_start',None,kind='frames')
            recipes[-1]['duration_s']=None
            recipes[-1]['notes']='Import only an already demonstrated motion with actual command timing; set task window/support transitions and success gate.'
    return recipes
