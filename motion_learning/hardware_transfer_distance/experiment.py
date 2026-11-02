"""Windows/macOS CLI for planned physical repeats, result annotation, and figures."""
import argparse
import csv
from datetime import datetime,timezone
import random
from pathlib import Path
from common import NAMES,read,save,sha

def config_template(out):
    ids=[6,7,8,9,10,6,7,8,9,10,1,2,4,1,2,4,0]
    cfg=dict(robot_port='COM3',robot_baud=1250000,power_port='COM4',power_baud=115200,
        mapping_verified=False,mixing_off_verified=False,servo_frame=1,
        max_current_A=None,min_voltage_V=None,max_initial_error_rad=None,
        max_command_lateness_s=.015,power_stale_s=.5,max_power_gap_s=.3,
        pre_s=2.,post_s=1.,power_timestamp_contract='legacy_receive',
        power_calibration=dict(verified=False,voltage_gain=None,voltage_offset=None,current_gain=None,current_offset=None,notes='Record reference meter/calibration evidence'),
        timing=dict(verified=False,receive_latency_s=None,uncertainty_s=None,notes='Measure serial acquisition latency; legacy firmware may repeat stale readings'),
        joints=[dict(name=n,id=i,sio=2 if n.startswith('right') else 1,zero_pos=None,sign=None,
                     units_per_rad=4000/__import__('math').radians(135),min_rad=None,max_rad=None,max_speed_rad_s=None)
                for n,i in zip(NAMES,ids)])
    if out.exists():raise FileExistsError(out)
    save(out,cfg)

def make_plan(bundle,out,weights,repeats,order_seed,seeds=(10,)):
    manifest=read(bundle/'manifest.json');items=manifest['motions'];trials=[];rng=random.Random(order_seed)
    if out.exists():raise FileExistsError(out)
    if repeats<1 or any(w != .3 for w in weights) or len(set(weights))!=len(weights):raise ValueError('Invalid repeats/weights')
    if not seeds or len(set(seeds))!=len(seeds) or any(s not in range(10,16) for s in seeds):raise ValueError('Use unique seeds 10-15')
    for w in weights:
        for seed in seeds:
            for repeat in range(1,repeats+1):
                block=[]
                for condition in ['R1','R2','R3','Task-only']:
                    item=next(x for x in items if x['condition']==condition and x['training_seed']==seed and x['weight']==(0 if condition=='Task-only' else w))
                    if sha(bundle/item['file'])!=item['sha256']:raise ValueError('Changed motion')
                    block.append(dict(trial_id=f'w{w:g}_s{seed}_r{repeat}_{condition}',condition=condition,weight=w,
                        training_seed=seed,repeat=repeat,motion=item['file'],motion_sha256=item['sha256']))
                rng.shuffle(block);trials.extend(block)
    save(out,dict(schema='khr-transfer-plan-v1',created_utc=datetime.now(timezone.utc).isoformat(),
                  training_seeds=list(seeds),target_distance_m=.2,task='20cm_distance_goal',
                  order_seed=order_seed,manifest_sha256=sha(bundle/'manifest.json'),repeats=repeats,weights=weights,trials=trials,
                  comparison='R1/R2/R3 vs one shared Task-only; repeats nested within training seed',
                  policy='Keep every attempted trial and annotate failure. No automatic replacement, no fabricated significance brackets.'))
    print(f'{len(trials)} planned physical trials. One trial is run at a time by trial_id.')

def annotate(raw,distance,outcome,method,note,goal_time_s=None,goal_time_uncertainty_s=None):
    import math
    if not math.isfinite(distance):raise ValueError('Finite signed forward displacement required')
    if not (raw/'meta.json').exists():raise ValueError('Unknown trial')
    motion=read(raw/'motion.json')
    if not method.strip():raise ValueError('Distance measurement method required')
    if outcome=='success' and distance<motion.get('target_distance_m',.2):
        raise ValueError('Success requires measured forward displacement >= 0.20 m; use shortfall otherwise')
    if (goal_time_s is None)!=(goal_time_uncertainty_s is None):raise ValueError('Provide both goal time and its uncertainty')
    if goal_time_s is not None:
        if (outcome!='success' or not math.isfinite(goal_time_s) or not 0<goal_time_s<=motion['duration_s']
            or not math.isfinite(goal_time_uncertainty_s) or goal_time_uncertainty_s<0):
            raise ValueError('Goal time must be an observed first crossing within the replay window of a successful trial')
    path=raw/'observation.json' 
    if path.exists():raise FileExistsError('Observation already exists; retain history before correcting')
    save(path,dict(distance_m=distance,outcome=outcome,distance_method=method,notes=note,
                   window='Displacement over meta.score_start to meta.score_end, along initial forward axis',
                   target_distance_m=.2,goal_time_s=goal_time_s,goal_time_uncertainty_s=goal_time_uncertainty_s,
                   goal_time_definition='Observed first forward crossing of 0.20 m; seconds after score_start, not inferred from average speed',
                   annotated_utc=datetime.now(timezone.utc).isoformat()))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('config');t.add_argument('--out',type=Path,default=Path('hardware.json'))
    t=sub.add_parser('plan');t.add_argument('--bundle',type=Path,default=Path('data'));t.add_argument('--out',type=Path,default=Path('plan_pilot.json'));t.add_argument('--weights',nargs='+',type=float,default=[.3]);t.add_argument('--repeats',type=int,default=3);t.add_argument('--order-seed',type=int,default=20261011)
    t.add_argument('--seeds',nargs='+',type=int,default=[10])
    t=sub.add_parser('run');t.add_argument('--bundle',type=Path,default=Path('data'));t.add_argument('--plan',type=Path,default=Path('plan_pilot.json'));t.add_argument('--trial',required=True);t.add_argument('--config',type=Path,default=Path('hardware.json'));t.add_argument('--out',type=Path,required=True);t.add_argument('--execute',action='store_true');t.add_argument('--mock',action='store_true')
    t=sub.add_parser('annotate');t.add_argument('--raw',type=Path,required=True);t.add_argument('--distance-m',type=float,required=True);t.add_argument('--outcome',choices=['success','fall','deviation','operator_stop','other_failure','shortfall'],required=True);t.add_argument('--method',required=True);t.add_argument('--note',default='')
    t.add_argument('--goal-time-s',type=float);t.add_argument('--goal-time-uncertainty-s',type=float)
    t=sub.add_parser('analyze');t.add_argument('--bundle',type=Path,default=Path('data'));t.add_argument('--plan',type=Path,default=Path('plan_pilot.json'));t.add_argument('--raw',type=Path,default=Path('raw'));t.add_argument('--out',type=Path,required=True);t.add_argument('--sim-evaluator',choices=['R1','R2','R3'],default='R2');t.add_argument('--allow-mock',action='store_true')
    a=p.parse_args()
    if a.command=='config':config_template(a.out)
    elif a.command=='plan':make_plan(a.bundle,a.out,a.weights,a.repeats,a.order_seed,a.seeds)
    elif a.command=='run':
        from capture import run
        run(a.bundle,a.plan,a.trial,a.config,a.out,a.execute,a.mock)
    elif a.command=='annotate':annotate(a.raw,a.distance_m,a.outcome,a.method,a.note,a.goal_time_s,a.goal_time_uncertainty_s)
    elif a.command=='analyze':
        from analyze import analyze
        analyze(a.bundle,a.plan,a.raw,a.out,a.sim_evaluator,a.allow_mock)

if __name__=='__main__':main()
