"""Reconstruct actual 50 Hz applied commands from archived 10 Hz policy actions."""
import argparse
from collections import deque
import importlib.util
from pathlib import Path
import shutil
import numpy as np
import yaml
from common import NAMES,read,save,sha,motion_arrays


def reconstruct(trace,cfg,gait_class):
    sim=cfg['simulation'];dt=sim['control_ms']/1000
    if sim['action_ms']%sim['control_ms']: raise ValueError('Nonintegral controller ratio')
    count=sim['action_ms']//sim['control_ms'];gait=gait_class(cfg['gait'])
    initial=gait.neutral().tolist()
    np.testing.assert_allclose(initial,trace[0]['cmd'],atol=1e-10,rtol=0)
    pending=deque([initial.copy() for _ in range(sim['command_latency_steps'])]);commands=[]
    for step,state in enumerate(trace[1:]):
        for sub in range(count):
            generated=gait.advance(state['action'],dt).tolist();pending.append(generated);applied=pending.popleft()
            commands.append(dict(t_s=len(commands)*dt,q_rad=applied))
        # The saved state was sampled AFTER five control steps. Do not play the
        # decimated cmd column at 50 Hz or interpolate actual sensed q as targets.
        np.testing.assert_allclose(applied,state['cmd'],rtol=0,atol=1e-10)
        np.testing.assert_allclose(gait.parameters,state['gait_parameters'],rtol=0,atol=1e-10)
        if abs(state['time']-trace[0]['time']-(step+1)*sim['action_ms']/1000)>1e-7:
            raise ValueError('Archived trajectory timing differs from config')
    return initial,commands,dt


def export_one(run,source,out,label,weight,seed,evaluation_seed):
    cfg=yaml.safe_load((run/'config.yaml').read_text())
    episode=read(run/'evaluation'/f'episode_{evaluation_seed}.json')
    tracepath=run/'evaluation'/f'trajectory_{evaluation_seed}.jsonl'
    trace=[__import__('json').loads(x) for x in tracepath.read_text().splitlines()]
    spec=importlib.util.spec_from_file_location('frozen_export_gait',source/'env/gait.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    initial,commands,dt=reconstruct(trace,cfg,mod.Gait)
    if (episode.get('task_mode')!='distance' or not episode.get('task_success') or not episode.get('reached_goal')
        or episode.get('termination_reason')!='goal' or episode['fallen'] or episode['escaped']
        or abs(episode['target_distance_m']-.2)>1e-9 or abs(len(commands)*dt-episode['duration_s'])>1e-7):
        raise ValueError('Export requires a complete successful 20 cm trajectory; no padding or truncation')
    m=dict(schema='khr-transfer-v1',motion_id=f'{label}_lambda{weight:g}_seed{seed}',condition=label,
           weight=weight,training_seed=seed,evaluation_seed=evaluation_seed,joint_names=NAMES,
           coordinate_system='anatomical simulator radians; requires independent hardware mapping',
           mode='open_loop_replay_of_archived_policy_commands',duration_s=len(commands)*dt,
           control_dt_s=dt,initial_pose_rad=initial,commands=commands,sim=episode,
           task='20cm_distance_goal',target_distance_m=.2,end_behavior='hold_last_target; no automatic neutral transition',
           source=dict(run=str(run),trajectory_sha256=sha(tracepath),policy_sha256=sha(run/'final.zip'),
                       gait_sha256=sha(source/'env/gait.py'),config_sha256=sha(run/'config.yaml')))
    motion_arrays(m);save(out/(m['motion_id']+'.json'),m)
    return dict(file='motions/'+m['motion_id']+'.json',motion_id=m['motion_id'],condition=label,
                weight=weight,training_seed=seed,sha256=sha(out/(m['motion_id']+'.json')))


def build(project,out,evaluation_seed):
    out.mkdir(parents=True,exist_ok=False);motions=out/'motions';motions.mkdir();manifest=[]
    root=project/'logs/distance_goal_20261011'
    if evaluation_seed!=20000:raise ValueError('Export evaluation seed is pre-fixed at 20000')
    for label in ['Task-only','R1','R2','R3']:
        weight=0. if label=='Task-only' else .3
        for seed in range(10,16):
            name=f'task_only-lambda0-seed{seed}' if label=='Task-only' else f'{label}-lambda0.3-seed{seed}'
            manifest.append(export_one(root/name,root/'source',motions,label,weight,seed,evaluation_seed))
    save(out/'manifest.json',dict(schema='khr-transfer-set-v1',experiment='distance_goal_20261011',motions=manifest,
        selection='All six training seeds 10-15; evaluation seed 20000 fixed; pilot uses seed10 in every condition without performance filtering',
        baseline='New shared distance-goal Task-only; no previous 20-second policy reused',
        note='Variable-duration open-loop replay. Hardware arrival must be measured independently. Sim energy is predicted, Real is measured.'))
    print(f'Exported {len(manifest)} motions with exact checks against every saved command endpoint: {out}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path,required=True);p.add_argument('--evaluation-seed',type=int,default=20000)
    a=p.parse_args();build(a.project.resolve(),a.out,a.evaluation_seed)
