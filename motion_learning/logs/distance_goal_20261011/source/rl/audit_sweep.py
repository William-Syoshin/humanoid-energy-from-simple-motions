"""Independently recompute saved trajectories and decompose frozen predictions.

Terms are signed regression contributions, not physical energy measurements.
This validates arithmetic; it does not validate transfer of the power predictor
from hardware to Webots.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from energy.power import FrozenPower, feature_matrix
from rl.common import write_json, sha256
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def audit(root):
    root=Path(root)
    runs=json.loads((root/'results.json').read_text())
    power=FrozenPower()
    groups=dict(intercept=(0,1),abs_velocity=(1,18),squared_velocity=(18,35),
                abs_tracking_error=(35,52),squared_tracking_error=(52,69),
                error_pairs=(69,89),tilt=(89,91),support=(91,95))
    checks=[];contributions=[]
    for run in runs:
        directory=Path(run['run'])
        tag_checks=[]
        for event in directory.glob('events.out.tfevents.*'):
            accumulator=EventAccumulator(str(event),size_guidance={'scalars':0})
            accumulator.Reload()
            tags=accumulator.Tags()['scalars']
            if not all(np.isfinite(e.value) for tag in tags for e in accumulator.Scalars(tag)):
                raise ValueError('Nonfinite TensorBoard value: '+str(event))
            rewards=[e.value for e in accumulator.Scalars('reward/energy')]
            if run['condition']=='task_only': assert all(v==0 for v in rewards)
            else: assert all(v<=0 for v in rewards)
            tag_checks.append(dict(event=str(event),scalar_tags=len(tags),energy_reward_min=min(rewards),
                                   energy_reward_max=max(rewards)))
        assert tag_checks, 'Missing TensorBoard event file'
        worst=0.
        for episode in run['episodes']:
            path=directory/'evaluation'/f"trajectory_{episode['seed']}.jsonl"
            trace=[json.loads(line) for line in path.read_text().splitlines()]
            q=np.array([s['q'] for s in trace])*power.sign+power.offset
            cmd=np.array([s['cmd'] for s in trace])*power.sign+power.offset
            tilt=np.arcsin(np.minimum(1.,np.linalg.norm(np.array([s['accel'][:2] for s in trace]),axis=1)/9.81))
            x=feature_matrix(q,cmd,tilt,power.edges,power.support,power.dt)
            predictions=[json.loads(line) for line in (directory/'evaluation'/f"power_{episode['seed']}.jsonl").read_text().splitlines()]
            for model,n in [('R1',69),('R2',89),('R3',95)]:
                terms=(x[:,:n]/power.params[model+'_scale'])*power.params[model+'_coef']
                offline=terms.sum(axis=1)
                logged=np.array([p[model] for p in predictions])
                np.testing.assert_allclose(offline,logged,rtol=1e-11,atol=1e-10)
                energy=float(np.trapezoid(offline,dx=.1))
                error=abs(energy-episode['predicted_energy_J'][model]);worst=max(worst,error)
                assert error<1e-8
                row=dict(condition=run['condition'],weight=run['weight'],training_seed=run['training_seed'],
                         evaluation_seed=episode['seed'],model=model,total_J=energy)
                for group,(lo,hi) in groups.items():
                    row[group+'_J']=float(np.trapezoid(terms[:,lo:min(hi,n)].sum(axis=1),dx=.1)) if lo<n else 0.
                assert abs(sum(row[g+'_J'] for g in groups)-energy)<1e-8
                contributions.append(row)
            expected=(-run['weight']*episode['positive_energy_J'][run['condition']]
                      if run['condition']!='task_only' else 0.)
            assert abs(episode['reward_terms']['energy']-expected)<1e-8
        checks.append(dict(condition=run['condition'],weight=run['weight'],training_seed=run['training_seed'],
                           max_energy_error_J=worst,tensorboard=tag_checks,policy_sha256=sha256(directory/'final.zip')))
    with (root/'energy_contributions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(contributions[0]));writer.writeheader();writer.writerows(contributions)
    result=dict(status='passed',runs=len(runs),episodes=sum(len(r['episodes']) for r in runs),
                max_energy_error_J=max(c['max_energy_error_J'] for c in checks),checks=checks,
                interpretation='Signed regression contributions are explanatory model terms, not physical energies.')
    write_json(root/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    audit(p.parse_args().directory)
