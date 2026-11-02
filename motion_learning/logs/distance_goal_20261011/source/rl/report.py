"""Matched comparisons: failed/slow/OOD policies cannot silently count as savings."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rl.common import write_json


def report(root, relative_distance=0.95):
    root=Path(root)
    runs=json.loads((root/'results.json').read_text())
    baselines={r['training_seed']:r for r in runs if r['condition']=='task_only'}
    comparisons=[]
    for run in runs:
        if run['condition']=='task_only':continue
        base={e['seed']:e for e in baselines[run['training_seed']]['episodes']}
        pairs=[]
        for candidate in run['episodes']:
            reference=base[candidate['seed']]
            task_eligible=bool(candidate['task_success'] and reference['task_success'] and
                               candidate['distance_m']>=relative_distance*reference['distance_m'])
            eligible=bool(task_eligible and candidate['in_distribution_success'] and reference['in_distribution_success'] and
                          candidate['negative_power_rate']==0 and reference['negative_power_rate']==0)
            changes={m:100*(candidate['predicted_energy_J'][m]/reference['predicted_energy_J'][m]-1)
                     if reference['predicted_energy_J'][m]>0 else None for m in ('R1','R2','R3')}
            pairs.append(dict(evaluation_seed=candidate['seed'],task_eligible=task_eligible,
                              fully_eligible=eligible,energy_change_pct=changes,
                              distance_ratio=candidate['distance_m']/reference['distance_m'] if reference['distance_m']>0 else None))
        comparisons.append(dict(condition=run['condition'],weight=run['weight'],training_seed=run['training_seed'],pairs=pairs))
    write_json(root/'paired_comparisons.json',dict(relative_distance_threshold=relative_distance,
               interpretation='Negative energy change means lower prediction, not measured electrical savings. '
               'Only fully_eligible pairs pass task, OOD and nonnegative-power gates. Shared predictors are not independent validation.',
               comparisons=comparisons))
    rows=[]
    for run in runs:
        e=run['episodes']
        row=dict(condition=run['condition'],weight=run['weight'],training_seed=run['training_seed'],
                 speed_m_s=float(np.mean([x['mean_speed_m_s'] for x in e])),
                 distance_m=float(np.mean([x['distance_m'] for x in e])),
                 success_rate=float(np.mean([x['task_success'] for x in e])),
                 fall_rate=float(np.mean([x['fallen'] for x in e])),
                 ood_rate=float(np.mean([x['ood_rate'] for x in e])))
        for m in ('R1','R2','R3'):
            row[m+'_J']=float(np.mean([x['predicted_energy_J'][m] for x in e]))
            good=[x['energy_per_m_J'][m] for x in e if x['energy_per_m_J'][m] is not None]
            row[m+'_J_m_successful']=float(np.mean(good)) if good else None
        rows.append(row)
    with (root/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=sorted({(r['condition'],r['weight']) for r in rows})
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,metric,label in zip(axes,['speed_m_s','R2_J_m_successful','ood_rate'],
                               ['Forward speed [m/s]','R2 predicted energy / distance [J/m]\n(task-successful episodes only)','OOD time fraction']):
        for k,(condition,weight) in enumerate(groups):
            values=[r[metric] for r in rows if (r['condition'],r['weight'])==(condition,weight) and r[metric] is not None]
            if values:
                ax.scatter(np.full(len(values),k),values,s=28)
                ax.errorbar(k,np.mean(values),yerr=np.std(values,ddof=1) if len(values)>1 else 0,
                            fmt='_',color='black',capsize=5)
        ax.set_xticks(range(len(groups)),[f'{c}\nλ={w:g}' for c,w in groups])
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Exploratory Webots comparison — points = independent training seeds')
    fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)
    print(root/'comparison.csv')
    return comparisons


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--relative-distance',type=float,default=.95)
    a=p.parse_args();report(a.directory,a.relative_distance)
