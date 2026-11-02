"""Seed-level analysis of the fixed coefficient sweep; no failed-seed deletion."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl.common import write_json


def interval(values, samples=20000, seed=84017):
    """Resample training-seed effects, never individual evaluation episodes."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (samples, len(x)))].mean(axis=1)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x)>1 else 0.,
                ci95=np.quantile(means,[.025,.975]).tolist(), n_training_seeds=len(x))


def sign_flip_p(values):
    """Exact, one-sided paired sign-flip test for a negative mean effect."""
    x = np.asarray(values, dtype=float)
    null_means = np.array([np.mean(x*np.array(s)) for s in itertools.product((-1.,1.),repeat=len(x))])
    return float(np.mean(null_means <= x.mean()+1e-12))


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    result = np.empty(len(p))
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, (len(p)-rank)*p[index])
        result[index] = min(1., previous)
    return result.tolist()


def paired_gates(reference, candidate, analysis, max_ood=.1):
    valid = (reference['task_success'] and candidate['task_success']
             and reference['ood_rate'] <= max_ood and candidate['ood_rate'] <= max_ood
             and reference['negative_power_rate'] == 0 and candidate['negative_power_rate'] == 0)
    ratio = candidate['distance_m']/reference['distance_m'] if reference['distance_m']>0 else None
    low,high = analysis['distance_equivalence']
    return dict(distance_ratio=ratio,
                equivalent=bool(valid and ratio is not None and low <= ratio <= high),
                noninferior=bool(valid and ratio is not None and low <= ratio))


def save_csv(path, rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def analyze(root):
    root=Path(root)
    design=json.loads((root/'design.json').read_text())
    runs=json.loads((root/'results.json').read_text())
    a=design['analysis']
    primary=a.get('primary_model','R2')
    if primary not in ('R1','R2','R3'):
        raise ValueError('Unknown primary energy model')
    expected={(j['seed'],j['condition'],j['weight']) for j in design['jobs']}
    observed={(r['training_seed'],r['condition'],r['weight']) for r in runs}
    if len(runs)!=len(expected) or observed!=expected:
        raise ValueError('Incomplete/duplicate sweep; do not silently drop missing runs')
    rows=[]
    for run in runs:
        episodes=run['episodes']
        expected_seeds=set(range(20000,20000+design['evaluation_episodes']))
        if len(episodes)!=len(expected_seeds) or {e['seed'] for e in episodes}!=expected_seeds:
            raise ValueError('Missing/duplicate evaluation seed')
        row=dict(condition=run['condition'],weight=run['weight'],training_seed=run['training_seed'])
        for key in ('distance_m','mean_speed_m_s','fallen','escaped','task_success','ood_rate',
                    'negative_power_rate','heading_error_rad','joint_velocity_rms_rad_s',
                    'tracking_error_rms_rad','tilt_rms_rad'):
            row[key]=float(np.mean([e[key] for e in episodes]))
        for model in ('R1','R2','R3'):
            row[model+'_J']=float(np.mean([e['predicted_energy_J'][model] for e in episodes]))
            # Only present unconditional J/m when EVERY episode is task-successful.
            row[model+'_J_per_m']=float(np.mean([e['energy_per_m_J'][model] for e in episodes])) if all(e['task_success'] for e in episodes) else None
        for i,name in enumerate(design['config']['gait']['parameter_names']):
            row['gait_'+name]=float(np.mean([e['mean_gait_parameters'][i] for e in episodes]))
        rows.append(row)
    lookup={(r['training_seed'],r['weight']):r for r in rows}
    run_lookup={(r['training_seed'],r['weight']):r for r in runs}
    effects=[];summary=[];pairs=[]
    for weight in design['weights']:
        selected=[]
        for seed in design['training_seeds']:
            ref,cand=lookup[seed,0.],lookup[seed,weight]
            base={e['seed']:e for e in run_lookup[seed,0.]['episodes']}
            gates=[]
            for episode in run_lookup[seed,weight]['episodes']:
                gate=paired_gates(base[episode['seed']],episode,a,design['config']['evaluation']['maximum_ood_rate'])
                pairs.append(dict(weight=weight,training_seed=seed,evaluation_seed=episode['seed'],**gate))
                gates.append(gate)
            item=dict(weight=weight,training_seed=seed,
                      equivalent_pair_rate=float(np.mean([g['equivalent'] for g in gates])),
                      noninferior_pair_rate=float(np.mean([g['noninferior'] for g in gates])),
                      distance_change_pct=100*(cand['distance_m']/ref['distance_m']-1))
            for model in ('R1','R2','R3'):
                item[model+'_energy_change_pct']=100*(cand[model+'_J']/ref[model+'_J']-1)
                item[model+'_J_per_m_change_pct']=(100*(cand[model+'_J_per_m']/ref[model+'_J_per_m']-1)
                    if cand[model+'_J_per_m'] is not None and ref[model+'_J_per_m'] is not None else None)
            selected.append(item);effects.append(item)
        res=dict(weight=weight,all_seeds_performance_gate=all(e['equivalent_pair_rate']>=a['minimum_paired_pass_rate'] for e in selected),
                 all_seeds_noninferior_gate=all(e['noninferior_pair_rate']>=a['minimum_paired_pass_rate'] for e in selected),
                 **{f'all_seeds_{primary}_decrease': all(e[primary+'_energy_change_pct']<0 for e in selected)},
                 equivalent_pairs=sum(p['equivalent'] for p in pairs if p['weight']==weight),
                 total_pairs=len(selected)*design['evaluation_episodes'])
        for model in ('R1','R2','R3'):
            values=[e[model+'_energy_change_pct'] for e in selected]
            res[model]=interval(values,a['bootstrap_samples'],a['bootstrap_seed'])
            res[model]['decreasing_training_seeds']=sum(v<0 for v in values)
        res['distance_change_pct']=interval([e['distance_change_pct'] for e in selected],a['bootstrap_samples'],a['bootstrap_seed'])
        res[primary+'_p_one_sided']=sign_flip_p([e[primary+'_energy_change_pct'] for e in selected])
        summary.append(res)
    for res,p in zip(summary,holm([s[primary+'_p_one_sided'] for s in summary])):
        res[primary+'_p_holm']=p
        res['primary_criterion_passed']=bool(len(design['training_seeds'])>=a['minimum_replicates']
            and res['all_seeds_performance_gate'] and res[f'all_seeds_{primary}_decrease']
            and p<a['familywise_alpha'] and all(res[m]['mean']<0 for m in ('R1','R2','R3') if m!=primary))
        res['practical_criterion_passed']=bool(res['primary_criterion_passed']
            and res[primary]['mean']<=-a['meaningful_energy_reduction_pct'])
    save_csv(root/'seed_metrics.csv',rows)
    save_csv(root/'paired_seed_effects.csv',effects)
    save_csv(root/'evaluation_pair_gates.csv',pairs)
    output=dict(primary_model=primary,independent_unit='training seed',n_training_seeds=len(design['training_seeds']),
                total_training_runs=len(runs),total_evaluation_episodes=sum(len(r['episodes']) for r in runs),
                interpretation='Negative change is lower predicted energy. Ineligible coefficients do not support savings. '
                'All seeds retained in effects; the performance gate is applied to conclusions, not by dropping failures. '
                'Bootstrap CI is descriptive with n=6; sign-flip inference assumes null sign exchangeability. '
                'No claim of measured hardware energy savings.',coefficients=summary)
    write_json(root/'sweep_summary.json',output)
    plot(root,rows,effects,summary,primary)
    print(json.dumps(output,indent=2))
    return output


def plot(root,rows,effects,summary,primary='R2'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    weights=sorted({r['weight'] for r in rows})
    labels=['Task-only' if w==0 else f'{primary} λ={w:g}' for w in weights]
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,metric,label in zip(axs.ravel(),['mean_speed_m_s',primary+'_J',primary+'_J_per_m','ood_rate'],
                             ['Forward speed [m/s]',f'{primary} predicted energy [J / 20 s]',
                              f'{primary} predicted energy / distance [J/m]\n(all 10 episodes must succeed)','OOD time fraction']):
        for i,w in enumerate(weights):
            values=[r[metric] for r in rows if r['weight']==w and r[metric] is not None]
            if values:
                ax.scatter(i+np.linspace(-.08,.08,len(values)),values,s=25)
                ax.errorbar(i,np.mean(values),yerr=np.std(values,ddof=1) if len(values)>1 else 0,
                            fmt='_',c='black',capsize=4)
        ax.set_xticks(range(len(weights)),labels);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    axs[0,0].axhline(.0108,color='gray',ls=':',label='Minimum task speed');axs[0,0].legend(fontsize=8)
    fig.suptitle('Fixed 100k budget: six independent training seeds per condition\nPoints = seed means; error bars = sample SD; failures retained')
    fig.savefig(root/'sweep_overview.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,model in zip(axs,['R1','R2','R3']):
        for i,res in enumerate(summary):
            seed_effects=[e for e in effects if e['weight']==res['weight']]
            for j,e in enumerate(seed_effects):
                marker='o' if e['equivalent_pair_rate']>=.9 else 'x'
                ax.scatter(i+(j-2.5)*.03,e[model+'_energy_change_pct'],marker=marker,c=f'C{i}',s=35)
            s=res[model];lo,hi=s['ci95']
            ax.errorbar(i,s['mean'],yerr=[[s['mean']-lo],[hi-s['mean']]],fmt='_',c='black',capsize=5)
        ax.axhline(0,c='gray',lw=1);ax.set_title(model);ax.set_ylabel('Predicted total energy change [%]')
        ax.set_xticks(range(len(summary)),[f"λ={s['weight']:g}" for s in summary]);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Paired with Task-only; negative = lower prediction\nDots = training seeds; crosses = performance gate failed; bars = bootstrap 95% CI')
    fig.savefig(root/'paired_energy_effects.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    analyze(parser.parse_args().directory)
