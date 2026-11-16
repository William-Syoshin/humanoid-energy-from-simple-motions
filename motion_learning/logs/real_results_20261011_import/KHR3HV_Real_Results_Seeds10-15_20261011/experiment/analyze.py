"""Measured V*I integration and seed-level Sim/Real plots, no synthetic Real data."""
import csv
from pathlib import Path
import numpy as np
from common import read,save,sha,rows,integrate_power


def write_csv(path,data):
    if not data:return
    fields=list(dict.fromkeys(k for row in data for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)


def analyze(bundle,plan_path,raw,out,evaluator='R2',allow_mock=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan=read(plan_path)
    if sha(bundle/'manifest.json')!=plan['manifest_sha256']:raise ValueError('Changed manifest')
    planned={x['trial_id']:x for x in plan['trials']};attempts={};unregistered=[]
    for p in raw.rglob('meta.json'):
        meta=read(p)
        if meta.get('trial_id') not in planned:unregistered.append(str(p.parent));continue
        tid=meta['trial_id'];attempts.setdefault(tid,[]).append(p.parent)
    if any(len(v)>1 for v in attempts.values()):raise ValueError('Multiple attempts for a trial: resolve in a new documented plan; never select the best attempt')
    out.mkdir(parents=True,exist_ok=False);table=[];motion_by_id={}
    any_mock=False
    for tid,tr in planned.items():
        motion=read(bundle/tr['motion']);motion_by_id[tid]=motion
        if sha(bundle/tr['motion'])!=tr['motion_sha256']:raise ValueError('Changed motion')
        row=dict(**tr,measurement_status='not_attempted',reason='',real_energy_J=None,real_speed_m_s=None,
                 real_distance_m=None,outcome='',mock=False,sim_energy_J=motion['sim']['predicted_energy_J'][evaluator],
                 sim_speed_m_s=motion['sim']['mean_speed_m_s'],replay_duration_s=motion['duration_s'],
                 sim_distance_m=motion['sim'].get('distance_m'),sim_goal_time_s=motion['duration_s'],
                 sim_goal_energy_J=motion['sim']['predicted_energy_J'][evaluator],
                 real_goal_time_s=None,real_goal_energy_J=None,goal_measurement_status='not_available')
        if tid in attempts:
            p=attempts[tid][0];meta=read(p/'meta.json');row['raw']=str(p)
            row['mock']=bool(meta.get('mock'));any_mock|=row['mock']
            if row['mock'] and not allow_mock:raise ValueError('Mock data are forbidden in a Real figure; use --allow-mock for a watermarked software demonstration')
            try:
                if meta['plan_sha256']!=sha(plan_path) or meta['motion_sha256']!=tr['motion_sha256']:raise ValueError('Plan/motion provenance differs')
                if read(p/'trial.json')!=tr:raise ValueError('Trial metadata mismatch')
                if meta['status']!='complete':raise ValueError('Acquisition failed: '+str(meta.get('error')))
                duration=meta['score_end']-meta['score_start']
                if abs(duration-motion['duration_s'])>1e-6:raise ValueError('Wrong measurement duration')
                commands=rows(p/'commands.csv')
                if len(commands)!=len(motion['commands']) or any(x['ok']!='1' or int(x['index'])!=k for k,x in enumerate(commands)):
                    raise ValueError('Incomplete command acknowledgements')
                cfg=read(p/'config.json')
                if sha(p/'config.json')!=meta['config_sha256'] or sha(p/'motion.json')!=tr['motion_sha256']:
                    raise ValueError('Changed raw motion/config')
                if not row['mock']:
                    for k,x in enumerate(commands):
                        due=meta['score_start']+motion['commands'][k]['t_s']
                        before=float(x['request_s']);after=float(x['response_s'])
                        if not all(np.isfinite(v) for v in [due,before,after]) or before<due or after<before or after-due>cfg['max_command_lateness_s']:
                            raise ValueError('Invalid command timing')
                if not row['mock'] and (not cfg['timing']['verified'] or not cfg['power_calibration']['verified']):raise ValueError('Unverified measurement calibration')
                metrics=integrate_power(rows(p/'power.csv'),meta['score_start'],meta['score_end'],cfg)
                row.update(real_energy_J=metrics['energy_J'],mean_power_W=metrics['mean_power_W'],timing_uncertainty_s=metrics['timing_uncertainty_s'],negative_power_samples=metrics['negative_power_samples'])
                if not (p/'observation.json').exists():raise ValueError('Distance/outcome annotation missing')
                obs=read(p/'observation.json');distance=float(obs['distance_m'])
                if not np.isfinite(distance) or obs['outcome'] not in ['success','fall','deviation','operator_stop','other_failure','shortfall']:raise ValueError('Invalid outcome annotation')
                if obs['outcome']=='success' and distance<.2:raise ValueError('Success annotated below 20 cm')
                row.update(measurement_status='valid',outcome=obs['outcome'],real_distance_m=distance,real_speed_m_s=distance/duration,
                           distance_method=obs['distance_method'],real_energy_per_m_J=metrics['energy_J']/distance if obs['outcome']=='success' and distance>0 else None)
                goal_time=obs.get('goal_time_s');uncertainty=obs.get('goal_time_uncertainty_s')
                if goal_time is not None:
                    if (obs['outcome']!='success' or not np.isfinite(goal_time) or not 0<goal_time<=duration+1e-7
                        or uncertainty is None or not np.isfinite(uncertainty) or uncertainty<0):
                        raise ValueError('Invalid independently observed first-arrival time')
                    goal=integrate_power(rows(p/'power.csv'),meta['score_start'],meta['score_start']+goal_time,cfg)
                    row.update(real_goal_time_s=goal_time,real_goal_energy_J=goal['energy_J'],
                               goal_time_uncertainty_s=uncertainty,goal_measurement_status='observed_success')
            except (ValueError,KeyError,TypeError,FileNotFoundError) as exc:
                row.update(measurement_status='invalid_or_incomplete',reason=str(exc))
        table.append(row)
    write_csv(out/'all_trials.csv',table)
    # Physical repetitions are nested under training seeds. A one-seed pilot
    # displays replay SD; a multi-seed design displays SD of complete seed means.
    summary=[];paired=[]
    conditions=['R1','R2','R3','Task-only']
    metrics=['speed_m_s','energy_J','goal_time_s','goal_energy_J']
    for condition in conditions:
        group=[r for r in table if r['condition']==condition]
        for seed in sorted({r['training_seed'] for r in group}):
            g=[r for r in group if r['training_seed']==seed]
            valid=[r for r in g if r['measurement_status']=='valid']
            goals=[r for r in valid if r['goal_measurement_status']=='observed_success']
            complete=len(valid)==len(g)==plan['repeats'];goal_complete=len(goals)==len(g)==plan['repeats']
            row=dict(condition=condition,training_seed=seed,planned_repeats=len(g),valid_repeats=len(valid),
                observed_goal_repeats=len(goals),real_complete=complete,goal_complete=goal_complete,
                successful_repeats=sum(r['outcome']=='success' for r in valid),
                outcome_success_fraction=sum(r['outcome']=='success' for r in valid)/len(g))
            for key in metrics:
                vals=goals if key.startswith('goal_') else valid
                usable=goal_complete if key.startswith('goal_') else complete
                row['sim_'+key]=g[0]['sim_'+key]
                row['real_'+key]=float(np.mean([r['real_'+key] for r in vals])) if usable else None
                row['repeat_sd_'+key]=float(np.std([r['real_'+key] for r in vals],ddof=1)) if usable and len(vals)>1 else None
            summary.append(row)
    for condition in conditions[:-1]:
        for a in [r for r in summary if r['condition']==condition]:
            b=next(r for r in summary if r['condition']=='Task-only' and r['training_seed']==a['training_seed'])
            if a['real_complete'] and b['real_complete']:
                paired.append(dict(condition=condition,training_seed=a['training_seed'],
                    replay_energy_change_J=a['real_energy_J']-b['real_energy_J'],
                    replay_energy_change_pct=100*(a['real_energy_J']/b['real_energy_J']-1) if b['real_energy_J']>0 else None,
                    both_all_repeats_successful=a['outcome_success_fraction']==b['outcome_success_fraction']==1,
                    observed_20cm_energy_change_J=a['real_goal_energy_J']-b['real_goal_energy_J'] if a['goal_complete'] and b['goal_complete'] else None))
    pilot=len({r['training_seed'] for r in table})==1
    plot_summary=[]
    for kind,keys,ylabels in [('replay',['speed_m_s','energy_J'],['Forward speed (m/s)','Energy during replay (J)']),
                             ('goal',['goal_time_s','goal_energy_J'],['Time to 20 cm (s)','Energy to first 20 cm crossing (J)'])]:
        fig,axes=plt.subplots(2,1,figsize=(9,7.5),sharex=True,layout='constrained')
        incomplete=False;ticklabels=[]
        for ci,condition in enumerate(conditions):
            group=[r for r in table if r['condition']==condition]
            seed_values=[r for r in summary if r['condition']==condition]
            complete_key='goal_complete' if kind=='goal' else 'real_complete'
            for domain,offset,color in [('sim',-.14,'#1877ad'),('real',.14,'#c34248')]:
                if domain=='sim':g=seed_values
                elif pilot:
                    g=[r for r in group if r['measurement_status']=='valid' and (kind!='goal' or r['goal_measurement_status']=='observed_success')]
                else:g=[r for r in seed_values if r[complete_key]]
                expected=len(group) if pilot and domain=='real' else len(seed_values)
                if domain=='real' and len(g)!=expected:incomplete=True
                for ax,key in zip(axes,keys):
                    vals=[r[domain+'_'+key] for r in g]
                    mean=float(np.mean(vals)) if vals else None
                    sd=float(np.std(vals,ddof=1)) if len(vals)>1 else None
                    plot_summary.append(dict(figure=kind,condition=condition,domain=domain,metric=key,n=len(vals),expected_n=expected,
                        mean=mean,sample_sd=sd,unit='physical replay of one fixed seed' if pilot and domain=='real' else 'training seed'))
                    if vals:
                        if len(vals)>1:ax.scatter(np.linspace(ci+offset-.035,ci+offset+.035,len(vals)),vals,color=color,s=18,alpha=.5)
                        ax.errorbar(ci+offset,mean,yerr=sd,fmt='o',color=color,capsize=6,elinewidth=1.6)
                    else:ax.text(ci+offset,.04,'no data',transform=ax.get_xaxis_transform(),rotation=90,color=color,ha='center',fontsize=8)
                axes[0].text(ci+offset,1.01,f'n={len(g)}/{expected}',transform=axes[0].get_xaxis_transform(),ha='center',fontsize=8,color=color)
            success=sum(r['measurement_status']=='valid' and r['outcome']=='success' for r in group)
            attempted=sum(r['measurement_status']!='not_attempted' for r in group)
            ticklabels.append(condition+'\n'+(f'Real success: {success}/{len(group)} planned' if attempted else 'Real: pending'))
        for ax,label in zip(axes,ylabels):
            ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.set_xlim(-.5,3.5);ax.margins(y=.2)
        axes[1].set_xticks(range(4),ticklabels,fontsize=9);axes[1].set_xlabel('Training reward (energy coefficient 0.3; Task-only 0)')
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([],[],marker='o',color='#1877ad',linestyle='none',label=f'Sim (energy predicted by common {evaluator})'),
            Line2D([],[],marker='o',color='#c34248',linestyle='none',label='MOCK — not hardware' if any_mock else 'Real (measured)')],loc='outside lower center',ncol=2,fontsize=8)
        units='One fixed training seed: Sim single trajectory; Real mean ± SD of physical repeats' if pilot else 'Mean ± SD across training seeds; physical repeats averaged within seed'
        title='Variable-duration command replay' if kind=='replay' else 'First 20 cm arrival (independently timed on hardware)'
        fig.suptitle(title+(' | DATA INCOMPLETE' if incomplete else '')+'\n'+units,fontsize=10)
        if any_mock:fig.text(.5,.5,'SOFTWARE TEST — NOT REAL DATA',ha='center',rotation=25,color='red',alpha=.3,fontsize=22)
        for suffix in ['png','pdf','svg']:fig.savefig(out/f'sim_real_{kind}.{suffix}',dpi=220)
        plt.close(fig)
    write_csv(out/'seed_means.csv',summary);write_csv(out/'paired_seed_effects.csv',paired);write_csv(out/'plot_summary.csv',plot_summary)
    save(out/'analysis.json',dict(sim_evaluator=evaluator,planned_trials=len(table),valid_trials=sum(x['measurement_status']=='valid' for x in table),
        observed_goal_trials=sum(x['goal_measurement_status']=='observed_success' and x['measurement_status']=='valid' for x in table),
        mock=any_mock,unregistered_attempts=unregistered,pilot_one_training_seed=pilot,
        independent_unit='training seed for learning generalization; physical repeats only characterize replay variability',
        pilot_plot='Sim one trajectory without SD; Real mean and sample SD of available physical repeats, incomplete n explicitly shown',
        energy='signed calibrated V*I trapezoidal integral; no idle subtraction; replay and observed first-arrival windows reported separately',
        interpretation='Replay is open loop. Arrival time must be independently observed, never inferred from average speed. Failures remain in replay tables/plots. Goal plots are conditional on observed success and must be read with success counts. No automatic significance tests.'))
    print(out)
