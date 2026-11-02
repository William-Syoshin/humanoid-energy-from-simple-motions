"""Audit complete distance experiments and report all conditions without selection."""
import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rl.run_sweep import atomic_json, verify_source, digest
from energy.power import FrozenPower, feature_matrix

MODELS=('R1','R2','R3')
CONDITIONS=('task_only',)+MODELS


def mean(values): return float(np.mean(values)) if len(values) else None

def sd(values): return float(np.std(values,ddof=1)) if len(values)>1 else None

def holm(ps):
    result=[None]*len(ps);running=0.
    for rank,i in enumerate(sorted(range(len(ps)),key=lambda i:ps[i])):
        running=max(running,min(1.,ps[i]*(len(ps)-rank)));result[i]=running
    return result


def paired_analysis(runs,plan):
    lookup={(r['condition'],r['training_seed']):r['episodes'] for r in runs}
    settings=plan['analysis']; comparisons=[]
    for condition in MODELS:
        seeds=[]
        for seed in plan['training_seeds']:
            a={e['seed']:e for e in lookup[condition,seed]}
            b={e['seed']:e for e in lookup['task_only',seed]}
            assert set(a)==set(b)
            good=[k for k in sorted(a) if a[k]['task_success'] and b[k]['task_success']]
            gate=(mean([e['task_success'] for e in a.values()])>=settings['minimum_success_rate_per_seed']
                  and mean([e['task_success'] for e in b.values()])>=settings['minimum_success_rate_per_seed']
                  and len(good)/len(a)>=settings['minimum_paired_success_fraction']
                  and all(e['ood_rate']<=settings['maximum_ood_rate'] and e['negative_power_rate']==0
                          for e in list(a.values())+list(b.values())))
            gate=gate and all(a[k]['goal_overshoot_m']<=settings['maximum_goal_overshoot_m']
                              and b[k]['goal_overshoot_m']<=settings['maximum_goal_overshoot_m'] for k in good)
            seeds.append(dict(training_seed=seed,paired_successes=len(good),total_pairs=len(a),performance_gate=bool(gate),
                delta_energy_J={m:mean([a[k]['predicted_energy_J'][m]-b[k]['predicted_energy_J'][m] for k in good]) for m in MODELS},
                delta_time_s=mean([a[k]['duration_s']-b[k]['duration_s'] for k in good])))
        complete=all(s['delta_energy_J']['R2'] is not None for s in seeds)
        delta=np.array([s['delta_energy_J']['R2'] for s in seeds],dtype=float) if complete else np.array([])
        p=1.;ci=None
        if complete:
            observed=delta.mean()
            p=float(np.mean([np.mean(delta*np.array(sign))<=observed+1e-12 for sign in itertools.product((-1,1),repeat=len(delta))]))
            rng=np.random.default_rng(settings['bootstrap_seed'])
            bootstrap=rng.choice(delta,size=(settings['bootstrap_samples'],len(delta)),replace=True).mean(axis=1)
            ci=np.quantile(bootstrap,[.025,.975]).tolist()
        comparisons.append(dict(condition=condition,seeds=seeds,R2_p_one_sided=p,R2_delta_bootstrap95_J=ci,
            mean_delta_energy_J={m:mean([s['delta_energy_J'][m] for s in seeds]) if complete else None for m in MODELS},
            all_seed_R2_reduction=bool(complete and np.all(delta<0)),
            performance_gate=all(s['performance_gate'] for s in seeds)))
    for c,p in zip(comparisons,holm([x['R2_p_one_sided'] for x in comparisons])):
        c['R2_p_holm']=p
        c['primary_criterion_passed']=bool(len(plan['training_seeds'])>=6 and c['performance_gate'] and c['all_seed_R2_reduction']
            and p<settings['alpha'] and all(v is not None and v<0 for v in c['mean_delta_energy_J'].values()))
    return comparisons


def audit(root,plan,runs):
    verify_source(root)
    expected={(j['condition'],j['seed']) for j in plan['jobs']}
    assert len(runs)==len(expected)==len(plan['jobs'])
    assert {(r['condition'],r['training_seed']) for r in runs}==expected
    errors=[];checked=0;maxerr=0.;power=FrozenPower(plan['config']['energy']['support'])
    for run in runs:
        folder=Path(run['run']);job=next(j for j in plan['jobs'] if j['condition']==run['condition'] and j['seed']==run['training_seed'])
        cfg=yaml.safe_load((folder/'config.yaml').read_text())
        wanted=json.loads(json.dumps(plan['config']));wanted.update(condition=job['condition'],seed=job['seed']);wanted['energy']['weight']=job['weight']
        assert cfg==wanted,'Config mismatch: '+str(folder)
        tr=json.loads((folder/'training_summary.json').read_text())
        n=cfg['training']['n_steps'];steps=((cfg['training']['total_timesteps']+n-1)//n)*n
        assert tr['status']=='complete' and tr['timesteps']==steps
        status=json.loads((root/'status'/(job['name']+'.json')).read_text())
        assert status['model_sha256']==digest(folder/'final.zip')
        assert len(run['episodes'])==plan['evaluation_episodes']
        assert {e['seed'] for e in run['episodes']}==set(range(20000,20000+plan['evaluation_episodes']))
        for e in run['episodes']:
            ev=folder/'evaluation';seed=e['seed']
            assert e==json.loads((ev/f'episode_{seed}.json').read_text())
            trace=[json.loads(t) for t in (ev/f'trajectory_{seed}.jsonl').read_text().splitlines()]
            preds=[json.loads(t) for t in (ev/f'power_{seed}.jsonl').read_text().splitlines()]
            assert len(trace)==len(preds)
            times=np.array([t['time'] for t in trace]);duration=(len(trace)-1)*.1
            assert np.allclose(np.diff(times),.1,atol=1e-6) and abs(duration-e['duration_s'])<1e-8
            distance=trace[-1]['position'][0]-trace[0]['position'][0]
            assert abs(distance-e['distance_m'])<1e-9
            reached=distance>=.2 and not e['fallen'] and not e['escaped']
            assert reached==e['task_success']==e['reached_goal']
            assert duration<=40.+1e-8
            assert all(t['position'][0]-trace[0]['position'][0]<.2 for t in trace[:-1])
            if not reached and not e['fallen'] and not e['escaped']: assert abs(duration-40.)<1e-8 and e['timed_out']
            q=np.array([t['q'] for t in trace])*power.sign+power.offset
            cmd=np.array([t['cmd'] for t in trace])*power.sign+power.offset
            tilt=[np.arcsin(min(1.,np.linalg.norm(t['accel'][:2])/9.81)) for t in trace]
            x=feature_matrix(q,cmd,tilt,power.edges,power.support,.1)
            bad=(x < power.bounds['feature_min']-1e-8) | (x > power.bounds['feature_max']+1e-8)
            ood=(np.any(bad[:,1:],axis=1) | np.any(q < power.bounds['q_min']-1e-8,axis=1)
                 | np.any(q > power.bounds['q_max']+1e-8,axis=1)
                 | np.any(cmd < power.bounds['cmd_min']-1e-8,axis=1)
                 | np.any(cmd > power.bounds['cmd_max']+1e-8,axis=1))
            assert np.array_equal(ood,[p['ood'] for p in preds])
            assert abs(float(np.trapezoid(ood.astype(float),dx=.1)/duration)-e['ood_rate'])<1e-8
            for m,nfeat in [('R1',69),('R2',89),('R3',95)]:
                predicted=(x[:,:nfeat]/power.params[m+'_scale'])@power.params[m+'_coef']
                assert np.allclose(predicted,[p[m] for p in preds],rtol=1e-9,atol=1e-8)
                integral=float(np.trapezoid(predicted,dx=.1));positive=float(np.trapezoid(np.maximum(0,predicted),dx=.1))
                maxerr=max(maxerr,abs(integral-e['predicted_energy_J'][m]))
                assert abs(integral-e['predicted_energy_J'][m])<1e-6
                assert abs(positive-e['positive_energy_J'][m])<1e-6
            energy=0. if run['condition']=='task_only' else -.3*e['positive_energy_J'][run['condition']]
            assert abs(e['reward_terms']['energy']-energy)<1e-6
            assert e['reward_terms']['speed']==e['reward_terms']['alive']==e['reward_terms']['upright']==0
            assert e['reward_terms']['goal']==(200. if reached else 0.)
            checked+=1
    assert checked==plan['planned_evaluation_episodes']
    out=dict(passed=True,training_runs=len(runs),evaluation_episodes=checked,source_verified=True,
             maximum_offline_energy_difference_J=maxerr,checked_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(root/'audit.json',out);return out


def finalize(root,project):
    plan=json.loads((root/'design.json').read_text());runs=json.loads((root/'results.json').read_text())
    assert json.loads((root/'completion.json').read_text())['all_complete']
    audit_record=audit(root,plan,runs)
    comparison=paired_analysis(runs,plan)
    rows=[];seeds=[];summary=[]
    for run in runs:
        good=[e for e in run['episodes'] if e['task_success']]
        r=dict(condition=run['condition'],training_seed=run['training_seed'],success_rate=mean([e['task_success'] for e in run['episodes']]),
               fall_rate=mean([e['fallen'] for e in run['episodes']]),timeout_rate=mean([e['timed_out'] for e in run['episodes']]),
               time_success_s=mean([e['duration_s'] for e in good]),distance_all_m=mean([e['distance_m'] for e in run['episodes']]),
               ood_rate=mean([e['ood_rate'] for e in run['episodes']]))
        for m in MODELS:
            r[m+'_energy_success_J']=mean([e['predicted_energy_J'][m] for e in good])
            r[m+'_energy_all_J']=mean([e['predicted_energy_J'][m] for e in run['episodes']])
        seeds.append(r)
        for e in run['episodes']:
            row=dict(condition=run['condition'],training_seed=run['training_seed'],evaluation_seed=e['seed'])
            row.update({k:e[k] for k in ('task_success','termination_reason','distance_m','duration_s','mean_speed_m_s','ood_rate','negative_power_rate','goal_overshoot_m')})
            row.update({m+'_energy_J':e['predicted_energy_J'][m] for m in MODELS});rows.append(row)
    for cond in CONDITIONS:
        ss=[s for s in seeds if s['condition']==cond];r=dict(condition=cond)
        for metric in ss[0]:
            if metric in ('condition','training_seed'):continue
            values=[s[metric] for s in ss if s[metric] is not None]
            r[metric]=dict(mean=mean(values),sd=sd(values),n_seeds=len(values))
        summary.append(r)
    for name,data in [('all_episodes.csv',rows),('seed_summary.csv',seeds)]:
        with (root/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    atomic_json(root/'analysis.json',dict(summary=summary,comparisons=comparison,analysis=plan['analysis'],
        caveat='Successful-only energy is conditional, never evidence of savings unless performance gates pass; all failures retained.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4.6))
    for ax,metric,title in zip(axes,('success_rate','time_success_s','R2_energy_success_J'),
                              ('Goal success rate','Time to goal (s), successful only','Predicted R2 energy (J), successful only')):
        for i,r in enumerate(summary):
            value=r[metric]
            if value['mean'] is not None:
                ax.errorbar(i,value['mean'],yerr=value['sd'] or 0,fmt='o',capsize=5)
                vals=[s[metric] for s in seeds if s['condition']==r['condition'] and s[metric] is not None]
                ax.scatter(np.full(len(vals),i),vals,alpha=.35,s=12)
        ax.set_xticks(range(4),['Task-only','R1','R2','R3']);ax.set_title(title,fontsize=10);ax.grid(alpha=.2)
    axes[0].set_ylim(-.05,1.05)
    fig.suptitle('20 cm goal | lambda = 0.3 | mean +/- SD across training seeds')
    fig.text(.5,.02,'Simulation predictions, not measured hardware energy. Failed episodes remain in all_episodes.csv.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    for ext in ('png','pdf','svg'):fig.savefig(root/f'distance_results.{ext}',dpi=180)
    plt.close(fig)
    label=root.name
    report=project/'docs'/f'{label}_RESULTS.md';report.parent.mkdir(exist_ok=True)
    def fmt(v):return 'NA' if v is None else f'{v:.3f}'
    lines=['# 20 cm到達課題：Task-only / R1 / R2 / R3','',
           f"{len(runs)}学習・{len(rows)}評価。λ=0.3、学習seed={plan['training_seeds']}。40秒以内の到達課題。",'',
           '速度追従報酬は使用しない。成功時は最初の10 Hz評価境界で終了。到達までの予測総エネルギーを積分する。',
           '共通R2評価器の対応比較を主要解析とし、R1/R3予測も全条件で報告する。実機省エネの実証ではない。','',
           '|条件|到達成功率|成功時所要時間 s|成功時R1 J|成功時R2 J|成功時R3 J|',
           '|---|---:|---:|---:|---:|---:|']
    for s in summary:
        values=[s[k] for k in ('success_rate','time_success_s','R1_energy_success_J','R2_energy_success_J','R3_energy_success_J')]
        lines.append('|'+s['condition']+'|'+'|'.join(fmt(v['mean'])+' ± '+fmt(v['sd']) for v in values)+'|')
    lines+=['','値は学習seed間の平均±標本SD。成功時の値は条件付き集計であり、未到達・転倒による低消費量を省エネ成功としない。',
            f"各{plan['evaluation_episodes']}評価はseed内で平均し、独立学習数として数えない。全試行はall_episodes.csv、seed別はseed_summary.csv。",'',
            '|比較 vs Task-only|対応R2差 J|Holm p（3比較）|低下seed数|主要基準|','|---|---:|---:|---:|---|']
    for c in comparison:
        n=sum(s['delta_energy_J']['R2'] is not None and s['delta_energy_J']['R2']<0 for s in c['seeds'])
        lines.append(f"|{c['condition']}|{fmt(c['mean_delta_energy_J']['R2'])}|{c['R2_p_holm']:.5f}|{n}/{len(c['seeds'])}|{'達成' if c['primary_criterion_passed'] else '未達'}|")
    winners=[c['condition'] for c in comparison if c['primary_criterion_passed']]
    lines+=['',('主要基準を満たした条件：'+', '.join(winners)) if winners else '**全条件で主要基準は未達。頑健な省エネ到達を確認できたとは結論しない。**','',
            '主要基準：全seedで両方策の成功率・対応成功割合90%以上、OOD率10%以下、負予測なし、成功対の超過距離5mm以内、全6seedでR2低下、3比較Holm p<0.05、R1/R3平均も低下。',
            '対応成功例のみの検定は選択の影響がある。成功率・所要時間・失敗を併記して解釈する。',
            'R1～R3は共通の実機データに由来し、複数予測器の一致は独立した実機検証ではない。','',
            f'監査：{audit_record["training_runs"]}学習・{audit_record["evaluation_episodes"]}評価、凍結ソース、条件、学習量、終端、全モデルのオフライン再積分を確認。',
            f'[全解析](../logs/{label}/analysis.json) / [図](../logs/{label}/distance_results.png) / [全試行](../logs/{label}/all_episodes.csv)','',
            '実機移行には可変長（到達まで）の指令書き出しと停止動作、サーボ対応・電力計測の検証が必要。既存の20秒固定Windowsパッケージへそのまま入れ替えない。']
    report.write_text('\n'.join(lines)+'\n')
    if not plan['smoke_test']:
        dest=project/'models'/label;dest.mkdir(exist_ok=True)
        for run in runs:
            folder=Path(run['run']);target=dest/folder.name;target.mkdir(exist_ok=True)
            for name in ('final.zip','config.yaml'):shutil.copy2(folder/name,target/name)
        for path in (project/'README.md',project/'docs/RL_QUICKSTART.md'):
            link=f'[{label}：20 cm到達実験結果]('+('docs/' if path.name=='README.md' else '')+report.name+')'
            text=path.read_text()
            if report.name not in text:path.write_text(text+'\n'+link+'\n')
    atomic_json(root/'finalization.json',dict(status='complete',report=str(report),audit=audit_record,
                finished_utc=datetime.now(timezone.utc).isoformat(),primary_passed=winners,gui_recheck='pending'))


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--project',type=Path,required=True);a=p.parse_args()
    finalize(a.root.resolve(),a.project.resolve())

if __name__=='__main__': main()
