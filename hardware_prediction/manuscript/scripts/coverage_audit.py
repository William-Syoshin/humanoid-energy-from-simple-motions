"""Retrospective coverage-controlled subsets; selection never reads test power."""
import json
import numpy as np
import pandas as pd
import transfer_audit as z


def main():
    f=z.read(z.R/'models/frozen.json'); manifest=z.a.manifest(z.R/'trials.csv')
    tr,_=z.a.load(manifest,z.R/'aligned',['train'],f['schema']);va,_=z.a.load(manifest,z.R/'aligned',['validation'],f['schema'])
    wb=[z.whole_body(r) for r in z.rows(z.PAPER/'data/primary_test_trials.csv')]
    G=np.array([d['X']['R3'].T@d['X']['R3'] for d in tr]);xy=np.array([d['X']['R3'].T@d['p'] for d in tr]);ns=np.array([len(d['p']) for d in tr])
    vx=np.concatenate([d['X']['R3'] for d in va]);vy=np.concatenate([d['p'] for d in va])
    single={j:[] for j in range(17)};holds={k:[] for k in ['UPRIGHT','SQUAT','SUPINE']}
    for i,d in enumerate(tr):
        level=d['meta']['level']
        if level=='L1':
            df=pd.read_csv(d['path']);amp=np.array([np.ptp(df[f'cmd_{j}']) for j in range(17)])
            assert (amp>1e-6).sum()==1
            single[int(np.argmax(amp))].append(i)
        if level=='L0':
            holds[next(k for k in holds if k in d['meta']['trial_id'])].append(i)
    assert all(len(v)==3 for v in single.values()) and all(len(v)==2 for v in holds.values())
    full=np.arange(len(tr));pools={'static_single':np.array([i for i,d in enumerate(tr) if d['meta']['level'] in ('L0','L1')]),
             'through_pairs':np.array([i for i,d in enumerate(tr) if d['meta']['level'] in ('L0','L1','L2')]),'all_levels':full}
    output=[];subsets=[]
    for seed in range(20):
        rng=np.random.default_rng(20260928+seed)
        mandatory=np.array([rng.choice(single[j]) for j in range(17)]+[rng.choice(holds[k]) for k in holds])
        remaining=rng.permutation(np.setdiff1d(full,mandatory));order=np.r_[mandatory,remaining]
        jobs=[('coverage_budget',n,order[:n]) for n in (20,24,48,80,116)]
        for pool,indices in pools.items():jobs.append(('coverage_'+pool,48,np.r_[mandatory,rng.permutation(np.setdiff1d(indices,mandatory))[:28]]))
        for experiment,n,idx in jobs:
            subsets.append(dict(seed=seed,experiment=experiment,n=n,trial_ids=[tr[i]['meta']['trial_id'] for i in idx]))
            for model in ('R1','R2','R3'):
                cols=z.COLS[model];g=G[idx].sum(0);v=xy[idx].sum(0);count=int(ns[idx].sum())
                candidates=[]
                for alpha in z.GRID:
                    beta=z.raw_fit(g,v,count,cols,alpha);val=np.sqrt(np.mean((vx[:,cols]@beta-vy)**2))
                    candidates.append((val,alpha,beta))
                val,alpha,beta=min(candidates,key=lambda r:r[0]);mm=[z.score(d,cols,beta) for d in wb]
                output.append(dict(experiment=experiment,seed=seed,n=n,model=model,alpha=alpha,validation_RMSE_W=val,
                   energy_error_pct=np.mean([m['energy_error_pct'] for m in mm]),mean_trial_RMSE_W=np.mean([m['power_RMSE_W'] for m in mm])))
    out=z.OUT/'coverage';out.mkdir(exist_ok=False)
    pd.DataFrame(output).to_csv(out/'metrics.csv',index=False)
    (out/'subsets.json').write_text(json.dumps(subsets,indent=2),encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps(dict(script_sha256=z.digest(__file__),
        mandatory='One L1 trial per joint plus one L0 hold per posture (20 trials); remaining trials uniformly sampled without replacement.',
        exploratory=True,selection_uses_test_power=False,seeds=list(range(20)),seed_offset=20260928),indent=2),encoding='utf-8')
    print(pd.DataFrame(output).groupby(['experiment','model','n'])[['energy_error_pct','mean_trial_RMSE_W']].median().round(3).to_string())


if __name__=='__main__':main()
