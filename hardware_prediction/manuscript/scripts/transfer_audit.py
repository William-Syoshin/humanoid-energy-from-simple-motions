"""Retrospective analyses: no hardware, no edits to frozen models or observations.
Hyperparameters use only the original validation set. Seeds/subsets are saved.
"""
import csv
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PAPER=Path(__file__).resolve().parents[1]
H=PAPER.parent/'hardware_experiment'  # consolidated package root
sys.path.insert(0,str(H))
import analysis as a
from processing import power_grid
from orientation import estimate_accelerometer
from core import read, rows, digest
R=H/'regression_results/ridge_20260915_143619'
OUT=PAPER/'data/transfer_audit_20260928'
GRID=[.001,.1,10.,100.]
COLS={'R1':np.arange(69),'R2':np.arange(89),'R2_tilt':np.arange(91),
      'R2_support':np.r_[np.arange(89),np.arange(91,95)],'R3':np.arange(95),
      'context_only':np.r_[0,np.arange(89,95)]}
NAMES=['Cheers','Clapping','Supine get-up','Safe walking','Kicking',
       'Forward crawling','Left crawling turn','Right crawling turn','Backward crawling']


def whole_body(row):
    raw=H/'session_v3_test_10hz/raw'/row['trial_id']
    meta,cfg=read(raw/'meta.json'),read(raw/'config.json')
    pw=rows(raw/'power.csv');st=rows(raw/'state.csv')
    pt=np.array([float(r['device_s']) for r in pw]);pc=np.array([float(r['pc_s']) for r in pw])
    pt+=np.median(pc-pt)
    end=min(meta['score_end_s'],pt[-1],min(max(float(r['pc_s']) for r in st if int(r['joint'])==i) for i in range(17)))
    t=meta['score_start_s']+np.arange(math.floor((end-meta['score_start_s'])*10)+1)/10
    power=power_grid(pt,np.array([float(r['voltage'])*float(r['current']) for r in pw]),t,.3)
    q,cmd=[],[]
    for i,j in enumerate(cfg['joints']):
        s=[r for r in st if int(r['joint'])==i];tt=np.array([float(r['pc_s']) for r in s])
        factor=math.radians(135)/4000*j['sign']
        q.append(np.interp(t,tt,[(int(r['position'])-j['zero_pos'])*factor for r in s]))
        cmd.append(np.interp(t,tt,[(int(r['target_register'])-j['zero_pos'])*factor for r in s]))
    q=np.array(q).T;cmd=np.array(cmd).T
    padded=np.pad(q,((2,2),(0,0)),mode='edge')
    vel=np.gradient(sum(padded[k:k+len(q)] for k in range(5))/5,t,axis=0);err=cmd-q
    ad=[]
    for r in rows(raw/'ad.csv'):
        ok,v=json.loads(r['values_json']);assert ok
        ad.append(dict(pc_s=float(r['pc_s']),ad3=v[3],ad4=v[4]))
    ori=estimate_accelerometer(ad,cfg['tilt_calibration'],.3)
    tilt=np.interp(t,[r['pc_s'] for r in ori],[r['tilt_rad'] for r in ori]);h=1-np.cos(tilt)
    ground='evaluation_ground' in row['evaluation_path']
    support=np.tile([0.,1.,0.,0.] if ground else [1.,0.,0.,0.],(len(t),1))
    x=np.column_stack([np.ones(len(t)),abs(vel),vel**2,abs(err),err**2,
          np.column_stack([abs(err[:,i]*err[:,j]) for i,j in a.EDGES]),h,h*h,support])
    d=dict(meta=dict(trial_id=row['trial_id'],task=row['motion'],success='unknown'),t=t-t[0],p=power,X={'R3':x})
    for m,n in [('R0',1),('R1',69),('R2',89)]:d['X'][m]=x[:,:n]
    return d


def raw_fit(g,xy,n,cols,alpha):
    gg=g[np.ix_(cols,cols)];s=np.maximum(np.sqrt(np.diag(gg)/n),1e-6);s[0]=1
    pen=np.eye(len(cols))*alpha;pen[0,0]=0
    coef=np.linalg.solve(gg/s[:,None]/s[None,:]+pen,xy[cols]/s)
    return coef/s


def score(d,cols,beta):
    yp=d['X']['R3'][:,cols]@beta
    return a.metrics(d,'audit',yp)


def main():
    OUT.mkdir(exist_ok=False)
    f=read(R/'models/frozen.json'); frozen=dict(np.load(R/'models/frozen_models.npz'))
    assert digest(H/'analysis.py')==f['analysis_sha256']
    assert digest(R/'models/frozen_models.npz')==f['model_sha256']
    for v in f['training_validation_hashes'].values():assert digest(R/'aligned'/v['file'])==v['sha256']
    manifest=a.manifest(R/'trials.csv')
    tr,_=a.load(manifest,R/'aligned',['train'],f['schema']);va,_=a.load(manifest,R/'aligned',['validation'],f['schema'])
    primary=rows(PAPER/'data/primary_test_trials.csv')
    wb=[whole_body(r) for r in primary]
    for d,name in zip(wb,NAMES):d['label']=name
    # Reconstruct all four published predictions before using any new fit.
    maxdiff=0.
    for d in wb:
        saved=pd.read_csv(PAPER/'data'/(d['meta']['trial_id']+'_predictions.csv'))
        assert np.allclose(d['p'],saved.actual_W,atol=1e-10)
        for m in a.MODELS:
            diff=np.max(abs(a.predict(d,m,(frozen[m+'_scale'],frozen[m+'_coef']))-saved[m+'_W']))
            maxdiff=max(maxdiff,float(diff));assert diff<1e-8,(d['meta'],m,diff)
    tests=[('whole_body',d) for d in wb]
    G=np.array([d['X']['R3'].T@d['X']['R3'] for d in tr])
    XY=np.array([d['X']['R3'].T@d['p'] for d in tr]);ns=np.array([len(d['p']) for d in tr])
    vx=np.concatenate([d['X']['R3'] for d in va]);vy=np.concatenate([d['p'] for d in va])
    def train(indices,cols,alpha):return raw_fit(G[indices].sum(0),XY[indices].sum(0),int(ns[indices].sum()),cols,alpha)
    def choose(indices,cols):
        fits=[(float(np.sqrt(np.mean((vx[:,cols]@(b:=train(indices,cols,alpha))-vy)**2))),alpha,b) for alpha in GRID]
        return min(fits,key=lambda r:r[0])
    full=np.arange(len(tr));ablation=[];selection=[];fitted={}
    for variant,cols in COLS.items():
        val,best,beta=choose(full,cols);selection.append(dict(variant=variant,alpha=best,validation_RMSE_W=val))
        fitted[variant+'_coef']=beta
        for alpha in GRID:
            b=train(full,cols,alpha)
            if variant in ('R1','R2','R3') and alpha==f['models'][variant]['alpha']:
                assert np.allclose(b,frozen[variant+'_coef']/frozen[variant+'_scale'],atol=1e-8)
            for group,d in tests:
                rec=score(d,cols,b);rec.update(group=group,label=d['label'],variant=variant,alpha=alpha,selected=(alpha==best));ablation.append(rec)
    pd.DataFrame(ablation).to_csv(OUT/'feature_regularization_metrics.csv',index=False)
    pd.DataFrame(selection).to_csv(OUT/'validation_selection.csv',index=False)
    np.savez(OUT/'exploratory_coefficients.npz',**fitted)
    # Diagnose error shape, represented pair signals and training-state coverage.
    stats=[];alltrain=np.concatenate([d['X']['R3'] for d in tr]);q99=np.quantile(alltrain,.99,axis=0)
    for group,d in tests:
        x=d['X']['R3'];pairs=x[:,69:89];p=d['p'];tt=d['t']
        b2=frozen['R2_coef']/frozen['R2_scale'];b3=frozen['R3_coef']/frozen['R3_scale']
        p2=x[:,:89]@b2;p3=x@b3
        pair2=pairs@b2[69:];ctx=x[:,89:]@b3[89:]
        rec=dict(group=group,trial_id=d['meta']['trial_id'],label=d['label'],power_sd=float(np.std(p)),
          R2_bias_W=float(np.mean(p2-p)),R2_centered_RMSE_W=float(np.std(p2-p)),
          R3_bias_W=float(np.mean(p3-p)),R3_centered_RMSE_W=float(np.std(p3-p)),
          pair_feature_mean=float(pairs.mean()), pair_direct_R2_RMS_W=float(np.sqrt(np.mean(pair2**2))),
          tilt_h_mean=float(x[:,89].mean()),tilt_h_sd=float(x[:,89].std()),
          context_direct_R3_J=float(a.integrate(ctx,tt)), R3_minus_R2_J=float(a.integrate(p3-p2,tt)),
          joint_features_above_training_q99_fraction=float((x[:,1:69]>q99[1:69]).mean()),
          pair_features_above_training_q99_fraction=float((pairs>q99[69:89]).mean()))
        stats.append(rec)
    pd.DataFrame(stats).to_csv(OUT/'mechanism_diagnostics.csv',index=False)
    print('Frozen predictions verified; feature analyses complete',flush=True)
    # Retrospective learning curves: whole trials, nested random subsets, 20 fixed seeds.
    # No held-out score enters subset or regularization selection.
    subsets=[];curve=[]
    pools={'static_single':np.array([i for i,d in enumerate(tr) if d['meta']['level'] in ('L0','L1')]),
           'through_pairs':np.array([i for i,d in enumerate(tr) if d['meta']['level'] in ('L0','L1','L2')]),'all_levels':full}
    for seed in range(20):
        rng=np.random.default_rng(20260928+seed);order=rng.permutation(full)
        jobs=[('budget',n,order[:n]) for n in (12,24,48,80,116)]
        for pool,indices in pools.items():jobs.append((pool,48,rng.permutation(indices)[:48]))
        for experiment,n,idx in jobs:
            subsets.append(dict(seed=seed,experiment=experiment,n=n,trial_ids=[tr[i]['meta']['trial_id'] for i in idx]))
            for model in ('R1','R2','R3'):
                val,alpha,beta=choose(idx,COLS[model])
                for group,ds in [('whole_body',wb)]:
                    mm=[score(d,COLS[model],beta) for d in ds]
                    curve.append(dict(experiment=experiment,seed=seed,n=n,model=model,alpha=alpha,
                       validation_RMSE_W=val,group=group,energy_error_pct=float(np.mean([m['energy_error_pct'] for m in mm])),
                       mean_trial_RMSE_W=float(np.mean([m['power_RMSE_W'] for m in mm]))))
        print('Completed subset seed',seed,flush=True)
    pd.DataFrame(curve).to_csv(OUT/'experience_subset_metrics.csv',index=False)
    (OUT/'subsets.json').write_text(json.dumps(subsets,indent=2),encoding='utf-8')
    provenance=dict(design='Retrospective exploratory audit; original training and validation only for fitting/selection; test sets previously inspected.',
      alphas=GRID,seeds=list(range(20)),seed_offset=20260928,budgets=[12,24,48,80,116],equal_budget=48,
      frozen_reconstruction_max_abs_W=maxdiff,frozen_model_sha256=f['model_sha256'],
      script_sha256=digest(__file__),training_validation_hashes=f['training_validation_hashes'],
      warning='No learning-curve band represents population uncertainty; bands reflect subset variability on fixed test recordings. Support labels unchanged and unaudited.')
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(pd.DataFrame(selection).to_string(index=False),flush=True)


if __name__=='__main__':main()
