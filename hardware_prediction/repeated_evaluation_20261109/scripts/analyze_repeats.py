"""Update manuscript statistics on 45 executions without changing primary models."""
from pathlib import Path
import importlib.util, sys, json, hashlib
import numpy as np
import pandas as pd

import argparse
B=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--part',choices=['windows','coverage'],required=True)
args=ap.parse_args()
S=B/'source'; O=args.out.resolve(); O.mkdir(parents=True,exist_ok=True)
H=B.parent
R=H/'hardware_experiment/regression_results/ridge_20260915_143619'
OLD=H/'manuscript/data/transfer_audit_20260928'
MODELS=['R0','R1','R2','R3']
COLS={'R1':np.arange(69),'R2':np.arange(89),'R2_tilt':np.arange(91),
      'R2_support':np.r_[np.arange(89),np.arange(91,95)],'R3':np.arange(95)}
GRID=[.001,.1,10.,100.]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0,str(S/'analysis_code_snapshot/legacy_analysis'))
spec=importlib.util.spec_from_file_location('repeat_source',S/'analysis_code_snapshot/repeat_analysis.py')
ra=importlib.util.module_from_spec(spec);spec.loader.exec_module(ra)
a=ra.original
frozen=dict(np.load(S/'original_models/frozen_models.npz'))
manifest=read(S/'export_manifest.json')['recordings']
df=pd.read_csv(B/'verified_data/per_trial_verified.csv')
motion=df.groupby(['motion','model'],sort=False).mean(numeric_only=True).reset_index()
summary={m:{'mean_energy_error_pct':float(g.energy_error_pct.mean()),
            'between_motion_sd_pp':float(g.energy_error_pct.std(ddof=1)),
            'mean_power_RMSE_W':float(g.power_RMSE_W.mean())} for m,g in motion.groupby('model')}
summary['selection']={}
for name in ['Clap','Bow']:
 g=df[(df.motion==name)&(df.model=='R3')].sort_values(['power_RMSE_W','repeat'])
 best=g.iloc[0];n=int(best.console_number);k=int(best['repeat'])
 trace=pd.read_csv(S/f'analysis/traces/{n:02}_{k}.csv')
 summary['selection'][name]={'repeat':k,'number':n,'criterion':'Minimum R3 power RMSE among five executions',
     'trial_id':best.trial_id,'duration_s':float(best.duration_s),
     'actual_J':float(best.actual_J),
     'models':{r.model:{'power_RMSE_W':float(r.power_RMSE_W),'energy_error_pct':float(r.energy_error_pct),
                       'predicted_J':float(r.predicted_J)} for r in df[(df.motion==name)&(df['repeat']==k)].itertuples()}}
windows=[];counts=[];durations=[]
for c in manifest:
 n,k=c['number'],c['repeat']; trace=pd.read_csv(S/f'analysis/traces/{n:02}_{k}.csv')
 t=trace.t_s.to_numpy();t=t-t[0];duration=t[-1];durations.append(duration)
 for length in [2,5,10]:
  for scheme in ['nonoverlap','overlap']:
   step=length if scheme=='nonoverlap' else .1
   starts=np.arange(int(np.floor((duration-length+1e-8)/step))+1)*step
   counts.append(dict(number=n,repeat=k,length=length,scheme=scheme,n_windows=len(starts)))
   errors={m:[] for m in MODELS}
   for start in starts:
    end=start+length; grid=np.r_[start,t[(t>start+1e-9)&(t<end-1e-9)],end]
    actual=float(np.trapezoid(np.interp(grid,t,trace.actual_W),grid))
    for m in MODELS:
     estimate=float(np.trapezoid(np.interp(grid,t,trace[m+'_W']),grid))
     errors[m].append(100*abs(estimate-actual)/actual)
   for m in MODELS:
    windows.append(dict(number=n,repeat=k,length=length,scheme=scheme,model=m,
                        mean_energy_error_pct=float(np.mean(errors[m])),n_windows=len(starts)))
win=pd.DataFrame(windows);win.to_csv(O/'window_per_trial.csv',index=False)
wc=pd.DataFrame(counts);wc.to_csv(O/'window_counts.csv',index=False)
wg=win.groupby(['scheme','length','number','model']).mean(numeric_only=True).reset_index()
wm=wg.groupby(['scheme','length','model']).mean(numeric_only=True).reset_index()
wm.to_csv(O/'window_summary.csv',index=False)
summary['windows']={scheme:{str(length):{
 'n_windows':int(wc[(wc.scheme==scheme)&(wc.length==length)].n_windows.sum()),
 'means':{row.model:float(row.mean_energy_error_pct) for row in wm[(wm.scheme==scheme)&(wm.length==length)].itertuples()}}
 for length in [2,5,10]} for scheme in ['nonoverlap','overlap']}
summary['total_test_time_s']=float(sum(durations))
print('Primary and window summaries complete',flush=True)
if args.part=='windows':
 save(O/'summary.json',summary)
 sys.exit(0)

# Exploratory refits use the SAME archived training/validation data and subset IDs.
# No primary model, raw recording, or train/validation split is modified.
f=read(R/'models/frozen.json')
assert digest(R/'models/frozen_models.npz')==digest(S/'original_models/frozen_models.npz')
for v in f['training_validation_hashes'].values():assert digest(R/'aligned'/v['file'])==v['sha256']
rs=a.manifest(R/'trials.csv')
tr,_=a.load(rs,R/'aligned',['train'],f['schema']);va,_=a.load(rs,R/'aligned',['validation'],f['schema'])
assert len(tr)==116 and len(va)==23
test=[]
for c in manifest:
 d,qc=ra.reconstruct(S/c['export_path']/'raw','ground' if c['number'] in [6,7,8,9] else 'feet',True)
 d.update(number=c['number'],repeat=c['repeat']);test.append(d)
G=np.array([d['X']['R3'].T@d['X']['R3'] for d in tr])
XY=np.array([d['X']['R3'].T@d['p'] for d in tr]);ns=np.array([len(d['p']) for d in tr])
vx=np.concatenate([d['X']['R3'] for d in va]);vy=np.concatenate([d['p'] for d in va])
tg=np.array([d['X']['R3'].T@d['X']['R3']/len(d['p']) for d in test])
tx=np.array([d['X']['R3'].T@d['p']/len(d['p']) for d in test])
ty=np.array([np.mean(d['p']**2) for d in test])
te=np.array([np.trapezoid(d['X']['R3'],d['t'],axis=0) for d in test])
ey=np.array([np.trapezoid(d['p'],d['t']) for d in test])
def fit(idx,cols,alpha):
 g=G[idx].sum(0)[np.ix_(cols,cols)];xy=XY[idx].sum(0)[cols];count=int(ns[idx].sum())
 scales=np.maximum(np.sqrt(np.diag(g)/count),1e-6);scales[0]=1
 pen=np.eye(len(cols))*alpha;pen[0,0]=0
 return np.linalg.solve(g/scales[:,None]/scales[None,:]+pen,xy/scales)/scales
def metrics(cols,beta):
 b=np.zeros(95);b[cols]=beta
 power=np.sqrt(np.maximum(0,np.einsum('i,kij,j->k',b,tg,b)-2*(tx@b)+ty))
 energy=100*abs(te@b-ey)/ey
 return power,energy
def choose(idx,cols):
 candidates=[]
 for alpha in GRID:
  beta=fit(idx,cols,alpha);val=float(np.sqrt(np.mean((vx[:,cols]@beta-vy)**2)))
  candidates.append((val,alpha,beta))
 return min(candidates,key=lambda x:x[0])
full=np.arange(116);ablation=[];beta_store={}
for variant,cols in COLS.items():
 for alpha in GRID:
  b=fit(full,cols,alpha);power,energy=metrics(cols,b)
  if variant in ['R1','R2','R3'] and alpha==f['models'][variant]['alpha']:
   assert np.allclose(b,frozen[variant+'_coef']/frozen[variant+'_scale'],atol=1e-8)
   saved=df[df.model==variant].set_index('trial_id')
   for d,rmse,err in zip(test,power,energy):
    ref=saved.loc[d['meta']['trial_id']]
    assert abs(rmse-ref.power_RMSE_W)<1e-7 and abs(err-ref.energy_error_pct)<1e-7
  beta_store[f'{variant}_alpha{alpha}']=b
  for d,rmse,err in zip(test,power,energy):
   ablation.append(dict(variant=variant,alpha=alpha,number=d['number'],repeat=d['repeat'],
       trial_id=d['meta']['trial_id'],power_RMSE_W=float(rmse),energy_error_pct=float(err)))
ab=pd.DataFrame(ablation);ab.to_csv(O/'body_context_per_trial.csv',index=False)
ag=ab.groupby(['variant','alpha']).mean(numeric_only=True).reset_index();ag.to_csv(O/'body_context_summary.csv',index=False)
summary['context_alpha10']={r.variant:{'power_RMSE_W':float(r.power_RMSE_W),'energy_error_pct':float(r.energy_error_pct)} for r in ag[ag.alpha==10].itertuples()}
print('Context comparison complete',flush=True)
index={d['meta']['trial_id']:i for i,d in enumerate(tr)}
curve=[];used=[]
for strategy,path,experiment in [('random',OLD/'subsets.json','budget'),('coverage',OLD/'coverage/subsets.json','coverage_budget')]:
 for job in read(path):
  if job['experiment']!=experiment:continue
  idx=np.array([index[t] for t in job['trial_ids']]);assert len(idx)==job['n']
  used.append(dict(strategy=strategy,**job))
  for model in ['R1','R2','R3']:
   val,alpha,b=choose(idx,COLS[model]);power,energy=metrics(COLS[model],b)
   beta_store[f'{strategy}_{job["seed"]}_{job["n"]}_{model}']=b
   curve.append(dict(strategy=strategy,seed=job['seed'],n=job['n'],model=model,
       alpha=alpha,validation_RMSE_W=val,energy_error_pct=float(energy.mean()),mean_trial_RMSE_W=float(power.mean())))
  if job['n']==116 and job['seed']%5==0:print(strategy,'seed',job['seed'],'complete',flush=True)
cv=pd.DataFrame(curve);assert len(cv)==600
cv.to_csv(O/'coverage_metrics.csv',index=False);save(O/'coverage_subsets.json',used)
np.savez(O/'exploratory_coefficients.npz',**beta_store)
summary['coverage_24']={m:{s:float(cv[(cv.model==m)&(cv.strategy==s)&(cv.n==24)].energy_error_pct.median()) for s in ['random','coverage']} for m in ['R1','R2','R3']}
save(O/'summary.json',summary)
save(O/'provenance.json',dict(test_recordings=45,motion_types=9,training_recordings=116,validation_recordings=23,
    primary_models_refitted=False,primary_model_sha256=digest(S/'original_models/frozen_models.npz'),
    exploratory_models_refitted=True,subset_selection_unchanged=True,test_power_used_for_fit_or_selection=False,
    original_subset_sha256={p.relative_to(H.parent).as_posix():digest(p) for p in [OLD/'subsets.json',OLD/'coverage/subsets.json']},script_sha256=digest(Path(__file__))))
print(json.dumps(summary,indent=2),flush=True)
