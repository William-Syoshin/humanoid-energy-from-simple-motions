"""Fit/evaluate the frozen reconstructed benchmark; never tune on test outcomes."""
from pathlib import Path
import sys,os,json,csv,hashlib,argparse,time
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from protocol import features,oracle,PAIRS
DIMS=[52,72,77,84];ALPHAS=[1e-5,.001,.1,10,100]
def writecsv(path,rows):
 if not rows:return
 with open(path,'w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def savefig(name):
 for ext in ['pdf','png']:plt.savefig(OUT/'figures'/f'{name}.{ext}',dpi=240,bbox_inches='tight')
 plt.close()
def load_trial(s):
 p=RUN/'raw'/f'{s["split"]}_{s["id"]}.npz'
 with np.load(p,allow_pickle=False) as z:d={k:z[k] for k in z.files}
 assert json.loads(str(d['spec_json']))==s
 x=features(d)[3];y,components=oracle(d)
 for key,value in d.items():
  if value.dtype.kind in 'fci':assert np.isfinite(value).all(),(p,key,'nonfinite raw values')
 assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(components).all(),(p,'nonfinite features/reference')
 noise=np.random.default_rng(s['seed']+800000).normal(0,.08,len(y)) if s['split'] in ['train','validation'] else np.zeros(len(y))
 nontip=(s['context']!='free' and (np.max(abs(d['h']-d['h'][0]))<.3 if s['context'] in ['lie_supine','lie_prone','hand'] else np.max(d['h'])<.3 and np.min(d['position'][:,2])>.20))
 t=d['t'];E=float(np.trapz(y,t));st=json.loads((RUN/'status'/f'{s["split"]}_{s["id"]}.json').read_text())
 assert hashlib.sha256(p.read_bytes()).hexdigest()==st['sha256']
 trace=dict(t=t,h=d['h'],position=d['position'],contact=d['contact'],yaw=d['yaw'])
 summary=dict(split=s['split'],trial=s['id'],action=s['action'],context=s['context'],samples=len(y),duration_s=s['duration'],integrated_window_s=float(t[-1]-t[0]),energy_J=E,residual_energy_J=float(np.trapz(components[:,-1],t)),nontip=bool(nontip),final_height_m=st['final_height'],final_h=st['final_h'],forward_m=st['forward_m'],yaw_change_rad=st['yaw_change'],object_displacement_m=st['object_displacement'],peak_power_W=float(y.max()))
 if s['split']=='train':return dict(spec=s,X=x[::4],y=(y+noise)[::4],summary=summary)
 return dict(spec=s,X=x,y=y+noise,t=t,summary=summary,trace=trace)
def fit_ridge(ids,cols):
 X=np.concatenate([TRAIN[i]['X'][:,cols] for i in ids]);y=np.concatenate([TRAIN[i]['y'] for i in ids]);scale=np.maximum(np.sqrt(np.mean(X*X,axis=0)),1e-6);scale[0]=1
 Z=X/scale;G=Z.T@Z;b=Z.T@y;best=None;records=[]
 for alpha in ALPHAS:
  A=G+np.diag(np.r_[0,np.full(len(cols)-1,alpha)]);beta=np.linalg.solve(A,b)
  pred=np.maximum(0,(VX[:,cols]/scale)@beta);rmse=float(np.sqrt(np.mean((pred-VY)**2)))
  records.append(dict(alpha=alpha,validation_power_RMSE_W=rmse))
  if best is None or rmse<best[0]:best=(rmse,beta,alpha)
 return dict(cols=np.array(cols),scale=scale,beta=best[1],alpha=best[2],validation_RMSE=best[0],selection=records)
def predict(model,X):return np.maximum(0,(X[:,model['cols']]/model['scale'])@model['beta'])
def score(name,preds):
 rows=[]
 for d,pred in zip(TEST,preds):
  y=d['y'];E=np.trapz(y,d['t']);Ep=np.trapz(pred,d['t']);e=pred-y
  rows.append(dict(model=name,trial=d['spec']['id'],action=d['spec']['action'],true_energy_J=float(E),pred_energy_J=float(Ep),energy_error_pct=float(100*abs(Ep-E)/E),power_RMSE_W=float(np.sqrt(np.mean(e*e))),signed_energy_error_J=float(Ep-E),absolute_power_error_integral_J=float(np.trapz(abs(e),d['t']))))
 err=np.concatenate([p-d['y'] for d,p in zip(TEST,preds)]);truth=np.concatenate([d['y'] for d in TEST])
 summary=dict(model=name,mean_energy_error_pct=float(np.mean([r['energy_error_pct'] for r in rows])),worst_energy_error_pct=max(r['energy_error_pct'] for r in rows),mean_trial_power_RMSE_W=float(np.mean([r['power_RMSE_W'] for r in rows])),pooled_power_RMSE_W=float(np.sqrt(np.mean(err*err))),pooled_R2=float(1-np.sum(err*err)/np.sum((truth-truth.mean())**2)))
 return summary,rows

def main():
 global RUN,OUT,TRAIN,TEST,VX,VY
 ap=argparse.ArgumentParser();ap.add_argument('--run',default='full_v2');ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();RUN=ROOT/'runs'/args.run;OUT=args.out;OUT.mkdir(parents=True,exist_ok=False);(OUT/'figures').mkdir(exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
 assert (RUN/'completion.json').exists(),'Collection incomplete'
 design=json.loads((RUN/'design.json').read_text());trials=design['trials'];excluded={'reference/main.tex','reference/figure_provenance.json'};mismatch=[f for f,h in design['source_hashes'].items() if f not in excluded and hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h];assert not mismatch,mismatch
 datasets={x:[] for x in ['train','validation','test','control']};summaries=[]
 for s in trials:
  d=load_trial(s);datasets[s['split']].append(d);summaries.append(d['summary'])
 TRAIN,TEST=datasets['train'],datasets['test'];VAL=datasets['validation'];assert [len(datasets[k]) for k in datasets]==[1000,90,21,3]
 VX=np.concatenate([d['X'] for d in VAL]);VY=np.concatenate([d['y'] for d in VAL]);ids=list(range(len(TRAIN)));allrows=[];models={};predictions={};metrics=[]
 for i,dim in enumerate(DIMS):
  name=f'M{i+1}';m=fit_ridge(ids,list(range(dim)));models[name]=m
  np.savez(OUT/'models'/f'{name}.npz',**{k:m[k] for k in ['cols','scale','beta','alpha']})
  predictions[name]=[predict(m,d['X']) for d in TEST];sm,rows=score(name,predictions[name]);sm['alpha']=m['alpha'];sm['validation_RMSE_W']=m['validation_RMSE'];metrics.append(sm);allrows.extend(rows);print(name,sm,flush=True)
 tree=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,learning_rate=.08,early_stopping=False,random_state=20260913)
 tree.fit(np.concatenate([d['X'] for d in TRAIN]),np.concatenate([d['y'] for d in TRAIN]));predictions['M5']=[np.maximum(0,tree.predict(d['X'])) for d in TEST]
 sm,rows=score('M5',predictions['M5']);sm.update(alpha='',validation_RMSE_W=float(np.sqrt(np.mean((np.maximum(0,tree.predict(VX))-VY)**2))));metrics.append(sm);allrows.extend(rows)
 import joblib;joblib.dump(tree,OUT/'models/M5.joblib')
 writecsv(OUT/'trial_summary.csv',summaries);writecsv(OUT/'model_summary.csv',metrics);writecsv(OUT/'per_trial_metrics.csv',allrows)
 selections={name:m['selection'] for name,m in models.items()};(OUT/'validation_selection.json').write_text(json.dumps(selections,indent=2))
 # Training experience comparisons, independent fixed order, no test selection.
 order=np.random.default_rng(20261012).permutation(len(TRAIN)).tolist();experience=[];subset_defs={};subset_preds={}
 for maxlevel in range(1,6):subset_defs[f'120_L0-L{maxlevel}']=[i for i in order if TRAIN[i]['spec']['level']<=maxlevel][:120]
 for omit in [2,3,4]:subset_defs[f'omit_L{omit}']=[i for i in ids if TRAIN[i]['spec']['level']!=omit]
 eligible=[i for i in ids if TRAIN[i]['summary']['nontip']]
 for level in [1,2,5]:subset_defs[f'nontip_L0-L{level}']=[i for i in eligible if TRAIN[i]['spec']['level']<=level]
 # State-only diagonal information gain; no target/validation/test values used.
 pooldiag=np.array([np.sum(TRAIN[i]['X']**2,axis=0) for i in eligible]);sc=np.maximum(pooldiag.sum(axis=0)/sum(len(TRAIN[i]['X']) for i in eligible),1e-12);diag=pooldiag/sc;information=np.ones(84);chosen=[]
 for _ in range(2):
  gain=np.log1p(diag/information).sum(axis=1);gain[chosen]=-np.inf;j=int(gain.argmax());chosen.append(j);information+=diag[j]
 subset_defs['nontip_selected_2']=[eligible[j] for j in chosen]
 for name,sel in subset_defs.items():
  assert sel
  m=fit_ridge(sel,list(range(84)));pr=[predict(m,d['X']) for d in TEST];sm,_=score(name,pr);sm.update(n_trials=len(sel),alpha=m['alpha']);experience.append(sm);subset_preds[name]=pr
 m=fit_ridge(ids,list(range(52))+list(range(72,84)));sm,_=score('no_pair_features',[predict(m,d['X']) for d in TEST]);sm.update(n_trials=1000,alpha=m['alpha']);experience.append(sm)
 writecsv(OUT/'experience_comparisons.csv',experience);(OUT/'subsets.json').write_text(json.dumps({k:[TRAIN[i]['spec']['id'] for i in v] for k,v in subset_defs.items()},indent=2))
 # Mechanical checks are independent of electrical fit.
 mechanical=[]
 for d in TEST:
  s=d['summary'];a=s['action'];upright=s['final_height_m']>.20 and s['final_h']<.3
  criteria={'walk':upright and s['forward_m']>.05,'stop':upright and s['forward_m']>.02,'get_up':upright,'lie_down':s['final_h']>.7 and s['final_height_m']<.18,'turn_90':upright and abs(s['yaw_change_rad'])>=np.pi/2-.2,'turn_180':upright and abs(s['yaw_change_rad'])>=np.pi-.2,'push_box':upright and s['object_displacement_m']>.05}
  mechanical.append(dict(trial=s['trial'],action=a,mechanically_passed=bool(criteria[a]),final_height_m=s['final_height_m'],final_h=s['final_h'],forward_m=s['forward_m'],yaw_change_rad=s['yaw_change_rad'],object_displacement_m=s['object_displacement_m']))
 writecsv(OUT/'mechanical_checks.csv',mechanical)
 audit=dict(source_hash_match=not mismatch,counts={k:len(v) for k,v in datasets.items()},raw_sha256_verified=len(trials),all_finite=True,mechanical_pass=sum(d['mechanically_passed'] for d in mechanical),mechanical_total=21,validation_only_hyperparameter_selection=True,excluded_historical_document_hashes=sorted(excluded))
 (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
 (OUT/'analysis.json').write_text(json.dumps(dict(models=metrics,experience=experience,audit=audit),indent=2))
 print('ANALYSIS_COMPLETE',OUT,flush=True)
if __name__=='__main__':main()
