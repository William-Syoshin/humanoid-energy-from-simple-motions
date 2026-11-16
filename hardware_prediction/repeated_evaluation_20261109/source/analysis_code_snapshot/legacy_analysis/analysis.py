"""Offline hardware energy analysis. No robot-control or raw-DAQ driver code.
fit reads training/validation traces only. evaluate reads test traces after freeze.
All inputs must already be calibrated, aligned and clipped to declared scoring windows.
"""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

MODELS=['R0','R1','R2','R3']; CATEGORIES=['feet','ground','fixture','object','other']
# Standard simulator column order; map real controller channels explicitly.
NAMES=['left_hip_roll','left_hip_pitch','left_knee_pitch','left_ankle_pitch','left_ankle_roll',
'right_hip_roll','right_hip_pitch','right_knee_pitch','right_ankle_pitch','right_ankle_roll',
'left_shoulder_pitch','left_shoulder_roll','left_elbow_pitch','right_shoulder_pitch','right_shoulder_roll','right_elbow_pitch','head_yaw']
EDGES=[]
for side in ['left','right']:
 for chain in [['hip_roll','hip_pitch','knee_pitch','ankle_pitch','ankle_roll'],['shoulder_pitch','shoulder_roll','elbow_pitch']]:
  EDGES += [(NAMES.index(side+'_'+a),NAMES.index(side+'_'+b)) for a,b in zip(chain,chain[1:])]
EDGES += [(NAMES.index(n),NAMES.index(n.replace('left_','right_'))) for n in NAMES if n.startswith('left_')]
EDGES=sorted(set(tuple(sorted(p)) for p in EDGES))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def design_rows(rs):return [{k:r[k] for k in ('trial_id','file','split','task','level')} for r in rs]
def write_json(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(p,rs):
 if not rs:return
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rs[0]);w.writeheader();w.writerows(rs)
def manifest(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:rs=list(csv.DictReader(f))
 required={'trial_id','file','split','task','level','attempted','usable','success'}
 if not rs or not required<=set(rs[0]):raise ValueError('Manifest needs '+str(required))
 if len({r['trial_id'] for r in rs})!=len(rs):raise ValueError('Trial IDs must be unique')
 for r in rs:
  if r['usable']=='1' and r['attempted']!='1':raise ValueError('Usable trial must have attempted=1')
  if r['split'] not in ['train','validation','test','qc']:raise ValueError('Invalid split')
 return rs

def read_trial(meta,data,expected=None):
 path=Path(data)/meta['file']
 qcpath=path.with_suffix('.qc.json')
 if not qcpath.is_file():raise ValueError('Missing per-trial quality approval: '+str(path))
 qc=json.loads(qcpath.read_text(encoding='utf-8'))
 if qc.get('usable') is not True or qc.get('aligned_sha256')!=digest(path):raise ValueError('Unapproved or changed aligned data')
 with path.open(encoding='utf-8-sig',newline='') as f:rs=list(csv.DictReader(f))
 cols=set(rs[0]) if rs else set()
 required={'t_s','voltage_V','current_A'}|{'cmd_'+str(i) for i in range(17)}
 if not required<=cols or len(rs)<7:raise ValueError(f'{path}: missing required columns or too few samples')
 haveq=all('q_'+str(i) in cols for i in range(17))
 if any('q_'+str(i) in cols for i in range(17)) and not haveq:raise ValueError('Supply all 17 q columns or none')
 schema={'feedback':haveq,'tilt':'tilt_rad' in cols,'support':'support' in cols,'offline_smoothing_samples':5,'power_target':'power_W' if 'power_W' in cols else 'VI'}
 def vals(ks):
  a=np.array([[float(r[k]) for k in ks] for r in rs])
  if not np.isfinite(a).all():raise ValueError(f'{path}: nonfinite data')
  return a
 t=vals(['t_s'])[:,0];dt=np.diff(t)
 if np.any(dt<=0) or dt.max()>2.5*np.median(dt):raise ValueError(f'{path}: nonmonotonic time or large dropout; inspect raw data')
 if not np.allclose(dt,np.median(dt),rtol=1e-4,atol=1e-7):raise ValueError('Uniform analysis grid required for fixed smoothing')
 schema['analysis_dt_s']=round(float(np.median(dt)),6)
 if expected is not None and schema!=expected:raise ValueError(f'Inconsistent sensors or analysis rates in {path}')
 vi=vals(['voltage_V','current_A']);p=vi[:,0]*vi[:,1]
 if 'power_W' in cols:p=vals(['power_W'])[:,0]
 if np.any(vi[:,0]<=0):raise ValueError('Supply voltage must be positive')
 cmd=vals(['cmd_'+str(i) for i in range(17)]);q=vals(['q_'+str(i) for i in range(17)]) if haveq else cmd
 # Fixed offline centered five-sample average. No power channel enters features.
 padded=np.pad(q,((2,2),(0,0)),mode='edge');smooth=sum(padded[k:k+len(q)] for k in range(5))/5
 vel=np.gradient(smooth,t,axis=0);local=[abs(vel),vel*vel];err=cmd-q
 if haveq:local.extend([abs(err),err*err])
 local=np.column_stack(local);a=err if haveq else vel
 pairs=np.column_stack([abs(a[:,i]*a[:,j]) for i,j in EDGES]);ctx=[]
 if schema['tilt']:
  h=1-np.cos(vals(['tilt_rad'])[:,0]);ctx.extend([h[:,None],(h*h)[:,None]])
 if schema['support']:
  s=[r['support'] for r in rs]
  if not set(s)<=set(CATEGORIES):raise ValueError('Unknown support category')
  # Drop final category to avoid a dummy-variable intercept dependency.
  ctx.append(np.column_stack([[z==c for z in s] for c in CATEGORIES[:-1]]).astype(float))
 X={'R0':np.ones((len(t),1))};X['R1']=np.c_[X['R0'],local];X['R2']=np.c_[X['R1'],pairs];X['R3']=np.column_stack([X['R2']]+ctx)
 return {'meta':meta,'path':path,'t':t,'p':p,'X':X,'schema':schema}
def load(rs,data,splits,expected=None):
 out=[]
 for r in rs:
  if r['split'] not in splits or r['usable']!='1':continue
  d=read_trial(r,data,expected);expected=d['schema'];out.append(d)
 if not out:raise ValueError('No usable trials in '+str(splits))
 return out,expected

def fit_one(ds,name,alpha):
 x=np.concatenate([d['X'][name] for d in ds]);y=np.concatenate([d['p'] for d in ds]);s=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-6);s[0]=1;z=x/s
 pen=np.eye(z.shape[1])*alpha;pen[0,0]=0
 c=np.linalg.solve(z.T@z+pen,z.T@y)
 return s,c
def predict(d,name,sc):return (d['X'][name]/sc[0])@sc[1]
def rmse(ds,name,sc):return float(np.sqrt(np.mean(np.concatenate([(predict(d,name,sc)-d['p'])**2 for d in ds]))))
def metrics(d,name,yp):
 t=d['t'];y=d['p'];e=float(integrate(y,t));ep=float(integrate(yp,t))
 if e<=0:raise ValueError('Nonpositive net energy requires separate metric convention')
 return {'trial_id':d['meta']['trial_id'],'task':d['meta']['task'],'success':d['meta']['success'],'model':name,'duration_s':float(t[-1]-t[0]),'actual_J':e,'predicted_J':ep,'energy_error_pct':100*abs(ep-e)/e,'power_RMSE_W':float(np.sqrt(np.mean((yp-y)**2)))}
def fit(args):
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False);rs=manifest(args.manifest)
 tr,schema=load(rs,args.data,['train']);va,_=load(rs,args.data,['validation'],schema);records={};arrays={}
 for name in MODELS:
  candidates=[(rmse(va,name,sc:=fit_one(tr,name,a)),a,sc) for a in [.001,.1,10,100]]
  score,a,sc=min(candidates,key=lambda x:x[0]);arrays[name+'_scale']=sc[0];arrays[name+'_coef']=sc[1];records[name]={'alpha':a,'validation_RMSE_W':score,'features':len(sc[0])}
 chosen=min(records,key=lambda n:records[n]['validation_RMSE_W']);np.savez(out/'frozen_models.npz',**arrays)
 write_json(out/'frozen.json',{'chosen':chosen,'schema':schema,'models':records,'train_count':len(tr),'validation_count':len(va),'design_manifest':design_rows(rs),'analysis_sha256':digest(__file__),'model_sha256':digest(out/'frozen_models.npz'),'training_validation_hashes':{d['meta']['trial_id']:{'file':d['meta']['file'],'sha256':digest(d['path'])} for d in tr+va},'joint_order':NAMES,'note':'Test traces not read during fit. Command-only R3 uses support when tilt is omitted; R0-R3 improvement is a hypothesis, not a guarantee.'})
 print('Frozen',chosen,'using',len(tr),'training and',len(va),'validation trials')
def plots_import():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 return plt
def evaluate(args):
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False);rs=manifest(args.manifest);f=json.loads((Path(args.models)/'frozen.json').read_text());params=dict(np.load(Path(args.models)/'frozen_models.npz'))
 if f.get('analysis_sha256')!=digest(__file__) or f.get('model_sha256')!=digest(Path(args.models)/'frozen_models.npz'):raise ValueError('Analysis code or model changed after freezing')
 if f.get('design_manifest')!=design_rows(rs):raise ValueError('Trial identities, splits or tasks changed after model freeze')
 for tid,v in f['training_validation_hashes'].items():
  if digest(Path(args.data)/v['file'])!=v['sha256']:raise ValueError('Training/validation changed since freezing: '+tid)
 ds,_=load(rs,args.data,['test'],f['schema']);rows=[];plt=plots_import()
 for d in ds:
  predictions={n:predict(d,n,(params[n+'_scale'],params[n+'_coef'])) for n in MODELS}
  for n,yp in predictions.items():rows.append(metrics(d,n,yp))
  trace=[{'t_s':float(t),'actual_W':float(d['p'][i]),**{n+'_W':float(predictions[n][i]) for n in MODELS}} for i,t in enumerate(d['t'])];write_csv(out/(d['meta']['trial_id']+'_predictions.csv'),trace)
  fig,ax=plt.subplots(figsize=(7,3));ax.plot(d['t'],d['p'],label='Measured');ax.plot(d['t'],predictions[f['chosen']],'--',label=f['chosen']);ax.set(xlabel='Time (s)',ylabel='Bus power (W)',title=d['meta']['task']);ax.legend();fig.tight_layout();fig.savefig(out/(d['meta']['trial_id']+'_power.png'),dpi=200);plt.close(fig)
 write_csv(out/'per_trial_metrics.csv',rows);fam=[]
 for task in sorted({d['meta']['task'] for d in ds}):
  attempted=sum(r['split']=='test' and r['task']==task and r['attempted']=='1' for r in rs);rr=[r for r in rows if r['task']==task and r['model']==f['chosen']];r0=[r for r in rows if r['task']==task and r['model']=='R0']
  fam.append({'task':task,'attempted':attempted,'valid':len(rr),'successful_valid':sum(r['success']=='1' for r in rr),'chosen_model':f['chosen'],'R0_mean_error_pct':float(np.mean([r['energy_error_pct'] for r in r0])),'chosen_mean_error_pct':float(np.mean([r['energy_error_pct'] for r in rr])),'worst_error_pct':float(max(r['energy_error_pct'] for r in rr)),'mean_power_RMSE_W':float(np.mean([r['power_RMSE_W'] for r in rr]))})
 write_csv(out/'per_task_metrics.csv',fam)
 agg={n:{'mean_error_pct':float(np.mean([r['energy_error_pct'] for r in rows if r['model']==n])),'worst_error_pct':float(max(r['energy_error_pct'] for r in rows if r['model']==n))} for n in MODELS}
 chosen=agg[f['chosen']]['mean_error_pct'];baseline=agg['R0']['mean_error_pct']
 write_json(out/'summary.json',{'models':agg,'chosen':f['chosen'],'mean_paired_error_difference_vs_R0_percentage_points':chosen-baseline,'exploratory_target_met':bool(chosen<10 and chosen<baseline and max(r['chosen_mean_error_pct'] for r in fam)<20),'valid_test_trials':len(ds),'all_test_attempts':sum(r['split']=='test' and r['attempted']=='1' for r in rs),'excluded_test_attempts':[r for r in rs if r['split']=='test' and r['attempted']=='1' and r['usable']!='1'],'note':'Main score includes all measurement-valid attempts; mechanical success is reported separately. Three repetitions are descriptive.'})
 fig,ax=plt.subplots(figsize=(4.5,4));rr=[r for r in rows if r['model']==f['chosen']];a=[r['actual_J'] for r in rr];b=[r['predicted_J'] for r in rr];ax.scatter(a,b);hi=max(a+b)*1.05;ax.plot([0,hi],[0,hi],'k--');ax.set(xlabel='Measured energy (J)',ylabel='Predicted energy (J)');fig.tight_layout();fig.savefig(out/'energy_scatter.png',dpi=220);plt.close(fig)
 print('Scored',len(ds),'valid attempts; frozen model',f['chosen'],'mean energy error',chosen)
def curve(args):
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False);rs=manifest(args.manifest);tr,schema=load(rs,args.data,['train']);ts,_=load(rs,args.data,['test'],schema);budgets=sorted(set([b for b in [10,20,40,80,120] if b<=len(tr)]+[len(tr)]));rows=[]
 for seed in range(20):
  order=np.random.default_rng(seed).permutation(len(tr))
  for b in budgets:
   sc=fit_one([tr[i] for i in order[:b]],'R3',.1);mm=[metrics(d,'R3',predict(d,'R3',sc)) for d in ts]
   rows.append({'seed':seed,'training_trials':b,'mean_energy_error_pct':float(np.mean([r['energy_error_pct'] for r in mm]))})
 write_csv(out/'learning_curve.csv',rows);plt=plots_import();fig,ax=plt.subplots(figsize=(5,3.5));v=np.array([[r['mean_energy_error_pct'] for r in rows if r['training_trials']==b] for b in budgets]);ax.plot(budgets,v.mean(1),'o-');ax.fill_between(budgets,np.maximum(0,v.mean(1)-v.std(1,ddof=1)),v.mean(1)+v.std(1,ddof=1),alpha=.15);ax.axhline(10,ls='--',color='gray');ax.set(xlabel='Complete training trials',ylabel='Mean energy error (%)',title='Retrospective random subsets; fixed R3 and ridge 0.1');fig.tight_layout();fig.savefig(out/'learning_curve.png',dpi=220);plt.close(fig)
 write_json(out/'interpretation.json',{'policy':'random offline subsets','seeds':20,'alpha':.1,'sensor_schema':schema,'warning':'No prospective acquisition savings or minimum-experience claim. Bands represent selection variability on one fixed dataset.'})
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fit','evaluate','learning-curve']);p.add_argument('--manifest',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--models');a=p.parse_args()
 if a.mode=='evaluate' and not a.models:p.error('evaluate requires --models')
 {'fit':fit,'evaluate':evaluate,'learning-curve':curve}[a.mode](a)
