"""Frozen-model evaluation; no fitting, no robot I/O, no test-data selection."""
from pathlib import Path
import csv,json,math,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/legacy_analysis'))
import analysis as original
from processing import power_grid
from orientation import estimate_accelerometer

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvsave(p,rr):
 if not rr:return
 with Path(p).open('x',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def parameters():
 f=read(ROOT/'frozen_models/frozen.json');p=ROOT/'frozen_models/frozen_models.npz'
 if sha(p)!=f['model_sha256']:raise ValueError('Frozen model hash mismatch')
 if sha(ROOT/'code/legacy_analysis/analysis.py')!=f['analysis_sha256']:raise ValueError('Original model implementation changed')
 return dict(np.load(p,allow_pickle=False))
def reconstruct(raw,support,strict=True):
 raw=Path(raw);meta=read(raw/'meta.json');cfg=read(raw/'config.json')
 if meta['status']!='complete':raise ValueError('Incomplete acquisition; retain but do not integrate as a full execution')
 pw=rows(raw/'power.csv');st=rows(raw/'state.csv');adraw=rows(raw/'ad.csv')
 if len(pw)<7:raise ValueError('Too few power samples')
 pt=np.array([float(r['device_s']) for r in pw]);pc=np.array([float(r['pc_s']) for r in pw]);pv=np.array([float(r['voltage'])*float(r['current']) for r in pw])
 if not all(np.isfinite(x).all() for x in [pt,pc,pv]) or np.any(np.diff(pt)<=0) or np.any(np.diff(pc)<=0):raise ValueError('Invalid/nonmonotonic power stream')
 jitter=float(np.max(np.abs(pc-pt-np.median(pc-pt))));pt=pt+np.median(pc-pt)
 end=min(meta['score_end_s'],pt[-1],min(max(float(r['pc_s']) for r in st if int(r['joint'])==i) for i in range(17)))
 start=meta['score_start_s'];t=start+np.arange(math.floor((end-start)*10)+1)/10
 if len(t)<7 or t[0]<pt[0]:raise ValueError('Insufficient aligned scoring interval')
 power=power_grid(pt,pv,t,.3)
 q,cmd,gaps=[],[],[]
 for i,j in enumerate(cfg['joints']):
  sub=[r for r in st if int(r['joint'])==i];tt=np.array([float(r['pc_s']) for r in sub])
  if len(tt)<2 or np.any(np.diff(tt)<=0) or tt[0]>t[0] or tt[-1]<t[-1]:raise ValueError('Invalid or uncovered joint stream '+str(i))
  gaps.append(float(np.max(np.diff(tt))))
  factor=math.radians(135)/4000*j['sign']
  q.append(np.interp(t,tt,[(int(r['position'])-j['zero_pos'])*factor for r in sub]))
  cmd.append(np.interp(t,tt,[(int(r['target_register'])-j['zero_pos'])*factor for r in sub]))
 q=np.array(q).T;cmd=np.array(cmd).T
 padded=np.pad(q,((2,2),(0,0)),mode='edge');vel=np.gradient(sum(padded[k:k+len(q)] for k in range(5))/5,t,axis=0);err=cmd-q
 ad=[]
 for r in adraw:
  ok,v=json.loads(r['values_json'])
  if not ok:raise ValueError('Accelerometer read failed')
  ad.append(dict(pc_s=float(r['pc_s']),ad3=v[3],ad4=v[4]))
 ori=estimate_accelerometer(ad,cfg['tilt_calibration'],.3)
 if not ori or ori[0]['pc_s']>t[0] or ori[-1]['pc_s']<t[-1]:raise ValueError('Inclination does not cover scoring interval')
 tilt=np.interp(t,[r['pc_s'] for r in ori],[r['tilt_rad'] for r in ori]);h=1-np.cos(tilt)
 sc=np.tile([0.,1.,0.,0.] if support=='ground' else [1.,0.,0.,0.],(len(t),1))
 x=np.column_stack([np.ones(len(t)),abs(vel),vel**2,abs(err),err**2,np.column_stack([abs(err[:,i]*err[:,j]) for i,j in original.EDGES]),h,h*h,sc])
 d=dict(meta=dict(trial_id=meta['trial_id'],task=f"M{meta['motion']:03d}",success='not_inferred'),t=t-t[0],p=power,X={'R0':x[:,:1],'R1':x[:,:69],'R2':x[:,:89],'R3':x})
 lost_end=float(meta['score_end_s']-t[-1])
 checks={'power_jitter_s':jitter,'max_joint_gap_s':max(gaps),'scoring_end_shortfall_s':lost_end,'support_label':support,'sample_count':len(t)}
 failures=[]
 if jitter>cfg['timing']['max_receive_jitter_s']:failures.append('Power timing exceeds the original 0.003 s jitter bound')
 if max(gaps)>.3:failures.append('Joint stream gap exceeds 0.3 s')
 if lost_end>.2:failures.append('More than 0.2 s missing at scoring end')
 checks['quality_issues']=failures
 if strict and failures:raise ValueError('; '.join(failures))
 return d,checks

def evaluate(raw,support,strict=True):
 d,qc=reconstruct(raw,support,strict);params=parameters()
 pred={m:original.predict(d,m,(params[m+'_scale'],params[m+'_coef'])) for m in original.MODELS}
 metrics=[original.metrics(d,m,pred[m]) for m in original.MODELS]
 trace=[{'t_s':float(t),'actual_W':float(d['p'][i]),**{m+'_W':float(pred[m][i]) for m in original.MODELS}} for i,t in enumerate(d['t'])]
 return metrics,trace,qc

def reference_check():
 means={m:[] for m in original.MODELS};worst=0.;qc=[]
 for item in rows(ROOT/'reference/original_metrics.csv'):
  support='ground' if 'evaluation_ground' in item['evaluation_path'] else 'feet'
  metrics,trace,checks=evaluate(ROOT/'reference/original_raw'/item['trial_id'],support,strict=False)
  saved=rows(ROOT/'reference/predictions'/(item['trial_id']+'.csv'))
  if len(saved)!=len(trace):raise AssertionError('Reference row count differs')
  for key in ['t_s','actual_W']+[m+'_W' for m in original.MODELS]:
   diff=float(np.max(abs(np.array([float(r[key]) for r in trace])-np.array([float(r[key]) for r in saved]))));worst=max(worst,diff)
   if diff>1e-8:raise AssertionError('Reference trace differs '+key)
  for mt in metrics:
   for k in ['predicted_J','energy_error_pct','power_RMSE_W']:
    if abs(mt[k]-float(item[mt['model']+'_'+k]))>1e-7:raise AssertionError('Reference metric differs '+k)
   means[mt['model']].append(mt['energy_error_pct'])
  qc.append({'trial_id':item['trial_id'],**checks})
 return {'passed':True,'recordings':9,'models':4,'max_trace_difference':worst,'mean_energy_error_pct':{m:float(np.mean(v)) for m,v in means.items()},'historical_quality_observations':qc,'hardware_connected':False}

def stats(v):
 a=np.array(v,dtype=float);return {'n':len(a),'mean':float(np.mean(a)) if len(a) else None,'sd':float(np.std(a,ddof=1)) if len(a)>1 else None,'variance':float(np.var(a,ddof=1)) if len(a)>1 else None}

def analyze_session(session,out,plot=True):
 plan=read(ROOT/'protocol/plan.json')['trials'];attempt_rows=[];primary=[];diagnostics=[]
 for tr in plan:
  attempts=sorted((session/'trials').glob(tr['trial_id']+'_A*'))
  if not attempts:
   attempt_rows.append({'trial_id':tr['trial_id'],'attempt':'','acquisition':'missing','outcome':'not_recorded','primary_eligible':False,'reason':'No attempt'});continue
  for folder in attempts:
   run=read(folder/'trial.json');anns=sorted(folder.glob('annotation_*.json'));ann=read(anns[-1]) if anns else {'outcome':'unknown'}
   status='missing';reason='';eligible=False
   try:
    captured=read(folder/'raw/meta.json');status=captured['status']
    if captured['motion']!=tr['motion'] or run['motion']!=tr['motion']:raise ValueError('Recorded motion does not match scheduled trial')
    hashes=read(folder/'raw_sha256.json')
    if any(not (folder/'raw'/k).is_file() or sha(folder/'raw'/k)!=v for k,v in hashes.items()):raise ValueError('Raw recording changed after capture')
    mt,trace,qc=evaluate(folder/'raw',tr['support'],strict=True)
    csvsave(out/(folder.name+'_predictions.csv'),trace)
    save(out/(folder.name+'_quality.json'),qc)
    # Mechanical failures are not silently removed. Completed data with a fall
    # remain in the primary first-attempt set, explicitly labelled by outcome.
    eligible=(run['attempt']==1 and ann['outcome'] not in ['unknown','wrong_motion','technical_failure'])
    reason='' if eligible else ('Additional attempt; primary remains attempt 1' if run['attempt']!=1 else 'Outcome unannotated or not the scheduled motion')
    for r in mt:
     r.update(trial_id=tr['trial_id'],attempt=run['attempt'],motion_name=tr['name'],block=tr['block'],outcome=ann['outcome'],primary_eligible=eligible)
     diagnostics.append(r)
     if eligible:primary.append(r)
   except Exception as e:reason=str(e)
   attempt_rows.append({'trial_id':tr['trial_id'],'attempt':run['attempt'],'acquisition':status,'outcome':ann['outcome'],'primary_eligible':eligible,'reason':reason})
 csvsave(out/'attempts_and_missing.csv',attempt_rows);csvsave(out/'all_evaluable_attempt_metrics.csv',diagnostics);csvsave(out/'primary_per_trial_metrics.csv',primary)
 summaries=[]
 for spec in read(ROOT/'protocol/motions.json'):
  for m in original.MODELS:
   rr=[r for r in primary if r['task']==f"M{spec['motion']:03d}" and r['model']==m];s=stats([r['energy_error_pct'] for r in rr])
   summaries.append({'motion':spec['motion'],'name':spec['name'],'model':m,'n':s['n'],'error_mean_pct':s['mean'],'error_sd_percentage_points':s['sd'],'error_variance_percentage_points_squared':s['variance'],'measured_energy_mean_J':stats([r['actual_J'] for r in rr])['mean'],'measured_energy_sd_J':stats([r['actual_J'] for r in rr])['sd'],'power_rmse_mean_W':stats([r['power_RMSE_W'] for r in rr])['mean']})
 csvsave(out/'per_motion_mean_sd_variance.csv',summaries)
 complete=all(x['n']==5 for x in summaries)
 overall={m:{'macro_mean_error_pct':float(np.mean([r['error_mean_pct'] for r in summaries if r['model']==m])) if complete else None} for m in original.MODELS}
 save(out/'summary.json',{'complete_45_first_attempts':complete,'primary_recordings':len(primary)//4,'planned_recordings':45,'overall':overall,'aggregation':'Average repeats within each motion, then average the nine motion means. Repeats are not nine new motion types or independent training seeds.','not_measured_here':'Repeatability of the RL hardware replay is outside this 45-trial protocol.'})
 if plot and any(r['n']>0 for r in summaries):
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,axs=plt.subplots(3,3,figsize=(10,8),sharey=True)
  colors=['#92999e','#d88f2e','#287eb2','#178d79']
  for ax,spec in zip(axs.flat,read(ROOT/'protocol/motions.json')):
   rr=[r for r in summaries if r['motion']==spec['motion']]
   vals=[r['error_mean_pct'] if r['n'] else np.nan for r in rr];sd=[r['error_sd_percentage_points'] if r['n']>1 else 0 for r in rr]
   ax.bar(original.MODELS,vals,color=colors,yerr=sd,capsize=4);ax.set_title(f"{spec['name']} (n={rr[0]['n']}/5)");ax.spines[['top','right']].set_visible(False)
   ax.set_ylabel('Energy error (%)')
  fig.suptitle('Repeated executions: mean ± sample SD (unrecorded trials left blank)');fig.tight_layout()
  fig.savefig(out/'repeated_energy_error.png',dpi=300);fig.savefig(out/'repeated_energy_error.pdf');plt.close(fig)
