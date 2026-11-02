"""Offline paper-result reproduction. Never opens Webots or a serial port."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,os,subprocess,sys
from datetime import datetime,timezone
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
HP=ROOT/'hardware_prediction'; H=HP/'hardware_experiment'; P=HP/'manuscript'
RIDGE=H/'regression_results/ridge_20260915_143619'
RL=ROOT/'motion_learning'; EXP=RL/'logs/distance_goal_20261011'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def mod(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def output(action):
 p=ROOT/'generated'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+action);p.mkdir(parents=True,exist_ok=False);return p
def hardware_modules():
 sys.path.insert(0,str(H));sys.path.insert(0,str(P/'scripts'))
 import analysis as a
 import transfer_audit as z
 return a,z
def equal_csv(actual,expected,keys):
 import numpy as np,pandas as pd
 a=pd.read_csv(actual).sort_values(keys).reset_index(drop=True)
 b=pd.read_csv(expected).sort_values(keys).reset_index(drop=True)
 assert a.shape==b.shape,(str(actual),a.shape,b.shape)
 for c in b:
  if pd.api.types.is_numeric_dtype(b[c]):assert np.allclose(a[c],b[c],atol=1e-7,rtol=1e-8,equal_nan=True),(str(actual),c)
  else:assert a[c].fillna('').tolist()==b[c].fillna('').tolist(),(str(actual),c)
def verify(out):
 entries=read(ROOT/'checksums.json');checked=0
 for rel,h in entries.items():
  p=ROOT/rel;assert p.is_file() and sha(p)==h,rel;checked+=1
 dump(out/'report.json',dict(files_verified=checked,passed=True))
 print('Verified',checked,'repository files')
def hardware(out):
 import numpy as np,pandas as pd
 from types import SimpleNamespace
 a,z=hardware_modules();f=read(RIDGE/'models/frozen.json');params=dict(np.load(RIDGE/'models/frozen_models.npz',allow_pickle=False))
 assert sha(H/'analysis.py')==f['analysis_sha256']
 assert sha(RIDGE/'models/frozen_models.npz')==f['model_sha256']
 for item in f['training_validation_hashes'].values():assert sha(RIDGE/'aligned'/item['file'])==item['sha256']
 schedule=a.manifest(RIDGE/'trials.csv');tr,_=a.load(schedule,RIDGE/'aligned',['train'],f['schema']);va,_=a.load(schedule,RIDGE/'aligned',['validation'],f['schema'])
 assert len(tr)==116 and len(va)==23
 assert not ({d['meta']['trial_id'] for d in tr}&{d['meta']['trial_id'] for d in va})
 a.fit(SimpleNamespace(manifest=str(RIDGE/'trials.csv'),data=str(RIDGE/'aligned'),out=str(out/'refitted_models')))
 new=dict(np.load(out/'refitted_models/frozen_models.npz',allow_pickle=False))
 diffs={k:float(np.max(abs(params[k]-new[k]))) for k in params}
 assert all(np.allclose(params[k],new[k],rtol=1e-8,atol=1e-8) for k in params),diffs
 metrics=[];worst=0.
 primary=rows(P/'data/primary_test_trials.csv');assert len(primary)==9
 for r in primary:
  d=z.whole_body(r);saved=pd.read_csv(P/'data'/(r['trial_id']+'_predictions.csv'))
  assert np.allclose(d['t'],saved.t_s,rtol=0,atol=1e-8)
  assert np.allclose(d['p'],saved.actual_W,rtol=0,atol=1e-8)
  for m in a.MODELS:
   pred=a.predict(d,m,(params[m+'_scale'],params[m+'_coef']));delta=float(np.max(abs(pred-saved[m+'_W'])));worst=max(worst,delta);assert delta<1e-8
   mt=a.metrics(d,m,pred)
   for k in ['predicted_J','energy_error_pct','power_RMSE_W']:assert np.isclose(mt[k],float(r[m+'_'+k]),atol=1e-7,rtol=0)
   metrics.append(mt)
 df=pd.DataFrame(metrics);df.to_csv(out/'per_motion_metrics.csv',index=False);means=df.groupby('model').energy_error_pct.mean().to_dict()
 assert {m:round(v,2) for m,v in means.items()}=={'R0':10.52,'R1':5.44,'R2':3.55,'R3':2.59}
 dump(out/'report.json',dict(passed=True,train=116,validation=23,held_out_motions=9,refit_max_differences=diffs,max_prediction_difference_W=worst,mean_energy_error_pct=means,validation_selected_model=f['chosen'],raw_test_signals_reconstructed=True))
 print('Hardware models refitted; nine raw recordings reconstructed:',means)
def coverage(out):
 a,z=hardware_modules();z.OUT=out/'retrospective';z.main()
 old=P/'data/transfer_audit_20260928'
 for n,keys in [('feature_regularization_metrics.csv',['variant','alpha','trial_id']),('experience_subset_metrics.csv',['experiment','seed','n','model','group']),('validation_selection.csv',['variant']),('mechanism_diagnostics.csv',['group','trial_id'])]:equal_csv(z.OUT/n,old/n,keys)
 import coverage_audit
 coverage_audit.main();equal_csv(z.OUT/'coverage/metrics.csv',old/'coverage/metrics.csv',['experiment','seed','n','model'])
 assert read(z.OUT/'subsets.json')==read(old/'subsets.json')
 assert read(z.OUT/'coverage/subsets.json')==read(old/'coverage/subsets.json')
 dump(out/'report.json',dict(passed=True,all_saved_subsets_and_metrics_match=True,seeds=20,design='retrospective; validation-only hyperparameter selection'))
def windows(out):
 import numpy as np,pandas as pd
 base=P/'frontiers_20261023';old=base/'analysis/integration_windows_20261101'
 text=(base/'scripts/analyze_integration_windows.py').read_text()
 text=text.replace("O=P/'analysis/integration_windows_20261101'",'O=OUTPUT')
 # The archived protocol hashes the original CSV bytes. The copied inputs are unchanged.
 (out/'protocol.json').write_bytes((old/'protocol.json').read_bytes())
 ns={'__file__':str(base/'scripts/analyze_integration_windows.py'),'OUTPUT':out,'__name__':'__main__'}
 exec(compile(text,str(base/'scripts/analyze_integration_windows.py'),'exec'),ns)
 equal_csv(out/'summary.csv',old/'summary.csv',['scheme','window_s','model'])
 dump(out/'report.json',dict(passed=True,all_window_summaries_match=True,intervals_s=[2,5,10]))
def rl(out):
 import numpy as np
 sys.path.insert(0,str(EXP/'source'))
 z=mod('distance_finalizer',EXP/'source/scripts/finalize_distance_experiment.py')
 plan=read(EXP/'design.json');runs=read(EXP/'results.json')
 for run in runs:
  name=Path(run['run']).name;run['run']=str(EXP/name)
 # Redirect only output, leaving all archived files immutable.
 z.atomic_json=lambda path,obj:dump(out/Path(path).name,obj)
 audit=z.audit(EXP,plan,runs);comp=z.paired_analysis(runs,plan);old=read(EXP/'analysis.json')['comparisons']
 def compare(a,b):
  if isinstance(b,dict):
   assert set(a)==set(b)
   for k in b:compare(a[k],b[k])
  elif isinstance(b,list):
   assert len(a)==len(b)
   for x,y in zip(a,b):compare(x,y)
  elif isinstance(b,(int,float)) and not isinstance(b,bool):assert np.isclose(a,b,rtol=1e-9,atol=1e-8),(a,b)
  else:assert a==b,(a,b)
 compare(comp,old)
 dump(out/'comparisons.json',comp);dump(out/'report.json',dict(passed=True,training_runs=audit['training_runs'],evaluation_episodes=audit['evaluation_episodes'],maximum_energy_difference_J=audit['maximum_offline_energy_difference_J'],all_pairwise_statistics_match=True,primary_criterion_passed={c['condition']:c['primary_criterion_passed'] for c in comp},ppo_retraining_performed=False))
 print('All 240 evaluation traces, source hashes, energies, OOD, and paired statistics match')
def replay(out):
 z=mod('real_replay_analysis',RL/'analysis/real_results_20261011/analyze_real.py')
 d=out/'recomputed';z.analyze(z.DEFAULT,d)
 for n,keys in [('all_trials.csv',['condition','seed']),('summary.csv',['condition','domain','metric']),('effect_summary.csv',['condition'])]:equal_csv(d/n,ROOT/'paper_results/rl_application'/n,keys)
 audit=read(d/'audit.json')
 dump(out/'report.json',dict(passed=True,physical_recordings=24,all_summary_values_match=True,maximum_reintegration_difference_J=audit['maximum_reintegration_difference_J'],physical_distance_measured=False))
 print('All 24 hardware replay integrations and recorded plans/commands match')
def synthetic(out):
 subprocess.run([sys.executable,str(ROOT/'synthetic_webots/analysis/analyze.py'),'--out',str(out/'analysis')],check=True)
 for name,keys in [('model_summary.csv',['model']),('per_trial_metrics.csv',['model','trial']),('experience_comparisons.csv',['model']),('mechanical_checks.csv',['trial'])]:equal_csv(out/'analysis'/name,ROOT/'synthetic_webots/runs/full_v2/analysis'/name,keys)
 dump(out/'report.json',dict(passed=True,train=1000,validation=90,test=21,separate_controls=3,all_refitted_metrics_match=True,webots_rerun_performed=False,power_targets='synthetic'))
def figures(out):
 subprocess.run([sys.executable,str(ROOT/'tools/plot_results.py'),'--out',str(out)],check=True)
 dump(out/'report.json',dict(passed=True,recreated='numeric panels for Figs. 4, 5, 6, 7, and 10; same saved statistics, portable typography'))
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['verify','hardware','coverage','windows','rl','replay','synthetic','figures','all']);args=parser.parse_args()
 if args.action=='all':
  for action in ['verify','hardware','coverage','windows','rl','replay','synthetic','figures']:subprocess.run([sys.executable,__file__,action],check=True)
  return
 out=output(args.action);globals()[args.action](out);print('Output:',out)
if __name__=='__main__':main()
