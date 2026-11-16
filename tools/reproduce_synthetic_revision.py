"""Reproduce the revised synthetic experiment without launching Webots."""
from pathlib import Path
import argparse, json, subprocess, sys, importlib.util, hashlib
import numpy as np
from reproduce import equal_csv
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'synthetic_webots/runs/revision_02'
OLD=ROOT/'synthetic_webots/runs/full_v2'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d): p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
 for rel,h in read(RUN/'auxiliary_hashes.json').items(): assert sha(RUN/rel)==h,rel
 reused=read(RUN/'reuse_manifest.json');assert len(reused)==1093
 for item in reused:
  name=item['key']+'.npz';assert sha(RUN/'raw'/name)==item['sha256']==sha(OLD/'raw'/name),name
 subprocess.run([sys.executable,str(RUN/'source/analysis/analyze.py'),'--run-dir',str(RUN),'--out',str(out/'analysis')],check=True)
 for n,k in [('model_summary.csv',['model']),('per_trial_metrics.csv',['model','trial']),('experience_comparisons.csv',['model']),('mechanical_checks.csv',['trial']),('trial_summary.csv',['split','trial'])]:equal_csv(out/'analysis'/n,RUN/'analysis'/n,k)
 assert read(out/'analysis/subsets.json')==read(RUN/'analysis/subsets.json')
 spec=importlib.util.spec_from_file_location('revised_motion_audit',RUN/'analysis_scripts/motion_audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert m.THRESHOLDS==read(RUN/'motion_thresholds.json')
 recorded={x['spec']['id']:x for x in read(RUN/'revision_report/motion_audit.json')};audits=[]
 for s in read(RUN/'design.json')['trials']:
  if s['split']!='test':continue
  d=m.audit_motion(RUN/'raw'/('test_'+s['id']+'.npz'));saved=recorded[s['id']]
  for k in ['legacy_endpoint_pass','revised_motion_pass','revised_gates','failed_revised_gates','metrics','command_diagnostics']:assert d[k]==saved[k],(s['id'],k)
  d['source']='raw/test_'+s['id']+'.npz';audits.append(d)
 assert len(audits)==21 and all(x['revised_motion_pass'] for x in audits)
 dump(out/'sustained_motion_audit.json',audits)
 for name in ['M1','M2','M3','M4']:
  with np.load(out/'analysis/models'/f'{name}.npz') as a,np.load(RUN/'analysis/models'/f'{name}.npz') as b,np.load(OLD/'analysis/models'/f'{name}.npz') as c:
   for key in ['cols','scale','beta','alpha']:assert np.allclose(a[key],b[key],rtol=1e-8,atol=1e-8) and np.array_equal(b[key],c[key]),(name,key)
 subprocess.run([sys.executable,str(RUN/'analysis_scripts/command_window_diagnostic.py'),str(RUN),'--out',str(out/'command_window')],check=True)
 for name,keys in [('model_summary.csv',['model','window']),('per_trial_metrics.csv',['model','trial','window'])]:
  equal_csv(out/'command_window'/name,RUN/'command_window'/name,keys)
 report=dict(passed=True,revision='revision_02',train=1000,validation=90,test=21,separate_controls=3,reused_records_verified=1093,raw_hashes_verified=1114,sustained_mechanical_pass=21,all_saved_numeric_metrics_match=True,original_ridge_coefficients_unchanged=True,webots_rerun_performed=False,power_targets='synthetic',mean_energy_error_pct={x['model']:x['mean_energy_error_pct'] for x in read(out/'analysis/analysis.json')['models']})
 dump(out/'report.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
