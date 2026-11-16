"""Retrospective, frozen-prediction window analysis; protocol.json is authoritative."""
from pathlib import Path
import csv, json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]; D=P.parent/'data'; O=P/'analysis/integration_windows_20261101'
protocol=json.loads((O/'protocol.json').read_text())
for rel,h in protocol['source_sha256'].items():
 assert hashlib.sha256((P/rel).read_bytes()).hexdigest()==h,rel
names={'M004':'Cheer','M008':'Clap','M015':'Supine get-up','M016':'Work','M026':'Kick','M073':'Crawl','M074':'Crawl turn left','M075':'Crawl turn right','M076':'Backward crawl'}
models=['R0','R1','R2','R3']; widths=[2,5,10]; records=[]; full=[]; coverage=[]; checks=[]
trapezoid=getattr(np,'trapezoid',np.trapz)
# Independent antiderivative for a piecewise-linear signal.
def primitive(x,t,y):
 c=np.r_[0,np.cumsum(np.diff(t)*(y[1:]+y[:-1])/2)]
 i=int(np.clip(np.searchsorted(t,x,side='right')-1,0,len(t)-2))
 dt=x-t[i]
 return c[i]+y[i]*dt+.5*(y[i+1]-y[i])/(t[i+1]-t[i])*dt*dt
max_quadrature_difference=0.
for r in csv.DictReader((D/'primary_test_trials.csv').open(encoding='utf-8-sig')):
 a=pd.read_csv(D/(r['trial_id']+'_predictions.csv'))
 t=a.t_s.to_numpy();t=t-t[0]; T=t[-1]; motion=names[r['motion']]
 assert np.all(np.diff(t)>0) and np.isfinite(a.to_numpy()).all()
 assert np.allclose(np.diff(t),.1,atol=1e-8,rtol=0)
 actual=a.actual_W.to_numpy(); ea=trapezoid(actual,t)
 assert np.isclose(ea,float(r['actual_J']),atol=1e-7,rtol=0)
 assert np.isclose(T,float(r['score_s']),atol=1e-8,rtol=0)
 for m in models:
  pred=a[m+'_W'].to_numpy(); ep=trapezoid(pred,t);ape=100*abs(ep-ea)/ea
  assert np.isclose(ep,float(r[m+'_predicted_J']),rtol=0,atol=1e-7)
  assert np.isclose(ape,float(r[m+'_energy_error_pct']),rtol=0,atol=1e-7)
  full.append(dict(motion=motion,model=m,duration_s=T,actual_J=ea,predicted_J=ep,error_pct=ape))
 for w in widths:
  n=int(np.floor((T+1e-8)/w)); coverage.append(dict(motion=motion,window_s=w,n_complete_windows=n,original_duration_s=T,included_duration_s=n*w,excluded_tail_s=max(0,T-n*w)))
  for scheme in ['nonoverlap','sliding_0.1s']:
   starts=np.arange(n)*w if scheme=='nonoverlap' else t[t+w<=T+1e-8]
   for idx,start in enumerate(starts):
    end=min(start+w,T);inside=t[(t>start+1e-9)&(t<end-1e-9)]
    tx=np.r_[start,inside,end];ev=trapezoid(np.interp(tx,t,actual),tx)
    assert ev>0
    for m in models:
     y=a[m+'_W'].to_numpy();pv=trapezoid(np.interp(tx,t,y),tx)
     delta=max(abs(pv-(primitive(end,t,y)-primitive(start,t,y))),abs(ev-(primitive(end,t,actual)-primitive(start,t,actual))))
     max_quadrature_difference=max(max_quadrature_difference,delta)
     assert delta<1e-8
     records.append(dict(scheme=scheme,motion=motion,model=m,window_s=w,window_index=idx,start_s=start,end_s=end,actual_J=ev,predicted_J=pv,signed_error_J=pv-ev,error_pct=100*abs(pv-ev)/ev))
 checks.append(dict(trial_id=r['trial_id'],motion=motion,n_samples=len(a),duration_s=T,actual_J=ea,full_trial_match=True))
f=pd.DataFrame(full);d=pd.DataFrame(records);c=pd.DataFrame(coverage)
per=d.groupby(['scheme','window_s','motion','model'],sort=False).agg(n_windows=('error_pct','size'),mean_error_pct=('error_pct','mean'),median_error_pct=('error_pct','median'),max_error_pct=('error_pct','max')).reset_index()
s=per.groupby(['scheme','window_s','model'],sort=False).agg(n_motions=('motion','size'),motion_balanced_mean_error_pct=('mean_error_pct','mean'),median_motion_error_pct=('mean_error_pct','median')).reset_index()
assert (s.n_motions==9).all()
for path,data in [('windows.csv',d),('per_motion.csv',per),('summary.csv',s),('coverage.csv',c),('full_motion.csv',f)]: data.to_csv(O/path,index=False)
fullmeans=f.groupby('model').error_pct.mean()
with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
 fig,ax=plt.subplots(figsize=(7.5,3.9))
 x=np.arange(4); colors=['#92999D','#DA912D','#287FB0','#168D7A']
 for i,(m,col) in enumerate(zip(models,colors)):
  vals=[s.query('scheme=="nonoverlap" and window_s==@w and model==@m').motion_balanced_mean_error_pct.iloc[0] for w in widths]+[fullmeans[m]]
  ax.bar(x+(i-1.5)*.19,vals,width=.17,color=col,label=m)
  for xx,v in zip(x+(i-1.5)*.19,vals): ax.text(xx,v+.3,f'{v:.2f}',ha='center',va='bottom',fontsize=8)
 ax.set_xticks(x,['2 s','5 s','10 s','Full motion'])
 ax.set_ylabel('Mean absolute relative energy error (%)');ax.set_xlabel('Integration interval')
 ax.set_ylim(0,ax.get_ylim()[1]+3);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
 ax.legend(frameon=False,ncol=4,loc='upper right')
 fig.subplots_adjust(left=.10,right=.99,top=.97,bottom=.25)
 fig.text(.1,.05,'Equal weight per motion (n = 9); frozen models; one recording per motion.\nComplete, non-overlapping windows; incomplete tails excluded.',fontsize=9,color='#444444')
 for ext in ['pdf','png','svg']:fig.savefig(O/('integration_window_errors.'+ext),dpi=220)
 plt.close(fig)
audit=dict(protocol_sha256=hashlib.sha256((O/'protocol.json').read_bytes()).hexdigest(),source_hashes_unchanged=True,all_nine_published_full_trial_values_reproduced=True,all_values_finite=True,no_refit=True,no_clipping=True,max_quadrature_difference_J=max_quadrature_difference,trials=checks,primary_window_counts={str(w):int(c[c.window_s==w].n_complete_windows.sum()) for w in widths},full_motion_mean_errors_pct=fullmeans.to_dict(),independent_executions_per_motion=1)
(O/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(s.to_string(index=False));print('FULL',fullmeans.to_dict());print('COVERAGE',c.groupby('window_s')[['n_complete_windows','included_duration_s','excluded_tail_s']].sum().to_string());print('QUADRATURE',max_quadrature_difference)
