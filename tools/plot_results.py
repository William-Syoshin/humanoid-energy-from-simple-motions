"""Portable numeric figure rebuilding; published layouts are in paper_figures/."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'hardware_prediction/manuscript/data'
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out;out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
COLORS=['#92999D','#DA912D','#287FB0','#168D7A'];MODELS=['R0','R1','R2','R3']
def save(fig,name):
 fig.tight_layout()
 for ext in ['pdf','png','svg']:fig.savefig(out/(name+'.'+ext),dpi=220)
 plt.close(fig)
trials=pd.read_csv(D/'primary_test_trials.csv').set_index('motion')
names=[('M016','Work'),('M008','Clap'),('M015','Supine get-up'),('M004','Cheer'),('M026','Kick'),('M073','Crawl'),('M074','Crawl turn left'),('M075','Crawl turn right'),('M076','Backward crawl')]
fig,axes=plt.subplots(3,3,figsize=(10,7),sharey=True)
for i,(ax,(key,label)) in enumerate(zip(axes.flat,names)):
 r=trials.loc[key];values=[r[m+'_energy_error_pct'] for m in MODELS];ax.bar(MODELS,values,color=COLORS);ax.axhline(values[0],color=COLORS[0],ls='--',lw=.8)
 ax.set_title(f'({chr(97+i)}) {label}',loc='left');ax.set_ylim(0,26)
 for x,v in enumerate(values):ax.text(x,v+.6,f'{v:.2f}',ha='center',fontsize=8)
 if i%3==0:ax.set_ylabel('Energy error (%)')
save(fig,'fig04_unseen_motions')
fig,axes=plt.subplots(2,2,figsize=(10,5.5),sharex='col')
for col,(key,label) in enumerate([('M073','Crawl'),('M015','Supine get-up')]):
 r=trials.loc[key];a=pd.read_csv(D/(r.trial_id+'_predictions.csv'));t=a.t_s.to_numpy()
 axes[0,col].plot(t,a.actual_W,color='black',lw=1,label='Measured')
 for m,c in [('R2',COLORS[2]),('R3',COLORS[3])]:
  y=a[m+'_W'].to_numpy();axes[0,col].plot(t,y,lw=1,color=c,label=m)
  e=y-a.actual_W.to_numpy();cum=np.r_[0,np.cumsum((e[1:]+e[:-1])*.5*np.diff(t))];axes[1,col].plot(t,cum,color=c,label=m)
 axes[0,col].set_title(label);axes[0,col].set_ylabel('Power (W)');axes[0,col].legend(frameon=False)
 axes[1,col].set_ylabel('Signed energy residual (J)');axes[1,col].set_xlabel('Time (s)')
save(fig,'fig05_power_traces')
c=pd.read_csv(D/'transfer_audit_20260928/coverage/metrics.csv');r=pd.read_csv(D/'transfer_audit_20260928/experience_subset_metrics.csv')
fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
for ax,m in zip(axes,['R2','R3']):
 for data,experiment,label,color in [(r,'budget','Random subset','#888888'),(c,'coverage_budget','Joint/posture coverage',COLORS[MODELS.index(m)])]:
  q=data[(data.model==m)&(data.experiment==experiment)];g=q.groupby('n').energy_error_pct;med=g.median();lo=g.quantile(.25);hi=g.quantile(.75)
  ax.plot(med.index,med.values,'o-',color=color,label=label);ax.fill_between(med.index,lo,hi,color=color,alpha=.15)
 ax.set_title(m);ax.set_xlabel('Number of training trials');ax.set_yscale('log');ax.legend(frameon=False,fontsize=8)
axes[0].set_ylabel('Energy error (%), median and IQR');save(fig,'fig06_experience_coverage')
s=pd.read_csv(ROOT/'synthetic_webots/runs/full_v2/analysis/model_summary.csv');fig,ax=plt.subplots(figsize=(6.4,3.2))
ax.bar(s.model,s.mean_energy_error_pct,color=['#92999D','#DA912D','#287FB0','#168D7A','#716585']);ax.set_ylabel('Mean energy error (%)')
ax.set_xticks(range(5),['M1\nLocal','M2\n+ Pairs','M3\n+ Body','M4\n+ Contact','M5\nBoosted trees'])
for i,v in enumerate(s.mean_energy_error_pct):ax.text(i,v+.3,f'{v:.3f}',ha='center',fontsize=9)
save(fig,'fig07_synthetic_features')
d=pd.read_csv(ROOT/'paper_results/rl_application/all_trials.csv');fig,ax=plt.subplots(figsize=(5.2,3.2));conds=['Task-only','R1','R2','R3'];x=np.arange(4)
for field,label,color,offset,marker in [('sim_R2_energy_J','Sim: common R2 predictor','#227EAA',-.15,'o'),('real_energy_J','Real: measured','#CC4B53',.15,'s')]:
 values=[d[d.condition==c][field].to_numpy() for c in conds];ax.errorbar(x+offset,[v.mean() for v in values],yerr=[v.std(ddof=1) for v in values],fmt=marker,color=color,capsize=4,label=label)
ax.set_xticks(x,conds);ax.set_ylabel('Replay energy (J)');ax.set_ylim(60,135);ax.legend(frameon=False,fontsize=8,loc='upper right');ax.grid(axis='y',alpha=.2);save(fig,'fig10_hardware_replay_energy')
print('Rebuilt five numeric result figures in',out)
