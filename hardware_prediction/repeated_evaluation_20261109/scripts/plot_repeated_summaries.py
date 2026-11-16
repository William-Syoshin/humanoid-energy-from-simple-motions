"""Publication figures for repeated execution results; uses verified saved statistics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,NullLocator

import argparse
from matplotlib import font_manager
B=Path(__file__).resolve().parents[1]; A=B/'analysis'
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
OUT=ap.parse_args().out.resolve();OUT.mkdir(parents=True,exist_ok=True)
FONT='Helvetica' if 'Helvetica' in {f.name for f in font_manager.fontManager.ttflist} else 'DejaVu Sans'
summary=json.loads((A/'summary.json').read_text())
MM=1/25.4
models=['R0','R1','R2','R3']
colors={'Measured':'#151515','R0':'#92999E','R1':'#D89035','R2':'#2879B0','R3':'#188A75'}
styles={'Measured':'-','R0':(0,(3,2)),'R1':'-','R2':(0,(4,2)),'R3':'-'}
plt.rcParams.update({'font.family':FONT,'font.size':9,'axes.labelsize':9.5,'axes.titlesize':10,
 'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.1,'xtick.major.width':1.1,
 'ytick.major.width':1.1,'savefig.facecolor':'white'})
def style(ax):
 ax.set_axisbelow(True);ax.grid(axis='y',color='#E8EBEE',lw=.7)
 for n in ['left','bottom']:ax.spines[n].set_color('#7C8388')
def save(fig,name,manuscript):
 for ext in ['pdf','svg','png']:fig.savefig(OUT/(name+'.'+ext),dpi=420)
 plt.close(fig)

# Fig.5: compute SD across the nine within-motion means, not across all 45 rows.
fig,ax=plt.subplots(figsize=(180*MM,100*MM))
means=np.array([summary[m]['mean_energy_error_pct'] for m in models]);sds=np.array([summary[m]['between_motion_sd_pp'] for m in models])
ax.bar(np.arange(4),means,color=[colors[m] for m in models],width=.58,
 yerr=sds,capsize=5,error_kw={'elinewidth':1.25,'capthick':1.25,'ecolor':'#30353A'},zorder=3)
for x,mean,sd in zip(range(4),means,sds):ax.text(x,mean+sd+.45,f'{mean:.2f} ± {sd:.2f}',ha='center',fontsize=10,weight='bold' if x==3 else 'normal')
ax.set_xticks(range(4),models);ax.set_ylim(0,15.5);ax.set_yticks([0,5,10,15]);ax.set_xlim(-.6,3.6)
ax.set_ylabel('Absolute relative energy error (%)');ax.set_xlabel('Power prediction model',labelpad=8);style(ax)
fig.text(.55,.96,'Nine motion means, each based on five executions',ha='center',fontsize=10)
fig.text(.55,.015,'Error bars: ±1 sample SD across the nine motion means',ha='center',fontsize=8.5)
fig.subplots_adjust(left=.12,right=.98,top=.88,bottom=.18)
save(fig,'Fig5_repeated_energy_error_summary','hardware_energy_error_mean_sd.pdf')

# Fig.6: exactly the lowest-R3-RMSE execution within each user-selected motion.
fig,axs=plt.subplots(2,2,figsize=(180*MM,110*MM),sharex='col',sharey='row',gridspec_kw={'height_ratios':[1.5,1]})
traces=[];power_values=[];residual_values=[]
for name in ['Clap','Bow']:
 sel=summary['selection'][name];d=pd.read_csv(B/f'source/analysis/traces/{sel["number"]:02}_{sel["repeat"]}.csv');traces.append(d)
 power_values.append(d[['actual_W']+[m+'_W' for m in models]].to_numpy())
for col,(name,d) in enumerate(zip(['Clap','Bow'],traces)):
 t=d.t_s.to_numpy();t=t-t[0];actual=d.actual_W.to_numpy();sel=summary['selection'][name]
 for model in ['Measured']+models:
  y=d['actual_W' if model=='Measured' else model+'_W'].to_numpy()
  axs[0,col].plot(t,y,color=colors[model],ls=styles[model],lw=.80 if model=='Measured' else .65,label=model,zorder=5 if model=='Measured' else 3)
  if model!='Measured':
   delta=y-actual;cum=np.r_[0,np.cumsum(.5*(delta[1:]+delta[:-1])*np.diff(t))];residual_values.append(cum)
   assert abs(cum[-1]-(sel['models'][model]['predicted_J']-sel['actual_J']))<1e-7
   axs[1,col].plot(t,cum,color=colors[model],ls=styles[model],lw=.70)
 axs[0,col].set_title(f'({chr(65+col)}) {name}',loc='left',fontweight='bold',pad=8)
 axs[1,col].axhline(0,color='#333333',lw=.55);axs[1,col].set_xlabel('Scoring time (s)')
 for ax in axs[:,col]:
  style(ax);ax.set_xlim(0,t[-1]);ax.grid(axis='y',linewidth=.45)
  ax.tick_params(width=.70)
  for spine in ['left','bottom']:ax.spines[spine].set_linewidth(.70)
 axs[1,col].set_xticks([0,10,20,30,40] if col==0 else [0,5,10,15,20,25])
pmin=min(v.min() for v in power_values);pmax=max(v.max() for v in power_values)
emin=min(v.min() for v in residual_values);emax=max(v.max() for v in residual_values)
axs[0,0].set_ylim(pmin-.35,pmax+.55);axs[0,0].set_yticks([7.5,9,10.5,12,13.5])
axs[1,0].set_ylim(min(-1,emin-1),emax+2);axs[1,0].set_yticks([0,10,20,30])
axs[0,0].set_ylabel('Power (W)');axs[1,0].set_ylabel('Cumulative energy\nerror (J)')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.99),ncol=5,frameon=False,fontsize=9,handlelength=2,columnspacing=1.25)
fig.subplots_adjust(left=.115,right=.985,top=.825,bottom=.13,wspace=.15,hspace=.20)
save(fig,'Fig6_Clap_B3_Bow_C3','six_page_hardware_traces.pdf')

# Fig.7: original subset identities and model-selection rules; new 45-recording evaluation.
cv=pd.read_csv(A/'coverage_metrics.csv')
fig,axs=plt.subplots(1,3,figsize=(180*MM,91*MM),sharey=True)
quant=[]
for i,(ax,model) in enumerate(zip(axs,['R1','R2','R3'])):
 for strategy,col,ls,label in [('random','#777777','--','Random trials'),('coverage','#2879A4','-','Joint + posture coverage')]:
  part=cv[(cv.model==model)&(cv.strategy==strategy)];budgets=sorted(part.n.unique())
  vv=[part[part.n==n].energy_error_pct.to_numpy() for n in budgets];assert all(len(v)==20 for v in vv)
  med=np.array([np.median(v) for v in vv]);q25=np.array([np.quantile(v,.25) for v in vv]);q75=np.array([np.quantile(v,.75) for v in vv])
  ax.plot(budgets,med,'o',ls=ls,color=col,lw=1.8,ms=3.5,label=label)
  ax.fill_between(budgets,q25,q75,color=col,alpha=.14,lw=0)
  quant.append(dict(model=model,strategy=strategy,budgets=[int(n) for n in budgets],median=med.tolist(),q25=q25.tolist(),q75=q75.tolist()))
 ax.set_yscale('log');ax.set_yticks([2,10,50,200]);ax.yaxis.set_major_formatter(ScalarFormatter());ax.yaxis.set_minor_locator(NullLocator())
 ax.set_xlim(8,121);ax.set_xticks([20,48,80,116]);ax.tick_params(axis='x',labelsize=8.5)
 ax.set_title(f'({chr(65+i)}) {model}',loc='left',weight='bold',pad=9);style(ax)
allq=[x for q in quant for x in q['q25']+q['q75']]
axs[0].set_ylim(min(1.8,min(allq)*.85),max(300,max(allq)*1.12))
axs[0].set_ylabel('Mean energy error (%)');fig.supxlabel('Training trials (8 s each)',fontsize=10,y=.06)
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.99),ncol=2,frameon=False,fontsize=9.5)
fig.text(.55,.018,'Median and interquartile range across 20 training subsets',ha='center',fontsize=8.5)
fig.text(.55,.875,'Evaluation: 9 motions × 5 executions (45 recordings)',ha='center',fontsize=8.7)
fig.subplots_adjust(left=.10,right=.985,bottom=.23,top=.75,wspace=.17)
save(fig,'Fig7_coverage_45_executions','six_page_experience_coverage.pdf')
(OUT/'figure_verification.json').write_text(json.dumps({'Fig5':{m:summary[m] for m in models},'Fig6':summary['selection'],'Fig7':quant},indent=2))
print('Generated numerical figures 5, 6 and 7 in the output directory.')
