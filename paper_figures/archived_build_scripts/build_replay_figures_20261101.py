"""Reorder the submitted energy figure; keep expanded comparison as preview only."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];D=P.parent/'data/rl_application_20261011';O=P/'previews/fig10_duration_power_20261101'
a=pd.read_csv(D/'all_trials.csv');saved=pd.read_csv(D/'summary.csv');conditions=['Task-only','R1','R2','R3']; x=np.arange(4);stats=[]
series=[('Sim (R2 predictor)','sim_R2_', '#227EAA','o',-.15,'Sim: common R2 predictor'),('Real','real_', '#CC4B53','s',.15,'Real: measured')]
def get(metric,domain,field):
 means=[];sds=[]
 for cond in conditions:
  r=a[a.condition==cond];assert sorted(r.seed.tolist())==list(range(10,16))
  v=r[field].to_numpy();mean=float(v.mean());sd=float(v.std(ddof=1));ss=saved[(saved.condition==cond)&(saved.domain==domain)&(saved.metric==metric)].iloc[0]
  assert np.isclose(mean,ss['mean'],atol=1e-10,rtol=0) and np.isclose(sd,ss['sd'],atol=1e-10,rtol=0)
  means.append(mean);sds.append(sd);stats.append(dict(condition=cond,domain=domain,metric=metric,mean=mean,sample_sd=sd,n=6))
 return means,sds
energy={dom:get('energy_J',dom,prefix+'energy_J') for dom,prefix,*_ in series}
power={dom:get('mean_power_W',dom,prefix+'mean_power_W') for dom,prefix,*_ in series}
duration=get('duration_s','Sim and Real matched','replay_duration_s')
rc={'font.family':'DejaVu Sans','font.size':7.2,'axes.labelsize':7.2,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6}
with plt.rc_context(rc):
 fig,ax=plt.subplots(figsize=(3.45,2.05))
 for dom,prefix,col,m,offset,label in series:
  mean,sd=energy[dom];ax.errorbar(x+offset,mean,yerr=sd,fmt=m,color=col,markersize=3.6,elinewidth=1,capsize=3,capthick=.8,linestyle='none',label=label)
 ax.set_xticks(x,conditions);ax.set_xlim(-.5,3.5);ax.set_ylim(60,135);ax.set_yticks([60,80,100,120]);ax.tick_params(length=2,pad=2)
 ax.set_ylabel('Replay energy (J)',labelpad=2);ax.set_xlabel('Training reward condition',labelpad=3)
 ax.set_axisbelow(True);ax.grid(axis='y',color='#DADDE2',lw=.5)
 # Moved from left to right so the legend does not obscure the now-leftmost baseline.
 ax.legend(loc='upper right',frameon=False,fontsize=6.4,handletextpad=.4,labelspacing=.3,borderaxespad=.1)
 fig.subplots_adjust(left=.15,right=.98,bottom=.23,top=.97)
 for ext in ['pdf','png']:fig.savefig(P/('figures/six_page_rl_replay_energy.'+ext),dpi=300)
 plt.close(fig)
with plt.rc_context({**rc,'font.size':10,'axes.labelsize':10,'axes.titlesize':11,'xtick.labelsize':9.5,'ytick.labelsize':9.5}):
 fig,axs=plt.subplots(1,3,figsize=(11.4,3.8))
 for ax in axs:
  ax.set_xticks(x,conditions);ax.set_xlim(-.5,3.5);ax.tick_params(length=3);ax.grid(axis='y',color='#DADDE2',lw=.6);ax.set_axisbelow(True)
 axs[0].errorbar(x,*[duration[0]],yerr=duration[1],fmt='D',color='#4B4B4B',ms=5.5,elinewidth=1.4,capsize=4,label='Matched simulation / hardware duration')
 axs[0].set_title('(a) Replay duration',loc='left',pad=9);axs[0].set_ylabel('Duration (s)');axs[0].set_ylim(0,15)
 for ax,vals,title,ylabel,lim in [(axs[1],power,'(b) Mean power','Mean power (W)',(7.8,9.5)),(axs[2],energy,'(c) Replay energy','Energy (J)',(60,135))]:
  for dom,prefix,col,m,offset,label in series:
   mean,sd=vals[dom];ax.errorbar(x+offset,mean,yerr=sd,fmt=m,color=col,ms=5.5,elinewidth=1.4,capsize=4,label=label)
  ax.set_title(title,loc='left',pad=9);ax.set_ylabel(ylabel);ax.set_ylim(*lim)
 fig.legend(*axs[1].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.68,1.01),frameon=False,ncol=2,fontsize=10)
 fig.text(.07,.955,'Duration matched per exported episode',fontsize=9.5,color='#444444')
 fig.supxlabel('Training reward condition',y=.13,fontsize=10)
 fig.text(.07,.025,'Mean ± sample SD across six training seeds; one hardware replay per policy.\nDuration is replay time; hardware displacement and time to 20 cm were not measured.',fontsize=9,color='#444444')
 fig.subplots_adjust(left=.07,right=.99,bottom=.25,top=.79,wspace=.37)
 for ext in ['pdf','png','svg']:fig.savefig(O/('Fig10_duration_power_energy_PREVIEW.'+ext),dpi=200)
 plt.close(fig)
# Vector equation for the revised overview.
fig=plt.figure(figsize=(4.5,.6));fig.text(.5,.45,r'$\widehat{P}(t)=\mathbf{w}^{\mathsf{T}}\mathbf{x}(t)$',fontsize=27,ha='center',va='center')
fig.savefig(P/'qa/20261101_revision/fig1_equation.pdf',transparent=True);plt.close(fig)
(O/'verification.json').write_text(json.dumps(dict(condition_order=conditions,all_means_and_sample_sd_match_saved_summary=True,statistics=stats,data_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [D/'all_trials.csv',D/'summary.csv']},expanded_figure_inserted_in_manuscript=False),indent=2)+'\n')
pd.DataFrame(stats).to_csv(O/'plotted_statistics.csv',index=False)
print('Main Fig10 reordered; expanded Fig10 saved only under previews. All statistics verified.')
