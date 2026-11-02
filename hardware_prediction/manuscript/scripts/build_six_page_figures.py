"""Compact publication layouts from unchanged saved statistics; no refitting."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
P=Path(__file__).resolve().parents[1]
Q=P/'qa/six_page_revision_20261021'
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
checks={}
rc={'font.family':'DejaVu Sans','font.size':7.2,'axes.labelsize':7.2,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6}
with plt.rc_context(rc):
    random=rows(P/'data/transfer_audit_20260928/experience_subset_metrics.csv')
    covered=rows(P/'data/transfer_audit_20260928/coverage/metrics.csv')
    fig,axs=plt.subplots(1,2,figsize=(3.45,2.08),sharey=True)
    curves=[]
    for ax,model in zip(axs,['R2','R3']):
        for src,exp,restrict,label,color,ls in [(random,'budget',True,'Random trials','#777777','--'),(covered,'coverage_budget',False,'Joint + posture coverage','#2C7FAB','-')]:
            rr=[r for r in src if r['model']==model and r['experiment']==exp and (not restrict or r['group']=='whole_body')]
            ns=sorted({int(r['n']) for r in rr})
            values=[np.array([float(r['energy_error_pct']) for r in rr if int(r['n'])==n]) for n in ns]
            med=np.array([np.median(v) for v in values]);lo=np.array([np.quantile(v,.25) for v in values]);hi=np.array([np.quantile(v,.75) for v in values])
            ax.plot(ns,med,marker='o',ls=ls,color=color,label=label,ms=2.5,lw=.9)
            ax.fill_between(ns,lo,hi,color=color,alpha=.15,linewidth=0)
            curves.append(dict(model=model,policy=label,budgets=ns,median=med.tolist(),q25=lo.tolist(),q75=hi.tolist()))
        ax.set_yscale('log');ax.set_yticks([2,10,50,200]);ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_title(model,pad=3);ax.set_xticks([20,48,80,116]);ax.tick_params(length=2,pad=2)
        ax.grid(axis='y',which='major',alpha=.18);ax.set_ylim(1.8,300);ax.set_xlim(8,120)
    axs[0].set_ylabel('Mean energy error (%)',labelpad=2)
    fig.supxlabel('Training trials (8 s each)',fontsize=7.2,y=.015)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,fontsize=6.7,columnspacing=.8,handlelength=1.6,bbox_to_anchor=(.53,1.01))
    fig.subplots_adjust(left=.15,right=.99,bottom=.22,top=.78,wspace=.15)
    fig.savefig(P/'figures/six_page_experience_coverage.pdf');fig.savefig(P/'figures/six_page_experience_coverage.png',dpi=240);plt.close(fig)
    checks['coverage']=curves
    trials=rows(P/'data/rl_application_20261011/all_trials.csv');summary=rows(P/'data/rl_application_20261011/summary.csv')
    fig,ax=plt.subplots(figsize=(3.45,2.05));stats=[]
    conditions=['R1','R2','R3','Task-only']
    for domain,field,color,marker,offset,label in [('Sim (R2 predictor)','sim_R2_energy_J','#227EAA','o',-.15,'Sim: common R2 predictor'),('Real','real_energy_J','#CC4B53','s',.15,'Real: measured')]:
        means=[];sds=[]
        for condition in conditions:
            rr=[r for r in trials if r['condition']==condition]
            assert sorted(int(r['seed']) for r in rr)==list(range(10,16))
            values=np.array([float(r[field]) for r in rr]);mean=values.mean();sd=values.std(ddof=1)
            saved=next(r for r in summary if r['condition']==condition and r['domain']==domain and r['metric']=='energy_J')
            assert np.isclose(mean,float(saved['mean']),atol=1e-10,rtol=0)
            assert np.isclose(sd,float(saved['sd']),atol=1e-10,rtol=0)
            means.append(mean);sds.append(sd);stats.append(dict(condition=condition,domain=domain,mean=float(mean),sd=float(sd),n=6))
        ax.errorbar(np.arange(4)+offset,means,yerr=sds,fmt=marker,color=color,markersize=3.6,elinewidth=1,capsize=3,capthick=.8,linestyle='none',label=label)
    ax.set_xticks(range(4),conditions);ax.set_xlim(-.5,3.5);ax.set_ylim(60,135);ax.set_yticks([60,80,100,120]);ax.tick_params(length=2,pad=2)
    ax.set_ylabel('Replay energy (J)',labelpad=2);ax.set_xlabel('Training reward condition',labelpad=3)
    ax.set_axisbelow(True);ax.grid(axis='y',color='#DADDE2',lw=.5)
    ax.legend(loc='upper left',frameon=False,fontsize=6.4,handletextpad=.4,labelspacing=.3,borderaxespad=.1)
    fig.subplots_adjust(left=.15,right=.98,bottom=.23,top=.97)
    fig.savefig(P/'figures/six_page_rl_replay_energy.pdf');fig.savefig(P/'figures/six_page_rl_replay_energy.png',dpi=240);plt.close(fig)
    checks['replay']=stats
files=['data/transfer_audit_20260928/experience_subset_metrics.csv','data/transfer_audit_20260928/coverage/metrics.csv','data/rl_application_20261011/all_trials.csv','data/rl_application_20261011/summary.csv']
checks.update(source_sha256={s:hashlib.sha256((P/s).read_bytes()).hexdigest() for s in files},data_refitted=False,statistics_changed=False)
(Q/'compact_figures_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print('Rendered two single-column figures; replay means and SDs match all saved summaries.')

# Preserve all nine panels and both trace comparisons; reduce only subplot whitespace.
import importlib.util
for src,fn,replacements in [
 ('build_trial_chart.py','build_trial_chart',{'figsize=(7.1,4.45)':'figsize=(7.1,4.10)','pad=18':'pad=16','hspace=.69':'hspace=.77','hardware_trial_errors.':'six_page_hardware_trial_errors.','qa/trial_chart_checks.json':'qa/six_page_revision_20261021/trial_chart_checks.json'}),
 ('build_power_traces.py','build_power_traces',{'figsize=(7.1,3.3)':'figsize=(7.1,3.0)','hardware_traces.':'six_page_hardware_traces.','qa/fig8_crawl_getup_verification.json':'qa/six_page_revision_20261021/trace_verification.json'})
]:
    original=P/'scripts'/src
    code=original.read_text()
    for before,after in replacements.items():
        assert before in code,(src,before)
        code=code.replace(before,after)
    namespace={'__file__':str(original),'__name__':'six_page_layout'}
    exec(compile(code,str(original),'exec'),namespace)
    namespace[fn]()
print('Nine-panel errors and trace layouts regenerated from identical saved values.')
