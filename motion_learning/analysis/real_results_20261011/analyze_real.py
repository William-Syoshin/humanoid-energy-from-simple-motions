"""Offline, independently audited analysis of the returned 24 hardware recordings.
No robot-control modules are imported. Source files are never modified.
"""
from pathlib import Path
import argparse,csv,hashlib,json,itertools,platform
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
DEFAULT=PROJECT/'logs/real_results_20261011_import/KHR3HV_Real_Results_Seeds10-15_20261011'
CONDITIONS=['R1','R2','R3','Task-only'];SEEDS=list(range(10,16))
BLUE='#2378A8';RED='#C44E52';GREY='#A7ADB4'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(p,data):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)))
        w.writeheader();w.writerows(data)
def require(test,message):
    if not test:raise ValueError(message)
def integrate(t,p,start,end):
    require(t[0]<=start<end<=t[-1],'Unbracketed integration interval')
    grid=np.unique(np.r_[start,t[(t>=start)&(t<=end)],end])
    return float(np.trapezoid(np.interp(grid,t,p),grid))
def clean(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#DDDFE2',lw=.5);ax.set_axisbelow(True)
def savefig(fig,out,name):
    for ext in ['pdf','svg','png']:fig.savefig(out/f'{name}.{ext}',dpi=350,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def analyze(source,out):
    out.mkdir(parents=True,exist_ok=False)
    hashes=read(source/'SHA256_MANIFEST.json')
    for rel,digest in hashes.items():
        p=(source/rel).resolve();require(p.is_relative_to(source.resolve()),'Unsafe manifest path')
        require(sha(p)==digest,'File hash mismatch: '+rel)
    records=read(source/'RESULTS.json')['trials']
    require(len(records)==24 and {(r['seed'],r['condition']) for r in records}==set(itertools.product(SEEDS,['R1','R2','R3','Baseline'])),'Incomplete/duplicate conditions')
    listed={r['raw_directory'] for r in records}
    actual={str(p.parent.relative_to(source)) for p in (source/'experiment/commissioning_raw').rglob('meta.json')}
    require(listed==actual,'Unlisted or missing acquisition attempts')
    plans={sha(p):(p,read(p)) for p in (source/'experiment').glob('plan*.json')}
    table=[];traces={};configs=[];bounds=[];joints=[];sensitivity=[];max_error=0
    for rec in records:
        p=source/rec['raw_directory'];meta=read(p/'meta.json');cfg=read(p/'config.json');motion=read(p/'motion.json');trial=read(p/'trial.json')
        cond='Task-only' if rec['condition']=='Baseline' else rec['condition'];seed=rec['seed']
        require(meta['status']=='complete' and not meta['mock'] and not meta['reader_errors'],'Incomplete/Mock acquisition')
        require(trial['training_seed']==seed and trial['condition']==cond and trial['repeat']==1,'Identity mismatch')
        source_motion=source/'experiment/data'/trial['motion'];local_motion=PROJECT/'hardware_transfer_distance/data'/trial['motion']
        require(sha(source_motion)==meta['motion_sha256']==trial['motion_sha256'],'Motion hash mismatch')
        require(motion==read(source_motion)==read(local_motion),'Motion differs from local Webots export')
        require(motion['condition']==cond and motion['training_seed']==seed and motion['evaluation_seed']==20000,'Motion identity mismatch')
        require(motion['weight']==(0 if cond=='Task-only' else .3),'Reward coefficient mismatch')
        require(sha(p/'config.json')==meta['config_sha256'],'Config snapshot changed')
        require(meta['plan_sha256'] in plans,'Missing executed plan snapshot')
        planfile,plan=plans[meta['plan_sha256']]
        require(trial in plan['trials'],'Trial not in executed plan')
        require(meta['capture_code_sha256']==sha(source/'experiment/capture.py'),'Capture code mismatch')
        job=f'task_only-lambda0-seed{seed}' if cond=='Task-only' else f'{cond}-lambda0.3-seed{seed}'
        saved_episode=read(PROJECT/'logs/distance_goal_20261011'/job/'evaluation/episode_20000.json')
        require(motion['sim']==saved_episode,'Sim metrics differ from archived episode')
        start,end=meta['score_start'],meta['score_end'];duration=end-start
        require(abs(duration-motion['duration_s'])<1e-7,'Replay duration mismatch')
        commands=rows(p/'commands.csv');require(len(commands)==len(motion['commands']),'Missing commands')
        late=[]
        for k,c in enumerate(commands):
            due=start+motion['commands'][k]['t_s'];before=float(c['request_s']);after=float(c['response_s'])
            require(int(c['index'])==k and int(c['ok'])==1,'Missing ACK or wrong sequence')
            require(abs(float(c['scheduled_s'])-due)<1e-7 and before>=due and after>=before,'Invalid command times')
            require(after-due<=cfg['max_command_lateness_s']+1e-9,'Command deadline exceeded')
            late.append(after-due)
        power=rows(p/'power.csv');t=np.array([float(x['pc_s'])-cfg['timing']['receive_latency_s'] for x in power])
        device=np.array([float(x['device_s']) for x in power]);v=np.array([float(x['voltage_V']) for x in power]);a=np.array([float(x['current_A']) for x in power]);watts=v*a
        require(all(np.isfinite(x).all() for x in [t,device,v,a]),'Nonfinite samples')
        require(np.all(np.diff(t)>0) and np.all(np.diff(device)>0),'Clock reset/repetition')
        i=max(0,np.searchsorted(t,start)-1);j=min(len(t),np.searchsorted(t,end)+1)
        gap=float(np.max(np.diff(t[i:j])));dgap=float(np.max(np.diff(device[i:j])))
        require(max(gap,dgap)<=cfg['max_power_gap_s'],'Power sampling gap')
        require(not any(int(x['clip']) for x in power[i:j]),'Clipped samples')
        cal=cfg['power_calibration']
        require(np.allclose(v,[float(x['voltage_raw'])*cal['voltage_gain']+cal['voltage_offset'] for x in power]),'Voltage scaling mismatch')
        require(np.allclose(a,[float(x['current_raw'])*cal['current_gain']+cal['current_offset'] for x in power]),'Current scaling mismatch')
        energy=integrate(t,watts,start,end);error=abs(energy-rec['energy_J']);max_error=max(max_error,error)
        require(error<1e-7 and abs(energy/duration-rec['mean_power_W'])<1e-7,'Energy disagrees with return summary')
        latency=cfg['timing']['receive_latency_s'];u=cfg['timing']['uncertainty_s']
        # Timing sensitivity is not a confidence interval or an accuracy bound.
        shifts=[integrate(t,watts,start+s,end+s) for s in [-u,0,u]]
        endpoints=[integrate(t,watts,start+s,end+e) for s in [-u,u] for e in [-u,u]]
        sensitivity.append(dict(condition=cond,seed=seed,timing_uncertainty_s=u,nominal_energy_J=energy,
            shifted_window_min_J=min(shifts),shifted_window_max_J=max(shifts),independent_endpoint_min_J=min(endpoints),independent_endpoint_max_J=max(endpoints)))
        fb=rows(p/'initial_feedback.csv');require(len(fb)==17 and all(int(x['ok']) for x in fb),'Invalid initial feedback')
        initial_error=max(abs(float(x['q_rad'])-motion['initial_pose_rad'][k]) for k,x in enumerate(fb))
        require(initial_error<=cfg['max_initial_error_rad'],'Initial pose error exceeds configured limit')
        mapping=[{k:x[k] for k in ['name','id','sio','zero_pos','sign','units_per_rad']} for x in cfg['joints']]
        joints.append(json.dumps(mapping,sort_keys=True));configs.append(cfg)
        for x in cfg['joints']:bounds.append(dict(condition=cond,seed=seed,joint=x['name'],min_rad=x['min_rad'],max_rad=x['max_rad'],max_speed_rad_s=x['max_speed_rad_s']))
        sim=motion['sim'];row=dict(condition=cond,seed=seed,physical_repeat=1,trial_id=trial['trial_id'],started_utc=meta['started_utc'],
            real_energy_J=energy,real_mean_power_W=energy/duration,replay_duration_s=duration,
            real_distance_m=None,real_speed_m_s=None,real_time_to_20cm_s=None,real_energy_to_20cm_J=None,
            physical_outcome=rec['physical_outcome'],mapping_verified=cfg['mapping_verified'],commissioning=meta['commissioning'],
            power_timestamp_contract=cfg['power_timestamp_contract'],receive_latency_s=latency,timing_uncertainty_s=u,
            sim_distance_m=sim['distance_m'],sim_ood_fraction=sim['ood_rate'],
            command_count=len(commands),power_sample_count=len(power),max_ack_lateness_s=max(late),max_power_gap_s=gap,
            max_initial_pose_error_rad=initial_error,mean_voltage_V=float(np.mean(v[i:j])),min_voltage_V=float(v[i:j].min()),
            max_current_A=float(a[i:j].max()),negative_power_samples=int((watts[i:j]<0).sum()),
            executed_plan=planfile.name,source_raw=rec['raw_directory'])
        for predictor in ['R1','R2','R3']:
            row[f'sim_{predictor}_energy_J']=sim['predicted_energy_J'][predictor]
            row[f'sim_{predictor}_mean_power_W']=sim['predicted_energy_J'][predictor]/sim['duration_s']
        table.append(row)
        grid=np.unique(np.r_[start,t[(t>=start)&(t<=end)],end]);traces[cond,seed]=(grid-start,np.interp(grid,t,watts))
    require(len(set(joints))==1,'Servo mapping changed across recorded conditions')
    table.sort(key=lambda r:(r['seed'],CONDITIONS.index(r['condition'])))
    ordered=sorted(table,key=lambda r:r['started_utc']);orders={str(s):[r['condition'] for r in ordered if r['seed']==s] for s in SEEDS}
    summary=[]
    for cond in CONDITIONS:
        g=[r for r in table if r['condition']==cond]
        for metric in ['energy_J','mean_power_W']:
            for domain,key in [('Real','real_'+metric),('Sim (R2 predictor)','sim_R2_'+metric)]:
                x=np.array([r[key] for r in g]);summary.append(dict(condition=cond,domain=domain,metric=metric,n_seeds=6,physical_repeats_per_seed=1,mean=float(x.mean()),sd=float(x.std(ddof=1)),min=float(x.min()),max=float(x.max())))
        x=np.array([r['replay_duration_s'] for r in g]);summary.append(dict(condition=cond,domain='Sim and Real matched',metric='duration_s',n_seeds=6,physical_repeats_per_seed=1,mean=float(x.mean()),sd=float(x.std(ddof=1)),min=float(x.min()),max=float(x.max())))
    paired=[];effects=[]
    for cond in CONDITIONS[:-1]:
        for seed in SEEDS:
            r=next(x for x in table if x['condition']==cond and x['seed']==seed);b=next(x for x in table if x['condition']=='Task-only' and x['seed']==seed)
            item=dict(condition=cond,seed=seed)
            for key in ['energy_J','mean_power_W']:
                item['delta_'+key]=r['real_'+key]-b['real_'+key]
                item['delta_'+key+'_pct']=100*(r['real_'+key]/b['real_'+key]-1)
            item['delta_duration_s']=r['replay_duration_s']-b['replay_duration_s'];paired.append(item)
        g=[r for r in table if r['condition']==cond];base=[r for r in table if r['condition']=='Task-only'];pairs=[r for r in paired if r['condition']==cond]
        effects.append(dict(condition=cond,n_pairs=6,
            energy_reduction_ratio_of_means_pct=100*(1-np.mean([r['real_energy_J'] for r in g])/np.mean([r['real_energy_J'] for r in base])),
            mean_paired_energy_reduction_pct=-float(np.mean([r['delta_energy_J_pct'] for r in pairs])),
            power_change_ratio_of_means_pct=100*(np.mean([r['real_mean_power_W'] for r in g])/np.mean([r['real_mean_power_W'] for r in base])-1),
            lower_energy_seeds=sum(r['delta_energy_J']<0 for r in pairs),lower_mean_power_seeds=sum(r['delta_mean_power_W']<0 for r in pairs)))
    write_csv(out/'all_trials.csv',table);write_csv(out/'summary.csv',summary);write_csv(out/'paired_effects.csv',paired);write_csv(out/'effect_summary.csv',effects)
    write_csv(out/'timing_sensitivity.csv',sensitivity);write_csv(out/'configured_joint_limits.csv',bounds)
    dump(out/'audit.json',dict(checked_utc=datetime.now(timezone.utc).isoformat(),source=str(source),files_verified=len(hashes),physical_recordings=24,
        matched_original_Webots_motions=24,matched_sim_evaluations=24,maximum_reintegration_difference_J=max_error,
        all_plans_commands_acks_configs_verified=True,servo_mapping_variants=len(set(joints)),acquisition_order=orders,
        all_commissioning=all(r['commissioning'] for r in table),all_mapping_unverified=all(not r['mapping_verified'] for r in table),
        real_distance_measurements=0,max_ack_lateness_s=max(r['max_ack_lateness_s'] for r in table),max_power_gap_s=max(r['max_power_gap_s'] for r in table),
        import_provenance=read(source.parent/'import_provenance.json'),analysis_sha256=sha(Path(__file__)),python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,'axes.titlesize':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.7,'savefig.transparent':False})
    # Fig 1: matched trajectory evaluation, common R2 predictor for all conditions.
    fig,axs=plt.subplots(2,1,figsize=(7.2,5.6),sharex=True)
    for ax,metric,title,label in zip(axs,['mean_power_W','energy_J'],['(a) Mean power during replay','(b) Total energy during replay'],['Mean power (W)','Replay energy (J)']):
        for k,cond in enumerate(CONDITIONS):
            g=[r for r in table if r['condition']==cond]
            sim=np.array([r['sim_R2_'+metric] for r in g]);real=np.array([r['real_'+metric] for r in g])
            for vals,offset,color,marker in [(sim,-.17,BLUE,'o'),(real,.17,RED,'s')]:
                ax.errorbar(k+offset,vals.mean(),yerr=vals.std(ddof=1),fmt=marker,markersize=6,color=color,elinewidth=1.6,capsize=6,capthick=1.3,zorder=4)
        ax.set_ylabel(label);ax.set_title(title,loc='left',fontweight='bold');clean(ax);ax.set_xlim(-.5,3.5)
    axs[0].set_ylim(7.8,9.6);axs[1].set_ylim(60,135)
    axs[1].set_xticks(range(4),CONDITIONS);axs[1].set_xlabel('Training reward condition')
    fig.legend(handles=[Line2D([],[],color=BLUE,marker='o',ls='',label='Sim: common R2 predictor'),Line2D([],[],color=RED,marker='s',ls='',label='Real: measured')],loc='upper center',bbox_to_anchor=(.55,1.005),ncol=2,frameon=False)
    fig.text(.5,.025,'Mean ± SD across six training seeds; one physical replay per seed and condition.',ha='center',fontsize=8)
    fig.text(.5,.003,'Replay durations differ. Hardware displacement and 20 cm attainment were not measured.',ha='center',fontsize=8)
    fig.subplots_adjust(top=.89,bottom=.16,left=.12,right=.98,hspace=.28);savefig(fig,out,'Fig1_SimReal_power_energy')
    # Fig 2: paired descriptive differences, no significance brackets.
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.45));colors=plt.get_cmap('tab10')(np.arange(6))
    for ax,key,title in zip(axs,['delta_energy_J_pct','delta_mean_power_W_pct'],['(a) Replay energy','(b) Mean power']):
        ax.axhline(0,color='#424B53',lw=.9,ls='--')
        for n,seed in enumerate(SEEDS):
            y=[next(r[key] for r in paired if r['condition']==c and r['seed']==seed) for c in CONDITIONS[:-1]]
            ax.plot(np.arange(3)+np.linspace(-.05,.05,6)[n],y,marker='o',markersize=4,color=colors[n],lw=.8,alpha=.85,label=f'Seed {seed}')
        ax.set_xticks(range(3),CONDITIONS[:-1]);ax.set_xlim(-.35,2.35);ax.set_title(title,loc='left',fontweight='bold');ax.set_ylabel('Change from same-seed Task-only (%)');clean(ax)
    fig.legend(*axs[0].get_legend_handles_labels(),ncol=6,loc='lower center',bbox_to_anchor=(.5,.035),frameon=False,fontsize=7.5)
    fig.text(.5,.012,'Negative values indicate lower measured values; comparisons are descriptive.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.10,right=.98,top=.88,bottom=.25,wspace=.42);savefig(fig,out,'Fig2_paired_hardware_changes')
    # Supplementary trajectories retain original durations and every seed.
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.3),sharex=True,sharey=True)
    for ax,cond,letter in zip(axs.ravel(),CONDITIONS,'abcd'):
        for n,seed in enumerate(SEEDS):
            t,p=traces[cond,seed];ax.plot(t,p,color=colors[n],lw=.85,alpha=.8,label=f'Seed {seed}')
        ax.set_title(f'({letter}) {cond}',loc='left',fontweight='bold');clean(ax);ax.set_xlim(0,13.6)
    for ax in axs[-1]:ax.set_xlabel('Time from replay start (s)')
    for ax in axs[:,0]:ax.set_ylabel('Measured power (W)')
    fig.legend(*axs[0,0].get_legend_handles_labels(),ncol=6,loc='lower center',bbox_to_anchor=(.5,.025),frameon=False,fontsize=8)
    fig.text(.5,.005,'Original ≈10 Hz samples with interpolated window boundaries; no smoothing or time normalization.',ha='center',fontsize=7.5)
    fig.subplots_adjust(top=.92,bottom=.18,left=.10,right=.98,hspace=.35,wspace=.13);savefig(fig,out,'FigS1_measured_power_traces')
    fig,ax=plt.subplots(figsize=(5.5,3.15))
    for k,cond in enumerate(CONDITIONS):
        x=np.array([r['replay_duration_s'] for r in table if r['condition']==cond])
        ax.scatter(k+np.linspace(-.06,.06,6),x,color='#566675',s=22,alpha=.65)
        ax.errorbar(k,x.mean(),yerr=x.std(ddof=1),fmt='D',color='#26394A',capsize=5,markersize=5)
    ax.set_xticks(range(4),CONDITIONS);ax.set_ylabel('Command replay duration (s)');ax.set_ylim(8.7,14.3);clean(ax)
    ax.set_title('Replay duration matched between simulation and hardware',fontsize=9)
    fig.text(.5,.01,'Mean ± SD across six seeds; not measured hardware time to 20 cm.',ha='center',fontsize=8)
    fig.subplots_adjust(bottom=.20,left=.14,right=.97,top=.88);savefig(fig,out,'FigS2_replay_duration')
    dump(out/'analysis.json',dict(summary=summary,effects=effects,statistics='Descriptive mean and sample SD across 6 seeds, 1 physical replay each; no inferential tests',
        sim_evaluation_seed=20000,sim_primary_predictor='R2',all_real_recordings_included=True,real_distance_unavailable=True,
        limitations=['Unmeasured displacement and task success','Provisional servo mapping, all recordings marked commissioning','Fixed R1-R3-R2-Task-only order within seed','One hardware replay per trained policy','Legacy receive timestamps and existing meter values; no new independent calibration','Calibration and return-to-pose energy excluded']))
    print(json.dumps(effects,indent=2));print('OUTPUT',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=DEFAULT);p.add_argument('--out',type=Path,default=HERE/'figures_v2')
    a=p.parse_args();analyze(a.source.resolve(),a.out.resolve())
