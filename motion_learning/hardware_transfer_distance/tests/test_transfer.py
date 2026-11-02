import copy
import json
import numpy as np
import pytest
from common import NAMES,parse_power,integrate_power,validate_config,read,save,sha
from experiment import config_template,make_plan,annotate
from capture import run


def motion():
    return dict(schema='khr-transfer-v1',motion_id='Task-only_lambda0_seed10',condition='Task-only',weight=0,training_seed=10,
                evaluation_seed=20000,joint_names=NAMES,mode='open_loop_replay_of_archived_policy_commands',
                duration_s=.2,control_dt_s=.02,initial_pose_rad=[0.]*17,
                commands=[dict(t_s=i*.02,q_rad=[.001*i]*17) for i in range(10)],
                sim=dict(predicted_energy_J={m:1. for m in ['R1','R2','R3']},mean_speed_m_s=.01))


def calibrated(tmp_path):
    p=tmp_path/'config.json';config_template(p);c=read(p)
    c.update(mapping_verified=True,mixing_off_verified=True,max_current_A=2.,min_voltage_V=8.,max_initial_error_rad=.1)
    for j in c['joints']:j.update(sign=1,zero_pos=7500,min_rad=-1.,max_rad=1.,max_speed_rad_s=1.)
    c['power_calibration'].update(verified=True,voltage_gain=1.,voltage_offset=0.,current_gain=1.,current_offset=0.)
    c['timing'].update(verified=True,receive_latency_s=0.,uncertainty_s=.01)
    save(p,c);return p,c


def bundle(tmp_path):
    b=tmp_path/'data';(b/'motions').mkdir(parents=True);items=[]
    for cond in ['R1','R2','R3','Task-only']:
        for seed in range(10,16):
            m=motion();m.update(condition=cond,weight=0 if cond=='Task-only' else .3,training_seed=seed,motion_id=f'{cond}_{seed}')
            p=b/'motions'/f'{cond}_{seed}.json';save(p,m)
            items.append(dict(file='motions/'+p.name,condition=cond,weight=m['weight'],training_seed=seed,sha256=sha(p)))
    save(b/'manifest.json',dict(motions=items));return b


def test_parser_clip_and_legacy():
    assert parse_power('1234,10,0.8')['device_s']==1.234
    assert parse_power('1234,10,0.8,1')['clip']==1
    assert parse_power('# FIRMWARE test') is None
    for s in ['1,nan,1','1,10,1,2','broken,data','-1,10,1']:
        with pytest.raises(ValueError):parse_power(s)


def test_exact_boundary_integral_and_clock_errors(tmp_path):
    _,cfg=calibrated(tmp_path)
    samples=[dict(pc_s=i*.1,device_s=i*.1,voltage_V=10,current_A=1+i*.1,clip=0) for i in range(11)]
    assert integrate_power(samples,.05,.95,cfg)['energy_J']==pytest.approx(13.5)
    for bad in ['clip','gap','duplicate','coverage']:
        s=copy.deepcopy(samples)
        if bad=='clip':s[5]['clip']=1
        if bad=='gap':s=s[:3]+s[8:]
        if bad=='duplicate':s[5]['device_s']=s[4]['device_s']
        if bad=='coverage':s=s[2:]
        with pytest.raises(ValueError):integrate_power(s,.05,.95,cfg)


def test_mapping_no_clamping_or_guessed_sign(tmp_path):
    p=tmp_path/'unverified.json';config_template(p)
    with pytest.raises(ValueError):validate_config(read(p),motion())
    _,c=calibrated(tmp_path);validate_config(c,motion())
    for change in ['range','speed','duplicate']:
        bad=copy.deepcopy(c)
        if change=='range':bad['joints'][0]['max_rad']=.001
        if change=='speed':bad['joints'][0]['max_speed_rad_s']=.001
        if change=='duplicate':bad['joints'][0].update(id=7,sio=1)
        with pytest.raises(ValueError):validate_config(bad,motion())


def test_mock_no_serial_and_missing_repeat_not_counted(tmp_path,monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'serial',None)
    b=bundle(tmp_path);plan=tmp_path/'plan.json';make_plan(b,plan,[.3],2,7)
    cfg,c=calibrated(tmp_path);tr=read(plan)['trials'][0];raw=tmp_path/'raw';out=raw/tr['trial_id']
    run(b,plan,tr['trial_id'],cfg,out,mock=True)
    assert read(out/'meta.json')['mock'] is True
    annotate(out,.20,'success','mock test','not hardware')
    from analyze import analyze
    with pytest.raises(ValueError,match='Mock data'):analyze(b,plan,raw,tmp_path/'forbidden')
    analyze(b,plan,raw,tmp_path/'demo',allow_mock=True)
    import csv
    values=list(csv.DictReader((tmp_path/'demo/seed_means.csv').open()))
    assert all(x['real_complete']=='False' for x in values)
    assert read(tmp_path/'demo/analysis.json')['valid_trials']==1
    with pytest.raises(FileExistsError):run(b,plan,tr['trial_id'],cfg,out,mock=True)


def test_dry_run_never_opens_ports(tmp_path,monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'serial',None)
    b=bundle(tmp_path);plan=tmp_path/'plan.json';make_plan(b,plan,[.3],1,7)
    cfg=tmp_path/'config.json';config_template(cfg)
    out=tmp_path/'no_output';run(b,plan,read(plan)['trials'][0]['trial_id'],cfg,out)
    assert not out.exists()


def test_complete_seed_aggregation_keeps_failed_walks(tmp_path):
    b=bundle(tmp_path);plan=tmp_path/'plan.json';make_plan(b,plan,[.3],1,7,seeds=range(10,16))
    cfg=tmp_path/'hardware.json';config_template(cfg)
    raw=tmp_path/'raw'
    for tr in read(plan)['trials']:
        p=raw/tr['trial_id'];run(b,plan,tr['trial_id'],cfg,p,mock=True)
        outcome='fall' if tr['condition']=='R1' and tr['training_seed']==10 else 'success'
        annotate(p,.20,outcome,'MOCK','')
    from analyze import analyze
    analyze(b,plan,raw,tmp_path/'fig',allow_mock=True)
    from common import rows
    seed=rows(tmp_path/'fig/seed_means.csv')
    assert len(seed)==24 and all(x['real_complete']=='True' for x in seed)
    assert all(float(x['real_energy_J'])==pytest.approx(1.6) for x in seed)
    failure=next(x for x in seed if x['condition']=='R1' and x['training_seed']=='10')
    assert float(failure['outcome_success_fraction'])==0
    assert read(tmp_path/'fig/analysis.json')['valid_trials']==24


def test_pilot_and_all_seed_plans(tmp_path):
    from collections import Counter
    b=bundle(tmp_path)
    for seeds,count,name in [([10],12,'pilot'),(list(range(10,16)),72,'all')]:
        path=tmp_path/(name+'.json');make_plan(b,path,[.3],3,20261011,seeds)
        p=read(path);assert len(p['trials'])==count
        assert set(p['training_seeds'])==set(seeds)
        assert set(Counter((t['condition'],t['training_seed']) for t in p['trials']).values())=={3}
        for i in range(0,count,4):assert {t['condition'] for t in p['trials'][i:i+4]}=={'Task-only','R1','R2','R3'}


def test_goal_timing_and_variable_window_integrals(tmp_path):
    from analyze import analyze
    from common import rows
    b=bundle(tmp_path)
    manifest=read(b/'manifest.json')
    for item in manifest['motions']:
        if item['condition']=='R3':
            path=b/item['file'];m=read(path);m['duration_s']=.4
            m['commands']=[dict(t_s=i*.02,q_rad=[.001*i]*17) for i in range(20)]
            save(path,m);item['sha256']=sha(path)
    save(b/'manifest.json',manifest)
    plan=tmp_path/'plan.json';make_plan(b,plan,[.3],3,7)
    cfg=tmp_path/'hardware.json';config_template(cfg)
    raw=tmp_path/'raw'
    for tr in read(plan)['trials']:
        p=raw/tr['trial_id'];run(b,plan,tr['trial_id'],cfg,p,mock=True)
        with pytest.raises(ValueError,match='Success requires'):annotate(p,.19,'success','video','')
        with pytest.raises(ValueError):annotate(p,.20,'success','video','',.5,.01)
        with pytest.raises(ValueError):annotate(p,.20,'success','video','',.1,None)
        # Different known constant powers exercise nonzero physical-repeat SD.
        import csv
        values=rows(p/'power.csv')
        for value in values:
            value['current_A']=float(value['current_A'])*tr['repeat']
            value['current_raw']=float(value['current_raw'])*tr['repeat']
        with (p/'power.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
        annotate(p,.20,'success','MOCK video','',.3 if tr['condition']=='R3' else .1,.01)
    analyze(b,plan,raw,tmp_path/'fig',allow_mock=True)
    data=rows(tmp_path/'fig/all_trials.csv')
    assert all(float(r['real_energy_J'])==pytest.approx((3.2 if r['condition']=='R3' else 1.6)*int(r['repeat'])) for r in data)
    assert all(float(r['real_goal_energy_J'])==pytest.approx((2.4 if r['condition']=='R3' else .8)*int(r['repeat'])) for r in data)
    values=rows(tmp_path/'fig/plot_summary.csv')
    assert all(r['n']=='3' and r['unit']=='physical replay of one fixed seed' for r in values if r['domain']=='real')
    assert all(r['n']=='1' and r['sample_sd']=='' for r in values if r['domain']=='sim')
    energy=next(r for r in values if r['domain']=='real' and r['condition']=='R1' and r['metric']=='energy_J')
    assert float(energy['mean'])==pytest.approx(3.2) and float(energy['sample_sd'])==pytest.approx(1.6)
    assert read(tmp_path/'fig/analysis.json')['observed_goal_trials']==12


def test_shortfall_energy_not_counted_as_goal(tmp_path):
    from analyze import analyze
    from common import rows
    b=bundle(tmp_path);plan=tmp_path/'plan.json';make_plan(b,plan,[.3],1,7)
    cfg=tmp_path/'hardware.json';config_template(cfg);tr=read(plan)['trials'][0]
    p=tmp_path/'raw'/tr['trial_id'];run(b,plan,tr['trial_id'],cfg,p,mock=True)
    annotate(p,.10,'shortfall','MOCK','')
    analyze(b,plan,tmp_path/'raw',tmp_path/'fig',allow_mock=True)
    r=next(r for r in rows(tmp_path/'fig/all_trials.csv') if r['trial_id']==tr['trial_id'])
    assert r['measurement_status']=='valid' and r['real_goal_energy_J']=='' and r['outcome']=='shortfall'
