"""Single robot-port owner, independent power reader; never runs on import."""
import csv
from datetime import datetime,timezone
import sys
import threading
import time
from pathlib import Path
import numpy as np
from common import read,save,sha,parse_power,validate_config,motion_arrays


def run(bundle,plan_path,trial_id,config_path,out,execute=False,mock=False,commissioning=False):
    if execute and mock:raise ValueError('Use --execute OR --mock')
    plan=read(plan_path)
    if sha(bundle/'manifest.json')!=plan['manifest_sha256']:raise ValueError('Changed motion manifest')
    trial=next(x for x in plan['trials'] if x['trial_id']==trial_id)
    path=bundle/trial['motion'];m=read(path);cfg=read(config_path)
    if sha(path)!=trial['motion_sha256']:raise ValueError('Motion hash mismatch')
    t,q,initial=motion_arrays(m)
    if not execute and not mock:
        print('DRY RUN: no serial port opened. Motion',trial_id,len(t),'commands;',m['duration_s'],'s')
        try:validate_config(cfg,m,commissioning=commissioning);print('Commissioning bounds passed; mapping remains provisional.' if commissioning else 'Mapping/config checks passed; initial physical pose still checked on execution.')
        except ValueError as exc:print('Before hardware execution:',exc)
        return
    if commissioning and plan.get('purpose')!='Operator-attended commissioning only; does not replace formal four-condition plan':
        raise ValueError('Commissioning requires a separate commissioning plan')
    positions,initial_pos=(None,None) if mock else validate_config(cfg,m,commissioning=commissioning)
    if mock:
        cfg['power_calibration'].update(verified=False,voltage_gain=1.,voltage_offset=0.,current_gain=1.,current_offset=0.)
        cfg['timing'].update(verified=False,receive_latency_s=0.,uncertainty_s=0.)
    if not (0<=cfg['pre_s']<=60 and 0<cfg['post_s']<=60):raise ValueError('Invalid pre/post window')
    out.mkdir(parents=True,exist_ok=False)
    save(out/'trial.json',trial);save(out/'motion.json',m);save(out/'config.json',cfg)
    meta=dict(status='failed',mock=mock,commissioning=commissioning,mapping_verified=cfg.get('mapping_verified',False),trial_id=trial_id,plan_sha256=sha(plan_path),motion_sha256=sha(path),
              started_utc=datetime.now(timezone.utc).isoformat(),score_start=None,score_end=None,
              end_behavior=m.get('end_behavior','hold_last_target'),target_distance_m=m.get('target_distance_m',.2),
              clock='time.perf_counter on this PC; device_s additionally logged',mode=m['mode'],
              config_sha256=sha(out/'config.json'),capture_code_sha256=sha(Path(__file__)))
    handles=[]
    def writer(name,fields):
        f=(out/name).open('x',newline='',encoding='utf-8');handles.append(f)
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        def append(row):w.writerow(row);f.flush()
        return append
    powerlog=writer('power.csv',['pc_s','device_s','voltage_raw','current_raw','voltage_V','current_A','clip'])
    events=writer('power_events.csv',['pc_s','line'])
    commands=writer('commands.csv',['index','scheduled_s','request_s','response_s','ok','max_lateness_s'])
    feedback=writer('initial_feedback.csv',['joint','position','q_rad','ok'])
    rcb=port=thread=None;stop=threading.Event();latest=[];errors=[];owned=False
    def read_power():
        previous_device=None
        try:
            while not stop.is_set():
                line=port.readline().decode('utf-8',errors='replace').strip();now=time.perf_counter()
                if not line:continue
                events(dict(pc_s=now,line=line))
                if line.startswith('# FIRMWARE '):meta['firmware']=line.split()[-1]
                if line.startswith('# TIMESTAMP '):meta['timestamp_source']=line.split()[-1]
                row=parse_power(line)
                if row is None:continue
                if previous_device is not None and row['device_s']<=previous_device:raise RuntimeError('Power clock reset/duplicate')
                previous_device=row['device_s'];cal=cfg['power_calibration']
                row.update(pc_s=now,voltage_V=row['voltage_raw']*cal['voltage_gain']+cal['voltage_offset'],current_A=row['current_raw']*cal['current_gain']+cal['current_offset'])
                powerlog(row);latest[:]=[row]
                if row['clip']:raise RuntimeError('Power sensor clipping')
                if abs(row['current_A'])>cfg['max_current_A']:raise RuntimeError('Current limit exceeded')
                if row['voltage_V']<cfg['min_voltage_V']:raise RuntimeError('Supply voltage below limit')
        except Exception as exc:errors.append(str(exc))
    def check():
        if errors:raise RuntimeError(errors[0])
        if not latest or time.perf_counter()-latest[0]['pc_s']>cfg['power_stale_s']:raise RuntimeError('Power stream stale')
    def wait_until(deadline):
        while time.perf_counter()<deadline:
            check();time.sleep(min(.002,max(0,deadline-time.perf_counter())))
    try:
        if mock:
            # Deterministic artificial data are kept separate and never called Real.
            meta.update(score_start=2.,score_end=2.+m['duration_s'],status='complete')
            for ts in np.arange(0,m['duration_s']+4,.1):
                powerlog(dict(pc_s=float(ts),device_s=float(ts),voltage_raw=10.,current_raw=.8,voltage_V=10.,current_A=.8,clip=0))
            for n,ts in enumerate(t):commands(dict(index=n,scheduled_s=2+ts,request_s=2+ts,response_s=2+ts+.001,ok=1,max_lateness_s=.001))
            return
        import serial
        sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
        from Rcb4BaseLib import Rcb4BaseLib
        rcb=Rcb4BaseLib()
        if not rcb.open(cfg['robot_port'],cfg['robot_baud'],.1) or not rcb.checkAcknowledge():raise RuntimeError('RCB connection failed')
        port=serial.Serial(cfg['power_port'],cfg['power_baud'],timeout=.1);time.sleep(2);port.reset_input_buffer()
        thread=threading.Thread(target=read_power,daemon=True);thread.start();port.write(b'START\n');port.flush()
        deadline=time.perf_counter()+5
        while not latest and not errors and time.perf_counter()<deadline:time.sleep(.01)
        check()
        if cfg['power_timestamp_contract']=='fresh_device' and (meta.get('firmware')!='KHR3HV_POWER_V2' or meta.get('timestamp_source')!='fresh_update_observed_ms'):
            raise RuntimeError('Fresh timestamp firmware not confirmed')
        old=rcb.getConfig()
        if old==65535:raise RuntimeError('Cannot read RCB configuration')
        # Same PC-control bit transition as archived hardware acquisition.
        new=(old & ~2)|1
        if not rcb.moveComToRamCmdSynchronize(Rcb4BaseLib.RamAddr.ConfigRamAddress.value,[new&255,new>>8]):raise RuntimeError('Cannot establish PC control')
        owned=True;meta['original_rcb_config']=old
        for k,j in enumerate(cfg['joints']):
            ok,pos=rcb.getSinglePos(j['id'],j['sio'])
            angle=(pos-j['zero_pos'])/j['units_per_rad']*j['sign']
            feedback(dict(joint=j['name'],position=pos,q_rad=angle,ok=int(bool(ok))))
            if not ok or not 3500<=pos<=11500 or abs(angle-initial[k])>cfg['max_initial_error_rad']:
                raise RuntimeError('Initial pose mismatch: '+j['name']+'; position robot using validated procedure before replay')
        check();score=time.perf_counter()+cfg['pre_s'];end=score+m['duration_s']
        meta.update(score_start=score,score_end=end)
        save(out/'meta.json',meta)
        print(f'MEASUREMENT START in {cfg["pre_s"]} s; duration {m["duration_s"]} s',flush=True)
        for n,(ts,pos) in enumerate(zip(t,positions)):
            due=score+ts;wait_until(due);check();before=time.perf_counter()
            if before-due>cfg['max_command_lateness_s']:raise RuntimeError('Command deadline missed; no catch-up burst')
            data=[Rcb4BaseLib.ServoData(j['id'],j['sio'],v) for j,v in zip(cfg['joints'],pos)]
            ok=rcb.setServoPos(data,cfg['servo_frame']);after=time.perf_counter()
            commands(dict(index=n,scheduled_s=due,request_s=before,response_s=after,ok=int(bool(ok)),max_lateness_s=after-due))
            if n==0:print('SCORE START',score,flush=True)
            if not ok:raise RuntimeError('Command acknowledgement failed')
            if after-due>cfg['max_command_lateness_s']:raise RuntimeError('Command acknowledgement too late')
        wait_until(end);print('SCORE END',end,flush=True)
        wait_until(end+cfg['post_s']);meta['status']='complete'
    except BaseException as exc:
        meta['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        # Stop issuing movement; retain last target. No automatic FREE, neutral
        # jump, or resuming a prior EEPROM motion on failure/KeyboardInterrupt.
        meta['pc_control_retained']=owned
        if port:
            try:port.write(b'STOP\n');port.flush()
            except Exception as exc:meta['power_stop_error']=str(exc)
        stop.set()
        if thread:thread.join(timeout=2)
        if port:port.close()
        if rcb:rcb.close()
        for f in handles:f.close()
        meta['reader_errors']=errors
        if errors:meta['status']='failed'
        save(out/'meta.json',meta)
