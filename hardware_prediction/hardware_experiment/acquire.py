"""Single owner of RCB serial port; independent power-reader thread."""
import csv
import math
import sys
import threading
import time
import shutil
from pathlib import Path

from core import compile_recipe, read, save, finite

def record(config_path, recipe_path, output, execute=False, purpose='primary', design_path=None):
    cfg,recipe = read(config_path),read(recipe_path)
    from protocol import validate_primary, audit_raw, verify_design
    if purpose not in ('primary','diagnostic'): raise ValueError('Unknown acquisition purpose')
    frozen=None
    if purpose == 'primary':
        validate_primary(cfg,recipe,execute)
        frozen=verify_design(design_path,config_path,recipe_path)
    trajectory = compile_recipe(cfg,recipe) if execute else []
    if execute:
        for key in ('max_initial_error_rad',):
            if not finite(cfg[key]) or cfg[key]<=0: raise ValueError(f'Set {key}')
        cal=cfg['power_calibration']
        if not cal['verified'] or not all(finite(cal[k]) for k in
                ('voltage_gain','voltage_offset','current_gain','current_offset')):
            raise ValueError('Calibrate power before motion')
        if not recipe.get('success_gate'): raise ValueError('Declare mechanical success gate')
    for k in ('ad_hz','feedback_hz'):
        if not finite(cfg[k]) or cfg[k]<=0: raise ValueError(f'Invalid {k}')
    focused_joints = recipe.get('focused_feedback_joints')
    if focused_joints is None:
        focused_joint = recipe.get('focused_feedback_joint')
        focused_joints = [] if focused_joint is None else [focused_joint]
    focused_hz = cfg.get('focused_feedback_hz', 0)
    if focused_joints:
        if (not isinstance(focused_joints, list) or len(set(focused_joints)) != len(focused_joints)
                or any(not isinstance(j, int) or not 0 <= j < 17 for j in focused_joints)):
            raise ValueError('focused_feedback_joints must contain unique indices from 0 to 16')
        if not finite(focused_hz) or focused_hz <= 0:
            raise ValueError('Set positive focused_feedback_hz')
    if not finite(recipe['duration_s']) or recipe['duration_s']<=0: raise ValueError('Invalid duration')
    import serial
    sys.path.insert(0,cfg['rcb_library'])
    from Rcb4BaseLib import Rcb4BaseLib
    out = Path(output)
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(config_path,out/'config.json')
    shutil.copyfile(recipe_path,out/'recipe.json')
    if frozen: save(out/'design.json',frozen)
    stop = threading.Event()
    latest,errors = [],[]
    handles=[]
    def writer(name,fields):
        f=(out/name).open('x',newline='',encoding='utf-8'); handles.append(f)
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        def append(row): w.writerow(row); f.flush()
        return append
    powerlog=writer('power.csv',['device_s','pc_s','voltage','current'])
    eventlog=writer('power_events.csv',['pc_s','line'])
    adlog=writer('ad.csv',['pc_s','request_s','response_s','ok','ad1','ad2','ad3','ad4'])
    fblog=writer('feedback.csv',['pc_s','request_s','response_s','joint','ok','q_rad'])
    cmdlog=writer('commands.csv',['pc_s','response_s','ok']+[f'cmd_{i}' for i in range(17)])
    rcb=Rcb4BaseLib(); port=None; thread=None; owned=False
    meta={'status':'failed','purpose':purpose,'execute':execute,'score_start':None,'score_end':None,
          'clock':'time.perf_counter, same host boot only','error':None}
    def read_power():
        try:
            while not stop.is_set():
                line=port.readline().decode('utf-8',errors='replace').strip()
                now=time.perf_counter()
                if not line: continue
                eventlog(dict(pc_s=now,line=line))
                if line.startswith('# FIRMWARE '): meta['power_firmware']=line.split()[-1]
                if line.startswith('# TIMESTAMP '): meta['power_timestamp_source']=line.split()[-1]
                if line.startswith('#'): continue
                try:
                    ts,v,i=map(float,line.split(','))
                    if not all(math.isfinite(x) for x in (ts,v,i)): raise ValueError()
                except ValueError:
                    # ESP32 emits a CSV header and occasional diagnostic lines.
                    # They are not samples and must not abort an otherwise valid run.
                    if line.lower().startswith('timestamp_ms,') or ',' not in line:
                        continue
                    errors.append('Malformed power sample: '+line); continue
                powerlog(dict(device_s=ts/1000,pc_s=now,voltage=v,current=i))
                latest[:] = [now,v,i]
        except Exception as e: errors.append(str(e))
    try:
        if not rcb.open(cfg['robot_port'],cfg['robot_baud'],0.15): raise RuntimeError('RCB connection failed')
        port=serial.Serial(cfg['power_port'],cfg['power_baud'],timeout=0.1)
        time.sleep(2)
        thread=threading.Thread(target=read_power,daemon=True); thread.start()
        port.write(b'START\n'); port.flush()
        deadline=time.perf_counter()+3
        while not latest and time.perf_counter()<deadline: time.sleep(.01)
        if not latest: raise RuntimeError('No power samples')
        if purpose=='primary' and cfg.get('power_timestamp_mode','fresh_device')=='fresh_device' and (
                                    meta.get('power_firmware')!=cfg.get('required_power_firmware') or
                                    meta.get('power_timestamp_source')!='fresh_update_observed_ms'):
            raise RuntimeError('Power firmware freshness/timestamp contract not confirmed')
        if execute:
            # Refuse a jump from an unknown pose. No automatic repositioning.
            for j,value in zip(cfg['joints'],recipe['center_rad']):
                ok,pos=rcb.getSinglePos(j['id'],j['sio'])
                angle=(pos-j['zero_pos'])*math.radians(135)/4000*j['sign']
                if not ok or not 3500<=pos<=11500 or abs(angle-value)>cfg['max_initial_error_rad']:
                    raise RuntimeError(f'Initial pose mismatch: {j["name"]}')
            old=rcb.getConfig()
            if old==65535: raise RuntimeError('Cannot read controller configuration')
            new=(old & ~2)|1
            if not rcb.moveComToRamCmdSynchronize(Rcb4BaseLib.RamAddr.ConfigRamAddress.value,[new&255,new>>8]):
                raise RuntimeError('Cannot establish PC control')
            owned=True
            meta['original_config']=old
        start=time.perf_counter()
        score=start+recipe['pre_s']
        finish=score+recipe['duration_s']+recipe['settle_s']
        meta.update(score_start=score,score_end=finish)
        n=0; nextad=start; nextfb=start; joint_index=0
        nextfocus={joint:start for joint in focused_joints}
        while time.perf_counter()<finish+recipe['post_s']:
            now=time.perf_counter()
            if errors: raise RuntimeError(errors[-1])
            if not latest or now-latest[0]>cfg['power_stale_s']: raise RuntimeError('Power stream stale')
            if execute and n<len(trajectory) and now>=score+trajectory[n]['t_s']:
                row=trajectory[n]
                if now-(score+row['t_s'])>1/cfg['command_hz']:
                    raise RuntimeError('Command deadline missed')
                positions=[round(j['zero_pos']+j['sign']*row[f'cmd_{k}']*4000/math.radians(135)) for k,j in enumerate(cfg['joints'])]
                data=[Rcb4BaseLib.ServoData(j['id'],j['sio'],p) for j,p in zip(cfg['joints'],positions)]
                before=time.perf_counter(); ok=rcb.setServoPos(data,1); after=time.perf_counter()
                cmdlog(dict(pc_s=before,response_s=after,ok=int(bool(ok)),
                    **{f'cmd_{k}':(positions[k]-j['zero_pos'])*math.radians(135)/4000*j['sign']
                       for k,j in enumerate(cfg['joints'])}))
                if not ok: raise RuntimeError('Command acknowledgement failed')
                if n==0:
                    score=after
                    finish=score+recipe['duration_s']+recipe['settle_s']
                    meta.update(score_start=score,score_end=finish)
                if n==len(trajectory)-1:
                    meta['last_command_ack_s']=after
                    if recipe['settle_s']>0:
                        finish=after+recipe['settle_s']
                        meta['score_end']=finish
                n+=1
            if now>=nextad:
                a=time.perf_counter()
                ok,buf=rcb.moveRamToComCmdSynchronize(rcb.adDataAddr(1),8)
                b=time.perf_counter(); valid=bool(ok and len(buf)==8)
                vals=[buf[k]+256*buf[k+1] for k in range(0,8,2)] if valid else ['']*4
                # Preserve exact RAM values; signed gyro interpretation remains offline.
                adlog(dict(pc_s=(a+b)/2,request_s=a,response_s=b,ok=int(valid),
                           **{f'ad{k+1}':v for k,v in enumerate(vals)}))
                nextad += (max(0,math.floor((b-nextad)*cfg['ad_hz']))+1)/cfg['ad_hz']
            if cfg['feedback'] and now>=nextfb:
                j=cfg['joints'][joint_index]; a=time.perf_counter()
                ok,pos=rcb.getSinglePos(j['id'],j['sio']); b=time.perf_counter()
                valid=bool(ok and 3500<=pos<=11500 and j['sign'] in (-1,1))
                value=(pos-j['zero_pos'])*math.radians(135)/4000*j['sign'] if valid else ''
                fblog(dict(pc_s=(a+b)/2,request_s=a,response_s=b,joint=joint_index,ok=int(valid),q_rad=value))
                joint_index=(joint_index+1)%17
                nextfb += (max(0,math.floor((b-nextfb)*17*cfg['feedback_hz']))+1)/(17*cfg['feedback_hz'])
            if cfg['feedback']:
                for focused_joint in focused_joints:
                    if now < nextfocus[focused_joint]:
                        continue
                    j=cfg['joints'][focused_joint]; a=time.perf_counter()
                    ok,pos=rcb.getSinglePos(j['id'],j['sio']); b=time.perf_counter()
                    valid=bool(ok and 3500<=pos<=11500 and j['sign'] in (-1,1))
                    value=(pos-j['zero_pos'])*math.radians(135)/4000*j['sign'] if valid else ''
                    fblog(dict(pc_s=(a+b)/2,request_s=a,response_s=b,joint=focused_joint,
                               ok=int(valid),q_rad=value))
                    due=nextfocus[focused_joint]
                    nextfocus[focused_joint] += (max(0,math.floor((b-due)*focused_hz))+1)/focused_hz
            time.sleep(.001)
        if execute and n!=len(trajectory): raise RuntimeError('Incomplete command trajectory')
        if errors: raise RuntimeError(errors[-1])
        meta['status']='complete'
    except BaseException as e:
        meta['error']=f'{type(e).__name__}: {e}'
        raise
    finally:
        # Do not resume an old EEPROM program or issue FREE on cleanup.
        # Last acknowledged position remains commanded; physical emergency stop remains available.
        meta['pc_control_retained']=owned
        if port:
            try: port.write(b'STOP\n'); port.flush(); time.sleep(.2)
            except Exception as e: meta['stop_error']=str(e)
        stop.set()
        if thread: thread.join(timeout=2)
        if port: port.close()
        rcb.close()
        for f in handles: f.close()
        save(out/'meta.json',meta)
        if meta['status']=='complete':
            save(out/'acquisition_qc.json',audit_raw(out))
    if purpose=='primary':
        from processing import prepare
        try:
            prepare(out,out/'config.json',out/'processed')
        except Exception as exc:
            save(out/'analysis_failure.json',{'usable':False,'error':str(exc)})
            raise
