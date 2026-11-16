"""Bounded physical knee-direction check, independent of simulator coordinates."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time

from common import parse_power, read, save


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--supported-initial-pose', action='store_true',
                    help='Verify the six pitch joints at the simulator start pose; hold endpoint')
    ap.add_argument('--roll-pair-check', action='store_true',
                    help='Supported +2 degree candidate hip roll, -2 degree ankle roll; return')
    args = ap.parse_args()
    if args.supported_initial_pose and args.roll_pair_check:
        raise ValueError('Select only one supported probe')
    if args.supported_initial_pose:
        motion=read('data/motions/R1_lambda0.3_seed10.json')
        cfg0=read('hardware.pending.json');joints={j['name']:j for j in cfg0['joints']}
        # Candidate anatomical signs inferred from archived real squat positions.
        # This supported probe does NOT mark simulator mapping as verified.
        signs={'left_hip_pitch':-1,'left_knee_pitch':1,'left_ankle_pitch':1,
               'right_hip_pitch':1,'right_knee_pitch':-1,'right_ankle_pitch':-1}
        angles=dict(zip(motion['joint_names'],motion['initial_pose_rad']))
        targets=[dict(name=n,id=joints[n]['id'],sio=joints[n]['sio'],
                      absolute_position=round(7500+s*angles[n]*joints[n]['units_per_rad']))
                 for n,s in signs.items()]
        if any(abs(t['absolute_position']-7500)>1100 for t in targets):
            raise ValueError('Supported start probe exceeds fixed 1100-count bound')
    elif args.roll_pair_check:
        targets=[{'name':'left_hip_roll','id':6,'sio':1,'delta_counts':59},
                 {'name':'right_hip_roll','id':6,'sio':2,'delta_counts':59},
                 {'name':'left_ankle_roll','id':10,'sio':1,'delta_counts':-59},
                 {'name':'right_ankle_roll','id':10,'sio':2,'delta_counts':-59}]
    else:
        targets=[{'id':8,'sio':1,'delta_counts':148}, {'id':8,'sio':2,'delta_counts':-148}]
    if not args.execute:
        print('Dry run; no ports opened. Targets:',json.dumps(targets))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parent/'vendor'))
    from Rcb4BaseLib import Rcb4BaseLib
    import serial
    cfg = read('hardware.pending.json')
    purpose=('supported six-pitch-joint start pose check' if args.supported_initial_pose else
             'supported candidate roll direction check' if args.roll_pair_check else 'supported physical knee-direction check')
    meta = dict(status='failed', purpose=purpose+', not a walking trial',
                created_utc=datetime.now(timezone.utc).isoformat(),
                targets=targets,
                basis=('Provisional roll signs; visual observation required, no verification claim' if args.roll_pair_check else
                       'Opposite raw knee signs in archived physical squat pose_20260915_065552_post_trim_squat_target.json'),
                visual_confirmation='pending')
    stop=threading.Event(); latest=[]; errors=[]; rcb=Rcb4BaseLib(); port=None; thread=None
    pf=(args.out/'power.csv').open('x',newline='',encoding='utf-8')
    cf=(args.out/'commands.csv').open('x',newline='',encoding='utf-8')
    pw=csv.DictWriter(pf,fieldnames=['pc_s','device_s','voltage_raw','current_raw','clip']);pw.writeheader()
    cw=csv.writer(cf);cw.writerow(['index','scheduled_s','response_s','raw_targets_json','ok'])
    def reader():
        previous=None
        try:
            while not stop.is_set():
                line=port.readline().decode('utf-8',errors='replace').strip()
                row=parse_power(line)
                if row is None:continue
                if previous is not None and row['device_s']<=previous:raise RuntimeError('Power clock not increasing')
                previous=row['device_s'];row['pc_s']=time.perf_counter()
                pw.writerow(row);pf.flush();latest[:]=[row]
                if row['clip'] or abs(row['current_raw'])>cfg['max_current_A']:raise RuntimeError('Current limit or clipping')
                if row['voltage_raw']<cfg['min_voltage_V']:raise RuntimeError('Low voltage')
        except Exception as e:errors.append(str(e))
    def check():
        if errors:raise RuntimeError(errors[0])
        if not latest or time.perf_counter()-latest[0]['pc_s']>.5:raise RuntimeError('Power stream stale')
    try:
        if not rcb.open(cfg['robot_port'],cfg['robot_baud'],.1) or not rcb.checkAcknowledge():raise RuntimeError('RCB connection failed')
        positions=[]
        for target in targets:
            ok,pos=rcb.getSinglePos(target['id'],target['sio'])
            if not ok or abs(pos-7500)>100:raise RuntimeError('Target joint not near neutral; no motion')
            positions.append(int(pos))
        endpoints=[t.get('absolute_position',p+t.get('delta_counts',0)) for t,p in zip(targets,positions)]
        meta['start_positions']=positions
        port=serial.Serial(cfg['power_port'],cfg['power_baud'],timeout=.1)
        time.sleep(2);port.reset_input_buffer();thread=threading.Thread(target=reader,daemon=True);thread.start()
        port.write(b'START\n');port.flush()
        timeout=time.perf_counter()+5
        while not latest and not errors and time.perf_counter()<timeout:time.sleep(.01)
        check()
        original=rcb.getConfig()
        if original==65535:raise RuntimeError('RCB config unavailable')
        new=(original & ~2)|1
        if not rcb.moveComToRamCmdSynchronize(Rcb4BaseLib.RamAddr.ConfigRamAddress.value,[new&255,new>>8]):raise RuntimeError('PC control failed')
        meta['original_rcb_config']=original
        begin=time.perf_counter()+.5
        for k in range(81):
            t=k*.1
            if args.supported_initial_pose:u=t/8
            elif t<3:u=t/3
            elif t<5:u=1.
            else:u=max(0.,(8-t)/3)
            blend=u*u*(3-2*u)
            values=[p+round((end-p)*blend) for p,end in zip(positions,endpoints)]
            due=begin+t
            while time.perf_counter()<due:
                check();time.sleep(min(.002,max(0,due-time.perf_counter())))
            check()
            if time.perf_counter()-due>.05:raise RuntimeError('Late dispatch; no catch-up')
            ok=rcb.setServoPos([Rcb4BaseLib.ServoData(tg['id'],tg['sio'],v) for tg,v in zip(targets,values)],1)
            now=time.perf_counter();cw.writerow([k,due,now,json.dumps(values),int(bool(ok))]);cf.flush()
            if not ok or now-due>.05:raise RuntimeError('Command ACK failure or lateness')
        meta['status']='complete';meta['returned_to_start_targets']=not args.supported_initial_pose
        meta['endpoint_behavior']='hold supported initial targets' if args.supported_initial_pose else 'returned to original targets'
    except BaseException as e:
        meta['error']=f'{type(e).__name__}: {e}'
        raise
    finally:
        if port:
            try:port.write(b'STOP\n');port.flush()
            except Exception:pass
        stop.set()
        if thread:thread.join(timeout=2)
        if port:port.close()
        rcb.close();pf.close();cf.close()
        meta['power_reader_errors']=errors
        if errors:meta['status']='failed'
        save(args.out/'meta.json',meta);print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
