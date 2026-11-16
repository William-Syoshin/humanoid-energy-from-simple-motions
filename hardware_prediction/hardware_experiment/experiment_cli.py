"""Hardware experiment CLI. record is read-only unless --execute is supplied."""
import argparse
from pathlib import Path
from core import initialize, read, save, csv_save, compile_recipe, linear_calibration, digest, sensor_summary, register, repeatability
from processing import prepare
from protocol import preflight, audit_session, freeze_design

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('init'); q.add_argument('session')
    for name in ('preview','record'):
        q=sub.add_parser(name)
        q.add_argument('--config',required=True); q.add_argument('--recipe',required=True)
        q.add_argument('--out',required=True)
        if name=='record':
            q.add_argument('--execute',action='store_true')
            q.add_argument('--purpose',choices=['primary','diagnostic'],default='primary')
            q.add_argument('--design')
    q=sub.add_parser('preflight'); q.add_argument('--config',required=True); q.add_argument('--recipe',required=True)
    q=sub.add_parser('audit'); q.add_argument('session'); q.add_argument('--out',required=True)
    q=sub.add_parser('calibrate-timing'); q.add_argument('--points',required=True); q.add_argument('--out',required=True)
    q.add_argument('--reference-uncertainty-s',type=float,required=True)
    q=sub.add_parser('calibrate'); q.add_argument('--points',required=True); q.add_argument('--out',required=True)
    q=sub.add_parser('prepare'); q.add_argument('--raw',required=True); q.add_argument('--config',required=True); q.add_argument('--out',required=True)
    q=sub.add_parser('freeze'); q.add_argument('session'); q.add_argument('--out',required=True)
    q=sub.add_parser('sensor-summary'); q.add_argument('--raw',required=True); q.add_argument('--out',required=True)
    q=sub.add_parser('register'); q.add_argument('session'); q.add_argument('--raw',required=True)
    q.add_argument('--processed',required=True); q.add_argument('--success',type=int,choices=[0,1],required=True)
    q.add_argument('--reason',default=''); q.add_argument('--out-manifest',required=True)
    q.add_argument('--manifest')
    q=sub.add_parser('repeatability'); q.add_argument('processed',nargs='+'); q.add_argument('--out',required=True)
    q=sub.add_parser('analyze'); q.add_argument('mode',choices=['fit','evaluate','learning-curve'])
    q.add_argument('--manifest',required=True); q.add_argument('--data',required=True)
    q.add_argument('--out',required=True); q.add_argument('--models')
    args=p.parse_args()
    if args.command=='init': initialize(args.session)
    elif args.command=='preview': csv_save(args.out,compile_recipe(read(args.config),read(args.recipe)))
    elif args.command=='record':
        from acquire import record
        record(args.config,args.recipe,args.out,args.execute,args.purpose,args.design)
    elif args.command=='preflight':
        import json
        result=preflight(read(args.config),read(args.recipe),True)
        print(json.dumps(result,indent=2,ensure_ascii=False))
        if not result['ready']: raise SystemExit(2)
    elif args.command=='audit': audit_session(args.session,args.out)
    elif args.command=='calibrate-timing':
        from calibration import timing_calibration
        timing_calibration(args.points,args.out,args.reference_uncertainty_s)
    elif args.command=='calibrate': linear_calibration(args.points,args.out)
    elif args.command=='prepare': prepare(args.raw,args.config,args.out)
    elif args.command=='sensor-summary': sensor_summary(args.raw,args.out)
    elif args.command=='register': register(args.session,args.raw,args.processed,args.success,args.reason,args.out_manifest,args.manifest)
    elif args.command=='repeatability': repeatability(args.processed,args.out)
    elif args.command=='analyze':
        import subprocess,sys
        command=[sys.executable,str(Path(__file__).with_name('analysis.py')),args.mode,'--manifest',args.manifest,'--data',args.data,'--out',args.out]
        if args.models: command+=['--models',args.models]
        subprocess.run(command,check=True)
    elif args.command=='freeze':
        freeze_design(args.session,args.out)

if __name__=='__main__': main()
