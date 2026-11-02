from pathlib import Path
import os,sys,socket,subprocess,json,time,argparse,fcntl,hashlib,traceback
from protocol import ROOT,specs,coefficients
PYTHON='/Users/yuchenzhou/.cache/khr3hv-rl/runtime-20261008/bin/python'
WEBOTS='/Applications/Webots.app/Contents/MacOS/webots'
def atomic(p,d):
 p=Path(p);p.with_suffix('.tmp').write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False));p.with_suffix('.tmp').replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',default='full');ap.add_argument('--smoke',action='store_true');ap.add_argument('--gui',action='store_true');args=ap.parse_args()
 run=ROOT/'runs'/args.run;run.mkdir(exist_ok=True)
 lock=(run/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 trials=specs()
 if args.smoke:
  selected=[]
  for key in [('L0','standing'),('L0','lie_supine'),('L3','crouching'),('L4','hand'),('L4','free'),('walk','whole-body'),('get_up','whole-body'),('push_box','whole-body')]:selected.append(next(x for x in trials if (x['action'],x['context'])==key))
  trials=selected
 design=dict(trials=trials,oracle=coefficients(),source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','controllers','plugins','worlds','reference'] for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.c','.wbt','.ini','.json','.csv','.tex']},exact_mujoco_replication=False,created=time.time())
 if (run/'design.json').exists():
  old=json.loads((run/'design.json').read_text())
  assert old['trials']==design['trials'] and old['source_hashes']==design['source_hashes'], 'Frozen design changed; use separate run'
 else: atomic(run/'design.json',design)
 listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(90)
 env=dict(os.environ,BENCH_PORT=str(listener.getsockname()[1]),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PATH=str(Path(PYTHON).parent)+os.pathsep+os.environ['PATH'])
 port=socket.socket();port.bind(('127.0.0.1',0));webport=port.getsockname()[1];port.close()
 cmd=[WEBOTS,'--batch','--stdout','--stderr',f'--port={webport}','--mode=fast']+([] if args.gui else ['--no-rendering','--minimize'])+[str(ROOT/'worlds/benchmark.wbt')]
 log=(run/'console.log').open('a');p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 caffeinate=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
 atomic(run/'launch.json',dict(pid=os.getpid(),webots_pid=p.pid,command=cmd,started=time.time()))
 stream=None;conn=None
 try:
  conn,_=listener.accept();conn.settimeout(300);stream=conn.makefile('rwb');hello=json.loads(stream.readline());atomic(run/'inventory.json',hello)
  for i,s in enumerate(trials):
   key=s['split']+'_'+s['id'];path=run/'raw'/f'{key}.npz';status=run/'status'/f'{key}.json';status.parent.mkdir(exist_ok=True)
   if path.exists():
    if not status.exists():raise RuntimeError('Raw exists without status; audit before recovery: '+key)
    continue
   atomic(run/'progress.json',dict(completed=i,total=len(trials),current=key,updated=time.time()))
   stream.write((json.dumps(dict(command='run',spec=s,path=str(path),snapshot=args.gui))+'\n').encode());stream.flush();reply=json.loads(stream.readline())
   if 'error'in reply:raise RuntimeError(reply['error'])
   atomic(status,dict(spec=s,**reply['result'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),completed=time.time()))
   print(f'{i+1}/{len(trials)} {key} h={reply["result"]["final_h"]:.3f}',flush=True)
  atomic(run/'completion.json',dict(completed=len(trials),finished=time.time()))
 except Exception:
  atomic(run/'error.json',dict(error=traceback.format_exc(),time=time.time()));raise
 finally:
  if stream:
   try:stream.write(b'{"command":"close"}\n');stream.flush()
   except OSError:pass
   stream.close()
  if conn:conn.close()
  listener.close()
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=10)
  log.close();caffeinate.terminate()
if __name__=='__main__':main()
