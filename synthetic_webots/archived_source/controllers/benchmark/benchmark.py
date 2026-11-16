from pathlib import Path
import os,sys,json,socket,struct,traceback,math,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from controller import Supervisor
from protocol import MAPPING,initial_context,pose,command
r=Supervisor();body=r.getFromDef('KHR');obj=r.getFromDef('OBJECT');world=r.getRoot().getField('children').getMFNode(0)
receiver=r.getDevice('contacts');receiver.enable(2)
names=[v['motor'] for v in MAPPING];sign=np.array([v['webots_sign'] for v in MAPPING]);motors=[r.getDevice(v) for v in names];sensors=[r.getDevice(v+'_sensor') for v in names];joints=[r.getFromDef('KHR3HV_'+v) for v in names]
for m,s in zip(motors,sensors):
 s.enable(2);m.enableTorqueFeedback(2);m.setVelocity(7.48);m.setAvailableTorque(min(1.373,m.getMaxTorque()));m.setControlPID(10,0,0)
body.saveState('initial');obj.saveState('initial')
def target(q):
 for m,x in zip(motors,q*sign):m.setPosition(float(np.clip(x,m.getMinPosition(),m.getMaxPosition())))
def advance():
 if r.step(10)==-1:raise RuntimeError('Webots stopped')
 vals=[]
 while receiver.getQueueLength():
  data=receiver.getBytes()
  if len(data)==56:vals.append(struct.unpack('7d',data))
  receiver.nextPacket()
 if not vals:raise RuntimeError('Contact plugin produced no packet')
 return np.array(vals[-1])
def run(s,path,snapshot=False):
 context=initial_context(s);body.loadState('initial');obj.loadState('initial')
 world.getField('gravity').setSFFloat(0.0 if context=='free' else 9.81)
 rotation=[0,1,0,0];position=[0,0,.34]
 if context=='lie_supine':rotation=[0,1,0,-math.pi/2];position=[0,0,.10]
 if context in ['lie_prone','hand']:rotation=[0,1,0,math.pi/2];position=[0,0,.10 if context=='lie_prone' else .17]
 if context=='free':position=[0,0,1.0]
 body.getField('translation').setSFVec3f(position);body.getField('rotation').setSFRotation(rotation)
 if context=='wall':obj.getField('translation').setSFVec3f([.12,0,.08])
 elif s['action']=='push_box':obj.getField('translation').setSFVec3f([.14,0,.08])
 q=command(s,0)
 if s['split'] in ['test','control']:q=q+np.random.default_rng(s['seed']).normal(0,.001,17)
 for j,x in zip(joints,q*sign):j.setJointPosition(float(x))
 body.resetPhysics();obj.resetPhysics();target(q)
 for _ in range(100):advance()
 assert abs(world.getField('gravity').getSFFloat()-(0.0 if context=='free' else 9.81))<1e-9, 'Gravity reset failed'
 t0=r.getTime();q0=np.array([x.getValue() for x in sensors])*sign
 origin=np.array(body.getPosition());obj0=np.array(obj.getPosition());records=[]
 for k in range(round(s['duration']*100)):
  target(command(s,k*.01));c=advance();q=np.array([x.getValue() for x in sensors])*sign
  tau=np.array([m.getTorqueFeedback() for m in motors])*sign
  R=np.array(body.getOrientation()).reshape(3,3);p=np.array(body.getPosition());rv=np.array(body.getVelocity());com=body.getCenterOfMass()
  records.append(np.r_[r.getTime()-t0,q,(q-q0)/.01,tau,1-R[2,2],com[2],rv[3:],p,c,np.arctan2(R[1,0],R[0,0]),np.array(obj.getPosition())]);q0=q
  if snapshot and k in [0,round(s['duration']*50),round(s['duration']*100)-1]:r.exportImage(str(Path(path).with_suffix(''))+f'_{k}.png',95)
 a=np.array(records)
 if not np.isfinite(a).all():raise RuntimeError('Nonfinite sensor data')
 d=dict(t=a[:,0],q=a[:,1:18],v=a[:,18:35],tau=a[:,35:52],h=a[:,52],com_z=a[:,53],root_omega=a[:,54:57],position=a[:,57:60],contact=a[:,60:67],yaw=a[:,67],object_position=a[:,68:71],origin=origin,object_origin=obj0,spec_json=json.dumps(s))
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():raise RuntimeError('Refuse to overwrite '+str(path))
 with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**d)
 path.with_suffix('.tmp').replace(path)
 return dict(samples=len(a),duration=s['duration'],final_height=float(a[-1,59]),final_h=float(a[-1,52]),forward_m=float(a[-1,57]-origin[0]),yaw_change=float(np.unwrap(a[:,67])[-1]-a[0,67]),object_displacement=float(np.linalg.norm(a[-1,68:71]-obj0)),max_tau=float(abs(a[:,35:52]).max()),contact_mean=a[:,60:67].mean(axis=0).tolist())
conn=socket.create_connection(('127.0.0.1',int(os.environ['BENCH_PORT'])));stream=conn.makefile('rwb')
def send(d):stream.write((json.dumps(d,allow_nan=False)+'\n').encode());stream.flush()
send({'ready':True,'python':sys.executable,'world_robot_plus_object_mass_kg':sum(float(x) for x in __import__('re').findall(r'\bmass ([0-9.]+)',(ROOT/'worlds/benchmark.wbt').read_text()))})
try:
 for line in stream:
  req=json.loads(line)
  if req['command']=='close':break
  if req['command']=='run':
   try:send({'ok':True,'result':run(req['spec'],req['path'],req.get('snapshot',False))})
   except Exception:send({'error':traceback.format_exc()})
finally:
 stream.close();conn.close();r.simulationQuit(0)
