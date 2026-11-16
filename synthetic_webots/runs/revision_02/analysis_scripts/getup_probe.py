"""Development-only motor sequences; frozen benchmark and physics are untouched."""
from pathlib import Path
import json,time,sys,numpy as np
from webots_session import Session,development_spec,mechanics
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/Users/yuchenzhou/Desktop/ヒューマノイド研究/02_実験コード・シミュレータ/023KHR3HVSimulator金曜/webots_official_khr3hv/data/verified_full_motion.json')
ORIGINAL=json.loads(SOURCE.read_text())
NAMES=['LeftFemurHead2','LeftFemur','LeftCrus','LeftAnkle','LeftFoot','RightFemurHead2','RightFemur','RightCrus','RightAnkle','RightFoot','LeftShoulder','LeftArm','LeftForearm','RightShoulder','RightArm','RightForearm','Head']
SIGNS=np.array([1,1,1,1,1,-1,1,1,1,-1,1,1,1,1,-1,1,1])
def q(p):return (np.array([p.get(n,0) for n in NAMES])*SIGNS).tolist()
def sequence(stages,initial=None,scale=1.,hold=2.):
 fs=[[0.,q(initial or {})]];t=0
 for f in stages:
  t+=f['duration']*scale;fs.append([t,q(f['pose'])])
 fs.append([t+hold,fs[-1][1]])
 return fs
ROLL=[f for f in ORIGINAL['frames'] if f['stage']=='right_roll']
UP=[f for f in ORIGINAL['frames'] if f['stage']=='get_up']
DOWN=[f for f in ORIGINAL['frames'] if f['stage']=='forward_roll'][:3]
STAND=UP[-1]['pose']
def metrics(d,kind):
 m=mechanics(d);v=d['root_velocity'][-100:];tilt=np.degrees(np.arccos(np.clip(1-d['h'][-100:],-1,1)));z=d['position'][-100:,2]
 m.update(tail_mean_linear_speed=float(np.linalg.norm(v[:,:3],axis=1).mean()),tail_mean_angular_speed=float(np.linalg.norm(v[:,3:],axis=1).mean()),tail_both_support_fraction=float(((d['contact'][-100:,4]>.05)&(d['contact'][-100:,5]>.05)).mean()))
 stable=m['tail_mean_linear_speed']<.02 and m['tail_mean_angular_speed']<.1
 m['passed']=bool(stable and ((z.min()>=.28 and tilt.max()<=20 and m['tail_both_support_fraction']>=.95) if kind=='up' else (z.max()<.18 and d['h'][-100:].min()>.7)))
 return m

def main():
 out=ROOT/'development'/('getup_probe_'+time.strftime('%Y%m%d_%H%M%S'));records=[]
 cases=[('prone_up','get_up_prone','lie_prone',sequence(UP),'up'),('supine_roll_up','get_up','whole-body',sequence(ROLL+UP),'up'),('stand_to_prone','lie_down','whole-body',sequence(DOWN,STAND,hold=4.),'down')]
 with Session(out) as session:
  (out/'development_design.json').write_text(json.dumps(dict(source=str(SOURCE),changes=['Native motor poses converted by joint axis; right pitch axis reversals accounted for','Keyframes may exceed legacy ±1.4 command clip, bounded by unchanged native ±2.356 limits','Original physics/selfCollision/PID/torque/friction preserved','Development seeds only; no power metrics used'],cases=[dict(name=n,action=a,context=c,frames=f,kind=k) for n,a,c,f,k in cases]),indent=2))
  for name,action,context,frames,kind in cases:
   spec=development_spec(action,seed=91001,duration=frames[-1][0],frames=frames);spec['context']=context
   result,d,path=session.run(spec,name+'_91001');m=metrics(d,kind);row=dict(name=name,spec=spec,metrics=m,result=result,path=str(path));records.append(row)
   print(name,json.dumps(m),flush=True);(out/'summary.json').write_text(json.dumps(records,indent=2))
 print('OUTPUT',out,flush=True)
if __name__=='__main__':main()
