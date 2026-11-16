"""Paper-specified quantities plus explicit reconstructions of missing recipes.
No MuJoCo outcome is used to choose parameters or trajectories.
"""
from pathlib import Path
import csv, json, math
import numpy as np
from gait import Gait
ROOT=Path(__file__).resolve().parents[1]
MAPPING=json.loads((ROOT/'src/joint_mapping.json').read_text())['joints']
# Adjacent edges on 2 five-joint legs and 2 three-joint arms + bilateral limbs.
PAIRS=[(i,i+1) for i in [0,1,2,3,5,6,7,8,10,11,13,14]]+[(i,i+5) for i in range(5)]+[(i,i+3) for i in range(10,13)]
CFG=json.loads((ROOT/'src/gait_config.json').read_text())

def pose(context):
 q=Gait(CFG).neutral()
 if context=='crouching':
  for i in [0,5]:q[i+1:i+4]=[-.65,1.3,-.65]
 elif context=='wide': q[[0,4,5,9]]=[.18,-.18,-.18,.18]
 elif context=='lean': q[[1,6]]-=.15
 elif context=='single': q[6:9]=[-.8,1.35,-.55]
 elif context in ['lie_supine','lie_prone','hand','free']:
  q[:]=0
  if context=='hand': q[[10,13]]=-1.1;q[[12,15]]=.8
 return q

def specs():
 result=[]
 for row in csv.DictReader((ROOT/'reference/all_trial_energies.csv').open()):
  split=row['split']; idx=row['trial_id']; test=split in ['test','control']
  seed=int(idx.split('_')[-1]) if test else int(idx)+(410 if split=='train' else 200410)
  rng=np.random.default_rng(seed); level=None if test else int(row['action'][1:])
  if level==0: joints=[]
  elif level==1:joints=[int(rng.integers(17))]
  elif level==2:joints=list(PAIRS[int(rng.integers(20))])
  elif level in [3,4]:joints=sorted(rng.choice(17,size=2,replace=False).tolist())
  elif level==5:joints=sorted(rng.choice(17,size=int(rng.choice([4,8,12,17])),replace=False).tolist())
  else:joints=[]
  result.append(dict(split=split,id=idx,action=row['action'],context=row['context'],duration=float(row['duration_s']),seed=seed,level=level,joints=joints,amplitude=float(rng.choice([.2,.35])),frequency=float(rng.choice([.3,.6,1.])),offset=float(rng.choice([0,.15])),phase=float(rng.choice([0,math.pi/2,math.pi]))))
 return result

def initial_context(s):
 if s['action']=='get_up':return 'lie_supine'
 return 'standing' if s['context']=='whole-body' else s['context']

def command(s,t):
 context=initial_context(s);q=pose(context)
 if s['level'] is not None:
  for k,i in enumerate(s['joints']):q[i]+=s['offset']+s['amplitude']*math.sin(2*math.pi*s['frequency']*t+k*s['phase'])
 else:
  a=s['action'];g=Gait(CFG)
  if a in ['walk','stop','turn_90','turn_180','push_box']:
   amplitude=min(1.,t/1.5)
   if a=='stop':amplitude*=max(0.,min(1.,(s['duration']-t)/2))
   q=g.pose(2*math.pi*t/CFG['nominal'][0],amplitude)
   if a.startswith('turn'):
    # Reconstructed asymmetric gait, no imposed root rotation or propulsion.
    q[1]+=.06*math.sin(2*math.pi*t/CFG['nominal'][0]);q[6]-=.06*math.sin(2*math.pi*t/CFG['nominal'][0])
   if a=='push_box':q[[10,13]]=-1.15;q[[12,15]]=.25
  elif a in ['get_up','lie_down']:
   # Reconstructed joint-only sequence; mechanical completion is measured,
   # never implied by the behavior label.
   u=t/s['duration'] if a=='get_up' else 1-t/s['duration']
   frames=[np.zeros(17),np.array([0,-1.1,1.4,-.5,0]*2+[-1.2,.2,.8,-1.2,-.2,.8,0]),pose('crouching'),pose('standing')]
   x=np.clip(u,0,1)*3;j=min(2,int(x));f=x-j;f=f*f*(3-2*f);q=(1-f)*frames[j]+f*frames[j+1]
 return np.clip(q,-1.4,1.4)

def features(d):
 tau=d['tau'];v=d['v'];h=d['h'];L=np.sum(tau*tau,axis=1);c=d['contact'];single=((c[:,4]>.05)^(c[:,5]>.05)).astype(float);both=((c[:,4]>.05)&(c[:,5]>.05)).astype(float)
 x1=np.column_stack([np.ones(len(v)),np.abs(tau*v),tau*tau,np.abs(v)])
 x2=np.column_stack([x1]+[np.abs(tau[:,i]*tau[:,j]) for i,j in PAIRS])
 x3=np.column_stack([x2,h,h*h,h*L,d['com_z'],np.sum(d['root_omega']**2,axis=1)])
 x4=np.column_stack([x3,c[:,:4],both,single,single*L]);assert x4.shape[1]==84
 return [x1,x2,x3,x4]

def coefficients():
 rng=np.random.default_rng(20260913)
 return dict(P0=2.2,a=rng.uniform(1.4,2.2,17).tolist(),b=rng.uniform(1,2,17).tolist(),c=rng.uniform(.035,.08,17).tolist(),d=rng.uniform(.09,.16,17).tolist(),k=rng.uniform(.15,.65,20).tolist(),rng='numpy.default_rng (original RNG/order unavailable)')

def oracle(d):
 z=coefficients();tau=d['tau'];v=d['v'];h=d['h'];L=np.sum(tau*tau,axis=1);c=d['contact'];single=((c[:,4]>.05)^(c[:,5]>.05))
 local=z['P0']+np.sum(np.abs(tau*v)*z['a']+tau*tau*z['b']+np.abs(v)*z['c']+np.array(z['d']),axis=1)
 pair=sum(z['k'][k]*np.abs(tau[:,i]*tau[:,j]) for k,(i,j) in enumerate(PAIRS))
 body=1.8*h*h+.35*h*L
 contact=c[:,0]*.025+c[:,1]*.035+c[:,2]*.075+c[:,3]*.05+.45*single*L
 T=np.zeros(len(L))
 for i in range(1,len(T)):T[i]=T[i-1]+.01*(L[i-1]-T[i-1])/25
 residual=.025*np.sum(np.abs(tau*v),axis=1)**1.3+.06*T
 return local+pair+body+contact+residual,np.column_stack([local,pair,body,contact,residual])
