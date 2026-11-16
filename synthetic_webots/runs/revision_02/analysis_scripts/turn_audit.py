"""Read-only kinematic development audit; never reads power/model outcomes."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
run=root/'development'/sys.argv[1]
rows=json.loads((run/'summary.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for row in rows:
    with np.load(row['raw'],allow_pickle=False) as data:
        t=data['t']; q=data['qcmd']; y=np.degrees(np.unwrap(data['yaw']));y-=y[0]
        moving=np.flatnonzero(np.max(np.abs(q-q[-1]),axis=1)>1e-5)
        row['command_motion_end_s']=float(t[moving[-1]]) if len(moving) else 0.
        row['max_tilt_deg']=float(np.degrees(np.arccos(np.clip(1-data['h'],-1,1))).max())
        row['net_translation_xy_m']=float(np.linalg.norm(data['position'][-1,:2]-data['position'][0,:2]))
        row['max_nonfoot_force_N']=float(data['contact'][:,2].max())
        row['max_abs_command_rad']=float(np.abs(q).max())
        row['raw_sha256']=hashlib.sha256(Path(row['raw']).read_bytes()).hexdigest()
        ax=axes[0 if row['target_deg']==90 else 1]
        ax.plot(t,y,lw=1,label=str(row['seed']))
for ax,target in zip(axes,[90,180]):
    ax.axhspan(target-10,target+10,color='#b8b8b8',alpha=.25)
    ax.axhline(target,color='black',ls='--',lw=.8)
    ax.set(xlabel='Time (s)',ylabel='Heading change (deg)',title=f'Target {target}°')
    ax.spines[['top','right']].set_visible(False)
fig.savefig(run/'development_heading.png',dpi=180)
fig.savefig(run/'development_heading.pdf')
(run/'kinematic_audit.json').write_text(json.dumps(rows,indent=2))
print(json.dumps({'n':len(rows),'passed':sum(r['acceptance_pass'] for r in rows),'by_target':{target:{'n':len([r for r in rows if r['target_deg']==target]),'yaw_deg':[min(r['yaw_deg'] for r in rows if r['target_deg']==target),max(r['yaw_deg'] for r in rows if r['target_deg']==target)],'max_command_end_s':max(r['command_motion_end_s'] for r in rows if r['target_deg']==target),'max_net_translation_xy_m':max(r['net_translation_xy_m'] for r in rows if r['target_deg']==target),'max_tilt_deg':max(r['max_tilt_deg'] for r in rows if r['target_deg']==target)} for target in [90,180]}},indent=2))
