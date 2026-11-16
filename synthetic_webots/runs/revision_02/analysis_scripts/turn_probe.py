from pathlib import Path
import json, sys, hashlib, shutil
from webots_session import Session, development_spec, mechanics

root=Path(__file__).resolve().parents[1]
out=root/'development'/sys.argv[1]
candidates=json.loads(Path(sys.argv[2]).read_text())
summary=[]
with Session(out) as session:
    src=root/'project/src/turn_revision.py'
    shutil.copy2(src,out/'turn_revision.py')
    (out/'source_manifest.json').write_text(json.dumps({'turn_revision.py':hashlib.sha256(src.read_bytes()).hexdigest(),'candidate_file':str(Path(sys.argv[2]).resolve()),'candidate_sha256':hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()},indent=2))
    for k,p in enumerate(candidates):
        p=dict(p)
        duration=p.pop('duration',12.)
        seed=p.pop('seed',91001)
        action=p.pop('action','turn_90')
        spec=development_spec(action,seed=seed,duration=duration,controller_module='turn_revision',turn_parameters=p)
        result,data,path=session.run(spec,f'candidate_{k:03d}')
        m=mechanics(data)
        m['target_deg']=p.get('target_deg',90. if action=='turn_90' else 180.)
        m['angle_error_deg']=m['yaw_deg']-m['target_deg']
        m['acceptance_pass']=bool(abs(m['angle_error_deg'])<=10 and m['tail_min_height_m']>.20 and m['tail_max_tilt_deg']<15 and m['tail_max_linear_speed']<.02 and m['tail_max_angular_speed']<.1 and m['max_h']<.3)
        m.update(candidate=k,parameters=p,seed=seed,duration=duration,raw=str(path))
        summary.append(m)
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(m),flush=True)
