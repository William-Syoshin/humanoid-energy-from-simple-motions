"""Verify packaged commands and plans without opening hardware ports."""
from pathlib import Path
import csv
import numpy as np
from common import read,sha,motion_arrays,NAMES


def verify(root):
    bundle=root/'data';manifest=read(bundle/'manifest.json');items=manifest['motions']
    expected={(c,s) for c in ['Task-only','R1','R2','R3'] for s in range(10,16)}
    if len(items)!=24 or {(m['condition'],m['training_seed']) for m in items}!=expected:
        raise ValueError('Expected all 24 distance-task motions')
    for item in items:
        path=bundle/item['file'];m=read(path);t,q,initial=motion_arrays(m)
        if sha(path)!=item['sha256']:raise ValueError('Motion hash mismatch: '+str(path))
        if m['evaluation_seed']!=20000 or m['task']!='20cm_distance_goal':raise ValueError('Wrong task/evaluation seed')
        if m['weight']!=(0 if m['condition']=='Task-only' else .3):raise ValueError('Wrong reward coefficient')
        if not m['sim']['task_success'] or abs(m['sim']['duration_s']-m['duration_s'])>1e-7:raise ValueError('Incomplete export')
        with path.with_suffix('.csv').open(newline='') as f:
            reader=csv.reader(f)
            if next(reader)!=['time_s']+NAMES:raise ValueError('Wrong CSV header')
            data=np.array([[float(v) for v in r] for r in reader])
        np.testing.assert_allclose(data,np.column_stack((t,q)),rtol=0,atol=1e-12)
        prov=root/'provenance'/m['motion_id']
        if sha(prov/'config.yaml')!=m['source']['config_sha256']:raise ValueError('Changed source configuration')
        if sha(root/'provenance/frozen_gait.py')!=m['source']['gait_sha256']:raise ValueError('Changed source gait')
    for name,seeds,count in [('plan_pilot.json',[10],12),('plan_all_seeds.json',list(range(10,16)),72)]:
        plan=read(root/name)
        if len(plan['trials'])!=count or plan['training_seeds']!=seeds or plan['repeats']!=3:raise ValueError('Wrong trial plan')
        if sha(bundle/'manifest.json')!=plan['manifest_sha256']:raise ValueError('Manifest mismatch')
        observed=set()
        for tr in plan['trials']:
            m=read(bundle/tr['motion']);key=(tr['condition'],tr['training_seed'],tr['repeat'])
            if key in observed:raise ValueError('Duplicate trial')
            observed.add(key)
            if tr['motion_sha256']!=sha(bundle/tr['motion']):raise ValueError('Trial motion hash mismatch')
            if (m['condition'],m['training_seed'])!=key[:2]:raise ValueError('Trial identity mismatch')
        if observed!={(c,s,r) for c in ['Task-only','R1','R2','R3'] for s in seeds for r in [1,2,3]}:raise ValueError('Missing repeat')
    print('Verified 24 motion JSON/CSV files and source hashes; pilot 12 and all-seed 72 trial plans. No hardware accessed.')

if __name__=='__main__':verify(Path(__file__).resolve().parent)
