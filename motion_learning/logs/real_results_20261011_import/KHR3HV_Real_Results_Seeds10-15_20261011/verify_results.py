"""Offline verification and energy recomputation; Python standard library only.
Never connects to robot hardware. Run: python3 verify_results.py
"""
from pathlib import Path
import csv, hashlib, json, math, bisect
root=Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
checks=read(root/'SHA256_MANIFEST.json')
for rel,digest in checks.items():
    assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest, 'Hash mismatch: '+rel
records=read(root/'RESULTS.json')['trials']
assert len(records)==24
assert {(r['seed'],r['condition']) for r in records}=={(s,c) for s in range(10,16) for c in ['R1','R2','R3','Baseline']}
for record in records:
    p=root/record['raw_directory']; meta=read(p/'meta.json');cfg=read(p/'config.json'); motion=read(p/'motion.json')
    commands=rows(p/'commands.csv'); power=rows(p/'power.csv')
    assert meta['status']=='complete' and not meta['mock'] and not meta['reader_errors']
    assert len(commands)==len(motion['commands']) and all(int(c['ok'])==1 for c in commands)
    trial=read(p/'trial.json')
    original_motion=root/'experiment/data'/trial['motion']
    assert hashlib.sha256(original_motion.read_bytes()).hexdigest()==meta['motion_sha256']
    assert read(original_motion)==motion, 'Re-serialized motion snapshot differs from original'
    assert hashlib.sha256((p/'config.json').read_bytes()).hexdigest()==meta['config_sha256']
    t=[float(r['pc_s'])-cfg['timing']['receive_latency_s'] for r in power]
    d=[float(r['device_s']) for r in power]
    watts=[float(r['voltage_V'])*float(r['current_A']) for r in power]
    assert all(math.isfinite(v) for arr in (t,d,watts) for v in arr)
    assert all(b>a for a,b in zip(t,t[1:])) and all(b>a for a,b in zip(d,d[1:]))
    start,end=meta['score_start'],meta['score_end']
    assert t[0]<=start<end<=t[-1]
    a=max(0,bisect.bisect_left(t,start)-1);b=min(len(t),bisect.bisect_left(t,end)+1)
    assert max(y-x for x,y in zip(t[a:b],t[a+1:b]))<=cfg['max_power_gap_s']
    assert max(y-x for x,y in zip(d[a:b],d[a+1:b]))<=cfg['max_power_gap_s']
    assert not any(int(r['clip']) for r in power[a:b])
    def interpolate(x):
        j=bisect.bisect_left(t,x)
        if j<len(t) and t[j]==x:return watts[j]
        return watts[j-1]+(watts[j]-watts[j-1])*(x-t[j-1])/(t[j]-t[j-1])
    grid=sorted(set([start,end]+[x for x in t if start<=x<=end]))
    energy=sum((y-x)*(interpolate(x)+interpolate(y))/2 for x,y in zip(grid,grid[1:]))
    assert abs(energy-record['energy_J'])<1e-7, (record['trial_id'],energy,record['energy_J'])
    assert abs(energy/(end-start)-record['mean_power_W'])<1e-7
    print(f"seed={record['seed']} {record['condition']:8s} {energy:.6f} J: PASS")
print(f'PASS: {len(checks)} file hashes and all 24 physical recordings verified. No robot commands sent.')
