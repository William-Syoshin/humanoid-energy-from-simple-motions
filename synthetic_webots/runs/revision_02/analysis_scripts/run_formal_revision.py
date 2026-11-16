"""Freeze and evaluate revised commanded motions without touching original data."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
import fcntl
import numpy as np
from webots_session import Session, ROOT, PYTHON
from motion_audit import audit_motion, THRESHOLDS

ORIGINAL = Path('/Users/yuchenzhou/Desktop/ヒューマノイド研究/07_GitHub公開用/humanoid-energy-from-simple-motions/synthetic_webots')

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))

def copy_reused_record(source, destination):
    # APFS clones are independent copy-on-write files, not hard links. They
    # retain complete recordings without duplicating their physical blocks.
    if sys.platform=='darwin':
        subprocess.run(['/bin/cp','-c','-p',str(source),str(destination)],check=True)
        return 'APFS independent copy-on-write clone'
    shutil.copy2(source,destination)
    return 'full independent copy'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--specs', type=Path, required=True, help='Action -> fixed full specification template or overrides')
    ap.add_argument('--revision', default='revision_01')
    args = ap.parse_args()
    lock = (ROOT/'formal_runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = ROOT/'runs'/args.revision
    run.mkdir(parents=True, exist_ok=False)
    oldrun = ORIGINAL/'runs/full_v2'
    olddesign = json.loads((oldrun/'design.json').read_text())
    overrides = json.loads(args.specs.read_text())
    original_hashes = json.loads((ROOT/'archive_manifest/raw_sha256.json').read_text())
    assert all(digest(oldrun/'raw'/name)==h for name,h in original_hashes.items()), 'Original raw data changed'
    shutil.copytree(ROOT/'project', run/'source', ignore=shutil.ignore_patterns('__pycache__','.DS_Store','runs'))
    shutil.copy2(args.specs, run/'frozen_motion_specs.json')
    shutil.copy2(ROOT/'plan.json', run/'plan.json')
    shutil.copytree(ROOT/'scripts',run/'analysis_scripts',ignore=shutil.ignore_patterns('__pycache__'))
    write(run/'motion_thresholds.json',THRESHOLDS)
    auxiliary_paths=list((run/'analysis_scripts').rglob('*.py'))+[run/'frozen_motion_specs.json',run/'plan.json',run/'motion_thresholds.json']
    if (ROOT/'reports/COMMAND_WINDOW_DIAGNOSTIC_RULES.json').is_file():
        shutil.copy2(ROOT/'reports/COMMAND_WINDOW_DIAGNOSTIC_RULES.json',run/'command_window_rules.json')
        auxiliary_paths.append(run/'command_window_rules.json')
    write(run/'auxiliary_hashes.json',{str(p.relative_to(run)):digest(p) for p in auxiliary_paths})
    (run/'raw').mkdir(); (run/'status').mkdir()
    trials = []
    reused = []
    for old in olddesign['trials']:
        s = dict(old)
        if s['split']=='test' and s['action'] in overrides:
            s.update(overrides[s['action']])
            # All perturbation identities are retained from the original design.
            s.update(split=old['split'], id=old['id'], seed=old['seed'], action=old['action'])
        trials.append(s)
        if old['split']!='test':
            key = old['split']+'_'+old['id']
            copy_method=copy_reused_record(oldrun/'raw'/f'{key}.npz', run/'raw'/f'{key}.npz')
            shutil.copy2(oldrun/'status'/f'{key}.json', run/'status'/f'{key}.json')
            assert digest(run/'raw'/f'{key}.npz')==original_hashes[f'{key}.npz']
            reused.append(dict(key=key,split=old['split'],sha256=original_hashes[f'{key}.npz'],copy_method=copy_method))
    source_hashes = {str(p.relative_to(run/'source')):digest(p)
                     for p in (run/'source').rglob('*') if p.is_file() and p.suffix in ['.py','.c','.wbt','.ini','.json','.csv','.dylib']}
    write(run/'design.json',dict(trials=trials,source_hashes=source_hashes,
        oracle=olddesign['oracle'],original_run=str(oldrun),
        reused_records=len(reused),new_test_records=21,
        frozen_before_evaluation=time.time(),
        selection='Mechanical development only; final prediction errors not inspected during selection'))
    write(run/'reuse_manifest.json',reused)
    write(run/'evaluation_order.json',dict(session='one persistent Webots session',
        keys=['test_'+s['id'] for s in trials if s['split']=='test'],
        note='Contact trajectories may depend on preceding trials despite identical reset commands. Differences from the original are recorded, not excluded.'))
    write(run/'runner.json',dict(pid=__import__('os').getpid(),started=time.time()))
    try:
        results=[]; mechanical=[]; unchanged_traces=[]
        with Session(run/'collection', project=run/'source') as sim:
            for i,s in enumerate(x for x in trials if x['split']=='test'):
                key = 'test_'+s['id']
                write(run/'progress.json',dict(stage='collecting_test',completed=i,total=21,current=key))
                result,data,path=sim.run(s,key)
                shutil.copy2(path,run/'raw'/f'{key}.npz')
                signs=np.array(sim.inventory['joint_signs'])
                bounds=np.array([sim.inventory['native_min_positions'],sim.inventory['native_max_positions']]).T*signs[:,None]
                q=data['qcmd']
                clipped=bool(np.any(q<bounds.min(axis=1)-1e-8) or np.any(q>bounds.max(axis=1)+1e-8))
                assert not clipped, f'Requested native joint range exceeded: {key}'
                mechanical.append(audit_motion(path))
                write(run/'motion_audit.json',mechanical)
                if s['action'] not in overrides:
                    with np.load(oldrun/'raw'/f'{key}.npz',allow_pickle=False) as old_data:
                        differences={k:float(np.max(abs(data[k]-old_data[k]))) for k in ['q','v','tau','h','position','contact','yaw','object_position']}
                    unchanged_traces.append(dict(key=key,max_abs_difference=differences,all_exactly_identical=all(v==0 for v in differences.values()),within_1e8_tolerance=all(v<=1e-8 for v in differences.values())))
                    write(run/'unchanged_motion_reproducibility.json',unchanged_traces)
                write(run/'status'/f'{key}.json',dict(spec=s,**result,sha256=digest(path),completed=time.time(),requested_commands_within_native_range=True))
                results.append(dict(key=key,**result))
                print(f'{i+1}/21 {key} height={result["final_height"]:.5f} h={result["final_h"]:.5f} yaw={result["yaw_change"]:.5f}',flush=True)
        write(run/'completion.json',dict(total_records=len(trials),newly_simulated_test_records=21,
            reused_training_records=1000,reused_validation_records=90,reused_control_records=3,finished=time.time()))
        write(run/'progress.json',dict(stage='fitting_and_scoring',completed=21,total=21))
        command=[PYTHON,str(run/'source/analysis/analyze.py'),'--run-dir',str(run),'--out',str(run/'analysis')]
        with (run/'analysis_console.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        after=all(digest(oldrun/'raw'/name)==h for name,h in original_hashes.items())
        write(run/'original_preservation_check.json',dict(all_1114_original_recordings_unchanged=after))
        assert after
        write(run/'progress.json',dict(stage='analysis_complete',completed=21,total=21))
        print('FORMAL_ANALYSIS_COMPLETE',run,flush=True)
    except Exception:
        write(run/'runner_error.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise

if __name__=='__main__': main()
