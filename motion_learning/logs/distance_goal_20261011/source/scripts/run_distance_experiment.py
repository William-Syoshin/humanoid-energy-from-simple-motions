"""Frozen 20 cm experiment: shared Task-only + R1/R2/R3, lambda=.3."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rl.run_sweep import prepare as freeze, run_job, verify_source, atomic_json, digest
from rl.common import load_config


def utc(): return datetime.now(timezone.utc).isoformat()


def prepare(root,smoke):
    steps=512 if smoke else 100000
    seeds=[10] if smoke else list(range(10,16))
    root.mkdir(parents=True,exist_ok=False)
    source=root/'source';source.mkdir()
    def copy_bytes(src,dst):
        Path(dst).write_bytes(Path(src).read_bytes())
        return dst
    for name in ('rl','env','energy','backends','controllers','scripts','config','protos','worlds','vendor','tests'):
        shutil.copytree(ROOT/name,source/name,copy_function=copy_bytes,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc','.*'))
    for name in ('requirements.txt','requirements-lock-macos.txt','pytest.ini'):
        copy_bytes(ROOT/name,source/name)
    hashes={str(p.relative_to(source)):digest(p) for p in sorted(source.rglob('*')) if p.is_file()}
    atomic_json(root/'source_manifest.json',hashes)
    with (root/'runtime_freeze.txt').open('w') as log:
        subprocess.run([sys.executable,'-m','pip','freeze'],stdout=log,check=True)
    plan=dict(created_utc=utc(),training_seeds=seeds,weights=[.3],evaluation_episodes=2 if smoke else 10,
              workers=3,source_manifest_sha256=digest(root/'source_manifest.json'))
    cfg=load_config(ROOT/'rl/distance_config.yaml')
    cfg['training']['total_timesteps']=steps
    cfg['evaluation']['episodes']=plan['evaluation_episodes']
    plan.update(config=cfg,experiment='distance_20cm',smoke_test=smoke,
                task='Reach x-origin_x >= 0.20 m without falling/escaping; terminal at first 10 Hz boundary; 40 s deadline.',
                baseline_policy='One shared, newly trained Task-only per seed; no reuse of speed-task baselines.',
                observation='Last scalar is normalized remaining distance, replacing speed target; elapsed fraction retained.',
                analysis=dict(primary_evaluator='R2',minimum_success_rate_per_seed=.9,
                    minimum_paired_success_fraction=.9,maximum_goal_overshoot_m=.005,
                    maximum_ood_rate=.1,alpha=.05,bootstrap_seed=20261011,bootstrap_samples=20000,
                    independent_unit='training seed; evaluations nested within seed',
                    note='Primary: common R2 predicted energy to goal on paired successful episodes, with success/OOD/overshoot gates. '
                         'Holm correction across three energy-aware vs shared Task-only comparisons. All failed trials retained. '
                         'All three predictors reported; no measured hardware savings claim.'))
    plan['jobs']=[dict(name=f'{c}-lambda{w:g}-seed{s}',condition=c,weight=w,seed=s)
                  for s in seeds for c,w in [('task_only',0.),('R1',.3),('R2',.3),('R3',.3)]]
    random.Random(20261011).shuffle(plan['jobs'])
    plan['planned_training_runs']=len(plan['jobs'])
    plan['planned_evaluation_episodes']=len(plan['jobs'])*plan['evaluation_episodes']
    atomic_json(root/'design.json',plan)
    for n in ('status','console'): (root/n).mkdir()
    return plan


def execute(root,project):
    plan=json.loads((root/'design.json').read_text())
    lock=(root/'runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_source(root)
    atomic_json(root/'runner.json',dict(pid=os.getpid(),started_utc=utc(),python=sys.executable))
    inhibitor=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]) if sys.platform=='darwin' else None
    results=[];finished=0
    try:
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            futures={pool.submit(run_job,root,plan,j):j for j in plan['jobs']}
            for future in as_completed(futures):
                result=future.result();finished+=1
                if result is not None: results.append(result)
                atomic_json(root/'results.json',sorted(results,key=lambda r:(r['training_seed'],r['condition'])))
                atomic_json(root/'progress.json',dict(updated_utc=utc(),complete=len(results),finished=finished,
                             failed=finished-len(results),planned=len(plan['jobs'])))
                print(utc(),futures[future]['name'],'complete' if result else 'FAILED',flush=True)
        verify_source(root)
        good=len(results)==len(plan['jobs'])
        atomic_json(root/'completion.json',dict(all_complete=good,complete=len(results),planned=len(plan['jobs']),
                    finished_utc=utc(),source_verified=True))
        if not good: raise RuntimeError('Partial or failed runs retained; do not overwrite or replace seeds')
        with (root/'finalize.log').open('a') as log:
            subprocess.run([sys.executable,str(root/'source/scripts/finalize_distance_experiment.py'),
                            str(root),'--project',str(project)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
    finally:
        if inhibitor is not None: inhibitor.terminate()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--project',type=Path,default=ROOT)
    p.add_argument('--smoke',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();root=a.out.resolve()
    if not a.resume: prepare(root,a.smoke)
    if not a.prepare_only:
        try: execute(root,a.project.resolve())
        except BaseException as exc:
            atomic_json(root/'runner_error.json',dict(time_utc=utc(),error=repr(exc)));raise

if __name__=='__main__': main()
