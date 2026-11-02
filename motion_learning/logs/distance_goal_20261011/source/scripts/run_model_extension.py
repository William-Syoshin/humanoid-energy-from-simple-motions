"""Repeat the archived R2 protocol for R1 and R3, with a global worker limit.

Training, simulator and predictor files come byte-for-byte from the archived
experiment. This runner never substitutes seeds or overwrites partial runs.
"""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl.run_sweep import atomic_json, digest, run_job, verify_source

MODELS = ('R1', 'R3')


def utc():
    return datetime.now(timezone.utc).isoformat()


def prepare(root, reference, steps=None, seeds=None, episodes=None):
    reference = reference.resolve()
    old = json.loads((reference/'design.json').read_text())
    manifest = json.loads((reference/'source_manifest.json').read_text())
    # Verify archived bytes before constructing a new experiment.
    verify_source(reference)
    root.mkdir(parents=True, exist_ok=False)
    driver = root/'driver'
    driver.mkdir()
    for file in ('scripts/run_model_extension.py', 'scripts/finalize_model_extension.py',
                 'rl/run_sweep.py', 'rl/analyze_sweep.py', 'rl/audit_sweep.py', 'rl/common.py'):
        dest=driver/file
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/file,dest)
    driver_hashes={str(p.relative_to(driver)):digest(p) for p in driver.rglob('*.py')}
    atomic_json(driver/'manifest.json',driver_hashes)
    plans={}
    for model in MODELS:
        child=root/model
        source=child/'source'
        source.mkdir(parents=True)
        # Copy only archived manifest entries, never the old runtime logs.
        for name,expected in manifest.items():
            target=source/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(reference/'source'/name,target)
            if digest(target)!=expected: raise RuntimeError('Archived source mismatch: '+name)
        atomic_json(child/'source_manifest.json',manifest)
        plan=json.loads(json.dumps(old))
        plan['created_utc']=utc()
        plan['extension_model']=model
        plan['reference_experiment']=str(reference)
        plan['reference_design_sha256']=digest(reference/'design.json')
        plan['source_manifest_sha256']=digest(child/'source_manifest.json')
        plan['config']=json.loads(json.dumps(old['config']))
        if steps is not None: plan['config']['training']['total_timesteps']=steps
        if seeds is not None: plan['training_seeds']=seeds
        if episodes is not None:
            plan['evaluation_episodes']=episodes
            plan['config']['evaluation']['episodes']=episodes
        plan['jobs']=[]
        for seed in plan['training_seeds']:
            for cond,w in [('task_only',0.)]+[(model,w) for w in plan['weights']]:
                plan['jobs'].append(dict(name=f'{cond}-lambda{w:g}-seed{seed}',condition=cond,weight=w,seed=seed))
        random.Random(20261008).shuffle(plan['jobs'])
        plan['analysis']['primary_model']=model
        others='/'.join(m for m in ('R1','R2','R3') if m!=model)
        plan['analysis']['note']=(f'Exact extension of R2 protocol. Primary outcome uses {model}; '
            f'{others} are cross-model sensitivity analyses. Holm across 3 weights within this model family. '
            'The combined extension additionally reports Holm across all 6 new model/weight hypotheses. '
            'The old R2 comparison is descriptive. Repeated Task-only seeds are reproducibility checks, '
            'not extra independent training seeds. No claim of measured hardware energy savings.')
        atomic_json(child/'design.json',plan)
        for name in ('status','console','analysis_source'): (child/name).mkdir()
        for name in ('audit_sweep.py','analyze_sweep.py'):
            shutil.copy2(driver/'rl'/name,child/'analysis_source'/name)
        atomic_json(child/'analysis_source/manifest.json',
                    {p.name:digest(p) for p in (child/'analysis_source').glob('*.py')})
        with (child/'runtime_freeze.txt').open('w') as f:
            subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
        plans[model]=plan
    sequence=[dict(model=model,job=j) for model,p in plans.items() for j in p['jobs']]
    random.Random(20261009).shuffle(sequence)
    plan=dict(created_utc=utc(),reference_experiment=str(reference),workers=3,
              models=list(MODELS),jobs=sequence,python=sys.executable,
              planned_training_runs=len(sequence),planned_evaluation_episodes=sum(p['evaluation_episodes']*len(p['jobs']) for p in plans.values()),
              baseline_policy='Train a fresh Task-only run for each model family, using identical seeds and budgets.',
              stage='prepared',driver_hashes=driver_hashes,
              smoke_test=steps is not None or seeds is not None or episodes is not None)
    atomic_json(root/'design.json',plan)
    return plan


def execute(root, project):
    plan=json.loads((root/'design.json').read_text())
    for name,expected in plan['driver_hashes'].items():
        if digest(root/'driver'/name)!=expected: raise RuntimeError('Frozen driver changed: '+name)
    lock=(root/'runner.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('This experiment already has an active runner')
    atomic_json(root/'runner.json',dict(pid=os.getpid(),started_utc=utc(),python=sys.executable))
    plans={m:json.loads((root/m/'design.json').read_text()) for m in MODELS}
    results={m:[] for m in MODELS}
    for model in MODELS: verify_source(root/model)
    # All 48 runs share this one pool. Never launch 3 workers per model.
    with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        futures={pool.submit(run_job,root/item['model'],plans[item['model']],item['job']):item
                 for item in plan['jobs']}
        for future in as_completed(futures):
            item=futures[future];model=item['model']
            result=future.result()
            if result is not None: results[model].append(result)
            atomic_json(root/model/'results.json',sorted(results[model],key=lambda r:(r['training_seed'],r['weight'])))
            count=sum(len(x) for x in results.values())
            atomic_json(root/'progress.json',dict(updated_utc=utc(),complete=count,planned=len(plan['jobs']),
                        by_model={m:len(v) for m,v in results.items()}))
            print(utc(),model,item['job']['name'],'complete' if result else 'FAILED',f'{count}/{len(plan["jobs"])}',flush=True)
    good=True
    for model in MODELS:
        verify_source(root/model)
        n=len(results[model]);all_complete=n==len(plans[model]['jobs']);good &= all_complete
        atomic_json(root/model/'completion.json',dict(planned=len(plans[model]['jobs']),complete=n,
                    all_complete=all_complete,source_verified=True,finished_utc=utc()))
    atomic_json(root/'completion.json',dict(planned=len(plan['jobs']),complete=sum(len(x) for x in results.values()),
                all_complete=good,finished_utc=utc()))
    if not good: raise RuntimeError('Incomplete runs retained. Diagnose before explicit resume.')
    with (root/'finalize.log').open('a') as output:
        subprocess.run([sys.executable,str(root/'driver/scripts/finalize_model_extension.py'),str(root),
                        '--project',str(project)],stdout=output,stderr=subprocess.STDOUT,check=True,timeout=2400)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=ROOT/'logs/coefficient_sweep_20261008')
    p.add_argument('--project',type=Path,default=ROOT)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true',help='512 steps, one seed, two evaluations; separate directory only')
    a=p.parse_args();root=a.out.resolve()
    if not a.resume:
        prepare(root,a.reference,steps=512 if a.smoke else None,seeds=[10] if a.smoke else None,
                episodes=2 if a.smoke else None)
    if not a.prepare_only:
        try: execute(root,a.project.resolve())
        except BaseException as exc:
            atomic_json(root/'runner_error.json',dict(time_utc=utc(),error=repr(exc)))
            raise


if __name__=='__main__': main()
