"""Fixed-budget coefficient sweep, isolated processes, immutable source copy.

This runs separate training replicates concurrently; it does not change PPO to
a vector environment. Failed runs stay in the record and are never replaced by
another seed. Resume only skips stages with complete, validated output files.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl.common import load_config
import yaml


def atomic_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root, steps, seeds, weights, episodes, workers):
    root.mkdir(parents=True, exist_ok=False)
    source = root / 'source'
    source.mkdir()
    for directory in ('rl', 'env', 'energy', 'backends', 'controllers', 'scripts',
                      'config', 'protos', 'worlds', 'vendor', 'tests'):
        shutil.copytree(ROOT / directory, source / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.*.wbproj'))
    for name in ('requirements.txt', 'requirements-lock-macos.txt', 'pytest.ini'):
        shutil.copy2(ROOT / name, source / name)
    # Only these pre-existing files are frozen. Runtime instance logs are not.
    hashes = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob('*')) if p.is_file()}
    atomic_json(root / 'source_manifest.json', hashes)
    cfg = load_config()
    cfg['training']['total_timesteps'] = steps
    cfg['training']['device'] = 'cpu'
    cfg['training']['model_save_interval'] = 25000
    cfg['evaluation']['seed_start'] = 20000
    cfg['evaluation']['episodes'] = episodes
    jobs = []
    for seed in seeds:
        for condition, weight in [('task_only', 0.)] + [('R2', w) for w in weights]:
            name = f'{condition}-lambda{weight:g}-seed{seed}'
            jobs.append(dict(name=name, condition=condition, weight=weight, seed=seed))
    random.Random(20261008).shuffle(jobs)
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), config=cfg, jobs=jobs,
                training_seeds=seeds, weights=weights, evaluation_episodes=episodes,
                workers=workers, source_manifest_sha256=digest(root/'source_manifest.json'),
                analysis=dict(distance_equivalence=[.95, 1.05], minimum_paired_pass_rate=.9,
                              minimum_replicates=6, bootstrap_samples=20000, bootstrap_seed=84017,
                              primary_model='R2', familywise_alpha=.05,
                              meaningful_energy_reduction_pct=1.,
                              note='All coefficients reported. Independent unit = training seed. '
                              'Primary claim requires performance gates in every seed, '
                              'negative R2 mean change in every seed, Holm-adjusted paired '
                              'sign-flip p < .05, and negative mean changes for R1/R3. '
                              'Stronger practical claim additionally requires >=1% R2 reduction. '
                              'Results are predicted energy, not measured electrical savings.'))
    atomic_json(root / 'design.json', plan)
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=(root/'runtime_freeze.txt').open('w'), check=True)
    return plan


def verify_source(root):
    hashes = json.loads((root/'source_manifest.json').read_text())
    changed = [name for name, value in hashes.items() if digest(root/'source'/name) != value]
    if changed:
        raise RuntimeError('Frozen source changed: ' + ', '.join(changed))


def run_job(root, plan, job):
    source = root / 'source'
    folder = root / job['name']
    config = json.loads(json.dumps(plan['config']))
    config['seed'] = job['seed']
    config['condition'] = job['condition']
    config['energy']['weight'] = job['weight']
    configs = root/'configs'
    configs.mkdir(exist_ok=True)
    config_path = configs/(job['name']+'.yaml')
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', PYTHONUNBUFFERED='1')
    start = time.monotonic()
    record = dict(**job, status='running', started_utc=datetime.now(timezone.utc).isoformat())
    status_path = root/'status'/(job['name']+'.json')
    atomic_json(status_path, record)
    try:
        verify_source(root)
        summary = folder/'training_summary.json'
        if not summary.exists() or json.loads(summary.read_text()).get('status') != 'complete':
            if folder.exists():
                raise RuntimeError('Incomplete training exists; preserved, not silently restarted: '+str(folder))
            with (root/'console'/(job['name']+'.train.log')).open('w') as output:
                subprocess.run([sys.executable, str(source/'rl/train.py'), '--config', str(config_path),
                                '--out', str(folder)], cwd=source, env=env, stdout=output,
                               stderr=subprocess.STDOUT, check=True, timeout=7200)
        training = json.loads(summary.read_text())
        expected = ((config['training']['total_timesteps']+config['training']['n_steps']-1)
                    //config['training']['n_steps'])*config['training']['n_steps']
        if training['timesteps'] != expected:
            raise RuntimeError('Training budget mismatch')
        evaluation = folder/'evaluation'
        if not (evaluation/'summary.json').exists():
            if evaluation.exists():
                raise RuntimeError('Incomplete evaluation preserved: '+str(evaluation))
            with (root/'console'/(job['name']+'.eval.log')).open('w') as output:
                subprocess.run([sys.executable, str(source/'rl/evaluate.py'), '--model', str(folder/'final.zip'),
                                '--episodes', str(plan['evaluation_episodes']), '--seed-start', '20000',
                                '--out', str(evaluation)], cwd=source, env=env, stdout=output,
                               stderr=subprocess.STDOUT, check=True, timeout=900)
        episodes = [json.loads((evaluation/f'episode_{seed}.json').read_text())
                    for seed in range(20000, 20000+plan['evaluation_episodes'])]
        result = dict(training_seed=job['seed'], condition=job['condition'], weight=job['weight'],
                      run=str(folder), episodes=episodes)
        record.update(status='complete', wall_s=time.monotonic()-start, training=training,
                      model_sha256=digest(folder/'final.zip'))
        atomic_json(status_path, record)
        return result
    except Exception as exc:
        record.update(status='failed', error=repr(exc), wall_s=time.monotonic()-start)
        atomic_json(status_path, record)
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(10,16)))
    p.add_argument('--weights', type=float, nargs='+', default=[.1,.3,1.])
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    root = a.out.resolve()
    if not a.resume:
        if len(set(a.seeds)) != len(a.seeds) or len(set(a.weights)) != len(a.weights):
            p.error('Duplicate seeds/weights')
        if min(a.weights) <= 0 or min(a.steps,a.episodes,a.workers) <= 0:
            p.error('Positive weights, steps, episodes and workers required')
        plan = prepare(root,a.steps,a.seeds,a.weights,a.episodes,a.workers)
    else:
        plan = json.loads((root/'design.json').read_text())
    for name in ('status','console'): (root/name).mkdir(exist_ok=True)
    verify_source(root)
    print('Frozen design:', root/'design.json', flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        futures = {pool.submit(run_job,root,plan,job): job for job in plan['jobs']}
        for future in as_completed(futures):
            result = future.result()
            if result is not None: results.append(result)
            atomic_json(root/'results.json', sorted(results,key=lambda r:(r['training_seed'],r['weight'])))
            print('Finished',futures[future]['name'], 'complete' if result else 'FAILED',
                  f'{len(results)}/{len(plan["jobs"])} successful',flush=True)
    verify_source(root)
    atomic_json(root/'completion.json',dict(planned=len(plan['jobs']),complete=len(results),
                all_complete=len(results)==len(plan['jobs']), source_verified=True,
                finished_utc=datetime.now(timezone.utc).isoformat()))
    if len(results)!=len(plan['jobs']): sys.exit(1)


if __name__ == '__main__': main()
