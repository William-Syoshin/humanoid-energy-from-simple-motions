"""Run equal-budget Task-only / energy conditions, then held-out paired evaluation."""
import argparse
import copy
from datetime import datetime
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl.common import load_config, write_json
from rl.train import train
from rl.evaluate import evaluate
from rl.report import report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path)
    p.add_argument('--steps', type=int)
    p.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--models', nargs='+', choices=['R1','R2','R3'], default=['R2'])
    p.add_argument('--weights', type=float, nargs='+', default=[0.1])
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if any(w < 0 for w in a.weights): p.error('weights must be nonnegative')
    cfg = load_config(a.config)
    if a.steps: cfg['training']['total_timesteps'] = a.steps
    root = a.out or ROOT/'logs/comparisons'/datetime.now().strftime('%Y%m%d-%H%M%S')
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'design.json', dict(config=cfg, seeds=a.seeds, models=a.models,
               weights=a.weights, evaluation_episodes=a.episodes, note='Training seeds are the replication units; evaluation seeds are paired.'))
    results = []
    for seed in a.seeds:
        conditions = [('task_only',0.)]+[(m,w) for m in a.models for w in a.weights]
        for model, weight in conditions:
            c = copy.deepcopy(cfg)
            c['seed'], c['condition'], c['energy']['weight'] = seed, model, weight
            folder = root/f'{model}-lambda{weight:g}-seed{seed}'
            policy = train(c, folder)
            rows = evaluate(c, folder/'evaluation', policy, a.episodes, cfg['evaluation']['seed_start'])
            results.append(dict(training_seed=seed, condition=model, weight=weight, run=str(folder), episodes=rows))
            write_json(root/'results.json', results)
    report(root)
    print('Comparison complete:', root)


if __name__ == '__main__':
    main()
