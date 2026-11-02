"""Paired-seed evaluation with all three frozen power models and raw trajectories."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.khr_env import KHR3HVEnv
from rl.common import load_config, provenance, write_json


def aggregate(rows):
    result = {'episodes': len(rows)}
    for k in ('distance_m','mean_speed_m_s','fallen','task_success','in_distribution_success',
              'heading_error_rad','ood_rate','negative_power_rate'):
        v = np.array([r[k] for r in rows], dtype=float)
        result[k] = {'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if len(v)>1 else 0.}
    # Failed/slow trajectories are not eligible for energy-per-distance savings.
    good = [r for r in rows if r['task_success']]
    result['eligible_episodes'] = len(good)
    result['predicted_energy_J_all'] = {m: float(np.mean([r['predicted_energy_J'][m] for r in rows])) for m in ('R1','R2','R3')}
    result['energy_per_m_J_successful'] = {m: float(np.mean([r['energy_per_m_J'][m] for r in good])) if good else None for m in ('R1','R2','R3')}
    return result


def evaluate(config, output, model_path=None, episodes=5, seed_start=10000, baseline=False, random_actions=False, screenshot=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'config.json', config)
    write_json(output/'provenance.json', provenance())
    model = None
    if model_path:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device='cpu')
    env = KHR3HVEnv(config)
    rows = []
    try:
        for episode in range(episodes):
            seed = seed_start+episode
            observation, _ = env.reset(seed=seed)
            env.action_space.seed(seed)
            while True:
                if model:
                    action, _ = model.predict(observation, deterministic=True)
                elif random_actions:
                    action = env.action_space.sample()
                else:
                    action = env.gait.nominal_action()
                observation, _, done, timeout, info = env.step(action)
                if done or timeout: break
            row = dict(seed=seed, policy=str(model_path) if model_path else ('random' if random_actions else 'nominal'), **info['summary'])
            rows.append(row)
            write_json(output/f'episode_{seed}.json', row)
            with (output/f'trajectory_{seed}.jsonl').open('w') as stream:
                for state in env.trace: stream.write(json.dumps(state, allow_nan=False)+'\n')
            with (output/f'power_{seed}.jsonl').open('w') as stream:
                for prediction in env.power.predictions: stream.write(json.dumps(prediction, allow_nan=False)+'\n')
            if screenshot and episode == 0:
                env.backend.screenshot(output/'webots.png')
            print(json.dumps(row, allow_nan=False), flush=True)
    finally:
        env.close()
    report = aggregate(rows)
    write_json(output/'summary.json', report)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path)
    p.add_argument('--baseline', action='store_true')
    p.add_argument('--random', action='store_true')
    p.add_argument('--config', type=Path)
    p.add_argument('--episodes', type=int)
    p.add_argument('--seed-start', type=int)
    p.add_argument('--gui', action='store_true')
    p.add_argument('--realtime', action='store_true')
    p.add_argument('--screenshot', action='store_true')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if sum([a.model is not None, a.baseline, a.random]) != 1:
        p.error('Select exactly one of --model, --baseline or --random')
    config_path = a.config or (a.model.parent/'config.yaml' if a.model else None)
    if a.model and not a.config and not config_path.exists():
        from stable_baselines3 import PPO
        saved = PPO.load(a.model,device='cpu')
        if not hasattr(saved,'experiment_config'):
            p.error('Model has no embedded config: supply --config')
        cfg = saved.experiment_config
    else:
        cfg = load_config(config_path)
    cfg['simulation']['gui'] = a.gui or a.screenshot
    cfg['simulation']['realtime'] = a.realtime
    episodes = a.episodes or cfg['evaluation']['episodes']
    output = a.out or ROOT/'logs/evaluation'/datetime.now().strftime('%Y%m%d-%H%M%S')
    evaluate(cfg, output, a.model, episodes, a.seed_start if a.seed_start is not None else cfg['evaluation']['seed_start'],
             baseline=a.baseline, random_actions=a.random, screenshot=a.screenshot)
    print('Evaluation:', output)


if __name__ == '__main__':
    main()
