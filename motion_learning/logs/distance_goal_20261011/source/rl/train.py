"""Train one fixed experimental condition; preserve config, checkpoints and logs."""
import argparse
import copy
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from env.khr_env import KHR3HVEnv
from rl.common import load_config, validate_config, provenance, write_json


class ExperimentCallback(BaseCallback):
    def __init__(self, directory, interval):
        super().__init__()
        self.directory, self.interval = directory, interval
        self.next_save = interval
        self.episodes = (directory/'episodes.jsonl').open('a')

    def _on_step(self):
        for info in self.locals['infos']:
            for key, value in info.get('reward_terms', {}).items():
                self.logger.record_mean('reward/'+key, value)
            self.logger.record_mean('task/forward_speed_m_s', info.get('forward_speed_m_s', 0.))
            if 'summary' in info:
                s = info['summary']
                self.episodes.write(json.dumps(dict(timesteps=self.num_timesteps, **s), allow_nan=False)+'\n')
                self.episodes.flush()
                for key in ('distance_m','mean_speed_m_s','fallen','task_success','ood_rate','negative_power_rate'):
                    self.logger.record_mean('episode/'+key, float(s[key]))
                for model, value in s['predicted_energy_J'].items():
                    self.logger.record_mean('energy/'+model+'_J', value)
        if self.num_timesteps >= self.next_save:
            self.model.save(self.directory/f'checkpoint_{self.num_timesteps}')
            self.next_save += self.interval
        return True

    def _on_training_end(self):
        self.episodes.close()


def train(config, directory, validate=True):
    validate_config(config)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    tr = config['training']
    torch.set_num_threads(tr['torch_threads'])
    versions = {n: importlib.metadata.version(n) for n in ['torch','numpy','gymnasium','stable-baselines3','tensorboard','PyYAML']}
    write_json(directory/'provenance.json', dict(files=provenance(), versions=versions, python=sys.version,
                                               seed=config['seed'], requested_device=tr['device']))
    env = KHR3HVEnv(config)
    model = callback = None
    start = time.monotonic()
    try:
        if validate:
            check_env(env, warn=True)
            env.reset(seed=config['seed'])
        wrapped = Monitor(env, str(directory/'monitor.csv'))
        model = PPO('MlpPolicy', wrapped, learning_rate=tr['learning_rate'], n_steps=tr['n_steps'],
                    batch_size=tr['batch_size'], gamma=tr['gamma'], gae_lambda=tr['gae_lambda'],
                    n_epochs=tr['n_epochs'], ent_coef=tr['ent_coef'], clip_range=tr['clip_range'],
                    seed=config['seed'], device=tr['device'], verbose=1,
                    policy_kwargs={'net_arch': tr['net_arch']})
        model.set_logger(configure(str(directory), ['stdout','csv','tensorboard']))
        model.experiment_config = copy.deepcopy(config)
        callback = ExperimentCallback(directory, tr['model_save_interval'])
        model.learn(total_timesteps=tr['total_timesteps'], callback=callback)
        model.save(directory/'final')
        write_json(directory/'simulator.json', dict(instance=str(env.backend.directory), inventory=env.backend.inventory))
        write_json(directory/'training_summary.json', dict(status='complete', timesteps=model.num_timesteps,
                   requested_timesteps=tr['total_timesteps'], wall_s=time.monotonic()-start,
                   actual_device=str(model.device), condition=config['condition'], seed=config['seed']))
    except BaseException as exc:
        if model is not None:
            model.save(directory/'interrupted')
        write_json(directory/'training_summary.json', dict(status='interrupted', error=str(exc),
                   wall_s=time.monotonic()-start))
        raise
    finally:
        if callback and not callback.episodes.closed:
            callback.episodes.close()
        env.close()
    return directory/'final.zip'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path)
    p.add_argument('--condition', choices=['task_only','R1','R2','R3'])
    p.add_argument('--steps', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--energy-weight', type=float)
    p.add_argument('--device')
    p.add_argument('--gui', action='store_true')
    p.add_argument('--out', type=Path)
    p.add_argument('--skip-check-env', action='store_true')
    a = p.parse_args()
    cfg = load_config(a.config)
    for key in ('condition','seed'):
        if getattr(a, key) is not None:
            cfg[key] = getattr(a, key)
    if a.steps is not None: cfg['training']['total_timesteps'] = a.steps
    if a.device is not None: cfg['training']['device'] = a.device
    if a.energy_weight is not None: cfg['energy']['weight'] = a.energy_weight
    if cfg['energy']['weight'] < 0: p.error('energy weight must be nonnegative')
    if a.gui: cfg['simulation']['gui'] = True
    directory = a.out or ROOT/cfg['training']['log_directory']/(
        datetime.now().strftime('%Y%m%d-%H%M%S')+f"-{cfg['condition']}-seed{cfg['seed']}")
    print('Saved:', train(cfg, directory, validate=not a.skip_check_env))


if __name__ == '__main__':
    main()
