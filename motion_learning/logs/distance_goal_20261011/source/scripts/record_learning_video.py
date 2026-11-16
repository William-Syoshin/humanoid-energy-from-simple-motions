"""Replay saved checkpoints as a movie; this does not record or rerun training."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from stable_baselines3 import PPO
from env.khr_env import KHR3HVEnv
from rl.common import write_json, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20000)
    p.add_argument('--acceleration',type=int,default=4)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    model=PPO.load(a.model,device='cpu')
    cfg=copy.deepcopy(model.experiment_config)
    cfg['simulation']['gui']=True
    cfg['simulation']['realtime']=False
    env=KHR3HVEnv(cfg)
    try:
        obs,_=env.reset(seed=a.seed)
        movie=(a.out/'webots.mp4').resolve()
        env.backend.request('movie_start',path=str(movie),width=640,height=480,
                            acceleration=a.acceleration,guides=True)
        while True:
            action,_=model.predict(obs,deterministic=True)
            obs,_,terminated,truncated,info=env.step(action)
            if terminated or truncated: break
        stopped=env.backend.request('movie_stop')
        assert stopped['simulation_time_before']==stopped['simulation_time_after']
        if not movie.exists() or movie.stat().st_size<1000:
            raise RuntimeError('Movie file missing or empty')
        result=dict(model=str(a.model.resolve()),model_sha256=sha256(a.model),
                    model_timesteps=model.num_timesteps,evaluation_seed=a.seed,
                    playback_acceleration=a.acceleration,summary=info['summary'],
                    recording_stop=stopped,description='Deterministic checkpoint replay; visual-only floor grid; not live training footage.')
        write_json(a.out/'metadata.json',result)
        with (a.out/'trajectory.jsonl').open('w') as stream:
            for s in env.trace: stream.write(json.dumps(s)+'\n')
        print(movie,flush=True)
    finally: env.close()


if __name__=='__main__': main()
