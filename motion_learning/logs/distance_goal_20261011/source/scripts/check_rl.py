"""Actual Webots integration checks. Fails loudly on nondeterministic reset."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from stable_baselines3.common.env_checker import check_env
from env.khr_env import KHR3HVEnv
from rl.common import load_config, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'logs/integration.json')
    a=p.parse_args()
    cfg=load_config();cfg['simulation']['episode_s']=2.
    e=KHR3HVEnv(cfg)
    try:
        check_env(e)
        sequences=[]
        for _ in range(2):
            o,_=e.reset(seed=123)
            seq=[o]
            for i in range(20):
                o,r,t,tr,info=e.step(e.gait.nominal_action());seq.append(o)
                if t or tr:break
            sequences.append(np.array(seq))
        np.testing.assert_allclose(sequences[0],sequences[1],atol=1e-5,rtol=1e-5)
        e.reset(seed=124);e.action_space.seed(124)
        for i in range(20):
            o,r,t,tr,info=e.step(e.action_space.sample())
            assert np.isfinite(o).all() and np.isfinite(r)
            if t or tr:break
        write_json(a.out,dict(check_env='passed',repeat_reset_max_difference=float(np.max(abs(sequences[0]-sequences[1]))),
                   random_action_steps=i+1,random_episode=info.get('summary'),instance=str(e.backend.directory)))
        print(a.out.read_text())
    finally:e.close()


if __name__=='__main__':main()
