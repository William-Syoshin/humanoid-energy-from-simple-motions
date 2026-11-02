"""Declared pre-training feasibility sweep; never selects on predicted energy."""
import copy
import itertools
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backends.socket_backend import WebotsProcess
from env.khr_env import KHR3HVEnv
from rl.common import load_config, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slow',action='store_true')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    cfg=load_config()
    cfg['simulation']['episode_s']=8.
    cfg['task']['target_speed_m_s']=0.015
    cfg['gait']['lower']=[0.9,0.005,0.002,0.005,-0.06]
    cfg['gait']['upper']=[3.4,0.055,0.020,0.22,0.10]
    out=args.out or ROOT/'logs'/('gait_pilot_slow_v2' if args.slow else 'gait_pilot')
    out.mkdir(parents=True,exist_ok=False)
    candidates=[list(x) for x in itertools.product([1.4,2.0],[0.025],[0.003,0.005],[0.025,0.05],[0.])]
    if args.slow:
        candidates=[list(x) for x in itertools.product([2.4,2.8],[0.020,0.025],[0.003],[0.01,0.02],[0.])]
    write_json(out/'design.json',dict(config=cfg,parameters=candidates,
               selection='First require no falls/heading failure, then low OOD and target speed. No energy-based selection.'))
    backend=WebotsProcess(cfg)
    rows=[]
    try:
        for parameters in candidates:
            c=copy.deepcopy(cfg);c['gait']['nominal']=parameters
            e=KHR3HVEnv(c,backend=backend)
            e.reset(seed=42)
            while True:
                _,_,t,tr,info=e.step(e.gait.nominal_action())
                if t or tr:break
            row=dict(parameters=parameters,**info['summary']);rows.append(row)
            write_json(out/'results.json',rows)
            print(parameters,'speed',round(row['mean_speed_m_s'],4),'ood',round(row['ood_rate'],3),
                  'fall',row['fallen'],'heading',round(row['heading_error_rad'],3),flush=True)
    finally:backend.close()


if __name__=='__main__':main()
