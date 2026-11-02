import copy
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env
from env.khr_env import KHR3HVEnv
from env.gait import Gait
from rl.common import load_config


class ReplayBackend:
    """Deterministic test fixture, not a training/electrical simulator."""
    def reset(self, seed, targets, yaw):
        self.t = 0.
        self.q = np.asarray(targets)
        return self.state()

    def state(self):
        return dict(time=self.t, q=self.q.tolist(), cmd=self.q.tolist(), rpy=[0.,0.,0.],
                    gyro=[0.,0.,0.], accel=[0.,0.,9.81], position=[.025*self.t,0.,.33],
                    velocity=[.025,0.,0.,0.,0.,0.])

    def step(self, targets):
        states=[]
        for q in targets:
            self.q=np.asarray(q);self.t+=.02;states.append(self.state())
        return states

    def close(self): pass


@pytest.mark.parametrize('condition', ['R1', 'R2', 'R3'])
def test_api_and_reward_ablation(condition):
    cfg=load_config();cfg['simulation']['episode_s']=1.
    e=KHR3HVEnv(cfg, backend=ReplayBackend())
    check_env(e)
    energy_cfg=copy.deepcopy(cfg);energy_cfg['condition']=condition
    f=KHR3HVEnv(energy_cfg, backend=ReplayBackend())
    x,_=e.reset(seed=3);y,_=f.reset(seed=3)
    np.testing.assert_equal(x,y)
    for _ in range(10):
        a=e.gait.nominal_action()
        x,r,t,tr,i=e.step(a);y,s,u,ur,j=f.step(a)
        np.testing.assert_equal(x,y)
        assert (t,tr)==(u,ur)
        assert s-r==pytest.approx(j['reward_terms']['energy'])
    assert e.last_episode['task_success']
    assert e.last_episode['predicted_energy_J']==f.last_episode['predicted_energy_J']
    assert sum(f.last_episode['reward_terms'][k] for k in ('energy',))==pytest.approx(-energy_cfg['energy']['weight']*f.power.positive_total[condition])
    with pytest.raises(RuntimeError): e.step(a)


def test_gait_extremes_are_reachable_and_limited():
    cfg=load_config()['gait'];g=Gait(cfg)
    for bits in range(32):
        g.parameters=np.array([cfg['upper'][j] if bits & (1<<j) else cfg['lower'][j] for j in range(5)])
        for phase in np.linspace(0, 2*np.pi, 100):
            q=g.pose(phase,1.)
            assert np.isfinite(q).all()
            assert max(abs(q))<=cfg['joint_limit_rad']


def test_fall_terminates_and_flushes_delayed_energy():
    class FallingBackend(ReplayBackend):
        def state(self):
            s=super().state()
            if self.t >= .5-1e-8: s['position'][2]=.1
            return s
    cfg=load_config();cfg['condition']='R2'
    e=KHR3HVEnv(cfg,backend=FallingBackend())
    e.reset(seed=0)
    energy_reward=0.
    for _ in range(20):
        _,_,terminated,truncated,info=e.step(e.gait.nominal_action())
        energy_reward+=info['reward_terms']['energy']
        if terminated or truncated:break
    assert terminated and not truncated
    assert info['summary']['fallen'] and not info['summary']['task_success']
    assert all(v is None for v in info['summary']['energy_per_m_J'].values())
    assert e.power.scored_samples==e.steps+1
    assert energy_reward==pytest.approx(-cfg['energy']['weight']*e.power.positive_total['R2'])
    e.reset(seed=0)
    assert e.power.scored_samples==0 and e.power.total['R2']==0


def test_invalid_configuration_rejected_before_webots_launch():
    cfg=load_config();cfg['simulation']['action_ms']=96
    with pytest.raises(ValueError): KHR3HVEnv(cfg)
    cfg=load_config();cfg['simulation']['motor_torque_fraction']=1.1
    with pytest.raises(ValueError): KHR3HVEnv(cfg)
