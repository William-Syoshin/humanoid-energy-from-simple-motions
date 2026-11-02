"""Distance goal, deadline, reward ablation and terminal energy conservation."""
import copy
import numpy as np
import pytest
from env.khr_env import KHR3HVEnv
from rl.common import load_config
from test_env import ReplayBackend


def cfg():
    return load_config('rl/distance_config.yaml')


def finish(env):
    observation,_=env.reset(seed=10)
    assert observation[-1] == pytest.approx(1.)
    total=0.
    while True:
        observation, reward, done, trunc, info=env.step(env.gait.nominal_action())
        total+=info['reward_terms']['energy']
        if done or trunc:
            return observation,done,trunc,info['summary'],total


@pytest.mark.parametrize('condition',['task_only','R1','R2','R3'])
def test_goal_stops_and_flushes_all_energy(condition):
    c=cfg();c['condition']=condition
    e=KHR3HVEnv(c,backend=ReplayBackend())
    obs,done,trunc,s,total=finish(e)
    assert done and not trunc and s['task_success'] and s['reached_goal']
    assert 8. <= s['duration_s'] <= 8.1
    assert s['distance_m'] >= .2 and s['goal_overshoot_m'] < .003
    assert obs[-1] == 0
    assert s['reward_terms']['goal'] == 200
    assert s['reward_terms']['fall'] == s['reward_terms']['timeout'] == 0
    assert s['reward_terms']['speed'] == s['reward_terms']['alive'] == s['reward_terms']['upright'] == 0
    assert s['reward_terms']['progress'] == pytest.approx(20)
    assert total == pytest.approx(0 if condition=='task_only' else -.3*s['positive_energy_J'][condition])
    assert e.power.scored_samples == e.steps+1
    with pytest.raises(RuntimeError): e.step(e.gait.nominal_action())


def test_stationary_deadline_is_failure_not_bootstrap_or_low_energy_success():
    class Stationary(ReplayBackend):
        def state(self):
            s=super().state();s['position'][0]=0;return s
    c=cfg();c['simulation']['episode_s']=1.
    _,done,trunc,s,_=finish(KHR3HVEnv(c,backend=Stationary()))
    assert done and not trunc and not s['task_success'] and s['timed_out']
    assert s['time_to_goal_s'] is None
    assert s['reward_terms']['timeout']==-200
    assert all(x is None for x in s['energy_per_m_J'].values())


def test_fall_at_goal_is_failure():
    class FallAtGoal(ReplayBackend):
        def state(self):
            s=super().state()
            if self.t >= 8.-1e-8:
                s['position'][0]=.201;s['position'][2]=.1
            return s
    _,done,trunc,s,_=finish(KHR3HVEnv(cfg(),backend=FallAtGoal()))
    assert done and not trunc and s['fallen'] and not s['task_success']
    assert s['reward_terms']['goal']==0 and s['reward_terms']['fall']==-200


def test_energy_is_only_condition_difference():
    c=cfg();c['simulation']['episode_s']=1.
    a=KHR3HVEnv(c,backend=ReplayBackend());d=copy.deepcopy(c);d['condition']='R3'
    b=KHR3HVEnv(d,backend=ReplayBackend())
    np.testing.assert_array_equal(a.reset(seed=2)[0], b.reset(seed=2)[0])
    for _ in range(10):
        x,r,t,tr,i=a.step(a.gait.nominal_action());y,q,u,ur,j=b.step(b.gait.nominal_action())
        np.testing.assert_array_equal(x,y)
        assert (t,tr)==(u,ur)
        assert q-r == pytest.approx(j['reward_terms']['energy'])


def test_invalid_distance_settings_rejected():
    c=cfg();c['training']['gamma']=.99
    with pytest.raises(ValueError,match='undiscounted'): KHR3HVEnv(c)
    c=cfg();c['reward']['alive']=.5
    with pytest.raises(ValueError,match='surviving'): KHR3HVEnv(c)
