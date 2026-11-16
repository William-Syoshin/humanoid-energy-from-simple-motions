import numpy as np
import pytest
from rl.analyze_sweep import holm, interval, paired_gates, sign_flip_p


def test_small_sample_sign_flip_and_multiple_comparison_correction():
    assert sign_flip_p([-1.]*6)==1/64
    assert sign_flip_p([0.]*6)==1.
    assert sign_flip_p([1.]*6)==1.
    assert holm([1/64]*3)==[3/64]*3
    assert holm([.04,.001,.02])==pytest.approx([.04,.003,.04])


def test_bootstrap_samples_training_seed_means():
    assert interval([-2.]*6)['ci95']==[-2.,-2.]
    x=interval([-1.,-2.,-3.,-4.,-5.,-6.])
    assert x['n_training_seeds']==6 and x['mean']==-3.5
    assert x==interval([-1.,-2.,-3.,-4.,-5.,-6.])


def test_performance_gate_does_not_reward_failure_or_slowing():
    reference=dict(task_success=True,ood_rate=.01,negative_power_rate=0.,distance_m=.24)
    a=dict(distance_equivalence=[.95,1.05])
    assert paired_gates(reference,reference,a)['equivalent']
    for changes in [dict(task_success=False),dict(ood_rate=.11),dict(negative_power_rate=.01),dict(distance_m=.20)]:
        assert not paired_gates(reference,dict(reference,**changes),a)['equivalent']
    faster=paired_gates(reference,dict(reference,distance_m=.27),a)
    assert not faster['equivalent'] and faster['noninferior']
