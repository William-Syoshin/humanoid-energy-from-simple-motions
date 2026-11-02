"""Scientific invariants: frozen features, causal delay and terminal accounting."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from energy.power import FrozenPower, feature_matrix
from rl.common import ROOT


@pytest.mark.parametrize('length', [2,3,4,5,7,21,81])
def test_delayed_energy_matches_offline_including_terminal(length):
    p = FrozenPower()
    rng = np.random.default_rng(912)
    states = [dict(q=rng.normal(0, .1, 17).tolist(), cmd=rng.normal(0, .1, 17).tolist(),
                   accel=[.1*i, .2, 9.81]) for i in range(length)]
    accumulated = dict.fromkeys(('R1','R2','R3'), 0.)
    ood_duration = 0.
    for i, s in enumerate(states):
        delta, _, ood_s, _ = p.push(s, terminal=i==length-1)
        if i < 3 and i != length-1:
            assert p.next_index == 0
        for m in accumulated: accumulated[m] += delta[m]
        ood_duration += ood_s
    x = feature_matrix(p.q, p.cmd, p.tilt, p.edges)
    for m,n in [('R1',69),('R2',89),('R3',95)]:
        values = (x[:,:n]/p.params[m+'_scale'])@p.params[m+'_coef']
        np.testing.assert_allclose([d[m] for d in p.predictions], values, rtol=1e-12, atol=1e-12)
        assert accumulated[m] == pytest.approx(np.trapezoid(values, dx=.1), rel=1e-12)
    assert p.scored_samples == length
    assert ood_duration == pytest.approx(p.summary()['ood_rate']*(length-1)*.1)


def test_features_and_predictions_match_archived_hardware():
    paper = ROOT.parent/'humanoid-energy-paper'
    source = paper/'hardware_experiment/analysis.py'
    if not source.exists():
        pytest.skip('Original paper checkout required for independent reference comparison')
    spec = importlib.util.spec_from_file_location('independent_reference', source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    ridge = paper/'hardware_experiment/regression_results/ridge_20260915_143619'
    rows = reference.manifest(ridge/'trials.csv')
    power = FrozenPower()
    import csv
    # Separate validation trial and two training trials, no fitting performed.
    selected = [next(r for r in rows if r['split']=='validation')]+[r for r in rows if r['split']=='train'][:2]
    for row in selected:
        d = reference.read_trial(row, ridge/'aligned', power.frozen['schema'])
        with d['path'].open(encoding='utf-8-sig') as stream:
            records = list(csv.DictReader(stream))
        q = np.array([[float(r[f'q_{i}']) for i in range(17)] for r in records])
        cmd = np.array([[float(r[f'cmd_{i}']) for i in range(17)] for r in records])
        tilt = np.array([float(r['tilt_rad']) for r in records])
        support = records[0]['support']
        assert all(r['support']==support for r in records)
        x = feature_matrix(q, cmd, tilt, power.edges, support, times=d['t'])
        np.testing.assert_allclose(x, d['X']['R3'], rtol=1e-10, atol=1e-12)
        for m,n in [('R1',69),('R2',89),('R3',95)]:
            expected = reference.predict(d, m, (power.params[m+'_scale'],power.params[m+'_coef']))
            np.testing.assert_allclose((x[:,:n]/power.params[m+'_scale'])@power.params[m+'_coef'], expected, atol=1e-10)
