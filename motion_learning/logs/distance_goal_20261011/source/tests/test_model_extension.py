"""Guard model-specific primary outcomes and complete-seed analysis."""
import json
from pathlib import Path
import pytest
import rl.analyze_sweep as analysis
from rl.common import load_config


@pytest.mark.parametrize('model', ['R1','R2','R3'])
def test_primary_test_uses_selected_model(tmp_path, monkeypatch, model):
    design=dict(config=load_config(),training_seeds=list(range(10,16)),weights=[.1,.3,1.],evaluation_episodes=2,
                analysis=dict(primary_model=model,distance_equivalence=[.95,1.05],minimum_paired_pass_rate=.9,
                              bootstrap_samples=100,bootstrap_seed=42,minimum_replicates=6,
                              familywise_alpha=.05,meaningful_energy_reduction_pct=1.),jobs=[])
    runs=[]
    for seed in design['training_seeds']:
        for w in [0.,.1,.3,1.]:
            condition='task_only' if w==0 else model
            design['jobs'].append(dict(seed=seed,weight=w,condition=condition))
            # Selected model decreases in all six seeds. Other models have
            # one positive seed but negative means; these are sensitivity checks.
            energies={m:(100. if w==0 else (95. if m==model or seed!=15 else 101.)) for m in ('R1','R2','R3')}
            episodes=[]
            for e in [20000,20001]:
                episodes.append(dict(seed=e,distance_m=.24,mean_speed_m_s=.012,fallen=False,escaped=False,
                    task_success=True,ood_rate=0.,negative_power_rate=0.,heading_error_rad=0.,
                    joint_velocity_rms_rad_s=.1,tracking_error_rms_rad=.01,tilt_rms_rad=.01,
                    predicted_energy_J=energies,energy_per_m_J={m:v/.24 for m,v in energies.items()},
                    mean_gait_parameters=[2.8,.02,.003,.02,0.]))
            runs.append(dict(training_seed=seed,weight=w,condition=condition,episodes=episodes))
    (tmp_path/'design.json').write_text(json.dumps(design));(tmp_path/'results.json').write_text(json.dumps(runs))
    monkeypatch.setattr(analysis,'plot',lambda *args: None)
    out=analysis.analyze(tmp_path)
    assert out['primary_model']==model
    for c in out['coefficients']:
        assert c[model+'_p_holm']==3/64
        assert c['primary_criterion_passed'] and c['practical_criterion_passed']
    # Failed performance cannot be removed to improve the conclusion.
    runs[1]['episodes'][0]['task_success']=False
    (tmp_path/'results.json').write_text(json.dumps(runs))
    out=analysis.analyze(tmp_path)
    assert not out['coefficients'][0]['primary_criterion_passed']
    runs.pop()
    (tmp_path/'results.json').write_text(json.dumps(runs))
    with pytest.raises(ValueError,match='Incomplete'):
        analysis.analyze(tmp_path)
