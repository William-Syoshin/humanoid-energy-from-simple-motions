import copy
from scripts.finalize_distance_experiment import paired_analysis


def fixture():
    plan=dict(training_seeds=list(range(10,16)),analysis=dict(minimum_success_rate_per_seed=.9,
        minimum_paired_success_fraction=.9,maximum_ood_rate=.1,maximum_goal_overshoot_m=.005,
        bootstrap_seed=3,bootstrap_samples=100,alpha=.05))
    runs=[]
    for seed in plan['training_seeds']:
        for cond in ('task_only','R1','R2','R3'):
            rows=[dict(seed=20000+k,task_success=True,ood_rate=0.,negative_power_rate=0.,goal_overshoot_m=.001,
                       duration_s=20.,predicted_energy_J={m:100. if cond=='task_only' else 95. for m in ('R1','R2','R3')}) for k in range(10)]
            runs.append(dict(condition=cond,training_seed=seed,episodes=rows))
    return plan,runs


def test_holm_and_failure_gates():
    plan,runs=fixture()
    result=paired_analysis(runs,plan)
    assert all(c['R2_p_holm']==3/64 and c['primary_criterion_passed'] for c in result)
    # Cheap failed episodes must not be interpreted as savings.
    for r in runs:
        if r['condition']=='R1':
            for e in r['episodes'][:2]:e['task_success']=False;e['predicted_energy_J']={m:0. for m in ('R1','R2','R3')}
    result=paired_analysis(runs,plan)
    assert result[0]['mean_delta_energy_J']['R2']==-5
    assert not result[0]['primary_criterion_passed']
    assert result[1]['primary_criterion_passed']


def test_missing_success_is_not_zero_energy():
    plan,runs=fixture()
    for r in runs:
        if r['condition']=='R3':
            for e in r['episodes']:e['task_success']=False
    result=paired_analysis(runs,plan)
    assert result[2]['mean_delta_energy_J']['R2'] is None
    assert not result[2]['primary_criterion_passed']
