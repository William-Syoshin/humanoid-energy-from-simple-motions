"""Audit both model families and report all outcomes without editing old R2 results."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rl.analyze_sweep import holm
from rl.run_sweep import atomic_json, digest, verify_source


def read(path): return json.loads(Path(path).read_text())
def mean(rows,key): return sum(float(r[key]) for r in rows)/len(rows)


def audit_design(child, reference):
    design=read(child/'design.json');old=read(reference/'design.json')
    assert read(child/'source_manifest.json')==read(reference/'source_manifest.json')
    verify_source(child)
    runs=read(child/'results.json')
    import yaml
    for r in runs:
        run=Path(r['run']);cfg=yaml.safe_load((run/'config.yaml').read_text())
        expected=json.loads(json.dumps(design['config']))
        expected.update(seed=r['training_seed'],condition=r['condition'])
        expected['energy']['weight']=r['weight']
        assert cfg==expected, 'Training configuration differs from design'
        trained=read(run/'training_summary.json')
        tr=cfg['training'];budget=((tr['total_timesteps']+tr['n_steps']-1)//tr['n_steps'])*tr['n_steps']
        assert trained['timesteps']==budget and trained['actual_device']=='cpu'
        # Runtime versions must also agree with the reference run.
        old_run=reference/f"task_only-lambda0-seed{r['training_seed']}"
        assert read(run/'provenance.json')['versions']==read(old_run/'provenance.json')['versions']
    return dict(status='passed',same_archived_source=True,all_run_configs_match=True,
                runtime_versions_match_reference=True,runs=len(runs))


def finalize_family(root, model, project, smoke):
    child=root/model;reference=Path(read(root/'design.json')['reference_experiment'])
    if not read(child/'completion.json')['all_complete']: raise RuntimeError('Incomplete '+model)
    design_audit=audit_design(child,reference)
    atomic_json(child/'design_audit.json',design_audit)
    for name,expected in read(child/'analysis_source/manifest.json').items():
        assert digest(child/'analysis_source'/name)==expected
    env=dict(os.environ,PYTHONPATH=str(child/'source'))
    for script in ('audit_sweep.py','analyze_sweep.py'):
        with (child/(script+'.log')).open('w') as out:
            subprocess.run([sys.executable,str(child/'analysis_source'/script),str(child)],
                           cwd=child/'source',env=env,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=900)
    summary=read(child/'sweep_summary.json');runs=read(child/'results.json');design=read(child/'design.json')
    rows=list(csv.DictReader((child/'seed_metrics.csv').open()))
    destination=root/'deliverables' if smoke else project
    docs=destination/'docs';docs.mkdir(parents=True,exist_ok=True)
    stem=f'{root.name}_{model}'
    assets=docs/stem;assets.mkdir(exist_ok=True)
    export=destination/'models'/root.name/model;export.mkdir(parents=True,exist_ok=True)
    exported=[]
    for r in runs:
        if r['training_seed']!=10: continue
        name='task_only_seed10.zip' if r['condition']=='task_only' else f"{model}_lambda{r['weight']:g}_seed10.zip"
        dst=export/name;shutil.copy2(Path(r['run'])/'final.zip',dst)
        exported.append(dict(path=str(dst),source=r['run'],sha256=digest(dst)))
    atomic_json(export/'manifest.json',exported)
    for name in ('sweep_summary.json','seed_metrics.csv','paired_seed_effects.csv','evaluation_pair_gates.csv',
                 'sweep_overview.png','paired_energy_effects.png','audit.json','design_audit.json',
                 'energy_contributions.csv','design.json','source_manifest.json'):
        shutil.copy2(child/name,assets/name)
    passed=[s['weight'] for s in summary['coefficients'] if s['primary_criterion_passed']]
    lines=[f'# {model}電力係数比較の結果','',f'対象実験: `{root.name}`。独立学習の単位はseed。',
           f"{summary['total_training_runs']}学習・{summary['total_evaluation_episodes']}評価を完了。主要基準を満たす係数: {passed or 'なし'}。",'',
           '実機の消費電力低減を実証した結果ではない。学習済み電力予測器の下での評価である。',
           '全seedと失敗を含める。低速化や転倒による低いJを成功とは扱わない。','',
           '## 実験条件','',
           f"- Task-onlyと{model} λ=0.1/0.3/1.0。学習seed: {design['training_seeds']}。",
           f"- 各{design['config']['training']['total_timesteps']:,}step指定。PPOロールアウト単位で切り上げ。",
           '- 前回R2と同一の凍結学習・環境・物理・予測器コードと依存バージョンを使用。',
           '- 20秒歩行、目標速度0.012m/s。20cm到達課題やローリングへの変更はしていない。',
           '- R1/R2/R3全予測器で再評価。Task-onlyはこの実験内で改めて学習。','',
           '## 全seedの平均','',
           '|条件|距離 m|速度 m/s|成功率|転倒率|OOD率|R1 J|R2 J|R3 J|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for w in [0.]+design['weights']:
        g=[r for r in rows if float(r['weight'])==w]
        label='Task-only' if w==0 else f'{model} λ={w:g}'
        lines.append(f"|{label}|{mean(g,'distance_m'):.6f}|{mean(g,'mean_speed_m_s'):.6f}|{mean(g,'task_success'):.1%}|{mean(g,'fallen'):.1%}|{mean(g,'ood_rate'):.2%}|"+
                     '|'.join(f'{mean(g,m+"_J"):.3f}' for m in ('R1','R2','R3'))+'|')
    lines += ['',f'![結果]({stem}/sweep_overview.png)','',
              '## 対応するTask-onlyからの変化','',
              '|λ|R1変化 %|R2変化 %|R3変化 %|性能同等の評価対|主モデル低下seed数|主モデルHolm p|主要基準|',
              '|---|---:|---:|---:|---:|---:|---:|---|']
    for s in summary['coefficients']:
        lines.append(f"|{s['weight']:g}|"+'|'.join(f"{s[m]['mean']:+.3f}" for m in ('R1','R2','R3'))+
                     f"|{s['equivalent_pairs']}/{s['total_pairs']}|{s[model]['decreasing_training_seeds']}/{summary['n_training_seeds']}|{s[model+'_p_holm']:.5f}|{'通過' if s['primary_criterion_passed'] else '未通過'}|")
    lines += ['',f'![seed別比較]({stem}/paired_energy_effects.png)','',
              '主要判定は前回と同じ基準を主モデルへ適用する。全6seedで評価対の90%以上が性能基準を満たし、',
              '各seedの主モデル予測エネルギーが低下、3係数のHolm補正p<0.05、他の2モデルの平均も低下を要求。',
              '距離比は0.95〜1.05、平均速度0.0108m/s以上、OOD率10%以下、転倒・逸脱・負電力なし。',
              '新規2モデル全6比較の追加Holm補正は統合レポートに記載する。',
              '6seedのbootstrap区間は記述的な推定。3つの電力モデルは同じ実機データに由来し、独立の実機検証ではない。','',
              '## 保存物','',f'- 監査: [{stem}/audit.json]({stem}/audit.json)。全軌道の予測値と報酬を再計算。',
              f'- seed別の全結果: [{stem}/seed_metrics.csv]({stem}/seed_metrics.csv)。',
              f'- 各学習モデルと生ログ: `{child}`。',f'- 固定seed10の再生用モデル: `{export}`。',
              '- source_manifest.jsonの全項目が前回R2のアーカイブと一致。','']
    if smoke: lines.insert(2,'**動作確認用の小規模テスト。研究結果には含めない。**')
    report=docs/(stem+'_RESULTS.md');report.write_text('\n'.join(lines))
    with tarfile.open(child/'frozen_source.tar.gz','w:gz') as tar:
        for name in read(child/'source_manifest.json'): tar.add(child/'source'/name,arcname='source/'+name)
        tar.add(child/'analysis_source',arcname='analysis_source')
        for name in ('design.json','source_manifest.json','runtime_freeze.txt'):tar.add(child/name,arcname=name)
    atomic_json(child/'finalization.json',dict(status='complete',report=str(report),exported=exported,
                source_archive_sha256=digest(child/'frozen_source.tar.gz'),primary_passing_weights=passed,
                finished_utc=datetime.now(timezone.utc).isoformat()))
    return summary,rows,report


def finish(root,project):
    design=read(root/'design.json');smoke=design['smoke_test']
    if not read(root/'completion.json')['all_complete']:raise RuntimeError('Incomplete model extension')
    completed={m:finalize_family(root,m,project,smoke) for m in ('R1','R3')}
    hypotheses=[(m,s) for m in ('R1','R3') for s in completed[m][0]['coefficients']]
    adjusted=holm([s[m+'_p_one_sided'] for m,s in hypotheses])
    global_tests=[dict(model=m,weight=s['weight'],p_one_sided=s[m+'_p_one_sided'],p_holm_all_six=p)
                  for (m,s),p in zip(hypotheses,adjusted)]
    atomic_json(root/'six_comparison_correction.json',global_tests)
    reference=Path(design['reference_experiment'])
    old=list(csv.DictReader((reference/'seed_metrics.csv').open()))
    groups=[('R1',completed['R1'][1]),('R2 (previous)',old),('R3',completed['R3'][1])]
    lines=['# R1・R3追加学習と前回R2の比較','',
           f"{design['planned_training_runs']}学習・{design['planned_evaluation_episodes']}評価の追加実験。",'',
           '前回と同じ20秒歩行課題。全条件で同一の凍結コードと設定を使用した。',
           'R1/R3でTask-onlyを再学習したが同じseedなので、新しい独立seedが増えたとは数えない。',
           'R2は前回の結果を参照した記述的比較。予測エネルギーは実測電力の代わりではない。','',
           '|学習モデル|λ|距離 m|成功率|R1予測 J|R2予測 J|R3予測 J|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for label,rows in groups:
        for w in [0.,.1,.3,1.]:
            g=[r for r in rows if float(r['weight'])==w]
            lines.append(f"|{label}|{w:g}|{mean(g,'distance_m'):.6f}|{mean(g,'task_success'):.1%}|"+
                         '|'.join(f'{mean(g,m+"_J"):.3f}' for m in ('R1','R2','R3'))+'|')
    lines+=['','## 各実験の主要基準','']
    for m in ('R1','R3'):
        passed=[s['weight'] for s in completed[m][0]['coefficients'] if s['primary_criterion_passed']]
        lines += [f"- {m}: 通過係数 {passed or 'なし'}。[詳細]({completed[m][2].name})。"]
    lines+=['','## 新規6比較全体の追加補正','',
            '各モデル内の3係数補正に加え、R1/R3の6比較を一つの族としてHolm補正した値を併記する。',
            '6seedの片側符号反転検定の最小pは1/64であるため、全6比較補正では5%未満に達しない。',
            'この小標本の限界を、事後的なseed追加や有利な比較の選別で回避しない。','',
            '|モデル|λ|未補正p|6比較Holm p|','|---|---:|---:|---:|']
    for t in global_tests:lines.append(f"|{t['model']}|{t['weight']:g}|{t['p_one_sided']:.5f}|{t['p_holm_all_six']:.5f}|")
    # Compare fresh baselines with the archived one. Equality is a reproducibility
    # observation, not extra independent replication.
    baseline=[]
    for m in ('R1','R3'):
        rows=completed[m][1]
        for r in rows:
            if float(r['weight'])!=0:continue
            ref=next(x for x in old if int(x['training_seed'])==int(r['training_seed']) and float(x['weight'])==0)
            baseline.append(dict(model_family=m,seed=int(r['training_seed']),
                differences={k:float(r[k])-float(ref[k]) for k in ('distance_m','mean_speed_m_s','R1_J','R2_J','R3_J')}))
    atomic_json(root/'baseline_reproducibility.json',baseline)
    lines += ['', '## 検証','', '- 全480評価の予測値・報酬、TensorBoard、学習量、依存バージョンを監査。' if not smoke else '- 動作確認用の小規模実行。',
              '- Task-onlyの再学習差は baseline_reproducibility.json に保存。','']
    docs=(root/'deliverables' if smoke else project)/'docs'
    report=docs/(root.name+'_RESULTS.md');report.write_text('\n'.join(lines))
    if not smoke:
        readme=project/'README.md';content=readme.read_text()
        link=f'[R1・R3電力係数比較の結果](docs/{report.name})'
        if link not in content:readme.write_text(content+'\n\n'+link+'\n')
    atomic_json(root/'finalization.json',dict(status='complete',report=str(report),
                reports={m:str(completed[m][2]) for m in ('R1','R3')},
                finished_utc=datetime.now(timezone.utc).isoformat()))
    print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--project',type=Path,required=True)
    a=p.parse_args();finish(a.root.resolve(),a.project.resolve())
