"""Audit, analyze and write a readable report when the fixed sweep completes."""
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
import time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finish(root):
    completion=json.loads((root/'completion.json').read_text())
    if not completion['all_complete']:
        raise RuntimeError('Incomplete sweep. Failed runs require diagnosis; no success report generated.')
    manifest=json.loads((root/'source_manifest.json').read_text())
    assert all(sha(root/'source'/p)==v for p,v in manifest.items())
    analysis=root/'analysis_source'
    hashes=json.loads((analysis/'manifest.json').read_text())
    assert all(sha(analysis/p)==v for p,v in hashes.items())
    env=dict(os.environ,PYTHONPATH=str(root/'source'))
    for script in ('audit_sweep.py','analyze_sweep.py'):
        with (root/(script+'.log')).open('w') as output:
            subprocess.run([sys.executable,str(analysis/script),str(root)],cwd=root/'source',
                           env=env,stdout=output,stderr=subprocess.STDOUT,check=True,timeout=600)
    result=json.loads((root/'sweep_summary.json').read_text())
    runs=json.loads((root/'results.json').read_text())
    rows=list(csv.DictReader((root/'seed_metrics.csv').open()))
    export=ROOT/'models/coefficient_sweep_20261008'
    export.mkdir(parents=True,exist_ok=True)
    exported=[]
    # Fixed seed 10 for all weights: no best-seed or best-checkpoint selection.
    for run in runs:
        if run['training_seed']!=10: continue
        name='task_only_seed10.zip' if run['condition']=='task_only' else f"R2_lambda{run['weight']:g}_seed10.zip"
        target=export/name;shutil.copy2(Path(run['run'])/'final.zip',target)
        exported.append(dict(file=name,source=run['run'],sha256=sha(target),training_seed=10,
                             condition=run['condition'],weight=run['weight']))
    (export/'manifest.json').write_text(json.dumps(exported,indent=2)+'\n')
    assets=ROOT/'docs/coefficient_sweep_20261008'
    assets.mkdir(exist_ok=True)
    for name in ('sweep_summary.json','seed_metrics.csv','paired_seed_effects.csv','evaluation_pair_gates.csv',
                 'sweep_overview.png','paired_energy_effects.png','audit.json','energy_contributions.csv','design.json'):
        shutil.copy2(root/name,assets/name)
    shutil.copy2(root/'source_manifest.json',assets/'source_manifest.json')
    with tarfile.open(root/'frozen_source.tar.gz','w:gz') as archive:
        for name in manifest: archive.add(root/'source'/name,arcname='source/'+name)
        archive.add(root/'analysis_source',arcname='analysis_source')
        for name in ('design.json','source_manifest.json','runtime_freeze.txt'):
            archive.add(root/name,arcname=name)
    def mean(group,key): return sum(float(x[key]) for x in group)/len(group)
    def formatted_effect(x):
        return f"{x['mean']:+.3f}% [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"
    lines=['# 電力係数と複数seedの比較結果（2026-10-08）','',
           '## 結論','']
    passing=[s for s in result['coefficients'] if s['primary_criterion_passed']]
    if passing:
        weights='、'.join(str(s['weight']) for s in passing)
        lines += [f'係数λ={weights}は、実行前に固定した主要判定基準を満たした。',
                  'このWebots設定と固定電力予測器の下で、同等の前進性能を保ちながら、',
                  '予測総エネルギーの小さい方策を複数seedで獲得したことを支持する。',
                  'これは実機で測定した省エネルギーの証明ではない。']
    else:
        lines += ['**今回の24学習では、事前に固定した主要判定基準を満たす係数はなかった。**',
                  '予測エネルギーだけが低下した条件や、一部のseedだけが成功した条件をもって、',
                  '「同等の歩行性能を維持した省エネルギー歩行を再現できた」とは結論づけない。',
                  '係数比較を完了した結果として、失敗・効果の小ささ・再現性も報告する。']
    practical=[s['weight'] for s in result['coefficients'] if s['practical_criterion_passed']]
    lines += ['',f'主要基準に加えて、事前の目安「R2平均1%以上低下」も満たした係数：{practical if practical else "なし"}。',
              '1%は任意の実験上の目安で、電力モデルの誤差を上回る保証ではない。','',
              '## 実行条件','',
              '- Task-only、R2 λ=0.1・0.3・1.0の4条件 × 学習seed 10〜15の6個 = 24学習。',
              '- 各100,000ステップ指定、実際は各100,352ステップ（合計2,408,448）。',
              '- 各方策を評価seed 20000〜20009で10回、合計240評価。独立学習反復は各条件6個。',
              '- 最終モデルのみ評価。良いcheckpointやseedだけを選んでいない。',
              '- 目標速度0.012m/s、20秒、既存の歩行生成器・モデル・物理・報酬は固定。',
              '- 実行前の[プロトコル](COEFFICIENT_SWEEP_PROTOCOL.md)とソースコピーを保存。','',
              '## 全seedを含む平均','',
              '|条件|速度 [m/s]|距離 [m]|転倒率|タスク成功率|OOD率|R2予測J|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for weight in [0.,.1,.3,1.]:
        group=[r for r in rows if float(r['weight'])==weight]
        label='Task-only' if weight==0 else f'R2 λ={weight:g}'
        lines.append(f"|{label}|{mean(group,'mean_speed_m_s'):.6f}|{mean(group,'distance_m'):.6f}|"
                     f"{mean(group,'fallen'):.1%}|{mean(group,'task_success'):.1%}|{mean(group,'ood_rate'):.2%}|{mean(group,'R2_J'):.3f}|")
    lines += ['', '早期終了がある条件の小さいJは、省エネルギー成功とは扱わない。',
              '![全条件の比較](coefficient_sweep_20261008/sweep_overview.png)','',
              '## 対応するTask-onlyからの予測総エネルギー変化','',
              '各seedで評価10回を平均した後、対応するTask-onlyに対する変化率を計算。',
              '以下は6seedの平均とbootstrap 95%区間。負値が予測エネルギーの低下。','',
              '|λ|R1変化|R2変化|R3変化|距離±5%などの基準を満たす対|R2 Holm補正p|主要基準|',
              '|---|---|---|---|---:|---:|---|']
    for s in result['coefficients']:
        lines.append(f"|{s['weight']:g}|{formatted_effect(s['R1'])}|{formatted_effect(s['R2'])}|{formatted_effect(s['R3'])}|"
                     f"{s['equivalent_pairs']}/{s['total_pairs']}|{s['R2_p_holm']:.5f}|{'通過' if s['primary_criterion_passed'] else '未通過'}|")
    lines += ['', '![seedごとの変化](coefficient_sweep_20261008/paired_energy_effects.png)','',
              '6seedは小標本。bootstrap区間は記述的な推定であり、符号反転検定は帰無仮説下の符号交換可能性を仮定する。',
              '3係数の検定をHolm補正した。R1/R3は同じデータで作られた関連モデルで、独立の実機検証ではない。','',
              '## 判定の内訳','']
    for s in result['coefficients']:
        lines += [f"- λ={s['weight']:g}：全seedで同等性能基準={s['all_seeds_performance_gate']}、"
                  f"全seedでR2低下={s['all_seeds_R2_decrease']}、R2低下seed={s['R2']['decreasing_training_seeds']}/6。"]
    lines += ['', '## 検証と保存物','',
              '- 全240軌道の予測値とエネルギーを、保存した関節データからオフラインで再計算して照合。',
              '- 全24学習のTensorBoard値が有限で、Task-onlyの電力報酬が0、R2条件の電力報酬が非正であることを確認。',
              '- 電力項のepisode総和と −λ × 正値化した予測エネルギーの一致を確認。',
              '- 学習前の自動テスト12件、実Webots check_env/reset/random action、集計用テスト3件が成功。',
              '- [seedごとの指標CSV](coefficient_sweep_20261008/seed_metrics.csv)、[対応する効果量CSV](coefficient_sweep_20261008/paired_seed_effects.csv)。',
              '- [検証記録](coefficient_sweep_20261008/audit.json)、[モデル各項の寄与](coefficient_sweep_20261008/energy_contributions.csv)。',
              '- 各項の寄与は回帰式の符号付き寄与であり、実測した部品ごとの消費電力ではない。',
              '- 生ログ・全24モデル・凍結コードは `logs/coefficient_sweep_20261008/`。','',
              '## 学習済みモデルを再生する','',
              '比較用に全条件の学習seed 10を固定してエクスポートした。最高性能seedを選んだものではない。',
              'プロジェクトの `khr3hv_rl` に移動して実行する。','',
              '```bash',
              '.venv/bin/python rl/evaluate.py --model models/coefficient_sweep_20261008/task_only_seed10.zip --gui --realtime --episodes 1',
              '.venv/bin/python rl/evaluate.py --model models/coefficient_sweep_20261008/R2_lambda0.3_seed10.zip --gui --realtime --episodes 1',
              '```','',
              '同じ場所にλ=0.1と1のモデルも保存。今回の夜間実行はiCloud退避を避けるため、',
              '`/Users/yuchenzhou/.cache/khr3hv-rl/runtime-20261008/bin/python` を使用した。',
              '依存バージョンはrun内の `runtime_freeze.txt` に保存している。','',
              '## 論文での扱い','',
              '本実験は、単純動作から全身動作の電力を予測するという主結果の応用検証である。',
              '電力予測器の実機からシミュレーターへの転用、サーボ応答、足滑り、自己衝突、',
              '関節対応の未校正という制約を伴う。未知の全身動作に対する予測精度は、',
              '元の実測との比較で別に示す必要がある。ここでの予測値の最小化で代用しない。','']
    report=ROOT/'docs/COEFFICIENT_SWEEP_RESULTS.md'
    report.write_text('\n'.join(lines))
    final=dict(status='complete',finished_utc=datetime.now(timezone.utc).isoformat(),
               report=str(report),source_archive_sha256=sha(root/'frozen_source.tar.gz'),
               primary_passing_weights=[s['weight'] for s in passing],
               practical_passing_weights=practical,exported=exported)
    (root/'finalization.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps(final,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--watch',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    if a.watch:
        last=None
        while not (root/'completion.json').exists():
            states=[json.loads(f.read_text()) for f in (root/'status').glob('*.json')]
            now=(sum(s['status']=='complete' for s in states),sum(s['status']=='failed' for s in states))
            if now!=last:
                print(dict(completed=now[0],failed=now[1],time_utc=datetime.now(timezone.utc).isoformat()),flush=True)
                last=now
            time.sleep(30)
    finish(root)
