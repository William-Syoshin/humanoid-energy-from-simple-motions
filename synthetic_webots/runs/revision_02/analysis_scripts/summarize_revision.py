"""Mechanical audit, unchanged-model checks, and standalone revision figures."""
from pathlib import Path
import argparse, csv, json, hashlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from motion_audit import audit_motion, THRESHOLDS
from webots_session import ROOT
from run_formal_revision import ORIGINAL

def readcsv(path):
    with Path(path).open() as f: return list(csv.DictReader(f))
def writejson(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
def writecsv(path, rows):
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args()
    run=args.run.resolve();out=run/'revision_report';out.mkdir(exist_ok=False)
    old=ORIGINAL/'runs/full_v2'
    design=json.loads((run/'design.json').read_text())
    tests=[s for s in design['trials'] if s['split']=='test']
    expected={'train':1000,'validation':90,'test':21,'control':3}
    assert {key:sum(s['split']==key for s in design['trials']) for key in expected}==expected
    assert json.loads((run/'analysis/audit.json').read_text())['counts']==expected
    assert json.loads((run/'completion.json').read_text())['newly_simulated_test_records']==21
    assert THRESHOLDS==json.loads((run/'motion_thresholds.json').read_text())
    auxiliary=json.loads((run/'auxiliary_hashes.json').read_text())
    assert all(hashlib.sha256((run/path).read_bytes()).hexdigest()==h for path,h in auxiliary.items())
    records=[]
    for s in tests:
        key='test_'+s['id'];a=audit_motion(run/'raw'/f'{key}.npz')
        a['trial']=key;records.append(a)
    writejson(out/'motion_audit.json',records);writejson(out/'thresholds.json',THRESHOLDS)
    old_records=[audit_motion(old/'raw'/f'test_{s["id"]}.npz') for s in tests]
    flat=[]
    for a,b in zip(records,old_records):
        flat.append(dict(trial=a['trial'],action=a['action'],seed=a['spec']['seed'],
            old_legacy_pass=b['legacy_endpoint_pass'],old_sustained_pass=b['revised_motion_pass'],
            new_legacy_pass=a['legacy_endpoint_pass'],new_sustained_pass=a['revised_motion_pass'],
            failed_gates=';'.join(a['failed_revised_gates']),**a['metrics']))
    writecsv(out/'mechanical_comparison.csv',flat)
    fits={}
    for model in ['M1','M2','M3','M4']:
        with np.load(old/'analysis/models'/f'{model}.npz') as a, np.load(run/'analysis/models'/f'{model}.npz') as b:
            differences={k:float(np.max(abs(a[k]-b[k]))) for k in ['cols','scale','beta','alpha']}
        fits[model]=dict(max_abs_differences=differences,all_arrays_identical=all(v==0 for v in differences.values()))
    writejson(out/'ridge_refit_comparison.json',fits)
    # Test unchanged tree fits on the same reused validation inputs, not on
    # changed test trajectories or serialisation metadata.
    sys.path.insert(0,str(run/'source/src'))
    from protocol import features
    import joblib
    old_tree=joblib.load(old/'analysis/models/M5.joblib')
    new_tree=joblib.load(run/'analysis/models/M5.joblib')
    tree_differences=[]
    for s in design['trials']:
        if s['split']!='validation':continue
        with np.load(run/'raw'/f'validation_{s["id"]}.npz',allow_pickle=False) as z:
            x=features(z)[3]
        tree_differences.append(float(np.max(abs(old_tree.predict(x)-new_tree.predict(x)))))
    tree_check=dict(validation_trials=len(tree_differences),max_abs_prediction_difference=max(tree_differences),
        all_validation_predictions_identical=all(v==0 for v in tree_differences))
    writejson(out/'M5_refit_comparison.json',tree_check)
    old_m=readcsv(old/'analysis/model_summary.csv');new_m=readcsv(run/'analysis/model_summary.csv')
    old_p=readcsv(old/'analysis/per_trial_metrics.csv');new_p=readcsv(run/'analysis/per_trial_metrics.csv')
    residual_share={}
    for name,base in [('original',old),('revised',run)]:
        rows=[r for r in readcsv(base/'analysis/trial_summary.csv') if r['split']=='test']
        residual_share[name]=100*sum(float(r['residual_energy_J']) for r in rows)/sum(float(r['energy_J']) for r in rows)
    writejson(out/'reference_residual_energy_share.json',residual_share)
    trial_ids={s['id'] for s in tests}
    assert len(trial_ids)==21
    for model in ['M1','M2','M3','M4','M5']:
        selected=[p for p in new_p if p['model']==model]
        assert len(selected)==21 and {p['trial'] for p in selected}==trial_ids
    plt.rcParams.update({'font.family':'sans-serif','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.1),layout='constrained')
    x=np.arange(5);width=.34
    for offset,summary,per,color,label in [(-width/2,old_m,old_p,'#999999','Original commanded motions'),(width/2,new_m,new_p,'#2166AC','Revised commanded motions')]:
        means=[float(r['mean_energy_error_pct']) for r in summary]
        sd=[np.std([float(p['energy_error_pct']) for p in per if p['model']==r['model']],ddof=1) for r in summary]
        axes[0].bar(x+offset,means,width,color=color,label=label)
        axes[0].errorbar(x+offset,means,yerr=sd,fmt='none',ecolor='black',elinewidth=.9,capsize=3)
        axes[1].bar(x+offset,[float(r['pooled_power_RMSE_W']) for r in summary],width,color=color,label=label)
    for ax in axes:
        ax.set_xticks(x,['M1','M2','M3','M4','M5']);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Absolute relative energy error (%)');axes[0].set_title('(a) Energy prediction: mean ± SD')
    axes[1].set_ylabel('Pooled power RMSE (W)');axes[1].set_title('(b) Power prediction')
    axes[1].legend(frameon=False,fontsize=8)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'old_vs_revised_models.{ext}',dpi=240)
    plt.close(fig)
    actions=['walk','stop','push_box','get_up','lie_down','turn_90','turn_180']
    names=dict(walk='歩行',stop='歩行→停止',push_box='箱押し',get_up='仰向けから起き上がり',lie_down='立位から横になる',turn_90='90°旋回',turn_180='180°旋回')
    lines=['# 補助シミュレーションの動作修正と再評価','',
        '旧実験を保持したまま、関節指令・動作段階を修正した別版の結果です。後半のPPO歩行実験とは別です。','',
        '## 動作の成立','',
        '|動作|旧・終了状態の基準|新・終了状態の基準|新・持続と停止を含む基準|新しい指令時間|',
        '|---|---:|---:|---:|---:|']
    for action in actions:
        a=[r for r in records if r['action']==action];b=[r for r in old_records if r['action']==action]
        lines.append(f'|{names[action]}|{sum(r["legacy_endpoint_pass"] for r in b)}/3|{sum(r["legacy_endpoint_pass"] for r in a)}/3|{sum(r["revised_motion_pass"] for r in a)}/3|{a[0]["spec"]["duration"]:.2f} s|')
    lines += ['',f'新しい持続基準の合格数：**{sum(r["revised_motion_pass"] for r in records)}/21**。失敗試行も全件を電力予測評価に含めています。','',
        '起立・旋回・停止では、最後1秒間の高さ28 cm以上・傾き20°以下、平均水平速度0.02 m/s未満、平均角速度0.1 rad/s未満、両足支持率95%以上を確認します。旋回は正方向の目標±10°、途中の傾き45°以下も必要です。横臥は初期直立と最後1秒の低い姿勢・静止を確認します。歩行・箱押しには停止を要求しません。足を擦らない歩行の成立を示す基準ではありません。','',
        '## 予測結果','',
        '|モデル|旧平均エネルギー誤差|新平均エネルギー誤差|新pooled電力RMSE|',
        '|---|---:|---:|---:|']
    for a,b in zip(old_m,new_m):
        lines.append(f'|{a["model"]}|{float(a["mean_energy_error_pct"]):.3f}%|{float(b["mean_energy_error_pct"]):.3f}%|{float(b["pooled_power_RMSE_W"]):.3f} W|')
    lines += ['','![旧・新の比較](old_vs_revised_models.png)','',
        'エネルギー誤差は21試行の等重み平均です。図のSDは7指令×3摂動間のばらつきであり、独立した21種類の動作や信頼区間を意味しません。電力RMSEは全時間サンプルの集約で、長い動作の重みが大きくなります。','',
        '## 変更と保持','',
        '- 物理条件、人工電力式、特徴量、M1〜M5の学習・モデル選択手順は維持。','- 1,000学習・90検証・3対照の原記録はSHA-256で一致するコピーを再利用。新しく実行したのは改訂後の21評価試行です。','- 初期状態を変えた開発用seed91001〜91010で動作を調整。候補選択に電力値・予測誤差は使っていません。','- 評価seed101/202/303は既知の旧評価seedを固定して再利用しており、新たな盲検評価とは呼びません。','- 起き上がり等の関節指令は、従来の一律±1.4 rad制限を超える場合があります。既存モーターの可動域内であることを別途監査します。','- 新しい全試行、開発中の失敗、凍結したソース、旧結果を保持。論文の現在のPDFは置換していません。','',
        f'人工電力の主要項はM4の入力に直接含まれます。線形特徴量で直接表現されない残差項は、新評価の総エネルギーの{residual_share["revised"]:.3f}%（旧{residual_share["original"]:.3f}%）です。動作が成立しても、実機の電気的消費の校正や実機省エネルギーの実証にはなりません。','',
        '旧新で予測対象の軌道と動作時間が異なります。誤差の変化は評価対象の変更を反映し、同じ動作に対して予測器自体が改善したことを意味しません。','',
        '`analysis/mechanical_checks.csv` とその `audit.json` の mechanical_pass は旧終了時判定です。新しい持続・停止条件を含む判定は `revision_report/motion_audit.json` を参照してください。','',
        '再現性・動画の監査は別ファイルに記録します。']
    (out/'RESULTS_JA.md').write_text('\n'.join(lines)+'\n')
    writejson(out/'summary.json',dict(old_legacy_pass=sum(r['legacy_endpoint_pass'] for r in old_records),
        new_legacy_pass=sum(r['legacy_endpoint_pass'] for r in records),new_revised_pass=sum(r['revised_motion_pass'] for r in records),
        total=21,all_ridge_fits_identical=all(v['all_arrays_identical'] for v in fits.values()),M5_refit_check=tree_check,
        residual_energy_share_pct=residual_share,original_models=old_m,revised_models=new_m))
    print(out)

if __name__=='__main__':main()
