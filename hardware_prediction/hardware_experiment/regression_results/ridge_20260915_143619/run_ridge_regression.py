import csv, json, sys, shutil, hashlib
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import numpy as np

ROOT=Path(r'C:\Users\BIRL-\OneDrive\Desktop\KHR_3HV_Ver.7.10_jp\hardware_experiment')
sys.path.insert(0,str(ROOT))
import analysis as a

def run():
    out=ROOT/'regression_results'/datetime.now().strftime('ridge_%Y%m%d_%H%M%S')
    data=out/'aligned';data.mkdir(parents=True)
    shutil.copy2(__file__,out/'run_ridge_regression.py')
    shutil.copy2(ROOT/'analysis.py',out/'analysis.py')
    rows=[];provenance=[]
    for session,split,count in [('session_v3_uniform_10hz','train',116),('session_v3_validation_10hz','validation',23)]:
        base=ROOT/session/'regression_data'
        source=list(csv.DictReader((base/'manifest.csv').open(encoding='utf-8-sig')))
        assert len(source)==count
        for r in source:
            p=base/'aligned'/r['file'];qc=Path(r['source_qc'])
            assert r['split']==split and r['processed_usable']=='1'
            q=json.loads(qc.read_text(encoding='utf-8'))
            assert q['usable'] is True and a.digest(p)==q['aligned_sha256']==r['sha256']
            name=r['trial_id']+'.csv'
            shutil.copy2(p,data/name);shutil.copy2(qc,(data/name).with_suffix('.qc.json'))
            rows.append(dict(trial_id=r['trial_id'],file=name,split=split,task=r['level'],level=r['level'],attempted='1',usable='1',success=r['mechanical_success']))
            provenance.append(dict(trial_id=r['trial_id'],source=str(p),sha256=a.digest(p),mechanical_success=r['mechanical_success']))
    assert len({r['trial_id'] for r in rows})==139
    a.write_csv(out/'trials.csv',rows);a.write_json(out/'input_provenance.json',provenance)
    args=SimpleNamespace(manifest=str(out/'trials.csv'),data=str(data),out=str(out/'models'))
    a.fit(args)
    ds,schema=a.load(rows,data,['train','validation'])
    # Confirm the model uses the velocities and errors actually exported by DAQ.
    for d in ds:
        raw=list(csv.DictReader(d['path'].open()))
        dq=np.array([[float(r[f'dq_{i}']) for i in range(17)] for r in raw])
        err=np.array([[float(r[f'e_{i}']) for i in range(17)] for r in raw])
        assert np.allclose(abs(dq),d['X']['R1'][:,1:18],atol=1e-8)
        assert np.allclose(abs(err),d['X']['R1'][:,35:52],atol=1e-8)
    frozen=json.loads((out/'models'/'frozen.json').read_text())
    params=dict(np.load(out/'models'/'frozen_models.npz'))
    metrics=[];summaries=[];grid=[]
    tr=[d for d in ds if d['meta']['split']=='train'];va=[d for d in ds if d['meta']['split']=='validation']
    for model in a.MODELS:
        sc=(params[model+'_scale'],params[model+'_coef'])
        for alpha in [.001,.1,10,100]:
            grid.append(dict(model=model,alpha=alpha,validation_RMSE_W=a.rmse(va,model,a.fit_one(tr,model,alpha))))
        for split,group in [('train',tr),('validation',va)]:
            mm=[]
            for d in group:
                pred=a.predict(d,model,sc)
                m=dict(split=split,**a.metrics(d,model,pred));metrics.append(m);mm.append(m)
            summaries.append(dict(model=model,split=split,alpha=frozen['models'][model]['alpha'],trials=len(group),power_RMSE_W=a.rmse(group,model,sc),mean_energy_error_pct=float(np.mean([m['energy_error_pct'] for m in mm])),worst_energy_error_pct=max(m['energy_error_pct'] for m in mm)))
    a.write_csv(out/'model_comparison.csv',summaries);a.write_csv(out/'per_trial_metrics.csv',metrics);a.write_csv(out/'alpha_search.csv',grid)
    names=['intercept']+[f'{term}:{n}' for term in ['abs_velocity','velocity_squared','abs_error','error_squared'] for n in a.NAMES]+[f'abs_error_product:{a.NAMES[i]}:{a.NAMES[j]}' for i,j in a.EDGES]+['1_minus_cos_tilt','1_minus_cos_tilt_squared']+['support:'+s for s in a.CATEGORIES[:-1]]
    coefficients=[]
    for model in a.MODELS:
        for name,s,c in zip(names,params[model+'_scale'],params[model+'_coef']):
            coefficients.append(dict(model=model,feature=name,training_RMS_scale=float(s),scaled_coefficient=float(c),original_units_coefficient=float(c/s)))
    a.write_csv(out/'coefficients.csv',coefficients)
    traces=out/'validation_predictions';traces.mkdir()
    for d in va:
        preds={n:a.predict(d,n,(params[n+'_scale'],params[n+'_coef'])) for n in a.MODELS}
        a.write_csv(traces/(d['meta']['trial_id']+'.csv'),[dict(t_s=float(t),actual_W=float(d['p'][i]),**{n+'_W':float(preds[n][i]) for n in a.MODELS}) for i,t in enumerate(d['t'])])
    plt=a.plots_import();fig,axes=plt.subplots(1,2,figsize=(10,4))
    vm=[s for s in summaries if s['split']=='validation']
    axes[0].bar([s['model'] for s in vm],[s['power_RMSE_W'] for s in vm]);axes[0].set(ylabel='Validation power RMSE (W)')
    for n in ['R0','R3']:
        rr=[m for m in metrics if m['split']=='validation' and m['model']==n]
        axes[1].scatter([m['actual_J'] for m in rr],[m['predicted_J'] for m in rr],label=n,alpha=.7)
    lo=min(m['actual_J'] for m in metrics if m['split']=='validation')*.95;hi=max(m['predicted_J'] for m in metrics if m['split']=='validation')*1.05
    axes[1].plot([lo,hi],[lo,hi],'k--');axes[1].set(xlabel='Measured energy (J)',ylabel='Predicted energy (J)');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'validation_comparison.png',dpi=180);plt.close(fig)
    text=['# 実機リッジ回帰結果','',f"検証電力RMSEによる選択モデル：{frozen['chosen']}",'','学習116試行、検証23試行。全試行81行。各試行の重みは等しい。目的変数はpower_W。係数とRMSスケールは学習データだけから推定し、正則化係数は事前設定の0.001、0.1、10、100から選択。検証後に学習＋検証で再学習していない。','', '|モデル|正則化係数|検証電力RMSE (W)|検証平均エネルギー誤差 (%)|','|---|---:|---:|---:|']
    for s in vm:text.append(f"|{s['model']}|{s['alpha']}|{s['power_RMSE_W']:.4f}|{s['mean_energy_error_pct']:.3f}|")
    text+=['','R0：定常電力。R1：17関節の角速度・追従誤差の絶対値と二乗。R2：身体の接続関係に基づく誤差積。R3：加速度センサー由来の傾きと支持カテゴリを追加。角度は追従誤差と角速度を通じて利用。電圧・電流・電力は説明変数に含めない。','','CSVのdq/eと解析特徴量の一致を139試行で検証済み。前処理は中心5点平滑化を使うオフライン推定。加速度由来の傾きは動的姿勢推定の保証ではない。支持カテゴリは既存収録ラベルを保持。試行の機械的成功欄はpendingを保持し、自動的に成功扱いしない。','','この表は正則化とモデル選択に使用した検証データ上の結果。歩行・起き上がりなど未使用の複合動作に対する精度は、今後の独立テストで評価する。学習時と検証時の電源・温度・時間変化の影響も未分離。','','再現：python run_ridge_regression.py（元のセッションを参照し、新しい日時フォルダへ保存）。']
    (out/'RESULTS_JA.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(out),chosen=frozen['chosen'],validation=vm),indent=2))

if __name__=='__main__':run()
