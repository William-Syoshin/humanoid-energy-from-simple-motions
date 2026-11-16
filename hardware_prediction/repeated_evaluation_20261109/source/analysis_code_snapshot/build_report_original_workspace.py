from pathlib import Path
import sys,csv,json,html,shutil
import numpy as np
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'outputs'))
import repeat9_console as c
import repeat_analysis as ra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=BASE/'outputs/bow_adopted9_20261109';OUT.mkdir(exist_ok=True)
for p in ['figures','tables','traces']:(OUT/p).mkdir(exist_ok=True)
MODELS=['R0','R1','R2','R3'];colors=['#78828f','#df921f','#327db8','#128b78']
motions=[(1,16,'Walk','歩く'),(2,8,'Clap','クラップ'),(10,1,'Bow','お辞儀'),(4,4,'Cheer','チア'),(5,26,'Kick','キック'),(6,73,'Crawl forward','前進クロール'),(7,74,'Crawl left','左クロール'),(8,75,'Crawl right','右クロール'),(9,76,'Crawl backward','後退クロール')]
def csvsave(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def csvread(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
c.backend.verify();c.extra.verify(c.backend)
metrics=[];quality=[];hashes={};selected=[]
for number,mid,name,ja in motions:
    for repeat in range(1,6):
        folder=max((c.SESSION/'trials').glob(f'B{repeat:02d}_M{mid:03d}_A*'),key=lambda p:c.read(p/'trial.json')['started_utc'])
        qc=c.recording_quality(folder)
        assert qc['state']=='pass',(folder,qc)
        quality.append(dict(number=number,repeat=repeat,capture_id=folder.name,**qc['quality']))
        selected.append(dict(number=number,motion=mid,name=name,repeat=repeat,capture_id=folder.name))
        for p,sha in c.read(folder/'raw_sha256.json').items():hashes[str(folder/'raw'/p)]=sha
        with c.quality_policy.session_evaluation_policy(c.SESSION):mm,trace,q=ra.evaluate(folder/'raw','ground' if 6<=number<=9 else 'feet',strict=True)
        for m in mm:metrics.append(dict(number=number,name=name,repeat=repeat,**m))
        csvsave(OUT/'traces'/f'{number:02d}_{repeat}.csv',trace)
summary=[]
for number,mid,name,ja in motions:
    for model in MODELS:
        vals=[r['energy_error_pct'] for r in metrics if r['number']==number and r['model']==model]
        assert len(vals)==5
        summary.append(dict(number=number,name=name,name_ja=ja,model=model,n=5,mean_error_pct=float(np.mean(vals)),sd_pp=float(np.std(vals,ddof=1))))
overall=[dict(model=m,n_motions=9,n_trials=45,mean_error_pct=float(np.mean([r['mean_error_pct'] for r in summary if r['model']==m]))) for m in MODELS]
previous=csvread(BASE/'outputs/latest_4_10_11_20261109/tables/per_trial_metrics.csv')
comparison=[]
for model in MODELS:
    old=[float(r['energy_error_pct']) for r in previous if int(r['number'])==4 and r['model']==model]
    now=[r for r in metrics if r['number']==4 and r['model']==model]
    comparison.append(dict(model=model,previous_mean_pct=float(np.mean(old)),latest_mean_pct=float(np.mean([r['energy_error_pct'] for r in now])),previous_trial1_pct=next(float(r['energy_error_pct']) for r in previous if int(r['number'])==4 and int(r['repeat'])==1 and r['model']==model),latest_trial1_pct=next(r['energy_error_pct'] for r in now if r['repeat']==1)))
for name,rows in [('per_trial_metrics',metrics),('per_motion',summary),('overall',overall),('quality',quality),('selected_captures',selected),('cheer_comparison',comparison)]:csvsave(OUT/'tables'/f'{name}.csv',rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def savefig(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
fig,ax=plt.subplots(figsize=(14,6));xx=np.arange(9);width=.19;markers=['o','s','^','D','v']
for j,model in enumerate(MODELS):
    ss=[next(r for r in summary if r['number']==n and r['model']==model) for n,_,_,_ in motions]
    pos=xx+(j-1.5)*width
    ax.bar(pos,[r['mean_error_pct'] for r in ss],width,yerr=[r['sd_pp'] for r in ss],capsize=3,color=colors[j],label=model,alpha=.85)
    for i,(n,_,_,_) in enumerate(motions):
        rr=sorted([r for r in metrics if r['number']==n and r['model']==model],key=lambda r:r['repeat'])
        for k,r in enumerate(rr):ax.scatter(pos[i]+np.linspace(-.06,.06,5)[k],r['energy_error_pct'],marker=markers[k],s=19,c='#233441',edgecolor='white',linewidth=.3,zorder=4)
ax.set(xticks=xx,xticklabels=[str(n)+' '+name.replace(' ','\n',1) for n,_,name,_ in motions],ylabel='Absolute energy error (%)',ylim=(0,None),title='Original R0–R3 | Bow replaces supine get-up; latest Cheer trial 1\nMean ± sample SD; all five trials shown')
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(ncol=4,loc='upper right')
fig.legend(handles=[plt.Line2D([],[],color='#233441',marker=m,linestyle='None',label=f'Trial {i+1}') for i,m in enumerate(markers)],loc='lower center',ncol=5,frameon=False)
fig.tight_layout(rect=(0,.07,1,1));savefig(fig,'01_adopted9_five_points')
fig,ax=plt.subplots(figsize=(6,4.5))
bars=ax.bar(MODELS,[r['mean_error_pct'] for r in overall],color=colors)
ax.bar_label(bars,fmt='%.2f%%',padding=4);ax.set(ylabel='Mean absolute energy error (%)',title='Adopted 9 motions | 45 trials\nEqual-weight mean across motions');ax.margins(y=.18)
fig.tight_layout();savefig(fig,'02_overall')
assert all(c.backend.sha(Path(p))==sha for p,sha in hashes.items())
c.save(OUT/'analysis.json',dict(models='original frozen R0-R3',model_sha256=c.backend.sha(c.PACKAGE/'frozen_models/frozen_models.npz'),selected_motions=[n for n,_,_,_ in motions],excluded_motions=[3,11,12],selected_captures=selected,quality_pass=45,original_3ms_pass=sum(r['original_quality_pass'] for r in quality),per_motion=summary,overall=overall,cheer_comparison=comparison,raw_hashes=hashes,note='User-selected revised evaluation set after viewing results; retrospective analysis, not independent first-attempt performance. Console numbering unchanged.'))
lines=['お辞儀を採用した9動作・最新チア取り直し反映','再学習前の旧モデルR0〜R3。仰向け起き上がりを外し、10番お辞儀（速度1/6）を採用。','採用番号: 1, 2, 10, 4, 5, 6, 7, 8, 9。コンソールではお辞儀は引き続き10番です。','各動作5回、計45試行が現在の通信品質（5 ms）に合格。','元の3 ms基準にも合格: '+str(sum(r['original_quality_pass'] for r in quality))+'/45。通信品質は機械的動作成功を保証しません。','最新4-1: '+next(r['capture_id'] for r in selected if r['number']==4 and r['repeat']==1),'','チア5回平均: 直前 → 最新']
for r in comparison:lines.append(f"{r['model']}: {r['previous_mean_pct']:.3f}% → {r['latest_mean_pct']:.3f}%（最新1回目 {r['latest_trial1_pct']:.3f}%）")
lines+=['','採用9動作の全体平均:']+[f"{r['model']}: {r['mean_error_pct']:.3f}%" for r in overall]
lines+=['','誤差=100×|予測エネルギー−実測エネルギー|/実測エネルギー。棒は5回平均、縦線は標本標準偏差。','評価動作の選択および再測定は結果を確認した後に行っています。旧9動作との平均差は、同一評価集合での改善を意味しません。','11番・12番は本集計に含めていません。元の記録と過去レポートは保持しています。']
text='\n'.join(lines);(OUT/'Summary_JA.txt').write_text(text,encoding='utf-8-sig')
body='<html lang="ja"><meta charset="utf-8"><title>お辞儀採用・最新9動作</title><style>body{font-family:Meiryo,sans-serif;max-width:1300px;margin:30px auto;line-height:1.8}img{width:100%}pre{white-space:pre-wrap}</style><h1>お辞儀を採用した9動作</h1><pre>'+html.escape(text)+'</pre>'
for name in ['01_adopted9_five_points','02_overall']:body+=f'<img src="figures/{name}.png">'
body+='<h2>各動作の平均誤差（%）</h2><table border="1"><tr><th>動作</th>'+''.join('<th>'+m+'</th>' for m in MODELS)+'</tr>'
for n,_,_,ja in motions:body+='<tr><td>'+str(n)+' '+ja+'</td>'+''.join(f"<td>{next(r['mean_error_pct'] for r in summary if r['number']==n and r['model']==m):.3f}</td>" for m in MODELS)+'</tr>'
body+='</table></html>';(OUT/'Report_JA.html').write_text(body,encoding='utf-8')
shutil.make_archive(str(OUT),'zip',OUT);print(text)
