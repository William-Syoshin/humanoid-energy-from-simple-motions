"""Compose actual Webots checkpoint recordings with Japanese explanatory cards."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import unicodedata
from PIL import Image, ImageDraw, ImageFont


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recordings',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    assets=a.out/'learning_progress_assets';assets.mkdir(exist_ok=True)
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg: raise RuntimeError('ffmpeg is required')
    font_path=next(f for f in Path('/System/Library/Fonts').glob('*.ttc')
                   if unicodedata.normalize('NFC',f.name)=='ヒラギノ角ゴシック W3.ttc')
    def font(size): return ImageFont.truetype(str(font_path),size)
    bg='#101826';fg='#edf3fa';muted='#b8c4d5';cyan='#69d8ff';green='#7be5b7';amber='#ffd080'
    def text(draw,xy,s,size=26,color=fg): draw.text(xy,s,font=font(size),fill=color)
    def encode(arguments):
        subprocess.run([ffmpeg,'-y','-hide_banner','-loglevel','error',*arguments,
                        '-c:v','libx264','-threads','2','-crf','19','-pix_fmt','yuv420p',
                        '-movflags','+faststart',str(assets/f'part_{len(parts):02}.mp4')],check=True)
        parts.append(assets/f'part_{len(parts):02}.mp4')
    def card(name,lines,duration):
        image=Image.new('RGB',(1280,800),bg);draw=ImageDraw.Draw(image)
        for y,s,size,col in lines: text(draw,(65,y),s,size,col)
        path=assets/(name+'.png');image.save(path)
        encode(['-loop','1','-framerate','25','-i',str(path),'-t',str(duration)])
    parts=[]
    card('intro',[(65,'強化学習で、歩き方はどう変わった？',42,fg),
                  (170,'左：歩行報酬のみ　　右：歩行報酬 ＋ R2電力ペナルティ',29,fg),
                  (260,'25,000 → 50,000 → 75,000 → 最終モデル',33,cyan),
                  (345,'保存した途中モデルを、同じ初期状態で20秒ずつ再生',28,fg),
                  (410,'4倍速 / 学習seed 10 / 評価seed 20000',28,fg),
                  (515,'学習中の画面録画ではなく、学習済みモデルの再生です。',25,muted),
                  (565,'低速・小振幅の歩行です。成功・失敗を含めて表示します。',25,muted)],4)
    records=[]
    for index,steps in enumerate([25000,50000,75000,100352]):
        label=f'{steps:,} steps'+('（最終）' if steps==100352 else '')
        overlay=Image.new('RGBA',(1280,800),(0,0,0,0));draw=ImageDraw.Draw(overlay)
        draw.rectangle((0,0,1279,159),fill=bg);draw.rectangle((0,640,1279,799),fill=bg)
        draw.rectangle((638,160,641,639),fill=bg)
        text(draw,(25,14),'保存モデルの比較  |  '+label,34)
        text(draw,(25,67),'4倍速・同じ20秒間の再生  /  床の目盛り10 cm  /  学習seed 10',23,muted)
        text(draw,(25,113),'Task-only：歩行報酬のみ',27,cyan)
        text(draw,(665,113),'R2-energy：電力係数 0.3',27,green)
        for x,kind in [(25,'task_only'),(665,'R2')]:
            folder=a.recordings/f'{kind}_{steps}'
            meta=json.loads((folder/'metadata.json').read_text());s=meta['summary'];records.append(meta)
            text(draw,(x,654),f"距離 {s['distance_m']*100:.2f} cm  |  R2予測 {s['predicted_energy_J']['R2']:.2f} J",23)
            text(draw,(x,691),'速度基準：'+('達成' if s['task_success'] else '未達')+f"  ({s['mean_speed_m_s']:.4f} m/s)",22,
                 green if s['task_success'] else amber)
        text(draw,(25,748),f'{index+1}/4  数値は20秒終了時の評価結果（保存モデルの再生）',22,muted)
        path=assets/f'overlay_{steps}.png';overlay.save(path)
        left=a.recordings/f'task_only_{steps}/webots.mp4';right=a.recordings/f'R2_{steps}/webots.mp4'
        filters=('[0:v]scale=640:480,setsar=1,fps=25,tpad=stop_mode=clone:stop_duration=1,trim=duration=5[l];'
                 '[1:v]scale=640:480,setsar=1,fps=25,tpad=stop_mode=clone:stop_duration=1,trim=duration=5[r];'
                 '[l][r]hstack=inputs=2,pad=1280:800:0:160:color=0x101826[base];'
                 '[base][2:v]overlay=0:0:shortest=1[out]')
        encode(['-i',str(left),'-i',str(right),'-loop','1','-framerate','25','-i',str(path),
                '-filter_complex',filters,'-map','[out]','-an','-t','5'])
    card('conclusion',[(65,'映像は「固定した1つの学習seed」の例です',38,fg),
                       (170,'このseedでは、電力係数0.3の最終モデルは速度不足でした。',27,amber),
                       (260,'全体の実験：4条件 × 6学習seed = 24学習・240評価',29,fg),
                       (345,'予測エネルギーは低下する傾向がありましたが、',28,fg),
                       (393,'同等の歩行性能を全seedで保つ基準は満たしていません。',28,fg),
                       (500,'実機の消費電力を削減したという結果ではありません。',25,muted),
                       (560,'詳細：docs/COEFFICIENT_SWEEP_RESULTS.md',25,cyan)],6)
    concat=assets/'concat.txt'
    concat.write_text(''.join(f"file '{f.resolve()}'\n" for f in parts))
    target=a.out/'learning_progress_seed10_4x.mp4'
    subprocess.run([ffmpeg,'-y','-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',str(concat),
                    '-c','copy','-movflags','+faststart',str(target)],check=True)
    (a.out/'learning_progress_metadata.json').write_text(json.dumps(dict(
        description='Checkpoint replay, NOT live training footage',training_seed=10,evaluation_seed=20000,
        speed=4,conditions=['task_only','R2 lambda=0.3'],checkpoints=[25000,50000,75000,100352],
        source_recordings=str(a.recordings.resolve()),clips=records),indent=2)+'\n')
    print(target)


if __name__=='__main__': main()
