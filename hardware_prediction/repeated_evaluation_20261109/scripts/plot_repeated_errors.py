"""Publication figure from verified repeated-execution metrics, without refitting."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

import argparse
BASE=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
OUT=ap.parse_args().out.resolve();OUT.mkdir(parents=True,exist_ok=True)
panels=json.loads((BASE/'verified_data/panels.json').read_text())
assert len(panels)==9
fonts={f.name for f in font_manager.fontManager.ttflist}
FONT='Helvetica' if 'Helvetica' in fonts else 'DejaVu Sans'
plt.rcParams.update({'font.family':FONT,'font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                     'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,
                     'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
                     'axes.linewidth':.85,'xtick.major.width':.8,'ytick.major.width':.8,'savefig.facecolor':'white'})
colors=['#92999E','#D89035','#2879B0','#188A75']
fig,axes=plt.subplots(3,3,figsize=(180/25.4,150/25.4),sharey=True)
labels=[];whisker_data=[]
for i,(ax,p) in enumerate(zip(axes.flat,panels)):
    mean=np.array([r['mean_error_pct'] for r in p['models']])
    sd=np.array([r['sd_percentage_points'] for r in p['models']])
    assert all(r['n']==5 for r in p['models'])
    assert (mean-sd>=0).all()
    ax.set_axisbelow(True);ax.grid(axis='y',color='#E5E8EB',linewidth=.65)
    ax.bar(np.arange(4),mean,width=.59,color=colors,zorder=3)
    ax.axhline(mean[0],color=colors[0],linewidth=.85,ls=(0,(3,3)),zorder=2)
    ax.errorbar(np.arange(4),mean,yerr=sd,fmt='none',ecolor='#181818',elinewidth=.9,capsize=3,capthick=.9,zorder=4)
    for j,(m,s) in enumerate(zip(mean,sd)):
        labels.append(ax.text(j,m+s+.45,f'{m:.2f}',fontsize=8.1,ha='center',va='bottom',
                              fontweight='bold' if j==int(mean.argmin()) else 'normal',
                              bbox=dict(facecolor='white',edgecolor='none',pad=.1),zorder=5))
    ax.set_xticks(np.arange(4),['R0','R1','R2','R3']);ax.set_xlim(-.6,3.6)
    ax.set_ylim(0,20.5);ax.set_yticks([0,5,10,15,20]);ax.tick_params(length=3,pad=3)
    for sp in ['left','bottom']:ax.spines[sp].set_color('#747B80')
    labels.append(ax.set_title(f"({chr(65+i)}) {p['name']}",loc='left',fontweight='bold',pad=21,fontsize=9.6))
    setting='native' if p['frame_multiplier']==1 else f"s = {p['frame_multiplier']}"
    note=f"{setting}; E = {p['measured_energy_mean_J']:.1f} ± {p['measured_energy_sd_J']:.1f} J"
    labels.append(ax.text(0,1.025,note,transform=ax.transAxes,fontsize=7.7,color='#42484D',va='bottom'))
    whisker_data.append(dict(motion=p['name'],mean=mean.tolist(),sd=sd.tolist(),lower=(mean-sd).tolist(),upper=(mean+sd).tolist()))
fig.supylabel('Absolute relative energy error (%)',x=.017,fontsize=10)
labels.append(fig.text(.535,.023,'Bars: mean; error bars: ±1 SD (n = 5 per motion).',ha='center',fontsize=8.5))
fig.subplots_adjust(left=.105,right=.985,bottom=.092,top=.914,wspace=.24,hspace=.69)
fig.canvas.draw()
renderer=fig.canvas.get_renderer();outside=[]
for text in labels:
    box=text.get_window_extent(renderer)
    if box.x0<0 or box.y0<0 or box.x1>fig.bbox.width or box.y1>fig.bbox.height:outside.append(text.get_text())
assert not outside,outside
for ext in ['pdf','svg']:fig.savefig(OUT/f'Fig4_nine_motions_mean_SD.{ext}')
fig.savefig(OUT/'Fig4_nine_motions_mean_SD.png',dpi=600)
fig.savefig(OUT/'Fig4_nine_motions_mean_SD_preview.png',dpi=240)
plt.close(fig)
(OUT/'plot_verification.json').write_text(json.dumps(dict(font=FONT,size_mm=[180,150],png_dpi=600,
    statistic='Mean ± sample SD within each motion/model, n=5; variance uses denominator 4.',
    panels=whisker_data,text_outside_canvas=outside),indent=2)+'\n')
caption=('Energy prediction errors for nine whole-body motions, each measured five times. '
         'Bars show the mean absolute relative energy error, calculated for each execution as '
         '100 × |predicted energy − measured energy| / measured energy. Error bars indicate ±1 sample '
         'standard deviation across the five executions (denominator n − 1, n = 5); they describe '
         'within-motion variability, not confidence intervals. Numbers above bars are means, with '
         'the smallest mean in each panel shown in bold. Dashed lines mark the mean R0 error. '
         'E denotes measured energy (mean ± SD), and s is the playback frame-duration multiplier; '
         'native denotes the stored playback timing. The original R0–R3 models are used without refitting.\n')
(OUT/'caption_EN.txt').write_text(caption,encoding='utf-8')
print(OUT/'Fig4_nine_motions_mean_SD.pdf')
