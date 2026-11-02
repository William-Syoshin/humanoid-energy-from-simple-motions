"""Original three-view simulation scene figure only; no refitting.
Use build_figure_revision.py for the current model-label and workflow figures."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/webots_simulation_20261012';FIG=ROOT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(FIG/f'{name}.{ext}',dpi=250,bbox_inches='tight')
 plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(7.4,2.1))
for ax,(name,title) in zip(axes,[('test_walk_101_600.png','(a) Walking'),('test_get_up_101_899.png','(b) Get-up not completed'),('test_push_box_101_1399.png','(c) Object pushing')]):
 ax.imshow(plt.imread(DATA/'screenshots'/name));ax.set_xlim(200,750);ax.set_ylim(550,180);ax.axis('off');ax.set_title(title,fontsize=9,pad=3)
fig.subplots_adjust(left=0,right=1,bottom=0,top=.87,wspace=.025);save(fig,'sim_webots_scenes')
