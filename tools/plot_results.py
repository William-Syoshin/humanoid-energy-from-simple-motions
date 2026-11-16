"""Rebuild current repeated-execution numerical figures without replacing artwork."""
from pathlib import Path
import argparse, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
out=ap.parse_args().out.resolve();out.mkdir(parents=True,exist_ok=True)
scripts=ROOT/'hardware_prediction/repeated_evaluation_20261109/scripts'
for name in ['plot_repeated_errors.py','plot_repeated_summaries.py']:
 subprocess.run([sys.executable,str(scripts/name),'--out',str(out)],check=True)
print('Rebuilt current Figures 4-7:',out)
