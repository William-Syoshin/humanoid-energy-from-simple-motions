"""Prepare/run the frozen 21-trial Webots revision in a fresh macOS workspace."""
from pathlib import Path
import argparse, json, sys, shutil, importlib.util
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'synthetic_webots/runs/revision_02'
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--webots',default='/Applications/Webots.app/Contents/MacOS/webots')
 ap.add_argument('--prepare-only',action='store_true',help='Copy project/config only; do not launch Webots')
 ap.add_argument('--gui',action='store_true')
 args=ap.parse_args();out=args.out.resolve()
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 project=out/'source';shutil.copytree(RUN/'source',project,ignore=shutil.ignore_patterns('__pycache__'))
 (project/'controllers/benchmark/runtime.ini').write_text('[python]\nCOMMAND = '+sys.executable+'\n')
 design=json.loads((RUN/'design.json').read_text());specs={s['split']+'_'+s['id']:s for s in design['trials'] if s['split']=='test'}
 order=json.loads((RUN/'evaluation_order.json').read_text())['keys'];assert set(order)==set(specs)
 (out/'planned_trials.json').write_text(json.dumps([specs[k] for k in order],indent=2)+'\n')
 if args.prepare_only:print('Prepared without launching Webots:',out);return
 if sys.platform!='darwin':raise RuntimeError('This launch adapter supports macOS; rebuild the physics plugin and adapt launch management for other platforms.')
 module_spec=importlib.util.spec_from_file_location('frozen_webots_session',RUN/'analysis_scripts/webots_session.py');m=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(m);m.WEBOTS=args.webots
 results=[]
 with m.Session(out/'collection',gui=args.gui,project=project) as sim:
  for key in order:
   result,_,_=sim.run(specs[key],key);results.append(dict(key=key,result=result));(out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print('Completed',key,flush=True)
 print('New simulations saved separately:',out)
if __name__=='__main__':main()
