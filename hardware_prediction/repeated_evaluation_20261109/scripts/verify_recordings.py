"""Read-only reconstruction of the supplied 45 recordings with frozen models."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import sys
import numpy as np

import argparse
BASE=Path(__file__).resolve().parents[1]
SOURCE=BASE/'source'
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out',type=Path,required=True)
OUT=ap.parse_args().out.resolve()
MODELS=['R0','R1','R2','R3']
NAMES={1:'Work',2:'Clap',10:'Bow',4:'Cheer',5:'Kick',6:'Crawl',7:'Crawl turn left',8:'Crawl turn right',9:'Backward crawl'}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvsave(path,rr):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    expected=read(SOURCE/'sha256_manifest.json')
    assert all(sha(SOURCE/p)==h for p,h in expected.items())
    frozen=read(SOURCE/'original_models/frozen.json')
    assert sha(SOURCE/'original_models/frozen_models.npz')==frozen['model_sha256']
    code=SOURCE/'analysis_code_snapshot'
    assert sha(code/'legacy_analysis/analysis.py')==frozen['analysis_sha256']
    sys.path.insert(0,str(code/'legacy_analysis'))
    spec=importlib.util.spec_from_file_location('repeat_analysis_archive',code/'repeat_analysis.py')
    ra=importlib.util.module_from_spec(spec);spec.loader.exec_module(ra)
    params=dict(np.load(SOURCE/'original_models/frozen_models.npz',allow_pickle=False))
    # Only reconstruction and frozen prediction are invoked. No fitting occurs.
    manifest=read(SOURCE/'export_manifest.json')
    captures=manifest['recordings'];assert len(captures)==45
    assert len({(x['number'],x['repeat']) for x in captures})==45
    old=rows(SOURCE/'analysis/tables/per_trial_metrics.csv')
    lookup={(int(x['number']),int(x['repeat']),x['model']):x for x in old}
    assert len(old)==len(lookup)==180
    result=[];quality=[];max_metric_difference=0.;max_trace_difference=0.
    for c in captures:
        folder=SOURCE/c['export_path'];raw=folder/'raw'
        assert all(sha(raw/p)==h for p,h in read(folder/'raw_sha256.json').items())
        d,qc=ra.reconstruct(raw,'ground' if c['number'] in [6,7,8,9] else 'feet',strict=True)
        # Recheck the original 3ms threshold without modifying stored settings.
        assert qc['power_jitter_s']<=.003
        trace=rows(SOURCE/f"analysis/traces/{c['number']:02d}_{c['repeat']}.csv")
        assert len(trace)==len(d['t'])
        for key,values in [('t_s',d['t']),('actual_W',d['p'])]:
            diff=float(np.max(abs(values-np.array([float(x[key]) for x in trace]))))
            max_trace_difference=max(max_trace_difference,diff);assert diff<1e-8,(c,key,diff)
        for model in MODELS:
            pred=ra.original.predict(d,model,(params[model+'_scale'],params[model+'_coef']))
            metric=ra.original.metrics(d,model,pred)
            ref=lookup[c['number'],c['repeat'],model]
            assert ref['trial_id']==c['capture_id']
            diff=float(np.max(abs(pred-np.array([float(x[model+'_W']) for x in trace]))))
            max_trace_difference=max(max_trace_difference,diff);assert diff<1e-8,(c,model,diff)
            for field in ['duration_s','actual_J','predicted_J','energy_error_pct','power_RMSE_W']:
                diff=abs(metric[field]-float(ref[field]));max_metric_difference=max(max_metric_difference,diff)
                assert diff<1e-7,(c,model,field,diff)
            result.append(dict(console_number=c['number'],motion=NAMES[c['number']],source_name=c['name'],repeat=c['repeat'],**metric))
        quality.append(dict(console_number=c['number'],repeat=c['repeat'],capture_id=c['capture_id'],**qc))
    summary=[];panels=[]
    old_summary={(int(x['number']),x['model']):x for x in rows(SOURCE/'analysis/tables/per_motion.csv')}
    for number in manifest['selected_motions']:
        all_rows=[r for r in result if r['console_number']==number]
        energy=[r['actual_J'] for r in all_rows if r['model']=='R0']
        spec=read(SOURCE/'motion_specs'/f"M{next(c['motion'] for c in captures if c['number']==number):03d}.json")
        panel=dict(number=number,name=NAMES[number],n=5,frame_multiplier=spec['frame_multiplier'],
                   measured_energy_mean_J=float(np.mean(energy)),measured_energy_sd_J=float(np.std(energy,ddof=1)),models=[])
        for model in MODELS:
            v=np.array([r['energy_error_pct'] for r in all_rows if r['model']==model]);assert len(v)==5
            stats=dict(console_number=number,motion=NAMES[number],model=model,n=5,mean_error_pct=float(v.mean()),
                       sd_percentage_points=float(v.std(ddof=1)),variance_percentage_points_squared=float(v.var(ddof=1)),
                       min_error_pct=float(v.min()),max_error_pct=float(v.max()))
            ref=old_summary[number,model]
            assert abs(stats['mean_error_pct']-float(ref['mean_error_pct']))<1e-7
            assert abs(stats['sd_percentage_points']-float(ref['sd_pp']))<1e-7
            summary.append(stats);panel['models'].append(stats)
        panels.append(panel)
    overall={m:float(np.mean([r['mean_error_pct'] for r in summary if r['model']==m])) for m in MODELS}
    csvsave(OUT/'per_trial_verified.csv',result);csvsave(OUT/'mean_sd_variance.csv',summary)
    save(OUT/'panels.json',panels)
    save(OUT/'audit.json',dict(status='passed',archive_hashes_checked=len(expected),recordings=45,model_results=180,
                             model_sha256=frozen['model_sha256'],refitting=False,original_3ms_quality_passes=45,
                             max_metric_difference=max_metric_difference,max_trace_difference=max_trace_difference,
                             overall_mean_error_pct=overall,statistic='Arithmetic mean and sample SD/variance (ddof=1) over 5 recordings within each motion/model.',
                             label_mapping={'Work':'Walk (source name); retained manuscript label'}))
    print(json.dumps(read(OUT/'audit.json'),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
