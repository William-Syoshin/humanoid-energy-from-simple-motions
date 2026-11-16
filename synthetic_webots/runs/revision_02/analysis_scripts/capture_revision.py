"""Capture the revised motions and verify them against the formal recordings.

This program never changes a formal run. It creates a new capture directory,
clones the frozen source, and edits only its Viewpoint and recording code.
No additional simulation step is inserted between settling and raw-data saving.
Each target uses a fresh session, replaying the preceding formal test trials
in their original order without recording. Encoding occurs only after the
target's raw data have been saved, and that session is then closed.

Example (run only after the formal run has completed)::

    python scripts/capture_revision.py --run runs/revision_01 \
        --out captures/revision_01_seed101

Use --prepare-only to create and inspect the separate capture project without
launching Webots; use --prepared <capture-directory> to execute it later.
"""
from pathlib import Path
import argparse
import ast
import difflib
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import traceback

import numpy as np

from webots_session import ROOT, Session

ACTIONS = ('get_up', 'lie_down', 'turn_90', 'turn_180')
TOLERANCE = 1e-8
VIEW_ORIENTATION = [-0.056929688627, 0.0442538840725, 0.997396914121, 1.82257358583]
VIEW_POSITION = [0.38, -1.1, 0.34]
VIEW_FOV = 0.88
REQUIRED_ARRAYS = ('t', 'q', 'v', 'tau', 'h', 'com_z', 'root_omega',
                   'position', 'contact', 'yaw', 'object_position', 'origin',
                   'object_origin', 'qcmd', 'root_velocity', 'root_orientation')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def source_hashes(source):
    return {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.name != '.DS_Store'}


def substitute_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise ValueError(f'Expected one stable controller anchor, found {count}: {old[:100]!r}')
    return text.replace(old, new, 1)


def capture_view(data):
    """Keep the previous oblique side angle, widening a FIXED view if needed.

    The camera is framed using only the formal physical trajectory. No trial,
    time point, or model is chosen by prediction accuracy. The 0.30 m padding
    covers the humanoid's limbs beyond its root-position envelope.
    """
    position = np.asarray(data['position'])
    lo, hi = position.min(axis=0), position.max(axis=0)
    axis = np.array(VIEW_ORIENTATION[:3], dtype=float)
    axis /= np.linalg.norm(axis)
    angle = VIEW_ORIENTATION[3]
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    rot = np.eye(3)*math.cos(angle) + (1-math.cos(angle))*np.outer(axis, axis) + math.sin(angle)*cross
    # Webots uses the camera's local +X direction in its z-up world convention.
    old_eye = np.array(VIEW_POSITION)
    old_target = old_eye + 1.15*rot[:, 0]
    # Root origin is near the upper body in standing; include the soles below
    # it and a full arm/leg extent horizontally while lying down or turning.
    padded_lo = lo - np.array([.30, .30, .34])
    padded_hi = hi + np.array([.30, .30, .11])
    padded_lo[2] = max(0., padded_lo[2])
    center = (padded_lo+padded_hi)/2
    corners = np.array([[x,y,z] for x in (padded_lo[0],padded_hi[0])
        for y in (padded_lo[1],padded_hi[1]) for z in (padded_lo[2],padded_hi[2])])
    # Project all eight padded envelope corners using Webots camera axes.
    # horizontal FOV = .88; vertical FOV is derived for 1280:720.
    relative = (corners-center) @ rot
    tan_horizontal = math.tan(VIEW_FOV/2)/1.08
    tan_vertical = tan_horizontal/(1280/720)
    distance = max(1.15, float(np.max(np.abs(relative[:,1])/tan_horizontal-relative[:,0])),
        float(np.max(np.abs(relative[:,2])/tan_vertical-relative[:,0])))
    eye = center + (old_eye-old_target)*(distance/1.15)
    return dict(position=eye.tolist(), orientation=VIEW_ORIENTATION,
                field_of_view=VIEW_FOV, fixed_camera=True,
                reference_position=VIEW_POSITION, trajectory_bounds=[lo.tolist(), hi.tolist()],
                framing='Fixed oblique side view; padded full formal trajectory envelope')


def patch_world(text):
    lines = text.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if line.startswith('Viewpoint {')]
    if len(matches) != 1 or '}' not in lines[matches[0]]:
        raise ValueError('Expected exactly one single-line Viewpoint declaration')
    lines[matches[0]] = ('DEF CAPTURE_VIEW Viewpoint { orientation ' +
        ' '.join(map(str, VIEW_ORIENTATION)) + ' position ' +
        ' '.join(map(str, VIEW_POSITION)) + f' fieldOfView {VIEW_FOV} }}\n')
    return ''.join(lines)


def patch_controller(text):
    helper = '''
# Capture-only helpers. No robot/world dynamics or trial specification changes.
CAPTURE_SETTINGS=json.loads((ROOT/'capture_settings.json').read_text())
CAPTURE_ACTIVE_ID=os.environ['SYNTHETIC_CAPTURE_ID']
def capture_begin(s,path):
 if s['id']!=CAPTURE_ACTIVE_ID:return None
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 cfg=CAPTURE_SETTINGS[s['id']];view=r.getFromDef('CAPTURE_VIEW')
 view.getField('position').setSFVec3f(cfg['view']['position'])
 view.getField('orientation').setSFRotation(cfg['view']['orientation'])
 view.getField('fieldOfView').setSFFloat(cfg['view']['field_of_view'])
 movie=path.with_suffix('.mp4')
 if path.exists() or movie.exists():raise RuntimeError('Refuse to overwrite capture '+str(path))
 # The 100 settling steps have finished. No additional step is taken here.
 r.movieStartRecording(str(movie),1280,720,0,90,1,False)
 return dict(movie=str(movie),snapshot_indices=cfg['snapshot_indices'],view=cfg['view'],
  dimensions=[1280,720],acceleration=1,caption=False,recording_start_simulation_time=r.getTime(),
  settling_seconds_excluded=1.0,snapshots=[])
def capture_finish(path,capture):
 # Called only AFTER the original raw-data save. These steps cannot enter raw.
 start=time.monotonic();steps=0
 while not r.movieIsReady():
  if time.monotonic()-start>240:raise RuntimeError('Movie encoder timeout')
  if r.step(10)==-1:raise RuntimeError('Webots stopped during movie encoding')
  steps+=1;time.sleep(.005)
 if r.movieFailed():raise RuntimeError('Webots movie encoding failed')
 movie=Path(capture['movie'])
 if not movie.is_file() or movie.stat().st_size==0:raise RuntimeError('Empty movie output')
 capture.update(encoding_wait_seconds=time.monotonic()-start,post_raw_encoding_steps=steps,
  movie_bytes=movie.stat().st_size)
 Path(path).with_suffix('.capture.json').write_text(json.dumps(capture,indent=2))
'''
    text = substitute_once(text, 'def run(s,path,snapshot=False):', helper+'\ndef run(s,path,snapshot=False):')
    text = substitute_once(text, ' t0=r.getTime();q0=', ' capture=capture_begin(s,path)\n t0=r.getTime();q0=')
    old = "  if snapshot and k in [0,round(s['duration']*50),round(s['duration']*100)-1]:r.exportImage(str(Path(path).with_suffix(''))+f'_{k}.png',95)"
    new = """  if capture is not None and snapshot and k in capture['snapshot_indices']:
   image_path=str(Path(path).with_suffix(''))+f'_frame_{k:05d}.png'
   r.exportImage(image_path,100)
   capture['snapshots'].append(dict(index=k,time_s=float(r.getTime()-t0),path=image_path))"""
    text = substitute_once(text, old, new)
    text = substitute_once(text, ' a=np.array(records)', " if capture is not None:\n  r.movieStopRecording()\n  capture['recording_end_simulation_time']=r.getTime()\n a=np.array(records)")
    text = substitute_once(text, " path.with_suffix('.tmp').replace(path)", " path.with_suffix('.tmp').replace(path)\n if capture is not None:capture_finish(path,capture)")
    ast.parse(text)
    return text


def load_record(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def compare_records(reference, captured):
    if set(reference) != set(captured):
        raise AssertionError(f'Raw keys differ: {set(reference)^set(captured)}')
    missing = set(REQUIRED_ARRAYS)-set(reference)
    if missing:
        raise AssertionError(f'Required audit arrays missing: {missing}')
    comparisons = {}
    for key in sorted(reference):
        a, b = reference[key], captured[key]
        if key == 'spec_json':
            same = json.loads(str(a)) == json.loads(str(b))
            comparisons[key] = dict(unchanged=same, passed=same)
        elif np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            shape_equal = a.shape == b.shape
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            delta = float(np.max(np.abs(a-b))) if shape_equal and finite and a.size else None
            passed = shape_equal and finite and (not a.size or delta <= TOLERANCE)
            comparisons[key] = dict(shape=list(a.shape), shape_equal=shape_equal,
                finite=finite, max_absolute_difference=delta, passed=bool(passed))
        else:
            same = bool(np.array_equal(a, b))
            comparisons[key] = dict(unchanged=same, passed=same)
    return dict(absolute_tolerance=TOLERANCE, arrays=comparisons,
                all_arrays_pass=all(item['passed'] for item in comparisons.values()))


def find_ffprobe():
    choices = [shutil.which('ffprobe'), '/opt/homebrew/bin/ffprobe', '/usr/local/bin/ffprobe']
    return next((str(p) for p in choices if p and Path(p).is_file()), None)


def audit_media(movie, spec, metadata):
    ffprobe = find_ffprobe()
    if not ffprobe:
        raise RuntimeError('ffprobe is required to verify movie dimensions and 1x duration')
    result = subprocess.run([ffprobe, '-v', 'error', '-show_streams', '-show_format',
                             '-of', 'json', str(movie)], capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    videos = [stream for stream in info['streams'] if stream.get('codec_type') == 'video']
    if len(videos) != 1:
        raise AssertionError('Expected one video stream')
    v = videos[0]
    duration = float(v.get('duration', info['format']['duration']))
    snaps = metadata['snapshots']
    conditions = dict(dimensions_1280x720=v['width']==1280 and v['height']==720,
        duration_1x=abs(duration-float(spec['duration'])) <= .20,
        no_caption=metadata['caption'] is False,
        five_snapshots=len(snaps)==5 and all(Path(s['path']).is_file() for s in snaps),
        settled_time_excluded=metadata['settling_seconds_excluded']==1.0)
    return dict(conditions=conditions, passed=all(conditions.values()),
                duration_seconds=duration, expected_seconds=float(spec['duration']),
                stream=v, movie_sha256=digest(movie), snapshots=snaps)


def prepare(run, out, actions=ACTIONS, history='formal-prefix'):
    run, out = Path(run).resolve(), Path(out).resolve()
    if run == out or run in out.parents:
        raise ValueError('The capture directory must be outside the formal run')
    if not (run/'completion.json').is_file():
        raise FileNotFoundError('Formal completion.json is required before capture preparation')
    source = run/'source'
    design = json.loads((run/'design.json').read_text())
    frozen_hashes = source_hashes(source)
    for name, sha in design['source_hashes'].items():
        if frozen_hashes.get(name) != sha:
            raise AssertionError(f'Frozen source mismatch: {name}')
    if history not in ('formal-prefix','isolated'):
        raise ValueError(f'Unknown session history mode: {history}')
    # The formal runner simulates test trials only, in its recorded order;
    # training/validation/control data were copied and did not step Webots.
    test_specs = {'test_'+s['id']:s for s in design['trials'] if s['split']=='test'}
    order_path = run/'evaluation_order.json'
    keys = json.loads(order_path.read_text())['keys'] if order_path.is_file() else list(test_specs)
    if len(keys)!=len(test_specs) or set(keys)!=set(test_specs):
        raise AssertionError('Recorded evaluation order does not contain each formal test exactly once')
    formal_order = []
    for key in keys:
        spec = test_specs[key]
        raw = run/'raw'/f'{key}.npz'
        data = load_record(raw)
        if json.loads(str(data['spec_json'])) != spec:
            raise AssertionError(f'Formal raw spec does not match design: {key}')
        formal_order.append(dict(key=key, action=spec['action'], raw=str(raw),
                                 raw_sha256=digest(raw), spec=spec))
    records, settings = [], {}
    for action in actions:
        key = f'test_{action}_101'
        raw = run/'raw'/f'{key}.npz'
        data = load_record(raw)
        spec = json.loads(str(data['spec_json']))
        if spec['action'] != action or spec['seed'] != 101 or spec['split'] != 'test':
            raise AssertionError(f'Unexpected fixed trial: {key}')
        design_spec = next(s for s in design['trials'] if s['split']=='test' and s['id']==spec['id'])
        if spec != design_spec:
            raise AssertionError(f'Raw spec does not match frozen design: {key}')
        n = len(data['t'])
        settings[spec['id']] = dict(view=capture_view(data),
            snapshot_indices=np.rint(np.linspace(0,n-1,5)).astype(int).tolist())
        index = next(i for i,item in enumerate(formal_order) if item['key']==key)
        prefix = [item['key'] for item in formal_order[:index]] if history=='formal-prefix' else []
        records.append(dict(key=key, action=action, raw=str(raw), raw_sha256=digest(raw),
                            spec=spec, formal_index=index, preceding_trials=prefix))
    out.mkdir(parents=True, exist_ok=False)
    project = out/'capture_project'
    shutil.copytree(source, project, ignore=shutil.ignore_patterns('__pycache__', '.DS_Store'))
    edits = []
    for relative, transform in [('worlds/benchmark.wbt',patch_world),
                                 ('controllers/benchmark/benchmark.py',patch_controller)]:
        original = (project/relative).read_text()
        updated = transform(original)
        (project/relative).write_text(updated)
        edits.extend(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),
            fromfile='frozen/'+relative,tofile='capture/'+relative))
    write_json(project/'capture_settings.json', settings)
    (out/'capture_source_changes.diff').write_text(''.join(edits))
    hashes = source_hashes(project)
    changed = [name for name in frozen_hashes if hashes.get(name)!=frozen_hashes[name]]
    allowed = ['controllers/benchmark/benchmark.py','worlds/benchmark.wbt']
    if sorted(changed) != sorted(allowed):
        raise AssertionError(f'Unexpected clone modifications: {changed}')
    manifest = dict(format_version=2,run=str(run), out=str(out), source=str(source), project=str(project),
        prepared_at=time.time(), formal_source_sha256=frozen_hashes,
        capture_source_sha256=hashes, changed_source_files=changed,
        added_source_files=sorted(set(hashes)-set(frozen_hashes)),
        records=records, formal_test_order=formal_order, session_history=history,
        evaluation_order_file=str(order_path) if order_path.is_file() else None,
        evaluation_order_sha256=digest(order_path) if order_path.is_file() else None,
        session_lifecycle='Fresh Webots per target; prefix without capture; target capture; encode; close',
        tolerance=TOLERANCE, actions=list(actions),
        physical_changes=False, modified_trial_specs=False, settings=settings)
    write_json(out/'capture_manifest.json', manifest)
    return manifest


def originals_preserved(manifest):
    order = manifest.get('evaluation_order_file')
    return (not order or digest(order)==manifest['evaluation_order_sha256']) and source_hashes(Path(manifest['source']))==manifest['formal_source_sha256'] and all(
        digest(record['raw'])==record['raw_sha256'] for record in manifest['formal_test_order'])


def execute(manifest):
    if manifest.get('format_version') != 2:
        raise ValueError('This capture manifest predates formal-history support; prepare a new directory')
    out, source = Path(manifest['out']), Path(manifest['source'])
    project = Path(manifest['project'])
    if not originals_preserved(manifest):
        raise AssertionError('Formal source or raw data changed since capture preparation')
    if source_hashes(project) != manifest['capture_source_sha256']:
        raise AssertionError('Prepared capture project changed')
    if (out/'collection').exists() or (out/'collections').exists():
        raise FileExistsError('Capture already started; retain this result and use a new output directory')
    if not find_ffprobe():
        raise RuntimeError('ffprobe is required before starting capture')
    reports, history_reports = [], []
    formal = {record['key']:record for record in manifest['formal_test_order']}
    prior_environment = os.environ.get('SYNTHETIC_CAPTURE_ID')
    try:
        for record in manifest['records']:
            session_out = out/'collections'/record['key']
            # Prefixes can contain another requested target. This selector keeps
            # those prior trials unrecorded, without changing their specifications.
            os.environ['SYNTHETIC_CAPTURE_ID'] = record['spec']['id']
            prefix_audits = []
            with Session(session_out, gui=True, project=project) as session:
                for index,key in enumerate(record['preceding_trials']):
                    prior = formal[key]
                    write_json(out/'progress.json',dict(stage='replaying_formal_prefix',
                        target=record['key'],current=key,prefix_completed=index,
                        prefix_total=len(record['preceding_trials']),
                        captures_completed=len(reports),captures_total=len(manifest['records'])))
                    _,data,path = session.run(prior['spec'],key,snapshot=False)
                    comparison = compare_records(load_record(prior['raw']),data)
                    audit = dict(key=key,reference_raw=prior['raw'],replayed_raw=str(path),
                                 comparison=comparison,recorded_movie=False)
                    write_json(session_out/f'{key}.warmup_audit.json',audit)
                    prefix_audits.append(audit)
                    write_json(session_out/'warmup_summary.json',dict(
                        formal_order=record['preceding_trials'],completed=len(prefix_audits),
                        all_pass=all(x['comparison']['all_arrays_pass'] for x in prefix_audits),
                        trials=prefix_audits))
                    print(record['key'],'PREFIX',index+1,'/',len(record['preceding_trials']),
                          key,'PASS' if comparison['all_arrays_pass'] else 'FAIL',flush=True)
                    if not comparison['all_arrays_pass']:
                        raise AssertionError(f'Formal-prefix reproduction failed at {key}; output retained')
                original_path = Path(record['raw'])
                write_json(out/'progress.json',dict(stage='capturing',key=record['key'],
                    completed=len(reports),total=len(manifest['records']),
                    prefix_trials_reproduced=len(prefix_audits)))
                result,data,path = session.run(record['spec'],record['key'],snapshot=True)
                comparison = compare_records(load_record(original_path),data)
                write_json(out/f'{record["key"]}.reproducibility.json',comparison)
                metadata = json.loads(path.with_suffix('.capture.json').read_text())
                media = audit_media(path.with_suffix('.mp4'),record['spec'],metadata)
                report = dict(key=record['key'],reference_raw=str(original_path),capture_raw=str(path),
                    movie=str(path.with_suffix('.mp4')),comparison=comparison,media=media,
                    controller_result=result,session_history=manifest['session_history'],
                    formal_index=record['formal_index'],preceding_trials=record['preceding_trials'],
                    prefix_audits=str(session_out/'warmup_summary.json') if prefix_audits else None,
                    passed=comparison['all_arrays_pass'] and media['passed'])
                write_json(out/f'{record["key"]}.audit.json',report)
                reports.append(report)
                history_reports.append(dict(target=record['key'],prefix_trials=len(prefix_audits),
                    all_prefix_trials_reproduced=True))
                write_json(out/'summary.json',dict(complete=False,records=reports,
                    session_history=manifest['session_history'],history=history_reports))
                print(record['key'],'PASS' if report['passed'] else 'FAIL',path.with_suffix('.mp4'),flush=True)
                if not report['passed']:
                    raise AssertionError(f'Capture verification failed: {record["key"]}; output retained')
            # Encoding steps occurred only after the target. Close this session
            # before opening the next one so they cannot alter its reset history.
        preserved = originals_preserved(manifest)
        write_json(out/'original_preservation_check.json',dict(formal_source_and_all_test_raw_unchanged=preserved))
        if not preserved:
            raise AssertionError('Original preservation check failed')
        write_json(out/'summary.json',dict(complete=True,all_reproduced=True,
            tolerance=TOLERANCE,formal_source_and_all_test_raw_unchanged=True,
            session_history=manifest['session_history'],history=history_reports,records=reports))
        write_json(out/'progress.json',dict(stage='complete',completed=len(reports),total=len(reports)))
    except Exception:
        write_json(out/'capture_error.json',dict(time=time.time(),traceback=traceback.format_exc(),
            records_completed=len(reports),session_history=manifest['session_history']))
        write_json(out/'original_preservation_check.json',dict(
            formal_source_and_all_test_raw_unchanged=originals_preserved(manifest)))
        raise
    finally:
        if prior_environment is None:
            os.environ.pop('SYNTHETIC_CAPTURE_ID',None)
        else:
            os.environ['SYNTHETIC_CAPTURE_ID']=prior_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run',type=Path,help='Completed formal run with frozen source and raw data')
    parser.add_argument('--out',type=Path,help='New capture directory outside the formal run')
    parser.add_argument('--actions',nargs='+',choices=ACTIONS,default=list(ACTIONS))
    parser.add_argument('--history',choices=['formal-prefix','isolated'],default='formal-prefix',
        help='Reproduce the original test-trial history (default); isolated is diagnostic only')
    parser.add_argument('--prepare-only',action='store_true',help='Copy and patch capture source; do not launch Webots')
    parser.add_argument('--prepared',type=Path,help='Execute an already prepared, unchanged capture directory')
    args = parser.parse_args()
    if args.prepared:
        if args.run or args.out or args.prepare_only:
            parser.error('--prepared cannot be combined with --run, --out, or --prepare-only')
        manifest=json.loads((args.prepared/'capture_manifest.json').read_text())
    else:
        if not args.run or not args.out:
            parser.error('--run and --out are required unless --prepared is provided')
        manifest=prepare(args.run,args.out,args.actions,args.history)
    print('CAPTURE_PROJECT',manifest['project'],flush=True)
    if not args.prepare_only:
        execute(manifest)


if __name__ == '__main__':
    main()
