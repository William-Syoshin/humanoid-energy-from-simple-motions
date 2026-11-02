"""Import immutable hardware predictors; generate a separate walking PROTO."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, default=ROOT.parent / 'humanoid-energy-paper')
    args = parser.parse_args()
    paper = args.paper.resolve()
    ridge = paper / 'hardware_experiment/regression_results/ridge_20260915_143619'
    frozen = json.loads((ridge / 'models/frozen.json').read_text())
    source = paper / 'hardware_experiment/analysis.py'
    assert digest(source) == frozen['analysis_sha256']
    assert digest(ridge / 'models/frozen_models.npz') == frozen['model_sha256']
    spec = importlib.util.spec_from_file_location('hardware_reference', source)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    manifest = ref.manifest(ridge / 'trials.csv')
    train, _ = ref.load(manifest, ridge / 'aligned', ['train'], frozen['schema'])
    assert len(train) == 116
    q, cmd = [], []
    for d in train:
        assert digest(d['path']) == frozen['training_validation_hashes'][d['meta']['trial_id']]['sha256']
        with d['path'].open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        q.extend([[float(r[f'q_{i}']) for i in range(17)] for r in rows])
        cmd.extend([[float(r[f'cmd_{i}']) for i in range(17)] for r in rows])
    features = np.concatenate([d['X']['R3'] for d in train])
    assets = ROOT / 'energy/assets'
    assets.mkdir(parents=True, exist_ok=True)
    for name in ('frozen.json', 'frozen_models.npz'):
        shutil.copy2(ridge / 'models' / name, assets / name)
    np.savez(assets / 'training_bounds.npz', feature_min=features.min(0), feature_max=features.max(0),
             q_min=np.min(q, axis=0), q_max=np.max(q, axis=0),
             cmd_min=np.min(cmd, axis=0), cmd_max=np.max(cmd, axis=0))
    pairs = [list(p) for p in ref.EDGES]
    motors = ['LeftFemurHead2','LeftFemur','LeftCrus','LeftAnkle','LeftFoot',
              'RightFemurHead2','RightFemur','RightCrus','RightAnkle','RightFoot',
              'LeftShoulder','LeftArm','LeftForearm','RightShoulder','RightArm','RightForearm','Head']
    signs = [1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,-1,1,1]
    power_signs = [1,-1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,1,1]
    mapping = {'coordinate_note': 'Anatomical simulation coordinates mapped to archived bent-leg/left-right hardware conventions. '
               'Inferred signs; independent hardware sign/zero calibration unverified. '
               'R1-R3 use absolute velocity/error and absolute error products, invariant to consistent per-joint sign flips.',
               'joints': [dict(name=n, motor=m, webots_sign=s, power_sign=p, power_offset_rad=0)
                          for n, m, s, p in zip(frozen['joint_order'], motors, signs, power_signs)]}
    (ROOT / 'config/joint_mapping.json').write_text(json.dumps(mapping, indent=2) + '\n')
    meta = dict(repository='https://github.com/William-Syoshin/humanoid-energy-paper',
                commit=subprocess.check_output(['git','-C',str(paper),'rev-parse','HEAD'],text=True).strip(),
                source_analysis_sha256=digest(source), model_sha256=digest(assets/'frozen_models.npz'),
                frozen_json_sha256=digest(assets/'frozen.json'),
                bounds_sha256=digest(assets/'training_bounds.npz'), train_trials=116, train_samples=len(q),
                bounds_source='training only; no validation or test samples', edges=pairs)
    (assets / 'provenance.json').write_text(json.dumps(meta, indent=2) + '\n')
    text = (ROOT / 'protos/Khr3hv17.proto').read_text().replace('PROTO Khr3hv17 [', 'PROTO Khr3hv17RL [', 1)
    changes = {'LeftFemur': '0 1 0', 'LeftCrus': '0 1 0',
               'RightFemur': '0 -1 0', 'RightCrus': '0 -1 0', 'RightAnkle': '0 -1 0'}
    for name, axis in changes.items():
        pattern = rf'(DEF KHR3HV_{name} HingeJoint \{{\s*jointParameters HingeJointParameters \{{)([^}}]*)'
        def replace(m):
            assert 'axis' not in m[2], 'Revisit axis fix if upstream changes'
            return m[1] + '\n                            axis ' + axis + m[2]
        text, count = re.subn(pattern, replace, text)
        assert count == 1
    text = text.replace('# keywords:', '# RL variant: five omitted pitch axes made explicit; see docs/EXPERIMENT.md.\n# keywords:', 1)
    (ROOT / 'protos/Khr3hv17RL.proto').write_text(text)
    print('Imported fixed predictors and training-only bounds; generated separate RL model.')


if __name__ == '__main__':
    main()
