"""Portable launcher. One unique log directory per process; no machine paths saved."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def webots_path():
    candidates = [os.environ.get('WEBOTS_EXECUTABLE'), shutil.which('webots')]
    home = os.environ.get('WEBOTS_HOME')
    if home:
        candidates += [str(Path(home)/'webots'), str(Path(home)/'Contents/MacOS/webots')]
    if sys.platform == 'darwin':
        candidates.append('/Applications/Webots.app/Contents/MacOS/webots')
    candidates.append('/usr/local/webots/webots')
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise SystemExit('Webots not found. Install R2025a or set WEBOTS_EXECUTABLE.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--world', type=Path, default=ROOT/'worlds/khr_training.wbt')
    p.add_argument('--config', type=Path, default=ROOT/'config/phase1.json')
    p.add_argument('--mode', choices=['hold', 'sweep'])
    p.add_argument('--joint', help='Sweep only this active motor, e.g. Head')
    p.add_argument('--duration', type=float)
    p.add_argument('--snapshot', action='store_true', help='Save final Webots view as PNG')
    p.add_argument('--batch', action='store_true', help='Exit after test; GUI otherwise pauses')
    p.add_argument('--headless', action='store_true', help='Linux only: xvfb-run and no rendering')
    p.add_argument('--port', type=int, default=1234)
    p.add_argument('--timeout', type=float, default=120, help='Wall-clock timeout for batch execution')
    args = p.parse_args()
    if args.headless and not sys.platform.startswith('linux'):
        p.error('--headless is supported only on Linux; use --batch on macOS')
    if not 1 <= args.port <= 65535:
        p.error('port out of range')
    config = json.loads(args.config.read_text())
    version = subprocess.check_output([webots_path(), '--version'], text=True).strip()
    if 'R2025a' not in version:
        p.error('This Phase 1 is pinned to Webots R2025a: ' + version)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    # Help relative Python commands resolve; an absolute Webots preference takes precedence.
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    env['KHR_SNAPSHOT'] = '1' if args.snapshot else '0'
    env['KHR_WEBOTS_VERSION'] = version
    env['KHR_WORLD'] = str(args.world.resolve())
    env['KHR_CONFIG'] = str(args.config.resolve())
    for key, value in [('KHR_MODE',args.mode),('KHR_DURATION',args.duration),('KHR_JOINT',args.joint)]:
        if value is not None:
            env[key] = str(value)
    batch = args.batch or args.headless
    env['KHR_QUIT'] = '1' if batch else '0'
    log_parent = ROOT/'logs'
    log_parent.mkdir(exist_ok=True)
    # Reserve a parent atomically. Controller creates the run child.
    session = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d-%H%M%S-'), dir=log_parent))
    output = session/'run'
    env['KHR_LOG_DIR'] = str(output)
    cmd = [webots_path(), '--stdout', '--stderr', '--mode=fast' if batch else '--mode=realtime', f'--port={args.port}']
    if batch:
        cmd += ['--batch']
    if args.headless:
        if not shutil.which('xvfb-run'):
            p.error('Install xvfb and xauth for Linux headless execution')
        cmd = ['xvfb-run', '-a'] + cmd + ['--no-rendering', '--minimize']
    cmd += [str(args.world.resolve())]
    # The world defines basicTimeStep; verify config before starting the simulator.
    import re
    basic = int(re.search(r'basicTimeStep\s+(\d+)', args.world.read_text())[1])
    if config['basic_timestep_ms'] != basic:
        p.error('basic_timestep_ms must match worlds/khr_training.wbt')
    (session/'launch.json').write_text(json.dumps({'command':cmd,'python':sys.executable,'webots':version}, indent=2)+'\n')
    print(f'Logs: {output}', flush=True)
    if not batch:
        return subprocess.call(cmd, env=env, cwd=ROOT)
    try:
        with (session/'console.log').open('w') as log:
            completed = subprocess.run(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
        print((session/'console.log').read_text())
    except subprocess.TimeoutExpired:
        print(f'Webots timed out; see {session}/console.log', file=sys.stderr)
        return 1
    summary = output/'summary.json'
    if not summary.exists():
        print('No summary produced. World loading or controller startup failed.', file=sys.stderr)
        return 1
    status = json.loads(summary.read_text())['status']
    return completed.returncode or (0 if status == 'passed' else 1)


if __name__ == '__main__':
    raise SystemExit(main())
