"""Isolated Webots session for the revised synthetic-power benchmark.

No physical hardware interfaces. Original archives are read only.
"""
from pathlib import Path
import json
import os
import socket
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/Users/yuchenzhou/.cache/khr3hv-rl/runtime-20261008/bin/python'
WEBOTS = '/Applications/Webots.app/Contents/MacOS/webots'

class Session:
    def __init__(self, out, gui=False, project=None):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.project = Path(project) if project else ROOT / 'project'
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen(1)
        self.listener.settimeout(100)
        with socket.socket() as port:
            port.bind(('127.0.0.1', 0))
            webport = port.getsockname()[1]
        env = dict(os.environ, BENCH_PORT=str(self.listener.getsockname()[1]),
                   PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        cmd = [WEBOTS, '--batch', '--mode=fast', '--stdout', '--stderr', f'--port={webport}']
        if not gui:
            cmd += ['--no-rendering', '--minimize']
        cmd += [str(self.project / 'worlds/benchmark.wbt')]
        self.log = (self.out / 'webots.log').open('w')
        self.process = subprocess.Popen(cmd, env=env, stdout=self.log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        self.caffeinate = subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(self.process.pid)])
        (self.out / 'launch.json').write_text(json.dumps(dict(pid=self.process.pid,
            runner_pid=os.getpid(), command=cmd, started=time.time()), indent=2))
        self.conn, _ = self.listener.accept()
        self.conn.settimeout(600)
        self.stream = self.conn.makefile('rwb')
        self.inventory = json.loads(self.stream.readline())
        (self.out / 'inventory.json').write_text(json.dumps(self.inventory, indent=2))

    def run(self, spec, key, snapshot=False):
        path = self.out / 'raw' / f'{key}.npz'
        if path.exists():
            raise FileExistsError(path)
        request = dict(command='run', spec=spec, path=str(path), snapshot=snapshot)
        (self.out / f'{key}.request.json').write_text(json.dumps(request, indent=2))
        self.stream.write((json.dumps(request)+'\n').encode())
        self.stream.flush()
        line = self.stream.readline()
        if not line:
            raise RuntimeError('Webots controller connection closed')
        reply = json.loads(line)
        (self.out / f'{key}.reply.json').write_text(json.dumps(reply, indent=2))
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        return reply['result'], data, path

    def close(self):
        try:
            self.stream.write(b'{"command":"close"}\n')
            self.stream.flush()
        except (OSError, AttributeError):
            pass
        try:
            self.process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=12)
        self.stream.close()
        self.conn.close()
        self.listener.close()
        self.log.close()
        self.caffeinate.terminate()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

def development_spec(action, seed=91001, duration=12.0, **kwargs):
    return dict(split='test', id=f'{action}_{seed}', action=action, context='whole-body',
                duration=float(duration), seed=int(seed), level=None, joints=[],
                amplitude=.2, frequency=.3, offset=0., phase=0., **kwargs)

def mechanics(data):
    tail = slice(-100, None)
    h = data['h']
    z = data['position'][:, 2]
    R = data['root_orientation']
    initial = R[0, :, 0].copy()
    final = R[-1, :, 0].copy()
    yaw = np.unwrap(data['yaw'])
    return dict(final_height_m=float(z[-1]), final_h=float(h[-1]),
        final_tilt_deg=float(np.degrees(np.arccos(np.clip(1-h[-1], -1, 1)))),
        tail_max_tilt_deg=float(np.degrees(np.arccos(np.clip(1-h[tail], -1, 1))).max()),
        tail_min_height_m=float(z[tail].min()),
        tail_max_linear_speed=float(np.linalg.norm(data['root_velocity'][tail, :3], axis=1).max()),
        tail_max_angular_speed=float(np.linalg.norm(data['root_velocity'][tail, 3:], axis=1).max()),
        forward_m=float(data['position'][-1, 0]-data['origin'][0]),
        yaw_deg=float(np.degrees(yaw[-1]-yaw[0])),
        min_height_m=float(z.min()), max_h=float(h.max()),
        tail_mean_nonfoot_force=float(data['contact'][tail,2].mean()))
