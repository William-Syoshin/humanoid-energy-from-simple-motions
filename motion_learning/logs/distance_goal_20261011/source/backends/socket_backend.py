"""Own one isolated Webots process; exchange JSON only over loopback TCP."""
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


class WebotsProcess:
    def __init__(self, config, run_dir=None):
        self.config = config
        self.process = self.connection = self.stream = self.log = None
        parent = ROOT / 'logs/instances'
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix='webots-', dir=run_dir or parent))
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(1)
        token = secrets.token_hex(24)
        from scripts.launch import webots_path
        executable = webots_path()
        version = subprocess.check_output([executable, '--version'], text=True).strip()
        if 'R2025a' not in version:
            raise RuntimeError('This model is validated against Webots R2025a: ' + version)
        # Separate Webots projects avoid wbproj/cache writes colliding in workers.
        for name in ('controllers', 'protos', 'vendor'):
            (self.directory / name).symlink_to(ROOT / name, target_is_directory=True)
        (self.directory / 'worlds').mkdir()
        world = (ROOT / 'worlds/khr_walk.wbt').read_text()
        world = world.replace('basicTimeStep 4', f"basicTimeStep {config['simulation']['physics_ms']}")
        world = world.replace('coulombFriction [ 1 ]', f"coulombFriction [ {config['simulation']['friction']} ]")
        target = self.directory / 'worlds/khr_walk.wbt'
        target.write_text(world)
        (self.directory / 'config.json').write_text(json.dumps(config))
        # Hold an OS-assigned port until launch. Webots' own port is separate
        # from our already-bound IPC listener. Startup failure is never ignored.
        port_reservation = socket.socket()
        port_reservation.bind(('127.0.0.1', 0))
        port = port_reservation.getsockname()[1]
        env = dict(os.environ, KHR_RL_PORT=str(listener.getsockname()[1]),
                   KHR_RL_TOKEN=token, KHR_RL_CONFIG=str(self.directory / 'config.json'),
                   KHR_RL_ROOT=str(ROOT), PYTHONUNBUFFERED='1')
        cmd = [executable, '--batch', '--stdout', '--stderr', f'--port={port}',
               '--mode=realtime' if config['simulation']['realtime'] else '--mode=fast']
        if not config['simulation']['gui']:
            cmd += ['--no-rendering', '--minimize']
            if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
                if not shutil.which('xvfb-run'):
                    raise RuntimeError('Linux without DISPLAY requires xvfb-run and xauth')
                cmd = ['xvfb-run', '-a'] + cmd
        cmd.append(str(target))
        (self.directory / 'launch.json').write_text(json.dumps({'command': cmd, 'version': version}, indent=2))
        try:
            self.log = (self.directory / 'console.log').open('w')
            port_reservation.close()
            self.process = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
            deadline = time.monotonic() + config['simulation']['startup_timeout_s']
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('Webots exited during startup; see ' + str(self.directory))
                try:
                    self.connection, _ = listener.accept()
                    break
                except socket.timeout:
                    continue
            else:
                raise TimeoutError('Webots controller did not connect; see ' + str(self.directory))
            self.connection.settimeout(config['simulation']['socket_timeout_s'])
            self.stream = self.connection.makefile('rwb')
            hello = self._read()
            if hello.get('token') != token:
                raise RuntimeError('Unexpected controller handshake')
            self.inventory = hello
        except Exception:
            self.close()
            raise
        finally:
            listener.close()
            port_reservation.close()

    def _read(self):
        raw = self.stream.readline(8_000_000)
        if not raw:
            raise RuntimeError('Webots connection closed; see ' + str(self.directory))
        reply = json.loads(raw)
        if 'error' in reply:
            raise RuntimeError('Controller: ' + reply['error'])
        return reply

    def request(self, command, **fields):
        self.stream.write((json.dumps(dict(command=command, **fields), allow_nan=False) + '\n').encode())
        self.stream.flush()
        return self._read()

    def reset(self, seed, targets, yaw):
        return self.request('reset', seed=int(seed), targets=list(targets), yaw=float(yaw))['state']

    def step(self, targets):
        return self.request('step', targets=targets)['states']

    def screenshot(self, path):
        return self.request('screenshot', path=str(Path(path).resolve()))

    def close(self):
        if self.stream:
            try:
                self.stream.write(b'{"command":"close"}\n')
                self.stream.flush()
            except (OSError, ValueError):
                pass
            self.stream.close()
            self.stream = None
        if self.connection:
            self.connection.close()
        if self.process:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
        if self.log:
            self.log.close()
