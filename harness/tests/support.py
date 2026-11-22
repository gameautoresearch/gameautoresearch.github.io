"""Offline stand-ins: a release fixture, a backend with no cloud, and a scripted researcher."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

from harness import policy, workspace
from harness.backend import ACTIVE, Backend

ROOT = Path(__file__).resolve().parents[2]
AR = ROOT / 'harness/box/ar.py'


def release_fixture(game, directory, bind=True):
    """A release holding only what the workspace needs: the game's real starter, configs, rules and source."""
    release = Path(directory) / game
    public = release / 'public'
    public.mkdir(parents=True)
    auto = ROOT / workspace.catalog()[game]['autoresearch']
    # Polytopia's constructor builds the worker runtime into the release; stand in for it.
    training = release / 'runtime' if game == 'polytopia' else public
    starter = auto / ('runtime/starter' if game == 'pokemon' else 'starter')  # Pokemon's native starter
    for native in workspace.TRAINING[game].values():
        source = starter / (native if game == 'polytopia' else Path(native).name)
        target = training / native
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if game == 'polytopia':
        shutil.copytree(auto / 'starter/src/polytopia_bot', training / 'src/polytopia_bot', dirs_exist_ok=True)
        (release / 'v1').mkdir()
        for name in ('train.py', 'policy.py', 'grade.py'):
            shutil.copyfile(auto / 'v1' / name, release / 'v1' / name)
    if game == 'clash':
        shutil.copyfile(auto / 'recipes/starter.json', public / 'config.json')
        shutil.copyfile(auto / 'recipes/smoke.json', public / 'smoke-config.json')
    else:
        (public / 'recipes').mkdir()
        for name in ('starter.json', 'smoke.json'):
            shutil.copyfile(auto / 'recipes' / name, public / 'recipes' / name)
    if game == 'polytopia':
        shutil.copyfile(auto / 'recipes/starter_arch.json', public / 'recipes/starter_arch.json')
    meta = workspace.stage(game, public, training_source=training)
    if bind:
        policy.bind_release(game, release)
    return release, meta


def host_runs(broker):
    """A broker whose `run` requests start training runs the way the host does (`Broker.train`).

    A game with leases refuses `ar run` to its researcher, but the host still trains a recipe with exactly that
    machinery, and these tests exercise it through the same requests."""
    handle = broker.handle

    def routed(request):
        if not isinstance(request, dict) or request.get('op') != 'run':
            return handle(request)
        try:
            return broker.train(request)
        except ValueError as exc:
            return {'error': str(exc)}
    broker.handle = routed
    return broker


class FakeBackend(Backend):
    """Ledger arithmetic and job states only. A run completes on the tick after it is queued."""
    game = 'doudizhu'
    opened = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workspace = self.root / 'workspace'
        self.path = self.root / 'fake.json'

    def open(self):
        if self.resume:
            self.state = json.loads(self.path.read_text())
        else:
            self.root.mkdir(parents=True)
            self.state = dict(budget=self.budget_seconds, jobs=[], submissions=[], finalized=False)
            self._save()
        FakeBackend.opened.append(self)

    def _save(self):
        self.path.write_text(json.dumps(self.state))

    def settle(self):
        for job in self.state['jobs']:
            if job['status'] in ACTIVE:
                job['status'] = 'stopped'
        self._save()

    def close(self):
        self.settle()

    def used(self):
        return sum(job['seconds'] for job in self.state['jobs'])

    def budget(self):
        return dict(budget_seconds=self.state['budget'], used_seconds=self.used(),
                    remaining_seconds=self.state['budget'] - self.used())

    def status(self):
        return dict(jobs=self.state['jobs'], active=[j['name'] for j in self.state['jobs'] if j['status'] in ACTIVE])

    def log(self, name, tail=80):
        return dict(name=name, stdout='trained')

    def run(self, name, seconds, config, seed=1):
        if seconds > self.state['budget'] - self.used():
            raise ValueError('Allocation exceeds remaining session budget')
        if self.status()['active']:
            raise ValueError('One job at a time')
        files = workspace.editable(self.release, self.workspace)
        self.state['jobs'].append(dict(name=name, kind='train', status='queued', seconds=seconds, config=config,
                                       seed=seed, files=files))
        self._save()
        return dict(queued=name)

    def stop(self, name):
        return dict(stopping=name)

    def evaluate(self, name, seconds=None):
        job = next((j for j in self.state['jobs'] if j['name'] == name and j['status'] == 'complete'), None)
        if job is None:
            raise ValueError('Practice requires a completed run')
        job['practice'] = dict(games=4, points=0.5)
        self._save()
        return dict(queued='eval-' + name)

    def submit(self, name):
        job = next((j for j in self.state['jobs'] if j['name'] == name and j['status'] == 'complete'), None)
        if job is None:
            raise ValueError('No registered trained artifact')
        self.state['submissions'].append(name)
        self._save()
        return dict(submission=len(self.state['submissions']), bundle_sha256=workspace.digest(job['files']))

    def finalize(self):
        self.state['finalized'] = True
        self._save()
        return dict(lifecycle='finalized')

    def tick(self):
        for job in self.state['jobs']:
            if job['status'] == 'queued':
                job['status'] = 'complete'
        self._save()

    def grade(self):
        job = next(j for j in self.state['jobs'] if j['name'] == self.state['submissions'][-1])
        return dict(reward=0.5, graded=job['name'], source_sha256=workspace.digest(job['files']))


class Researcher:
    """Drives the real in-box `ar` against a directory standing in for the box."""

    def __init__(self, task_dir, box_dir):
        self.ar = Path(box_dir) / 'ar'
        self.workspace = Path(box_dir) / 'workspace'
        shutil.copytree(Path(task_dir) / 'environment/ar', self.ar)
        shutil.copytree(Path(task_dir) / 'environment/workspace', self.workspace)
        for name in ('requests', 'replies'):
            # The broker's polling thread may find the box (and create these) as soon as `ar` exists.
            (self.ar / name).mkdir(exist_ok=True)

    def __call__(self, *args, expect=0):
        env = dict(os.environ, AR_ROOT=str(self.ar), AR_WORKSPACE=str(self.workspace))
        done = subprocess.run([sys.executable, '-I', str(self.ar / 'ar.py'), *args], env=env, capture_output=True,
                              text=True, timeout=60, cwd=self.workspace)
        if done.returncode != expect:
            raise AssertionError(f'ar {args}: exit {done.returncode}\n{done.stdout}\n{done.stderr}')
        return json.loads(done.stdout) if done.stdout.strip() else {'error': done.stderr}

    def edit(self, name, text):
        (self.workspace / name).write_text(text)

    def launch(self, *args, expect=0):
        """Run the box's `training/launch.py` in the foreground, with `ar` on its PATH. Returns what it printed."""
        tools = self.ar.parent / 'bin'
        tools.mkdir(exist_ok=True)
        shim = tools / 'ar'
        shim.write_text(f'#!/bin/sh\nexec {sys.executable} -I {self.ar / "ar.py"} "$@"\n')
        shim.chmod(0o755)
        env = dict(os.environ, AR_ROOT=str(self.ar), AR_WORKSPACE=str(self.workspace), AR_LAUNCH_POLL='0.2',
                   AR_LAUNCH_STATE=str(self.ar.parent / 'launch'), PATH=f'{tools}:{os.environ["PATH"]}')
        done = subprocess.run([sys.executable, '-I', str(self.workspace / 'training/launch.py'), *args, '--foreground'],
                              env=env, capture_output=True, text=True, timeout=120, cwd=self.workspace)
        if done.returncode != expect:
            raise AssertionError(f'launch {args}: exit {done.returncode}\n{done.stdout}\n{done.stderr}')
        return done.stdout


class Process:
    """Stands in for the `harbor run` subprocess: a researcher script on a thread."""

    def __init__(self, script):
        self.error = None

        def work():
            try:
                script()
            except BaseException as exc:
                self.error = exc
        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def wait(self):
        self.thread.join(timeout=120)
        if self.error:
            raise self.error
        return 0
