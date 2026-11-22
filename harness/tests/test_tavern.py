"""Tavern backend against its real controller, ledger, registry and final-bank code; the provider is stubbed.

Run with the game's research interpreter: .venv-tavern-ar/bin/python -m unittest harness.tests.test_tavern
"""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from unittest import mock

from harness import budget as budgets, handoff, policy, registry, session, workers, workspace
from harness.tests import test_handoff
from harness.transport import LocalBox
from harness.broker import Broker
from harness.tests.support import ROOT, Process, Researcher, release_fixture

try:
    import numpy as np
    from simulators.tavern.autoresearch.evaluation_v2.release import GAME, RUNTIME, sha, write
    from simulators.tavern.autoresearch.ledger import Ledger
    from harness.games.tavern import TRAIN_SHAPE, controller as native
    from harness.games.tavern.ledger import DollarLedger, recorded_mode
except ImportError as exc:  # wrong interpreter
    raise unittest.SkipTest(f'Tavern research dependencies unavailable: {exc}')

BINDING = {'benchmark_id': 'tavern-four-small-v1', 'native_source_hash': 'stub', 'rules': {'num_players': 4}}
PROFILES = ('scripts', 'mixed', 'learned')
SEATS = 4
SELECTED = ROOT / 'simulators/tavern/autoresearch/evaluation_v2/selected_registry-four-small-v1.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tavern_release(directory):
    """The shared workspace fixture plus the release identity the controller verifies before every evaluation."""
    release, meta = release_fixture('tavern', directory, bind=False)
    pool = release / 'private/registry'
    pool.mkdir(parents=True)
    opponents = [dict(policy_id='ppo', kind='learned')] + [dict(policy_id=s, kind='script', style=s)
                                                            for s in ('tempo', 'economy', 'synergy')]
    write(pool / 'registry.json', dict(binding=BINDING, opponents=opponents, profiles={
        name: ['ppo', 'tempo', 'economy'] for name in PROFILES}))
    write(pool / 'manifest.json', {'registry.json': sha(pool / 'registry.json')})
    write(release / 'binding.json', BINDING)
    write(release / 'public/practice.json', dict(label='public-practice', seeds=[50101, 50102, 50103, 50104]))
    write(release / 'release.json', dict(
        evaluator_id='tavern-four-small-v1', frozen_files={}, binding_sha256=sha(release / 'binding.json'),
        registry_manifest_sha256=sha(pool / 'manifest.json'),
        decision_source={name: sha(GAME / 'evaluation_readiness' / name) for name in RUNTIME}))
    policy.bind_release('tavern', release)
    return release, meta


class FolderModal:
    """The modal module's Volume lookup, offline: each named Volume is a local folder (harness.tests.test_handoff)."""

    def __init__(self, root):
        outer = self
        self.root = Path(root)

        class Volume:
            @staticmethod
            def from_name(name, environment_name=None, **options):
                return test_handoff.FolderVolume(outer.root / name)

        self.Volume = Volume


def local_publish(command, mount):
    """The worker's publish command, rewritten to run here against a local folder (no `sync` on this host)."""
    code = command[2].replace(repr(handoff.MOUNT), repr(str(mount)), 1)
    return [sys.executable, '-c', code.replace("subprocess.run(['sync',mount],check=True,timeout=120)", 'pass'),
            *command[3:]]


def step(args):
    """What a worker command is, by its script or entry point."""
    if args[1] == '-m':
        return 'evaluate'
    if args[1] != '-c':
        return args[1].rsplit('/', 1)[-1].replace('_entry.py', '')
    return {handoff.PREPARE: 'prepare', native.RUN_AND_PUBLISH: 'train', native.PACK: 'publish'}.get(args[2], 'other')


class Sandbox:
    def __init__(self, number, gpu, timeout, queue=0, volume=None):
        self.id = f'sb-{number}'
        self.gpu, self.timeout, self.volume = gpu, timeout, volume
        self.files, self.outputs, self.commands, self.live = {}, {}, [], []
        self.started, self.ended, self.terminated = 1000.0 * number, 0.0, 0
        self.entered, self.released = threading.Event(), threading.Event()
        # A GPU worker can wait in the provider's queue: no start for `queue` receipt reads (None: never).
        self.queued, self.reads, self.number = (queue if gpu else 0), 0, number
        if self.queued != 0:
            self.started = 0.0


class Provider:
    """Stands in for Modal: records what each worker was given and answers as the provider would."""
    app_id = 'app-stub'

    def __init__(self, train='fail', seconds=30.0):
        self.train, self.seconds = train, seconds
        self.sandboxes, self.readied, self.withhold = [], 0, False
        self.queue = 0  # receipt reads before a GPU worker starts; None: it never does
        self.exported = False  # whether a trainer that is ended early had exported a network
        self.gate = None  # an Event withholding training workers' end receipts until it is set
        self.unpublished = False  # the worker's publish to the hand-off Volume fails
        self.lose_reply = False  # the training command's reply is lost after the worker published
        self.temp = tempfile.TemporaryDirectory()
        self.modal = FolderModal(Path(self.temp.name) / 'volumes')
        self.volumes = []

    def ready(self):
        self.readied += 1

    def volume(self, root):
        volume = handoff.SessionVolume('tavern', root, modal_module=self.modal)
        self.volumes.append(volume)
        return volume

    def create(self, gpu, timeout, volume=None):
        sandbox = Sandbox(len(self.sandboxes) + 1, gpu, timeout, self.queue, volume)
        self.sandboxes.append(sandbox)
        return sandbox

    def publish(self, sandbox, spec):
        """Run the real in-worker packing and publish scripts on a local copy of what the worker holds."""
        if self.unpublished or sandbox.terminated:
            raise RuntimeError('Remote command failed: publish')
        root = Path(self.temp.name) / 'workers' / sandbox.id
        for folder, files in sandbox.outputs.items():
            for name, value in files.items():
                path = root / folder.lstrip('/') / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        if getattr(sandbox, 'trainer_log', None) is not None:
            (root / 'job').mkdir(parents=True, exist_ok=True)
            (root / native.TRAINER_LOG.lstrip('/')).write_text(sandbox.trainer_log)
        local = lambda path: str(root / path.lstrip('/'))
        spec = dict(spec, files={name: local(path) for name, path in spec['files'].items()},
                    tars={name: dict(tar, root=local(tar['root']), path=local(tar['path']))
                          for name, tar in spec['tars'].items()},
                    publish=local_publish(spec['publish'], self.modal.root / sandbox.volume.name))
        for tar in spec['tars'].values():
            Path(tar['path']).parent.mkdir(parents=True, exist_ok=True)
        done = subprocess.run([sys.executable, '-c', native.PACK, json.dumps(spec)], capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError('Remote command failed: ' + done.stderr[-2000:])
        return done.stdout

    def identity(self, sandbox):
        return sandbox.id

    def upload(self, sandbox, source, dest):
        source = Path(source)
        sandbox.files[dest] = {p.relative_to(source).as_posix(): p.read_bytes()
                               for p in sorted(source.rglob('*')) if p.is_file()}

    def copy(self, sandbox, local, remote):
        sandbox.files[remote] = Path(local).read_bytes()

    def command(self, sandbox, args, timeout=220):
        sandbox.commands.append((list(args), timeout))
        kind = step(args)
        if kind == 'prepare':
            return ''
        if kind == 'publish':
            return self.publish(sandbox, json.loads(args[3]))
        if kind == 'train':  # the worker's runner: the trainer, then (if it finished) its publish
            spec = json.loads(args[3])
            sandbox.entered.set()
            if self.train == 'block':
                sandbox.released.wait(30)
                raise RuntimeError('Remote command failed: worker terminated')
            if self.train == 'overrun':
                raise RuntimeError('Remote command failed: time limit')
            if self.train == 'fail':
                raise RuntimeError('Remote command failed: stub provider trains nothing')
            self._train(sandbox)
            sandbox.trainer_log = 'trained\n{"updates": 1}\n'
            try:
                marker = spec['marks'][0] + self.publish(sandbox, spec['pack']).strip().splitlines()[-1]
            except RuntimeError as exc:
                marker = spec['marks'][1] + json.dumps(str(exc))
            if self.lose_reply:
                raise ConnectionError('the host link dropped')
            return sandbox.trainer_log + '\n' + marker + '\n'
        if args[1].endswith('/finalize_entry.py'):  # packages what an ended trainer last exported, if anything
            if sandbox.terminated or not self.exported:
                raise RuntimeError('Remote command failed: No complete export to package')
            self._train(sandbox, ended_early=True)
            return json.dumps(sandbox.outputs['/job/train']['summary.json'])
        if args[1] == '-m':
            lobbies = SEATS * len(PROFILES) * len(json.loads(sandbox.files['/opt/evaluator-panel.json'])['seeds'])
            result = dict(complete_valid_schedule=True, expected_cases=lobbies, observed_cases=lobbies, raw_score=40.0,
                          mean_placement=2.6, top_half_rate=0.4, failures={}, confidence_interval_95=[30.0, 50.0])
            sandbox.outputs['/private/evaluation'] = {
                'lifecycle.json': dict(status='completed'),
                'qualification-privacy.json': dict(uid=65534, private_paths_denied=3, credential_env_absent=True),
                'qualification-terminal-replay.json': dict(all_lobby_actual_terminals_retained=True,
                                                           recorded_lobbies=lobbies)}
            sandbox.outputs['/private/evaluation/screen'] = {'result.json': result, 'episodes.json': []}
            return ''
        return '[]'  # listings of retained files

    def kill_trainer(self, sandbox):
        """End the trainer process only: the blocked training command returns and the worker stays up."""
        running = sandbox.entered.is_set() and not sandbox.released.is_set()
        sandbox.kills = getattr(sandbox, 'kills', 0) + 1
        sandbox.released.set()
        return int(running)

    def _train(self, sandbox, **marks):
        """What train_entry.py exports for the uploaded source and recipe."""
        buffer = io.BytesIO()
        np.savez(buffer, weight=np.ones(3, np.float32))
        identity = json.dumps(dict(binding=BINDING, config=json.loads(sandbox.files['/job/recipe.json']),
                                   source={name: digest(data) for name, data in
                                           sandbox.files['/job/source'].items()})).encode()
        summary = dict(updates=1, samples=10, finite_losses=True, changed_weights=True, accounted_seconds=5.0,
                       identity_sha256=digest(identity), weights_sha256=digest(buffer.getvalue()), **marks)
        metrics = [dict(update=1, learning=dict(loss=0.5, samples=10, parameter_delta_l2=0.1),
                        episodes=[dict(placement=3, valid=True), dict(placement=8, valid=False)],
                        counts=dict(invalid_lobbies=1, actor_calls=7))]
        sandbox.outputs['/job/train'] = {'weights.npz': buffer.getvalue(), 'identity.json': identity,
                                         'summary.json': summary, 'metrics.json': metrics, 'resume.pt': b'resume'}

    def download(self, sandbox, source, dest, names):
        sandbox.live.append(source)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            value = sandbox.outputs[source][name]
            (dest / name).write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())

    def read(self, sandbox, remote):
        sandbox.live.append(remote)
        raise FileNotFoundError(remote)

    def terminate(self, sandbox):
        sandbox.terminated += 1
        sandbox.ended = sandbox.ended or (sandbox.started + self.seconds if sandbox.started else 1.0)
        sandbox.released.set()

    def terminate_id(self, sandbox_id):
        self.terminate(next(s for s in self.sandboxes if s.id == sandbox_id))

    def allocation(self, app_id, sandbox_id):
        sandbox = next(s for s in self.sandboxes if s.id == sandbox_id)
        if not sandbox.started and not sandbox.terminated:
            sandbox.reads += 1
            if sandbox.queued is not None and sandbox.reads >= sandbox.queued:
                sandbox.started = 1000.0 * sandbox.number  # a GPU finally became free
        finished = 0.0 if self.withhold or (sandbox.gpu and self.gate is not None and not self.gate.is_set()) \
            else sandbox.ended
        return dict(provider_id=sandbox_id, started_at=sandbox.started, finished_at=finished,
                    gpu_type='L40S' if sandbox.gpu else '')


class TavernBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.release, cls.meta = tavern_release(cls.shared.name)

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / 'session'
        self.out.mkdir()
        self.provider = Provider()
        self.addCleanup(self.provider.temp.cleanup)
        self.backend = self.open()
        self.broker = Broker(self.backend, self.out, 'network', research_seconds=600)
        self.backend.controller.start_poll_seconds = self.backend.controller.abandon_poll_seconds = 0.01
        base = workspace.public(self.release) / workspace.WORKSPACE
        self.files = {name: (base / name).read_text() for name in self.meta['writable']}

    def open(self, name='research', **kwargs):
        kwargs.setdefault('budget_seconds', None if kwargs.get('resume') else 600)
        backend = registry.backend('tavern')(self.release, self.out / name, smoke=True,
                                             options={'_provider': self.provider}, **kwargs)
        backend.open()
        backend.controller.collect_wait = 0
        self.addCleanup(backend.close)
        return backend

    def drain(self, backend=None):
        backend = backend or self.backend
        for _ in range(20):
            backend.tick()
            if backend.controller.thread:
                backend.controller.thread.join(timeout=20)
            if not backend.status()['active']:
                return
        self.fail('work did not settle: ' + json.dumps(backend.status()))

    def run_job(self, name, seconds=240, **changes):
        reply = self.broker.handle(dict(op='run', name=name, seconds=seconds, files=dict(self.files, **changes)))
        self.assertNotIn('error', reply, reply)
        return reply

    def jobs(self, backend=None):
        return [(job['name'], job['kind'], job['status']) for job in (backend or self.backend).status()['jobs']]

    def test_run_carries_exactly_the_researchers_files_and_chosen_config(self):
        self.assertEqual(self.broker.handle({'op': 'budget'})['remaining_seconds'], 600)
        files = dict(self.files)
        files['training/policy.py'] += '\n# researcher change\n'
        files['configs/train.json'] = json.dumps(dict(json.loads(files['configs/train.json']), n_games=16, seed=77))
        reply = self.broker.handle({'op': 'run', 'name': 'a', 'seconds': 240, 'config': 'configs/train.json',
                                    'files': files})
        self.assertNotIn('error', reply, reply)
        self.assertEqual((reply['train_seconds'], reply['reserved_seconds'], reply['seed']), (90, 240, 77))
        self.assertEqual(self.broker.handle({'op': 'status'})['active'], ['a'])
        self.assertEqual(self.provider.sandboxes, [])  # queued: nothing allocated by the request itself
        self.drain()
        worker, = self.provider.sandboxes
        self.assertEqual((worker.gpu, worker.timeout, self.provider.readied), (True, 230, 1))  # inside its 240 reserved
        self.assertEqual({name: data.decode() for name, data in worker.files['/job/source'].items()},
                         {name.split('/')[1]: files[name] for name in files if name.endswith('.py')})
        self.assertEqual(json.loads(worker.files['/job/recipe.json']), json.loads(files['configs/train.json']))
        self.assertEqual(set(worker.files), {'/job/source', '/job/recipe.json'})
        self.assertEqual([step(command) for command, _ in worker.commands], ['prepare', 'train'])
        runner, timeout = worker.commands[1]
        spec = json.loads(runner[3])
        command = spec['train']
        self.assertEqual(command[command.index('--seconds') + 1], '90.0')
        # the worker ends the trainer at 155 s itself; the command may run on for the in-datacenter publish
        self.assertEqual((command[-1], spec['limit'], timeout, worker.terminated), ('cuda:0', 155, 155 + native.PUBLISH_SECONDS, 1))
        self.assertEqual(self.jobs(), [('a', 'train', 'failed')])
        budget = self.broker.handle({'op': 'budget'})
        self.assertEqual((budget['used_seconds'], budget['charged_seconds'], budget['remaining_seconds']),
                         (30, 30, 570))  # the failed allocation stays charged at the provider's own duration
        log = self.broker.handle({'op': 'log', 'name': 'a'})
        self.assertNotIn('error', log)  # reading a failed run's log is a successful command
        self.assertIn('stub provider', log['job_error'])
        self.assertIn('error', self.broker.handle({'op': 'submit', 'name': 'a'}))  # nothing trained
        self.assertIn('Unknown run', self.broker.handle({'op': 'log', 'name': 'zzz'})['error'])

    def test_seconds_mode_is_one_job_at_a_time_charged_in_training_worker_seconds(self):
        settings = json.loads((self.out / 'research/session.json').read_text())
        self.assertEqual((settings['budget_mode'], settings['gpu_limit_seconds'], self.backend.budget_mode),
                         ('seconds', 600, 'seconds'))
        self.assertEqual(recorded_mode(self.out / 'research/worker-ledger.sqlite'), 'seconds')
        budget = self.broker.handle({'op': 'budget'})
        self.assertEqual((budget['budget_mode'], budget['concurrency'], budget['budget_seconds']), ('seconds', 1, 600))
        self.assertNotIn('budget_usd', budget)
        worker = self.training()
        self.assertIn('One training run at a time',
                      self.broker.handle(dict(op='run', name='b', seconds=160, files=self.files))['error'])
        self.assertEqual(len(self.provider.sandboxes), 1)
        self.broker.handle({'op': 'stop', 'name': 'a'})
        self.backend.controller.thread.join(timeout=20)
        self.assertEqual(worker.terminated, 1)
        budget = self.backend.budget()  # the provider's 30 s of the training worker, not dollars
        self.assertEqual((budget['used_seconds'], budget['charged_seconds'], budget['remaining_seconds']), (30, 30, 570))
        with self.assertRaisesRegex(ValueError, 'created in seconds mode'):
            self.open(resume=True, budget_usd=1.0)

    def test_bad_requests_never_reach_the_ledger_or_a_worker(self):
        config = json.loads(self.files['configs/train.json'])
        changed_rules = json.dumps(dict(config, rules=dict(config['rules'], num_players=2)))
        missing = json.dumps({key: value for key, value in config.items() if key != 'gamma'})
        good = dict(op='run', name='a', seconds=240, files=self.files)
        for change in (dict(files=dict(self.files, **{'training/learner.py': 'def broken(:'})),
                       dict(files=dict(self.files, extra='x')),
                       dict(seconds=100000),            # more than the session holds
                       dict(seconds=400),               # more than one smoke run may take
                       dict(seconds=100),               # less than the setup/teardown reserve plus training
                       dict(name='../a'), dict(name='a-b'),  # Tavern run names are letters and digits
                       dict(seed=5),                    # the seed belongs to the config
                       dict(config='configs/other.json'),
                       dict(files=dict(self.files, **{'configs/train.json': changed_rules})),
                       dict(files=dict(self.files, **{'configs/train.json': missing}))):
            reply = self.broker.handle(dict(good, **change))
            self.assertIn('error', reply, change)
            self.assertNotIn('Host fault', reply['error'], change)
        self.assertEqual(self.broker.handle({'op': 'grade'})['error'], 'Unknown command')  # grading is host-only
        self.assertEqual(Ledger(self.out / 'research/worker-ledger.sqlite', 600).snapshot()['allocations'], [])
        self.assertEqual(self.backend.status()['jobs'], [])
        self.assertFalse((self.out / 'research/runs').exists())
        self.assertEqual((self.provider.sandboxes, self.provider.readied), ([], 0))

    def test_budget_counts_queued_reserved_and_charged_time(self):
        self.run_job('a')
        budget = self.backend.budget()
        self.assertEqual((budget['budget_seconds'], budget['used_seconds'], budget['remaining_seconds']),
                         (600, 240, 360))
        self.assertEqual((budget['min_run_seconds'], budget['max_run_seconds'], budget['concurrency']), (160, 330, 1))
        self.assertIn('One training run at a time',
                      self.broker.handle(dict(op='run', name='b', seconds=240, files=self.files))['error'])
        self.provider.seconds = 200.0
        self.drain()
        budget = self.backend.budget()
        self.assertEqual((budget['used_seconds'], budget['reserved_seconds'], budget['remaining_seconds']),
                         (200, 0, 400))
        self.run_job('b', seconds=330)
        self.assertEqual(self.backend.budget()['remaining_seconds'], 70)
        self.drain()
        self.assertIn('error', self.broker.handle(dict(op='run', name='c', seconds=160, files=self.files)))
        self.assertEqual(self.jobs(), [('a', 'train', 'failed'), ('b', 'train', 'failed')])  # smoke: two attempts

    def test_network_session_from_training_to_private_grade(self):
        self.provider.train = 'train'
        edited = self.files['training/policy.py'] + '\n# the submitted network\n'
        self.run_job('a', **{'training/policy.py': edited})
        self.drain()
        job, = self.backend.status()['jobs']
        self.assertEqual((job['status'], job['trained'], job['charged_seconds'], job['summary']['updates']),
                         ('complete', True, 30, 1))
        log = self.broker.handle({'op': 'log', 'name': 'a'})
        self.assertEqual(log['stdout'].splitlines()[0], 'trained')
        self.assertEqual(log['progress'], [dict(update=1, learning=dict(loss=0.5, samples=10, parameter_delta_l2=0.1),
                                                episodes=2, natural=1, mean_placement=3.0,
                                                faults=dict(invalid_lobbies=1))])
        reply = self.broker.handle({'op': 'eval', 'name': 'a'})
        self.assertEqual((reply['queued'], reply['charged']), ('practice-a', False))
        self.assertEqual(self.backend.status()['active'], ['practice-a'])
        self.drain()
        self.assertEqual(self.jobs(), [('a', 'train', 'complete'), ('practice-a', 'practice', 'complete')])
        practice = self.provider.sandboxes[1]
        self.assertEqual((practice.gpu, practice.timeout), (False, 900))
        self.assertEqual(practice.files['/opt/evaluator-assets']['learned/entrant-a/policy.py'].decode(), edited)
        self.assertEqual(json.loads(practice.files['/opt/evaluator-panel.json'])['seeds'], [50101, 50102, 50103, 50104])
        self.assertEqual(self.backend.status()['jobs'][0]['practice']['raw_score'], 40.0)
        self.assertEqual(self.backend.budget()['used_seconds'], 30)  # practice runs on a CPU worker, off the ledger
        self.assertIn('complete', self.broker.handle({'op': 'eval', 'name': 'a'})['error'])

        self.assertNotIn('error', self.broker.handle({'op': 'submit', 'name': 'a'}))
        self.run_job('b')
        self.drain()
        self.assertEqual(self.broker.handle({'op': 'submit', 'name': 'b'})['submitted']['native']['submission'], 'b')
        again = self.broker.handle({'op': 'submit', 'name': 'a'})  # an immutable submission is reselected as is
        self.assertEqual(again['submitted']['native']['selected'], 'a')
        self.assertEqual(self.backend.status()['selected_submission'], 'a')
        with self.assertRaisesRegex(ValueError, 'Finalize before private grading'):
            self.backend.grade()
        self.assertEqual(self.broker.handle({'op': 'finalize'})['submission']['name'], 'a')
        self.assertTrue(self.backend.budget()['closed'])
        self.assertIn('error', self.broker.handle(dict(op='run', name='c', seconds=160, files=self.files)))
        self.backend.close()

        graded = self.open(resume=True)  # the host reopens the same session; no new allowance
        self.assertEqual(graded.budget()['budget_seconds'], 600)
        result = graded.grade()
        self.assertEqual((result['observed_cases'], result['benchmark_id']), (288, 'tavern-four-small-v1'))
        worker = self.provider.sandboxes[-1]
        self.assertEqual((worker.gpu, worker.timeout), (False, 1800))
        seeds = json.loads(worker.files['/opt/evaluator-panel.json'])['seeds']
        self.assertEqual(len(set(seeds)), 24)
        self.assertFalse({50101, 50102, 50103, 50104} & set(seeds))
        self.assertEqual(worker.files['/opt/evaluator-assets']['learned/entrant-a/policy.py'].decode(), edited)
        commitment = json.loads((self.out / 'research/grading-private/bank-commitment.json').read_text())
        self.assertEqual(commitment['required_lobbies'], 288)
        graded.close()

        target = self.out / 'research/submissions/a/policy.py'
        target.chmod(0o644)
        target.write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'Submission mutated'):
            self.open(resume=True).grade()

    def test_recipe_final_training_registers_a_network_without_practice(self):
        """What harness.session.final_training does with a fresh root and the final-training allowance."""
        self.provider.train = 'train'
        final = self.open('final', budget_seconds=240)
        workspace.apply(final.release, final.workspace, self.files)
        final.run('final', 240, 'configs/smoke.json', 1)
        self.drain(final)
        self.assertEqual(self.jobs(final), [('final', 'train', 'complete')])
        self.assertEqual(final.submit('final')['submission'], 'final')
        self.assertEqual(final.finalize(), dict(lifecycle='finalized', final_submission='final'))
        self.assertEqual(json.loads(self.provider.sandboxes[0].files['/job/recipe.json']),
                         json.loads(self.files['configs/smoke.json']))
        strict = self.open('strict', budget_seconds=240)
        strict.options['require_practice'] = True
        workspace.apply(strict.release, strict.workspace, self.files)
        strict.run('final', 240, 'configs/smoke.json', 1)
        self.drain(strict)
        with self.assertRaisesRegex(ValueError, 'public practice required'):
            strict.submit('final')

    def test_settle_cancels_queued_work_without_charge(self):
        self.run_job('a')
        self.backend.settle()
        self.assertEqual(self.jobs(), [('a', 'train', 'stopped')])
        self.assertEqual((self.backend.budget()['used_seconds'], self.provider.sandboxes), (0, []))
        self.assertEqual(self.broker.finalize('host')['submission'], None)

    def training(self, name='a'):
        """Start run `name` and return its worker once the training command is running on it."""
        self.provider.train = 'block'
        before = len(self.provider.sandboxes)
        self.run_job(name)
        self.backend.tick()
        deadline = time.monotonic() + 20
        while len(self.provider.sandboxes) == before or not self.provider.sandboxes[before].entered.wait(0.01):
            self.assertLess(time.monotonic(), deadline, 'the training command never started')
        return self.provider.sandboxes[before]

    def test_stop_keeps_the_network_a_running_trainer_last_exported(self):
        self.provider.exported = True
        worker = self.training()
        reply = self.broker.handle({'op': 'stop', 'name': 'a'})
        self.assertEqual((reply['stopping'], 'error' in reply), ('a', False), reply)
        self.assertIn('keeps that network', reply['note'])
        self.backend.controller.thread.join(timeout=20)
        job, = self.backend.status()['jobs']
        self.assertEqual((job['status'], job.get('stopped_early'), job['charged_seconds'], job['summary']['updates']),
                         ('complete', True, 30, 1), job)
        self.assertNotIn('error', job)
        # only the trainer was ended; the worker packaged and published its last export, then was released once
        self.assertEqual((worker.terminated, [step(command) for command, _ in worker.commands]),
                         (1, ['prepare', 'train', 'finalize', 'publish']))
        self.assertEqual((self.backend.controller.snapshot()['runs']['a']['handoff'], worker.live), ('volume', []))
        log = self.broker.handle({'op': 'log', 'name': 'a'})
        self.assertEqual((log['stopped_early'], 'error' in log), (True, False), log)
        self.assertIn('Packaged its last export', log['stdout'])
        self.assertEqual((self.backend.budget()['used_seconds'], self.backend.status()['active']), (30, []))
        self.assertNotIn('error', self.broker.handle({'op': 'submit', 'name': 'a'}))  # a kept network is an answer

    def test_a_trainer_cut_off_at_its_time_limit_keeps_its_last_export_and_a_failed_one_does_not(self):
        self.provider.train, self.provider.exported = 'overrun', True
        self.run_job('a')
        with mock.patch.object(self.backend.controller, 'overran', return_value=True):
            self.drain()
        job, = self.backend.status()['jobs']
        self.assertEqual((job['status'], job.get('cut_short'), job['charged_seconds']), ('complete', True, 30), job)
        # the same failure before the limit is the trainer's own: nothing is packaged for it
        self.run_job('b')
        self.drain()
        self.assertEqual(self.jobs()[1], ('b', 'train', 'failed'))
        self.assertEqual([step(command) for command, _ in self.provider.sandboxes[1].commands], ['prepare', 'train'])

    def test_stop_ends_a_running_worker_and_charges_what_it_used(self):
        worker = self.training()  # its trainer has exported nothing yet
        self.assertEqual(self.jobs(), [('a', 'train', 'running')])
        self.assertEqual(self.backend.budget()['used_seconds'], 240)  # held until the provider reports the end
        self.assertEqual(self.broker.handle({'op': 'stop', 'name': 'a'})['stopping'], 'a')
        self.backend.controller.thread.join(timeout=20)
        self.assertEqual(self.jobs(), [('a', 'train', 'stopped')])
        self.assertEqual(worker.terminated, 1)
        self.assertEqual((self.backend.budget()['used_seconds'], self.backend.status()['active']), (30, []))
        self.assertIn('not active', self.broker.handle({'op': 'stop', 'name': 'a'})['error'])
        # settle() ends whatever is still running when the researcher leaves, and keeps nothing: the worker itself
        # is ended at once, even when its trainer had exported a network
        self.provider.exported = True
        worker = self.training('b')
        self.backend.settle()
        self.assertEqual(self.jobs()[1], ('b', 'train', 'stopped'))
        self.assertEqual((getattr(worker, 'kills', 0), worker.terminated > 0), (0, True))
        self.assertEqual(self.backend.status()['active'], [])

    def test_unknown_worker_end_holds_the_reservation_until_the_provider_reports_it(self):
        self.provider.withhold = True
        self.run_job('a')
        with mock.patch.object(native.time, 'sleep', lambda seconds: None):
            self.backend.tick()
            self.backend.controller.thread.join(timeout=20)
        self.assertEqual(self.jobs(), [('a', 'train', 'uncertain')])
        self.assertEqual(self.backend.status()['active'], ['a'])
        budget = self.backend.budget()
        self.assertEqual((budget['used_seconds'], budget['charged_seconds'], budget['reserved_seconds']),
                         (240, 0, 240))
        self.assertIn('Unresolved allocation',
                      self.broker.handle(dict(op='run', name='b', seconds=160, files=self.files))['error'])
        self.assertIn('error', self.broker.handle({'op': 'finalize'}))
        self.provider.withhold = False  # the recorded worker is asked again; nothing is relaunched
        self.backend.controller.recover_at = 0
        self.drain()
        self.assertEqual(self.jobs(), [('a', 'train', 'failed')])
        self.assertEqual((self.backend.budget()['used_seconds'], len(self.provider.sandboxes)), (30, 1))

    def test_a_queued_worker_is_waited_for_and_charged_only_from_its_start(self):
        self.provider.train, self.provider.queue = 'train', 30
        self.run_job('a')
        self.drain()
        job, = self.backend.status()['jobs']
        self.assertEqual((job['status'], job['charged_seconds']), ('complete', 30), job)
        self.assertGreaterEqual(self.provider.sandboxes[0].reads, 30)

    def test_a_worker_that_never_starts_fails_at_zero_and_the_session_continues(self):
        self.provider.train, self.provider.queue = 'train', None
        self.run_job('a')
        with mock.patch.object(workers, 'START_PATIENCE_SECONDS', 0.3):
            self.drain()
        job, = self.backend.status()['jobs']
        self.assertEqual((job['status'], job['charged_seconds'], job['never_started']), ('failed', 0, True), job)
        self.assertIn('never started', job['error'])
        log = self.broker.handle({'op': 'log', 'name': 'a'})
        self.assertEqual((log['status'], log['charged_seconds']), ('failed', 0))
        self.assertNotIn('error', log)
        self.assertIn('never started', log['job_error'])
        worker, = self.provider.sandboxes
        self.assertEqual((worker.terminated, worker.commands), (1, []))  # nothing ran on it
        row, = Ledger(self.out / 'research/worker-ledger.sqlite', 600).snapshot()['allocations']
        self.assertEqual((row['actual'], json.loads(row['receipt'])['never_started']), (0, True))
        self.assertEqual(self.backend.budget()['used_seconds'], 0)
        self.provider.queue = 0  # the next run is accepted at once
        self.run_job('b')
        self.drain()
        self.assertEqual(self.jobs()[1], ('b', 'train', 'complete'))
        self.assertNotIn('error', self.broker.handle({'op': 'submit', 'name': 'b'}))
        self.assertNotIn('error', self.broker.handle({'op': 'finalize'}))

    def test_stop_while_queued_ends_the_wait_and_charges_nothing(self):
        self.provider.queue = None
        self.run_job('a')
        self.backend.tick()
        deadline = time.monotonic() + 10
        while not self.provider.sandboxes:
            self.assertLess(time.monotonic(), deadline, 'no worker was created')
            time.sleep(0.01)
        self.assertEqual(self.broker.handle({'op': 'stop', 'name': 'a'}), {'stopping': 'a'})
        self.backend.controller.thread.join(timeout=20)
        job, = self.backend.status()['jobs']
        self.assertEqual((job['status'], job['charged_seconds'], job['never_started']), ('stopped', 0, True), job)
        self.assertEqual((self.backend.status()['active'], self.backend.budget()['used_seconds']), ([], 0))

    def interrupted(self, status):
        """A recorded worker whose job lost its thread (the earlier ~60-second give-up, or a host restart)."""
        controller = self.backend.controller
        self.run_job('a')
        with controller.lock:
            controller.state['queue'].remove(['train', 'a'])
            controller.state['runs']['a'].update(status=status, error='authoritative allocation start unavailable')
            controller.save()
        worker = self.provider.create(True, 180)
        write(self.out / 'research/runs/a/sandbox.json', dict(id=worker.id, app_id='app-stub', gpu='L40S',
                                                               timeout_seconds=180))
        controller.ledger.reserve('a', 240)
        return worker

    def test_recovery_settles_a_never_started_uncertain_worker_at_zero(self):
        self.provider.queue = None
        worker = self.interrupted('failed')
        self.assertEqual((self.jobs(), self.backend.status()['active']), ([('a', 'train', 'uncertain')], ['a']))
        self.assertIn('Unresolved allocation',
                      self.broker.handle(dict(op='run', name='b', seconds=160, files=self.files))['error'])
        self.drain()  # the periodic reconciliation terminates it by its recorded identity
        job, = self.backend.status()['jobs']
        self.assertEqual((job['status'], job['charged_seconds'], job['never_started']), ('failed', 0, True), job)
        self.assertEqual((worker.terminated, self.backend.budget()['used_seconds']), (1, 0))
        self.provider.queue = 0
        self.run_job('b')

    def test_a_worker_still_running_after_a_host_restart_keeps_its_hold_until_stopped(self):
        worker = self.interrupted('running')  # started, and its job never ended: reconciliation leaves it running
        self.backend.controller.recover()
        self.assertEqual((self.jobs(), worker.terminated), ([('a', 'train', 'uncertain')], 0))
        self.assertEqual(self.backend.budget()['reserved_seconds'], 240)
        self.backend.settle()  # closing the session stops it and charges its real interval
        self.assertEqual((self.jobs(), worker.terminated), ([('a', 'train', 'stopped')], 1))
        self.assertEqual((self.backend.budget()['reserved_seconds'], self.backend.budget()['charged_seconds']), (0, 30))

    def test_stop_resolves_an_uncertain_job_now(self):
        worker = self.interrupted('failed')  # it started, then its job gave up on it
        reply = self.broker.handle({'op': 'stop', 'name': 'a'})
        self.assertEqual((reply['resolved'], reply['charged_seconds']), (True, 30), reply)
        self.assertNotIn('error', reply)  # the stop succeeded; the run's own failure is `job_error`
        self.assertEqual((worker.terminated, self.backend.status()['active']), (1, []))
        self.assertEqual(self.backend.budget()['used_seconds'], 30)

    def test_finalize_waits_for_a_completed_job_to_finish_settling(self):
        self.provider.train, self.provider.gate = 'train', threading.Event()
        final = self.open('final', budget_seconds=240)
        final.controller.start_poll_seconds = final.controller.abandon_poll_seconds = 0.01
        workspace.apply(final.release, final.workspace, self.files)
        final.run('final', 240, 'configs/smoke.json', 1)
        final.tick()
        deadline = time.monotonic() + 20
        while final.controller.snapshot()['runs']['final']['status'] != 'completed':
            self.assertLess(time.monotonic(), deadline, 'training never completed')
            time.sleep(0.01)
        self.assertTrue(final.controller.busy())  # still waiting for the provider's end receipt
        self.assertEqual((self.jobs(final), final.status()['active']), ([('final', 'train', 'complete')], []))
        self.assertEqual(final.submit('final')['submission'], 'final')
        threading.Timer(0.3, self.provider.gate.set).start()
        self.assertEqual(final.finalize(), dict(lifecycle='finalized', final_submission='final'))
        self.assertEqual(final.budget()['charged_seconds'], 30)

    def test_a_finished_run_travels_through_the_volume_after_its_worker_ended(self):
        self.provider.train = 'train'
        self.backend.controller.settings['record_actions'] = True
        self.run_job('a')
        self.drain()
        worker, = self.provider.sandboxes
        row = self.backend.controller.snapshot()['runs']['a']
        self.assertEqual((row['status'], row['handoff'], worker.terminated, worker.live), ('completed', 'volume', 1, []))
        self.assertNotIn('handoff_fallback', row)
        self.assertEqual(worker.volume.name, row['handoff_volume'])
        self.assertEqual(json.loads((self.out / 'research' / handoff.BINDING).read_text())['volume'], row['handoff_volume'])
        # the worker published as root inside its own training command, before the host ended it
        self.assertEqual([step(command) for command, _ in worker.commands], ['prepare', 'train'])
        published = self.provider.modal.root / worker.volume.name / row['handoff_job']
        manifest = json.loads((published / handoff.MANIFEST).read_text())
        self.assertEqual(set(manifest), set(native.ARTIFACT) | {'trainer-stdout.log', 'diagnostics.tar'})
        artifact = self.out / 'research/runs/a/artifact'
        for name in native.ARTIFACT:
            self.assertEqual(handoff.sha256(artifact / name), manifest[name]['sha256'])
        self.assertEqual((self.out / 'research/runs/a/train.log').read_text(), 'trained\n{"updates": 1}\n')
        self.assertTrue((self.out / 'research/runs/a/diagnostics-handoff.json').is_file())
        self.assertNotIn('error', self.broker.handle({'op': 'submit', 'name': 'a'}))
        self.assertFalse(any(p.name.endswith('.log') for p in (self.out / 'research/submissions/a').iterdir()))

    def test_without_a_publication_a_live_worker_is_read_directly_and_the_job_says_why(self):
        self.provider.train, self.provider.unpublished = 'train', True
        self.run_job('a')
        self.drain()
        worker, = self.provider.sandboxes
        row = self.backend.controller.snapshot()['runs']['a']
        self.assertEqual((row['status'], row['handoff'], worker.live, worker.terminated),
                         ('completed', 'live', ['/job/train'], 1))
        self.assertIn('publish failed in the worker', row['handoff_fallback'])
        self.assertEqual((self.out / 'research/runs/a/train.log').read_text(), 'trained\n{"updates": 1}\n')

    def test_a_lost_reply_does_not_cost_a_run_the_worker_already_published(self):
        self.provider.train, self.provider.lose_reply = 'train', True
        self.run_job('a')
        self.drain()
        worker, = self.provider.sandboxes
        row = self.backend.controller.snapshot()['runs']['a']
        self.assertEqual((row['status'], row['handoff'], worker.live, worker.terminated), ('completed', 'volume', [], 1))
        self.assertEqual((self.out / 'research/runs/a/train.log').read_text(), 'trained\n{"updates": 1}\n')

    def test_recovery_collects_a_run_published_before_its_host_was_interrupted(self):
        controller = self.backend.controller
        worker = self.interrupted('running')
        job = 'train-a-0123456789ab'
        with controller.lock:
            controller.state['runs']['a'].update(handoff_job=job)
            controller.save()
        worker.volume = controller.handoff_volume()
        run = self.out / 'research/runs/a'
        self.provider.upload(worker, run / 'source', '/job/source')
        self.provider.copy(worker, run / 'recipe.json', '/job/recipe.json')
        self.provider._train(worker)
        worker.trainer_log = 'trained\n'
        spec = controller.training_pack(job, False)
        self.provider.publish(worker, spec)
        self.provider.terminate(worker)  # the worker ended by itself; the host never collected
        controller.recover()
        job_row = self.backend.status()['jobs'][0]
        self.assertEqual((job_row['status'], job_row['charged_seconds']), ('complete', 30), job_row)
        row = controller.snapshot()['runs']['a']
        self.assertEqual((row['handoff'], row['recovered']), ('volume', True))
        self.assertNotIn('error', self.broker.handle({'op': 'submit', 'name': 'a'}))

    def test_practice_recordings_come_back_in_one_published_tar(self):
        self.provider.train = 'train'
        self.run_job('a')
        self.drain()
        self.broker.handle({'op': 'eval', 'name': 'a'})
        original = self.provider.command

        def recording(sandbox, args, timeout=220):  # the evaluator also leaves lobby recordings behind
            reply = original(sandbox, args, timeout)
            if step(args) == 'evaluate':
                sandbox.outputs['/private/evaluation/screen/traces/mixed-50101-0'] = {'actual-terminal-game.json': {}}
            return reply
        with mock.patch.object(self.provider, 'command', recording):
            self.drain()
        self.assertEqual(self.jobs()[1], ('practice-a', 'practice', 'complete'))
        practice = self.provider.sandboxes[1]
        self.assertEqual(([step(command) for command, _ in practice.commands], practice.live, practice.terminated),
                         (['prepare', 'evaluate', 'publish'], [], 1))
        self.assertNotEqual(practice.volume.name, self.provider.sandboxes[0].volume.name)
        dest = self.out / 'research/practice-a'
        self.assertTrue((dest / 'retained/screen/traces/mixed-50101-0/actual-terminal-game.json').is_file())
        self.assertEqual(json.loads((dest / 'retained/lifecycle.json').read_text()), dict(status='completed'))
        self.assertEqual(json.loads((dest / 'result.json').read_text())['raw_score'], 40.0)
        self.assertEqual(json.loads((dest / 'handoff.json').read_text())['handoff'], 'volume')
        self.assertFalse((dest / 'retention-blocker.json').exists())
        self.assertFalse((dest / 'handoff/retained.tar').exists())
        self.assertEqual(self.backend.controller.snapshot()['practice']['a']['handoff'], 'volume')

    def test_a_cpu_device_config_is_refused_for_a_gpu_worker(self):
        config = json.loads(self.files['configs/train.json'])
        self.assertEqual(config['device'], 'cuda')
        cpu = dict(self.files, **{'configs/train.json': json.dumps(dict(config, device='cpu'))})
        for request in ({'op': 'check', 'files': cpu}, dict(op='run', name='a', seconds=240, files=cpu)):
            self.assertIn('"device" must be "cuda"', self.broker.handle(request)['error'])
        self.assertEqual((self.provider.sandboxes, self.backend.status()['jobs']), ([], []))
        explicit = dict(self.files, **{'configs/train.json': json.dumps(dict(config, device='cuda:0'))})
        self.assertTrue(self.broker.handle({'op': 'check', 'files': explicit})['checked'])


class TavernDollarMode(unittest.TestCase):
    """The dollar mode: parallel jobs within a dollar budget, each queued or running run holding its whole grant."""
    RATE = budgets.rate(TRAIN_SHAPE)

    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.release, cls.meta = tavern_release(cls.shared.name)

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / 'session'
        self.out.mkdir()
        self.provider = Provider()
        self.addCleanup(self.provider.temp.cleanup)
        base = workspace.public(self.release) / workspace.WORKSPACE
        self.files = {name: (base / name).read_text() for name in self.meta['writable']}

    def open(self, name='research', smoke=False, **kwargs):
        if not kwargs.get('resume'):
            kwargs.setdefault('budget_usd', 1.0)
        backend = registry.backend('tavern')(self.release, self.out / name, smoke=smoke,
                                             options={'_provider': self.provider}, **kwargs)
        backend.open()
        backend.controller.collect_wait = 0
        backend.controller.start_poll_seconds = backend.controller.abandon_poll_seconds = 0.01
        self.addCleanup(backend.close)
        self.backend, self.broker = backend, Broker(backend, self.out, 'network', research_usd=kwargs.get('budget_usd'))
        return backend

    def run_job(self, name, seconds=240, **changes):
        reply = self.broker.handle(dict(op='run', name=name, seconds=seconds, files=dict(self.files, **changes)))
        self.assertNotIn('error', reply, reply)
        return reply

    def refused(self, request):
        return self.broker.handle(dict(dict(op='run', seconds=240, files=self.files), **request))['error']

    def drain(self):
        for _ in range(20):
            self.backend.tick()
            self.backend.controller.join(20)
            if not self.backend.status()['active']:
                return
        self.fail('work did not settle: ' + json.dumps(self.backend.status()))

    def jobs(self):
        return [(job['name'], job['kind'], job['status']) for job in self.backend.status()['jobs']]

    def entered(self, count):
        """Tick, and return the training workers created now once `count` of them are inside their training commands."""
        before = len(self.provider.sandboxes)
        self.backend.tick()
        deadline = time.monotonic() + 20
        while True:
            workers = [s for s in self.provider.sandboxes[before:] if s.gpu]
            if len(workers) >= count and all(s.entered.is_set() for s in workers):
                return workers
            self.assertLess(time.monotonic(), deadline, 'the training commands never started')
            time.sleep(0.01)

    def within_budget(self):
        budget = self.backend.budget()
        self.assertLessEqual(budget['used_usd'], budget['budget_usd'] + 1e-9, budget)
        ledger = self.backend.controller.budget()
        self.assertLessEqual(ledger['charged_usd'] + ledger['unresolved_usd'], ledger['cap_usd'] + 1e-9, ledger)
        return budget

    def test_the_mode_and_its_price_are_recorded_and_never_switched(self):
        self.open(budget_usd=2.5)
        settings = json.loads((self.out / 'research/session.json').read_text())
        self.assertEqual((settings['budget_mode'], settings['budget_usd'], settings['usd_per_second']),
                         ('usd', 2.5, self.RATE))
        self.assertEqual(settings['budget_limits'], budgets.PARALLEL)
        self.assertNotIn('gpu_limit_seconds', settings)
        self.assertEqual(recorded_mode(self.out / 'research/worker-ledger.sqlite'), 'usd')
        budget = self.broker.handle({'op': 'budget'})
        self.assertEqual((budget['budget_mode'], budget['budget_usd'], budget['used_usd'], budget['concurrency'],
                          budget['concurrent_gpus']), ('usd', 2.5, 0, 4, 2))
        self.assertEqual(budget['affordable_seconds'], {'ar run (L40S, 4 CPU, 16 GiB)': int(2.5 / self.RATE)})
        self.assertNotIn('budget_seconds', budget)
        self.assertEqual(self.broker.usable_seconds(), int(2.5 / self.RATE))
        self.backend.close()
        with self.assertRaisesRegex(ValueError, 'created in usd mode'):
            self.open(resume=True, budget_seconds=600)
        reopened = self.open(resume=True)  # host grading: no amount given, the recorded mode stands
        self.assertEqual((reopened.budget_mode, reopened.budget()['budget_usd']), ('usd', 2.5))
        with self.assertRaisesRegex(ValueError, 'Cannot reset budget'):
            DollarLedger(self.out / 'research/worker-ledger.sqlite', 3.0, self.RATE)
        Ledger(self.out / 'seconds.sqlite', 600)
        with self.assertRaisesRegex(ValueError, 'created in seconds mode'):
            DollarLedger(self.out / 'seconds.sqlite', 2.5, self.RATE)
        self.assertEqual(registry.spec('tavern')['research_usd'], round(7200 * self.RATE, 2))
        self.assertEqual(registry.spec('tavern')['budget_mode'], 'seconds')

    def test_training_runs_in_parallel_up_to_the_gpu_limit_each_holding_its_grant(self):
        self.open(budget_usd=0.6)  # 744 s of the training worker
        self.provider.train = 'block'
        self.run_job('a')
        self.run_job('b')
        self.assertIn('At most 2 GPUs', self.refused(dict(name='c', seconds=160)))
        budget = self.within_budget()  # queued runs hold their whole grants before any worker exists
        self.assertAlmostEqual(budget['used_usd'], 480 * self.RATE)
        self.assertEqual((budget['charged_usd'], budget['affordable_seconds']['ar run (L40S, 4 CPU, 16 GiB)']), (0, 264))
        self.assertEqual(self.provider.sandboxes, [])
        workers = {s.id: s for s in self.entered(2)}  # both training commands run at the same time, on two GPU workers
        first, second = (workers[json.loads((self.out / 'research/runs' / name / 'sandbox.json').read_text())['id']]
                         for name in 'ab')
        self.assertEqual((first.gpu, second.gpu, first.terminated, second.terminated), (True, True, 0, 0))
        self.assertEqual(self.backend.status()['active'], ['a', 'b'])
        ledger = self.backend.controller.budget()
        self.assertEqual(sorted((row['name'], row['reserved'], row['actual']) for row in ledger['allocations']),
                         [('a', 240, None), ('b', 240, None)])
        rows = self.backend.controller.snapshot()['runs']
        self.assertNotEqual(rows['a']['handoff_job'], rows['b']['handoff_job'])
        self.assertEqual({self.broker.handle({'op': 'log', 'name': name})['status'] for name in 'ab'}, {'running'})
        # stopping one run ends only its own worker; the other keeps training
        self.assertEqual(self.broker.handle({'op': 'stop', 'name': 'a'})['stopping'], 'a')
        self.backend.controller.threads[('train', 'a')].join(timeout=20)
        self.assertEqual(self.jobs(), [('a', 'train', 'stopped'), ('b', 'train', 'running')])
        self.assertEqual((first.terminated, second.terminated, second.released.is_set()), (1, 0, False))
        budget = self.within_budget()  # a's 30 s are charged from its receipt; b still holds its 240 s
        self.assertAlmostEqual(budget['charged_usd'], 30 * self.RATE)
        self.assertAlmostEqual(budget['used_usd'], 270 * self.RATE)
        self.assertEqual(self.broker.handle({'op': 'log', 'name': 'a'})['status'], 'stopped')
        self.run_job('c')  # a GPU is free again and the grant fits: 30 + 240 + 240 seconds
        self.assertIn('At most 2 GPUs', self.refused(dict(name='d', seconds=160)))
        self.assertEqual(self.broker.handle({'op': 'stop', 'name': 'c'}), dict(stopped='c', charged=False))
        self.assertIn('exceeds the remaining session budget', self.refused(dict(name='d', seconds=500)))
        self.within_budget()
        self.backend.settle()  # closing the session ends every running worker and keeps nothing
        self.assertEqual(self.jobs(), [('a', 'train', 'stopped'), ('b', 'train', 'stopped'), ('c', 'train', 'stopped')])
        self.assertEqual((second.terminated > 0, len(self.provider.sandboxes)), (True, 2))
        budget = self.within_budget()
        self.assertEqual((budget['reserved_usd'], self.backend.status()['active']), (0, []))
        self.assertAlmostEqual(budget['used_usd'], 60 * self.RATE)

    def test_practice_runs_beside_training_within_four_jobs_and_finalize_waits_for_all(self):
        self.open(budget_usd=1.0)
        self.provider.train = 'train'
        self.run_job('a')
        self.run_job('x')
        self.drain()
        self.assertEqual(self.jobs(), [('a', 'train', 'complete'), ('x', 'train', 'complete')])
        self.provider.train = 'block'
        self.run_job('b')
        self.run_job('c')
        self.assertEqual(self.broker.handle({'op': 'eval', 'name': 'a'})['queued'], 'practice-a')
        self.assertEqual(self.broker.handle({'op': 'eval', 'name': 'x'})['queued'], 'practice-x')
        self.assertIn('At most 4 jobs', self.refused(dict(name='d', seconds=160)))
        self.within_budget()
        b, c = self.entered(2)  # two trainings and two practices start together
        for name in ('a', 'x'):
            self.backend.controller.threads[('practice', name)].join(timeout=20)
        self.assertEqual(self.jobs(), [('a', 'train', 'complete'), ('x', 'train', 'complete'), ('b', 'train', 'running'),
                                       ('c', 'train', 'running'), ('practice-a', 'practice', 'complete'),
                                       ('practice-x', 'practice', 'complete')])
        practices = [s for s in self.provider.sandboxes if not s.gpu]
        self.assertEqual((len(practices), b.terminated, c.terminated), (2, 0, 0))
        budget = self.within_budget()  # practice is not charged: two finished runs' 30 s plus two held grants
        self.assertAlmostEqual(budget['charged_usd'], 60 * self.RATE)
        self.assertAlmostEqual(budget['used_usd'], 540 * self.RATE)
        self.assertEqual(self.broker.handle({'op': 'log', 'name': 'practice-a'})['practice']['raw_score'], 40.0)
        self.assertNotIn('error', self.broker.handle({'op': 'submit', 'name': 'a'}))
        self.assertIn('Resolve active work', self.broker.handle({'op': 'finalize'})['error'])
        for name in ('b', 'c'):
            self.broker.handle({'op': 'stop', 'name': name})
        self.backend.controller.join(20)
        self.assertEqual(self.broker.handle({'op': 'finalize'})['submission']['name'], 'a')
        self.assertTrue(self.backend.budget()['closed'])
        self.assertAlmostEqual(self.within_budget()['used_usd'], 120 * self.RATE)

    def test_an_unknown_end_holds_its_grant_and_a_slot_without_blocking_other_runs(self):
        self.open(budget_usd=0.6)
        self.provider.withhold = True
        self.addCleanup(setattr, self.provider, 'withhold', False)  # before close: a held end would keep it waiting
        self.run_job('a')
        with mock.patch.object(native.time, 'sleep', lambda seconds: None):
            self.backend.tick()
            self.backend.controller.threads[('train', 'a')].join(timeout=20)
        self.assertEqual(self.jobs(), [('a', 'train', 'uncertain')])
        self.assertAlmostEqual(self.within_budget()['reserved_usd'], 240 * self.RATE)
        self.run_job('b')  # seconds mode would refuse this: there an unknown end blocks every launch
        self.assertIn('At most 2 GPUs', self.refused(dict(name='c', seconds=160)))
        self.assertIn('exceeds the remaining session budget', self.refused(dict(name='c', seconds=300)))
        self.provider.withhold = False  # the recorded worker is asked again; nothing is relaunched for it
        self.backend.controller.recover_at = 0
        self.drain()
        self.assertEqual(self.jobs(), [('a', 'train', 'failed'), ('b', 'train', 'failed')])
        self.assertEqual(len(self.provider.sandboxes), 2)
        self.assertAlmostEqual(self.within_budget()['used_usd'], 60 * self.RATE)

    def test_dollar_session_with_the_real_ar_command(self):
        """harness.session in dollar mode: the instructions, `ar budget` in dollars, parallel admission, a final grade."""
        provider = self.provider
        provider.train = 'train'
        out, box = Path(self.temp.name) / 'dollar', Path(self.temp.name) / 'box'

        class Stubbed(registry.backend('tavern')):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.options['_provider'] = provider

        def script(ar):
            budget = ar('budget')
            self.assertEqual((budget['budget_mode'], budget['budget_usd'], budget['concurrency']), ('usd', 0.48, 4))
            self.assertEqual(ar('run', 'one', '--seconds', '240', '--config', 'configs/smoke.json')['train_seconds'], 90)
            self.assertEqual(ar('run', 'two', '--seconds', '240', '--config', 'configs/smoke.json')['train_seconds'], 90)
            status = ar('wait')
            self.assertEqual([(job['name'], job['status']) for job in status['jobs']],
                             [('one', 'complete'), ('two', 'complete')])
            self.assertAlmostEqual(ar('budget')['used_usd'], 60 * self.RATE)
            self.assertEqual(ar('submit', 'two')['submitted']['native']['submission'], 'two')
            self.assertTrue(ar('finalize')['finalized'])

        def launcher(session_dir):
            researcher = Researcher(session_dir / 'task', box)
            return Process(lambda: script(researcher))
        with mock.patch.object(registry, 'backend', lambda game: Stubbed):
            result = session.run('tavern', 'network', self.release, out, model='test/model', smoke=True,
                                 budget_mode='usd', launcher=launcher,
                                 find_box=lambda: LocalBox(box / 'ar') if (box / 'ar').is_dir() else None)
        self.assertEqual((result['status'], result['error'], result['budget_mode']), ('researched', None, 'usd'))
        self.assertEqual((result['research_usd'], result['research_seconds'], result['wall_seconds']), (0.48, None, 4800))
        self.assertAlmostEqual(result['budget']['used_usd'], 60 * self.RATE)
        self.assertIn('up to 4 at once', (out / 'task/instruction.md').read_text())
        self.assertFalse(session.official('tavern', result))


class WorkerScripts(unittest.TestCase):
    """The in-worker runner and packer, run for real against local folders."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.mount = self.root / 'volume'

    def run_trainer(self, trainer, limit=30):
        out = self.root / 'out'
        out.mkdir(exist_ok=True)
        pack = native.pack_spec('job1', {'weights.npz': str(out / 'weights.npz'), 'trainer-stdout.log': str(self.root / 'log')},
                                ['trainer-stdout.log'], {'diagnostics.tar': dict(root=str(out), include=['faults'],
                                                                                   path=str(self.root / 'diag.tar'))})
        pack['publish'] = local_publish(pack['publish'], self.mount)
        spec = dict(train=[sys.executable, '-c', trainer, str(out)], limit=limit, log=str(self.root / 'log'), pack=pack,
                    pack_script=native.PACK, marks=[native.PUBLISHED, native.UNPUBLISHED])
        return subprocess.run([sys.executable, '-c', native.RUN_AND_PUBLISH, json.dumps(spec)], capture_output=True,
                              text=True, timeout=60)

    def test_a_finished_trainer_is_published_and_its_output_passes_through(self):
        done = self.run_trainer("import pathlib,sys;o=pathlib.Path(sys.argv[1]);(o/'weights.npz').write_bytes(b'w');"
                                "(o/'faults').mkdir();(o/'faults/u1.json').write_text('{}');print('trained');print('x')")
        self.assertEqual(done.returncode, 0, done.stderr)
        text, manifest, reason = native.split_publication(done.stdout)
        self.assertEqual((text, reason), ('trained\nx\n', None))
        self.assertEqual(set(manifest), {'weights.npz', 'trainer-stdout.log', 'diagnostics.tar'})
        self.assertEqual((self.mount / 'job1/files/trainer-stdout.log').read_text(), 'trained\nx\n')
        native.unpack(self.mount / 'job1/files/diagnostics.tar', self.root / 'diagnostics')
        self.assertEqual((self.root / 'diagnostics/faults/u1.json').read_text(), '{}')

    def test_a_failed_or_overrunning_trainer_publishes_nothing(self):
        done = self.run_trainer("print('partial');raise SystemExit(3)")
        self.assertEqual((done.returncode, done.stdout), (3, 'partial\n'))
        late = self.run_trainer("import time;print('started',flush=True);time.sleep(30)", limit=0.5)
        self.assertEqual(late.returncode, 137)
        self.assertIn('at its time limit', late.stderr)
        self.assertFalse(self.mount.exists())

    def test_a_missing_export_is_reported_by_the_last_line_only(self):
        # a trainer that prints the marker itself cannot claim a publication: only the runner's last line counts
        done = self.run_trainer(f"print({native.PUBLISHED + '{}'!r})")
        text, manifest, reason = native.split_publication(done.stdout)
        self.assertEqual(done.returncode, 0)
        self.assertIsNone(manifest)
        self.assertIn('publish failed in the worker', reason)

    def test_unpack_refuses_paths_outside_its_folder(self):
        archive = self.root / 'bad.tar'
        with tarfile.open(archive, 'w') as tar:
            info = tarfile.TarInfo('../escape')
            info.size = 1
            tar.addfile(info, io.BytesIO(b'x'))
        with self.assertRaisesRegex(ValueError, 'Invalid retained path'):
            native.unpack(archive, self.root / 'dest')
        self.assertFalse((self.root / 'escape').exists())


class Session(unittest.TestCase):
    def test_recipe_session_with_the_real_ar_command_final_training_and_grade(self):
        """harness.session end to end: the in-box `ar`, the broker thread, the worker thread and a fresh final root."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        release, _ = tavern_release(temp.name)
        provider = Provider(train='train')
        self.addCleanup(provider.temp.cleanup)
        out, box = Path(temp.name) / 'session', Path(temp.name) / 'box'

        class Stubbed(registry.backend('tavern')):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.options['_provider'] = provider

        def script(ar):
            budget = ar('budget')
            self.assertEqual((budget['variant'], budget['remaining_seconds'], budget['final_train_seconds']),
                             ('recipe', 600.0, 240.0))
            self.assertTrue(ar('check')['checked'])
            ar.edit('training/learner.py', (ar.workspace / 'training/learner.py').read_text() + '\n# explored\n')
            self.assertIn('at least 160', ar('run', 'short', '--seconds', '60', expect=1)['error'])
            self.assertEqual(ar('run', 'explore', '--seconds', '240', '--config', 'configs/smoke.json')['train_seconds'], 90)
            self.assertEqual(ar('wait')['active'], [])
            self.assertEqual(ar('eval', 'explore')['queued'], 'practice-explore')
            status = ar('wait')
            self.assertEqual([(job['name'], job['status']) for job in status['jobs']],
                             [('explore', 'complete'), ('practice-explore', 'complete')])
            self.assertEqual(ar('log', 'explore')['practice']['raw_score'], 40.0)
            ar.edit('training/learner.py', (ar.workspace / 'training/learner.py').read_text() + '\n# final recipe\n')
            self.assertEqual(ar('submit', '--note', 'final')['submitted']['kind'], 'recipe')
            self.assertTrue(ar('finalize')['finalized'])

        def launcher(session_dir):
            researcher = Researcher(session_dir / 'task', box)
            return Process(lambda: script(researcher))
        with mock.patch.object(registry, 'backend', lambda game: Stubbed):
            result = session.run('tavern', 'recipe', release, out, model='test/model', smoke=True, launcher=launcher,
                                 find_box=lambda: LocalBox(box / 'ar') if (box / 'ar').is_dir() else None)
            self.assertEqual((result['status'], result['error'], result['closed_by']), ('researched', None, 'researcher'))
            self.assertEqual(result['budget']['used_seconds'], 30)
            final = result['final_training']
            self.assertEqual((final['status'], final['error'], final['job']['status']), ('complete', None, 'complete'))
            self.assertEqual(final['native'], dict(lifecycle='finalized', final_submission='final'))
            research, practice, fresh = provider.sandboxes
            self.assertEqual((research.gpu, practice.gpu, fresh.gpu, fresh.timeout), (True, False, True, 230))
            self.assertTrue(research.files['/job/source']['learner.py'].decode().endswith('# explored\n'))
            self.assertTrue(fresh.files['/job/source']['learner.py'].decode().endswith('# final recipe\n'))
            self.assertEqual(set(fresh.files), {'/job/source', '/job/recipe.json'})  # nothing from the research session
            # each root hands off through its own Volume; evaluation workers never mount a training worker's Volume
            names = {research.volume.name, practice.volume.name, fresh.volume.name}
            self.assertEqual(len(names), 3, names)
            self.assertTrue(fresh.volume.name.startswith('tavern-handoff-session-final-'))
            self.assertFalse((out / 'research/submissions').exists())  # research networks are never promoted
            with mock.patch.object(policy, 'require_modal', lambda: None):
                grade = session.grade(out)
        self.assertEqual((grade['observed_cases'], grade['complete_valid_schedule']), (288, True))
        self.assertEqual(json.loads((out / 'grade.json').read_text())['graded_root'], 'final')
        graded = provider.sandboxes[-1]
        self.assertTrue(graded.files['/opt/evaluator-assets']['learned/entrant-final/learner.py'].decode()
                        .endswith('# final recipe\n'))


class Constructor(unittest.TestCase):
    def test_release_holds_the_workspace_and_nothing_for_older_harnesses(self):
        receipt = json.loads(SELECTED.read_text())
        if not (ROOT / receipt['path'] / 'manifest.json').is_file() or not (ROOT / '.venv-tavern/bin/python').is_file():
            self.skipTest('needs the private selected registry and the game interpreter .venv-tavern')
        with tempfile.TemporaryDirectory() as temp:
            release = Path(temp) / 'release'
            done = subprocess.run([sys.executable, '-m', 'harness', 'prepare', 'tavern', '--out', str(release)],
                                  cwd=ROOT, capture_output=True, text=True, timeout=600)
            self.assertEqual(done.returncode, 0, done.stderr[-3000:])
            self.assertEqual(policy.verify_release('tavern', release)['game'], 'tavern')
            public = release / 'public'
            self.assertEqual({p.name for p in public.iterdir()},
                             {'agent_workspace', 'workspace_manifest.json', 'starter', 'recipes', 'practice.json',
                              'practice-full.json', 'benchmark.json', 'CONTRACT.md'})
            self.assertEqual(set(workspace.verify(release)['writable']), set(workspace.TRAINING['tavern'])
                             | set(workspace.CONFIGS))
            from simulators.tavern.autoresearch.rules_enabled.release import verify
            manifest = verify(release)
            frozen = set(manifest['frozen_files'])
            for needed in ('simulators/tavern/autoresearch/train_entry.py', 'simulators/tavern/autoresearch/contract.py',
                           'simulators/tavern/autoresearch/policy_worker.py', 'simulators/tavern/rollout.py',
                           'simulators/tavern/evaluation_readiness/five_hour/guest.py',
                           'vendor/engines/AutoTavern/Cargo.lock'):
                self.assertIn(needed, frozen)
            names = {Path(name).name for name in frozen if name.startswith('simulators/tavern/autoresearch/')}
            self.assertFalse(names & {'driver.py', 'manage.py', 'model_client.py', 'instruction.md', 'ar.py',
                                      'rules_reader.py', 'rules_manifest.json'})
            self.assertTrue((release / 'private/registry/registry.json').is_file())
            self.assertFalse((release / 'private/panel-final.json').exists())


if __name__ == '__main__':
    unittest.main()
