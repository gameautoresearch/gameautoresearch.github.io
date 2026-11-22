"""Dou Dizhu backend against its real controller and ledger; the provider is stubbed before any allocation.

Run with the game's research interpreter: .venv-doudizhu-ar/bin/python -m unittest harness.tests.test_doudizhu
"""
import functools
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from harness import budget as budgets, handoff, policy, registry, session, task, workers, workspace
from harness.backend import STATUSES
from harness.transport import LocalBox
from harness.broker import Broker
from harness.tests.support import ROOT, Process, Researcher, release_fixture
from harness.tests.test_handoff import FolderVolume, publish as publish_locally

try:
    from simulators.doudizhu.autoresearch.bundle import FILES, archive_bundle, verify
    from simulators.doudizhu.autoresearch.contract import DEFAULT, digest, frozen_manifest, sha, write_json
    from simulators.doudizhu.autoresearch.ledger import Ledger
except ImportError as exc:  # wrong interpreter
    raise unittest.SkipTest(f'Dou Dizhu research dependencies unavailable: {exc}')

SIMPLE = [{'name': 'random', 'kind': 'random'}, {'name': 'cooperating-shedder', 'kind': 'heuristic'}]


def private(release):
    """The private half of a release that practice and grading read."""
    folder = Path(release) / 'private'
    folder.mkdir()
    write_json(folder / 'panel.json', {'practice': SIMPLE, 'development': SIMPLE, 'official': None})
    write_json(folder / 'anchors.json', {'qualified': False})
    write_json(folder / 'grade-seeds.json', {'split': 'heldout', 'deal_seed': 7, 'policy_seed': 11, 'deals': 256})


def sealed(folder, code):
    """A source-bound policy package as the worker's exporter leaves it (weights are never loaded on the host)."""
    folder.mkdir(parents=True)
    for name, text in code.items():
        (folder / name).write_text(text)
    (folder / 'model.json').write_text('{}')
    (folder / 'weights.pt').write_bytes(b'stub weights')
    manifest = {'format': 1, 'contract': DEFAULT['version'], 'files': {name: sha(folder / name) for name in FILES},
                'tensors': 1, 'engine': digest(frozen_manifest()), 'model': {}, 'provenance': {}}
    manifest['bundle_sha256'] = digest(manifest)
    write_json(folder / 'manifest.json', manifest)
    return verify(folder, check_tensors=False)['bundle_sha256']


class Provider:
    """Records the exact payload and settles the allocation as a provider would."""

    def __init__(self, outcome='failed'):
        self.outcome = outcome
        self.payloads = []
        self.evaluations = []
        self.entered = threading.Event()
        self.gate = None
        self.lock = threading.Lock()
        self.waiting = 0  # workers claimed and held at the gate right now
        self.most = 0  # the most ever held at once

    def execute(self, ledger, job, stop=None):
        payload = json.loads(job['payload'])
        self.payloads.append(payload)
        folder = ledger.root / 'jobs' / job['name']
        folder.mkdir(parents=True)
        self.entered.set()
        with self.lock:
            self.waiting += 1
            self.most = max(self.most, self.waiting)
        if self.gate is not None:  # a worker that has been claimed but whose allocation has not started yet
            self.gate.wait(10)
        with self.lock:
            self.waiting -= 1
        now = time.time()
        ledger.start(job['name'], 'stub', now)
        if stop and stop():
            ledger.finish(job['name'], now + 7, status='failed', error='stub provider: stopped before any export')
        elif self.outcome == 'failed':
            ledger.finish(job['name'], now + 30, status='failed', error='stub provider: no worker was allocated')
        elif payload['kind'] == 'practice':
            write_json(folder / 'practice.json', {'status': 'development_only', 'raw_score': 41.5, 'games': 48})
            ledger.finish(job['name'], now + 17)
        else:
            (folder / 'progress.jsonl').write_text('{"waves": 1}\nnot json\n{"waves": 2, "mse": 0.5}\n')
            (folder / 'trainer.log').write_text('\n'.join(f'line {i}' for i in range(200)))
            identity = sealed(folder / 'bundle', payload['code'])
            ledger.finish(job['name'], now + job['grant'] - 5, 'complete', folder / 'bundle', identity)

    def evaluate(self, bundle, panel, seeds, anchors=None, development=True):
        self.evaluations.append(dict(bundle=str(bundle), panel=panel, seeds=seeds, development=development))
        return {'public': {'status': 'development_only', 'raw_score': 55.0, 'reward': None},
                'private': {'rows': []}, 'allocation': {'gpu_count': 0}}


class DoudizhuBackend(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.release, self.meta = release_fixture('doudizhu', self.temp.name, bind=False)
        private(self.release)
        policy.bind_release('doudizhu', self.release)
        self.out = Path(self.temp.name) / 'session'
        self.out.mkdir()
        base = workspace.public(self.release) / workspace.WORKSPACE
        self.files = {name: (base / name).read_text() for name in self.meta['writable']}

    def open(self, outcome='failed', variant='network', root='research', budget_seconds=600, smoke=True, **kwargs):
        self.provider = Provider(outcome)
        self.backend = registry.backend('doudizhu')(self.release, self.out / root, smoke=smoke,
                                                    budget_seconds=budget_seconds,
                                                    options={'_provider': self.provider}, **kwargs)
        self.backend.open()
        self.addCleanup(self.backend.close)
        self.broker = Broker(self.backend, self.out, variant, research_seconds=budget_seconds,
                             research_usd=kwargs.get('budget_usd'))
        return self.broker

    def open_usd(self, outcome='complete', usd=0.48):
        """A smoke session in dollar mode: parallel jobs, a dollar budget."""
        return self.open(outcome, budget_seconds=None, budget_usd=usd)

    def held(self, count):
        """Wait until `count` claimed workers are held at the provider's gate at the same time."""
        deadline = time.time() + 10
        while time.time() < deadline and self.provider.waiting < count:
            time.sleep(0.01)
        self.assertEqual(self.provider.waiting, count)

    def work(self):
        self.backend.tick()
        self.backend.controller.join(timeout=10)

    def run_job(self, name, seconds=120, **extra):
        reply = self.broker.handle(dict({'op': 'run', 'name': name, 'seconds': seconds, 'config': 'configs/smoke.json',
                                         'files': self.files}, **extra))
        self.assertNotIn('error', reply, reply)
        self.work()
        return reply

    def test_spec_is_complete_and_importing_the_game_is_cheap(self):
        spec = registry.spec('doudizhu')
        self.assertEqual((spec['research_seconds'], spec['smoke_research_seconds']), (7200, 600))
        self.assertEqual(spec['budget_mode'], 'seconds')  # the game's default mode is unchanged
        # Dollar mode buys the same worker time at the frozen list price, rounded to cents.
        train = {'gpu': 1, 'cpu_cores': 4, 'memory_mib': 16384}
        self.assertEqual((spec['research_usd'], spec['smoke_research_usd']),
                         (round(7200 * budgets.rate(train), 2), round(600 * budgets.rate(train), 2)))
        self.assertEqual((spec['research_usd'], spec['smoke_research_usd']), (5.81, 0.48))
        self.assertEqual((spec['wall_seconds'], spec['smoke_wall_seconds']), (18000, 4800))
        self.assertNotIn('at a time', spec['hardware'])
        self.assertEqual(registry.research('doudizhu', budget_mode='usd'), dict(mode='usd', seconds=None, usd=5.81))
        for mode, sentence in (('seconds', 'one at a time'), ('usd', 'up to 4 at once')):
            text = task.instruction('doudizhu', 'network', research_seconds=600 if mode == 'seconds' else None,
                                    research_usd=0.48 if mode == 'usd' else None, final_train_seconds=180,
                                    wall_seconds=4800, hardware=spec['hardware'], smoke=True, budget_mode=mode)
            self.assertIn(sentence, text)
        self.assertEqual(registry.constructor('doudizhu').__module__, 'simulators.doudizhu.autoresearch.prepare')
        json.dumps(spec)
        code = ("import sys, harness.games.doudizhu as g; "
                "bad = [m for m in ('modal', 'torch', 'numpy', 'simulators.doudizhu') if m in sys.modules]; "
                "print(bad); sys.exit(1 if bad else 0)")
        done = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_run_carries_exactly_the_researchers_files_and_is_charged(self):
        broker = self.open()
        budget = broker.handle({'op': 'budget'})
        self.assertEqual((budget['budget_seconds'], budget['remaining_seconds'], budget['allocatable_seconds']),
                         (600, 600, 580))  # the smoke contract holds 20 seconds back for teardown
        self.files['training/policy.py'] += '\n# researcher change\n'
        self.files['configs/smoke.json'] = json.dumps({'architecture': {'hidden': 64}, 'researcher': True})
        reply = broker.handle({'op': 'run', 'name': 'a', 'seconds': 100, 'seed': 7, 'config': 'configs/smoke.json',
                               'files': self.files})
        self.assertNotIn('error', reply, reply)
        self.assertEqual(reply['reserved_seconds'], 100)
        self.assertEqual(broker.handle({'op': 'status'})['active'], ['a'])
        self.assertEqual(broker.handle({'op': 'budget'})['used_seconds'], 100)  # queued work is reserved
        self.work()
        payload = self.provider.payloads[0]
        self.assertEqual(payload['code'], {name.split('/')[1]: text for name, text in self.files.items()
                                           if name.endswith('.py')})
        self.assertEqual(payload['config'], {'architecture': {'hidden': 64}, 'researcher': True})
        self.assertEqual((payload['kind'], payload['seed']), ('train', 7))
        self.assertEqual(set(payload), {'kind', 'code', 'config', 'seed'})
        status = broker.handle({'op': 'status'})
        self.assertEqual([(j['name'], j['kind'], j['status']) for j in status['jobs']], [('a', 'train', 'failed')])
        self.assertEqual(status['active'], [])
        budget = broker.handle({'op': 'budget'})
        self.assertAlmostEqual(budget['used_seconds'], 30, places=3)  # failed allocations stay charged
        self.assertAlmostEqual(budget['remaining_seconds'], 570, places=3)
        self.assertAlmostEqual(budget['allocatable_seconds'], 550, places=3)
        self.assertIn('error', broker.handle({'op': 'submit', 'name': 'a'}))  # no trained artifact
        self.assertIn('error', broker.handle({'op': 'eval', 'name': 'a'}))
        self.assertIn('Unknown run', broker.handle({'op': 'log', 'name': 'zzz'})['error'])
        log = broker.handle({'op': 'log', 'name': 'a'})
        self.assertEqual((log['status'], log['job_error']), ('failed', 'stub provider: no worker was allocated'))
        self.assertNotIn('error', log)  # a top-level error would make the researcher's command fail

    def test_bad_requests_never_reach_the_ledger(self):
        broker = self.open()
        broken = dict(self.files, **{'training/learner.py': 'def broken(:'})
        for request in ({'op': 'run', 'name': 'a', 'seconds': 120, 'files': broken},
                        {'op': 'run', 'name': 'a', 'seconds': 120, 'files': dict(self.files, extra='x')},
                        {'op': 'run', 'name': 'a', 'seconds': 100000, 'files': self.files},  # over the allowance
                        {'op': 'run', 'name': 'a', 'seconds': 400, 'files': self.files},  # over the smoke per-run cap
                        {'op': 'run', 'name': 'a', 'seconds': 10, 'files': self.files},  # under the minimum run
                        {'op': 'run', 'name': '../a', 'seconds': 120, 'files': self.files},
                        {'op': 'run', 'name': 'a', 'seconds': 120, 'seed': 'x', 'files': self.files},
                        {'op': 'run', 'name': 'a', 'seconds': 120,
                         'files': dict(self.files, **{'configs/train.json': '[1, 2]'})},
                        {'op': 'run', 'name': 'a', 'seconds': 120,
                         'files': dict(self.files, **{'training/policy.py': '#' * (DEFAULT['source_bytes_cap'] + 1)})},
                        {'op': 'eval', 'name': 'a'}, {'op': 'stop', 'name': 'a'}, {'op': 'submit', 'name': 'a'},
                        {'op': 'grade'}, {'op': 'calibrate'}):
            self.assertIn('error', broker.handle(request), request)
        self.assertEqual(Ledger(self.out / 'research').jobs(), [])
        self.assertEqual(self.provider.payloads, [])
        self.assertFalse((self.out / 'sources').exists())
        self.assertEqual(broker.handle({'op': 'budget'})['used_seconds'], 0)

    def test_check_is_free_and_enforces_the_transport_caps(self):
        broker = self.open()
        reply = broker.handle({'op': 'check', 'files': self.files})
        self.assertTrue(reply['checked'])
        self.assertEqual(reply['policy_tensor_cap'], 500_000)
        big = dict(self.files, **{'training/train.py': '#' * (DEFAULT['source_bytes_cap'] + 1)})
        self.assertIn('cap', broker.handle({'op': 'check', 'files': big})['error'])
        listed = dict(self.files, **{'configs/smoke.json': '[]'})
        self.assertIn('JSON object', broker.handle({'op': 'check', 'config': 'configs/smoke.json', 'files': listed})['error'])
        self.assertEqual(Ledger(self.out / 'research').jobs(), [])

    def test_train_practice_submit_and_host_only_grade(self):
        broker = self.open('complete')
        self.run_job('a', 120)
        self.assertIn('Run name already used', broker.handle({'op': 'run', 'name': 'a', 'seconds': 60,
                                                              'files': self.files})['error'])
        reply = broker.handle({'op': 'eval', 'name': 'a'})
        self.assertNotIn('error', reply, reply)
        self.assertEqual(reply['reserved_seconds'], 60)
        self.assertIn('One job at a time', broker.handle({'op': 'run', 'name': 'b', 'seconds': 60,
                                                          'files': self.files})['error'])
        self.work()
        practice = self.provider.payloads[1]
        self.assertEqual((practice['kind'], practice['panel'], practice['seeds']['deals'], practice['seeds']['split']),
                         ('practice', SIMPLE, 8, 'practice'))  # public practice seeds and panel, smoke size
        status = broker.handle({'op': 'status'})
        self.assertEqual([(j['kind'], j['status'], j.get('evaluates')) for j in status['jobs']],
                         [('train', 'complete', None), ('practice', 'complete', 'a')])
        self.assertTrue(all(j['status'] in STATUSES for j in status['jobs']))
        self.assertEqual(status['jobs'][0]['progress'], {'waves': 2, 'mse': 0.5})
        self.assertEqual(status['jobs'][1]['practice']['raw_score'], 41.5)
        log = broker.handle({'op': 'log', 'name': 'a', 'tail': 3})
        self.assertEqual(log['trainer_log'], 'line 197\nline 198\nline 199')
        self.assertEqual(list(log['practice_runs'].values()), [{'status': 'development_only', 'raw_score': 41.5,
                                                               'games': 48}])
        self.assertNotIn('payload', json.dumps(log))
        budget = broker.handle({'op': 'budget'})
        self.assertAlmostEqual(budget['used_seconds'], 115 + 17, places=3)
        self.assertAlmostEqual(budget['budget_seconds'] - budget['used_seconds'], budget['remaining_seconds'], places=3)
        submitted = broker.handle({'op': 'submit', 'name': 'a'})['submitted']
        self.assertEqual(submitted['native']['bundle_sha256'], status['jobs'][0]['artifact_sha'])
        self.assertTrue(broker.handle({'op': 'finalize'})['finalized'])
        self.assertIn('finalized', broker.handle({'op': 'run', 'name': 'c', 'seconds': 60, 'files': self.files})['error'])
        self.assertEqual(self.provider.evaluations, [])  # nothing the researcher did reached private grading
        self.backend.close()

        with self.assertRaises(ValueError):  # an existing session is never reopened with a new allowance
            registry.backend('doudizhu')(self.release, self.out / 'research', smoke=True, budget_seconds=9999)
        grader = Provider()
        reopened = registry.backend('doudizhu')(self.release, self.out / 'research', smoke=True, resume=True,
                                                options={'_provider': grader})
        with reopened:
            self.assertEqual(reopened.budget()['budget_seconds'], 600)
            self.assertAlmostEqual(reopened.budget()['used_seconds'], 132, places=3)
            grade = reopened.grade()
        self.assertEqual((grade['development'], grade['private_deals'], grade['final']['raw_score']), (True, 16, 55.0))
        self.assertEqual(grade['final']['artifact_sha256'], submitted['native']['bundle_sha256'])
        self.assertEqual(len(grader.evaluations), 1)
        self.assertEqual((grader.evaluations[0]['seeds']['deals'], grader.evaluations[0]['panel']), (16, SIMPLE))
        self.assertTrue(grader.evaluations[0]['bundle'].endswith('jobs/a/bundle'))
        seeds = json.loads((self.release / 'private/grade-seeds.json').read_text())
        self.assertEqual(seeds['deals'], 256)  # a development prefix never rewrites the release

    def test_a_full_allowance_session_is_not_graded_officially_without_calibration(self):
        self.open('complete', budget_seconds=None, smoke=False)
        self.assertEqual(self.backend.budget()['budget_seconds'], 7200)
        self.assertEqual(self.backend.budget()['teardown_reserve_seconds'], 0)
        self.run_job('a', 120, config='configs/train.json')
        self.broker.handle({'op': 'submit', 'name': 'a'})
        self.broker.handle({'op': 'finalize'})
        self.backend.official = True  # the host marks a session run at the declared budgets
        with self.assertRaisesRegex(ValueError, 'calibration'):
            self.backend.grade()
        self.assertEqual(self.provider.evaluations, [])

    def test_fresh_final_training_fits_the_smoke_allowance(self):
        seconds = registry.budgets('doudizhu', smoke=True)[1]
        self.open('complete', variant='recipe', root='final', budget_seconds=seconds)
        self.assertEqual(workspace.apply(self.release, self.backend.workspace, self.files),
                         workspace.validate(self.meta, self.files))
        reply = self.backend.run('final', seconds, 'configs/smoke.json', 1)
        self.assertEqual(reply['reserved_seconds'], seconds - 20)  # the ledger's teardown reserve is never granted
        self.assertIn('reserved for worker teardown', reply['note'])
        self.work()
        status = self.backend.status()
        self.assertEqual([(j['name'], j['status']) for j in status['jobs']], [('final', 'complete')])
        self.assertEqual(self.backend.submit('final')['submission'], 1)
        self.assertEqual(len(self.backend.finalize()['submissions']), 1)
        self.assertLessEqual(self.backend.budget()['used_seconds'], seconds)

    def test_settle_cancels_queued_work_without_charge(self):
        broker = self.open()
        broker.handle({'op': 'run', 'name': 'a', 'seconds': 120, 'files': self.files})
        self.backend.settle()
        status = self.backend.status()
        self.assertEqual([(j['name'], j['status']) for j in status['jobs']], [('a', 'stopped')])
        self.assertEqual(self.backend.budget()['used_seconds'], 0)
        self.assertEqual(self.provider.payloads, [])
        self.assertEqual(broker.finalize('host')['submission'], None)
        self.backend.settle()  # idempotent

    def test_settle_stops_a_claimed_worker_through_the_provider_and_charges_it(self):
        broker = self.open()
        self.provider.gate = threading.Event()
        broker.handle({'op': 'run', 'name': 'a', 'seconds': 120, 'files': self.files})
        self.backend.tick()
        self.assertTrue(self.provider.entered.wait(10))
        self.assertEqual(Ledger(self.out / 'research').jobs()[0]['status'], 'queued')  # claimed, not yet started
        threading.Timer(0.2, self.provider.gate.set).start()
        self.backend.settle()
        job = self.backend.status()['jobs'][0]
        self.assertEqual((job['status'], job['error']), ('failed', 'stub provider: stopped before any export'))
        self.assertAlmostEqual(self.backend.budget()['used_seconds'], 7, places=3)
        self.assertEqual(self.backend.status()['active'], [])

    def test_controller_takes_only_the_source_whitelist_and_has_no_grading_action(self):
        self.open()
        controller = self.backend.controller
        for request in ({'action': 'run', 'name': 'bad', 'seconds': 60, 'code': {'engine.cpp': 'bad'}, 'config': {}},
                        {'action': 'run', 'name': 'bad', 'seconds': 60, 'code': {'policy.py': 'x = 1'}, 'config': {}},
                        {'action': 'eval', 'name': 'arbitrary', 'seconds': 60},
                        {'action': 'grade'}, {'action': 'rules'}, {'action': 'workspace'}):
            with self.assertRaises(ValueError, msg=request):
                controller.handle(request)
        self.assertEqual(controller.ledger.jobs(), [])

    def test_reopening_settles_an_interrupted_lease_only_from_provider_receipts(self):
        from harness.games.doudizhu import provider as module

        class Recovering(Provider):
            recover_jobs = module.ModalProvider.recover_jobs  # the real recovery; only provider metadata is patched
            recover = module.ModalProvider.recover

        self.open()
        ledger = Ledger(self.out / 'research')
        ledger.queue('lost', 100, {'kind': 'train'})
        ledger.start('lost', 'sandbox', 200)  # the host died while this worker was allocated
        write_json(self.out / 'research/jobs/lost/provider.json', {'app_id': 'app', 'sandbox_id': 'sandbox'})

        def reopen(receipt):
            backend = registry.backend('doudizhu')(self.release, self.out / 'research', smoke=True, resume=True,
                                                   options={'_provider': Recovering()})
            patcher = mock.patch.object(module, 'allocation_info', lambda *args: receipt)
            patcher.start()  # kept through close, which also settles unresolved leases from provider metadata
            self.addCleanup(patcher.stop)
            backend.open()
            self.addCleanup(backend.close)
            return backend
        held = reopen(None)  # the provider cannot yet say when the lease ended: the whole grant stays held
        self.assertEqual([(j['status'], j['charged']) for j in held.status()['jobs']], [('uncertain', 0)])
        self.assertEqual(held.status()['active'], ['lost'])
        self.assertEqual(held.budget()['used_seconds'], 100)
        with self.assertRaises(ValueError):
            held.finalize()
        settled = reopen({'started_at': 200, 'finished_at': 245})
        self.assertEqual([(j['status'], j['charged']) for j in settled.status()['jobs']], [('failed', 45)])
        self.assertEqual(settled.status()['active'], [])
        self.assertEqual(settled.budget()['used_seconds'], 45)  # the exact provider lease, never a refund

    def test_seconds_mode_runs_one_job_at_a_time_charged_in_worker_seconds(self):
        broker = self.open('complete')
        budget = broker.handle({'op': 'budget'})
        self.assertEqual((budget['budget_mode'], budget['concurrency'], budget['budget_seconds']), ('seconds', 1, 600))
        self.assertFalse(any(key.endswith('_usd') for key in budget))
        config = Ledger(self.out / 'research').config
        self.assertEqual((config['budget_mode'], config['max_concurrent_jobs'], config['max_concurrent_gpus']),
                         ('seconds', 1, 1))
        self.assertNotIn('budget_usd', config)
        self.provider.gate = threading.Event()
        reply = broker.handle({'op': 'run', 'name': 'a', 'seconds': 120, 'files': self.files})
        self.assertEqual(reply['reserved_seconds'], 120)
        self.assertFalse(any(key.endswith('_usd') or key.startswith('usd') for key in reply))
        self.backend.tick()
        self.held(1)
        self.assertIn('One job at a time', broker.handle({'op': 'run', 'name': 'b', 'seconds': 60,
                                                          'files': self.files})['error'])
        self.backend.tick()  # nothing else is claimed while the one worker thread lives
        self.assertEqual(len(self.provider.payloads), 1)
        self.provider.gate.set()
        self.backend.controller.join(timeout=10)
        self.assertEqual(self.provider.most, 1)
        job = broker.handle({'op': 'status'})['jobs'][0]
        self.assertEqual((job['status'], job['charged']), ('complete', 115))  # the training worker's lease seconds
        self.assertNotIn('charged_usd', job)
        self.assertAlmostEqual(broker.handle({'op': 'budget'})['used_seconds'], 115, places=3)

    def test_a_running_job_holds_its_whole_grant_in_dollar_mode_only(self):
        for mode in ('seconds', 'usd'):
            with self.subTest(mode=mode):
                self.out = Path(self.temp.name) / ('held-' + mode)
                self.out.mkdir()
                self.open_usd() if mode == 'usd' else self.open()
                ledger = Ledger(self.out / 'research')
                ledger.queue('a', 300, {'kind': 'train'})
                ledger.start('a', 'stub', time.time() - 50)  # running for 50 s of a 300 s grant
                budget = self.backend.budget()
                if mode == 'seconds':
                    self.assertAlmostEqual(budget['used_seconds'], 50, delta=1)  # unchanged: time so far
                else:
                    self.assertAlmostEqual(budget['used_usd'], 300 * budgets.rate(TRAIN), places=9)
                    self.assertAlmostEqual(ledger.progress(), 50, delta=1)  # milestones still follow time so far
                ledger.finish('a', ledger.jobs()[0]['started'] + 80, 'failed', error='stub')
                self.backend.close()

    def test_dollar_mode_runs_jobs_side_by_side_within_the_dollar_budget(self):
        broker = self.open_usd()
        rate = budgets.rate(TRAIN)
        budget = broker.handle({'op': 'budget'})
        self.assertEqual((budget['budget_mode'], budget['budget_usd'], budget['concurrency'],
                          budget['concurrent_gpus']), ('usd', 0.48, 4, 2))
        self.assertFalse(any(key in budget for key in ('budget_seconds', 'used_seconds', 'remaining_seconds')))
        self.assertAlmostEqual(budget['allocatable_usd'], 0.48 - 20 * rate, places=9)  # the teardown reserve
        self.assertEqual(budget['affordable_seconds'], {'run or eval (L40S, 4 CPU, 16 GiB)': int(0.48 / rate - 20)})
        config = Ledger(self.out / 'research').config
        self.assertEqual((config['budget_mode'], config['budget_usd'], config['usd_per_second'],
                          config['max_concurrent_jobs'], config['max_concurrent_gpus']), ('usd', 0.48, rate, 4, 2))
        self.provider.gate = threading.Event()
        reply = broker.handle({'op': 'run', 'name': 'a', 'seconds': 300, 'files': self.files})
        self.assertNotIn('error', reply, reply)
        self.assertEqual((reply['reserved_seconds'], reply['usd_per_second']), (300, rate))
        self.assertAlmostEqual(reply['reserved_usd'], 300 * rate, places=9)
        # A grant beyond what the budget still holds is refused: 300 + 300 s is more than the $0.48 buys.
        self.assertIn('exceeds remaining', broker.handle({'op': 'run', 'name': 'b', 'seconds': 300,
                                                          'files': self.files})['error'])
        self.assertNotIn('error', broker.handle({'op': 'run', 'name': 'b', 'seconds': 270, 'files': self.files}))
        # Every worker has a GPU, so a third is refused by the 2-GPU limit before the 4-job one.
        self.assertIn('At most 2 GPUs', broker.handle({'op': 'run', 'name': 'c', 'seconds': 30,
                                                       'files': self.files})['error'])
        grants = [(j['grant'], j['status']) for j in Ledger(self.out / 'research').jobs()]
        self.assertEqual(grants, [(300, 'queued'), (270, 'queued')])
        self.assertLessEqual(sum(g for g, _ in grants) * rate + 20 * rate, 0.48 + 1e-9)
        self.assertAlmostEqual(broker.handle({'op': 'budget'})['used_usd'], 570 * rate, places=9)
        self.backend.tick()
        self.held(2)  # both workers claimed at once, each on its own thread
        self.assertEqual(sorted(self.backend.status()['active']), ['a', 'b'])
        self.provider.gate.set()
        self.backend.controller.join(timeout=10)
        self.assertEqual(self.provider.most, 2)
        jobs = {j['name']: j for j in broker.handle({'op': 'status'})['jobs']}
        self.assertEqual({name: (j['status'], j['charged']) for name, j in jobs.items()},
                         {'a': ('complete', 295), 'b': ('complete', 265)})
        self.assertAlmostEqual(jobs['a']['charged_usd'], 295 * rate, places=9)
        budget = broker.handle({'op': 'budget'})
        self.assertAlmostEqual(budget['used_usd'], 560 * rate, places=9)  # dollars from the provider's seconds
        self.assertAlmostEqual(budget['remaining_usd'], 0.48 - 560 * rate, places=9)
        # Practice of one run beside training of another.
        self.provider.gate = threading.Event()
        self.provider.gate.set()
        self.assertNotIn('error', broker.handle({'op': 'eval', 'name': 'a', 'seconds': 10}))
        self.assertIn('exceeds remaining', broker.handle({'op': 'run', 'name': 'c', 'seconds': 60,
                                                          'files': self.files})['error'])
        self.backend.tick()
        self.backend.controller.join(timeout=10)
        self.assertNotIn('error', broker.handle({'op': 'submit', 'name': 'b'}))
        self.assertTrue(broker.handle({'op': 'finalize'})['finalized'])
        native = json.loads((self.out / 'final.json').read_text())['native']
        self.assertAlmostEqual(native['used_usd'], (560 + 17) * rate, places=9)

    def test_stop_log_and_settle_with_several_active_jobs(self):
        broker = self.open_usd(usd=5.81)
        self.provider.gate = threading.Event()
        for name in ('a', 'b'):
            self.assertNotIn('error', broker.handle({'op': 'run', 'name': name, 'seconds': 120,
                                                     'config': 'configs/smoke.json', 'files': self.files}))
        self.backend.tick()
        self.held(2)
        reply = broker.handle({'op': 'stop', 'name': 'a'})  # stops only a
        self.assertNotIn('error', reply, reply)
        for name in ('a', 'b'):
            log = broker.handle({'op': 'log', 'name': name})
            self.assertEqual((log['name'], log['status']), (name, 'queued'))
        self.assertIn('Resolve active work', broker.handle({'op': 'finalize'})['error'])
        self.provider.gate.set()
        self.backend.controller.join(timeout=10)
        jobs = {j['name']: (j['status'], j['charged']) for j in broker.handle({'op': 'status'})['jobs']}
        self.assertEqual(jobs, {'a': ('failed', 7), 'b': ('complete', 115)})  # a stopped through its worker
        # Settle stops every active job at once, queued (claimed) and unclaimed alike, and waits for all.
        self.provider.gate = threading.Event()
        for name in ('c', 'd'):
            self.assertNotIn('error', broker.handle({'op': 'run', 'name': name, 'seconds': 120,
                                                     'config': 'configs/smoke.json', 'files': self.files}))
        self.backend.tick()
        self.held(2)
        threading.Timer(0.2, self.provider.gate.set).start()
        self.backend.settle()
        status = self.backend.status()
        self.assertEqual(status['active'], [])
        self.assertEqual({j['name']: j['status'] for j in status['jobs'] if j['name'] in 'cd'},
                         {'c': 'failed', 'd': 'failed'})
        self.assertEqual(broker.finalize('host')['submission']['name'], 'b')  # the newest complete run

    def test_a_session_resumes_only_in_its_own_mode(self):
        self.open_usd()
        self.backend.close()
        with self.assertRaisesRegex(ValueError, 'usd mode'):
            registry.backend('doudizhu')(self.release, self.out / 'research', smoke=True, resume=True,
                                         budget_seconds=600, options={'_provider': Provider()}).open()
        reopened = registry.backend('doudizhu')(self.release, self.out / 'research', smoke=True, resume=True,
                                                options={'_provider': Provider()})
        with reopened:
            self.assertEqual((reopened.budget_mode, reopened.limits['max_concurrent_jobs']), ('usd', 4))
            self.assertEqual((reopened.budget()['budget_mode'], reopened.budget()['budget_usd']), ('usd', 0.48))
        with self.assertRaisesRegex(ValueError, 'immutable'):  # the ledger refuses another recorded mode as well
            Ledger(self.out / 'research', dict(Ledger(self.out / 'research').config, budget_mode='seconds'))


TRAIN = {'gpu': 1, 'cpu_cores': 4, 'memory_mib': 16384}


class FakeClock:
    """Monotonic time for the provider's queue wait: each poll advances it without really waiting."""

    def __init__(self):
        self.now = 0.

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        time.sleep(0.001)  # let a concurrent stop request in


class FakeProcess:
    def __init__(self):
        self.returncode = 0
        self.stdout = io.StringIO('')
        self.stderr = io.StringIO('')

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode


class FakeFilesystem:
    def __init__(self, sandbox):
        self.sandbox = sandbox

    def write_text(self, text, path):
        self.sandbox.files[path] = text

    def read_text(self, path):
        self.sandbox.commands.append(('read', path))
        if self.sandbox.gone:
            raise RuntimeError('NotFoundError: Modal Sandbox not found. This means this Sandbox has already shut down.')
        if path not in self.sandbox.files:
            raise FileNotFoundError(path)
        return self.sandbox.files[path]

    def copy_from_local(self, local, remote):
        self.sandbox.files[remote] = Path(local).read_bytes()

    def copy_to_local(self, remote, local):
        self.sandbox.commands.append(('copy_to_local', remote))
        if self.sandbox.gone:
            raise RuntimeError('NotFoundError: Modal Sandbox not found. This means this Sandbox has already shut down.')
        if not isinstance(self.sandbox.files.get(remote), bytes):
            raise FileNotFoundError(remote)
        Path(local).write_bytes(self.sandbox.files[remote])


class LoggedVolume(FolderVolume):
    """The session's hand-off Volume: a local folder whose reads are logged (and can fail like a broken link)."""

    def __init__(self, cloud, root):
        super().__init__(root)
        self.cloud = cloud

    def read_file(self, path):
        self.cloud.events.append(('read', path))
        if self.cloud.broken_reads > 0:
            self.cloud.broken_reads -= 1
            raise ConnectionError('host link dropped')
        yield from super().read_file(path)


class FakeSandbox:
    def __init__(self, cloud, object_id, options=None):
        self.cloud, self.object_id, self.options = cloud, object_id, options or {}
        self.files, self.commands = {}, []
        self.filesystem = FakeFilesystem(self)
        self.created = cloud.clock.now
        self.started_at = self.finished_at = 0.
        self.terminated = False
        self.gone = False  # True: every later call fails as if the sandbox had hit its timeout
        self.spec = None  # the chain's arguments, once the host has started it
        self.chain_end = None
        self.prepared = False

    def exec(self, *command, **kwargs):
        self.commands.append(command)
        process = FakeProcess()
        if self.gone:
            raise RuntimeError('NotFoundError: Modal Sandbox not found. This means this Sandbox has already shut down.')
        if command[0] == 'tail':  # the provider's bounded log read: tail -c BYTES PATH
            if self.cloud.tail_gate is not None and command[-1] == '/job/output/progress.jsonl':
                # A read that stalls (3-hour finals: whole-file reads hung for minutes) until its exec timeout.
                self.cloud.stalled += 1
                answered = self.cloud.tail_gate.wait(kwargs.get('timeout'))
                self.cloud.stalled -= 1
                if not answered:
                    process.returncode = 124
                    return process
            if self.gone:
                raise RuntimeError('NotFoundError: Modal Sandbox not found. This means this Sandbox has already shut down.')
            text = self.files.get(command[-1])
            process.returncode = 0 if text is not None else 1
            process.stdout = io.StringIO((text or '')[-int(command[2]):])
        elif command == self.cloud.module.KILL_RESEARCHER:  # ends the trainer: the chain moves on to its export
            if self.chain_end is not None:
                self.chain_end = min(self.chain_end, time.time())
        elif command[1:3] == ('-c', handoff.PREPARE):
            self.prepared = True
        elif command[1:3] == ('-c', self.cloud.module.CHAIN):
            self.chain(json.loads(command[3]), process)
        elif 'integrations.doudizhu.autoresearch.evaluate_entry' in command:
            self.files['/job/evaluation-public.json'] = json.dumps({'status': 'development_only', 'raw_score': 12.0})
        return process

    def chain(self, spec, process):
        """The worker's own train -> kill -> export -> publish sequence, publishing with the real hand-off script."""
        self.spec = spec
        self.files['/job/trainer.log'] = 'trainer line\n' * 3
        self.files['/job/output/progress.jsonl'] = '{"waves": 1}\n{"waves": 2}\n'
        process.returncode = None
        self.chain_end = time.time() + self.cloud.train_seconds

        def finish():
            from simulators.doudizhu.autoresearch.contract import EDITABLE
            self.cloud.export_while_stalled = self.cloud.stalled > 0
            self.files['/job/supervisor.json'] = json.dumps({'status': 'completed', 'returncode': 0})
            stage = Path(self.cloud.temp) / self.object_id / 'stage'
            stage.mkdir(parents=True)
            published = {}
            for name, path in spec['logs'].items():
                (stage / name).write_text(self.files[path])
                published[name] = stage / name
            if self.cloud.export_fails:
                record = {'train': 0, 'export': 1}
                self.files[spec['stage'] + '/export.err'] = 'ValueError: Policy tensor cap exceeded\n'
                (stage / 'export-error.log').write_text(self.files[spec['stage'] + '/export.err'])
                published['export-error.log'] = stage / 'export-error.log'
            else:
                record = {'train': 0, 'export': 0}
                folder = Path(self.cloud.temp) / self.object_id / 'export'
                sealed(folder, {name: self.files['/job/input/' + name] for name in EDITABLE})
                archive_bundle(folder, stage / 'bundle.tar')
                self.files[spec['archive']] = (stage / 'bundle.tar').read_bytes()
                published['bundle.tar'] = stage / 'bundle.tar'
            self.files[spec['stage'] + '/chain.json'] = json.dumps(record)
            (stage / 'chain.json').write_text(json.dumps(record))
            published['chain.json'] = stage / 'chain.json'
            if self.cloud.publish_fails:
                process.returncode, process.stderr = 1, io.StringIO('sync: Volume commit failed\n')
            else:
                done = publish_locally(self.options['volumes'][handoff.MOUNT].root, spec['publish'][3], published)
                assert done.returncode == 0, done.stderr
                process.returncode = 0
                process.stdout = io.StringIO('' if self.cloud.reply_lost else done.stdout)
            if self.cloud.vanish_after_export:
                self.gone = True

        def poll():
            if self.cloud.poll_errors > 0:
                self.cloud.poll_errors -= 1
                raise ConnectionError('host link dropped')
            if process.returncode is None and time.time() >= self.chain_end:
                finish()
            return process.returncode
        process.poll = poll

    def terminate(self, wait=True):
        self.cloud.events.append(('terminate', self.object_id))
        self.terminated = True
        if self.started_at and not self.finished_at:
            self.finished_at = time.time()


class Cloud:
    """Modal stand-in: created sandboxes start `start_after` queue seconds later (None: never)."""

    def __init__(self, temp, clock, module, start_after=None):
        self.temp, self.clock, self.module, self.start_after = temp, clock, module, start_after
        self.vanish_after_export = False
        self.tail_gate = None  # a threading.Event the progress tail waits on (None: answers at once)
        self.train_seconds = 0  # how long the trainer runs inside the chain (0: it has already exited)
        self.stalled = 0  # progress reads currently stalled
        self.export_while_stalled = None
        self.export_fails = False  # the worker's export refuses the trainer's output
        self.publish_fails = False  # the in-worker publish fails: nothing reaches the Volume
        self.reply_lost = False  # the chain published but its reply never reached the host
        self.poll_errors = 0  # host polls of the chain that fail as over a broken link
        self.broken_reads = 0  # Volume reads that fail as over a broken link
        self.events = []  # ('terminate', sandbox) and ('read', Volume path), in order
        self.volumes = {}
        self.sandboxes = {}
        self.created = threading.Event()
        self.lock = threading.Lock()  # parallel workers create sandboxes at the same time

    def add(self, object_id, options=None):
        self.sandboxes[object_id] = FakeSandbox(self, object_id, options)
        return self.sandboxes[object_id]

    def create(self, *command, **kwargs):
        with self.lock:
            sandbox = self.add('sb-' + str(len(self.sandboxes)), kwargs)
        self.created.set()
        return sandbox

    def from_id(self, object_id):
        return self.sandboxes[object_id]

    def volume(self, name, environment_name=None, **options):
        self.volumes.setdefault(name, []).append((environment_name, options))
        root = Path(self.temp) / 'volumes' / name
        root.mkdir(parents=True, exist_ok=True)
        return LoggedVolume(self, root)

    def info(self, app_id, object_id):
        sandbox = self.sandboxes.get(object_id)
        if sandbox is None:
            return None
        if (not sandbox.started_at and not sandbox.terminated and self.start_after is not None
                and self.clock.now - sandbox.created >= self.start_after):
            sandbox.started_at = time.time()
        return {'provider_id': object_id, 'task_id': 'ta-' + object_id if sandbox.started_at else '',
                'created_at': 1., 'started_at': sandbox.started_at, 'finished_at': sandbox.finished_at,
                'gpu_type': 'L40S' if sandbox.started_at else ''}


class DoudizhuWorkerQueue(unittest.TestCase):
    """The real provider, controller and ledger with Modal stubbed: a worker queued for a GPU is never charged."""

    def setUp(self):
        from harness.games.doudizhu import provider as module
        import modal  # the pinned SDK; only its Sandbox entry point is replaced, so nothing reaches Modal
        self.module = module
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.release, self.meta = release_fixture('doudizhu', self.temp.name, bind=False)
        private(self.release)
        policy.bind_release('doudizhu', self.release)
        self.clock = FakeClock()
        self.cloud = Cloud(self.temp.name, self.clock, module)
        sandbox = types.SimpleNamespace(create=self.cloud.create, from_id=self.cloud.from_id)
        volume = types.SimpleNamespace(from_name=self.cloud.volume)
        for patcher in (mock.patch.object(modal, 'Sandbox', sandbox),
                        mock.patch.object(modal, 'Volume', volume),
                        mock.patch.multiple(module.ModalProvider, PUBLISHED_WAIT=0., COLLECT_WAIT=0.),
                        mock.patch.object(module, 'allocation_info', self.cloud.info),
                        mock.patch.object(workers, 'await_start', functools.partial(
                            workers.await_start, poll=5., sleep=self.clock.sleep, clock=self.clock.clock)),
                        mock.patch.object(workers, 'abandon', functools.partial(
                            workers.abandon, sleep=self.clock.sleep, clock=self.clock.clock)),
                        mock.patch.object(workers, 'START_PATIENCE_SECONDS', 600.)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.out = Path(self.temp.name) / 'session'
        self.out.mkdir()
        base = workspace.public(self.release) / workspace.WORKSPACE
        self.files = {name: (base / name).read_text() for name in self.meta['writable']}
        self.open()

    def provider(self):
        provider = self.module.ModalProvider.__new__(self.module.ModalProvider)
        provider.__dict__.update(release=Path(self.release), app=types.SimpleNamespace(app_id='app'), image=None,
                                 export_reserve_seconds=15)
        return provider

    def open(self, resume=False, usd=None):
        self.backend = registry.backend('doudizhu')(self.release, self.out / 'research', smoke=True,
                                                    budget_seconds=None if resume or usd else 600, budget_usd=usd,
                                                    resume=resume, options={'_provider': self.provider()})
        self.backend.open()
        self.addCleanup(self.backend.close)
        self.broker = Broker(self.backend, self.out, 'network', research_seconds=None if usd else 600,
                             research_usd=usd)
        return self.backend

    def run_job(self, name, wait=True):
        reply = self.broker.handle({'op': 'run', 'name': name, 'seconds': 120, 'config': 'configs/smoke.json',
                                    'files': self.files})
        self.assertNotIn('error', reply, reply)
        self.backend.tick()
        if wait:
            self.backend.controller.join(timeout=60)
            self.assertFalse(self.backend.controller.busy())

    def job(self, name):
        return next(j for j in self.broker.handle({'op': 'status'})['jobs'] if j['name'] == name)

    def events(self):
        return [row['kind'] for row in Ledger(self.out / 'research').db.execute('SELECT kind FROM events')]

    def recorded_uncertain(self, name):
        """A job an older controller gave up on: its sandbox was created and recorded but never started."""
        ledger = Ledger(self.out / 'research')
        ledger.queue(name, 120, {'kind': 'train'})
        ledger.uncertain(name, 'RuntimeError: Provider did not expose an authoritative GPU allocation start')
        sandbox = self.cloud.add('sb-' + name)
        write_json(self.out / 'research/jobs' / name / 'provider.json', {'app_id': 'app', 'sandbox_id': sandbox.object_id})
        self.assertEqual(self.broker.handle({'op': 'status'})['active'], [name])
        self.assertIn('unresolved provider lease', self.broker.handle(
            {'op': 'run', 'name': 'blocked', 'seconds': 120, 'files': self.files})['error'])
        return sandbox

    def test_a_worker_that_starts_after_a_long_queue_is_waited_for_and_trains(self):
        self.cloud.start_after = 90  # longer than the old 60-second give-up
        self.files['training/policy.py'] += '\n# researcher change\n'
        self.run_job('a')
        self.assertGreaterEqual(self.clock.now, 90)
        job = self.job('a')
        self.assertEqual((job['status'], job['error']), ('complete', None), job)
        sandbox = self.cloud.sandboxes['sb-0']
        self.assertEqual(job['started'], sandbox.started_at)  # charging starts at the provider's start, not creation
        self.assertAlmostEqual(job['charged'], sandbox.finished_at - sandbox.started_at, places=6)
        self.assertLess(job['charged'], 30)
        self.assertTrue(sandbox.terminated)
        self.assertEqual(sandbox.files['/job/input/policy.py'], self.files['training/policy.py'])
        self.assertEqual(self.broker.handle({'op': 'submit', 'name': 'a'})['submitted']['native']['bundle_sha256'],
                         job['artifact_sha'])

    def test_the_export_reserve_grows_with_the_grant(self):
        provider = self.module.ModalProvider.__new__(self.module.ModalProvider)
        provider.export_reserve_seconds = 60
        self.assertEqual([provider.export_reserve(g) for g in (120, 300, 600, 1200, 3600)], [15, 15, 30, 60, 60])

    def allocation(self, name):
        return json.loads((self.out / 'research/jobs' / name / 'allocation.json').read_text())

    def chain_index(self, commands):
        return next(i for i, c in enumerate(commands) if c[1:3] == ('-c', self.module.CHAIN))

    def test_the_worker_publishes_its_run_and_the_host_collects_it_after_terminating(self):
        # On Nov 13 runs lost their networks because the host streamed them out of the running worker over a
        # 16 KB/s link inside the paid time. The worker now trains, exports and publishes to the session's private
        # Volume itself; the host terminates it and only then collects, so a sandbox gone after its publish
        # costs nothing.
        self.cloud.start_after = 0
        self.cloud.vanish_after_export = True
        self.run_job('a')
        job = self.job('a')
        self.assertEqual((job['status'], job['error']), ('complete', None), job)
        sandbox = self.cloud.sandboxes['sb-0']
        options = sandbox.options
        self.assertEqual(list(options['volumes']), [handoff.MOUNT])
        self.assertEqual((options['block_network'], options['secrets'], options['include_oidc_identity_token']),
                         (True, [], False))
        binding = json.loads((self.out / 'research' / handoff.BINDING).read_text())
        self.assertEqual((binding['game'], binding['root']), ('doudizhu', str((self.out / 'research').resolve())))
        self.assertEqual(self.cloud.volumes[binding['volume']][0],
                         ('game-autoresearch-project', dict(version=2, create_if_missing=True)))
        commands = sandbox.commands
        prepare = next(i for i, c in enumerate(commands) if c[1:3] == ('-c', handoff.PREPARE))
        chain = self.chain_index(commands)
        self.assertLess(prepare, chain)  # /handoff is root-only before any researcher code runs
        self.assertFalse(any(c[0] in ('copy_to_local', 'tail') and c[-1] != '/job/output/progress.jsonl'
                             for c in commands))  # no live read of the bundle or the final logs
        spec = sandbox.spec
        self.assertEqual(spec['publish'][:2], [self.module.PYTHON, '-c'])
        self.assertEqual(spec['publish'][3], self.module.handoff_job('a'))
        self.assertLessEqual(float(spec['train'][spec['train'].index('--seconds') + 1]), 120 - 15)
        self.assertLessEqual(spec['budget'], 120 - 3)
        self.assertEqual(spec['publish_seconds'], self.module.ModalProvider.PUBLISH_SECONDS)
        terminated = self.cloud.events.index(('terminate', 'sb-0'))
        bundle = self.cloud.events.index(('read', spec['publish'][3] + '/files/bundle.tar'))
        self.assertLess(terminated, bundle)  # collected outside the worker's paid time
        record = self.allocation('a')
        self.assertEqual(record['handoff'], 'volume')
        self.assertEqual(set(record['handoff_detail']['files']),
                         {'bundle.tar', 'trainer.log', 'progress.jsonl', 'supervisor.json', 'chain.json'})
        self.assertIsNone(record['handoff_detail']['reason'])
        folder = self.out / 'research/jobs/a'
        self.assertEqual(verify(folder / 'bundle', job['artifact_sha'], check_tensors=False)['bundle_sha256'],
                         job['artifact_sha'])
        log = self.broker.handle({'op': 'log', 'name': 'a'})
        self.assertEqual((log['trainer_log'], log['supervisor']['status']),
                         ('trainer line\ntrainer line\ntrainer line', 'completed'))
        self.assertEqual(self.job('a')['progress'], {'waves': 2})

    def test_a_lost_publish_reply_is_found_in_the_volume_before_the_worker_ends(self):
        self.cloud.start_after = 0
        self.cloud.reply_lost = True
        self.run_job('a')
        job = self.job('a')
        self.assertEqual((job['status'], job['error']), ('complete', None), job)
        manifest = self.module.handoff_job('a') + '/' + handoff.MANIFEST
        self.assertLess(self.cloud.events.index(('read', manifest)), self.cloud.events.index(('terminate', 'sb-0')))
        self.assertEqual(self.allocation('a')['handoff'], 'volume')
        self.assertNotIn(('copy_to_local', '/job/export.tar'), self.cloud.sandboxes['sb-0'].commands)

    def test_nothing_published_falls_back_to_the_live_read_while_the_worker_lives(self):
        self.cloud.start_after = 0
        self.cloud.publish_fails = True
        self.run_job('a')
        job = self.job('a')
        self.assertEqual((job['status'], job['error']), ('complete', None), job)
        commands = self.cloud.sandboxes['sb-0'].commands
        self.assertIn(('copy_to_local', '/job/export.tar'), commands)
        self.assertTrue(any(c[0] == 'tail' and c[-1] == '/job/trainer.log' for c in commands))
        record = self.allocation('a')
        self.assertEqual(record['handoff'], 'live')
        self.assertIn('nothing in the hand-off Volume', record['handoff_detail']['reason'])
        self.assertIn('sync: Volume commit failed', record['handoff_detail']['chain_error'])
        verify(self.out / 'research/jobs/a/bundle', job['artifact_sha'], check_tensors=False)

    def test_a_broken_host_link_does_not_cost_a_finished_run(self):
        # The worker carries on alone while the host cannot reach it; collection retries after it has ended.
        self.cloud.start_after = 0
        self.cloud.poll_errors = 4
        self.cloud.broken_reads = 3
        self.run_job('a')
        job = self.job('a')
        self.assertEqual((job['status'], job['error']), ('complete', None), job)
        record = self.allocation('a')
        self.assertEqual(record['handoff'], 'volume')
        self.assertEqual(sum('poll: ConnectionError' in note for note in record['handoff_detail']['host_errors']), 4)
        self.assertNotIn(('copy_to_local', '/job/export.tar'), self.cloud.sandboxes['sb-0'].commands)

    def test_a_stopped_run_keeps_its_last_network_through_the_volume(self):
        self.cloud.start_after = 0
        self.cloud.train_seconds = 30
        self.run_job('a', wait=False)
        deadline = time.time() + 10
        while time.time() < deadline and not (self.cloud.sandboxes.get('sb-0') and self.cloud.sandboxes['sb-0'].spec):
            time.sleep(0.01)
        began = time.time()
        reply = self.broker.handle({'op': 'stop', 'name': 'a'})
        self.assertNotIn('error', reply, reply)
        self.backend.controller.join(timeout=20)
        self.assertFalse(self.backend.controller.busy())
        self.assertLess(time.time() - began, 10)  # the stop ended the trainer; the run did not wait out its time
        job = self.job('a')
        self.assertEqual((job['status'], job.get('stopped_early'), job['error']), ('complete', True, None), job)
        self.assertIn(self.module.KILL_RESEARCHER, self.cloud.sandboxes['sb-0'].commands)
        self.assertEqual(Ledger(self.out / 'research').jobs()[0]['status'], 'stopped')
        self.assertEqual(self.allocation('a')['handoff'], 'volume')
        self.assertNotIn('error', self.broker.handle({'op': 'submit', 'name': 'a'}))

    def test_a_failed_export_is_reported_from_the_volume(self):
        self.cloud.start_after = 0
        self.cloud.export_fails = True
        self.run_job('a')
        job = self.job('a')
        self.assertEqual(job['status'], 'failed')
        self.assertIn('Trainer did not leave a valid policy package', job['error'])
        self.assertIsNone(job['artifact_sha'])
        log = self.broker.handle({'op': 'log', 'name': 'a'})
        self.assertIn('Policy tensor cap exceeded', log['export_error'])
        self.assertEqual(self.allocation('a')['handoff'], 'volume')
        self.assertGreater(job['charged'], 0)  # the provider lease is charged as before

    def test_practice_mounts_no_volume_and_returns_its_score_over_exec(self):
        self.cloud.start_after = 0
        self.run_job('a')
        reply = self.broker.handle({'op': 'eval', 'name': 'a'})
        self.assertNotIn('error', reply, reply)
        self.backend.tick()
        self.backend.controller.join(timeout=60)
        practice = next(j for j in self.broker.handle({'op': 'status'})['jobs'] if j['kind'] == 'practice')
        self.assertEqual((practice['status'], practice['practice']['raw_score']), ('complete', 12.0), practice)
        self.assertEqual(self.cloud.sandboxes['sb-1'].options['volumes'], {})

    def test_a_stalled_progress_read_cannot_delay_the_export(self):
        # Seven 3-hour finals lost their bundles: the host re-read the whole progress log (14-21 MB) every
        # 5 s on the deadline loop, a read stalled past the stop point and the sandbox timed out before the
        # export. The mirror now runs beside the loop and reads a bounded tail; a read that never returns
        # costs only the progress copy.
        self.cloud.start_after = 0
        self.cloud.tail_gate = threading.Event()
        self.addCleanup(self.cloud.tail_gate.set)
        self.cloud.train_seconds = 1.5
        with mock.patch.object(self.module.ModalProvider, 'PROGRESS_POLL_SECONDS', 0.05), \
                mock.patch.object(self.module.ModalProvider, 'TAIL_TIMEOUT', 6):
            self.run_job('a')
        job = self.job('a')
        self.assertEqual((job['status'], job['error']), ('complete', None), job)
        self.assertTrue(self.cloud.export_while_stalled)  # the export did not wait for the stalled read
        commands = self.cloud.sandboxes['sb-0'].commands
        self.chain_index(commands)
        tails = [c for c in commands if c[0] == 'tail']
        self.assertTrue(tails)
        self.assertTrue(all(c[1:3] == ('-c', str(self.module.ModalProvider.TAIL_BYTES)) for c in tails))
        self.assertNotIn(('read', '/job/output/progress.jsonl'), commands)  # never a whole-file read

    def test_the_tail_keeps_whole_lines_within_its_byte_bound(self):
        sandbox = self.cloud.add('sb-tail')
        sandbox.files['/job/output/progress.jsonl'] = ''.join(f'{{"waves": {i}}}\n' for i in range(100000))
        with mock.patch.object(self.module.ModalProvider, 'TAIL_BYTES', 1000):
            text = self.module.ModalProvider.tail(sandbox, '/job/output/progress.jsonl')
        self.assertLessEqual(len(text), 1000)
        self.assertTrue(text.startswith('{"waves": ') and text.endswith('{"waves": 99999}\n'))
        self.assertIsNone(self.module.ModalProvider.tail(sandbox, '/job/missing.log'))

    def test_a_never_started_worker_fails_uncharged_and_the_session_continues(self):
        self.run_job('a')
        self.assertGreaterEqual(self.clock.now, 600)
        job = self.job('a')
        self.assertEqual((job['status'], job['charged']), ('failed', 0))
        self.assertIn('never started', job['error'])
        self.assertTrue(self.cloud.sandboxes['sb-0'].terminated)
        self.assertEqual(self.broker.handle({'op': 'status'})['active'], [])
        self.assertEqual(self.broker.handle({'op': 'budget'})['used_seconds'], 0)
        log = self.broker.handle({'op': 'log', 'name': 'a'})
        self.assertNotIn('error', log)
        self.assertIn('never started', log['job_error'])
        self.assertIn('abandoned_unstarted', self.events())
        self.assertNotIn('provider_uncertain', self.events())
        self.cloud.start_after = 0  # capacity is back: the next run is accepted at once and trains
        self.run_job('b')
        self.assertEqual(self.job('b')['status'], 'complete')
        self.assertNotIn('error', self.broker.handle({'op': 'submit', 'name': 'b'}))
        self.assertTrue(self.broker.handle({'op': 'finalize'})['finalized'])

    def test_stop_while_queued_for_a_gpu_is_stopped_uncharged(self):
        with mock.patch.object(workers, 'START_PATIENCE_SECONDS', 10.**9):
            self.run_job('a', wait=False)
            self.assertTrue(self.cloud.created.wait(10))
            self.assertEqual(self.job('a')['status'], 'queued')
            reply = self.broker.handle({'op': 'stop', 'name': 'a'})
            self.assertNotIn('error', reply, reply)
            self.backend.controller.join(timeout=60)
            self.assertFalse(self.backend.controller.busy())
        job = self.job('a')
        self.assertEqual((job['status'], job['charged']), ('stopped', 0))
        row = Ledger(self.out / 'research').jobs()[0]
        self.assertEqual((row['status'], row['provider_id']), ('stopped', 'sb-0'))  # abandoned by its worker, not cancelled
        self.assertIn('Stopped while waiting in the provider queue', row['error'])
        self.assertTrue(self.cloud.sandboxes['sb-0'].terminated)
        self.assertEqual(self.broker.handle({'op': 'budget'})['used_seconds'], 0)
        self.assertEqual(self.broker.handle({'op': 'status'})['active'], [])

    def test_parallel_runs_have_their_own_workers_and_hand_off_jobs_in_one_volume(self):
        self.backend.close()
        self.out = Path(self.temp.name) / 'parallel'
        self.out.mkdir()
        self.open(usd=0.48)
        self.cloud.start_after = 0
        self.cloud.train_seconds = 0.5
        self.run_job('a', wait=False)
        self.run_job('b', wait=False)
        self.backend.controller.join(timeout=60)
        self.assertFalse(self.backend.controller.busy())
        for name in ('a', 'b'):
            job = self.job(name)
            self.assertEqual((job['status'], job['error']), ('complete', None), job)
            self.assertEqual(self.allocation(name)['handoff'], 'volume')
            self.assertTrue((self.out / 'research/jobs' / name / 'bundle/manifest.json').is_file())
        first, second = self.cloud.sandboxes['sb-0'], self.cloud.sandboxes['sb-1']
        self.assertLess(max(first.started_at, second.started_at), min(first.finished_at, second.finished_at))
        records = [json.loads((self.out / 'research/jobs' / name / 'handoff.json').read_text()) for name in 'ab']
        self.assertEqual(records[0]['volume'], records[1]['volume'])  # one session Volume, bound once
        self.assertNotEqual(records[0]['job'], records[1]['job'])
        self.assertEqual(len(self.cloud.volumes), 1)
        rate = budgets.rate(TRAIN)
        charged = sum(self.job(name)['charged'] for name in 'ab')
        self.assertAlmostEqual(self.broker.handle({'op': 'budget'})['used_usd'], charged * rate, places=9)
        self.assertNotIn('error', self.broker.handle({'op': 'submit', 'name': 'b'}))
        self.assertTrue(self.broker.handle({'op': 'finalize'})['finalized'])

    def test_reopening_settles_a_recorded_never_started_uncertain_job_at_zero(self):
        sandbox = self.recorded_uncertain('lost')
        self.open(resume=True)  # host recovery reads the provider's record of the recorded sandbox
        job = self.job('lost')
        self.assertEqual((job['status'], job['charged']), ('failed', 0))
        self.assertIn('never started', job['error'])
        self.assertTrue(sandbox.terminated)
        self.assertEqual(self.broker.handle({'op': 'budget'})['used_seconds'], 0)
        self.cloud.start_after = 0
        self.run_job('next')
        self.assertEqual(self.job('next')['status'], 'complete')

    def test_stop_settles_a_recorded_never_started_uncertain_job_and_unblocks_finalize(self):
        sandbox = self.recorded_uncertain('lost')
        reply = self.broker.handle({'op': 'stop', 'name': 'lost'})
        self.assertNotIn('error', reply, reply)
        self.assertEqual((reply['status'], reply['charged']), ('stopped', 0))
        self.assertTrue(sandbox.terminated)
        self.assertEqual(self.broker.handle({'op': 'status'})['active'], [])
        self.assertEqual(self.broker.handle({'op': 'budget'})['used_seconds'], 0)
        self.assertIn('abandoned_unstarted', self.events())
        self.assertEqual(self.broker.finalize('host')['submission'], None)

    def test_stop_terminates_an_abandoned_started_worker_now_and_charges_its_lease(self):
        sandbox = self.recorded_uncertain('lost')
        sandbox.started_at = time.time() - 40  # it started late, after the controller had given up on it
        reply = self.broker.handle({'op': 'stop', 'name': 'lost'})
        self.assertNotIn('error', reply, reply)
        self.assertTrue(sandbox.terminated)  # not left to idle until its grant deadline
        job = self.job('lost')
        self.assertEqual(job['status'], 'failed')
        self.assertAlmostEqual(job['charged'], sandbox.finished_at - sandbox.started_at, places=6)
        self.assertGreaterEqual(job['charged'], 40)
        self.assertEqual(self.broker.handle({'op': 'status'})['active'], [])


class DoudizhuChain(unittest.TestCase):
    """The real in-worker chain, run locally: stand-in train/export commands and the real publish script."""

    def setUp(self):
        from harness.games.doudizhu import provider as module
        self.module = module
        self.temp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__('shutil').rmtree, self.temp, True)
        self.mount = self.temp / 'mount'

    def spec(self, train, export, budget=60, publish_seconds=5, tail_bytes=1000):
        code = handoff.PUBLISH.replace(repr(handoff.MOUNT), repr(str(self.mount)), 1)
        code = code.replace("subprocess.run(['sync',mount],check=True,timeout=120)", 'pass')
        logs = {'trainer.log': self.temp / 'trainer.log', 'progress.jsonl': self.temp / 'progress.jsonl',
                'supervisor.json': self.temp / 'supervisor.json'}
        return {'train': [sys.executable, '-c', train], 'train_timeout': 30,
                'kill': [sys.executable, '-c', 'pass'],  # never pkill on the host running the tests
                'export': [sys.executable, '-c', export], 'archive': str(self.temp / 'export.tar'),
                'logs': {name: str(path) for name, path in logs.items()}, 'stage': str(self.temp / 'stage'),
                'budget': budget, 'publish_seconds': publish_seconds, 'tail_bytes': tail_bytes, 'error_bytes': 50,
                'publish': [sys.executable, '-c', code, 'train-a-0123']}

    def chain(self, spec):
        return subprocess.run([sys.executable, '-c', self.module.CHAIN, json.dumps(spec)], capture_output=True,
                              text=True, timeout=60)

    def train(self):
        return ('import pathlib;root=pathlib.Path(%r)\n'
                "(root/'trainer.log').write_text(''.join(f'line {i}\\n' for i in range(500)))\n"
                "(root/'progress.jsonl').write_text('{\"waves\": 1}\\n')\n"
                "(root/'supervisor.json').write_text('{\"status\": \"completed\"}')") % str(self.temp)

    def published(self, name):
        return (self.mount / 'train-a-0123/files' / name).read_bytes()

    def test_it_trains_exports_and_publishes_bounded_whole_line_tails(self):
        export = 'import pathlib;pathlib.Path(%r).write_bytes(b"bundle" * 100);print("{}")' % str(self.temp / 'export.tar')
        done = self.chain(self.spec(self.train(), export))
        self.assertEqual(done.returncode, 0, done.stderr)
        manifest = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertEqual(set(manifest), {'bundle.tar', 'trainer.log', 'progress.jsonl', 'supervisor.json', 'chain.json'})
        self.assertEqual(self.published('bundle.tar'), b'bundle' * 100)
        log = self.published('trainer.log')
        self.assertLessEqual(len(log), 1000)
        self.assertTrue(log.startswith(b'line ') and log.endswith(b'line 499\n'))
        record = json.loads(self.published('chain.json'))
        self.assertEqual((record['train'], record['kill'], record['export']), (0, 0, 0))
        self.assertEqual(oct((self.mount / 'train-a-0123').stat().st_mode & 0o777), oct(0o700))
        self.assertEqual(oct((self.temp / 'stage').stat().st_mode & 0o777), oct(0o700))
        got = handoff.SessionVolume('doudizhu', self.temp / 'session/research', modal_module=types.SimpleNamespace(
            Volume=types.SimpleNamespace(from_name=lambda *a, **k: FolderVolume(self.mount)))).collect(
            'train-a-0123', self.temp / 'collected', wait=0)
        self.assertEqual({k: (v['bytes'], v['sha256']) for k, v in got.items()},
                         {k: (v['bytes'], v['sha256']) for k, v in manifest.items()})

    def test_a_failed_export_publishes_its_error_and_no_bundle_and_links_are_not_followed(self):
        secret = self.temp / 'secret'
        secret.write_text('root only\n')
        train = self.train() + "\n(root/'progress.jsonl').unlink();(root/'progress.jsonl').symlink_to(root/'secret')"
        export = 'import pathlib,sys;pathlib.Path(%r).write_bytes(b"partial");sys.stderr.write("x" * 80 + "tensor cap");sys.exit(3)' % (
            str(self.temp / 'export.tar'))
        done = self.chain(self.spec(train, export))
        self.assertEqual(done.returncode, 0, done.stderr)
        manifest = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertEqual(set(manifest), {'export-error.log', 'trainer.log', 'supervisor.json', 'chain.json'})
        error = self.published('export-error.log')
        self.assertEqual((len(error), error.endswith(b'tensor cap')), (50, True))
        self.assertEqual(json.loads(self.published('chain.json'))['export'], 3)

    def test_a_hung_export_still_leaves_time_to_publish(self):
        export = 'import time;time.sleep(60)'
        began = time.time()
        done = self.chain(self.spec(self.train(), export, budget=4, publish_seconds=2))
        self.assertLess(time.time() - began, 10)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn('bundle.tar', json.loads(done.stdout.strip().splitlines()[-1]))
        self.assertEqual(json.loads(self.published('chain.json'))['export'], 'timeout')


class DoudizhuSession(unittest.TestCase):
    """A whole recipe session through harness.session with the real `ar`, broker, controller and ledger."""

    def test_recipe_session_trains_the_submitted_recipe_in_a_fresh_ledger_and_grades_it(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        release, meta = release_fixture('doudizhu', temp.name, bind=False)
        private(release)
        policy.bind_release('doudizhu', release)
        providers = []

        class Offline(registry.backend('doudizhu')):
            def __init__(self, *args, **kwargs):
                providers.append(Provider('complete'))
                kwargs['options'] = dict(kwargs.get('options') or {}, _provider=providers[-1])
                super().__init__(*args, **kwargs)
        patcher = mock.patch.object(registry, 'backend', lambda game: Offline)
        patcher.start()
        self.addCleanup(patcher.stop)
        out, box = Path(temp.name) / 'session', Path(temp.name) / 'box'
        seen = {}

        def script(ar):
            seen['budget'] = ar('budget')
            ar('check')
            ar.edit('training/learner.py', (ar.workspace / 'training/learner.py').read_text() + '\n# explored\n')
            ar('run', 'explore', '--seconds', '60', '--config', 'configs/smoke.json')
            ar('wait')
            ar('eval', 'explore')
            ar('wait')
            seen['status'] = ar('status')
            seen['log'] = ar('log', 'explore')
            ar.edit('training/learner.py', (ar.workspace / 'training/learner.py').read_text() + '\n# final\n')
            ar.edit('configs/smoke.json', json.dumps({'architecture': {'hidden': 96, 'depth': 3, 'history': 64}}))
            seen['submit'] = ar('submit', '--note', 'final')
            ar('finalize')
        result = session.run('doudizhu', 'recipe', release, out, model='test/model', smoke=True,
                             launcher=lambda folder: Process(lambda: script(Researcher(folder / 'task', box))),
                             find_box=lambda: LocalBox(box / 'ar') if (box / 'ar').is_dir() else None)
        self.assertEqual((result['status'], result['error'], result['closed_by']), ('researched', None, 'researcher'))
        self.assertEqual((seen['budget']['budget_seconds'], seen['budget']['final_train_seconds']), (600, 180))
        self.assertEqual([(j['kind'], j['status']) for j in seen['status']['jobs']],
                         [('train', 'complete'), ('practice', 'complete')])
        self.assertIn('practice_runs', seen['log'])
        research, final = providers
        self.assertTrue(research.payloads[0]['code']['learner.py'].endswith('# explored\n'))
        self.assertEqual(Ledger(out / 'research').public()['submissions'], [])  # research networks are never promoted
        trained = result['final_training']
        self.assertEqual((trained['status'], trained['run']['reserved_seconds']), ('complete', 160))
        self.assertEqual(len(final.payloads), 1)
        self.assertTrue(final.payloads[0]['code']['learner.py'].endswith('# explored\n\n# final\n'))
        self.assertEqual(final.payloads[0]['config'], {'architecture': {'hidden': 96, 'depth': 3, 'history': 64}})
        ledger = Ledger(out / 'final').public()
        self.assertEqual((ledger['config']['gpu_minutes'], [j['name'] for j in ledger['jobs']]), (3, ['final']))
        self.assertEqual(len(ledger['submissions']), 1)
        with mock.patch.object(policy, 'require_modal', lambda: None):
            grade = session.grade(out)
        self.assertEqual((grade['development'], grade['private_deals'], grade['final']['raw_score']), (True, 16, 55.0))
        self.assertEqual(json.loads((out / 'grade.json').read_text())['graded_root'], 'final')
        self.assertEqual([len(p.evaluations) for p in providers], [0, 0, 1])  # only the host's reopened final root


    def test_a_dollar_mode_network_session_runs_two_jobs_at_once(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        release, meta = release_fixture('doudizhu', temp.name, bind=False)
        private(release)
        policy.bind_release('doudizhu', release)
        provider = Provider('complete')

        class Offline(registry.backend('doudizhu')):
            def __init__(self, *args, **kwargs):
                kwargs['options'] = dict(kwargs.get('options') or {}, _provider=provider)
                super().__init__(*args, **kwargs)
        patcher = mock.patch.object(registry, 'backend', lambda game: Offline)
        patcher.start()
        self.addCleanup(patcher.stop)
        out, box = Path(temp.name) / 'session', Path(temp.name) / 'box'
        seen = {}

        def script(ar):
            seen['budget'] = ar('budget')
            ar('run', 'a', '--seconds', '120', '--config', 'configs/smoke.json')
            seen['second'] = ar('run', 'b', '--seconds', '120', '--config', 'configs/smoke.json')
            ar('wait')
            seen['status'] = ar('status')
            ar('submit', 'b')
            ar('finalize')
        result = session.run('doudizhu', 'network', release, out, model='test/model', smoke=True, budget_mode='usd',
                             launcher=lambda folder: Process(lambda: script(Researcher(folder / 'task', box))),
                             find_box=lambda: LocalBox(box / 'ar') if (box / 'ar').is_dir() else None)
        self.assertEqual((result['status'], result['error'], result['budget_mode']), ('researched', None, 'usd'))
        self.assertEqual((result['research_usd'], result['wall_seconds']), (0.48, 4800))
        self.assertIn('up to 4 at once', (out / 'task/instruction.md').read_text())
        self.assertEqual((seen['budget']['budget_mode'], seen['budget']['budget_usd'], seen['budget']['concurrency']),
                         ('usd', 0.48, 4))
        self.assertNotIn('error', seen['second'])  # accepted while the first is still queued
        self.assertEqual([(j['name'], j['status']) for j in seen['status']['jobs']], [('a', 'complete'), ('b', 'complete')])
        ledger = Ledger(out / 'research')
        self.assertEqual((ledger.mode(), ledger.config['budget_usd']), ('usd', 0.48))
        self.assertAlmostEqual(result['budget']['used_usd'], 230 * budgets.rate(TRAIN), places=9)


class DoudizhuConstructor(unittest.TestCase):
    @unittest.skipUnless((ROOT / 'vendor/engines/DouZero/.git').exists(), 'offline DouZero metadata is not restored')
    def test_release_has_the_workspace_and_nothing_for_older_harnesses(self):
        from simulators.doudizhu.autoresearch.prepare import audit_release
        with tempfile.TemporaryDirectory() as temp:
            release = Path(temp) / 'release'
            manifest = registry.constructor('doudizhu')(release, include_existing=False)
            self.assertEqual((manifest['gpu_jobs_started'], manifest['model_calls']), (0, 0))
            self.assertEqual({p.name for p in release.iterdir()}, {'frozen', 'public', 'private', 'release.json'})
            names = {p.relative_to(release / 'public').as_posix() for p in (release / 'public').rglob('*') if p.is_file()}
            self.assertEqual({name for name in names if not name.startswith(workspace.WORKSPACE + '/')},
                             {'CONTRACT.md', 'benchmark.json', 'engine-manifest.json', 'practice-opponents.json',
                              'session.json', 'workspace_manifest.json', 'agent/policy.py', 'agent/learner.py',
                              'agent/train.py', 'recipes/starter.json', 'recipes/smoke.json'})
            base = release / 'public' / workspace.WORKSPACE
            self.assertEqual({p.name for p in base.iterdir()}, set(workspace.TOP))
            self.assertEqual((base / 'training/train.py').read_bytes(),
                             (ROOT / 'simulators/doudizhu/autoresearch/starter/train.py').read_bytes())
            frozen = {p.relative_to(release / 'frozen/integrations/doudizhu/autoresearch').as_posix()
                      for p in (release / 'frozen/integrations/doudizhu/autoresearch').rglob('*') if p.is_file()}
            self.assertEqual(frozen, {'__init__.py', 'contract.py', 'bundle.py', 'wire.py', 'policy_worker.py',
                                      'arena.py', 'grading.py', 'train_entry.py', 'evaluate_entry.py',
                                      'export_entry.py', 'build_cuda.py', 'starter/policy.py'})
            self.assertEqual(audit_release(release)['hidden_files_staged'], 0)
            record = policy.bind_release('doudizhu', release)
            self.assertEqual(policy.verify_release('doudizhu', release)['sha256'], record['sha256'])
            panel = json.loads((release / 'private/panel.json').read_text())
            self.assertEqual((panel['practice'], panel['official']), (SIMPLE, None))
            (release / 'public/agent/extra.py').write_text('unmanifested')
            with self.assertRaisesRegex(ValueError, 'Unmanifested'):
                audit_release(release)
            (release / 'public/agent/extra.py').unlink()
            # The constructed release opens as a session and takes a run without any cloud call.
            provider = Provider()
            backend = registry.backend('doudizhu')(release, Path(temp) / 'session/research', smoke=True,
                                                   budget_seconds=600, options={'_provider': provider})
            with backend:
                files = workspace.editable(release, backend.workspace)
                self.assertEqual(set(files), set(workspace.manifest(release)['writable']))
                backend.run('a', 60, 'configs/train.json', 3)
                backend.tick()
                backend.controller.join(timeout=10)
            self.assertEqual(provider.payloads[0]['config'], json.loads(files['configs/train.json']))
            self.assertEqual(provider.payloads[0]['seed'], 3)


if __name__ == '__main__':
    unittest.main()
