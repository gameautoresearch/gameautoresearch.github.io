"""Clash backend against its real constructor, controller, ledger and bundle code; only the provider is stubbed.

Run with the game's research interpreter: .venv-clash-ar/bin/python -m unittest harness.tests.test_clash
"""
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from harness import budget as budgets, policy, registry, session, workers, workspace
from harness.transport import LocalBox
from harness.broker import Broker
from harness.tests.support import Process, Researcher
from simulators.clash.autoresearch.common import STOP_REQUEST

try:
    import numpy as np
    from simulators.clash.autoresearch.common import HERE, MATCH_FORM, digest, dump, sha
    from simulators.clash.autoresearch.short_release import BENCHMARK_ID, PRACTICE_SEEDS, prepare, used_seeds, verify
except ImportError as exc:  # wrong interpreter
    raise unittest.SkipTest(f'Clash research dependencies unavailable: {exc}')

SOURCE = ('policy.py', 'learner.py', 'train.py', 'config.json')
RATE = budgets.rate(dict(gpu=1, cpu_cores=4, memory_mib=16384))  # dollars per second of the training worker
RUN_SHAPE = 'run (L40S, 4 CPU, 16 GiB)'
SENTINEL = '\n# researcher change\n'


def tree(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}


def starter_checkpoint(directory):
    """Call the starter's own Trainer.checkpoint on a stand-in trainer (no torch here); return payload and bytes."""
    import pickle
    from simulators.clash.autoresearch.common import load_module
    module = load_module(HERE / 'starter/train.py', 'starter_train_under_test')
    saved = {}

    class Torch:
        cuda = types.SimpleNamespace(get_rng_state_all=lambda: ['cuda-rng'])

        @staticmethod
        def get_rng_state():
            return 'cpu-rng'

        @staticmethod
        def save(payload, stream):
            saved.update(payload)
            pickle.dump(payload, stream)
    counters = types.SimpleNamespace(cpu=lambda: [0] * 11)
    trainer = object.__new__(module.Trainer)
    trainer.__dict__.update(torch=Torch, c={'envs': 1024}, binding={'source': {'native/train.py': '0' * 64}},
                            model=types.SimpleNamespace(state_dict=lambda: {'w': [1.0, 2.0]}),
                            optimizer=types.SimpleNamespace(state_dict=lambda: {'state': {}, 'param_groups': []}),
                            update=3, samples=96, prior_seconds=0.0, counters=counters, bucket_usage=counters, gpu=True)
    path = Path(directory) / 'checkpoint.pt'
    diagnostic = trainer.checkpoint(path, 12.5, entrant_binding={'source': {'train.py': '1' * 64}},
                                    entrant_state={'optimizer_steps': 3, 'losses': []})
    return saved, diagnostic, path


class Provider:
    """Follows FastProvider's ledger protocol (reserve, record the worker, settle from a receipt) with no worker."""
    reservation_overhead_seconds = 30
    minimum_run_seconds = 90
    max_practice_evaluations = 6

    def __init__(self):
        self.trains, self.practices = [], []
        self.outcome = 'learn'
        self.hold = None  # an Event keeps the stand-in worker "running" until it is stopped or released
        self.started = threading.Event()
        self.checkpoint = b'stand-in checkpoint'

    def train(self, ledger, name, source, out, seconds, stop=None):
        source, out = Path(source), Path(out)
        self.trains.append(dict(name=name, seconds=seconds, files={n: (source / n).read_text() for n in SOURCE}))
        ledger.reserve(name, seconds + self.reservation_overhead_seconds)
        out.mkdir(parents=True)
        ledger.record(name, status='running', app_id='ap-stub', sandbox_id='sb-' + name)
        self.started.set()
        receipt = dict(provider_id='sb-' + name, started_at=1000.0, finished_at=1100.0)
        while self.hold is not None and not self.hold.is_set():
            if stop():
                ledger.settle(name, dict(receipt, finished_at=1040.0), status='failed', error='stopped by request')
                raise RuntimeError('stopped by request')
            self.hold.wait(0.01)
        if self.outcome == 'unknown-end':
            ledger.record(name, status='uncertain', error='provider end unavailable')
            raise RuntimeError('unresolved allocation; hold remains')
        if self.outcome == 'crash':
            ledger.settle(name, dict(receipt, finished_at=1040.0), status='failed', error='worker failed: stub')
            raise RuntimeError('worker failed: stub')
        ledger.settle(name, receipt, status='complete')
        artifact = out / 'artifact'
        artifact.mkdir()
        for n in SOURCE:
            shutil.copyfile(source / n, artifact / n)
        np.savez(artifact / 'weights.npz', layer=np.ones(8, np.float32))
        (artifact / 'checkpoint.pt').write_bytes(self.checkpoint)
        result = dict(updates=3, samples=96, natural_games=2, optimizer_steps=3, finite_losses=True,
                      weight_l1_change=0.5 if self.outcome == 'learn' else 0.0, seconds=80.0,
                      loss_summary={'last': [0.1]})
        dump(artifact / 'result.json', result)
        return result

    def recover(self, ledger, name):
        ledger.settle(name, dict(provider_id='sb-' + name, started_at=1000.0, finished_at=1090.0), status='failed',
                      error='settled later from the provider receipt')
        return dict(resolved=True, adopted=False)

    def cpu_grade(self, bundle, schedule, opponents, out, seconds=600, split='public-development', stop=None):
        value = json.loads(Path(schedule).read_text())
        self.practices.append(dict(schedule=value, opponents=dict(opponents), seconds=seconds, split=split))
        for actor in value['opponents']:
            if actor['kind'] == 'checkpoint':
                actor['bundle'] = '/private/' + actor['name']
        identity = json.loads((Path(bundle) / 'manifest.json').read_text())['identity']
        result = dict(games=80, raw_score_0_100=37.5, interval_95_percent=[30.0, 45.0], by_opponent={'standard': 50.0},
                      by_seat={'0': 40.0, '1': 35.0}, natural_games=80, truncations=0, illegal_actions=0,
                      simulator_faults=0, split=split, seconds=200.0, submission_identity=identity,
                      schedule_commitment=digest(value), rows=[{'opponent': 'standard', 'points': 1.0}])
        dump(out, result)
        return result


class Release(unittest.TestCase):
    """One real release for the whole module: the real standalone constructor -> harness binding."""

    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        root = Path(cls.shared.name)
        cls.release = root / 'release'
        cls.prepared = prepare(cls.release)
        policy.bind_release('clash', cls.release)
        cls.meta = workspace.manifest(cls.release)
        staged = workspace.public(cls.release) / workspace.WORKSPACE
        cls.files = {name: (staged / name).read_text() for name in cls.meta['writable']}

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()


class Constructor(Release):
    def test_release_holds_the_starter_the_short_panel_and_only_the_uniform_workspace(self):
        public = self.release / 'public'
        self.assertEqual({p.name for p in public.iterdir()},
                         {'policy.py', 'learner.py', 'train.py', 'config.json', 'smoke-config.json', 'benchmark.json',
                          'short-contract.md',
                          'interface.json', 'practice.json', 'opponent-registry.json', 'observation-schema.json',
                          'agent_workspace', 'workspace_manifest.json'})
        for name in ('policy.py', 'learner.py', 'train.py'):
            self.assertEqual((public / name).read_bytes(), (HERE / 'starter' / name).read_bytes())
        self.assertEqual((public / 'config.json').read_bytes(), (HERE / 'recipes/starter.json').read_bytes())
        self.assertEqual((public / 'benchmark.json').read_bytes(), (HERE / 'benchmark-short-v1.json').read_bytes())
        self.assertEqual(self.meta['game'], 'clash')
        self.assertEqual(self.files['configs/train.json'], (HERE / 'recipes/starter.json').read_text())
        for name in ('configs/train.json', 'configs/smoke.json'):
            config = json.loads(self.files[name])
            self.assertEqual((config['match_form'], config['rules']['decision_ticks']), (MATCH_FORM, 10))
        record = verify(self.release)
        self.assertEqual((record['format'], record['benchmark_id'], record['grading']), (5, BENCHMARK_ID, 'parallel3'))
        self.assertEqual(record['workspace_sha256'], self.meta['sha256'])
        self.assertEqual(set(record['runtime_source_sha256']), {p.name for p in HERE.glob('*.py')})
        self.assertEqual(record['engine']['match_form']['name'], MATCH_FORM)
        policy.verify_release('clash', self.release)

    def test_short_interface_panel_and_new_schedules(self):
        interface = json.loads((self.release / 'public/interface.json').read_text())
        self.assertEqual((interface['decisions_per_second'], interface['decision_ticks'], interface['regulation_seconds'],
                          interface['overtime_seconds'], interface['observation_ints'], interface['actions']),
                         (2, 10, 60, 30, 7024, 2307))
        scripted = [{'kind': 'scripted', 'name': 'standard'}, {'kind': 'scripted', 'name': 'pressure'},
                    {'kind': 'intermediate', 'name': 'intermediate'}, {'kind': 'teacher', 'name': 'teacher'}]
        practice = json.loads((self.release / 'public/practice.json').read_text())
        protected = json.loads((self.release / 'private/grade.json').read_text())
        registry = json.loads((self.release / 'public/opponent-registry.json').read_text())
        self.assertEqual((practice['opponents'], protected['opponents'], registry['fixed_panel'], registry['learned']),
                         (scripted, scripted, scripted, []))
        self.assertEqual(set(registry['dropped']), {'retained-deepseek', 'incumbent-v4'})
        self.assertEqual((practice['match_form'], protected['match_form']), (MATCH_FORM, MATCH_FORM))
        self.assertEqual(practice['engine_seeds'], PRACTICE_SEEDS)
        self.assertEqual(len(set(protected['engine_seeds'])), 32)
        self.assertFalse(set(protected['engine_seeds']) & used_seeds(), 'protected seeds are fresh secrets')
        self.assertEqual(oct((self.release / 'private').stat().st_mode & 0o777), oct(0o700))
        other = Path(self.shared.name) / 'other'
        prepare(other)  # every release draws its own protected seeds
        self.assertNotEqual(json.loads((other / 'private/grade.json').read_text())['engine_seeds'],
                            protected['engine_seeds'])

    def test_changed_release_is_refused(self):
        root = Path(self.shared.name)
        copy = root / 'copy'
        shutil.copytree(self.release, copy)
        verify(copy)
        (copy / 'public/config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'public release drift'):
            verify(copy)
        rescheduled = root / 'rescheduled'
        shutil.copytree(self.release, rescheduled)
        schedule = json.loads((rescheduled / 'private/grade.json').read_text())
        schedule['engine_seeds'].append(7)
        dump(rescheduled / 'private/grade.json', schedule)
        with self.assertRaisesRegex(ValueError, 'schedule drift'):
            verify(rescheduled)
        reformed = root / 'reformed'
        shutil.copytree(self.release, reformed)
        schedule = json.loads((reformed / 'private/grade.json').read_text())
        schedule['match_form'] = 'full'
        dump(reformed / 'private/grade.json', schedule)
        record = json.loads((reformed / 'release.json').read_text())
        record['private_schedule_commitment'] = digest(schedule)
        dump(reformed / 'release.json', record)
        with self.assertRaisesRegex(ValueError, 'match form'):
            verify(reformed)
        with self.assertRaises(FileExistsError):  # a release is never rebuilt in place
            prepare(self.release)


class ClashBackend(Release):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.provider = Provider()
        self.out = Path(self.temp.name) / 'session'
        self.out.mkdir()
        self.backend = self.open(self.out / 'research', 600)
        self.broker = Broker(self.backend, self.out, 'network', research_seconds=600)

    def open(self, root, seconds, **kwargs):
        backend = registry.backend('clash')(self.release, root, smoke=True, budget_seconds=seconds,
                                            options={'_provider': self.provider}, **kwargs)
        backend.open()
        self.addCleanup(backend.close)
        return backend

    def work(self, backend=None):
        backend = backend or self.backend
        backend.tick()
        self.assertEqual(backend.controller.wait(10), [])

    def failure(self, call, *args):
        with self.assertRaises(ValueError) as caught:
            call(*args)
        return str(caught.exception)

    def run_a(self, **extra):
        reply = self.broker.handle({'op': 'run', 'name': 'a', 'seconds': 120, 'config': 'configs/smoke.json',
                                    'files': self.files, **extra})
        self.assertNotIn('error', reply, reply)
        return reply

    def ledger(self):
        return json.loads((self.out / 'research/allocation-ledger.json').read_text())

    def test_run_carries_exactly_the_researchers_files_and_is_charged(self):
        budget = self.broker.handle({'op': 'budget'})
        self.assertEqual((budget['remaining_seconds'], budget['allocatable_seconds']), (600, 570))
        self.assertEqual((budget['budget_mode'], budget['concurrency']), ('seconds', 1))
        self.assertNotIn('budget_usd', budget)
        files = dict(self.files)
        files['training/learner.py'] += SENTINEL
        files['training/train.py'] += SENTINEL
        chosen = json.loads(files['configs/smoke.json'])
        chosen['learning_rate'] = 0.00123
        files['configs/smoke.json'] = json.dumps(chosen)
        reply = self.broker.handle({'op': 'run', 'name': 'a', 'seconds': 120, 'config': 'configs/smoke.json',
                                    'files': files})
        self.assertNotIn('error', reply, reply)
        self.assertEqual((reply['reserved_seconds'], reply['worker_seconds']), (150, 120))
        self.assertEqual(self.broker.handle({'op': 'status'})['active'], ['a'])
        budget = self.broker.handle({'op': 'budget'})  # a queued run is already a liability, reserve included
        self.assertEqual((budget['used_seconds'], budget['remaining_seconds'], budget['allocatable_seconds']),
                         (150, 450, 420))
        self.assertIn('One job at a time', self.broker.handle({'op': 'run', 'name': 'b', 'seconds': 120,
                                                              'files': files})['error'])
        self.work()
        sent = self.provider.trains[0]
        self.assertEqual((sent['name'], sent['seconds']), ('a', 120))
        self.assertEqual(sent['files'], {'policy.py': files['training/policy.py'],
                                         'learner.py': files['training/learner.py'],
                                         'train.py': files['training/train.py'],
                                         'config.json': files['configs/smoke.json']})
        status = self.broker.handle({'op': 'status'})
        self.assertEqual([(j['name'], j['kind'], j['status']) for j in status['jobs']], [('a', 'train', 'complete')])
        self.assertEqual(status['active'], [])
        budget = self.broker.handle({'op': 'budget'})  # charged the provider's own start-to-end interval
        self.assertEqual((budget['used_seconds'], budget['remaining_seconds']), (100, 500))
        bundle = self.out / 'research/jobs/a/bundle'
        self.assertEqual((bundle / 'learner.py').read_text(), files['training/learner.py'])
        self.assertEqual(status['jobs'][0]['result']['identity'],
                         json.loads((bundle / 'manifest.json').read_text())['identity'])
        self.assertIn('Unknown run', self.broker.handle({'op': 'log', 'name': 'zzz'})['error'])
        self.assertEqual(self.broker.handle({'op': 'log', 'name': 'a'})['result']['updates'], 3)

    def test_practice_submit_finalize_and_protected_development_grade(self):
        self.run_a()
        self.assertIn('completed training run', self.broker.handle({'op': 'eval', 'name': 'a'})['error'])
        self.work()
        queued = self.broker.handle({'op': 'eval', 'name': 'a'})
        self.assertEqual((queued['charged_to_allocation'], queued['practice_evaluations_remaining']), (False, 5))
        self.assertEqual((queued['queued'], self.broker.handle({'op': 'status'})['active']), ('eval-a', ['eval-a']))
        self.work()
        asked = self.provider.practices[0]
        public = json.loads((self.release / 'public/practice.json').read_text())
        self.assertEqual(asked['schedule']['engine_seeds'], public['engine_seeds'])
        self.assertEqual(asked['opponents'], {})  # the short panel is scripted: no opponent bundle is sent
        self.assertEqual((asked['seconds'], asked['split']), (600, 'public-development'))
        status = self.broker.handle({'op': 'status'})
        self.assertEqual([(j['kind'], j['status']) for j in status['jobs']], [('train', 'complete'),
                                                                               ('practice', 'complete')])
        self.assertEqual(status['jobs'][0]['practice']['raw_score_0_100'], 37.5)
        self.assertNotIn('rows', self.broker.handle({'op': 'log', 'name': 'a'})['practice'])
        self.assertEqual(self.broker.handle({'op': 'log', 'name': 'eval-a'})['practice']['raw_score_0_100'], 37.5)
        again = self.broker.handle({'op': 'eval', 'name': 'a'})  # the same network is never practiced twice
        self.assertTrue(again['cached'])
        self.assertEqual(len(self.provider.practices), 1)
        budget = self.broker.handle({'op': 'budget'})
        self.assertEqual((budget['used_seconds'], budget['practice_evaluations_remaining']), (100, 5))
        submitted = self.broker.handle({'op': 'submit', 'name': 'a'})['submitted']
        self.assertTrue(submitted['native']['practiced'])
        identity = submitted['native']['identity']
        self.assertEqual(json.loads((self.out / 'research/submissions' / identity / 'manifest.json').read_text())
                         ['identity'], identity)
        self.assertEqual(self.broker.handle({'op': 'finalize'})['submission']['name'], 'a')
        self.assertEqual(self.ledger()['milestones']['final']['identity'], identity)
        self.assertIn('finalized', self.broker.handle({'op': 'run', 'name': 'b', 'seconds': 120,
                                                       'files': self.files})['error'])
        self.backend.close()
        graded = self.open(self.out / 'research', None, resume=True)  # reopening grants nothing new
        self.assertEqual(graded.budget()['budget_seconds'], 600)
        result = graded.grade()  # development grade: the final plays the protected schedule once on CPU
        protected = json.loads((self.release / 'private/grade.json').read_text())
        played = self.provider.practices[-1]
        self.assertEqual((played['split'], played['seconds'], played['schedule']['engine_seeds']),
                         ('protected-development', 3600, protected['engine_seeds']))
        self.assertEqual((result['status'], result['schedule'], result['official'], result['submission_identity'],
                          result['raw_score_0_100']), ('development_only', 'protected', False, identity, 37.5))
        with self.assertRaises(FileExistsError):  # graded once; never replayed for a better draw
            graded.grade()

    def test_bad_requests_never_reach_the_ledger(self):
        def changed(name, text):
            return dict(self.files, **{name: text})
        config = json.loads(self.files['configs/train.json'])
        bigger = json.dumps(dict(config, hardware_allocation=dict(config['hardware_allocation'], cpu_threads=32,
                                                                  ram_gib=128)))
        capped = json.dumps(dict(config, max_ticks=600))
        impure = 'import os\ndef create(config):\n    return None\ndef act(model, observation, mask):\n' \
                 '    return model(observation, mask)[0].argmax(-1)\n'
        run = {'op': 'run', 'name': 'a', 'seconds': 120, 'files': self.files}
        for request, reason in (
                (dict(run, files=changed('training/learner.py', 'def broken(:')), 'line 1'),
                (dict(run, files=dict(self.files, extra='x')), 'editable file set'),
                (dict(run, seconds=100000), '90 to 3300 worker seconds'),
                (dict(run, seconds=60), '90 to 3300 worker seconds'),
                (dict(run, seconds=3300), 'only 600.0 remain'),
                (dict(run, seconds=580), 'settlement reserve'),  # 580 fits the budget, 580 + 30 does not
                (dict(run, name='../a'), 'Run names'),
                (dict(run, name='eval-a'), 'Run names'),
                (dict(run, seed=3), 'seed'),
                (dict(run, files=changed('configs/train.json', bigger)), '4 CPU cores and 16 GiB'),
                (dict(run, files=changed('configs/train.json', capped)), 'natural match horizon'),
                (dict(run, files=changed('training/policy.py', impure)), 'policy import'),
                ({'op': 'eval', 'name': 'a'}, 'Unknown run'),
                ({'op': 'submit', 'name': 'a'}, 'Unknown run'),
                ({'op': 'stop', 'name': 'a'}, 'Unknown run')):
            reply = self.broker.handle(request)
            self.assertIn(reason, reply.get('error', ''), request.get('name'))
        with self.assertRaisesRegex(ValueError, 'eval- prefix'):  # practice job names stay unambiguous
            self.backend.run('eval-a', 120.0, 'configs/train.json')
        self.assertEqual(self.ledger()['allocations'], [])
        self.assertEqual(self.backend.status(), {'jobs': [], 'active': []})
        self.assertEqual((self.provider.trains, self.provider.practices), ([], []))
        self.assertFalse((self.out / 'research/sources').exists())
        self.assertEqual(self.backend.budget()['used_seconds'], 0)
        check = self.broker.handle({'op': 'check', 'files': self.files})
        self.assertTrue(check['checked'] and check['policy_review']['greedy_entry_checked'], check)
        self.assertIn('4 CPU cores', self.broker.handle({'op': 'check',
                                                         'files': changed('configs/train.json', bigger)})['error'])

    def test_failed_and_unlearned_runs_stay_charged_and_cannot_be_submitted(self):
        self.provider.outcome = 'crash'
        self.run_a()
        self.work()
        self.provider.outcome = 'flat'
        self.assertNotIn('error', self.broker.handle({'op': 'run', 'name': 'b', 'seconds': 120, 'files': self.files}))
        self.work()
        jobs = self.backend.status()['jobs']
        self.assertEqual([(j['name'], j['status']) for j in jobs], [('a', 'failed'), ('b', 'failed')])
        self.assertIn('worker failed', jobs[0]['error'])
        self.assertIn('no verified learning update', jobs[1]['error'])
        self.assertEqual(self.backend.budget()['used_seconds'], 40 + 100)
        self.assertIn('already used', self.broker.handle({'op': 'run', 'name': 'a', 'seconds': 120,
                                                          'files': self.files})['error'])
        self.assertIn('completed training run', self.broker.handle({'op': 'submit', 'name': 'b'})['error'])
        with self.assertRaisesRegex(ValueError, 'without submission'):
            self.backend.finalize()

    def test_settle_cancels_queued_work_without_charge(self):
        self.run_a()
        self.backend.settle()
        status = self.backend.status()
        self.assertEqual([(j['name'], j['status']) for j in status['jobs']], [('a', 'stopped')])
        self.assertEqual(status['active'], [])
        self.assertEqual((self.backend.budget()['used_seconds'], self.ledger()['allocations']), (0, []))
        self.assertEqual(self.broker.finalize('host')['submission'], None)

    def test_settle_stops_a_running_worker_and_charges_its_real_interval(self):
        self.provider.hold = threading.Event()
        self.run_a()
        self.backend.tick()
        self.assertTrue(self.provider.started.wait(10))
        self.assertEqual(self.backend.status()['jobs'][0]['status'], 'running')
        self.assertIn('active', self.broker.handle({'op': 'finalize'})['error'])
        self.backend.settle()
        self.backend.settle()  # idempotent
        status = self.backend.status()
        self.assertEqual([(j['name'], j['status'], j['charged_seconds']) for j in status['jobs']],
                         [('a', 'stopped', 40)])
        self.assertEqual((status['active'], self.backend.budget()['used_seconds']), ([], 40))

    def test_unknown_worker_end_keeps_its_full_hold_until_the_provider_reports(self):
        self.provider.outcome = 'unknown-end'
        self.run_a()
        self.work()
        status = self.backend.status()
        self.assertEqual((status['jobs'][0]['status'], status['active']), ('uncertain', ['a']))
        self.assertEqual(self.backend.budget()['used_seconds'], 150)
        self.assertIn('One job at a time', self.broker.handle({'op': 'run', 'name': 'b', 'seconds': 120,
                                                              'files': self.files})['error'])
        self.backend.settle()  # asks the provider again; the receipt settles it as failed
        status = self.backend.status()
        self.assertEqual((status['jobs'][0]['status'], status['active']), ('failed', []))
        self.assertEqual(self.backend.budget()['used_seconds'], 90)

    def test_seconds_mode_is_recorded_one_job_at_a_time_and_charged_in_training_worker_seconds(self):
        ledger = self.ledger()
        self.assertEqual((ledger['budget_mode'], ledger['limit_seconds']), ('seconds', 600))
        self.assertNotIn('limit_usd', ledger)
        self.assertEqual((self.backend.budget_mode, self.backend.limits['max_concurrent_jobs']), ('seconds', 1))
        self.run_a()
        self.work()
        self.provider.hold = threading.Event()
        self.assertNotIn('error', self.broker.handle({'op': 'run', 'name': 'b', 'seconds': 120, 'files': self.files}))
        self.backend.tick()
        self.assertTrue(self.provider.started.wait(10))
        for request in ({'op': 'eval', 'name': 'a'}, {'op': 'run', 'name': 'c', 'seconds': 90, 'files': self.files}):
            self.assertIn('One job at a time; still active: b', self.broker.handle(request)['error'])
        self.provider.hold.set()
        self.assertEqual(self.backend.controller.wait(10), [])
        self.assertEqual(self.backend.budget()['used_seconds'], 200)  # each run its provider interval, 1 s for 1 s
        self.assertNotIn('charged_usd', self.backend.status()['jobs'][0])
        self.backend.close()
        with self.assertRaisesRegex(ValueError, 'seconds mode'):  # a resumed session keeps its mode
            registry.backend('clash')(self.release, self.out / 'research', smoke=True, resume=True, budget_usd=1.0,
                                      options={'_provider': self.provider}).open()
        reopened = self.open(self.out / 'research', None, resume=True)
        self.assertEqual((reopened.budget_mode, reopened.budget()['budget_seconds']), ('seconds', 600))

    def test_starter_checkpoint_is_model_and_optimizer_only_and_still_sealed(self):
        payload, diagnostic, path = starter_checkpoint(self.temp.name)
        self.assertEqual(set(payload), {'format', 'config', 'binding', 'model', 'optimizer', 'update', 'samples',
                                        'allocated_seconds', 'counters', 'bucket_usage', 'python_rng', 'numpy_rng',
                                        'torch_rng', 'cuda_rng', 'entrant_binding', 'entrant_state'})
        self.assertFalse({'engine_state', 'engine_outputs'} & set(payload))  # no native world image
        self.assertEqual((diagnostic['format'], diagnostic['engine_state_saved'], diagnostic['file_bytes']),
                         ('clash-fast-model-optimizer-v1', False, path.stat().st_size))
        self.assertFalse(path.with_suffix('.tmp').exists())
        self.provider.checkpoint = path.read_bytes()  # what the worker returns for the starter
        self.run_a()
        self.work()
        self.assertEqual(self.backend.status()['jobs'][0]['status'], 'complete')
        manifest = json.loads((self.out / 'research/jobs/a/bundle/manifest.json').read_text())
        self.assertEqual((manifest['provenance']['checkpoint_sha256'], manifest['provenance']['checkpoint_storage']),
                         (hashlib.sha256(path.read_bytes()).hexdigest(), 'local'))

    def test_fresh_final_training_fits_its_own_allowance_exactly(self):
        """The recipe variant's second root: one run of everything allocatable, then submit and finalize."""
        final = self.open(self.out / 'final', 120)
        files = dict(self.files)
        files['training/learner.py'] += SENTINEL
        self.assertEqual(workspace.apply(final.release, final.workspace, files),
                         workspace.validate(self.meta, files))
        seconds = final.budget()['allocatable_seconds']  # what the host's final training asks for
        self.assertEqual(seconds, 90)
        self.assertIn('settlement reserve', self.failure(final.run, 'final', 120.0, 'configs/smoke.json', 1))
        self.assertEqual(final.run('final', seconds, 'configs/smoke.json', 1)['reserved_seconds'], 120)
        self.assertEqual(final.budget()['remaining_seconds'], 0)
        self.work(final)
        self.assertEqual(self.provider.trains[0]['files']['learner.py'], files['training/learner.py'])
        self.assertEqual(final.status()['jobs'][0]['status'], 'complete')
        self.assertFalse(final.submit('final')['practiced'])
        self.assertEqual(final.finalize()['lifecycle'], 'finalized')
        self.assertEqual(self.ledger()['allocations'], [], 'the research ledger is a different account')


class DollarMode(Release):
    """Dollar mode: parallel jobs within a dollar budget, through the same backend, controller and ledger."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.provider = Provider()
        self.out = Path(self.temp.name) / 'session'
        self.out.mkdir()
        self.backend = self.open(self.out / 'research', 1.0)
        self.broker = Broker(self.backend, self.out, 'network', research_usd=1.0)

    def open(self, root, usd, **kwargs):
        backend = registry.backend('clash')(self.release, root, smoke=True, budget_usd=usd,
                                            options={'_provider': self.provider}, **kwargs)
        backend.open()
        self.addCleanup(backend.close)
        return backend

    def launch(self, name, seconds=120):
        return self.broker.handle({'op': 'run', 'name': name, 'seconds': seconds, 'files': self.files})

    def ledger(self):
        return json.loads((self.out / 'research/allocation-ledger.json').read_text())

    def jobs(self):
        return {job['name']: job for job in self.backend.status()['jobs']}

    def running(self, count):
        """Wait until `count` allocations are running at the same moment."""
        deadline = time.monotonic() + 10
        while sum(row['status'] == 'running' for row in self.ledger()['allocations']) < count:
            self.assertLess(time.monotonic(), deadline, 'the workers never ran side by side')
            time.sleep(0.01)

    def test_training_runs_in_parallel_up_to_the_gpu_limit_and_is_charged_in_dollars(self):
        budget = self.broker.handle({'op': 'budget'})
        self.assertEqual((budget['budget_mode'], budget['budget_usd'], budget['used_usd'], budget['concurrency'],
                          budget['concurrent_gpus']), ('usd', 1.0, 0.0, 4, 2))
        self.assertEqual(budget['affordable_seconds'], {RUN_SHAPE: int(1.0 / RATE)})
        self.assertNotIn('budget_seconds', budget)
        self.provider.hold = threading.Event()
        first = self.launch('a')
        self.assertEqual(first['reserved_seconds'], 150)
        self.assertAlmostEqual(first['reserved_usd'], 150 * RATE)
        self.assertNotIn('error', self.launch('b'))
        self.assertIn('At most 2 GPUs', self.launch('c')['error'])  # one GPU per training run
        self.assertAlmostEqual(self.broker.handle({'op': 'budget'})['used_usd'], 300 * RATE)  # both grants held
        self.backend.tick()
        self.running(2)
        self.assertEqual(self.broker.handle({'op': 'status'})['active'], ['a', 'b'])
        self.assertEqual(self.broker.handle({'op': 'log', 'name': 'b'})['status'], 'running')
        self.assertEqual(self.broker.handle({'op': 'stop', 'name': 'a'})['stopping'], 'a')  # only a is asked to stop
        self.backend.controller.threads['a'].join(timeout=10)
        jobs = self.jobs()
        self.assertEqual((jobs['a']['status'], jobs['b']['status']), ('stopped', 'running'))
        self.assertAlmostEqual(jobs['a']['charged_usd'], 40 * RATE)
        self.provider.hold.set()
        self.assertEqual(self.backend.controller.wait(10), [])
        jobs = self.jobs()
        self.assertEqual((jobs['b']['status'], jobs['b']['charged_seconds']), ('complete', 100))
        self.assertAlmostEqual(jobs['b']['charged_usd'], 100 * RATE)
        budget = self.broker.handle({'op': 'budget'})  # each run charged its own provider interval at the price
        self.assertAlmostEqual(budget['used_usd'], 140 * RATE)
        self.assertAlmostEqual(budget['remaining_usd'], 1.0 - 140 * RATE)
        self.assertEqual(budget['used_worker_seconds'], 140)
        self.assertEqual([(t['name'], t['seconds']) for t in self.provider.trains], [('a', 120), ('b', 120)])
        # Practice (CPU only, uncharged) runs beside a training run; one practice of a run at a time.
        self.provider.hold = threading.Event()
        self.assertNotIn('error', self.launch('c'))
        practice = self.broker.handle({'op': 'eval', 'name': 'b'})
        self.assertEqual((practice['queued'], practice['practice_evaluations_remaining']), ('eval-b', 5))
        self.assertIn('already queued or running', self.broker.handle({'op': 'eval', 'name': 'b'})['error'])
        self.assertEqual(self.broker.handle({'op': 'budget'})['practice_evaluations_remaining'], 5)
        self.backend.tick()
        self.backend.controller.threads['eval-b'].join(timeout=10)
        self.assertEqual(self.jobs()['c']['status'], 'running')
        self.provider.hold.set()
        self.assertEqual(self.backend.controller.wait(10), [])
        self.assertEqual(self.jobs()['eval-b']['status'], 'complete')
        self.assertAlmostEqual(self.broker.handle({'op': 'budget'})['used_usd'], 240 * RATE)  # practice is free
        submitted = self.broker.handle({'op': 'submit', 'name': 'b'})['submitted']['native']
        self.assertAlmostEqual(submitted['charged_or_reserved_usd'], 240 * RATE)
        self.assertNotIn('error', self.broker.handle({'op': 'finalize'}))
        ledger = self.ledger()
        self.assertEqual((ledger['budget_mode'], ledger['limit_usd'], ledger['usd_per_second'], ledger['limits'],
                          ledger['finalized']), ('usd', 1.0, RATE, budgets.PARALLEL, True))
        self.assertNotIn('limit_seconds', ledger)

    def test_grants_never_add_up_past_the_budget(self):
        self.backend.close()
        self.out = Path(self.temp.name) / 'small'
        self.out.mkdir()
        self.backend = self.open(self.out / 'research', 0.2)
        self.broker = Broker(self.backend, self.out, 'network', research_usd=0.2)
        self.assertNotIn('error', self.launch('a'))  # $0.12 of $0.20 held
        for seconds in (120, 90):
            error = self.launch('b', seconds)['error']
            self.assertIn('nothing was launched', error)
            self.assertIn('only $0.0790 remain', error)
        budget = self.broker.handle({'op': 'budget'})
        self.assertLessEqual(budget['used_usd'], budget['budget_usd'])
        self.assertLess(budget['allocatable_seconds'], budget['min_run_seconds'])
        self.backend.tick()
        self.assertEqual(self.backend.controller.wait(10), [])
        self.assertAlmostEqual(self.broker.handle({'op': 'budget'})['used_usd'], 100 * RATE)
        self.assertNotIn('error', self.launch('b', 90))  # the settled grant gave back what it did not use

    def test_ledger_holds_each_grant_until_it_settles_and_limits_unresolved_workers(self):
        from harness.games.clash.controller import Account
        path = Path(self.temp.name) / 'account/ledger.json'
        ledger = Account(path, usd=0.3)
        self.addCleanup(ledger.lock.close)
        ledger.reserve('x', 150)
        ledger.reserve('y', 150)
        with self.assertRaisesRegex(ValueError, '2 unresolved'):  # one GPU per allocation, two GPUs at once
            ledger.reserve('z', 60)
        for name in ('x', 'y'):
            ledger.record(name, status='running', sandbox_id='sb-' + name)
        ledger.settle('x', dict(provider_id='sb-x', started_at=1000.0, finished_at=1100.0))
        ledger.reserve('z', 60)
        ledger.settle('y', dict(provider_id='sb-y', started_at=1000.0, finished_at=1100.0))
        self.assertEqual(set(ledger.read()['milestones']), {str(1 / 3)})  # 200 s of $0.30 / RATE = 372 s
        with self.assertRaisesRegex(ValueError, r'only \$0\.0903 remain'):  # 200 charged + 60 held of 372
            ledger.reserve('w', 150)
        self.assertLessEqual(ledger.used() * RATE, 0.3)
        with self.assertRaisesRegex(ValueError, 'usd mode'):
            Account(path, seconds=600)
        with self.assertRaisesRegex(ValueError, 'cannot be reset'):
            Account(path, usd=0.4)

    def test_host_settle_stops_every_running_job_and_charges_each_its_real_interval(self):
        self.provider.hold = threading.Event()
        self.assertNotIn('error', self.launch('a'))
        self.assertNotIn('error', self.launch('b'))
        self.backend.tick()
        self.running(2)
        self.assertIn('active', self.broker.handle({'op': 'finalize'})['error'])
        self.backend.settle()
        self.backend.settle()  # idempotent
        jobs = self.jobs()
        self.assertEqual([(j['status'], j['charged_seconds']) for j in jobs.values()], [('stopped', 40)] * 2)
        self.assertEqual(self.backend.status()['active'], [])
        self.assertAlmostEqual(self.backend.budget()['used_usd'], 80 * RATE)

    def test_an_unresolved_worker_keeps_its_grant_and_its_gpu_until_the_provider_reports(self):
        self.provider.outcome = 'unknown-end'
        self.assertNotIn('error', self.launch('a'))
        self.backend.tick()
        self.assertEqual(self.backend.controller.wait(10), [])
        self.assertEqual(self.jobs()['a']['status'], 'uncertain')
        self.assertAlmostEqual(self.backend.budget()['used_usd'], 150 * RATE)
        self.provider.outcome, self.provider.hold = 'learn', threading.Event()
        self.assertNotIn('error', self.launch('b'))
        self.assertIn('At most 2 GPUs', self.launch('c')['error'])
        self.backend.tick()  # starts b and asks the provider for a's end
        self.running(1)
        self.backend.settle()  # stops b; a is settled from its receipt
        jobs = self.jobs()
        self.assertEqual([(j['status'], j['charged_seconds']) for j in jobs.values()], [('failed', 90), ('stopped', 40)])
        self.assertAlmostEqual(self.backend.budget()['used_usd'], 130 * RATE)

    def test_the_mode_is_recorded_and_a_reopened_session_keeps_it(self):
        self.assertEqual((self.backend.budget_mode, self.backend.limits), ('usd', budgets.PARALLEL))
        self.backend.close()
        root = self.out / 'research'
        with self.assertRaisesRegex(ValueError, 'usd mode'):
            registry.backend('clash')(self.release, root, smoke=True, resume=True, budget_seconds=600,
                                      options={'_provider': self.provider}).open()
        reopened = registry.backend('clash')(self.release, root, smoke=True, resume=True,
                                             options={'_provider': self.provider})  # host grading names no amount
        reopened.open()
        self.addCleanup(reopened.close)
        self.assertEqual((reopened.budget_mode, reopened.limits['max_concurrent_jobs']), ('usd', 4))
        self.assertEqual(reopened.budget()['budget_usd'], 1.0)


class FakeModal:
    """A stand-in `modal` module: one in-memory volume store, sandboxes that "run" the worker instantly or never."""
    __version__ = 'stand-in'

    def __init__(self):
        self.volumes, self.sandboxes, self.finishes = {}, {}, True
        self.answers_stop = False  # a running worker ends at its next update when asked, as the trusted loop does
        # GPU sandboxes stay queued (no start) for this many receipt reads; None: the provider never starts them.
        self.queue = 0
        self.lock = threading.Lock()  # dollar mode creates workers from several threads
        fake = self

        class Image:
            object_id = 'im-stand-in'

            def __getattr__(self, name):  # from_registry(...).apt_install(...) ... all return the image
                return lambda *a, **k: self

        class App:
            app_id = 'ap-stand-in'

            def __init__(self, name):
                self.name = name

            def run(self, environment_name):
                fake.environment = environment_name
                return self

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                fake.left = True

        class Volume:
            def __init__(self):
                self.files = {}

            @staticmethod
            def from_name(name, **kwargs):
                return fake.volumes.setdefault(name, Volume())

            def batch_upload(self):
                return self

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

            def put_directory(self, local, remote):
                for path in sorted(Path(local).rglob('*')):
                    if path.is_file():
                        self.files[remote.strip('/') + '/' + path.relative_to(local).as_posix()] = path.read_bytes()

            def iterdir(self, path, recursive=True):
                return [types.SimpleNamespace(path='/' + name, type=types.SimpleNamespace(name='FILE'),
                                              size=len(data)) for name, data in self.files.items()]

            def read_file(self, remote):
                yield self.files[remote.lstrip('/')]

        class Process:
            def __init__(self, code):
                self.returncode, self._code = None, code
                self.stdout = self.stderr = types.SimpleNamespace(read=lambda: '')

            def poll(self):
                self.returncode = self._code() if callable(self._code) else self._code
                return self.returncode

        class Sandbox:
            def __init__(self, options):
                with fake.lock:
                    self.object_id = f'sb-{len(fake.sandboxes) + 1}'
                    fake.sandboxes[self.object_id] = self
                self.options = options
                self.started, self.finished, self.fs = time.time(), 0, {}
                self.queued, self.reads, self.killed = fake.queue if 'gpu' in options else 0, 0, False
                if self.queued != 0:
                    self.started = 0
                self.filesystem = types.SimpleNamespace(
                    make_directory=lambda path: None, write_bytes=lambda data, path: self.fs.__setitem__(path, data),
                    write_text=self.write_text,
                    read_text=lambda path: self.fs[path].decode())

            def write_text(self, text, path):
                self.fs[path] = text.encode()
                if path == STOP_REQUEST and fake.answers_stop:
                    self.train(self.options['volumes']['/retained'])  # the loop ended and exported
                    self.ended = True

            @staticmethod
            def create(*command, **options):
                return Sandbox(options)

            @staticmethod
            def from_id(identity):
                return fake.sandboxes[identity]

            def exec(self, *command, timeout=None):
                self.commands = getattr(self, 'commands', []) + [command]
                if 'simulators.clash.autoresearch.fast_worker' in command and fake.finishes:
                    self.train(self.options['volumes']['/retained'])
                elif command[:3] == ('python', '-u', '-c'):
                    schedule = json.loads(self.fs['/private/schedule.json'])
                    identity = json.loads(self.fs['/private/candidate/manifest.json'])['identity']
                    self.fs['/private/result.json'] = json.dumps(dict(
                        games=80, raw_score_0_100=25.0, submission_identity=identity, split='public-development',
                        schedule_commitment=digest(schedule), rows=[])).encode()
                if fake.finishes or 'chmod' in command:
                    return Process(0)
                return Process(lambda: 0 if getattr(self, 'ended', False) else None)

            def train(self, volume):
                """What a completed fast worker leaves on the volume: the source it ran, weights and its manifest."""
                out = {'output/artifact/' + name.split('/', 2)[2]: data for name, data in volume.files.items()
                       if name.startswith('inputs/source/')}
                weights = io.BytesIO()
                np.savez(weights, layer=np.ones(8, np.float32))
                out['output/artifact/weights.npz'] = weights.getvalue()
                out['output/artifact/checkpoint.pt'] = b'stand-in checkpoint'
                out['output/artifact/progress.jsonl'] = b'{"updates": 1}\n'
                out['output/artifact/result.json'] = json.dumps(dict(
                    updates=1, samples=32, natural_games=1, optimizer_steps=1, finite_losses=True,
                    weight_l1_change=0.25, seconds=50.0, loss_summary={'last': [0.5]})).encode()
                out['output/starter.log'] = b'update 1\n'
                volume.files.update(out)
                volume.files['output-manifest.json'] = json.dumps(dict(
                    complete=True, files={name: hashlib.sha256(data).hexdigest() for name, data in out.items()})).encode()

            def terminate(self, wait=False):
                self.killed = True
                self.finished = self.finished or (self.started + 42.0 if self.started else time.time())

        self.Image, self.App, self.Volume, self.Sandbox = Image(), App, Volume, Sandbox

    def allocation_info(self, app_id, sandbox_id):
        box = self.sandboxes[sandbox_id]
        if not box.started and not box.killed:
            box.reads += 1
            if box.queued is not None and box.reads >= box.queued:
                box.started = time.time()  # a GPU finally became free
        return dict(provider_id=sandbox_id, started_at=box.started, finished_at=box.finished)


class RealProvider(Release):
    """The real FastProvider and Backend.open against a stand-in Modal SDK: ledger protocol, collection, stop."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.modal = FakeModal()
        patches = [mock.patch.dict(sys.modules, {'modal': self.modal}),
                   mock.patch.multiple('simulators.generals.autoresearch.provider', configure_sandbox_waits=lambda: None,
                                       read_rpc=lambda call: call(), allocation_info=self.modal.allocation_info,
                                       wait_process=lambda process: process.poll())]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.out = Path(self.temp.name) / 'session'
        self.out.mkdir()
        self.backend = registry.backend('clash')(self.release, self.out / 'research', smoke=True, budget_seconds=600)
        self.backend.open()
        self.addCleanup(self.backend.close)
        self.fast(self.backend)
        self.broker = Broker(self.backend, self.out, 'network', research_seconds=600)

    @staticmethod
    def fast(backend):
        backend.controller.provider.start_poll_seconds = backend.controller.provider.abandon_poll_seconds = 0.01

    def work(self):
        self.backend.tick()
        self.assertEqual(self.backend.controller.wait(20), [])

    def test_training_worker_request_collection_bundle_and_practice(self):
        root = self.out / 'research'
        self.assertEqual(self.modal.environment, 'game-autoresearch-project')
        self.assertEqual(json.loads((root / 'app.json').read_text())['hardware']['gpu'], 'L40S')
        files = dict(self.files)
        files['training/learner.py'] += SENTINEL
        self.assertNotIn('error', self.broker.handle({'op': 'run', 'name': 'a', 'seconds': 120, 'files': files}))
        self.work()
        job = self.backend.status()['jobs'][0]
        self.assertEqual((job['status'], job['charged_seconds']), ('complete', 42.0), job)
        train = self.modal.sandboxes['sb-1'].options
        self.assertEqual((train['gpu'], train['cpu'], train['memory'], train['timeout'], train['block_network']),
                         ('L40S', (4, 4), (16384, 16384), 120, True))
        self.assertEqual(train['environment_name'], 'game-autoresearch-project')
        request = json.loads((root / 'jobs/a/request.json').read_text())
        self.assertEqual((request['hard_worker_seconds'], request['reserved_seconds']), (120, 150))
        volume = train['volumes']['/retained']
        self.assertEqual(volume.files['inputs/source/learner.py'].decode(), files['training/learner.py'])
        self.assertEqual(volume.files['inputs/source/config.json'].decode(), files['configs/train.json'])
        self.assertEqual((root / 'jobs/a/bundle/learner.py').read_text(), files['training/learner.py'])
        self.assertIn('update 1', self.backend.log('a')['trainer'])
        self.assertEqual(self.backend.budget()['used_seconds'], 42.0)
        self.assertNotIn('error', self.broker.handle({'op': 'eval', 'name': 'a'}))
        self.work()
        practice = self.modal.sandboxes['sb-2']
        self.assertNotIn('gpu', practice.options)
        self.assertEqual((practice.options['cpu'], practice.options['timeout']), ((8, 8), 600))
        sent = json.loads(practice.fs['/private/schedule.json'])
        self.assertEqual([actor['name'] for actor in sent['opponents']],
                         ['standard', 'pressure', 'intermediate', 'teacher'])  # scripted panel: no bundles sent
        self.assertFalse(any(path.startswith('/private/') and path.endswith('weights.npz') and 'candidate' not in path
                             for path in practice.fs))
        self.assertEqual(self.backend.status()['jobs'][0]['practice']['raw_score_0_100'], 25.0)
        self.assertEqual(self.backend.budget()['used_seconds'], 42.0)  # practice is not on the GPU ledger
        self.backend.close()
        self.assertTrue(self.modal.left)

    def test_stop_terminates_a_worker_that_does_not_end_within_the_grace(self):
        self.modal.finishes = False
        self.backend.controller.provider.stop_grace_seconds = 0.2
        self.assertNotIn('error', self.broker.handle({'op': 'run', 'name': 'a', 'seconds': 120, 'files': self.files}))
        self.backend.tick()
        for _ in range(200):
            if self.modal.sandboxes:
                break
            time.sleep(0.01)
        reply = self.broker.handle({'op': 'stop', 'name': 'a'})
        self.assertEqual((reply['stopping'], 'error' in reply), ('a', False), reply)
        self.backend.controller.wait(20)
        box = self.modal.sandboxes['sb-1']
        self.assertEqual(box.fs[STOP_REQUEST], b'stop\n')  # it was asked first, and only then terminated
        job = self.backend.status()['jobs'][0]
        self.assertEqual((job['status'], job['charged_seconds']), ('stopped', 42.0), job)
        self.assertIn('stopped by request', job['error'])
        self.assertTrue(box.finished)
        self.assertEqual(self.backend.status()['active'], [])

    def running(self):
        self.modal.finishes = False
        self.launch()
        self.backend.tick()
        deadline = time.monotonic() + 10
        while not any(command[:1] == ('python',) for command in getattr(self.modal.sandboxes.get('sb-1'), 'commands', [])):
            self.assertLess(time.monotonic(), deadline, 'the worker never started training')
            time.sleep(0.01)
        return self.modal.sandboxes['sb-1']

    def test_stop_ends_training_at_its_next_update_and_keeps_the_network(self):
        self.modal.answers_stop = True
        box = self.running()
        reply = self.broker.handle({'op': 'stop', 'name': 'a'})
        self.assertIn('keeps the network', reply['note'])
        self.backend.controller.wait(20)
        job = self.backend.status()['jobs'][0]
        self.assertEqual((job['status'], job.get('stopped_early'), job['charged_seconds']), ('complete', True, 42.0), job)
        self.assertNotIn('error', job)
        self.assertTrue(box.killed)  # the worker is still released once its outputs are final
        self.assertEqual(self.backend.status()['active'], [])
        self.assertNotIn('error', self.broker.handle({'op': 'submit', 'name': 'a'}))  # a kept network is an answer

    def test_closing_the_session_ends_a_running_worker_at_once(self):
        self.modal.answers_stop = True  # even a loop that would answer is not waited for: nothing unsubmitted is kept
        box = self.running()
        self.backend.settle()
        job = self.backend.status()['jobs'][0]
        self.assertEqual((job['status'], job['charged_seconds']), ('stopped', 42.0), job)
        self.assertNotIn(STOP_REQUEST, box.fs)

    def launch(self, name='a'):
        reply = self.broker.handle({'op': 'run', 'name': name, 'seconds': 120, 'files': self.files})
        self.assertNotIn('error', reply, reply)

    def wait_for_sandbox(self, name):
        deadline = time.monotonic() + 10
        while name not in self.modal.sandboxes:
            self.assertLess(time.monotonic(), deadline, 'no worker was created')
            time.sleep(0.01)

    def test_a_queued_worker_is_waited_for_and_charged_only_from_its_start(self):
        self.modal.queue = 30  # far beyond the old ~60-second give-up in real polls; queued time is not charged
        self.launch()
        self.work()
        job = self.backend.status()['jobs'][0]
        self.assertEqual((job['status'], job['charged_seconds']), ('complete', 42.0), job)
        self.assertGreaterEqual(self.modal.sandboxes['sb-1'].reads, 30)
        self.assertEqual(self.backend.budget()['used_seconds'], 42.0)

    def test_a_worker_that_never_starts_fails_at_zero_and_the_session_continues(self):
        self.modal.queue = None
        self.launch()
        with mock.patch.object(workers, 'START_PATIENCE_SECONDS', 0.3):
            self.work()
        job = self.backend.status()['jobs'][0]
        self.assertEqual((job['status'], job['charged_seconds'], job['never_started']), ('failed', 0, True), job)
        self.assertIn('never started', job['error'])
        log = self.broker.handle({'op': 'log', 'name': 'a'})
        self.assertNotIn('error', log)  # reading a failed run's log is a successful command
        self.assertIn('never started', log['job_error'])
        self.assertEqual(self.backend.status()['active'], [])
        row, = json.loads((self.out / 'research/allocation-ledger.json').read_text())['allocations']
        self.assertEqual((row['status'], row['charged_seconds'], row['never_started'], row['receipt']['provider_id']),
                         ('failed', 0, True, 'sb-1'))
        self.assertTrue(self.modal.sandboxes['sb-1'].killed)
        self.assertEqual(self.backend.budget()['used_seconds'], 0)
        self.modal.queue = 0  # the next run is accepted at once and runs normally
        self.launch('b')
        self.work()
        self.assertEqual(self.backend.status()['jobs'][1]['status'], 'complete')
        self.assertNotIn('error', self.broker.handle({'op': 'submit', 'name': 'b'}))
        self.assertNotIn('error', self.broker.handle({'op': 'finalize'}))
        self.assertTrue(json.loads((self.out / 'research/allocation-ledger.json').read_text())['finalized'])

    def test_stop_while_queued_ends_the_wait_and_charges_nothing(self):
        self.modal.queue = None
        self.launch()
        self.backend.tick()
        self.wait_for_sandbox('sb-1')
        self.assertEqual(self.broker.handle({'op': 'stop', 'name': 'a'})['stopping'], 'a')
        self.backend.controller.wait(20)
        job = self.backend.status()['jobs'][0]
        self.assertEqual((job['status'], job['charged_seconds'], job['never_started']), ('stopped', 0, True), job)
        self.assertIn('stopped by request', job['error'])
        self.assertTrue(self.modal.sandboxes['sb-1'].killed)
        self.assertEqual((self.backend.status()['active'], self.backend.budget()['used_seconds']), ([], 0))

    def uncertain(self, error='RuntimeError: authoritative allocation start unavailable'):
        """What the earlier ~60-second give-up left behind: a recorded sandbox and an uncertain job."""
        controller = self.backend.controller
        box = self.modal.Sandbox.create('/bin/sleep', gpu='L40S')
        controller.ledger.reserve('a', 150)
        controller.ledger.record('a', status='uncertain', app_id='ap-stand-in', sandbox_id=box.object_id, error=error)
        controller._add(dict(name='a', kind='train', status='uncertain', seconds=120, reserved_seconds=150, error=error,
                             queued_at=time.time()))
        return box

    def test_recovery_terminates_a_never_started_uncertain_worker_and_settles_it_at_zero(self):
        self.modal.queue = None
        box = self.uncertain()
        self.assertEqual(self.backend.status()['active'], ['a'])
        self.assertIn('One job at a time', self.broker.handle({'op': 'run', 'name': 'b', 'seconds': 120,
                                                              'files': self.files})['error'])
        self.work()  # the periodic reconciliation
        job = self.backend.status()['jobs'][0]
        self.assertEqual((job['status'], job['charged_seconds'], job['never_started']), ('failed', 0, True), job)
        self.assertTrue(box.killed)
        self.assertEqual(self.backend.budget()['used_seconds'], 0)
        self.modal.queue = 0
        self.launch('b')

    def test_stop_resolves_an_uncertain_job_now(self):
        box = self.uncertain('RuntimeError: provider end unavailable')  # it started, then was abandoned
        self.assertTrue(box.started and not box.finished)
        reply = self.broker.handle({'op': 'stop', 'name': 'a'})
        self.assertEqual((reply['resolved'], reply['status'], reply['charged_seconds']), (True, 'stopped', 42.0), reply)
        self.assertNotIn('error', reply)  # the stop succeeded; the job's own failure is `job_error`
        self.assertIn('provider end unavailable', reply['job_error'])
        self.assertTrue(box.killed)
        self.assertEqual((self.backend.status()['active'], self.backend.budget()['used_seconds']), ([], 42.0))
        self.modal.queue = None  # and one that never started is settled at zero the same way
        self.launch('b')
        with mock.patch.object(workers, 'START_PATIENCE_SECONDS', 0.3):
            self.work()
        self.assertEqual(self.backend.status()['jobs'][1]['charged_seconds'], 0)

    def test_dollar_mode_runs_workers_side_by_side_on_their_own_volumes(self):
        out = Path(self.temp.name) / 'dollars'
        out.mkdir()
        backend = registry.backend('clash')(self.release, out / 'research', smoke=True, budget_usd=1.0)
        backend.open()
        self.addCleanup(backend.close)
        self.fast(backend)
        broker = Broker(backend, out, 'network', research_usd=1.0)
        self.assertEqual(json.loads((out / 'research/app.json').read_text())['budget_mode'], 'usd')
        files = dict(self.files)
        for name in ('a', 'b'):
            files['training/learner.py'] = self.files['training/learner.py'] + f'\n# run {name}\n'
            self.assertNotIn('error', broker.handle({'op': 'run', 'name': name, 'seconds': 120, 'files': files}))
        backend.tick()
        self.assertEqual(backend.status()['active'], ['a', 'b'])  # both started by one tick
        self.assertEqual(backend.controller.wait(20), [])
        jobs = backend.status()['jobs']
        self.assertEqual([(j['name'], j['status'], j['charged_seconds']) for j in jobs],
                         [('a', 'complete', 42.0), ('b', 'complete', 42.0)])
        requests = [json.loads((out / 'research/jobs' / name / 'request.json').read_text()) for name in ('a', 'b')]
        self.assertNotEqual(requests[0]['volume'], requests[1]['volume'])
        for name in ('a', 'b'):
            self.assertTrue((out / f'research/jobs/{name}/bundle/learner.py').read_text().endswith(f'# run {name}\n'))
        self.assertAlmostEqual(backend.budget()['used_usd'], 84 * RATE)

    def test_grade_after_reopening_creates_its_own_cpu_worker(self):
        self.launch()
        self.work()
        identity = self.broker.handle({'op': 'submit', 'name': 'a'})['submitted']['native']['identity']
        self.assertNotIn('error', self.broker.handle({'op': 'finalize'}))
        self.backend.close()
        graded = registry.backend('clash')(self.release, self.out / 'research', smoke=True, resume=True)
        graded.open()
        self.addCleanup(graded.close)
        self.assertIsNone(graded.controller.provider)  # nothing is created until grading needs a worker
        before = set(self.modal.sandboxes)
        result = graded.grade()
        worker, = [self.modal.sandboxes[name] for name in set(self.modal.sandboxes) - before]
        self.assertNotIn('gpu', worker.options)
        self.assertEqual(worker.options['timeout'], 3600)
        self.assertEqual((result['status'], result['schedule'], result['submission_identity']),
                         ('development_only', 'protected', identity))
        record = json.loads((self.out / 'research/grade-app.json').read_text())
        self.assertEqual(record['environment'], 'game-autoresearch-project')
        self.assertTrue((self.out / 'research/grade/result.json').is_file())


class RecipeSession(Release):
    """A whole recipe session through the real core: `ar` in a local box, research root, fresh final root."""

    def test_recipe_session_trains_the_final_recipe_in_its_own_root_and_grades_it_on_the_protected_schedule(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        provider, out, box = Provider(), Path(temp.name) / 'session', Path(temp.name) / 'box'

        class Stubbed(registry.backend('clash')):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **dict(kwargs, options={'_provider': provider}))
        patch = mock.patch.object(registry, 'backend', lambda game: Stubbed)
        patch.start()
        self.addCleanup(patch.stop)
        seen = {}

        def script(ar):
            budget = ar('budget')
            self.assertEqual((budget['variant'], budget['remaining_seconds'], budget['final_train_seconds']),
                             ('recipe', 600.0, 120.0))
            ar.edit('training/learner.py', (ar.workspace / 'training/learner.py').read_text() + SENTINEL)
            self.assertTrue(ar('check')['checked'])
            self.assertEqual(ar('run', 'explore', '--seconds', '120', '--config', 'configs/smoke.json')
                             ['reserved_seconds'], 150)
            self.assertEqual(ar('wait')['active'], [])
            ar('eval', 'explore')
            ar('wait')
            seen['status'] = ar('status')
            self.assertEqual(ar('submit', '--note', 'recipe')['submitted']['kind'], 'recipe')
            self.assertTrue(ar('finalize')['finalized'])

        def launcher(session_dir):
            researcher = Researcher(session_dir / 'task', box)
            return Process(lambda: script(researcher))
        result = session.run('clash', 'recipe', self.release, out, model='test/model', smoke=True, launcher=launcher,
                             find_box=lambda: LocalBox(box / 'ar') if (box / 'ar').is_dir() else None)
        self.assertEqual((result['status'], result['error'], result['closed_by']), ('researched', None, 'researcher'))
        self.assertEqual([(j['kind'], j['status']) for j in seen['status']['jobs']],
                         [('train', 'complete'), ('practice', 'complete')])
        self.assertEqual(seen['status']['jobs'][0]['practice']['raw_score_0_100'], 37.5)
        final = result['final_training']
        self.assertEqual((final['status'], final['error'], final['allowance_seconds'], final['seconds'], final['config']),
                         ('complete', None, 120.0, 90.0, 'configs/smoke.json'))
        self.assertEqual([(run['name'], run['seconds']) for run in provider.trains], [('explore', 120), ('final', 90)])
        self.assertTrue(provider.trains[1]['files']['learner.py'].endswith(SENTINEL))
        research = json.loads((out / 'research/allocation-ledger.json').read_text())
        fresh = json.loads((out / 'final/allocation-ledger.json').read_text())
        self.assertEqual((research['limit_seconds'], research['finalized'], research['submissions']), (600, False, []))
        self.assertEqual((fresh['limit_seconds'], fresh['finalized'], fresh['submissions'][0]['name']),
                         (120, True, 'final'))
        self.assertEqual(final['budget']['remaining_seconds'], 20)  # charged the 100-second receipt, not the hold
        with mock.patch.object(policy, 'require_modal', lambda: None):
            graded = session.grade(out)
        self.assertEqual((graded['status'], graded['schedule'], graded['official']), ('development_only', 'protected', False))
        record = json.loads((out / 'grade.json').read_text())
        self.assertEqual((record['graded_root'], record['official']), ('final', False))
        self.assertEqual(provider.practices[-1]['split'], 'protected-development')

    def test_dollar_mode_session_runs_research_in_parallel_and_the_fresh_final_in_seconds(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        provider, out, box = Provider(), Path(temp.name) / 'session', Path(temp.name) / 'box'

        class Stubbed(registry.backend('clash')):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **dict(kwargs, options={'_provider': provider}))
        patch = mock.patch.object(registry, 'backend', lambda game: Stubbed)
        patch.start()
        self.addCleanup(patch.stop)

        def script(ar):
            budget = ar('budget')
            self.assertEqual((budget['variant'], budget['budget_mode'], budget['budget_usd'], budget['concurrency'],
                              budget['final_train_seconds']), ('recipe', 'usd', 0.48, 4, 120.0))
            ar.edit('training/learner.py', (ar.workspace / 'training/learner.py').read_text() + SENTINEL)
            for name in ('explore', 'wide'):
                self.assertAlmostEqual(ar('run', name, '--seconds', '120', '--config', 'configs/smoke.json')
                                       ['reserved_usd'], 150 * RATE)
            self.assertEqual(ar('wait')['active'], [])
            self.assertAlmostEqual(ar('budget')['used_usd'], 200 * RATE)
            self.assertEqual(ar('submit', '--note', 'recipe')['submitted']['kind'], 'recipe')
            self.assertTrue(ar('finalize')['finalized'])

        def launcher(session_dir):
            researcher = Researcher(session_dir / 'task', box)
            return Process(lambda: script(researcher))
        result = session.run('clash', 'recipe', self.release, out, model='test/model', smoke=True, budget_mode='usd',
                             launcher=launcher,
                             find_box=lambda: LocalBox(box / 'ar') if (box / 'ar').is_dir() else None)
        self.assertEqual((result['status'], result['error']), ('researched', None))
        plan = json.loads((out / 'plan.json').read_text())
        self.assertEqual((plan['budget_mode'], plan['research_usd'], plan['research_seconds'], plan['wall_seconds']),
                         ('usd', 0.48, None, 4800.0))
        self.assertIn('up to 4 at once', (out / 'task/instruction.md').read_text())
        final = result['final_training']
        self.assertEqual((final['status'], final['allowance_seconds'], final['seconds']), ('complete', 120.0, 90.0))
        self.assertEqual(sorted(run['name'] for run in provider.trains), ['explore', 'final', 'wide'])
        research = json.loads((out / 'research/allocation-ledger.json').read_text())
        fresh = json.loads((out / 'final/allocation-ledger.json').read_text())
        self.assertEqual((research['budget_mode'], research['limit_usd']), ('usd', 0.48))
        self.assertEqual((fresh['budget_mode'], fresh['limit_seconds'], fresh['finalized']), ('seconds', 120, True))


class Download(unittest.TestCase):
    def test_disconnected_download_never_becomes_a_complete_file(self):
        from harness.games.clash.provider import download_atomic

        class Lost:
            def read_file(self, path):
                yield b'partial'
                raise OSError('controller disconnected')

        class Complete:
            def read_file(self, path):
                yield b'complete'
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'result.json'
            with self.assertRaisesRegex(OSError, 'disconnected'):
                download_atomic(Lost(), '/result.json', target)
            self.assertEqual(list(Path(temp).iterdir()), [])
            download_atomic(Complete(), '/result.json', target)
            self.assertEqual(target.read_bytes(), b'complete')


del Release  # a fixture base class, not a test case of its own

if __name__ == '__main__':
    unittest.main()
