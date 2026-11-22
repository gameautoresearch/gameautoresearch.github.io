"""Generals backend against its real controller and ledger; the provider is stubbed before any allocation.

Run with the game's research interpreter: .venv-generals-ar/bin/python -m unittest harness.tests.test_generals
"""
import functools
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from harness import handoff, policy, registry, session, task, workers, workspace
from harness.broker import Broker, VARIANTS
from harness.tests.test_handoff import FolderVolume, publish
from harness.transport import LocalBox
from harness.tests.support import Process, Researcher, host_runs, release_fixture

try:
    from simulators.generals.autoresearch.ledger import Ledger
    from simulators.generals.autoresearch.contract import TRAIN_SHAPE, rate
    from simulators.generals.autoresearch import bundle  # noqa: F401  the controller's first non-stdlib import
except ImportError as exc:  # wrong interpreter
    raise unittest.SkipTest(f'Generals research dependencies unavailable: {exc}')

# Dollars per second of the training worker. Test sessions are opened with a budget written in its seconds.
R = rate(TRAIN_SHAPE)


def private(release):
    """The private half of a release that practice reads: a scripted-only panel."""
    folder = Path(release) / 'private'
    folder.mkdir()
    (folder / 'panel.json').write_text(json.dumps({'practice': [{'name': 'random', 'kind': 'random'}],
                                                   'official': [{'name': 'random', 'kind': 'random'}]}))


def sealed(folder, code):
    """A source-bound policy package as the worker's exporter leaves it."""
    import numpy as np
    from simulators.generals.autoresearch.bundle import seal
    source = folder.with_name(folder.name + '-source')
    source.mkdir(parents=True)
    for name, text in code.items():
        (source / name).write_text(text)
    (source / 'model.json').write_text('{}')
    np.savez(source / 'weights.npz', w=np.zeros(4, np.float32))
    return seal(source, folder)


class Provider:
    """Records the exact payload and settles the allocation as a provider would."""

    def __init__(self, outcome='failed'):
        self.outcome = outcome
        self.payloads = []
        self.live = {}

    def lease_live(self, name):
        return name in self.live

    def lease_exec(self, name, command):
        self.live[name]['commands'].append(command)
        return {'exec': f"{len(self.live[name]['commands']):04d}", 'state': 'running'}

    def lease_poll(self, name, exec_id, tail=None):
        return {'exec': exec_id, 'state': 'exited', 'exit_code': 0, 'output': 'ran ' + self.live[name]['commands'][-1]}

    def lease_push(self, name, files):
        self.live[name]['files'] = dict(files)
        return {'pushed': sorted(files)}

    def lease_sync(self, name, reload=False):
        return {'committed': '/scratch', 'reloaded': reload}

    def lease_package(self, target, name, path, code, config):
        return target / 'bundle', sealed(target / 'bundle', code)

    def execute(self, ledger, job, stop=None):
        payload = json.loads(job['payload'])
        self.payloads.append(payload)
        folder = ledger.root / 'jobs' / job['name']
        folder.mkdir(parents=True)
        now = time.time()
        ledger.start(job['name'], 'stub', now)
        if payload['kind'] == 'lease':  # held until stopped; the stub charges 40 worker seconds
            self.live[job['name']] = {'files': dict(payload['files']), 'commands': []}
            while not (stop and stop()):
                time.sleep(0.01)
            del self.live[job['name']]
            (folder / 'exec').mkdir()
            (folder / 'exec/0001.json').write_text(json.dumps({'exec': '0001', 'state': 'ended with the lease',
                                                               'output': 'late'}))
            ledger.finish(job['name'], now + 40, status='stopped')
        elif self.outcome == 'failed':
            ledger.finish(job['name'], now + 30, status='failed', error='stub provider: no worker was allocated')
        elif payload['kind'] == 'practice':
            (folder / 'practice.json').write_text(json.dumps({'status': 'development_only', 'reward': 0.5, 'games': 4}))
            ledger.finish(job['name'], now + 17)
        else:
            identity = sealed(folder / 'bundle', payload['code'])
            ledger.finish(job['name'], now + job['grant'] - 5, 'complete', folder / 'bundle', identity)


class GeneralsBackend(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.release, self.meta = release_fixture('generals', self.temp.name, bind=False)
        private(self.release)
        policy.bind_release('generals', self.release)
        self.out = Path(self.temp.name) / 'session'
        self.out.mkdir()
        base = workspace.public(self.release) / workspace.WORKSPACE
        self.files = {name: (base / name).read_text() for name in self.meta['writable']}
        self.open()

    def open(self, outcome='failed', budget_seconds=360, mode='usd'):
        """A smoke session; in dollar mode (the v2 contract these tests were written for) its budget is what
        `budget_seconds` of the training worker cost."""
        if getattr(self, 'backend', None) is not None:
            self.backend.close()
            self.out = Path(self.temp.name) / ('session-' + outcome + str(budget_seconds) + mode)
            self.out.mkdir()
        self.provider = Provider(outcome)
        amount = dict(budget_usd=budget_seconds * R) if mode == 'usd' else dict(budget_seconds=budget_seconds)
        self.backend = registry.backend('generals')(self.release, self.out / 'research', smoke=True,
                                                    options={'_provider': self.provider}, **amount)
        self.backend.open()
        self.addCleanup(self.backend.close)
        self.broker = host_runs(Broker(self.backend, self.out, 'network', research_seconds=amount.get('budget_seconds'),
                                       research_usd=amount.get('budget_usd')))
        return self.broker

    def settle(self):
        self.backend.tick()
        self.backend.controller.join(timeout=10)

    def test_a_host_that_cannot_verify_the_engine_pin_opens_no_session(self):
        # On 2026-11-13 a host copy without the engine's git metadata opened sessions normally and then failed
        # every `ar run` and `ar package` at `git rev-parse HEAD`, in front of the researcher.
        out = Path(self.temp.name) / 'unverifiable'
        out.mkdir()
        backend = registry.backend('generals')(self.release, out / 'research', smoke=True, budget_seconds=360,
                                               options={'_provider': Provider('failed')})
        fault = subprocess.CalledProcessError(128, ['git', 'rev-parse', 'HEAD'])
        with mock.patch('simulators.generals.autoresearch.contract.frozen_manifest', side_effect=fault):
            with self.assertRaisesRegex(RuntimeError, 'cannot verify the pinned Generals engine.*nothing was spent'):
                backend.open()
        self.assertIsNone(backend.controller)
        self.assertFalse((out / 'research').exists())

    def test_run_carries_exactly_the_researchers_files_and_is_charged(self):
        budget = self.broker.handle({'op': 'budget'})
        self.assertAlmostEqual(budget['remaining_usd'], 360 * R)
        self.assertEqual((budget['concurrency'], budget['concurrent_gpus']), (4, 2))
        self.assertAlmostEqual(budget['usd_per_hour']['lease --gpu 1 --cpu 4 --memory 16 (and eval)'], 2.9031, places=3)
        self.assertEqual(budget['affordable_seconds']['lease --gpu 1 --cpu 4 --memory 16 (and eval)'], 360)
        self.assertNotIn('budget_seconds', budget)
        files = dict(self.files)
        files['training/policy.py'] += '\n# researcher change\n'
        reply = self.broker.handle({'op': 'run', 'name': 'a', 'seconds': 120, 'config': 'configs/smoke.json',
                                    'files': files})
        self.assertNotIn('error', reply, reply)
        self.assertEqual(self.broker.handle({'op': 'status'})['active'], ['a'])
        self.settle()
        payload = self.provider.payloads[0]
        self.assertEqual(payload['code'], {name.split('/')[1]: files[name] for name in files if name.endswith('.py')})
        self.assertEqual(payload['config'], json.loads(files['configs/smoke.json']))
        self.assertEqual((payload['shape'], payload['rate']), ({'gpu': 1, 'cpu_cores': 4, 'memory_mib': 16384}, R))
        status = self.broker.handle({'op': 'status'})
        self.assertEqual([(j['name'], j['kind'], j['status']) for j in status['jobs']], [('a', 'train', 'failed')])
        self.assertEqual(status['active'], [])
        budget = self.broker.handle({'op': 'budget'})
        self.assertAlmostEqual(budget['used_usd'], 30 * R)  # failed allocations stay charged
        self.assertIn('error', self.broker.handle({'op': 'submit', 'name': 'a'}))  # no trained artifact
        self.assertIn('Unknown run', self.broker.handle({'op': 'log', 'name': 'zzz'})['error'])

    def test_bad_requests_never_reach_the_ledger(self):
        broken = dict(self.files, **{'training/learner.py': 'def broken(:'})
        for request in ({'op': 'run', 'name': 'a', 'seconds': 120, 'files': broken},
                        {'op': 'run', 'name': 'a', 'seconds': 120, 'files': dict(self.files, extra='x')},
                        {'op': 'run', 'name': 'a', 'seconds': 100000, 'files': self.files},
                        {'op': 'run', 'name': 'a', 'seconds': 120,
                         'files': dict(self.files, **{'configs/train.json': '[1, 2]'})},
                        {'op': 'run', 'name': '../a', 'seconds': 120, 'files': self.files}):
            self.assertIn('error', self.broker.handle(request))
        self.assertEqual(Ledger(self.out / 'research').jobs(), [])
        self.assertEqual(self.provider.payloads, [])

    def test_check_is_free_and_enforces_the_transport_caps(self):
        from simulators.generals.autoresearch.contract import DEFAULT
        self.assertTrue(self.broker.handle({'op': 'check', 'files': self.files})['checked'])
        big = dict(self.files, **{'training/train.py': '#' * (DEFAULT['source_bytes_cap'] + 1)})
        self.assertIn('cap', self.broker.handle({'op': 'check', 'files': big})['error'])
        self.assertEqual(Ledger(self.out / 'research').jobs(), [])

    def test_settle_cancels_queued_work_without_charge(self):
        self.broker.handle({'op': 'run', 'name': 'a', 'seconds': 120, 'files': self.files})
        self.backend.settle()
        status = self.backend.status()
        self.assertEqual([(j['name'], j['status']) for j in status['jobs']], [('a', 'stopped')])
        self.assertEqual(self.backend.budget()['used_usd'], 0)
        self.assertEqual(self.broker.finalize('host')['submission'], None)

    def lease(self, name='gpu1', **shape):
        reply = self.broker.handle(dict({'op': 'lease', 'name': name, 'seconds': 120, 'gpu': 1, 'cpu': 4, 'memory': 16,
                                         'files': self.files}, **shape))
        self.assertNotIn('error', reply, reply)
        self.backend.tick()
        deadline = time.time() + 10
        while not self.provider.lease_live(name) and time.time() < deadline:
            time.sleep(0.01)
        return reply

    def test_a_lease_runs_commands_beside_other_jobs_and_its_network_is_packaged_and_submitted(self):
        broker = self.open('complete', budget_seconds=600)
        reply = self.lease(cpu=8, memory=32)
        expected = rate({'gpu': 1, 'cpu_cores': 8, 'memory_mib': 32768})
        self.assertEqual((reply['usd_per_second'], reply['shape']),
                         (expected, {'gpu': 1, 'cpu_cores': 8, 'memory_mib': 32768}))
        self.assertAlmostEqual(reply['reserved_usd'], 120 * expected)
        self.assertGreater(expected, R)  # more cores and memory than the training worker cost more per second
        self.assertEqual(self.provider.live['gpu1']['files'], self.files)  # the editable files arrive with the lease
        status = broker.handle({'op': 'status'})['jobs'][0]
        self.assertEqual((status['kind'], status['status'], status['ready']), ('lease', 'running', True))

        started = broker.handle({'op': 'exec', 'name': 'gpu1', 'command': 'python training/train.py --out /scratch/a'})
        self.assertEqual(started, {'exec': '0001', 'state': 'running'})
        done = broker.handle({'op': 'poll', 'name': 'gpu1', 'exec': '0001'})
        self.assertEqual((done['state'], done['exit_code']), ('exited', 0))
        edited = dict(self.files, **{'training/learner.py': self.files['training/learner.py'] + '\n# pushed\n'})
        self.assertNotIn('error', broker.handle({'op': 'push', 'name': 'gpu1', 'files': edited}))
        self.assertEqual(self.provider.live['gpu1']['files'], edited)
        self.assertEqual(broker.handle({'op': 'sync', 'name': 'gpu1', 'reload': True})['reloaded'], True)
        self.assertTrue((self.out / 'sources/gpu1/pushes/001/training/learner.py').read_text().endswith('# pushed\n'))

        # A second job runs beside the lease; the session's GPU limit still holds.
        self.assertNotIn('error', broker.handle({'op': 'run', 'name': 'side', 'seconds': 120, 'files': self.files}))
        self.assertIn('At most 2 GPUs', broker.handle({'op': 'run', 'name': 'third', 'seconds': 120,
                                                       'files': self.files})['error'])
        self.assertNotIn('error', broker.handle({'op': 'lease', 'name': 'cpu1', 'seconds': 120, 'gpu': 0, 'cpu': 2,
                                                 'memory': 4, 'files': self.files}))

        packaged = broker.handle({'op': 'package', 'name': 'net1', 'lease': 'gpu1', 'path': '/scratch/a',
                                  'files': edited})
        self.assertNotIn('error', packaged, packaged)
        bundle = self.out / 'research/jobs/net1/bundle'
        self.assertTrue((bundle / 'learner.py').read_text().endswith('# pushed\n'))  # sealed with the source sent
        self.assertIn('already used', broker.handle({'op': 'package', 'name': 'net1', 'lease': 'gpu1',
                                                     'path': '/scratch/a', 'files': edited})['error'])
        self.assertIn('Unknown lease', broker.handle({'op': 'exec', 'name': 'side', 'command': 'ls'})['error'])

        self.assertNotIn('error', broker.handle({'op': 'stop', 'name': 'gpu1'}))
        self.backend.settle()
        jobs = {j['name']: j for j in broker.handle({'op': 'status'})['jobs']}
        self.assertEqual((jobs['gpu1']['status'], jobs['net1']['kind'], jobs['net1']['status']),
                         ('stopped', 'package', 'complete'))
        self.assertAlmostEqual(jobs['gpu1']['charged_usd'], 40 * expected)  # held time x the shape's price
        self.assertEqual((jobs['gpu1']['usd_per_second'], jobs['net1']['charged_usd']), (expected, 0))
        self.assertIn('ended', broker.handle({'op': 'exec', 'name': 'gpu1', 'command': 'ls'})['error'])
        late = broker.handle({'op': 'poll', 'name': 'gpu1', 'exec': '0001'})  # the host's record outlives the lease
        self.assertEqual(late['output'], 'late')
        self.assertEqual(broker.handle({'op': 'log', 'name': 'gpu1'})['commands'][0]['state'], 'ended with the lease')

        # The packaged network is practised and submitted like a trained one.
        self.assertNotIn('error', broker.handle({'op': 'eval', 'name': 'net1', 'seconds': 120}))
        self.settle()
        practice = self.provider.payloads[-1]
        self.assertEqual((practice['kind'], practice['rate'], practice['seeds']['map_indices']), ('practice', R, [0, 1]))
        self.assertEqual(list(broker.handle({'op': 'log', 'name': 'net1'})['practice_runs'].values())[0]['games'], 4)
        submitted = broker.handle({'op': 'submit', 'name': 'net1'})
        self.assertEqual(submitted['submitted']['native']['bundle_sha256'], packaged['bundle_sha256'])
        self.assertTrue(broker.handle({'op': 'finalize'})['finalized'])

    def test_lease_requests_outside_the_limits_never_reach_the_ledger(self):
        broker = self.open(budget_seconds=200)
        base = {'op': 'lease', 'name': 'a', 'seconds': 100, 'gpu': 0, 'cpu': 4, 'memory': 16, 'files': self.files}
        for change, message in (({'gpu': 2}, '--gpu is 0 or 1'), ({'cpu': 33}, 'between 1 and 32'),
                                ({'memory': 1}, 'between 2 and 64'), ({'cpu': 2.5}, 'whole number'),
                                ({'seconds': 10}, 'shorter than minimum'), ({'name': 'Bad_name'}, 'lowercase'),
                                ({'files': {}}, 'editable')):
            reply = broker.handle(dict(base, **change))
            self.assertIn(message, reply.get('error', ''), (change, reply))
        self.assertEqual(self.backend.status()['jobs'], [])
        # A cheap shape stretches the budget: 150 seconds of one core cost less than 20 of the training worker.
        cheap = broker.handle(dict(base, seconds=150, cpu=1, memory=2))
        self.assertNotIn('error', cheap, cheap)
        self.assertLess(cheap['reserved_usd'], 20 * R)
        self.assertIn('exceeds remaining', broker.handle(dict(base, name='b', seconds=150, gpu=1, cpu=32,
                                                              memory=64))['error'])
        self.assertIn('error', broker.handle({'op': 'exec', 'name': 'a', 'command': 'ls'}))  # not started yet
        self.backend.final = True
        self.assertIn('no leases', broker.handle(dict(base, name='c'))['error'])


    def test_seconds_mode_runs_one_job_at_a_time_and_charges_training_worker_seconds(self):
        from harness import budget as budgets
        broker = self.open('complete', budget_seconds=600, mode='seconds')
        budget = broker.handle({'op': 'budget'})
        self.assertEqual((budget['budget_mode'], budget['budget_seconds'], budget['remaining_seconds'],
                          budget['concurrency'], budget['concurrent_gpus']), ('seconds', 600, 600, 1, 1))
        self.assertEqual(budget['seconds_weight']['lease --gpu 1 --cpu 4 --memory 16 (and eval)'], 1.0)
        self.assertLess(budget['seconds_weight']['lease --gpu 0 --cpu 8 --memory 16'], 1.0)
        for key in ('budget_usd', 'used_usd', 'usd_per_hour'):
            self.assertNotIn(key, budget)
        config = Ledger(self.out / 'research').config
        self.assertEqual((config['budget_mode'], config['budget_seconds'], config['max_concurrent_jobs']),
                         ('seconds', 600, 1))
        self.assertEqual(config['prices'], budgets.PRICE)

        reply = broker.handle({'op': 'run', 'name': 'a', 'seconds': 120, 'files': self.files})
        self.assertEqual((reply['seconds_weight'], reply['reserved_seconds']), (1.0, 120), reply)
        self.assertNotIn('reserved_usd', reply)
        # One job at a time: a second run, a practice or a lease waits until the first has ended.
        for request in ({'op': 'run', 'name': 'b', 'seconds': 120, 'files': self.files},
                        {'op': 'lease', 'name': 'c', 'seconds': 120, 'gpu': 0, 'cpu': 1, 'memory': 2,
                         'files': self.files}):
            self.assertIn('One job at a time', broker.handle(request)['error'])
        self.assertAlmostEqual(broker.handle({'op': 'budget'})['used_seconds'], 120)  # the grant is held
        self.settle()
        job = broker.handle({'op': 'status'})['jobs'][0]
        self.assertEqual((job['status'], job['charged_seconds'], job['seconds_weight']), ('complete', 115, 1.0))
        self.assertNotIn('charged_usd', job)
        self.assertAlmostEqual(broker.handle({'op': 'budget'})['used_seconds'], 115)  # its wall seconds

        # A lease of another shape is charged its seconds x its price relative to the training worker's.
        shape = {'gpu': 0, 'cpu_cores': 8, 'memory_mib': 16384}
        weight = budgets.seconds_weight(shape, TRAIN_SHAPE)
        reply = self.lease('cpu1', gpu=0, cpu=8, memory=16)
        self.assertAlmostEqual(reply['seconds_weight'], weight)
        self.assertAlmostEqual(reply['reserved_seconds'], 120 * weight, places=4)
        self.assertIn('One job at a time', broker.handle({'op': 'eval', 'name': 'a', 'seconds': 120})['error'])
        self.assertNotIn('error', broker.handle({'op': 'stop', 'name': 'cpu1'}))
        self.backend.settle()
        lease = next(j for j in broker.handle({'op': 'status'})['jobs'] if j['name'] == 'cpu1')
        self.assertAlmostEqual(lease['charged_seconds'], 40 * weight, places=4)  # 40 held seconds, weighted
        budget = broker.handle({'op': 'budget'})
        self.assertAlmostEqual(budget['used_seconds'], 115 + 40 * weight, places=4)
        self.assertAlmostEqual(budget['remaining_seconds'], 600 - 115 - 40 * weight, places=4)
        # Practice now runs: the lease has ended. A grant past what is left is refused.
        self.assertNotIn('error', broker.handle({'op': 'eval', 'name': 'a', 'seconds': 120}))
        self.settle()
        # 180 s of the largest lease weigh about 500 training-worker seconds, more than the ~450 left.
        self.assertIn('exceeds remaining', broker.handle({'op': 'lease', 'name': 'big', 'seconds': 180, 'gpu': 1,
                                                          'cpu': 32, 'memory': 64, 'files': self.files})['error'])
        self.assertNotIn('error', broker.handle({'op': 'submit', 'name': 'a'}))
        self.assertIn('used_seconds', self.backend.finalize())

    def test_dollar_mode_runs_jobs_side_by_side_up_to_the_limit_and_grants_never_pass_the_budget(self):
        broker = self.open(budget_seconds=600)
        self.assertEqual(Ledger(self.out / 'research').config['budget_mode'], 'usd')
        for name in ('a', 'b'):  # two training runs side by side, each holding its whole grant in dollars
            self.assertNotIn('error', broker.handle({'op': 'run', 'name': name, 'seconds': 120, 'files': self.files}))
        lease = {'op': 'lease', 'seconds': 120, 'gpu': 0, 'cpu': 1, 'memory': 2, 'files': self.files}
        big = broker.handle(dict(lease, name='c', seconds=180, cpu=32, memory=64))
        self.assertIn('exceeds remaining', big['error'])  # it would push the held grants past the budget
        for name in ('c', 'd'):
            self.assertNotIn('error', broker.handle(dict(lease, name=name)))
        self.assertIn('At most 4 jobs', broker.handle(dict(lease, name='e'))['error'])
        budget = broker.handle({'op': 'budget'})
        self.assertEqual((budget['budget_mode'], budget['concurrency']), ('usd', 4))
        held = 240 * R + 240 * rate({'gpu': 0, 'cpu_cores': 1, 'memory_mib': 2048})
        self.assertAlmostEqual(budget['used_usd'], held)
        self.assertLessEqual(budget['used_usd'], budget['budget_usd'])
        self.assertEqual(sorted(broker.handle({'op': 'status'})['active']), ['a', 'b', 'c', 'd'])
        self.backend.tick()  # all four start at once; the stub fails the runs after 30 s and holds the leases
        deadline = time.time() + 10
        while not all(self.provider.lease_live(name) for name in ('c', 'd')) and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(self.provider.payloads), 4)
        for name in ('c', 'd'):
            broker.handle({'op': 'stop', 'name': name})
        self.backend.settle()
        jobs = {j['name']: j for j in broker.handle({'op': 'status'})['jobs']}
        self.assertAlmostEqual(jobs['a']['charged_usd'], 30 * R)
        self.assertAlmostEqual(jobs['c']['charged_usd'], 40 * rate({'gpu': 0, 'cpu_cores': 1, 'memory_mib': 2048}))

    def test_a_session_keeps_the_budget_mode_it_was_opened_in(self):
        from simulators.generals.autoresearch.contract import DEFAULT
        self.open(budget_seconds=600, mode='seconds')
        root = self.out / 'research'
        with self.assertRaisesRegex(ValueError, 'seconds budget mode'):
            Ledger(root, dict(DEFAULT, budget_usd=1.0))
        with self.assertRaisesRegex(ValueError, 'one job at a time'):  # seconds mode is never parallel
            Ledger(Path(self.temp.name) / 'parallel', dict(DEFAULT, budget_mode='seconds', budget_seconds=600))
        self.backend.close()
        backend = registry.backend('generals')
        with self.assertRaisesRegex(ValueError, 'seconds budget mode'):
            backend(self.release, root, smoke=True, resume=True, budget_usd=1.0,
                    options={'_provider': self.provider}).open()
        reopened = backend(self.release, root, smoke=True, resume=True, options={'_provider': self.provider})
        reopened.open()
        self.addCleanup(reopened.close)
        self.assertEqual((reopened.budget_mode, reopened.limits['max_concurrent_jobs']), ('seconds', 1))
        self.assertEqual(reopened.budget()['budget_seconds'], 600)

    def test_both_modes_are_declared_and_seconds_is_the_default(self):
        from harness import budget as budgets
        from simulators.generals.autoresearch.contract import PRICE
        spec = registry.spec('generals')
        self.assertEqual((spec['budget_mode'], spec['research_seconds'], spec['smoke_research_seconds'],
                          spec['research_usd'], spec['smoke_research_usd']), ('seconds', 16200, 360, 10.0, 0.5))
        self.assertEqual(registry.research('generals'), dict(mode='seconds', seconds=16200.0, usd=None))
        self.assertEqual(registry.research('generals', budget_mode='usd'), dict(mode='usd', seconds=None, usd=10.0))
        self.assertEqual((PRICE, rate(TRAIN_SHAPE)), (budgets.PRICE, budgets.rate(TRAIN_SHAPE)))
        for phrase in ('$', 'one run at a time', 'Up to 4 jobs'):
            self.assertNotIn(phrase, spec['hardware'])
        brief = task.brief('generals')
        for phrase in ('$2.90', 'Up to 4 jobs run at once', 'Your allocation is in US dollars'):
            self.assertNotIn(phrase, brief)
        settings = dict(smoke=False, budget_mode='seconds', research_seconds=16200.0, variant='recipe',
                        final_train_seconds=3600.0)
        self.assertTrue(session.official('generals', settings))
        self.assertFalse(session.official('generals', dict(settings, budget_mode='usd', research_seconds=None,
                                                           research_usd=10.0)))

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
    def __init__(self, returncode=0, stdout=''):
        self.returncode = returncode  # None while the command is still running
        self.stdout = io.StringIO(stdout)
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
        self.sandbox.reads.append(path)  # a whole-file read: no size bound and no timeout
        if self.sandbox.gone():
            raise RuntimeError('NotFoundError: Modal Sandbox not found. This Sandbox has already shut down.')
        if path not in self.sandbox.files:
            raise FileNotFoundError(path)
        return self.sandbox.files[path]

    def make_directory(self, path):
        pass

    def copy_from_local(self, local, remote):
        self.sandbox.files[remote] = Path(local).read_bytes()

    def copy_to_local(self, remote, local):
        from simulators.generals.autoresearch.bundle import archive_bundle
        self.sandbox.copies.append(remote)
        cloud = self.sandbox.cloud
        if cloud.transfer_gate is not None:  # a transfer that stalls until the gate opens
            cloud.transfer_gate.wait()
        if cloud.lost_in_transfer:  # the worker reached its own timeout while the bundle was on its way
            raise RuntimeError('NotFoundError: The Sandbox is unavailable. This Sandbox may have already shut down.')
        if remote != '/job/export.tar.gz':
            raise FileNotFoundError(remote)  # the package comes back as one archive, nothing else
        staged = Path(cloud.temp) / (self.sandbox.object_id + '.tar.gz')
        archive_bundle(self.sandbox.package(), staged)
        Path(local).write_bytes(staged.read_bytes())
        self.sandbox.exported = True


class FakeSandbox:
    def __init__(self, cloud, object_id):
        self.cloud, self.object_id = cloud, object_id
        self.files, self.commands, self.export = {}, [], None
        self.reads, self.copies, self.exported = [], [], False
        self.filesystem = FakeFilesystem(self)
        self.created = cloud.clock.now
        self.started_at = self.finished_at = 0.
        self.terminated = False
        self.volumes = {}

    def gone(self):
        """True once the bundle is out and the cloud is set to lose the worker then (its timeout, a provider fault)."""
        return self.cloud.vanish_after_export and self.exported

    def exec(self, *command, **kwargs):
        from harness.games.generals import provider as module
        self.commands.append(command)
        text = ' '.join(command)
        if command == tuple(handoff.prepare_command(module.PYTHON)):
            self.prepared = '/handoff' in self.volumes
            return FakeProcess()
        if command[:3] == (module.PYTHON, '-c', module.FINISH):
            return self.finish(json.loads(command[3]))
        if command[:3] == (module.PYTHON, '-c', handoff.PUBLISH):  # a lease publishing a package it sealed
            return self.publish_package(command[3], dict(item.partition('=')[::2] for item in command[4:]))
        if command[0] == 'tail':  # the provider's bounded log read: tail -c BYTES PATH
            process = FakeProcess()
            if self.cloud.tail_gate is not None and command[-1] == '/job/output/progress.jsonl':
                # A read that stalls, as whole-file reads of a long run's progress did, until its exec timeout.
                self.cloud.stalled += 1
                answered = self.cloud.tail_gate.wait(kwargs.get('timeout'))
                self.cloud.stalled -= 1
                if not answered:
                    process.returncode = 124
                    return process
            if self.gone():
                raise RuntimeError('NotFoundError: Modal Sandbox not found. This Sandbox has already shut down.')
            found = self.files.get(command[-1])
            process.returncode = 0 if found is not None else 1
            process.stdout = io.StringIO((found or '')[-int(command[2]):])
            return process
        if 'train_entry' in text:
            # 'ends': the trainer reaches its own end (after `train_seconds`). 'runs': it trains until something
            # kills it. 'overruns': the worker's supervisor ended it at its command limit and recorded that.
            self.files['/job/output/progress.jsonl'] = '{"update": 1}\n{"update": 2}\n'
            self.files['/job/output/result.json'] = '{"update": 2}\n'
            self.files['/job/trainer-stdout.log'] = 'step 1\nstep 2\n'
            self.files['/job/trainer-stderr.log'] = 'a warning\n'
            self.trainer = FakeProcess(None if self.cloud.trainer == 'runs' or self.cloud.train_seconds else 0)
            if self.cloud.trainer == 'overruns':
                self.files['/job/supervisor.json'] = json.dumps({'status': 'deadline', 'returncode': -15})
            if self.cloud.trainer != 'runs' and self.cloud.train_seconds:
                trainer, end = self.trainer, time.time() + self.cloud.train_seconds

                def poll():
                    if trainer.returncode is None and time.time() >= end:
                        trainer.returncode = 0
                    return trainer.returncode
                trainer.poll = poll
            self.cloud.training.set()
            return self.trainer
        if 'pkill' in text and getattr(self, 'trainer', None) is not None and self.trainer.returncode is None:
            self.trainer.returncode = -9
            self.cloud.killed_at = self.cloud.provider_time()
        if 'export_entry' in text:
            self.cloud.export_while_stalled = self.cloud.stalled > 0
            if not self.cloud.saved:
                return FakeProcess(1)  # nothing valid to seal: the trainer never saved a network
        return FakeProcess()

    def finish(self, spec):
        """The worker's trusted end of a run: export, bounded copies and the real publish into its mounted Volume."""
        from simulators.generals.autoresearch.bundle import archive_bundle, ARCHIVE
        cloud = self.cloud
        cloud.export_while_stalled = cloud.stalled > 0
        staging = Path(cloud.temp) / (self.object_id + '-staging')
        staging.mkdir(exist_ok=True)
        summary = {'export_returncode': 0 if cloud.saved else 1}  # unsaved: nothing valid to seal
        if cloud.saved:
            archive_bundle(self.package(), staging / ARCHIVE)
        for source, name, limit in spec['copies']:
            if source in self.files:
                data = self.files[source]
                (staging / name).write_bytes((data.encode() if isinstance(data, str) else data)[-limit:])
        (staging / 'finish.json').write_text(json.dumps(summary))
        self.files[spec['folder'] + '/finish.json'] = json.dumps(summary)
        self.exported = True
        command = spec['publish']
        job, files = command[3], {}
        for item in command[4:]:
            name, _, path = item.partition('=')
            assert path == ('/job/' + ARCHIVE if name == ARCHIVE else spec['folder'] + '/' + name), item
            if (staging / name).is_file():
                files[name] = staging / name
        reply = dict(summary, published=not cloud.publish_fails, published_files=sorted(files))
        if cloud.publish_fails:  # e.g. the mount was unusable: nothing reaches the Volume
            reply['publish_error'] = 'OSError: Volume unavailable'
        else:
            done = publish(self.volumes['/handoff'].root, job, files)
            assert done.returncode == 0, done.stderr
        process = FakeProcess(0 if reply['published'] else 3, json.dumps(reply) + '\n')
        if cloud.reply_lost:  # the host's link dropped while the worker finished: the reply never arrives
            def lost():
                raise ConnectionError('host link lost')
            process.poll = lost
        return process

    def publish_package(self, job, paths):
        """The lease's root-side publish of the archive its export just sealed."""
        from simulators.generals.autoresearch.bundle import archive_bundle, ARCHIVE
        assert paths == {ARCHIVE: '/job/' + ARCHIVE}, paths
        if self.cloud.publish_fails:
            return FakeProcess(1)
        staged = Path(self.cloud.temp) / (self.object_id + '-' + job + '.tar.gz')
        archive_bundle(self.package(), staged)
        done = publish(self.volumes['/handoff'].root, job, {ARCHIVE: staged})
        assert done.returncode == 0, done.stderr
        return FakeProcess(0, done.stdout)

    def terminate(self, wait=True):
        self.terminated = True
        if self.started_at and not self.finished_at:
            self.finished_at = time.time()

    def package(self):
        """The trainer's sealed package, built from exactly the source the host wrote into the worker."""
        if self.export is None:
            import numpy as np
            from simulators.generals.autoresearch.bundle import seal
            from simulators.generals.autoresearch.contract import EDITABLE
            work = Path(self.cloud.temp) / self.object_id
            (work / 'source').mkdir(parents=True)
            for name in EDITABLE:
                (work / 'source' / name).write_text(self.files['/job/input/' + name])
            (work / 'source/model.json').write_text('{}')
            np.savez(work / 'source/weights.npz', w=np.zeros(4, np.float32))
            seal(work / 'source', work / 'export')
            self.export = work / 'export'
        return self.export


class Cloud:
    """Modal stand-in: created sandboxes start `start_after` queue seconds later (None: never)."""

    def __init__(self, temp, clock, start_after=None):
        self.temp, self.clock, self.start_after = temp, clock, start_after
        self.sandboxes = {}
        self.created = threading.Event()
        self.trainer, self.saved = 'ends', True  # how the trainer command behaves; whether it saved a network
        self.train_seconds = 0  # how long a trainer that ends by itself runs (0: it has already exited)
        self.training = threading.Event()
        self.tail_gate = None  # a threading.Event the progress tail waits on (None: answers at once)
        self.stalled = 0  # progress reads currently stalled
        self.export_while_stalled = None
        self.vanish_after_export = False  # the worker is lost as soon as the bundle is on the host
        self.transfer_gate = None  # a threading.Event the bundle transfer waits on (None: answers at once)
        self.lost_in_transfer = False  # the bundle transfer fails: the worker is already gone
        self.mount = Path(temp) / 'volumes'  # one folder per hand-off Volume name
        self.publish_fails = False  # the worker's publish fails: nothing reaches the Volume
        self.reply_lost = False  # the worker publishes but the host never hears back from the finish exec
        self.volume_failures = 0  # host reads of the Volume that fail (a slow or broken link) before one succeeds
        self.volume_reads = []  # (path, whether every worker had already ended)
        self.provider_time, self.killed_at = time.time, None  # when, on the provider's clock, the trainer was killed

    def add(self, object_id):
        self.sandboxes[object_id] = FakeSandbox(self, object_id)
        return self.sandboxes[object_id]

    def create(self, *command, **kwargs):
        sandbox = self.add('sb-' + str(len(self.sandboxes)))
        sandbox.timeout = kwargs.get('timeout')  # the worker's own lifetime limit
        sandbox.volumes = kwargs.get('volumes') or {}
        self.created.set()
        return sandbox

    def from_id(self, object_id):
        return self.sandboxes[object_id]

    def volume(self, name, environment_name=None, **options):
        """modal.Volume.from_name: a local folder per Volume name; host reads can be made to fail first."""
        assert environment_name == 'game-autoresearch-project'
        cloud = self

        class Volume(FolderVolume):
            def read_file(self, path):
                cloud.volume_reads.append((path, all(s.terminated for s in cloud.sandboxes.values())))
                if cloud.volume_failures:
                    cloud.volume_failures -= 1
                    raise ConnectionError('slow host link')
                return super().read_file(path)
        folder = self.mount / name
        folder.mkdir(parents=True, exist_ok=True)
        return Volume(folder)

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


class GeneralsWorkerQueue(unittest.TestCase):
    """The real provider, controller and ledger with Modal stubbed: a worker queued for a GPU is never charged."""

    def setUp(self):
        from harness.games.generals import provider as module
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.release, self.meta = release_fixture('generals', self.temp.name)
        self.clock = FakeClock()
        self.cloud = Cloud(self.temp.name, self.clock)
        import modal  # the pinned SDK; only its Sandbox entry point is replaced, so nothing reaches Modal
        sandbox = types.SimpleNamespace(create=self.cloud.create, from_id=self.cloud.from_id)
        for patcher in (mock.patch.object(modal, 'Sandbox', sandbox),
                        mock.patch.object(modal, 'Volume', types.SimpleNamespace(from_name=self.cloud.volume)),
                        mock.patch.object(module.ModalProvider, 'COLLECT_WAIT', 0.),
                        mock.patch.object(module.ModalProvider, 'PROBE_WAIT', 0.),
                        mock.patch.object(module, 'allocation_info', self.cloud.info),
                        mock.patch.object(workers, 'await_start', functools.partial(
                            workers.await_start, poll=5., sleep=self.clock.sleep, clock=self.clock.clock)),
                        mock.patch.object(workers, 'abandon', functools.partial(
                            workers.abandon, sleep=self.clock.sleep, clock=self.clock.clock)),
                        mock.patch.object(workers, 'START_PATIENCE_SECONDS', 600.)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.provider = module.ModalProvider.__new__(module.ModalProvider)
        self.provider.__dict__.update(release=Path(self.release), app=types.SimpleNamespace(app_id='app'), image=None,
                                      cash=None, allocation_budget=None, accounting=None,
                                      cpu_inference_profile='default', live={}, lock=threading.Lock())
        self.out = Path(self.temp.name) / 'session'
        self.out.mkdir()
        self.backend = registry.backend('generals')(self.release, self.out / 'research', smoke=True,
                                                    budget_usd=360 * R, options={'_provider': self.provider})
        self.backend.open()
        self.addCleanup(self.backend.close)
        self.broker = host_runs(Broker(self.backend, self.out, 'network', research_usd=360 * R))
        base = workspace.public(self.release) / workspace.WORKSPACE
        self.files = {name: (base / name).read_text() for name in self.meta['writable']}

    def run_job(self, name, wait=True):
        reply = self.broker.handle({'op': 'run', 'name': name, 'seconds': 120, 'config': 'configs/smoke.json',
                                    'files': self.files})
        self.assertNotIn('error', reply, reply)
        self.backend.tick()
        if wait:
            self.backend.controller.join(timeout=60)
            self.assertFalse(self.backend.controller.claimed(name))

    def job(self, name):
        return next(j for j in self.broker.handle({'op': 'status'})['jobs'] if j['name'] == name)

    def recorded_uncertain(self, name):
        """A job an older controller gave up on: its sandbox was created and recorded but never started."""
        ledger = Ledger(self.out / 'research')
        ledger.queue(name, 120, {'kind': 'train'})
        ledger.uncertain(name, 'RuntimeError: Provider did not expose an authoritative GPU allocation start')
        sandbox = self.cloud.add('sb-' + name)
        (self.out / 'research/jobs' / name).mkdir(parents=True)
        (self.out / 'research/jobs' / name / 'provider.json').write_text(
            json.dumps({'app_id': 'app', 'sandbox_id': sandbox.object_id}))
        self.assertEqual(self.broker.handle({'op': 'status'})['active'], [name])
        self.assertEqual(self.broker.handle({'op': 'budget'})['used_usd'], 120 * R)  # its grant stays held
        self.assertIn('error', self.broker.handle({'op': 'finalize'}))
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
        self.assertAlmostEqual(job['charged_usd'], (sandbox.finished_at - sandbox.started_at) * R)
        self.assertLess(job['charged_usd'], 30 * R)
        self.assertTrue(sandbox.terminated)
        self.assertEqual(sandbox.files['/job/input/policy.py'], self.files['training/policy.py'])
        self.assertEqual(self.broker.handle({'op': 'submit', 'name': 'a'})['submitted']['native']['bundle_sha256'],
                         job['artifact_sha'])

    def test_a_never_started_worker_fails_uncharged_and_the_session_continues(self):
        self.run_job('a')
        self.assertGreaterEqual(self.clock.now, 600)
        job = self.job('a')
        self.assertEqual((job['status'], job['charged_usd']), ('failed', 0))
        self.assertIn('never started', job['error'])
        self.assertTrue(self.cloud.sandboxes['sb-0'].terminated)
        self.assertEqual(self.broker.handle({'op': 'status'})['active'], [])
        self.assertEqual(self.broker.handle({'op': 'budget'})['used_usd'], 0)
        log = self.broker.handle({'op': 'log', 'name': 'a'})
        self.assertNotIn('error', log)
        self.assertIn('never started', log['job_error'])
        self.assertEqual((log['host_error']['type'], log['host_error']['charged_usd']), ('NeverStarted', 0))
        events = [row['kind'] for row in Ledger(self.out / 'research').db.execute('SELECT kind FROM events')]
        self.assertIn('abandoned_unstarted', events)
        self.assertNotIn('provider_uncertain', events)
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
            self.assertFalse(self.backend.controller.claimed('a'))
        job = self.job('a')
        self.assertEqual((job['status'], job['charged_usd']), ('stopped', 0))
        row = Ledger(self.out / 'research').jobs()[0]
        self.assertEqual((row['status'], row['provider_id']), ('stopped', 'sb-0'))  # abandoned by its worker, not cancelled
        self.assertIn('Stopped while waiting in the provider queue', row['error'])
        self.assertTrue(self.cloud.sandboxes['sb-0'].terminated)
        self.assertEqual(self.broker.handle({'op': 'budget'})['used_usd'], 0)
        self.assertEqual(self.broker.handle({'op': 'status'})['active'], [])

    def test_recovery_settles_a_recorded_never_started_uncertain_job_at_zero(self):
        sandbox = self.recorded_uncertain('lost')
        self.backend.tick()  # an unresolved job is settled from the provider's record without an operator
        self.backend.controller.join(timeout=60)
        job = self.job('lost')
        self.assertEqual((job['status'], job['charged_usd']), ('failed', 0))
        self.assertIn('never started', job['error'])
        self.assertTrue(sandbox.terminated)
        self.assertEqual(self.broker.handle({'op': 'budget'})['used_usd'], 0)
        self.cloud.start_after = 0
        self.run_job('next')
        self.assertEqual(self.job('next')['status'], 'complete')

    def test_stop_settles_a_recorded_never_started_uncertain_job_and_unblocks_finalize(self):
        sandbox = self.recorded_uncertain('lost')
        reply = self.broker.handle({'op': 'stop', 'name': 'lost'})
        self.assertNotIn('error', reply, reply)
        self.assertEqual((reply['status'], reply['charged_usd']), ('stopped', 0))
        self.assertTrue(sandbox.terminated)
        self.assertEqual(self.broker.handle({'op': 'status'})['active'], [])
        self.assertEqual(self.broker.handle({'op': 'budget'})['used_usd'], 0)
        self.assertEqual(self.broker.finalize('host')['submission'], None)

    def stop_while_training(self, name):
        self.cloud.start_after, self.cloud.trainer = 0, 'runs'
        self.run_job(name, wait=False)
        self.assertTrue(self.cloud.training.wait(30))
        self.assertEqual(self.job(name)['status'], 'running')
        reply = self.broker.handle({'op': 'stop', 'name': name})
        self.assertNotIn('error', reply, reply)
        self.assertIn('stopped_early', reply['note'])
        self.backend.controller.join(timeout=60)
        self.assertFalse(any(thread.is_alive() for thread in self.backend.controller.threads.values()))
        sandbox = self.cloud.sandboxes['sb-0']
        self.assertEqual(sandbox.trainer.returncode, -9)  # the trainer was ended before its output was read
        self.assertTrue(sandbox.terminated)
        return sandbox

    def test_a_stopped_run_keeps_its_last_network_and_is_complete_stopped_early(self):
        sandbox = self.stop_while_training('a')
        job = self.job('a')
        self.assertEqual((job['status'], job.get('stopped_early'), job['error']), ('complete', True, None), job)
        self.assertNotIn('cut_short', job)
        self.assertAlmostEqual(job['charged_usd'], (sandbox.finished_at - sandbox.started_at) * R, places=6)  # charged as run
        log = self.broker.handle({'op': 'log', 'name': 'a'})
        self.assertEqual((log['status'], log.get('stopped_early'), 'error' in log), ('complete', True, False), log)
        allocation = json.loads((self.out / 'research/jobs/a/allocation.json').read_text())
        self.assertEqual((allocation['status'], allocation['stopped_early']), ('complete', True))
        # It can be practised and submitted like any other complete run.
        private = Path(self.release) / 'private'  # the fixture release has no panel; practice needs its list
        private.mkdir(exist_ok=True)
        (private / 'panel.json').write_text(json.dumps({'practice': [{'name': 'random', 'kind': 'random'}]}))
        practice = self.broker.handle({'op': 'eval', 'name': 'a'})
        self.assertNotIn('error', practice, practice)
        self.assertEqual(self.broker.handle({'op': 'stop', 'name': practice['queued']})['stopping'], practice['queued'])
        self.assertEqual(self.broker.handle({'op': 'submit', 'name': 'a'})['submitted']['native']['bundle_sha256'],
                         job['artifact_sha'])
        self.assertTrue(self.broker.handle({'op': 'finalize'})['finalized'])

    def test_a_stopped_run_that_saved_nothing_stays_stopped(self):
        self.cloud.saved = False
        self.stop_while_training('a')
        job = self.job('a')
        self.assertEqual((job['status'], job['artifact_sha']), ('stopped', None), job)
        self.assertNotIn('stopped_early', job)
        self.assertIn('valid policy package', job['error'])
        self.assertGreater(job['charged_usd'], 0)
        self.assertIn('error', self.broker.handle({'op': 'eval', 'name': 'a'}))
        self.assertIn('error', self.broker.handle({'op': 'submit', 'name': 'a'}))

    def test_a_trainer_ended_by_its_supervisor_at_the_limit_is_complete_cut_short(self):
        self.cloud.start_after, self.cloud.trainer = 0, 'overruns'
        self.run_job('a')
        job = self.job('a')
        self.assertEqual((job['status'], job.get('cut_short'), job['error']), ('complete', True, None), job)
        self.assertNotIn('stopped_early', job)
        self.assertEqual(self.broker.handle({'op': 'log', 'name': 'a'}).get('cut_short'), True)
        self.assertNotIn('error', self.broker.handle({'op': 'submit', 'name': 'a'}))

    def cut_off_by_the_host(self):
        from harness.games.generals import provider as module

        class Leap:
            """The provider's clock: each half-second poll of a running trainer moves it 27 s on."""
            offset = 0.

            def time(self):
                return time.time() + self.offset

            def sleep(self, seconds):
                self.offset += 27. if seconds == .5 else 0.
                time.sleep(.001)

        self.cloud.start_after, self.cloud.trainer = 0, 'runs'
        with mock.patch.object(module, 'time', Leap()):
            self.run_job('a')  # nobody stops it: the 120 s grant runs out
        self.assertEqual(self.cloud.sandboxes['sb-0'].trainer.returncode, -9)
        return self.job('a')

    def test_a_trainer_still_running_at_the_time_limit_is_complete_cut_short(self):
        job = self.cut_off_by_the_host()
        self.assertEqual((job['status'], job.get('cut_short'), job['error']), ('complete', True, None), job)
        self.assertNotIn('stopped_early', job)
        self.assertNotIn('error', self.broker.handle({'op': 'submit', 'name': 'a'}))

    def test_a_cut_off_run_that_saved_nothing_fails(self):
        self.cloud.saved = False
        job = self.cut_off_by_the_host()
        self.assertEqual((job['status'], job['artifact_sha']), ('failed', None), job)
        self.assertNotIn('cut_short', job)
        self.assertIn('valid policy package', job['error'])

    def test_a_stalled_progress_read_cannot_delay_the_stop_or_the_export(self):
        # A 900 s final lost its network: the host re-read the whole progress log (megabytes at one row per
        # update) on every pass of the loop that must end the trainer and export before the grant runs out; a
        # read stalled and the loop came back with no time left. The mirror now runs beside that loop and reads
        # a bounded tail with a timeout: a read that never answers costs only the progress copy.
        from harness.games.generals import provider as module
        self.cloud.start_after, self.cloud.train_seconds = 0, 1.5
        self.cloud.tail_gate = threading.Event()
        self.addCleanup(self.cloud.tail_gate.set)
        started = time.monotonic()
        with mock.patch.object(module.ModalProvider, 'PROGRESS_POLL_SECONDS', 0.05), \
                mock.patch.object(module.ModalProvider, 'TAIL_TIMEOUT', 3):
            self.run_job('a')
        job = self.job('a')
        self.assertEqual((job['status'], job['error']), ('complete', None), job)
        self.assertFalse({'stopped_early', 'cut_short'} & set(job), job)
        self.assertTrue(self.cloud.export_while_stalled)  # the export did not wait for the stalled read
        self.assertLess(time.monotonic() - started, 1.5 + 2 * 3)  # one stalled mirror read, one stalled final read
        sandbox = self.cloud.sandboxes['sb-0']
        tails = [c for c in sandbox.commands if c[0] == 'tail']
        self.assertTrue(tails)
        self.assertTrue(all(c[1:3] == ('-c', str(module.ModalProvider.TAIL_BYTES)) for c in tails))
        for path in ('/job/output/progress.jsonl', '/job/output/result.json'):
            self.assertNotIn(path, sandbox.reads)  # never a whole-file read
        self.assertNotIn('error', self.broker.handle({'op': 'submit', 'name': 'a'}))

    def test_a_stop_is_obeyed_while_a_progress_read_is_stalled(self):
        from harness.games.generals import provider as module
        self.cloud.tail_gate = threading.Event()
        self.addCleanup(self.cloud.tail_gate.set)
        with mock.patch.object(module.ModalProvider, 'PROGRESS_POLL_SECONDS', 0.05), \
                mock.patch.object(module.ModalProvider, 'TAIL_TIMEOUT', 3):
            self.cloud.start_after, self.cloud.trainer = 0, 'runs'
            self.run_job('a', wait=False)
            self.assertTrue(self.cloud.training.wait(30))
            deadline = time.monotonic() + 10
            while not self.cloud.stalled and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(self.cloud.stalled)  # the mirror is stuck in a read right now
            self.assertNotIn('error', self.broker.handle({'op': 'stop', 'name': 'a'}))
            self.backend.controller.join(timeout=60)
        job = self.job('a')
        self.assertEqual((job['status'], job.get('stopped_early')), ('complete', True), job)
        self.assertTrue(self.cloud.export_while_stalled)

    def test_the_tail_keeps_whole_lines_within_its_byte_bound(self):
        from harness.games.generals import provider as module
        sandbox = self.cloud.add('sb-tail')
        sandbox.files['/job/output/progress.jsonl'] = ''.join(f'{{"update": {i}}}\n' for i in range(100000))
        with mock.patch.object(module.ModalProvider, 'TAIL_BYTES', 1000):
            text = module.ModalProvider.tail(sandbox, '/job/output/progress.jsonl')
        self.assertLessEqual(len(text), 1000)
        self.assertTrue(text.startswith('{"update": ') and text.endswith('{"update": 99999}\n'))
        self.assertIsNone(module.ModalProvider.tail(sandbox, '/job/missing.log'))
        self.assertEqual(sandbox.commands[-1][0], 'tail')

    def finish_index(self, sandbox):
        from harness.games.generals import provider as module
        return next(i for i, c in enumerate(sandbox.commands) if c[:3] == (module.PYTHON, '-c', module.FINISH))

    def test_progress_result_and_console_come_back_through_the_volume_after_the_worker_ends(self):
        self.cloud.start_after = 0
        self.run_job('a')
        self.assertEqual(self.job('a')['status'], 'complete')
        sandbox, folder = self.cloud.sandboxes['sb-0'], self.out / 'research/jobs/a'
        self.assertEqual((folder / 'progress.jsonl').read_text(), sandbox.files['/job/output/progress.jsonl'])
        self.assertEqual(json.loads((folder / 'result.json').read_text()), {'update': 2})
        self.assertEqual((folder / 'trainer-stdout.log').read_text(), 'step 1\nstep 2\n')
        self.assertEqual((folder / 'trainer-stderr.log').read_text(), 'a warning\n')
        self.assertEqual(json.loads((folder / 'training-lifecycle.json').read_text())['parent_exec_returncode'], 0)
        # Nothing is read out of the worker after its trusted end; the host reads the Volume once it has ended.
        finish = self.finish_index(sandbox)
        self.assertFalse([c for c in sandbox.commands[finish + 1:] if c[0] == 'tail'])
        self.assertEqual(sandbox.copies, [])
        self.assertNotIn('/job/output/progress.jsonl', sandbox.reads)
        self.assertTrue(self.cloud.volume_reads and all(ended for _, ended in self.cloud.volume_reads))
        self.assertEqual(self.job('a')['progress'], {'update': 2})
        allocation = json.loads((folder / 'allocation.json').read_text())
        self.assertEqual(allocation['handoff'], 'volume')
        self.assertNotIn('handoff_fallback', allocation)
        record = json.loads((folder / 'handoff.json').read_text())
        self.assertEqual((record['handoff'], record['unconfirmed_publish'], record['finish']['published']),
                         ('volume', None, True))
        self.assertEqual(set(record['manifest']), {'export.tar.gz', 'finish.json', 'progress.jsonl', 'result.json',
                                                   'trainer-stdout.log', 'trainer-stderr.log'})

    def test_the_training_worker_mounts_its_session_volume_privately_before_researcher_code(self):
        from harness.games.generals import provider as module
        self.cloud.start_after = 0
        self.run_job('a')
        sandbox = self.cloud.sandboxes['sb-0']
        binding = json.loads((self.out / 'research' / handoff.BINDING).read_text())
        self.assertEqual(list(sandbox.volumes), ['/handoff'])  # never /scratch
        self.assertEqual(sandbox.volumes['/handoff'].root.name, binding['volume'])
        self.assertTrue(binding['volume'].startswith('generals-handoff-session-research-'))
        self.assertTrue(sandbox.prepared)
        prepare = sandbox.commands.index(tuple(handoff.prepare_command(module.PYTHON)))
        train = next(i for i, c in enumerate(sandbox.commands) if 'integrations.generals_autoresearch.train_entry' in c)
        self.assertLess(prepare, train)  # root-only before any researcher code runs
        # The trainer's console goes to root-owned files; the supervisor is still the exec's own process.
        self.assertEqual(sandbox.commands[train][:2], ('/bin/sh', '-c'))
        self.assertIn('exec "$@"', sandbox.commands[train][2])
        self.assertLess(train, self.finish_index(sandbox))
        # A fresh final has its own backend root and so its own Volume.
        final = handoff.SessionVolume('generals', self.out / 'final', modal_module=types.SimpleNamespace())
        self.assertNotEqual(final.name, binding['volume'])

    def test_a_failed_publish_falls_back_to_the_live_read_while_the_worker_is_alive(self):
        self.cloud.start_after, self.cloud.publish_fails = 0, True
        self.files['training/policy.py'] += '\n# researcher change\n'
        self.run_job('a')
        job = self.job('a')
        self.assertEqual((job['status'], job['error']), ('complete', None), job)
        sandbox, folder = self.cloud.sandboxes['sb-0'], self.out / 'research/jobs/a'
        self.assertEqual(sandbox.copies, ['/job/export.tar.gz'])  # the old single live transfer
        self.assertEqual((folder / 'bundle/policy.py').read_text(), self.files['training/policy.py'])
        self.assertEqual(json.loads((folder / 'result.json').read_text()), {'update': 2})
        self.assertEqual((folder / 'trainer-stdout.log').read_text(), 'step 1\nstep 2\n')
        finish = self.finish_index(sandbox)
        self.assertTrue([c for c in sandbox.commands[finish + 1:] if c[0] == 'tail'])  # bounded tails after it
        allocation = json.loads((folder / 'allocation.json').read_text())
        self.assertEqual(allocation['handoff'], 'live')
        self.assertIn('Volume unavailable', allocation['handoff_fallback'])
        self.assertEqual(json.loads((folder / 'handoff.json').read_text())['handoff'], 'live')
        self.assertNotIn('error', self.broker.handle({'op': 'submit', 'name': 'a'}))

    def test_a_lost_finish_reply_is_collected_from_the_volume(self):
        self.cloud.start_after, self.cloud.reply_lost = 0, True
        self.run_job('a')
        job = self.job('a')
        self.assertEqual((job['status'], job['error']), ('complete', None), job)
        sandbox, folder = self.cloud.sandboxes['sb-0'], self.out / 'research/jobs/a'
        self.assertEqual(sandbox.copies, [])
        record = json.loads((folder / 'handoff.json').read_text())
        self.assertEqual(record['handoff'], 'volume')
        self.assertIn('host link lost', record['unconfirmed_publish'])
        self.assertNotIn('handoff_fallback', json.loads((folder / 'allocation.json').read_text()))

    def test_a_slow_host_link_after_the_publish_does_not_cost_the_run_or_paid_time(self):
        self.cloud.start_after, self.cloud.volume_failures = 0, 7  # more than one collection attempt fails
        self.run_job('a')
        job = self.job('a')
        self.assertEqual((job['status'], job['error']), ('complete', None), job)
        sandbox = self.cloud.sandboxes['sb-0']
        self.assertTrue(all(ended for _, ended in self.cloud.volume_reads))  # every read after the worker ended
        self.assertAlmostEqual(job['charged_usd'], (sandbox.finished_at - sandbox.started_at) * R)
        self.assertNotIn('error', self.broker.handle({'op': 'submit', 'name': 'a'}))

    def test_a_stopped_run_that_saved_nothing_still_hands_over_its_logs(self):
        self.cloud.saved = False
        self.stop_while_training('a')
        folder = self.out / 'research/jobs/a'
        self.assertEqual(self.job('a')['status'], 'stopped')
        self.assertEqual((folder / 'trainer-stderr.log').read_text(), 'a warning\n')
        record = json.loads((folder / 'handoff.json').read_text())
        self.assertEqual((record['handoff'], record['finish']['export_returncode']), ('volume', 1))
        self.assertNotIn('export.tar.gz', record['manifest'])
        self.assertFalse((folder / 'bundle').exists())

    def test_recovery_collects_a_result_the_worker_published_before_the_controller_was_interrupted(self):
        from simulators.generals.autoresearch.bundle import archive_bundle, ARCHIVE
        from simulators.generals.autoresearch.contract import EDITABLE
        sandbox = self.recorded_uncertain('lost')
        sandbox.started_at = time.time() - 40
        for name in EDITABLE:
            sandbox.files['/job/input/' + name] = self.files['training/' + name]
        staged = Path(self.temp.name) / 'staged'
        staged.mkdir()
        archive_bundle(sandbox.package(), staged / ARCHIVE)
        (staged / 'finish.json').write_text(json.dumps({'export_returncode': 0}))
        (staged / 'supervisor.json').write_text(json.dumps({'status': 'completed', 'returncode': 0}))
        session_volume = handoff.SessionVolume('generals', self.out / 'research')
        done = publish(self.cloud.mount / session_volume.name, 'lost',
                       {n: staged / n for n in (ARCHIVE, 'finish.json', 'supervisor.json')})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.backend.tick()
        self.backend.controller.join(timeout=60)
        job = self.job('lost')
        self.assertTrue(sandbox.terminated)
        self.assertEqual((job['status'], job['error']), ('complete', None), job)
        self.assertGreaterEqual(job['charged_usd'], 40 * R)
        recovery = json.loads((self.out / 'research/jobs/lost/recovery.json').read_text())
        self.assertEqual((recovery['handoff'], recovery['artifact_sha256']), ('volume', job['artifact_sha']))
        self.assertNotIn('error', self.broker.handle({'op': 'submit', 'name': 'lost'}))

    def test_a_worker_lost_right_after_its_publish_still_leaves_a_complete_run(self):
        self.cloud.start_after, self.cloud.vanish_after_export = 0, True
        self.files['training/policy.py'] += '\n# researcher change\n'
        self.run_job('a')
        job = self.job('a')
        self.assertEqual((job['status'], job['error']), ('complete', None), job)
        folder = self.out / 'research/jobs/a'
        self.assertEqual(json.loads((folder / 'result.json').read_text()), {'update': 2})  # from the Volume
        self.assertEqual((folder / 'bundle/policy.py').read_text(), self.files['training/policy.py'])
        self.assertEqual(self.broker.handle({'op': 'submit', 'name': 'a'})['submitted']['native']['bundle_sha256'],
                         job['artifact_sha'])

    def test_the_export_reserve_grows_with_the_grant_inside_the_sandbox_timeout(self):
        from harness.games.generals import provider as module
        reserve = module.ModalProvider.export_reserve
        self.assertEqual([reserve(g) for g in (120, 300, 800, 900, 1200, 3600, 14400)], [40, 40, 40, 45, 60, 60, 60])
        from simulators.generals.autoresearch.contract import DEFAULT, EDITABLE

        class Leap:
            """The provider's clock: each half-second poll of a running trainer moves it `step` seconds on."""
            def __init__(self, step):
                self.step, self.offset = step, 0.

            def time(self):
                return time.time() + self.offset

            def sleep(self, seconds):
                self.offset += self.step if seconds == .5 else 0.
                time.sleep(.001)

        self.cloud.start_after, self.cloud.trainer = 0, 'runs'  # never ends by itself: the host must end it
        code = {name: self.files['training/' + name] for name in EDITABLE}
        # grant -> (seconds the trainer is told at most, host stop point, both from the worker's start)
        for index, (grant, told, stop_at) in enumerate(((120, 80, 83), (300, 260, 263), (900, 855, 858), (3600, 3540, 3543))):
            ledger = Ledger(Path(self.temp.name) / f'long-{grant}', dict(DEFAULT, budget_usd=grant * R))
            ledger.queue('final', grant, {'kind': 'train', 'code': code, 'config': {}, 'seed': 1})
            leap = Leap(grant / 240)
            self.cloud.provider_time, self.cloud.killed_at = leap.time, None
            with mock.patch.object(module, 'time', leap):
                job = self.provider.execute(ledger, ledger.jobs()[0])
            sandbox = self.cloud.sandboxes[f'sb-{index}']
            train = next(c for c in sandbox.commands if 'integrations.generals_autoresearch.train_entry' in c)
            seconds = float(train[train.index('--seconds') + 1])
            self.assertTrue(told - 5 < seconds <= told, (grant, seconds))  # the grant less setup and the reserve
            stopped = self.cloud.killed_at - sandbox.started_at
            self.assertTrue(stop_at <= stopped < stop_at + grant / 240 + 2, (grant, stopped))
            self.assertEqual(sandbox.timeout, grant)  # the worker's own timeout stays the grant: never charged past it
            self.assertGreaterEqual(grant - stopped, reserve(grant) - 3 - grant / 240 - 2)  # slack left for the export
            self.assertEqual((job['status'], job['error']), ('complete', None), (grant, job))
            self.assertEqual(ledger.early(), {'final': 'cut_short'})

    def test_the_bundle_comes_back_as_one_archive_through_the_volume(self):
        # Four of six live 120 s runs lost their network: six copies in a row (the weights alone took 12-29 s
        # on a slow link) did not fit the 15 s reserve before the worker's own timeout. On 2026-11-13 a slow link
        # lost more within the 40 s window. The package now leaves the worker through the session's Volume.
        self.cloud.start_after = 0
        self.files['training/policy.py'] += '\n# researcher change\n'
        self.run_job('a')
        job = self.job('a')
        self.assertEqual((job['status'], job['error']), ('complete', None), job)
        sandbox, folder = self.cloud.sandboxes['sb-0'], self.out / 'research/jobs/a'
        self.assertEqual(sandbox.copies, [])  # nothing streamed out of the running worker
        self.assertEqual(sorted(p.name for p in (folder / 'bundle').iterdir()),
                         ['learner.py', 'manifest.json', 'model.json', 'policy.py', 'train.py', 'weights.npz'])
        self.assertEqual((folder / 'bundle/policy.py').read_text(), self.files['training/policy.py'])
        for leftover in ('export.tar.gz', 'handoff/export.tar.gz', 'bundle.partial'):
            self.assertFalse((folder / leftover).exists(), leftover)
        train = next(c for c in sandbox.commands if 'integrations.generals_autoresearch.train_entry' in c)
        self.assertLessEqual(float(train[train.index('--seconds') + 1]), 120 - 40)  # the 40 s reserve is kept

    def test_a_transfer_is_given_up_on_at_its_client_side_limit(self):
        from harness.games.generals import provider as module
        sandbox = self.cloud.add('sb-slow')
        self.cloud.transfer_gate = threading.Event()
        self.addCleanup(self.cloud.transfer_gate.set)
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, 'did not finish within'):
            module.ModalProvider.fetch(sandbox, '/job/export.tar.gz', Path(self.temp.name) / 'late.tar.gz', 0.2)
        self.assertLess(time.monotonic() - started, 3)

    def lose_the_worker_mid_transfer(self, name):
        # Only the live fallback transfers anything: the worker could not publish, then was lost mid-transfer.
        self.cloud.start_after, self.cloud.lost_in_transfer, self.cloud.publish_fails = 0, True, True
        self.run_job(name)
        self.cloud.lost_in_transfer = self.cloud.publish_fails = False
        return self.job(name)

    def test_a_worker_lost_mid_transfer_leaves_a_failed_run_with_no_partial_bundle(self):
        job = self.lose_the_worker_mid_transfer('a')
        self.assertEqual((job['status'], job['artifact_sha']), ('failed', None), job)
        self.assertIn('Sandbox is unavailable', job['error'])
        self.assertGreater(job['charged_usd'], 0)
        folder = self.out / 'research/jobs/a'
        self.assertFalse((folder / 'bundle').exists() or (folder / 'bundle.partial').exists())
        self.assertIsNone(Ledger(self.out / 'research').jobs()[0]['artifact'])
        self.assertEqual(json.loads((folder / 'allocation.json').read_text())['handoff'], 'live')

    def test_practice_and_submit_on_a_run_without_a_policy_are_refused_not_host_faults(self):
        # Live: `ar eval` on a run that failed mid-export answered with a host fault (FileNotFoundError on the
        # missing manifest) instead of telling the researcher the run has no policy.
        self.lose_the_worker_mid_transfer('a')
        for op in ('eval', 'submit'):
            reply = self.broker.handle({'op': op, 'name': 'a'})
            self.assertEqual(reply.get('error'), 'Run a did not finish with a policy: status failed', (op, reply))
        self.assertEqual(self.broker.handle({'op': 'eval', 'name': 'zzz'})['error'], 'Unknown run: zzz')
        # An older ledger row that kept the path of a bundle directory which was never completed.
        ledger = Ledger(self.out / 'research')
        ledger.db.execute("UPDATE jobs SET artifact=? WHERE name='a'", (str(self.out / 'research/jobs/a/bundle'),))
        self.assertEqual(self.broker.handle({'op': 'eval', 'name': 'a'})['error'],
                         'Run a did not finish with a policy: status failed')
        # A run that is still training is not a policy yet.
        self.cloud.trainer = 'runs'
        self.cloud.training.clear()
        self.run_job('b', wait=False)
        self.assertTrue(self.cloud.training.wait(30))
        for op in ('eval', 'submit'):
            self.assertEqual(self.broker.handle({'op': op, 'name': 'b'})['error'], 'Run b has not finished: status running')
        self.assertNotIn('error', self.broker.handle({'op': 'stop', 'name': 'b'}))
        self.backend.controller.join(timeout=60)
        self.assertEqual(self.job('b')['status'], 'complete')
        # A complete run whose package has since gone missing is refused the same plain way.
        (self.out / 'research/jobs/b/bundle/manifest.json').unlink()
        for op in ('eval', 'submit'):
            reply = self.broker.handle({'op': op, 'name': 'b'})
            self.assertIn('policy package of run b is missing or no longer valid', reply['error'], (op, reply))
        requests = [json.loads(line) for line in (self.out / 'requests.jsonl').read_text().splitlines()]
        self.assertFalse([r for r in requests if 'Host fault' in (r.get('error') or '')])

    def test_the_shortest_run_is_two_minutes(self):
        from simulators.generals.autoresearch.contract import DEFAULT
        self.assertEqual((self.broker.handle({'op': 'budget'})['min_run_seconds'], DEFAULT['min_run_seconds']), (120, 120))
        reply = self.broker.handle({'op': 'run', 'name': 'short', 'seconds': 60, 'config': 'configs/smoke.json',
                                    'files': self.files})
        self.assertIn('shorter than minimum', reply['error'])
        self.assertEqual(Ledger(self.out / 'research').jobs(), [])

    def test_a_run_that_reaches_its_own_end_carries_no_flag(self):
        self.cloud.start_after = 0
        self.run_job('a')
        job = self.job('a')
        self.assertEqual(job['status'], 'complete')
        self.assertFalse({'stopped_early', 'cut_short'} & set(job), job)

    def test_closing_the_session_ends_a_training_worker_at_once(self):
        self.cloud.start_after, self.cloud.trainer = 0, 'runs'
        self.run_job('a', wait=False)
        self.assertTrue(self.cloud.training.wait(30))
        started = time.monotonic()
        self.backend.settle()
        self.assertLess(time.monotonic() - started, 10)  # no waiting for the trainer's own end or its grant
        sandbox = self.cloud.sandboxes['sb-0']
        self.assertEqual(sandbox.trainer.returncode, -9)
        self.assertTrue(sandbox.terminated)
        status = self.broker.handle({'op': 'status'})
        self.assertEqual(status['active'], [])
        job = status['jobs'][0]
        self.assertEqual((job['status'], job.get('stopped_early')), ('complete', True), job)  # its last network is kept
        self.assertLess(job['charged_usd'], 10 * R)

    def test_recovery_terminates_an_abandoned_started_worker_now_and_charges_its_lease(self):
        sandbox = self.recorded_uncertain('lost')
        sandbox.started_at = time.time() - 40  # it started late, after the controller had given up on it
        self.backend.tick()
        self.backend.controller.join(timeout=60)
        job = self.job('lost')
        self.assertTrue(sandbox.terminated)  # not left to idle until its grant deadline
        self.assertEqual(job['status'], 'failed')
        self.assertAlmostEqual(job['charged_usd'], (sandbox.finished_at - sandbox.started_at) * R)
        self.assertGreaterEqual(job['charged_usd'], 40 * R)
        self.assertEqual(self.broker.handle({'op': 'status'})['active'], [])


    def test_a_lease_on_the_real_provider_runs_commands_packages_a_network_and_is_charged_at_its_rate(self):
        from harness.games.generals import provider as module
        self.cloud.start_after = 0
        self.provider.scratch = lambda root: 'scratch-volume'  # no Modal Volume is looked up
        shape = {'gpu': 0, 'cpu_cores': 8, 'memory_mib': 16384}
        reply = self.broker.handle({'op': 'lease', 'name': 'cpu1', 'seconds': 120, 'gpu': 0, 'cpu': 8, 'memory': 16,
                                    'files': self.files})
        self.assertEqual((reply.get('shape'), reply.get('usd_per_second')), (shape, rate(shape)), reply)
        self.backend.tick()
        deadline = time.time() + 20
        while not self.provider.lease_live('cpu1') and time.time() < deadline:
            time.sleep(0.01)
        self.assertTrue(self.job('cpu1')['ready'])
        sandbox = self.cloud.sandboxes['sb-0']
        self.assertEqual(sandbox.files['/work/training/train.py'], self.files['training/train.py'])

        hostile = 'echo hi; chmod 777 /job; $(id)'
        started = self.broker.handle({'op': 'exec', 'name': 'cpu1', 'command': hostile})
        self.assertEqual(started['exec'], '0001', started)
        shell, flag, script, argument = sandbox.commands[-1]
        self.assertEqual((shell, flag, argument), ('/bin/sh', '-c', hostile))  # an argument, never part of the root script
        self.assertNotIn('chmod 777', script)
        self.assertIn('setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs', script)
        self.assertIn('bash -c "$0"', script)
        self.assertIn('OMP_NUM_THREADS=8', script)
        self.assertIn('JAX_PLATFORMS=cpu', script)  # a lease without a GPU keeps JAX on the CPU
        self.assertIn('PYTHONPATH=/opt/gr:/opt/gr/candidates/generals-bots:/work/training', script)
        done = self.broker.handle({'op': 'poll', 'name': 'cpu1', 'exec': '0001'})
        self.assertEqual((done['state'], done['exit_code']), ('exited', 0))
        self.assertNotIn('error', self.broker.handle({'op': 'sync', 'name': 'cpu1'}))

        edited = dict(self.files, **{'training/policy.py': self.files['training/policy.py'] + '\n# packaged\n'})
        packaged = self.broker.handle({'op': 'package', 'name': 'net1', 'lease': 'cpu1',
                                       'path': "/scratch/run 1'; rm -rf /x", 'files': edited})
        self.assertNotIn('error', packaged, packaged)
        copy = next(c[2] for c in sandbox.commands if 'head -c' in c[2])
        self.assertIn("setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs head -c 134217729 -- "
                      "'/scratch/run 1'\"'\"'; rm -rf /x'/weights.npz > /job/output/weights.npz", copy)
        self.assertTrue(any('integrations.generals_autoresearch.export_entry' in c[2] for c in sandbox.commands))
        bundle = self.out / 'research/jobs/net1/bundle'
        self.assertTrue((bundle / 'policy.py').read_text().endswith('# packaged\n'))  # sealed with the source sent
        # The lease mounts the hand-off Volume root-only beside /scratch and hands the package over through it.
        self.assertEqual(set(sandbox.volumes), {'/scratch', '/handoff'})
        self.assertTrue(sandbox.prepared)
        prepare = sandbox.commands.index(tuple(handoff.prepare_command(module.PYTHON)))
        self.assertLess(prepare, sandbox.commands.index(next(c for c in sandbox.commands if 'setpriv' in ' '.join(c))))
        self.assertEqual(sandbox.copies, [])
        record = json.loads((self.out / 'research/jobs/net1/handoff.json').read_text())
        self.assertEqual((record['handoff'], record['unconfirmed_publish']), ('volume', None))
        self.assertTrue(record['job'].startswith('net1-'))
        self.assertFalse((self.out / 'research/jobs/net1/handoff/export.tar.gz').exists())
        self.assertIn('absolute directory', self.broker.handle({'op': 'package', 'name': 'net2', 'lease': 'cpu1',
                                                                'path': 'relative/dir', 'files': edited})['error'])
        self.assertTrue(list((self.out / 'research/jobs').glob('net2.failed-*')))  # the failed attempt is kept

        self.assertNotIn('error', self.broker.handle({'op': 'stop', 'name': 'cpu1'}))
        self.backend.controller.join(timeout=60)
        job = self.job('cpu1')
        self.assertEqual(job['status'], 'stopped')
        self.assertAlmostEqual(job['charged_usd'], (sandbox.finished_at - sandbox.started_at) * rate(shape))
        self.assertTrue(sandbox.terminated)
        scripts = [c[2] for c in sandbox.commands if c[0] == '/bin/sh']
        self.assertEqual(scripts[-1], 'sync /scratch')  # the last thing a lease does is commit its storage
        self.assertIn('pkill -KILL -u 65534 || true', scripts)
        saved = json.loads((self.out / 'research/jobs/cpu1/exec/0001.json').read_text())
        self.assertEqual((saved['command'], saved['state']), (hostile, 'exited'))  # kept for the run archive
        self.assertEqual(self.job('net1')['kind'], 'package')
        self.assertNotIn('error', self.broker.handle({'op': 'submit', 'name': 'net1'}))
        self.assertTrue(self.broker.handle({'op': 'finalize'})['finalized'])


    def test_a_lease_that_cannot_publish_a_package_hands_it_over_live(self):
        self.cloud.start_after, self.cloud.publish_fails = 0, True
        self.provider.scratch = lambda root: 'scratch-volume'  # no Modal Volume is looked up
        reply = self.broker.handle({'op': 'lease', 'name': 'cpu1', 'seconds': 120, 'gpu': 0, 'cpu': 8, 'memory': 16,
                                    'files': self.files})
        self.assertNotIn('error', reply, reply)
        self.backend.tick()
        deadline = time.time() + 20
        while not self.provider.lease_live('cpu1') and time.time() < deadline:
            time.sleep(0.01)
        packaged = self.broker.handle({'op': 'package', 'name': 'net1', 'lease': 'cpu1', 'path': '/scratch/run1',
                                       'files': self.files})
        self.assertNotIn('error', packaged, packaged)
        sandbox, folder = self.cloud.sandboxes['sb-0'], self.out / 'research/jobs/net1'
        self.assertEqual(sandbox.copies, ['/job/export.tar.gz'])  # the old live copy
        record = json.loads((folder / 'handoff.json').read_text())
        self.assertEqual((record['handoff'], record['unconfirmed_publish']),
                         ('live', 'The lease did not confirm its publish'))
        self.assertEqual(verify_bundle(folder / 'bundle'), packaged['bundle_sha256'])
        self.assertNotIn('error', self.broker.handle({'op': 'stop', 'name': 'cpu1'}))
        self.backend.controller.join(timeout=60)


def verify_bundle(folder):
    from simulators.generals.autoresearch.bundle import verify
    return verify(folder)['bundle_sha256']


class GeneralsLeaseSession(unittest.TestCase):
    """A network session that works only through leases, with the real `ar`, broker, controller and ledger."""

    def test_network_session_packages_and_submits_a_network_from_a_lease(self):
        budget = self.session()  # the default: seconds mode
        self.assertEqual((budget['budget_mode'], budget['budget_seconds'], budget['concurrency']), ('seconds', 360, 1))
        self.assertNotIn('budget_usd', budget)

    def test_the_same_session_in_dollar_mode(self):
        budget = self.session(research_usd=0.5)
        self.assertEqual((budget['budget_mode'], budget['budget_usd'], budget['concurrency']), ('usd', 0.5, 4))
        self.assertNotIn('budget_seconds', budget)

    def session(self, **amount):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        release, meta = release_fixture('generals', temp.name, bind=False)
        private(release)
        policy.bind_release('generals', release)
        providers = []

        class Offline(registry.backend('generals')):
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
            seen['lease'] = ar('lease', 'cpu1', '--seconds', '120', '--gpu', '0', '--cpu', '8', '--memory', '16')
            seen['ready'] = ar('wait', 'cpu1', '--started', '--timeout', '30')
            seen['exec'] = ar('exec', 'cpu1', '--wait', '10', '--', 'python', 'training/train.py', '--out', '/scratch/a b')
            seen['shell'] = ar('exec', 'cpu1', '--', 'cd /scratch && ls')
            ar('push', 'cpu1')
            ar('sync', 'cpu1')
            seen['package'] = ar('package', 'cpu1', 'net1', '--path', '/scratch/a', '--config', 'configs/smoke.json')
            ar('stop', 'cpu1')
            ar('wait', 'cpu1')
            ar('eval', 'net1', '--seconds', '120')
            ar('wait')
            seen['status'] = ar('status')
            ar('submit', 'net1')
            ar('finalize')
        result = session.run('generals', 'network', release, out, model='test/model', smoke=True, **amount,
                             launcher=lambda folder: Process(lambda: script(Researcher(folder / 'task', box))),
                             find_box=lambda: LocalBox(box / 'ar') if (box / 'ar').is_dir() else None)
        self.assertEqual((result['status'], result['error'], result['closed_by']), ('researched', None, 'researcher'))
        self.assertEqual(seen['budget']['lease']['scratch'], '/scratch')
        self.assertEqual(seen['lease']['shape'], {'gpu': 0, 'cpu_cores': 8, 'memory_mib': 16384})
        self.assertTrue(any(job['name'] == 'cpu1' and job['ready'] for job in seen['ready']['jobs']))
        # Separate words are quoted back into one command; a single string is passed to the shell as written.
        self.assertEqual(seen['exec']['output'], "ran python training/train.py --out '/scratch/a b'")
        self.assertEqual((seen['shell']['state'], seen['shell']['output']), ('exited', 'ran cd /scratch && ls'))
        self.assertEqual([(j['name'], j['kind'], j['status']) for j in seen['status']['jobs'][:2]],
                         [('cpu1', 'lease', 'stopped'), ('net1', 'package', 'complete')])
        self.assertEqual(seen['status']['jobs'][2]['evaluates'], 'net1')
        ledger = Ledger(out / 'research').public()
        self.assertEqual([s['artifact_sha'] for s in ledger['submissions']], [seen['package']['bundle_sha256']])
        self.assertTrue((out / 'sources/net1/training/policy.py').is_file())  # the packaged source is kept
        mode = seen['budget']['budget_mode']
        self.assertEqual((json.loads((out / 'plan.json').read_text())['budget_mode'], ledger['budget_mode'],
                          ledger['config']['budget_mode']), (mode, mode, mode))
        return seen['budget']


class GeneralsConstructor(unittest.TestCase):
    """The real constructor, offline: sealed release -> harness binding -> Harbor task -> worker tree."""

    @classmethod
    def setUpClass(cls):
        from simulators.generals.autoresearch.contract import frozen_manifest
        try:
            frozen_manifest()
        except Exception as exc:  # a fresh clone before tools/restore_upstream_metadata.py
            raise unittest.SkipTest(f'Pinned Generals engine metadata unavailable: {exc}')
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.release = Path(cls.temp.name) / 'release'
        # Two private maps and a scripted-only panel keep this independent of evidence/ checkpoints.
        cls.manifest = registry.constructor('generals')(cls.release, include_existing=False, private_maps=2)
        cls.binding = policy.bind_release('generals', cls.release)

    def test_release_is_bound_and_carries_nothing_for_older_harnesses(self):
        from simulators.generals.autoresearch.prepare import audit_release
        self.assertEqual(policy.verify_release('generals', self.release)['sha256'], self.binding['sha256'])
        self.assertEqual(audit_release(self.release)['hidden_files_staged'], 0)
        self.assertNotIn('task_files', self.manifest)
        self.assertEqual(sorted(p.name for p in self.release.iterdir()), ['frozen', 'private', 'public', 'release.json'])
        public = self.release / 'public'
        for legacy in ('ar.py', 'INSTRUCTIONS.md', 'BRIEF.md', 'NOTES.md', 'rules_reader.py', 'rules_manifest.json'):
            self.assertFalse((public / legacy).exists(), legacy)
        base = public / workspace.WORKSPACE
        self.assertEqual({p.name for p in base.iterdir()}, set(workspace.TOP))
        for name in ('policy.py', 'learner.py', 'train.py'):
            self.assertEqual((base / 'training' / name).read_bytes(), (public / 'agent' / name).read_bytes())
        self.assertEqual((base / 'configs/train.json').read_bytes(), (public / 'recipes/starter.json').read_bytes())
        panel = json.loads((self.release / 'private/panel.json').read_text())
        self.assertEqual([p['name'] for p in panel['official']], ['random', 'expansion', 'hunter'])
        self.assertFalse(self.manifest['grading_ready'])  # fewer private maps than the contract's schedule

    def test_both_variants_build_a_task_with_the_game_brief(self):
        research, final_train = registry.budgets('generals')  # seconds mode is the default
        self.assertEqual((research, registry.research_usd('generals')), (16200.0, None))
        self.assertEqual(registry.research_usd('generals', budget_mode='usd'), 10.0)
        for mode, amount, allocation, jobs in (
                ('seconds', dict(research_seconds=research), 'Research allocation: 270 minutes of allocated worker time',
                 'one at a time'),
                ('usd', dict(research_seconds=None, research_usd=10.0),
                 'Research allocation: $10.00 of allocated worker time', 'up to 4 at once')):
            for variant in VARIANTS:
                out = Path(self.temp.name) / ('task-' + variant + '-' + mode)
                result = task.build('generals', variant, self.release, out, final_train_seconds=final_train,
                                    wall_seconds=18000, hardware=registry.spec('generals')['hardware'],
                                    budget_mode=mode, **amount)
                self.assertEqual(result['workspace_sha256'], self.binding['workspace_sha256'])
                text = (out / 'instruction.md').read_text()
                self.assertIn(task.brief('generals'), text)
                self.assertIn(allocation, text)
                self.assertIn('- Jobs: ' + jobs, text)
                self.assertEqual({p.name for p in (out / 'environment/workspace').iterdir()}, set(workspace.TOP))

    def test_worker_tree_runs_every_module_the_provider_names(self):
        frozen = (self.release / 'frozen').resolve()
        package = 'integrations.generals_autoresearch.'
        source = ((Path(__file__).resolve().parents[1] / 'games/generals/provider.py').read_text()
                  + (frozen / 'integrations/generals_autoresearch/arena.py').read_text())
        modules = sorted(set(re.findall(re.escape(package) + r'(\w+)', source)))
        self.assertTrue({'train_entry', 'export_entry', 'evaluate_entry', 'policy_worker'} <= set(modules), modules)
        # Only the staged tree is importable: a module missing from CONTROL fails here, not on a paid worker.
        code = ('import importlib,sys\n'
                f'for name in {modules!r}+["rollout","starter.policy"]:\n'
                f'    module=importlib.import_module({package!r}+name)\n'
                f'    assert module.__file__.startswith({str(frozen)!r}),module.__file__\n'
                'import integrations.generals.environment as game\n'
                f'assert game.__file__.startswith({str(frozen)!r})\n'
                'assert not [n for n in sys.modules if n.split(".")[0] in ("harness","simulators","tools")]\n')
        env = {'PATH': os.environ.get('PATH', ''), 'JAX_PLATFORMS': 'cpu',
               'PYTHONPATH': f'{frozen}{os.pathsep}{frozen / "candidates/generals-bots"}'}
        done = subprocess.run([sys.executable, '-c', code], cwd=frozen, env=env, capture_output=True, text=True,
                              timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])


if __name__ == '__main__':
    unittest.main()
