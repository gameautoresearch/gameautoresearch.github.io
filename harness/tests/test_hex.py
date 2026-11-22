"""Hex backend against its real controller and ledger; the provider is stubbed before any allocation.

Run with the game's research interpreter: .venv-hex/bin/python -m unittest harness.tests.test_hex
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from unittest import mock

from harness import budget as budgets, handoff, policy, registry, session, workspace
from harness.backend import STATUSES
from harness.broker import Broker
from harness.transport import LocalBox
from harness.tests.support import ROOT, Process, Researcher, host_runs, release_fixture
from harness.tests.test_handoff import FakeModal, FolderVolume, publish

try:
    from simulators.hex.autoresearch.bundle import FILES, verify
    from simulators.hex.autoresearch.contract import (DEFAULT, PANEL, PRACTICE_SHAPE, PRICE, SECONDS_RULE, TRAIN_SHAPE, digest,
                                                      frozen_manifest, rate, sha, write_json)
    from simulators.hex.autoresearch.ledger import Ledger
    from simulators.hex.autoresearch import ledger as hex_ledger
except ImportError as exc:  # wrong interpreter
    raise unittest.SkipTest(f'Hex research dependencies unavailable: {exc}')

PRACTICE = [dict(p) for p in PANEL]
T = rate(TRAIN_SHAPE)  # dollars per second of the training worker


def private(release):
    """The private half of a release that practice and grading read."""
    folder = Path(release) / 'private'
    folder.mkdir()
    write_json(folder / 'panel.json', {'practice': PRACTICE, 'official': PRACTICE})
    write_json(folder / 'grade-seeds.json', {'split': 'heldout', 'opening_seed': 7, 'policy_seed': 11, 'openings': 64})


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


CODE = {'policy.py': 'p = 1\n', 'learner.py': 'l = 1\n', 'train.py': 't = 1\n'}


def sealed_archive(folder, code=CODE):
    """The worker's /job/export.tar for a sealed package, and its identity."""
    identity = sealed(Path(folder) / 'sealed', code)
    with tarfile.open(Path(folder) / 'export.tar', 'w') as archive:
        for name in (*FILES, 'manifest.json'):
            archive.add(Path(folder) / 'sealed' / name, arcname=name)
    return Path(folder) / 'export.tar', identity


def volume_for(root, mount):
    """A backend root's hand-off Volume, kept in a local folder."""
    return handoff.SessionVolume('hex', root, modal_module=FakeModal(mount))


class Provider:
    """Records the exact payload and settles the allocation as a provider would."""

    def __init__(self, outcome='failed'):
        self.outcome = outcome
        self.payloads = []
        self.evaluations = []
        self.entered = threading.Event()
        self.gate = None
        self.live = {}

    def handoff_volume(self, root):  # recovery reads it; an empty one unless a test publishes into it
        return volume_for(root, Path(root) / '.handoff-volume')

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
        self.entered.set()
        if self.gate is not None:  # a worker that has been claimed but whose allocation has not started yet
            self.gate.wait(10)
        now = time.time()
        ledger.start(job['name'], 'stub', now)
        if payload['kind'] == 'lease':  # held until stopped; the stub charges 40 worker seconds
            self.live[job['name']] = {'files': dict(payload['files']), 'commands': []}
            while not (stop and stop()):
                time.sleep(0.01)
            del self.live[job['name']]
            write_json(folder / 'exec/0001.json', {'exec': '0001', 'state': 'ended with the lease', 'output': 'late'})
            ledger.finish(job['name'], now + 40, status='stopped')
        elif stop and stop():
            ledger.finish(job['name'], now + 7, status='failed', error='stub provider: stopped before any export')
        elif self.outcome == 'failed':
            ledger.finish(job['name'], now + 30, status='failed', error='stub provider: no worker was allocated')
        elif payload['kind'] == 'practice':
            write_json(folder / 'practice.json', {'status': 'development_only', 'raw_score': 41.5, 'games': 16,
                                                  'per_opponent': {'random': 1.0}, 'rows_hidden': True,
                                                  'anchor': {'elo_vs_anchor': -812.0}})
            ledger.finish(job['name'], now + 17)
        else:
            (folder / 'progress.jsonl').write_text('{"wave": 1}\nnot json\n{"wave": 2, "entropy": 3.1}\n')
            (folder / 'trainer.log').write_text('\n'.join(f'line {i}' for i in range(200)))
            identity = sealed(folder / 'bundle', payload['code'])
            ledger.finish(job['name'], now + job['grant'] - 5, 'complete', folder / 'bundle', identity)

    def evaluate(self, bundle, panel, seeds, development=True):
        self.evaluations.append(dict(bundle=str(bundle), panel=panel, seeds=seeds, development=development))
        return {'public': {'status': 'development_only' if development else 'complete', 'raw_score': 55.0},
                'private': {'rows': []}, 'allocation': {'gpu_count': 0}}


class HexBackend(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.release, self.meta = release_fixture('hex', self.temp.name, bind=False)
        private(self.release)
        policy.bind_release('hex', self.release)
        self.out = Path(self.temp.name) / 'session'
        self.out.mkdir()
        base = workspace.public(self.release) / workspace.WORKSPACE
        self.files = {name: (base / name).read_text() for name in self.meta['writable']}

    def open(self, outcome='failed', variant='network', root='research', budget_seconds=600, smoke=True, mode='usd',
             **kwargs):
        self.provider = Provider(outcome)
        # Unless a test asks for seconds mode, research gets an explicit dollar budget sized in seconds of the
        # training worker (T dollars each). Seconds mode, and fresh final training, are granted in seconds (None:
        # the game's default).
        budget = (dict(budget_seconds=budget_seconds) if root == 'final' or mode == 'seconds'
                  else dict(budget_usd=budget_seconds * T))
        self.backend = registry.backend('hex')(self.release, self.out / root, smoke=smoke,
                                               options={'_provider': self.provider}, **budget, **kwargs)
        self.backend.open()
        self.addCleanup(self.backend.close)
        self.broker = host_runs(Broker(self.backend, self.out, variant, research_usd=budget.get('budget_usd'),
                                       research_seconds=budget.get('budget_seconds')))
        return self.broker

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
        spec = registry.spec('hex')
        self.assertEqual((spec['research_usd'], spec['final_train_seconds'], spec['smoke_research_usd']),
                         (10.0, 3600, 0.5))
        # Both modes: seconds (the default, the v2 contract's 7,200 seconds) and dollars (v3's $10).
        self.assertEqual((spec['budget_mode'], spec['research_seconds'], spec['smoke_research_seconds']),
                         ('seconds', 7200, 600))
        self.assertEqual(registry.research('hex'), dict(mode='seconds', seconds=7200.0, usd=None))
        self.assertEqual(registry.research('hex', budget_mode='usd'), dict(mode='usd', seconds=None, usd=10.0))
        self.assertEqual(registry.research('hex', smoke=True, research_usd=0.5)['mode'], 'usd')
        self.assertEqual(session.wall_limit('hex', research_seconds=7200, budget_mode='seconds'), 18000)
        self.assertEqual(session.wall_limit('hex', budget_mode='usd'), 18000)
        for text in ('one at a time', 'Up to 4', '$10', 'jobs at once'):
            self.assertNotIn(text, spec['hardware'])  # the instructions state the mode's concurrency
        self.assertEqual(registry.constructor('hex').__module__, 'simulators.hex.autoresearch.prepare')
        self.assertIn('L40S', spec['hardware'])
        json.dumps(spec)
        code = ("import sys, harness.games.hex as g; "
                "bad = [m for m in ('modal', 'torch', 'numpy', 'simulators.hex') if m in sys.modules]; "
                "print(bad); sys.exit(1 if bad else 0)")
        done = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_workspace_shows_the_engine_but_neither_mohex_nor_private_material(self):
        base = workspace.public(self.release) / workspace.WORKSPACE
        names = set(self.meta['files'])
        self.assertIn('simulator/simulators/hex/engine.py', names)
        self.assertIn('simulator/simulators/__init__.py', names)
        self.assertIn('simulator/simulators/hex/heuristic.py', names)
        self.assertFalse(any('benzene' in n or 'mohex' in n.lower() for n in names))  # held out: no source to build
        self.assertEqual({n for n in names if n.startswith('training/')},
                         {'training/policy.py', 'training/learner.py', 'training/train.py'})
        self.assertFalse(any('autoresearch' in n or 'grade-seeds' in n or n.endswith('.pt') for n in names))
        self.assertTrue((base / 'game_rules/evaluation.md').is_file())

    def test_the_researcher_has_no_run_command_and_the_host_still_trains(self):
        self.open()
        request = {'op': 'run', 'name': 'a', 'seconds': 60, 'config': 'configs/smoke.json', 'files': self.files}
        with self.assertRaisesRegex(ValueError, 'no `ar run`.*training/launch.py'):
            self.broker.op_run(dict(request))
        self.assertEqual(self.backend.status()['jobs'], [])
        self.assertNotIn('error', self.broker.train(dict(request)))  # the host's own path, as for a recipe's final

    def test_run_carries_exactly_the_researchers_files_and_is_charged(self):
        broker = self.open()
        budget = broker.handle({'op': 'budget'})
        self.assertAlmostEqual(budget['budget_usd'], 600 * T)
        self.assertAlmostEqual(budget['remaining_usd'], 600 * T)
        self.assertAlmostEqual(budget['allocatable_usd'], 580 * T)  # the smoke contract holds 20 seconds back for teardown
        self.assertNotIn('budget_seconds', budget)
        self.assertEqual(budget['usd_per_hour']['lease --gpu 1 --cpu 4 --memory 16'], round(3600 * T, 4))
        self.assertEqual(budget['affordable_seconds']['lease --gpu 1 --cpu 4 --memory 16'], 580)
        self.assertEqual(budget['practice_opponents'], [p['name'] for p in PANEL])
        self.assertEqual(budget['default_practice_seconds'], 300)  # smoke practice fits a smoke allowance
        self.files['training/policy.py'] += '\n# researcher change\n'
        self.files['configs/smoke.json'] = json.dumps({'architecture': {'hidden': 64}, 'researcher': True})
        reply = broker.handle({'op': 'run', 'name': 'a', 'seconds': 100, 'seed': 7, 'config': 'configs/smoke.json',
                               'files': self.files})
        self.assertNotIn('error', reply, reply)
        self.assertAlmostEqual(reply['reserved_usd'], 100 * T)
        self.assertEqual(broker.handle({'op': 'status'})['active'], ['a'])
        self.assertAlmostEqual(broker.handle({'op': 'budget'})['used_usd'], 100 * T)  # queued work is reserved
        self.work()
        payload = self.provider.payloads[0]
        self.assertEqual(payload['code'], {name.split('/')[1]: text for name, text in self.files.items()
                                           if name.endswith('.py')})
        self.assertEqual(payload['config'], {'architecture': {'hidden': 64}, 'researcher': True})
        self.assertEqual((payload['kind'], payload['seed']), ('train', 7))
        self.assertEqual(set(payload), {'kind', 'code', 'config', 'seed', 'shape', 'rate'})
        self.assertEqual((payload['shape'], payload['rate']), ({'gpu': 1, 'cpu_cores': 4, 'memory_mib': 16384}, T))
        status = broker.handle({'op': 'status'})
        self.assertEqual([(j['name'], j['kind'], j['status']) for j in status['jobs']], [('a', 'train', 'failed')])
        budget = broker.handle({'op': 'budget'})
        self.assertAlmostEqual(budget['used_usd'], 30 * T, places=6)  # failed allocations stay charged
        self.assertIn('error', broker.handle({'op': 'submit', 'name': 'a'}))  # no trained artifact
        self.assertIn('error', broker.handle({'op': 'eval', 'name': 'a'}))
        log = broker.handle({'op': 'log', 'name': 'a'})
        self.assertEqual((log['status'], log['job_error']), ('failed', 'stub provider: no worker was allocated'))
        self.assertNotIn('error', log)

    def test_bad_requests_never_reach_the_ledger(self):
        broker = self.open()
        broken = dict(self.files, **{'training/learner.py': 'def broken(:'})
        for request in ({'op': 'run', 'name': 'a', 'seconds': 120, 'files': broken},
                        {'op': 'run', 'name': 'a', 'seconds': 120, 'files': dict(self.files, extra='x')},
                        {'op': 'run', 'name': 'a', 'seconds': 100000, 'files': self.files},  # over the allowance
                        {'op': 'run', 'name': 'a', 'seconds': 400, 'files': self.files},  # over the smoke per-run cap
                        {'op': 'run', 'name': 'a', 'seconds': 10, 'files': self.files},  # under the minimum run
                        {'op': 'run', 'name': '../a', 'seconds': 120, 'files': self.files},
                        {'op': 'run', 'name': 'a', 'seconds': 120,
                         'files': dict(self.files, **{'configs/train.json': '[1, 2]'})},
                        {'op': 'run', 'name': 'a', 'seconds': 120,
                         'files': dict(self.files, **{'training/policy.py': '#' * (DEFAULT['source_bytes_cap'] + 1)})},
                        {'op': 'eval', 'name': 'a'}, {'op': 'stop', 'name': 'a'}, {'op': 'submit', 'name': 'a'},
                        {'op': 'grade'}):
            self.assertIn('error', broker.handle(request), request)
        self.assertEqual(Ledger(self.out / 'research').jobs(), [])
        self.assertEqual(self.provider.payloads, [])
        self.assertEqual(broker.handle({'op': 'budget'})['used_usd'], 0)

    def test_check_is_free_and_enforces_the_transport_caps(self):
        broker = self.open()
        reply = broker.handle({'op': 'check', 'files': self.files})
        self.assertTrue(reply['checked'])
        big = dict(self.files, **{'training/train.py': '#' * (DEFAULT['source_bytes_cap'] + 1)})
        self.assertIn('cap', broker.handle({'op': 'check', 'files': big})['error'])
        listed = dict(self.files, **{'configs/smoke.json': '[]'})
        self.assertIn('JSON object', broker.handle({'op': 'check', 'config': 'configs/smoke.json', 'files': listed})['error'])
        self.assertEqual(Ledger(self.out / 'research').jobs(), [])

    def test_train_practice_submit_and_host_only_grade(self):
        broker = self.open('complete')
        self.run_job('a', 120)
        reply = broker.handle({'op': 'eval', 'name': 'a', 'seconds': 60})
        self.assertNotIn('error', reply, reply)
        practice_rate = rate(PRACTICE_SHAPE)  # an 8-CPU worker second costs about half a training worker second
        self.assertAlmostEqual(practice_rate / T, 0.52, places=2)
        self.assertEqual((reply['worker_seconds'], reply['usd_per_second']), (60, practice_rate))
        self.assertAlmostEqual(reply['reserved_usd'], 60 * practice_rate)
        self.work()
        practice = self.provider.payloads[1]
        self.assertEqual((practice['kind'], practice['panel'], practice['seeds']['openings'], practice['seeds']['split']),
                         ('practice', PRACTICE, 2, 'practice'))  # public practice seeds and panel, smoke size
        status = broker.handle({'op': 'status'})
        self.assertEqual([(j['kind'], j['status'], j.get('evaluates')) for j in status['jobs']],
                         [('train', 'complete', None), ('practice', 'complete', 'a')])
        self.assertTrue(all(j['status'] in STATUSES for j in status['jobs']))
        self.assertEqual(status['jobs'][0]['progress'], {'wave': 2, 'entropy': 3.1})
        self.assertEqual(status['jobs'][1]['practice'], {'status': 'development_only', 'raw_score': 41.5, 'games': 16,
                                                         'per_opponent': {'random': 1.0}, 'elo_vs_anchor': -812.0})
        log = broker.handle({'op': 'log', 'name': 'a', 'tail': 3})
        self.assertEqual(log['trainer_log'], 'line 197\nline 198\nline 199')
        self.assertEqual(list(log['practice_runs'].values())[0]['raw_score'], 41.5)
        self.assertNotIn('payload', json.dumps(log))
        budget = broker.handle({'op': 'budget'})
        self.assertAlmostEqual(budget['used_usd'], 115 * T + 17 * practice_rate, places=6)
        submitted = broker.handle({'op': 'submit', 'name': 'a'})['submitted']
        self.assertEqual(submitted['native']['bundle_sha256'], status['jobs'][0]['artifact_sha'])
        self.assertTrue(broker.handle({'op': 'finalize'})['finalized'])
        self.assertEqual(self.provider.evaluations, [])  # nothing the researcher did reached private grading
        self.backend.close()

        with self.assertRaises(ValueError):  # an existing session is never reopened with a new allowance
            registry.backend('hex')(self.release, self.out / 'research', smoke=True, budget_usd=99.0)
        grader = Provider()
        reopened = registry.backend('hex')(self.release, self.out / 'research', smoke=True, resume=True,
                                           options={'_provider': grader})
        with reopened:
            self.assertAlmostEqual(reopened.budget()['used_usd'], 115 * T + 17 * practice_rate, places=6)
            grade = reopened.grade()
        self.assertEqual((grade['development'], grade['private_openings'], grade['final']['raw_score']), (True, 4, 55.0))
        self.assertEqual(grade['final']['artifact_sha256'], submitted['native']['bundle_sha256'])
        self.assertEqual(len(grader.evaluations), 1)
        self.assertEqual((grader.evaluations[0]['seeds']['openings'], grader.evaluations[0]['panel']), (4, PRACTICE))
        seeds = json.loads((self.release / 'private/grade-seeds.json').read_text())
        self.assertEqual(seeds['openings'], 64)  # a development prefix never rewrites the release

    def test_an_official_session_is_graded_on_the_full_private_schedule(self):
        self.open('complete', budget_seconds=None, smoke=False, mode='seconds')  # the default mode and allocation
        self.assertEqual((self.backend.budget()['budget_seconds'], self.backend.budget()['teardown_reserve_seconds']),
                         (7200, 0))
        self.assertEqual(self.backend.budget()['budget_mode'], 'seconds')
        self.assertEqual(self.backend.budget()['default_practice_seconds'], 1800)
        self.run_job('a', 120, config='configs/train.json')
        self.broker.handle({'op': 'submit', 'name': 'a'})
        self.broker.handle({'op': 'finalize'})
        self.backend.official = True  # the host marks a session run at the declared budgets
        grade = self.backend.grade()
        self.assertEqual((grade['development'], grade['private_openings']), (False, 64))
        self.assertEqual([(e['development'], e['seeds']['openings']) for e in self.provider.evaluations], [(False, 64)])
        self.assertEqual(grade['final']['status'], 'complete')

    def test_fresh_final_training_fits_the_smoke_allowance(self):
        seconds = registry.budgets('hex', smoke=True)[1]
        self.open('complete', variant='recipe', root='final', budget_seconds=seconds)
        workspace.apply(self.release, self.backend.workspace, self.files)
        reply = self.backend.run('final', seconds, 'configs/smoke.json', 1)
        self.assertAlmostEqual(reply['worker_seconds'], seconds - 20)  # the teardown reserve is never granted
        self.work()
        self.assertEqual([(j['name'], j['status']) for j in self.backend.status()['jobs']], [('final', 'complete')])
        self.assertEqual(self.backend.submit('final')['submission'], 1)
        budget = self.backend.budget()  # one run of the allowance, whatever the research mode was
        self.assertEqual((budget['budget_mode'], budget['budget_seconds'], budget['concurrency']), ('seconds', seconds, 1))
        self.assertLessEqual(budget['used_seconds'], seconds)
        self.assertNotIn('budget_usd', budget)

    def test_seconds_mode_runs_one_job_at_a_time_charged_in_training_worker_seconds(self):
        broker = self.open('complete', mode='seconds')
        budget = broker.handle({'op': 'budget'})
        self.assertEqual((budget['budget_mode'], budget['budget_seconds'], budget['remaining_seconds'],
                          budget['allocatable_seconds']), ('seconds', 600, 600, 580))
        self.assertEqual((budget['concurrency'], budget['concurrent_gpus']), (1, 1))
        self.assertFalse({'budget_usd', 'used_usd', 'remaining_usd', 'usd_per_hour'} & set(budget))
        weight = rate(PRACTICE_SHAPE) / T
        self.assertEqual(budget['seconds_weight']['lease --gpu 1 --cpu 4 --memory 16'], 1.0)
        self.assertEqual(budget['seconds_weight']['eval (8 CPU, 16 GiB)'], round(weight, 4))
        self.assertEqual(budget['affordable_seconds']['lease --gpu 1 --cpu 4 --memory 16'], 580)
        self.assertEqual(PRICE, budgets.PRICE)  # the weights are the core's: price relative to the training worker
        self.assertAlmostEqual(weight, budgets.seconds_weight(PRACTICE_SHAPE, TRAIN_SHAPE))
        config = self.backend.controller.ledger.config  # the ledger records the mode, its limits and its unit
        self.assertEqual((config['budget_mode'], config['budget_seconds'], config['max_concurrent_jobs'],
                          config['allocation_rule']), ('seconds', 600, 1, SECONDS_RULE))
        self.assertNotIn('budget_usd', config)

        reply = broker.handle({'op': 'run', 'name': 'a', 'seconds': 120, 'files': self.files})
        self.assertEqual((reply['reserved_seconds'], reply['seconds_weight']), (120, 1.0))
        self.assertNotIn('reserved_usd', reply)
        # A second job of any kind waits for the first to end: a run, a lease and (once there is a network) practice.
        self.assertIn('One job at a time', broker.handle({'op': 'run', 'name': 'b', 'seconds': 60,
                                                          'files': self.files})['error'])
        self.assertIn('One job at a time', broker.handle({'op': 'lease', 'name': 'c', 'seconds': 60, 'gpu': 0, 'cpu': 1,
                                                          'memory': 2, 'files': self.files})['error'])
        self.assertEqual(broker.handle({'op': 'budget'})['used_seconds'], 120)  # queued work is reserved
        self.work()
        reply = broker.handle({'op': 'eval', 'name': 'a', 'seconds': 60})
        self.assertAlmostEqual(reply['reserved_seconds'], 60 * weight)
        self.assertIn('One job at a time', broker.handle({'op': 'run', 'name': 'b', 'seconds': 60,
                                                          'files': self.files})['error'])
        self.work()
        jobs = {j['name']: j for j in broker.handle({'op': 'status'})['jobs']}
        practice = next(j for j in jobs.values() if j['kind'] == 'practice')
        self.assertEqual((jobs['a']['charged_seconds'], jobs['a']['seconds_weight']), (115, 1.0))
        self.assertAlmostEqual(practice['charged_seconds'], 17 * weight)  # its worker seconds x its weight
        self.assertAlmostEqual(broker.handle({'op': 'budget'})['used_seconds'], 115 + 17 * weight)

        # A lease is the one job while it is held; stopping it frees the session for the next.
        shape = {'gpu': 1, 'cpu_cores': 8, 'memory_mib': 32768}
        reply = self.lease(cpu=8, memory=32)
        self.assertAlmostEqual(reply['seconds_weight'], rate(shape) / T)
        self.assertIn('One job at a time', broker.handle({'op': 'run', 'name': 'b', 'seconds': 60,
                                                          'files': self.files})['error'])
        self.assertNotIn('error', broker.handle({'op': 'stop', 'name': 'gpu1'}))
        self.backend.settle()
        jobs = {j['name']: j for j in broker.handle({'op': 'status'})['jobs']}
        self.assertAlmostEqual(jobs['gpu1']['charged_seconds'], 40 * rate(shape) / T)
        left = broker.handle({'op': 'budget'})['allocatable_seconds']
        big = {'gpu': 1, 'cpu_cores': 16, 'memory_mib': 32768}  # 290 worker seconds of it cost more than is left
        self.assertGreater(290 * rate(big) / T, left)
        self.assertIn('exceeds remaining', broker.handle({'op': 'lease', 'name': 'big', 'seconds': 290, 'gpu': 1,
                                                          'cpu': 16, 'memory': 32, 'files': self.files})['error'])
        self.assertNotIn('error', broker.handle({'op': 'run', 'name': 'b', 'seconds': 60, 'files': self.files}))
        self.work()
        jobs = broker.handle({'op': 'status'})['jobs']
        self.assertAlmostEqual(self.backend.budget()['used_seconds'], sum(j['charged_seconds'] for j in jobs))
        self.assertLessEqual(self.backend.budget()['used_seconds'], 600)

    def test_dollar_mode_runs_jobs_in_parallel_and_its_grants_never_exceed_the_budget(self):
        broker = self.open('complete', budget_seconds=400)
        budget = broker.handle({'op': 'budget'})
        self.assertEqual((budget['budget_mode'], budget['concurrency'], budget['concurrent_gpus'],
                          budget['concurrent_cpu_cores']), ('usd', 4, 2, 64))
        self.assertFalse({'budget_seconds', 'used_seconds', 'seconds_weight'} & set(budget))
        config = self.backend.controller.ledger.config
        self.assertEqual((config['budget_mode'], config['price'], config['allocation_rule']),
                         ('usd', PRICE, DEFAULT['allocation_rule']))
        self.assertEqual(budgets.rate(TRAIN_SHAPE), T)
        held = 0.
        for name, cpu in (('gpu1', 4), ('gpu2', 4)):
            held += self.lease(name, cpu=cpu)['reserved_usd']
        self.assertTrue(self.provider.lease_live('gpu1') and self.provider.lease_live('gpu2'))  # side by side
        self.assertIn('At most 2 GPUs', broker.handle({'op': 'run', 'name': 'r', 'seconds': 60,
                                                       'files': self.files})['error'])
        cpu = {'op': 'lease', 'seconds': 60, 'gpu': 0, 'cpu': 1, 'memory': 2, 'files': self.files}
        # Each running job holds its whole grant, so a grant beyond what is left is refused.
        left = budget['allocatable_usd'] - held
        wide = {'gpu': 0, 'cpu_cores': 32, 'memory_mib': 65536}
        self.assertGreater(100 * rate(wide), left)
        self.assertIn('exceeds remaining', broker.handle(dict(cpu, name='c0', seconds=100, cpu=32, memory=64))['error'])
        held += self.lease('c1', gpu=0, cpu=1, memory=2, seconds=60)['reserved_usd']
        held += self.lease('c2', gpu=0, cpu=1, memory=2, seconds=60)['reserved_usd']
        self.assertIn('At most 4 jobs', broker.handle(dict(cpu, name='c3'))['error'])
        self.assertEqual(len(self.provider.live), 4)
        budget = broker.handle({'op': 'budget'})
        self.assertAlmostEqual(budget['used_usd'], held)
        self.assertLessEqual(budget['used_usd'], budget['budget_usd'])
        for name in ('gpu1', 'gpu2', 'c1', 'c2'):
            broker.handle({'op': 'stop', 'name': name})
        self.backend.settle()
        jobs = broker.handle({'op': 'status'})['jobs']
        self.assertEqual({j['status'] for j in jobs}, {'stopped'})
        self.assertAlmostEqual(self.backend.budget()['used_usd'], sum(40 * j['usd_per_second'] for j in jobs))

    def test_a_session_never_continues_in_another_budget_mode(self):
        self.open(mode='seconds')
        self.backend.close()
        with self.assertRaisesRegex(ValueError, 'opened in seconds budget mode'):
            registry.backend('hex')(self.release, self.out / 'research', smoke=True, resume=True, budget_usd=1.0,
                                    options={'_provider': Provider()}).open()
        with self.assertRaisesRegex(ValueError, 'opened in seconds budget mode'):
            Ledger(self.out / 'research', dict(DEFAULT, budget_usd=1.0))
        reopened = registry.backend('hex')(self.release, self.out / 'research', smoke=True, resume=True,
                                           options={'_provider': Provider()})
        with reopened:  # the ledger's mode rules a reopened session
            self.assertEqual((reopened.budget_mode, reopened.limits['max_concurrent_jobs']), ('seconds', 1))
            self.assertEqual(reopened.budget()['budget_seconds'], 600)

    def lease(self, name='gpu1', **shape):
        reply = self.broker.handle(dict({'op': 'lease', 'name': name, 'seconds': 100, 'gpu': 1, 'cpu': 4, 'memory': 16,
                                         'files': self.files}, **shape))
        self.assertNotIn('error', reply, reply)
        self.backend.tick()
        deadline = time.time() + 10
        while not self.provider.lease_live(name) and time.time() < deadline:
            time.sleep(0.01)
        return reply

    def test_a_lease_runs_commands_beside_other_jobs_and_its_network_is_packaged_and_submitted(self):
        broker = self.open('complete')
        reply = self.lease(cpu=8, memory=32)
        expected = rate({'gpu': 1, 'cpu_cores': 8, 'memory_mib': 32768})
        self.assertEqual((reply['usd_per_second'], reply['shape']), (expected, {'gpu': 1, 'cpu_cores': 8, 'memory_mib': 32768}))
        self.assertGreater(expected, T)  # more cores and memory than the training worker cost more per second
        # Running: its whole grant is held until it ends, so parallel grants cannot add up to more than the budget.
        self.assertAlmostEqual(broker.handle({'op': 'budget'})['used_usd'], reply['reserved_usd'])
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
        self.assertNotIn('error', broker.handle({'op': 'run', 'name': 'side', 'seconds': 60, 'files': self.files}))
        self.assertIn('At most 2 GPUs', broker.handle({'op': 'run', 'name': 'third', 'seconds': 60,
                                                       'files': self.files})['error'])
        self.assertNotIn('error', broker.handle({'op': 'lease', 'name': 'cpu1', 'seconds': 60, 'gpu': 0, 'cpu': 2,
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
        self.assertAlmostEqual(jobs['gpu1']['charged_usd'], 40 * expected, places=6)  # held time x the shape's rate
        self.assertEqual(jobs['net1']['charged_usd'], 0)
        self.assertIn('ended', broker.handle({'op': 'exec', 'name': 'gpu1', 'command': 'ls'})['error'])
        late = broker.handle({'op': 'poll', 'name': 'gpu1', 'exec': '0001'})  # the host's record outlives the lease
        self.assertEqual(late['output'], 'late')
        self.assertEqual(broker.handle({'op': 'log', 'name': 'gpu1'})['commands'][0]['state'], 'ended with the lease')
        submitted = broker.handle({'op': 'submit', 'name': 'net1'})
        self.assertEqual(submitted['submitted']['native']['bundle_sha256'], packaged['bundle_sha256'])

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
        # A cheap shape stretches the allowance: 250 worker seconds of one core fit a 200-second budget.
        cheap = broker.handle(dict(base, seconds=250, cpu=1, memory=2))
        self.assertNotIn('error', cheap, cheap)
        self.assertLess(cheap['reserved_usd'], 20 * T)
        self.assertIn('exceeds remaining', broker.handle(dict(base, name='b', seconds=250, gpu=1))['error'])
        long = 'x' * 70000
        self.assertIn('error', broker.handle({'op': 'exec', 'name': 'a', 'command': long}))  # not started yet
        self.backend.final = True
        self.assertIn('no leases', broker.handle(dict(base, name='c'))['error'])

    def test_settle_cancels_queued_work_without_charge(self):
        broker = self.open()
        broker.handle({'op': 'run', 'name': 'a', 'seconds': 120, 'files': self.files})
        self.backend.settle()
        self.assertEqual([(j['name'], j['status']) for j in self.backend.status()['jobs']], [('a', 'stopped')])
        self.assertEqual(self.backend.budget()['used_usd'], 0)
        self.assertEqual(self.provider.payloads, [])
        self.backend.settle()  # idempotent

    def test_settle_stops_a_claimed_worker_through_the_provider_and_charges_it(self):
        broker = self.open()
        self.provider.gate = threading.Event()
        broker.handle({'op': 'run', 'name': 'a', 'seconds': 120, 'files': self.files})
        self.backend.tick()
        self.assertTrue(self.provider.entered.wait(10))
        threading.Timer(0.2, self.provider.gate.set).start()
        self.backend.settle()
        job = self.backend.status()['jobs'][0]
        self.assertEqual((job['status'], job['error']), ('failed', 'stub provider: stopped before any export'))
        self.assertAlmostEqual(self.backend.budget()['used_usd'], 7 * T, places=6)

    def test_controller_takes_only_the_source_whitelist_and_has_no_grading_action(self):
        self.open()
        controller = self.backend.controller
        for request in ({'action': 'run', 'name': 'bad', 'seconds': 60, 'code': {'engine.py': 'bad'}, 'config': {}},
                        {'action': 'run', 'name': 'bad', 'seconds': 60, 'code': {'policy.py': 'x = 1'}, 'config': {}},
                        {'action': 'eval', 'name': 'arbitrary', 'seconds': 60},
                        {'action': 'grade'}, {'action': 'rules'}):
            with self.assertRaises(ValueError, msg=request):
                controller.handle(request)
        self.assertEqual(controller.ledger.jobs(), [])

    def test_reopening_settles_an_interrupted_lease_only_from_provider_receipts(self):
        from harness.games.hex import provider as module

        class Recovering(Provider):
            recover = module.ModalProvider.recover  # the real recovery; only provider metadata is patched
            recover_jobs = module.ModalProvider.recover_jobs

        self.open()
        ledger = Ledger(self.out / 'research')
        ledger.queue('lost', 100, {'kind': 'train'})
        ledger.start('lost', 'sandbox', 200)  # the host died while this worker was allocated
        write_json(self.out / 'research/jobs/lost/provider.json', {'app_id': 'app', 'sandbox_id': 'sandbox'})

        def reopen(receipt):
            backend = registry.backend('hex')(self.release, self.out / 'research', smoke=True, resume=True,
                                              options={'_provider': Recovering()})
            with mock.patch.object(module, 'allocation_info', lambda *args: receipt):
                backend.open()
            self.addCleanup(backend.close)
            return backend
        held = reopen(None)
        self.assertEqual([(j['status'], j['charged_usd']) for j in held.status()['jobs']], [('uncertain', 0)])
        self.assertAlmostEqual(held.budget()['used_usd'], 100 * T)
        settled = reopen({'started_at': 200, 'finished_at': 245})
        self.assertEqual([j['status'] for j in settled.status()['jobs']], ['failed'])
        self.assertAlmostEqual(settled.status()['jobs'][0]['charged_usd'], 45 * T)
        self.assertAlmostEqual(settled.budget()['used_usd'], 45 * T)  # the exact provider lease, never a refund

    def test_reopening_settles_a_run_its_worker_published_with_that_result(self):
        from harness.games.hex import provider as module

        class Recovering(Provider):
            recover = module.ModalProvider.recover
            recover_jobs = module.ModalProvider.recover_jobs

        self.open()
        root = self.out / 'research'
        ledger = Ledger(root)
        for name in ('lost', 'tampered'):
            ledger.queue(name, 100, {'kind': 'train'})
        for name in ('lost', 'tampered'):
            ledger.start(name, 'sandbox', 200)  # the host died while these workers were allocated
            write_json(root / 'jobs' / name / 'provider.json', {'app_id': 'app', 'sandbox_id': 'sandbox'})
        ledger.uncertain('tampered', 'RuntimeError: the host lost the worker')
        archive, identity = sealed_archive(self.out)
        mount = root / '.handoff-volume'
        (self.out / 'trainer.log').write_text('trained\n')
        for name in ('lost', 'tampered'):  # each worker published before it ended
            done = publish(mount, name + '.sandbox', {'bundle.tar': archive, 'trainer.log': self.out / 'trainer.log'})
            self.assertEqual(done.returncode, 0, done.stderr)
        (mount / 'tampered.sandbox/files/bundle.tar').write_bytes(b'not the published bytes')
        backend = registry.backend('hex')(self.release, root, smoke=True, resume=True, options={'_provider': Recovering()})
        with mock.patch.object(module, 'allocation_info', lambda *args: {'started_at': 200, 'finished_at': 245}):
            backend.open()
        self.addCleanup(backend.close)
        jobs = {j['name']: j for j in backend.status()['jobs']}
        rate_usd = rate(TRAIN_SHAPE)
        self.assertEqual((jobs['lost']['status'], jobs['lost']['charged_usd'], jobs['lost']['artifact_sha']),
                         ('complete', 45 * rate_usd, identity))  # the finished run is kept, charged its exact lease
        self.assertEqual((root / 'jobs/lost/trainer.log').read_text(), 'trained\n')
        self.assertEqual((jobs['tampered']['status'], jobs['tampered']['charged_usd']), ('failed', 45 * rate_usd))
        self.assertIn('differs from its manifest', jobs['tampered']['error'])  # verified as strictly as ever
        self.assertEqual(backend.submit('lost')['bundle_sha256'], identity)

    def test_reopening_settles_a_sandbox_modal_never_started_at_zero_charge(self):
        from harness.games.hex import provider as module

        class Recovering(Provider):
            recover = module.ModalProvider.recover
            recover_jobs = module.ModalProvider.recover_jobs

        self.open()
        ledger = Ledger(self.out / 'research')
        ledger.queue('pending', 100, {'kind': 'train'})  # the host died while Modal had not placed the task
        write_json(self.out / 'research/jobs/pending/provider.json', {'app_id': 'app', 'sandbox_id': 'sandbox'})
        terminated = []
        backend = registry.backend('hex')(self.release, self.out / 'research', smoke=True, resume=True,
                                          options={'_provider': Recovering()})
        with self.modal_records(module, terminated):
            backend.open()
        self.addCleanup(backend.close)
        self.assertEqual(terminated, ['sandbox'])  # it can no longer start behind the ledger's back
        self.assertEqual([(j['status'], j['charged_usd']) for j in backend.status()['jobs']], [('failed', 0)])
        self.assertEqual(backend.budget()['used_usd'], 0)

    @staticmethod
    def modal_records(module, terminated, started_at=0.0, finished_at=0.0):
        """Modal's sandbox list for one recorded sandbox; terminating it is recorded, not sent."""
        from contextlib import ExitStack
        record = {'provider_id': 'sandbox', 'task_id': 'ta-1' if started_at else '', 'created_at': 150.0,
                  'started_at': started_at, 'finished_at': finished_at, 'gpu_type': 'L40S' if started_at else ''}
        stack = ExitStack()
        stack.enter_context(mock.patch.object(module, 'allocation_info', lambda *args: record))
        stack.enter_context(mock.patch.object(module.ModalProvider, 'terminate', staticmethod(terminated.append)))
        stack.enter_context(mock.patch.object(module, 'time', Clock()))  # abandon's settle polls use no real time
        return stack

    def stuck(self):
        """A session whose only job was left `uncertain` with a recorded sandbox Modal never started."""
        from harness.games.hex import provider as module

        class Recovering(Provider):
            recover = module.ModalProvider.recover

        self.open()
        self.backend.controller.provider = Recovering()
        ledger = self.backend.controller.ledger
        ledger.queue('stuck', 100, {'kind': 'train'})
        write_json(self.out / 'research/jobs/stuck/provider.json', {'app_id': 'app', 'sandbox_id': 'sandbox'})
        ledger.uncertain('stuck', 'RuntimeError: Provider did not expose an authoritative allocation start')
        with self.assertRaises(ValueError):  # its grant stays held against the budget until it is settled
            ledger.queue('next', 550, {'kind': 'train'})
        return module

    def test_stop_settles_an_uncertain_never_started_job_and_unblocks_the_session(self):
        module = self.stuck()
        terminated = []
        with self.modal_records(module, terminated):
            reply = self.backend.stop('stuck')
        self.assertEqual(terminated, ['sandbox'])
        self.assertEqual((reply['status'], reply['charged_usd']), ('stopped', 0))
        self.assertEqual(self.backend.budget()['used_usd'], 0)
        self.backend.controller.ledger.queue('next', 550, {'kind': 'train'})

    def test_tick_settles_an_uncertain_job_without_an_operator(self):
        module = self.stuck()
        terminated = []
        controller = self.backend.controller
        with self.modal_records(module, terminated):
            self.backend.tick()
            controller.join(10)
            self.backend.tick()  # nothing left to recover, and no job is queued
        self.assertEqual(controller.threads, {})
        job = next(j for j in controller.ledger.jobs() if j['name'] == 'stuck')
        self.assertEqual((job['status'], job['charged']), ('failed', 0))
        self.assertIn('never started', job['error'])

    def test_tick_charges_a_worker_that_did_run_from_its_receipt(self):
        module = self.stuck()
        terminated = []
        controller = self.backend.controller
        with self.modal_records(module, terminated, started_at=200.0, finished_at=260.0):
            self.backend.tick()
            controller.join(10)
        self.assertEqual(terminated, [])  # Modal already reports it finished
        job = next(j for j in controller.ledger.jobs() if j['name'] == 'stuck')
        self.assertEqual(job['status'], 'failed')
        self.assertAlmostEqual(job['charged'], 60 * T)  # the ledger keeps dollars

    def test_tick_asks_modal_again_only_after_the_retry_interval(self):
        from harness.games.hex import controller as controller_module
        module = self.stuck()
        controller = self.backend.controller
        with mock.patch.object(module, 'allocation_info', lambda *args: None):
            self.backend.tick()
            controller.join(10)
            self.backend.tick()
        self.assertEqual(controller.threads, {})  # within the interval: no second Modal listing
        self.assertEqual(controller.ledger.jobs()[0]['status'], 'uncertain')  # no record: the hold is kept
        self.assertIn('stuck', controller.next_recovery)
        self.assertEqual(controller_module.RECOVERY_RETRY_SECONDS, 30)


class Clock:
    """Host time that advances only when the provider sleeps."""
    def __init__(self, now=1000.0):
        self.now = now

    def time(self):
        return self.now

    def time_ns(self):
        return int(self.now * 1e9)

    def sleep(self, seconds):
        self.now += seconds


class Sandbox:
    """A Modal sandbox as the provider sees it; `record` is what Modal's sandbox list reports for it."""
    object_id = 'sb-test'

    def __init__(self, clock, start_after=None, evaluator_exit=None):
        self.clock, self.start_after, self.evaluator_exit = clock, start_after, evaluator_exit
        self.created, self.started, self.finished = clock.now, 0.0, 0.0
        self.terminated = False

    def record(self, *args):
        if not self.terminated and self.start_after is not None and self.clock.now >= self.created + self.start_after:
            self.started = self.started or self.clock.now
        return {'provider_id': self.object_id, 'task_id': 'ta-1' if self.started else '', 'created_at': self.created,
                'started_at': self.started, 'finished_at': self.finished, 'gpu_type': 'CPU' if self.started else ''}

    def terminate(self, wait=True):
        self.terminated = True
        if self.started:
            self.finished = self.clock.now

    def exec(self, *args, timeout=None):
        clock, killed_at, exit_code = self.clock, self.clock.now + timeout, self.evaluator_exit

        class Process:
            returncode = None
            stderr = mock.Mock(read=lambda: '')

            def poll(self):
                if clock.now >= killed_at:  # Modal's exec timeout kills the evaluator without output
                    self.returncode = exit_code
                return self.returncode
        return Process()


class ModalProviderAccounting(unittest.TestCase):
    """The real ModalProvider.execute against a fake sandbox, Modal record and clock; nothing reaches Modal."""

    def setUp(self):
        from harness.games.hex import provider as module
        self.module = module
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.ledger = Ledger(Path(temp.name) / 'research', {})  # the default contract
        self.clock = Clock()
        self.provider = object.__new__(module.ModalProvider)  # no image build
        self.provider.app = mock.Mock(app_id='ap-test')
        self.provider.export_reserve_seconds = 15

    def execute(self, sandbox, name, seconds, payload, stop=None, never_started=False):
        from harness import workers
        self.ledger.queue(name, seconds, payload, minimum_seconds=30)
        job = next(j for j in self.ledger.jobs() if j['name'] == name)
        self.provider._sandbox = lambda *args: sandbox
        self.provider.stage_evaluation = lambda *args: None
        with mock.patch.object(self.module, 'time', self.clock), \
                mock.patch.object(self.module, 'allocation_info', sandbox.record):
            if never_started:
                with self.assertRaises(workers.NeverStarted):  # the controller records it; nothing is held
                    self.provider.execute(self.ledger, job, stop)
            else:
                self.provider.execute(self.ledger, job, stop)
        return next(j for j in self.ledger.jobs() if j['name'] == name)

    def test_a_worker_modal_never_starts_is_given_up_and_charged_nothing(self):
        total = self.ledger.public()['remaining_usd']
        job = self.execute(Sandbox(self.clock), 'never', 300, {'kind': 'train'}, never_started=True)
        self.assertEqual((job['status'], job['charged']), ('failed', 0))
        self.assertIn('Gave up after 1800 s waiting in the provider queue', job['error'])
        self.assertIn('charged 0 seconds', job['error'])
        self.assertGreaterEqual(self.clock.now - 1000, 1800)  # waited the shared start window, not 60 s
        self.assertEqual(self.ledger.public()['remaining_usd'], total)  # the grant is released, not held
        self.ledger.queue('next', 300, {'kind': 'train'})  # and the session is not blocked

    def test_a_worker_stopped_while_queued_is_stopped_and_charged_nothing(self):
        stop = lambda: self.clock.now >= 1100  # noqa: E731 - the researcher runs `ar stop` after 100 s
        job = self.execute(Sandbox(self.clock), 'waiting', 300, {'kind': 'train'}, stop=stop, never_started=True)
        self.assertEqual((job['status'], job['charged']), ('stopped', 0))
        self.assertIn('Stopped while waiting in the provider queue', job['error'])
        self.assertLess(self.clock.now - 1000, 1800)  # the stop ended the wait early

    def test_a_start_racing_the_give_up_is_charged_from_modals_receipt(self):
        class Racing(Sandbox):
            def terminate(self, wait=True):  # Modal placed the task just as the host gave up
                self.started, self.finished, self.terminated = self.clock.now - 3, self.clock.now, True
        job = self.execute(Racing(self.clock), 'race', 300, {'kind': 'train'})
        self.assertEqual(job['status'], 'failed')
        self.assertAlmostEqual(job['charged'], 3 * T)  # never left uncertain, never refunded

    def test_a_slow_start_is_waited_for_and_charged_from_the_provider_start(self):
        sandbox = Sandbox(self.clock, start_after=400, evaluator_exit=0)
        sandbox.filesystem = mock.Mock(read_text=lambda path: '{"raw_score": 50}')
        job = self.execute(sandbox, 'slow', 120, {'kind': 'practice', 'bundle': 'b', 'panel': [], 'seeds': {}})
        self.assertEqual(job['status'], 'complete')
        self.assertEqual(job['started'], sandbox.started)
        self.assertGreaterEqual(sandbox.started - sandbox.created, 400)

    def test_practice_killed_by_its_grant_says_so(self):
        sandbox = Sandbox(self.clock, start_after=0, evaluator_exit=-1)
        job = self.execute(sandbox, 'eval-slow', 60, {'kind': 'practice', 'bundle': 'b', 'panel': [], 'seeds': {}})
        self.assertEqual(job['status'], 'failed')
        self.assertGreater(job['charged'], 0)  # the grant was used, so it is charged
        self.assertIn('Practice did not finish within its grant (60 s)', job['error'])


class TrainSandbox(Sandbox):
    """A training worker. Its one root-side sequence trains for 100 s (or until the host's pkill), then publishes
    the sealed archive and its logs to the hand-off Volume (or reports that the publish failed)."""

    def __init__(self, clock, mount, archive, publish_exit=0, publishes=True, link_down=False):
        super().__init__(clock, start_after=0)
        self.mount, self.archive, self.publish_exit, self.publishes = mount, archive, publish_exit, publishes
        self.link_down, self.killed, self.calls, self.copied, self.reads = link_down, False, [], [], []
        outer = self

        class Files:
            def write_text(self, text, path):
                pass

            def read_text(self, path):
                if path.endswith('/export.json'):
                    return '{"exit_code": 0}'
                raise FileNotFoundError(path)

            def copy_to_local(self, remote, local):
                outer.copied.append((remote, outer.terminated))
                Path(local).write_bytes(Path(outer.archive).read_bytes())
        self.filesystem = Files()

    def record(self, *args):
        return dict(super().record(*args), gpu_type='L40S' if self.started else '')

    def publish(self):
        if self.publishes:
            log = Path(self.archive).with_name('trainer.log')
            log.write_text('trained until stopped\n' if self.killed else 'trained\n')
            done = publish(self.mount, 'run1.' + self.object_id, {'bundle.tar': self.archive, 'trainer.log': log})
            assert done.returncode == 0, done.stderr

    def exec(self, *args, timeout=None):
        self.calls.append(args)
        script, outer = args[2] if len(args) > 2 else '', self
        if 'train_entry' not in script:
            if 'pkill' in script:
                self.killed = True
            return mock.Mock(wait=lambda: 0, poll=lambda: 0, stderr=mock.Mock(read=lambda: ''))
        done_at = self.clock.now + 100
        if self.link_down:  # the host never hears back; the worker finishes its sequence by itself
            self.publish()

        class Process:
            returncode = None
            stderr = mock.Mock(read=lambda: 'sync: Volume commit failed')

            def poll(self):
                if outer.link_down:
                    raise ConnectionError('host link down')
                if self.returncode is None and (outer.clock.now >= done_at or outer.killed):
                    self.returncode = outer.publish_exit
                    if outer.publish_exit == 0:
                        outer.publish()
                return self.returncode
        return Process()


class TrainingHandOff(unittest.TestCase):
    """The real ModalProvider.execute for a training run: its result travels through the hand-off Volume and is
    read after the worker has ended; nothing reaches Modal."""

    def setUp(self):
        from harness.games.hex import provider as module
        self.module = module
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)
        self.ledger = Ledger(self.temp / 'research', {})
        self.clock = Clock()
        self.archive, self.identity = sealed_archive(self.temp)
        self.mount = self.temp / 'volume'
        self.provider = object.__new__(module.ModalProvider)
        self.provider.app = mock.Mock(app_id='ap-test')
        self.provider.export_reserve_seconds = 15
        self.provider.handoff_wait = 0

    def train(self, sandbox, stop=None):
        reads = sandbox.reads

        class Watched(FolderVolume):  # every host read of the Volume, and whether the worker had ended by then
            def read_file(self, path):
                reads.append((path, sandbox.terminated))
                return super().read_file(path)

        class Modal:
            class Volume:
                @staticmethod
                def from_name(name, environment_name=None, **options):
                    return Watched(self.mount)
        self.provider.handoff_volume = lambda root: handoff.SessionVolume('hex', root, modal_module=Modal)
        self.provider._sandbox = lambda *args: sandbox
        payload = {'kind': 'train', 'code': CODE, 'config': {}, 'seed': 1}
        self.ledger.queue('run1', 300, payload)
        job = next(j for j in self.ledger.jobs() if j['name'] == 'run1')
        with mock.patch.object(self.module, 'time', self.clock), \
                mock.patch.object(self.module, 'allocation_info', sandbox.record):
            self.provider.execute(self.ledger, job, stop)
        record = json.loads((self.ledger.root / 'jobs/run1/allocation.json').read_text())
        return next(j for j in self.ledger.jobs() if j['name'] == 'run1'), record

    def test_a_finished_run_reaches_the_host_through_the_volume_after_its_worker_ends(self):
        sandbox = TrainSandbox(self.clock, self.mount, self.archive)
        job, record = self.train(sandbox)
        self.assertEqual((job['status'], job['artifact_sha']), ('complete', self.identity))
        # the provider receipt, nothing added (dollars: worker seconds x the shape's rate)
        self.assertAlmostEqual(job['charged'], (sandbox.finished - sandbox.started) * hex_ledger.terms(job)[1])
        self.assertEqual((record['handoff'], record['transfer']['job']), ('volume', 'run1.sb-test'))
        self.assertEqual(sandbox.copied, [])  # nothing was pulled out of the running worker
        self.assertTrue(sandbox.reads and all(ended for _, ended in sandbox.reads))  # read after the worker ended
        self.assertEqual((self.ledger.root / 'jobs/run1/trainer.log').read_text(), 'trained\n')
        scripts = [call[2] for call in sandbox.calls if len(call) > 2]
        self.assertIn('/handoff', scripts[0])  # made root-only before the trainer starts
        sequence = scripts[1]
        self.assertLess(sequence.index('train_entry'), sequence.index('pkill -KILL -u 65534'))
        self.assertLess(sequence.index('pkill -KILL -u 65534'), sequence.index('export_entry'))
        self.assertLess(sequence.index('export_entry'), sequence.index("os.chmod(mount,0o700)"))

    def test_a_lost_host_link_does_not_cost_the_finished_run(self):
        sandbox = TrainSandbox(self.clock, self.mount, self.archive, link_down=True)
        job, record = self.train(sandbox)
        self.assertEqual((job['status'], job['artifact_sha']), ('complete', self.identity))
        self.assertIn('host link down', record['transfer']['link_error'])
        self.assertEqual(sandbox.copied, [])

    def test_a_stopped_run_keeps_its_last_network(self):
        sandbox = TrainSandbox(self.clock, self.mount, self.archive)
        job, record = self.train(sandbox, stop=lambda: self.clock.now >= 1040)
        self.assertTrue(sandbox.killed)
        self.assertEqual((job['status'], job['artifact_sha'], record['handoff']), ('stopped', self.identity, 'volume'))
        self.assertLess(job['charged'], 100)
        self.assertEqual((self.ledger.root / 'jobs/run1/trainer.log').read_text(), 'trained until stopped\n')

    def test_a_failed_publish_falls_back_to_reading_the_running_worker(self):
        sandbox = TrainSandbox(self.clock, self.mount, self.archive, publish_exit=1)
        job, record = self.train(sandbox)
        self.assertEqual((job['status'], job['artifact_sha']), ('complete', self.identity))
        self.assertEqual(record['handoff'], 'live')
        self.assertIn('In-worker publish exited 1: sync: Volume commit failed', record['transfer']['fallback_reason'])
        self.assertEqual(sandbox.copied, [('/job/export.tar', False)])  # while the worker was still up
        self.assertEqual(sandbox.reads, [])

    def test_a_result_missing_from_the_volume_fails_the_run_charged_from_its_receipt(self):
        sandbox = TrainSandbox(self.clock, self.mount, self.archive, publishes=False)
        job, record = self.train(sandbox)
        self.assertEqual((job['status'], job['artifact_sha']), ('failed', None))
        self.assertIn('NotPublished', job['error'])
        self.assertAlmostEqual(job['charged'], (sandbox.finished - sandbox.started) * hex_ledger.terms(job)[1])
        self.assertEqual((record['handoff'], sandbox.copied), ('volume', []))


class WorkerSequence(unittest.TestCase):
    """The training worker's real root-side shell sequence, run locally with stand-ins for its Python steps."""

    def run_sequence(self, fail_export=False):
        from harness.games.hex import provider as module
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        temp = Path(temp.name)
        job, mount = temp / 'job', temp / 'volume'
        (job / 'output').mkdir(parents=True)
        archive, identity = sealed_archive(temp)
        fake = temp / 'python'
        fake.write_text(f'''#!/bin/sh
case "$2" in
  *train_entry) echo trained > {job}/trainer.log; echo '{{"status": "completed"}}' > {job}/supervisor.json
                echo '{{"wave": 1}}' > {job}/output/progress.jsonl;;
  *export_entry) {'echo no weights >&2; exit 3' if fail_export else f'cp {archive} {job}/export.tar'};;
  *) exec {sys.executable} "$@";;
esac
''')
        fake.chmod(0o755)
        local = handoff.PUBLISH.replace(repr(handoff.MOUNT), repr(str(mount)), 1)
        local = local.replace("subprocess.run(['sync',mount],check=True,timeout=120)", 'pass')
        with mock.patch.object(module, 'PYTHON', str(fake)), mock.patch.object(module, 'UNPRIVILEGED', ''), \
                mock.patch.object(handoff, 'PUBLISH', local):
            script = module.train_script(1.5, 'run1.sb-1')
        script = script.replace('/job/', f'{job}/').replace('pkill -KILL -u 65534', 'true')
        if shutil.which('timeout') is None:
            script = script.replace('timeout -s KILL 10 ', '')
        done = subprocess.run(['/bin/sh', '-c', script], capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(oct((mount / 'run1.sb-1').stat().st_mode & 0o777), oct(0o700))
        out = temp / 'host'
        out.mkdir()
        return volume_for(temp / 'research', mount), out, identity

    def test_the_sequence_publishes_the_sealed_archive_and_the_logs(self):
        from harness.games.hex import provider as module
        volume, out, identity = self.run_sequence()
        bundle, sha256, files = module.collect_published(volume, 'run1.sb-1', out, wait=0)
        self.assertEqual((bundle, sha256), (out / 'bundle', identity))
        self.assertEqual(files, ['bundle.tar', 'export.json', 'progress.jsonl', 'supervisor.json', 'trainer.log'])
        self.assertEqual(json.loads((out / 'export.json').read_text()), {'exit_code': 0})
        self.assertEqual((out / 'progress.jsonl').read_text(), '{"wave": 1}\n')

    def test_a_failed_export_publishes_its_diagnostics_and_no_archive(self):
        from harness.games.hex import provider as module
        volume, out, _ = self.run_sequence(fail_export=True)
        with self.assertRaises(module.NoPolicyPackage):
            module.collect_published(volume, 'run1.sb-1', out, wait=0)
        self.assertEqual(json.loads((out / 'export.json').read_text()), {'exit_code': 3})
        self.assertEqual((out / 'export-error.log').read_text(), 'no weights\n')
        self.assertFalse((out / 'bundle.tar').exists())


class WorkerMounts(unittest.TestCase):
    def test_training_workers_and_leases_mount_the_handoff_volume_and_practice_does_not(self):
        from harness.games.hex import provider as module
        modal = mock.MagicMock()
        provider = object.__new__(module.ModalProvider)
        provider.app, provider.image, provider.research = mock.Mock(), 'evaluation image', 'research image'
        provider.handoff_volume = lambda root: mock.Mock(mounts=lambda: {'/handoff': 'hand-off'})
        provider.scratch = lambda root: 'scratch'
        shape = {'gpu': 1, 'cpu_cores': 4, 'memory_mib': 16384}
        with mock.patch.dict(sys.modules, {'modal': modal}):
            for kind in ('train', 'lease', 'practice'):
                provider._sandbox('job', 100, kind, shape, 'root')
        calls = [call.kwargs for call in modal.Sandbox.create.call_args_list]
        self.assertEqual([call['volumes'] for call in calls],
                         [{'/handoff': 'hand-off'}, {'/scratch': 'scratch', '/handoff': 'hand-off'}, {}])
        self.assertEqual([call['image'] for call in calls], ['research image', 'research image', 'evaluation image'])
        for call in calls:  # no network, secrets or identity token in any worker
            self.assertEqual((call['block_network'], call['secrets'], call['include_oidc_identity_token']),
                             (True, [], False))


class LeaseSandbox:
    """A sandbox that records what the host ran as root and answers like a small shell."""

    def __init__(self, mount=None, publish_exit=0):
        self.calls, self.written, self.reloads, self.copied = [], {}, 0, []
        self.mount, self.publish_exit, self.identity = mount, publish_exit, None
        outer = self

        class Files:
            def write_text(self, text, path):
                outer.written[path] = text

            def copy_to_local(self, remote, local):
                outer.copied.append(remote)
                Path(local).write_bytes(outer.archive)
        self.filesystem = Files()

    def exec(self, *args, timeout=None):
        self.calls.append(args)
        if args[2] == handoff.PUBLISH:  # the root-side publish, run for real into the local Volume folder
            if self.publish_exit == 0:
                with tempfile.TemporaryDirectory() as temp:
                    (Path(temp) / 'export.tar').write_bytes(self.archive)
                    done = publish(self.mount, args[3], {'bundle.tar': Path(temp) / 'export.tar'})
                    assert done.returncode == 0, done.stderr
            return mock.Mock(wait=lambda: self.publish_exit, stderr=mock.Mock(read=lambda: 'sync failed'))
        text = 'tail output' if 'tail -c' in args[2] else ''
        if 'export_entry' in args[2] and self.identity:
            text = json.dumps({'bundle_sha256': self.identity}) + '\n'
        fail = 'missing' in args[2]
        return mock.Mock(wait=lambda: 1 if fail else 0, poll=lambda: 0, stdout=mock.Mock(read=lambda: text),
                         stderr=mock.Mock(read=lambda: 'No such file' if fail else ''))

    def reload_volumes(self):
        self.reloads += 1


class LeaseOnTheWorker(unittest.TestCase):
    """The real Lease against a fake sandbox: what runs as root, what runs as the researcher."""

    def setUp(self):
        from harness.games.hex import provider as module
        self.module = module
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.sandbox = LeaseSandbox(self.root / 'volume')
        self.lease = module.Lease(self.sandbox, self.root / 'jobs/gpu1', {'gpu': 1, 'cpu_cores': 8, 'memory_mib': 16384},
                                  volume_for(self.root, self.root / 'volume'))

    def test_a_command_is_an_argument_to_an_unprivileged_shell_never_part_of_the_root_script(self):
        hostile = 'echo hi; chmod 777 /job; $(id)'
        started = self.lease.start(hostile)
        shell, flag, script, argument = self.sandbox.calls[-1]
        self.assertEqual((shell, flag, argument), ('/bin/sh', '-c', hostile))
        self.assertNotIn('chmod', script)
        self.assertIn('setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs', script)
        self.assertIn('bash -c "$0"', script)
        self.assertIn('OMP_NUM_THREADS=8', script)
        self.assertEqual(started['exec'], '0001')
        record = self.lease.poll('0001', tail=50)
        self.assertEqual((record['state'], record['exit_code'], record['output']), ('exited', 0, 'tail output'))
        saved = json.loads((self.root / 'jobs/gpu1/exec/0001.json').read_text())
        self.assertEqual((saved['command'], saved['state']), (hostile, 'exited'))  # kept for the run archive
        with self.assertRaises(ValueError):
            self.lease.poll('0009')

    def test_push_sync_and_close(self):
        self.lease.push({'training/policy.py': 'p', 'configs/train.json': '{}'})
        self.assertEqual(self.sandbox.written, {'/work/training/policy.py': 'p', '/work/configs/train.json': '{}'})
        self.assertIn('chown -R 65534:65534 /work', self.sandbox.calls[-1][2])
        self.assertEqual(self.lease.sync(reload=True), {'committed': '/scratch', 'reloaded': True})
        self.assertEqual(self.sandbox.reloads, 1)
        self.lease.start('sleep 1000')
        self.lease.close()
        scripts = [call[2] for call in self.sandbox.calls]
        self.assertLess(scripts.index('pkill -KILL -u 65534 || true'), len(scripts) - 1)
        self.assertEqual(scripts[-1], 'sync /scratch')  # the last thing a lease does is commit its storage

    def package(self, name, path='/scratch/run1', code=CODE):
        target = self.root / 'jobs' / name
        target.mkdir(parents=True)
        sealed = self.lease.seal(target, path, code, {'a': 1}, name + '.attempt1')
        return target, sealed, self.lease.collect(target, sealed, wait=0)

    def test_package_reads_the_researchers_files_unprivileged_and_seals_with_the_source_sent(self):
        with tempfile.TemporaryDirectory() as temp:
            archive, identity = sealed_archive(temp)
            self.sandbox.archive, self.sandbox.identity = archive.read_bytes(), identity
        target, sealed, (bundle, sha256) = self.package('net1', "/scratch/run 1'; rm -rf /x")
        self.assertEqual((bundle, sha256), (target / 'bundle', identity))
        copy = self.sandbox.calls[0][2]
        self.assertIn("setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs head -c 134217729 -- "
                      "'/scratch/run 1'\"'\"'; rm -rf /x'/weights.pt > /job/output/weights.pt", copy)
        self.assertEqual(self.sandbox.written['/job/input/policy.py'], 'p = 1\n')
        self.assertIn('export_entry', self.sandbox.calls[1][2])
        # Sealed as root, then published to the hand-off Volume as root and read from it, never out of the worker.
        self.assertEqual(self.sandbox.calls[2][1:4], ('-c', handoff.PUBLISH, 'net1.attempt1'))
        self.assertEqual(self.sandbox.calls[2][4], 'bundle.tar=/job/export.tar')
        self.assertEqual((sealed['handoff'], self.sandbox.copied), ('volume', []))
        self.assertEqual(json.loads((target / 'handoff.json').read_text())['handoff'], 'volume')
        for path in ('relative/dir', '/scratch/missing'):
            with self.assertRaises(ValueError):
                self.lease.seal(self.root / 'jobs/net2', path, CODE, {}, 'net2.attempt1')

    def test_package_falls_back_to_the_live_copy_only_while_the_lease_serves(self):
        with tempfile.TemporaryDirectory() as temp:
            archive, identity = sealed_archive(temp)
            self.sandbox.archive, self.sandbox.identity = archive.read_bytes(), identity
        self.sandbox.publish_exit = 1  # the worker could not commit the Volume
        target, sealed, (_, sha256) = self.package('net1')
        self.assertEqual((sealed['handoff'], sha256, self.sandbox.copied), ('live', identity, ['/job/export.tar']))
        self.assertIn('In-worker publish exited 1: sync failed', sealed['fallback_reason'])
        self.sandbox.publish_exit, self.sandbox.mount = 0, self.root / 'elsewhere'  # published, but not visible
        target, sealed, _ = self.package('net2')
        record = json.loads((target / 'handoff.json').read_text())
        self.assertEqual((record['handoff'], len(self.sandbox.copied)), ('live', 2))
        self.assertIn('NotPublished', record['fallback_reason'])
        self.lease.closed = True  # the lease has ended: there is no worker left to read
        target = self.root / 'jobs/net3'
        target.mkdir()
        sealed = self.lease.seal(target, '/scratch/run1', CODE, {}, 'net3.attempt1')
        with self.assertRaisesRegex(RuntimeError, 'lease has ended'):
            self.lease.collect(target, sealed, wait=0)
        self.sandbox.identity = 'f' * 64  # whatever arrives must be the package the export sealed
        self.sandbox.mount = self.root / 'volume'
        target = self.root / 'jobs/net4'
        target.mkdir()
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.lease.collect(target, self.lease.seal(target, '/scratch/run1', CODE, {}, 'net4.attempt1'), wait=0)


class HexSession(unittest.TestCase):
    """A whole recipe session through harness.session with the real `ar`, broker, controller and ledger."""

    def test_recipe_session_trains_the_submitted_recipe_in_a_fresh_ledger_and_grades_it(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        release, meta = release_fixture('hex', temp.name, bind=False)
        private(release)
        policy.bind_release('hex', release)
        providers = []

        class Offline(registry.backend('hex')):
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
            seen['launch'] = ar.launch('explore', '--seconds', '150', '--config', 'configs/smoke.json', '--seed', '3',
                                       '--no-check')  # the dry run needs the box's engine
            seen['run'] = ar('run', 'again', '--seconds', '60', expect=2)
            ar('wait')
            ar('eval', 'explore', '--seconds', '60')
            ar('wait')
            seen['status'] = ar('status')
            ar.edit('training/learner.py', (ar.workspace / 'training/learner.py').read_text() + '\n# final\n')
            seen['submit'] = ar('submit', '--note', 'final')
            ar('finalize')
        result = session.run('hex', 'recipe', release, out, model='test/model', smoke=True, research_usd=0.5,
                             launcher=lambda folder: Process(lambda: script(Researcher(folder / 'task', box))),
                             find_box=lambda: LocalBox(box / 'ar') if (box / 'ar').is_dir() else None)
        self.assertEqual((result['status'], result['error'], result['closed_by']), ('researched', None, 'researcher'))
        self.assertEqual((seen['budget']['budget_usd'], seen['budget']['final_train_seconds']), (0.5, 180))
        self.assertEqual((seen['budget']['budget_mode'], seen['budget']['concurrency']), ('usd', 4))
        self.assertEqual(json.loads((out / 'plan.json').read_text())['budget_mode'], 'usd')
        self.assertEqual(Ledger(out / 'final').mode, 'seconds')  # fresh final training is one run of its seconds
        # The launcher leased a worker, ran the starter's training command there, sealed the result and stopped it.
        self.assertEqual([(j['name'], j['kind'], j['status']) for j in seen['status']['jobs'][:2]],
                         [('explorew', 'lease', 'stopped'), ('explore', 'package', 'complete')])
        self.assertEqual([(j['kind'], j['status']) for j in seen['status']['jobs'][2:]], [('practice', 'complete')])
        self.assertRegex(seen['launch'], r'training: mkdir -p /scratch/runs/explore && python training/train\.py '
                                         r'--config configs/smoke\.json --out /scratch/runs/explore --seed 3 --seconds \d+\n')
        self.assertIn('packaged /scratch/runs/explore as run explore', seen['launch'])
        # `ar run` is not a researcher command in a game with leases, in the client or at the broker.
        self.assertIn('invalid choice', seen['run']['error'])
        research, final = providers
        self.assertEqual(Ledger(out / 'research').public()['submissions'], [])  # research networks are never promoted
        self.assertEqual(result['final_training']['status'], 'complete')
        self.assertTrue(final.payloads[0]['code']['learner.py'].endswith('# final\n'))
        with mock.patch.object(policy, 'require_modal', lambda: None):
            grade = session.grade(out)
        self.assertEqual((grade['development'], grade['private_openings'], grade['final']['raw_score']), (True, 4, 55.0))
        self.assertEqual([len(p.evaluations) for p in providers], [0, 0, 1])  # only the host's reopened final root


class HexLeaseSession(unittest.TestCase):
    """A network session that works only through leases, with the real `ar`, broker, controller and ledger."""

    def test_network_session_packages_and_submits_a_network_from_a_lease(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        release, meta = release_fixture('hex', temp.name, bind=False)
        private(release)
        policy.bind_release('hex', release)
        providers = []

        class Offline(registry.backend('hex')):
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
            ar('eval', 'net1', '--seconds', '60')
            ar('wait')
            seen['status'] = ar('status')
            ar('submit', 'net1')
            ar('finalize')
        result = session.run('hex', 'network', release, out, model='test/model', smoke=True,
                             launcher=lambda folder: Process(lambda: script(Researcher(folder / 'task', box))),
                             find_box=lambda: LocalBox(box / 'ar') if (box / 'ar').is_dir() else None)
        self.maxDiff = None
        self.assertEqual((result['status'], result['error'], result['closed_by']), ('researched', None, 'researcher'))
        # The default mode: one job at a time (the lease, then its practice) within training-worker seconds.
        self.assertEqual((seen['budget']['budget_mode'], seen['budget']['budget_seconds'], seen['budget']['concurrency']),
                         ('seconds', 600, 1))
        self.assertIn('one at a time', (out / 'task/instruction.md').read_text())
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


class HexConstructor(unittest.TestCase):
    def test_release_binds_and_opens_as_a_session(self):
        from simulators.hex.autoresearch.prepare import audit_release
        with tempfile.TemporaryDirectory() as temp:
            release = Path(temp) / 'release'
            manifest = registry.constructor('hex')(release)
            self.assertEqual((manifest['gpu_jobs_started'], manifest['model_calls']), (0, 0))
            self.assertEqual({p.name for p in release.iterdir()}, {'frozen', 'public', 'private', 'release.json'})
            names = {p.relative_to(release / 'public').as_posix() for p in (release / 'public').rglob('*') if p.is_file()}
            self.assertEqual({name for name in names if not name.startswith(workspace.WORKSPACE + '/')},
                             {'CONTRACT.md', 'benchmark.json', 'engine-manifest.json', 'practice-opponents.json',
                              'session.json', 'workspace_manifest.json', 'agent/policy.py', 'agent/learner.py',
                              'agent/train.py', 'recipes/starter.json', 'recipes/smoke.json'})
            record = policy.bind_release('hex', release)
            self.assertEqual(policy.verify_release('hex', release)['sha256'], record['sha256'])
            (release / 'public/agent/extra.py').write_text('unmanifested')
            with self.assertRaisesRegex(ValueError, 'Unmanifested'):
                audit_release(release)
            (release / 'public/agent/extra.py').unlink()
            provider = Provider()
            backend = registry.backend('hex')(release, Path(temp) / 'session/research', smoke=True,
                                              budget_usd=600 * T, options={'_provider': provider})
            with backend:
                files = workspace.editable(release, backend.workspace)
                backend.run('a', 60, 'configs/train.json', 3)
                backend.tick()
                backend.controller.join(timeout=10)
            self.assertEqual(provider.payloads[0]['config'], json.loads(files['configs/train.json']))
            self.assertEqual(provider.payloads[0]['seed'], 3)


if __name__ == '__main__':
    unittest.main()
