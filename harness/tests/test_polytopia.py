"""Polytopia backend against its real controller and ledger; the provider is stubbed before any allocation.

Run with the game's research interpreter: .venv-polytopia-ar/bin/python -m unittest harness.tests.test_polytopia
"""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from harness import budget as budgets, handoff, policy, registry, session, workers, workspace
from harness.backend import STATUSES
from harness.transport import LocalBox
from harness.broker import Broker
from harness.tests.support import Process, Researcher, host_runs, release_fixture
from harness.tests.test_handoff import FolderVolume

try:
    import numpy  # noqa: F401  (the panel score needs it)
    from simulators.polytopia.autoresearch import build_context
    from simulators.polytopia.autoresearch.contract import DEFAULT, PRICE, TRAIN_SHAPE, rate
    from simulators.polytopia.autoresearch.ledger import Ledger
    from simulators.polytopia.autoresearch.v1.core import EDITABLE, HERE, MODE, seal, verify, write
    from harness.games.polytopia import controller as control, provider as transport
except ImportError as exc:  # wrong interpreter
    raise unittest.SkipTest(f'Polytopia research dependencies unavailable: {exc}')

NATIVE = {'training/policy.py': 'src/polytopia_bot/rl/network.py', 'training/learner.py': 'src/polytopia_bot/rl/ppo.py',
          'training/train.py': 'src/polytopia_bot/rl/gpu_resident.py',
          'training/collector.py': 'experiments/gpu_resident_train.py', 'configs/architecture.json': 'arch.json'}
PANEL = ('starter-200m', 'starter-1h', 'research-mid', 'research-top')
# Dollars per second of the training worker. Test sessions are opened with a budget written in its seconds.
R = rate(TRAIN_SHAPE)


def tree(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): p.read_text() for p in sorted(root.rglob('*')) if p.is_file()}


class Provider:
    """Stands in for Modal. It reserves, starts and settles on the real ledger exactly where the real provider does."""

    def __init__(self, succeed=True, seconds=200.0, receipt=True):
        self.succeed, self.seconds, self.receipt = succeed, seconds, receipt
        self.counts = dict(policy=191587, critic=191587, policy_cap=500000, critic_cap=500000, errors=[])
        self.preflights, self.checked, self.trained, self.evaluated, self.stopped = 0, [], [], [], []
        self.receipts = {}
        self.entered, self.gate = threading.Event(), None
        self.ended = {}  # run name -> how the real provider would report a kept network that ended early
        self.live = {}   # lease name -> its files and commands, while it is held
        self.packaged = []

    def preflight(self, out):
        self.preflights += 1

    def parameter_counts(self, workspace):
        self.checked.append(tree(workspace))
        return dict(self.counts)

    def stop(self, name):
        self.stopped.append(name)
        if self.gate:
            self.gate.set()

    def lookup(self, app_id, sandbox_id):
        return self.receipts.get(sandbox_id)

    def train(self, ledger, name, grant, workspace, out, smoke=True, stop=None):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        shutil.copytree(workspace, out / 'input')
        self.trained.append(dict(name=name, grant=grant, smoke=smoke, files=tree(out / 'input')))
        if stop and stop():
            raise RuntimeError('Stopped before any GPU allocation')
        ledger.reserve(name, grant)
        write(out / 'provider.json', dict(sandbox_id='sb-' + name, app_id='ap-stub', ledger_job_name=name))
        ledger.start(name, 'sb-' + name, 1000.0)
        self.entered.set()
        if self.gate:
            self.gate.wait(timeout=30)
        receipt = (dict(provider_id='sb-' + name, started_at=1000.0, finished_at=1000.0 + self.seconds)
                   if self.receipt else None)
        if not self.succeed or (stop and stop()):
            ledger.settle(name, receipt, 'failed')
            raise RuntimeError('stub provider: training did not complete')
        bundle = out / 'bundle'
        bundle.mkdir()
        (bundle / 'final.npz').write_bytes(b'stub weights for ' + name.encode())
        shutil.copyfile(out / 'input' / EDITABLE[0], bundle / 'network.py')
        seal(bundle)
        ledger.settle(name, receipt, 'complete')
        return dict(bundle=str(bundle), learning=dict(updates=3, changed_policy_tensors=5), budget=ledger.public(),
                    ended=self.ended.get(name, 'complete'))

    def evaluate(self, bundle, seeds, opponent_seed, out, opponent_bundle=None):
        verify(bundle)
        self.evaluated.append(dict(seeds=list(seeds), opponent_seed=opponent_seed, out=str(out),
                                   opponent=Path(opponent_bundle).name if opponent_bundle else None))
        rows = [dict(seed=seed, seat=seat, reward=1.0 if seat == 1 else -1.0, capped=False)
                for seed in seeds for seat in (1, 2)]
        summary = dict(raw_score_0_100=50.0, games=len(rows), natural_games=len(rows), capped_games=0)
        write(out, dict(summary, details=rows, cpu_seconds=1.0))
        return summary

    # A lease is held until it is stopped; the stub charges 40 worker seconds.
    def hold(self, ledger, name, grant, shape, files, out, stop=None):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        ledger.reserve(name, grant)
        ledger.start(name, 'sb-' + name, 2000.0)
        self.live[name] = dict(files=dict(files), commands=[], shape=shape)
        while not (stop and stop()):
            time.sleep(0.01)
        del self.live[name]
        write(out / 'exec/0001.json', dict(exec='0001', state='ended with the lease', output='late', command='x'))
        ledger.settle(name, dict(provider_id='sb-' + name, started_at=2000.0, finished_at=2040.0), 'stopped')

    def lease_live(self, name):
        return name in self.live

    def lease_exec(self, name, command):
        self.live[name]['commands'].append(command)
        return dict(exec=f"{len(self.live[name]['commands']):04d}", state='running')

    def lease_poll(self, name, exec_id, tail=None):
        return dict(exec=exec_id, state='exited', exit_code=0, output='ran ' + self.live[name]['commands'][-1])

    def lease_push(self, name, files):
        self.live[name]['files'] = dict(files)
        return dict(pushed=sorted(files))

    def lease_sync(self, name, reload=False):
        return dict(committed='/scratch', reloaded=reload)

    def lease_package(self, target, name, path, files):
        self.packaged.append(dict(lease=name, path=path, files=dict(files)))
        target = Path(target)
        target.mkdir(parents=True)
        value = io.BytesIO()
        numpy.savez(value, p0=numpy.ones((4, 4), dtype=numpy.float32))
        record = json.dumps(dict(sha256=hashlib.sha256(value.getvalue()).hexdigest())).encode()
        transport.package_bundle(target / 'bundle', {'final.npz': value.getvalue(), 'final.json': record}, files,
                                 'release', dict(lease=name, path=path))
        return target / 'bundle'


class Case(unittest.TestCase):
    """`budget` is the research allocation in seconds of the training worker (None: the game's default). `mode` is
    the session's budget mode: most of these tests exercise dollar mode, opened with that many seconds' worth of
    dollars (an explicit dollar budget); the seconds-mode cases set mode = 'seconds'."""
    smoke, budget, bind, mode = True, 600, True, 'usd'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.release, self.meta = release_fixture('polytopia', self.temp.name, bind=self.bind)
        self.prepare_release()
        self.provider = self.make_provider()
        self.out = Path(self.temp.name).resolve() / 'session'
        self.out.mkdir()
        self.root = self.out / 'research'
        allowance = self.allowance()
        self.backend = self.open(self.root, **allowance)
        self.broker = host_runs(Broker(self.backend, self.out, 'network', research_seconds=allowance.get('budget_seconds'),
                                       research_usd=allowance.get('budget_usd')))
        base = workspace.public(self.release) / workspace.WORKSPACE
        self.files = {name: (base / name).read_text() for name in self.meta['writable']}

    def prepare_release(self):
        pass

    def allowance(self):
        prefix = 'smoke_' if self.smoke else ''
        if self.mode == 'seconds':
            return dict(budget_seconds=self.budget or registry.spec('polytopia')[prefix + 'research_seconds'])
        return dict(budget_usd=self.budget * R if self.budget else registry.spec('polytopia')[prefix + 'research_usd'])

    def make_provider(self):
        return Provider()

    def open(self, root, **kwargs):
        backend = registry.backend('polytopia')(self.release, root, smoke=self.smoke,
                                                options={'_provider': self.provider}, **kwargs)
        backend.open()
        self.addCleanup(backend.close)
        return backend

    def work(self, backend=None):
        """One broker tick, then wait for the worker thread it started."""
        backend = backend or self.backend
        backend.tick()
        backend.controller.join(timeout=20)

    def seconds(self, key='used_usd', backend=None):
        """A dollar figure of the budget, in seconds of the training worker."""
        return round((backend or self.backend).budget()[key] / R, 6)

    def row(self, name, backend=None):
        """A job's ledger row, with the worker seconds it was charged for."""
        row = Ledger((backend or self.backend).root).job(name)
        return dict(row, seconds=row['ended'] - row['started'] if row['started'] and row['ended'] else 0)

    def run_job(self, name='a', seconds=360, files=None, config='configs/smoke.json'):
        return self.broker.handle({'op': 'run', 'name': name, 'seconds': seconds, 'config': config,
                                   'files': files or self.files})

    def jobs(self):
        status = self.broker.handle({'op': 'status'})
        self.assertTrue(all(job['status'] in STATUSES for job in status['jobs']), status)
        return [(job['name'], job['kind'], job['status']) for job in status['jobs']], status['active']


class PolytopiaBackend(Case):
    def test_run_carries_exactly_the_researchers_files_and_is_charged(self):
        self.provider.succeed, self.provider.seconds = False, 30.0
        self.assertEqual(self.provider.preflights, 1)  # the CPU gate ran before any GPU work could be requested
        budget = self.broker.handle({'op': 'budget'})
        self.assertEqual((budget['budget_usd'], budget['used_usd'], budget['remaining_usd']), (600 * R, 0, 600 * R))
        self.assertEqual(budget['usd_per_hour']['lease --gpu 1 --cpu 4 --memory 16'], round(3600 * R, 4))
        files = dict(self.files)
        files['training/policy.py'] += '\n# researcher change\n'
        files['training/collector.py'] += '\n# researcher option\n'
        recipe = json.loads(files['configs/smoke.json'])
        recipe['learning_rate'] = 0.001
        files['configs/smoke.json'] = json.dumps(recipe)
        reply = self.run_job(files=files)
        self.assertNotIn('error', reply, reply)
        self.assertEqual(self.jobs(), ([('a', 'train', 'queued')], ['a']))
        self.assertEqual((reply['usd_per_second'], reply['reserved_usd']), (R, 360 * R))
        self.assertEqual(self.seconds('remaining_usd'), 240)  # the queued grant is held, in dollars
        self.work()
        payload = self.provider.trained[0]
        expected = {native: files[name] for name, native in NATIVE.items()}
        expected['recipe.json'] = files['configs/smoke.json']
        self.assertEqual(payload['files'], expected)  # the edited sources, the chosen config and nothing else
        self.assertEqual((payload['name'], payload['grant'], payload['smoke']), ('a', 360, True))
        self.assertEqual(self.jobs(), ([('a', 'train', 'failed')], []))
        self.assertEqual((self.seconds(), self.seconds('remaining_usd')), (30, 570))  # failures stay charged
        job = self.broker.handle({'op': 'status'})['jobs'][0]
        self.assertEqual((job['worker_seconds'], job['usd_per_second']), (30.0, R))
        self.assertAlmostEqual(job['charged_usd'], 30 * R)
        self.assertIn('error', self.broker.handle({'op': 'submit', 'name': 'a'}))  # no trained artifact
        self.assertIn('did not complete', self.broker.handle({'op': 'log', 'name': 'a'})['failure'])
        self.assertIn('Unknown run', self.broker.handle({'op': 'log', 'name': 'zzz'})['error'])
        self.assertIn('already used', self.run_job()['error'])
        self.assertIn('Unknown command', self.broker.handle({'op': 'grade'})['error'])  # grading is host-only

    def test_bad_requests_never_reach_the_ledger(self):
        def changed(name, **fields):
            return dict(self.files, **{name: json.dumps(dict(json.loads(self.files[name]), **fields))})
        no_trainer = json.loads(self.files['configs/smoke.json'])
        del no_trainer['trainer']
        requests = (dict(files=dict(self.files, **{'training/learner.py': 'def broken(:'})),
                    dict(files=dict(self.files, extra='x')),
                    dict(seconds=100000), dict(seconds=10), dict(name='../a'), dict(name='a_b'), dict(name='eval-a'),
                    dict(files=changed('configs/smoke.json', size=14)),  # the board, maps and round limit are frozen
                    dict(files=changed('configs/smoke.json', maps='proposed-v1')),
                    dict(files=changed('configs/smoke.json', max_turns=200)),
                    dict(files=changed('configs/smoke.json', max_turns=MODE.ROUND_LIMIT + 1)),
                    dict(files=changed('configs/smoke.json', learning_rate=1.0)),
                    dict(files=dict(self.files, **{'configs/smoke.json': json.dumps(no_trainer)})),
                    dict(files=changed('configs/architecture.json', width=64)),  # smoke keeps the 2x32 network
                    dict(config='configs/other.json'))
        for request in requests:
            self.assertIn('error', self.run_job(**request), request)
        self.assertEqual((Ledger(self.root).jobs(), self.provider.trained), ([], []))
        self.assertFalse((self.root / 'queue').exists())
        self.work()
        self.assertEqual(self.provider.trained, [])

    def test_check_counts_parameters_without_allocation(self):
        files = dict(self.files)
        files['training/policy.py'] += '\n# wider\n'
        reply = self.broker.handle({'op': 'check', 'files': files, 'config': 'configs/smoke.json'})
        self.assertEqual(reply['parameters']['policy'], 191587, reply)
        self.assertEqual(self.provider.checked[0]['src/polytopia_bot/rl/network.py'], files['training/policy.py'])
        self.provider.counts.update(policy=700000, errors=['policy parameter cap exceeded'])
        self.assertIn('cap exceeded', self.broker.handle({'op': 'check', 'files': files})['error'])
        self.assertEqual(self.backend.budget()['used_usd'], 0)
        self.assertEqual(self.backend.status()['jobs'], [])

    def test_network_session_practice_submit_finalize_and_host_grade(self):
        self.assertNotIn('error', self.run_job())
        self.assertIn('completed run', self.broker.handle({'op': 'eval', 'name': 'a'})['error'])  # still queued
        self.work()
        self.assertEqual(self.jobs(), ([('a', 'train', 'complete')], []))
        self.assertIn('One successful', self.run_job('b', 120)['error'])  # the smoke allows one successful run
        practice = self.broker.handle({'op': 'eval', 'name': 'a'})
        self.assertEqual(practice['queued'], 'eval-a-1', practice)
        self.assertIn('Resolve active work', self.broker.handle({'op': 'finalize'})['error'])
        self.work()
        self.assertEqual(self.jobs(), ([('a', 'train', 'complete'), ('eval-a-1', 'practice', 'complete')], []))
        self.assertEqual(self.provider.evaluated[0]['seeds'], [700000])  # the public smoke map
        self.assertEqual(self.broker.handle({'op': 'log', 'name': 'eval-a-1'})['result']['games'], 2)
        submitted = self.broker.handle({'op': 'submit', 'name': 'a'})['submitted']['native']
        self.assertEqual(set(submitted), {'run', 'sha256', 'submitted_at'})  # no host path reaches the researcher
        self.assertEqual(self.broker.handle({'op': 'status'})['sealed_submission'], submitted)
        sealed = self.root / 'submissions' / submitted['sha256']
        self.assertEqual(verify(sealed), submitted['sha256'])
        self.assertFalse([p for p in sealed.rglob('*') if p.is_file() and p.stat().st_mode & 0o222])  # read-only
        self.assertEqual(self.broker.handle({'op': 'finalize'})['finalized'], True)
        self.assertIn('finalized', self.run_job('c', 60)['error'])
        self.assertEqual((self.seconds(), self.seconds('remaining_usd'), self.backend.budget()['lifecycle']),
                         (200, 400, 'finalized'))
        self.backend.close()
        # Host grading reopens the same root: no new allowance, the private schedule fixed when the session opened.
        with self.assertRaises(ValueError):
            registry.backend('polytopia')(self.release, self.root, smoke=True, options={'_provider': self.provider})
        graded = self.open(self.root, resume=True)
        self.assertEqual(graded.budget()['budget_usd'], 600 * R)
        schedule = json.loads((self.root / 'private/schedule.json').read_text())
        self.assertEqual(len(schedule['seeds']), 2)
        result = graded.grade()
        self.assertEqual(result['games'], 4)
        self.assertEqual(self.provider.evaluated[-1]['seeds'], schedule['seeds'])
        self.assertTrue(self.provider.evaluated[-1]['out'].startswith(str(self.root / 'private')))
        self.assertEqual(graded.grade(), result)  # never graded twice
        self.assertEqual(len(self.provider.evaluated), 2)

    def test_settle_cancels_queued_work_without_charge(self):
        self.run_job()
        self.backend.settle()
        self.assertEqual(self.jobs(), ([('a', 'train', 'stopped')], []))
        self.assertEqual(self.backend.budget()['used_usd'], 0)
        self.work()
        self.assertEqual(self.provider.trained, [])
        self.assertEqual(self.broker.finalize('host')['submission'], None)

    def test_settle_stops_a_running_worker_and_its_lifetime_stays_charged(self):
        self.provider.gate, self.provider.seconds = threading.Event(), 95.0
        self.run_job()
        self.backend.tick()
        self.assertTrue(self.provider.entered.wait(timeout=10))
        self.assertEqual(self.jobs(), ([('a', 'train', 'running')], ['a']))
        self.assertEqual(self.seconds('remaining_usd'), 240)  # a running worker holds its whole grant
        self.assertIn('exceeds the remaining', self.run_job('b', 300)['error'])
        self.assertEqual(self.broker.handle({'op': 'check', 'files': self.files})['parameters']['policy'], 191587)
        self.backend.settle()
        self.assertEqual(self.provider.stopped, ['a'])
        self.assertEqual(self.jobs(), ([('a', 'train', 'stopped')], []))
        self.assertEqual((self.seconds(), self.seconds('remaining_usd')), (95, 505))
        self.backend.settle()  # idempotent
        self.assertEqual(self.provider.stopped, ['a'])

    def test_unknown_provider_lifetime_holds_the_grant_until_its_receipt(self):
        self.provider.succeed, self.provider.receipt = False, False
        self.run_job()
        self.work()
        self.assertEqual(self.jobs(), ([('a', 'train', 'uncertain')], ['a']))
        self.assertEqual(self.seconds('remaining_usd'), 240)
        self.assertIn('exceeds the remaining', self.run_job('b', 300)['error'])
        with self.assertRaises(RuntimeError):
            self.broker.finalize('host')
        self.backend.controller.next_recovery.clear()
        self.work()  # the provider still has no receipt
        self.assertEqual(self.jobs()[1], ['a'])
        self.provider.receipts['sb-a'] = dict(provider_id='sb-a', started_at=1000.0, finished_at=1410.0)
        self.backend.controller.next_recovery.clear()
        self.work()
        self.assertEqual(self.jobs(), ([('a', 'train', 'failed')], []))
        self.assertEqual((self.seconds(), self.seconds('remaining_usd')), (410, 190))
        receipt = json.loads((self.root / 'jobs/a/recovered-interruption.json').read_text())
        self.assertIs(receipt['unsubmitted_artifact_adopted'], False)

    def test_fresh_final_training_of_a_recipe(self):
        """What harness.session.final_training does: a second root with its own allowance and one run named final."""
        files = dict(self.files)
        files['training/learner.py'] += '\n# the submitted recipe\n'
        final = self.open(self.out / 'final', budget_seconds=360)
        workspace.apply(final.release, final.workspace, files)
        final.run('final', 360, 'configs/smoke.json', 1)
        self.work(final)
        status = final.status()
        self.assertEqual([(job['name'], job['status']) for job in status['jobs']], [('final', 'complete')])
        self.assertEqual(self.provider.trained[0]['files']['src/polytopia_bot/rl/ppo.py'], files['training/learner.py'])
        self.assertEqual(final.submit('final')['run'], 'final')
        self.assertEqual(final.finalize()['finalized'], True)
        # A fresh final is always seconds mode: one run of the final's seconds of the training worker.
        self.assertEqual((final.budget()['budget_mode'], final.budget()['budget_seconds']), ('seconds', 360))
        self.assertEqual(final.budget()['used_seconds'], 200)
        self.assertIn('no leases', self.assert_refused(final))
        self.assertEqual(self.backend.budget()['used_usd'], 0)  # nothing reached the research session

    def test_a_continued_session_adopts_its_scratch_copy_and_a_final_never_does(self):
        self.assertFalse((self.root / 'scratch-volume.json').exists())  # created lazily by the first lease
        options = {'_provider': self.provider, 'scratch_volume': 'polytopia-scratch-copy1'}
        research = registry.backend('polytopia')(self.release, self.out / 'research2', smoke=self.smoke,
                                                 options=options, **self.allowance())
        research.open()
        self.addCleanup(research.close)
        record = json.loads((self.out / 'research2/scratch-volume.json').read_text())
        self.assertEqual((record['volume'], record['mount'], record['adopted']), ('polytopia-scratch-copy1', '/scratch', True))
        final = registry.backend('polytopia')(self.release, self.out / 'final2', smoke=self.smoke, options=options,
                                              budget_seconds=360)
        final.final = True
        final.open()
        self.addCleanup(final.close)
        self.assertFalse((self.out / 'final2/scratch-volume.json').exists())
        with self.assertRaisesRegex(ValueError, 'polytopia-scratch-'):
            registry.backend('polytopia')(self.release, self.out / 'research3', smoke=self.smoke,
                                          options=dict(options, scratch_volume='ar-coordinator-records'),
                                          **self.allowance()).open()

    def assert_refused(self, final):
        final.final = True
        with self.assertRaises(ValueError) as refused:
            final.lease('gpu1', 120, 1, 4, 16)
        return str(refused.exception)


def checkpoint(folder, stem, value, pad=0):
    """One checkpoint as the trainer's runner.save_checkpoint leaves it: STEM.npz and STEM.json naming its digest.
    `pad` adds that many incompressible bytes (a real checkpoint is 0.7-1.8 MB and does not compress)."""
    import numpy as np
    stream = io.BytesIO()
    extra = dict(o0=np.random.default_rng(len(stem)).integers(0, 256, pad, dtype=np.uint8)) if pad else {}
    np.savez(stream, p0=np.full((4, 4), value, dtype=np.float32), **extra)
    (folder / (stem + '.npz')).write_bytes(stream.getvalue())
    (folder / (stem + '.json')).write_text(json.dumps(dict(sha256=hashlib.sha256(stream.getvalue()).hexdigest())))


def write_run(folder, updates=3, final=None, exports=(), cut_line=False, clocks=(), real_sizes=False):
    """A trainer's run folder: what it has at its first update, then its whole exports and, with `final`, its own end
    (`final` is the `stopped` field of training-complete.json). `clocks` are its 5- and 10-minute snapshots."""
    folder.mkdir(parents=True, exist_ok=True)
    policy, critic = (1_840_000, 715_000) if real_sizes else (0, 0)  # bytes, as collected on 2026-11-13
    checkpoint(folder, 'initial', 0.0, critic)
    (folder / 'protocol.json').write_text('{}')
    # A real update row is about 1.3 KB of metrics.
    filler = {f'metric_{k}': 0.123456789 + k for k in range(48)} if real_sizes else {}
    rows = ''.join(json.dumps(dict(update=i + 1, policy_loss=0.5, value_loss=0.25, samples=(i + 1) * 65536,
                                   elapsed=0.6253 * (i + 1), return_mean=0.0663 + i * 1e-7, **filler)) + '\n'
                   for i in range(updates))
    for minutes in clocks:
        checkpoint(folder, f'clock-{minutes}m', float(minutes), policy)
        checkpoint(folder, f'clock-{minutes}m-critic', float(minutes), critic)
        (folder / f'clock-{minutes}m-receipt.json').write_text(json.dumps(dict(requested_minutes=minutes)))
    (folder / 'training.jsonl').write_text(rows + ('{"update": %d, "policy_lo' % (updates + 1) if cut_line else ''))
    for update in exports:
        export = folder / 'exports' / f'update-{update:09d}'
        export.mkdir(parents=True)
        checkpoint(export, 'final', float(update))
        checkpoint(export, 'final-critic', float(update))
        (export / 'export.json').write_text(json.dumps(dict(update=update, samples=update * 16384, elapsed=1.0)))
    if final is not None:
        checkpoint(folder, 'final', 100.0, policy)
        checkpoint(folder, 'final-critic', 100.0, critic)
        (folder / 'training-complete.json').write_text(json.dumps(dict(updates=updates, seconds=1.0, stopped=final)))


class Cloud:
    """Stands in for Modal's sandboxes and their provider records: a created worker may wait in a GPU queue.

    `start_after` is how many reads of its record pass before the provider starts it (None: never);
    `start_on_terminate` starts it in the same moment it is terminated (a start racing an abandon).
    `trainer` is what the training command does: 'finishes'; 'stops' when asked; 'ignores' a stop request and runs
    on; is 'cut' off at its time limit; 'fails' by itself. `exports` are the whole periodic exports it has written
    by then, `run_age` how long ago the worker started (a run near its time limit).
    """

    def __init__(self, start_after=0, start_on_terminate=False, lifetime=123.0, output=None):
        self.start_after, self.start_on_terminate, self.lifetime = start_after, start_on_terminate, lifetime
        self.sandboxes, self.terminated = {}, []
        # The hand-off Volumes, one folder each; `volume_fails` breaks them: 'publish' (the in-worker publish fails),
        # 'invisible' (published, but the host never sees it), 'unreadable' (the host sees the manifest, not the files).
        self.volumes, self.volume_fails, self.volume_reads = Path(output).with_name('volumes'), None, []
        self.output, self.trainer, self.exports, self.run_age = output, 'finishes', (), 0.0
        self.updates, self.clocks, self.stdout_lines, self.real_sizes = 3, (), 5, False
        self.stream_bytes_per_second = float('inf')  # how fast a collection's archive reaches the host
        self.slow_parts = {}                         # part -> its own rate
        self.damage = None                           # a file of the finished run that is cut short afterwards
        self.checkpoint = None                       # what a lease's directory holds for `ar package`
        self.training = threading.Event()

    def create(self, role, seconds, gpu=False, shape=None, volumes=None):
        sandbox = Sandbox(self, 'sb-' + role)
        sandbox.mounts = dict(volumes or {})
        sandbox.shape, sandbox.volumes = shape, volumes
        self.sandboxes[sandbox.object_id] = sandbox
        return sandbox

    def info(self, app_id, sandbox_id):
        sandbox = self.sandboxes.get(sandbox_id)
        if sandbox is None:
            return None
        sandbox.reads += 1
        if (sandbox.started_at is None and sandbox.finished_at is None and self.start_after is not None
                and sandbox.reads > self.start_after):
            sandbox.started_at = time.time() - self.run_age
        return dict(provider_id=sandbox_id, task_id='ta-1', created_at=sandbox.created_at,
                    started_at=sandbox.started_at or 0, finished_at=sandbox.finished_at or 0,
                    gpu_type='L40S' if sandbox.started_at else '')


class Volume(FolderVolume):
    """One hand-off Volume as the host reads it (a folder of the cloud), and as a worker mounts it."""

    def __init__(self, cloud, name, options):
        super().__init__(cloud.volumes / name)
        self.cloud, self.name, self.options = cloud, name, options

    def read_file(self, path):
        fails = self.cloud.volume_fails
        if fails == 'invisible' or (fails == 'unreadable' and not path.endswith(handoff.MANIFEST)):
            raise FileNotFoundError(path)
        # When the host read it: was every training worker already ended?
        self.cloud.volume_reads.append((path, all(sb.finished_at for sb in self.cloud.sandboxes.values())))
        yield from super().read_file(path)

    def remove_file(self, path, recursive=False):
        shutil.rmtree(self.root / path)


class Exec:
    def __init__(self, stdout, returncode=0, stderr=''):
        self.stdout = io.BytesIO(stdout) if isinstance(stdout, bytes) else io.StringIO(stdout)
        self.stderr, self.returncode = io.StringIO(stderr), returncode

    def poll(self):
        return self.returncode


class Unread:
    """The trainer command's own output stream. Reading it after a long run took the collection's time on a live
    worker (2026-11-13), so the provider must not touch it: the trainer's output is in files in the worker."""

    def read(self):
        raise AssertionError('the trainer command\'s stream must not be read')


class Trainer(Exec):
    """The training command of one worker, behaving as its cloud says (Cloud.trainer)."""

    def __init__(self, sandbox, folder, args, env):
        super().__init__('', None)
        self.stdout = self.stderr = Unread()
        self.sandbox, self.folder, self.args, self.env, self.kind = sandbox, folder, args, env, sandbox.cloud.trainer
        self.plan = json.loads(args[3])
        # The wrapper makes the hand-off Volume root-only before the trainer (the researcher's code) starts.
        assert self.plan['prepare'] == handoff.prepare_command('python'), self.plan['prepare']
        sandbox.prepared = True
        write_run(folder, updates=sandbox.cloud.updates, final=False if self.kind == 'finishes' else None,
                  exports=sandbox.cloud.exports, cut_line=self.kind in ('cut', 'ignores'), clocks=sandbox.cloud.clocks,
                  real_sizes=sandbox.cloud.real_sizes)
        if sandbox.cloud.damage:
            (folder / sandbox.cloud.damage).write_bytes((folder / sandbox.cloud.damage).read_bytes()[:40])
        # What the shell wrapper redirects the trainer's output to.
        sandbox.local(transport.TRAINER_STDOUT).write_text('progress\n' * sandbox.cloud.stdout_lines + 'RESULT {"ok": true}\n')
        sandbox.local(transport.TRAINER_STDERR).write_text('Traceback: own learner\n' if self.kind == 'fails' else '')
        sandbox.cloud.training.set()

    def poll(self):
        if self.returncode is None:
            if self.kind in ('finishes', 'quits'):  # 'quits': exit code 0 without the trainer's own end
                self.returncode = 0
            elif self.kind == 'fails':
                self.returncode = 1
            elif self.kind == 'cut':
                self.returncode = -1  # the SDK's report of a command that reached its time limit
            elif self.kind == 'stops' and self.env['POLYTOPIA_STOP_REQUEST'] in self.sandbox.files:
                write_run(self.folder, final=True)  # its loop ended at the request: final checkpoint, own end
                self.returncode = 0
            if self.returncode == 0:  # the wrapper publishes a trainer that ended by itself, inside the same command
                self.sandbox.handoff(self.plan['handoff'], chained=True)
        return self.returncode


class Sandbox:
    """One worker: a training command (Trainer) and the provider's real export script run on its run folder."""

    def __init__(self, cloud, object_id):
        self.cloud, self.object_id = cloud, object_id
        self.created_at, self.started_at, self.finished_at, self.reads, self.files = time.time(), None, None, 0, {}
        self.filesystem, self.trainer, self.collections, self.handoffs, self.scripts = self, None, [], [], []
        self.mounts, self.prepared = {}, False

    def write_text(self, text, path):
        self.files[path] = text

    def terminate(self, wait=False):
        if self.finished_at is not None:
            return
        if self.started_at is None and self.cloud.start_on_terminate:
            self.started_at = time.time()
        self.finished_at = self.started_at + self.cloud.lifetime if self.started_at else time.time()
        self.cloud.terminated.append(self.object_id)

    def local(self, path):
        """Where a worker path under /output lives on this machine."""
        return Path(self.cloud.output) / self.object_id / Path(path).relative_to('/output')

    def handoff(self, argv, chained=False):
        """The provider's in-worker hand-off step, run on this machine: worker paths become local ones, /handoff is
        the mounted Volume's folder, and `sync` (which commits a real Volume) is skipped."""
        assert argv[:3] == ['python', '-c', transport.HANDOFF_SCRIPT], argv[:2]
        run, mode, kind, out_log, err_log, out_limit, err_limit, staging = argv[3:11]
        python, flag, code, job, *items = argv[11:]
        assert (python, flag, code) == ('python', '-c', handoff.PUBLISH)
        code = code.replace(repr(handoff.MOUNT), repr(str(self.mounts[handoff.MOUNT].root)), 1)
        code = code.replace("subprocess.run(['sync',mount],check=True,timeout=120)", 'pass')
        if self.cloud.volume_fails == 'publish':
            code = 'raise SystemExit("no Volume is mounted at /handoff")'
        items = [f'{key}={self.local(path)}' for key, _, path in (item.partition('=') for item in items)]
        done = subprocess.run([sys.executable, '-c', argv[2], str(self.local(run)), mode, kind, str(self.local(out_log)),
                               str(self.local(err_log)), out_limit, err_limit, str(self.local(staging)),
                               sys.executable, flag, code, job, *items], capture_output=True, text=True)
        reply = json.loads(done.stdout.strip().splitlines()[-1]) if done.returncode == 0 else {}
        self.handoffs.append(dict(kind=kind, mode=mode, job=job, chained=chained, returncode=done.returncode,
                                  reused=bool(reply.get('reused')), alive=self.finished_at is None))
        return Exec(done.stdout, done.returncode, done.stderr)

    def copy_to_local(self, remote, local):
        Path(local).write_bytes(self.files[remote])

    def reload_volumes(self):
        self.reloaded = True

    def exec(self, *args, **kwargs):
        if args[:2] == ('/bin/sh', '-c'):
            # A lease: the host's own root scripts and the researcher's commands, recorded, not run.
            self.scripts.append(args[2:])
            if transport.PACKAGE_STAGE in args[2] and self.cloud.checkpoint is None:
                return Exec('', 1, 'head: cannot open final.npz')
            if transport.PACKAGE_STAGE in args[2]:
                self.files.update({f'{transport.PACKAGE_STAGE}/{name}': data for name, data in self.cloud.checkpoint.items()})
            return Exec('command output\n' if 'tail -c' in args[2] else '')
        if '/opt/v1/train.py' in args:
            # The trainer runs under the provider's wrapper, which sends its output to two files; its stream carries
            # nothing. The hand-off Volume is mounted into this worker, and only this one.
            assert args[:3] == ('python', '-c', transport.TRAIN_WRAPPER) and args[4:6] == ('python', '/opt/v1/train.py')
            plan = json.loads(args[3])
            assert (plan['stdout'], plan['stderr']) == (transport.TRAINER_STDOUT, transport.TRAINER_STDERR)
            assert list(self.mounts) == [handoff.MOUNT], self.mounts
            name = args[args.index('--run-id') + 1]
            self.trainer = Trainer(self, self.local('/output/' + name), args, kwargs['env'])
            return self.trainer
        if len(args) > 2 and args[2] == transport.HANDOFF_SCRIPT:
            assert kwargs['timeout'] <= transport.PUBLISH_SECONDS_MAX
            return self.handoff(list(args))
        # A collection: the provider's own script, as the worker would run it, on this worker's files.
        assert args[:2] == ('python', '-c') and args[2] == transport.EXPORT_SCRIPT, args[:2]
        run, mode, part, out_log, err_log, *limits = args[3:]
        done = subprocess.run([sys.executable, '-c', args[2], str(self.local(run)), mode, part, str(self.local(out_log)),
                               str(self.local(err_log)), *limits], capture_output=True)
        self.collections.append(dict(mode=mode, part=part, timeout=kwargs['timeout'], bytes=len(done.stdout)))
        # As the pinned SDK does: a stream that has not finished at the command's time limit ends silently there, with
        # what had arrived, and the command is reported as ended with code -1.
        arrives = self.cloud.slow_parts.get(part, self.cloud.stream_bytes_per_second) * kwargs['timeout']
        if len(done.stdout) > arrives:
            return Exec(done.stdout[:int(arrives)], -1, '')
        return Exec(done.stdout, done.returncode, done.stderr.decode())


def real_provider(cloud, directory, patience=0.5):
    """The real Polytopia provider (its train, start wait, abandon and settlement) against the stand-in cloud."""
    release = Path(directory) / 'provider-release'
    release.mkdir()
    (release / 'frozen.txt').write_text('frozen')
    seal(release)
    value = object.__new__(transport.Provider)
    value.release, value.app, value.account, value._tokens, value._training = release, type('App', (), {'app_id': 'ap-test'}), None, {}, {}
    value.live, value.lock, value.lease_tick = {}, threading.Lock(), 0.005
    value.cpu_timeout = 300
    value.start_patience, value.start_poll, value.abandon_poll = patience, 0.001, 0.001
    value.stop_poll, value.visible_wait, value.collect_wait = 0.005, 0.001, 0.001
    stub = Provider()
    value._create, value.parameter_counts, value.preflight, value.evaluate = cloud.create, stub.parameter_counts, stub.preflight, stub.evaluate
    return value


class Workers(Case):
    """Sessions on the real Polytopia provider, with the stand-in cloud in place of Modal."""
    start_after = 0

    def make_provider(self):
        self.cloud = Cloud(start_after=self.start_after, output=Path(self.temp.name) / 'workers')
        modal = type(sys)('modal')
        modal.Sandbox = type('Sandbox', (), dict(from_id=staticmethod(self.cloud.sandboxes.__getitem__)))
        cloud = self.cloud
        self.volumes = []

        def from_name(name, **options):
            # Hand-off Volumes are folders the in-worker publish writes; lease scratch Volumes are recorded by name.
            if '-handoff-' in name:
                return Volume(cloud, name, {k: v for k, v in options.items() if k != 'environment_name'})
            self.volumes.append((name, options))
            return name
        modal.Volume = type('Volume', (), dict(from_name=staticmethod(from_name)))
        for patcher in (mock.patch.dict(sys.modules, {'modal': modal}),
                        mock.patch.object(transport, 'allocation_info', self.cloud.info),
                        mock.patch.object(transport, 'reset_rpc_channels', lambda: None)):
            patcher.start()
            self.addCleanup(patcher.stop)
        return real_provider(self.cloud, self.temp.name)

    def ledger(self, name):
        return self.row(name)


class QueuedWorkers(Workers):
    """A created GPU worker can sit in the provider's queue: the real provider waits, abandons and settles it."""
    start_after = 40

    def test_a_long_queue_then_a_start_trains_and_charges_only_the_started_lifetime(self):
        self.provider.start_patience = 30.0
        self.assertNotIn('error', self.run_job())
        self.work()
        self.assertEqual(self.jobs(), ([('a', 'train', 'complete')], []))
        sandbox = self.cloud.sandboxes['sb-training-a']
        self.assertGreater(sandbox.reads, 40)  # it waited through the queue instead of giving up
        entry = self.ledger('a')
        self.assertEqual((entry['status'], entry['seconds']), ('complete', 123.0))  # start to finish, never the queue
        self.assertFalse((self.root / 'jobs/a/abandoned.json').exists())
        self.assertEqual(self.seconds(), 123.0)
        self.assertAlmostEqual(entry['charged'], 123.0 * R)  # dollars: the lifetime at the training worker's price

    def test_a_worker_that_never_starts_fails_uncharged_and_the_session_continues(self):
        self.cloud.start_after = None
        self.assertNotIn('error', self.run_job('queued'))
        self.work()
        self.assertEqual(self.jobs(), ([('queued', 'train', 'failed')], []))  # never `uncertain`
        entry = self.ledger('queued')
        self.assertEqual((entry['status'], entry['seconds'], entry['charged'], entry['started']),
                         ('failed', 0, 0, None))
        self.assertEqual(self.cloud.terminated, ['sb-training-queued'])
        self.assertEqual(self.backend.budget()['used_usd'], 0)
        status = self.broker.handle({'op': 'status'})['jobs'][0]
        self.assertIn('never started', status['error'])
        self.assertIn('charged 0 seconds', self.broker.handle({'op': 'log', 'name': 'queued'})['failure'])
        self.cloud.start_after = 0
        self.assertNotIn('error', self.run_job('b'))  # accepted at once
        self.work()
        self.assertEqual(self.jobs()[0][-1], ('b', 'train', 'complete'))
        self.assertIn('run', self.broker.handle({'op': 'submit', 'name': 'b'})['submitted']['native'])
        self.assertTrue(self.broker.handle({'op': 'finalize'})['finalized'])
        self.assertEqual(self.seconds(), 123.0)

    def test_stop_while_queued_is_stopped_and_uncharged(self):
        self.cloud.start_after = None
        self.provider.start_patience = 30.0
        self.run_job()
        self.backend.tick()
        deadline = time.time() + 10
        while 'sb-training-a' not in self.cloud.sandboxes and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.jobs(), ([('a', 'train', 'queued')], ['a']))  # created, waiting for a GPU
        self.assertEqual(self.broker.handle({'op': 'stop', 'name': 'a'})['stopping'], 'a')
        self.backend.controller.join(timeout=10)
        self.assertEqual(self.jobs(), ([('a', 'train', 'stopped')], []))
        entry = self.ledger('a')
        self.assertEqual((entry['status'], entry['seconds'], entry['charged']), ('stopped', 0, 0))
        self.assertIn('Stopped while queued', self.broker.handle({'op': 'log', 'name': 'a'})['failure'])
        self.assertEqual(self.backend.budget()['used_usd'], 0)

    def test_a_start_racing_the_abandon_is_charged_from_its_receipt(self):
        self.cloud.start_after, self.cloud.start_on_terminate = None, True
        self.run_job()
        self.work()
        self.assertEqual(self.jobs(), ([('a', 'train', 'failed')], []))
        entry = self.ledger('a')
        self.assertEqual((entry['status'], entry['seconds']), ('failed', 123.0))
        self.assertAlmostEqual(entry['charged'], 123.0 * R)
        self.assertIn('started only after', self.broker.handle({'op': 'log', 'name': 'a'})['failure'])

    def interrupted(self, name, started):
        """A run whose controller died after creating its worker: the ledger holds it, nothing drives it."""
        self.cloud.start_after = 0 if started else None
        sandbox = self.cloud.create('training-' + name, 360, True)
        self.cloud.info('ap-test', sandbox.object_id)
        self.backend.controller.ledger.queue(name, 360., dict(kind='train', files={}, shape=dict(TRAIN_SHAPE), rate=R))
        write(self.root / 'jobs' / name / 'provider.json', dict(sandbox_id=sandbox.object_id, app_id='ap-test'))
        self.assertEqual(self.jobs(), ([(name, 'train', 'uncertain')], [name]))
        self.assertEqual(self.seconds('remaining_usd'), 240)  # its whole grant stays held
        return sandbox

    def test_recovery_settles_a_never_started_uncertain_job_at_zero(self):
        self.interrupted('a', started=False)
        self.assertIn('exceeds the remaining', self.run_job('b')['error'])
        self.backend.controller.next_recovery.clear()
        self.work()
        self.assertEqual(self.jobs(), ([('a', 'train', 'failed')], []))
        entry = self.ledger('a')
        self.assertEqual((entry['status'], entry['seconds'], entry['charged']), ('failed', 0, 0))
        self.assertIs(json.loads((self.root / 'jobs/a/recovered-interruption.json').read_text())['never_started'], True)
        self.assertEqual(self.cloud.terminated, ['sb-training-a'])
        self.assertIn('never started', self.broker.handle({'op': 'log', 'name': 'a'})['failure'])
        self.assertNotIn('error', self.run_job('b'))

    def test_recovery_terminates_a_started_abandoned_worker_now_and_charges_its_receipt(self):
        sandbox = self.interrupted('a', started=True)
        self.backend.controller.next_recovery.clear()
        self.work()
        self.assertEqual(self.jobs(), ([('a', 'train', 'failed')], []))
        self.assertEqual(self.cloud.terminated, [sandbox.object_id])  # not left to idle for its grant
        self.assertEqual(self.ledger('a')['seconds'], 123.0)

    def test_stop_resolves_an_uncertain_job_instead_of_refusing(self):
        self.interrupted('a', started=False)
        self.assertEqual(self.broker.handle({'op': 'stop', 'name': 'a'}), {'stopped': 'a', 'charged_usd': 0})
        self.assertEqual(self.jobs(), ([('a', 'train', 'stopped')], []))
        self.assertEqual((self.ledger('a')['seconds'], self.ledger('a')['status']), (0, 'stopped'))
        self.assertIn('Stopped while queued', self.broker.handle({'op': 'log', 'name': 'a'})['failure'])
        self.assertNotIn('error', self.run_job('b'))


class EarlyEnds(Workers):
    """A training run the researcher stops, or one cut off at its time limit, keeps the last network it wrote whole
    and is complete (`stopped_early`, `cut_short`); one that wrote none fails. A closing session keeps nothing."""
    NEAR_LIMIT = 360 - transport.TEARDOWN_SECONDS - transport.COLLECT_SECONDS + 1  # a 360 s run past its trainer's limit

    def start(self, trainer, exports=(), run_age=0.0, backend=None):
        """Queue run `a` (or `final` on another backend) and wait until its training command is running."""
        self.cloud.trainer, self.cloud.exports, self.cloud.run_age = trainer, exports, run_age
        if backend is None:
            self.assertNotIn('error', self.run_job())
        (backend or self.backend).tick()
        self.assertTrue(self.cloud.training.wait(timeout=10))

    def job(self, name='a', backend=None):
        (backend or self.backend).controller.join(timeout=20)
        return next(job for job in (backend or self.backend).status()['jobs'] if job['name'] == name)

    def kept(self, name='a', root=None):
        """The value the kept policy holds (write_run: an export holds its update number, the trainer's own final 100)."""
        import numpy as np
        with np.load((root or self.root) / 'jobs' / name / 'bundle/final.npz') as weights:
            return float(weights['p0'][0, 0])

    def usable(self, name='a'):
        """A kept network is an ordinary complete run: it can be practised and submitted."""
        self.assertEqual(self.broker.handle({'op': 'eval', 'name': name})['queued'], f'eval-{name}-1')
        self.work()
        self.assertEqual(self.backend.status()['latest_practice']['run'], name)
        self.assertEqual(self.broker.handle({'op': 'submit', 'name': name})['submitted']['native']['run'], name)

    def test_a_stop_request_ends_the_trainer_at_its_next_step_and_the_run_is_complete_stopped_early(self):
        self.start('stops')
        self.assertEqual(self.jobs(), ([('a', 'train', 'running')], ['a']))
        sandbox = self.cloud.sandboxes['sb-training-a']
        self.assertEqual(sandbox.trainer.env['POLYTOPIA_STOP_REQUEST'], transport.STOP_REQUEST)  # the trainer is told
        reply = self.broker.handle({'op': 'stop', 'name': 'a'})
        self.assertEqual(reply['stopping'], 'a')
        self.assertIn('stopped_early', reply['note'])
        job = self.job()
        self.assertEqual((job['status'], job.get('stopped_early'), job.get('cut_short'), job['worker_seconds']),
                         ('complete', True, None, 123.0), job)
        self.assertEqual(sandbox.files[transport.STOP_REQUEST], 'stop\n')  # asked, not terminated: it ended by itself
        self.assertEqual(sandbox.trainer.returncode, 0)
        record = json.loads((self.root / 'jobs/a/export.json').read_text())
        self.assertEqual((record['ended'], record['network'], self.kept()), ('stopped_early', 'final', 100.0))
        self.assertEqual(self.ledger('a')['status'], 'complete')
        self.assertIs(self.broker.handle({'op': 'log', 'name': 'a'})['stopped_early'], True)
        self.assertEqual(self.jobs(), ([('a', 'train', 'complete')], []))
        self.usable()

    def test_a_trainer_that_ignores_the_stop_is_ended_after_the_grace_and_keeps_its_last_whole_export(self):
        self.provider.stop_grace = 0.05
        self.start('ignores', exports=(1, 2))
        self.broker.handle({'op': 'stop', 'name': 'a'})
        job = self.job()
        self.assertEqual((job['status'], job.get('stopped_early')), ('complete', True), job)
        self.assertIsNone(self.cloud.sandboxes['sb-training-a'].trainer.returncode)  # it never ended by itself
        self.assertEqual(self.cloud.terminated, ['sb-training-a'])
        # The newest whole export, judged on the updates it holds: the log's third row and its cut line are later.
        self.assertEqual((self.kept(), job['learning']['updates'], job['learning']['ended']), (2.0, 2, 'stopped_early'))
        self.assertEqual(json.loads((self.root / 'jobs/a/export.json').read_text())['last_export']['update'], 2)
        self.assertFalse((self.root / 'jobs/a/bundle/training-complete.json').exists())
        # The host published its newest whole export from the still-running worker, through the hand-off Volume.
        handoffs = self.cloud.sandboxes['sb-training-a'].handoffs
        self.assertEqual([(h['kind'], h['mode'], h['chained']) for h in handoffs], [('network', 'early', False)])
        self.assertEqual(json.loads((self.root / 'jobs/a/export.json').read_text())['handoff'], 'volume')
        self.usable()

    def test_grace_expiry_without_a_whole_export_falls_back_to_termination_and_the_run_is_lost(self):
        self.provider.stop_grace = 0.05
        self.start('ignores')
        self.broker.handle({'op': 'stop', 'name': 'a'})
        job = self.job()
        self.assertEqual((job['status'], job['worker_seconds']), ('stopped', 123.0), job)
        self.assertNotIn('stopped_early', job)
        self.assertIn('no network to keep', job['error'])
        self.assertEqual(self.cloud.terminated, ['sb-training-a'])
        self.assertAlmostEqual(self.ledger('a')['charged'], 123.0 * R)  # stopped, and its worker time is charged
        self.assertIn('error', self.broker.handle({'op': 'submit', 'name': 'a'}))
        self.assertIn('error', self.broker.handle({'op': 'eval', 'name': 'a'}))

    def test_a_run_cut_off_at_its_time_limit_is_packaged_from_its_last_whole_export_as_cut_short(self):
        self.start('cut', exports=(1, 2), run_age=self.NEAR_LIMIT)
        folder = self.cloud.sandboxes['sb-training-a'].local('/output/a')
        # A newer export whose checkpoint does not match its own record is not whole, and one still being written
        # has not been renamed into place: neither is taken.
        torn = folder / 'exports/update-000000003'
        shutil.copytree(folder / 'exports/update-000000002', torn)
        (torn / 'final.npz').write_bytes((torn / 'final.npz').read_bytes()[:40])
        shutil.copytree(folder / 'exports/update-000000002', folder / 'exports/writing-4')
        job = self.job()
        self.assertEqual((job['status'], job.get('cut_short'), job.get('stopped_early'), job['worker_seconds']),
                         ('complete', True, None, 123.0), job)
        self.assertEqual((self.kept(), job['learning']['updates']), (2.0, 2))
        record = json.loads((self.root / 'jobs/a/export.json').read_text())
        self.assertEqual((record['ended'], record['network']), ('cut_short', 'last complete export'))
        self.assertNotIn(transport.STOP_REQUEST, self.cloud.sandboxes['sb-training-a'].files)
        self.assertIs(self.broker.handle({'op': 'status'})['jobs'][0]['cut_short'], True)  # visible to the researcher
        handoffs = self.cloud.sandboxes['sb-training-a'].handoffs
        self.assertEqual([(h['kind'], h['mode'], h['chained']) for h in handoffs], [('network', 'early', False)])
        self.usable()

    def test_a_run_cut_off_before_it_wrote_a_whole_export_fails(self):
        self.start('cut', run_age=self.NEAR_LIMIT)
        job = self.job()
        self.assertEqual((job['status'], job['worker_seconds']), ('failed', 123.0), job)
        self.assertNotIn('cut_short', job)
        self.assertIn('cut off at its time limit', job['error'])
        self.assertIn('no network to keep', job['error'])
        self.assertIn('error', self.broker.handle({'op': 'submit', 'name': 'a'}))

    def test_a_trainer_that_fails_by_itself_stays_failed_whatever_it_exported(self):
        self.start('fails', exports=(1,))
        job = self.job()
        self.assertEqual(job['status'], 'failed', job)
        self.assertIn('Training failed', job['error'])
        self.assertFalse((self.root / 'jobs/a/bundle').exists())

    def test_an_unchanged_or_non_finite_export_is_still_refused(self):
        self.start('cut', exports=(0,), run_age=self.NEAR_LIMIT)  # write_run: export 0 holds the initial weights
        job = self.job()
        self.assertEqual(job['status'], 'failed', job)
        self.assertIn('Learning evidence failed', job['error'])

    def test_closing_the_session_ends_a_training_worker_at_once_and_keeps_nothing(self):
        self.start('ignores', exports=(1, 2))
        self.backend.settle()
        job = self.job()
        self.assertEqual((job['status'], job['worker_seconds']), ('stopped', 123.0), job)
        self.assertNotIn(transport.STOP_REQUEST, self.cloud.sandboxes['sb-training-a'].files)  # not asked: ended
        self.assertEqual(self.cloud.terminated, ['sb-training-a'])
        self.assertFalse((self.root / 'jobs/a/bundle').exists())

    def test_fresh_final_training_that_is_cut_off_but_exported_counts_as_trained(self):
        """harness.session.final_training takes a `complete` job named final; `cut_short` rides on it."""
        final = self.open(self.out / 'final', budget_seconds=360)
        workspace.apply(final.release, final.workspace, self.files)
        final.run('final', 360, 'configs/smoke.json', 1)
        self.start('cut', exports=(5,), run_age=self.NEAR_LIMIT, backend=final)
        job = self.job('final', final)
        self.assertEqual((job['status'], job.get('cut_short')), ('complete', True), job)
        self.assertEqual(self.kept('final', self.out / 'final'), 5.0)
        # The fresh final has its own hand-off Volume, bound to its own backend root.
        binding = json.loads((self.out / 'final' / handoff.BINDING).read_text())
        self.assertEqual(binding['root'], str(self.out / 'final'))
        self.assertEqual(self.cloud.sandboxes['sb-training-final'].mounts[handoff.MOUNT].name, binding['volume'])
        self.assertNotEqual(binding['volume'], handoff.SessionVolume('polytopia', self.root).name)
        self.assertEqual(final.submit('final')['run'], 'final')
        self.assertEqual(final.finalize()['finalized'], True)


class Collection(Workers):
    """Collecting a finished run from its worker. On 2026-11-13 every time-limited final of more than a few minutes
    failed with a bare "Bounded archive export failed": the host read the trainer command's whole output stream
    first, that read ran into the command's time limit (where the SDK ends a stream silently), and the archive was
    left 19 seconds, which the SDK cut the same way. Now nothing waits on that stream, the archive has its own
    window, and a failed collection says why. Later that day all nine one-hour trainings lost their network to a
    16 KB/s host link: the run now travels through the session's hand-off Volume, and the live read (the tests
    marked `live`, with a publish that fails) is only its fallback."""
    WORST_RATE = 380_000  # bytes a second: the slowest archive stream measured that day (9.7 MB in 25.4 s)
    SLOW_LINK = 16_000    # bytes a second: the host's link to Modal when the nine one-hour trainings were lost

    def train(self, **cloud):
        for key, value in cloud.items():
            setattr(self.cloud, key, value)
        self.assertNotIn('error', self.run_job())
        self.work()
        job = self.backend.status()['jobs'][0]
        return job, self.root / 'jobs/a', self.cloud.sandboxes['sb-training-a']

    def test_schedule_leaves_the_collection_its_own_window(self):
        """The timing budget, for a trainer command started 8 s after the worker (measured 16-32 s to the loop)."""
        self.assertEqual((transport.TEARDOWN_SECONDS, transport.COLLECT_SECONDS, transport.TRAINER_MARGIN_SECONDS),
                         (15, 65, 115))
        for grant, seconds in ((900, 762), (3600, 3462), (6000, 5862), (300, 162)):
            plan = transport.schedule(1000.0, grant, 1008.0)
            self.assertEqual((plan['deadline'] - 1000, plan['limit'] - 1000, plan['seconds']),
                             (grant - 15, grant - 80, seconds), grant)
            # The trainer's loop ends `seconds` after it began; with 25 s to start and 3 s to exit, it has ended by
            # itself 22 s before its command's limit, and the archive stream that starts then may take 82 s.
            ended = 1008.0 + 25 + plan['seconds'] + 3
            self.assertEqual(plan['limit'] - ended, 22)
            self.assertEqual(transport.collect_seconds(plan['deadline'], ended), 82)
            # A trainer cut off at its command's limit still leaves a full minute.
            self.assertEqual(transport.collect_seconds(plan['deadline'], plan['limit']), 60)
        self.assertEqual(transport.schedule(1000.0, 360, 1008.0, smoke=True)['seconds'], 180)  # the smoke run
        self.assertEqual(transport.collect_seconds(5000.0, 1000.0), transport.COLLECT_SECONDS_MAX)

    def test_the_trainer_command_stream_is_not_read_and_its_output_arrives_as_bounded_tails(self):
        job, folder, sandbox = self.train(stdout_lines=60_000)  # 540 KB of progress lines, then the RESULT line
        self.assertEqual(job['status'], 'complete', job)       # Unread.read would have failed the run
        log = (folder / 'stdout.log').read_bytes()
        self.assertEqual(len(log), transport.STDOUT_TAIL_BYTES)
        self.assertTrue(log.endswith(b'RESULT {"ok": true}\n'))
        self.assertEqual((folder / 'stderr.log').read_text(), '')
        self.assertIn('RESULT', self.broker.handle({'op': 'log', 'name': 'a'})['stdout'])
        self.assertFalse((folder / 'bundle/stdout.tail').exists())  # logs are the job's, not part of the sealed bundle
        self.assertEqual(verify(folder / 'bundle'), self.broker.handle({'op': 'submit', 'name': 'a'})['submitted']['native']['sha256'])

    def test_a_one_hour_run_fits_its_window_several_times_over_at_the_slowest_rate_measured(self):
        """5,600 updates, both clock snapshots, real file sizes, and a stream of 0.38 MB a second. The trainer ended
        by itself 33 s before its command's limit, as the failed 3,600-second finals did."""
        age = 360 - transport.TEARDOWN_SECONDS - transport.COLLECT_SECONDS - 33
        job, folder, sandbox = self.train(updates=5600, clocks=(5, 10), real_sizes=True, run_age=age,
                                          stream_bytes_per_second=self.WORST_RATE, volume_fails='publish')  # live
        self.assertEqual(job['status'], 'complete', job)
        network, extras = sandbox.collections
        self.assertEqual((network['part'], extras['part']), ('network', 'extras'))
        self.assertGreaterEqual(network['timeout'], 85)        # what was 19 s
        log_bytes = (folder / 'bundle/training.jsonl').stat().st_size
        self.assertGreater(log_bytes, 6_000_000)               # the hour's training log, about 7 MB ...
        self.assertLess(network['bytes'], 5_000_000)           # ... travels compressed: the whole stream is under 5 MB
        self.assertLess(network['bytes'] / self.WORST_RATE, network['timeout'] / 6)
        self.assertLess(extras['bytes'] / self.WORST_RATE, extras['timeout'] / 5)
        record = json.loads((folder / 'export.json').read_text())
        self.assertEqual(len(record['clock_snapshots']['collected']), 10)
        self.assertEqual([c['returncode'] for c in record['collections']], [0, 0])
        self.assertEqual((record['handoff'], [c['handoff'] for c in record['collections']]), ('live', ['live', 'live']))
        self.assertIn('the publish step ended with code 1', record['handoff_fallback'])
        self.assertEqual(job['learning']['updates'], 5600)
        self.assertTrue((folder / 'bundle/clock-10m.npz').is_file())

    def test_a_one_hour_run_survives_the_slowest_host_link_through_the_hand_off_volume(self):
        """The run of the test above over a 16 KB/s link, which no collection window could carry: the worker
        publishes it to the Volume itself, is ended, and only then does anything cross the host's link."""
        age = 360 - transport.TEARDOWN_SECONDS - transport.COLLECT_SECONDS - 33
        job, folder, sandbox = self.train(updates=5600, clocks=(5, 10), real_sizes=True, run_age=age,
                                          stream_bytes_per_second=self.SLOW_LINK)
        self.assertEqual(job['status'], 'complete', job)
        self.assertEqual(sandbox.collections, [])  # nothing was read live out of the worker
        # The trainer's wrapper published the run inside its own command; the host's publish found it done.
        chained, host = sandbox.handoffs
        self.assertEqual((chained['chained'], chained['kind'], chained['mode'], chained['returncode']),
                         (True, 'network', 'final', 0))
        self.assertEqual((host['chained'], host['reused'], host['alive']), (False, True, True))
        self.assertEqual(chained['job'], 'a.sb-training-a')
        # While the worker lived, the host read only the small manifest (is the publish visible?); every file was read
        # after the worker had ended, and the job left the Volume afterwards.
        early = [path for path, ended in self.cloud.volume_reads if not ended]
        self.assertEqual(early, ['a.sb-training-a/MANIFEST.json'])
        reads = [path for path, ended in self.cloud.volume_reads if ended]
        self.assertEqual(sorted(path for path in reads if '/files/' in path),
                         ['a.sb-training-a/files/' + name for name in sorted(transport.HANDOFF_FILES)])
        route = json.loads((folder / 'handoff.json').read_text())
        self.assertEqual((route['handoff'], route['worker_ended_before_collection'], route['removed_from_volume']),
                         ('volume', True, True))
        self.assertFalse((self.cloud.volumes / route['volume'] / 'a.sb-training-a').exists())
        self.assertEqual(sorted(p.name for p in (folder / 'handoff').iterdir()), ['status.json'])
        record = json.loads((folder / 'export.json').read_text())
        self.assertEqual((record['handoff'], record['handoff_fallback']), ('volume', None))
        self.assertEqual(len(record['clock_snapshots']['collected']), 10)
        self.assertEqual(job['learning']['updates'], 5600)
        self.assertEqual(verify(folder / 'bundle'), self.broker.handle({'op': 'submit', 'name': 'a'})['submitted']['native']['sha256'])
        # Only the training worker mounts the Volume, which is this session's own and created as a v2 Volume.
        volume = sandbox.mounts[handoff.MOUNT]
        self.assertEqual((volume.name, volume.options), (route['volume'], dict(version=2, create_if_missing=True)))
        self.assertTrue(sandbox.prepared)
        self.assertEqual(json.loads((folder / 'provider.json').read_text())['handoff_job'], 'a.sb-training-a')

    def test_a_publish_the_volume_does_not_show_falls_back_to_the_live_read_while_the_worker_is_alive(self):
        job, folder, sandbox = self.train(volume_fails='invisible')
        self.assertEqual(job['status'], 'complete', job)
        self.assertEqual([c['part'] for c in sandbox.collections], ['network'])
        record = json.loads((folder / 'export.json').read_text())
        self.assertEqual(record['handoff'], 'live')
        self.assertIn('did not show in the hand-off Volume', record['handoff_fallback'])
        route = json.loads((folder / 'handoff.json').read_text())
        self.assertEqual(route['publish']['returncode'], 0)
        self.assertNotIn('worker_ended_before_collection', route)  # it was still alive for the live read

    def test_a_published_run_the_host_cannot_read_after_the_worker_ended_fails_and_stays_in_the_volume(self):
        job, folder, sandbox = self.train(volume_fails='unreadable')
        self.assertEqual(job['status'], 'failed', job)
        self.assertIn('Bounded archive export failed: the run was published to the hand-off Volume', job['error'])
        self.assertEqual(sandbox.collections, [])  # the worker had ended: no live read
        route = json.loads((folder / 'handoff.json').read_text())
        self.assertEqual((route['handoff'], route['worker_ended_before_collection']), ('volume', True))
        self.assertTrue((self.cloud.volumes / route['volume'] / 'a.sb-training-a/MANIFEST.json').is_file())
        self.assertEqual(json.loads((folder / 'export-error.json').read_text())['handoff'], 'volume')

    def test_the_clock_snapshots_are_not_needed_for_the_run_to_count(self):
        job, folder, sandbox = self.train(clocks=(5, 10), slow_parts={'extras': 1.0}, volume_fails='publish')  # live
        self.assertEqual(job['status'], 'complete', job)
        record = json.loads((folder / 'export.json').read_text())
        self.assertEqual(len(record['clock_snapshots']['in_worker']), 10)
        self.assertEqual(record['clock_snapshots']['collected'], [])
        self.assertIn('cut off at its', record['collections'][1]['not_kept'])
        self.assertFalse(list((folder / 'bundle').glob('clock-*')))
        self.assertEqual(verify(folder / 'bundle'), self.broker.handle({'op': 'submit', 'name': 'a'})['submitted']['native']['sha256'])

    def test_a_collection_cut_off_at_its_time_limit_says_so_with_the_bytes_that_arrived(self):
        job, folder, sandbox = self.train(real_sizes=True, slow_parts={'network': 100.0}, volume_fails='publish')  # live
        self.assertEqual(job['status'], 'failed', job)
        self.assertRegex(job['error'], r'Bounded archive export failed: the collection was cut off at its 240-second time '
                                       r'limit \(24000 bytes had arrived after \d+ s\)')
        failure = json.loads((folder / 'export-error.json').read_text())
        first = failure['collections'][0]
        self.assertEqual((first['part'], first['returncode'], first['reached_time_limit'], first['bytes']),
                         ('network', -1, True, 24000))
        self.assertEqual(failure['trainer_returncode'], 0)
        # The trainer's own last lines are still fetched (a short stream of their own), and the researcher sees both.
        self.assertTrue((folder / 'stdout.log').read_text().endswith('RESULT {"ok": true}\n'))
        log = self.broker.handle({'op': 'log', 'name': 'a'})
        self.assertIn('cut off at its 240-second time limit', log['failure'])
        self.assertIn('RESULT', log['stdout'])
        self.assertFalse((folder / 'bundle').exists())

    def test_a_collection_command_that_fails_reports_what_it_wrote_to_stderr(self):
        job, folder, _ = self.train(damage='final.npz')  # the final checkpoint does not match its own record
        self.assertEqual(job['status'], 'failed', job)
        self.assertIn('Bounded archive export failed: the collection command ended with code 1: training-complete.json '
                      'is there, but final.npz / final-critic.npz and their records are missing or do not match', job['error'])
        failure = json.loads((folder / 'export-error.json').read_text())
        self.assertIn('do not match their digests', failure['collections'][0]['stderr'])

    def test_a_trainer_that_exits_cleanly_without_its_own_end_is_named_as_that(self):
        job, folder, _ = self.train(trainer='quits', exports=(1, 2))
        self.assertEqual(job['status'], 'failed', job)
        self.assertIn('no complete policy export: the trainer ended without training-complete.json', job['error'])
        self.assertEqual(json.loads((folder / 'trainer.json').read_text())['returncode'], 0)

    def test_a_trainer_that_fails_by_itself_leaves_its_own_last_output(self):
        job, folder, sandbox = self.train(trainer='fails')
        self.assertEqual(job['status'], 'failed', job)
        self.assertIn('Training failed', job['error'])
        self.assertEqual((folder / 'stderr.log').read_text(), 'Traceback: own learner\n')
        self.assertEqual([(h['kind'], h['chained']) for h in sandbox.handoffs], [('logs', False)])  # the tails only
        self.assertEqual(sandbox.collections, [])
        self.assertEqual(self.broker.handle({'op': 'log', 'name': 'a'})['stderr'], 'Traceback: own learner')

    def test_a_trainer_that_fails_by_itself_leaves_its_own_last_output_through_the_live_fallback(self):
        job, folder, sandbox = self.train(trainer='fails', volume_fails='publish')
        self.assertEqual(job['status'], 'failed', job)
        self.assertEqual((folder / 'stderr.log').read_text(), 'Traceback: own learner\n')
        self.assertEqual([c['part'] for c in sandbox.collections], ['logs'])
        self.assertLessEqual(sandbox.collections[0]['timeout'], transport.LOGS_SECONDS)

    def test_a_cut_off_trainer_whose_export_cannot_be_collected_is_not_called_exportless(self):
        age = 360 - transport.TEARDOWN_SECONDS - transport.COLLECT_SECONDS + 1
        job, folder, _ = self.train(trainer='cut', exports=(1, 2), run_age=age, slow_parts={'network': 1.0},
                                    volume_fails='publish')  # live
        self.assertEqual(job['status'], 'failed', job)
        self.assertIn('Training was cut off at its time limit, and its last export could not be collected: the collection '
                      'was cut off at its', job['error'])
        self.assertNotIn('no network to keep', job['error'])

    def test_the_trainer_wrapper_prepares_then_trains_then_publishes_inside_its_own_command(self):
        """TRAIN_WRAPPER as the worker runs it: a failed prepare is recorded, not fatal; the trainer's output goes to
        its two files; a trainer that succeeds is published by the wrapper itself; its code is the trainer's."""
        folder = Path(self.temp.name) / 'wrapper'
        staging, marker = folder / 'staging', folder / 'published'
        def plan(code):
            return json.dumps(dict(prepare=[sys.executable, '-c', 'raise SystemExit("no /handoff here")'],
                                   staging=str(staging), stdout=str(folder / 'out'), stderr=str(folder / 'err'),
                                   handoff=[sys.executable, '-c', f'open({str(marker)!r}, "w").write("yes")'],
                                   handoff_seconds=30))
        folder.mkdir()
        trainer = [sys.executable, '-c', 'import sys;print("trained");print("warn",file=sys.stderr);sys.exit(int(sys.argv[1]))']
        done = subprocess.run([sys.executable, '-c', transport.TRAIN_WRAPPER, plan(0), *trainer, '0'], capture_output=True)
        self.assertEqual((done.returncode, done.stdout, done.stderr), (0, b'', b''))
        self.assertEqual(((folder / 'out').read_text(), (folder / 'err').read_text()), ('trained\n', 'warn\n'))
        self.assertIn('no /handoff here', (staging / 'prepare-error.txt').read_text())
        self.assertEqual(marker.read_text(), 'yes')
        marker.unlink()
        done = subprocess.run([sys.executable, '-c', transport.TRAIN_WRAPPER, plan(0), *trainer, '3'], capture_output=True)
        self.assertEqual(done.returncode, 3)
        self.assertFalse(marker.exists())  # a trainer that failed is left to the host

    def test_an_archive_the_host_cannot_unpack_stays_in_the_volume(self):
        """A finished run whose archive the host refuses (here: a member over the size cap, as the training log of
        every 8-hour starter final was under the old 50 MiB cap) fails, and its published files are kept: in the
        hand-off Volume and beside the job. They are the only copy of the run once the worker has ended."""
        age = 360 - transport.TEARDOWN_SECONDS - transport.COLLECT_SECONDS - 33
        with mock.patch.object(transport, 'MEMBER_BYTES_MAX', 64):
            job, folder, sandbox = self.train(updates=5600, clocks=(5, 10), real_sizes=True, run_age=age,
                                              stream_bytes_per_second=self.SLOW_LINK)
        self.assertEqual(job['status'], 'failed', job)
        self.assertRegex(job['error'], r'archive member \S+ is \d+ bytes, over the 0 MiB cap')
        route = json.loads((folder / 'handoff.json').read_text())
        self.assertEqual(route['removed_from_volume'], False)
        self.assertIn('network', route['kept_unread'])
        published = self.cloud.volumes / route['volume'] / 'a.sb-training-a' / 'files'
        self.assertTrue((published / 'network.tgz').is_file())
        self.assertTrue((folder / 'handoff/network.tgz').is_file())
        # the kept archive is the run: with the real cap it unpacks to the trained network
        self.assertIn('final.npz', transport.unpack((folder / 'handoff/network.tgz').read_bytes()))

    def test_unpack_refuses_a_cut_or_altered_archive(self):
        import gzip
        import tarfile
        sandbox = Sandbox(self.cloud, 'sb-direct')
        write_run(sandbox.local('/output/a'), final=False)
        proc = sandbox.exec('python', '-c', transport.EXPORT_SCRIPT, '/output/a', 'final', 'network',
                            transport.TRAINER_STDOUT, transport.TRAINER_STDERR, '100', '100', text=False, timeout=30)
        payload = proc.stdout.read()
        files = transport.unpack(payload)
        self.assertTrue(set(transport.ALWAYS + transport.NETWORK) <= set(files))
        with self.assertRaisesRegex(ValueError, r'incomplete or damaged \(\d+ bytes arrived\)'):
            transport.unpack(payload[:len(payload) // 2])       # what a stream cut at its time limit delivers
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w') as archive:
            for name, data in dict(files, **{'final.npz': b'other weights'}).items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        with self.assertRaisesRegex(ValueError, 'checksum mismatch for final.npz'):
            transport.unpack(gzip.compress(buffer.getvalue()))


class FullSession(Case):
    smoke, budget, bind = False, None, False

    def prepare_release(self):
        write(self.release / 'host-private-panel/inventory.json',
              dict(panel=[dict(name=name, directory='host-private-panel/' + name) for name in PANEL]))

    def test_check_refuses_a_segment_the_ring_cannot_hold(self):
        def with_segment(segment, ring):
            recipe = json.loads(self.files['configs/train.json'])
            recipe['segment'] = segment
            recipe['trainer']['ring'] = ring
            return dict(self.files, **{'configs/train.json': json.dumps(recipe)})
        for segment, ring in ((64, 160), (64, 192), (48, 239)):  # the first two each lost a run on 2026-11-14
            reply = self.broker.handle(dict(op='check', files=with_segment(segment, ring), config='configs/train.json'))
            self.assertIn(f'trainer.ring >= {5 * segment}', reply.get('error', ''))
        self.assertEqual(self.backend.budget()['used_usd'], 0)
        for segment, ring in ((64, 320), (16, 160)):
            reply = self.broker.handle(dict(op='check', files=with_segment(segment, ring), config='configs/train.json'))
            self.assertNotIn('error', reply)
        self.assertNotIn('error', self.broker.handle(dict(op='check', files=self.files, config='configs/train.json')))

    def test_refused_finalize_quotes_dollars_not_seconds_of_the_cheapest_lease(self):
        out = Path(self.temp.name).resolve() / 'budgeted'
        out.mkdir()
        broker = host_runs(Broker(self.open(out / 'research', budget_usd=23.0), out, 'network', research_usd=23.0,
                                  use_budget=True))
        refused = broker.handle({'op': 'finalize'})['error']
        self.assertIn('Not yet: $23.00 of your $23.00 research allocation is still usable', refused)
        self.assertNotIn('seconds of worker time', refused)

    def test_budget_shows_held_grants_apart_from_spent_money(self):
        self.backend.run('a', 3600, 'configs/train.json')
        budget = self.backend.budget()  # queued: the whole grant is held, nothing is spent
        self.assertEqual((budget['spent_usd'], set(budget['held_by_job'])), (0, {'a'}))
        self.assertAlmostEqual(budget['held_usd'], 3600 * R)
        self.assertAlmostEqual(budget['remaining_usd'], 23.0 - 3600 * R)
        self.assertIn('NOT spent', budget['reading'])
        self.work()
        budget = self.backend.budget()  # ended: charged for the seconds its worker lived, the rest came back
        self.assertEqual((budget['held_usd'], budget['held_by_job']), (0, {}))
        self.assertAlmostEqual(budget['spent_usd'], budget['used_usd'])
        self.assertLess(budget['spent_usd'], 3600 * R)
        self.assertAlmostEqual(budget['remaining_usd'], 23.0 - budget['spent_usd'])

    def test_contract_budget_public_practice_and_private_panel(self):
        budget = self.backend.budget()
        self.assertEqual((budget['budget_usd'], registry.spec('polytopia')['research_usd']), (23.0, 23.0))
        self.assertEqual(budget['affordable_seconds']['lease --gpu 1 --cpu 4 --memory 16'], int(23.0 / R))
        self.assertNotIn('error', self.run_job(seconds=3600, config='configs/train.json'))
        self.work()
        self.assertEqual(self.provider.trained[0]['files']['recipe.json'], self.files['configs/train.json'])
        self.assertNotIn('error', self.broker.handle({'op': 'eval', 'name': 'a'}))
        self.work()
        practice = self.backend.status()['latest_practice']
        self.assertEqual((practice['games'], set(practice['by_opponent'])), (4 * 64 * 2, set(PANEL)))
        public = sorted(seed for call in self.provider.evaluated for seed in call['seeds'])
        self.assertEqual(public, sorted(list(range(700000, 700064)) * 4))
        self.assertTrue(all(len(call['seeds']) <= 4 for call in self.provider.evaluated))
        self.assertNotIn('error', self.run_job('b', 600, config='configs/train.json'))  # no single-run limit here
        self.work()
        self.broker.handle({'op': 'submit', 'name': 'b'})
        self.assertEqual(self.broker.handle({'op': 'finalize'})['submission']['name'], 'b')
        self.assertEqual(self.seconds(), 400)
        final = next(point for point in self.backend.controller.ledger.public()['points'] if float(point['fraction']) == 1)
        self.assertEqual(self.backend.controller.ledger.submissions()[final['submission'] - 1]['job'], 'b')
        before = len(self.provider.evaluated)
        result = self.backend.grade()
        calls = self.provider.evaluated[before:]
        schedule = json.loads((self.root / 'private/schedule.json').read_text())['seeds']
        self.assertEqual((len(schedule), result['games']), (128, 4 * 128 * 2))
        self.assertTrue(all(seed >= 1_000_000 for seed in schedule))  # disjoint from the public practice maps
        self.assertEqual({call['opponent'] for call in calls}, set(PANEL))
        self.assertEqual(sorted(seed for call in calls for seed in call['seeds']), sorted(schedule * 4))
        self.assertTrue(all(call['out'].startswith(str(self.root / 'private')) for call in calls))

    def test_practice_runs_alongside_training_in_both_directions(self):
        self.assertNotIn('error', self.run_job(seconds=600, config='configs/train.json'))
        self.work()
        # Practice holds its CPU lane; a training run is still accepted and trains to completion meanwhile.
        practice_gate, evaluate = threading.Event(), self.provider.evaluate
        self.provider.evaluate = lambda *args, **kwargs: (practice_gate.wait(timeout=30), evaluate(*args, **kwargs))[1]
        self.assertEqual(self.broker.handle({'op': 'eval', 'name': 'a'})['queued'], 'eval-a-1')
        self.assertNotIn('error', self.run_job('b', 600, config='configs/train.json'))
        self.backend.tick()
        self.backend.controller.threads['b'].join(timeout=20)
        self.assertEqual(self.jobs(), ([('a', 'train', 'complete'), ('eval-a-1', 'practice', 'running'),
                                        ('b', 'train', 'complete')], ['eval-a-1']))
        self.assertIn('One practice job at a time', self.broker.handle({'op': 'eval', 'name': 'b'})['error'])
        self.assertIn('Resolve active work', self.broker.handle({'op': 'finalize'})['error'])
        practice_gate.set()
        self.work()
        # Training holds the GPU lane; practice of a completed run starts and finishes meanwhile.
        self.provider.gate = threading.Event()
        self.provider.entered.clear()
        self.assertNotIn('error', self.run_job('c', 600, config='configs/train.json'))
        self.backend.tick()
        self.assertTrue(self.provider.entered.wait(timeout=10))
        self.assertEqual(self.broker.handle({'op': 'eval', 'name': 'b'})['queued'], 'eval-b-1')
        self.backend.tick()
        self.backend.controller.threads['eval-b-1'].join(timeout=20)
        jobs = dict((name, status) for name, _, status in self.jobs()[0])
        self.assertEqual((jobs['c'], jobs['eval-a-1'], jobs['eval-b-1']), ('running', 'complete', 'complete'))
        self.provider.gate.set()
        self.work()
        self.assertEqual(self.jobs()[1], [])
        self.assertEqual(self.seconds(), 600)  # practice is never charged to the budget
        practice = self.row('eval-b-1')
        self.assertEqual((practice['status'], practice['charged']), ('complete', 0))


class Leases(Case):
    """Leases, parallel jobs and the dollar budget on the real controller and ledger; the provider is the stub."""
    smoke, budget, bind = False, 6000, False
    prepare_release = FullSession.prepare_release

    def lease(self, name='gpu1', seconds=120, **shape):
        reply = self.broker.handle(dict({'op': 'lease', 'name': name, 'seconds': seconds, 'gpu': 1, 'cpu': 4,
                                         'memory': 16, 'files': self.files}, **shape))
        self.assertNotIn('error', reply, reply)
        self.backend.tick()
        deadline = time.time() + 10
        while not self.provider.lease_live(name) and time.time() < deadline:
            time.sleep(0.01)
        return reply

    def test_a_lease_runs_commands_beside_other_jobs_and_its_checkpoint_is_packaged_practised_and_submitted(self):
        reply = self.lease(cpu=8, memory=32)
        shape = {'gpu': 1, 'cpu_cores': 8, 'memory_mib': 32768}
        expected = rate(shape)
        self.assertEqual((reply['usd_per_second'], reply['shape']), (expected, shape))
        self.assertAlmostEqual(reply['reserved_usd'], 120 * expected)
        self.assertGreater(expected, R)  # more cores and memory than the 4-core training worker cost more per second
        self.assertEqual(self.provider.live['gpu1']['files'], self.files)  # the editable files arrive with the lease
        status = self.broker.handle({'op': 'status'})['jobs'][0]
        self.assertEqual((status['kind'], status['status'], status['ready'], status['shape']),
                         ('lease', 'running', True, shape))

        command = 'python experiments/gpu_resident_train.py --config configs/train.json --run-root /scratch/runs'
        self.assertEqual(self.broker.handle({'op': 'exec', 'name': 'gpu1', 'command': command}),
                         {'exec': '0001', 'state': 'running'})
        done = self.broker.handle({'op': 'poll', 'name': 'gpu1', 'exec': '0001'})
        self.assertEqual((done['state'], done['exit_code']), ('exited', 0))
        edited = dict(self.files, **{'training/policy.py': self.files['training/policy.py'] + '\n# pushed\n'})
        self.assertNotIn('error', self.broker.handle({'op': 'push', 'name': 'gpu1', 'files': edited}))
        self.assertEqual(self.provider.live['gpu1']['files'], edited)
        self.assertEqual(self.broker.handle({'op': 'sync', 'name': 'gpu1', 'reload': True})['reloaded'], True)
        self.assertTrue((self.out / 'sources/gpu1/pushes/001/training/policy.py').read_text().endswith('# pushed\n'))

        # A training run is accepted beside the lease; the session's GPU limit still holds.
        self.assertNotIn('error', self.run_job('side', 600, config='configs/train.json'))
        self.assertIn('At most 2 GPUs', self.run_job('third', 600, config='configs/train.json')['error'])
        self.assertNotIn('error', self.broker.handle({'op': 'lease', 'name': 'cpu1', 'seconds': 120, 'gpu': 0, 'cpu': 2,
                                                      'memory': 4, 'files': self.files}))
        self.assertAlmostEqual(self.backend.budget()['used_usd'],
                               120 * expected + 600 * R + 120 * rate({'gpu': 0, 'cpu_cores': 2, 'memory_mib': 4096}))

        packaged = self.broker.handle({'op': 'package', 'name': 'net1', 'lease': 'gpu1', 'path': '/scratch/runs/r1',
                                       'files': edited})
        self.assertNotIn('error', packaged, packaged)
        bundle = self.root / 'jobs/net1/bundle'
        self.assertEqual(verify(bundle), packaged['sha256'])
        self.assertTrue((bundle / 'network.py').read_text().endswith('# pushed\n'))  # sealed with the source sent
        self.assertEqual((bundle / 'source/src/polytopia_bot/rl/ppo.py').read_text(), self.files['training/learner.py'])
        self.assertEqual(set(self.provider.packaged[0]['files']), set(control.NATIVE))
        self.assertEqual(json.loads((bundle / 'binding.json').read_text())['packaged_from'],
                         dict(lease='gpu1', path='/scratch/runs/r1'))
        self.assertIn('already used', self.broker.handle({'op': 'package', 'name': 'net1', 'lease': 'gpu1',
                                                          'path': '/scratch/runs/r1', 'files': edited})['error'])
        self.assertIn('Unknown lease', self.broker.handle({'op': 'exec', 'name': 'side', 'command': 'ls'})['error'])
        self.assertIn('not started yet', self.broker.handle({'op': 'exec', 'name': 'cpu1', 'command': 'ls'})['error'])

        self.assertEqual(self.broker.handle({'op': 'stop', 'name': 'gpu1'}), {'stopping': 'gpu1'})
        self.backend.settle()  # the lease ends; the queued run and the queued lease are cancelled uncharged
        jobs = {job['name']: job for job in self.broker.handle({'op': 'status'})['jobs']}
        self.assertEqual([jobs[name]['status'] for name in ('gpu1', 'side', 'cpu1', 'net1')],
                         ['stopped', 'stopped', 'stopped', 'complete'])
        self.assertAlmostEqual(jobs['gpu1']['charged_usd'], 40 * expected)  # held time x the shape's price
        self.assertEqual((jobs['gpu1']['usd_per_second'], jobs['gpu1']['worker_seconds']), (expected, 40.0))
        self.assertEqual((jobs['net1']['kind'], jobs['net1']['charged_usd'], jobs['net1']['lease']), ('package', 0, 'gpu1'))
        self.assertAlmostEqual(self.backend.budget()['used_usd'], 40 * expected)
        self.assertIn('ended', self.broker.handle({'op': 'exec', 'name': 'gpu1', 'command': 'ls'})['error'])
        late = self.broker.handle({'op': 'poll', 'name': 'gpu1', 'exec': '0001'})  # the host's record outlives the lease
        self.assertEqual(late['output'], 'late')
        self.assertEqual(self.broker.handle({'op': 'log', 'name': 'gpu1'})['commands'][0]['state'], 'ended with the lease')

        # The packaged network is practised and submitted like a trained one.
        self.assertEqual(self.broker.handle({'op': 'eval', 'name': 'net1'})['queued'], 'eval-net1-1')
        self.work()
        self.assertEqual(self.backend.status()['latest_practice']['run'], 'net1')
        self.assertEqual(self.broker.handle({'op': 'log', 'name': 'net1'})['practice_runs']['eval-net1-1']['games'], 512)
        self.assertEqual(self.broker.handle({'op': 'submit', 'name': 'net1'})['submitted']['native']['run'], 'net1')
        self.assertTrue(self.broker.handle({'op': 'finalize'})['finalized'])
        self.assertEqual(self.backend.grade()['games'], 4 * 128 * 2)

    def test_lease_requests_outside_the_limits_never_reach_the_ledger(self):
        base = {'op': 'lease', 'name': 'a', 'seconds': 100, 'gpu': 0, 'cpu': 4, 'memory': 16, 'files': self.files}
        for change, message in (({'gpu': 2}, '--gpu is 0 or 1'), ({'cpu': 33}, 'between 1 and 32'),
                                ({'memory': 1}, 'between 2 and 64'), ({'cpu': 2.5}, 'whole number'),
                                ({'seconds': 10}, 'at least 60 seconds'), ({'name': 'Bad_name'}, 'lowercase'),
                                ({'files': {}}, 'editable')):
            reply = self.broker.handle(dict(base, **change))
            self.assertIn(message, reply.get('error', ''), (change, reply))
        self.assertEqual(self.backend.status()['jobs'], [])
        # A cheap shape stretches the budget: 500 seconds of one core cost less than 40 of the training worker.
        cheap = self.broker.handle(dict(base, seconds=500, cpu=1, memory=2))
        self.assertNotIn('error', cheap, cheap)
        self.assertLess(cheap['reserved_usd'], 40 * R)
        self.assertIn('exceeds the remaining', self.broker.handle(dict(base, name='b', seconds=6000, gpu=1, cpu=32,
                                                                       memory=64))['error'])

    def test_a_session_lease_cap_refuses_longer_leases(self):
        self.backend.controller.ledger.config['max_lease_seconds'] = 900
        base = {'op': 'lease', 'name': 'capped', 'gpu': 0, 'cpu': 1, 'memory': 2, 'files': self.files}
        self.assertIn('at most 900 seconds', self.broker.handle(dict(base, seconds=901)).get('error', ''))
        self.assertNotIn('error', self.broker.handle(dict(base, seconds=900)))
        self.assertEqual(self.broker.handle({'op': 'budget'})['max_lease_seconds'], 900)

    def test_two_training_runs_at_once_each_hold_their_whole_grant(self):
        self.provider.gate = threading.Event()
        self.assertNotIn('error', self.run_job('a', 1200, config='configs/train.json'))
        self.assertNotIn('error', self.run_job('b', 1800, config='configs/train.json'))
        self.backend.tick()
        deadline = time.time() + 10
        while time.time() < deadline and [status for _, _, status in self.jobs()[0]] != ['running', 'running']:
            time.sleep(0.01)
        self.assertEqual(self.jobs(), ([('a', 'train', 'running'), ('b', 'train', 'running')], ['a', 'b']))
        self.assertEqual((self.seconds(), self.seconds('remaining_usd')), (3000, 3000))  # both grants, not time so far
        self.assertIn('At most 2 GPUs', self.run_job('c', 600, config='configs/train.json')['error'])
        self.assertIn('Resolve active work', self.broker.handle({'op': 'finalize'})['error'])
        self.provider.gate.set()
        self.backend.controller.join(timeout=20)
        self.assertEqual(self.jobs(), ([('a', 'train', 'complete'), ('b', 'train', 'complete')], []))
        self.assertEqual(self.seconds(), 400)  # each settled on its own 200-second lifetime
        self.assertNotIn('error', self.run_job('c', 600, config='configs/train.json'))

    def test_a_session_of_the_earlier_contract_is_refused(self):
        old = self.out / 'earlier'
        old.mkdir()
        write(old / 'backend.json', dict(game='polytopia', smoke=False, limit_seconds=14400))
        write(old / 'gpu-ledger.json', dict(limit_seconds=14400, jobs=[], closed=True))
        backend = registry.backend('polytopia')(self.release, old, smoke=False, resume=True,
                                                options={'_provider': self.provider})
        with self.assertRaisesRegex(ValueError, 'earlier contract budgeted in seconds'):
            backend.open()

    def test_the_declared_format_matches_generals_and_hex(self):
        spec = {game: registry.module(game).SPEC for game in ('polytopia', 'generals', 'hex')}
        # The same format; the dollar amount is Polytopia's own (its 16-core, 64 GiB training worker costs twice theirs).
        for key in ('leases', 'agents', 'wall_seconds', 'smoke_wall_seconds'):
            self.assertEqual(spec['polytopia'][key], spec['generals'][key], key)
            self.assertEqual(spec['polytopia'][key], spec['hex'][key], key)
        # Both modes are declared; seconds is the default (the small form's 14,400 seconds of one worker).
        self.assertEqual(registry.spec('polytopia')['budget_mode'], 'seconds')
        self.assertEqual(registry.research('polytopia'), dict(mode='seconds', seconds=14400.0, usd=None))
        self.assertEqual(registry.research('polytopia', smoke=True), dict(mode='seconds', seconds=600.0, usd=None))
        self.assertEqual(registry.research('polytopia', research_seconds=10800), dict(mode='seconds', seconds=10800.0,
                                                                                      usd=None))
        self.assertEqual((registry.research_usd('polytopia', budget_mode='usd'),
                          registry.research_usd('polytopia', smoke=True, budget_mode='usd')), (23.0, 1.0))
        self.assertEqual(registry.budgets('polytopia'), (14400.0, 3600.0))
        self.assertEqual(registry.budgets('polytopia', budget_mode='usd'), (None, 3600.0))
        self.assertEqual(session.wall_limit('polytopia', research_seconds=10800, budget_mode='seconds'), 2 * 10800 + 3600)
        self.assertEqual(session.wall_limit('polytopia', budget_mode='usd'), 18000)
        with self.assertRaisesRegex(ValueError, 'not both'):
            registry.budgets('polytopia', research_seconds=7200, research_usd=5)
        # The smoke budget buys the smoke's one 360-second run with room for a short lease, in either mode.
        self.assertLess(360 * R, 1.0)
        for mode, kwargs, key, amount in (('usd', dict(budget_usd=1.0), 'budget_usd', 1.0),
                                          ('seconds', {}, 'budget_seconds', 600.0)):
            smoke = registry.backend('polytopia')(self.release, self.out / ('smoke-' + mode), smoke=True,
                                                  options={'_provider': self.provider}, **kwargs)
            smoke.open()
            self.addCleanup(smoke.close)
            self.assertEqual((smoke.budget()['budget_mode'], smoke.budget()[key]), (mode, amount))


    def test_dollar_mode_runs_up_to_four_jobs_and_their_grants_never_exceed_the_budget(self):
        budget = self.backend.budget()
        self.assertEqual((budget['budget_mode'], budget['concurrency'], budget['concurrent_gpus']), ('usd', 4, 2))
        self.assertNotIn('budget_seconds', budget)
        config = Ledger(self.root).config
        self.assertEqual((config['budget_mode'], config['price'], config['max_concurrent_jobs']), ('usd', PRICE, 4))
        self.assertEqual(rate(TRAIN_SHAPE), budgets.rate(TRAIN_SHAPE))  # the harness's frozen prices are the contract's
        small = dict(gpu=0, cpu=2, memory=4)
        grants = [self.run_job('a', 1200, config='configs/train.json')['reserved_usd']]
        for name, seconds, shape in (('g', 600, dict(gpu=1, cpu=4, memory=16)), ('c1', 600, small), ('c2', 600, small)):
            reply = self.broker.handle(dict({'op': 'lease', 'name': name, 'seconds': seconds, 'files': self.files}, **shape))
            self.assertNotIn('error', reply, reply)
            grants.append(reply['reserved_usd'])
        self.assertIn('At most 4 jobs', self.broker.handle(dict({'op': 'lease', 'name': 'c3', 'seconds': 60,
                                                                 'files': self.files}, **small))['error'])
        self.assertAlmostEqual(self.backend.budget()['used_usd'], sum(grants))  # every queued grant is held
        self.broker.handle({'op': 'stop', 'name': 'c2'})  # cancelled uncharged: a slot and its grant come back
        left, cheap = self.backend.budget()['remaining_usd'], rate({'gpu': 0, 'cpu_cores': 2, 'memory_mib': 4096})
        request = dict({'op': 'lease', 'name': 'c3', 'files': self.files}, **small)
        self.assertIn('exceeds the remaining $', self.broker.handle(dict(request, seconds=int(left / cheap) + 60))['error'])
        self.assertNotIn('error', self.broker.handle(dict(request, seconds=int(left / cheap))))
        self.assertLessEqual(self.backend.budget()['used_usd'], self.backend.budget()['budget_usd'] + 1e-9)


class SecondsMode(Case):
    """The default mode: seconds of the training worker, one training run or lease at a time; practice stays
    outside the allocation with its own one-at-a-time rule."""
    smoke, budget, bind, mode = False, 6000, False, 'seconds'
    prepare_release = FullSession.prepare_release
    lease = Leases.lease

    def test_one_job_at_a_time_charged_in_training_worker_seconds(self):
        budget = self.broker.handle({'op': 'budget'})
        self.assertEqual((budget['budget_mode'], budget['budget_seconds'], budget['remaining_seconds'],
                          budget['allocatable_seconds'], budget['concurrency']), ('seconds', 6000, 6000, 6000, 1))
        self.assertNotIn('budget_usd', budget)
        self.assertEqual(budget['seconds_weight']['lease --gpu 1 --cpu 4 --memory 16'], 1.0)
        config = Ledger(self.root).config
        self.assertEqual((config['budget_mode'], config['train_shape'], config['max_concurrent_jobs']),
                         ('seconds', TRAIN_SHAPE, 1))
        self.assertNotIn('budget_usd', config)
        self.provider.gate = threading.Event()
        reply = self.run_job('a', 1200, config='configs/train.json')
        self.assertEqual((reply['seconds_weight'], reply['reserved_seconds']), (1.0, 1200), reply)
        self.assertEqual(self.backend.budget()['remaining_seconds'], 4800)  # the queued grant is held
        refused = self.run_job('b', 600, config='configs/train.json')['error']
        self.assertIn('One training run or lease at a time', refused)
        self.assertIn(': a', refused)
        self.assertIn('One training run or lease at a time', self.broker.handle(
            {'op': 'lease', 'name': 'l', 'seconds': 120, 'gpu': 0, 'cpu': 2, 'memory': 4, 'files': self.files})['error'])
        self.backend.tick()
        self.assertTrue(self.provider.entered.wait(timeout=10))
        self.assertEqual(self.jobs(), ([('a', 'train', 'running')], ['a']))
        self.assertIn('One training run or lease at a time', self.run_job('b', 600, config='configs/train.json')['error'])
        self.provider.gate.set()
        self.backend.controller.join(timeout=20)
        job = self.broker.handle({'op': 'status'})['jobs'][0]
        self.assertEqual((job['status'], job['seconds_weight'], job['charged_seconds']), ('complete', 1.0, 200.0))
        self.assertNotIn('charged_usd', job)
        budget = self.backend.budget()
        self.assertEqual((budget['used_seconds'], budget['remaining_seconds']), (200, 5800))  # exact worker seconds
        # Practice is outside the allocation: it runs beside the one training run, one practice job at a time.
        practice_gate, evaluate = threading.Event(), self.provider.evaluate
        self.provider.evaluate = lambda *args, **kwargs: (practice_gate.wait(timeout=30), evaluate(*args, **kwargs))[1]
        self.assertEqual(self.broker.handle({'op': 'eval', 'name': 'a'})['queued'], 'eval-a-1')
        self.backend.tick()
        self.assertNotIn('error', self.run_job('b', 600, config='configs/train.json'))
        self.assertIn('One practice job at a time', self.broker.handle({'op': 'eval', 'name': 'a'})['error'])
        self.backend.tick()
        self.backend.controller.threads['b'].join(timeout=20)
        jobs = dict((name, status) for name, _, status in self.jobs()[0])
        self.assertEqual((jobs['b'], jobs['eval-a-1']), ('complete', 'running'))
        practice_gate.set()
        self.work()
        self.assertEqual(self.backend.budget()['used_seconds'], 400)  # practice is never charged
        self.broker.handle({'op': 'submit', 'name': 'b'})
        self.assertEqual(self.backend.finalize()['used_seconds'], 400)
        self.assertGreaterEqual(self.backend.controller.ledger.public()['points'][-1]['charged'], 400)

    def test_a_lease_is_charged_its_seconds_weighted_by_its_price(self):
        shape = {'gpu': 0, 'cpu_cores': 8, 'memory_mib': 16384}  # cheaper than the training worker
        weight = budgets.seconds_weight(shape, TRAIN_SHAPE)
        self.assertAlmostEqual(weight, rate(shape) / R)
        self.assertLess(weight, 1.0)
        reply = self.lease(gpu=0, cpu=8, memory=16)
        self.assertEqual((reply['seconds_weight'], reply['shape']), (weight, shape))
        self.assertAlmostEqual(reply['reserved_seconds'], 120 * weight)
        self.assertIn('One training run or lease at a time', self.run_job('a', 600, config='configs/train.json')['error'])
        self.assertAlmostEqual(self.backend.budget()['seconds_weight']['lease --gpu 0 --cpu 8 --memory 16'], weight, 6)
        self.broker.handle({'op': 'stop', 'name': 'gpu1'})
        self.backend.controller.join(timeout=20)
        job = self.broker.handle({'op': 'status'})['jobs'][0]
        self.assertEqual((job['status'], job['worker_seconds']), ('stopped', 40.0))
        self.assertAlmostEqual(job['charged_seconds'], 40 * weight)  # the stub's 40 worker seconds, weighted
        self.assertAlmostEqual(self.backend.budget()['used_seconds'], 40 * weight)
        refused = self.run_job('a', 6000, config='configs/train.json')['error']
        self.assertRegex(refused, r'Allocation of 6000 s exceeds the remaining \d+ s')
        self.assertNotIn('error', self.run_job('a', 5900, config='configs/train.json'))

    def test_a_resumed_session_keeps_its_mode_and_refuses_another(self):
        self.backend.close()
        record = json.loads((self.root / 'backend.json').read_text())
        self.assertEqual((record['budget_mode'], record['budget_seconds']), ('seconds', 6000))
        resumed = self.open(self.root, resume=True)
        self.assertEqual((resumed.budget_mode, resumed.limits['max_concurrent_jobs']), ('seconds', 1))
        self.assertEqual(resumed.budget()['budget_seconds'], 6000)
        with self.assertRaisesRegex(ValueError, 'opened in seconds budget mode'):
            self.open(self.root, resume=True, budget_usd=5.0)
        with self.assertRaisesRegex(ValueError, 'opened in seconds budget mode'):
            Ledger(self.root, dict(DEFAULT))
        dollars = self.open(self.out / 'dollars', budget_usd=5.0)
        dollars.close()
        with self.assertRaisesRegex(ValueError, 'opened in usd budget mode'):
            self.open(self.out / 'dollars', resume=True, budget_seconds=600)
        self.assertEqual(self.open(self.out / 'dollars', resume=True).budget()['budget_usd'], 5.0)


class PackagedCheckpoints(unittest.TestCase):
    """`ar package` seals a checkpoint as data: nothing the researcher wrote is imported on the host."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.files = {name: '# ' + name + '\n' for name in EDITABLE}
        self.files.update({'recipe.json': '{"learning_rate": 0.001}', 'arch.json': '{"width": 32}'})

    def checkpoint(self, record=None, **arrays):
        stream = io.BytesIO()
        numpy.savez(stream, **(arrays or dict(p0=numpy.ones((4, 4), dtype=numpy.float32))))
        record = record if record is not None else json.dumps(dict(sha256=hashlib.sha256(stream.getvalue()).hexdigest()))
        return {'final.npz': stream.getvalue(), 'final.json': record.encode()}

    def seal(self, checkpoint, name='bundle'):
        return transport.package_bundle(Path(self.temp.name) / name, checkpoint, self.files, 'release-sha',
                                        dict(lease='l1', path='/scratch/r1'))

    def test_a_whole_checkpoint_becomes_the_bundle_a_training_run_leaves(self):
        identity = self.seal(self.checkpoint(p0=numpy.ones((4, 4), dtype=numpy.float32),
                                             o0=numpy.zeros(700_000, dtype=numpy.float32)))  # optimizer state is not policy
        bundle = Path(self.temp.name) / 'bundle'
        self.assertEqual(verify(bundle), identity)
        for name in ('final.npz', 'final.json', 'network.py', 'ar_arch.json'):  # what the evaluator copies to a worker
            self.assertTrue((bundle / name).is_file(), name)
        self.assertEqual((bundle / 'network.py').read_text(), self.files[EDITABLE[0]])
        self.assertEqual(tree(bundle / 'source'), self.files)
        binding = json.loads((bundle / 'binding.json').read_text())
        self.assertEqual((binding['benchmark_id'], binding['policy_parameters'], binding['release_sha256']),
                         (transport.BENCHMARK, 16, 'release-sha'))

    def test_a_checkpoint_over_the_cap_or_not_whole_is_refused(self):
        cap = DEFAULT['policy_parameter_cap']
        refused = ((self.checkpoint(p0=numpy.zeros(cap + 1, dtype=numpy.float32)), 'Policy parameter cap exceeded'),
                   (self.checkpoint(p0=numpy.zeros(cap // 2 + 1), p1=numpy.zeros(cap // 2 + 1)), 'cap exceeded'),
                   (self.checkpoint(p0=numpy.array([1.0, float('nan')])), 'not finite'),
                   (self.checkpoint(o0=numpy.ones(4)), 'no policy parameters'),
                   (self.checkpoint(record='{"sha256": "0"}'), 'does not name the digest'),
                   (self.checkpoint(record='not json'), 'not JSON'),
                   ({'final.npz': b'not an archive', 'final.json': json.dumps(dict(
                       sha256=hashlib.sha256(b'not an archive').hexdigest())).encode()}, 'not a readable checkpoint'))
        for index, (checkpoint, message) in enumerate(refused):
            with self.assertRaisesRegex(ValueError, message):
                self.seal(checkpoint, f'bundle{index}')
            self.assertFalse((Path(self.temp.name) / f'bundle{index}').exists(), message)  # nothing half-sealed
        self.seal(self.checkpoint(p0=numpy.zeros(cap, dtype=numpy.float32)), 'at-the-cap')


class LeaseWorkers(Workers):
    """A lease on the real Polytopia provider, with the stand-in cloud in place of Modal."""
    SHAPE = {'gpu': 1, 'cpu_cores': 16, 'memory_mib': 65536}

    def lease(self, seconds=120, wait=True):
        reply = self.broker.handle({'op': 'lease', 'name': 'l1', 'seconds': seconds, 'gpu': 1, 'cpu': 16, 'memory': 64,
                                    'files': self.files})
        self.assertNotIn('error', reply, reply)
        self.backend.tick()
        deadline = time.time() + 10
        while wait and not self.provider.lease_live('l1') and time.time() < deadline:
            time.sleep(0.01)
        return reply

    def job(self):
        self.backend.controller.join(timeout=20)
        return self.backend.status()['jobs'][0]

    def test_a_lease_serves_unprivileged_commands_packages_a_checkpoint_and_settles_on_its_receipt(self):
        reply = self.lease()
        self.assertEqual((reply['shape'], reply['usd_per_second']), (self.SHAPE, rate(self.SHAPE)))
        sandbox = self.cloud.sandboxes['sb-lease-l1']
        self.assertEqual(sandbox.shape, self.SHAPE)
        # One scratch Volume per session, in the project environment, mounted at /scratch and recorded for the archive.
        record = json.loads((self.root / 'scratch-volume.json').read_text())
        self.assertEqual(sandbox.volumes, {'/scratch': record['volume']})
        self.assertTrue(record['volume'].startswith('polytopia-scratch-'))
        self.assertEqual(self.volumes, [(record['volume'], dict(environment_name=policy.ENVIRONMENT,
                                                                create_if_missing=True, version=2))])
        # /work is a writable copy of the worker's package; the pushed files sit under their workspace names and the
        # trainer paths are links to them.
        setup = sandbox.scripts[0][0]
        self.assertIn('cp -a /workspace/src /workspace/experiments /work/', setup)
        self.assertIn('chown -R 65534:65534 /work', setup)
        self.assertEqual(sandbox.files['/work/training/policy.py'], self.files['training/policy.py'])
        self.assertEqual(sandbox.files['/work/configs/train.json'], self.files['configs/train.json'])
        links = next(script[0] for script in sandbox.scripts if 'ln -sfn' in script[0])
        for name, native in NATIVE.items():
            if name.startswith('training/'):
                self.assertIn(f'ln -sfn /work/{name} /work/{native}', links)
        job = self.backend.status()['jobs'][0]
        self.assertEqual((job['kind'], job['status'], job['ready']), ('lease', 'running', True))

        hostile = 'python experiments/gpu_resident_train.py; chmod 777 /private; $(id)'
        started = self.broker.handle({'op': 'exec', 'name': 'l1', 'command': hostile})
        self.assertEqual((started['exec'], started['state']), ('0001', 'running'), started)
        script, argument = sandbox.scripts[-1]
        self.assertEqual(argument, hostile)  # passed as an argument, never interpolated into the root script
        self.assertNotIn('chmod 777', script)
        for text in ('setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs', 'cd /work',
                     'PYTHONPATH=/work/src:/work/experiments', 'JAX_PLATFORMS=cuda', 'POLYTOPIA_PC_BUILD=/ar/engine-build',
                     'OMP_NUM_THREADS=16', 'setsid --wait bash -c "$0"'):
            self.assertIn(text, script)
        done = self.broker.handle({'op': 'poll', 'name': 'l1', 'exec': '0001'})
        self.assertEqual((done['state'], done['exit_code'], done['output']), ('exited', 0, 'command output\n'))
        self.assertEqual(self.broker.handle({'op': 'sync', 'name': 'l1', 'reload': True}),
                         {'committed': '/scratch', 'reloaded': True})
        self.assertTrue(sandbox.reloaded)

        # A directory without a checkpoint is refused and leaves no run behind; a whole one is sealed.
        request = {'op': 'package', 'name': 'net1', 'lease': 'l1', 'path': '/scratch/runs/r1', 'files': self.files,
                   'config': 'configs/smoke.json'}
        self.assertIn('needs a readable final.npz and final.json', self.broker.handle(request)['error'])
        self.assertFalse((self.root / 'jobs/net1').exists())
        self.assertEqual(len(list((self.root / 'jobs').glob('net1.failed-*'))), 1)
        stream = io.BytesIO()
        numpy.savez(stream, p0=numpy.full((4, 4), 7.0, dtype=numpy.float32))
        self.cloud.checkpoint = {'final.npz': stream.getvalue(), 'final.json': json.dumps(dict(
            sha256=hashlib.sha256(stream.getvalue()).hexdigest())).encode()}
        packaged = self.broker.handle(request)
        self.assertNotIn('error', packaged, packaged)
        fetch = next(script[0] for script in reversed(sandbox.scripts) if transport.PACKAGE_STAGE in script[0])
        self.assertIn("head -c 52428801 -- /scratch/runs/r1/final.npz > /private/ar-package/final.npz", fetch)
        self.assertIn('setpriv --reuid=65534', fetch)  # the researcher's files are read as the researcher's user
        bundle = self.root / 'jobs/net1/bundle'
        self.assertEqual(verify(bundle), packaged['sha256'])
        self.assertEqual((bundle / 'final.npz').read_bytes(), self.cloud.checkpoint['final.npz'])
        self.assertEqual((bundle / 'source/recipe.json').read_text(), self.files['configs/smoke.json'])
        self.assertEqual(self.broker.handle({'op': 'submit', 'name': 'net1'})['submitted']['native']['sha256'],
                         packaged['sha256'])

        self.assertEqual(self.broker.handle({'op': 'stop', 'name': 'l1'}), {'stopping': 'l1'})
        job = self.job()
        self.assertEqual((job['status'], job['worker_seconds']), ('stopped', 123.0), job)
        self.assertAlmostEqual(job['charged_usd'], 123.0 * rate(self.SHAPE))
        self.assertEqual(self.cloud.terminated, ['sb-lease-l1'])
        closing = [script[0] for script in sandbox.scripts[-4:]]
        self.assertTrue(any('pkill -KILL -u 65534' in script for script in closing))  # no researcher process survives
        self.assertTrue(any(script == 'sync /scratch' for script in closing))        # and /scratch is committed
        self.assertIn('ended', self.broker.handle({'op': 'exec', 'name': 'l1', 'command': 'ls'})['error'])
        self.assertEqual(self.broker.handle({'op': 'poll', 'name': 'l1', 'exec': '0001'})['exit_code'], 0)
        self.assertEqual(self.broker.handle({'op': 'log', 'name': 'l1'})['commands'][0]['command'], hostile)
        self.assertTrue(self.broker.handle({'op': 'finalize'})['finalized'])

    def test_a_lease_ends_with_its_grant_and_is_complete(self):
        # Started 26 seconds ago with a 60-second grant: past the point where the host stops serving commands.
        self.cloud.run_age = 60 - transport.TEARDOWN_SECONDS - transport.LEASE_TEARDOWN_SECONDS + 1
        self.lease(seconds=60, wait=False)
        job = self.job()
        self.assertEqual((job['status'], job['worker_seconds']), ('complete', 123.0), job)
        self.assertFalse(self.provider.lease_live('l1'))
        self.assertEqual(self.cloud.terminated, ['sb-lease-l1'])
        self.assertEqual(self.jobs()[1], [])

    def test_a_lease_whose_worker_never_starts_is_uncharged(self):
        self.cloud.start_after = None
        self.lease(wait=False)
        job = self.job()
        self.assertEqual((job['status'], job['charged_usd']), ('failed', 0), job)
        self.assertIn('never started', job['error'])
        self.assertEqual(self.backend.budget()['used_usd'], 0)

    def test_closing_the_session_ends_a_lease_and_charges_its_lifetime(self):
        self.lease()
        self.backend.settle()
        job = self.backend.status()['jobs'][0]
        self.assertEqual((job['status'], job['worker_seconds']), ('stopped', 123.0), job)
        self.assertEqual(self.cloud.terminated, ['sb-lease-l1'])


class RecipeSession(unittest.TestCase):
    def test_whole_recipe_session_through_the_box_command(self):
        """The real `ar`, broker and session loop; only Harbor, the box transport and Modal are stood in for."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        base = Path(temp.name).resolve()
        release, _ = release_fixture('polytopia', base)
        provider, backend = Provider(), registry.backend('polytopia')
        provider.ended['final'] = 'cut_short'  # the fresh final is cut off at its time limit but had exported
        out, box = base / 'session', base / 'box'

        def factory(*args, **kwargs):  # a provider object cannot travel through the session's JSON settings
            return backend(*args, **dict(kwargs, options={'_provider': provider}))

        def idle(ar):
            for _ in range(100):
                status = ar('status')
                if not status['active']:
                    return status
                time.sleep(0.2)
            raise AssertionError('work never settled')

        def script(ar):
            budget = ar('budget')
            # The default mode: the smoke's 600 seconds of the training worker, one job at a time.
            self.assertEqual((budget['variant'], budget['budget_mode'], budget['remaining_seconds'], budget['concurrency'],
                              budget['final_train_seconds']), ('recipe', 'seconds', 600.0, 1, 360.0))
            self.assertEqual(ar('check', '--config', 'configs/smoke.json')['parameters']['policy'], 191587)
            ar.launch('explore', '--seconds', '360', '--config', 'configs/smoke.json')
            # The launcher leased a worker, trained there, sealed the result as `explore` and stopped the worker.
            self.assertEqual([(job['name'], job['kind'], job['status']) for job in idle(ar)['jobs']],
                             [('explorew', 'lease', 'stopped'), ('explore', 'package', 'complete')])
            self.assertEqual(ar('eval', 'explore')['queued'], 'eval-explore-1')
            self.assertEqual(idle(ar)['latest_practice']['games'], 2)
            ar.edit('training/learner.py', (ar.workspace / 'training/learner.py').read_text() + '\n# final recipe\n')
            self.assertEqual(ar('submit', '--note', 'final')['submitted']['kind'], 'recipe')
            self.assertTrue(ar('finalize')['finalized'])

        def launcher(session_dir):
            return Process(lambda: script(Researcher(session_dir / 'task', box)))

        with mock.patch.object(registry, 'backend', lambda game: factory), \
                mock.patch.object(policy, 'require_modal', lambda: None):
            result = session.run('polytopia', 'recipe', release, out, model='test/model', smoke=True, launcher=launcher,
                                 find_box=lambda: LocalBox(box / 'ar') if (box / 'ar').is_dir() else None)
            self.assertEqual((result['status'], result['error'], result['closed_by']), ('researched', None, 'researcher'))
            self.assertEqual(result['final_training']['status'], 'complete', result['final_training'])
            self.assertIs(result['final_training']['job']['cut_short'], True)  # trained, and says how it ended
            graded = session.grade(out)
        # Research trained on a lease and packaged; the only fixed training run is the host's final one.
        (final,), (explore,) = provider.trained, provider.packaged
        self.assertEqual((explore['lease'], explore['path'], final['name'], final['grant']),
                         ('explorew', '/scratch/runs/explore', 'final', 360))
        self.assertTrue(final['files']['src/polytopia_bot/rl/ppo.py'].endswith('# final recipe\n'))
        self.assertFalse(explore['files']['src/polytopia_bot/rl/ppo.py'].endswith('# final recipe\n'))
        self.assertEqual(final['files']['recipe.json'], explore['files']['recipe.json'])
        research = Ledger(out / 'research')
        self.assertEqual((research.submissions(), research.lifecycle()), ([], 'active'))  # no research network counts
        self.assertEqual(research.mode, 'seconds')
        self.assertAlmostEqual(research.used(), 40)  # the lease, at the training worker's shape (weight 1)
        self.assertEqual(json.loads((out / 'plan.json').read_text())['budget_mode'], 'seconds')
        self.assertTrue(session.official('polytopia', dict(json.loads((out / 'plan.json').read_text()), smoke=False,
                                                           research_seconds=14400.0, final_train_seconds=3600.0)))
        trained = Ledger(out / 'final')
        self.assertEqual((trained.submissions()[-1]['job'], trained.lifecycle()), ('final', 'finalized'))
        self.assertEqual((trained.mode, trained.budget()), ('seconds', 360))
        self.assertEqual((graded['games'], json.loads((out / 'grade.json').read_text())['graded_root']), (4, 'final'))
        self.assertEqual(provider.preflights, 3)  # research, final training and grading each pass the CPU gate


@unittest.skipUnless((build_context.CACHE / build_context.MAP_NAME).is_file()
                     and all((build_context.PANEL_DIR / name / 'final.npz').is_file() for name in PANEL),
                     "this game mode's map table and practice checkpoints (git-ignored .cache/) are not on this machine")
class Constructor(unittest.TestCase):
    def test_release_holds_what_workers_need_and_nothing_of_older_harnesses(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        release = Path(temp.name) / 'release'
        identity = registry.constructor('polytopia')(release)
        self.assertEqual(verify(release), identity)
        with self.assertRaises(FileExistsError):
            registry.constructor('polytopia')(release)
        policy.bind_release('polytopia', release)
        policy.verify_release('polytopia', release)
        self.assertEqual({p.name for p in release.iterdir()},
                         {'runtime', 'maps', 'v1', 'public', 'benchmark.json', 'base.Dockerfile', 'base-context',
                          'host-private-panel', 'manifest.json'})
        self.assertEqual({p.name for p in (release / 'public').iterdir()},
                         {'recipes', workspace.WORKSPACE, workspace.MARKER})
        # Every path the provider mounts or names on a worker.
        self.assertEqual({p.name for p in (release / 'v1').iterdir()},
                         {'core.py', 'train.py', 'grade.py', 'policy.py', 'preflight.py'})
        for name in ('runtime/src/polytopia_bot/rl/runner.py', 'runtime/experiments/gpu_resident_train.py',
                     'base-context/repo/engine-rs/Cargo.toml', 'base-context/repo/engine-cu/Makefile',
                     'maps/' + build_context.MAP_NAME):
            self.assertTrue((release / name).is_file(), name)
        starter = HERE / 'starter'
        for name in EDITABLE:
            self.assertEqual((release / 'runtime' / name).read_bytes(), (starter / name).read_bytes(), name)
        panel = json.loads((release / 'host-private-panel/inventory.json').read_text())['panel']
        self.assertEqual([entry['name'] for entry in panel], list(PANEL))
        meta = workspace.verify(release)
        self.assertEqual(set(meta['writable']), {*NATIVE, 'configs/train.json', 'configs/smoke.json'})
        staged = [p.relative_to(release).as_posix() for p in release.rglob('*') if p.is_file()]
        public = [name for name in staged if name.startswith(('public/', 'runtime/', 'v1/', 'maps/'))]
        self.assertFalse([name for name in public if 'reference' in Path(name).name or '/hidden/' in name])
        self.assertFalse([name for name in staged if name.startswith('public/') and name.endswith('.npz')])
        engine = ('public/agent_workspace/simulator/', 'runtime/', 'base-context/')  # pinned game source, not harness
        self.assertFalse([name for name in staged if not name.startswith(engine) and Path(name).name in (
            'ar.py', 'ar.sh', 'instruction.md', 'brief.md', 'rules_reader.py', 'rules_manifest.json', 'task.toml',
            'session.json', 'driver.py', 'model.py', 'provider.py', 'Dockerfile')])
        # An untouched workspace trains exactly the release's own starter and smoke recipe.
        provider = Provider()
        backend = registry.backend('polytopia')(release, Path(temp.name) / 'session', smoke=True,
                                                options={'_provider': provider})
        backend.open()
        self.addCleanup(backend.close)
        base = workspace.public(release) / workspace.WORKSPACE
        workspace.apply(release, backend.workspace, {name: (base / name).read_text() for name in meta['writable']})
        backend.run('a', 360, 'configs/smoke.json')
        backend.tick()
        backend.controller.join(timeout=20)
        expected = {name: (release / 'runtime' / name).read_text() for name in EDITABLE}
        expected.update({'recipe.json': (release / 'public/recipes/smoke.json').read_text(),
                         'arch.json': (release / 'public/recipes/starter_arch.json').read_text()})
        self.assertEqual(provider.trained[0]['files'], expected)


class ConstructorInputs(unittest.TestCase):
    """The real constructor on stand-in local inputs of the current game mode (a blank map table, placeholder
    opponents): what a release needs, and the bootstrap release that has no opponents yet."""

    def setUp(self):
        import numpy as np
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        cache = self.base / 'cache'
        cache.mkdir()
        for name, value in (('CACHE', cache), ('PANEL_DIR', cache / build_context.PANEL_DIR.name)):
            patcher = mock.patch.object(build_context, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        maps, tiles = MODE.MAP_SEED_HIGH - MODE.MAP_SEED_LOW, MODE.SIZE ** 2
        np.savez_compressed(cache / MODE.MAP_TABLE, lo=np.int64(MODE.MAP_SEED_LOW), hi=np.int64(MODE.MAP_SEED_HIGH),
                            types=np.array([MODE.MAPS]), kind=np.zeros(maps, np.int8),
                            capitals=np.zeros((maps, MODE.PLAYERS), np.int16),
                            **{key: np.zeros((maps, tiles), np.int16) for key in ('terrain', 'structure', 'resource')})

    def opponents(self, labels=PANEL):
        for label in labels:
            folder = build_context.PANEL_DIR / label
            folder.mkdir(parents=True)
            (folder / 'final.npz').write_bytes(b'weights of ' + label.encode())
            digest = hashlib.sha256((folder / 'final.npz').read_bytes()).hexdigest()
            write(folder / 'final.json', dict(config=dict(size=MODE.SIZE, players=MODE.PLAYERS), sha256=digest))
            (folder / 'network.py').write_text('# network\n')
            (folder / 'ar_arch.json').write_text('{}\n')

    def test_a_bootstrap_release_trains_and_refuses_full_practice_and_a_full_release_carries_the_opponents(self):
        with self.assertRaises(FileNotFoundError) as refused:  # no opponents of this mode yet
            registry.constructor('polytopia')(self.base / 'release')
        self.assertIn('Missing practice opponent starter-200m', str(refused.exception))
        self.assertFalse((self.base / 'release').exists())
        boot = self.base / 'bootstrap'
        identity = registry.constructor('polytopia')(boot, panel=False)
        self.assertEqual(verify(boot), identity)
        policy.bind_release('polytopia', boot)
        policy.verify_release('polytopia', boot)
        self.assertFalse((boot / 'host-private-panel').exists())
        self.assertTrue((boot / 'maps' / MODE.MAP_TABLE).is_file())
        self.assertEqual((boot / 'runtime/src/polytopia_bot/rl/mode.py').read_bytes(),
                         (HERE / 'starter/src/polytopia_bot/rl/mode.py').read_bytes())
        meta = workspace.verify(boot)
        base = workspace.public(boot) / workspace.WORKSPACE
        # The researcher reads the mode module and the mode's rules; the starter config is the small mode's.
        self.assertTrue((base / 'simulator/runtime/src/polytopia_bot/rl/mode.py').is_file())
        train = json.loads((base / 'configs/train.json').read_text())
        self.assertEqual({key: train[key] for key in MODE.FROZEN}, MODE.FROZEN)
        self.assertIn(f'an{MODE.SIZE}×{MODE.SIZE} board', (base / 'game_rules/rules/mode-and-maps.md').read_text())
        # It trains (the unmodified starter, as `harness retrain --starter` does); full practice has no opponents.
        provider = Provider()
        backend = registry.backend('polytopia')(boot, self.base / 'session', smoke=False, options={'_provider': provider})
        backend.open()
        self.addCleanup(backend.close)
        workspace.apply(boot, backend.workspace, {name: (base / name).read_text() for name in meta['writable']})
        backend.run('final', 600, 'configs/train.json')
        backend.tick()
        backend.controller.join(timeout=20)
        self.assertEqual(backend.status()['jobs'][0]['status'], 'complete')
        self.assertEqual(provider.trained[0]['files']['recipe.json'], (boot / 'public/recipes/starter.json').read_text())
        backend.evaluate('final')
        backend.tick()
        backend.controller.join(timeout=20)
        practice = backend.status()['jobs'][1]
        self.assertEqual(practice['status'], 'failed', practice)
        self.assertIn('prepared without practice opponents', practice['error'])
        self.assertEqual(provider.evaluated, [])
        # With the opponents in place the ordinary release carries them, host-only.
        self.opponents()
        release = self.base / 'release'
        registry.constructor('polytopia')(release)
        panel = json.loads((release / 'host-private-panel/inventory.json').read_text())['panel']
        self.assertEqual([entry['name'] for entry in panel], list(PANEL))
        self.assertFalse([p for p in (release / 'public').rglob('*') if p.suffix == '.npz'])

    def test_an_empty_starter_release_on_the_ladder_withholds_the_reference_trainer(self):
        """starter=empty: the runtime (box, leases and worker image alike) carries the boilerplate, not the PPO
        starter; the box withholds the browser player and anything matching the learning-code pattern; the ladder
        panel is host-only; the release records its starter, and the PPO ring rule does not bind its recipes."""
        from harness.games.polytopia.controller import recipe_of, starter_of
        self.opponents(build_context.LADDER)
        release = self.base / 'empty'
        registry.constructor('polytopia')(release, panel='ladder', starter='empty')
        policy.bind_release('polytopia', release)
        policy.verify_release('polytopia', release)
        self.assertEqual(starter_of(release), 'empty')
        self.assertEqual(json.loads((release / 'starter.json').read_text())['panel'], list(build_context.LADDER))
        panel = json.loads((release / 'host-private-panel/inventory.json').read_text())['panel']
        self.assertEqual([entry['name'] for entry in panel], list(build_context.LADDER))
        empty = HERE / 'starter_empty'
        for name in ('src/polytopia_bot/rl/network.py', 'src/polytopia_bot/rl/ppo.py',
                     'src/polytopia_bot/rl/gpu_resident.py', 'experiments/gpu_resident_train.py'):
            self.assertEqual((release / 'runtime' / name).read_bytes(), (empty / name).read_bytes(), name)
        runtime = '\n'.join(p.read_text() for p in (release / 'runtime').rglob('*.py'))
        for marker in ('adam_step', 'segment_gae', 'ppo.loss', 'value_and_grad', 'family_embedding', 'own-potential'):
            self.assertNotIn(marker, runtime, marker)
        meta = json.loads((workspace.public(release) / workspace.MARKER).read_text())
        self.assertTrue(meta['withheld'] and all('engine-wasm' in name for name in meta['withheld']), meta['withheld'])
        base = workspace.public(release) / workspace.WORKSPACE
        self.assertFalse(list(base.rglob('net.rs')))
        self.assertTrue((workspace.public(release) / 'brief.md').read_text().startswith('# Polytopia research brief'))
        self.assertIn('boilerplate only', (workspace.public(release) / 'brief.md').read_text())
        files = {'recipe.json': json.dumps(dict(json.loads((base / 'configs/train.json').read_text()), segment=64)),
                 'arch.json': '{}'}
        recipe_of(files, ring=False)
        with self.assertRaises(ValueError):
            recipe_of(files)
        # The PPO release of the same code differs from it only in the starter's trainer files and its choice.
        ppo = self.base / 'ppo'
        registry.constructor('polytopia')(ppo, panel='ladder')
        self.assertEqual(starter_of(ppo), 'ppo')
        a, b = (json.loads((r / 'manifest.json').read_text())['files'] for r in (release, ppo))
        played = {k for k in set(a) | set(b) if a.get(k) != b.get(k) and not k.startswith('public/')}
        self.assertEqual(played, {'starter.json', 'runtime/experiments/gpu_resident_train.py',
                                  'runtime/src/polytopia_bot/rl/gpu_resident.py',
                                  'runtime/src/polytopia_bot/rl/network.py', 'runtime/src/polytopia_bot/rl/ppo.py'})


if __name__ == '__main__':
    unittest.main()
