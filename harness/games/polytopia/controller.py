"""Host-owned Polytopia session authority: budget ledger, job queue, leases, practice, sealed submissions, grading.

The session's budget mode (harness/budget.py, recorded in the ledger config) sets the unit and the concurrency:
in dollar mode training runs, leases and practice run side by side up to the ledger's limits and every worker
second is charged its shape's list price; in seconds mode one training run or lease runs at a time, a training
worker second is one second of allocation and a lease's second is weighted by budget.seconds_weight. Every job has
its own worker thread. A training run or a lease holds its whole grant, in the session's unit, from the request
until the provider's own start/finish record settles it; this file never invents a charge. Public practice plays
on separate CPU workers outside the budget, one practice job at a time, in either mode.
"""
import ast
import json
from pathlib import Path
import shutil
import threading
import time

from harness import budget as budgets, workers
from simulators.polytopia.autoresearch.contract import (DEFAULT, PRACTICE_SHAPE, PRICE, TRAIN_SHAPE, lease_shape,
                                                        rate)
from simulators.polytopia.autoresearch.ledger import ACTIVE, Ledger, terms
from simulators.polytopia.autoresearch.v1.core import EDITABLE, validate_config, verify, write

# How a run that kept a network ended before its own end (harness.broker.EARLY); both are `complete`.
EARLY = ('stopped_early', 'cut_short')
SMOKE_PRACTICE = dict(seeds=[700_000], opponent_seed=1741)
# 64 public maps per opponent since 2026-11-15 (32 before): the old interval (about ±4 points) was as wide as the
# differences researchers selected on. Practice and grading play their four-map shards on EVAL_PARALLEL CPU workers at
# once, so practice takes about as long as the old 32-map schedule played one shard at a time.
FULL_PRACTICE = dict(seeds=list(range(700000, 700064)), opponent_seed=0)
EVAL_PARALLEL = 8
NATIVE = (*EDITABLE, 'recipe.json', 'arch.json')
TRAINER_KEYS = ('games', 'sets', 'ring')
RING_PER_SEGMENT = 5  # the starter's own ratio (ring 160, segment 32); see check_ring
# How often tick() re-asks the provider about the same unresolved worker (each ask lists Modal sandboxes).
RECOVERY_RETRY_SECONDS = 30.0

STOPPED_QUEUED = ('Stopped while queued for a GPU: the provider never started this worker; it was terminated and '
                  'charged 0 seconds')
STOP_NOTE = ('A run that is training ends at its next step and keeps the network trained so far: it finishes as '
             'complete with stopped_early, or as stopped if it had not written a complete policy yet, and the time '
             'until then is charged. A worker still waiting for a GPU is cancelled uncharged. Check ar status.')


class Stopped(RuntimeError):
    pass


class _Stoppable:
    """Lets a multi-worker practice schedule end between CPU workers; one worker is never interrupted."""

    def __init__(self, provider, stop):
        self.provider, self.stop = provider, stop

    def evaluate(self, *args, **kwargs):
        if self.stop():
            raise Stopped('Stopped by request')
        return self.provider.evaluate(*args, **kwargs)


class Allocation:
    """What a provider needs of the ledger for one worker (Provider.train, Provider.hold).

    The request already holds the grant, so `reserve` only confirms it; the worker's start is recorded when the
    provider reports it, and the provider's own receipt settles the job. A receipt without both times leaves the
    whole grant held (`uncertain`). A failure the researcher asked for is `stopped`.
    """

    def __init__(self, ledger, stopped):
        self.ledger, self.stopped = ledger, stopped

    def reserve(self, name, grant):
        row = self.ledger.job(name)
        if not row or row['status'] != 'queued' or grant > row['grant']:
            raise ValueError('No queued allocation of that size for this job')

    def start(self, name, provider_id, started_at):
        self.ledger.start(name, provider_id, started_at)

    def settle(self, name, receipt, status):
        if not (receipt and receipt.get('started_at') and receipt.get('finished_at')):
            self.ledger.uncertain(name, 'The provider has no final record of this worker yet; its grant stays held')
            return
        if status == 'failed' and self.stopped():
            status = 'stopped'
        self.ledger.settle(name, receipt, status)

    def settle_unstarted(self, name, receipt, stopped=False, reason=None):
        self.ledger.abandon_unstarted(name, receipt, str(reason or workers.NEVER_STARTED), stopped=stopped)

    def public(self):
        return self.ledger.public()


def kind_of(row):
    return json.loads(row['payload']).get('kind', 'train')


def check_ring(recipe):
    """Refuse a segment the trainer's device ring cannot hold. A seat's rows stay in the ring until its segment has
    closed and been trained on, which takes about four segments' worth of steps (two seats alternate, and closed
    segments wait for the next batch). The starter's 32-decision segments fit its 160-step ring; three sessions
    of 2026-11-14 raised `segment` to 64 with `ring` at 160 or 192 and each lost a run about three minutes in
    ('row N steps old has left the device ring'), while 64 with 320 trained for 95 minutes. This is the starter
    trainer's ratio, kept as a rule: it is not derived from a rewritten training loop."""
    segment, ring = recipe.get('segment', 32), recipe['trainer']['ring']
    if isinstance(segment, bool) or not isinstance(segment, int) or segment < 1:
        raise ValueError('segment must be a positive whole number')
    if isinstance(ring, bool) or not isinstance(ring, int) or ring < RING_PER_SEGMENT * segment:
        raise ValueError(f'trainer.ring must be at least {RING_PER_SEGMENT} x segment: segment {segment} needs '
                         f'trainer.ring >= {RING_PER_SEGMENT * segment}, and this configuration has {ring}. With a '
                         'smaller ring the run fails a few minutes in, when a row leaves the device ring before '
                         'it is trained on. Ring memory grows with trainer.ring x trainer.games.')


def starter_of(release):
    """The starter a release was prepared with (its `starter.json`; releases from before the option are `ppo`)."""
    record = Path(release) / 'starter.json'
    return json.loads(record.read_text())['starter'] if record.is_file() else 'ppo'


def recipe_of(files, ring=True):
    """The native recipe and architecture, checked against the frozen task fields and the trainer's launch options.
    `ring` applies the PPO starter's ring rule (`check_ring`), which says nothing about a trainer written from the empty
    starter."""
    try:
        recipe = json.loads(files['recipe.json'])
        if not isinstance(recipe, dict):
            raise ValueError('The configuration must be a JSON object')
        validate_config(recipe)
        trainer = recipe.get('trainer')
        if not isinstance(trainer, dict) or any(key not in trainer for key in TRAINER_KEYS):
            raise ValueError('The configuration needs trainer.games, trainer.sets and trainer.ring')
        if ring:
            check_ring(recipe)
        if not isinstance(json.loads(files['arch.json']), dict):
            raise ValueError('configs/architecture.json must be a JSON object')
    except KeyError as exc:
        raise ValueError('Missing configuration field ' + str(exc)) from None
    except TypeError as exc:
        raise ValueError('Invalid configuration value: ' + str(exc)) from None
    return recipe


class Controller:
    def __init__(self, root, release, provider, config=None, smoke=False):
        self.root = Path(root)
        self.release = Path(release)
        self.provider = provider
        self.smoke = bool(smoke)
        self.starter = starter_of(self.release)
        self.ledger = Ledger(self.root, config)
        self.stops = set()
        self.halts = set()  # stopped runs whose worker is ended at once, keeping nothing (the session is closing)
        self.threads = {}   # job name -> the thread that owns its worker (or its recovery)
        self.next_recovery = {}

    # ---- views ----------------------------------------------------------------------------------------------------

    def claimed(self, name):
        """True while a worker thread owns this job, including while its worker waits in the provider queue."""
        thread = self.threads.get(name)
        return thread is not None and thread.is_alive()

    def join(self, timeout=None):
        """Wait for every worker thread, up to `timeout` seconds in total."""
        deadline = None if timeout is None else time.monotonic() + timeout
        for thread in list(self.threads.values()):
            thread.join(None if deadline is None else max(0., deadline - time.monotonic()))

    def _json(self, name, filename):
        path = self.root / 'jobs' / name / filename
        return json.loads(path.read_text()) if path.is_file() else None

    def _orphan(self, row):
        """A training run or lease whose worker no thread of this process drives (a lost host, a provider fault):
        its grant stays held until its recorded worker's provider record settles it."""
        if kind_of(row) not in ('train', 'lease') or self.claimed(row['name']):
            return False
        recorded = (self.root / 'jobs' / row['name'] / 'provider.json').is_file()
        return row['status'] in ('running', 'uncertain') or (row['status'] == 'queued' and recorded)

    def jobs(self):
        """Every job with a status from harness.backend.STATUSES."""
        result = []
        for row in self.ledger.jobs():
            name, payload = row['name'], json.loads(row['payload'])
            kind = payload.get('kind', 'train')
            status = 'uncertain' if self._orphan(row) else row['status']
            job = dict(name=name, kind=kind, status=status)
            if kind == 'practice':
                job['run'] = payload['run']
                practice = self._json(name, 'practice.json')
                if practice is not None:
                    job['result'] = practice
            else:
                ended = row['started'] and row['ended']
                job.update(grant_seconds=row['grant'], worker_seconds=row['ended'] - row['started'] if ended else None)
                job[('usd_per_second' if self.ledger.mode == budgets.USD else 'seconds_weight')] = terms(row)[1]
                job['charged_' + self.ledger.unit] = row['charged'] if status not in ACTIVE else None
            if kind == 'train':
                job['config'] = payload.get('config')
                outcome = self._json(name, 'outcome.json') or {}
                if outcome.get('learning') is not None:
                    job['learning'] = outcome['learning']
                if outcome.get('ended') in EARLY:  # complete, on the network the trainer last wrote whole
                    job[outcome['ended']] = True
            elif kind == 'lease':
                job['shape'] = payload['shape']
                job['ready'] = bool(self.provider.lease_live(name))
            elif kind == 'package':
                job['lease'], job['path'], job['config'] = payload.get('lease'), payload.get('path'), payload.get('config')
            if row['error']:
                job['error'] = row['error']
            result.append(job)
        return result

    def active(self):
        return [job['name'] for job in self.jobs() if job['status'] in ACTIVE]

    def latest_practice(self):
        for row in reversed(self.ledger.jobs()):
            if kind_of(row) == 'practice' and row['status'] == 'complete':
                return self._json(row['name'], 'practice.json')
        return None

    def charge(self, shape):
        """What one wall second of a worker of `shape` costs, in the session's unit."""
        if self.ledger.mode == budgets.USD:
            return rate(shape)
        return budgets.seconds_weight(shape, TRAIN_SHAPE)

    def costs(self, seconds, charge):
        """A request's per-second charge and held grant, under the session unit's names."""
        if self.ledger.mode == budgets.USD:
            return dict(usd_per_second=charge, reserved_usd=seconds * charge)
        return dict(seconds_weight=charge, reserved_seconds=seconds * charge)

    def budget(self):
        state = self.ledger.public()
        config = state['config']
        # The first is the shape `training/launch.py` leases by default and a recipe's final training gets.
        shapes = {'lease --gpu 1 --cpu 4 --memory 16': TRAIN_SHAPE,
                  'lease --gpu 1 --cpu 16 --memory 64': dict(gpu=1, cpu_cores=16, memory_mib=65536),
                  'lease --gpu 0 --cpu 8 --memory 16': dict(gpu=0, cpu_cores=8, memory_mib=16384),
                  'lease --gpu 0 --cpu 32 --memory 64': dict(gpu=0, cpu_cores=32, memory_mib=65536),
                  'lease --gpu 0 --cpu 1 --memory 2': dict(gpu=0, cpu_cores=1, memory_mib=2048)}
        # Held budget is shown apart from spent budget: a researcher that read a running job's whole grant as
        # spent stopped 31 runs to "save" a budget that was never short (2026-11-14).
        unit = self.ledger.unit
        held = {row['name']: row['grant'] * terms(row)[1]
                for row in self.ledger.jobs() if row['status'] in ACTIVE and terms(row)[1]}
        spent = max(0., state['charged_or_reserved_' + unit] - sum(held.values()))
        common = {'budget_mode': self.ledger.mode, 'spent_' + unit: spent, 'held_' + unit: sum(held.values()),
                  'held_by_job': {name: round(amount, 4) for name, amount in held.items()},
                  'reading': f'spent_{unit} is gone. held_{unit} is NOT spent: it is the whole grant of each job '
                             'still queued or running (held_by_job). When such a job ends it is charged only for '
                             'the seconds its worker lived and the rest of its grant comes back, so stopping a job '
                             f'early to "save" its held grant only throws away its training. remaining_{unit} '
                             f'(= allocatable_{unit}) is what you can grant to new work right now: budget - spent '
                             '- held.',
                  'min_run_seconds': config['min_run_seconds'], 'min_lease_seconds': config['min_lease_seconds'],
                  'max_lease_seconds': config.get('max_lease_seconds'),
                  'lease': dict(gpu='0 or 1', max_cpu=config['lease_max_cpu_cores'],
                                max_memory_gib=config['lease_max_memory_mib'] // 1024, scratch='/scratch',
                                work='/work'),
                  'policy_parameter_cap': config['policy_parameter_cap'],
                  'critic_parameter_cap': config['critic_parameter_cap'], 'lifecycle': state['lifecycle'],
                  'smoke': self.smoke}
        if self.ledger.mode == budgets.SECONDS:
            weights = {name: round(budgets.seconds_weight(shape, TRAIN_SHAPE), 6) for name, shape in shapes.items()}
            return dict(common, budget_seconds=config['budget_seconds'], used_seconds=state['charged_or_reserved_seconds'],
                        remaining_seconds=state['remaining_seconds'], allocatable_seconds=state['allocatable_seconds'],
                        concurrency=self.ledger.limit('max_concurrent_jobs'), practice_concurrency=1,
                        charging='budget, used and remaining are seconds of the training worker (L40S, 4 CPU, '
                                 '16 GiB). One lease at a time. A lease is charged its worker seconds x the weight of '
                                 'its shape (its list price relative to the training worker). A queued or running job '
                                 'holds its whole grant until it ends.',
                        seconds_weight=weights,
                        affordable_seconds={name: int(state['allocatable_seconds'] / budgets.seconds_weight(
                            shape, TRAIN_SHAPE)) for name, shape in shapes.items()},
                        practice='CPU-only public practice is reported separately and is not charged to this budget; '
                                 'one practice job at a time, and it may run beside your one lease')
        return dict(common, budget_usd=config['budget_usd'], used_usd=state['charged_or_reserved_usd'],
                    remaining_usd=state['remaining_usd'], allocatable_usd=state['allocatable_usd'],
                    concurrency=self.ledger.limit('max_concurrent_jobs'),
                    concurrent_gpus=self.ledger.limit('max_concurrent_gpus'),
                    concurrent_cpu_cores=self.ledger.limit('max_concurrent_cpu_cores'), practice_concurrency=1,
                    charging='budget, used and remaining are US dollars. A worker is charged its '
                             'seconds x the price per second of its shape: GPU $%.2f/h, CPU core $%.4f/h, GiB '
                             'of memory $%.4f/h. Queued and running work holds its whole grant until it ends.'
                             % tuple(3600 * PRICE[key] for key in ('gpu', 'cpu_core', 'memory_gib')),
                    usd_per_hour={name: round(3600 * rate(shape), 4) for name, shape in shapes.items()},
                    affordable_seconds={name: int(state['allocatable_usd'] / rate(shape))
                                        for name, shape in shapes.items()},
                    practice='CPU-only public practice is reported separately and is not charged to this budget; '
                             'one practice job at a time, alongside your other jobs')

    def log(self, name, tail=80):
        result = next((job for job in self.jobs() if isinstance(name, str) and job['name'] == name), None)
        if result is None:
            raise ValueError('Unknown run')
        if 'error' in result:
            result['failure'] = result.pop('error')  # a reply's own `error` key means the request itself failed
        folder = self.root / 'jobs' / name
        if result['kind'] == 'train':
            for label, filename in (('stdout', 'stdout.log'), ('stderr', 'stderr.log')):
                path = folder / filename
                if path.is_file():
                    result[label] = '\n'.join(path.read_text(errors='replace').splitlines()[-tail:])[-20000:]
            counts = self._json(name, 'parameter-counts.json')
            if counts is not None:
                result['parameters'] = counts
        if (folder / 'exec').is_dir():  # a lease's commands, as last recorded by the host
            commands = [json.loads(path.read_text()) for path in sorted((folder / 'exec').glob('*.json'))]
            result['commands'] = [dict({key: c.get(key) for key in ('exec', 'state', 'exit_code')},
                                       command=c.get('command', '')[:200]) for c in commands[-tail:]]
        if result['kind'] in ('train', 'package'):
            practice = {job['name']: job.get('result', job['status']) for job in self.jobs()
                        if job['kind'] == 'practice' and job['run'] == name}
            if practice:
                result['practice_runs'] = practice
        return result

    # ---- validation -----------------------------------------------------------------------------------------------

    def _lease(self, name):
        """The provider must be holding this lease's worker for a command to reach it."""
        row = self.ledger.job(name) if isinstance(name, str) else None
        if not row or kind_of(row) != 'lease':
            raise ValueError('Unknown lease')
        if row['status'] == 'queued':
            raise ValueError('The lease has not started yet; ar wait ' + name + ' --started')
        if row['status'] != 'running':
            raise ValueError('The lease has ended (' + row['status'] + ')')
        return row

    def _native(self, files):
        """The exact native trainer files of one request: the editable sources, the recipe and the architecture."""
        if set(files) != set(NATIVE) or any(not isinstance(text, str) for text in files.values()):
            raise ValueError('Exact editable source whitelist required')
        for name in EDITABLE:
            try:
                ast.parse(files[name], filename=name)
            except SyntaxError as exc:
                raise ValueError(f'{name} does not parse: {exc}') from None
        recipe_of(files, ring=self.starter == 'ppo')
        return dict(files)

    def check(self, files):
        """Free of allocation. The parameter count builds the networks on a short CPU worker."""
        files = self._native(files)
        folder = self.root / 'checks' / str(time.time_ns())
        self._write_tree(folder, files)
        counts = self.provider.parameter_counts(folder)
        if counts.get('errors'):
            raise ValueError('; '.join(counts['errors']) + f" (policy {counts['policy']}/{counts['policy_cap']}, "
                             f"critic {counts['critic']}/{counts['critic_cap']})")
        return dict(checked=True, parameters=counts)

    @staticmethod
    def _write_tree(folder, files):
        for relative, text in files.items():
            target = folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)

    # ---- requests -------------------------------------------------------------------------------------------------

    def queue_run(self, name, seconds, files, config=None):
        """Queue one training run of exactly these files. Its whole grant is held, in the session's unit, until the
        worker ends."""
        files = self._native(files)
        if (self.root / 'jobs' / name).exists():
            raise ValueError('Fresh run name required; failed and stopped runs keep their names')
        if self.smoke and any(kind_of(row) == 'train' and row['status'] == 'complete' for row in self.ledger.jobs()):
            raise ValueError('One successful training run suffices for this smoke')
        seconds, charge = float(int(seconds)), self.charge(TRAIN_SHAPE)
        self.ledger.queue(name, seconds, {'kind': 'train', 'files': files, 'config': config,
                                          'shape': dict(TRAIN_SHAPE), 'rate': charge},
                          minimum=self.ledger.config['min_run_seconds'])
        return dict(queued=name, worker_seconds=seconds, **self.costs(seconds, charge))

    def lease(self, name, seconds, gpu, cpu, memory, files):
        shape = lease_shape(self.ledger.config, gpu, cpu, memory)
        cap = self.ledger.config.get('max_lease_seconds')
        if cap and int(seconds) > cap:
            raise ValueError(f'A lease is at most {int(cap)} seconds in this session')
        if (self.root / 'jobs' / name).exists():
            raise ValueError('Fresh run name required; failed and stopped runs keep their names')
        seconds, charge = float(int(seconds)), self.charge(shape)
        self.ledger.queue(name, seconds, {'kind': 'lease', 'files': files, 'shape': shape, 'rate': charge},
                          minimum=self.ledger.config['min_lease_seconds'])
        return dict(queued=name, worker_seconds=seconds, **self.costs(seconds, charge), shape=shape)

    def lease_exec(self, name, command):
        self._lease(name)
        if len(command.encode()) > self.ledger.config.get('lease_command_bytes', DEFAULT['lease_command_bytes']):
            raise ValueError('Command too long; put it in a file under training/ and ar push')
        return self.provider.lease_exec(name, command)

    def lease_poll(self, name, exec_id, tail=None):
        # A finished lease's commands stay readable from the host's record.
        saved = self.root / 'jobs' / str(name) / 'exec' / (exec_id + '.json')
        if not self.provider.lease_live(name) and saved.is_file():
            return json.loads(saved.read_text())
        self._lease(name)
        return self.provider.lease_poll(name, exec_id, tail)

    def lease_push(self, name, files):
        self._lease(name)
        return self.provider.lease_push(name, files)

    def lease_sync(self, name, reload=False):
        self._lease(name)
        return self.provider.lease_sync(name, reload=bool(reload))

    def package(self, name, lease, path, files, config=None):
        """Seal a checkpoint a lease produced, with exactly these files, as a run that practice and submit accept."""
        lease = self._lease(lease)
        files = self._native(files)
        if self.ledger.job(name) or (self.root / 'jobs' / name).exists():
            raise ValueError('Fresh run name required; failed and stopped runs keep their names')
        if self.ledger.lifecycle() != 'active':
            raise ValueError('Session finalized; no new networks')
        bundle = self.provider.lease_package(self.root / 'jobs' / name, lease['name'], path, files)
        identity = verify(bundle)
        self.ledger.register(name, bundle, identity, {'kind': 'package', 'lease': lease['name'], 'path': path,
                                                      'config': config})
        return dict(packaged=name, lease=lease['name'], sha256=identity)

    def queue_practice(self, name):
        run = self.ledger.job(name) if isinstance(name, str) else None
        if not run or run['status'] != 'complete' or not run['artifact']:
            raise ValueError('Practice requires a completed run')
        rows = [row for row in self.ledger.jobs() if kind_of(row) == 'practice']
        busy = [row['name'] for row in rows if row['status'] in ACTIVE]
        if busy:
            raise ValueError('One practice job at a time; wait for or stop: ' + ', '.join(busy))
        job = f"eval-{name}-{1 + sum(json.loads(row['payload'])['run'] == name for row in rows)}"
        # Practice holds CPU workers outside the budget: rate 0, no grant.
        self.ledger.queue(job, 0., {'kind': 'practice', 'run': name, 'bundle': run['artifact'],
                                    'shape': dict(PRACTICE_SHAPE), 'rate': 0.})
        return dict(queued=job, run=name, charged_to_allocation=False)

    def stop(self, name, now=False):
        """Stop a job. A training run that is training is asked to end at its next step and keeps its network; `now`
        (the session is closing) ends its worker at once and keeps nothing.

        Queued work no worker was created for is cancelled uncharged. A worker waiting for a GPU is abandoned
        uncharged through its thread. An unresolved job is settled now from its recorded worker's provider record
        (never started: nothing charged)."""
        row = self.ledger.job(name) if isinstance(name, str) else None
        if row is None:
            raise ValueError('Unknown run')
        kind = kind_of(row)
        if self._orphan(row):
            self.stops.add(name)
            if not (self.root / 'jobs' / name / 'provider.json').is_file():
                raise ValueError('Run is unresolved but no worker was recorded for it, so nothing can be terminated; '
                                 'its hold stays until the provider record is reconciled')
            settled = self._resolve(name)
            if settled:
                return dict(stopped=name, **{'charged_' + self.ledger.unit: settled['charged']})
            return dict(stopping=name, note='Its worker was asked to stop; the allocation settles from the provider '
                                            'record as soon as that record is final. Check ar status.')
        if row['status'] not in ACTIVE:
            raise ValueError('Run is not active')
        if not self.claimed(name):
            if row['status'] == 'queued':
                self.ledger.cancel_queued(name, 'Cancelled before any worker started')
            else:  # practice whose thread is gone: it holds no allocation, so it is simply lost
                self._lost(row)
            return dict(stopped=name, **{'charged_' + self.ledger.unit: 0})
        self.stops.add(name)
        if kind == 'train' and now:
            self.halts.add(name)
            self.provider.stop(name)  # as the deadline timer does; the provider receipt still settles the run
        elif kind == 'train':
            # The provider's wait sees the request (provider.train): the trainer is asked to end, then collected.
            return dict(stopping=name, note=STOP_NOTE)
        return dict(stopping=name)

    def submit(self, name):
        run = self.ledger.job(name) if isinstance(name, str) else None
        if not run or run['status'] != 'complete' or not run['artifact']:
            raise ValueError('Only a finished run can be submitted')
        if self.ledger.lifecycle() != 'active':
            raise ValueError('Session finalized')
        if verify(run['artifact']) != run['artifact_sha']:
            raise ValueError('The run differs from its sealed identity')
        target = self.sealed(run['artifact_sha'])
        if not target.exists():
            shutil.copytree(run['artifact'], target)
        verify(target)
        for path in target.rglob('*'):
            if path.is_file():
                path.chmod(0o444)
        entry = self.ledger.submit(name)
        return dict(run=name, sha256=entry['artifact_sha'], submitted_at=entry['at'])

    def sealed(self, identity):
        """Where the read-only copy of a submitted bundle is kept."""
        return self.root / 'submissions' / identity

    def submission(self):
        """The latest submission (never its host path), or None."""
        entries = self.ledger.submissions()
        if not entries:
            return None
        return dict(run=entries[-1]['job'], sha256=entries[-1]['artifact_sha'], submitted_at=entries[-1]['at'])

    def finalize(self):
        if self.ledger.lifecycle() != 'active':
            raise ValueError('Session finalized')
        latest = self.submission()
        if not latest:
            raise ValueError('Submit a completed run before finalization')
        if self.active():
            raise ValueError('Allocation unresolved')
        verify(self.sealed(latest['sha256']))
        state = self.ledger.finalize()
        unit = self.ledger.unit
        return dict(finalized=True, submission=latest['sha256'], **{'used_' + unit: state['charged_or_reserved_' + unit]})

    # ---- workers --------------------------------------------------------------------------------------------------

    def tick(self):
        """Start whatever can start, and settle a worker no thread drives from its provider record. Never blocks."""
        for name in [name for name, thread in self.threads.items() if not thread.is_alive()]:
            del self.threads[name]
        for row in self.ledger.jobs():
            name, kind = row['name'], kind_of(row)
            if name in self.threads or row['status'] not in ACTIVE:
                continue
            if kind == 'practice':
                if row['status'] == 'queued':
                    self._start(name, self._practice, row)
                else:
                    self._lost(row)
            elif self._orphan(row):
                # Only the recorded worker is asked about; a tick never queues a replacement allocation or clears
                # a hold whose end is unknown.
                if time.time() >= self.next_recovery.get(name, 0) and hasattr(self.provider, 'lookup'):
                    self.next_recovery[name] = time.time() + RECOVERY_RETRY_SECONDS
                    self._start(name, self._resolve, name)
            elif row['status'] == 'queued':
                self._start(name, self._work, row)

    def _start(self, name, target, argument):
        self.threads[name] = threading.Thread(target=target, args=(argument,), daemon=True)
        self.threads[name].start()

    def _lost(self, row):
        """A practice job whose thread is gone: a CPU practice worker holds no allocation, so it is simply lost."""
        now = time.time()
        Ledger(self.root).settle(row['name'], dict(started_at=row['started'] or now, finished_at=now), 'failed',
                                 'Host interruption; this practice job was lost and nothing was charged')

    def _work(self, row):
        """One training run or lease, on its own thread with its own SQLite connection."""
        ledger = Ledger(self.root)
        name, payload = row['name'], json.loads(row['payload'])
        hold = Allocation(ledger, lambda: name in self.stops)
        out = self.root / 'jobs' / name
        try:
            if payload['kind'] == 'lease':
                self.provider.hold(hold, name, int(row['grant']), payload['shape'], payload['files'], out,
                                   stop=lambda: name in self.stops)
            else:
                snapshot = self.root / 'queue' / name
                self._write_tree(snapshot, payload['files'])
                result = self.provider.train(hold, name, int(row['grant']), snapshot, out, smoke=self.smoke,
                                             stop=lambda: 'now' if name in self.halts else name in self.stops)
                # A run the researcher stopped, or one cut off at its time limit, that left a whole verified
                # network is complete, and says how it ended.
                write(out / 'outcome.json', dict(learning=result['learning'], ended=result.get('ended')))
                ledger.attach(name, result['bundle'], verify(result['bundle']))
        except workers.NeverStarted as exc:
            # Created but never given a worker: terminated and settled uncharged, never left uncertain.
            if ledger.job(name)['status'] in ('queued', 'uncertain'):
                ledger.abandon_unstarted(name, exc.receipt, str(exc), stopped=name in self.stops)
        except BaseException as exc:
            message, stopped = str(exc)[:2000], name in self.stops
            status = ledger.job(name)['status']
            if status == 'queued':  # no worker was created: a refused network, a stop, a host fault
                ledger.cancel_queued(name, message, 'stopped' if stopped else 'failed')
            elif status == 'running':
                ledger.uncertain(name, message)
            elif status == 'complete':  # settled as complete, then its bundle could not be bound
                write(out / 'host-error.json', dict(type=type(exc).__name__, message=message))
            elif status in ('failed', 'stopped'):
                ledger.annotate(name, message, stopped=stopped)

    def _practice(self, row):
        ledger = Ledger(self.root)
        name, payload = row['name'], json.loads(row['payload'])
        folder = self.root / 'jobs' / name
        started = time.time()
        ledger.start(name, 'cpu-practice-workers', started)
        try:
            folder.mkdir(parents=True, exist_ok=False)
            if self.smoke:
                result = self.provider.evaluate(payload['bundle'], SMOKE_PRACTICE['seeds'],
                                                SMOKE_PRACTICE['opponent_seed'], folder / 'smoke.json')
            else:
                write(folder / 'schedule.json', FULL_PRACTICE)
                result = grade_submission(_Stoppable(self.provider, lambda: name in self.stops), payload['bundle'],
                                          self.release, folder, False)
            write(folder / 'practice.json', dict(result, run=payload['run']))
            status, error = 'complete', None
        except BaseException as exc:
            status, error = ('stopped' if name in self.stops else 'failed'), str(exc)[:2000]
        ledger.settle(name, dict(started_at=started, finished_at=time.time()), status, error)

    def _resolve(self, name):
        """Settle an unresolved hold from its recorded worker's provider record. Never adopts an artifact or restarts work.

        The worker no thread of this process drives is terminated first (harness.workers): never started settles
        uncharged; started settles from its start/finish receipt, without waiting out the grant.
        """
        record = self.root / 'jobs' / name / 'provider.json'
        if not record.is_file():
            return None
        data = json.loads(record.read_text())
        if hasattr(self.provider, 'abandon'):
            receipt = self.provider.abandon(data['app_id'], data['sandbox_id'])
        else:  # a provider that can only read receipts waits for the worker's own end
            receipt = self.provider.lookup(data['app_id'], data['sandbox_id'])
        receipt = receipt or {}
        stopped = name in self.stops
        ledger = Ledger(self.root)
        try:
            if receipt.get('provider_id') and workers.never_started(receipt):
                ledger.abandon_unstarted(name, receipt, STOPPED_QUEUED if stopped else workers.NEVER_STARTED,
                                         stopped=stopped)
                never = True
            elif receipt.get('started_at') and receipt.get('finished_at'):
                ledger.settle(name, receipt, 'stopped' if stopped else 'failed',
                              'Stopped while unresolved; its recorded worker was terminated and charged from its '
                              'provider receipt' if stopped else
                              'Controller interruption; authoritative lifetime recovered; no artifact auto-submitted')
                never = False
            else:
                return None  # not final yet: stays held, asked again later
        except ValueError:
            return None  # settled meanwhile by its own worker thread
        write(record.with_name('recovered-interruption.json'),
              {'receipt': receipt, 'status': 'assisted-failure', 'never_started': never, 'stop_requested': stopped,
               'unsubmitted_artifact_adopted': False})
        return {'job': name, 'charged': ledger.job(name)['charged']}

    def settle(self, timeout=None):
        """Cancel queued work, end running work at once and wait for the worker threads. Returns what is still active.

        Nothing unsubmitted is kept at a close, so a training worker is terminated instead of asked to finish."""
        for name in self.active():
            try:
                self.stop(name, now=True)
            except ValueError:
                pass  # an unresolved hold with no recorded worker has nothing to stop; it stays held
        self.join(timeout)
        return self.active()


# A shard whose CPU worker was lost (Modal shut the sandbox down, or the connection to it dropped) is played again on
# a fresh worker: same maps, same seats and same opponent, so the score cannot depend on the loss. Anything the
# evaluation itself reported (a failed grade.py, a policy error, a stop request) is raised as before, and a shard that
# already wrote its result is never replayed. The lost attempt's worker records are kept beside the shard as
# <shard>.lost-<n>.*, with the error in <shard>.lost-<n>.error.json.
SHARD_ATTEMPTS = 3
LOST_WORKER = ('NotFoundError', 'SandboxTerminatedError', 'StreamTerminatedError')


def _lost_worker(exc):
    chain = [e for e in (exc, exc.__cause__, exc.__context__) if e is not None]
    return any(isinstance(e, ConnectionError) or type(e).__name__ in LOST_WORKER for e in chain)


def _evaluate_shard(provider, bundle, seeds, path, opponent):
    path = Path(path)
    for attempt in range(1, SHARD_ATTEMPTS + 1):
        try:
            return provider.evaluate(bundle, seeds, 0, path, opponent)
        except Exception as exc:
            if attempt == SHARD_ATTEMPTS or path.exists() or not _lost_worker(exc):
                raise
            for f in list(path.parent.glob(path.stem + '.*')):
                if not f.name.startswith(path.stem + '.lost-'):
                    f.rename(f.with_name(f'{path.stem}.lost-{attempt}{f.name[len(path.stem):]}'))
            write(path.with_name(f'{path.stem}.lost-{attempt}.error.json'),
                  {'type': type(exc).__name__, 'message': str(exc)[:2000], 'replayed': True})


def grade_submission(provider,bundle,release,private,smoke):
 schedule=json.loads((Path(private)/'schedule.json').read_text())
 if smoke:return provider.evaluate(bundle,schedule['seeds'],schedule['opponent_seed'],Path(private)/'grade.json')
 panel=Path(release)/'host-private-panel/inventory.json'
 if not panel.is_file():raise ValueError('This release was prepared without practice opponents (panel=false): it can train, but full practice and grading need a release prepared with them')
 inventory=json.loads(panel.read_text());rows=[];breakdown={};cpu_seconds=0.
 # Every (opponent, four-map shard) is one CPU worker; up to EVAL_PARALLEL run at once. Results are read back in
 # schedule order, so the score does not depend on which worker finished first. A lost worker is replayed
 # (_evaluate_shard); any other failure (or a stop request, which _Stoppable raises before a shard starts) cancels
 # the shards not yet started and is raised.
 from concurrent.futures import ThreadPoolExecutor
 shards=[(opponent,start,Path(private)/'grades'/opponent['name']/(str(start)+'.json'))
         for opponent in inventory['panel'] for start in range(0,len(schedule['seeds']),4)]
 pool=ThreadPoolExecutor(max_workers=EVAL_PARALLEL)
 try:
  futures=[pool.submit(_evaluate_shard,provider,bundle,schedule['seeds'][start:start+4],path,Path(release)/opponent['directory'])
           for opponent,start,path in shards]
  for future in futures:future.result()
 finally:pool.shutdown(wait=True,cancel_futures=True)
 for opponent in inventory['panel']:
  name=opponent['name'];oprows=[]
  for _,start,path in (s for s in shards if s[0] is opponent):
   detail=json.loads(path.read_text());oprows.extend(detail['details']);cpu_seconds+=detail['cpu_seconds']
  for row in oprows:row['opponent']=name
  rows.extend(oprows);breakdown[name]={'raw_score_0_100':100*sum((r['reward']+1)/2 for r in oprows)/len(oprows),'games':len(oprows)}
 import numpy as np
 by_map={seed:np.mean([(r['reward']+1)/2 for r in rows if r['seed']==seed]) for seed in schedule['seeds']}
 values=np.array(list(by_map.values()));boots=np.random.default_rng(0).choice(values,(2000,len(values)),replace=True).mean(axis=1)
 result={'raw_score_0_100':float(values.mean()*100),'normalized_reference_score':None,'games':len(rows),'by_opponent':breakdown,'by_seat':{str(s):100*float(np.mean([(r['reward']+1)/2 for r in rows if r['seat']==s])) for s in (1,2)},'natural_games':sum(not r['capped'] for r in rows),'capped_games':sum(r['capped'] for r in rows),'illegal_actions':0,'simulator_faults':0,'ci95_map_cluster':[float(x*100) for x in np.quantile(boots,[.025,.975])],'cpu_seconds':cpu_seconds}
 write(Path(private)/'full-grade.json',dict(result,details=rows));return result
