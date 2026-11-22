"""Host-owned Hex session authority: ledger, job queue, leases, practice and submissions."""
import ast
import json
from pathlib import Path
import threading
import time
import uuid

from harness import workers
from simulators.hex.autoresearch.bundle import verify
from simulators.hex.autoresearch.contract import DEFAULT, EDITABLE, PRACTICE_SHAPE, TRAIN_SHAPE, lease_shape, write_json
from simulators.hex.autoresearch.ledger import Ledger
from simulators.hex.autoresearch.manage import panel

PRACTICE_MIN_SECONDS = 30
# Practice openings and seeds are public and fixed, so practice results are comparable across runs.
PRACTICE_SEEDS = {'split': 'practice', 'opening_seed': 20210407, 'policy_seed': 31130}
# How often tick() re-asks the provider about the same unresolved job (each ask lists Modal sandboxes).
RECOVERY_RETRY_SECONDS = 30


class Controller:
    def __init__(self, root, release, provider, config=None):
        self.root = Path(root)
        self.release = Path(release)
        self.provider = provider
        self.ledger = Ledger(root, config)
        self.stops = set()
        self.threads = {}  # job name -> the thread that owns its worker (or its recovery)
        self.next_recovery = {}

    def claimed(self, name):
        """True while a worker thread owns this job, including before the provider reports its start."""
        thread = self.threads.get(name)
        return thread is not None and thread.is_alive()

    def join(self, timeout=None):
        """Wait for every worker thread, up to `timeout` seconds in total."""
        deadline = None if timeout is None else time.monotonic() + timeout
        for thread in list(self.threads.values()):
            thread.join(None if deadline is None else max(0., deadline - time.monotonic()))

    def _code(self, code):
        for name, text in code.items():
            ast.parse(text, filename=name)
        if set(code) != set(EDITABLE) or any(not isinstance(v, str) for v in code.values()):
            raise ValueError('Exact editable source whitelist required')
        if sum(len(v.encode()) for v in code.values()) > DEFAULT['source_bytes_cap']:
            raise ValueError('Source transport cap exceeded')
        return code

    def _charge(self, shape, seconds):
        """The rate a worker of `shape` is charged in this session's unit, and how a reply states its grant:
        dollars per second in dollar mode, training-worker seconds per second (`seconds_weight`) in seconds mode."""
        charge = self.ledger.rate(shape)
        return charge, {'worker_seconds': seconds, ('usd_per_second' if self.ledger.unit == 'usd' else 'seconds_weight'):
                        charge, 'reserved_' + self.ledger.unit: seconds * charge}

    def _lease(self, name):
        """The provider must be holding this lease's worker for a command to reach it."""
        row = next((r for r in self.ledger.jobs() if r['name'] == name), None)
        if not row or json.loads(row['payload']).get('kind') != 'lease':
            raise ValueError('Unknown lease')
        if row['status'] == 'queued':
            raise ValueError('The lease has not started yet; ar wait ' + name + ' --started')
        if row['status'] != 'running':
            raise ValueError('The lease has ended (' + row['status'] + ')')
        return row

    def handle(self, request):
        # Runs in one broker thread. Source/config/timing returned by a worker never
        # replace session settings, immutable snapshots or provider receipts.
        action = request.get('action')
        if action in ('budget', 'status'):
            return self.ledger.public()
        if action == 'run':
            code = self._code(request.get('code', {}))
            limit = self.ledger.config.get('max_training_jobs')
            if limit is not None and sum(json.loads(j['payload']).get('kind') == 'train'
                                         for j in self.ledger.jobs()) >= limit:
                raise ValueError('Session training-job cap reached')
            if len(json.dumps(request['config']).encode()) > 65536:
                raise ValueError('Recipe exceeds cap')
            name, seconds = request['name'], float(request['seconds'])
            if seconds > self.ledger.config.get('max_run_seconds', float('inf')):
                raise ValueError('Session per-run cap exceeded')
            charge, grant = self._charge(TRAIN_SHAPE, seconds)
            self.ledger.queue(name, seconds, {'kind': 'train', 'code': code, 'config': request['config'],
                                              'seed': int(request.get('seed', 1)), 'shape': dict(TRAIN_SHAPE),
                                              'rate': charge})
            return dict(grant, queued=name)
        if action == 'lease':
            if self.ledger.config.get('leases') is False:
                raise ValueError('This session has no leases')
            shape = lease_shape(self.ledger.config, request['gpu'], request['cpu'], request['memory'])
            name, seconds = request['name'], float(request['seconds'])
            if seconds > self.ledger.config.get('max_run_seconds', float('inf')):
                raise ValueError('Session per-run cap exceeded')
            charge, grant = self._charge(shape, seconds)
            self.ledger.queue(name, seconds, {'kind': 'lease', 'files': request['files'], 'shape': shape,
                                              'rate': charge})
            return dict(grant, queued=name, shape=shape)
        if action in ('exec', 'poll', 'push', 'sync'):
            name = request['name']
            if action == 'poll':  # a finished lease's commands stay readable from the host's record
                saved = self.root / 'jobs' / name / 'exec' / (request['exec'] + '.json')
                if not self.provider.lease_live(name) and saved.is_file():
                    return json.loads(saved.read_text())
            self._lease(name)
            if action == 'exec':
                command = request['command']
                if len(command.encode()) > self.ledger.config.get('lease_command_bytes', DEFAULT['lease_command_bytes']):
                    raise ValueError('Command too long; put it in a file under training/ and ar push')
                return self.provider.lease_exec(name, command)
            if action == 'poll':
                return self.provider.lease_poll(name, request['exec'], request.get('tail'))
            if action == 'push':
                return self.provider.lease_push(name, request['files'])
            return self.provider.lease_sync(name, reload=bool(request.get('reload')))
        if action == 'package':
            lease = self._lease(request['lease'])
            code = self._code(request.get('code', {}))
            if len(json.dumps(request['config']).encode()) > 65536:
                raise ValueError('Recipe exceeds cap')
            name = request['name']
            if any(r['name'] == name for r in self.ledger.jobs()):
                raise ValueError('Run name already used')
            artifact, identity = self.provider.lease_package(self.root / 'jobs' / name, lease['name'], request['path'],
                                                             code, request['config'])
            self.ledger.register(name, artifact, identity, {'kind': 'package', 'lease': lease['name'],
                                                            'path': request['path']})
            return {'packaged': name, 'lease': lease['name'], 'bundle_sha256': identity}
        if action == 'eval':
            trained = next((r for r in self.ledger.jobs() if r['name'] == request['name']), None)
            if not trained or not trained['artifact']:
                raise ValueError('Practice requires a completed registered run')
            verify(trained['artifact'], trained['artifact_sha'], check_tensors=False)
            name = 'eval-' + uuid.uuid4().hex[:12]
            seconds = float(request['seconds'])
            charge, grant = self._charge(PRACTICE_SHAPE, seconds)
            self.ledger.queue(name, seconds, {
                'kind': 'practice', 'bundle': trained['artifact'], 'panel': panel(self.release, 'practice'),
                'seeds': dict(PRACTICE_SEEDS, openings=self.ledger.config['practice_openings']),
                'shape': dict(PRACTICE_SHAPE), 'rate': charge},
                minimum_seconds=PRACTICE_MIN_SECONDS)
            return dict(grant, queued=name, budgeted_practice=True)
        if action == 'stop':
            r = next((r for r in self.ledger.jobs() if r['name'] == request['name']), None)
            if not r:
                raise ValueError('Unknown run')
            return dict(self.release_job(r) or {}, stopping=r['name'])
        if action == 'submit':
            r = next((r for r in self.ledger.jobs() if r['name'] == request['name']), None)
            if not r or not r['artifact']:
                raise ValueError('No registered trained artifact')
            verify(r['artifact'], r['artifact_sha'], check_tensors=False)
            return {'submission': self.ledger.submit(r['name']), 'bundle_sha256': r['artifact_sha']}
        if action == 'finalize':
            return self.ledger.finalize()
        raise ValueError('Unsupported broker action')

    def release_job(self, job):
        """Stop or settle one job now, never leaving the session blocked by it.

        Unclaimed queued -> cancelled, uncharged. Claimed queued (waiting for a GPU) or running -> stopped through
        its worker thread (a worker Modal never started is abandoned uncharged). Uncertain -> its recorded worker
        is terminated and settled from Modal's record (never started: 0 charged).
        """
        if job['status'] == 'queued' and not self.claimed(job['name']):
            self.ledger.cancel_queued(job['name'])
        elif job['status'] in ('queued', 'running'):
            self.stops.add(job['name'])
        elif job['status'] == 'uncertain':
            self.stops.add(job['name'])
            if self.claimed(job['name']):
                return {'note': 'Already being reconciled with the provider'}
            if not hasattr(self.provider, 'recover'):
                raise ValueError('This provider cannot reconcile unresolved work')
            self.provider.recover(Ledger(self.root), job, stopped=True)
            settled = next(r for r in self.ledger.jobs() if r['name'] == job['name'])
            return {'status': settled['status'], 'charged_' + self.ledger.unit: settled['charged'],
                    'error': settled['error']}
        else:
            raise ValueError('Run is not active')
        return {}

    def tick(self):
        """Start whatever can start. Each job has its own worker thread, so leases, training and practice run
        side by side up to the ledger's concurrency limits (one job at a time in seconds mode)."""
        for name in [name for name, thread in self.threads.items() if not thread.is_alive()]:
            del self.threads[name]
        for job in self.ledger.jobs():
            name = job['name']
            if name in self.threads:
                continue
            if job['status'] in ('running', 'uncertain'):
                # No thread owns it (a lost host, or a provider fault). Settle it from Modal's record only; a tick
                # never clears a hold whose end is unknown. Ask again at most every RECOVERY_RETRY_SECONDS.
                if not hasattr(self.provider, 'recover') or time.time() < self.next_recovery.get(name, 0):
                    continue
                self.next_recovery[name] = time.time() + RECOVERY_RETRY_SECONDS
                self._start(name, self._recover, job)
            elif job['status'] == 'queued':
                self._start(name, self._work, job)

    def _start(self, name, target, job):
        self.threads[name] = threading.Thread(target=target, args=(job,), daemon=True)
        self.threads[name].start()

    def _recover(self, job):
        self.provider.recover(Ledger(self.root), job, stopped=job['name'] in self.stops)

    def _work(self, job):
        # Each worker thread opens its own SQLite connection; writes serialize via BEGIN IMMEDIATE.
        ledger = Ledger(self.root)
        name = job['name']
        try:
            self.provider.execute(ledger, job, lambda: name in self.stops)
        except workers.NeverStarted as exc:
            # The provider terminated a worker Modal never started and settled it uncharged; nothing is held.
            write_json(self.root / 'jobs' / name / 'host-error.json',
                       {'type': 'NeverStarted', 'message': str(exc), 'provider': exc.receipt,
                        'charged_' + ledger.unit: 0})
            current = next(j for j in ledger.jobs() if j['name'] == name)
            if current['status'] in ('queued', 'uncertain'):
                ledger.abandon_unstarted(name, exc.receipt, str(exc), stopped=name in self.stops)
        except BaseException as exc:
            ledger.uncertain(name, type(exc).__name__ + ': ' + str(exc))
