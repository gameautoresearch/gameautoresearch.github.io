"""Host-owned Generals session authority: ledger, job queue, leases, practice, submissions and grading."""
import ast
import json
from pathlib import Path
import threading
import time
import uuid

from harness import workers
from simulators.generals.autoresearch.bundle import verify
from simulators.generals.autoresearch.contract import (BUNDLED, DEFAULT, EDITABLE, PRACTICE_SHAPE, TRAIN_SHAPE, lease_shape,
                                                       rate, write_json)
from simulators.generals.autoresearch.ledger import Ledger

# How often tick() re-asks the provider about the same unresolved job (each ask lists Modal sandboxes).
RECOVERY_RETRY_SECONDS = 30
BUDGET_STOPS = ('grant exhausted', 'ceiling exhausted', 'reservation would cross', 'campaign deadline')


def panel_paths(release, which):
    root = Path(release) / 'private'
    catalog = json.loads((root / 'panel.json').read_text())
    if catalog.get(which) is None:
        raise ValueError('Official panel is not calibrated')
    return [dict(p, bundle=str((root / p['bundle']).resolve())) if p['kind'] in BUNDLED else p
            for p in catalog[which]]


class Controller:
    def __init__(self, root, release, provider, config=None):
        self.root = Path(root)
        self.release = Path(release)
        self.provider = provider
        self.ledger = Ledger(root, config)
        self.stops = set()
        self.threads = {}  # job name -> the thread that owns its worker (or its recovery)
        self.next_recovery = {}
        self.grade_thread = None

    def claimed(self, name):
        """True while a worker thread owns this job, including while its worker waits in the provider queue."""
        thread = self.threads.get(name)
        return thread is not None and thread.is_alive()

    def join(self, timeout=None):
        """Wait for every worker thread, up to `timeout` seconds in total."""
        deadline = None if timeout is None else time.monotonic() + timeout
        for thread in list(self.threads.values()):
            thread.join(None if deadline is None else max(0., deadline - time.monotonic()))

    def release_job(self, job):
        """Stop or settle one job now, never leaving the session blocked by it.

        Unclaimed queued -> cancelled, uncharged. Claimed queued (waiting for a GPU) or running -> stopped
        through its worker thread (a queued worker is abandoned uncharged). Uncertain -> its recorded
        worker is terminated and settled from the provider's record (never started: 0 charged).
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
            return {'status': settled['status'], 'charged_usd': settled['charged'], 'error': settled['error']}
        else:
            raise ValueError('Run is not active')
        return {}

    def _reserved(self, seconds, shape):
        """What a queued job holds, in the session's budget unit: dollars at its shape's price (dollar mode), or
        training-worker seconds at its shape's weight (seconds mode: 1.0 for the training shape)."""
        charge = rate(shape)
        if self.ledger.mode == 'seconds':
            weight = self.ledger.seconds(charge)
            return {'seconds_weight': weight, 'reserved_seconds': round(seconds * weight, 6)}
        return {'usd_per_second': charge, 'reserved_usd': seconds * charge}

    def _code(self, code):
        for name, text in code.items():
            ast.parse(text, filename=name)
        if set(code) != set(EDITABLE) or any(not isinstance(v, str) for v in code.values()):
            raise ValueError('Exact editable source whitelist required')
        if sum(len(v.encode()) for v in code.values()) > DEFAULT['source_bytes_cap']:
            raise ValueError('Source transport cap exceeded')
        return code

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

    def policy(self, name):
        """The ledger row of a training run or packaged network that finished with a verified policy package.

        Practice and submission need one. Anything else is the researcher's request to refuse (a ValueError
        naming the run and how it ended), never a host fault: a failed run has no package to open.
        """
        row = next((r for r in self.ledger.jobs() if r['name'] == name), None)
        if row is None:
            raise ValueError('Unknown run: ' + str(name))
        if json.loads(row['payload'] or '{}').get('kind', 'train') not in ('train', 'package'):
            raise ValueError('Run ' + name + ' is not a training run or a packaged network')
        if row['status'] in ('queued', 'running', 'uncertain'):
            raise ValueError('Run ' + name + ' has not finished: status ' + row['status'])
        if not row['artifact'] or not row['artifact_sha']:
            raise ValueError('Run ' + name + ' did not finish with a policy: status '
                             + {'cancelled': 'stopped'}.get(row['status'], row['status']))
        try:
            verify(row['artifact'], row['artifact_sha'], check_tensors=False)
        except (ValueError, OSError, KeyError) as exc:
            raise ValueError('The policy package of run ' + name + ' is missing or no longer valid ('
                             + type(exc).__name__ + ')') from exc
        return row

    def handle(self, request):
        # Runs in one broker thread. Source/config/timing returned by a worker never
        # replace session settings, immutable snapshots or provider receipts.
        action = request.get('action')
        if action in ('budget', 'status', 'log'):
            state = self.ledger.public()
            for job in state['jobs']:
                folder = self.root / 'jobs' / job['name']
                progress = folder / 'progress.jsonl'
                if progress.exists():
                    for line in reversed(progress.read_text().splitlines()):
                        try:
                            job['progress'] = json.loads(line)
                            break
                        except ValueError:
                            continue
                if (folder / 'practice.json').exists():
                    job['practice'] = json.loads((folder / 'practice.json').read_text())
            return state
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
            self.ledger.queue(name, seconds, {'kind': 'train', 'code': code, 'config': request['config'],
                                              'seed': int(request.get('seed', 1)), 'shape': dict(TRAIN_SHAPE),
                                              'rate': rate(TRAIN_SHAPE)})
            return dict({'queued': name, 'worker_seconds': seconds}, **self._reserved(seconds, TRAIN_SHAPE))
        if action == 'lease':
            if self.ledger.config.get('leases') is False:
                raise ValueError('This session has no leases')
            shape = lease_shape(self.ledger.config, request['gpu'], request['cpu'], request['memory'])
            name, seconds = request['name'], float(request['seconds'])
            if seconds > self.ledger.config.get('max_run_seconds', float('inf')):
                raise ValueError('Session per-run cap exceeded')
            charge = rate(shape)
            self.ledger.queue(name, seconds, {'kind': 'lease', 'files': request['files'], 'shape': shape,
                                              'rate': charge})
            return dict({'queued': name, 'worker_seconds': seconds, 'shape': shape}, **self._reserved(seconds, shape))
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
            trained = self.policy(request['name'])
            name = 'eval-' + uuid.uuid4().hex[:12]
            seconds = float(request['seconds'])
            self.ledger.queue(name, seconds, {
                'kind': 'practice', 'bundle': trained['artifact'], 'panel': panel_paths(self.release, 'practice'),
                'seeds': {'split': 'practice', 'pool_path': str(self.release / 'public/practice-maps.npz'),
                          'map_indices': list(range(self.ledger.config['practice_maps']))},
                'shape': dict(PRACTICE_SHAPE), 'rate': rate(PRACTICE_SHAPE)})
            return dict({'queued': name, 'worker_seconds': seconds, 'budgeted_practice': True},
                        **self._reserved(seconds, PRACTICE_SHAPE))
        if action == 'stop':
            r = next((r for r in self.ledger.jobs() if r['name'] == request['name']), None)
            if not r:
                raise ValueError('Unknown run')
            training = r['status'] in ('queued', 'running') and json.loads(r['payload']).get('kind') == 'train'
            settled = self.release_job(r)
            if training and not settled and r['name'] in self.stops:
                settled = {'note': 'A run that is training is ended now and keeps the network it last saved: it '
                                   'finishes as complete with stopped_early, or as stopped if it had saved none, '
                                   'and its worker time is charged. A worker still waiting for a GPU is cancelled '
                                   'uncharged. Check ar status.'}
            # Keep a resolved job's own failure out of the reply's top-level `error` (a failed command).
            if settled and 'error' in settled:
                settled['job_error'] = settled.pop('error')
            return dict(settled or {}, stopping=r['name'])
        if action == 'submit':
            r = self.policy(request['name'])
            return {'submission': self.ledger.submit(r['name']), 'bundle_sha256': r['artifact_sha']}
        if action == 'finalize':
            if self.ledger.config.get('require_smoke_practice') and not any(
                    (self.root / 'jobs' / r['name'] / 'practice.json').exists() and
                    json.loads((self.root / 'jobs' / r['name'] / 'practice.json').read_text()).get('games', 0) > 0
                    for r in self.ledger.jobs() if r['status'] == 'complete'):
                raise ValueError('This session requires a successful practice check before finalization: '
                                 'ar eval NAME --seconds 180')
            return self.ledger.finalize()
        if action in ('grade', 'grade_status'):
            if not self.ledger.config.get('allow_agent_grade', True):
                raise ValueError('Private grading is host-only for this research session; no private feedback')
            if self.ledger.public()['lifecycle'] != 'finalized':
                raise ValueError('Finalize immutable submissions before hidden grading')
            if (self.root / 'grades.json').exists():
                return {'status': 'complete', 'grades': json.loads((self.root / 'grades.json').read_text())}
            if (self.root / 'grade-error.json').exists():
                return {'status': 'failed', 'error': json.loads((self.root / 'grade-error.json').read_text())}
            if action == 'grade' and self.grade_thread is None:
                def work():
                    try:
                        from simulators.generals.autoresearch.manage import grade_session
                        grade_session(self.release, self.root, self.provider,
                                      self.ledger.config['budget_usd'] != DEFAULT['budget_usd'],
                                      maps=self.ledger.config['grade_maps'],
                                      timeout=self.ledger.config.get('host_grade_timeout_seconds'))
                    except BaseException as exc:
                        write_json(self.root / 'grade-error.json', {'type': type(exc).__name__,
                                                                    'message': str(exc)[:1000]})
                self.grade_thread = threading.Thread(target=work, daemon=True)
                self.grade_thread.start()
            return {'status': 'running' if self.grade_thread else 'not_started'}
        raise ValueError('Unsupported broker action')

    def tick(self):
        """Start whatever can start. Each job has its own worker thread, so leases, training and practice run
        side by side up to the ledger's concurrency limits (one job at a time in seconds mode: the ledger refuses
        a second one)."""
        for name in [name for name, thread in self.threads.items() if not thread.is_alive()]:
            del self.threads[name]
        for job in self.ledger.jobs():
            name = job['name']
            if name in self.threads:
                continue
            if job['status'] in ('running', 'uncertain'):
                # No thread owns it (a lost host, or a provider fault). Reattach only to the recorded provider
                # worker; a tick never queues a replacement allocation or clears a hold whose end is unknown.
                # Ask again at most every RECOVERY_RETRY_SECONDS.
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
            # The provider terminated a worker that never got a GPU and settled it uncharged; nothing is held.
            write_json(self.root / 'jobs' / name / 'host-error.json',
                       dict(type='NeverStarted', message=str(exc), provider=exc.receipt, charged_usd=0))
            current = next(j for j in ledger.jobs() if j['name'] == name)
            if current['status'] in ('queued', 'uncertain'):
                ledger.abandon_unstarted(name, exc.receipt, str(exc), stopped=name in self.stops)
        except BaseException as exc:
            current = next(j for j in ledger.jobs() if j['name'] == name)
            message = type(exc).__name__ + ': ' + str(exc)
            write_json(self.root / 'jobs' / name / 'host-error.json', dict(type=type(exc).__name__, message=str(exc)))
            if any(word in str(exc).lower() for word in BUDGET_STOPS):
                write_json(self.root / 'budget-stop.json', dict(job=name, message=message, paid_retry_allowed=False))
            if current['status'] == 'queued':
                ledger.cancel_queued(name)
            else:
                ledger.uncertain(name, message)

    def local_poll(self, transport):
        transport = Path(transport)
        requests = transport / 'requests'
        requests.mkdir(parents=True, exist_ok=True)
        for p in sorted(requests.glob('*.json')):
            if p.stat().st_size > 2 * 1024 * 1024:
                result = {'error': 'Request exceeds cap'}
            else:
                try:
                    result = self.handle(json.loads(p.read_text()))
                except Exception as exc:
                    result = {'error': type(exc).__name__ + ': ' + str(exc)}
            write_json(transport / 'replies' / p.name, result)
            p.unlink()
        self.tick()
