"""Host side of a research session: one request handler for every game and variant.

The broker is the only thing that reads the researcher's requests. It validates
the editable files, applies the task variant's submission rules and forwards
the rest to the game's backend. All state lives in the session directory.
"""
import json
import math
from pathlib import Path
import re
import time

from harness import workspace
from harness.backend import ACTIVE

VARIANTS = ('recipe', 'network')
# A new run name is plain so every game's workers accept it. Jobs a backend derives
# from it (practice, retries) add hyphenated prefixes or suffixes and can be referenced.
NEW_NAME = re.compile(r'[a-z0-9]{1,40}')
JOB_NAME = re.compile(r'[a-z0-9][a-z0-9-]{0,79}')
# A full session (opt-in) keeps the researcher working: `ar finalize` is accepted only once
# little allocation or little wall-clock time remains.
FULL_SESSION = dict(allocation_fraction=0.10, final_seconds=1200.0)
GRADED = {'recipe': 'The latest submitted recipe is retrained from scratch by the host and that network is graded.',
          'network': 'The latest submitted network from your own runs is graded as is.'}
# With `use_budget`, the researcher may finalize only once less than this share of the allocation is usable ...
FINALIZE_FLOOR = 0.10
# ... or the wall-clock limit is this close.
WALL_MARGIN_SECONDS = 600.0
# Per-run setup time a backend reports in `budget()`, under the name its own contract uses.
OVERHEAD_KEYS = ('run_reserve_seconds', 'run_overhead_seconds', 'setup_and_publication_seconds_per_run')
EARLY = ('stopped_early', 'cut_short')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')
    temporary.replace(path)


def positive(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(what + ' must be a positive number')
    return float(value)


class Broker:
    def __init__(self, backend, root, variant, *, research_seconds=None, research_usd=None,
                 final_train_seconds=None, deadline=None, use_budget=False, full_session=False):
        """`use_budget`: refuse the researcher's finalize while a useful share of the allocation is unspent.
        `full_session`: the stricter rule of `_finalize_blocker` (little allocation or session time left)."""
        if variant not in VARIANTS:
            raise ValueError('variant must be recipe or network')
        self.backend = backend
        self.root = Path(root)
        self.variant = variant
        self.meta = workspace.manifest(backend.release)
        # With leases the researcher has one way to get compute, the lease; `ar run` is not offered.
        from harness import registry  # registry imports this module
        self.leases = bool(registry.spec(self.meta['game']).get('leases'))
        self.path = self.root / 'session.json'
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state['variant'] != variant or self.state['game'] != self.meta['game']:
                raise ValueError('Session was created for a different game or variant')
        else:
            self.state = dict(schema=1, game=self.meta['game'], variant=variant, finalized=False, closed_by=None,
                              research_seconds=research_seconds, final_train_seconds=final_train_seconds,
                              deadline=deadline, submissions=[], use_budget=bool(use_budget),
                              full_session=bool(full_session))
            if research_usd is not None:  # the research budget is dollars of worker time, not seconds
                self.state['research_usd'] = research_usd
            self.save()

    def save(self):
        write_json(self.path, self.state)

    @property
    def submission(self):
        return self.state['submissions'][-1] if self.state['submissions'] else None

    def handle(self, request):
        """Never raises: the researcher always gets a reply."""
        started = time.time()
        op = request.get('op') if isinstance(request, dict) else None
        try:
            if not isinstance(op, str) or not hasattr(self, 'op_' + op.lstrip('_')):
                raise ValueError('Unknown command')
            reply = getattr(self, 'op_' + op.lstrip('_'))(request)
        except ValueError as exc:
            reply = {'error': str(exc)}
        except Exception as exc:  # backend fault: report, keep serving
            reply = {'error': 'Host fault, not charged to your judgement: ' + type(exc).__name__ + ': ' + str(exc)[:500]}
        record = {k: v for k, v in request.items() if k != 'files'} if isinstance(request, dict) else {'raw': str(request)[:200]}
        with (self.root / 'requests.jsonl').open('a') as log:
            log.write(json.dumps(dict(time=started, request=record, error=reply.get('error'),
                                      seconds=round(time.time() - started, 3)), sort_keys=True) + '\n')
        return reply

    def _open(self):
        if self.state['finalized']:
            raise ValueError('The session is finalized; no further work is accepted')

    def _in_time(self):
        deadline = self.state['deadline']
        if deadline and time.time() > deadline:
            raise ValueError('The wall-clock limit of this session has passed: no new work is accepted. '
                             'Submit your best answer and finalize.')

    def usable_seconds(self):
        """Allocation the researcher could still spend on a run; 0 when what is left is too little or too late.

        Under a dollar budget this is the worker time the unspent dollars still buy (of the cheapest shape)."""
        budget = self.backend.budget()
        need = float(budget.get('min_run_seconds') or 0) + sum(float(budget.get(key) or 0) for key in OVERHEAD_KEYS)
        if 'budget_usd' in budget:
            usd = float(budget.get('allocatable_usd', budget.get('remaining_usd', 0)) or 0)
            if usd < FINALIZE_FLOOR * float(budget['budget_usd']):
                return 0.0
            left = float(max((budget.get('affordable_seconds') or {}).values(), default=0))
            floor = 0.0
        else:
            left = budget.get('allocatable_seconds')
            left = float(budget.get('remaining_seconds') or 0) if left is None else float(left)
            floor = FINALIZE_FLOOR * float(budget.get('budget_seconds') or self.state['research_seconds'] or 0)
        if left < max(need, floor, 1.0):
            return 0.0
        deadline = self.state['deadline']
        if deadline and deadline - time.time() < max(WALL_MARGIN_SECONDS, need):
            return 0.0
        return left

    def resumable(self):
        """True while an unfinished session still has work a researcher could do: the host restarts one then."""
        if self.state['finalized'] or not self.state.get('use_budget'):
            return False
        return bool(self.backend.status().get('active')) or self.usable_seconds() > 0

    def _name(self, request, new=False):
        name = request.get('name')
        if not isinstance(name, str) or not (NEW_NAME if new else JOB_NAME).fullmatch(name):
            raise ValueError('Run names are 1-40 lowercase letters and digits' if new else 'Unknown job name')
        return name

    def _files(self, request):
        """Validate the researcher's editable payload and install it in the host workspace copy."""
        hashes = workspace.apply(self.backend.release, self.backend.workspace, request.get('files'))
        return hashes

    def _config(self, request):
        config = request.get('config', workspace.CONFIGS[0])
        if config not in workspace.CONFIGS:
            raise ValueError('config must be configs/train.json or configs/smoke.json')
        return config

    def op_budget(self, request):
        reply = dict(self.backend.budget())
        reply.update(variant=self.variant, graded=GRADED[self.variant], finalized=self.state['finalized'],
                     submissions=len(self.state['submissions']))
        if self.variant == 'recipe':
            reply['final_train_seconds'] = self.state['final_train_seconds']
        if self.state['deadline']:
            reply['session_seconds_until_deadline'] = max(0.0, round(self.state['deadline'] - time.time(), 1))
        if self.state.get('full_session'):
            reply['finalize_accepted'] = self._finalize_blocker() is None
        return reply

    def _finalize_blocker(self):
        """Why a full session may not be finalized by the researcher yet, or None."""
        if not self.state.get('full_session'):
            return None
        budget = self.backend.budget()
        if 'budget_usd' in budget:
            total = float(budget['budget_usd'])
            remaining = float(budget.get('allocatable_usd', budget.get('remaining_usd', 0)) or 0)
            unspent = f'${remaining:.2f}'
        else:
            total = float(budget.get('budget_seconds') or self.state['research_seconds'] or 0)
            remaining = float(budget.get('allocatable_seconds', budget.get('remaining_seconds', 0)) or 0)
            unspent = f'{remaining:.0f} s'
        left = self.state['deadline'] - time.time() if self.state['deadline'] else 0.0
        if remaining <= FULL_SESSION['allocation_fraction'] * total or left <= FULL_SESSION['final_seconds']:
            return None
        return (f'This is a full session: {unspent} of allocation and {left / 60:.0f} minutes remain. '
                f'Keep researching to improve your agent; your latest submission stays registered. '
                f'`ar finalize` is accepted once at most {FULL_SESSION["allocation_fraction"]:.0%} of the '
                f'allocation or {FULL_SESSION["final_seconds"] / 60:.0f} minutes of session time remain.')

    def op_status(self, request):
        reply = dict(self.backend.status())
        reply.update(editable=self.meta['writable'], finalized=self.state['finalized'], submission=self.submission)
        return reply

    def op_check(self, request):
        hashes = self._files(request)
        config = self._config(request)
        return dict(self.backend.check(config), files=hashes, config=config)

    def op_run(self, request):
        if self.leases:
            raise ValueError('This game has no `ar run`: lease a worker and train on it. '
                             '`python training/launch.py NAME --seconds S` does the plain case.')
        return self.train(request)

    def train(self, request):
        """One fixed training run of the request's files. A game with leases keeps this for the host alone."""
        self._open()
        self._in_time()
        name = self._name(request, new=True)
        seconds = positive(request.get('seconds'), '--seconds')
        seed = request.get('seed', 1)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError('--seed must be an integer')
        config = self._config(request)
        source = self.root / 'sources' / name
        if source.exists():
            raise ValueError('Run name already used; failed and stopped runs keep their names')
        hashes = self._files(request)
        reply = dict(self.backend.run(name, seconds, config, seed))
        for path, text in request['files'].items():
            target = source / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        write_json(source / 'run.json', dict(name=name, seconds=seconds, config=config, seed=seed, files=hashes,
                                             requested_at=time.time()))
        return dict(reply, name=name, config=config, files=hashes)

    def _source(self, name, request, **record):
        """Keep the editable files a job was given, under the job's name."""
        source = self.root / 'sources' / name
        for path, text in request['files'].items():
            target = source / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        write_json(source / 'run.json', dict(record, name=name, requested_at=time.time()))

    def _whole(self, request, key, default):
        value = request.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'--{key} must be a whole number')
        return value

    def op_lease(self, request):
        self._open()
        name = self._name(request, new=True)
        seconds = positive(request.get('seconds'), '--seconds')
        shape = {key: self._whole(request, key, None) for key in ('gpu', 'cpu', 'memory')}
        if (self.root / 'sources' / name).exists():
            raise ValueError('Run name already used; failed and stopped runs keep their names')
        hashes = self._files(request)
        reply = dict(self.backend.lease(name, seconds, **shape))
        self._source(name, request, kind='lease', seconds=seconds, files=hashes, **shape)
        return dict(reply, name=name, files=hashes)

    def op_exec(self, request):
        self._open()
        command = request.get('command')
        if not isinstance(command, str) or not command.strip():
            raise ValueError('Give a command: ar exec NAME -- COMMAND')
        return self.backend.lease_exec(self._name(request), command)

    def op_poll(self, request):
        exec_id = request.get('exec')
        if not isinstance(exec_id, str) or not re.fullmatch(r'[0-9]{4}', exec_id):
            raise ValueError('Unknown command id')
        tail = request.get('tail')
        if tail is not None and (isinstance(tail, bool) or not isinstance(tail, int) or not 1 <= tail <= 20000):
            raise ValueError('--tail must be between 1 and 20000 characters')
        return self.backend.lease_poll(self._name(request), exec_id, tail)

    def op_push(self, request):
        self._open()
        name = self._name(request)
        hashes = self._files(request)
        reply = dict(self.backend.lease_push(name))
        pushes = self.root / 'sources' / name / 'pushes'
        index = len(list(pushes.glob('*'))) + 1 if pushes.is_dir() else 1
        folder = pushes / f'{index:03d}'
        for path, text in request['files'].items():
            target = folder / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        return dict(reply, files=hashes)

    def op_sync(self, request):
        self._open()
        return self.backend.lease_sync(self._name(request), bool(request.get('reload')))

    def op_package(self, request):
        self._open()
        name = self._name(request, new=True)
        lease = request.get('lease')
        if not isinstance(lease, str) or not JOB_NAME.fullmatch(lease):
            raise ValueError('Unknown job name')
        path = request.get('path')
        if not isinstance(path, str) or not 1 <= len(path) <= 1024:
            raise ValueError('--path is the directory on the lease holding the network files (see the game brief)')
        config = self._config(request)
        if (self.root / 'sources' / name).exists():
            raise ValueError('Run name already used; failed and stopped runs keep their names')
        hashes = self._files(request)
        reply = dict(self.backend.package(name, lease, path, config))
        self._source(name, request, kind='package', lease=lease, path=path, config=config, files=hashes)
        return dict(reply, name=name, config=config, files=hashes)

    def op_log(self, request):
        tail = request.get('tail', 80)
        if isinstance(tail, bool) or not isinstance(tail, int) or not 1 <= tail <= 2000:
            raise ValueError('--tail must be between 1 and 2000')
        return self.backend.log(self._name(request), tail)

    def op_stop(self, request):
        return self.backend.stop(self._name(request))

    def op_eval(self, request):
        self._open()
        self._in_time()
        seconds = request.get('seconds')
        return self.backend.evaluate(self._name(request), None if seconds is None else positive(seconds, '--seconds'))

    def _register(self, note, *, files=None, name=None, **extra):
        """Record one submission: a recipe's files, or a completed run's network."""
        index = len(self.state['submissions']) + 1
        if self.variant == 'recipe':
            hashes = workspace.validate(self.meta, files)
            folder = self.root / 'recipes' / f'{index:03d}'
            for path, text in files.items():
                target = folder / 'files' / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
            entry = dict(index=index, kind='recipe', files=hashes, sha256=workspace.digest(hashes),
                         path=str(folder.relative_to(self.root)), note=note, time=time.time(), **extra)
            write_json(folder / 'recipe.json', entry)
        else:
            native = self.backend.submit(name)
            entry = dict(index=index, kind='network', name=name, native=native, note=note, time=time.time(), **extra)
        self.state['submissions'].append(entry)
        self.save()
        return entry

    def op_submit(self, request):
        self._open()
        note = str(request.get('note', ''))[:2000]
        if self.variant == 'recipe':
            entry = self._register(note, files=request.get('files'))
        else:
            entry = self._register(note, name=self._name(request))
        return dict(submitted=entry, final_if_you_finalize_now=True)

    def op_finalize(self, request):
        self._open()
        left = self.usable_seconds() if self.state.get('use_budget') else 0.0
        if left:
            budget = self.backend.budget()
            if 'budget_usd' in budget:
                # Seconds of the cheapest shape (hundreds of thousands of them) read as if the session had days left.
                usd = float(budget.get('allocatable_usd', budget.get('remaining_usd', 0)) or 0)
                unspent = f'${usd:.2f} of your ${float(budget["budget_usd"]):.2f} research allocation is still usable'
            else:
                unspent = f'{left:.0f} seconds of worker time in your research allocation are still usable'
            raise ValueError(
                f'Not yet: {unspent}. Keep experimenting '
                'and submit again whenever a result is better; the latest submission counts. A run made only to '
                'use up allocation gains you nothing: test another idea, or train your submitted recipe again '
                'on a different seed to see how reliably it comes out. `ar finalize` is '
                f'accepted once less than {FINALIZE_FLOOR:.0%} of the allocation (or less than one minimum run) '
                f'is left, or within {WALL_MARGIN_SECONDS / 60:g} minutes of the wall-clock limit.')
        blocker = self._finalize_blocker()
        if blocker:
            raise ValueError(blocker)
        return self.finalize('researcher')

    def op_verify(self, request):
        return dict(finalized=self.state['finalized'], submitted=bool(self.state['submissions']),
                    resume=self.resumable())

    def automatic_answer(self):
        """Host close with nothing submitted: the newest complete run of the graded configuration is the answer.

        A run that reached its own end is preferred to one that was stopped or cut off. Returns the new
        submission, or None when the session has no such run.
        """
        config = workspace.CONFIGS[1] if self.backend.smoke else workspace.CONFIGS[0]
        complete = {job['name']: job for job in self.backend.status().get('jobs', [])
                    if job.get('kind', 'train') in ('train', 'package') and job.get('status') == 'complete'}
        runs = []
        for path in sorted((self.root / 'sources').glob('*/run.json')):
            run = json.loads(path.read_text())
            job = complete.get(run['name'])
            if job is not None and run.get('config') == config:
                runs.append((not any(job.get(flag) for flag in EARLY), run['requested_at'], run))
        if not runs:
            return None
        run = max(runs, key=lambda item: item[:2])[2]
        note = f'Registered by the host at close: nothing was submitted; {run["name"]} is the newest complete {config} run.'
        if self.variant == 'recipe':
            files = {path: (self.root / 'sources' / run['name'] / path).read_text() for path in run['files']}
            return self._register(note, files=files, auto=True, run=run['name'])
        return self._register(note, name=run['name'], auto=True)

    def finalize(self, closed_by):
        """Close the session. The latest explicit submission is the answer; a session the host closes with
        none is answered by its newest complete run of the graded configuration."""
        active = self.backend.status().get('active', [])
        if active:
            if closed_by == 'researcher':
                raise ValueError('Resolve active work before finalizing: ' + ', '.join(active))
            raise RuntimeError('Host close with unresolved work: ' + ', '.join(active))
        if closed_by == 'researcher' and not self.state['submissions']:
            raise ValueError('Nothing submitted yet; ar submit first')
        if closed_by == 'host' and not self.state['submissions']:
            try:
                self.automatic_answer()
            except Exception as exc:  # the close itself must still happen
                self.state['automatic_answer_error'] = type(exc).__name__ + ': ' + str(exc)[:500]
        native = None
        if self.variant == 'network' and self.state['submissions']:
            native = self.backend.finalize()
        self.state.update(finalized=True, closed_by=closed_by, finalized_at=time.time())
        self.save()
        write_json(self.root / 'final.json', dict(variant=self.variant, closed_by=closed_by, submission=self.submission,
                                                  native=native))
        return dict(finalized=True, submission=self.submission)


def active(status):
    return [job['name'] for job in status.get('jobs', []) if job.get('status') in ACTIVE]
