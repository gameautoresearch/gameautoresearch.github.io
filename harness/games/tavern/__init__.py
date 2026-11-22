"""Tavern: four-player AutoTavern lobbies (small form), mixed-lobby evaluator. Controller and provider live beside this file.

Both research budget modes (harness/budget.py). `seconds` (the default): one job at a time, training runs charged
their training-worker seconds. `usd`: up to `budget.PARALLEL` jobs at once (at most two training runs, one GPU
each), every training-worker second charged at `budget.rate(TRAIN_SHAPE)`, and each queued or running run holding
its whole grant until it ends. Public practice runs on a separate CPU-only worker outside the allocation in both
modes; in usd mode it takes one of the job slots.

Backend options: `session_config` (overrides for the session settings, e.g. {"record_actions": true}),
`require_practice` (a run must have a complete public practice before it can be submitted; off by
default because the recipe variant submits its fresh final training directly) and `_provider` (tests).
"""
import json
import shutil
import time

from harness import budget as budgets
from harness.backend import ACTIVE, Backend as Base

# The mixed-v2 contract's workers: training on one L40S with 4 CPU cores and 16 GiB, practice and grading on the
# same CPU and RAM without a GPU.
TRAIN_SHAPE = dict(gpu=1, cpu_cores=4, memory_mib=16384)
PRACTICE_SHAPE = dict(gpu=0, cpu_cores=4, memory_mib=16384)
RESEARCH_SECONDS, SMOKE_RESEARCH_SECONDS = 7200, 600

SPEC = dict(
    game='tavern',
    # How `harness retrain` sets a replicate's training seed: the run seed, the config's `seed` field, or the
    # backend's own fresh host seeds for a final.
    retrain_seed='config',
    # What `ar` offers in this game: `ar run --seed`, `ar eval --seconds`, and the box's CPU torch version.
    run_seed=False, eval_seconds=False, box_torch='2.15.0',
    # The graded policy's size limit (parameters and buffers), the same in every game.
    parameter_cap=500_000,
    venv='.venv-tavern-ar',
    constructor='simulators.tavern.autoresearch.rules_enabled.release:prepare',
    constructor_options={
        'registry': 'directory of a compatible frozen mixed-v2 reference registry; default: the pool named by '
                    'evaluation_v2/selected_registry-four-small-v1.json (private local evidence, not in the repository)'},
    contract='simulators/tavern/autoresearch/evaluation_v2/benchmark.json',
    mode='four-player AutoTavern 37.0.2 implemented-content lobbies (tavern tiers 1-4, three minion types); '
         'scripted training opponents, mixed-lobby evaluator',
    hardware='one L40S GPU with 4 CPU cores and 16 GiB per training run; practice games on a separate CPU-only '
             'worker with 4 CPU cores and 16 GiB, not charged to the research allocation (the mixed-v2 contract)',
    # Tavern is seconds-native: a session that names no mode gets seconds mode.
    budget_mode='seconds',
    research_seconds=RESEARCH_SECONDS,
    smoke_research_seconds=SMOKE_RESEARCH_SECONDS,
    # Dollar mode's defaults: the same training-worker time at the frozen list price, and the dollar contracts'
    # wall clocks.
    research_usd=round(RESEARCH_SECONDS * budgets.rate(TRAIN_SHAPE), 2),
    smoke_research_usd=round(SMOKE_RESEARCH_SECONDS * budgets.rate(TRAIN_SHAPE), 2),
    wall_seconds=18000,
    smoke_wall_seconds=4800,
    final_train_seconds=3600,
    smoke_final_train_seconds=240,
)
# The repository's GPU with the mixed-v2 contract's CPU and RAM.
HARDWARE = dict(gpu='L40S', cpu_cores=4, memory_gib=16)
LIMITS = dict(parameter_cap=SPEC['parameter_cap'], weights_byte_cap=32000000, inference_deadline_seconds=15)
# controller status -> harness.backend.STATUSES
STATUS = dict(queued='queued', reserved='running', running='running', completed='complete', failed='failed',
              stopped='stopped', cancelled='stopped')
PRACTICE = 'practice-'
RUN_SHAPE = 'ar run (L40S, 4 CPU, 16 GiB)'
SUMMARY = ('complete_valid_schedule', 'raw_score', 'mean_placement', 'top_half_rate', 'observed_cases', 'expected_cases',
           'failures', 'confidence_interval_95')


def tail_text(text, tail):
    return '\n'.join(text.splitlines()[-tail:])


class Backend(Base):
    game = 'tavern'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workspace = self.root / 'workspace'  # native trainer tree: starter/ and recipes/
        self.controller = None
        self._app = None

    def open(self):
        from harness.policy import ENVIRONMENT
        from simulators.tavern.autoresearch.contract import write
        from .controller import Controller
        if not self.resume:
            if self.budget_mode == budgets.USD:
                budget = dict(budget_usd=float(self.budget_usd), usd_per_second=budgets.rate(TRAIN_SHAPE))
            else:
                budget = dict(gpu_limit_seconds=float(self.budget_seconds or SPEC['smoke_research_seconds' if self.smoke
                                                                                  else 'research_seconds']))
            settings = dict(budget, budget_mode=self.budget_mode, budget_limits=self.limits,
                            mode='smoke' if self.smoke else 'research', record_actions=False, **HARDWARE)
            settings.update(self.options.get('session_config', {}))
            self.root.mkdir(parents=True)
            for folder in ('starter', 'recipes'):
                shutil.copytree(self.release / 'public' / folder, self.workspace / folder)
            write(self.root / 'session.json', settings)
        provider = self.options.get('_provider')  # tests stand in for Modal here
        try:
            if provider is None:
                import modal
                from .provider import ModalProvider
                app = modal.App('game-autoresearch-tavern')
                self._app = app.run(environment_name=ENVIRONMENT)
                self._app.__enter__()
                provider = ModalProvider(self.release, app)
                with (self.root / 'apps.jsonl').open('a') as log:
                    log.write(json.dumps(dict(app_id=app.app_id, environment=ENVIRONMENT, opened=time.time())) + '\n')
            self.controller = Controller(self.release, self.root, provider)
            # A reopened session keeps the mode it was created in: an explicit amount of the other mode is refused.
            if self.resume and (self.budget_usd is not None or self.budget_seconds is not None) \
                    and self.controller.mode != self.budget_mode:
                raise ValueError(f'This session was created in {self.controller.mode} mode; it cannot be reopened '
                                 f'in {self.budget_mode} mode')
            self.budget_mode, self.limits = self.controller.mode, dict(self.controller.limits)
        except BaseException:
            self._leave()
            raise

    def _leave(self):
        if self._app is not None:
            context, self._app = self._app, None
            context.__exit__(None, None, None)

    def settle(self):
        controller = self.controller
        if controller is None:
            return
        for kind, name in list(controller.snapshot()['queue']):
            self._stop(kind, name)
        for kind, name in controller.running():  # one job in seconds mode; every running one in usd mode
            self._stop(kind, name)
        deadline = time.time() + (120 if self.smoke else 600)
        controller.join(max(1.0, deadline - time.time()))
        # A stopped worker settles from its provider receipt. Ask again for any end time still outstanding;
        # a reservation with no recorded worker cannot be asked about and keeps its hold, as the ledger requires.
        # Nothing will use a worker left over from an interrupted host, so it is stopped too.
        with controller.lock:
            controller.stops.update(('train', name) for name in controller.unresolved())
        while not controller.busy() and time.time() < deadline:
            known = {name for name in controller.unresolved() if (self.root / 'runs' / name / 'sandbox.json').exists()}
            if not known:
                break
            controller.recover()
            if known & set(controller.unresolved()):
                time.sleep(2)

    def _stop(self, kind, name):
        try:
            self.controller.stop(kind, name, now=True)  # the session is closing: nothing unsubmitted is kept
        except ValueError:
            pass  # already finished

    def close(self):
        try:
            self.settle()
        finally:
            self._leave()

    def budget(self):
        from .controller import MIN_TRAIN, OVERHEAD, SMOKE_TRAIN
        if self.controller.parallel:
            return self._budget_usd()
        ledger = self.controller.budget()
        runs = self.controller.snapshot()['runs'].values()
        queued = sum(run['seconds'] + OVERHEAD for run in runs if run['status'] == 'queued')
        held = ledger['unresolved_seconds'] + queued
        used = ledger['allocated_seconds'] + held
        cap = ledger['cap_seconds']
        return dict(budget_mode=budgets.SECONDS, budget_seconds=cap, used_seconds=used, remaining_seconds=max(0.0, cap - used),
                    charged_seconds=ledger['allocated_seconds'], reserved_seconds=held,
                    run_overhead_seconds=OVERHEAD, min_run_seconds=MIN_TRAIN + OVERHEAD,
                    max_run_seconds=SMOKE_TRAIN + OVERHEAD if self.smoke else cap, concurrency=1,
                    practice='48 public lobbies on a separate CPU-only worker; not charged to this allocation',
                    closed=ledger['closed'], **LIMITS)

    def _budget_usd(self):
        """Dollar mode: every queued, reserved or running run holds its whole grant, an unknown end keeps its hold."""
        from .controller import MIN_TRAIN, OVERHEAD, SMOKE_TRAIN
        controller = self.controller
        ledger = controller.budget()
        rate, cap, limits = ledger['usd_per_second'], ledger['cap_usd'], controller.limits
        held = ledger['unresolved_seconds'] + controller.queued_seconds()
        used = ledger['charged_usd'] + held * rate
        remaining = max(0.0, cap - used)
        return dict(budget_mode=budgets.USD, budget_usd=cap, used_usd=round(used, 6), remaining_usd=round(remaining, 6),
                    allocatable_usd=round(remaining, 6), charged_usd=round(ledger['charged_usd'], 6),
                    reserved_usd=round(held * rate, 6), charged_seconds=ledger['allocated_seconds'],
                    reserved_seconds=held, usd_per_second=rate, usd_per_hour={RUN_SHAPE: round(3600 * rate, 4)},
                    affordable_seconds={RUN_SHAPE: int(remaining / rate + 1e-6)},
                    run_overhead_seconds=OVERHEAD, min_run_seconds=MIN_TRAIN + OVERHEAD,
                    max_run_seconds=SMOKE_TRAIN + OVERHEAD if self.smoke else int(controller.cap + 1e-6),
                    concurrency=limits['max_concurrent_jobs'], concurrent_gpus=limits['max_concurrent_gpus'],
                    concurrent_cpu_cores=limits['max_concurrent_cpu_cores'],
                    charging='budget, used and remaining are US dollars. A training run is charged its worker '
                             'seconds x $%.7f (one L40S, 4 CPU cores, 16 GiB at the frozen list price); `ar run '
                             '--seconds` still grants worker seconds, and a queued or running run holds its whole '
                             'grant (seconds + %d s overhead) until it ends.' % (rate, OVERHEAD),
                    practice='48 public lobbies on a separate CPU-only worker; not charged to this allocation, but '
                             'it takes one of the job slots while it runs',
                    closed=ledger['closed'], **LIMITS)

    def status(self):
        state = self.controller.snapshot()
        ledger = {row['name']: row for row in self.controller.budget()['allocations']}
        unresolved = set(self.controller.unresolved())
        jobs = []
        for name, run in state['runs'].items():
            row = ledger.get(name, {})
            job = dict(name=name, kind='train', status=STATUS[run['status']], train_seconds=run['seconds'],
                       reserved_seconds=row.get('reserved'), charged_seconds=row.get('actual'),
                       trained=run['status'] == 'completed', submitted=(self.root / 'submissions' / name).exists())
            if name in unresolved:  # the worker's end time is unknown: its reservation is still held
                job['status'] = 'uncertain'
            if run.get('never_started'):  # created but never given a GPU: terminated and charged 0 seconds
                job['never_started'] = True
            for flag in ('stopped_early', 'cut_short'):  # complete, on the network the ended trainer last exported
                if run.get(flag):
                    job[flag] = True
            if run.get('error'):
                job['error'] = run['error'][-600:]
            if run.get('summary'):
                job['summary'] = {key: run['summary'].get(key) for key in ('updates', 'samples', 'finite_losses',
                                                                           'changed_weights', 'accounted_seconds')}
            if run.get('practice'):
                job['practice'] = {key: run['practice'].get(key) for key in SUMMARY}
            jobs.append(job)
        for name, practice in state['practice'].items():
            job = dict(name=PRACTICE + name, kind='practice', status=STATUS[practice['status']], run=name,
                       attempt=practice.get('attempt', 1))
            if practice.get('error'):
                job['error'] = practice['error'][-600:]
            jobs.append(job)
        return dict(jobs=jobs, active=[job['name'] for job in jobs if job['status'] in ACTIVE],
                    selected_submission=state['submission'], lifecycle='finalized' if state['finalized'] else 'open')

    def log(self, name, tail=80):
        state = self.controller.snapshot()
        run_name = name[len(PRACTICE):] if name.startswith(PRACTICE) else name
        run = state['runs'].get(run_name)
        if run is None or (name != run_name and run_name not in state['practice']):
            raise ValueError('Unknown run')
        result = dict(name=name)
        if name != run_name:
            practice = state['practice'][run_name]
            result.update(status=STATUS[practice['status']], run=run_name)
            if practice.get('error'):  # the job's own failure; a top-level `error` is a failed command
                result['job_error'] = practice['error'][-6000:]
            if practice['status'] == 'completed':
                result['practice'] = run.get('practice')
            return result
        folder = self.root / 'runs' / name
        result.update(status=STATUS[run['status']], train_seconds=run['seconds'])
        if name in self.controller.unresolved():
            result['status'] = 'uncertain'
        if run.get('never_started'):
            result.update(never_started=True, charged_seconds=0)
        for flag in ('stopped_early', 'cut_short'):
            if run.get(flag):
                result[flag] = True
        if run.get('error'):  # the job's own failure; a top-level `error` is a failed command
            result['job_error'] = run['error'][-6000:]
        if (folder / 'train.log').is_file():
            result['stdout'] = tail_text((folder / 'train.log').read_text(), tail)
        if run.get('summary'):
            result['summary'] = run['summary']
        if (folder / 'artifact/metrics.json').is_file():
            result['progress'] = [self._progress(row) for row in
                                  json.loads((folder / 'artifact/metrics.json').read_text())[-tail:]]
        if run.get('practice'):
            result['practice'] = run['practice']
        return result

    @staticmethod
    def _progress(row):
        """One line per update: the learner's own statistics and how the training lobbies ended."""
        if not isinstance(row, dict):
            return row
        compact = {key: value for key, value in row.items() if key not in ('episodes', 'counts')}
        episodes = [e for e in row.get('episodes') or [] if isinstance(e, dict)]
        placements = [e['placement'] for e in episodes if e.get('valid') and isinstance(e.get('placement'), int)]
        compact.update(episodes=len(episodes), natural=len(placements),
                       mean_placement=round(sum(placements) / len(placements), 3) if placements else None)
        counts = row.get('counts')
        if isinstance(counts, dict):
            compact['faults'] = {key: value for key, value in counts.items()
                                 if key.startswith('fault_') or key in ('invalid_lobbies', 'rejected')}
        return compact

    def _recipe(self):
        from simulators.tavern.autoresearch.contract import validate
        native = self.workspace / ('recipes/smoke.json' if self.smoke else 'recipes/starter.json')
        from .controller import require_gpu_device
        try:
            recipe = json.loads(native.read_text())
            validate(recipe)
            int(recipe['seed']), int(recipe['updates']), float(recipe['learning_rate'])
        except (KeyError, TypeError) as exc:
            raise ValueError('Config is missing a required field or has the wrong type: ' + str(exc)) from None
        require_gpu_device(recipe)  # every training run gets a GPU worker
        return recipe

    def check(self, config):
        chosen = self.sync(config)
        recipe = self._recipe()
        return dict(checked=True, config=chosen, seed=recipe['seed'], network=recipe['network'], **LIMITS,
                    note='Rules, bounds and config fields were checked on the host. The parameter cap and the '
                         'training code itself are only exercised on a worker.')

    def run(self, name, seconds, config, seed=1):
        from .controller import MIN_TRAIN, OVERHEAD, SMOKE_TRAIN
        if seed != 1:
            raise ValueError('Tavern takes the training seed from the "seed" field of the config; edit it there '
                             'instead of passing --seed')
        train = float(seconds) - OVERHEAD
        if train < MIN_TRAIN:
            raise ValueError(f'--seconds must be at least {MIN_TRAIN + OVERHEAD}: {OVERHEAD} seconds of every run '
                             'are reserved for worker setup, export and teardown')
        if self.smoke and train > SMOKE_TRAIN:
            raise ValueError(f'A smoke run is at most {SMOKE_TRAIN + OVERHEAD} seconds')
        budget = self.budget()
        if (seconds * budget['usd_per_second'] > budget['remaining_usd'] + 1e-9 if self.controller.parallel
                else seconds > budget['remaining_seconds']):
            raise ValueError('Allocation exceeds the remaining session budget')
        chosen = self.sync(config)
        recipe = self._recipe()
        reply = self.controller.admit(name, train)
        return dict(queued=name, reserved_seconds=reply['reserved_seconds'], train_seconds=reply['train_seconds'],
                    config=chosen, seed=recipe['seed'])

    def stop(self, name):
        if name.startswith(PRACTICE):
            return self.controller.stop('practice', name[len(PRACTICE):])
        return self.controller.stop('train', name)

    def evaluate(self, name, seconds=None):
        # The public practice schedule and its worker lifetime are fixed by the evaluator; `seconds` is not used.
        return dict(self.controller.practice(name), lobbies=48, charged=False)

    def submit(self, name):
        """The named run becomes the selected submission; an earlier immutable submission is reselected as is."""
        if (self.root / 'submissions' / name).exists():
            return dict(self.controller.select(name), submission=name)
        return self.controller.submit(name, require_practice=bool(self.options.get('require_practice', False)))

    def finalize(self):
        state = self.controller.finalize()['state']
        return dict(lifecycle='finalized', final_submission=state['final_submission'])

    def tick(self):
        self.controller.tick()

    def grade(self):
        return self.controller.grade()
