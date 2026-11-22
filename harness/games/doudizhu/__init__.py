"""Dou Dizhu: fixed-role, no-bidding three-player game. Controller and provider live beside this file."""
import json

from harness import budget as budgets, workspace
from harness.backend import ACTIVE, Backend as Base

# Every Dou Dizhu worker, training and practice alike, has this shape (simulators/doudizhu/autoresearch/contract.py).
TRAIN_SHAPE = dict(gpu=1, cpu_cores=4, memory_mib=16384)

SPEC = dict(
    game='doudizhu',
    # How `harness retrain` sets a replicate's training seed: the run seed, the config's `seed` field, or the
    # backend's own fresh host seeds for a final.
    retrain_seed='run',
    # What `ar` offers in this game: `ar run --seed`, `ar eval --seconds`, and the box's CPU torch version.
    run_seed=True, eval_seconds=True, box_torch='2.10.0',
    # The graded policy's size limit (parameters and buffers), the same in every game.
    parameter_cap=500_000,
    venv='.venv-doudizhu-ar',
    constructor='simulators.doudizhu.autoresearch.prepare:prepare',
    constructor_options={
        'include_existing': 'true (default) converts the three historical development policies under '
                            'evidence/doudizhu-implementation into practice/development panel bots; this needs '
                            'torch, so construct with .venv-doudizhu/bin/python. false builds the release with '
                            'only the random and heuristic panel bots and runs under either interpreter.'},
    backend_options={
        'development': 'grade: true forces a development grade (raw score, no reference normalization). '
                       'Sessions not at the 7200-second contract in seconds mode are always development.',
        'catalog': 'grade: path to the calibrated-panel.json written by manage.calibrate for this release; '
                   'required for an official reference-normalized grade.',
        'private_deals': 'grade: development-only prefix of the private schedule (smoke default 16).',
        'session_config': 'open: explicit overrides of the session contract (host use only).'},
    contract='simulators/doudizhu/autoresearch/benchmark.json',
    mode='fixed-role no-bidding three-player Dou Dizhu, full 54-card deck',
    hardware='one L40S GPU with 4 CPU cores and 16 GiB per worker, for training runs and practice alike '
             '(the Dou Dizhu contract)',
    # Seconds mode (the default): one job at a time within this many seconds of the worker.
    budget_mode='seconds',
    research_seconds=7200,
    smoke_research_seconds=600,
    # Dollar mode: the same worker time at the frozen list price (budget.rate, about $2.90 an hour), rounded to
    # cents, with jobs side by side; its wall clock is the dollar games' (5 h, smoke 80 min).
    research_usd=round(7200 * budgets.rate(TRAIN_SHAPE), 2),
    smoke_research_usd=round(600 * budgets.rate(TRAIN_SHAPE), 2),
    wall_seconds=18000,
    smoke_wall_seconds=4800,
    final_train_seconds=3600,
    smoke_final_train_seconds=180,
)
# The contract's bounded end-to-end smoke (benchmark.json: 600 s, 8 practice deals, 20 s teardown reserve).
SMOKE = dict(min_run_seconds=30, practice_deals=8, teardown_reserve_seconds=20, max_run_seconds=360)
SMOKE_PRIVATE_DEALS = 16
PRACTICE_SECONDS = 60
STATUS = {'cancelled': 'stopped'}
LOG_CHARS = 20000


def _tail(path, lines):
    text = '\n'.join(path.read_text(errors='replace').splitlines()[-lines:])
    return text[-LOG_CHARS:]


class Backend(Base):
    game = 'doudizhu'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workspace = self.root / 'workspace'  # anchors the host workspace copy; source travels in the request
        self.controller = None
        self._app = None

    def open(self):
        from harness.policy import ENVIRONMENT
        from simulators.doudizhu.autoresearch.contract import DEFAULT
        from .controller import Controller
        config = None
        if not self.resume:
            config = dict(DEFAULT)
            if self.smoke:
                config.update(SMOKE)
            if self.budget_mode == budgets.USD:
                # Dollar mode: the ledger keeps worker seconds (every worker has the training shape) and holds
                # each queued or running job's whole grant; `gpu_minutes` is the worker time the dollars buy.
                rate = budgets.rate(TRAIN_SHAPE)
                config.update(budget_usd=float(self.budget_usd), usd_per_second=rate,
                              gpu_minutes=float(self.budget_usd) / rate / 60, worker_shape=dict(TRAIN_SHAPE))
            elif self.budget_seconds:
                config['gpu_minutes'] = self.budget_seconds / 60
            config.update(budget_mode=self.budget_mode, **self.limits)
            config.update(self.options.get('session_config', {}))
        provider = self.options.get('_provider')  # tests stand in for Modal here
        try:
            if provider is None:
                import modal
                from simulators.doudizhu.autoresearch.transport import install_transport_guard
                from .provider import ModalProvider
                install_transport_guard()
                app = modal.App('game-autoresearch-doudizhu')
                self._app = app.run(environment_name=ENVIRONMENT)
                self._app.__enter__()
                provider = ModalProvider(self.release, app)
            self.controller = Controller(self.root, self.release, provider, config)
            if self.resume:
                # The ledger's mode is the session's; a resume never changes it.
                ledger = self.controller.ledger
                given = self.budget_usd is not None or bool(self.budget_seconds)
                if given and ledger.mode() != self.budget_mode:
                    raise ValueError(f'This session is in {ledger.mode()} mode; it cannot be resumed in '
                                     f'{self.budget_mode} mode')
                self.budget_mode = ledger.mode()
                self.limits = {key: ledger.limit(key) for key in budgets.LINEAR}
            if self.resume and hasattr(provider, 'recover_jobs'):
                # A lease left unresolved by an interrupted host is settled only from provider receipts.
                provider.recover_jobs(self.controller.ledger)
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
        for job in controller.ledger.jobs():
            if job['status'] in ('queued', 'running'):
                controller.release_job(job)
        controller.join(timeout=150 if self.smoke else 7350)
        for job in controller.ledger.jobs():
            if job['status'] == 'uncertain':
                try:  # terminate the recorded worker and settle it from the provider's record
                    controller.release_job(job)
                except Exception:
                    pass  # stays uncertain (and visible) when the provider has no usable record

    def close(self):
        try:
            self.settle()
        finally:
            self._leave()

    def _handle(self, action, **arguments):
        return self.controller.handle(dict(arguments, action=action))

    def budget(self):
        from .controller import PRACTICE_MIN_SECONDS
        ledger = self.controller.ledger
        state = ledger.public()
        config = state['config']
        if ledger.mode() == budgets.USD:
            rate = ledger.rate()
            amounts = dict(budget_usd=config['budget_usd'], used_usd=state['charged_or_reserved_usd'],
                           remaining_usd=state['remaining_usd'], allocatable_usd=state['allocatable_usd'],
                           usd_per_second=rate, usd_per_hour=round(3600 * rate, 4),
                           affordable_seconds={'run or eval (L40S, 4 CPU, 16 GiB)':
                                               int(state['allocatable_usd'] / rate)},
                           concurrent_gpus=ledger.limit('max_concurrent_gpus'),
                           concurrent_cpu_cores=ledger.limit('max_concurrent_cpu_cores'),
                           charging='budget, used and remaining are US dollars: a worker is charged its seconds x '
                                    '$%.4f/h (every worker, practice included, has one L40S, 4 CPU cores and '
                                    '16 GiB, so at most %d run at once). Queued and running work is reserved in '
                                    'full until it ends.' % (3600 * rate, min(ledger.limit('max_concurrent_jobs'),
                                                                              ledger.limit('max_concurrent_gpus'))))
        else:
            amounts = dict(budget_seconds=config['gpu_minutes'] * 60, used_seconds=state['charged_or_reserved_seconds'],
                           remaining_seconds=state['remaining_seconds'],
                           allocatable_seconds=state['allocatable_seconds'])
        return dict(amounts, budget_mode=ledger.mode(), concurrency=ledger.limit('max_concurrent_jobs'),
                    teardown_reserve_seconds=config.get('teardown_reserve_seconds', 0),
                    min_run_seconds=config['min_run_seconds'], max_run_seconds=config.get('max_run_seconds'),
                    min_practice_seconds=PRACTICE_MIN_SECONDS, default_practice_seconds=PRACTICE_SECONDS,
                    practice_deals=config['practice_deals'],
                    policy_tensor_cap=config['policy_tensor_cap'], source_bytes_cap=config['source_bytes_cap'],
                    artifact_bytes_cap=config['artifact_bytes_cap'],
                    inference_batch=config['inference_batch'],
                    inference_timeout_seconds=config['inference_timeout_seconds'])

    def _jobs(self):
        """Ledger rows with the harness's kind, normalized status and the run a practice job evaluated."""
        ledger = self.controller.ledger.jobs()
        bundles = {row['artifact']: row['name'] for row in ledger if row['artifact']}
        rows = []
        for row in ledger:
            payload = json.loads(row['payload'])
            job = {key: row[key] for key in ('name', 'grant', 'started', 'ended', 'charged', 'artifact_sha', 'error')}
            if self.controller.ledger.mode() == budgets.USD:
                job['charged_usd'] = row['charged'] * self.controller.ledger.rate()
            job['status'] = STATUS.get(row['status'], row['status'])
            job['kind'] = 'practice' if payload.get('kind') == 'practice' else 'train'
            if job['kind'] == 'train' and job['status'] == 'stopped' and row['artifact_sha']:
                # A run the researcher stopped still left a validated bundle (the ledger lets it be practised and
                # submitted): complete, flagged like every other game's stopped run.
                job.update(status='complete', stopped_early=True)
            if job['kind'] == 'practice':
                job['evaluates'] = bundles.get(payload.get('bundle'))
            rows.append(job)
        return rows

    def status(self):
        jobs = self._jobs()
        for job in jobs:
            folder = self.root / 'jobs' / job['name']
            if (folder / 'progress.jsonl').is_file():
                for line in reversed((folder / 'progress.jsonl').read_text(errors='replace').splitlines()):
                    try:
                        job['progress'] = json.loads(line)
                        break
                    except ValueError:
                        continue
            if (folder / 'practice.json').is_file():
                job['practice'] = json.loads((folder / 'practice.json').read_text())
        return dict(jobs=jobs, active=[job['name'] for job in jobs if job['status'] in ACTIVE])

    def log(self, name, tail=80):
        jobs = self._jobs()
        job = next((job for job in jobs if job['name'] == name), None)
        if job is None:
            raise ValueError('Unknown run')
        folder = self.root / 'jobs' / name
        # A reply with a top-level `error` is a failed command; the job's own failure is reported as job_error.
        result = {key: value for key, value in job.items() if key != 'error'}
        if job['error']:
            result['job_error'] = job['error']
        for label, filename in (('trainer_log', 'trainer.log'), ('progress', 'progress.jsonl'),
                                ('export_error', 'export-error.log'), ('practice_error', 'practice-error.log')):
            if (folder / filename).is_file():
                result[label] = _tail(folder / filename, tail)
        for label, filename in (('supervisor', 'supervisor.json'), ('practice', 'practice.json')):
            if (folder / filename).is_file():
                result[label] = json.loads((folder / filename).read_text())
        if job['kind'] == 'train':
            evaluations = {}
            for other in jobs:
                path = self.root / 'jobs' / other['name'] / 'practice.json'
                if other.get('evaluates') == name:
                    evaluations[other['name']] = json.loads(path.read_text()) if path.is_file() else other['status']
            if evaluations:
                result['practice_runs'] = evaluations
        return result

    def check(self, config):
        from simulators.doudizhu.autoresearch.contract import DEFAULT, EDITABLE
        files = workspace.editable(self.release, self.workspace)
        source_bytes = sum(len(files['training/' + name].encode()) for name in EDITABLE)
        if source_bytes > DEFAULT['source_bytes_cap']:
            raise ValueError(f'training/ source is {source_bytes} bytes; the cap is {DEFAULT["source_bytes_cap"]}')
        recipe = json.loads(files[config])
        if not isinstance(recipe, dict):
            raise ValueError(config + ' must be a JSON object')
        if len(json.dumps(recipe).encode()) > 65536:
            raise ValueError(config + ' exceeds the 65536-byte configuration cap')
        return dict(checked=True, source_bytes=source_bytes, source_bytes_cap=DEFAULT['source_bytes_cap'],
                    policy_tensor_cap=DEFAULT['policy_tensor_cap'], artifact_bytes_cap=DEFAULT['artifact_bytes_cap'],
                    note='Syntax and transport caps only. The tensor and weight-file caps are enforced on the '
                         'worker when a run is exported; nothing is imported or executed here.')

    def run(self, name, seconds, config, seed=1):
        from simulators.doudizhu.autoresearch.contract import EDITABLE
        files = workspace.editable(self.release, self.workspace)
        recipe = json.loads(files[config])
        if not isinstance(recipe, dict):
            raise ValueError(config + ' must be a JSON object')
        state = self.controller.ledger.public()
        reserve = state['config'].get('teardown_reserve_seconds', 0)
        granted = float(seconds)
        if reserve and state['allocatable_seconds'] < granted <= state['remaining_seconds'] + 1e-9:
            # The request fits the allowance but not the teardown reserve the ledger holds back: grant what is left.
            granted = state['allocatable_seconds']
            if granted < state['config']['min_run_seconds']:
                raise ValueError(f'Only {granted:.0f} seconds remain after the {reserve:g}-second teardown reserve; '
                                 'that is below the minimum run')
        reply = self._handle('run', name=name, seconds=granted, seed=seed, config=recipe,
                             code={source: files['training/' + source] for source in EDITABLE})
        if granted != seconds:
            reply['note'] = (f'Granted {granted:g} of the requested {seconds:g} seconds: '
                             f'{reserve:g} seconds of the allowance are reserved for worker teardown.')
        return reply

    def stop(self, name):
        return self._handle('stop', name=name)

    def evaluate(self, name, seconds=None):
        return self._handle('eval', name=name, seconds=seconds or PRACTICE_SECONDS)

    def submit(self, name):
        return self._handle('submit', name=name)

    def finalize(self):
        state = self._handle('finalize')
        if self.controller.ledger.mode() == budgets.USD:
            return dict(submissions=state['submissions'], points=state['points'],
                        used_usd=state['charged_or_reserved_usd'])
        return dict(submissions=state['submissions'], points=state['points'],
                    used_seconds=state['charged_or_reserved_seconds'])

    def tick(self):
        self.controller.tick()

    def grade(self):
        from simulators.doudizhu.autoresearch.manage import grade_session
        config = self.controller.ledger.config
        # Official only when the host says this session ran at the declared budgets
        # (a network session's research allowance, or a recipe's fresh final training).
        development = bool(self.options.get('development')) or not self.official
        deals = self.options.get('private_deals', SMOKE_PRIVATE_DEALS if self.smoke else None) if development else None
        points = grade_session(self.release, self.root, self.controller.provider, development,
                               self.options.get('catalog'), deals)
        final = next((points[key] for key in points if float(key) == 1), None)
        return dict(development=development, private_deals=deals or config['grade_deals'], final=final, points=points)
