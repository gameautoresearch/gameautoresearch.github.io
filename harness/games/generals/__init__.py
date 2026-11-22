"""Generals: mid-size fog-of-war 1v1 (sides 12-15). Controller and provider live beside this file."""
import json

from harness import budget as budgets, workspace
from harness.backend import ACTIVE, Backend as Base

SPEC = dict(
    game='generals',
    # How `harness retrain` sets a replicate's training seed: the run seed, the config's `seed` field, or the
    # backend's own fresh host seeds for a final.
    retrain_seed='run',
    # What `ar` offers in this game: `ar run --seed`, `ar eval --seconds`.
    run_seed=True, eval_seconds=True,
    # `ar lease`, `exec`, `poll`, `push`, `sync` and `package`: workers of a chosen shape running the researcher's
    # own commands, with the session's scratch Volume.
    leases=True,
    # Generals also runs the installed Claude Code researcher, as Hex does; the model key then enters the box.
    agents=('terminus-2', 'claude-code'),
    # The graded policy's size limit (parameters and buffers), the same in every game.
    # Compute per move limits a submitted policy (simulators/generals/autoresearch/flops.py); parameters only have
    # a storage ceiling. harness/task.py states both.
    parameter_cap=32_000_000,
    move_flops_cap=1_000_000_000,
    venv='.venv-generals-ar',
    constructor='simulators.generals.autoresearch.prepare:prepare',
    constructor_options={
        'opponent_release': 'an earlier sealed Generals release whose verified panel bundles are reused',
        'panel': 'a sealed, rated opponent panel (`simulators.generals.autoresearch.ladder seal`): learned '
                 'opponents with their own network code join random, expansion and hunter, and grades also '
                 'report Elo on the panel ladder; every package must be bound to this release\'s frozen engine',
        'include_existing': 'false builds a scripted-only panel (random, expansion, hunter); the default converts '
                            'historical checkpoints under evidence/ and fails closed if they no longer bind to the '
                            'simulator source',
        'private_maps': 'private grading and calibration maps per schedule, 2..512; fewer than the default 128 '
                        'gives a release that is not ready for official grading',
        'starter': 'ppo (default, the working PPO starter) or empty (boilerplate only: self-play plumbing with no '
                   'learning, an empty learner, a linear placeholder network; its own brief and configs, and the '
                   'project training code, reward helpers and scripted players withheld from the box and from '
                   'unprivileged worker processes), or slowsim (the empty starter on an editable simulator: '
                   'the box\'s simulator/ is a plain-Python implementation of the rules, checked tick for tick '
                   'against the engine, that the researcher may edit and that the boilerplate trains on; the JAX '
                   'engine and wrapper are withheld from the box and from unprivileged worker processes; network '
                   'variant only)'},
    contract='simulators/generals/autoresearch/benchmark.json',
    mode='mid-size fog-of-war 1v1 (map sides 12-15, padded to 16)',
    hardware='workers you lease, of a shape you choose (0 or 1 L40S GPU, up to 32 CPU cores and 64 GiB), with '
             'shared /scratch storage; `training/launch.py` uses one L40S with 4 CPU cores and 16 GiB unless told '
             'otherwise, which is also what a recipe\'s final training gets; public practice on one L40S '
             'with 4 CPU cores and 16 GiB. One research allocation pays for all of it: every worker is charged by '
             'its wall seconds and its shape (`ar budget` shows the unit, what each shape costs and what you have '
             'left)',
    # Both research budget modes (harness/budget.py); a session names one with --budget-mode, else gets this one.
    budget_mode='seconds',
    # Seconds mode: one job at a time (practice or lease) within this many seconds of the training worker; a
    # worker of another shape is charged its seconds x its price relative to the training worker's. These are the
    # session contract's last seconds budget (v1, before the dollar contract).
    research_seconds=16200,
    smoke_research_seconds=360,
    # Dollar mode (session contract generals-autoresearch-v2): up to 4 jobs at once within US dollars of worker time
    # at the contract's frozen list prices.
    research_usd=10.0,
    smoke_research_usd=0.5,
    wall_seconds=18000,
    smoke_wall_seconds=4800,
    final_train_seconds=3600,
    smoke_final_train_seconds=120,
)
# A smoke session's shortest run is the contract's (120 s): anything shorter is setup and export reserve only.
SMOKE = dict(practice_maps=2, grade_maps=2, max_training_jobs=2, max_run_seconds=180)
PRACTICE_SECONDS = 300
SMOKE_PRACTICE_SECONDS = 180
STATUS = {'cancelled': 'stopped'}


class Backend(Base):
    game = 'generals'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workspace = self.root / 'workspace'  # anchors the host workspace copy; Generals sends source directly
        self.controller = None
        self._app = None

    def open(self):
        from harness.policy import ENVIRONMENT
        from simulators.generals.autoresearch.contract import DEFAULT, frozen_manifest
        from .controller import Controller
        # The host re-derives the engine pin with git every time it seals or reads a network (`ar run`,
        # `ar package`, practice, grading). A host copy without the engine's git metadata would fail each of
        # those in front of the researcher, after allocation was spent, so it is refused here instead.
        try:
            frozen_manifest()
        except Exception as exc:
            raise RuntimeError(
                'This host cannot verify the pinned Generals engine (vendor/engines/generals-bots needs its git '
                'metadata at the pinned commit; tools/restore_upstream_metadata.py restores it after a fresh '
                'clone). No session was opened and nothing was spent: ' + type(exc).__name__ + ': '
                + str(exc)[:300]) from exc
        config = None
        if not self.resume:
            config = dict(DEFAULT, allow_agent_grade=False)  # private grading is host-only
            if self.smoke:
                config.update(SMOKE)
            # The session's budget mode and its concurrency limits; the ledger records both. In seconds mode
            # (also every fresh final training) the ledger holds the seconds as dollars of the training worker.
            config.update(self.limits)
            if self.budget_mode == budgets.USD:
                config['budget_usd'] = self.budget_usd
            else:
                config.update(budget_mode=budgets.SECONDS, budget_seconds=self.budget_seconds or
                              SPEC['smoke_research_seconds' if self.smoke else 'research_seconds'])
            config.update(self.options.get('session_config', {}))
        provider = self.options.get('_provider')  # tests stand in for Modal here
        try:
            if provider is None:
                import modal
                from .provider import ModalProvider
                app = modal.App('game-autoresearch-generals')
                self._app = app.run(environment_name=ENVIRONMENT)
                self._app.__enter__()
                provider = ModalProvider(self.release, app)
            self.controller = Controller(self.root, self.release, provider, config)
            mode = self.controller.ledger.mode
            if self.resume and mode != self.budget_mode and (self.budget_usd or self.budget_seconds):
                raise ValueError(f'This session was opened in {mode} budget mode; it cannot continue in '
                                 f'{self.budget_mode} mode')
            # A reopened session (host grading) keeps the mode its ledger was opened in.
            self.budget_mode = mode
            self.limits = {key: self.controller.ledger.limit(key) for key in self.limits}
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
                # Unclaimed queued work is cancelled; a claimed or running worker is stopped through its thread
                # (one still waiting for a GPU is abandoned uncharged).
                controller.release_job(job)
        controller.join(timeout=240 if self.smoke else 7500)
        for job in controller.ledger.jobs():
            if job['status'] == 'uncertain':
                try:  # terminate the recorded worker and settle it from the provider's record
                    controller.release_job(job)
                except Exception:
                    pass  # stays uncertain (and visible) when the provider has no usable record
        controller.join(timeout=240 if self.smoke else 7500)

    def close(self):
        try:
            self.settle()
        finally:
            self._leave()

    def _handle(self, action, **arguments):
        return self.controller.handle(dict(arguments, action=action))

    def budget(self):
        from simulators.generals.autoresearch.contract import PRICE, TRAIN_SHAPE, rate
        ledger = self.controller.ledger
        state = ledger.public()
        config = state['config']
        shapes = {'lease --gpu 1 --cpu 4 --memory 16 (and eval)': TRAIN_SHAPE,
                  'lease --gpu 1 --cpu 16 --memory 32': dict(gpu=1, cpu_cores=16, memory_mib=32768),
                  'lease --gpu 0 --cpu 8 --memory 16': dict(gpu=0, cpu_cores=8, memory_mib=16384),
                  'lease --gpu 0 --cpu 32 --memory 64': dict(gpu=0, cpu_cores=32, memory_mib=65536),
                  'lease --gpu 0 --cpu 1 --memory 2': dict(gpu=0, cpu_cores=1, memory_mib=2048)}
        if ledger.mode == budgets.SECONDS:
            weights = {name: budgets.seconds_weight(shape, TRAIN_SHAPE) for name, shape in shapes.items()}
            account = dict(budget_seconds=state['budget_seconds'], used_seconds=state['charged_or_reserved_seconds'],
                           remaining_seconds=state['remaining_seconds'],
                           allocatable_seconds=state['allocatable_seconds'],
                           charging='budget, used and remaining are seconds of the training worker (L40S, 4 CPU, '
                                    '16 GiB). A worker of another shape is charged its seconds x its seconds_weight '
                                    '(its list price per second relative to the training worker\'s). One job at a '
                                    'time; queued and running work is reserved in full until it ends.',
                           seconds_weight={name: round(weight, 4) for name, weight in weights.items()},
                           affordable_seconds={name: int(state['allocatable_seconds'] / weight)
                                               for name, weight in weights.items()})
        else:
            account = dict(budget_usd=config['budget_usd'], used_usd=state['charged_or_reserved_usd'],
                           remaining_usd=state['remaining_usd'], allocatable_usd=state['allocatable_usd'],
                           charging='budget, used and remaining are US dollars. A worker is charged its seconds x '
                                    'the price per second of its shape: GPU $%.2f/h, CPU core $%.4f/h, GiB of memory '
                                    '$%.4f/h. Queued and running work is reserved in full until it ends.'
                                    % tuple(3600 * PRICE[key] for key in ('gpu', 'cpu_core', 'memory_gib')),
                           usd_per_hour={name: round(3600 * rate(shape), 4) for name, shape in shapes.items()},
                           affordable_seconds={name: int(state['allocatable_usd'] / rate(shape))
                                               for name, shape in shapes.items()})
        return dict(account, budget_mode=ledger.mode,
                    min_run_seconds=config['min_run_seconds'], max_run_seconds=config.get('max_run_seconds'),
                    default_practice_seconds=self._practice_seconds(), practice_maps=config['practice_maps'],
                    concurrency=ledger.limit('max_concurrent_jobs'),
                    concurrent_gpus=ledger.limit('max_concurrent_gpus'),
                    concurrent_cpu_cores=ledger.limit('max_concurrent_cpu_cores'),
                    lease=dict(gpu='0 or 1', max_cpu=config['lease_max_cpu_cores'],
                               max_memory_gib=config['lease_max_memory_mib'] // 1024,
                               scratch='/scratch', work='/work', train_maps='/data/train-maps.npz'),
                    policy_tensor_cap=config['policy_tensor_cap'], source_bytes_cap=config['source_bytes_cap'],
                    artifact_bytes_cap=config['artifact_bytes_cap'], lifecycle=state['lifecycle'])

    def _jobs(self):
        """Ledger rows with the harness's kind, normalized status and the run a practice job evaluated."""
        rows = self.controller.ledger.jobs()
        payloads = {row['name']: json.loads(row['payload']) for row in rows}
        bundles = {row['artifact']: row['name'] for row in rows if row['artifact']}
        jobs = []
        for job in self._handle('status')['jobs']:
            payload = payloads[job['name']]
            kind = payload.get('kind')
            job = dict(job, status=STATUS.get(job['status'], job['status']),
                       kind=kind if kind in ('practice', 'lease', 'package') else 'train')
            charged, charge = job.pop('charged'), job.pop('rate')
            if 'charged_seconds' not in job:  # seconds mode reports charged_seconds and seconds_weight instead
                job['charged_usd'], job['usd_per_second'] = charged, charge
            if job['kind'] == 'practice':
                job['evaluates'] = bundles.get(payload.get('bundle'))
            elif job['kind'] == 'lease':
                job['shape'] = payload['shape']
                job['ready'] = self.controller.provider.lease_live(job['name'])
            elif job['kind'] == 'package':
                job['lease'], job['path'] = payload.get('lease'), payload.get('path')
            jobs.append(job)
        return jobs

    def status(self):
        jobs = self._jobs()
        return dict(jobs=jobs, active=[job['name'] for job in jobs if job['status'] in ACTIVE])

    def log(self, name, tail=80):
        jobs = self._jobs()
        job = next((job for job in jobs if job['name'] == name), None)
        if job is None:
            raise ValueError('Unknown run')
        folder = self.root / 'jobs' / name
        result = {'name': name, 'status': job['status'], 'kind': job['kind']}
        for early in ('stopped_early', 'cut_short'):  # complete, on the network the ended trainer last saved
            if job.get(early):
                result[early] = True
        if job['error']:  # the job's own failure; a top-level `error` would make the command fail
            result['job_error'] = job['error']
        for label, filename in (('stdout', 'trainer-stdout.log'), ('stderr', 'trainer-stderr.log'),
                                ('progress', 'progress.jsonl'), ('export_error', 'export-error.log')):
            if (folder / filename).is_file():
                result[label] = '\n'.join((folder / filename).read_text(errors='replace').splitlines()[-tail:])
        for label, filename in (('result', 'result.json'), ('practice', 'practice.json'),
                                ('host_error', 'host-error.json')):
            if (folder / filename).is_file():
                result[label] = json.loads((folder / filename).read_text())
        if (folder / 'exec').is_dir():  # a lease's commands, as last recorded by the host
            commands = [json.loads(path.read_text()) for path in sorted((folder / 'exec').glob('*.json'))]
            result['commands'] = [dict({key: c.get(key) for key in ('exec', 'state', 'exit_code')},
                                       command=c.get('command', '')[:200]) for c in commands[-tail:]]
        if job['kind'] in ('train', 'package'):
            evaluations = {other['name']: other.get('practice', other['status'])
                           for other in jobs if other.get('evaluates') == name}
            if evaluations:
                result['practice_runs'] = evaluations
        return result

    def check(self, config):
        from simulators.generals.autoresearch.contract import DEFAULT, EDITABLE
        files = workspace.editable(self.release, self.workspace)
        source_bytes = sum(len(files['training/' + name].encode()) for name in EDITABLE)
        if source_bytes > DEFAULT['source_bytes_cap']:
            raise ValueError(f'training/ source is {source_bytes} bytes; the cap is {DEFAULT["source_bytes_cap"]}')
        self._recipe(files, config)
        return dict(checked=True, source_bytes=source_bytes, source_bytes_cap=DEFAULT['source_bytes_cap'],
                    policy_tensor_cap=DEFAULT['policy_tensor_cap'], artifact_bytes_cap=DEFAULT['artifact_bytes_cap'],
                    note='Host checks are syntax and transport caps; the parameter and weight-file caps are enforced '
                         'on the worker at export.')

    @staticmethod
    def _recipe(files, config):
        recipe = json.loads(files[config])
        if not isinstance(recipe, dict):
            raise ValueError(config + ' must be a JSON object')
        if len(json.dumps(recipe).encode()) > 65536:
            raise ValueError(config + ' exceeds the 65536-byte configuration cap')
        return recipe

    def run(self, name, seconds, config, seed=1):
        from simulators.generals.autoresearch.contract import EDITABLE
        files = workspace.editable(self.release, self.workspace)
        return self._handle('run', name=name, seconds=seconds, seed=seed, config=self._recipe(files, config),
                            code={source: files['training/' + source] for source in EDITABLE})

    def stop(self, name):
        return self._handle('stop', name=name)

    def lease(self, name, seconds, gpu, cpu, memory):
        if self.final:
            raise ValueError('Fresh final training runs the recipe alone; it has no leases')
        for value, flag in ((gpu, '--gpu'), (cpu, '--cpu'), (memory, '--memory')):
            if value is None:
                raise ValueError(flag + ' is required')
        files = workspace.editable(self.release, self.workspace)
        return self._handle('lease', name=name, seconds=float(seconds), gpu=gpu, cpu=cpu, memory=memory, files=files)

    def lease_exec(self, name, command):
        return self._handle('exec', name=name, command=command)

    def lease_poll(self, name, exec_id, tail=None):
        return self._handle('poll', name=name, exec=exec_id, tail=tail)

    def lease_push(self, name):
        return self._handle('push', name=name, files=workspace.editable(self.release, self.workspace))

    def lease_sync(self, name, reload=False):
        return self._handle('sync', name=name, reload=reload)

    def package(self, name, lease, path, config):
        from simulators.generals.autoresearch.contract import EDITABLE
        files = workspace.editable(self.release, self.workspace)
        return self._handle('package', name=name, lease=lease, path=path, config=self._recipe(files, config),
                            code={source: files['training/' + source] for source in EDITABLE})

    def evaluate(self, name, seconds=None):
        return self._handle('eval', name=name, seconds=seconds or self._practice_seconds())

    def _practice_seconds(self):
        return SMOKE_PRACTICE_SECONDS if self.smoke else PRACTICE_SECONDS

    def submit(self, name):
        return self._handle('submit', name=name)

    def finalize(self):
        state = self._handle('finalize')
        used = (dict(used_seconds=state['charged_or_reserved_seconds']) if state['budget_mode'] == budgets.SECONDS
                else dict(used_usd=state['charged_or_reserved_usd']))
        return dict(used, lifecycle=state['lifecycle'], submissions=state['submissions'], points=state['points'])

    def tick(self):
        self.controller.tick()

    def grade(self):
        from simulators.generals.autoresearch.manage import grade_session
        config = self.controller.ledger.config
        # Official only when the host says this session ran at the declared budgets;
        # anything else (smoke, custom budgets) is graded as development.
        return grade_session(self.release, self.root, self.controller.provider, not self.official,
                             maps=config['grade_maps'], timeout=config.get('host_grade_timeout_seconds'),
                             contract_usd=config['budget_usd'] if self.official else None)
