"""Polytopia: Bardur 1v1 on a small drylands board with fog. Controller, provider and transport live beside this file."""
import json
import secrets
import time

from harness import budget as budgets, workspace
from harness.backend import ACTIVE, Backend as Base
# The frozen game mode (board, map type, round limit) is defined once, in the worker package (rl/mode.py).
from simulators.polytopia.autoresearch.v1.core import MODE

SPEC = dict(
    game='polytopia',
    # How `harness retrain` sets a replicate's training seed: the run seed, the config's `seed` field, or the
    # backend's own fresh host seeds for a final.
    retrain_seed='config',
    # What `ar` offers in this game: `ar run --seed`, `ar eval --seconds`.
    run_seed=False, eval_seconds=False,
    # The graded policy's size limit (parameters and buffers), the same in every game.
    parameter_cap=500_000,
    # `ar lease`, `exec`, `poll`, `push`, `sync` and `package`: workers of a chosen shape running the researcher's
    # own commands, with the session's scratch Volume.
    leases=True,
    # Polytopia also runs the installed Claude Code researcher, as Hex and Generals do; the model key then enters
    # the box.
    agents=('terminus-2', 'claude-code'),
    venv='.venv-polytopia-ar',
    constructor='simulators.polytopia.autoresearch.prepare:prepare',
    constructor_options={
        'panel': 'false builds a bootstrap release without the four practice opponents (they are starter runs of this '
                 'game mode, so the first release of a mode cannot have them): it trains and runs smoke sessions, and '
                 'full practice and grading refuse it; ladder builds a release with the earlier ladder (random, '
                 'starter-20m, starter-65m, starter-200m) in place of the current panel',
        'starter': 'ppo (default, the working PPO starter) or empty (boilerplate only: rollout plumbing with no '
                   'learning, an empty learner, a linear placeholder network; its own brief and recipes, and the '
                   'reference learning code withheld from the box)'},
    backend_options={
        'session_config': 'open: explicit overrides of the session contract (host use only), e.g. '
                          '{"max_lease_seconds": 900}.',
        'scratch_volume': 'open: the research session adopts this existing Volume as its /scratch instead of '
                          'creating one (a continued session gets a copy of its earlier scratch; host use only). '
                          'Never another live session\'s Volume; a fresh final never mounts scratch.'},
    contract='simulators/polytopia/autoresearch/benchmark.json',
    mode=f'Bardur vs Bardur 1v1, {MODE.SIZE}x{MODE.SIZE}, fog of war, {MODE.MAPS} maps, {MODE.ROUND_LIMIT}-round cap',
    hardware='workers you lease, of a shape you choose (0 or 1 L40S GPU, up to 32 CPU cores and 64 GiB), with shared '
             '/scratch storage; `training/launch.py` uses one L40S with 4 CPU cores and 16 GiB unless told '
             'otherwise, which is also what a recipe\'s final training gets. The research budget pays for every '
             'worker by its wall seconds and shape (`ar budget` gives the unit, the charge of each shape and what is '
             'left). Public practice runs on separate 4-core CPU workers outside the budget and beside your '
             'workers, one practice job at a time',
    # Both research budget modes (harness/budget.py). The default is seconds of the training worker, one job at a
    # time (the 14,400 seconds of the small form before the dollar contract); dollar mode is the contract's
    # US dollars of worker time at its frozen list prices, parallel jobs.
    budget_mode='seconds',
    research_seconds=14400,
    smoke_research_seconds=600,
    research_usd=23.0,
    smoke_research_usd=1.0,
    wall_seconds=18000,
    smoke_wall_seconds=4800,
    final_train_seconds=3600,
    smoke_final_train_seconds=360,
)
PRIVATE_MAPS = {True: 2, False: 128}  # smoke, full
PROVIDER_OPTIONS = ('base_image_id', 'reuse_image_id', 'cpu_timeout', 'training_updates')
# A session root of the earlier contract (v1, budgeted in seconds of one worker with its own ledger files).
EARLIER = ('gpu-ledger.json', 'state.json')


class Backend(Base):
    game = 'polytopia'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workspace = self.root / 'workspace'  # native trainer tree: src/..., experiments/..., recipe.json, arch.json
        self.controller = None
        self._app = None

    def open(self):
        from simulators.polytopia.autoresearch.contract import DEFAULT, SECONDS_DEFAULT
        from simulators.polytopia.autoresearch.v1.core import write
        from .controller import Controller
        record = self.root / 'backend.json'
        config = None
        prefix = 'smoke_' if self.smoke else ''
        if self.resume:
            if any((self.root / name).exists() for name in EARLIER) and not (self.root / 'ledger.sqlite3').exists():
                raise ValueError('This session was opened under the earlier contract budgeted in seconds; this '
                                 'contract cannot continue or grade it')
            saved = json.loads(record.read_text())
            if saved['smoke'] != self.smoke:
                raise ValueError('Session was created in a different mode')
            # A resumed session keeps the budget mode it was opened in (a record without one is a dollar session);
            # it grants nothing new, and an amount in another mode is refused.
            mode = saved.get('budget_mode', budgets.USD)
            if (self.budget_usd is not None or self.budget_seconds is not None) and self.budget_mode != mode:
                raise ValueError(f'This session was opened in {mode} budget mode; it cannot continue in '
                                 f'{self.budget_mode} mode')
            self.budget_mode, self.limits = mode, budgets.limits(mode)
        elif self.budget_mode == budgets.USD:
            config = dict(DEFAULT, budget_usd=float(self.budget_usd), **self.limits)
        else:
            # Seconds of the training worker, one job at a time. Fresh final training is always this mode.
            config = dict(SECONDS_DEFAULT, budget_seconds=float(self.budget_seconds or SPEC[prefix + 'research_seconds']),
                          **self.limits)
        if not self.resume:
            config.update(self.options.get('session_config', {}))
            self.root.mkdir(parents=True)
            amount = 'budget_usd' if config['budget_mode'] == budgets.USD else 'budget_seconds'
            write(record, dict(game=self.game, smoke=self.smoke, contract=config['version'],
                               budget_mode=config['budget_mode'], **{amount: config[amount]}, created_at=time.time()))
            # A fresh independent host schedule, fixed before any research; never shown to the researcher.
            private = self.root / 'private'
            private.mkdir()
            private.chmod(0o700)
            seeds = [int.from_bytes(secrets.token_bytes(4), 'big') % 1_000_000_000 + 1_000_000
                     for _ in range(PRIVATE_MAPS[self.smoke])]
            write(private / 'schedule.json', dict(seeds=seeds, opponent_seed=int.from_bytes(secrets.token_bytes(4), 'big')))
            adopted = self.options.get('scratch_volume')
            if adopted and not self.final:
                if not isinstance(adopted, str) or not adopted.startswith('polytopia-scratch-'):
                    raise ValueError('scratch_volume names a polytopia-scratch-... Volume')
                from harness.policy import ENVIRONMENT
                # The provider's `scratch` reads this record and creates nothing.
                write(self.root / 'scratch-volume.json', dict(volume=adopted, environment=ENVIRONMENT,
                                                              mount='/scratch', created_unix=time.time(),
                                                              adopted=True))
        provider = self.options.get('_provider')  # tests stand in for Modal here
        try:
            if provider is None:
                import modal
                from harness.policy import ENVIRONMENT
                from .provider import Provider
                self._app = modal.App('game-autoresearch-polytopia').run(environment_name=ENVIRONMENT)
                app = self._app.__enter__()
                write(self.root / f'app-{time.time_ns()}.json', dict(app_id=app.app_id))
                provider = Provider(self.release, app, **{key: self.options[key] for key in PROVIDER_OPTIONS
                                                          if key in self.options})
            # CPU-only gate before any GPU work: a legal engine transition through the UID-isolated policy process.
            provider.preflight(self.root / 'cpu-preflight.json')
            self.controller = Controller(self.root, self.release, provider, config, smoke=self.smoke)
        except BaseException:
            self._leave()
            raise

    def _leave(self):
        if self._app is not None:
            context, self._app = self._app, None
            context.__exit__(None, None, None)

    def settle(self):
        if self.controller is not None:
            # A stopped worker ends within its teardown; a practice job ends between its CPU workers.
            self.controller.settle(timeout=900 if self.smoke else 7500)

    def close(self):
        try:
            self.settle()
        finally:
            self._leave()

    def budget(self):
        return self.controller.budget()

    def status(self):
        jobs = self.controller.jobs()
        return dict(jobs=jobs, active=[job['name'] for job in jobs if job['status'] in ACTIVE],
                    sealed_submission=self.controller.submission(),
                    latest_practice=self.controller.latest_practice())

    def log(self, name, tail=80):
        return self.controller.log(name, tail)

    def _native(self, config):
        """The synced native trainer files of this request: the editable sources, the chosen recipe, the architecture."""
        from .controller import NATIVE
        self.sync(config)
        return {name: (self.workspace / name).read_text() for name in NATIVE}

    def check(self, config):
        return self.controller.check(self._native(config))

    def run(self, name, seconds, config, seed=1):
        # The training seed is the `seed` field of the chosen configuration; `ar run --seed` does not replace it.
        return dict(self.controller.queue_run(name, seconds, self._native(config), config),
                    seed='the seed field of ' + config)

    def stop(self, name):
        return self.controller.stop(name)

    def lease(self, name, seconds, gpu, cpu, memory):
        if self.final:
            raise ValueError('Fresh final training runs the recipe alone; it has no leases')
        for value, flag in ((gpu, '--gpu'), (cpu, '--cpu'), (memory, '--memory')):
            if value is None:
                raise ValueError(flag + ' is required')
        files = workspace.editable(self.release, self.workspace)
        return self.controller.lease(name, seconds, gpu, cpu, memory, files)

    def lease_exec(self, name, command):
        return self.controller.lease_exec(name, command)

    def lease_poll(self, name, exec_id, tail=None):
        return self.controller.lease_poll(name, exec_id, tail)

    def lease_push(self, name):
        return self.controller.lease_push(name, workspace.editable(self.release, self.workspace))

    def lease_sync(self, name, reload=False):
        return self.controller.lease_sync(name, reload)

    def package(self, name, lease, path, config):
        return self.controller.package(name, lease, path, self._native(config), config)

    def evaluate(self, name, seconds=None):
        # Practice is a fixed public schedule on CPU workers; it takes no time grant and no allocation.
        return self.controller.queue_practice(name)

    def submit(self, name):
        return self.controller.submit(name)

    def finalize(self):
        return self.controller.finalize()

    def tick(self):
        self.controller.tick()

    def grade(self):
        from simulators.polytopia.autoresearch.v1.core import verify, write
        from .controller import grade_submission
        latest = self.controller.submission()
        if self.controller.ledger.lifecycle() != 'finalized' or not latest:
            raise ValueError('Nothing to grade: the session has no finalized submission')
        public = self.root / 'grade-public.json'
        if public.is_file():
            return json.loads(public.read_text())
        bundle = self.controller.sealed(latest['sha256'])
        if verify(bundle) != latest['sha256']:
            raise ValueError('The sealed submission differs from its identity')
        result = grade_submission(self.controller.provider, str(bundle), self.release, self.root / 'private', self.smoke)
        write(public, result)
        return result
