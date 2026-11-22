"""Clash: fixed 8-card mirror deck at level 11, short matches (60+30 s, 2 decisions/s), on the native fast trainer.
Controller and provider live beside this file."""
import os
import time

from harness import budget, workspace
from harness.backend import ACTIVE, Backend as Base

# The training worker of every run (and of the fresh final): one L40S with 4 CPU cores and 16 GiB.
TRAIN_SHAPE = dict(gpu=1, cpu_cores=4, memory_mib=16384)

SPEC = dict(
    game='clash',
    # How `harness retrain` sets a replicate's training seed: the run seed, the config's `seed` field, or the
    # backend's own fresh host seeds for a final.
    retrain_seed='config',
    # What `ar` offers in this game: `ar run --seed`, `ar eval --seconds`, and the box's CPU torch version.
    run_seed=False, eval_seconds=False, box_torch='2.9.1',
    # The graded policy's size limit (parameters and buffers), the same in every game.
    parameter_cap=500_000,
    venv='.venv-clash-ar',
    # The short-v1 benchmark (60 s + 30 s overtime, 2 decisions per simulated second). Its release is standalone:
    # a new scripted panel, public practice and protected schedule. The full-length fast-v4 form is at git tag
    # full-size-20261113.
    constructor='simulators.clash.autoresearch.short_release:prepare',
    constructor_options={},
    contract='simulators/clash/autoresearch/benchmark-short-v1.json',
    mode='fixed 8-card mirror deck, level 11, 60+30 s matches at 2 decisions/s, native fast trainer',
    hardware='one L40S GPU with 4 CPU cores and 16 GiB per training run (the Clash short-v1 contract); public '
             'practice runs on a CPU-only worker',
    # Research budget modes (harness/budget.py). The short-v1 contract is seconds mode: one job at a time within
    # 7,200 training-worker seconds. Dollar mode runs up to budget.PARALLEL jobs (two training runs at once, one GPU
    # each) within the same training-worker time at its frozen price, on the dollar games' wall clock. CPU practice is
    # limited by count and outside the allocation in both modes.
    budget_mode='seconds',
    research_seconds=7200,
    smoke_research_seconds=600,
    research_usd=round(7200 * budget.rate(TRAIN_SHAPE), 2),  # $5.81
    smoke_research_usd=round(600 * budget.rate(TRAIN_SHAPE), 2),  # $0.48
    wall_seconds=18000,
    smoke_wall_seconds=4800,
    # One launch: at most 3300 worker seconds plus the ledger's 30-second settlement reserve.
    final_train_seconds=3330,
    smoke_final_train_seconds=120,
    evaluation='public practice; `grade` plays the final once on the protected schedule (fresh secret seeds) as a '
               'development grade; there is no official private final bank',
)
APP = 'game-autoresearch-clash'
SETTLE_SECONDS = 300

class Backend(Base):
    game = 'clash'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workspace = self.root / 'workspace'  # native trainer tree: policy.py, learner.py, train.py, config.json
        self.controller = None
        self._app = None

    def open(self):
        from simulators.clash.autoresearch.short_release import verify
        from .controller import Controller
        verify(self.release)  # before any cloud object exists
        provider = self.options.get('_provider')  # tests stand in for Modal here
        limit = usd = None  # reopening keeps the existing ledger's mode and ceiling
        ceiling = dict(gpu_ceiling_seconds=None)
        if not self.resume:
            if self.budget_mode == budget.USD:
                usd = float(self.budget_usd)
                ceiling.update(budget_mode=budget.USD, research_usd=usd, usd_per_second=budget.rate(TRAIN_SHAPE))
            else:
                limit = self.budget_seconds or SPEC['smoke_research_seconds' if self.smoke else 'research_seconds']
                ceiling.update(gpu_ceiling_seconds=limit, budget_mode=budget.SECONDS)
            self.root.mkdir(parents=True)
        # A reopened session given an amount must be in that amount's mode; given none, it keeps the ledger's.
        expected = self.budget_mode if self.budget_usd is not None or self.budget_seconds is not None else None
        try:
            # A reopened session needs no worker until it grades; `grade` creates the provider then.
            if provider is None and not self.resume:
                provider = self._modal(ceiling)
            self.controller = Controller(self.root, self.release, provider, limit, usd=usd, mode=expected)
        except BaseException:
            self._leave()
            raise
        self.budget_mode = self.controller.ledger.mode
        self.limits = self.controller.limits()

    def _modal(self, ceiling, record='app.json', provider_root='provider'):
        os.environ.setdefault('MODAL_DISABLE_API_PROXY', '1')
        import modal
        from harness import hardware
        from harness.policy import ENVIRONMENT
        from simulators.clash.autoresearch.common import dump, sha
        from .provider import FastProvider, snapshot
        archive = snapshot(self.root / 'uploads')
        app = modal.App(APP)
        self._app = app.run(environment_name=ENVIRONMENT)
        self._app.__enter__()
        dump(self.root / record, {
            'app_id': app.app_id, 'environment': ENVIRONMENT, 'source_archive_sha256': sha(archive),
            **ceiling, 'runtime_base_image_override': os.getenv('CLASH_AR_BASE_IMAGE_ID'),
            'hardware': {'gpu': hardware.GPU, 'cpu_cores': 4, 'memory_gib': 16,
                         'selection': 'repository default GPU (harness.hardware)'},
            'at': time.time()})
        return FastProvider(app, archive, self.root / provider_root)

    def _leave(self):
        if self._app is not None:
            context, self._app = self._app, None
            context.__exit__(None, None, None)

    def settle(self):
        if self.controller is not None:
            self.controller.settle(SETTLE_SECONDS)

    def close(self):
        try:
            self.settle()
        finally:
            self._leave()
            if self.controller is not None:
                self.controller.close()

    def budget(self):
        return self.controller.budget()

    def status(self):
        jobs = self.controller.status()
        return dict(jobs=jobs, active=[job['name'] for job in jobs if job['status'] in ACTIVE])

    def log(self, name, tail=80):
        return self.controller.log(name, tail)

    def check(self, config):
        from .controller import preflight
        root, meta = workspace.session_root(self.release, self.workspace)
        paths = {native: root / name for name, native in meta['training_map'].items()}
        paths['config.json'] = root / config
        return dict(checked=True, **preflight(paths))

    def run(self, name, seconds, config, seed=1):
        if seed != 1:
            raise ValueError('Clash takes its training seed from the `seed` field of the config; --seed is not used')
        self.sync(config)
        return self.controller.run(name, seconds, config)

    def stop(self, name):
        return self.controller.stop(name)

    def evaluate(self, name, seconds=None):
        reply = self.controller.evaluate(name)
        if seconds is not None:
            reply['note'] = 'Clash practice is one fixed public schedule in a CPU-only worker; --seconds is ignored'
        return reply

    def submit(self, name):
        return self.controller.submit(name)

    def finalize(self):
        return self.controller.finalize()

    def tick(self):
        self.controller.tick()

    def grade(self):
        if self.controller.provider is None:
            # A reopened session: the protected-schedule grade runs on a CPU worker of its own app. The research
            # app record and image cache are left as they were.
            self.controller.attach(self._modal(dict(gpu_ceiling_seconds=None), record='grade-app.json',
                                               provider_root='grade-provider'))
        return self.controller.grade()
