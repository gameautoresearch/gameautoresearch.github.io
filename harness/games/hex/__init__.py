"""Hex: 13x13, no swap rule, graded against a MoHex ladder (the method of Jones 2021). Controller and provider live beside this file."""
import json

from harness import budget as budgets, workspace
from harness.backend import ACTIVE, Backend as Base

SPEC = dict(
    game='hex',
    # How `harness retrain` sets a replicate's training seed: the run seed, the config's `seed` field, or the
    # backend's own fresh host seeds for a final.
    retrain_seed='run',
    # What `ar` offers in this game: `ar run --seed`, `ar eval --seconds`, and the box's CPU torch version.
    run_seed=True, eval_seconds=True, box_torch='2.15.0',
    # The graded policy's size limit (parameters and buffers): Hex's own 10,000,000, enforced by
    # `simulators/hex/autoresearch/policy_worker.py` and stated in its brief (the other games use 500,000).
    parameter_cap=10_000_000,
    # `ar lease`, `exec`, `poll`, `push`, `sync` and `package`: workers of a chosen shape running the researcher's
    # own commands, with the session's scratch Volume.
    leases=True,
    # Hex is the first game to run an installed CLI researcher as well as Terminus-2.
    agents=('terminus-2', 'claude-code'),
    venv='.venv-hex',
    constructor='simulators.hex.autoresearch.prepare:prepare',
    backend_options={
        'development': 'grade: true forces a development grade. Sessions not at the declared budgets are always '
                       'development.',
        'private_openings': 'grade: development-only prefix of the private opening schedule (smoke default 4).',
        'session_config': 'open: explicit overrides of the session contract (host use only).'},
    contract='simulators/hex/autoresearch/benchmark.json',
    mode='13x13 Hex without the swap rule; MoHex 2.0 anchor (the method of Jones 2021, "Scaling Scaling Laws with Board Games")',
    hardware='workers you lease, of a shape you choose (0 or 1 L40S GPU, up to 32 CPU cores and 64 GiB), with '
             'shared /scratch storage; `training/launch.py` uses one L40S with 4 CPU cores and 16 GiB unless told '
             'otherwise, which is also what a recipe\'s final training gets; public practice on an 8-CPU,'
             ' 16 GiB worker without a GPU. One research budget pays for all of it, each worker second charged by '
             'its shape (the 8-CPU practice worker costs about half the training worker per second); `ar budget` '
             'lists the rates and what is left',
    # Both research budget modes (harness/budget.py). `seconds` (the default): one job at a time within 7,200
    # seconds of the training worker, other shapes weighted by their price relative to it (the v2 contract's
    # allocation). `usd`: up to 4 jobs at once within $10.00 at the frozen list prices.
    budget_mode='seconds',
    research_seconds=7200,
    smoke_research_seconds=600,
    research_usd=10.0,
    smoke_research_usd=0.5,
    wall_seconds=18000,
    smoke_wall_seconds=4800,
    final_train_seconds=3600,
    smoke_final_train_seconds=180,
)
# The contract's bounded end-to-end smoke: short runs, a few practice openings, a teardown reserve.
SMOKE = dict(min_run_seconds=30, practice_openings=2, teardown_reserve_seconds=20, max_run_seconds=300)
SMOKE_PRIVATE_OPENINGS = 4
# 288 practice games on 13x13. The anchor's moves dominate: about 37 a game at 1 to 2 s of one core on a
# laptop, in 32 games over 7 MoHex processes. Not yet timed on a Modal worker; a schedule that overruns
# its grant fails, still charged, so the default is generous and `ar eval --seconds` can lower it.
# Smoke practice is 36 games.
PRACTICE_SECONDS = 1800
SMOKE_PRACTICE_SECONDS = 300
STATUS = {'cancelled': 'stopped'}
LOG_CHARS = 20000


def _tail(path, lines):
    text = '\n'.join(path.read_text(errors='replace').splitlines()[-lines:])
    return text[-LOG_CHARS:]


class Backend(Base):
    game = 'hex'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workspace = self.root / 'workspace'  # anchors the host workspace copy; source travels in the request
        self.controller = None
        self._app = None

    def open(self):
        from harness.policy import ENVIRONMENT
        from simulators.hex.autoresearch.contract import DEFAULT, PRICE, SECONDS_RULE
        from .controller import Controller
        config = None
        if not self.resume:
            # The ledger records the mode, its concurrency limits and its unit; it never changes afterwards.
            config = dict(DEFAULT)
            if self.smoke:
                config.update(SMOKE)
            config.update(self.limits, budget_mode=self.budget_mode)
            if self.budget_mode == budgets.USD:
                config.update(budget_usd=self.budget_usd, price=dict(PRICE))
            else:  # research in seconds mode, and every fresh final training (one run of its allowance)
                del config['budget_usd']
                config.update(budget_seconds=self.budget_seconds or SPEC[('smoke_' if self.smoke else '')
                                                                         + 'research_seconds'],
                              price=dict(PRICE), allocation_rule=SECONDS_RULE)
            config.update(self.options.get('session_config', {}))
        provider = self.options.get('_provider')  # tests stand in for Modal here
        try:
            if provider is None:
                import modal
                from simulators.hex.autoresearch.transport import install_transport_guard
                from .provider import ModalProvider
                install_transport_guard()
                app = modal.App('game-autoresearch-hex')
                self._app = app.run(environment_name=ENVIRONMENT)
                self._app.__enter__()
                provider = ModalProvider(self.release, app)
            self.controller = Controller(self.root, self.release, provider, config)
            if self.resume:
                # A reopened session keeps the mode its ledger was opened in; it never continues in another.
                recorded = self.controller.ledger.mode
                if (self.budget_seconds is not None or self.budget_usd is not None) and recorded != self.budget_mode:
                    raise ValueError(f'This session was opened in {recorded} budget mode; it cannot continue in '
                                     f'{self.budget_mode} mode')
                self.budget_mode = recorded
                self.limits = {key: self.controller.ledger.limit(key) for key in self.limits}
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
                try:  # terminate the recorded worker and settle it from Modal's record
                    controller.release_job(job)
                except Exception:
                    pass  # stays uncertain (and visible) when Modal has no usable record

    def close(self):
        try:
            self.settle()
        finally:
            self._leave()

    def _handle(self, action, **arguments):
        return self.controller.handle(dict(arguments, action=action))

    def budget(self):
        from .controller import PRACTICE_MIN_SECONDS
        from simulators.hex.autoresearch.contract import DEFAULT, PANEL, PRACTICE_SHAPE, PRICE, TRAIN_SHAPE, rate
        ledger = self.controller.ledger
        state = ledger.public()
        config = state['config']
        shapes = {'lease --gpu 1 --cpu 4 --memory 16': TRAIN_SHAPE, 'eval (8 CPU, 16 GiB)': PRACTICE_SHAPE,
                  'lease --gpu 1 --cpu 16 --memory 32': dict(gpu=1, cpu_cores=16, memory_mib=32768),
                  'lease --gpu 0 --cpu 8 --memory 16': dict(gpu=0, cpu_cores=8, memory_mib=16384),
                  'lease --gpu 0 --cpu 32 --memory 64': dict(gpu=0, cpu_cores=32, memory_mib=65536),
                  'lease --gpu 0 --cpu 1 --memory 2': dict(gpu=0, cpu_cores=1, memory_mib=2048)}
        unit = ledger.unit
        if unit == 'usd':
            reply = dict(budget_usd=config['budget_usd'], used_usd=state['charged_or_reserved_usd'],
                         remaining_usd=state['remaining_usd'], allocatable_usd=state['allocatable_usd'])
        else:
            reply = dict(budget_seconds=config['budget_seconds'], used_seconds=state['charged_or_reserved_seconds'],
                         remaining_seconds=state['remaining_seconds'],
                         allocatable_seconds=state['allocatable_seconds'])
        reply.update(budget_mode=ledger.mode,
                     teardown_reserve_seconds=config.get('teardown_reserve_seconds', 0),
                     min_run_seconds=config['min_run_seconds'], max_run_seconds=config.get('max_run_seconds'),
                     min_practice_seconds=PRACTICE_MIN_SECONDS, default_practice_seconds=self._practice_seconds(),
                     practice_openings=config['practice_openings'],
                     practice_games=config['practice_openings'] * 2 * len(PANEL),
                     practice_opponents=[p['name'] for p in PANEL],
                     concurrency=ledger.limit('max_concurrent_jobs'),
                     concurrent_gpus=ledger.limit('max_concurrent_gpus'),
                     concurrent_cpu_cores=ledger.limit('max_concurrent_cpu_cores'))
        if unit == 'usd':
            reply.update(charging='budget, used and remaining are US dollars. A worker is charged its seconds x the '
                                  'price per second of its shape: GPU $%.2f/h, CPU core $%.4f/h, GiB of memory '
                                  '$%.4f/h. Queued and running work is reserved in full until it ends.'
                                  % tuple(3600 * PRICE[key] for key in ('gpu', 'cpu_core', 'memory_gib')),
                         usd_per_hour={name: round(3600 * rate(shape), 4) for name, shape in shapes.items()})
        else:
            reply.update(charging='budget, used and remaining are seconds of the training worker (L40S, 4 CPU, '
                                  '16 GiB). A worker of another shape is charged its seconds x its '
                                  '`seconds_weight`, its list price relative to the training worker. One job at a '
                                  'time: a practice or lease is refused while another is active. Queued and '
                                  'running work is reserved in full until it ends.',
                         seconds_weight={name: round(ledger.rate(shape), 4) for name, shape in shapes.items()})
        reply.update(affordable_seconds={name: int(state['allocatable_' + unit] / ledger.rate(shape))
                                         for name, shape in shapes.items()},
                     lease=dict(gpu='0 or 1', max_cpu=config.get('lease_max_cpu_cores', DEFAULT['lease_max_cpu_cores']),
                                max_memory_gib=config.get('lease_max_memory_mib', DEFAULT['lease_max_memory_mib']) // 1024,
                                scratch='/scratch', work='/work'),
                     policy_tensor_cap=config['policy_tensor_cap'], source_bytes_cap=config['source_bytes_cap'],
                     artifact_bytes_cap=config['artifact_bytes_cap'], inference_batch=config['inference_batch'],
                     position_seconds=config['position_seconds'], call_seconds=config['call_seconds'])
        return reply

    def _jobs(self):
        """Ledger rows with the harness's kind, normalized status and the run a practice job evaluated."""
        ledger = self.controller.ledger.jobs()
        unit = self.controller.ledger.unit
        bundles = {row['artifact']: row['name'] for row in ledger if row['artifact']}
        rows = []
        for row in ledger:
            payload = json.loads(row['payload'])
            job = {key: row[key] for key in ('name', 'grant', 'started', 'ended', 'artifact_sha', 'error')}
            job['charged_' + unit] = row['charged']
            job['status'] = STATUS.get(row['status'], row['status'])
            job['kind'] = payload.get('kind') if payload.get('kind') in ('practice', 'lease', 'package') else 'train'
            job['usd_per_second' if unit == 'usd' else 'seconds_weight'] = payload.get('rate')
            if job['kind'] == 'practice':
                job['evaluates'] = bundles.get(payload.get('bundle'))
            elif job['kind'] == 'lease':
                job['shape'] = payload['shape']
                job['ready'] = self.controller.provider.lease_live(row['name'])
            elif job['kind'] == 'package':
                job['lease'], job['path'] = payload.get('lease'), payload.get('path')
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
                job['practice'] = _practice_summary(json.loads((folder / 'practice.json').read_text()))
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
                                ('export_error', 'export-error.log')):
            if (folder / filename).is_file():
                result[label] = _tail(folder / filename, tail)
        for label, filename in (('supervisor', 'supervisor.json'), ('practice', 'practice.json')):
            if (folder / filename).is_file():
                result[label] = json.loads((folder / filename).read_text())
        if (folder / 'exec').is_dir():  # a lease's commands, as last recorded by the host
            commands = [json.loads(path.read_text()) for path in sorted((folder / 'exec').glob('*.json'))]
            result['commands'] = [dict({key: c.get(key) for key in ('exec', 'state', 'exit_code')},
                                       command=c.get('command', '')[:200]) for c in commands[-tail:]]
        if job['kind'] in ('train', 'package'):
            evaluations = {}
            for other in jobs:
                path = self.root / 'jobs' / other['name'] / 'practice.json'
                if other.get('evaluates') == name:
                    evaluations[other['name']] = (_practice_summary(json.loads(path.read_text()))
                                                  if path.is_file() else other['status'])
            if evaluations:
                result['practice_runs'] = evaluations
        return result

    def check(self, config):
        from simulators.hex.autoresearch.contract import DEFAULT, EDITABLE
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
                    note='Host checks are syntax and transport caps; the tensor and weight-file caps are enforced on '
                         'the worker at export. The CPU dry run in your box exercises train.py and policy.py.')

    def run(self, name, seconds, config, seed=1):
        from simulators.hex.autoresearch.contract import EDITABLE
        files = workspace.editable(self.release, self.workspace)
        recipe = json.loads(files[config])
        if not isinstance(recipe, dict):
            raise ValueError(config + ' must be a JSON object')
        from simulators.hex.autoresearch.contract import TRAIN_SHAPE
        state = self.controller.ledger.public()
        reserve = state['config'].get('teardown_reserve_seconds', 0)
        granted = self._granted(float(seconds), self.controller.ledger.rate(TRAIN_SHAPE))
        if granted < min(float(seconds), state['config']['min_run_seconds']):
            raise ValueError(f'Only {granted:.0f} seconds of the training worker remain after the {reserve:g}-second '
                             'teardown reserve; that is below the minimum run')
        reply = self._handle('run', name=name, seconds=granted, seed=seed, config=recipe,
                             code={source: files['training/' + source] for source in EDITABLE})
        if granted != seconds:
            reply['note'] = (f'Granted {granted:g} of the requested {seconds:g} seconds: '
                             f'{reserve:g} seconds of the training worker are reserved for worker teardown.')
        return reply

    def stop(self, name):
        return self._handle('stop', name=name)

    def _granted(self, seconds, charge):
        """Worker seconds to grant: the request, or what the teardown reserve leaves when the request overlaps it.
        `charge` is the worker's rate in the session's unit."""
        state = self.controller.ledger.public()
        reserve = state['config'].get('teardown_reserve_seconds', 0)
        unit = self.controller.ledger.unit
        if reserve and state['allocatable_' + unit] < seconds * charge <= state['remaining_' + unit] + 1e-9:
            return state['allocatable_' + unit] / charge
        return seconds

    def lease(self, name, seconds, gpu, cpu, memory):
        from simulators.hex.autoresearch.contract import lease_shape
        if self.final:
            raise ValueError('Fresh final training runs the recipe alone; it has no leases')
        for value, flag in ((gpu, '--gpu'), (cpu, '--cpu'), (memory, '--memory')):
            if value is None:
                raise ValueError(flag + ' is required')
        charge = self.controller.ledger.rate(lease_shape(self.controller.ledger.config, gpu, cpu, memory))
        files = workspace.editable(self.release, self.workspace)
        return self._handle('lease', name=name, seconds=self._granted(float(seconds), charge), gpu=gpu, cpu=cpu,
                            memory=memory, files=files)

    def lease_exec(self, name, command):
        return self._handle('exec', name=name, command=command)

    def lease_poll(self, name, exec_id, tail=None):
        return self._handle('poll', name=name, exec=exec_id, tail=tail)

    def lease_push(self, name):
        return self._handle('push', name=name, files=workspace.editable(self.release, self.workspace))

    def lease_sync(self, name, reload=False):
        return self._handle('sync', name=name, reload=reload)

    def package(self, name, lease, path, config):
        from simulators.hex.autoresearch.contract import EDITABLE
        files = workspace.editable(self.release, self.workspace)
        recipe = json.loads(files[config])
        if not isinstance(recipe, dict):
            raise ValueError(config + ' must be a JSON object')
        return self._handle('package', name=name, lease=lease, path=path, config=recipe,
                            code={source: files['training/' + source] for source in EDITABLE})

    def evaluate(self, name, seconds=None):
        return self._handle('eval', name=name, seconds=seconds or self._practice_seconds())

    def _practice_seconds(self):
        return SMOKE_PRACTICE_SECONDS if self.smoke else PRACTICE_SECONDS

    def submit(self, name):
        return self._handle('submit', name=name)

    def finalize(self):
        state = self._handle('finalize')
        unit = self.controller.ledger.unit
        return {'submissions': state['submissions'], 'points': state['points'],
                'used_' + unit: state['charged_or_reserved_' + unit]}

    def tick(self):
        self.controller.tick()

    def grade(self):
        from simulators.hex.autoresearch.manage import grade_session
        config = self.controller.ledger.config
        # Official only when the host says this session ran at the declared budgets.
        development = bool(self.options.get('development')) or not self.official
        openings = (self.options.get('private_openings', SMOKE_PRIVATE_OPENINGS if self.smoke else None)
                    if development else None)
        points = grade_session(self.release, self.root, self.controller.provider, development, openings)
        final = next((points[key] for key in points if float(key) == 1), None)
        return dict(development=development, private_openings=openings or config['grade_openings'], final=final,
                    points=points)


def _practice_summary(result):
    """The parts of a practice result a status line needs; `ar log` shows the whole result."""
    keys = ('status', 'raw_score', 'raw_score_ci95', 'per_opponent', 'games', 'error')
    summary = {key: result[key] for key in keys if key in result}
    if 'anchor' in result:
        summary['elo_vs_anchor'] = result['anchor'].get('elo_vs_anchor')
    return summary
