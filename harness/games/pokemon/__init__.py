"""Pokémon: Gen 9 random battle 3v3, trained on the admitted native engine. Controller and provider live beside this file."""
import json

from harness import budget as budgets, workspace
from harness.backend import ACTIVE, Backend as Base

# Every research job (a training run or a practice panel) runs on this one worker profile.
TRAIN_SHAPE = dict(gpu=1, cpu_cores=4, memory_mib=16384)
SPEC = dict(
    game='pokemon',
    # How `harness retrain` sets a replicate's training seed: the run seed, the config's `seed` field, or the
    # backend's own fresh host seeds for a final.
    retrain_seed='host',
    # What `ar` offers in this game: `ar run --seed`, `ar eval --seconds`, and the box's CPU torch version.
    run_seed=False, eval_seconds=True, box_torch='2.15.0',
    # The graded policy's size limit (parameters and buffers). The other games use 500,000; here a hashed
    # name-embedding table alone is 131k-524k elements, and under 500,000 the best recipe known reached only
    # +11.6 lift over the starter (headroom check, 2026-11-13), so this game's limit is 1,000,000.
    parameter_cap=1_000_000,
    venv='.venv-pokemon',
    constructor='simulators.pokemon.autoresearch.prepare:prepare',
    constructor_options={'base_release': 'an earlier sealed Pokémon release whose native runtime and evaluator inputs are reused '
                                         '(audited and every digest re-checked); needed when the local evidence they came from is gone',
                         'starter': 'ppo (default, the working clipped-PPO starter) or empty (boilerplate only: run plumbing, '
                                    'a learner with no update, a one-layer placeholder policy; its own brief)'},
    contract='simulators/pokemon/autoresearch/benchmark.json',
    mode='Gen 9 random battle 3v3 (Max Team Size = 3): admitted native C++ engine for training, pinned Showdown for evaluation',
    hardware='one L40S GPU with 4 physical CPU cores and 16 GiB per job (the Pokémon native training contract); '
             'public practice plays its games on a CPU worker with the same CPU/RAM cap and counts against the allocation per worker '
             'second at the same rate',
    # Seconds mode (this game's default): one job at a time within 14,400 worker seconds.
    budget_mode=budgets.SECONDS,
    research_seconds=14400,
    final_train_seconds=3600,
    smoke_research_seconds=600,
    smoke_final_train_seconds=240,
    # Dollar mode: the same allocation priced at the frozen list rate of the training worker ($11.61; smoke $0.48),
    # spent by parallel jobs within the dollar games' wall clock.
    research_usd=round(14400 * budgets.rate(TRAIN_SHAPE), 2),
    smoke_research_usd=round(600 * budgets.rate(TRAIN_SHAPE), 2),
    wall_seconds=18000,
    smoke_wall_seconds=4800,
)
RUN_SHAPE = 'run or practice (L40S, 4 CPU, 16 GiB)'
# One worker lease: the trusted supervisor accepts at most an hour of training.
MAX_RUN_SECONDS = 3600
# Private grading on a separate CPU-only worker: 64 fresh seeds x both seats x four opponents.
GRADING = dict(clusters=64, seconds=7200)
SMOKE_GRADING = dict(clusters=2, seconds=600)
PARAMETER_CAP = SPEC['parameter_cap']
# What `ar log` says about a training run whose trainer the worker's trusted supervisor ended.
PACKAGED = dict(
    stopped_early='Stopped on request: the trainer was ended and this run was packaged from the last complete policy '
                  'it had written. It is complete and can be practised and submitted.',
    cut_short='The trainer was still running at its stage time limit and was ended there: this run was packaged from '
              'the last complete policy it had written. It is complete and can be practised and submitted.')
NOT_PACKAGED = ('before it had written a complete checkpoint.pt, graph.json, result.json and progress.jsonl, so there '
                'was no policy to package. The run failed; its worker time stays charged.')


class Backend(Base):
    game = 'pokemon'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workspace = self.root / 'workspace'  # anchors the host workspace copy; Pokémon sends source directly
        self.controller = None

    def open(self):
        from .controller import Controller
        limits = None
        if not self.resume:
            grading = SMOKE_GRADING if self.smoke else GRADING
            limits = dict(budget_mode=self.budget_mode, research_shape=TRAIN_SHAPE,
                          max_lease_seconds=self.options.get('max_run_seconds', MAX_RUN_SECONDS),
                          grading_seconds=grading['seconds'], **self.limits)
            if self.budget_mode == budgets.USD:
                limits.update(research_usd=self.budget_usd, usd_per_second=budgets.rate(TRAIN_SHAPE),
                              prices=budgets.PRICE)
            else:
                limits['research_seconds'] = (self.budget_seconds
                                              or SPEC['smoke_research_seconds' if self.smoke else 'research_seconds'])
        self.root.mkdir(parents=True, exist_ok=self.resume)
        provider = self.options.get('_provider')  # tests stand in for Modal here
        if provider is None:
            from harness import policy
            from .provider import Provider
            policy.require_modal()
            provider = Provider(self.release, self.root)
        controller = Controller(self.release, self.root, provider, limits,
                                poll=self.options.get('poll_seconds', 3.0), final=self.final)
        if self.resume:
            # A reopened session keeps the mode its ledger was bound with; an amount of the other mode is refused.
            if (self.budget_usd is not None or self.budget_seconds is not None) \
                    and controller.ledger.mode != self.budget_mode:
                raise ValueError(f'This session was opened in {controller.ledger.mode} mode; it cannot be resumed '
                                 f'in {self.budget_mode} mode')
            self.budget_mode = controller.ledger.mode
            self.limits = budgets.limits(self.budget_mode)
        self.controller = controller
        # Whether this root is a fresh final training is the root's own state; a reopen reads it back.
        self.final = self.controller.final

    def settle(self):
        if self.controller is not None:
            self.controller.settle()

    def close(self):
        # Workers are finite sandboxes in the session's own app and Volume; nothing else is held open.
        self.settle()

    def _payload(self, config):
        """The researcher's current editable files as the native trainer takes them."""
        from simulators.pokemon.autoresearch.runtime.sdk import EDITABLE
        files = workspace.editable(self.release, self.workspace)
        return {name: files['training/' + name] for name in EDITABLE}, json.loads(files[config])

    def budget(self):
        from simulators.pokemon.autoresearch.runtime.practice import startup_summary
        from .controller import (MIN_PRACTICE_SECONDS, MIN_TRAIN_SECONDS, PRACTICE_SECONDS, PRACTICE_SEEDS,
                                 TRAINING_RESERVE_SECONDS)
        state = self.controller.ledger.snapshot()
        limits = state['limits']
        if state['budget_mode'] == budgets.USD:
            price, allocatable = limits['usd_per_second'], state['research_usd_remaining']
            reply = dict(budget_mode=budgets.USD, budget_usd=limits['research_usd'],
                         used_usd=round(state['research_usd_used_or_reserved'], 6),
                         remaining_usd=round(allocatable, 6), allocatable_usd=round(allocatable, 6),
                         unresolved_usd=round(state['unresolved_usd'], 6),
                         concurrency=limits['max_concurrent_jobs'], concurrent_gpus=limits['max_concurrent_gpus'],
                         concurrent_cpu_cores=limits['max_concurrent_cpu_cores'], jobs_at_once=state['jobs_at_once'],
                         charging='budget, used and remaining are US dollars. Every run and practice is one L40S '
                                  'worker with 4 CPU cores and 16 GiB, charged its seconds x $%.4f/h; queued and '
                                  'running work is reserved in full (its whole lease) until it ends. Each job takes '
                                  'one GPU, so at most %d run at once.' % (3600 * price, state['jobs_at_once']),
                         usd_per_hour={RUN_SHAPE: round(3600 * price, 4)},
                         affordable_seconds={RUN_SHAPE: int(allocatable / price + 1e-6)})
        else:
            reply = dict(budget_mode=budgets.SECONDS, budget_seconds=limits['research_seconds'],
                         used_seconds=state['research_seconds_used_or_reserved'],
                         remaining_seconds=state['research_seconds_remaining'],
                         unresolved_seconds=state['unresolved_seconds'], concurrency=1)
        return dict(reply, min_run_seconds=MIN_TRAIN_SECONDS, max_run_seconds=limits['max_lease_seconds'],
                    setup_and_publication_seconds_per_run=TRAINING_RESERVE_SECONDS,
                    min_practice_seconds=MIN_PRACTICE_SECONDS, default_practice_seconds=PRACTICE_SECONDS,
                    practice_public_seeds=len(PRACTICE_SEEDS), practice_counts_against_allocation=True,
                    parameter_cap=PARAMETER_CAP,
                    worker_startup=startup_summary([j for j in state['jobs'] if j['kind'] == 'research']))

    def status(self):
        controller = self.controller
        state = controller.ledger.snapshot()
        price = state['limits'].get('usd_per_second') if state['budget_mode'] == budgets.USD else None
        jobs = []
        for job in state['jobs']:
            if job['kind'] != 'research':
                continue  # the host's private grading worker is not the researcher's job
            record = controller.record(job['name']) or {}
            entry = dict(name=job['name'], kind=record.get('kind', 'train'), status=controller.state(job),
                         reserved_seconds=job['seconds'], charged_seconds=job['actual_seconds'])
            if price:
                entry.update(reserved_usd=round(job['seconds'] * price, 6),
                             charged_usd=None if job['actual_seconds'] is None else round(job['actual_seconds'] * price, 6))
            for key in ('run', 'bundle_sha256', 'error', 'score_valid', 'score_0_100', 'evaluation_status'):
                if record.get(key) is not None:
                    entry[key] = record[key]
            for flag in ('stopped_early', 'cut_short'):  # complete, on the last policy the ended trainer wrote
                if record.get(flag) and entry['status'] == 'complete':
                    entry[flag] = True
            jobs.append(entry)
        return dict(jobs=jobs, active=[job['name'] for job in jobs if job['status'] in ACTIVE],
                    submissions=[{k: v for k, v in row.items() if k != 'bundle'} for row in state['submissions']])

    def log(self, name, tail=80):
        from simulators.pokemon.autoresearch.runtime.recovery import training_outcome
        from .diagnostics import live_output, stage_diagnostics
        controller = self.controller
        job = next((job for job in self.status()['jobs'] if job['name'] == name), None)
        if job is None:
            raise ValueError('Unknown run')
        record = controller.record(name) or {}
        result = dict(name=name, kind=job['kind'], status=job['status'], job_error=record.get('error'))
        if job['status'] in ACTIVE:
            live = controller.live(name)
            if live is None:
                return dict(result, note='Output is collected when the worker stops.')
            if not live['available']:
                return dict(result, live=False, note=live['reason'] + '. Look again in a little while.')
            rows, text = live_output(live['text'], tail, cut=live.get('cut', False))
            return dict(result, live=True, printed_bytes=live['bytes'], read_seconds_ago=live['age_seconds'],
                        progress_samples=rows, trainer_log_tail=text,
                        note='What train.py has printed so far, read from the worker\'s storage: it trails the '
                             'trainer by up to a minute or so. progress_samples are the JSON rows it printed '
                             '(evenly sampled from the recent ones); full diagnostics follow when the worker stops.')
        output = controller.root / 'jobs' / record['operation'] / 'output'
        if job['kind'] == 'practice':
            keys = ('run', 'execution_success', 'panel_complete', 'score_valid', 'score_0_100', 'evaluation_status',
                    'scheduled_games', 'natural_games', 'faults', 'truncations', 'unstarted_games',
                    'fault_categories', 'candidate_binding_valid', 'note')
            result.update({key: record[key] for key in keys if key in record})
            if (output / 'summary.json').is_file():
                result['public_practice'] = json.loads((output / 'summary.json').read_bytes())
            return result
        if record.get('output'):
            measured = stage_diagnostics(controller.root, name, record, max(1, min(64, tail)))
            measured['own_log_tails'] = {key: '\n'.join(text.splitlines()[-tail:])
                                         for key, text in measured['own_log_tails'].items()}
            if controller.final:
                # A trainer may print its config; a fresh final's host seeds never leave the host's own files.
                measured['own_log_tails'] = 'Raw logs of a fresh final training stay with the host.'
            outcome = training_outcome(output, record.get('successful', False))
            result.update(measured, training_outcome=outcome)
            for flag, note in PACKAGED.items():
                if job.get(flag):
                    result.update({flag: True}, note=note)
            if job['status'] != 'complete' and outcome['termination_reason'] in ('stop_requested', 'hard_deadline'):
                ended = 'on the stop request' if outcome['termination_reason'] == 'stop_requested' else 'at its stage time limit'
                result['note'] = f'The trainer was ended {ended} {NOT_PACKAGED}'
        if job['kind'] == 'train' and job['status'] != 'complete' and record.get('stop_grace_expired_at'):
            from .controller import STOP_GRACE_SECONDS
            result['note'] = (f'The worker had not finished within {STOP_GRACE_SECONDS} seconds of the stop request and '
                              'was terminated; nothing was packaged.')
        result['practice'] = [dict(name=entry['name'], status=entry['status'], score_0_100=entry.get('score_0_100'))
                              for entry in self.status()['jobs'] if entry.get('run') == name]
        return result

    def check(self, config):
        return self.controller.check(*self._payload(config))

    def run(self, name, seconds, config, seed=1):
        if seed != 1:
            raise ValueError('Pokémon seeds come from the `seed` and `training_seed_start` fields of the config; '
                             'edit the config instead of passing --seed')
        return self.controller.run(name, seconds, *self._payload(config))

    def stop(self, name):
        return self.controller.stop(name)

    def evaluate(self, name, seconds=None):
        return self.controller.evaluate(name, seconds)

    def submit(self, name):
        record = self.controller.submit(name)
        return {key: value for key, value in record.items() if key != 'bundle'}

    def finalize(self):
        selected = self.controller.finalize()
        return dict(finalized=True, final_submission=selected['id'], job=selected['job'],
                    bundle_sha256=selected['bundle_sha256'])

    def tick(self):
        self.controller.tick()

    def grade(self):
        from simulators.pokemon.autoresearch.prepare import audit
        grading = SMOKE_GRADING if self.smoke else GRADING
        record = self.controller.grade(grading['clusters'], grading['seconds'])
        keys = ('execution_success', 'panel_complete', 'score_valid', 'score_0_100', 'evaluation_status',
                'scheduled_games', 'natural_games', 'faults', 'truncations', 'unstarted_games', 'fault_categories',
                'candidate_identity', 'candidate_binding_valid', 'bundle_sha256', 'private_bank_commitment',
                'profile_verified')
        return dict({key: record.get(key) for key in keys}, raw_score_0_100=record.get('score_0_100'),
                    official=bool(self.official), panel='smoke' if self.smoke else 'grade',
                    seed_clusters=grading['clusters'], release_sha256=audit(self.release)['release_sha256'],
                    normalized_reference_score=None, calibration_qualified=False)
