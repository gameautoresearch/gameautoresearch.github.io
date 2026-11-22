"""Pokémon native backend: real constructor, controller, ledger, bundles and transport; no cloud, no model.

The provider is stubbed before any allocation, or the real provider runs against
a stand-in for the Modal SDK. The native engine itself is exercised offline by
simulators/pokemon/autoresearch/runtime/tests.

Run with the game's research interpreter: .venv-pokemon/bin/python -m unittest harness.tests.test_pokemon
"""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from harness import budget as budgets, policy, registry, session, task, workspace
from harness.transport import LocalBox
from harness.broker import Broker
from harness.tests.support import Process, Researcher

try:
    import torch
    import harness.games.pokemon as game
    from harness.games.pokemon import controller as control, provider as transport
    from simulators.pokemon.autoresearch import prepare as constructor
    from simulators.pokemon.autoresearch.contract import ROOT, sha
    from simulators.pokemon.autoresearch.ledger import Ledger
    from simulators.pokemon.autoresearch.runtime import bootstrap, bundle, graph_export, sdk, worker
    from simulators.pokemon.autoresearch.runtime.starter import policy as starter_policy
except ImportError as exc:  # wrong interpreter
    raise unittest.SkipTest(f'Pokémon research dependencies unavailable: {exc}')

SENTINEL = '\n# researcher change\n'
RETAINED = constructor.NATIVE.is_dir() and constructor.EVALUATOR.is_dir()
# Without that local evidence, a sealed release kept in this checkout supplies the same inputs (`base_release`).
BASE = ROOT / 'evidence/validity-20261112/releases/pokemon-p1'
REUSED = BASE if not RETAINED and (BASE / 'release.json').is_file() else None
DRY_RUN = ROOT / 'simulators/pokemon/autoresearch/task/dryrun.py'


class Provider:
    """Stands in for the cloud: records the exact payload and settles as the provider's receipts would."""

    def __init__(self, root, seconds=100, outcome='complete'):
        self.root, self.seconds, self.outcome = Path(root).resolve(), seconds, outcome
        self.launches, self.jobs, self.stopped, self.profiles = [], {}, set(), []
        self.asked, self.receipt = set(), {}  # stop requests delivered; what the worker's supervisor reports
        self.printed, self.peeks = None, 0  # what a running trainer has printed so far
        self.entered, self.create, self.finish = threading.Event(), threading.Event(), threading.Event()
        self.create.set()
        self.finish.set()

    def hold(self):
        self.create.clear()
        self.finish.clear()

    def launch(self, ledger, operation, kind, spec, files, seconds, stopped=None):
        self.launches.append(dict(kind=kind, spec=spec, seconds=seconds, names=sorted(name for _, name in files),
                                  candidate={name: Path(source).read_bytes() for source, name in files
                                             if name.startswith('candidate/')}))
        self.entered.set()
        self.create.wait(10)
        if stopped and stopped():
            ledger.cancel_unattached(operation, 'no provider create attempted: Stopped')
            raise transport.Stopped('Stopped before a worker was created')
        if self.outcome == 'create-interrupted':
            ledger.unknown(operation, dict(reason='Provider create interrupted/rejected; reconcile exact operation'))
            raise RuntimeError('stub: provider create interrupted')
        self.jobs[operation] = (kind, spec, {name: Path(source) for source, name in files})
        # As the real provider records a created worker; the live log is read only once this exists.
        transport.private_write(self.root / 'jobs' / operation / 'provider.json',
                                dict(operation=operation, sandbox_id='sb-' + operation[:8]))
        ledger.attach(operation, 'sb-' + operation[:8])

    def stop(self, operation):
        self.stopped.add(operation)
        self.finish.set()

    def request_stop(self, operation):
        """The worker's supervisor ends the trainer and seals what it had written; the worker then finishes."""
        self.asked.add(operation)
        self.receipt = dict(stopped_early=True, termination_reason='stop_requested')
        self.finish.set()
        return True

    def peek(self, operation, limit=64 * 1024):
        self.peeks += 1
        if self.printed is None:
            raise FileNotFoundError(operation)
        text = self.printed[-limit:]
        return dict(bytes=len(self.printed), cut=len(self.printed) > len(text), text=text)

    def collect(self, ledger, operation):
        if not self.finish.is_set():
            return dict(status='active-or-unknown', operation=operation)
        kind, spec, files = self.jobs[operation]
        receipt = dict(provider_id='sb-' + operation[:8], created_at=1000.0, started_at=1010.0,
                       finished_at=1010.0 + self.seconds, gpu_type='L40S' if kind == 'research' else '')
        ledger.settle(operation, self.seconds, dict(receipt, provider_verified_stopped=True))
        output = self.root / 'jobs' / operation / 'output'
        output.mkdir(parents=True, exist_ok=True)
        if operation in self.stopped or self.outcome == 'failed':
            (output / 'worker-error.log').write_text('stub: the worker did not finish its work\n')
            return dict(status='settled', operation=operation, output=str(output), provider=receipt, successful=False)
        if spec['action'] == 'train':
            self.stage(output / 'stage', files['candidate/policy.py'].parent, spec['config'], **self.receipt)
        else:
            request = spec['assessment']
            games = 2 * len(request['seeds']) * len(request['panel'])
            (output / 'summary.json').write_text(json.dumps(dict(
                status='complete', scheduled_games=games, natural_games=games, faults=0, raw_score_0_100=62.5,
                candidate_identity={k: v for k, v in request['candidate'].items() if k != 'path'})))
        (output / 'completed.json').write_text(json.dumps(dict(status='complete', body_seconds=self.seconds - 12)))
        return dict(status='settled', operation=operation, output=str(output), provider=receipt, successful=True)

    @staticmethod
    def stage(stage, source, config, **receipt):
        """What a successful training worker leaves: an exported, sealed, source-bound policy.

        `receipt` is what its supervisor adds when it ended the trainer itself (`stopped_early`, `cut_short`).
        """
        stage.mkdir()
        model = starter_policy.create(config['network']).eval()
        graph_export.export(model, stage / 'graph.json')
        torch.save(dict(model=model.state_dict(), iteration=1, benchmark_id=sdk.BENCHMARK_ID,
                        source_sha256=sdk.source_signature(source), runtime_sha256=sdk.runtime_signature(),
                        config_sha256=sdk.identity(config)), stage / 'checkpoint.pt')
        bundle.seal(stage / 'bundle', source, config, stage / 'checkpoint.pt', {'stub': True})
        (stage / 'result.json').write_text(json.dumps(dict(iterations=1, trained_samples=64, optimizer_updates=2)))
        (stage / 'progress.jsonl').write_text(json.dumps(dict(iteration=1, samples=64, seconds=1.5)) + '\n')
        (stage / 'supervisor.json').write_text(json.dumps(dict(dict(termination_reason='completed', phase='complete'),
                                                               **receipt)))
        (stage / 'trainer.log').write_text('trained\n')

    def verify_profile(self, operation, gpu):
        self.profiles.append((self.jobs[operation][0], gpu))


class Parallel(Provider):
    """Workers that run at once, each until it is let finish (`released`), asked to stop or terminated."""

    def __init__(self, root, seconds=100):
        super().__init__(root, seconds)
        self.released, self.serial = set(), threading.Lock()

    def request_stop(self, operation):
        self.asked.add(operation)
        return True

    def collect(self, ledger, operation):
        if operation not in self.released | self.stopped | self.asked:
            return dict(status='active-or-unknown', operation=operation)
        with self.serial:  # the stub's supervisor receipt is one attribute: one worker publishes at a time
            self.receipt = dict(stopped_early=True, termination_reason='stop_requested') if operation in self.asked else {}
            return super().collect(ledger, operation)


def until(condition, seconds=30):
    deadline = time.time() + seconds
    while not condition() and time.time() < deadline:
        time.sleep(0.01)
    return condition()


def open_backend(case, release, root, provider, **kwargs):
    backend = registry.backend('pokemon')(release, root, smoke=True,
                                          options={'_provider': provider, 'poll_seconds': 0.01}, **kwargs)
    backend.open()
    case.addCleanup(backend.close)
    return backend


def join(backend):
    backend.tick()
    backend.controller.thread.join(timeout=30)


class Cheap(unittest.TestCase):
    def test_importing_the_game_module_needs_no_game_or_cloud_dependency(self):
        code = ("import sys, harness.games.pokemon as game\n"
                "assert not {'torch', 'modal', 'numpy'} & set(sys.modules), sorted(sys.modules)\n"
                "print(game.SPEC['research_seconds'], game.SPEC['smoke_final_train_seconds'])")
        done = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual((done.returncode, done.stdout.split()), (0, ['14400', '240']), done.stderr[-2000:])
        spec = registry.spec('pokemon')
        self.assertIn('L40S', spec['hardware'])
        self.assertEqual(registry.constructor('pokemon'), constructor.prepare)
        self.assertGreaterEqual(spec['smoke_final_train_seconds'], control.MIN_TRAIN_SECONDS)

    def test_both_budget_modes_have_defaults_and_seconds_stays_the_default(self):
        spec = registry.spec('pokemon')
        rate = budgets.rate(game.TRAIN_SHAPE)
        self.assertEqual(game.TRAIN_SHAPE, transport.SHAPE)  # what a worker is created with is what is priced
        self.assertEqual((spec['budget_mode'], spec['research_seconds'], spec['smoke_research_seconds']),
                         ('seconds', 14400, 600))
        self.assertEqual((spec['research_usd'], spec['smoke_research_usd'], spec['wall_seconds'],
                          spec['smoke_wall_seconds']), (round(14400 * rate, 2), round(600 * rate, 2), 18000, 4800))
        self.assertEqual((spec['research_usd'], spec['smoke_research_usd']), (11.61, 0.48))
        self.assertEqual(registry.research('pokemon'), dict(mode='seconds', seconds=14400.0, usd=None))
        self.assertEqual(registry.research('pokemon', budget_mode='usd', smoke=True), dict(mode='usd', seconds=None, usd=0.48))
        for words in ('at a time', '$', 'one job'):
            self.assertNotIn(words, spec['hardware'])
        contract = json.loads((ROOT / 'simulators/pokemon/autoresearch/benchmark.json').read_text())['budget']
        self.assertEqual((contract['default_budget_mode'], contract['budget_modes']['usd']['research_usd'],
                          contract['budget_modes']['usd']['smoke_research_usd'],
                          contract['budget_modes']['usd']['usd_per_worker_second']),
                         (spec['budget_mode'], spec['research_usd'], spec['smoke_research_usd'], rate))
        brief = (ROOT / 'simulators/pokemon/autoresearch/task/brief.md').read_text()
        for words in ('$', 'one run at a time', 'one job at a time', 'One job'):
            self.assertNotIn(words, brief)

    def test_a_dollar_ledger_never_grants_past_its_budget_or_its_jobs_at_once(self):
        rate = budgets.rate(game.TRAIN_SHAPE)
        with tempfile.TemporaryDirectory() as temp:
            ledger = Ledger(Path(temp) / 'usd.sqlite3', dict(
                budget_mode='usd', research_usd=0.5, usd_per_second=rate, max_lease_seconds=3600, grading_seconds=600,
                research_shape=game.TRAIN_SHAPE, prices=budgets.PRICE, **budgets.PARALLEL))
            self.assertEqual((ledger.mode, ledger.capacity), ('usd', 2))  # four jobs, but each takes one of two GPUs
            first = ledger.reserve('a', 'research', 300)
            ledger.reserve('b', 'research', 300)
            with self.assertRaisesRegex(ValueError, 'At most 2 jobs at once'):
                ledger.reserve('c', 'research', 60)
            ledger.attach(first, 'sb-a')
            ledger.settle(first, 50, dict(provider_verified_stopped=True))
            with self.assertRaisesRegex(ValueError, 'remaining allocation'):  # 0.5 - 350 s held = 270 s left
                ledger.reserve('c', 'research', 271)
            ledger.reserve('c', 'research', 270)
            state = ledger.snapshot()
            self.assertAlmostEqual(state['research_usd_used_or_reserved'], 620 * rate)
            self.assertLessEqual(state['research_usd_used_or_reserved'], 0.5)
            self.assertAlmostEqual(state['unresolved_usd'], 570 * rate)
            self.assertNotIn('research_seconds_remaining', state)
            reopened = Ledger(Path(temp) / 'usd.sqlite3')
            self.assertEqual((reopened.mode, reopened.capacity, reopened.limits['research_usd']), ('usd', 2, 0.5))
            # Seconds mode: one worker at a time, whatever is still held.
            seconds = Ledger(Path(temp) / 'seconds.sqlite3', dict(
                budget_mode='seconds', research_seconds=600, max_lease_seconds=3600, grading_seconds=600,
                research_shape=game.TRAIN_SHAPE, **budgets.LINEAR))
            self.assertEqual((seconds.mode, seconds.capacity), ('seconds', 1))
            seconds.reserve('a', 'research', 240)
            with self.assertRaisesRegex(ValueError, 'blocks dispatch'):
                seconds.reserve('b', 'research', 240)
            # A ledger bound before the modes existed is a seconds-mode ledger.
            legacy = Ledger(Path(temp) / 'legacy.sqlite3', dict(research_seconds=600, max_lease_seconds=3600,
                                                                  grading_seconds=600))
            self.assertEqual((legacy.mode, legacy.capacity, 'budget_mode' in legacy.limits), ('seconds', 1, False))


class DryRun(unittest.TestCase):
    """`ar check`'s CPU dry run and the live log reader, on this checkout's own starter: no release, no cloud."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.box = Path(self.temp.name) / 'workspace'
        research = ROOT / 'simulators/pokemon/autoresearch'
        self.starter = {'training/' + name: (research / 'runtime/starter' / name).read_text()
                        for name in ('policy.py', 'learner.py', 'train.py')}
        self.starter.update({'configs/train.json': (research / 'recipes/starter.json').read_text(),
                             'configs/smoke.json': (research / 'recipes/smoke.json').read_text()})
        for name, text in self.starter.items():
            self.edit(name, text)
        # The box holds public simulator source under its repository names; the dry run stages only from there.
        shown = workspace.source_files('pokemon')
        namespace = {}
        exec(compile(DRY_RUN.read_text().split('# The identity and file helpers')[0], str(DRY_RUN), 'exec'), namespace)
        for origin in namespace['PUBLIC'].values():
            self.assertIn(origin, shown)
            target = self.box / 'simulator' / origin
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(shown[origin].read_bytes())

    def edit(self, name, text):
        path = self.box / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def dry(self, config='configs/smoke.json'):
        done = subprocess.run([sys.executable, '-I', str(DRY_RUN), '--workspace', str(self.box), '--config', config],
                              capture_output=True, text=True, timeout=300)
        return json.loads(done.stdout)

    def test_dry_run_passes_the_starter_and_catches_what_a_worker_would_refuse(self):
        passed = self.dry('configs/train.json')
        self.assertTrue(passed['passed'], passed)
        self.assertEqual(passed['reduced_for_cpu'], dict(environments=8, rollout_steps=8))
        self.assertEqual(passed['progress_rows'], 1)
        model = starter_policy.create(json.loads(self.starter['configs/train.json'])['network'])
        self.assertEqual(passed['parameter_and_buffer_elements'], sum(v.numel() for v in model.state_dict().values()))
        learner = self.starter['training/learner.py']
        self.assertEqual(learner.count('self.faint_reward=0;'), 1)
        self.edit('training/learner.py', learner.replace('self.faint_reward=0;', 'self.faint_reward=.1;'))
        failed = self.dry()
        self.assertEqual((failed['passed'], failed['stage']), (False, 'train'))
        self.assertIn('Frozen terminal rewards only', failed['error'])
        self.assertIn('training/learner.py', failed['error'])
        self.assertNotIn(self.temp.name, failed['error'])  # paths are the researcher's own names
        # A learner that answers for a seat that is not to move, as a worker's engine would fault.
        self.edit('training/learner.py', learner.replace('torch.where(ready,actions,-1)', 'actions'))
        self.assertIn('not to move', self.dry()['error'])
        self.edit('training/learner.py', learner)
        policy_source = self.starter['training/policy.py']
        self.assertEqual(policy_source.count('        return logits.masked_fill'), 1)
        self.edit('training/policy.py', policy_source.replace(
            '        return logits.masked_fill',
            '        logits = logits + torch.rand_like(logits)\n        return logits.masked_fill'))
        self.assertIn('GraphError', self.dry()['error'])  # randomness in the scored forward cannot be exported
        self.edit('training/policy.py', policy_source)
        self.edit('configs/smoke.json', json.dumps(dict(json.loads(self.starter['configs/smoke.json']),
                                                        network=dict(hidden=512, embedding=512))))
        self.assertIn('State exceeds ten million', self.dry()['error'])  # the worker's own export refuses it
        self.edit('configs/smoke.json', '[]')
        self.assertIn('JSON object', self.dry()['error'])

    def test_empty_starter_passes_the_dry_run_without_learning(self):
        from simulators.pokemon.autoresearch.prepare import STARTERS
        research = ROOT / 'simulators/pokemon/autoresearch'
        empty = STARTERS['empty']
        for name in ('policy.py', 'learner.py', 'train.py'):
            self.edit('training/' + name, (research / 'runtime' / empty['source'] / name).read_text())
        for config, native in (('configs/train.json', 'starter'), ('configs/smoke.json', 'smoke')):
            self.edit(config, (research / 'recipes' / (empty['recipes'][native] + '.json')).read_text())
            passed = self.dry(config)
            self.assertTrue(passed['passed'], passed)
            self.assertEqual(passed['last_progress_row']['optimizer_updates'], 0)  # boilerplate: nothing learns
        self.assertNotIn('PPO', (research / 'task' / empty['brief']).read_text().split('**Keep a complete')[0]
                         .split('**Starter')[1])

    def test_empty_starter_box_holds_no_reference_learning_code_and_still_dry_runs(self):
        import re
        research = ROOT / 'simulators/pokemon/autoresearch'
        empty = constructor.STARTERS['empty']
        public = Path(self.temp.name) / 'release/public'
        for name in sdk.EDITABLE:
            target = public / 'agent' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((research / 'runtime' / empty['source'] / name).read_text())
        for name, native in empty['recipes'].items():
            target = public / 'recipes' / f'{name}.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((research / 'recipes' / f'{native}.json').read_text())
        record = workspace.stage('pokemon', public, withhold=empty['withhold'],
                                 views={name: make() for name, make in empty['views'].items()})
        box = public / workspace.WORKSPACE
        shown = {p.relative_to(box / 'simulator').as_posix(): p for p in (box / 'simulator').rglob('*') if p.is_file()}
        pattern = re.compile(constructor.LEARNING_CODE)
        self.assertEqual([name for name, path in shown.items() if pattern.search(path.read_text(errors='replace'))], [])
        for name in ('simulators/pokemon/native/resident_ppo.py', 'simulators/pokemon/native/train.py'):
            self.assertNotIn(name, shown)
            self.assertIn(name, record['withheld'])
        view = shown[constructor.NETWORK].read_text()
        self.assertIn('KEYS = ', view)
        self.assertNotIn('class Policy', view)
        self.assertEqual(record['views'], {constructor.NETWORK: sha(shown[constructor.NETWORK])})
        for name in ('simulators/pokemon/efficient.py', 'simulators/pokemon/autoresearch/runtime/graph_export.py',
                     'simulators/pokemon/native/engine.cpp'):
            self.assertIn(name, shown)  # the engine and the runtime interface stay readable
        # The box as staged, not this checkout, is what the dry run imports from.
        done = subprocess.run([sys.executable, '-I', str(DRY_RUN), '--workspace', str(box), '--config', 'configs/smoke.json'],
                              capture_output=True, text=True, timeout=300)
        passed = json.loads(done.stdout)
        self.assertTrue(passed['passed'], passed)

    def test_empty_starter_recipes_may_import_only_the_runtime_interface(self):
        research = ROOT / 'simulators/pokemon/autoresearch'
        empty = constructor.STARTERS['empty']
        source = {name: (research / 'runtime' / empty['source'] / name).read_text() for name in sdk.EDITABLE}
        config = json.loads((research / 'recipes' / 'empty.json').read_text())
        control.inspect_source(source, config, 'empty')  # the boilerplate itself is admitted
        for line, refused in (('from integrations.pokemon.network import Policy', 'only KEYS'),
                              ('import integrations.pokemon.network', 'not available'),
                              ('from integrations.pokemon.native.resident_ppo import decision_gae', 'not available'),
                              ('import integrations.pokemon.native.resident_ppo as ppo', 'not available'),
                              ('from integrations.pokemon.autoresearch.runtime.starter import learner', 'not available'),
                              ('from integrations.pokemon import network', 'not available'),
                              ('import importlib', 'dynamic imports'),
                              ('x = __import__("integrations.pokemon.rl")', 'dynamic imports')):
            changed = dict(source, **{'learner.py': line + '\n' + source['learner.py']})
            with self.assertRaisesRegex(ValueError, refused):
                control.inspect_source(changed, config, 'empty')
            if 'dynamic' not in refused:
                control.inspect_source(changed, config, 'ppo')  # the PPO starter's task is unchanged
        allowed = ('from integrations.pokemon.network import KEYS, batch\nfrom . import policy as local\n'
                   'from integrations.pokemon.autoresearch.runtime import sdk, graph_export\n'
                   'from integrations.pokemon import efficient\n')
        control.inspect_source(dict(source, **{'learner.py': allowed + source['learner.py']}), config, 'empty')
        for starter in ('ppo', 'empty'):  # `from` imports of encoded-state modules are refused like plain ones
            with self.assertRaisesRegex(ValueError, 'encoded-state'):
                control.inspect_source(dict(source, **{'learner.py': 'from pickle import loads\n' + source['learner.py']}),
                                       config, starter)

    def test_empty_starter_workers_import_a_network_module_without_the_reference_policy(self):
        frozen = Path(self.temp.name) / 'frozen'
        frozen.mkdir()
        _, manifest = bootstrap.source_snapshot(frozen)
        bootstrap.packed_source(frozen, manifest)
        reduced, pack = constructor.reduce_public_network(frozen, manifest)
        public = frozen / 'public-source' / constructor.PUBLIC_NETWORK
        self.assertNotIn('class Policy', public.read_text())
        self.assertEqual(public.read_text(), constructor.network_view())  # what the box shows is what workers import
        # The protected tree keeps the native-certified module byte for byte; only the public import copy changes.
        self.assertEqual(sha(frozen / 'source' / constructor.PUBLIC_NETWORK), sha(ROOT / constructor.NETWORK))
        self.assertEqual(reduced['files'], manifest['files'])
        self.assertEqual(reduced['public_files'][constructor.PUBLIC_NETWORK], sha(public))
        self.assertEqual(json.loads((frozen / 'source-manifest.json').read_text()), reduced)
        with tempfile.TemporaryDirectory() as temp:  # the rebuilt pack unpacks under the worker's own checks
            bootstrap.unpack_source_pack(pack, temp, reduced, sha(pack))
        namespace = {}
        exec(compile(public.read_text(), str(public), 'exec'), namespace)
        self.assertEqual(namespace['KEYS'][0], 'tokens')
        self.assertNotIn('Policy', namespace)

    def test_a_release_with_an_unknown_starter_is_refused(self):
        release = Path(self.temp.name) / 'release'
        release.mkdir()
        (release / 'release.json').write_text(json.dumps(dict(starter='mystery')))
        with self.assertRaisesRegex(ValueError, 'Unknown release starter'):
            control.Controller(release, Path(self.temp.name) / 'root', None)

    def test_dry_run_rejects_a_policy_over_the_parameter_cap_naming_the_count_and_the_cap(self):
        from harness.games import pokemon as game
        smoke = json.loads(self.starter['configs/smoke.json'])

        def elements(network):
            return sum(v.numel() for v in starter_policy.create(network).state_dict().values())
        over = dict(hidden=32, embedding=58)   # the graph exporter admits it (under ten million); the cap does not
        under = dict(hidden=32, embedding=57)
        self.assertTrue(elements(under) <= 1_000_000 < elements(over), (elements(under), elements(over)))
        self.edit('configs/smoke.json', json.dumps(dict(smoke, network=over)))
        failed = self.dry()
        self.assertEqual((failed['passed'], failed['stage']), (False, 'outputs'), failed)
        self.assertEqual(failed['error'], f'ValueError: {elements(over):,} parameter and buffer elements exceed the 1,000,000 cap')
        self.edit('configs/smoke.json', json.dumps(dict(smoke, network=under)))
        passed = self.dry()
        self.assertEqual((passed['passed'], passed['parameter_and_buffer_elements']), (True, elements(under)), passed)
        # The dry run, the backend and the sealer state one cap.
        namespace = {}
        exec(compile(DRY_RUN.read_text().split('# The identity and file helpers')[0], str(DRY_RUN), 'exec'), namespace)
        self.assertEqual((namespace['PARAMETER_CAP'], game.PARAMETER_CAP, game.SPEC['parameter_cap']), (1_000_000,) * 3)
        self.assertIn('> 1_000_000', (ROOT / 'simulators/pokemon/autoresearch/runtime/bundle.py').read_text())

    def test_starter_keeps_a_complete_policy_on_disk_from_its_first_update(self):
        text = self.starter['training/train.py']
        early = text.index("export(copy.deepcopy(model),out/'graph.json')")
        loop = text.index('while time.monotonic()<stop')
        self.assertLess(early, loop)  # the graph is written before the first update
        body = text[loop:text.index('result=summary()')]
        self.assertLess(body.index("replace(out/'checkpoint.pt')"), body.index("write(out/'result.json',summary())"))
        self.assertIn("export(model,out/'graph.tmp');(out/'graph.tmp').replace(out/'graph.json')", text[loop:])

    def test_starter_prints_each_progress_row_for_the_live_log(self):
        self.assertIn('print(json.dumps(row),flush=True)', self.starter['training/train.py'])

    def test_live_log_reads_only_a_created_training_worker_and_never_a_final_one(self):
        root = Path(self.temp.name) / 'research'
        root.mkdir()
        provider = Provider(root)
        controller = control.Controller(self.temp.name, root, provider,
                                        dict(research_seconds=600, max_lease_seconds=3600, grading_seconds=600))
        operation = 'ab' * 16
        self.assertIsNone(controller.live('missing'))
        controller.update('a', name='a', kind='train', operation=operation, status='reserved')
        controller.update('eval-00000000', name='eval-00000000', kind='practice', operation='cd' * 16)
        self.assertIsNone(controller.live('eval-00000000'))
        self.assertEqual((controller.live('a')['available'], provider.peeks), (False, 0))  # no worker: nothing is read
        (root / 'jobs' / operation).mkdir(parents=True)
        (root / 'jobs' / operation / 'provider.json').write_text('{}')
        self.assertIn('not started', controller.live('a')['reason'])
        rows = [dict(iteration=i, samples=64, seconds=1.5, policy_loss=1 / i, seed=7, note='x') for i in range(1, 41)]
        provider.printed = 'starting\n' + ''.join(json.dumps(row) + '\n' for row in rows) + 'Traceback: boom\n'
        self.assertFalse(controller.live('a')['available'])  # the earlier look is reused for a few seconds
        controller.peeked.clear()
        backend = registry.backend('pokemon')(self.temp.name, Path(self.temp.name) / 'unopened')
        backend.controller = controller
        running = dict(jobs=[dict(name='a', kind='train', status='running')], active=['a'])
        with mock.patch.object(backend, 'status', lambda: running):
            live = backend.log('a', tail=5)
            peeks = provider.peeks
            backend.log('a')
        self.assertEqual(provider.peeks, peeks)
        self.assertEqual((live['live'], live['status']), (True, 'running'))
        self.assertEqual(live['trainer_log_tail'], 'starting\nTraceback: boom')  # metric rows are not repeated as text
        self.assertEqual(len(live['progress_samples']), 12)
        self.assertEqual(live['progress_samples'][0], dict(iteration=1, samples=64, seconds=1.5, policy_loss=1.0))
        self.assertEqual(live['progress_samples'][-1]['iteration'], 40)  # measured scalars only, newest kept
        self.assertEqual(live['printed_bytes'], len(provider.printed))
        # Only the end of a long log is read; its first, partial line is dropped.
        from harness.games.pokemon.diagnostics import live_output
        self.assertEqual(live_output('eration": 3}\n{"iteration": 4}\nlast', cut=True), ([dict(iteration=4)], 'last'))
        provider.printed = None
        controller.peeked.clear()
        provider.peek = mock.Mock(side_effect=RuntimeError('volume unavailable'))
        self.assertIn('volume unavailable', controller.live('a')['reason'])
        controller.final = True
        self.assertIsNone(controller.live('a'))  # a fresh final's output could echo its host seeds


@unittest.skipUnless(RETAINED or REUSED, 'neither the retained native admission and evaluator inputs nor a sealed '
                                        'release to reuse them from is in this checkout')
class Native(unittest.TestCase):
    """The real constructor, then sessions on its release."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.release = Path(cls.temp.name).resolve() / 'release'
        cls.manifest = constructor.prepare(cls.release, base_release=REUSED)
        policy.bind_release('pokemon', cls.release)
        cls.meta = workspace.manifest(cls.release)
        base = workspace.public(cls.release) / workspace.WORKSPACE
        cls.starter = {name: (base / name).read_text() for name in cls.meta['writable']}

    @classmethod
    def tearDownClass(cls):
        for path in Path(cls.temp.name).rglob('*'):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)
        cls.temp.cleanup()

    def setUp(self):
        self.out = Path(tempfile.mkdtemp(dir=self.temp.name))
        self.root = self.out / 'research'
        self.provider = Provider(self.root)
        self.files = dict(self.starter)

    def session(self, **kwargs):
        self.backend = open_backend(self, self.release, self.root, self.provider, **kwargs)
        self.broker = Broker(self.backend, self.out, 'network', research_seconds=600)
        return self.broker

    def run_request(self, name='a', seconds=240, **extra):
        return self.broker.handle(dict({'op': 'run', 'name': name, 'seconds': seconds, 'files': self.files}, **extra))

    def jobs(self):
        status = self.broker.handle({'op': 'status'})
        return [(job['name'], job['kind'], job['status']) for job in status['jobs']], status['active']

    def spent(self):
        budget = self.backend.budget()
        return budget['used_seconds'], budget['remaining_seconds']

    # Constructor -----------------------------------------------------------------------------------------
    def test_constructor_freezes_the_native_runtime_and_stages_only_the_workspace(self):
        release = self.release
        self.assertEqual(constructor.audit(release)['release_sha256'], self.manifest['release_sha256'])
        policy.verify_release('pokemon', release)
        self.assertEqual({path.name for path in release.iterdir()}, {'frozen', 'private', 'public', 'release.json'})
        self.assertEqual({path.name for path in (release / 'public').iterdir()},
                         {'agent', 'recipes', 'agent_workspace', 'workspace_manifest.json', 'benchmark.json',
                          'CONTRACT.md', 'GRAPH_CONTRACT.md'})  # nothing for an older harness
        base = release / 'public/agent_workspace'
        self.assertEqual({path.name for path in base.iterdir()}, set(workspace.TOP))
        for name in sdk.EDITABLE:
            self.assertEqual((base / 'training' / name).read_bytes(), (sdk.STARTER / name).read_bytes())
        for staged, native in (('train', 'starter'), ('smoke', 'smoke')):
            self.assertEqual((base / f'configs/{staged}.json').read_bytes(),
                             (constructor.HERE / f'recipes/{native}.json').read_bytes())
        self.assertFalse(any('private' in Path(name).parts or name.endswith(('.pt', '.bin', '.whl'))
                             for name in self.meta['files']))
        self.assertEqual((self.manifest['hardware'], self.manifest['image_id'], self.manifest['private_seeds_created']),
                         (dict(gpu='L40S', physical_cpu_cores=4, memory_gib=16), bootstrap.IMAGE, False))
        # The worker source tree: the runtime identity, the native-certified files and the qualified evaluator.
        frozen = release / 'frozen'
        source = frozen / 'source/integrations/pokemon'
        self.assertEqual(self.manifest['protected_runtime_sha256'], sdk.runtime_signature())
        self.assertEqual(sdk.identity({name: sha(source / 'autoresearch/runtime' / name) for name in sdk.PROTECTED}),
                         sdk.runtime_signature())
        certificate = json.loads((frozen / 'runtime/native-admission.json').read_bytes())
        for name, digest in certificate['external_runtime_sha256'].items():
            self.assertEqual(sha(source / name), digest, name)
        registry_ = json.loads((release / 'private/assessment/registry.json').read_bytes())
        request = registry_['evaluator_request']
        for folder, key in (('evaluation_v2', 'qualified_core_file_sha256'), ('strong_bots', 'search_extension_file_sha256')):
            for name, digest in request[key].items():
                self.assertEqual(sha(source / 'autoresearch' / folder / name), digest, name)
        self.assertEqual([entry['kind'] for entry in registry_['panel']],
                         ['power', 'checkpoint', 'checkpoint', 'search'])  # 3v3: the two retrained learned opponents
        self.assertFalse(any(str(value).startswith('/') for value in request.values() if isinstance(value, str)))
        sealed = json.loads((frozen / 'source-manifest.json').read_bytes())
        self.assertEqual(set(sealed['public_files']), set(bootstrap.PUBLIC_SOURCE_FILES))
        self.assertNotIn('ROOT', (frozen / 'public-source/integrations/pokemon/autoresearch/contract.py').read_text())
        with tempfile.TemporaryDirectory() as temp:  # the pack every worker verifies and unpacks
            proof = bootstrap.unpack_source_pack(frozen / 'source-pack.tar', temp, sealed, self.manifest['source_pack_sha256'])
            self.assertEqual(proof['files'], len(sealed['files']) + len(sealed['public_files']))
        key = release / 'private/env-state-key.bin'
        self.assertEqual((key.stat().st_size, key.stat().st_mode & 0o077), (32, 0))
        self.assertEqual({path.name for path in (release / 'private').iterdir()}, {'assessment', 'env-state-key.bin'})
        request, files = transport.assessment_files(release)
        self.assertTrue(all(Path(source).is_file() for source, _ in files))
        self.assertTrue(all(Path(source).is_file() for source, _ in transport.runtime_files(release)))
        with self.assertRaises(FileExistsError):
            constructor.prepare(release)

    def test_frozen_source_trees_import_on_their_own(self):
        """A worker has only these trees: no harness, no tools, no repository checkout."""
        frozen = self.release / 'frozen'
        runtime = 'integrations.pokemon.autoresearch.runtime.'
        protected = ('import sys\n' + ''.join(f'import {runtime}{name}\n' for name in (
            'worker', 'supervisor', 'env_service', 'fastenv', 'arena', 'practice', 'qualification', 'policy_process',
            'graph_worker', 'bundle', 'bootstrap')) + f'import {runtime}worker as module\n'
            "assert not {'harness', 'tools', 'simulators'} & set(sys.modules)\nprint(module.__file__)")
        public = ('import sys\nimport integrations.pokemon.network, integrations.pokemon.efficient\n'
                  f'from {runtime}sdk import BENCHMARK_ID,identity,write,source_signature,runtime_signature,validate_recipe,vector_for\n'
                  f'from {runtime}graph_export import export\nimport {runtime}env_client, {runtime}graph_worker\n'
                  f'import {runtime}sdk as module\n'
                  "assert not {'harness', 'tools', 'simulators'} & set(sys.modules)\nprint(module.__file__)")
        for tree, code in (('source', protected), ('public-source', public)):
            done = subprocess.run([sys.executable, '-c', code], cwd=self.out, capture_output=True, text=True, timeout=120,
                                  env=dict(os.environ, PYTHONPATH=str(frozen / tree)))
            self.assertEqual(done.returncode, 0, done.stderr[-3000:])
            self.assertTrue(done.stdout.strip().startswith(str(frozen / tree)), done.stdout)
        # Everything the editable starter imports from the runtime is in the public closure.
        public_modules = {name[:-3].replace('/', '.') for name in bootstrap.PUBLIC_SOURCE_FILES}
        for name in sdk.EDITABLE:
            for line in (sdk.STARTER / name).read_text().splitlines():
                words = line.split()
                if words[:1] == ['from'] and words[1].startswith('integrations.'):
                    self.assertIn(words[1], public_modules, line)

    def test_real_provider_binds_one_private_app_and_volume_to_the_session_without_any_cloud_call(self):
        self.root.mkdir()
        provider = transport.Provider(self.release, self.root)
        binding = json.loads((self.root / 'provider-binding.json').read_bytes())
        self.assertEqual((binding['environment'], binding['image_id'], binding['volume']),
                         (policy.ENVIRONMENT, bootstrap.IMAGE, binding['app'] + '-private'))
        self.assertLessEqual(len(binding['volume']), 64)
        self.assertEqual(transport.Provider(self.release, self.root).binding, provider.binding)  # a reopen keeps it
        other = Path(tempfile.mkdtemp(dir=self.temp.name))
        self.assertNotEqual(transport.Provider(self.release, other).binding['app'], binding['app'])
        with mock.patch.object(sdk, 'runtime_signature', return_value='0' * 64):
            with self.assertRaisesRegex(ValueError, 'prepare a new release'):
                transport.Provider(self.release, self.root)

    # Research sessions -----------------------------------------------------------------------------------
    def test_run_carries_exactly_the_researchers_files_and_is_charged(self):
        broker = self.session()
        self.assertEqual(broker.handle({'op': 'budget'})['remaining_seconds'], 600)
        self.files['training/learner.py'] += SENTINEL
        config = dict(json.loads(self.files['configs/smoke.json']), learning_rate=0.001)
        self.files['configs/smoke.json'] = json.dumps(config)
        reply = self.run_request(config='configs/smoke.json')
        self.assertNotIn('error', reply, reply)
        join(self.backend)
        launch, = self.provider.launches
        source = {name.split('/')[1]: self.files[name].encode() for name in self.files if name.endswith('.py')}
        self.assertEqual({name.split('/')[1]: data for name, data in launch['candidate'].items() if name.endswith('.py')},
                         source)
        self.assertEqual(json.loads(launch['candidate']['candidate/config.json']), config)
        self.assertEqual(set(launch['candidate']), {'candidate/' + name for name in (*sdk.EDITABLE, 'config.json')})
        self.assertNotEqual(config, json.loads(self.files['configs/train.json']))
        spec = launch['spec']
        self.assertEqual((launch['kind'], launch['seconds'], spec['action'], spec['config'], spec['training_seconds']),
                         ('research', 240, 'train', config, 120))
        self.assertNotIn('resume', spec)
        self.assertNotIn('fork', spec)
        self.assertNotIn('fresh_final', spec)  # a research run trains with the recipe's own seeds
        self.assertEqual((spec['config']['seed'], spec['config']['training_seed_start']), (1729, 100000))
        self.assertFalse(self.backend.final or (self.root / 'final-training.json').exists())
        self.assertEqual(set(launch['names']) - set(launch['candidate']),
                         {'native-admission.json', 'env-state-key.bin', 'runtime/generators.json',
                          'runtime/constructors.json', 'runtime/chart.json'})
        self.assertEqual(self.jobs(), ([('a', 'train', 'complete')], []))
        self.assertEqual(self.spent(), (100, 500))  # settled at the provider's interval, not the lease
        stage = self.backend.controller.record('a')
        self.assertEqual(stage['source_sha256'], spec['source_sha256'])
        package = Path(stage['output']) / 'stage/bundle'
        self.assertEqual((package / 'learner.py').read_text(), self.files['training/learner.py'])
        self.assertEqual(bundle.verify(package)['bundle_sha256'], stage['bundle_sha256'])
        log = broker.handle({'op': 'log', 'name': 'a'})
        self.assertEqual((log['training_totals']['trained_samples'], log['training_outcome']['termination_reason']),
                         (64, 'completed'))
        self.assertIn('Unknown run', broker.handle({'op': 'log', 'name': 'zzz'})['error'])
        self.assertIn('already used', self.run_request()['error'])  # attempts keep their names

    def test_bad_requests_never_reach_the_ledger(self):
        broker = self.session()

        def config(**change):
            value = dict(json.loads(self.files['configs/train.json']), **change)
            return dict(self.files, **{'configs/train.json': json.dumps(value)})

        def source(text):
            return dict(self.files, **{'training/policy.py': self.files['training/policy.py'] + text})
        for request in ({'files': source('\ndef broken(:')},
                        {'files': dict(self.files, extra='x')},
                        {'files': config(rollout_steps=0)},
                        {'files': config(environments=100000)},
                        {'files': config(training_seed_start=900_000_000)},  # reserved evaluation seed domain
                        {'files': config(resume_checkpoint='x')},            # a recipe is never state
                        {'files': source('\nimport pickle\n')},
                        {'files': source('\nBLOB = "' + 'A' * 5000 + '"\n')},  # no encoded payloads in source
                        {'files': source('\nTABLE = [' + '0.5, ' * 300 + ']\n')},
                        {'seconds': 100000},
                        {'seconds': 120},  # below the 180-second minimum training lease
                        {'name': '../a'},
                        {'name': 'eval-0123abcd'},  # reserved for practice jobs
                        {'name': 'private-grade'},
                        {'seed': 7}):
            reply = self.run_request(**request)
            self.assertIn('error', reply, request)
            self.assertNotIn('Host fault', reply['error'], request)
        self.assertEqual(Ledger(self.root / 'ledger.sqlite3').snapshot()['jobs'], [])
        self.assertEqual(self.provider.launches, [])
        self.assertFalse(list((self.root / 'snapshots').glob('*')) if (self.root / 'snapshots').exists() else [])
        self.assertFalse((self.root / 'stages').exists())
        self.assertEqual(self.spent(), (0, 600))
        self.assertIn('error', broker.handle({'op': 'check', 'files': config(minibatch=-1)}))
        self.assertTrue(broker.handle({'op': 'check', 'files': self.files})['passed'])
        self.assertIn('error', broker.handle({'op': 'eval', 'name': 'a'}))
        self.assertIn('error', broker.handle({'op': 'submit', 'name': 'a'}))

    def test_one_job_at_a_time_and_status_normalization(self):
        broker = self.session()
        self.provider.hold()
        self.assertNotIn('error', self.run_request(seconds=300))
        self.assertTrue(self.provider.entered.wait(10))
        self.assertEqual(self.jobs(), ([('a', 'train', 'queued')], ['a']))  # reserved, no worker yet
        self.assertEqual(self.spent(), (300, 300))                          # the whole lease is held
        self.assertIn('One job at a time', self.run_request(name='b')['error'])
        self.assertIn('Resolve active work', broker.handle({'op': 'finalize'})['error'])
        self.provider.create.set()
        deadline = time.time() + 10
        while self.jobs()[0] != [('a', 'train', 'running')] and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.jobs(), ([('a', 'train', 'running')], ['a']))
        self.provider.outcome = 'failed'
        self.provider.finish.set()
        join(self.backend)
        self.assertEqual(self.jobs(), ([('a', 'train', 'failed')], []))
        self.assertEqual(self.spent(), (100, 500))  # a failed worker's allocated time stays charged
        self.assertIn('error', broker.handle({'op': 'submit', 'name': 'a'}))
        self.assertIn('did not finish', broker.handle({'op': 'log', 'name': 'a'})['own_log_tails']['worker-error.log'])

    def test_log_follows_a_running_trainer_and_a_final_training_shows_nothing(self):
        broker = self.session()
        self.provider.hold()
        self.run_request(seconds=300)
        self.assertTrue(self.provider.entered.wait(10))
        waiting = broker.handle({'op': 'log', 'name': 'a'})
        self.assertEqual((waiting['live'], self.provider.peeks), (False, 0))  # no worker yet: nothing is read
        self.provider.create.set()
        deadline = time.time() + 10
        while self.jobs()[0] != [('a', 'train', 'running')] and time.time() < deadline:
            time.sleep(0.01)
        self.assertIn('not started', broker.handle({'op': 'log', 'name': 'a'})['note'])
        rows = [dict(iteration=i, samples=64, seconds=1.5, policy_loss=1 / i, seed=7, note='x') for i in range(1, 41)]
        self.provider.printed = 'starting\n' + ''.join(json.dumps(row) + '\n' for row in rows) + 'Traceback: boom\n'
        self.backend.controller.peeked.clear()
        live = broker.handle({'op': 'log', 'name': 'a', 'tail': 5})
        self.assertTrue(live['live'])
        self.assertEqual(live['status'], 'running')
        self.assertEqual(live['trainer_log_tail'], 'starting\nTraceback: boom')  # metric rows are not repeated as text
        self.assertEqual(len(live['progress_samples']), 12)
        self.assertEqual(live['progress_samples'][0], dict(iteration=1, samples=64, seconds=1.5, policy_loss=1.0))
        self.assertEqual(live['progress_samples'][-1]['iteration'], 40)  # measured scalars only, newest kept
        self.assertEqual(live['printed_bytes'], len(self.provider.printed))
        peeks = self.provider.peeks
        broker.handle({'op': 'log', 'name': 'a'})
        self.assertEqual(self.provider.peeks, peeks)  # the Volume is not re-read on every look
        self.provider.finish.set()
        join(self.backend)
        done = broker.handle({'op': 'log', 'name': 'a'})
        self.assertNotIn('live', done)
        self.assertIn('progress_samples', done)

        final = open_backend(self, self.release, self.out / 'final', Provider(self.out / 'final'))
        self.assertIsNone(final.controller.live('missing'))
        final.controller.final = True
        self.assertIsNone(final.controller.live('a'))  # a fresh final's output could echo its host seeds

    def test_task_ships_the_dry_run_and_every_public_module_it_stages(self):
        out = self.out / 'task'
        task.build('pokemon', 'recipe', self.release, out, research_seconds=600, final_train_seconds=240,
                   wall_seconds=900, hardware='test')
        self.assertEqual((out / 'environment/ar/dryrun.py').read_bytes(), DRY_RUN.read_bytes())
        ar = Researcher(out, self.out / 'box')
        passed = json.loads(subprocess.run([sys.executable, '-I', str(ar.ar / 'dryrun.py'), '--workspace',
                                            str(ar.workspace)], capture_output=True, text=True, timeout=300).stdout)
        self.assertTrue(passed['passed'], passed)

    def test_stop_before_creation_is_a_proven_zero_and_stop_after_is_charged(self):
        self.session()
        self.provider.hold()
        self.run_request(name='early')
        self.assertTrue(self.provider.entered.wait(10))
        self.assertEqual(self.broker.handle({'op': 'stop', 'name': 'early'}), {'stopping': 'early'})
        self.provider.create.set()
        join(self.backend)
        self.assertEqual(self.jobs(), ([('early', 'train', 'stopped')], []))
        self.assertEqual(self.spent(), (0, 600))
        receipt = json.loads(Ledger(self.root / 'ledger.sqlite3').snapshot()['jobs'][0]['receipt'])
        self.assertTrue(receipt['provider_creation_not_attempted'])
        self.provider.finish.clear()
        self.run_request(name='late')
        deadline = time.time() + 10
        while self.jobs()[0][-1][2] != 'running' and time.time() < deadline:
            time.sleep(0.01)
        self.backend.settle()  # the host closing a session stops and settles the researcher's worker
        self.assertEqual(self.jobs(), ([('early', 'train', 'stopped'), ('late', 'train', 'stopped')], []))
        self.assertEqual(self.spent(), (100, 500))
        self.backend.settle()  # idempotent
        self.assertIn('not active', self.broker.handle({'op': 'stop', 'name': 'late'})['error'])
        self.assertEqual(self.broker.finalize('host')['submission'], None)
        self.assertFalse(Ledger(self.root / 'ledger.sqlite3').snapshot()['finalized'])

    def running(self, name):
        deadline = time.time() + 10
        while (self.backend.controller.record(name) or {}).get('status') != 'active' and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.backend.controller.record(name)['status'], 'active')

    def test_stopped_and_cut_off_runs_that_sealed_are_complete_flagged_and_usable(self):
        broker = self.session()
        self.provider.seconds = 40  # every worker here is charged 40 seconds of the 600
        self.provider.finish.clear()
        self.run_request(name='stopped')
        self.running('stopped')
        reply = broker.handle({'op': 'stop', 'name': 'stopped'})
        self.assertEqual(reply['stopping'], 'stopped')
        self.assertIn('stopped_early', reply['note'])
        join(self.backend)
        operation = self.backend.controller.record('stopped')['operation']
        self.assertEqual((self.provider.asked, self.provider.stopped), ({operation}, set()))  # asked, never terminated
        job, = broker.handle({'op': 'status'})['jobs']
        self.assertEqual((job['status'], job['stopped_early'], job['charged_seconds'], 'cut_short' in job),
                         ('complete', True, 40, False), job)
        log = broker.handle({'op': 'log', 'name': 'stopped'})
        self.assertEqual((log['stopped_early'], log['training_outcome']['termination_reason'],
                          log['training_outcome']['stopped_early']), (True, 'stop_requested', True))
        self.assertIn('Stopped on request', log['note'])
        # A trainer still running at its stage limit, sealed by the worker: complete, flagged cut_short.
        self.provider.receipt = dict(cut_short=True, termination_reason='hard_deadline')
        self.provider.finish.set()
        self.run_request(name='cut')
        join(self.backend)
        jobs = {job['name']: job for job in broker.handle({'op': 'status'})['jobs']}
        self.assertEqual((jobs['cut']['status'], jobs['cut']['cut_short'], 'stopped_early' in jobs['cut']),
                         ('complete', True, False), jobs['cut'])
        self.assertIn('stage time limit', broker.handle({'op': 'log', 'name': 'cut'})['note'])
        # Both can be practised and submitted like any complete run.
        self.provider.receipt = {}
        for name in ('stopped', 'cut'):
            practice = broker.handle({'op': 'eval', 'name': name, 'seconds': 240})
            self.assertEqual((practice.get('games'), practice.get('run')), (128, name), practice)
            join(self.backend)
            self.assertEqual(broker.handle({'op': 'log', 'name': practice['queued']})['score_valid'], True)
        self.assertEqual(broker.handle({'op': 'submit', 'name': 'cut'})['submitted']['name'], 'cut')
        self.assertEqual(broker.handle({'op': 'submit', 'name': 'stopped'})['submitted']['name'], 'stopped')
        # A practice panel has nothing to package: stopping it ends its worker at once.
        self.provider.finish.clear()
        label = broker.handle({'op': 'eval', 'name': 'cut', 'seconds': 240})['queued']
        self.running(label)
        self.assertEqual(broker.handle({'op': 'stop', 'name': label}), {'stopping': label})
        join(self.backend)
        self.assertEqual(self.provider.stopped, {self.backend.controller.record(label)['operation']})
        self.assertEqual(len(self.provider.asked), 1)
        self.assertEqual(self.jobs()[0][-1], (label, 'practice', 'stopped'))

    def test_uncertain_creation_and_overrun_keep_their_hold_and_block_allocation(self):
        self.session()
        self.provider.outcome = 'create-interrupted'
        self.run_request()
        join(self.backend)
        self.assertEqual(self.jobs(), ([('a', 'train', 'uncertain')], ['a']))
        self.assertEqual(self.spent(), (240, 360))
        self.assertIn('unknown allocation', self.run_request(name='b')['error'])
        self.assertIn('Resolve active work', self.broker.handle({'op': 'finalize'})['error'])
        self.assertEqual(len(self.provider.launches), 1)
        # A worker measured beyond its reservation is never forgiven.
        other = Path(tempfile.mkdtemp(dir=self.temp.name)) / 'research'
        provider = Provider(other, seconds=250)
        backend = open_backend(self, self.release, other, provider, budget_seconds=600)
        backend.run('a', 240, 'configs/smoke.json')
        join(backend)
        self.assertEqual([(j['name'], j['status']) for j in backend.status()['jobs']], [('a', 'uncertain')])
        with self.assertRaisesRegex(ValueError, 'exceeded its reservation'):
            backend.run('b', 240, 'configs/smoke.json')

    def test_resume_reopens_the_same_allowance(self):
        self.session(budget_seconds=600)
        self.run_request()
        join(self.backend)
        self.backend.close()
        with self.assertRaises(ValueError):
            registry.backend('pokemon')(self.release, self.root, smoke=True, budget_seconds=600)
        again = registry.backend('pokemon')(self.release, self.root, smoke=True, budget_seconds=99999, resume=True,
                                            options={'_provider': self.provider, 'poll_seconds': 0.01})
        again.final = True  # a research root stays a research root, whatever a reopen is told
        again.open()
        self.addCleanup(again.close)
        self.assertFalse(again.final)
        budget = again.budget()
        self.assertEqual((budget['budget_seconds'], budget['used_seconds'], budget['remaining_seconds']), (600, 100, 500))
        self.assertEqual([(j['name'], j['status']) for j in again.status()['jobs']], [('a', 'complete')])

    # Budget modes ----------------------------------------------------------------------------------------
    def dollars(self, budget_usd):
        """A dollar-mode session whose workers run at once until each is let finish."""
        self.provider = Parallel(self.root)
        self.backend = open_backend(self, self.release, self.root, self.provider, budget_usd=budget_usd)
        self.broker = Broker(self.backend, self.out, 'network', research_usd=budget_usd)
        return self.broker

    def finish(self, name):
        self.provider.released.add(self.backend.controller.record(name)['operation'])

    def status(self, name):
        return dict((job[0], job[2]) for job in self.jobs()[0]).get(name)

    def test_seconds_mode_is_the_default_one_job_charged_in_training_worker_seconds(self):
        broker = self.session()
        budget = broker.handle({'op': 'budget'})
        self.assertEqual((budget['budget_mode'], budget['concurrency'], budget['budget_seconds']), ('seconds', 1, 600))
        self.assertFalse({'budget_usd', 'used_usd', 'remaining_usd'} & set(budget))
        limits = Ledger(self.root / 'ledger.sqlite3').snapshot()['limits']
        self.assertEqual((limits['budget_mode'], limits['max_concurrent_jobs'], limits['research_shape']),
                         ('seconds', 1, game.TRAIN_SHAPE))
        self.provider.finish.clear()
        reply = self.run_request(seconds=300)
        self.assertNotIn('reserved_usd', reply)
        self.running('a')
        self.assertIn('One job at a time; active: a', self.run_request(name='b')['error'])
        self.provider.finish.set()
        join(self.backend)
        self.provider.finish.clear()
        self.assertNotIn('error', broker.handle({'op': 'eval', 'name': 'a', 'seconds': 240}))
        self.assertIn('One job at a time', self.run_request(name='b')['error'])  # a practice is the one job too
        self.provider.finish.set()
        join(self.backend)
        self.assertEqual(self.spent(), (200, 400))  # practice is charged the same training-worker seconds
        self.assertNotIn('charged_usd', broker.handle({'op': 'status'})['jobs'][0])

    def test_dollar_mode_runs_jobs_at_once_within_its_limits_and_charges_dollars(self):
        rate = budgets.rate(game.TRAIN_SHAPE)
        broker = self.dollars(0.6)
        budget = broker.handle({'op': 'budget'})
        self.assertEqual((budget['budget_mode'], budget['budget_usd'], budget['remaining_usd'], budget['concurrency'],
                          budget['concurrent_gpus'], budget['jobs_at_once']), ('usd', 0.6, 0.6, 4, 2, 2))
        self.assertFalse({'budget_seconds', 'used_seconds', 'remaining_seconds'} & set(budget))
        self.assertEqual(list(budget['affordable_seconds'].values()), [int(0.6 / rate)])
        limits = Ledger(self.root / 'ledger.sqlite3').snapshot()['limits']
        self.assertEqual((limits['budget_mode'], limits['research_usd'], limits['usd_per_second'], limits['prices'],
                          limits['max_concurrent_jobs'], limits['max_concurrent_gpus']),
                         ('usd', 0.6, rate, budgets.PRICE, 4, 2))
        self.assertEqual(self.run_request('a')['reserved_usd'], round(240 * rate, 6))
        self.assertNotIn('error', self.run_request('b'))
        self.running('a')
        self.running('b')
        self.assertEqual(self.jobs(), ([('a', 'train', 'running'), ('b', 'train', 'running')], ['a', 'b']))
        self.assertIn('At most 2 jobs at once', self.run_request('c', seconds=180)['error'])  # one GPU each
        self.assertAlmostEqual(broker.handle({'op': 'budget'})['used_usd'], 480 * rate, places=6)  # whole grants held
        # Each job is its own: a log and a stop by name, and the other job is untouched.
        self.assertEqual(broker.handle({'op': 'log', 'name': 'a'})['status'], 'running')
        self.assertIn('stopped_early', broker.handle({'op': 'stop', 'name': 'b'})['note'])
        self.assertTrue(until(lambda: self.status('b') == 'complete'))
        self.assertEqual((self.status('a'), self.provider.asked, self.provider.stopped),
                         ('running', {self.backend.controller.record('b')['operation']}, set()))
        self.assertTrue(broker.handle({'op': 'log', 'name': 'b'})['stopped_early'])
        self.assertIn('Resolve active work before finalizing: a', broker.handle({'op': 'finalize'})['error'])
        # Grants never add up past the budget: 340 s charged or held leaves $0.326, 404 s of the worker.
        self.assertIn('remaining allocation', self.run_request('c', seconds=420)['error'])
        self.assertNotIn('error', self.run_request('c', seconds=300))
        self.finish('a')
        self.assertTrue(until(lambda: self.status('a') == 'complete'))
        practice = broker.handle({'op': 'eval', 'name': 'b', 'seconds': 240})  # beside the run still training
        self.assertEqual(practice['reserved_usd'], round(240 * rate, 6), practice)
        self.running(practice['queued'])
        self.assertIn('At most 2 jobs at once', self.run_request('d', seconds=180)['error'])
        self.assertLessEqual(broker.handle({'op': 'budget'})['used_usd'], 0.6)
        self.finish('c')
        self.finish(practice['queued'])
        self.backend.controller.join(30)
        jobs = broker.handle({'op': 'status'})['jobs']
        self.assertEqual([(job['name'], job['status'], job['charged_usd']) for job in jobs],
                         [(name, 'complete', round(100 * rate, 6)) for name in ('a', 'b', 'c', practice['queued'])])
        budget = broker.handle({'op': 'budget'})
        self.assertAlmostEqual(budget['used_usd'], 400 * rate, places=6)  # settled at the provider's intervals
        self.assertEqual(broker.handle({'op': 'log', 'name': practice['queued']})['score_valid'], True)
        self.assertEqual(broker.handle({'op': 'submit', 'name': 'b'})['submitted']['name'], 'b')
        self.assertTrue(broker.handle({'op': 'finalize'})['finalized'])

    def test_dollar_mode_closes_reattaches_and_holds_every_active_job(self):
        rate = budgets.rate(game.TRAIN_SHAPE)
        broker = self.dollars(1.0)
        for name in ('a', 'b'):
            self.run_request(name)
            self.running(name)
        controller = self.backend.controller
        self.backend.settle()  # closing the session ends every worker at once, without asking a trainer
        self.assertEqual(self.jobs(), ([('a', 'train', 'stopped'), ('b', 'train', 'stopped')], []))
        self.assertEqual((self.provider.stopped, self.provider.asked),
                         ({controller.record(name)['operation'] for name in 'ab'}, set()))
        # What a host that stopped while two workers ran leaves behind: a reopen follows both.
        for name in ('c', 'd'):
            operation = controller.ledger.reserve(name, 'research', 240)
            transport.private_write(self.root / 'jobs' / operation / 'provider.json',
                                    dict(operation=operation, sandbox_id='sb-' + operation[:8]))
            controller.ledger.attach(operation, 'sb-' + operation[:8])
            controller.update(name, name=name, kind='train', operation=operation, status='active', seconds=240)
            self.provider.jobs[operation] = ('research', dict(action='train'), {})
        controller.tick()
        self.assertEqual(sorted(controller.running()), ['c', 'd'])
        self.provider.outcome = 'failed'
        for name in ('c', 'd'):
            self.finish(name)
        controller.join(30)
        self.assertEqual(self.jobs()[0][2:], [('c', 'train', 'failed'), ('d', 'train', 'failed')])
        # An uncertain allocation keeps its whole grant and its place among the jobs at once, without blocking.
        operation = controller.ledger.reserve('x', 'research', 240)
        controller.ledger.unknown(operation, dict(reason='stub: creation unresolved'))
        controller.update('x', name='x', kind='train', operation=operation, status='unknown', seconds=240, successful=False)
        self.provider.outcome = 'complete'
        self.assertNotIn('error', self.run_request('e'))
        self.assertIn('At most 2 jobs at once', self.run_request('f')['error'])
        self.assertAlmostEqual(broker.handle({'op': 'budget'})['unresolved_usd'], 480 * rate, places=6)
        self.running('e')
        self.finish('e')
        self.assertTrue(until(lambda: self.status('e') == 'complete'))
        self.assertEqual(self.jobs()[1], ['x'])
        self.assertEqual(broker.handle({'op': 'submit', 'name': 'e'})['submitted']['name'], 'e')
        self.assertIn('Resolve active work before finalizing: x', broker.handle({'op': 'finalize'})['error'])

    def test_a_session_keeps_its_budget_mode_when_reopened(self):
        self.dollars(1.0)
        self.backend.close()
        options = {'_provider': self.provider, 'poll_seconds': 0.01}
        with self.assertRaisesRegex(ValueError, 'opened in usd mode'):
            registry.backend('pokemon')(self.release, self.root, smoke=True, budget_seconds=600, resume=True,
                                        options=options).open()
        again = registry.backend('pokemon')(self.release, self.root, smoke=True, resume=True, options=options)
        again.open()
        self.addCleanup(again.close)
        self.assertEqual((again.budget_mode, again.limits['max_concurrent_jobs'], again.budget()['budget_usd']),
                         ('usd', 4, 1.0))
        root = self.out / 'seconds'
        open_backend(self, self.release, root, Provider(root), budget_seconds=600).close()
        with self.assertRaisesRegex(ValueError, 'opened in seconds mode'):
            registry.backend('pokemon')(self.release, root, smoke=True, budget_usd=1.0, resume=True,
                                        options=options).open()

    def test_final_root_trains_with_host_seeds_that_never_reach_a_reply(self):
        options = {'_provider': self.provider, 'poll_seconds': 0.01}
        backend = registry.backend('pokemon')(self.release, self.root, smoke=True, budget_seconds=240, options=options)
        backend.final = True  # as harness.session.final_training does before open()
        backend.open()
        self.addCleanup(backend.close)
        recipe = json.loads(self.starter['configs/smoke.json'])
        with mock.patch.object(control.secrets, 'randbelow', side_effect=[612345677, 598765431]):
            replies = [backend.run('final', 240, 'configs/smoke.json')]
        join(backend)
        launch, = self.provider.launches
        spec, config = launch['spec'], launch['spec']['config']
        self.assertEqual((config['seed'], config['training_seed_start']), (612345678, 598765432))
        self.assertNotEqual((config['seed'], config['training_seed_start']), (recipe['seed'], recipe['training_seed_start']))
        self.assertEqual(dict(config, seed=recipe['seed'], training_seed_start=recipe['training_seed_start']), recipe)
        self.assertEqual((spec['fresh_final'], spec['no_research_state'], 'resume' in spec, 'fork' in spec),
                         (True, True, False, False))
        # The source-only payload is the recipe itself; only the trusted request carries the host's seeds.
        self.assertEqual(json.loads(launch['candidate']['candidate/config.json']), recipe)
        self.assertEqual(set(launch['candidate']), {'candidate/' + name for name in (*sdk.EDITABLE, 'config.json')})
        replies += [backend.status(), backend.log('final'), backend.log('final', 2000), backend.budget()]
        self.assertEqual((replies[1]['jobs'][0]['status'], replies[0]['seed_source']), ('complete', 'fresh-host-secrets'))
        visible = json.dumps(replies)
        for secret in ('612345678', '598765432'):
            self.assertNotIn(secret, visible)
        self.assertIsInstance(replies[2]['own_log_tails'], str)  # raw trainer output could echo a config
        retained = self.root / 'final-training.json'
        intent = json.loads(retained.read_bytes())
        self.assertEqual((intent['effective_config'], intent['seed_source'], intent['status'], retained.stat().st_mode & 0o077),
                         (config, 'fresh-host-secrets', 'settled', 0))
        package = Path(backend.controller.record('final')['output']) / 'stage/bundle'
        self.assertEqual(json.loads((package / 'config.json').read_bytes()), config)  # the policy is bound to what it trained on
        with self.assertRaisesRegex(ValueError, 'One fresh final attempt only'):
            backend.run('again', 180, 'configs/smoke.json')
        self.assertEqual(len(self.provider.launches), 1)
        backend.submit('final')
        backend.finalize()
        backend.close()
        # Grading reopens the root; that it is a final root is the root's own state.
        graded = registry.backend('pokemon')(self.release, self.root, smoke=True, resume=True, options=options)
        self.assertFalse(graded.final)
        graded.open()
        self.addCleanup(graded.close)
        self.assertTrue(graded.final)
        result = graded.grade()
        self.assertEqual((result['score_valid'], result['raw_score_0_100'], result['scheduled_games']), (True, 62.5, 16))
        self.assertEqual(self.provider.profiles, [('research', 'L40S'), ('final', None)])
        self.assertNotIn('612345678', json.dumps(result))

    def test_network_session_trains_practices_submits_and_is_graded_by_the_host(self):
        broker = self.session()
        self.assertNotIn('error', self.run_request(seconds=300))
        join(self.backend)
        practice = broker.handle({'op': 'eval', 'name': 'a'})
        self.assertEqual((practice['games'], practice['reserved_seconds']), (128, 360), practice)
        self.assertEqual((len(control.PRACTICE_SEEDS), control.PRACTICE_SEEDS[0], control.PRACTICE_SEEDS[-1]),
                         (16, 900_420_000, 900_420_015))  # one contiguous public block, as the contract states
        contract = json.loads((ROOT / 'simulators/pokemon/autoresearch/benchmark.json').read_text())
        self.assertEqual((contract['evaluation']['practice_public_seeds'], contract['evaluation']['practice_games'],
                          contract['budget']['min_practice_lease_seconds'], contract['budget']['default_practice_lease_seconds']),
                         (list(control.PRACTICE_SEEDS), practice['games'], control.MIN_PRACTICE_SECONDS, control.PRACTICE_SECONDS))
        join(self.backend)
        label = practice['queued']
        self.assertEqual(self.jobs(), ([('a', 'train', 'complete'), (label, 'practice', 'complete')], []))
        self.assertEqual(self.spent(), (200, 400))  # practice is allocated worker time too
        launch = self.provider.launches[-1]
        request = launch['spec']['assessment']
        package = Path(self.backend.controller.record('a')['output']) / 'stage/bundle'
        self.assertEqual((launch['kind'], launch['spec']['action'], request['public_development'], request['seeds']),
                         ('research', 'assess', True, list(control.PRACTICE_SEEDS)))
        self.assertEqual(request['candidate'], dict(kind='bundle-native', name=label, path='/pilotjob/input/candidate',
                                                    bundle_sha256=bundle.verify(package)['bundle_sha256']))
        self.assertEqual(launch['candidate']['candidate/weights.pt'], (package / 'weights.pt').read_bytes())
        self.assertTrue({'peer-1/weights.pt', 'peer-2/weights.pt', 'foul-play.tar.gz', 'public-sets.json'} <= set(launch['names']))
        self.assertNotIn('env-state-key.bin', launch['names'])
        log = broker.handle({'op': 'log', 'name': label})
        self.assertEqual((log['score_valid'], log['score_0_100'], log['run']), (True, 62.5, 'a'))
        self.assertEqual(broker.handle({'op': 'log', 'name': 'a'})['practice'],
                         [dict(name=label, status='complete', score_0_100=62.5)])
        self.assertIn('at least 240', broker.handle({'op': 'eval', 'name': 'a', 'seconds': 120})['error'])
        budget = broker.handle({'op': 'budget'})
        self.assertEqual((budget['min_practice_seconds'], budget['default_practice_seconds'], budget['practice_public_seeds']),
                         (240, 360, 16))
        self.assertIn('error', broker.handle({'op': 'finalize'}))  # nothing submitted yet
        submitted = broker.handle({'op': 'submit', 'name': 'a'})['submitted']
        self.assertEqual(submitted['native']['bundle_sha256'], bundle.verify(package)['bundle_sha256'])
        self.assertNotIn('bundle', submitted['native'])  # host paths stay with the host
        published = self.root / 'submissions/a'
        self.assertEqual(bundle.verify(published)['bundle_sha256'], submitted['native']['bundle_sha256'])
        with self.assertRaisesRegex(ValueError, 'lock before private'):
            self.backend.controller.ledger.private_bank(2)
        self.assertTrue(broker.handle({'op': 'finalize'})['finalized'])
        self.assertIn('error', self.run_request(name='b'))
        self.assertEqual(len(self.provider.launches), 2)
        self.backend.close()
        graded = open_backend(self, self.release, self.root, self.provider, resume=True)
        graded.official = False
        result = graded.grade()
        launch = self.provider.launches[-1]
        request = launch['spec']['assessment']
        self.assertEqual((launch['kind'], launch['seconds'], request['public_development'], len(request['seeds'])),
                         ('final', 600, False, 2))
        self.assertTrue(all(1_900_000_000 <= seed < 2_000_000_000 for seed in request['seeds']))
        self.assertEqual(launch['candidate']['candidate/weights.pt'], (published / 'weights.pt').read_bytes())
        self.assertEqual(self.provider.profiles, [('research', 'L40S'), ('final', None)])
        self.assertEqual((result['score_valid'], result['raw_score_0_100'], result['official'], result['panel'],
                          result['scheduled_games'], result['profile_verified']), (True, 62.5, False, 'smoke', 16, True))
        self.assertEqual(result['release_sha256'], self.manifest['release_sha256'])
        self.assertNotIn(str(request['seeds'][0]), json.dumps(result))  # only a commitment leaves the ledger
        self.assertEqual(self.spent(), (200, 400))                      # grading is not research allocation
        self.assertEqual(graded.status()['jobs'][-1]['name'], label)    # nor a researcher job
        with self.assertRaises(FileExistsError):
            graded.grade()  # private games are never replayed

    def test_recipe_session_retrains_the_submitted_recipe_on_a_fresh_allowance(self):
        out = self.out / 'recipe'
        box = self.out / 'box'
        providers = {}
        Base = registry.backend('pokemon')

        class Stubbed(Base):
            def __init__(self, release, root, **kwargs):
                stub = providers.setdefault(Path(root).name, Provider(root))
                super().__init__(release, root, **dict(kwargs, options={'_provider': stub, 'poll_seconds': 0.01}))

        def script(ar):
            self.assertEqual(ar('budget')['remaining_seconds'], 600)
            ar.edit('training/train.py', (ar.workspace / 'training/train.py').read_text() + SENTINEL)
            ar('check', '--config', 'configs/smoke.json')
            self.assertEqual(ar('run', 'a', '--seconds', '240', '--config', 'configs/smoke.json')['name'], 'a')
            self.assertEqual(ar('wait', 'a')['active'], [])
            ar('submit', '--note', 'starter with a marker')
            ar('finalize')
        with mock.patch.object(registry, 'backend', lambda game: Stubbed), \
                mock.patch.object(control.secrets, 'randbelow', side_effect=[612345677, 598765431]):
            result = session.run('pokemon', 'recipe', self.release, out, model='test/model', smoke=True,
                                 launcher=lambda folder: Process(lambda: script(Researcher(folder / 'task', box))),
                                 find_box=lambda: LocalBox(box / 'ar') if (box / 'ar').is_dir() else None)
            self.assertEqual((result['status'], result['error']), ('researched', None), result)
            final = result['final_training']
            self.assertEqual((final['status'], final['seconds'], final['config']), ('complete', 240.0, 'configs/smoke.json'), final)
            research, = providers['research'].launches
            trained, = providers['final'].launches
            self.assertTrue(trained['candidate']['candidate/train.py'].decode().endswith(SENTINEL))
            self.assertEqual(trained['candidate'], research['candidate'])  # exactly the submitted recipe
            # Research trained with the recipe's seeds; the host's fresh final did not, and kept its seeds to itself.
            recipe = json.loads(research['candidate']['candidate/config.json'])
            self.assertEqual(research['spec']['config'], recipe)
            self.assertNotIn('fresh_final', research['spec'])
            self.assertEqual((trained['spec']['fresh_final'], trained['spec']['config']),
                             (True, dict(recipe, seed=612345678, training_seed_start=598765432)))
            for name in ('final-train.json', 'session-result.json', 'requests.jsonl', 'session.json'):
                self.assertNotIn('612345678', (out / name).read_text(), name)
            self.assertEqual((trained['seconds'], trained['spec']['training_seconds']), (240, 120))
            ledger = Ledger(out / 'final/ledger.sqlite3').snapshot()
            self.assertEqual((ledger['limits']['research_seconds'], ledger['finalized'], [j['name'] for j in ledger['jobs']]),
                             (240, True, ['final']))  # nothing from research reaches the fresh root
            self.assertFalse(Ledger(out / 'research/ledger.sqlite3').snapshot()['finalized'])
            graded = Stubbed(self.release, out / 'final', smoke=True, resume=True)
            graded.open()
            self.addCleanup(graded.close)
            self.assertEqual(graded.grade()['bundle_sha256'], ledger['submissions'][0]['bundle_sha256'])


class Volume:
    """A session Volume: uploads are recorded, results are read back."""

    def __init__(self):
        self.commits, self.objects = 0, {}

    def batch_upload(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.commits += 1

    def put_file(self, source, name):
        self.objects[name] = Path(source).read_bytes()

    def read_file(self, name):
        if name not in self.objects:
            raise FileNotFoundError(name)
        yield self.objects[name]


class ShortReadVolume(Volume):
    """Lists sizes like a Modal Volume. A results archive's first read breaks off with an error partway, its second
    ends early without one, and only its third is whole."""

    def __init__(self):
        super().__init__()
        self.short = set()
        self.reads = {}

    def listdir(self, folder):
        prefix = folder.rstrip('/') + '/'
        return [SimpleNamespace(path=name, size=len(data)) for name, data in self.objects.items()
                if name.startswith(prefix) and '/' not in name[len(prefix):]]

    def read_file(self, name):
        data = self.objects[name]
        if name.endswith('/results.tar'):
            self.reads[name] = self.reads.get(name, 0) + 1
            if self.reads[name] == 1:
                yield data[:300]
                raise ConnectionError('stream reset')
            if self.reads[name] == 2:
                self.short.add(name)
                data = data[:700]  # inside the second member
        yield data


class Transport(unittest.TestCase):
    """The trusted bootstrap and the real provider, against a stand-in for the cloud SDK."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def snapshot(self, name='frozen'):
        target = self.root / 'release' / name
        files = {'source/integrations/pokemon/a.py': b'print("frozen")\n',
                 'public-source/integrations/pokemon/public.py': b'PUBLIC=1\n'}
        for path, content in files.items():
            (target / path).parent.mkdir(parents=True, exist_ok=True)
            (target / path).write_bytes(content)
        manifest = dict(source_sha256='fixture', image_id=bootstrap.IMAGE,
                        files={'integrations/pokemon/a.py': sha(target / 'source/integrations/pokemon/a.py')},
                        public_files={'integrations/pokemon/public.py': sha(target / 'public-source/integrations/pokemon/public.py')})
        (target / 'source-manifest.json').write_text(json.dumps(manifest))
        return target, manifest

    def provider(self, session='session'):
        """The real provider on a fixture release, without the release audit of its constructor."""
        target, manifest = self.snapshot()
        bootstrap.packed_source(target, manifest)
        value = object.__new__(transport.Provider)
        value.release, value.root, value.manifest = target.parent, self.root / session, manifest
        value.root.mkdir()
        value.binding = dict(app='pokemon-ar-test', volume='pokemon-ar-test-private')
        return value

    def malicious_pack(self, path, name, content=b'bad', kind=None):
        with tarfile.open(path, 'w') as archive:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            if kind is not None:
                info.type, info.linkname = kind, '/outside'
            archive.addfile(info, io.BytesIO(content))

    def test_exact_roundtrip_deterministic_pack(self):
        target, manifest = self.snapshot()
        pack = bootstrap.packed_source(target, manifest)
        digest, data = sha(pack), pack.read_bytes()
        self.assertEqual(bootstrap.unpack_source_pack(pack, self.root / 'out', manifest, digest)['files'], 2)
        self.assertEqual((self.root / 'out/source/integrations/pokemon/a.py').read_bytes(), b'print("frozen")\n')
        self.assertEqual(bootstrap.packed_source(target, manifest).read_bytes(), data)

    def test_tampered_unsafe_or_unexpected_packs_are_rejected_before_extraction(self):
        target, manifest = self.snapshot()
        pack = bootstrap.packed_source(target, manifest)
        digest = sha(pack)
        with pack.open('ab') as stream:
            stream.write(b'tamper')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            bootstrap.unpack_source_pack(pack, self.root / 'out', manifest, digest)
        for index, (name, kind) in enumerate([('../escape', None), ('/absolute', None),
                                              ('source/integrations/pokemon/a.py', tarfile.SYMTYPE), ('unexpected', None)]):
            bad = self.root / f'bad{index}.tar'
            self.malicious_pack(bad, name, kind=kind)
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                bootstrap.unpack_source_pack(bad, self.root / f'out{index}', manifest, sha(bad))
            self.assertFalse((self.root / f'out{index}').exists())
        bad = self.root / 'changed.tar'
        self.malicious_pack(bad, 'source/integrations/pokemon/a.py', b'changed')
        with self.assertRaisesRegex(ValueError, 'member SHA256'):
            bootstrap.unpack_source_pack(bad, self.root / 'out', manifest, sha(bad))
        with tarfile.open(bad, 'w') as archive:
            archive.add(target / 'source/integrations/pokemon/a.py', arcname='source/integrations/pokemon/a.py')
        with self.assertRaisesRegex(ValueError, 'exact file set'):
            bootstrap.unpack_source_pack(bad, self.root / 'out', manifest, sha(bad))
        self.assertFalse((self.root / 'out').exists())

    def test_lease_is_bound_to_allocation_entry_not_to_upload_time(self):
        spec = dict(absolute_stop_unix=100, allocation_seconds=60, allocation_margin_seconds=5,
                    original_campaign_deadline_unix=None)
        proof = bootstrap.worker_deadline(spec, 5000)
        self.assertEqual((proof['effective_absolute_stop_unix'], proof['upload_preparation_excluded']), (5055, True))
        for seconds in (float('inf'), 0, 59, 86401):
            with self.assertRaises(ValueError):
                bootstrap.worker_deadline(dict(spec, allocation_seconds=seconds), 5000)
        self.assertEqual(worker.timing_reserves('assess'), dict(publication=15, start_guard=20))
        self.assertEqual(worker.timing_reserves('train'), dict(publication=30, start_guard=60))

    def test_worker_gives_the_supervisor_the_jobs_stop_request_and_a_stage_that_ends_before_sealing_and_publication(self):
        from simulators.pokemon.autoresearch.runtime import supervisor
        job = self.root / 'job'
        candidate = job / 'input/candidate'
        candidate.mkdir(parents=True)
        for name in sdk.EDITABLE:
            (candidate / name).write_text('# fixture\n')
        (job / 'input/env-state-key.bin').write_bytes(b'k' * 32)
        config = json.loads((constructor.HERE / 'recipes/smoke.json').read_text())
        stop = time.time() + 500
        (job / 'input/request.json').write_text(json.dumps(dict(
            action='train', arm=transport.SCOPE, source_sha256=sdk.source_signature(candidate), config=config,
            training_seconds=3480, absolute_stop_unix=stop)))
        with mock.patch.object(worker, 'runtime'), mock.patch.object(worker, 'publish_results') as publish, \
                mock.patch.object(supervisor, 'train') as train, mock.patch.dict(os.environ), \
                mock.patch.object(sys, 'argv', ['worker', '--job', str(job)]):
            worker.main()
        (source, used, out, seconds, device, resume), options = train.call_args
        self.assertEqual((source, used, out, device, resume), (candidate, config, job / 'output/stage', 'cuda:0', None))
        self.assertEqual(options['stop_request'], job / transport.STOP_REQUEST)
        self.assertEqual((transport.MOUNT, transport.STOP_REQUEST), ('/campaign', 'stop-request'))
        self.assertIn("link=pathlib.Path('/pilotjob')", bootstrap.ENTRY)       # the worker's job folder ...
        self.assertIn('job=campaign/operation;job.chmod(0o700)', bootstrap.ENTRY)  # ... is the root-only Volume folder
        # The stage's own deadline is 60 s before the worker's absolute stop: 30 s to end the trainer, validate and
        # seal (also when the stage is cut off there), then the worker's alarm and its 30 s publication reserve.
        self.assertTrue(stop - time.time() - 60 - 5 < seconds <= stop - time.time() - 60 + 5, seconds)
        self.assertEqual((worker.STAGE_END_RESERVE_SECONDS, worker.PUBLICATION_RESERVE_SECONDS), (60, 30))
        self.assertTrue((job / 'output/completed.json').is_file())
        publish.assert_called_once()

    def test_image_engine_is_restored_without_the_images_old_python(self):
        image = self.root / 'image'
        engine = image / 'vendor/engines/pokemon-showdown'
        for name in ('battle-stream.js', 'battle.js', 'index.js'):
            (engine / 'dist/sim').mkdir(parents=True, exist_ok=True)
            (engine / 'dist/sim' / name).write_text('module.exports={};')
        (engine / 'node_modules').mkdir()
        (image / 'candidates').symlink_to('vendor/engines', target_is_directory=True)
        (image / 'old-private-recipe.py').write_text('OLD=1')
        destination = self.root / 'fresh'
        destination.mkdir()
        self.assertTrue(bootstrap.restore_image_engine(image, destination)['original_and_copied_bytes_identical'])
        self.assertEqual((destination / 'candidates/pokemon-showdown/dist/sim/battle.js').read_text(), 'module.exports={};')
        self.assertFalse((destination / 'old-private-recipe.py').exists())
        (engine / 'node_modules/escape').symlink_to(self.root / 'outside')
        other = self.root / 'other'
        other.mkdir()
        with self.assertRaisesRegex(ValueError, 'link escapes'):
            bootstrap.restore_image_engine(image, other)

    def test_bootstrap_admits_only_the_packed_source_before_the_trusted_worker(self):
        provider, volume = self.provider(), Volume()
        reference = self.root / 'reference.pt'
        reference.write_bytes(b'reference')
        layout = provider.stage_shared(volume, [(reference, 'peer-1/weights.pt')])
        mount, operation = self.root / 'mount', 'ab' * 16
        inputs = mount / operation / 'input'
        inputs.mkdir(parents=True)
        for name, content in volume.objects.items():
            (mount / name).parent.mkdir(parents=True, exist_ok=True)
            (mount / name).write_bytes(content)
        (inputs / 'input-layout.json').write_text(json.dumps(layout))
        (inputs / 'source-manifest.json').write_bytes((provider.release / 'frozen/source-manifest.json').read_bytes())
        original = dict(action='assess', allocation_seconds=60, allocation_margin_seconds=5,
                        absolute_stop_unix=time.time() + 1000, original_campaign_deadline_unix=None)
        (inputs / 'request.json').write_text(json.dumps(original))
        alias, private, public = self.root / 'job-alias', self.root / 'opt-pk', self.root / 'public'
        alias.mkdir()
        (alias / 'historical-image-file').write_bytes(b'preserve-image-cache')
        private.mkdir()
        (private / 'old-module.py').write_bytes(b'OLD_PRIVATE=1')
        for name in ('battle-stream.js', 'battle.js', 'index.js'):
            (private / 'candidates/pokemon-showdown/dist/sim').mkdir(parents=True, exist_ok=True)
            (private / 'candidates/pokemon-showdown/dist/sim' / name).write_text('module.exports={};')
        (private / 'candidates/pokemon-showdown/node_modules').mkdir()
        (private / 'FROZEN.json').write_text('{"pinned":true}')
        public.mkdir()
        (public / 'old-module.py').write_bytes(b'OLD_PUBLIC=1')
        entry = bootstrap.ENTRY
        self.assertLess(entry.index('unpack_source_pack(shared'), entry.index('p=subprocess.run'))
        self.assertIn('integrations.pokemon.autoresearch.runtime.worker', entry)
        replacements = sorted([('/campaign', mount), ('/pilotjob', alias), ('/opt/pilot-public', public), ('/opt/pk', private),
                               ('/runtime', self.root / 'runtime'), ('/job', self.root / 'old-job'),
                               ('/private', self.root / 'private'), ('/image-pilot-cache', self.root / 'image-cache'),
                               ('/opt/pk-image-cache', self.root / 'protected-image-cache'),
                               ('/opt/pilot-public-image-cache', self.root / 'public-image-cache')],
                              key=lambda item: -len(item[0]))
        for index, (before, _) in enumerate(replacements):
            entry = entry.replace(before, f'__ROOT_{index}__')
        for index, (_, after) in enumerate(replacements):
            entry = entry.replace(f'__ROOT_{index}__', str(after))
        cwd = Path.cwd()
        try:
            with mock.patch.dict(os.environ, {'POKEMON_OPERATION_ID': operation}), mock.patch('os.chown'), \
                    mock.patch('subprocess.run', return_value=SimpleNamespace(returncode=0)) as run:
                with self.assertRaises(SystemExit) as stopped:
                    exec(compile(entry, '<bootstrap-test>', 'exec'), {})
                self.assertEqual(stopped.exception.code, 0)
                self.assertEqual(run.call_args.args[0][:3], ['python', '-m', 'integrations.pokemon.autoresearch.runtime.worker'])
        finally:
            os.chdir(cwd)
        self.assertEqual((private / 'integrations/pokemon/a.py').read_bytes(), b'print("frozen")\n')
        self.assertEqual((public / 'integrations/pokemon/public.py').read_bytes(), b'PUBLIC=1\n')
        self.assertEqual((inputs / 'peer-1/weights.pt').read_bytes(), b'reference')
        self.assertFalse((public / 'old-module.py').exists() or (private / 'old-module.py').exists())
        effective = json.loads((inputs / 'effective-request.json').read_text())
        timing = json.loads((inputs.parent / 'output/effective-worker-deadline.json').read_text())
        self.assertEqual(effective['absolute_stop_unix'], timing['allocation_entry_unix'] + 55)

    def test_real_provider_reserves_publishes_creates_and_settles_from_receipts(self):
        # The archive's first read comes back short (as over a congested link); collection reads it again.
        provider, volume = self.provider(), ShortReadVolume()
        ledger = Ledger(provider.root / 'ledger.sqlite3', dict(research_seconds=600, max_lease_seconds=3600, grading_seconds=600))
        candidate, key, peer = self.root / 'policy.py', self.root / 'key.bin', self.root / 'weights.pt'
        candidate.write_text('x = 1\n')
        key.write_bytes(b'k' * 32)
        peer.write_bytes(b'reference')
        files = [(candidate, 'candidate/policy.py'), (key, 'env-state-key.bin'), (peer, 'peer-1/weights.pt')]
        created = []

        def create(*command, **options):
            created.append((command, options))
            return SimpleNamespace(object_id='sb-test')
        modal = SimpleNamespace(Volume=SimpleNamespace(from_name=lambda *a, **k: volume),
                                App=SimpleNamespace(lookup=lambda *a, **k: SimpleNamespace(app_id='ap-test')),
                                Image=SimpleNamespace(from_id=lambda value: ('image', value)),
                                Sandbox=SimpleNamespace(create=create))
        spec = dict(action='train', source_sha256='s', config={}, training_seconds=120)
        with mock.patch.dict(sys.modules, {'modal': modal}):
            operation = ledger.reserve('a', 'research', 240)
            with mock.patch.object(transport, 'allocation_info', return_value=dict(provider_id='sb-test', started_at=100.0)):
                provider.launch(ledger, operation, 'research', spec, files, 240)  # returns once the worker started
            # The candidate and the private state key are per-job inputs; only reference assets are shared.
            self.assertEqual(volume.objects[operation + '/input/candidate/policy.py'], b'x = 1\n')
            self.assertEqual(volume.objects[operation + '/input/env-state-key.bin'], b'k' * 32)
            shared = [name for name in volume.objects if name.startswith('_shared/')]
            self.assertEqual(len(shared), 2)
            self.assertFalse(any(volume.objects[name] in (b'x = 1\n', b'k' * 32) for name in shared))
            request = json.loads(volume.objects[operation + '/input/request.json'])
            self.assertEqual((request['allocation_seconds'], request['allocation_margin_seconds'], request['arm']),
                             (240, 30, transport.SCOPE))
            (command, options), = created
            self.assertEqual(command, ('python', '-c', bootstrap.ENTRY))
            self.assertEqual((options['gpu'], options['cpu'], options['memory'], options['timeout'], options['block_network'],
                              options['secrets'], options['environment_name'], options['image']),
                             ('L40S', (4, 4), (16384, 16384), 220, True, [], policy.ENVIRONMENT, ('image', bootstrap.IMAGE)))
            self.assertEqual(options['env'], {'POKEMON_OPERATION_ID': operation})
            self.assertEqual(ledger.job(operation)['status'], 'active')
            # Not finished: nothing settles. Finished: charged at the provider's own interval.
            receipt = dict(provider_id='sb-test', created_at=90.0, started_at=100.0, finished_at=0, gpu_type='L40S',
                           initialization_failure_status=0, resource_gpu_type='L40S', cpu_milli=4000,
                           cpu_milli_max=4000, memory_mib=16384, memory_mib_max=16384)
            with mock.patch.object(transport, 'allocation_info', return_value=receipt):
                self.assertEqual(provider.collect(ledger, operation)['status'], 'active-or-unknown')
            archive = self.root / 'results.tar'
            with tarfile.open(archive, 'w') as tar:
                for name, content in (('completed.json', b'{}'), ('stage/result.json', b'{"iterations": 1}')):
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    tar.addfile(info, io.BytesIO(content))
            volume.objects[operation + '/output/results.tar'] = archive.read_bytes()
            with mock.patch.object(transport, 'allocation_info', return_value=dict(receipt, finished_at=287.5)), \
                    mock.patch.object(transport.time, 'sleep'):
                result = provider.collect(ledger, operation)
                self.assertIn(operation + '/output/results.tar', volume.short)
                self.assertEqual(volume.reads[operation + '/output/results.tar'], 3)
                self.assertEqual(provider.verify_profile(operation, 'L40S')['cpu_milli'], 4000)
                with self.assertRaisesRegex(ValueError, 'GPU profile differs'):
                    provider.verify_profile(operation, None)
            self.assertEqual((result['status'], result['successful']), ('settled', True))
            self.assertEqual(json.loads((Path(result['output']) / 'stage/result.json').read_text()), {'iterations': 1})
            self.assertEqual((ledger.job(operation)['status'], ledger.job(operation)['actual_seconds']), ('settled', 187.5))
            # A failure before the provider is asked settles at a proven zero; one during creation stays held.
            second = ledger.reserve('b', 'research', 240)
            with self.assertRaises(transport.Stopped):
                provider.launch(ledger, second, 'research', spec, files, 240, stopped=lambda: True)
            self.assertEqual((ledger.job(second)['status'], ledger.job(second)['actual_seconds'], len(created)), ('settled', 0.0, 1))
            modal.Sandbox.create = mock.Mock(side_effect=ConnectionError('lost'))
            third = ledger.reserve('c', 'research', 240)
            with self.assertRaises(ConnectionError):
                provider.launch(ledger, third, 'research', spec, files, 240)
            self.assertEqual(ledger.job(third)['status'], 'unknown')
            with self.assertRaisesRegex(ValueError, 'unknown allocation'):
                ledger.reserve('d', 'research', 240)
        self.assertEqual(ledger.snapshot()['research_seconds_used_or_reserved'], 187.5 + 0 + 240)

    def test_practice_runs_on_a_right_sized_cpu_worker_and_charges_the_research_ledger(self):
        # A practice is charged to the research allocation (ledger kind `research`) but holds no GPU: it reserves
        # what an arena uses, keeps the contract shape as its cap, and settles from a CPU receipt.
        provider, volume = self.provider(), ShortReadVolume()
        ledger = Ledger(provider.root / 'ledger.sqlite3', dict(research_seconds=600, max_lease_seconds=3600, grading_seconds=600))
        candidate = self.root / 'policy.py'
        candidate.write_text('x = 1\n')
        created = []

        def create(*command, **options):
            created.append(options)
            return SimpleNamespace(object_id='sb-practice')
        modal = SimpleNamespace(Volume=SimpleNamespace(from_name=lambda *a, **k: volume),
                                App=SimpleNamespace(lookup=lambda *a, **k: SimpleNamespace(app_id='ap-test')),
                                Image=SimpleNamespace(from_id=lambda value: ('image', value)),
                                Sandbox=SimpleNamespace(create=create))
        with mock.patch.dict(sys.modules, {'modal': modal}):
            operation = ledger.reserve('eval-1', 'research', 240)
            with mock.patch.object(transport, 'allocation_info', return_value=dict(provider_id='sb-practice', started_at=100.0)):
                provider.launch(ledger, operation, 'practice', dict(action='assess', assessment={}),
                                [(candidate, 'candidate/policy.py')], 240)
            options, = created
            self.assertEqual((options['gpu'], options['cpu'], options['memory']),
                             (None, (transport.CPU_JOB_REQUEST_CORES, 4), (transport.CPU_JOB_REQUEST_MIB, 16384)))
            receipt = dict(provider_id='sb-practice', created_at=90.0, started_at=100.0, finished_at=210.0, gpu_type='CPU',
                           initialization_failure_status=0, resource_gpu_type='', cpu_milli=1000, cpu_milli_max=4000,
                           memory_mib=2048, memory_mib_max=16384)
            archive = self.root / 'practice.tar'
            with tarfile.open(archive, 'w') as tar:
                info = tarfile.TarInfo('completed.json')
                info.size = 2
                tar.addfile(info, io.BytesIO(b'{}'))
            volume.objects[operation + '/output/results.tar'] = archive.read_bytes()
            with mock.patch.object(transport, 'allocation_info', return_value=receipt), mock.patch.object(transport.time, 'sleep'):
                self.assertEqual(provider.collect(ledger, operation)['status'], 'settled')
                self.assertEqual(provider.verify_profile(operation, None)['cpu_milli'], 1000)
                with self.assertRaisesRegex(ValueError, 'GPU profile differs'):
                    provider.verify_profile(operation, 'L40S')
            self.assertEqual((ledger.job(operation)['kind'], ledger.job(operation)['actual_seconds']), ('research', 110.0))

    def test_shared_names_and_collected_archives_cannot_escape(self):
        for name in ('../secret', '/secret', 'candidate/../../secret', 'a\\b'):
            with self.assertRaises(ValueError):
                transport.shared_file(name)
        self.assertFalse(transport.shared_file('candidate/weights.pt') or transport.shared_file('env-state-key.bin'))
        for index, (name, kind) in enumerate((('../escape', None), ('/absolute', None), ('link', tarfile.SYMTYPE))):
            archive = self.root / f'results{index}.tar'
            self.malicious_pack(archive, name, kind=kind)
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                transport.unpack_results(archive, self.root / f'collected{index}')
            self.assertFalse((self.root / f'collected{index}').exists())


class Cloud:
    """Stands in for Modal: sandboxes that may wait in a GPU queue, their provider records, and their results.

    `start_after` is how many reads of a worker's record pass before the provider starts it (None: never);
    once started it runs for `runs_for` more reads and then publishes a successful training result.
    """

    def __init__(self, volume, start_after=0, runs_for=3, lifetime=100.0):
        self.volume, self.start_after, self.runs_for, self.lifetime = volume, start_after, runs_for, lifetime
        self.sandboxes, self.terminated = {}, []
        # Files the host wrote into workers; whether a write arrives; whether a worker asked to stop ends by itself.
        self.written, self.deliverable, self.answers_stop, self.stopped_results = {}, True, True, None

    def create(self, *command, **options):
        sandbox = SimpleNamespace(object_id=f'sb-{len(self.sandboxes)}', operation=options['env']['POKEMON_OPERATION_ID'],
                                  created_at=time.time(), started_at=None, finished_at=None, reads=0, started_read=None,
                                  asked=False)
        sandbox.terminate = lambda: self.finish(sandbox, terminated=True)
        sandbox.filesystem = SimpleNamespace(write_text=lambda data, path: self.write(sandbox, data, path))
        self.sandboxes[sandbox.object_id] = sandbox
        return sandbox

    def write(self, sandbox, data, path):
        if not self.deliverable:
            raise ConnectionError('stub: the worker could not be reached')
        self.written[path] = (data, list(self.terminated))
        sandbox.asked = True

    def finish(self, sandbox, terminated=False):
        if sandbox.finished_at is not None:
            return
        if terminated:
            self.terminated.append(sandbox.object_id)
        sandbox.finished_at = sandbox.started_at + self.lifetime if sandbox.started_at else time.time()
        if sandbox.started_at and not terminated:
            self.publish(sandbox)

    def publish(self, sandbox):
        """What a successful training worker leaves on the session Volume: results.tar with a sealed stage."""
        self.volume.objects[sandbox.operation + '/output/results.tar'] = (self.stopped_results if sandbox.asked
                                                                          else self.results)

    @staticmethod
    def results_archive(source, config, **receipt):
        """A successful worker's results.tar for this source and config (built on the main thread: graph export)."""
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            (temp / 'out').mkdir()
            Provider.stage(temp / 'out/stage', source, config, **receipt)
            (temp / 'out/completed.json').write_text('{}')
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode='w') as archive:
                for path in sorted((temp / 'out').rglob('*')):
                    if path.is_file():
                        archive.add(path, arcname=path.relative_to(temp / 'out').as_posix())
            return stream.getvalue()

    def info(self, app_id, sandbox_id):
        sandbox = self.sandboxes.get(sandbox_id)
        if sandbox is None:
            return None
        sandbox.reads += 1
        if (sandbox.started_at is None and sandbox.finished_at is None and self.start_after is not None
                and sandbox.reads > self.start_after):
            sandbox.started_at, sandbox.started_read = time.time(), sandbox.reads
        elif sandbox.started_at and (sandbox.reads >= sandbox.started_read + self.runs_for
                                     or sandbox.asked and self.answers_stop):
            self.finish(sandbox)
        return dict(provider_id=sandbox_id, task_id='ta-1', created_at=sandbox.created_at,
                    started_at=sandbox.started_at or 0, finished_at=sandbox.finished_at or 0,
                    gpu_type='L40S' if sandbox.started_at else '', initialization_failure_status=0,
                    initialization_failure='', resource_gpu_type='L40S', cpu_milli=4000, cpu_milli_max=4000,
                    memory_mib=16384, memory_mib_max=16384)


class QueuedWorkers(unittest.TestCase):
    """The real controller and provider against a cloud whose GPU workers can sit queued, never start, or lag."""

    snapshot = Transport.snapshot
    results = None  # one successful worker's results.tar, shared by every test
    early = None    # the same for a trainer its supervisor ended: on a stop request, at the stage time limit

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        target, manifest = self.snapshot()
        bootstrap.packed_source(target, manifest)
        release = target.parent
        (release / 'private').mkdir()
        (release / 'private/env-state-key.bin').write_bytes(b'k' * 32)
        (target / 'runtime').mkdir()
        for name in ('native-admission', 'generators', 'constructors', 'chart'):  # what runtime_files sends
            (target / 'runtime' / (name + '.json')).write_text('{}')
        self.volume = Volume()
        self.cloud = Cloud(self.volume, start_after=40)
        modal = SimpleNamespace(Volume=SimpleNamespace(from_name=lambda *a, **k: self.volume),
                                App=SimpleNamespace(lookup=lambda *a, **k: SimpleNamespace(app_id='ap-test')),
                                Image=SimpleNamespace(from_id=lambda value: ('image', value)),
                                Sandbox=SimpleNamespace(create=self.cloud.create, from_id=self.cloud.sandboxes.__getitem__))
        for patcher in (mock.patch.dict(sys.modules, {'modal': modal}),
                        mock.patch.object(transport, 'allocation_info', self.cloud.info)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.release, self.manifest = release, manifest
        provider = self.real_provider(self.root / 'research')
        self.provider = provider
        self.backend = registry.backend('pokemon')(release, provider.root, smoke=True, budget_seconds=600,
                                                   options={'_provider': provider, 'poll_seconds': 0.01})
        self.backend.open()
        self.addCleanup(self.backend.close)
        self.controller = self.backend.controller
        self.source = {name: (sdk.STARTER / name).read_text() for name in sdk.EDITABLE}
        self.config = json.loads((constructor.HERE / 'recipes/smoke.json').read_text())
        if QueuedWorkers.results is None:
            QueuedWorkers.results = Cloud.results_archive(sdk.STARTER, self.config)
            QueuedWorkers.early = {flag: Cloud.results_archive(sdk.STARTER, self.config, termination_reason=reason,
                                                               **{flag: True})
                                   for flag, reason in (('stopped_early', 'stop_requested'), ('cut_short', 'hard_deadline'))}
        self.cloud.results, self.cloud.stopped_results = QueuedWorkers.results, QueuedWorkers.early['stopped_early']

    def real_provider(self, root):
        provider = object.__new__(transport.Provider)
        provider.release, provider.root, provider.manifest = self.release, root, self.manifest
        provider.binding = dict(app='pokemon-ar-test', volume='pokemon-ar-test-private')
        provider.start_patience, provider.start_poll, provider.abandon_poll = 0.5, 0.001, 0.001
        return provider

    def training(self, name='a'):
        """A run whose worker has started and would train on."""
        self.cloud.start_after, self.cloud.runs_for = 0, 10 ** 6
        self.train(name)
        deadline = time.time() + 30
        while (self.controller.record(name) or {}).get('status') != 'active' and time.time() < deadline:
            time.sleep(0.01)
        self.assertJobs(([(name, 'running')], [name]))
        return self.row(name)['id']

    def train(self, name='a', seconds=240):
        return self.controller.run(name, seconds, self.source, self.config)

    def join(self):
        self.controller.thread.join(timeout=60)
        self.assertFalse(self.controller.thread.is_alive())

    def jobs(self):
        status = self.backend.status()
        return [(job['name'], job['status']) for job in status['jobs']], status['active']

    def assertJobs(self, expected):
        status = self.backend.status()
        self.assertEqual(self.jobs(), expected, status)

    def row(self, name):
        return next(job for job in self.controller.ledger.snapshot()['jobs'] if job['name'] == name)

    def test_a_long_queue_then_a_start_trains_and_charges_only_the_started_lifetime(self):
        self.provider.start_patience = 30.0
        self.train()
        self.join()
        self.assertJobs(([('a', 'complete')], []))
        self.assertGreater(self.cloud.sandboxes['sb-0'].reads, 40)  # it waited through the queue
        self.assertEqual((self.row('a')['status'], self.row('a')['actual_seconds']), ('settled', 100.0))
        self.assertFalse((self.root / 'research/jobs' / self.row('a')['id'] / 'abandoned.json').exists())

    def test_a_worker_that_never_starts_fails_uncharged_and_the_session_continues(self):
        self.cloud.start_after = None
        self.train('queued')
        self.join()
        self.assertJobs(([('queued', 'failed')], []))  # never `uncertain`
        row = self.row('queued')
        receipt = json.loads(row['receipt'])
        self.assertEqual((row['status'], row['actual_seconds'], receipt['provider_never_started'], receipt['started_at']),
                         ('settled', 0, True, 0))
        self.assertEqual(self.cloud.terminated, ['sb-0'])
        self.assertEqual(self.backend.budget()['used_seconds'], 0)
        self.assertIn('never started', self.backend.status()['jobs'][0]['error'])
        self.assertIn('charged 0 seconds', self.backend.log('queued')['job_error'])
        self.cloud.start_after = 0
        self.assertEqual(self.train('b')['queued'], 'b')  # accepted at once
        self.join()
        self.assertJobs(([('queued', 'failed'), ('b', 'complete')], []))
        self.backend.submit('b')
        self.assertEqual(self.backend.finalize()['job'], 'b')
        self.assertEqual(self.backend.budget()['used_seconds'], 100)

    def test_stop_while_queued_is_stopped_and_uncharged(self):
        self.cloud.start_after = None
        self.provider.start_patience = 30.0
        self.train()
        deadline = time.time() + 10
        while not self.cloud.sandboxes and time.time() < deadline:
            time.sleep(0.01)
        self.assertJobs(([('a', 'running')], ['a']))
        self.assertEqual(self.backend.stop('a'), {'stopping': 'a'})
        self.join()
        self.assertJobs(([('a', 'stopped')], []))
        self.assertEqual(self.row('a')['actual_seconds'], 0)
        self.assertIn('Stopped while queued', self.backend.log('a')['job_error'])

    def test_stop_asks_the_trainer_to_end_and_its_sealed_run_is_complete_and_stopped_early(self):
        operation = self.training()
        reply = self.backend.stop('a')
        self.assertEqual(reply['stopping'], 'a')
        self.assertIn('stopped_early', reply['note'])
        self.join()
        # The request is one file in the job's own root-only folder of the worker; nothing was terminated.
        self.assertEqual(self.cloud.written, {f'/campaign/{operation}/stop-request': ('stop\n', [])})
        self.assertEqual(self.cloud.terminated, [])
        job, = self.backend.status()['jobs']
        self.assertEqual((job['status'], job['stopped_early'], job['charged_seconds']), ('complete', True, 100.0), job)
        self.assertEqual(self.backend.status()['active'], [])
        record = self.controller.record('a')
        self.assertEqual((record['successful'], record['stopped_early'], 'stop_grace_expired_at' in record), (True, True, False))
        self.assertTrue((self.root / 'research/jobs' / operation / 'stop-request.json').is_file())
        log = self.backend.log('a')
        self.assertEqual((log['status'], log['stopped_early'], log['training_outcome']['termination_reason']),
                         ('complete', True, 'stop_requested'))
        self.assertIn('Stopped on request', log['note'])
        self.backend.submit('a')
        self.assertEqual(self.backend.finalize()['job'], 'a')

    def test_a_trainer_that_has_not_finished_within_the_grace_is_terminated(self):
        self.cloud.answers_stop = False  # the worker never acts on the request
        self.controller.stop_grace = 0.3
        operation = self.training()
        asked = time.time()
        self.assertIn('note', self.backend.stop('a'))
        self.join()
        (data, terminated_before), = self.cloud.written.values()
        self.assertEqual((data, terminated_before, self.cloud.terminated), ('stop\n', [], ['sb-0']))  # asked first
        record = self.controller.record('a')
        self.assertGreaterEqual(record['stop_grace_expired_at'] - asked, 0.3)
        self.assertEqual(round(record['stop_grace_until'] - record['stop_request_sent_at'], 3), 0.3)
        self.assertJobs(([('a', 'stopped')], []))
        job, = self.backend.status()['jobs']
        self.assertEqual((job['charged_seconds'], 'stopped_early' in job), (100.0, False))
        self.assertIn('terminated; nothing was packaged', self.backend.log('a')['note'])
        with self.assertRaisesRegex(ValueError, 'No completed registered run'):
            self.backend.submit('a')
        self.assertTrue((self.root / 'research/jobs' / operation / 'stop-request.json').is_file())

    def test_a_request_that_cannot_be_delivered_and_a_session_close_end_the_worker_at_once(self):
        self.cloud.deliverable = False
        operation = self.training()
        self.assertEqual(self.backend.stop('a'), {'stopping': 'a'})  # no promise of a packaged run
        self.join()
        self.assertEqual((self.cloud.written, self.cloud.terminated), ({}, ['sb-0']))
        self.assertTrue((self.root / 'research/jobs' / operation / 'stop-request-error.json').is_file())
        self.assertJobs(([('a', 'stopped')], []))
        # Closing the session never waits for a trainer: no request is written, the worker is ended.
        self.cloud.deliverable = True
        self.cloud.start_after, self.cloud.runs_for = 0, 10 ** 6
        self.train('b')
        deadline = time.time() + 30
        while (self.controller.record('b') or {}).get('status') != 'active' and time.time() < deadline:
            time.sleep(0.01)
        self.backend.settle()
        self.assertEqual((self.cloud.written, self.cloud.terminated), ({}, ['sb-0', 'sb-1']))
        self.assertJobs(([('a', 'stopped'), ('b', 'stopped')], []))

    def test_a_host_restarted_inside_the_grace_still_terminates_at_its_end(self):
        """What a host that stopped after sending a stop request left behind: an active job nobody follows."""
        self.cloud.answers_stop, self.cloud.runs_for = False, 10 ** 6
        operation = self.controller.ledger.reserve('a', 'research', 240)
        sandbox = self.cloud.create(env={'POKEMON_OPERATION_ID': operation})
        sandbox.started_at, sandbox.started_read = time.time(), 0
        transport.private_write(self.root / 'research/jobs' / operation / 'provider.json',
                                dict(operation=operation, sandbox_id=sandbox.object_id, app_id='ap-test'))
        self.controller.ledger.attach(operation, sandbox.object_id)
        self.controller.update('a', name='a', kind='train', operation=operation, status='active', seconds=240,
                               stop_requested_at=time.time(), stop_request_sent_at=time.time(),
                               stop_grace_until=time.time() + 0.3)
        self.assertEqual(self.controller.grace, {})  # nothing in memory: the grace is read from the job's record
        self.controller.tick()                        # reattaches to the recorded worker
        self.join()
        self.assertEqual(self.cloud.terminated, ['sb-0'])
        self.assertIn('stop_grace_expired_at', self.controller.record('a'))
        self.assertJobs(([('a', 'stopped')], []))

    def test_a_trainer_cut_off_at_its_limit_but_sealed_is_complete_and_a_final_counts_as_trained(self):
        self.cloud.start_after = 0
        self.cloud.results = QueuedWorkers.early['cut_short']
        self.train()
        self.join()
        job, = self.backend.status()['jobs']
        self.assertEqual((job['status'], job['cut_short'], 'stopped_early' in job), ('complete', True, False), job)
        log = self.backend.log('a')
        self.assertEqual((log['cut_short'], log['training_outcome']['termination_reason'], log['training_outcome']['cut_short']),
                         (True, 'hard_deadline', True))
        self.assertEqual(self.cloud.terminated, [])
        # The host's fresh final training of a recipe, cut off the same way: trained, not a forfeit.
        root = self.root / 'final'
        final = registry.backend('pokemon')(self.release, root, smoke=True, budget_seconds=240,
                                            options={'_provider': self.real_provider(root), 'poll_seconds': 0.01})
        final.final = True  # as harness.session.final_training does before open()
        final.open()
        self.addCleanup(final.close)
        final.controller.run('final', 240, self.source, self.config)
        final.controller.thread.join(timeout=60)
        job, = final.status()['jobs']
        self.assertEqual((job['status'], job['cut_short']), ('complete', True), job)  # what final_training requires
        intent = json.loads((root / 'final-training.json').read_bytes())
        self.assertEqual((intent['status'], intent['successful'], intent['cut_short']), ('settled', True, True))
        final.submit('final')
        self.assertEqual(final.finalize()['job'], 'final')

    def test_dollar_mode_workers_run_at_once_each_in_its_own_volume_folder(self):
        root = self.root / 'usd'
        backend = registry.backend('pokemon')(self.release, root, smoke=True, budget_usd=1.0,
                                              options={'_provider': self.real_provider(root), 'poll_seconds': 0.01})
        backend.open()
        self.addCleanup(backend.close)
        controller = backend.controller
        self.cloud.start_after, self.cloud.runs_for = 0, 10 ** 6
        controller.run('a', 240, self.source, self.config)
        self.assertTrue(until(lambda: len(self.cloud.sandboxes) == 1))
        controller.run('b', 240, self.source, self.config)
        self.assertTrue(until(lambda: all((controller.record(name) or {}).get('status') == 'active' for name in 'ab')))
        self.assertEqual([(job['name'], job['status']) for job in backend.status()['jobs']],
                         [('a', 'running'), ('b', 'running')])
        with self.assertRaisesRegex(ValueError, 'At most 2 jobs at once'):
            controller.run('c', 240, self.source, self.config)
        operations = {name: controller.record(name)['operation'] for name in 'ab'}
        for operation in operations.values():  # each job's own request and inputs, under its own operation
            self.assertIn(operation + '/input/request.json', self.volume.objects)
            self.assertEqual(json.loads(self.volume.objects[operation + '/input/request.json'])['allocation_seconds'], 240)
        # A stop is delivered to that job's own folder only; its worker publishes there and settles alone.
        self.assertIn('note', backend.stop('a'))
        self.assertTrue(until(lambda: dict((j['name'], j['status']) for j in backend.status()['jobs'])['a'] == 'complete'))
        self.assertEqual(list(self.cloud.written), [f'/campaign/{operations["a"]}/stop-request'])
        self.assertIn(operations['a'] + '/output/results.tar', self.volume.objects)
        self.assertNotIn(operations['b'] + '/output/results.tar', self.volume.objects)
        self.assertEqual(backend.status()['active'], ['b'])
        self.assertEqual(backend.log('b')['status'], 'running')
        backend.settle()  # the close ends the other worker
        self.assertEqual(self.cloud.terminated, ['sb-1'])
        jobs = backend.status()['jobs']
        self.assertEqual([(job['name'], job['status'], job['charged_usd']) for job in jobs],
                         [('a', 'complete', round(100 * budgets.rate(game.TRAIN_SHAPE), 6)),
                          ('b', 'stopped', round(100 * budgets.rate(game.TRAIN_SHAPE), 6))])
        self.assertTrue(all((root / 'jobs' / operation / 'provider-stopped.json').is_file()
                            for operation in operations.values()))

    def unresolved(self, name):
        """A job whose worker was created and recorded, then left `unknown` (the host stopped following it)."""
        self.cloud.start_after = None
        operation = self.controller.ledger.reserve(name, 'research', 240)
        sandbox = self.cloud.create(env={'POKEMON_OPERATION_ID': operation})
        transport.private_write(self.root / 'research/jobs' / operation / 'provider.json',
                                dict(operation=operation, sandbox_id=sandbox.object_id, app_id='ap-test'))
        self.controller.ledger.attach(operation, sandbox.object_id)
        self.controller.ledger.unknown(operation, dict(reason='Provider start unresolved'))
        self.controller.update(name, name=name, kind='train', operation=operation, status='unknown', seconds=240,
                               successful=False)
        self.assertJobs(([(name, 'uncertain')], [name]))
        with self.assertRaisesRegex(ValueError, 'unknown allocation'):
            self.train('blocked')
        return operation

    def test_recovery_settles_a_never_started_uncertain_job_at_zero(self):
        self.unresolved('a')
        self.controller.tick()
        self.join()
        self.assertJobs(([('a', 'failed')], []))
        self.assertEqual((self.row('a')['status'], self.row('a')['actual_seconds']), ('settled', 0))
        self.assertEqual(self.cloud.terminated, ['sb-0'])
        self.assertIn('never started', self.backend.log('a')['job_error'])
        self.cloud.start_after = 0
        self.train('b')
        self.join()
        self.assertEqual(self.jobs()[0][-1], ('b', 'complete'))

    def test_recovery_terminates_a_started_abandoned_worker_now_and_charges_its_receipt(self):
        operation = self.unresolved('a')
        sandbox = self.cloud.sandboxes['sb-0']
        sandbox.started_at, sandbox.started_read, self.cloud.runs_for = time.time(), 0, 10 ** 6  # it would run on
        self.controller.tick()
        self.join()
        self.assertEqual(self.cloud.terminated, ['sb-0'])
        self.assertJobs(([('a', 'failed')], []))
        self.assertEqual((self.row('a')['status'], self.row('a')['actual_seconds']), ('settled', 100.0))
        self.assertTrue((self.root / 'research/jobs' / operation / 'resolved.json').is_file())

    def test_stop_resolves_an_uncertain_job_instead_of_refusing(self):
        self.unresolved('a')
        self.controller.recovered_at = time.monotonic()  # no recovery pass of its own: the stop does it
        self.assertEqual(self.backend.stop('a')['stopping'], 'a')
        self.join()
        self.assertJobs(([('a', 'stopped')], []))
        self.assertEqual(self.row('a')['actual_seconds'], 0)
        self.assertIn('Stopped while queued', self.backend.log('a')['job_error'])

    def test_a_job_whose_record_lags_the_ledger_never_reads_failed(self):
        self.cloud.start_after = 0
        gate, complete, seen = threading.Event(), self.controller.complete, []

        def late(name, result):  # the ledger row is settled; the outcome record is not written yet
            seen.append(self.jobs())
            gate.wait(timeout=30)
            return complete(name, result)
        self.controller.complete = late
        self.train()
        deadline = time.time() + 30
        while not seen and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.row('a')['status'], 'settled')
        self.assertEqual(seen[0], ([('a', 'running')], ['a']))
        self.assertJobs(([('a', 'running')], ['a']))
        with self.assertRaisesRegex(ValueError, 'Resolve active work'):
            self.backend.finalize()
        gate.set()
        self.join()
        self.assertJobs(([('a', 'complete')], []))
        # A host that stopped inside that window: a reopen finishes the record from the provider's receipt.
        self.controller.complete = mock.Mock(side_effect=KeyboardInterrupt)
        self.train('b')
        with mock.patch.object(threading, 'excepthook'):
            self.join()
        self.assertJobs(([('a', 'complete'), ('b', 'running')], ['b']))
        self.controller.complete = complete
        self.controller.tick()
        self.join()
        self.assertJobs(([('a', 'complete'), ('b', 'complete')], []))


if __name__ == '__main__':
    unittest.main()
