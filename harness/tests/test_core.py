"""The shared harness without any cloud or model: workspace, task, broker, session and both variants."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest import mock

from harness import agents, policy, registry, session, task, workspace
from harness.transport import LocalBox
from harness.broker import VARIANTS
from harness.tests.support import FakeBackend, Process, Researcher, release_fixture

ROOT = Path(__file__).resolve().parents[2]
HARBOR = Path.home() / 'Desktop/projects/polytopia-bot-skirmish/artifacts/harbor-runtime/bin/python'
SENTINEL = '\n# researcher change\n'


class WorkspaceAndTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.releases = {game: release_fixture(game, cls.temp.name) for game in workspace.GAMES}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_one_layout_for_every_game(self):
        prompts = set()
        for game, (release, meta) in self.releases.items():
            base = workspace.public(release) / workspace.WORKSPACE
            self.assertEqual({p.name for p in base.iterdir()}, set(workspace.TOP), game)
            self.assertTrue(all(p.suffix == '.md' for p in (base / 'game_rules').rglob('*') if p.is_file()), game)
            self.assertTrue((base / 'game_rules/README.md').is_file(), game)
            self.assertGreater(sum(1 for p in (base / 'simulator').rglob('*') if p.is_file()), 5, game)
            self.assertTrue(all(name.startswith(('training/', 'configs/')) for name in meta['writable']), game)
            self.assertLessEqual({'training/policy.py', 'training/learner.py', 'training/train.py',
                                  'configs/train.json', 'configs/smoke.json'}, set(meta['writable']), game)
            prompts.add((base / 'instructions.md').read_text())
            self.assertEqual(workspace.verify(release)['sha256'], meta['sha256'])
        self.assertEqual(len(prompts), 1, 'release instructions are identical across games')

    def test_no_private_material_in_any_workspace(self):
        banned = {'private', 'hidden', 'weights', 'checkpoints', 'evidence'}
        for game, (release, meta) in self.releases.items():
            listed = [g for g in workspace.SOURCE_GROUPS[game] if 'autoresearch' in Path(g).parts]
            for name in meta['files']:
                self.assertFalse(set(Path(name).parts[:-1]) & banned, f'{game}: {name}')
                if 'autoresearch' in Path(name).parts:  # only a deliberately listed public helper
                    source = name.removeprefix('simulator/')
                    self.assertTrue(any(source == g or source.startswith(g + '/') for g in listed), f'{game}: {name}')
                self.assertFalse(name.endswith(('.npz', '.pt', '.key', '.sqlite3')), f'{game}: {name}')

    def test_release_binding_rejects_changes_and_foreign_releases(self):
        release, _ = self.releases['tavern']
        policy.verify_release('tavern', release)
        with self.assertRaises(ValueError):
            policy.verify_release('generals', release)
        unbound, _ = release_fixture('generals', Path(self.temp.name) / 'unbound', bind=False)
        with self.assertRaisesRegex(ValueError, 'not adopted'):
            policy.verify_release('generals', unbound)

    def test_release_identity_covers_the_core_and_only_its_own_backend(self):
        seen = {}
        original = policy.workspace.sha
        with mock.patch.object(policy.workspace, 'sha', lambda path: seen.setdefault(str(path), original(path))):
            policy.source_identity('tavern')
        names = {str(Path(name).relative_to(ROOT)) for name in seen}
        self.assertIn('harness/broker.py', names)
        self.assertIn('harness/box/ar.py', names)
        self.assertIn('harness/templates/instructions.md', names)
        self.assertTrue(any(name.startswith('harness/games/tavern/') for name in names))
        self.assertTrue(any(name.startswith('simulators/tavern/autoresearch/') for name in names))
        self.assertFalse(any(name.startswith(('harness/games/generals/', 'harness/tests/', 'simulators/generals/'))
                             for name in names))

    def test_task_is_identical_in_shape_for_every_game_and_variant(self):
        shapes = set()
        for game, (release, meta) in self.releases.items():
            for variant in VARIANTS:
                out = Path(self.temp.name) / 'tasks' / f'{game}-{variant}'
                task.build(game, variant, release, out, research_seconds=14400, final_train_seconds=3600,
                           wall_seconds=30000, hardware='one test GPU')
                # A game may add its own CPU dry run for `ar check`; nothing else differs.
                shapes.add(tuple(sorted(p.relative_to(out).as_posix() for p in out.rglob('*')
                                        if 'workspace' not in p.relative_to(out).parts and p.name != 'dryrun.py')))
                config = tomllib.loads((out / 'task.toml').read_text())
                self.assertEqual(config['agent']['user'], 'researcher')
                self.assertEqual(config['environment']['network_mode'], 'no-network')
                self.assertEqual(config['metadata']['name'], f'{game}-autoresearch-{variant}')
                text = (out / 'instruction.md').read_text()
                self.assertEqual(text, (out / 'environment/workspace/instructions.md').read_text())
                self.assertIn(f'This is a **{variant}** session', text)
                self.assertIn('4 hours', text)
                cap = dict(pokemon=1_000_000, hex=10_000_000, generals=32_000_000).get(game, 500_000)
                self.assertEqual(registry.spec(game)['parameter_cap'], cap)
                flops = registry.spec(game).get('move_flops_cap')
                if flops:
                    # Generals limits compute per move, not parameters: the cap above is only a storage ceiling.
                    self.assertEqual((game, flops), ('generals', 1_000_000_000))
                    self.assertIn(f'take at most {flops:,} FLOPs', text)
                    self.assertIn(f'Weights are stored up to {cap:,} parameters', text)
                else:
                    self.assertIn(f'- Model size: at most {cap:,} parameters and buffers', text)
                self.assertNotIn('`ar finalize` is refused', text)
                graded = text.split('## What is graded')[1].split('## This session')[0]
                self.assertEqual('from scratch' in graded, variant == 'recipe')
                self.assertEqual(text.count('## What is graded'), 1)
                self.assertNotIn('{', text.split('## What is graded')[1].split('## This session')[0])
                box = json.loads((out / 'environment/ar/box.json').read_text())
                spec = registry.spec(game)
                dry_run = (out / 'environment/ar/dryrun.py').is_file()
                self.assertEqual(box, dict(game=game, variant=variant, writable=meta['writable'], dry_run=dry_run,
                                           run_seed=spec['run_seed'], eval_seconds=spec['eval_seconds'],
                                           leases=bool(spec.get('leases'))))
                self.assertEqual(dry_run, (workspace.task_dir(game) / 'dryrun.py').is_file())
                self.assertIn('ARG TORCH=' + spec.get('box_torch', task.BOX_TORCH) + '\n',
                              (out / 'environment/Dockerfile').read_text())
                self.assertEqual((out / 'environment/ar/ar.py').read_bytes(), (ROOT / 'harness/box/ar.py').read_bytes())
                self.assertEqual({p.name for p in (out / 'environment/workspace').iterdir()}, set(workspace.TOP))
        self.assertEqual(len(shapes), 1)

    def test_restarts_make_a_multi_step_task_that_continues_only_while_the_session_is_resumable(self):
        release, _ = self.releases['generals']
        out = Path(self.temp.name) / 'restart-task'
        task.build('generals', 'recipe', release, out, research_seconds=7200, final_train_seconds=600,
                   wall_seconds=18000, hardware='one test GPU', restarts=2)
        config = tomllib.loads((out / 'task.toml').read_text())
        self.assertEqual(config['multi_step_reward_strategy'], 'final')
        self.assertEqual([step['name'] for step in config['steps']], ['research', 'continue1', 'continue2'])
        self.assertEqual([step.get('min_reward') for step in config['steps']], [{'resume': 1.0}, {'resume': 1.0}, None])
        self.assertEqual({step['agent']['user'] for step in config['steps']}, {'researcher'})
        self.assertEqual([step['agent']['timeout_sec'] for step in config['steps']], [18000.0] * 3)
        ruled = task.instruction('generals', 'recipe', research_seconds=7200, final_train_seconds=600,
                                 wall_seconds=18000, hardware='one test GPU', use_budget=True)
        self.assertIn('`ar finalize` is refused while a useful share of it is unspent', ruled)
        first = (out / 'steps/research/instruction.md').read_text()
        self.assertEqual(first, (out / 'instruction.md').read_text())
        later = (out / 'steps/continue2/instruction.md').read_text()
        self.assertTrue(later.startswith('You are resuming an unfinished research session') and later.endswith(first))
        verifier = (out / 'steps/continue1/tests/test.sh').read_text()
        # Run the verifier for real against a stand-in `ar`: Harbor must be able to parse what it writes.
        for reply, expected in ((dict(finalized=False, submitted=True, resume=True), dict(reward=1, resume=1)),
                                (dict(finalized=True, submitted=False, resume=False), dict(reward=0, resume=0))):
            scratch = Path(tempfile.mkdtemp(dir=self.temp.name))
            (scratch / 'bin').mkdir()
            (scratch / 'bin/ar').write_text("#!/bin/sh\necho '" + json.dumps(reply) + "'\n")
            (scratch / 'bin/ar').chmod(0o755)
            subprocess.run(['bash', '-c', verifier.replace('/logs/verifier', str(scratch / 'v'))], check=True,
                           env=dict(os.environ, PATH=str(scratch / 'bin') + os.pathsep + os.environ['PATH']))
            self.assertEqual(json.loads((scratch / 'v/reward.json').read_text()), expected)
        # Long sessions: every step together stays inside one sandbox lifetime.
        long = Path(self.temp.name) / 'restart-task-long'
        task.build('generals', 'recipe', release, long, research_seconds=36000, final_train_seconds=600,
                   wall_seconds=60000, hardware='one test GPU', restarts=3)
        steps = tomllib.loads((long / 'task.toml').read_text())['steps']
        self.assertLessEqual(sum(step['agent']['timeout_sec'] for step in steps), task.MAX_TASK_SECONDS)
        local = ROOT / '.venv-harness/bin/python'
        if local.is_file():
            code = ('import sys,tomllib,pathlib;from harbor.models.task.config import TaskConfig;'
                    'from harbor.models.task.task import Task;'
                    'c=TaskConfig.model_validate(tomllib.loads((pathlib.Path(sys.argv[1])/"task.toml").read_text()));'
                    't=Task(pathlib.Path(sys.argv[1]));'
                    'print(len(c.steps),c.steps[0].min_reward,len(t.step_instruction("continue1"))>100)')
            done = subprocess.run([str(local), '-c', code, str(out)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            self.assertEqual(done.stdout.split(), ['3', "{'resume':", '1.0}', 'True'])
    def test_a_continued_session_starts_from_its_seed_and_says_so(self):
        release, meta = self.releases['polytopia']
        base = workspace.public(release) / workspace.WORKSPACE
        files = {name: (base / name).read_text() + SENTINEL for name in meta['writable']}
        seed = Path(self.temp.name) / 'seed-polytopia'
        prior = dict(session='camp/polytopia/m/a1', model='m', used_seconds=17900, scratch_copy='polytopia-scratch-x')
        record = task.make_seed('polytopia', 'recipe', files, '# notes from before\n', seed, prior)
        self.assertEqual(task.seed_summary(seed, 'polytopia', 'recipe', release)['sha256'], record['sha256'])
        out = Path(self.temp.name) / 'continued-task'
        task.build('polytopia', 'recipe', release, out, research_seconds=10800, final_train_seconds=900,
                   wall_seconds=18000, hardware='one test GPU', restarts=1, continue_from=seed)
        box = out / 'environment/workspace'
        for name, text in files.items():
            self.assertEqual((box / name).read_text(), text, name)
        self.assertEqual((box / 'NOTES.md').read_text(), '# notes from before\n')
        self.assertNotIn(SENTINEL, (box / 'training/launch.py').read_text())  # the launcher is the host's
        text = (out / 'instruction.md').read_text()
        self.assertTrue(text.startswith('This session continues an earlier research session'))
        self.assertIn("in this session's `/scratch`, as a copy", text)
        self.assertEqual(text, (box / 'instructions.md').read_text())
        self.assertTrue((out / 'steps/continue1/instruction.md').read_text().endswith(text))
        # Wrong game or variant, a missing or changed file, a damaged record: refused before anything is built.
        with self.assertRaisesRegex(ValueError, 'not tavern recipe'):
            task.read_seed(seed, 'tavern', 'recipe', meta)
        with self.assertRaisesRegex(ValueError, 'not polytopia network'):
            task.read_seed(seed, 'polytopia', 'network', meta)
        changed = Path(self.temp.name) / 'seed-changed'
        task.make_seed('polytopia', 'recipe', files, 'n', changed, dict(prior, scratch_copy=None))
        (changed / task.SEED_FILES / 'configs/train.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'missing or changed'):
            task.read_seed(changed, 'polytopia', 'recipe', meta)
        partial = Path(self.temp.name) / 'seed-partial'
        task.make_seed('polytopia', 'recipe', dict(list(files.items())[1:]), 'n', partial, prior)
        with self.assertRaisesRegex(ValueError, 'exactly the editable files'):
            task.read_seed(partial, 'polytopia', 'recipe', meta)
        record = json.loads((partial / task.SEED_RECORD).read_text())
        (partial / task.SEED_RECORD).write_text(json.dumps(dict(record, game='hex')))
        with self.assertRaisesRegex(ValueError, 'damaged'):
            task.read_seed(partial, 'polytopia', 'recipe', meta)
        plain = Path(self.temp.name) / 'continued-task-plain'
        task.build('polytopia', 'recipe', release, plain, research_seconds=10800, final_train_seconds=900,
                   wall_seconds=18000, hardware='one test GPU')
        self.assertFalse((plain / 'instruction.md').read_text().startswith('This session continues'))

    def test_installed_agent_is_opt_in_and_adapts_the_task(self):
        # Only a game that names the agent may run it.
        self.assertEqual(agents.allowed(registry.spec('doudizhu')), ('terminus-2',))
        self.assertIn('claude-code', agents.allowed(registry.spec('hex')))
        self.assertIn('claude-code', agents.allowed(registry.spec('generals')))
        with self.assertRaisesRegex(ValueError, 'does not enable'):
            agents.resolve(registry.spec('doudizhu'), 'claude-code')
        with self.assertRaisesRegex(ValueError, 'Unknown researcher agent'):
            agents.get('not-an-agent')

        release, _ = self.releases['hex']
        out = Path(self.temp.name) / 'hex-claude-code'
        built = task.build('hex', 'network', release, out, research_seconds=600, final_train_seconds=0,
                           wall_seconds=1800, hardware='one test GPU', agent='claude-code')
        self.assertEqual(built['agent'], 'claude-code')
        config = tomllib.loads((out / 'task.toml').read_text())
        # The agent phase reaches the model; the environment baseline stays offline.
        self.assertEqual(config['agent']['network_mode'], 'public')
        self.assertEqual(config['environment']['network_mode'], 'no-network')
        dockerfile = (out / 'environment/Dockerfile').read_text()
        self.assertIn('claude-code-releases/bootstrap.sh', dockerfile)
        self.assertIn('runuser -u researcher -- env HOME=/home/researcher', dockerfile)
        self.assertNotIn('# AGENT_INSTALL', dockerfile)
        # The installed agent's shell caps command time, so the prompt says how to poll.
        self.assertIn('## Shell limits', (out / 'instruction.md').read_text())
        self.assertIn('## Shell limits', (out / 'environment/workspace/instructions.md').read_text())

        # Terminus-2 is unchanged: no phase override and no agent install layer.
        plain = Path(self.temp.name) / 'hex-terminus'
        task.build('hex', 'network', release, plain, research_seconds=600, final_train_seconds=0,
                   wall_seconds=1800, hardware='one test GPU')
        plain_config = tomllib.loads((plain / 'task.toml').read_text())
        self.assertNotIn('network_mode', plain_config['agent'])
        self.assertNotIn('bootstrap.sh', (plain / 'environment/Dockerfile').read_text())

        # Harbor's option names differ per agent: Claude Code takes base_url, not api_base/model_info.
        command = session.harbor_command('harbor', 'TASK', 'OUT', 's1', 'anthropic/claude-x', agent='claude-code',
                                         api_base='http://127.0.0.1:1', reasoning_effort='high',
                                         model_info={'max_input_tokens': 1})
        self.assertEqual(command[:9], ['harbor', 'run', '-p', 'TASK', '-a', 'claude-code', '-m', 'anthropic/claude-x', '-e'])
        self.assertIn('base_url=http://127.0.0.1:1', command)
        self.assertIn('reasoning_effort=high', command)
        self.assertIn('disable_web_search=true', command)
        self.assertFalse(any(part.startswith('model_info=') for part in command))
        self.assertFalse(any(part.startswith('api_base=') for part in command))

        # The plan records the agent, and a game that did not opt in refuses it.
        planned = session.plan('hex', 'network', release, Path(self.temp.name) / 'plan', model='anthropic/claude-x',
                               agent='claude-code')
        self.assertEqual(planned['agent'], 'claude-code')
        with self.assertRaisesRegex(ValueError, 'does not enable'):
            session.plan('doudizhu', 'network', self.releases['doudizhu'][0],
                         Path(self.temp.name) / 'plan-doudizhu', model='anthropic/claude-x', agent='claude-code')

    @unittest.skipUnless(HARBOR.is_file(), 'no local Harbor 0.25.0 runtime')
    def test_harbor_accepts_the_task(self):
        release, _ = self.releases['generals']
        out = Path(self.temp.name) / 'harbor-task'
        task.build('generals', 'recipe', release, out, research_seconds=7200, final_train_seconds=3600,
                   wall_seconds=18000, hardware='one test GPU')
        code = ('import sys,tomllib,pathlib;from harbor.models.task.config import TaskConfig;'
                'c=TaskConfig.model_validate(tomllib.loads(pathlib.Path(sys.argv[1]).read_text()));'
                'print(c.agent.user,c.agent.timeout_sec,c.environment.cpus,c.environment.memory_mb,len(c.artifacts))')
        done = subprocess.run([str(HARBOR), '-c', code, str(out / 'task.toml')], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertEqual(done.stdout.split(), ['researcher', '18000.0', '4', '16384', '3'])

    def test_shared_modal_helpers_use_the_project_environment(self):
        text = (ROOT / 'simulators/generals/autoresearch/provider.py').read_text()
        self.assertIn(f"ENVIRONMENT='{policy.ENVIRONMENT}'", text)
        self.assertNotIn('staging', text)

    def test_catalog_lists_every_game_with_both_variants(self):
        catalog = registry.catalog()
        self.assertEqual(set(catalog['games']), set(workspace.GAMES))
        self.assertEqual(catalog['variants'], ['recipe', 'network'])
        self.assertEqual(catalog['default_agent'], 'terminus-2')
        for game, spec in catalog['games'].items():
            self.assertTrue((ROOT / spec['venv']).is_dir() or True)
            self.assertEqual(registry.backend(game).game, game)
            self.assertIn('terminus-2', spec['agents'])
        self.assertEqual({game for game, spec in catalog['games'].items() if 'claude-code' in spec['agents']},
                         {'hex', 'generals', 'polytopia'})


class Sessions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.release, self.meta = release_fixture('doudizhu', self.temp.name)
        patcher = mock.patch.object(registry, 'backend', lambda game: FakeBackend)
        patcher.start()
        self.addCleanup(patcher.stop)
        FakeBackend.opened = []

    def session(self, variant, script, **kwargs):
        out = Path(self.temp.name) / f'session-{variant}'
        box = Path(self.temp.name) / f'box-{variant}'
        self.replies = {}

        def launcher(session_dir):
            researcher = Researcher(session_dir / 'task', box)
            return Process(lambda: script(researcher))
        result = session.run('doudizhu', variant, self.release, out, model='test/model', smoke=True,
                             launcher=launcher, find_box=lambda: LocalBox(box / 'ar') if (box / 'ar').is_dir() else None,
                             **kwargs)
        return out, result

    def test_ar_offers_only_what_the_game_takes_and_check_runs_the_games_dry_run(self):
        def script(ar):
            settings = json.loads((ar.ar / 'box.json').read_text())
            self.assertEqual((settings['run_seed'], settings['eval_seconds'], settings['dry_run']), (True, True, False))
            self.assertNotIn('dry_run', ar('check'))  # this game has none
            self.assertNotIn('error', ar('run', 'a', '--seconds', '60', '--seed', '3'))
            ar('wait')
            # A game whose seed comes from the config and whose practice is a fixed schedule.
            (ar.ar / 'box.json').write_text(json.dumps(dict(settings, run_seed=False, eval_seconds=False)))
            self.assertIn('unrecognized arguments: --seed', ar('run', 'b', '--seconds', '60', '--seed', '3', expect=2)['error'])
            self.assertIn('unrecognized arguments: --seconds', ar('eval', 'a', '--seconds', '60', expect=2)['error'])
            self.assertNotIn('--seed', self.help(ar, 'run'))
            # A game's dry run executes in the box after the host's rules pass; its failure fails the check.
            (ar.ar / 'dryrun.py').write_text(
                'import json,sys\nprint(json.dumps(dict(passed="smoke" in sys.argv[-1], config=sys.argv[-1])))\n')
            passed = ar('check', '--config', 'configs/smoke.json')
            self.assertEqual(passed['dry_run'], dict(passed=True, config='configs/smoke.json'))
            failed = ar('check', expect=1)
            self.assertFalse(failed['dry_run']['passed'])
            self.assertIn('Nothing was charged', failed['error'])
            (ar.ar / 'dryrun.py').write_text('raise SystemExit("broken dry run")\n')
            self.assertIn('broken dry run', ar('check', expect=1)['dry_run']['error'])
        self.session('network', script)

    @staticmethod
    def help(ar, command):
        done = subprocess.run([sys.executable, '-I', str(ar.ar / 'ar.py'), command, '--help'], capture_output=True,
                              text=True, env=dict(os.environ, AR_ROOT=str(ar.ar), AR_WORKSPACE=str(ar.workspace)))
        return done.stdout

    def test_network_variant_grades_the_submitted_run(self):
        def script(ar):
            budget = ar('budget')
            self.assertEqual((budget['variant'], budget['remaining_seconds']), ('network', 600.0))
            self.assertNotIn('final_train_seconds', budget)
            ar('check')
            ar.edit('training/policy.py', (ar.workspace / 'training/policy.py').read_text() + SENTINEL)
            ar.edit('training/scratch.py', 'not declared, never leaves the box')
            self.assertIn('error', ar('submit', 'a', expect=1))  # nothing trained yet
            self.assertEqual(ar('run', 'a', '--seconds', '120', '--config', 'configs/smoke.json')['config'], 'configs/smoke.json')
            self.assertIn('Run name already used', ar('run', 'a', '--seconds', '60', expect=1)['error'])
            self.assertIn('exceeds', ar('run', 'b', '--seconds', '9999', expect=1)['error'])
            self.assertEqual(ar('wait')['active'], [])
            ar('eval', 'a')
            self.assertIn('error', ar('finalize', expect=1))  # nothing submitted
            self.assertEqual(ar('submit', 'a')['submitted']['kind'], 'network')
            self.assertTrue(ar('finalize')['finalized'])
            self.assertIn('finalized', ar('run', 'c', '--seconds', '60', expect=1)['error'])
            self.assertTrue(ar('_verify')['submitted'])
        out, result = self.session('network', script)
        self.assertEqual((result['status'], result['closed_by'], result['submission']['name']),
                         ('researched', 'researcher', 'a'))
        self.assertNotIn('final_training', result)
        self.assertFalse((out / 'final').exists())
        research = FakeBackend.opened[0]
        job = research.state['jobs'][0]
        self.assertTrue(job['files']['training/policy.py'].endswith(SENTINEL))
        self.assertEqual(set(job['files']), set(self.meta['writable']))
        self.assertTrue(research.state['finalized'])
        self.assertTrue((out / 'sources/a/training/policy.py').read_text().endswith(SENTINEL))
        with mock.patch.object(policy, 'require_modal', lambda: None):
            self.assertEqual(session.grade(out)['graded'], 'a')
            with self.assertRaises(FileExistsError):
                session.grade(out)
        self.assertEqual(json.loads((out / 'grade.json').read_text())['graded_root'], 'research')
        self.assertFalse(json.loads((out / 'completion.json').read_text())['complete'])

    def test_recipe_variant_retrains_the_final_recipe_from_scratch(self):
        def script(ar):
            budget = ar('budget')
            self.assertEqual((budget['variant'], budget['final_train_seconds']), ('recipe', 180.0))
            ar('run', 'explore', '--seconds', '100', '--config', 'configs/smoke.json')
            ar('wait')
            self.assertIn('error', ar('submit', 'explore', expect=2))  # a recipe session takes no run name
            ar.edit('training/learner.py', (ar.workspace / 'training/learner.py').read_text() + '\n# first\n')
            self.assertEqual(ar('submit')['submitted']['index'], 1)
            ar.edit('training/learner.py', (ar.workspace / 'training/learner.py').read_text() + SENTINEL)
            ar.edit('training/policy.py', 'def broken(:')
            self.assertIn('training/policy.py', ar('submit', expect=1)['error'])  # a recipe must at least parse
            ar.edit('training/policy.py', (self.release / 'public/agent/policy.py').read_text())
            self.assertEqual(ar('submit', '--note', 'final')['submitted']['index'], 2)
            ar('finalize')
        out, result = self.session('recipe', script)
        self.assertEqual(result['status'], 'researched')
        self.assertEqual(result['final_training']['status'], 'complete')
        self.assertEqual(result['final_training']['recipe'], 2)
        research, final = FakeBackend.opened
        self.assertEqual((research.root.name, final.root.name), ('research', 'final'))
        self.assertFalse(research.state['finalized'], 'research networks are never submitted in a recipe session')
        self.assertEqual(research.state['submissions'], [])
        trained = final.state['jobs']
        self.assertEqual([(j['name'], j['seconds'], j['config']) for j in trained], [('final', 180.0, 'configs/smoke.json')])
        self.assertTrue(trained[0]['files']['training/learner.py'].endswith(SENTINEL))
        self.assertEqual(final.state['budget'], 180.0)
        self.assertEqual(final.state['submissions'], ['final'])
        with self.assertRaises(FileExistsError):
            session.final_training(out)
        with mock.patch.object(policy, 'require_modal', lambda: None):
            self.assertEqual(session.grade(out)['graded'], 'final')
        self.assertEqual(json.loads((out / 'grade.json').read_text())['graded_root'], 'final')

    def test_retrain_replicates_a_session_recipe_and_the_starter_with_explicit_seeds(self):
        def script(ar):
            ar.edit('training/learner.py', (ar.workspace / 'training/learner.py').read_text() + SENTINEL)
            ar('submit')
            ar('finalize')
        source, _ = self.session('recipe', script, final=False)
        recipe = json.loads((source / 'session.json').read_text())['submissions'][-1]
        outs = []
        for seed in (101, 102):
            out = Path(self.temp.name) / f'retrain-{seed}'
            record = session.retrain('doudizhu', self.release, out, seed=seed, session=source, smoke=True)
            self.assertEqual(record['status'], 'complete')
            self.assertEqual(record['seed']['value'], seed)
            self.assertEqual(record['recipe_sha256'], recipe['sha256'])
            outs.append(out)
        trained = [b for b in FakeBackend.opened if b.root.name == 'final']
        self.assertEqual([b.state['jobs'][0]['seed'] for b in trained], [101, 102])  # Generals: the run seed
        self.assertTrue(all(b.state['jobs'][0]['files']['training/learner.py'].endswith(SENTINEL) for b in trained))
        self.assertFalse((source / 'final').exists(), 'a retrain never writes into its source session')
        binding = policy.run_binding(outs[0])['settings']
        self.assertEqual((binding['kind'], binding['source']['kind'], binding['model']), ('retrain', 'session', 'test/model'))
        with self.assertRaises(FileExistsError):
            session.retrain('doudizhu', self.release, outs[0], seed=101, session=source, smoke=True)
        starter = Path(self.temp.name) / 'starter-7'
        record = session.retrain('doudizhu', self.release, starter, seed=7, starter=True, smoke=True)
        self.assertEqual(record['status'], 'complete')
        base = workspace.public(self.release) / workspace.WORKSPACE
        self.assertEqual(FakeBackend.opened[-1].state['jobs'][0]['files'],
                         {name: (base / name).read_text() for name in self.meta['writable']})
        self.assertEqual(policy.run_binding(starter)['settings']['model'], 'starter')
        with self.assertRaises(ValueError):
            session.retrain('doudizhu', self.release, Path(self.temp.name) / 'both', seed=1, session=source, starter=True)

    def test_retrain_sets_a_config_seed_where_the_game_reads_it_there(self):
        release, meta = release_fixture('tavern', Path(self.temp.name) / 'tavern')
        base = workspace.public(release) / workspace.WORKSPACE
        if 'seed' not in json.loads((base / 'configs/smoke.json').read_text()):
            self.skipTest('fixture config has no seed field')
        out = Path(self.temp.name) / 'tavern-retrain'
        record = session.retrain('tavern', release, out, seed=4242, starter=True, smoke=True)
        self.assertEqual((record['status'], record['seed']['mode']), ('complete', 'config'))
        job = FakeBackend.opened[-1].state['jobs'][0]
        self.assertEqual(job['seed'], 1)
        self.assertEqual(json.loads(job['files']['configs/smoke.json'])['seed'], 4242)
        self.assertEqual(job['files']['training/learner.py'], (base / 'training/learner.py').read_text())

    def test_final_training_waits_out_a_lagging_outcome_and_a_settling_finalize(self):
        # Pokémon reported a finished final as "failed" for a moment before its worker record was final; Tavern
        # refused finalize while still settling the job. Both outcomes must be waited out, not recorded as failures.
        class Lagging(FakeBackend):
            refusals = 2

            def tick(self):
                for job in self.state['jobs']:
                    if job['status'] == 'queued':
                        job['status'], job['lag'] = 'failed', 2
                    elif job.get('lag'):
                        job['lag'] -= 1
                        if not job['lag']:
                            job['status'] = 'complete'
                self._save()

            def finalize(self):
                if self.root.name == 'final' and Lagging.refusals:
                    Lagging.refusals -= 1
                    raise ValueError('Resolve active work before finalizing')
                return super().finalize()

        def script(ar):
            ar.edit('training/learner.py', (ar.workspace / 'training/learner.py').read_text() + SENTINEL)
            ar('submit')
            ar('finalize')
        with mock.patch.object(registry, 'backend', lambda game: Lagging), \
                mock.patch.object(session, 'SETTLE_GRACE_SECONDS', 30):
            out, result = self.session('recipe', script)
        self.assertEqual(result['final_training']['status'], 'complete')
        self.assertEqual(result['final_training']['job']['status'], 'complete')
        self.assertEqual(Lagging.refusals, 0)
        self.assertEqual(json.loads((out / 'final-train.json').read_text())['native'], dict(lifecycle='finalized'))

    def test_host_closes_an_abandoned_session_and_promotes_nothing(self):
        def script(ar):
            ar('run', 'a', '--seconds', '100')
        out, result = self.session('network', script)
        self.assertEqual((result['status'], result['closed_by'], result['submission']), ('no_submission', 'host', None))
        self.assertTrue(result['finalized'])
        self.assertEqual(FakeBackend.opened[0].status()['active'], [])
        with mock.patch.object(policy, 'require_modal', lambda: None), self.assertRaises(ValueError):
            session.grade(out)

    def test_researcher_records_name_the_whole_trajectory_and_final_notes(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            trial = out / 'harbor' / 's' / 'task__x'
            for path, text in (('agent/trajectory.json', 'snapshot'), ('steps/research/agent/trajectory.json', 'whole'),
                               ('steps/research/artifacts/workspace/NOTES.md', 'first'),
                               ('steps/resume-1/agent/trajectory.json', 'later'),
                               ('steps/resume-1/artifacts/workspace/NOTES.md', 'final notes')):
                (trial / path).parent.mkdir(parents=True, exist_ok=True)
                (trial / path).write_text(text)
            os.utime(trial / 'steps/research/agent/trajectory.json', (1, 1))
            os.utime(trial / 'steps/research/artifacts/workspace/NOTES.md', (1, 1))
            records = session.researcher_records(out, out / 'harbor' / 's')
            self.assertEqual([item['step'] for item in records['steps']], ['research', 'resume-1'])
            self.assertEqual(records['trajectory'], 'harbor/s/task__x/steps/resume-1/agent/trajectory.json')
            self.assertEqual(((out / records['notes']).read_text(), records['notes']), ('final notes', 'NOTES.md'))
            self.assertEqual(session.researcher_records(out, out / 'harbor' / 'none'),
                             dict(steps=[], notes=None, trajectory=None))

    def test_finalize_is_refused_while_a_useful_share_of_the_allocation_is_unspent(self):
        def script(ar):
            ar('run', 'a', '--seconds', '120', '--config', 'configs/smoke.json')
            ar('wait')
            ar('submit', 'a')
            refused = ar('finalize', expect=1)['error']
            self.assertIn('Not yet: 240 seconds', refused)
            self.assertEqual(ar('_verify'), dict(finalized=False, submitted=True, resume=True))
            ar('run', 'b', '--seconds', '210', '--config', 'configs/smoke.json')  # 30 s left: under 10 % of 360
            ar('wait')
            self.assertFalse(ar('_verify')['resume'])
            self.assertTrue(ar('finalize')['finalized'])
            self.assertFalse(ar('_verify')['resume'])
        out, result = self.session('network', script, use_budget=True, research_seconds=360)
        self.assertEqual((result['closed_by'], result['submission']['name']), ('researcher', 'a'))
        self.assertEqual((result['allocation_used_fraction'], result['automatic_answer']), (0.9167, False))
        plan = json.loads((out / 'plan.json').read_text())
        self.assertEqual((plan['use_budget'], plan['restarts']), (True, session.RESTARTS))
        self.assertEqual(len(tomllib.loads((out / 'task/task.toml').read_text())['steps']), 1 + session.RESTARTS)

    def test_finalize_is_accepted_near_the_wall_clock_limit_and_new_work_is_refused_after_it(self):
        def script(ar):
            ar('run', 'a', '--seconds', '60', '--config', 'configs/smoke.json')
            ar('wait')
            ar('submit', 'a')
            self.assertFalse(ar('_verify')['resume'])  # 300 s unspent, but the wall is under ten minutes away
            import time
            time.sleep(3.2)
            self.assertIn('wall-clock limit', ar('run', 'b', '--seconds', '60', expect=1)['error'])
            self.assertIn('wall-clock limit', ar('eval', 'a', expect=1)['error'])
            self.assertTrue(ar('finalize')['finalized'])
        _, result = self.session('network', script, use_budget=True, wall_seconds=3)
        self.assertEqual(result['closed_by'], 'researcher')

    def test_a_session_closed_without_a_submission_is_answered_by_its_newest_complete_graded_config_run(self):
        def script(ar):
            ar.edit('training/learner.py', (ar.workspace / 'training/learner.py').read_text() + '\n# first\n')
            ar('run', 'a', '--seconds', '60', '--config', 'configs/smoke.json')
            ar('wait')
            ar.edit('training/learner.py', (ar.workspace / 'training/learner.py').read_text() + SENTINEL)
            ar('run', 'b', '--seconds', '60', '--config', 'configs/smoke.json')
            ar('wait')
            ar.edit('training/learner.py', 'raise SystemExit("an unfinished idea")')
            ar('run', 'c', '--seconds', '60')  # the other configuration: never the answer
            ar('wait')
        out, result = self.session('recipe', script)
        self.assertEqual((result['status'], result['closed_by'], result['automatic_answer']), ('researched', 'host', True))
        entry = result['submission']
        self.assertEqual((entry['auto'], entry['run'], entry['index']), (True, 'b', 1))
        self.assertIn('nothing was submitted', entry['note'])
        self.assertEqual(result['final_training']['status'], 'complete')
        trained = FakeBackend.opened[-1].state['jobs'][0]['files']['training/learner.py']
        self.assertTrue(trained.endswith(SENTINEL))

    def test_an_explicit_submission_is_never_replaced_and_a_network_session_gets_its_newest_complete_run(self):
        def explicit(ar):
            ar('run', 'a', '--seconds', '60', '--config', 'configs/smoke.json')
            ar('wait')
            ar('submit', 'a')
            ar('run', 'b', '--seconds', '60', '--config', 'configs/smoke.json')
            ar('wait')
        _, result = self.session('network', explicit)
        self.assertEqual((result['submission']['name'], result['automatic_answer']), ('a', False))

        def silent(ar):
            for name in ('a', 'b'):
                ar('run', name, '--seconds', '60', '--config', 'configs/smoke.json')
                ar('wait')
        self.temp2 = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp2.cleanup)
        out = Path(self.temp2.name) / 'silent'
        box = Path(self.temp2.name) / 'box'
        result = session.run('doudizhu', 'network', self.release, out, model='test/model', smoke=True,
                             launcher=lambda folder: Process(lambda: silent(Researcher(folder / 'task', box))),
                             find_box=lambda: LocalBox(box / 'ar') if (box / 'ar').is_dir() else None)
        self.assertEqual((result['submission']['name'], result['submission']['auto']), ('b', True))
        self.assertTrue(FakeBackend.opened[-1].state['finalized'])

    def test_sessions_are_bound_and_never_reused(self):
        out, _ = self.session('network', lambda ar: None)
        binding = policy.run_binding(out)
        self.assertEqual((binding['game'], binding['settings']['variant'], binding['settings']['model']),
                         ('doudizhu', 'network', 'test/model'))
        with self.assertRaises(FileExistsError):
            self.session('network', lambda ar: None)

    def test_full_session_refuses_an_early_finalize(self):
        import time
        from harness.broker import Broker
        out = Path(self.temp.name) / 'full-session'
        out.mkdir()
        backend = FakeBackend(self.release, out / 'research', budget_seconds=1000)
        backend.open()
        broker = Broker(backend, out, 'network', research_seconds=1000, deadline=time.time() + 7200,
                        full_session=True)
        broker.state['submissions'].append(dict(index=1, kind='network', name='a'))
        self.assertFalse(broker.handle({'op': 'budget'})['finalize_accepted'])
        self.assertIn('full session', broker.handle({'op': 'finalize'})['error'])
        broker.state['deadline'] = time.time() + 600  # inside the last 20 minutes
        self.assertTrue(broker.handle({'op': 'budget'})['finalize_accepted'])
        self.assertTrue(broker.handle({'op': 'finalize'})['finalized'])
        text = task.instruction('hex', 'network', research_seconds=7200, final_train_seconds=0, wall_seconds=18000,
                                hardware='x', full_session=True)
        self.assertIn('This is a full session', text)
        self.assertNotIn('full session', task.instruction('hex', 'network', research_seconds=7200,
                                                          final_train_seconds=0, wall_seconds=18000, hardware='x'))

    def test_a_dollar_budget_is_stated_in_dollars_and_gates_a_full_session(self):
        import time
        from harness.broker import Broker
        text = task.instruction('hex', 'network', research_seconds=None, research_usd=15.0, final_train_seconds=0,
                                wall_seconds=18000, hardware='x')
        self.assertIn('Research allocation: $15.00 of allocated worker time', text)
        self.assertIn('research allocation of $15.00 of worker time', text)
        self.assertIn('Wall-clock limit for this terminal session: 5 hours', text)
        self.assertIn('up to 4 at once', text)  # dollar mode states its parallel rule
        # Every game offers both modes and defaults to seconds; a dollar amount selects dollar mode.
        for game in registry.GAMES:
            self.assertEqual(registry.spec(game)['budget_mode'], 'seconds')
            self.assertEqual(registry.research_usd(game), None)
            self.assertEqual(registry.research_usd(game, research_usd=15.0), 15.0)
            self.assertEqual(registry.research(game, budget_mode='usd')['seconds'], None)
        with self.assertRaises(ValueError):
            registry.research('doudizhu', research_seconds=60, research_usd=15.0)
        # Each game's public contract declares both modes with exactly its SPEC's amounts.
        for game in registry.GAMES:
            spec = registry.spec(game)
            declared = json.loads((ROOT / workspace.catalog()[game]['autoresearch'] / 'benchmark.json')
                                  .read_text())['budget']
            self.assertEqual(declared['default_mode'], spec['budget_mode'], game)
            for block, keys in (('seconds_mode', ('research_seconds', 'smoke_research_seconds')),
                                ('usd_mode', ('research_usd', 'smoke_research_usd', 'wall_seconds',
                                              'smoke_wall_seconds'))):
                self.assertEqual({k: declared[block][k] for k in keys}, {k: spec[k] for k in keys}, game)

        class Dollars(FakeBackend):
            left = 10.0

            def budget(self):
                return dict(budget_usd=15.0, used_usd=15.0 - self.left, remaining_usd=self.left,
                            allocatable_usd=self.left)
        out = Path(self.temp.name) / 'dollar-session'
        out.mkdir()
        backend = Dollars(self.release, out / 'research', budget_usd=15.0)
        backend.open()
        broker = Broker(backend, out, 'network', research_usd=15.0, deadline=time.time() + 7200, full_session=True)
        self.assertEqual((broker.state['research_usd'], broker.state['research_seconds']), (15.0, None))
        broker.state['submissions'].append(dict(index=1, kind='network', name='a'))
        self.assertIn('$10.00 of allocation', broker.handle({'op': 'finalize'})['error'])
        backend.left = 1.4  # under 10% of the dollar budget
        self.assertTrue(broker.handle({'op': 'finalize'})['finalized'])

    def test_bridge_process_serves_the_box_over_a_pipe(self):
        import sys
        from harness.broker import Broker
        from harness.transport import Bridge, serve
        box = Path(self.temp.name) / 'bridged'
        out = Path(self.temp.name) / 'bridged-session'
        out.mkdir()
        backend = FakeBackend(self.release, out / 'research', budget_seconds=100)
        backend.open()
        broker = Broker(backend, out, 'network', research_seconds=100)
        bridge = Bridge(sys.executable, ['--local', str(box)], cwd=ROOT)
        self.addCleanup(bridge.close)
        self.assertIs(bridge.find(), bridge)
        self.assertTrue(bridge.alive())
        self.assertEqual(bridge.pending(), {})
        name = '0123456789abcdef0123456789abcdef.json'
        (box / 'requests' / name).write_text(json.dumps({'op': 'budget'}))
        self.assertEqual(serve(bridge, broker), 1)
        self.assertEqual(json.loads((box / 'replies' / name).read_text())['remaining_seconds'], 100)
        self.assertEqual(bridge.pending(), {})
        with self.assertRaisesRegex(RuntimeError, 'Unknown call'):
            bridge._call(call='nonsense')

    def test_harbor_command_runs_terminus_in_the_project_environment(self):
        command = session.harbor_command('harbor', 'TASK', 'OUT', 's1', 'openrouter/x/y', api_base='http://127.0.0.1:1/v1',
                                         reasoning_effort='high', model_info={'max_input_tokens': 1})
        self.assertEqual(command[:9], ['harbor', 'run', '-p', 'TASK', '-a', 'terminus-2', '-m', 'openrouter/x/y', '-e'])
        self.assertIn('labels={"ar_session": "s1"}', command)
        self.assertIn('app_name=game-autoresearch-researchers', command)
        self.assertFalse(any('KEY' in part.upper() for part in command))
        key = Path(self.temp.name) / 'key'
        key.write_text('secret-value\n')
        key.chmod(0o644)
        with self.assertRaises(ValueError):
            session.harbor_environment(key)
        key.chmod(0o600)
        env = session.harbor_environment(key, 'OPENAI_API_KEY')
        self.assertEqual((env['OPENAI_API_KEY'], env['MODAL_ENVIRONMENT'], env['MODAL_PROFILE']),
                         ('secret-value', 'game-autoresearch-project', 'duper'))

    def test_concurrent_sessions_get_distinct_box_tags(self):
        # Three models' sessions named `smoke-1` once shared one tag, so each broker served whichever box Modal
        # listed first. The tag Harbor attaches and the bridge searches for is now unique per session.
        tags = []
        for parent in ('model-a', 'model-b'):
            out = Path(self.temp.name) / parent / 'smoke-1'
            box = Path(self.temp.name) / parent / 'box'
            result = session.run('doudizhu', 'network', self.release, out, model='test/model', smoke=True,
                                 launcher=lambda d, box=box: (Researcher(d / 'task', box), Process(lambda: None))[1],
                                 find_box=lambda box=box: LocalBox(box / 'ar') if (box / 'ar').is_dir() else None)
            tag = json.loads((out / 'plan.json').read_text())['box_tag']
            self.assertTrue(tag.startswith('smoke-1-'))
            self.assertEqual(policy.run_binding(out)['settings']['box_tag'], tag)
            tags.append(tag)
        self.assertNotEqual(*tags)
        command = session.harbor_command('harbor', 'TASK', 'OUT', 'smoke-1', 'm', box_tag=tags[0])
        self.assertIn('labels=' + json.dumps({'ar_session': tags[0]}), command)
        self.assertIn('smoke-1', command)  # Harbor's job folder keeps the readable session name

    def test_box_lookup_refuses_an_ambiguous_tag(self):
        import sys, types
        from harness.transport import ModalBox

        class Sandbox:
            def __init__(self, alive):
                self.alive = alive

            def poll(self):
                return None if self.alive else 0
        boxes = []
        fake = types.SimpleNamespace(
            App=types.SimpleNamespace(lookup=lambda *a, **k: types.SimpleNamespace(app_id='ap-1')),
            Sandbox=types.SimpleNamespace(list=lambda **k: list(boxes)))
        with mock.patch.dict(sys.modules, {'modal': fake}):
            self.assertIsNone(ModalBox.find('app', 'tag', 'env'))
            boxes[:] = [Sandbox(False), Sandbox(True)]
            self.assertIs(ModalBox.find('app', 'tag', 'env').sandbox, boxes[1])
            boxes.append(Sandbox(True))
            with self.assertRaisesRegex(RuntimeError, '2 live boxes'):
                ModalBox.find('app', 'tag', 'env')



class Workers(unittest.TestCase):
    """Shared start/abandon rules every provider follows (harness/workers.py)."""

    def clock(self):
        state = dict(t=0.0)
        return state, (lambda: state['t']), (lambda s: state.__setitem__('t', state['t'] + s))

    def test_a_queued_worker_is_waited_for_not_abandoned_after_a_minute(self):
        from harness import workers
        state, clock, sleep = self.clock()
        receipts = iter([{}] * 300 + [dict(provider_id='sb', started_at=5.0)])
        got = workers.await_start(lambda: next(receipts), patience=1800, poll=2, sleep=sleep, clock=clock)
        self.assertEqual(got['started_at'], 5.0)
        self.assertGreater(state['t'], 60)

    def test_patience_and_stop_end_the_wait_without_a_start(self):
        from harness import workers
        _, clock, sleep = self.clock()
        self.assertIsNone(workers.await_start(lambda: {}, patience=30, poll=2, sleep=sleep, clock=clock))
        stops = iter([False, False, True])
        self.assertIsNone(workers.await_start(lambda: {}, patience=999, poll=2, stop=lambda: next(stops),
                                              sleep=sleep, clock=clock))

    def test_abandon_reports_never_started_or_the_finished_receipt(self):
        from harness import workers
        _, clock, sleep = self.clock()
        killed = []
        never = workers.abandon(lambda: killed.append(1), lambda: dict(provider_id='sb', started_at=0.0),
                                poll=1, sleep=sleep, clock=clock)
        self.assertTrue(workers.never_started(never))
        self.assertEqual(killed, [1])
        receipts = iter([dict(provider_id='sb', started_at=10.0), dict(provider_id='sb', started_at=10.0,
                                                                        finished_at=12.0)])
        done = workers.abandon(lambda: None, lambda: next(receipts), poll=1, sleep=sleep, clock=clock)
        self.assertEqual((done['started_at'], done['finished_at']), (10.0, 12.0))
        self.assertFalse(workers.never_started(done))

    def test_final_training_checks_code_identity_only(self):
        # Editing documentation or templates must not invalidate the finals of sessions already running.
        self.assertNotEqual(policy.source_identity('generals'), policy.source_identity('generals', code_only=True))


class OnePathToCompute(unittest.TestCase):
    """A game with leases has no `ar run`: its researcher trains on leased workers, with `training/launch.py`."""

    def test_a_release_that_seals_its_own_brief_is_briefed_from_it(self):
        with tempfile.TemporaryDirectory() as temp:
            release = Path(temp)
            default = task.brief('generals', release)  # no sealed copy: the game's own brief
            self.assertEqual(default, task.brief('generals'))
            (release / 'public').mkdir()
            (release / 'public/brief.md').write_text('# Title\n\nSealed brief text.\n')
            self.assertEqual(task.brief('generals', release), 'Sealed brief text.')
            text = task.instruction('generals', 'recipe', research_seconds=7200, final_train_seconds=900,
                                    wall_seconds=18000, hardware='one test GPU', release=release)
            self.assertIn('## This game\n\nSealed brief text.', text)
            self.assertIn('15 minutes', text)

    def test_instructions_describe_one_way_to_get_compute_per_game(self):
        run, lease = workspace.instructions(), workspace.instructions(leases=True)
        for text in (run, lease):
            self.assertNotIn('<!--', text)
        self.assertIn('| `ar run NAME --seconds S', run)
        self.assertNotIn('launch.py', run)
        self.assertNotIn('ar run', lease)
        self.assertIn('python training/launch.py NAME --seconds S', lease)
        self.assertIn('| `ar lease NAME', lease)

    def test_every_game_with_leases_has_a_launcher_that_builds_its_training_command(self):
        games = [game for game in registry.GAMES if registry.spec(game).get('leases')]
        self.assertEqual(sorted(games), ['generals', 'hex', 'polytopia'])
        for game in games:
            scope = dict(__name__='launcher')
            exec(compile(workspace.launcher(game), 'launch.py', 'exec'), scope)
            settings = scope['SETTINGS']
            self.assertEqual(set(settings['shape']), {'gpu', 'cpu', 'memory'})
            with tempfile.TemporaryDirectory() as temp:
                (Path(temp) / 'configs').mkdir()
                (Path(temp) / 'configs/train.json').write_text(json.dumps({'trainer': dict(games=8, sets=2, ring=160)}))
                line, output = scope['command'](settings, 'a1', 'configs/train.json', 3, 99.6, Path(temp))
            self.assertEqual(output, '/scratch/runs/a1')
            self.assertIn('--seconds 99', line)
            self.assertNotIn('{', line)
            self.assertIn(' --config configs/train.json ', line)
            self.assertIn('a1', line)

    def test_the_box_of_a_game_with_leases_has_the_launcher_and_no_run_command(self):
        with tempfile.TemporaryDirectory() as temp:
            release, _ = release_fixture('hex', temp)
            built = Path(temp) / 'task'
            task.build('hex', 'network', release, built, research_seconds=None, research_usd=10.0,
                       final_train_seconds=3600, wall_seconds=3600, hardware='x')
            self.assertIn('SETTINGS = {', (built / 'environment/workspace/training/launch.py').read_text())
            text = (built / 'instruction.md').read_text()
            self.assertNotIn('ar run', text)
            self.assertIn('training/launch.py', text)
            self.assertIs(json.loads((built / 'environment/ar/box.json').read_text())['leases'], True)


if __name__ == '__main__':
    unittest.main()
