"""The one researcher workspace, identical in shape for all six games.

    workspace/
      instructions.md   read-only task instructions
      game_rules/       read-only Markdown rules and lookup indexes
      simulator/        read-only public simulator source and engine dependencies
      training/         editable starter network and training code
      configs/          editable experiment configurations
      NOTES.md          the researcher's own notes

`stage` builds the tree inside a NEW release before it is sealed. The manifest
(inventory, checksums, native path mapping) stays outside the researcher root.
The source export covers the complete public engine trees for the pinned game
mode and excludes private evaluation state, weights, binaries and artwork.
No game/framework imports and no cloud work happen here.
"""
import ast
import copy
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = Path(__file__).resolve().parent / 'templates'
MARKER = 'workspace_manifest.json'
WORKSPACE = 'agent_workspace'
TOP = ('instructions.md', 'game_rules', 'simulator', 'training', 'configs', 'NOTES.md')
NOTES = 'NOTES.md'
CONFIGS = ('configs/train.json', 'configs/smoke.json')
SOURCE_SUFFIXES = {'.py', '.pyi', '.js', '.cjs', '.ts', '.rs', '.cpp', '.c', '.hpp', '.h', '.cu', '.cuh', '.java',
                   '.kt', '.kts', '.xml', '.properties', '.toml', '.lock', '.json', '.yaml', '.yml', '.md', '.txt',
                   '.sh', '.cmake', '.gradle', '.cff'}
SOURCE_NAMES = {'LICENSE', 'NOTICE', 'Makefile', 'CMakeLists.txt', 'build'}
EXCLUDE = {'.git', '__pycache__', 'node_modules', 'target', 'build', '.build', '.cache', 'dist', 'checkpoints',
           'weights', 'assets', 'sprites', 'logs', 'evidence', '.local', 'private', 'hidden', 'autoresearch'}
RULE_EXCLUDE = {'private', 'hidden', 'evidence', '.local', 'assets', 'checkpoints', '__pycache__'}


def _under(prefix, names):
    return tuple(prefix + name for name in names)


# Read-only public simulator source shown to the researcher, by repository path.
# A `release:` entry is read from the release being built instead.
SOURCE_GROUPS = {
    'generals': ('simulators/generals/__init__.py', 'simulators/generals/environment.py',
                 'simulators/generals/packed.py', 'vendor/engines/generals-bots/generals',
                 'vendor/engines/generals-bots/LICENSE',
                 # Public helpers the starter imports on the training worker.
                 'simulators/generals/autoresearch/rollout.py', 'simulators/generals/autoresearch/bundle.py'),
    'doudizhu': _under('simulators/doudizhu/', ('__init__.py', 'environment.py', 'model.py',
                                                'native_backend.py', 'native_rollout.py', 'graph.py',
                                                'artifacts.py', 'workers.py', 'rollout.py', 'native', 'cuda'))
                + ('vendor/engines/DouZero/douzero/env', 'vendor/engines/DouZero/LICENSE'),
    # Training runs on the native engine; evaluation plays on Showdown, whose view.cjs
    # is the reference observation encoder. The runtime files are the interface the
    # starter is written against.
    'pokemon': _under('simulators/pokemon/', ('env.py', 'view.cjs', 'worker.cjs', 'engine_fast.cjs', 'native',
                                              'efficient.py', 'network.py'))
               + _under('simulators/pokemon/autoresearch/runtime/',
                        ('sdk.py', 'env_client.py', 'fastenv.py', 'graph_export.py', 'graph_worker.py', 'bundle.py',
                         'supervisor.py', 'GRAPH_CONTRACT.md'))
               + _under('vendor/engines/pokemon-showdown/', ('sim', 'data', 'lib', 'config/formats.ts', 'package.json',
                                                             'tsconfig.json', 'build', 'LICENSE')),
    'clash': ('simulators/clash/native', 'simulators/clash/model.py', 'simulators/clash/rules.lock.json',
              # Trusted wrappers the starter imports on the training worker.
              'simulators/clash/autoresearch/fast_policy.py', 'simulators/clash/autoresearch/fast_adapter.py',
              'simulators/clash/autoresearch/fast_entry.py')
             + _under('vendor/engines/crforge/', ('core', 'env', 'python/crforge_env', 'tables', 'build.gradle.kts',
                                                  'settings.gradle.kts', 'gradle.properties', 'LICENSE', 'NOTICE')),
    # Polytopia workers run a pruned package built into each release. The researcher
    # reads that exact tree (and how a run is trained, loaded and played), not the
    # development sources it was derived from.
    'polytopia': ('release:runtime', 'release:v1/train.py', 'release:v1/policy.py', 'release:v1/grade.py',
                  'simulators/polytopia/THIRD_PARTY_NOTICES.md'),
    'tavern': _under('vendor/engines/AutoTavern/', ('crates', 'python', 'Cargo.toml', 'Cargo.lock', 'pyproject.toml',
                                                    'rust-toolchain.toml', 'LICENSE', 'NOTICE'))
              + ('simulators/tavern/adapter.py', 'simulators/tavern/rollout.py'),
    # The engine the starter imports (and the box's dry run runs), the BFS rules check and the opponent the
    # researcher may run: the shortest-path heuristic. MoHex is held out. Its source and build script stay
    # on the host, because a researcher who can run commands on a lease could otherwise build the engine there.
    'hex': ('simulators/__init__.py', 'simulators/hex/__init__.py', 'simulators/hex/engine.py',
            'simulators/hex/reference.py', 'simulators/hex/heuristic.py'),
}
GAMES = tuple(SOURCE_GROUPS)

# workspace path -> path inside the game's native trainer tree.
_NATIVE_PREFIX = {'tavern': 'starter/', 'clash': ''}
TRAINING = {game: {'training/' + name: _NATIVE_PREFIX.get(game, 'agent/') + name
                   for name in ('policy.py', 'learner.py', 'train.py')} for game in SOURCE_GROUPS}
TRAINING['polytopia'] = {
    'training/policy.py': 'src/polytopia_bot/rl/network.py',
    'training/learner.py': 'src/polytopia_bot/rl/ppo.py',
    'training/train.py': 'src/polytopia_bot/rl/gpu_resident.py',
    'training/collector.py': 'experiments/gpu_resident_train.py'}
SIMULATOR_README = ('# Simulator source\n\nRead-only public implementation and engine dependencies for this task. '
                    'Paths below preserve their repository names. No private banks, run state, weights, credentials, '
                    'artwork or compiled binaries are included. The game mode is defined in '
                    '../game_rules/README.md.\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def public(release):
    return Path(release) / 'public'


def enabled(release):
    return (public(release) / MARKER).is_file()


def catalog(root=ROOT):
    return json.loads((Path(root) / 'simulators/catalog.json').read_text())['games']


def task_dir(game, root=ROOT):
    return Path(root) / catalog(root)[game]['autoresearch'] / 'task'


def rulebook_files(game, root=ROOT, rules_dir=None):
    """Each game keeps its own chapter structure; every entry is Markdown."""
    base = Path(rules_dir) if rules_dir else task_dir(game, root) / 'game_rules'
    result = []
    for path in sorted(base.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(base)
        if RULE_EXCLUDE.intersection(relative.parts):
            continue
        if path.suffix != '.md':
            raise ValueError(f'Researcher rulebook must contain Markdown only: {path}')
        if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):
            raise ValueError(f'Rulebook entry escapes its folder: {path}')
        result.append(relative.as_posix())
    if 'README.md' not in result:
        raise ValueError(f'{game}: game_rules/README.md is required')
    return sorted(result, key=lambda name: (name != 'README.md', name))


def source_files(game, root=ROOT, release=None):
    """{path under simulator/: source file}. `release` resolves `release:` entries."""
    found = {}
    for name in SOURCE_GROUPS[game]:
        origin = Path(root)
        if name.startswith('release:'):
            if release is None:
                raise ValueError('This source group is read from a release: ' + name)
            origin, name = Path(release), name.removeprefix('release:')
        base = origin / name
        if not base.exists():
            raise ValueError('Missing simulator source root: ' + name)
        for path in ([base] if base.is_file() else sorted(base.rglob('*'))):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(origin)
            # Exclusions apply below a named root. The root itself is listed
            # deliberately (Showdown's `build` script, a game's public trainer helpers).
            if any(part in EXCLUDE for part in path.relative_to(base).parts[:-1]):
                continue
            if path.suffix not in SOURCE_SUFFIXES and path.name not in SOURCE_NAMES:
                continue
            found[relative.as_posix()] = path
    if not found:
        raise ValueError('Empty simulator source export: ' + game)
    return dict(sorted(found.items()))


SWITCH = re.compile(r'<!-- run -->\n(.*?)<!-- lease -->\n(.*?)<!-- end -->\n', re.S)


# A release whose simulator is one of the researcher's editable files (its manifest says `editable_simulator`)
# describes `simulator/` as such; the template's wording for a read-only simulator is replaced exactly.
EDITABLE_SIMULATOR_TEXT = (
    ('| `simulator/` | Complete public simulator implementation and engine dependencies | Read only |',
     '| `simulator/` | The game simulator your training runs on | Edit |'),
    ('`simulator/README.md` to locate the engine sources. Full source access reveals the\n'
     'implementation, not hidden live state, private evaluation data or opponent weights.',
     '`simulator/README.md` for what the simulator holds and how your training uses it. Nothing in\n'
     'the workspace reveals hidden live state, private evaluation data or opponent weights.'),
    ('everything and to edit the starter files in `training/` and `configs/` in place.\n'
     'Only those starter files, under their existing names, enter a training run:',
     'everything and to edit the starter files in `training/`, `simulator/` and `configs/` in\n'
     'place. Only those starter files, under their existing names, enter a training run:'),
    ('exercise your own modules, but not the game engine or GPU: training runs for real',
     'exercise your own modules and the simulator, but not the GPU: training runs for real'),
    ('Do not change or work around the frozen simulator, rules, observations, legal\n'
     'masks or scoring,',
     'Do not change or work around the frozen rules, observations, legal masks or\n'
     'scoring that practice and grading play by (your `simulator/` is your own copy),'),
)


def instructions(leases=False, editable_simulator=False):
    """Common, variant-neutral research instructions. A game with leases has no `ar run`: its researcher
    trains on workers it leases, so the passages about getting compute differ."""
    text = SWITCH.sub(lambda match: match.group(2 if leases else 1), (TEMPLATES / 'instructions.md').read_text())
    if editable_simulator:
        for old, new in EDITABLE_SIMULATOR_TEXT:
            if text.count(old) != 1:
                raise ValueError('instructions.md lost a passage the editable simulator replaces: ' + old[:60])
            text = text.replace(old, new)
    return text


def launcher(game):
    """`training/launch.py` of a game with leases: the shared script with this game's training command and
    worker shape written in. It is the researcher's own helper, outside the editable set that is trained."""
    settings = json.loads((task_dir(game) / 'launch.json').read_text())
    text = (TEMPLATES / 'launch.py').read_text()
    if text.count('\nSETTINGS = None\n') != 1:
        raise ValueError('launch.py lost its settings line')
    return text.replace('\nSETTINGS = None\n', '\nSETTINGS = ' + repr(settings) + '\n')


def _copy(source, target):
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise ValueError('Missing regular input file: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    target.chmod(0o444)


def stage(game, public_root, training_source=None, flat=False, withhold=None, views=None, withhold_paths=None,
          extra_editable=None, simulator_readme=None, rules=None):
    """Build the workspace inside a new release, before its outer manifest is sealed.

    `public_root` is the release's public directory, which already holds the
    game's native starter and recipes. `training_source` overrides where the
    native training files are read from; `flat` means they sit directly in it.

    `withhold` (a regular expression) leaves out of `simulator/` every source file whose text matches it, and
    `views` ({path under simulator/: text}) shows a file as that text instead of its source. A release whose
    starter is deliberately empty uses them to keep reference learning code out of the researcher's reading
    material; workers still run the full source. `withhold_paths` (a regular expression) leaves out every source
    file whose path under `simulator/` matches it, whatever its text. All three are recorded in the workspace manifest.

    `extra_editable` ({workspace path: path under `public_root`}) adds editable files outside `training/`, such as
    a simulator the researcher may change (the manifest then says `editable_simulator`); they travel with the
    training files wherever the editable payload goes. `simulator_readme` replaces `simulator/README.md` and `rules`
    (a folder) the game's own rulebook.
    """
    pattern = re.compile(withhold) if withhold else None
    path_pattern = re.compile(withhold_paths) if withhold_paths else None
    views = dict(views or {})
    public_root = Path(public_root)
    base = public_root / WORKSPACE
    sealed = any((public_root.parent / name).exists() for name in ('release.json', 'manifest.json'))
    if sealed or base.exists() or (public_root / MARKER).exists():
        raise FileExistsError('Never retrofit a prepared release; choose a fresh release directory')
    base.mkdir()
    training_source = Path(training_source) if training_source else public_root
    editable = copy.deepcopy(TRAINING[game])
    origins = {}
    for name, native in editable.items():
        source = training_source / (Path(native).name if flat else native)
        _copy(source, base / name)
        origins[name] = str(source)
    # Clash keeps its native single config at the public root, with the smoke one beside it.
    native = ('config.json', 'smoke-config.json') if game == 'clash' else ('recipes/starter.json', 'recipes/smoke.json')
    configs = {'configs/train.json': training_source / 'config.json' if flat else public_root / native[0],
               'configs/smoke.json': training_source / 'config.json' if flat else public_root / native[1]}
    if game == 'polytopia':
        configs['configs/architecture.json'] = public_root / 'recipes/starter_arch.json'
    for name, source in configs.items():
        _copy(source, base / name)
        origins[name] = str(source)
    for name, native in (extra_editable or {}).items():
        if not name.startswith('simulator/') or name in editable:
            raise ValueError('An extra editable file belongs under simulator/: ' + name)
        _copy(public_root / native, base / name)
        origins[name] = str(public_root / native)
        editable[name] = native
    rules = Path(rules) if rules else task_dir(game) / 'game_rules'
    for name in rulebook_files(game, rules_dir=rules):
        _copy(rules / name, base / 'game_rules' / name)
    # Editable files appear once, under training/, never as a stale read-only copy.
    shown_as_training = {Path(origins[name]).resolve() for name in editable}
    sources = source_files(game, release=public_root.parent)
    unknown = sorted(set(views) - set(sources))
    if unknown:
        raise ValueError('Workspace views name no simulator source: ' + ', '.join(unknown))
    withheld, shown_views = [], {}
    for name, source in sources.items():
        if source.resolve() in shown_as_training:
            continue
        target = base / 'simulator' / name
        if name in views:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(views[name])
            target.chmod(0o444)
            shown_views[name] = sha(target)
        elif (path_pattern and path_pattern.search(name)) or (pattern and pattern.search(source.read_text(errors='replace'))):
            withheld.append(name)
        else:
            _copy(source, target)
    for name, text in (('instructions.md', instructions(editable_simulator=bool(extra_editable))), ('simulator/README.md', simulator_readme or SIMULATOR_README),
                       (NOTES, '')):
        (base / name).write_text(text)
        (base / name).chmod(0o444)
    files = {p.relative_to(base).as_posix(): sha(p) for p in sorted(base.rglob('*')) if p.is_file()}
    record = {'schema': 2, 'game': game, 'files': files, 'writable': sorted([*editable, *configs]), 'notes': NOTES,
              'training_map': editable, 'source_groups': list(SOURCE_GROUPS[game]), 'source_origins': origins,
              'root': WORKSPACE}
    if extra_editable:
        record['editable_simulator'] = True
    if rules != task_dir(game) / 'game_rules':
        record['rules'] = rules.name
    if pattern or views or path_pattern:
        record.update(withhold_pattern=withhold, withheld=withheld, views=shown_views)
        if path_pattern:
            record['withhold_paths'] = withhold_paths
    record['sha256'] = digest(record)
    (public_root / MARKER).write_text(json.dumps(record, indent=2) + '\n')
    (public_root / MARKER).chmod(0o444)
    for path in sorted((p for p in base.rglob('*') if p.is_dir()), reverse=True):
        path.chmod(0o555)
    base.chmod(0o555)
    return record


def manifest(release):
    path = public(release) / MARKER
    if path.is_symlink() or not path.is_file():
        raise ValueError('This release has no researcher workspace; prepare a new release')
    data = json.loads(path.read_text())
    body = {k: v for k, v in data.items() if k != 'sha256'}
    if data.get('schema') != 2 or digest(body) != data.get('sha256'):
        raise ValueError('Workspace manifest changed or predates the current harness; prepare a new release')
    return data


def verify(release):
    """Every staged file still matches the sealed manifest."""
    meta = manifest(release)
    base = public(release) / WORKSPACE
    actual = {p.relative_to(base).as_posix(): sha(p) for p in sorted(base.rglob('*')) if p.is_file()}
    if actual != meta['files']:
        raise ValueError('Release workspace differs from its manifest')
    return meta


def safe(root, name):
    if not isinstance(name, str) or '\\' in name or '\x00' in name:
        raise ValueError('Invalid workspace path')
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Path outside workspace')
    target = Path(root).joinpath(*relative.parts)
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Path outside workspace')
    probe = target
    while probe != Path(root):
        if probe.is_symlink():
            raise ValueError('Symlinked workspace path')
        probe = probe.parent
    return target


def session_root(release, native_workspace):
    """The host's per-session copy of the workspace, next to the native trainer tree."""
    meta = manifest(release)
    base = public(release) / WORKSPACE
    session = Path(native_workspace).resolve().parent
    session.mkdir(parents=True, exist_ok=True)
    root = session / WORKSPACE
    binding = session / 'workspace-binding.json'
    expected = {'schema': 3, 'game': meta['game'], 'release': str(Path(release).resolve()),
                'native_workspace': str(Path(native_workspace).resolve()), 'workspace_sha256': meta['sha256'],
                'release_files': {name: sha(Path(release) / name)
                                  for name in ('release.json', 'manifest.json', 'binding.json', 'benchmark.json')
                                  if (Path(release) / name).is_file()}}
    with (session / 'workspace-binding.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if binding.is_symlink() or root.is_symlink():
            raise ValueError('Symlinked workspace identity')
        if binding.exists():
            if json.loads(binding.read_text()) != expected or not root.is_dir():
                raise ValueError('Workspace release/session binding differs; use a fresh session')
        else:
            if root.exists():
                raise ValueError('Unbound existing workspace; use a fresh session')
            # Publish only a complete tree: a crash between tree and binding
            # publication fails closed on restart instead of adopting a partial copy.
            with tempfile.TemporaryDirectory(prefix='.workspace-init-', dir=session) as temp:
                staged = Path(temp) / WORKSPACE
                shutil.copytree(base, staged)
                for name in (*meta['writable'], meta['notes']):
                    (staged / name).chmod(0o600)
                staged.chmod(0o755)  # macOS needs write permission on a directory being renamed.
                staged.rename(root)
                root.chmod(0o555)
            temporary = binding.with_suffix('.tmp')
            temporary.write_text(json.dumps(expected, sort_keys=True) + '\n')
            temporary.chmod(0o400)
            temporary.replace(binding)
    return root, meta


def validate(meta, files):
    """Check a researcher's editable payload without touching any tree.

    The payload is exactly the declared editable set: Python must parse and
    configs must be finite JSON. Nothing else ever enters a training run.
    """
    if not isinstance(files, dict) or set(files) != set(meta['writable']):
        raise ValueError('Exact declared editable file set required: ' + ', '.join(meta['writable']))
    total = 0
    for name, text in files.items():
        if not isinstance(text, str):
            raise ValueError('Text content required: ' + name)
        total += len(text.encode())
        if name.endswith('.py'):
            try:
                ast.parse(text, filename=name)
            except SyntaxError as exc:
                raise ValueError(f'{name}: line {exc.lineno}: {exc.msg}') from None
        else:
            try:
                json.dumps(json.loads(text), allow_nan=False)
            except ValueError as exc:
                raise ValueError(f'{name}: invalid JSON: {exc}') from None
    if total > 4 * 1024 * 1024:
        raise ValueError('Editable payload exceeds 4 MiB')
    return {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sorted(files.items())}


def apply(release, native_workspace, files):
    """Write a validated editable payload into the host's session workspace copy."""
    root, meta = session_root(release, native_workspace)
    hashes = validate(meta, files)
    for name, text in files.items():
        safe(root, name).write_text(text)
    return hashes


def editable(release, native_workspace):
    """The current editable payload held in the host's session workspace copy."""
    root, meta = session_root(release, native_workspace)
    return {name: safe(root, name).read_text() for name in meta['writable']}


def sync(controller, config=None):
    """Copy only the declared training/config payload into the game's native trainer tree.

    `controller` exposes `release` and `workspace` (the native trainer tree) and
    optionally `smoke` / `default_recipe`.
    """
    root, meta = session_root(controller.release, controller.workspace)
    smoke = getattr(controller, 'smoke', False) or getattr(controller, 'default_recipe', '').endswith('/smoke.json')
    if config is None:
        config = getattr(controller, '_workspace_config', None) or CONFIGS[1 if smoke else 0]
    if config not in CONFIGS:
        raise ValueError('Choose configs/train.json or configs/smoke.json')
    # Validate the entire payload before changing any native training file.
    for name in meta['files']:
        if name in meta['writable'] or name == meta['notes']:
            continue
        if sha(safe(root, name)) != meta['files'][name]:
            raise ValueError('Read-only source changed: ' + name)
    validate(meta, {name: safe(root, name).read_text() for name in meta['writable']})
    game = meta['game']
    if game == 'polytopia' and smoke:
        architecture = json.loads(safe(root, 'configs/architecture.json').read_text())
        if architecture.get('width', 32) != 32 or architecture.get('blocks', 2) != 2:
            raise ValueError('Smoke preserves the minimal 2x32 architecture')
    if game == 'clash':
        native_config = 'config.json'
    elif game == 'polytopia':
        native_config = 'recipe.json'
    else:
        native_config = getattr(controller, 'default_recipe', 'recipes/smoke.json' if smoke else 'recipes/starter.json')
    copies = {**meta['training_map'], config: native_config}
    if game == 'polytopia':
        copies['configs/architecture.json'] = 'arch.json'
    for name, native in copies.items():
        target = safe(controller.workspace, native)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.chmod(0o600)
        shutil.copyfile(safe(root, name), target)
    controller._workspace_config = config
    return config
