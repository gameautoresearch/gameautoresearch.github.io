"""Stage the pinned worker runtime of a new release.

`stage(repo)` writes into a NEW directory:
  engine-rs, engine-cu                    engine sources (built into the worker base image)
  src/polytopia_bot                       the Python package, pruned to the modules the trainer and the worker
                                          entrypoints (v1/train.py, grade.py, policy.py, preflight.py) import, with
                                          the starter files of autoresearch/starter/ in place of main's
  experiments/gpu_resident_train.py       (the starter's)
Never staged: the hidden reference recipes, our research modules, configs, docs, checkpoints or run logs.

`map_table()` and `panel()` return the two local inputs of a release from the git-ignored cache: the training map
table and the practice opponents of the current game mode (rl/mode.py). Both refuse a missing input and one made for
another mode, and say how to make it. Nothing here contacts a provider.
"""
import json
import shutil
from pathlib import Path

from .v1.core import MODE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAP_NAME = MODE.MAP_TABLE
CACHE = HERE / '.cache'
# The practice opponents are kept per game mode, so the checkpoints of another board are never picked up or replaced.
# The stronger panel of 2026-11-15 (owner's request): the old one (random, starter-20m/65m/200m) was saturated, every
# research network scored 100 and ~96 against its two weakest rungs. Bradley-Terry Elo against starter-200m in the open
# tournament of 2026-11-15 (runs/tournaments/polytopia-open-20261115): starter-1h +86, research-mid +128,
# research-top +190. The three new rungs are networks trained in earlier sessions of this game mode, copied unchanged
# from their sealed bundles (PROVENANCE); the old rungs stay in the cache for older releases.
PANEL = ('starter-200m', 'starter-1h', 'research-mid', 'research-top')
# The earlier ladder (until 2026-11-15): an untrained network, then the unmodified starter after 20, 65 and 200 million
# samples. Saturated by researchers who start from the PPO starter with an hour or more of final training, it spans
# what a short final from an empty starter reaches; a release chooses it with `prepare(..., panel='ladder')`.
LADDER = ('random', 'starter-20m', 'starter-65m', 'starter-200m')
PANELS = {'strong': PANEL, 'ladder': LADDER}
PROVENANCE = {
    'random': 'an untrained network of the starter architecture, a practice opponent of this game mode',
    'starter-20m': 'starter checkpoint after 20 million samples, trained for this game mode as a practice opponent',
    'starter-65m': 'starter checkpoint after 65 million samples, trained for this game mode as a practice opponent',
    'starter-200m': 'starter checkpoint after 200 million samples, trained for this game mode as a practice opponent',
    'starter-1h': 'the unmodified starter recipe trained fresh by the host for 3,600 s (polytopia-recipe-20261114, '
                  'Sol final, 337 M samples); open-tournament Elo +86 against starter-200m',
    'research-mid': 'research network of polytopia-20261114 (Sol, 48x3 normalized-v2, network variant); '
                    'open-tournament Elo +128 against starter-200m',
    'research-top': 'research network of polytopia-20261114 (Astra, 48x3 context-v3, network variant), the strongest '
                    'known network of this mode; open-tournament Elo +190 against starter-200m',
}
PANEL_DIR = CACHE / f'practice-{MODE.MAPS}-{MODE.SIZE}'
PANEL_FILES = ('final.npz', 'final.json', 'network.py', 'ar_arch.json')
MAP_TABLE_VOLUME = 'polytopia-map-tables'   # in the project environment (experiments/modal_map_table.py)
IGNORE = shutil.ignore_patterns('__pycache__', '*.pyc', 'target', 'build', '*.so', '.pytest_cache')


STARTER = HERE / 'starter'
# The empty starter's editable files (network.py, ppo.py, gpu_resident.py, experiments/gpu_resident_train.py), laid
# over STARTER's fixed plumbing (rl/mode.py, infra.py, data.py) when a release chooses `starter='empty'`.
STARTER_EMPTY = HERE / 'starter_empty'
WORKER = tuple(HERE / 'v1' / name for name in ('train.py', 'grade.py', 'policy.py', 'preflight.py'))
# where the import walk starts: the starter's trainer, the evaluator's and grader's modules
SEEDS = ('polytopia_bot.rl.gpu_resident', 'polytopia_bot.rl.network', 'polytopia_bot.rl.ppo', 'polytopia_bot.rl.infra',
         'polytopia_bot.rl.runner', 'polytopia_bot.rl.rust_engine', 'polytopia_bot.rl.elo', 'polytopia_bot.rl.gpu_engine',
         'polytopia_bot.maps.stream', 'polytopia_bot.maps.state')
# research modules that must never reach a worker or the researcher (a check, after the walk)
FORBIDDEN = ('distributed', 'vector_actor', 'inference_server', 'search', 'belief', 'four_player', 'full_rl', 'learning',
             'studio', 'policy', 'cpu_arena')


# comments in frozen modules that describe our research (architecture options, critic design, our training code):
# rewritten at staging; each target must exist, so a change in main fails the build instead of leaking
SCRUB = (
    ('rl/runner.py', "(docs/rules-alignment-2026-10-27.md); training is rl/distributed.py.", "; training is rl/gpu_resident.py."),
    ('rl/runner.py', "    extra_blocks: int = 0           # identity-initialized pre-activation blocks (network.grow); 0 for older checkpoints\n",
     "    extra_blocks: int = 0           # architecture fields of older checkpoints (the evaluator reads them)\n"),
    ('rl/runner.py', "    global_pool: bool = False       # extra blocks carry a global pooling bias (network.init global_pool)\n",
     "    global_pool: bool = False\n"),
    ('rl/runner.py', "    block_pool: int = 0             # global pooling in the trunk blocks (network.init block_pool): 0 none, 1 / true =\n"
                     "                                    # farsight's layout, 2 = the gated normalized layout (network.BLOCK_POOL_LAYOUTS)\n",
     "    block_pool: int = 0\n"),
    ('rl/runner.py', "    attention: int = 0              # transformer blocks over the tiles after the convolutions (network.init attention)\n",
     "    attention: int = 0\n"),
    ('rl/observation.py', "version 2: without the history; the omniscient critic\n    reads several views).",
     "version 2: without the history)."),
    ('rl/cpu_engine.py', "(both_views=True); for an omniscient critic", "(both_views=True)"),
    ('rl/rust_engine.py', "served=True (the policy runs on the node's inference server, rl/inference_server.py): ",
     "served=True (the policy runs elsewhere): "),
    ('ruleset.py', "which is how the old champions serve as evaluation references.", "so older checkpoints can serve as references."),
    # Retired-trainer metadata and comments that describe the reference trainer's design (all releases alike, so the
    # files that reach play stay identical between starters).
    ('rl/runner.py', 'The single-machine PPO loop that used to live here ran on the retired JAX engine and was removed on 2026-10-27\n; training is rl/gpu_resident.py.',
     'The trainer is rl/gpu_resident.py.'),
    ('rl/runner.py', "    environments: int = 8\n    horizon: int = 32\n    updates: int = 32\n    epochs: int = 2\n    minibatch: int = 64\n    learning_rate: float = 3e-4\n    entropy_coef: float = .01\n    gamma: float = 1.\n    gae_lambda: float = .95\n    target_kl: float = .03\n    bc_epochs: int = 2\n    pool_size: int = 32\n    eval_maps: int = 2\n    eval_actions: int = 512\n    training_seconds: int = 1200\n\n    def validate(self):\n        if self.reward_mode not in ('native', 'own-potential-v1'):\n            raise ValueError('unknown reward mode')\n        if self.architecture not in ('legacy', 'normalized-v2'):\n            raise ValueError('unknown architecture')\n        if self.obs_version not in (1, 2, 3):\n            raise ValueError('unknown observation version')\n        if int(self.block_pool) not in (0, 1, 2):\n            raise ValueError('unknown block_pool layout')\n        if self.size not in BOARD_SIZES or self.extra_blocks < 0 or self.attention < 0 or min(self.width, self.environments, self.horizon, self.updates,\n                self.epochs, self.minibatch, self.pool_size, self.eval_maps, self.eval_actions, self.training_seconds) < 1:\n            raise ValueError('invalid allocation')\n        if self.blocks < 0 or self.bc_epochs < 0 or self.environments * self.horizon % self.minibatch:\n            raise ValueError('invalid model or minibatch size')\n        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1 or self.learning_rate <= 0 or self.entropy_coef < 0 or self.target_kl <= 0:\n            raise ValueError('invalid PPO hyperparameters')\n",
     '    # Trainer settings a checkpoint records as metadata; the evaluator reads none of them, and a trainer passes its own.\n    epochs: int = 0\n    minibatch: int = 0\n    learning_rate: float = 0.\n    entropy_coef: float = 0.\n    gamma: float = 0.\n    gae_lambda: float = 0.'),
    ('rl/actions.py', '(family logits, per-tile source logits, per-family target embeddings)',
     '(family, source and target)'),
    ('rl/observation.py', "Accept observe()'s output, never State. Both actor and critic use this.",
     "Accept observe()'s output, never State."),
    ('rl/cpu_engine.py', 'per-seat features of a truncated final state (bootstrap)',
     'per-seat features of a truncated final state'),
)


def scrub(src):
    for rel, old, new in SCRUB:
        p = src / 'polytopia_bot' / rel
        text = p.read_text()
        if old not in text:
            raise SystemExit(f'scrub target missing in {rel}: {old[:60]!r} (update SCRUB)')
        p.write_text(text.replace(old, new))


def _imports(path, module):
    """polytopia_bot modules a file imports (static: import / from-import, relative or absolute)."""
    import ast
    pkg = module if path.name == '__init__.py' else module.rsplit('.', 1)[0]
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names if a.name.startswith('polytopia_bot')}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split('.')[:len(pkg.split('.')) - node.level + 1]
                mod = '.'.join(base + ([node.module] if node.module else []))
            elif node.module and node.module.startswith('polytopia_bot'):
                mod = node.module
            else:
                continue
            out.add(mod)
            out |= {f'{mod}.{a.name}' for a in node.names}
    return out


def prune(src, trainer=None):
    """Delete every module of the staged package that the seeds, the worker entrypoints and the staged trainer CLI
    (`trainer`, default the starter's) do not import."""
    def path_of(mod):
        p = src.joinpath(*mod.split('.'))
        return p.with_suffix('.py') if p.with_suffix('.py').exists() else (p / '__init__.py' if (p / '__init__.py').exists() else None)
    extra = [*WORKER, trainer or STARTER / 'experiments' / 'gpu_resident_train.py']
    stack = list(SEEDS) + [m for f in extra for m in _imports(f, 'x.y')]
    keep = set()
    while stack:
        mod = stack.pop()
        p = path_of(mod)
        if p is None or p in keep:
            continue
        keep.add(p)
        parts = mod.split('.')
        stack += ['.'.join(parts[:i]) for i in range(1, len(parts))]       # parent packages
        stack += list(_imports(p, mod))
    for p in sorted(src.rglob('*.py')):
        if p not in keep:
            p.unlink()
    for d in sorted((d for d in src.rglob('*') if d.is_dir()), reverse=True):   # packages left without modules go
        if d.exists() and not any(d.rglob('*.py')):
            shutil.rmtree(d)
    left = sorted(str(p.relative_to(src)) for p in src.rglob('*.py'))
    bad = [p for p in left if any(part.removesuffix('.py') in FORBIDDEN for part in Path(p).parts)]
    if bad:
        raise SystemExit(f'research modules would be staged: {bad}')
    return left


def map_table():
    """The training map table of the game mode. A missing table, or one of another mode, stops the release."""
    path = CACHE / MAP_NAME
    lo, hi = MODE.MAP_SEED_LOW, MODE.MAP_SEED_HIGH
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing the training map table {path}.\n'
            f'Build it once for this game mode (CPU workers in the project environment), then fetch it:\n'
            f'  cd simulators/polytopia && MODAL_PROFILE=duper MODAL_ENVIRONMENT=game-autoresearch-project \\\n'
            f'    ../../.venv-polytopia-ar/bin/modal run --env game-autoresearch-project '
            f'experiments/modal_map_table.py::build \\\n'
            f'    --mix {MODE.MAPS} --size {MODE.SIZE} --players {MODE.PLAYERS} --lo {lo} --hi {hi}\n'
            f'  MODAL_PROFILE=duper ../../.venv-polytopia-ar/bin/modal volume get --env game-autoresearch-project '
            f'{MAP_TABLE_VOLUME} \\\n    /maps/{MAP_NAME} {path}')
    import numpy as np
    with np.load(path, allow_pickle=False) as table:
        found = dict(lo=int(table['lo']), hi=int(table['hi']), types=[str(kind) for kind in table['types']],
                     maps=int(table['terrain'].shape[0]), tiles=int(table['terrain'].shape[1]),
                     players=int(table['capitals'].shape[1]))
    expected = dict(lo=lo, hi=hi, types=[MODE.MAPS], maps=hi - lo, tiles=MODE.SIZE ** 2, players=MODE.PLAYERS)
    if found != expected:
        raise ValueError(f'{path} is not the map table of this game mode: it holds {found}, the mode needs {expected}')
    return path


def panel(labels=PANEL):
    """The practice opponents of the game mode, {name: folder}: `labels` (PANEL, or LADDER). A missing checkpoint,
    a changed one, or one trained for another board stops the release."""
    import hashlib
    folders = {}
    for label in labels:
        folder = PANEL_DIR / label
        missing = [name for name in PANEL_FILES if not (folder / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f'Missing practice opponent {label}: {folder} lacks {", ".join(missing)}. The four opponents '
                f'({", ".join(labels)}) are trained for this game mode with the starter and placed under {PANEL_DIR}/, '
                f'each with {", ".join(PANEL_FILES)}; see "Practice opponents" in the README beside this file.')
        meta = json.loads((folder / 'final.json').read_text())
        if meta.get('sha256') != hashlib.sha256((folder / 'final.npz').read_bytes()).hexdigest():
            raise ValueError(f'Practice opponent {label}: final.npz does not match the digest in its final.json')
        config = meta.get('config', {})
        if (config.get('size'), config.get('players', 2)) != (MODE.SIZE, MODE.PLAYERS):
            raise ValueError(f"Practice opponent {label} was trained for a {config.get('size')}x{config.get('size')} "
                             f"board with {config.get('players', 2)} players; this game mode is "
                             f'{MODE.SIZE}x{MODE.SIZE} with {MODE.PLAYERS}. Retrain it and place it under {PANEL_DIR}/')
        folders[label] = folder
    return folders


def stage_source(repo, starter='ppo'):
    """The Python side only: the pruned, scrubbed package with the starter overlaid, and the starter's trainer CLI.
    `starter='empty'` lays the empty starter's editable files over the starter's."""
    if starter not in ('ppo', 'empty'):
        raise ValueError('starter is ppo or empty')
    repo = Path(repo)
    if (repo / 'src').exists() or (repo / 'experiments').exists():
        raise FileExistsError('Stage into a new directory')
    shutil.copytree(ROOT / 'src' / 'polytopia_bot', repo / 'src' / 'polytopia_bot', ignore=IGNORE)
    (repo / 'experiments').mkdir(parents=True)
    shutil.copytree(STARTER, repo, dirs_exist_ok=True, ignore=IGNORE)      # the minimal starter over main's files
    if starter == 'empty':
        shutil.copytree(STARTER_EMPTY, repo, dirs_exist_ok=True, ignore=IGNORE)
    modules = prune(repo / 'src', repo / 'experiments' / 'gpu_resident_train.py')
    scrub(repo / 'src')
    leaked = [p for p in repo.rglob('*') if 'hidden' in p.parts or p.name.startswith('reference')]
    if leaked:
        raise SystemExit(f'hidden files staged: {leaked}')
    return modules


def stage(repo, starter='ppo'):
    """The whole worker runtime: both engine source trees and the Python side."""
    repo = Path(repo)
    if repo.exists():
        raise FileExistsError('Stage into a new directory')
    shutil.copytree(ROOT / 'engine-rs', repo / 'engine-rs', ignore=IGNORE)
    shutil.copytree(ROOT / 'engine-cu', repo / 'engine-cu', ignore=IGNORE)
    return stage_source(repo, starter)
