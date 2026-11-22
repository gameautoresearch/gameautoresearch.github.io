"""The Hex research contract: session defaults, opponents, editable and frozen files.

Mode `hex13-no-swap-v1`: 13x13 Hex without the swap rule, every evaluation pairing played from both
seats, graded against a ladder of opponents up to MoHex 2.0, which is fixed at 0 Elo. The method is that
of Jones, "Scaling Scaling Laws with Board Games" (2021); the board is larger than the paper's 9x9, which
research sessions saturated (reports/suite-2026-11-13.md), so MoHex here is a strong reference and not
perfect play.
"""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
GAME = HERE.parent
ROOT = HERE.parents[2]
EDITABLE = ('policy.py', 'learner.py', 'train.py')

# MoHex opponent levels. Every MoHex process starts with MOHEX_STARTUP; a level is a
# complete setting of the same LEVEL_KEYS, so one process can serve any level by
# re-sending its lines. Strength is set by simulated games per move, not wall time, so
# it does not depend on how busy the evaluation machine is; max_time is only a
# backstop. `mohex-anchor` is boardlaw's MoHexAgent() configuration (parallel DFPN
# solver on, knowledge threshold 0) with 50,000 games per move in place of MoHex's
# default 10-second clock. On 13x13 more games buy little: 10,000 games won 14 of 32
# against it, and a move costs 1 to 2 seconds of one core, which keeps practice affordable.
MOHEX_STARTUP = ('param_mohex num_threads 1', 'param_mohex max_time 300', 'param_mohex max_memory 400000000',
                 'param_mohex reuse_subtree 0', 'param_mohex ponder 0', 'param_game allow_swap 0')
LEVEL_KEYS = ('param_mohex max_games', 'param_mohex perform_pre_search', 'param_mohex knowledge_threshold',
              'param_mohex use_parallel_solver', 'param_dfpn threads')


def _level(games, presearch, knowledge, solver, dfpn_threads):
    return tuple(f'{key} {value}' for key, value in zip(LEVEL_KEYS, (games, presearch, knowledge, solver, dfpn_threads)))


MOHEX_LEVELS = {
    'mohex-100': _level(100, 0, 256, 0, 1),
    'mohex-2k': _level(2000, 1, 256, 0, 1),
    'mohex-anchor': _level(50000, 1, 0, 1, 2),
}
# Equal-weight opponents from random play to the anchor, 180 to 420 Elo apart, so the score rises
# with skill over the whole range. `epsilon` is the chance of a uniformly random move instead (the
# strength dial of boardlaw's MoHex wrapper). `elo` is each opponent's rating relative to the
# anchor, fitted by `ladder.py` from opponent-vs-opponent games on 13x13
# (reports/hex/panel-calibration-13x13-2026-11-14.json); grading uses it to estimate a candidate's
# own Elo. Practice and private grading use the same opponents on different openings; the
# private openings and seeds never leave the host.
PANEL = (
    {'name': 'random', 'kind': 'random', 'elo': -2580},
    {'name': 'heuristic-e90', 'kind': 'heuristic', 'epsilon': 0.9, 'elo': -2164},
    {'name': 'heuristic-e80', 'kind': 'heuristic', 'epsilon': 0.8, 'elo': -1852},
    {'name': 'heuristic-e60', 'kind': 'heuristic', 'epsilon': 0.6, 'elo': -1486},
    {'name': 'heuristic', 'kind': 'heuristic', 'elo': -1155},
    {'name': 'mohex-100-e15', 'kind': 'mohex', 'level': 'mohex-100', 'epsilon': 0.15, 'elo': -917},
    {'name': 'mohex-100', 'kind': 'mohex', 'elo': -498},
    {'name': 'mohex-2k', 'kind': 'mohex', 'elo': -177},
    {'name': 'mohex-anchor', 'kind': 'mohex', 'elo': 0},
)
ANCHOR = 'mohex-anchor'

DEFAULT = {
    'version': 'hex13-autoresearch-v1', 'mode': 'hex13-no-swap-v1',
    'gpu': 'L40S', 'budget_usd': 10.0, 'cpu_cores': 4, 'memory_mib': 16384,
    'min_run_seconds': 60,
    'policy_tensor_cap': 10_000_000, 'source_bytes_cap': 524288, 'artifact_bytes_cap': 134217728,
    'grading_points': [1 / 3, 2 / 3, 1.0],
    'practice_openings': 16, 'grade_openings': 64,
    'evaluation_cpu_cores': 8, 'evaluation_memory_mib': 16384,
    # The per-move inference cap: a policy call covering n positions (n <= inference_batch) must answer within
    # call_seconds + n * position_seconds of wall time, on one CPU thread.
    'inference_batch': 64, 'position_seconds': 0.01, 'call_seconds': 0.1,
    'policy_startup_seconds': 60, 'bootstrap_replicates': 2000,
    # Leases: workers of a shape the researcher chooses, running the researcher's own commands. These bound
    # what one session may hold at once and what one lease may ask for.
    'max_concurrent_jobs': 4, 'max_concurrent_gpus': 2, 'max_concurrent_cpu_cores': 64,
    'lease_max_cpu_cores': 32, 'lease_min_memory_mib': 2048, 'lease_max_memory_mib': 65536,
    'lease_command_bytes': 65536, 'lease_running_commands': 16,
    'allocation_rule': 'all allocated worker wall time, including setup, compile, simulation, transfers, '
                       'learning, practice, checkpoints, idle GPU and teardown, charged in US dollars at the list '
                       'price per second of the worker shape; failed allocations are charged',
    'evaluation': 'every 2-ply opening (a black stone, then a white stone) played twice, the candidate taking '
                  'black once and white once, against each panel opponent; opponents weigh equally',
    'submission': 'immutable checkpoint + exact policy source + model config; latest explicitly submitted run',
    'score': '100 x mean over opponents of the candidate win rate across both seats; the candidate Elo fitted '
             'against the calibrated panel and Elo against the MoHex anchor (fixed at 0) are reported alongside',
}

# One dollar budget for every worker shape. A worker second is charged what its shape costs at Modal's sandbox
# list prices in USD per second (modal.com/pricing, read 2026-11-13). These frozen prices are the contract's
# unit of account; Modal's invoice may differ from them.
PRICE = {'gpu': 0.000542, 'cpu_core': 0.00003942, 'memory_gib': 0.00000667}
TRAIN_SHAPE = {'gpu': 1, 'cpu_cores': DEFAULT['cpu_cores'], 'memory_mib': DEFAULT['memory_mib']}
PRACTICE_SHAPE = {'gpu': 0, 'cpu_cores': DEFAULT['evaluation_cpu_cores'], 'memory_mib': DEFAULT['evaluation_memory_mib']}


def _cost(shape):
    return (shape['gpu'] * PRICE['gpu'] + shape['cpu_cores'] * PRICE['cpu_core']
            + shape['memory_mib'] / 1024 * PRICE['memory_gib'])


def rate(shape):
    """US dollars charged per wall second of a worker of this shape."""
    return round(_cost(shape), 10)


def seconds_weight(shape):
    """Seconds-mode allocation seconds charged per wall second of a worker of this shape: its price relative
    to the training worker's (1.0 for the training worker), as `harness.budget.seconds_weight`."""
    return rate(shape) / rate(TRAIN_SHAPE)


# The two research budget modes a session's ledger can be opened in (harness/budget.py). `usd` is this
# contract's own: up to four jobs at once within `budget_usd`, each worker second charged `rate(shape)`.
# `seconds` runs one job at a time within `budget_seconds` of the training worker, each worker second charged
# `seconds_weight(shape)`. A ledger without `budget_mode` is a dollar session.
USD, SECONDS = 'usd', 'seconds'
SECONDS_RULE = ('all allocated worker wall time, including setup, compile, simulation, transfers, learning, '
                'practice, checkpoints, idle GPU and teardown, charged in seconds of the training worker: a '
                'worker of another shape is charged its seconds x its list price relative to the training '
                'worker; one job at a time; failed allocations are charged')


def budget_mode(config):
    return config.get('budget_mode', USD)


def lease_shape(config, gpu, cpu_cores, memory_gib):
    """A validated lease shape within the session's limits."""
    for value, what in ((gpu, '--gpu'), (cpu_cores, '--cpu'), (memory_gib, '--memory')):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(what + ' must be a whole number')
    c = dict(DEFAULT, **config)
    if gpu not in (0, 1):
        raise ValueError('--gpu is 0 or 1: a lease has at most one ' + c['gpu'])
    if not 1 <= cpu_cores <= c['lease_max_cpu_cores']:
        raise ValueError(f'--cpu must be between 1 and {c["lease_max_cpu_cores"]} cores')
    memory_mib = memory_gib * 1024
    if not c['lease_min_memory_mib'] <= memory_mib <= c['lease_max_memory_mib']:
        raise ValueError(f'--memory must be between {c["lease_min_memory_mib"] // 1024} and '
                         f'{c["lease_max_memory_mib"] // 1024} GiB')
    return {'gpu': gpu, 'cpu_cores': cpu_cores, 'memory_mib': memory_mib}


RUNTIME = ('__init__.py', 'engine.py', 'reference.py', 'heuristic.py', 'mohex.py', 'build_mohex.sh', 'source.lock.json')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_manifest():
    return {name: sha(GAME / name) for name in RUNTIME}


def validate_session(config):
    c = dict(DEFAULT, **config)
    if c['gpu'] != 'L40S' or c['cpu_cores'] != 4 or c['memory_mib'] != 16384:
        raise ValueError('The v1 resource contract is one L40S + 4 CPU cores + 16 GiB per run')
    if 'gpu_minutes' in c:
        raise ValueError('This session was opened under an earlier contract budgeted in seconds; the '
                         'dollar-budget contract cannot continue or grade it')
    mode = budget_mode(c)
    if mode == SECONDS:
        if 'budget_usd' in config:
            raise ValueError('A seconds-mode session is budgeted in seconds of the training worker, not dollars')
        del c['budget_usd']  # the contract's default dollar budget does not apply
        allocation = c.get('budget_seconds', 0)
        bound = 86400
    elif mode == USD:
        if 'budget_seconds' in c:
            raise ValueError('A dollar-mode session is budgeted in dollars, not seconds')
        allocation, bound = c['budget_usd'], 1000
    else:
        raise ValueError('budget_mode is usd or seconds')
    if not 0 < allocation <= bound or c['min_run_seconds'] <= 0:
        raise ValueError('Invalid bounded allocation')
    return c


def write_json(path, value, exclusive=False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if exclusive:
        with p.open('x') as f:
            f.write(data)
    else:
        tmp = p.with_name(p.name + '.tmp')
        tmp.write_text(data)
        tmp.replace(p)
