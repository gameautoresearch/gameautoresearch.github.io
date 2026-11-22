"""The Polytopia session contract: research budget, worker shapes, list prices and lease limits.

The benchmark (game mode, bundle manifests, frozen configuration fields) is `v1/core.py` and keeps its id, so
sealed bundles and practice opponents still verify. This file is the session contract around it, built the way
Generals' and Hex's are.

A session's research budget is in one of the harness's two modes (harness/budget.py), recorded in its ledger
config as `budget_mode`:
`usd`      (DEFAULT, the contract's own) US dollars of worker time, each worker second at its shape's list
           price; up to 4 jobs at once, at most 2 with a GPU and 64 CPU cores.
`seconds`  (SECONDS_DEFAULT) seconds of the training worker: a second of the training worker is a second of
           allocation and a lease's second is weighted by its price relative to the training worker's; one
           training run or lease at a time.
Either way public practice runs on separate CPU workers outside the budget, one practice job at a time.
"""
import math

from simulators.polytopia.autoresearch.v1.core import BENCHMARK

DEFAULT = {
    'version': 'polytopia-autoresearch-v2', 'benchmark_id': BENCHMARK,
    'gpu': 'L40S', 'budget_mode': 'usd', 'budget_usd': 23.0, 'cpu_cores': 4, 'memory_mib': 16384,
    # A run spends ~130 s compiling and allocating before its first update, and the provider keeps ~130 s of every
    # grant for worker setup, collecting the run's files and teardown: a 300 s run has about half a minute of updates.
    'min_run_seconds': 300, 'min_lease_seconds': 60,
    'policy_parameter_cap': 500_000, 'critic_parameter_cap': 500_000,
    'artifact_bytes_cap': 52428800, 'metadata_bytes_cap': 1048576,
    'grading_points': [1 / 3, 2 / 3, 1.0],
    'practice_cpu_cores': 4, 'practice_memory_mib': 16384,
    # Leases: workers of a shape the researcher chooses, running the researcher's own commands. These bound
    # what one session may hold at once and what one lease may ask for.
    'max_concurrent_jobs': 4, 'max_concurrent_gpus': 2, 'max_concurrent_cpu_cores': 64,
    'lease_max_cpu_cores': 32, 'lease_min_memory_mib': 2048, 'lease_max_memory_mib': 65536,
    'lease_command_bytes': 65536, 'lease_running_commands': 16,
    'allocation_rule': 'all allocated worker wall time of training runs and leases, including setup, compile, '
                       'simulation, transfers, learning, checkpoints, idle GPU and teardown, charged in US dollars '
                       'at the list price per second of the worker shape; failed allocations are charged. Public '
                       'practice and private grading run on separate CPU workers outside the budget',
    'submission': 'latest explicitly submitted sealed bundle of a completed or packaged run: final checkpoint, '
                  'exact policy source and architecture; finalization requires no active or unresolved jobs',
}

# One dollar budget for every worker shape. A worker second is charged what its shape costs at Modal's sandbox
# list prices in USD per second (modal.com/pricing, read 2026-11-13; the prices Generals and Hex froze). These
# frozen prices are the contract's unit of account; Modal's invoice may differ from them.
PRICE = {'gpu': 0.000542, 'cpu_core': 0.00003942, 'memory_gib': 0.00000667}
TRAIN_SHAPE = {'gpu': 1, 'cpu_cores': DEFAULT['cpu_cores'], 'memory_mib': DEFAULT['memory_mib']}
# Practice, private grading and the parameter check run the trusted evaluator on CPU workers of this shape.
PRACTICE_SHAPE = {'gpu': 0, 'cpu_cores': DEFAULT['practice_cpu_cores'], 'memory_mib': DEFAULT['practice_memory_mib']}
# A dollar session records the prices it is charged at.
DEFAULT['price'] = dict(PRICE)
# The seconds mode of the same contract: an allocation in seconds of the training worker, one job at a time (the
# 14,400 seconds the small form was given before the dollar contract). Its charging weights are
# harness.budget.seconds_weight(shape, TRAIN_SHAPE), each recorded with its job.
SECONDS_DEFAULT = {key: value for key, value in DEFAULT.items() if key not in ('budget_usd', 'price')}
SECONDS_DEFAULT.update({
    'budget_mode': 'seconds', 'budget_seconds': 14400.0, 'train_shape': dict(TRAIN_SHAPE),
    'max_concurrent_jobs': 1, 'max_concurrent_gpus': 1, 'max_concurrent_cpu_cores': None,
    'allocation_rule': 'all allocated worker wall time of training runs and leases, including setup, compile, '
                       'simulation, transfers, learning, checkpoints, idle GPU and teardown, in seconds of the '
                       'training worker: a training run is charged its worker seconds, a lease its worker seconds '
                       'times its shape\'s list price relative to the training worker\'s; failed allocations are '
                       'charged. Public practice and private grading run on separate CPU workers outside the budget',
})
MODES = {'usd': DEFAULT, 'seconds': SECONDS_DEFAULT}


def rate(shape):
    """US dollars charged per wall second of a worker of this shape."""
    return round(shape['gpu'] * PRICE['gpu'] + shape['cpu_cores'] * PRICE['cpu_core']
                 + shape['memory_mib'] / 1024 * PRICE['memory_gib'], 10)


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


def validate_session(config):
    mode = config.get('budget_mode', 'usd')
    if mode not in MODES:
        raise ValueError('budget_mode is usd or seconds')
    c = dict(MODES[mode], **config)
    # 4 cores and 16 GiB since 2026-11-14: the engine and the learner both run on the GPU (the earlier shape was 16
    # cores and 64 GiB, about twice the price; see runs/polytopia/shape-check-20261114).
    if c['gpu'] != 'L40S' or c['cpu_cores'] != 4 or c['memory_mib'] != 16384:
        raise ValueError('The training worker is one L40S + 4 CPU cores + 16 GiB')
    amount = 'budget_usd' if mode == 'usd' else 'budget_seconds'
    if any(not math.isfinite(float(c[key])) or float(c[key]) <= 0
           for key in (amount, 'min_run_seconds', 'min_lease_seconds')):
        raise ValueError('Invalid bounded allocation')
    cap = c.get('max_lease_seconds')
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < c['min_lease_seconds']):
        raise ValueError('max_lease_seconds is a whole number of seconds, at least min_lease_seconds')
    if (c['budget_usd'] if mode == 'usd' else c['budget_seconds'] * rate(TRAIN_SHAPE)) > 1000:
        raise ValueError('Session exceeds the infrastructure limit')
    if mode == 'seconds' and 'budget_usd' in c:
        raise ValueError('A seconds session has no dollar budget')
    return c
