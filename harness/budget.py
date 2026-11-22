"""The two research budget modes every game offers, chosen per session.

`seconds`  Linear: one job at a time (a training run, a practice or a lease), and the research allocation is a
           number of worker seconds. A worker of the game's training shape is charged its wall seconds; a
           worker of another shape (a CPU practice worker, a lease) is charged the training-shape seconds its
           price buys, so one second of the training worker is always one second of allocation.
`usd`      Parallel: up to `PARALLEL` jobs at once, and the research allocation is US dollars of worker time,
           each worker second charged at its shape's frozen list price (`PRICE`).

A session picks its mode with `--budget-mode`, or implicitly with `--research-seconds` / `--research-usd`;
without either it uses the game's `budget_mode` default. Fresh final training is always one run of
`final_train_seconds`, whatever the research mode.
"""

SECONDS, USD = 'seconds', 'usd'
MODES = (SECONDS, USD)

# Frozen Modal list prices in US dollars per second (L40S GPU, one CPU core, one GiB of memory): the prices the
# dollar contracts of Generals, Hex and Polytopia were written against.
PRICE = {'gpu': 0.000542, 'cpu_core': 0.00003942, 'memory_gib': 0.00000667}

# Dollar mode's concurrency limits (the Generals / Hex / Polytopia contracts).
PARALLEL = dict(max_concurrent_jobs=4, max_concurrent_gpus=2, max_concurrent_cpu_cores=64)
LINEAR = dict(max_concurrent_jobs=1, max_concurrent_gpus=1, max_concurrent_cpu_cores=None)


def rate(shape):
    """US dollars per wall second of a worker of `shape` ({'gpu', 'cpu_cores', 'memory_mib'})."""
    return round(shape['gpu'] * PRICE['gpu'] + shape['cpu_cores'] * PRICE['cpu_core']
                 + shape['memory_mib'] / 1024 * PRICE['memory_gib'], 10)


def seconds_weight(shape, train_shape):
    """Allocation seconds charged per wall second of `shape` in seconds mode (1.0 for the training shape)."""
    return rate(shape) / rate(train_shape)


def choose(spec, budget_mode=None, research_seconds=None, research_usd=None):
    """The session's mode: explicit, implied by the amount given, or the game's default."""
    if budget_mode is not None and budget_mode not in MODES:
        raise ValueError(f'--budget-mode is one of {", ".join(MODES)}')
    if research_seconds and research_usd:
        raise ValueError('Give --research-seconds (seconds mode) or --research-usd (dollar mode), not both')
    implied = SECONDS if research_seconds else USD if research_usd else None
    if budget_mode and implied and budget_mode != implied:
        raise ValueError(f'--budget-mode {budget_mode} does not take --research-{"seconds" if implied == SECONDS else "usd"}')
    return budget_mode or implied or spec['budget_mode']


def limits(mode):
    return dict(PARALLEL if mode == USD else LINEAR)


def describe(mode, research_seconds=None, research_usd=None):
    """The researcher-facing sentence for how jobs run in this mode."""
    if mode == USD:
        return (f'- Jobs: up to {PARALLEL["max_concurrent_jobs"]} at once (at most {PARALLEL["max_concurrent_gpus"]} '
                'GPUs); each worker second is charged at the price of its shape, and a queued or running job holds '
                'its whole grant until it ends.')
    return ('- Jobs: one at a time. A new lease or run (where there is `ar run`), and a practice where practice is charged to the allocation, '
            'is refused until the current one has ended; `ar budget` says which work is charged.')
