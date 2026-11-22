"""Shared worker start/abandon semantics for every game's provider.

A created Modal sandbox can sit in the provider's queue for many minutes when GPU capacity is short. Providers
used to give up after ~60 s and mark the job `uncertain`; recovery then needed a recorded start that never came,
so the session was blocked (no runs, no finalize) and a late-starting sandbox idled for its whole grant.

Rules, identical in every backend:
- `await_start` keeps waiting while the worker is queued, up to START_PATIENCE_SECONDS, and stops waiting early
  if the job is stopped. Queued time is not charged.
- If it never starts, `abandon` terminates it and returns the provider's final record. A worker the provider
  never started (`never_started`) is settled as failed with 0 seconds charged and no artifact; it is never left
  `uncertain`. One that started while being abandoned is charged from its start/finish receipt as usual.
- Recovery after a host interruption applies the same rule to a recorded sandbox: never started -> terminate and
  settle at 0; started -> the existing receipt-based settlement.
"""
import time

# How long a created worker may wait in the provider's queue for a GPU before the job is abandoned (uncharged).
START_PATIENCE_SECONDS = 1800.0
# After terminating a worker, how long to wait for the provider to report its final state.
ABANDON_SETTLE_SECONDS = 60.0

NEVER_STARTED = 'The provider never started this worker (no GPU was allocated); it was terminated and charged 0 seconds'


class NeverStarted(RuntimeError):
    """The worker was created but never started; it has been terminated. `receipt` is the provider's record."""

    def __init__(self, receipt, reason=NEVER_STARTED):
        super().__init__(reason)
        self.receipt = dict(receipt or {})


def started(receipt):
    return bool((receipt or {}).get('started_at'))


def never_started(receipt):
    """True when the provider's record shows no start: no allocation, nothing to charge."""
    return not started(receipt)


def await_start(info, *, patience=None, poll=2.0, stop=None, sleep=time.sleep, clock=time.monotonic):
    """Poll `info()` (a provider receipt dict or None) until it reports a start.

    Returns the receipt once started, or None when `stop()` becomes true or `patience` runs out first.
    """
    deadline = clock() + (START_PATIENCE_SECONDS if patience is None else patience)
    while True:
        receipt = info() or {}
        if started(receipt):
            return receipt
        if (stop is not None and stop()) or clock() >= deadline:
            return None
        sleep(poll)


def abandon(terminate, info, *, settle=None, poll=1.0, sleep=time.sleep, clock=time.monotonic):
    """Terminate a worker the backend will not use and return its final provider record.

    `terminate()` stops the sandbox; `info()` reads its receipt. Returns as soon as the receipt is final: either
    finished (it had started: charge start->finish) or still unstarted after termination (charge nothing).
    """
    try:
        terminate()
    except Exception:  # already gone: the receipt below is still authoritative
        pass
    begun = clock()
    deadline = begun + (ABANDON_SETTLE_SECONDS if settle is None else settle)
    unstarted_grace = begun + 5 * poll  # a start racing the termination shows up within a few polls
    receipt = info() or {}
    while clock() < deadline:
        if receipt.get('finished_at'):
            break  # it had started: charge from the receipt
        if never_started(receipt) and clock() >= unstarted_grace:
            break  # never started: nothing to charge
        sleep(poll)
        receipt = info() or {}
    return receipt
