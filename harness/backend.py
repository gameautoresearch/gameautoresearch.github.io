"""What a game supplies to the harness.

A backend wraps one game's native trainer, evaluator, ledger and worker
dispatch behind the session operations below. It never talks to a language
model and never reads the researcher's box: the broker hands it validated
files through `harness.workspace` and plain arguments.

Every method returns a JSON-serializable dict and raises ValueError for a
request the researcher got wrong (the message is shown to the researcher).
"""
from pathlib import Path

from harness import budget as budgets

STATUSES = ('queued', 'running', 'complete', 'failed', 'stopped', 'uncertain')
ACTIVE = ('queued', 'running', 'uncertain')


class Backend:
    game = None

    def __init__(self, release, root, *, smoke=False, budget_seconds=None, budget_usd=None, resume=False,
                 options=None):
        """`root` is this backend's own session directory.

        The research budget mode (harness/budget.py) follows from which amount is given:
        `budget_usd` selects dollar mode (parallel jobs, each worker second at its shape's price, limits
        `budget.PARALLEL`); otherwise seconds mode (one job at a time, `budget_seconds` of the training
        worker; None means the game's default). Every backend supports both and reports its mode's keys from
        `budget()`. A fresh final training is always seconds mode. `resume=True` reopens an existing root
        without granting any new allowance (used for host grading). `options` are game-specific.
        """
        self.release = Path(release).resolve()
        self.root = Path(root).resolve()
        self.smoke = bool(smoke)
        self.budget_seconds = budget_seconds
        self.budget_usd = budget_usd
        self.budget_mode = budgets.USD if budget_usd is not None else budgets.SECONDS
        self.limits = budgets.limits(self.budget_mode)
        self.resume = bool(resume)
        self.options = dict(options or {})
        if self.root.exists() != self.resume:
            raise ValueError('Reopen an existing session root with resume=True; a new session needs a fresh root')

    # The native trainer tree that harness.workspace.sync copies editable files into.
    workspace = None

    # Set by the host before `open()`: True for the separate session that trains a
    # submitted recipe from scratch. A backend whose contract gives fresh final
    # training host-chosen seeds (so a recipe cannot be tuned to known ones) reads it.
    final = False

    # Set by the host before `grade()`: True only when the session ran at the
    # game's declared budgets (not smoke, no overrides). A backend whose evaluator
    # distinguishes official from development grading reads it.
    official = False

    def open(self):
        """Create the provider/controller. No research allocation is spent until `run`."""
        raise NotImplementedError

    def settle(self):
        """Cancel queued work, stop running work and wait until nothing is active. Idempotent."""
        raise NotImplementedError

    def close(self):
        """Settle, then release the provider/app. Idempotent."""
        raise NotImplementedError

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def budget(self):
        """`budget_mode` and, in seconds mode, at least budget_seconds, used_seconds and remaining_seconds
        (floats); in dollar mode budget_usd, used_usd, remaining_usd and allocatable_usd instead. Both report
        `concurrency` (the most jobs that may run at once)."""
        raise NotImplementedError

    def status(self):
        """{'jobs': [{'name', 'kind', 'status', ...}], 'active': [names]} with status in STATUSES."""
        raise NotImplementedError

    def log(self, name, tail=80):
        """Researcher-visible output and metrics for one of their own jobs."""
        raise NotImplementedError

    def check(self, config):
        """Free validation of the synced files beyond syntax (parameter caps, config schema)."""
        return {'checked': True}

    def run(self, name, seconds, config, seed=1):
        """Queue one training run of the current editable files; sync them first."""
        raise NotImplementedError

    def stop(self, name):
        raise NotImplementedError

    def evaluate(self, name, seconds=None):
        """Queue public practice games for a completed run."""
        raise NotImplementedError

    def submit(self, name):
        """Register a completed run's trained artifact as an immutable submission."""
        raise NotImplementedError

    # Leases: a game whose SPEC sets `leases=True` lets the researcher hold a worker of a chosen shape and run
    # their own commands on it. The broker validates names, numbers and the editable files; the backend owns
    # the shape limits, the charging rate and what a lease can reach.

    def lease(self, name, seconds, gpu, cpu, memory):
        """Queue a worker with `gpu` GPUs (0 or 1), `cpu` cores and `memory` GiB for `seconds`; push the files."""
        raise ValueError('This game has no leases')

    def lease_exec(self, name, command):
        """Start a shell command on a running lease. Returns at once with {'exec': id}."""
        raise ValueError('This game has no leases')

    def lease_poll(self, name, exec_id, tail=None):
        """State, exit code and the end of the output of one command."""
        raise ValueError('This game has no leases')

    def lease_push(self, name):
        """Copy the current editable files to a running lease."""
        raise ValueError('This game has no leases')

    def lease_sync(self, name, reload=False):
        """Commit the lease's scratch storage; `reload` also brings in what other leases committed."""
        raise ValueError('This game has no leases')

    def package(self, name, lease, path, config):
        """Seal a network a lease produced, with the current policy source, as run `name` (for eval and submit)."""
        raise ValueError('This game has no leases')

    def finalize(self):
        """Close the native ledger. Requires no active work."""
        raise NotImplementedError

    def tick(self):
        """Advance queued work. Called about once a second by the broker; must not block."""

    def grade(self):
        """Host-only private grading of the final submission."""
        raise NotImplementedError

    def sync(self, config=None):
        from harness import workspace
        return workspace.sync(self, config)
