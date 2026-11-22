"""Evaluation-only worker path. Training encoder/runtime and source bindings stay unchanged."""
from pathlib import Path
import subprocess
from integrations.pokemon.env import Worker, Vector, team_size_args


class EvaluationWorker(Worker):
    def __init__(self, timeout=60, team_size=6):
        # Initialize transport exactly as Worker, replacing only its executable path.
        self.metrics = {}; self.decode_seconds = 0.; self.pipe_wait_seconds = 0.
        self.binary = False; self.timeout = timeout; self.lazy = False
        self.pending = bytearray(); self.shared = None; self.shared_file = None; self.shared_dir = None
        self.process = subprocess.Popen(['node', str(Path(__file__).with_name('worker.cjs')), *team_size_args(team_size)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)


class EvaluationVector(Vector):
    def __init__(self, size=12, timeout=60, team_size=6):
        self.workers = [EvaluationWorker(timeout=timeout, team_size=team_size)]
        self.per_worker = size; self.batched = True; self.optimized = True
        self.packed = False; self.token_cache = False; self.mon_cache = False; self.fast_callbacks = False
        self.states = None; self.seconds = 0.; self.calls = 0

    def visible(self, seats, offsets=None):
        return self.call([{'op': 'visible', 'seats': seats, 'offsets': offsets or [0] * self.size}])

    def audit(self):
        return self.call([{'op': 'audit'}])
