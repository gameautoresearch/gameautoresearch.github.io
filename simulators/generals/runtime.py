"""Source binding, safe non-pickle checkpoints, allocated worker wall-time ledger."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from generals.core.game import GameState
from .environment import RULESET
from .learning import Carry

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PIN = "954927f508aa47899019d66317cc313af3e9cc19"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True)+"\n")
    temp.replace(path)


def sources():
    repo = ROOT/"candidates/generals-bots"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if commit != PIN:
        raise RuntimeError(f"Engine commit mismatch: expected {PIN}, got {commit}")
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True)
    if status:
        raise RuntimeError("Upstream tracked source changed; refuse frozen-rules run")
    # Bind every tracked file, including license, game code and assets. No external data download.
    files = subprocess.check_output(["git", "ls-files"], cwd=repo, text=True).splitlines()
    tree = {f: digest((repo/f).read_bytes()) for f in files if (repo/f).is_file()}
    integration = {str(p.relative_to(HERE)): digest(p.read_bytes()) for p in sorted(HERE.rglob("*"))
                   if p.is_file() and p.suffix in (".py", ".json", ".lock")
                   and "autoresearch" not in p.relative_to(HERE).parts}
    return dict(repository="https://github.com/strakam/generals-bots.git", commit=commit,
                upstream_license="MIT, Copyright (c) 2024 Matej Straka", tracked_sha256=tree,
                tree_sha256=digest(canonical(tree)), integration_sha256=integration,
                implementation_sha256=digest(canonical(integration)), rules=RULESET, rules_sha256=digest(canonical(RULESET)))


def hardware(cfg, requested):
    devices = jax.devices()
    backend = jax.default_backend()
    contract = cfg["hardware"]
    if requested == "gpu":
        if backend != "gpu":
            raise RuntimeError("GPU requested but JAX is not running on a GPU")
        expected = contract["gpu_model"]
        if not expected or any(expected not in d.device_kind for d in devices) or len(devices) != 1:
            raise RuntimeError(f"Requires one {expected}; actual devices: {devices}")
        # Scheduler/container is responsible for physical CPU/RAM isolation; don't claim enforced here.
        if os.environ.get("GENERALS_ALLOCATION_CONFIRMED") != "1":
            raise RuntimeError("Set GENERALS_ALLOCATION_CONFIRMED=1 only inside the declared CPU/RAM/GPU allocation")
        if "GENERALS_ALLOCATION_START_UNIX" not in os.environ:
            raise RuntimeError("GPU commands must run under integrations.generals.lease so the complete allocation is timed")
    return dict(backend=backend, devices=[str(d) for d in devices], device_kinds=[d.device_kind for d in devices],
                machine=platform.machine(), platform=platform.platform(), python=platform.python_version(),
                jax=jax.__version__, process_cpu_count=os.cpu_count(), contract=contract,
                allocation_enforced_externally=(requested == "gpu"),
                gpu_measurement=backend == "gpu", requested=requested)


class Ledger:
    def __init__(self, start, limit_seconds, previous_seconds=0):
        self.start, self.limit, self.previous = start, limit_seconds, previous_seconds
        self.lease_start = os.environ.get("GENERALS_ALLOCATION_START_UNIX")
        self.lease_prior = float(os.environ.get("GENERALS_ALLOCATION_PRIOR_SECONDS", "0"))

    def seconds(self):
        if self.lease_start is not None:
            return self.lease_prior + time.time()-float(self.lease_start)
        return self.previous + time.monotonic()-self.start

    def expired(self, reserve=0):
        return self.seconds()+reserve >= self.limit

    def report(self):
        elapsed = self.seconds()
        return dict(allocated_worker_seconds=elapsed, allocated_worker_hours=elapsed/3600,
                    allocation_id=os.environ.get("GENERALS_ALLOCATION_ID"),
                    limit_seconds=self.limit, overrun_seconds=max(0., elapsed-self.limit),
                    accounting="Full allocated worker elapsed time: imports, CPU simulation, transfer, compilation, inference, updates, evaluation, checkpoint IO. GPU idle time counts. Local CPU runs are not measured GPU-hours.",
                    stopping="Cooperative at phase/update boundaries; compilation may overrun. External hard worker deadline required for strict allocation.")


@jax.jit
def _history_fingerprints(history):
    """Cheap device-side row index; exact reconstruction is independently checked."""
    leaves = jax.tree.leaves(history)
    b = leaves[0].shape[0]
    first, second = jnp.zeros(b, jnp.uint32), jnp.zeros(b, jnp.uint32)
    for i, leaf in enumerate(leaves):
        bits = jax.lax.bitcast_convert_type(leaf, jnp.uint32).reshape((b, -1))
        salt = jnp.arange(bits.shape[1], dtype=jnp.uint32)*jnp.uint32(2654435761)+jnp.uint32(i+1)
        first ^= jnp.sum(bits*salt, -1, dtype=jnp.uint32)
        second += jnp.sum((bits ^ (bits >> 16))*(salt ^ jnp.uint32(2246822519)), -1, dtype=jnp.uint32)
    return jnp.stack([first, second], -1)


def compact_history(history):
    """Deduplicate immutable per-episode weights before the device-to-host copy.

    Fingerprints are only an index: integer bit-for-bit reconstruction of every
    row is checked on device. A collision falls back to the original full cache.
    No probabilistic checksum substitutes for the checkpoint's payload SHA.
    """
    fingerprints = np.asarray(_history_fingerprints(history))
    _, indices, inverse = np.unique(fingerprints, axis=0, return_index=True, return_inverse=True)
    selected = jax.tree.map(lambda p: p[jnp.asarray(indices, jnp.int32)], history)
    ids = jnp.asarray(inverse, jnp.int32)
    exact = all(bool(jnp.array_equal(jax.lax.bitcast_convert_type(p, jnp.uint32),
                                  jax.lax.bitcast_convert_type(q[ids], jnp.uint32)))
                for p, q in zip(jax.tree.leaves(history), jax.tree.leaves(selected)))
    return (selected, ids) if exact else (history, None)


def save_checkpoint(path, train_state, bank, carry, pool, cfg, binding, update, ledger, bank_cursor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(params=train_state.params, opt_state=serialization.to_state_dict(train_state.opt_state),
                   step=train_state.step, bank=bank, carry={**carry._asdict(), "states": carry.states._asdict()},
                   pool=pool._asdict())
    cache_rows = None
    if cfg.get('compact_checkpoint', False):
        selected, ids = compact_history(carry.history_params)
        if ids is not None:
            payload['carry']['history_params'] = selected
            payload['carry']['history_row_ids'] = ids
            cache_rows = int(jax.tree.leaves(selected)[0].shape[0])
    binary = serialization.msgpack_serialize(jax.device_get(payload))
    temp = path.with_suffix(".tmp")
    temp.write_bytes(binary)
    temp.replace(path)
    meta = dict(format=2 if cache_rows is not None else 1,
                unique_history_rows=cache_rows, config=cfg, source_binding=binding, update=update, bank_cursor=bank_cursor,
                budget=ledger.report(), payload_sha256=digest(binary))
    write_json(str(path)+".json", meta)
    return meta


def read_checkpoint(path, template=None, expected_cfg=None, binding=None):
    path = Path(path)
    meta = json.loads(Path(str(path)+".json").read_text())
    raw = path.read_bytes()
    if digest(raw) != meta["payload_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    if binding is not None:
        for field in ("commit", "tree_sha256", "rules_sha256", "implementation_sha256"):
            if binding[field] != meta["source_binding"][field]:
                raise ValueError(f"Checkpoint {field} differs from current source")
    if expected_cfg is not None and expected_cfg != meta["config"]:
        raise ValueError("Resume recipe differs from checkpoint; exact resume requires unchanged recipe")
    data = jax.tree.map(jnp.asarray, serialization.msgpack_restore(raw))
    if meta['format'] == 2:
        ids = data['carry'].pop('history_row_ids')
        if ids.shape != data['carry']['seats'].shape:
            raise ValueError('Malformed indexed history cache')
        rows = jax.tree.leaves(data['carry']['history_params'])[0].shape[0]
        if bool(jnp.any((ids < 0) | (ids >= rows))):
            raise ValueError('History cache row index out of bounds')
        data['carry']['history_params'] = jax.tree.map(lambda p: p[ids], data['carry']['history_params'])
    elif meta['format'] != 1:
        raise ValueError('Unsupported checkpoint format')
    if template is None:
        return meta, data
    train = template.replace(params=data["params"], step=data["step"],
                             opt_state=serialization.from_state_dict(template.opt_state, data["opt_state"]))
    c = data["carry"]
    c["states"] = GameState(**c["states"])
    return meta, (train, data["bank"], Carry(**c), GameState(**data["pool"]))
