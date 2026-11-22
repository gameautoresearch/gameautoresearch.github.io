"""Content pins, safe checkpoint I/O, manifests and full allocated-worker clock."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .environment import PIN, ROOT, RULESET, SOURCE, digest, verify_checkout
from .model import Policy

HERE = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_source():
    return verify_checkout()


def contract_hash():
    """Rules, upstream file pins and deal/codec specification, not execution knobs."""
    return digest({name:sha256(HERE/name) for name in
                   ('source.lock.json','rules.lock.json','public-data-seeds.json')})


def manifest(config):
    lock = verify_source()
    files = {str(p.relative_to(HERE)): sha256(p) for p in sorted(HERE.rglob('*'))
             if p.is_file() and (p.suffix in {'.py', '.cpp','.h','.cu', '.json', '.md', '.txt', '.sh'} or p.name.startswith('LICENSE'))
             and '__pycache__' not in p.parts}
    public_config = dict(config)
    if 'private_seed_manifest_sha256' in config:
        public_config.update(seed='evaluator-private',deal_seed='evaluator-private')
    return {
        'format': 1, 'ruleset': RULESET, 'source': lock,
        'integration_files': files, 'config': public_config, 'config_sha256': digest(config),
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'torch': str(torch.__version__), 'platform': platform.platform(),
                    'processor': platform.processor(), 'logical_cpus': os.cpu_count(),
                    'torch_threads': torch.get_num_threads(),
                    'head_tf32_allowed':torch.backends.cuda.matmul.allow_tf32,
                    'cudnn_tf32_allowed':torch.backends.cudnn.allow_tf32,
                    'cuda_available': torch.cuda.is_available(),
                    'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        'simulation_device': 'cuda:0' if config.get('engine_backend')=='cuda' else 'cpu',
        'simulation_backend':config.get('engine_backend','view'),
        'policy_device': config.get('device', 'cpu'),
    }


class Budget:
    def __init__(self, config, started_perf, started_unix, prior=0.0, allocated_start_unix=None):
        self.started_perf = started_perf
        self.prior = prior
        self.extra = max(0., started_unix - allocated_start_unix) if allocated_start_unix is not None else 0.
        self.gpus = config['allocation']['gpus']
        self.limit = float(config['max_seconds'])
        if 'budget_gpu_hours' in config and self.gpus:
            self.limit = min(self.limit, float(config['budget_gpu_hours']) * 3600)
        if self.limit <= 0:
            raise ValueError('A positive full worker elapsed-time limit is required')

    def elapsed(self):
        return self.prior + self.extra + time.monotonic() - self.started_perf

    def expired(self):
        return self.elapsed() >= self.limit

    def report(self):
        elapsed = self.elapsed()
        return {'allocated_worker_seconds': elapsed, 'limit_seconds': self.limit,
                'allocated_gpu_hours': elapsed / 3600 if self.gpus else 0.,
                'local_cpu_worker_hours': elapsed / 3600 if not self.gpus else 0.,
                'allocated_gpus': self.gpus, 'includes_prior_invocations': self.prior,
                'prelaunch_allocation_seconds': self.extra,
                'charge_rule': 'full allocated worker wall time; includes idle GPU and CPU simulation, '
                               'imports, I/O, transfer, inference, updates, checkpoints and development evaluation',
                'deadline_overshoot_seconds': max(0., elapsed - self.limit)}


def save_checkpoint(path, policy, optimizers, state, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format': 1, 'ruleset': RULESET, 'source_commit': PIN,
               'contract_sha256': contract_hash(),
               'architecture': policy.architecture, 'config': config,
               'model': policy.state_dict(), 'state': state,
               'torch_rng': torch.get_rng_state(),
               'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               'optimizers': {r: o.state_dict() for r, o in optimizers.items()}}
    if config.get('engine_backend')=='native':
        payload['native_source_sha256']=sha256(HERE/'native/engine.cpp')
    if config.get('engine_backend')=='cuda':
        from .cuda.backend import source_hash
        payload['cuda_source_sha256']=source_hash()
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device='cpu', expected_sha256=None):
    path = Path(path).resolve()
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError(f'Checkpoint hash changed: {path}')
    # Never opt into arbitrary pickle execution; only tensors and plain state types.
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get('format') != 1 or data.get('ruleset') != RULESET or data.get('source_commit') != PIN:
        raise ValueError('Checkpoint format/rules/source mismatch')
    verify_source()
    # Historical format-1 checkpoints retain their original source/rules checks.
    # New checkpoints additionally bind the complete frozen contract files.
    if 'contract_sha256' in data and data['contract_sha256'] != contract_hash():
        raise ValueError('Checkpoint source/rules/deal contract mismatch')
    if 'native_source_sha256' in data and data['native_source_sha256'] != sha256(HERE/'native/engine.cpp'):
        raise ValueError('Checkpoint native source mismatch')
    if 'cuda_source_sha256' in data:
        from .cuda.backend import source_hash
        if data['cuda_source_sha256']!=source_hash():
            raise ValueError('Checkpoint CUDA source mismatch')
    # Loading an opponent must not perturb the learner's RNG stream. Constructor
    # initialization is overwritten immediately by the checkpoint.
    with torch.random.fork_rng(devices=[]):
        policy = Policy(data['architecture']).to(device)
    policy.load_state_dict(data['model'], strict=True)
    policy.compute_dtype=data['config'].get('compute_dtype','float32')
    policy.eval()
    return policy, data


def checkpoint_descriptor(path):
    path = Path(path).resolve()
    return {'kind': 'checkpoint', 'path': str(path), 'sha256': sha256(path)}
