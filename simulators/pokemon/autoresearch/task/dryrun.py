#!/usr/bin/env python3
"""CPU dry run of a Pokémon recipe inside the researcher's box. `ar check` runs it.

It runs the researcher's real `train.py` for one update, the way a worker's
supervisor starts it, with one substitution: the trusted environment service is
replaced by a stand-in that returns random observations of the contract's shape
and enforces the same call rules (frozen terminal reward, action shape, rollout
bounds). Everything else is the worker's own public code from `simulator/`:
the recipe validation, the observation layout and the graph export with its
operator, node, size and parameter limits.

A pass means the code imports, steps, updates, checkpoints and exports a graph
the contract admits. It says nothing about learning, speed, GPU memory or
anything that depends on real battles. Nothing here is charged or sent anywhere.

Standard library and torch only; prints one JSON object.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

EDITABLE = ('policy.py', 'learner.py', 'train.py')
OUTPUTS = ('checkpoint.pt', 'result.json', 'progress.jsonl', 'graph.json')
PARAMETER_CAP = 1_000_000
# Small enough for a CPU, large enough that most rollouts contain an ended and a cut-off battle.
ENVIRONMENTS = 8
ROLLOUT_STEPS = 8
TRAINER_SECONDS = 420
# Public worker modules, as the trainer imports them, and where the box keeps them.
PUBLIC = {'integrations/pokemon/network.py': 'simulators/pokemon/network.py',
          'integrations/pokemon/efficient.py': 'simulators/pokemon/efficient.py',
          'integrations/pokemon/autoresearch/runtime/sdk.py': 'simulators/pokemon/autoresearch/runtime/sdk.py',
          'integrations/pokemon/autoresearch/runtime/graph_export.py':
              'simulators/pokemon/autoresearch/runtime/graph_export.py',
          'integrations/pokemon/autoresearch/runtime/graph_worker.py':
              'simulators/pokemon/autoresearch/runtime/graph_worker.py'}

# The identity and file helpers a worker's public `contract.py` holds.
CONTRACT = '''from pathlib import Path
import hashlib, json, os, tempfile
EDITABLE = ('policy.py', 'learner.py', 'train.py')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\\n')

def source_signature(directory):
    return identity({name: sha(Path(directory) / name) for name in EDITABLE})
'''

# Stands in for runtime/env_client.py: same attributes, methods, shapes, dtypes and refusals; random battles.
ENV_CLIENT = '''"""Dry-run stand-in for the environment client: contract-shaped random observations, no engine."""
import torch
from integrations.pokemon.efficient import SLICES, SHAPES, WIDTH

TOKENS = ('tokens', 'global_tokens', 'action_tokens')


class Client:
    backend = 'dry_run_random_observations'

    def __init__(self, config, device):
        self.n = config['environments']; self.device = torch.device(device); self.steps = None; self.closed = False
        self.random = torch.Generator().manual_seed(config['seed'])
        for name in ('source_signature', 'binary_sha256', 'rules_sha256', 'initial_sha256',
                     'integration_signature', 'runtime_manifest_sha256'):
            setattr(self, name, '0' * 64)
        self.obs = self._observations(self.n).to(self.device)

    def _rows(self, count, ready):
        """`count` seat observations; a seat that is not to move has an empty legal mask."""
        rows = torch.zeros((count, WIDTH), dtype=torch.float32)
        for key in SHAPES:
            width = SLICES[key].stop - SLICES[key].start
            if key in TOKENS:
                rows[:, SLICES[key]] = torch.randint(0, 16384, (count, width), generator=self.random).float()
            elif key != 'mask':
                rows[:, SLICES[key]] = torch.rand((count, width), generator=self.random)
        mask = torch.rand((count, 14), generator=self.random) < .5
        mask[torch.arange(count), torch.randint(0, 14, (count,), generator=self.random)] = True
        rows[:, SLICES['mask']] = (mask & ready[:, None]).float()
        return rows

    def _observations(self, battles):
        # Either seat alone or both are to move, as in a real battle.
        kind = torch.randint(0, 3, (battles,), generator=self.random)
        ready = torch.stack((kind != 1, kind != 0), dim=1).reshape(-1)
        return self._rows(battles * 2, ready).reshape(battles, 2, WIDTH)

    def allocate_rollout(self, steps):
        if type(steps) is not int or not 1 <= steps <= 256: raise ValueError('Bounded rollout required')
        self.steps = steps
        self.flags = torch.zeros((steps, self.n, 3), dtype=torch.int32, device=self.device)
        self.rewards = torch.zeros((steps, self.n, 2), dtype=torch.float32, device=self.device)

    def step(self, actions, index, faint_reward=0):
        if faint_reward != 0: raise ValueError('Frozen terminal rewards only')
        if self.steps is None or type(index) is not int or not 0 <= index < self.steps:
            raise ValueError('Rollout index outside allocation')
        if tuple(actions.shape) != (self.n, 2): raise ValueError('Expected actions [environments,2]')
        chosen = actions.detach().to(device='cpu', dtype=torch.int64)
        mask = self.obs[..., SLICES['mask']].cpu() > 0
        ready = mask.any(-1)
        legal = torch.gather(mask, 2, chosen.clamp(0, 13)[..., None]).squeeze(-1) & (chosen >= 0) & (chosen < 14)
        fault = ((ready & ~legal) | (~ready & (chosen != -1))).any(-1)
        if fault.any():
            raise RuntimeError('Dry run: an action was illegal for its seat, or a seat that was not to move did not '
                               'answer -1. A worker reports this as a native transition fault.')
        ended = torch.rand((self.n,), generator=self.random) < .15
        cut = ~ended & (torch.rand((self.n,), generator=self.random) < .1)
        outcome = torch.randint(0, 3, (self.n,), generator=self.random).float() - 1
        reward = torch.stack((outcome, -outcome), dim=1) * ended[:, None]
        flags = torch.stack((ended, cut, torch.zeros_like(ended)), dim=1).to(torch.int32)
        self.obs.copy_(self._observations(self.n))
        self.flags[index].copy_(flags); self.rewards[index].copy_(reward)

    def truncated_views(self):
        selected = self.flags[..., 1].bool()[..., None].expand(-1, -1, 2).reshape(-1)
        count = int(selected.sum())
        return self._rows(count, torch.ones(count, dtype=torch.bool)).to(self.device), selected

    def snapshot(self):
        return b'dry-run'

    def restore(self, state):
        if not isinstance(state, bytes): raise ValueError('Authenticated opaque state required')

    def close(self):
        self.closed = True
'''


def stage(workspace, root, config):
    """Lay the files out as a worker's supervisor does: read-only source beside its config, public imports apart."""
    source = root / 'input'; source.mkdir()
    for name in EDITABLE:
        shutil.copyfile(workspace / 'training' / name, source / name)
    (source / 'config.json').write_text(json.dumps(config))
    public = root / 'public'
    for target, origin in PUBLIC.items():
        path = public / target
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / 'simulator' / origin, path)
    runtime = public / 'integrations/pokemon/autoresearch'
    (runtime / 'contract.py').write_text(CONTRACT)
    (runtime / 'runtime/env_client.py').write_text(ENV_CLIENT)
    return source, public


def measure(out):
    """What the exported graph and checkpoint weigh against the contract's limits."""
    import torch
    graph = json.loads((out / 'graph.json').read_text())
    found = dict(graph_nodes=len(graph['nodes']), graph_bytes=(out / 'graph.json').stat().st_size)
    state = torch.load(out / 'checkpoint.pt', map_location='cpu', weights_only=True).get('model')
    if not isinstance(state, dict):
        raise ValueError('checkpoint.pt has no "model" state dictionary; keep the bindings the starter writes')
    found['parameter_and_buffer_elements'] = sum(int(value.numel()) for value in state.values())
    if found['parameter_and_buffer_elements'] > PARAMETER_CAP:
        raise ValueError(f'{found["parameter_and_buffer_elements"]:,} parameter and buffer elements exceed the '
                         f'{PARAMETER_CAP:,} cap')
    rows = [json.loads(line) for line in (out / 'progress.jsonl').read_text().splitlines() if line.strip()]
    found['progress_rows'] = len(rows)
    found['last_progress_row'] = {key: value for key, value in (rows[-1] if rows else {}).items()
                                  if type(value) in (int, float)}
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', default='/workspace')
    parser.add_argument('--config', default='configs/train.json')
    args = parser.parse_args()
    workspace = Path(args.workspace)
    started = time.monotonic()
    result = dict(passed=False, stage='read', config=args.config,
                  scope='One update of your train.py on CPU against random contract-shaped observations, then the '
                        'real graph export. Not evidence of learning, speed or GPU memory.')
    try:
        config = json.loads((workspace / args.config).read_text())
        if not isinstance(config, dict):
            raise ValueError('The config must be a JSON object')
        reduced = dict(config, environments=min(config.get('environments', ENVIRONMENTS), ENVIRONMENTS),
                       rollout_steps=min(config.get('rollout_steps', ROLLOUT_STEPS), ROLLOUT_STEPS))
        result['reduced_for_cpu'] = {key: reduced[key] for key in ('environments', 'rollout_steps')
                                     if reduced[key] != config.get(key)}
        with tempfile.TemporaryDirectory(prefix='pokemon-dry-run-') as temporary:
            root = Path(temporary)
            source, public = stage(workspace, root, reduced)
            out = root / 'output/train'; out.parent.mkdir()
            environment = {key: value for key, value in os.environ.items() if key in ('PATH', 'LANG', 'LC_ALL')}
            environment.update(HOME=str(root), TMPDIR=str(root), PYTHONPATH=str(public), PYTHONDONTWRITEBYTECODE='1',
                               POKEMON_PILOT_RELEASE_SHA256='dry-run', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            result['stage'] = 'train'
            done = subprocess.run([sys.executable, 'train.py', '--config', 'config.json', '--out', str(out),
                                   '--seconds', '120', '--device', 'cpu', '--updates', '1'],
                                  cwd=source, env=environment, capture_output=True, text=True, timeout=TRAINER_SECONDS)
            if done.returncode != 0:
                output = (done.stdout + done.stderr).replace(str(source) + '/', 'training/').replace(str(public) + '/', '')
                raise RuntimeError('train.py exited with ' + str(done.returncode) + ':\n' + output[-2500:])
            result['stage'] = 'outputs'
            missing = [name for name in OUTPUTS if not (out / name).is_file()]
            if missing:
                raise ValueError('train.py finished without ' + ', '.join(missing))
            result.update(measure(out))
        result.update(passed=True, stage='complete')
    except subprocess.TimeoutExpired:
        result['error'] = f'train.py did not finish one small CPU update within {TRAINER_SECONDS} seconds'
    except Exception as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
    result['seconds'] = round(time.monotonic() - started, 1)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
