#!/usr/bin/env python3
"""CPU dry run of a Hex recipe inside the researcher's box. `ar check` runs it.

Hex's engine is plain PyTorch, so nothing is stubbed: the researcher's real
`train.py` runs for a short time against the real engine from `simulator/`, the
way a worker's supervisor starts it but on CPU. Then the files it left are loaded
through the evaluation API (`policy.create` and `policy.act`) and asked for moves
in real positions, with the evaluator's legality, batch and tensor checks.

A pass means the code imports, trains, saves a loadable package and answers legal
moves. It says nothing about learning, GPU speed or memory, or playing strength.
Nothing here is charged or sent anywhere. Standard library and torch only; prints
one JSON object.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

EDITABLE = ('policy.py', 'learner.py', 'train.py')
TRAINER_SECONDS = 45
GRACE_SECONDS = 60
TENSOR_CAP = 10_000_000
BATCH = 64
# The evaluator's per-move inference cap: a call covering n positions has CALL_SECONDS + n * POSITION_SECONDS.
POSITION_SECONDS = 0.01
CALL_SECONDS = 0.1

# Runs in a fresh interpreter beside the trained files: the evaluator's view of a package.
PROBE = r'''
import importlib.util, json, sys, time
from pathlib import Path
import torch
from simulators.hex.engine import HexGames
root = Path(sys.argv[1]); out = {}
torch.set_num_threads(1)
state = torch.load(root / 'out/weights.pt', map_location='cpu', weights_only=True)
if not isinstance(state, dict) or not state or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
    raise SystemExit(json.dumps({'error': 'weights.pt must be a nonempty tensor-only state_dict'}))
out['weights_tensors'] = sum(v.numel() for v in state.values())
if any(v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):
    raise SystemExit(json.dumps({'error': 'weights.pt holds non-finite values'}))
spec = importlib.util.spec_from_file_location('entrant_policy', root / 'source/policy.py')
module = importlib.util.module_from_spec(spec); sys.path.insert(0, str(root / 'source')); spec.loader.exec_module(module)
started = time.monotonic()
policy = module.create(json.loads((root / 'out/model.json').read_text()))
if not isinstance(policy, torch.nn.Module):
    raise SystemExit(json.dumps({'error': 'create() must return a torch.nn.Module'}))
out['policy_tensors'] = sum(p.numel() for p in policy.parameters()) + sum(b.numel() for b in policy.buffers())
policy.load_state_dict(state, strict=True); policy.eval()
games = HexGames(int(sys.argv[2])); generator = torch.Generator().manual_seed(0)
slowest = 0.; first = None; answered = 0; late = 0; calls = 0
position_seconds, call_seconds = float(sys.argv[3]), float(sys.argv[4])
while not bool(games.done.all()):
    running = (~games.done).nonzero().squeeze(1)
    legal = games.legal()
    t = time.monotonic()
    with torch.inference_mode():
        cells = list(module.act(policy, games.board[running].numpy(), games.to_play[running].numpy(), legal[running].numpy()))
    elapsed = time.monotonic() - t
    first = elapsed + (t - started) if first is None else first
    slowest = max(slowest, elapsed)
    calls += 1
    late += elapsed > call_seconds + len(running) * position_seconds
    if len(cells) != len(running):
        raise SystemExit(json.dumps({'error': f'act returned {len(cells)} moves for {len(running)} positions'}))
    for i, c in zip(running.tolist(), cells):
        if isinstance(c, bool) or int(c) != c or not 0 <= int(c) < games.board.shape[1] or not bool(legal[i, int(c)]):
            raise SystemExit(json.dumps({'error': f'act chose an illegal cell {c!r}'}))
    answered += len(cells)
    # Alternate the policy's move with a random one so positions vary.
    actions = torch.tensor([int(c) for c in cells])
    full = torch.zeros(games.n, dtype=torch.long); full[running] = actions
    games.step(full)
    if not bool(games.done.all()):
        running = (~games.done).nonzero().squeeze(1)
        noise = torch.multinomial(games.legal()[running].float(), 1, generator=generator).squeeze(1)
        full = torch.zeros(games.n, dtype=torch.long); full[running] = noise
        games.step(full)
out.update(positions_answered=answered, first_call_seconds=round(first, 3), slowest_batch_seconds=round(slowest, 3),
           calls=calls, late_calls=int(late))
print(json.dumps(out))
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', default='/workspace')
    p.add_argument('--config', default='configs/train.json')
    a = p.parse_args()
    workspace = Path(a.workspace)
    result = {'passed': False, 'config': a.config, 'stage': 'read', 'warnings': []}
    try:
        with tempfile.TemporaryDirectory(prefix='hex-dryrun-') as temp:
            root = Path(temp)
            (root / 'source').mkdir()
            for name in EDITABLE:
                (root / 'source' / name).write_text((workspace / 'training' / name).read_text())
            config = json.loads((workspace / a.config).read_text())
            (root / 'source/config.json').write_text(json.dumps(config))
            env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': temp, 'TMPDIR': temp,
                   'PYTHONPATH': str(workspace / 'simulator'), 'CUDA_VISIBLE_DEVICES': '',
                   'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', '4')}
            result['stage'] = 'train'
            command = [sys.executable, 'train.py', '--config', 'config.json', '--out', str(root / 'out'),
                       '--seed', '1', '--seconds', str(TRAINER_SECONDS)]
            started = time.monotonic()
            try:
                done = subprocess.run(command, cwd=root / 'source', env=env, capture_output=True, text=True,
                                      timeout=TRAINER_SECONDS + GRACE_SECONDS)
                result['trainer_returncode'] = done.returncode
                log = done.stdout + done.stderr
            except subprocess.TimeoutExpired as exc:
                result['trainer_returncode'] = None
                result['warnings'].append(f'train.py was still running {GRACE_SECONDS} s after its --seconds limit '
                                          'and was killed; a worker kills it at the limit and packages what is saved')
                log = ((exc.stdout or b'').decode(errors='replace') if isinstance(exc.stdout, bytes) else exc.stdout or '') + \
                      ((exc.stderr or b'').decode(errors='replace') if isinstance(exc.stderr, bytes) else exc.stderr or '')
            result['trainer_seconds'] = round(time.monotonic() - started, 1)
            result['trainer_output_tail'] = log[-3000:]
            if result['trainer_returncode'] not in (0, None):
                result['error'] = f'train.py exited with code {result["trainer_returncode"]}'
                return result
            out = root / 'out'
            for name in ('model.json', 'weights.pt'):
                if not (out / name).is_file():
                    result['error'] = f'train.py left no {name} in --out'
                    return result
            if (out / 'progress.jsonl').is_file():
                lines = (out / 'progress.jsonl').read_text().splitlines()
                result['progress_rows'] = len(lines)
                result['last_progress'] = lines[-1][:1500] if lines else None
            else:
                result['warnings'].append('no progress.jsonl: `ar log` will show only the trainer output')
            if (out / 'weights.pt').stat().st_size > 128 * 1024 * 1024:
                result['error'] = 'weights.pt exceeds 128 MiB'
                return result
            result['stage'] = 'evaluate'
            done = subprocess.run([sys.executable, '-c', PROBE, str(root), str(BATCH), str(POSITION_SECONDS),
                                   str(CALL_SECONDS)], cwd=root, env=env,
                                  capture_output=True, text=True, timeout=300)
            if done.returncode != 0:
                try:
                    result['error'] = json.loads(done.stdout.strip().splitlines()[-1])['error']
                except (ValueError, IndexError, KeyError):
                    result['error'] = 'Loading or calling the policy failed: ' + (done.stderr or done.stdout)[-2000:]
                return result
            result.update(json.loads(done.stdout.strip().splitlines()[-1]))
            if max(result['weights_tensors'], result['policy_tensors']) > TENSOR_CAP:
                result['error'] = f'The policy holds more than {TENSOR_CAP:,} tensor elements'
                return result
            result['move_clock'] = (f'{CALL_SECONDS:g} s + {POSITION_SECONDS:g} s per position in the call '
                                    f'({CALL_SECONDS + BATCH * POSITION_SECONDS:g} s for {BATCH} positions)')
            if result['late_calls']:
                result['warnings'].append(f'{result["late_calls"]} of {result["calls"]} act() calls were slower than '
                                          'the evaluator\'s move clock on this CPU (one thread). In evaluation a late '
                                          'call is replaced by random moves for every game in it. The evaluation '
                                          'CPU is not this one: leave a margin')
            result['stage'] = 'done'
            result['passed'] = True
            return result
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        return result


if __name__ == '__main__':
    print(json.dumps(main()))
