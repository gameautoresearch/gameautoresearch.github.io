"""Unprivileged policy process. Only the wire's public arrays enter this process."""
import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path


def main():
    from .wire import CELLS, read_frame, decode, reply
    import numpy as np
    import torch
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', required=True)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    root = Path(a.bundle)
    sys.path.insert(0, str(root))
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed % 2 ** 32)
    random.seed(a.seed)
    config = json.loads((root / 'model.json').read_text())
    spec = importlib.util.spec_from_file_location('entrant_policy', root / 'policy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = module.create(config)
    if not isinstance(policy, torch.nn.Module):
        raise ValueError('create() must return a torch.nn.Module')
    if sum(p.numel() for p in policy.parameters()) + sum(p.numel() for p in policy.buffers()) > 10_000_000:
        raise ValueError('Constructed policy exceeds tensor cap')
    policy.load_state_dict(torch.load(root / 'weights.pt', map_location='cpu', weights_only=True), strict=True)
    policy.eval()
    sys.stdout.buffer.write(reply([]))  # loaded: the move clock starts with the first request
    sys.stdout.buffer.flush()
    while (raw := read_frame(sys.stdin.buffer)) is not None:
        board, to_play, legal = decode(raw)
        with torch.inference_mode():
            cells = list(module.act(policy, board, to_play, legal))
        if len(cells) != len(to_play):
            raise ValueError(f'Policy returned {len(cells)} cells for {len(to_play)} positions')
        if any(isinstance(c, bool) or int(c) != c or not 0 <= int(c) < CELLS or not legal[i, int(c)]
               for i, c in enumerate(cells)):
            raise ValueError('Illegal policy action: not an empty cell 0..168')
        sys.stdout.buffer.write(reply(cells))
        sys.stdout.buffer.flush()


if __name__ == '__main__':
    main()
