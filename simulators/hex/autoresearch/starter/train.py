"""Editable, deliberately plain self-play starter: one network plays both sides, no search, no history.

Run as `train.py --config FILE --out DIR --seed N --seconds S`. Leaves `model.json`
and `weights.pt` in DIR from the start and after every wave, and appends one JSON
line per wave to `progress.jsonl`.
"""
import argparse
import json
from pathlib import Path
import time

import torch

import learner
import policy as network
from simulators.hex.engine import HexGames, canonical, canonical_legal, from_canonical


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--seconds', type=float, required=True)
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    end = time.monotonic() + max(0., a.seconds - 2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    policy = network.create(c['architecture']).to(device)
    optimizer = learner.make_optimizer(policy, c)
    (out / 'model.json').write_text(json.dumps(c['architecture']))

    def save():
        path = out / 'weights.pt.tmp'
        torch.save({k: v.detach().cpu() for k, v in policy.state_dict().items()}, path)
        path.replace(out / 'weights.pt')

    save()
    games = HexGames(c['games_per_wave'], device)
    started = time.monotonic()
    total_games = total_moves = waves = 0
    while time.monotonic() < end:
        games.reset()
        planes, masks, actions, movers, rows = [], [], [], [], []
        with torch.no_grad():
            while not bool(games.done.all()):
                running = (~games.done).nonzero().squeeze(1)
                board, to_play = games.board[running], games.to_play[running]
                x = canonical(board, to_play)
                mask = canonical_legal(games.legal()[running], to_play)
                logits, _ = policy(x)
                choice = torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9)).sample()
                cells = torch.zeros(games.n, dtype=torch.long, device=device)
                cells[running] = from_canonical(choice, to_play)
                games.step(cells)
                planes.append(x)
                masks.append(mask)
                actions.append(choice)
                movers.append(to_play)
                rows.append(running)
        rows, movers = torch.cat(rows), torch.cat(movers)
        outcome = torch.where(games.winner[rows] == movers, 1., -1.)
        batch = {'planes': torch.cat(planes), 'mask': torch.cat(masks), 'action': torch.cat(actions),
                 'outcome': outcome}
        metrics = learner.learn(policy, optimizer, batch, c)
        save()
        waves += 1
        total_games += games.n
        total_moves += len(rows)
        elapsed = time.monotonic() - started
        row = {'wave': waves, 'games': total_games, 'moves': total_moves, 'seconds': round(elapsed, 2),
               'games_per_second': round(total_games / elapsed, 1), 'mean_plies': len(rows) / games.n,
               'black_win_rate': float((games.winner == 1).float().mean()), **metrics}
        with (out / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
