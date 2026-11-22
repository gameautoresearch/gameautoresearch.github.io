"""Export fresh, public demonstration self-play; never read evaluation trajectories.

Run with the game's venv. Only the frozen policy adapter is executed; checkpoint
source and runtime hashes are verified before loading tensor-only weights.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from integrations.doudizhu.environment import Game, ROLES, RULESET, deal, replay, digest
from integrations.doudizhu.autoresearch.arena import TrustedPanelPolicy
from integrations.doudizhu.autoresearch.bundle import verify
from douzero.env.move_detector import get_move_type

DEFAULT_BUNDLE = ROOT / 'evidence/doudizhu-implementation/autoresearch-deepseek-research30-20261108/jobs/improved_v2/bundle'
PUBLIC_DEMO_SEED = 202611080901
MOVE_NAMES = ('Pass', 'Single', 'Pair', 'Triple', 'Bomb', 'Rocket',
              'Triple + single', 'Triple + pair', 'Straight', 'Consecutive pairs',
              'Airplane', 'Airplane + singles', 'Airplane + pairs',
              'Four + two singles', 'Four + two pairs')


def record_game(policy, identity: str, index: int, *, observer=None, demo_seed=None):
    seed = PUBLIC_DEMO_SEED if demo_seed is None else demo_seed
    cards = deal('practice', seed, index)
    # Produce with the unchanged upstream oracle; verify against the view engine.
    game = Game(cards, record=True, engine_backend='upstream')
    hands = {r: list(cards[r]) for r in ROLES}
    frames = [{'hands': deepcopy(hands), 'nextRole': ROLES[0], 'event': None,
               'trick': None, 'passes': 0, 'bombs': 0}]
    trick, passes, bombs = None, 0, 0
    obs = game.observe()
    while obs is not None:
        choice = policy.choose([obs])[0]
        if observer is not None:
            observer(game, obs, choice)
        action = list(obs.legal[choice])
        move_type = get_move_type(action)['type']
        if move_type >= len(MOVE_NAMES):
            raise ValueError('Unexpected move type')
        event = {'role': obs.role, 'cards': action, 'kind': MOVE_NAMES[move_type]}
        for card in action:
            hands[obs.role].remove(card)
        if action:
            trick = deepcopy(event)
            passes = 0
            bombs += int(move_type in (4, 5))
        else:
            passes += 1
        obs, outcome = game.step(choice)
        if passes == 2:
            trick = None
            passes = 0
        if obs is not None and [len(hands[r]) for r in ROLES] != list(obs.remaining):
            raise ValueError('Replay hand count disagrees with engine')
        frames.append({'hands': deepcopy(hands), 'nextRole': obs.role if obs else None,
                       'event': event, 'trick': deepcopy(trick), 'passes': passes,
                       'bombs': bombs})
    trajectory = game.trajectory()
    assert replay(trajectory) == trajectory  # All public hashes, legal moves, outcome.
    assert bombs == outcome['bombs']
    return {
        'schemaVersion': 1, 'gameId': 'doudizhu', 'id': f'deepseek-selfplay-{index + 1:02d}',
        'title': f'Table {index + 1:02d}',
        'subtitle': 'DeepSeek selected · self-play',
        'policy': {'name': 'DeepSeek research bot', 'checkpoint': identity,
                   'description': 'Selected checkpoint from the 30-minute research run. The same frozen policy controls all three seats.'},
        'provenance': {'kind': 'fresh-public-demo', 'ruleset': RULESET,
                       'demoSeed': seed, 'dealIndex': index,
                       'dealSha256': digest(cards), 'trajectorySha256': digest(trajectory),
                       'verification': 'upstream generation + full observation/legal-action/outcome replay on view engine'},
        'bottom': cards['three_landlord_cards'], 'frames': frames,
        'outcome': outcome,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--output', type=Path, default=ROOT / 'replay-site/public/replays')
    args = parser.parse_args()
    identity = verify(args.bundle)['bundle_sha256']
    policy = TrustedPanelPolicy(args.bundle)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index in range(2):
        recording = record_game(policy, identity, index)
        path = args.output / f"{recording['id']}.json"
        # Never silently overwrite demonstration records.
        with path.open('x') as target:
            json.dump(recording, target, separators=(',', ':'))
            target.write('\n')
        manifest.append({k: recording[k] for k in ('id', 'gameId', 'title', 'subtitle')}
                        | {'url': f'replays/{path.name}', 'moves': len(recording['frames']) - 1})
        print(f"{path.name}: {len(recording['frames']) - 1} legal moves, natural finish, replay verified")
    with (args.output / 'index.json').open('x') as target:
        json.dump(manifest, target, indent=2)
        target.write('\n')


if __name__ == '__main__':
    main()
