"""What one player sees of a game, and the action codec (game_rules/observations.md, game_rules/actions.md).

`observe(game, player)` is the player's fog-of-war view; `encode(view)` turns it into the `[16, 16, 14]` float32
input a policy receives and `legal_mask(view)` into the `[2049]` legal-action mask. Action ids: 0 is pass, then
1 + ((row * 16 + col) * 4 + direction) * 2 + split.
"""
import math
import numpy as np
from .game import SIZE, PLAYERS, DIRECTIONS, TRUNCATION

BOARD_SHAPE = (SIZE, SIZE)
CHANNELS = 14
ACTION_COUNT = 1 + SIZE * SIZE * 8
SPATIAL = ('armies', 'generals', 'castles', 'mountains', 'neutral_cells', 'owned_cells', 'opponent_cells',
           'fog_cells', 'structures_in_fog')
SCALARS = ('owned_land_count', 'owned_army_count', 'opponent_land_count', 'opponent_army_count', 'timestep')


def observe(game, player):
    """The view of `player`: a dict of 16 x 16 grids (lists of rows) and five public totals.

    A player sees every cell within one step (the 3 x 3 block) of a cell it holds. Unseen cells show 0 armies and
    no ownership; an unseen mountain or castle shows only as `structures_in_fog`, other unseen cells as `fog_cells`.
    """
    opponent = 1 - player
    visible = [[False] * SIZE for _ in range(SIZE)]
    for r in range(SIZE):
        for c in range(SIZE):
            if game.ownership[player][r][c]:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < SIZE and 0 <= cc < SIZE:
                            visible[rr][cc] = True
    view = {name: [[0] * SIZE for _ in range(SIZE)] for name in SPATIAL}
    for r in range(SIZE):
        for c in range(SIZE):
            seen = visible[r][c]
            structure = game.mountains[r][c] or game.castles[r][c]
            view['armies'][r][c] = game.armies[r][c] if seen else 0
            view['generals'][r][c] = game.generals[r][c] and seen
            view['castles'][r][c] = game.castles[r][c] and seen
            view['mountains'][r][c] = game.mountains[r][c] and seen
            view['neutral_cells'][r][c] = game.ownership_neutral[r][c] and seen
            view['owned_cells'][r][c] = game.ownership[player][r][c]
            view['opponent_cells'][r][c] = game.ownership[opponent][r][c] and seen
            view['fog_cells'][r][c] = (not seen) and not structure
            view['structures_in_fog'][r][c] = (not seen) and structure
    view['owned_land_count'] = game.land(player)
    view['owned_army_count'] = game.army(player)
    view['opponent_land_count'] = game.land(opponent)
    view['opponent_army_count'] = game.army(opponent)
    view['timestep'] = game.time
    return view


def encode(view):
    """The `[16, 16, 14]` float32 input of a policy: log-scaled armies, eight 0/1 planes, five scaled totals."""
    cells = SIZE * SIZE
    totals = [view['owned_land_count'] / cells,
              math.log1p(view['owned_army_count']) / 8,
              view['opponent_land_count'] / cells,
              math.log1p(view['opponent_army_count']) / 8,
              view['timestep'] / TRUNCATION]
    rows = []
    for r in range(SIZE):
        row = []
        for c in range(SIZE):
            planes = [math.log1p(view['armies'][r][c]) / 6.0]
            for name in SPATIAL[1:]:
                planes.append(1.0 if view[name][r][c] else 0.0)
            planes.extend(totals)
            row.append(planes)
        rows.append(row)
    return np.array(rows, dtype=np.float32)


def legal_mask(view):
    """The `[2049]` boolean mask of legal actions. Pass is always legal; a move needs an own cell with more than
    one army and a destination on the board that is not a mountain. Both split choices of a move are legal alike."""
    mask = [True]
    for r in range(SIZE):
        for c in range(SIZE):
            source = view['owned_cells'][r][c] and view['armies'][r][c] > 1
            for dr, dc in DIRECTIONS:
                rr, cc = r + dr, c + dc
                ok = source and 0 <= rr < SIZE and 0 <= cc < SIZE and not view['mountains'][rr][cc]
                mask.append(ok)
                mask.append(ok)
    return np.array(mask, dtype=bool)


def decode_action(action_id):
    """[pass, row, col, direction, split] of an action id."""
    action_id = int(action_id)
    index = max(action_id - 1, 0)
    return [1 if action_id == 0 else 0, index // (SIZE * 8), (index // 8) % SIZE, (index // 2) % 4, index % 2]


def encode_action(action):
    passed, row, col, direction, split = action
    return 0 if passed == 1 else 1 + ((row * SIZE + col) * 4 + direction) * 2 + split


def player_inputs(game):
    """Both players' encoded views and legal masks: (`[2, 16, 16, 14]` float32, `[2, 2049]` bool)."""
    views = [observe(game, p) for p in range(PLAYERS)]
    return np.stack([encode(v) for v in views]), np.stack([legal_mask(v) for v in views])
