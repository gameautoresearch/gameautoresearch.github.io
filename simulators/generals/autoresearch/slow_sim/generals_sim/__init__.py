"""The Generals simulator for this task: mid-size fog-of-war 1v1 (game_rules/).

    from generals_sim import load_maps, player_inputs, decode_action
    maps = load_maps('/data/train-maps.npz')   # the 256 training maps, one starting Game each
    game = maps[0].copy()
    x, mask = player_inputs(game)               # [2, 16, 16, 14] float32 inputs and [2, 2049] legal masks
    terminated, truncated, reward = game.step([decode_action(i) for i in (0, 0)])

`game.py` holds the rules of one game, `view.py` what a player sees of it and the action codec.
"""
from .game import Game, load_maps, SIZE, PLAYERS, DIRECTIONS, TRUNCATION, FIELDS
from .view import (observe, encode, legal_mask, decode_action, encode_action, player_inputs, BOARD_SHAPE,
                   CHANNELS, ACTION_COUNT, SPATIAL, SCALARS)
