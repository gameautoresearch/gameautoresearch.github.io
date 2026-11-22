"""The frozen game mode of this task (fixed; not editable, not research code).

This is the one place the board, the map type and the round limit are defined. The trainer's defaults, the evaluator
(v1/grade.py, v1/policy.py, v1/preflight.py), the host's configuration check (v1/core.py) and the release constructor
all read these names; recipes, the contract and the rulebook are checked against them by the tests.
"""
SIZE = 11                    # board side (the engine supports 11, 14 and 18)
PLAYERS = 2                  # both Bardur
MAPS = 'drylands'            # the single map type (maps.stream): land only
ROUND_LIMIT = 30             # `max_turns`: a game that reaches this round is cut and goes to the higher in-game score
MAP_SEED_LOW, MAP_SEED_HIGH = 200000, 300000   # the training layouts
MAP_TABLE = f'table-{MAPS}-{SIZE}-{PLAYERS}p-{MAP_SEED_LOW}-{MAP_SEED_HIGH}.npz'
# The configuration fields a recipe may not change.
FROZEN = dict(size=SIZE, players=PLAYERS, maps=MAPS, max_turns=ROUND_LIMIT, map_seed_low=MAP_SEED_LOW,
              map_seed_high=MAP_SEED_HIGH)
