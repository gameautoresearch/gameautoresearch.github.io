# Interface

## What a policy receives

The evaluator calls `act(policy, board, to_play, legal)` with NumPy arrays for a batch of up to 64
positions where your policy is to move:

| Argument | Shape and type | Meaning |
|---|---|---|
| `board` | (n, 169) int8 | 0 empty, 1 black stone, 2 white stone, cells in row-major order |
| `to_play` | (n,) int8 | 1 if black is to move, 2 if white is to move |
| `legal` | (n, 169) bool | True on every empty cell |

The board is the complete game state: Hex has no hidden information, no move counter rules and no
history-dependent rules, so nothing else is needed. The batch mixes positions from different games and
both colours.

## What a policy returns

`act` returns a list (or 1-D array) of n integers, one cell from 0 to 168 per position, each with
`legal[i, cell]` True. Cells are on the real board, not in the canonical frame. `create(model_config)`
receives the parsed `model.json` and must return a `torch.nn.Module` whose `state_dict` loads `weights.pt`
strictly. The policy process has one CPU thread, no GPU and no network, and is seeded once at start.
A call with n positions must return within 0.1 + 0.01 x n seconds (see [Evaluation](evaluation.md)).

## The canonical frame

The engine's helpers turn any position into the point of view of the side to move:

| Helper | Result |
|---|---|
| `canonical(board, to_play)` | (n, 2, 13, 13) float planes: plane 0 the mover's stones, plane 1 the opponent's; transposed when white is to move |
| `canonical_legal(legal, to_play)` | the legal mask in the same frame |
| `from_canonical(cells, to_play)` | real cells for canonical cells (also `to_canonical`; the transpose is its own inverse) |

In the canonical frame the side to move always joins the top edge to the bottom edge, so one network can
play both colours. Black positions are unchanged; white positions are transposed with colours swapped.

## The engine for training

`HexGames(n, device)` holds n independent games as tensors on one device:

| Member | Meaning |
|---|---|
| `board` (n, 169) int8, `to_play` (n,) int8 | as above |
| `winner` (n,) int8 | 0 while a game runs, else 1 or 2 |
| `moves` (n,) int16 | stones placed |
| `done`, `legal()` | finished games, and legal cells (none once a game is finished) |
| `step(actions, active=None)` | place the mover's stone in every running (or selected) game; returns the colour that won with this move, else 0 |
| `reset(where=None)`, `clone()`, `select(index)`, `play(cells)` | restart games, copy the batch, copy some games, play a fixed sequence |

`step` raises `IllegalMove` on an occupied cell, a cell outside the board or a move in a finished game. It
is safe to run thousands of games in one batch on the GPU; finished games stay finished until reset. The
module also has `SIZE` (13), `CELLS` (169), `NEIGHBOURS` (169 x 6, with 169 for "no neighbour"), `ROTATE`, `TRANSPOSE`, `cell_name`,
`parse_cell` and `render`. `simulators.hex.reference` is an independent breadth-first winner check.

## The training run

The starter's training command is `train.py --config FILE --out DIR --seed N --seconds S`, run with your three
training files and the config: `training/launch.py` runs it on a worker you lease, and the host runs it for a
recipe's final training. The GPU worker imports `simulators.hex` from the same frozen engine as the
evaluator. Your process must leave in `DIR`:

| File | Content |
|---|---|
| `model.json` | the JSON value passed to `create` |
| `weights.pt` | `torch.save` of a tensor-only `state_dict`, at most 128 MiB |
| `progress.jsonl` (optional) | one JSON object per line: your own record of progress, read on the worker |

At the time limit the process is stopped and whatever `model.json` and `weights.pt` hold is exported with
your exact source as a sealed package. `ar check` runs `train.py` for about 45 seconds on CPU in your
box against the real engine and then asks the resulting package for moves the way the evaluator does.
