# Hex 13x13

13x13 Hex without the swap rule, evaluated by the method of Andy Jones, *Scaling Scaling Laws with Board
Games* ([arXiv:2104.03113](https://arxiv.org/abs/2104.03113)). Black moves first and joins top to bottom,
white joins left to right; no draws. The board was 9x9, the paper's largest, until 2026-11-14: research
sessions reached the near-perfect 9x9 anchor, so the benchmark moved to 13x13 with a per-move inference
cap ([readout](../../reports/hex/hex13-2026-11-14.md)). The 9x9 form is in git history at `48f226ee`. Research sessions run through the single harness (`harness/`, backend
`harness/games/hex/`); the research substrate is [`autoresearch/`](autoresearch/README.md).

| Path | Purpose |
|---|---|
| `engine.py` | The frozen engine: batched PyTorch `HexGames` (GPU and CPU), canonical frame, symmetries, notation |
| `reference.py` | Independent breadth-first rules check, used only by tests |
| `heuristic.py` | One-ply shortest-path player, the panel's rung between random and MoHex |
| `mohex.py` | GTP client for MoHex 2.0, the evaluation opponent and anchor |
| `build_mohex.sh` | Builds MoHex from `vendor/engines/benzene` (worker image and local) |
| `source.lock.json` | MoHex pin: upstream commit, file hashes and the one recorded local change |
| `requirements.txt`, `requirements-universal-lock.txt` | Worker Python dependencies and their hash lock |
| `tests/` | Engine tests against the reference |

## Setup

```sh
uv venv --python 3.12.9 .venv-hex
uv pip install --python .venv-hex/bin/python torch==2.15.0 numpy==2.4.2 "modal[api-proxy-support]==1.5.2" grpclib==0.4.9 pytest
sh simulators/hex/build_mohex.sh                       # optional locally; prints the binary path
export HEX_MOHEX=$PWD/.cache/hex/mohex-build/src/mohex/mohex
.venv-hex/bin/python -m pytest -q simulators/hex/tests simulators/hex/autoresearch/tests
.venv-hex/bin/python -m unittest harness.tests.test_hex
```

MoHex needs cmake, a C++ compiler, Boost and Berkeley DB: on Ubuntu `libboost-all-dev libdb-dev`, on
macOS `brew install cmake boost berkeley-db`. Tests that play MoHex skip when no binary is found.
`.venv-hex` is both the game interpreter and the research interpreter.

## The engine

Each game is a row of an (n, 169) int8 board (0 empty, 1 black, 2 white; `cell = 13 * row + col`). Every
stone carries two flags: whether its group touches its owner's first edge and its second edge, like
boardlaw's TOP/BOT/LEFT/RIGHT group labels. A move wins exactly when the new stone's group has both, so
the win test is a gather over six neighbours; a flag spreads through the mover's group only when the
new stone just gained it. (On the 9x9 board, 4,096 random games in one batch ran at about 460,000 moves a second on an
Apple M-series CPU; 13x13 has not been timed.) The tests check every result against `reference.py` on full random games and
after every move.

## MoHex

`vendor/engines/benzene` is benzene-vanilla-cmake at `95614769`, the repository boardlaw built. The paper
cites version `d450c01` of 25 February 2020, which is not in the public history. One local change, in
`src/util/Misc.cpp`, replaces two Boost filesystem calls that Boost 1.85 removed; it only affects how
data files are found. The evaluation runs MoHex through GTP: each move reloads the game with
`clear_board`/`play` and asks `reg_genmove`, which reseeds the search, so a level's choice depends only
on the position. One process serves any level by re-sending the level's `param_*` lines. The default
benzene build supports boards up to 13x13, so no build change was needed.
