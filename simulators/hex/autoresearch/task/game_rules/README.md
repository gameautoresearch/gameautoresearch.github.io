# Hex 13x13 rulebook

Mode `hex13-no-swap-v1`. Everything here describes the frozen engine `simulators.hex.engine` and the
evaluator exactly; the engine source is in `simulator/simulators/hex/engine.py`.

| Chapter | What it covers |
|---|---|
| [Rules](rules.md) | Board, cells, neighbours, edges, moves, winning, why there are no draws, symmetries |
| [Interface](interface.md) | The arrays a policy receives and returns, the canonical frame, the engine API for training |
| [Evaluation](evaluation.md) | Openings, seats, the opponents, the score, Elo against the anchor, practice and faults |
| [Scope and limitations](scope-and-limitations.md) | What this mode leaves out and what the anchor is and is not |

## The game in brief

- A 13x13 rhombus of 169 hexagonal cells, all empty at the start.
- Black moves first; then the players alternate. A move places one stone of your colour on any empty
  cell. There is no swap (pie) rule and no passing.
- Black wins by completing a chain of black stones from the top edge (row 1) to the bottom edge (row 13).
  White wins with a chain from the left edge (column a) to the right edge (column m).
- A full board always contains exactly one such chain, so there are no draws, and every game ends after
  at most 169 moves.

## Coordinates

Cells are numbered 0 to 168 in row-major order from the top-left: `cell = 13 * row + col`, with `row` and
`col` from 0 to 12. Written names use a column letter (a to m, the letter i included) and a 1-based row, as
MoHex and HexGui do: cell 0 is `a1`, cell 12 is `m1`, cell 84 (the centre) is `g7` and cell 168 is `m13`.

Each row is shifted half a cell to the right of the one above, so the board leans:

        a b c d e f g h i j k l m
     1   . . . . . . . . . . . . .
     2    . . . . . . . . . . . . .
     3     . . . . . . . . . . . . .
     4      . . . . . . . . . . . . .
     5       . . . . . . . . . . . . .
     6        . . . . . . . . . . . . .
     7         . . . . . . . . . . . . .
     8          . . . . . . . . . . . . .
     9           . . . . . . . . . . . . .
    10            . . . . . . . . . . . . .
    11             . . . . . . . . . . . . .
    12              . . . . . . . . . . . . .
    13               . . . . . . . . . . . . .

The two obtuse corners are `m1` and `a13`; the two acute corners are `a1` and `m13`.
