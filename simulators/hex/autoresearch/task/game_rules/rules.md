# Rules

## Board and neighbours

The board has 169 cells in 13 rows and 13 columns. Cell (row r, column c), numbered `13r + c`, touches up to
six cells:

| Direction | Neighbour |
|---|---|
| up | (r - 1, c) |
| up-right | (r - 1, c + 1) |
| left | (r, c - 1) |
| right | (r, c + 1) |
| down-left | (r + 1, c - 1) |
| down | (r + 1, c) |

Neighbours outside the board do not exist. Interior cells have six neighbours, non-corner edge cells
four, the obtuse corners `m1` (cell 12) and `a13` (cell 156) three, and the acute corners `a1` (cell 0) and
`m13` (cell 168) two. Note that (r - 1, c - 1) and (r + 1, c + 1) are **not** neighbours.

## Edges and colours

| Colour | Moves | First edge | Second edge |
|---|---|---|---|
| Black (1) | first | top: row 1, cells 0 to 12 | bottom: row 13, cells 156 to 168 |
| White (2) | second | left: column a, cells 0, 13, ..., 156 | right: column m, cells 12, 25, ..., 168 |

A corner cell lies on one edge of each colour; it counts for whichever colour occupies it.

## Moves

The players alternate, black first. A move places one stone of the mover's colour on an empty cell.
Stones never move or leave the board. There is no swap (pie) rule: black's first stone stays black, and
white cannot take it over. There is no passing. The engine rejects a move on an occupied cell, a cell
outside 0 to 168, or any move in a finished game.

## Winning

A chain is a set of same-coloured stones connected through neighbouring cells. A player wins as soon as
one of their chains contains a cell on each of their two edges. The game ends immediately; the winner is
the player who just moved. Only the mover can complete a chain, so the engine tests each new stone: it
wins exactly when its chain then touches both of its owner's edges.

## No draws, and the first-move advantage

Once every cell is filled, exactly one player has a winning chain (a classical theorem of Hex), so a
game never ends without a winner, and it ends after at most 169 moves; in practice usually well before.
By a strategy-stealing argument the first player (black) has a winning strategy from the empty board.
That is why tournament Hex uses the swap rule; this mode does not, and the evaluation compensates by
starting every game from a 2-ply opening played from both seats (see [Evaluation](evaluation.md)).
Some opening cells, many of them on the edges, lose for the side that played them.

## Symmetries

Two transformations of the board preserve the neighbour relation:

- **Half-turn**: (r, c) to (12 - r, 12 - c), that is cell `x` to `168 - x`. Each colour keeps its edges, so a
  position and its half-turn have the same winner with the same side to move.
- **Transpose with colour swap**: (r, c) to (c, r), that is cell `13r + c` to `13c + r`, together with
  swapping black and white stones. It exchanges the top and bottom edges with the left and right edges,
  so it turns a position for one colour into the mirror position for the other.

The engine exposes both as index tables, `ROTATE` and `TRANSPOSE`. The canonical frame in
[Interface](interface.md) uses the transpose so that the side to move always joins top to bottom.
