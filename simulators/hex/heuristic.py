"""A classical one-ply Hex heuristic: the shortest-path ("two-edge distance") player.

For a colour, the distance of a position is the fewest empty cells that colour must
still fill to join its two edges (its own stones cost 0, empty cells 1, opponent
stones block). The player tries every empty cell and keeps those that maximize
(opponent's distance - own distance) after the move, then picks among them with a
seeded generator. A move that wins at once has own distance 0 and is always kept.
It is a weak but sensible opponent: much stronger than random, far below MoHex.
"""
import numpy as np

from .engine import BLACK, CELLS, FIRST_EDGE, NEIGHBOURS, SECOND_EDGE, WHITE

INF = 10 ** 6


def distances(boards, colour):
    """(c, 169) boards -> (c,) shortest-path distance for `colour` (0 when already connected, INF when cut off)."""
    boards = np.asarray(boards)
    cost = np.where(boards == colour, 0, np.where(boards == 0, 1, INF)).astype(np.int64)
    dist = np.where(FIRST_EDGE[colour][None, :], cost, INF)
    padded = np.full((len(boards), CELLS + 1), INF, dtype=np.int64)
    for _ in range(CELLS):
        padded[:, :CELLS] = dist
        relaxed = np.minimum(dist, np.minimum(cost + padded[:, NEIGHBOURS].min(2), INF))
        if np.array_equal(relaxed, dist):
            break
        dist = relaxed
    return np.where(SECOND_EDGE[colour][None, :], dist, INF).min(1)


def choose(board, colour, rng):
    """One move for `colour` on a single (169,) board; `rng` is a numpy Generator."""
    board = np.asarray(board)
    empty = np.flatnonzero(board == 0)
    after = np.repeat(board[None, :], len(empty), 0)
    after[np.arange(len(empty)), empty] = colour
    other = WHITE if colour == BLACK else BLACK
    own, theirs = distances(after, colour), distances(after, other)
    if (own == 0).any():
        return int(empty[np.flatnonzero(own == 0)[0]])
    score = theirs - own
    best = np.flatnonzero(score == score.max())
    return int(empty[rng.choice(best)])
