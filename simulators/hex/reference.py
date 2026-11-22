"""Independent pure-Python Hex rules, used only to check the batched engine.

Winner detection is a breadth-first search from each player's first edge over the
whole board, sharing nothing with `engine.py` except the cell numbering.
"""
from collections import deque

SIZE = 13
EMPTY, BLACK, WHITE = 0, 1, 2


def neighbours(cell, size=SIZE):
    row, col = divmod(cell, size)
    for dr, dc in ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)):
        r, c = row + dr, col + dc
        if 0 <= r < size and 0 <= c < size:
            yield r * size + c


def connects(board, colour, size=SIZE):
    """Does `colour` have a chain between its two edges? Black: top-bottom; white: left-right."""
    if colour == BLACK:
        start = [c for c in range(size) if board[c] == BLACK]
        goal = lambda cell: cell // size == size - 1
    else:
        start = [r * size for r in range(size) if board[r * size] == WHITE]
        goal = lambda cell: cell % size == size - 1
    seen, queue = set(start), deque(start)
    while queue:
        cell = queue.popleft()
        if goal(cell):
            return True
        for other in neighbours(cell, size):
            if other not in seen and board[other] == colour:
                seen.add(other)
                queue.append(other)
    return False


def play(moves, size=SIZE):
    """Play cells alternately from black. Returns (board, winner, moves played until the win)."""
    board = [EMPTY] * (size * size)
    colour = BLACK
    for index, cell in enumerate(moves):
        if board[cell] != EMPTY:
            raise ValueError('Occupied cell')
        board[cell] = colour
        if connects(board, colour, size):
            return board, colour, index + 1
        colour = WHITE if colour == BLACK else BLACK
    return board, EMPTY, len(moves)
