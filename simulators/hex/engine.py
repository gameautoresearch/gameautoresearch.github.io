"""Batched 13x13 Hex in PyTorch: the one engine used for training, practice and grading.

Mode `hex13-no-swap-v1`: the rules and seat-paired evaluation of Jones, "Scaling Scaling Laws with
Board Games" (2021), on a 13x13 board (the paper stops at 9x9, which research sessions saturated):

- a 13x13 rhombus of hexagonal cells, indexed `cell = 13 * row + col` from the top-left;
- black moves first and connects the top edge (row 0) to the bottom edge (row 12);
- white connects the left edge (col 0) to the right edge (col 12);
- players alternate placing one stone on any empty cell; there is no swap (pie) rule
  and no passing. A game always ends with exactly one winner, after at most 169 stones.

Cell (r, c) touches (r-1, c), (r-1, c+1), (r, c-1), (r, c+1), (r+1, c-1) and (r+1, c).

The engine keeps, for every stone, whether its group already touches its owner's
first edge and its owner's second edge (as boardlaw's CUDA kernel labels groups
TOP/BOT/LEFT/RIGHT). A move wins exactly when the new stone joins both. Every tensor
lives on one device, so the same code runs batched on a GPU worker and on CPU.
"""
import numpy as np
import torch

SIZE = 13
CELLS = SIZE * SIZE
EMPTY, BLACK, WHITE = 0, 1, 2
COLOURS = {BLACK: 'black', WHITE: 'white'}
DIRECTIONS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))
PAD = CELLS  # neighbour slots beyond the board point here; that slot is never a stone


def neighbour_table(size=SIZE):
    """(size*size, 6) neighbour cells, PAD (= size*size) where a direction leaves the board."""
    table = np.full((size * size, 6), size * size, dtype=np.int64)
    for row in range(size):
        for col in range(size):
            for k, (dr, dc) in enumerate(DIRECTIONS):
                r, c = row + dr, col + dc
                if 0 <= r < size and 0 <= c < size:
                    table[row * size + col, k] = r * size + c
    return table


NEIGHBOURS = neighbour_table()
_rows, _cols = np.divmod(np.arange(CELLS), SIZE)
# [colour, cell]: does this cell lie on the colour's first / second edge? Index 0 (empty) is unused.
FIRST_EDGE = np.stack([np.zeros(CELLS, bool), _rows == 0, _cols == 0])
SECOND_EDGE = np.stack([np.zeros(CELLS, bool), _rows == SIZE - 1, _cols == SIZE - 1])
# The transpose (r, c) -> (c, r) preserves adjacency and swaps the two players' edges.
TRANSPOSE = (_cols * SIZE + _rows).astype(np.int64)
# The half-turn (r, c) -> (12 - r, 12 - c) preserves adjacency and each player's edges.
ROTATE = (CELLS - 1 - np.arange(CELLS)).astype(np.int64)


class IllegalMove(ValueError):
    pass


class HexGames:
    """`n` independent games on `device`.

    board    (n, 169) int8: EMPTY, BLACK or WHITE
    to_play  (n,)   int8: BLACK or WHITE, whose turn it is (meaningless once a game is over)
    winner   (n,)   int8: EMPTY while a game is running, else the winning colour
    moves    (n,)   int16: stones on the board
    """

    def __init__(self, n, device='cpu'):
        self.n = int(n)
        self.device = torch.device(device)
        d = self.device
        self._neighbours = torch.as_tensor(NEIGHBOURS, device=d)
        self._first_edge = torch.as_tensor(FIRST_EDGE, device=d)
        self._second_edge = torch.as_tensor(SECOND_EDGE, device=d)
        self._rows = torch.arange(self.n, device=d)
        self.board = torch.zeros((self.n, CELLS), dtype=torch.int8, device=d)
        self.to_play = torch.full((self.n,), BLACK, dtype=torch.int8, device=d)
        self.winner = torch.zeros((self.n,), dtype=torch.int8, device=d)
        self.moves = torch.zeros((self.n,), dtype=torch.int16, device=d)
        # Edge-connection flags per stone, with a PAD column that stays False.
        self._first = torch.zeros((self.n, CELLS + 1), dtype=torch.bool, device=d)
        self._second = torch.zeros((self.n, CELLS + 1), dtype=torch.bool, device=d)

    # ----- queries
    @property
    def done(self):
        return self.winner != EMPTY

    def legal(self):
        """(n, 169) bool: empty cells of running games. A finished game has no legal move."""
        return (self.board == EMPTY) & ~self.done[:, None]

    def clone(self):
        other = HexGames.__new__(HexGames)
        other.__dict__.update({k: (v.clone() if torch.is_tensor(v) else v) for k, v in self.__dict__.items()})
        return other

    def select(self, index):
        """A new batch holding copies of the games at `index` (a bool mask or integer indices)."""
        index = torch.as_tensor(index, device=self.device)
        if index.dtype == torch.bool:
            index = index.nonzero().squeeze(1)
        other = HexGames(len(index), self.device)
        for name in ('board', 'to_play', 'winner', 'moves', '_first', '_second'):
            setattr(other, name, getattr(self, name)[index].clone())
        return other

    # ----- transitions
    def reset(self, where=None):
        """Empty the board of the selected games (all by default); black to play."""
        where = torch.ones(self.n, dtype=torch.bool, device=self.device) if where is None else where
        self.board[where] = EMPTY
        self.to_play[where] = BLACK
        self.winner[where] = EMPTY
        self.moves[where] = 0
        self._first[where] = False
        self._second[where] = False

    def step(self, actions, active=None):
        """Place a stone for the side to move in every selected running game.

        `actions` (n,) are cells 0..168; entries of unselected games are ignored.
        `active` defaults to every game that is still running. Returns (n,) int8:
        the colour that won with this move, EMPTY otherwise. An occupied cell, an
        out-of-range cell or a move in a finished game raises IllegalMove.
        """
        actions = torch.as_tensor(actions, device=self.device).long()
        if actions.shape != (self.n,):
            raise IllegalMove(f'Expected {self.n} actions, got shape {tuple(actions.shape)}')
        active = ~self.done if active is None else torch.as_tensor(active, device=self.device).bool()
        result = torch.zeros(self.n, dtype=torch.int8, device=self.device)
        if not bool(active.any()):
            return result
        rows = self._rows[active]
        cells = actions[active]
        if bool(((cells < 0) | (cells >= CELLS)).any()):
            raise IllegalMove('Cell outside the 13x13 board')
        if bool(self.done[rows].any()):
            raise IllegalMove('Move in a finished game')
        if bool((self.board[rows, cells] != EMPTY).any()):
            raise IllegalMove('Cell already occupied')
        colour = self.to_play[rows]
        colour_index = colour.long()
        self.board[rows, cells] = colour
        own = torch.cat([self.board[rows] == colour[:, None],
                         torch.zeros((len(rows), 1), dtype=torch.bool, device=self.device)], 1)
        around = self._neighbours[cells]                                            # (k, 6)
        own_around = own.gather(1, around)
        first = self._first_edge[colour_index, cells] | (self._first[rows].gather(1, around) & own_around).any(1)
        second = self._second_edge[colour_index, cells] | (self._second[rows].gather(1, around) & own_around).any(1)
        self._first[rows, cells] = first
        self._second[rows, cells] = second
        self._spread(self._first, rows, own, first)
        self._spread(self._second, rows, own, second)
        won = first & second
        self.winner[rows] = torch.where(won, colour, self.winner[rows])
        result[rows] = torch.where(won, colour, result[rows])
        self.to_play[rows] = torch.where(won, colour, (BLACK + WHITE) - colour).to(torch.int8)
        self.moves[rows] += 1
        return result

    def _spread(self, flags, rows, own, changed):
        """Extend a newly set edge flag through the mover's group (only in games where it was set)."""
        if not bool(changed.any()):
            return
        rows, own = rows[changed], own[changed]
        current = flags[rows]
        mine = current & own  # the mover's flagged stones; the opponent's flags never spread
        while True:
            reached = mine[:, self._neighbours].any(2)                               # (k, 169)
            grown = mine.clone()
            grown[:, :CELLS] |= reached & own[:, :CELLS]
            if torch.equal(grown, mine):
                break
            mine = grown
        flags[rows] = current | mine

    def play(self, moves):
        """Play the same cell sequence in every game (openings, tests)."""
        for cell in moves:
            self.step(torch.full((self.n,), int(cell), device=self.device))
        return self


# ----- observation and action frames
def canonical(board, to_play):
    """(n, 2, 13, 13) float32 planes [stones of the side to move, opponent stones].

    For white to move the board is transposed, so the side to move always connects
    top to bottom; map a chosen canonical cell back with `from_canonical`.
    """
    board = torch.as_tensor(board)
    to_play = torch.as_tensor(to_play, device=board.device).to(board.dtype)
    white = to_play == WHITE
    transpose = torch.as_tensor(TRANSPOSE, device=board.device)
    oriented = torch.where(white[:, None], board[:, transpose], board)
    mine = oriented == to_play[:, None]
    theirs = (oriented != EMPTY) & ~mine
    return torch.stack([mine, theirs], 1).float().reshape(-1, 2, SIZE, SIZE)


def canonical_legal(legal, to_play):
    """Legal mask in the canonical frame of `canonical`."""
    legal = torch.as_tensor(legal)
    white = torch.as_tensor(to_play, device=legal.device) == WHITE
    transpose = torch.as_tensor(TRANSPOSE, device=legal.device)
    return torch.where(white[:, None], legal[:, transpose], legal)


def from_canonical(cells, to_play):
    """Board cells for canonical-frame cells (the transpose is its own inverse)."""
    cells = torch.as_tensor(cells).long()
    white = torch.as_tensor(to_play, device=cells.device) == WHITE
    transpose = torch.as_tensor(TRANSPOSE, device=cells.device)
    return torch.where(white, transpose[cells], cells)


to_canonical = from_canonical


# ----- notation and display
LETTERS = 'abcdefghijklm'  # benzene's notation keeps the letter i


def cell_name(cell):
    """MoHex/HexGui notation: column letter then 1-based row, e.g. cell 0 -> 'a1', cell 84 -> 'g7'."""
    row, col = divmod(int(cell), SIZE)
    return LETTERS[col] + str(row + 1)


def parse_cell(name):
    name = name.strip().lower()
    col, row = LETTERS.index(name[0]), int(name[1:]) - 1  # a letter beyond the board raises ValueError
    if not 0 <= row < SIZE:
        raise ValueError('Row outside the board: ' + name)
    return row * SIZE + col


def render(board):
    """ASCII rhombus of one board (169 values): '.', 'B', 'W'; row r is indented by r spaces."""
    values = [int(v) for v in (board.tolist() if hasattr(board, 'tolist') else board)]
    lines = ['  ' + ' '.join(LETTERS)]
    for row in range(SIZE):
        cells = ' '.join('.BW'[values[row * SIZE + col]] for col in range(SIZE))
        lines.append(f'{row + 1:>2}' + ' ' * (row + 1) + cells)
    return '\n'.join(lines)
