"""The batched engine against the independent BFS reference. CPU only; no GPU or MoHex needed."""
import random

import numpy as np
import pytest
import torch

from simulators.hex import engine, reference
from simulators.hex.engine import BLACK, CELLS, EMPTY, LETTERS, SIZE, WHITE, HexGames

LAST = SIZE - 1          # index of the last row and column
MID = LETTERS[SIZE // 2]  # the middle column's letter


def random_games(n, seed):
    rng = random.Random(seed)
    return [rng.sample(range(CELLS), CELLS) for _ in range(n)]


def run_batch(orders, device='cpu'):
    """Play each permutation until its game ends; return per-game (board, winner, length)."""
    games = HexGames(len(orders), device)
    orders_t = torch.tensor(orders, device=device)
    pointer = torch.zeros(len(orders), dtype=torch.long, device=device)
    while not bool(games.done.all()):
        running = ~games.done
        actions = orders_t[torch.arange(len(orders), device=device), pointer.clamp(max=CELLS - 1)]
        games.step(actions, running)
        pointer += running.long()
    return games


def test_neighbour_table_is_symmetric_and_six_connected():
    table = engine.NEIGHBOURS
    for cell in range(CELLS):
        real = [c for c in table[cell] if c != engine.PAD]
        assert sorted(real) == sorted(reference.neighbours(cell))
        for other in real:
            assert cell in table[other]
    assert (table != engine.PAD).sum(1).tolist().count(6) == (SIZE - 2) ** 2  # interior cells
    assert sorted((table[[0, LAST, SIZE * LAST, CELLS - 1]] != engine.PAD).sum(1).tolist()) == [2, 2, 3, 3]  # acute/obtuse corners


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_random_games_match_reference(seed):
    orders = random_games(400, seed)
    games = run_batch(orders)
    for i, order in enumerate(orders):
        board, winner, length = reference.play(order)
        assert games.winner[i].item() == winner != EMPTY
        assert games.moves[i].item() == length
        assert games.board[i].tolist() == board


def test_every_prefix_agrees_on_who_has_won():
    """Step one move at a time and compare the winner after every move."""
    for order in random_games(60, 9):
        games = HexGames(1)
        board = [EMPTY] * CELLS
        colour = BLACK
        for cell in order:
            won = games.step(torch.tensor([cell])).item()
            board[cell] = colour
            assert (won == colour) == reference.connects(board, colour)
            assert not reference.connects(board, BLACK if colour == WHITE else WHITE)
            if won:
                break
            colour = WHITE if colour == BLACK else BLACK


def test_straight_chains_and_edges():
    black_column = [engine.parse_cell(f'{MID}{r}') for r in range(1, SIZE + 1)]
    white_row = [engine.parse_cell(f'{c}5') for c in LETTERS]
    # Black fills the middle column top to bottom while white plays harmlessly in columns a and b.
    filler = [engine.parse_cell(f'{c}{r}') for c in 'ab' for r in range(1, SIZE + 1)][:SIZE]
    moves = [x for pair in zip(black_column, filler) for x in pair]
    games = HexGames(1).play(moves[:2 * SIZE - 2])
    assert games.winner.item() == EMPTY
    games.step(torch.tensor([black_column[-1]]))
    assert games.winner.item() == BLACK
    # A full white row wins for white; the same cells never win for black (wrong edges).
    board = [EMPTY] * CELLS
    for cell in white_row:
        board[cell] = BLACK
    assert not reference.connects(board, BLACK)
    for cell in white_row:
        board[cell] = WHITE
    assert reference.connects(board, WHITE)


def test_win_through_a_group_joined_in_the_middle():
    """The flag must spread when a stone joins a top group and a bottom group."""
    half = SIZE // 2
    top = [engine.parse_cell(f'{MID}{r}') for r in range(1, half + 1)]
    bottom = [engine.parse_cell(f'{MID}{r}') for r in range(half + 2, SIZE + 1)]
    junk = [engine.parse_cell(f'a{r}') for r in range(1, SIZE + 1)]
    moves = [x for pair in zip(top + bottom, junk) for x in pair]
    games = HexGames(1).play(moves)
    assert games.winner.item() == EMPTY
    games.step(torch.tensor([engine.parse_cell(f'{MID}{half + 1}')]))
    assert games.winner.item() == BLACK


def test_illegal_moves_raise_and_finished_games_have_no_moves():
    games = HexGames(2)
    games.step(torch.tensor([0, 1]))
    with pytest.raises(engine.IllegalMove):
        games.step(torch.tensor([0, 2]))  # occupied in game 0
    with pytest.raises(engine.IllegalMove):
        games.step(torch.tensor([CELLS, 2]))
    finished = run_batch(random_games(3, 4))
    assert not finished.legal().any()
    with pytest.raises(engine.IllegalMove):
        finished.step(torch.zeros(3, dtype=torch.long), torch.ones(3, dtype=torch.bool))


def test_partial_steps_leave_other_games_untouched():
    games = HexGames(3)
    games.step(torch.tensor([10, 20, 30]), torch.tensor([True, False, True]))
    assert games.moves.tolist() == [1, 0, 1]
    assert games.to_play.tolist() == [WHITE, BLACK, WHITE]
    assert games.board[1].abs().sum().item() == 0


def test_reset_select_and_clone_are_independent():
    games = run_batch(random_games(4, 5))
    copy = games.clone()
    games.reset(torch.tensor([True, False, True, False]))
    assert games.moves.tolist()[0] == 0 and games.to_play[0].item() == BLACK
    assert copy.moves[0].item() > 0
    picked = copy.select(torch.tensor([3, 1]))
    assert picked.board.tolist() == [copy.board[3].tolist(), copy.board[1].tolist()]


def test_symmetries_preserve_the_winner():
    """Transpose + colour swap and the half-turn rotation map games to equivalent games."""
    for order in random_games(80, 11):
        _, winner, length = reference.play(order)
        rotated = [int(engine.ROTATE[c]) for c in order]
        assert reference.play(rotated)[1:] == (winner, length)
        # Transposing swaps the edges, so the same move sequence played from white's
        # perspective: prepend nothing, but compare connection of the swapped colour.
        board, _, _ = reference.play(order[:length])
        swapped = [EMPTY] * CELLS
        for cell, value in enumerate(board):
            if value:
                swapped[int(engine.TRANSPOSE[cell])] = WHITE if value == BLACK else BLACK
        other = WHITE if winner == BLACK else BLACK
        assert reference.connects(swapped, other)


def test_canonical_frame_round_trip():
    games = HexGames(2).play([engine.parse_cell('c2')])
    games.step(torch.tensor([engine.parse_cell('g3'), engine.parse_cell('b8')]))
    games.step(torch.tensor([engine.parse_cell(f'a{SIZE}'), engine.parse_cell(f'{LETTERS[-1]}1')]))  # white to move now
    assert games.to_play.tolist() == [WHITE, WHITE]
    planes = engine.canonical(games.board, games.to_play)
    assert planes.shape == (2, 2, SIZE, SIZE)
    # Mover (white) stones appear transposed in plane 0.
    white_cell = engine.parse_cell('g3')
    row, col = divmod(int(engine.TRANSPOSE[white_cell]), SIZE)
    assert planes[0, 0, row, col] == 1
    legal = engine.canonical_legal(games.legal(), games.to_play)
    assert torch.equal(legal, (planes.sum(1) == 0).reshape(2, CELLS))
    back = engine.from_canonical(torch.tensor([5, 77]), games.to_play)
    assert back.tolist() == [int(engine.TRANSPOSE[5]), int(engine.TRANSPOSE[77])]
    black = engine.from_canonical(torch.tensor([5]), torch.tensor([BLACK]))
    assert black.tolist() == [5]


def test_notation():
    assert engine.cell_name(0) == 'a1' and engine.cell_name(168) == 'm13' and engine.cell_name(84) == 'g7'
    assert engine.cell_name(1) == 'b1' and engine.cell_name(13) == 'a2' and engine.cell_name(8) == 'i1'
    for bad in ('n1', 'a14', 'a0'):
        with pytest.raises(ValueError):
            engine.parse_cell(bad)
    assert all(engine.parse_cell(engine.cell_name(c)) == c for c in range(CELLS))
    text = engine.render(np.zeros(CELLS, dtype=int))
    assert text.splitlines()[0].strip() == 'a b c d e f g h i j k l m'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device')
def test_cuda_matches_cpu():
    orders = random_games(256, 21)
    cpu, gpu = run_batch(orders), run_batch(orders, 'cuda')
    assert torch.equal(cpu.winner, gpu.winner.cpu()) and torch.equal(cpu.board, gpu.board.cpu())


# ----- the shortest-path heuristic opponent
def test_heuristic_distances():
    from simulators.hex import heuristic
    empty = np.zeros((1, CELLS), np.int64)
    assert heuristic.distances(empty, BLACK).tolist() == [SIZE] and heuristic.distances(empty, WHITE).tolist() == [SIZE]
    column = empty.copy()
    column[0, [r * SIZE + 4 for r in range(SIZE)]] = BLACK
    assert heuristic.distances(column, BLACK).tolist() == [0]
    assert heuristic.distances(column, WHITE).tolist() == [heuristic.INF]
    partial = empty.copy()
    partial[0, [r * SIZE + 4 for r in range(5)]] = BLACK  # e1..e5: the rest of the column is left
    assert heuristic.distances(partial, BLACK).tolist() == [SIZE - 5]


def test_heuristic_takes_a_win_and_blocks_a_one_move_threat():
    from simulators.hex import heuristic
    rng = np.random.default_rng(0)
    board = np.zeros(CELLS, np.int64)
    board[[r * SIZE + 4 for r in range(LAST)]] = BLACK  # e1..e12: e13 and d13 (both touch e12) win for black
    board[[r * SIZE for r in range(LAST)]] = WHITE
    wins = (engine.parse_cell(f'e{SIZE}'), engine.parse_cell(f'd{SIZE}'))
    assert heuristic.choose(board, BLACK, rng) in wins
    # With d13 already white, e13 is black's only winning cell, and white must take it.
    board[engine.parse_cell(f'd{SIZE}')] = WHITE
    assert heuristic.choose(board, WHITE, rng) == engine.parse_cell(f'e{SIZE}')
    for order in random_games(5, 3):
        board, _, length = reference.play(order[:20])
        cell = heuristic.choose(np.array(board), BLACK, np.random.default_rng(1))
        assert board[cell] == EMPTY
