"""The rules of one game of mid-size Generals 1v1, written out one cell and one move at a time.

A `Game` holds the true board of one game as plain Python lists (row-major, 16 x 16 including the padding, which is
mountains). `Game.step(actions)` plays one joint tick: both players' moves resolve in the rulebook's order
(game_rules/joint-order-and-trades.md), then the clock advances and production is paid
(game_rules/clock-and-outcomes.md). Everything a player may see of it is computed in `view.py`.
"""
import numpy as np

SIZE = 16                     # every board is padded to 16 x 16; cells beyond the map are mountains
PLAYERS = 2
TRUNCATION = 1000             # the game is a draw once the clock reaches this many ticks
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))   # up, down, left, right
FIELDS = ('armies', 'ownership', 'ownership_neutral', 'generals', 'castles', 'mountains', 'passable',
          'general_positions', 'teams', 'eliminated', 'time', 'winner', 'pool_idx')


def load_maps(path='/data/train-maps.npz'):
    """The map bank as a list of starting boards (one `Game` each). Copy one with `game.copy()` before playing it."""
    with np.load(path, allow_pickle=False) as bank:
        columns = {name: bank[name].tolist() for name in FIELDS}
    return [Game(**{name: columns[name][i] for name in FIELDS}) for i in range(len(columns['time']))]


class Game:
    """The true state of one game. Grids are lists of rows; `ownership[p]` is player p's grid of booleans."""

    def __init__(self, armies, ownership, ownership_neutral, generals, castles, mountains, passable,
                 general_positions, teams, eliminated, time, winner, pool_idx=0):
        self.armies = [[int(v) for v in row] for row in armies]
        self.ownership = [[[bool(v) for v in row] for row in grid] for grid in ownership]
        self.ownership_neutral = [[bool(v) for v in row] for row in ownership_neutral]
        self.generals = [[bool(v) for v in row] for row in generals]
        self.castles = [[bool(v) for v in row] for row in castles]
        self.mountains = [[bool(v) for v in row] for row in mountains]
        self.passable = [[bool(v) for v in row] for row in passable]
        self.general_positions = [[int(v) for v in row] for row in general_positions]
        self.teams = [int(v) for v in teams]
        self.eliminated = [bool(v) for v in eliminated]
        self.time = int(time)
        self.winner = int(winner)
        self.pool_idx = int(pool_idx)

    def copy(self):
        return Game(**{name: getattr(self, name) for name in FIELDS})

    def as_arrays(self):
        """The state as NumPy arrays, field by field."""
        dtypes = dict(armies=np.int32, general_positions=np.int32, teams=np.int32, time=np.int32,
                      winner=np.int32, pool_idx=np.int32)
        return {name: np.asarray(getattr(self, name), dtype=dtypes.get(name, bool)) for name in FIELDS}

    # ----------------------------------------------------------------------------------------------------- totals
    def land(self, player):
        count = 0
        for r in range(SIZE):
            for c in range(SIZE):
                if self.ownership[player][r][c]:
                    count += 1
        return count

    def army(self, player):
        total = 0
        for r in range(SIZE):
            for c in range(SIZE):
                if self.ownership[player][r][c]:
                    total += self.armies[r][c]
        return total

    def owner(self, r, c):
        """The player holding cell (r, c), or -1."""
        for p in range(PLAYERS):
            if self.ownership[p][r][c]:
                return p
        return -1

    # ------------------------------------------------------------------------------------------------- one tick
    def step(self, actions):
        """Play one joint tick. `actions` is one [pass, row, col, direction, split] per player.

        Returns (terminated, truncated, reward): terminated when a general was captured this tick (the game is
        decided), truncated when the clock reached the cap without that, reward +1/-1 for the winner/loser of a
        decided game and 0 otherwise. The game is changed in place and is not reset.
        """
        actions = [[int(v) for v in a] for a in actions]
        done_before = self.winner >= 0
        order = self.move_order(actions)
        self.moves_with_general_trades(actions, order)
        if not done_before:
            self.time += 1
        if self.winner < 0:
            self.winner = self.last_team_standing()
        self.production(active=self.winner < 0)
        terminated = self.winner >= 0
        truncated = self.time >= TRUNCATION and not terminated
        if terminated:
            reward = [1.0 if p == self.winner else -1.0 for p in range(PLAYERS)]
        else:
            reward = [0.0] * PLAYERS
        return terminated, truncated, reward

    def production(self, active=True):
        """Every 50th tick every owned cell gains one army; every 2nd tick every owned general and castle does."""
        if not active:
            return
        every_cell = self.time % 50 == 0
        structures = self.time % 2 == 0
        for r in range(SIZE):
            for c in range(SIZE):
                owned = False
                for p in range(PLAYERS):
                    if self.ownership[p][r][c]:
                        owned = True
                if not owned:
                    continue
                if every_cell:
                    self.armies[r][c] += 1
                if structures and (self.generals[r][c] or self.castles[r][c]):
                    self.armies[r][c] += 1

    def last_team_standing(self):
        alive = [not e for e in self.eliminated]
        first_alive = 0
        for p in range(PLAYERS):
            if alive[p]:
                first_alive = p
                break
        team = self.teams[first_alive]
        if not any(alive):
            return -1
        for p in range(PLAYERS):
            if alive[p] and self.teams[p] != team:
                return -1
        return team

    # --------------------------------------------------------------------------------------------- move order
    def move_order(self, actions):
        """The order in which this tick's moves resolve (game_rules/joint-order-and-trades.md)."""
        n = PLAYERS
        passes = [a[0] != 0 for a in actions]
        src = [(a[1], a[2]) for a in actions]
        dst = [(a[1] + DIRECTIONS[a[3]][0], a[2] + DIRECTIONS[a[3]][1]) for a in actions]
        clipped_src = [(_clip(r), _clip(c)) for r, c in src]
        clipped_dst = [(_clip(r), _clip(c)) for r, c in dst]
        defensive, general_attack, army = [], [], []
        for i in range(n):
            r, c = clipped_dst[i]
            mover_team = self.teams[i]
            held_by_team = False
            for owner in range(PLAYERS):
                if self.ownership[owner][r][c] and self.teams[owner] == mover_team:
                    held_by_team = True
            defensive.append(held_by_team and not passes[i])
            general_attack.append(self.generals[r][c] and not passes[i])
            sr, sc = clipped_src[i]
            army.append(-1 if passes[i] else self.armies[sr][sc])

        def ahead(j, i):
            """True when move j resolves before move i."""
            if passes[j] != passes[i]:
                return not passes[j]
            if defensive[j] != defensive[i]:
                return defensive[j]
            if general_attack[j] != general_attack[i]:
                return not general_attack[j]
            if army[j] != army[i]:
                return army[j] > army[i]
            return j < i if self.time % 2 == 0 else j > i

        rank = [sum(1 for j in range(n) if ahead(j, i)) for i in range(n)]
        enters = [[dst[j] == src[i] and not passes[j] and not passes[i] for i in range(n)] for j in range(n)]
        waits = [[enters[j][i] and not (enters[j][i] and enters[i][j]) and j != i for i in range(n)]
                 for j in range(n)]   # waits[j][i]: move i waits for move j
        order, queued = [], [False] * n
        for _ in range(n):
            free = [i for i in range(n) if not queued[i]
                    and not any(waits[j][i] and not queued[j] for j in range(n))]
            pool = free if free else [i for i in range(n) if not queued[i]]
            pick = min(pool, key=lambda i: (rank[i], i))
            order.append(pick)
            queued[pick] = True
        return order

    def moves_with_general_trades(self, actions, order):
        """Play the moves in `order`. Two players who capture each other's general on the same tick trade them."""
        consumed = [False] * PLAYERS
        for k in range(PLAYERS):
            player = order[k]
            for m in range(k + 1, PLAYERS):
                other = order[m]
                if not consumed[player] and not consumed[other] and self.is_general_trade(actions, player, other):
                    self.execute_general_trade(actions, player, other)
                    consumed[player] = consumed[other] = True
            if not consumed[player]:
                self.execute_action(player, actions[player])

    # ------------------------------------------------------------------------------------------------- moves
    def move_geometry(self, player, action):
        """(valid, destination row, destination col, source row, source col, army moved, army left behind)."""
        pass_turn, si, sj, direction, split = action
        in_bounds = 0 <= si < SIZE and 0 <= sj < SIZE
        di, dj = si + DIRECTIONS[direction][0], sj + DIRECTIONS[direction][1]
        dest_in_bounds = 0 <= di < SIZE and 0 <= dj < SIZE
        si, sj, di, dj = _clip(si), _clip(sj), _clip(di), _clip(dj)
        source_army = self.armies[si][sj]
        moved = source_army // 2 if split == 1 else source_army - 1
        moved = max(0, min(moved, source_army - 1))
        can_move = moved > 0 or source_army == 1
        valid = (pass_turn == 0 and in_bounds and dest_in_bounds and self.ownership[player][si][sj] and can_move
                 and self.passable[di][dj] and not self.eliminated[player])
        return valid, di, dj, si, sj, moved, source_army - moved

    def execute_action(self, player, action):
        valid, di, dj, si, sj, moved, _ = self.move_geometry(player, action)
        if valid:
            self.apply_move(player, si, sj, di, dj, moved)

    def apply_move(self, player, si, sj, di, dj, moved):
        """Move `moved` armies from (si, sj) to (di, dj): reinforce an own cell, otherwise attack it."""
        owners = [self.ownership[p][di][dj] for p in range(PLAYERS)]
        friendly = False
        for p in range(PLAYERS):
            if owners[p] and self.teams[p] == self.teams[player]:
                friendly = True
        ally_general = friendly and not owners[player] and self.generals[di][dj]
        target = self.armies[di][dj]
        attacker_wins = moved > target
        takes_cell = (not ally_general) if friendly else attacker_wins
        self.armies[di][dj] = target + moved if friendly else abs(target - moved)
        self.armies[si][sj] -= moved
        if takes_cell:
            for p in range(PLAYERS):
                self.ownership[p][di][dj] = p == player
            self.ownership_neutral[di][dj] = False
        captured = attacker_wins and not friendly and self.generals[di][dj] and any(owners)
        if captured:
            captured_player = owners.index(True)
            self.generals[di][dj] = False
            self.castles[di][dj] = True
            self.eliminate(captured_player, player)

    def eliminate(self, captured, capturer):
        """A captured general: the loser's cells pass to the capturer with their armies halved, rounded up."""
        for r in range(SIZE):
            for c in range(SIZE):
                if self.ownership[captured][r][c]:
                    self.armies[r][c] = (self.armies[r][c] + 1) // 2
                    for p in range(PLAYERS):
                        self.ownership[p][r][c] = p == capturer
                    if self.generals[r][c]:
                        self.castles[r][c] = True
                        self.generals[r][c] = False
        self.eliminated[captured] = True
        if self.winner < 0:
            standing = all(self.eliminated[p] or self.teams[p] == self.teams[capturer] for p in range(PLAYERS))
            if standing:
                self.winner = self.teams[capturer]

    # ------------------------------------------------------------------------------------------ general trades
    def general_cell(self, player):
        """The first cell, in row-major order, holding a live general of `player`; (0, 0) when there is none."""
        for r in range(SIZE):
            for c in range(SIZE):
                if self.generals[r][c] and self.ownership[player][r][c]:
                    return r, c
        return 0, 0

    def is_general_trade(self, actions, e, t):
        ve, dei, dej, sei, sej, me, re_ = self.move_geometry(e, actions[e])
        vt, dti, dtj, sti, stj, mt, rt = self.move_geometry(t, actions[t])
        gei, gej = self.general_cell(e)
        gti, gtj = self.general_cell(t)
        hits = (dei, dej) == (gti, gtj) and (dti, dtj) == (gei, gej)
        enemies = self.teams[e] != self.teams[t]
        garrison_t = rt if (sti, stj) == (gti, gtj) else self.armies[gti][gtj]
        garrison_e = re_ if (sei, sej) == (gei, gej) else self.armies[gei][gej]
        return ve and vt and hits and enemies and me > garrison_t and mt > garrison_e

    def execute_general_trade(self, actions, e, t):
        """Both generals fall on the same tick: each player's land passes to the other (armies halved, rounded up),
        each attacking army lands on the other's general with its surplus, and the two generals change hands."""
        _, _, _, sei, sej, me, re_ = self.move_geometry(e, actions[e])
        _, _, _, sti, stj, mt, rt = self.move_geometry(t, actions[t])
        gei, gej = self.general_cell(e)
        gti, gtj = self.general_cell(t)
        garrison_t = rt if (sti, stj) == (gti, gtj) else self.armies[gti][gtj]
        garrison_e = re_ if (sei, sej) == (gei, gej) else self.armies[gei][gej]
        surplus_e = max(0, me - garrison_t)
        surplus_t = max(0, mt - garrison_e)
        self.armies[sei][sej] = re_
        self.armies[sti][stj] = rt
        own_e = [row[:] for row in self.ownership[e]]
        own_t = [row[:] for row in self.ownership[t]]
        for r in range(SIZE):
            for c in range(SIZE):
                general_cell = (r, c) == (gei, gej) or (r, c) == (gti, gtj)
                if (own_e[r][c] or own_t[r][c]) and not general_cell:
                    self.armies[r][c] = (self.armies[r][c] + 1) // 2
        self.armies[gti][gtj] = surplus_e
        self.armies[gei][gej] = surplus_t
        for r in range(SIZE):
            for c in range(SIZE):
                general_cell = (r, c) == (gei, gej) or (r, c) == (gti, gtj)
                to_t = (own_e[r][c] and not general_cell) or (r, c) == (gei, gej)
                to_e = (own_t[r][c] and not general_cell) or (r, c) == (gti, gtj)
                if to_t:
                    for p in range(PLAYERS):
                        self.ownership[p][r][c] = p == t
                elif to_e:
                    for p in range(PLAYERS):
                        self.ownership[p][r][c] = p == e
        self.general_positions[e] = [gti, gtj]
        self.general_positions[t] = [gei, gej]


def _clip(v):
    return min(max(v, 0), SIZE - 1)
