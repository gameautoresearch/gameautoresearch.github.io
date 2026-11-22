"""Trusted evaluator. The submitted policy sees only public arrays over a pipe; MoHex runs as GTP subprocesses.

Every game starts from a 2-ply opening (a black stone, then a white stone), as in
boardlaw's reference evaluation, and is played twice: once with the candidate as
black and once as white. All games of a schedule advance together, one ply per
round, so the candidate answers in batches and MoHex moves run in parallel.

Inference is capped per move. A policy call covering n positions must answer within
`call_seconds + n * position_seconds` (`move_clock`): a fixed allowance per position,
plus one small constant for the pipe round trip so that a one-position call is not
decided by scheduling noise. A call answered late is replaced by a uniformly random
legal move in every game it covered, the stalled policy process is restarted, and play
continues; the substitutions are counted, never hidden.
"""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
from pathlib import Path
import queue
import resource
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

from .bundle import FILES, verify
from .contract import DEFAULT, MOHEX_LEVELS, MOHEX_STARTUP, ROOT
from .wire import CELLS, encode

SEATS = {1: 'black', 2: 'white'}


class PolicyFailure(RuntimeError):
    pass


class PolicyTimeout(Exception):
    """One policy call missed its move clock. The process has already been restarted."""


def _drop_privileges():
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_NO_NEW_PRIVS failed')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))


def move_clock(positions, position_seconds=None, call_seconds=None):
    """Seconds one policy call covering `positions` positions has to answer."""
    position_seconds = DEFAULT['position_seconds'] if position_seconds is None else position_seconds
    call_seconds = DEFAULT['call_seconds'] if call_seconds is None else call_seconds
    return call_seconds + positions * position_seconds


class PolicyProcess:
    def __init__(self, bundle, development=False, position_seconds=None, seed=0, startup_seconds=None,
                 call_seconds=None):
        verify(bundle)
        # The candidate runs as an unprivileged child whenever the evaluator can arrange it (practice
        # included); `development` only permits running elsewhere, such as a developer's machine.
        isolate = self.isolate = sys.platform == 'linux' and os.geteuid() == 0
        if not development and not isolate:
            raise RuntimeError('Official grading requires a root-owned Linux evaluator with an unprivileged policy child')
        self.tmp = tempfile.TemporaryDirectory(prefix='hex-policy-')
        root = Path(self.tmp.name)
        root.chmod(0o755)
        target = root / 'bundle'
        target.mkdir(mode=0o755)
        for name in FILES:
            shutil.copyfile(Path(bundle) / name, target / name)
            (target / name).chmod(0o444)
        scratch = root / 'scratch'
        scratch.mkdir(mode=0o700)
        if isolate:
            os.chown(scratch, 65534, 65534)
        self.position_seconds = DEFAULT['position_seconds'] if position_seconds is None else position_seconds
        self.call_seconds = DEFAULT['call_seconds'] if call_seconds is None else call_seconds
        self.startup_seconds = DEFAULT['policy_startup_seconds'] if startup_seconds is None else startup_seconds
        self.timeouts = 0
        self.err = (root / 'stderr.log').open('wb')
        code = (f'import sys;sys.path.insert(0,{str(ROOT)!r});'
                'from simulators.hex.autoresearch.policy_worker import main;main()')
        env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': str(scratch), 'TMPDIR': str(scratch),
               'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''}
        self.command = dict(args=[sys.executable, '-I', '-c', code, '--bundle', str(target), '--seed', str(seed)],
                            cwd=scratch, env=env)
        self.process = None
        try:
            self._start()
        except BaseException:
            self.close()
            raise

    def _start(self):
        """Start the policy process and wait (up to startup_seconds, not the move clock) until it has loaded."""
        self.process = subprocess.Popen(stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err,
                                        start_new_session=True, preexec_fn=_drop_privileges if self.isolate else None,
                                        **self.command)
        deadline = time.monotonic() + self.startup_seconds
        try:
            size = struct.unpack('>I', self._read(4, deadline))[0]
            if size > 64 or json.loads(self._read(size, deadline)) != []:
                raise PolicyFailure('Policy process broke its startup handshake')
        except PolicyTimeout:
            raise PolicyFailure(f'Policy did not load within {self.startup_seconds} s: ' + self.stderr_tail()) from None
        except (ValueError, OSError) as exc:
            raise PolicyFailure(type(exc).__name__ + ': ' + str(exc)) from exc

    def _stop(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        for pipe in (self.process.stdin, self.process.stdout):
            try:
                pipe.close()
            except OSError:
                pass

    def _read(self, n, deadline):
        chunks = []
        while n:
            ready, _, _ = select.select([self.process.stdout], [], [], max(0., deadline - time.monotonic()))
            if not ready:
                raise PolicyTimeout('Policy answer deadline exceeded')
            value = os.read(self.process.stdout.fileno(), n)
            if not value:
                raise PolicyFailure('Policy process exited or broke its response protocol: ' + self.stderr_tail())
            chunks.append(value)
            n -= len(value)
        return b''.join(chunks)

    def stderr_tail(self, limit=1500):
        try:
            self.err.flush()
            return Path(self.err.name).read_bytes()[-limit:].decode(errors='replace')
        except OSError:
            return ''

    def choose(self, board, to_play, legal):
        raw = encode(board, to_play, legal)
        deadline = time.monotonic() + move_clock(len(to_play), self.position_seconds, self.call_seconds)
        try:
            # Nonblocking writes enforce the same deadline if a policy stops reading stdin.
            fd = self.process.stdin.fileno()
            os.set_blocking(fd, False)
            view = memoryview(raw)
            while view:
                _, ready, _ = select.select([], [fd], [], max(0., deadline - time.monotonic()))
                if not ready:
                    raise PolicyTimeout('Policy request write deadline exceeded')
                sent = os.write(fd, view)
                view = view[sent:]
            size = struct.unpack('>I', self._read(4, deadline))[0]
            if size > 65536:
                raise PolicyFailure('Oversized policy response')
            cells = json.loads(self._read(size, deadline))
            if not isinstance(cells, list) or len(cells) != len(to_play):
                raise PolicyFailure('Wrong policy response length')
            if any(type(c) is not int or not 0 <= c < CELLS or not legal[i, c] for i, c in enumerate(cells)):
                raise PolicyFailure('Illegal policy action')
            return cells
        except PolicyTimeout:
            # The late answer would desynchronize the pipe: replace the process before anything else is asked.
            self.timeouts += 1
            self._stop()
            self._start()
            raise
        except (BrokenPipeError, EOFError, ValueError, OSError) as exc:
            raise PolicyFailure(type(exc).__name__ + ': ' + str(exc)) from exc

    def close(self):
        self._stop()
        self.err.close()
        self.tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class MoHexPool:
    """Up to `workers` MoHex processes, started on first use. Each request sets its level and reloads the game.

    `reg_genmove` reseeds from the process seed and does not play the move, so a
    level's choice depends only on the position and seed, not on which process or
    what earlier request served it (the anchor's parallel solver threads aside).

    MoHex very rarely crashes (a segfault seen once in about 300 games, with the
    anchor's multi-threaded solver running). The game is unchanged by a crash, so the
    request is asked again of a fresh process, at most RETRIES times; every restart
    is counted in `restarts` and reported. A request that keeps crashing is a
    simulator fault and ends the evaluation without a grade.
    """
    RETRIES = 2

    def __init__(self, workers, seed, binary=None):
        import threading
        self.workers, self.seed, self.binary = workers, seed, binary
        self.idle = queue.Queue()
        self.started = []
        self.restarts = []
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(workers)

    def _process(self):
        from simulators.hex.mohex import MoHex
        try:
            return self.idle.get_nowait()
        except queue.Empty:
            process = MoHex(MOHEX_STARTUP, seed=self.seed, binary=self.binary)
            process.level = None
            with self.lock:
                self.started.append(process)
            return process

    def _ask(self, process, level, moves, colour):
        from simulators.hex.engine import parse_cell
        from simulators.hex.mohex import MoHexFault
        if process.level != level:
            process.level = None
            for line in MOHEX_LEVELS[level]:
                process.query(line)
            process.level = level
        process.new_game()
        for index, cell in enumerate(moves):
            process.play(1 if index % 2 == 0 else 2, cell)
        answer = process.query('reg_genmove ' + ('b' if colour == 1 else 'w')).lower()
        try:
            return parse_cell(answer)
        except (ValueError, IndexError):
            raise MoHexFault(f'MoHex answered {answer!r} instead of a cell') from None

    def _genmove(self, level, moves, colour):
        from simulators.hex.mohex import MoHexFault
        for attempt in range(self.RETRIES + 1):
            process = self._process()
            try:
                cell = self._ask(process, level, moves, colour)
            except MoHexFault as exc:
                if process.process.poll() is None or attempt == self.RETRIES:
                    self.idle.put(process)  # a live process that refused, or the last attempt: a real fault
                    raise
                with self.lock:
                    self.restarts.append({'level': level, 'plies': len(moves), 'error': str(exc)[:300]})
                process.close()  # dead: replace it and ask again
                continue
            self.idle.put(process)
            return cell

    def genmoves(self, requests):
        """[(level, moves, colour)] -> [cell], in parallel."""
        return list(self.executor.map(lambda r: self._genmove(*r), requests))

    def close(self):
        self.executor.shutdown(wait=True)
        for process in self.started:
            process.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def opening_bank(seed, count):
    """`count` distinct 2-ply openings (black cell, white cell), drawn without replacement from all 169 x 168."""
    pairs = np.array([(b, w) for b in range(CELLS) for w in range(CELLS) if b != w])
    rng = np.random.default_rng(int(seed) % 2 ** 128)
    if not 0 < count <= len(pairs):
        raise ValueError('Invalid opening count')
    return [tuple(int(x) for x in pairs[i]) for i in rng.choice(len(pairs), count, replace=False)]


def check_panel(panel):
    """Opponent specs: {'name', 'kind': random | heuristic | mohex, 'level' (mohex; defaults to the name),
    'epsilon' (optional: the chance of a uniformly random move instead, the strength dial boardlaw's
    MoHex wrapper had)}."""
    if not panel or len({p['name'] for p in panel}) != len(panel):
        raise ValueError('Panel must be nonempty with unique names')
    for p in panel:
        level = p.get('level', p['name'])
        if p['kind'] not in ('random', 'heuristic', 'mohex') or (p['kind'] == 'mohex' and level not in MOHEX_LEVELS):
            raise ValueError('Unknown panel opponent: ' + p['name'])
        if not 0 <= p.get('epsilon', 0) < 1:
            raise ValueError('Opponent epsilon must be in [0, 1): ' + p['name'])


def start_games(openings):
    """A batch of games with each 2-ply opening played, and each game's move list."""
    import torch
    from simulators.hex.engine import HexGames
    games = HexGames(len(openings))
    games.step(torch.tensor([o[0] for o in openings]))
    won = games.step(torch.tensor([o[1] for o in openings]))
    if bool((won != 0).any()):
        raise RuntimeError('An opening ended a game')
    return games, [list(o) for o in openings]


def opponent_moves(movers, games, legal, rngs, history, pool):
    """{game index: cell} for [(game index, opponent spec)] whose side is to move. MoHex requests run in parallel.

    Each game's own generator decides an epsilon move first (when the spec has one), then
    random and heuristic choices, so a game's moves never depend on the other games.
    """
    from simulators.hex import heuristic
    moves, requests, targets = {}, [], []
    for index, spec in movers:
        rng = rngs[index]
        if spec.get('epsilon') and rng.random() < spec['epsilon'] or spec['kind'] == 'random':
            moves[index] = int(rng.choice(legal[index].nonzero().squeeze(1).numpy()))
        elif spec['kind'] == 'heuristic':
            moves[index] = heuristic.choose(games.board[index].numpy(), int(games.to_play[index]), rng)
        else:
            requests.append((spec.get('level', spec['name']), history[index], int(games.to_play[index])))
            targets.append(index)
    for index, cell in zip(targets, pool.genmoves(requests)):
        if not legal[index, cell]:
            raise RuntimeError(f'Simulator fault: MoHex chose an occupied cell in game {index}')
        moves[index] = cell
    return moves


def evaluate(bundle, panel, *, opening_seed, policy_seed, openings, split='heldout', development=False,
             workers=None, binary=None, batch_size=None):
    import torch
    from simulators.hex.engine import IllegalMove
    torch.set_num_threads(1)
    if split not in ('practice', 'heldout') or openings <= 0:
        raise ValueError('Invalid evaluation schedule')
    check_panel(panel)
    batch_size = batch_size or DEFAULT['inference_batch']
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    bank = opening_bank(opening_seed, openings)
    jobs = [(p, i, seat) for p in panel for i in range(openings) for seat in (1, 2)]
    n = len(jobs)
    games, history = start_games([bank[i] for _, i, _ in jobs])
    seat = torch.tensor([s for _, _, s in jobs], dtype=torch.int8)
    rngs = [np.random.default_rng([int(policy_seed) % 2 ** 64, index]) for index in range(n)]
    # Random moves that replace late candidate answers draw from their own streams, so opponents' draws never shift.
    stand_ins = [np.random.default_rng([int(policy_seed) % 2 ** 64, index, 1]) for index in range(n)]
    random_moves = [0] * n
    started = time.monotonic()
    decisions = 0
    with PolicyProcess(bundle, development, seed=int(policy_seed) % 2 ** 31) as candidate, \
            MoHexPool(workers, int(policy_seed) % 2 ** 31, binary) as pool:
        while not bool(games.done.all()):
            running = ~games.done
            actions = torch.zeros(n, dtype=torch.long)
            legal = games.legal()
            mine = (running & (games.to_play == seat)).nonzero().squeeze(1).tolist()
            for cursor in range(0, len(mine), batch_size):
                ids = mine[cursor:cursor + batch_size]
                try:
                    chosen = candidate.choose(games.board[ids].numpy(), games.to_play[ids].numpy(),
                                              legal[ids].numpy())
                except PolicyTimeout:
                    chosen = [int(stand_ins[i].choice(legal[i].nonzero().squeeze(1).numpy())) for i in ids]
                    for i in ids:
                        random_moves[i] += 1
                actions[ids] = torch.tensor(chosen)
                decisions += len(ids)
            theirs = (running & (games.to_play != seat)).nonzero().squeeze(1).tolist()
            for index, cell in opponent_moves([(i, jobs[i][0]) for i in theirs], games, legal, rngs, history,
                                              pool).items():
                actions[index] = cell
            try:
                games.step(actions, running)
            except IllegalMove as exc:
                raise RuntimeError('Simulator fault: ' + str(exc)) from exc
            for index in running.nonzero().squeeze(1).tolist():
                history[index].append(int(actions[index]))
        move_timeouts = candidate.timeouts
    restarts = list(pool.restarts)
    rows = []
    for index, (p, opening, s) in enumerate(jobs):
        rows.append({'opponent': p['name'], 'opening_index': opening, 'opening': list(bank[opening]),
                     'seat': SEATS[s], 'won': int(games.winner[index]) == s, 'moves': history[index],
                     'plies': len(history[index]), 'natural': True, 'truncated': False,
                     'random_moves': random_moves[index]})
    if len(rows) != n or not bool(games.done.all()):
        raise RuntimeError('Incomplete evaluation; no partial grade awarded')
    return {'rows': rows, 'seconds': time.monotonic() - started, 'games': n, 'candidate_decisions': decisions,
            'faults': 0, 'illegal_actions': 0, 'truncations': 0, 'mohex_restarts': restarts,
            'position_seconds': candidate.position_seconds, 'call_seconds': candidate.call_seconds,
            'move_timeouts': move_timeouts,
            'random_moves': sum(random_moves),
            'development_only': development, 'split': split}
