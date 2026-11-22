"""CPU process/native thread workers and lazy resident-CUDA rollout selection."""
from __future__ import annotations

import multiprocessing as mp
import time
import traceback

from .environment import Game, CompactObservation


def _serve(pipe, compact=False):
    games = {}
    try:
        while True:
            op, payload = pipe.recv()
            if op == 'close':
                break
            if op == 'reset':
                games = {i: Game(cards, objective, record, cached, backend)
                         for i, cards, objective, record, cached, backend in payload}
                result = {i: (g.observe(), None, None) for i, g in games.items()}
            elif op == 'step':
                result = {}
                for i, action in payload:
                    game = games[i]
                    obs, outcome = game.step(action)
                    trace = game.trajectory() if outcome is not None and game.record else None
                    result[i] = obs, outcome, trace
            else:
                raise ValueError(op)
            if compact:
                result = {i:(CompactObservation.from_observation(obs),outcome,trace)
                          for i,(obs,outcome,trace) in result.items()}
            pipe.send(('ok', result))
    except EOFError:
        pass
    except BaseException:
        pipe.send(('error', traceback.format_exc()))
    finally:
        pipe.close()


class VectorGames:
    def __init__(self, workers=0, timeout=120, cache_observations=True, engine_backend='view',
                 compact=False,record_limit=None):
        if workers < 0 or timeout <= 0:
            raise ValueError('workers must be nonnegative and timeout positive')
        self.workers = workers
        self.timeout = timeout
        self.processes = []
        self.pipes = []
        self.games = {}
        self.cache_observations = cache_observations
        self.engine_backend = engine_backend
        self.closed = False
        self.record_limit = record_limit
        self.native = None
        self.cuda_mode = engine_backend=='cuda'
        self.cuda = None
        if self.cuda_mode:
            return
        if engine_backend == 'native':
            from .native_backend import NativeVectorGames
            self.native = NativeVectorGames(max(1,workers),record_limit=record_limit)
            return
        if workers:
            ctx = mp.get_context('spawn')
            for _ in range(workers):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_serve, args=(child,compact), daemon=True)
                process.start()
                child.close()
                self.processes.append(process)
                self.pipes.append(parent)

    def _call(self, op, payload):
        if self.closed:
            raise RuntimeError('Simulator workers are closed')
        shards = [[] for _ in self.pipes]
        for item in payload:
            shards[item[0] % len(shards)].append(item)
        # Finished/idle shards need no step round trip. Reset must clear every shard.
        requests = [(pipe,shard) for pipe,shard in zip(self.pipes,shards) if shard or op == 'reset']
        deadline = time.monotonic()+self.timeout
        for pipe, shard in requests:
            pipe.send((op, shard))
        result = {}
        for pipe, _ in requests:
            if not pipe.poll(max(0.,deadline-time.monotonic())):
                raise TimeoutError('Simulator worker timeout; no game scores awarded')
            status, data = pipe.recv()
            if status != 'ok':
                raise RuntimeError(f'Simulator worker fault: {data}')
            result.update(data)
        return result

    def reset(self, deals, objective='wp', record=False):
        if self.closed:
            raise RuntimeError('Simulator workers are closed')
        if self.cuda_mode:
            raise RuntimeError('Resident simulation uses collect() and public device buffers; use CudaBatch for low-level reset/step')
        if self.native is not None:
            return self.native.reset(deals,objective,record)
        payload = [(i, cards, objective, record and (self.record_limit is None or i<self.record_limit),
                    self.cache_observations, self.engine_backend)
                   for i, cards in enumerate(deals)]
        if self.workers:
            return self._call('reset', payload)
        self.games = {i: Game(cards, obj, rec, cached, backend)
                      for i, cards, obj, rec, cached, backend in payload}
        return {i: (g.observe(), None, None) for i, g in self.games.items()}

    def step(self, actions):
        if self.closed:
            raise RuntimeError('Simulator workers are closed')
        if self.cuda_mode:
            raise RuntimeError('Resident simulation uses collect() and public device buffers; use CudaBatch for low-level reset/step')
        if self.native is not None:
            return self.native.step(actions)
        if self.workers:
            return self._call('step', list(actions.items()))
        result = {}
        for i, action in actions.items():
            game = self.games[i]
            obs, outcome = game.step(action)
            result[i] = obs, outcome, game.trajectory() if outcome and game.record else None
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.cuda_mode:
            if self.cuda is not None:self.cuda.close()
            self.cuda=None
            return
        if self.native is not None:
            self.native.close()
            return
        # EOF is nonblocking, even if a failed worker stopped draining its pipe.
        for pipe in self.pipes:
            pipe.close()
        deadline = time.monotonic()+3
        for process in self.processes:
            process.join(timeout=max(0.,deadline-time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic()+3
        for process in self.processes:
            process.join(timeout=max(0.,deadline-time.monotonic()))
            if process.is_alive():
                process.kill()
                process.join(timeout=1)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
