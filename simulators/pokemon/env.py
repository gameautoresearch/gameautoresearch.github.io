"""Persistent CPU simulators. One pipe call per worker per vector phase."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import select
import struct
import mmap
import tempfile
import time

HERE = Path(__file__).resolve().parent

class WorkerError(RuntimeError):
    def __init__(self, message, category):
        super().__init__(message)
        self.category = category

def team_size_args(team_size):
    """worker.cjs arguments for a format's Max Team Size; six is unmodified gen9randombattle and adds none."""
    if type(team_size) is not int or not 1 <= team_size <= 6:
        raise ValueError('Team size must be 1..6')
    return [] if team_size == 6 else [f'--team-size={team_size}']


class Worker:
    MAX_FRAME_BYTES = 128 * 1024 * 1024
    def __init__(self, binary=False, timeout=60, shared_bytes=0, lazy=False, team_size=6):
        self.metrics = {}
        self.decode_seconds = 0.0
        self.pipe_wait_seconds = 0.0
        self.binary, self.timeout = binary, timeout
        self.lazy=lazy
        self.pending = bytearray()
        self.shared=None;self.shared_file=None;self.shared_dir=None
        if shared_bytes:
            if not binary or shared_bytes>self.MAX_FRAME_BYTES:raise ValueError('Invalid shared slab size')
            self.shared_dir=tempfile.TemporaryDirectory(prefix='pokemon-wire-')
            self.shared_path=str(Path(self.shared_dir.name)/'observations')
            self.shared_file=open(self.shared_path,'w+b');self.shared_file.truncate(shared_bytes)
            self.shared=mmap.mmap(self.shared_file.fileno(),shared_bytes)
        self.process = subprocess.Popen(["node", str(HERE / "worker.cjs"), *(['--binary'] if binary else []), *team_size_args(team_size)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
    def send(self, message):
        if self.shared is not None and message.get('op')=='reset':message=dict(message,shared_path=self.shared_path,shared_bytes=len(self.shared))
        data=(json.dumps(message, separators=(",", ":")) + "\n").encode()
        try:
            while data:
                written=self.process.stdin.write(data)
                if not written:raise BrokenPipeError('Simulator pipe closed')
                data=data[written:]
        except (OSError,ValueError) as exc:
            raise WorkerError(f"Simulator request failed: {exc}","infrastructure_failure") from exc
    def _read_more(self, deadline):
        remaining=deadline-time.monotonic()
        if remaining<=0 or not select.select([self.process.stdout],[],[],max(0,remaining))[0]:
            raise WorkerError(f"Simulator response timed out after {self.timeout}s", "infrastructure_failure")
        chunk=os.read(self.process.stdout.fileno(),65536)
        if not chunk:
            raise WorkerError(f"Simulator worker exited: {self.process.poll()}","infrastructure_failure")
        self.pending.extend(chunk)
        if len(self.pending)>self.MAX_FRAME_BYTES:
            raise WorkerError("Simulator response exceeds bounded frame size", "infrastructure_failure")
    def _read_exact(self, n, deadline):
        while len(self.pending)<n:self._read_more(deadline)
        value=bytes(self.pending[:n]);del self.pending[:n]
        return value
    def receive(self):
        waiting=time.perf_counter();deadline=time.monotonic()+self.timeout
        if self.binary:
            meta_size,data_size=struct.unpack('<II',self._read_exact(8,deadline))
            if meta_size+data_size>self.MAX_FRAME_BYTES:
                raise WorkerError("Simulator frame header exceeds limit", "infrastructure_failure")
            metadata=self._read_exact(meta_size,deadline)
            # Worker publishes metadata only after a synchronous slab write.
            # Snapshot once before the next request: trajectories must never
            # alias a slab that the simulator will overwrite on the next step.
            if self.shared is not None:
                if data_size>len(self.shared):raise WorkerError('Shared slab overflow','infrastructure_failure')
                data=self.shared[:data_size]
            else:data=self._read_exact(data_size,deadline)
        else:
            while b'\n' not in self.pending:self._read_more(deadline)
            end=self.pending.index(b'\n');metadata=bytes(self.pending[:end]);del self.pending[:end+1]
            data=None
        self.pipe_wait_seconds+=time.perf_counter()-waiting
        decoding=time.perf_counter()
        try:response=json.loads(metadata)
        except (ValueError,UnicodeError) as exc:
            raise WorkerError("Malformed simulator response", "infrastructure_failure") from exc
        if not response["ok"]:
            raise WorkerError(response["error"],response["category"])
        self.metrics=response.get('worker_metrics',{})
        result=response['result']
        if isinstance(result,list):
            from .efficient import decode_observation, PackedObservation, LazyPackedObservation, WIDTH
            import numpy as np
            count=0
            for game in result:
                if game and 'observations' in game:
                    observations=[]
                    for obs in game['observations']:
                        if 'packed_index' in obs:
                            index=obs['packed_index']
                            if data is None or index!=count or (index+1)*WIDTH*8>len(data):
                                raise WorkerError("Invalid binary observation offset", "infrastructure_failure")
                            observations.append((LazyPackedObservation if self.lazy else PackedObservation)(np.frombuffer(data,dtype='<f8',count=WIDTH,offset=index*WIDTH*8)))
                            count+=1
                        else:observations.append(decode_observation(obs,lazy=self.lazy))
                    game['observations']=observations
            if data is not None and count*WIDTH*8!=len(data):
                raise WorkerError("Unexpected binary observation payload", "infrastructure_failure")
        self.decode_seconds+=time.perf_counter()-decoding
        return result
    def call(self, message):
        self.send(message)
        return self.receive()
    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        if self.shared is not None:self.shared.close();self.shared=None
        if self.shared_file is not None:self.shared_file.close();self.shared_file=None
        if self.shared_dir is not None:self.shared_dir.cleanup();self.shared_dir=None

class Vector:
    def __init__(self, workers=1, per_worker=4, batched=True, optimized=True, packed=False, binary=False, token_cache=False, mon_cache=False, fast_callbacks=False, shared=False, lazy=False, timeout=60, team_size=6):
        if workers < 1 or per_worker < 1:
            raise ValueError("Positive workers and environments_per_worker required")
        self.workers = []
        try:
            for _ in range(workers):self.workers.append(Worker(binary=binary,timeout=timeout,shared_bytes=per_worker*2*700*8 if shared else 0,lazy=lazy,team_size=team_size))
        except BaseException:
            self.close()
            raise
        self.token_cache=token_cache
        self.mon_cache=mon_cache
        self.fast_callbacks=fast_callbacks
        self.per_worker = per_worker
        self.batched = batched
        self.optimized = optimized
        self.packed = packed
        self.states = None
        self.seconds = 0.0
        self.calls = 0
    @property
    def worker_metrics(self):
        keys={k for w in self.workers for k in w.metrics}
        result={k:sum(w.metrics.get(k,0) for w in self.workers) for k in keys}
        result.update(python_decode_seconds=sum(w.decode_seconds for w in self.workers),
                      pipe_wait_seconds=sum(w.pipe_wait_seconds for w in self.workers))
        return result
    @property
    def size(self):
        return len(self.workers) * self.per_worker
    def call(self, messages):
        start = time.perf_counter()
        # Dispatch all CPU workers before reading any; no thread/async-loop per observation.
        if self.batched:
            for w, m in zip(self.workers, messages, strict=True):
                w.send(m)
            result = [w.receive() for w in self.workers]
        else:
            result = [w.call(m) for w, m in zip(self.workers, messages, strict=True)]
        self.seconds += time.perf_counter() - start
        self.calls += 1
        return [game for group in result for game in group]
    def reset(self, seeds, limit, swapped=None):
        if len(seeds) != self.size:
            raise ValueError("Reset requires exactly vector.size seeds")
        swapped = swapped or [False] * self.size
        n = self.per_worker
        self.states = self.call([{"op": "reset", "token_cache":self.token_cache,"mon_cache":self.mon_cache,"fast_callbacks":self.fast_callbacks,"packed":self.packed,"optimized": self.optimized, "seeds": seeds[i:i+n], "swapped": swapped[i:i+n], "limit": limit}
                          for i in range(0, self.size, n)])
        return self.states
    def step(self, actions):
        n = self.per_worker
        updates = self.call([{"op": "step", "actions": actions[i:i+n]} for i in range(0, self.size, n)])
        self.states = [new if new is not None else old for new, old in zip(updates, self.states, strict=True)]
        return self.states
    def reset_slots(self, updates, limit):
        """Replace only completed episodes; leave every other live simulator intact."""
        if len(updates)!=self.size:raise ValueError('Slot updates must match vector size')
        messages=[];n=self.per_worker
        for start in range(0,self.size,n):
            group=updates[start:start+n]
            messages.append({'op':'reset_slots','token_cache':self.token_cache,'mon_cache':self.mon_cache,'fast_callbacks':self.fast_callbacks,
                'updates':[dict(seed=u[0],swapped=u[1],limit=limit) if u is not None else None for u in group]})
        changes=self.call(messages)
        self.states=[new if new is not None else old for new,old in zip(changes,self.states,strict=True)]
        return self.states
    def hashes(self):
        return self.call([{"op": "hashes"} for _ in self.workers])
    def trace(self):
        return self.call([{"op": "trace"} for _ in self.workers])
    def close(self):
        for w in self.workers:
            w.close()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()


def vector_for(cfg):
    """Shared production construction: transport/encoder mode is checkpoint-bound."""
    h=cfg['hardware'];mode=cfg.get('optimization','baseline')
    return Vector(h['workers'],h['environments_per_worker'],packed=mode!='baseline',
                  binary=mode in ('stream','shared'),shared=mode=='shared',token_cache=mode=='stream',mon_cache=cfg.get('mon_cache',False),fast_callbacks=cfg.get('fast_callbacks',False),
                  timeout=cfg.get('worker_timeout_seconds',60),lazy=cfg.get('lazy_observations',False))


def evaluation_vector_for(cfg,games):
    """Avoid padding a small practice panel to a thousand training environments."""
    import math
    h=dict(cfg['hardware']);h['workers']=min(h['workers'],max(1,math.ceil(games/16)))
    h['environments_per_worker']=max(1,math.ceil(games/h['workers']))
    return vector_for(dict(cfg,hardware=h))
