"""Batched environment: many eight-player games stepped together.

Each game is a full lobby. You control the seats in `controlled` (e.g. `(0,)` for one
learner against seven scripted bots, or `range(8)` for self-play); scripted bots play the
others. A *slot* is one controlled seat of one game: slot `i` is seat `controlled[i % k]` of
game `i // k`, with `k = len(controlled)`.

    env = BatchEnv(n_games=64, seed=0, controlled=(0,), threads=8)
    obs = env.observe()
    while training:
        actions = policy(obs["features"], obs["cards"], obs["action_mask"])  # int per slot
        rewards, dones, info = env.step(actions)
        obs = env.observe()

`observe()["ready"]` says which slots must act now; actions of the other slots are ignored.
Several seats of one game can be ready at once (the recruit phase is simultaneous). The
reward is the placement reward (1st = 1, last = -1) when a seat's episode ends, plus
`illegal_penalty` for a rejected action. A game whose controlled seats have all finished
starts again with a new seed. Results do not depend on `threads`.
"""

import json

import numpy as np

from taverngym._core import N_ACTIONS, OBS_CARDS, OBS_FLOATS
from taverngym._core import BatchEnv as _BatchEnv
from taverngym._core import Game


class BatchEnv:
    def __init__(self, n_games, seed=0, controlled=(0,), config=None, threads=1):
        config_json = json.dumps(config) if config is not None else None
        self._env = _BatchEnv(int(n_games), int(seed), [int(p) for p in controlled],
                              config_json, int(threads))

    @property
    def n_games(self):
        return self._env.n_games

    @property
    def n_slots(self):
        return self._env.n_slots

    @property
    def controlled(self):
        return tuple(self._env.controlled)

    @property
    def illegal_penalty(self):
        return self._env.illegal_penalty

    @illegal_penalty.setter
    def illegal_penalty(self, value):
        self._env.illegal_penalty = float(value)

    def slot(self, i):
        """(game index, seat) of slot `i`."""
        k = len(self.controlled)
        return i // k, self.controlled[i % k]

    def observe(self, copy=True, active_games=None):
        """With copy=False, returns read-only views of immutable Python-owned bytes;
        they remain valid across steps/observations. Per slot: `features` float32 [n_slots, OBS_FLOATS], `cards` int32
        [n_slots, OBS_CARDS] (card embedding indices, 0 = empty), `action_mask` bool
        [n_slots, N_ACTIONS] and `ready` bool [n_slots]."""
        f, c, m, r = self._env.observe(self._active(active_games))
        own = lambda a: a.copy() if copy else a
        n = self.n_slots
        return {
            "features": own(np.frombuffer(f, dtype="<f4").reshape(n, OBS_FLOATS)),
            "cards": own(np.frombuffer(c, dtype="<i4").reshape(n, OBS_CARDS)),
            "action_mask": own(np.frombuffer(m, dtype=np.bool_).reshape(n, N_ACTIONS)),
            "ready": own(np.frombuffer(r, dtype=np.bool_)),
        }

    def step(self, actions, active_games=None):
        """Apply one action per slot. Returns (rewards float32 [n_slots], dones bool
        [n_slots], info) with `info["placement"]` (uint8, the final placement where done;
        0 if a simulator guard stopped the game), `info["rejected"]` (bool),
        `info["applied"]` and `info["deferred"]`. A response invalidated by an earlier
        seat's action is deferred without a penalty; observe again before responding.
        Only applied responses form learning transitions."""
        a = np.ascontiguousarray(np.asarray(actions).reshape(-1), dtype="<i8")
        if a.shape[0] != self.n_slots:
            raise ValueError(f"{a.shape[0]} actions for {self.n_slots} slots")
        rw, dn, pl, rj, stop, applied, deferred = self._env.step_admission(a.tobytes(), self._active(active_games))
        return (
            np.frombuffer(rw, dtype="<f4").copy(),
            np.frombuffer(dn, dtype=np.uint8).astype(bool),
            {
                "placement": np.frombuffer(pl, dtype=np.uint8).copy(),
                "rejected": np.frombuffer(rj, dtype=np.uint8).astype(bool),
                "applied": np.frombuffer(applied, dtype=np.uint8).astype(bool),
                "deferred": np.frombuffer(deferred, dtype=np.uint8).astype(bool),
                "stop_reason": np.frombuffer(stop, dtype=np.uint8).copy(),
                "terminated": np.frombuffer(stop, dtype=np.uint8) == 1,
                "truncated": np.frombuffer(stop, dtype=np.uint8) >= 2,
            },
        )

    def observe_ready(self, copy=False, active_games=None):
        """Compact flat training observations and their original full-slot indices.

        Only ready seats are encoded. Immutable Python-owned bytes retain their
        lifetime across subsequent observe/step calls, as in observe(copy=False).
        """
        f,c,m,indices=self._env.observe_ready(self._active(active_games))
        own=lambda a:a.copy() if copy else a
        slots=np.frombuffer(indices,dtype='<i8')
        n=len(slots)
        return {'features':own(np.frombuffer(f,dtype='<f4').reshape(n,OBS_FLOATS)),
                'cards':own(np.frombuffer(c,dtype='<i4').reshape(n,OBS_CARDS)),
                'action_mask':own(np.frombuffer(m,dtype=np.bool_).reshape(n,N_ACTIONS)),
                'ready':np.ones(n,dtype=np.bool_), 'slots':own(slots)}

    def _active(self, active_games):
        if active_games is None:
            return None
        active = np.ascontiguousarray(active_games, dtype=np.bool_).reshape(-1)
        if len(active) != self.n_games:
            raise ValueError("active game mask length mismatch")
        return active.tobytes()

    def save(self, i):
        """Game `i` and its episode state as JSON; `load(i, s)` restores it exactly."""
        return self._env.save(int(i))

    def warm_start(self, target_turn):
        """Run seat-ordered scripted bots to a target turn on the native workers.

        For constructing occupied continuation profiles, not a learning shortcut.
        Full eight-player rules, RNG, combat and fault guards remain active.
        """
        self._env.warm_start(int(target_turn))

    def load(self, i, state):
        self._env.load(int(i), state)

    def episode_index(self, i):
        return self._env.episode_index(int(i))

    def game(self, i) -> Game:
        """An independent copy of game `i` (inspect it with `view`, `state_json`...)."""
        return self._env.game(int(i))
