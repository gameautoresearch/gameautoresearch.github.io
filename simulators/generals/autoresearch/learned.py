"""A submitted policy must be learned: its moves have to come from its weights.

The check replays the policy's own decisions with its weights re-initialized (`model.init` under host keys). A
network chooses differently once its weights are replaced; a hand-written player dressed as a network, or one with
a small learned correction on top of fixed move scores, keeps choosing the same moves. Only boards with a real
choice count (at least MIN_LEGAL legal actions), and the verdict is given once, after BOARDS of them.

None of this is on the policy's clock (found live 2026-11-19: building the probe took 13 s for a large network,
inside the 15 s the evaluator allows a policy's first answer, so every evaluation of that network was rejected).
The probe is built and compiled before the policy process reports ready (`warm`), it looks at a batch only after
the policy's answer has been sent, and it always runs one fixed batch shape so nothing compiles mid-game.
"""
import numpy as np

MIN_LEGAL = 17      # pass plus at least eight source/direction pairs
BOARDS = 2048       # counted boards before the verdict
LIMIT = 0.25        # largest share of counted boards on which re-initialized weights may choose the same move
KEYS = (101, 202)   # two independent re-initializations; the larger agreement counts
BATCH = 32          # the one batch shape the probe compiles and runs


class NotLearned(ValueError):
    pass


class Probe:
    def __init__(self, model, params):
        import jax
        import jax.numpy as jnp
        from integrations.generals.environment import BOARD_SHAPE, ACTION_COUNT
        shapes = lambda tree: sorted((path, np.shape(leaf)) for path, leaf in _leaves(tree))
        self.fresh = []
        for key in KEYS:
            new = model.init(jax.random.PRNGKey(key), jnp.zeros((1, *BOARD_SHAPE, 14)), jnp.ones((1, ACTION_COUNT), bool))['params']
            if shapes(new) != shapes(params):
                raise NotLearned('The saved weights do not have the shape of the policy\'s own initialization')
            self.fresh.append(new)

        @jax.jit
        def choose(p, x, mask):
            def one(a, m):
                return model.apply({'params': p}, a[None], m[None])[0][0]
            return jnp.argmax(jnp.where(mask, jax.vmap(one)(x, mask), -jnp.inf), axis=-1)
        self.choose = choose
        self.shape = ((BATCH, *BOARD_SHAPE, 14), (BATCH, ACTION_COUNT))
        self.counted = 0
        self.same = np.zeros(len(KEYS), np.int64)
        self.done = False

    def warm(self):
        """Compile the probe's one batch shape now, before the policy is asked anything."""
        x, mask = np.zeros(self.shape[0], np.float32), np.ones(self.shape[1], bool)
        for params in self.fresh:
            np.asarray(self.choose(params, x, mask))
        return self

    def see(self, x, mask, ids):
        """One answered batch: the visible boards, their legal masks and the moves the policy chose."""
        if self.done:
            return
        x, mask, ids = np.asarray(x), np.asarray(mask), np.asarray(ids)
        real = mask.sum(-1) >= MIN_LEGAL
        if not real.any():
            return
        x, mask, ids = x[real], mask[real], ids[real]
        for start in range(0, len(x), BATCH):
            count = len(x[start:start + BATCH])
            pad = np.resize(np.arange(start, start + count), BATCH)  # a short batch repeats its own boards
            for i, params in enumerate(self.fresh):
                chosen = np.asarray(self.choose(params, x[pad], mask[pad]))[:count]
                self.same[i] += int((chosen == ids[start:start + count]).sum())
        self.counted += len(x)
        if self.counted >= BOARDS:
            self.done = True
            share = float(self.same.max()) / self.counted
            if share > LIMIT:
                raise NotLearned(f'Policy is not learned: with its weights re-initialized it still chooses the same move on '
                                 f'{100 * share:.0f}% of {self.counted} boards (at most {100 * LIMIT:.0f}% is accepted). Moves '
                                 'must come from trained weights, not from move scores written into policy.py.')

    def share(self):
        return float(self.same.max()) / self.counted if self.counted else None


def _leaves(tree, prefix=()):
    if isinstance(tree, dict):
        for name in sorted(tree):
            yield from _leaves(tree[name], prefix + (name,))
    else:
        yield prefix, tree
