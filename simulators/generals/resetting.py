"""Sparse exact reset/cache refresh with a dense fallback for synchronized caps."""
import jax
import jax.numpy as jnp
from .environment import reset_done


def opponent_weights(params, history, kinds):
    b = kinds.shape[0]
    return jax.tree.map(lambda current, old: jnp.where(
        (kinds == 1).reshape((b,)+(1,)*current.ndim), old, current[None]), params, history)


def refresh(final, done, pool, map_ids, bank, history_ids, old_history,
            params, kinds, old_opponent=None, capacity=0):
    """Every random draw/index is supplied by the existing deterministic stream.

    Ordinary sparse refresh gathers K policy rows, rather than B policy rows,
    and permits XLA to scatter into loop-carried buffers. No reset is dropped:
    cohorts larger than K use the original dense rule. Unique OOB padding makes
    the scatter contract valid even when fewer than K episodes end.
    """
    b = done.shape[0]

    def dense(_):
        states = reset_done(final, done, pool, map_ids)
        hp = jax.tree.map(lambda p, old: jnp.where(
            done.reshape((b,)+(1,)*(old.ndim-1)), p[history_ids], old), bank, old_history)
        op = None
        if old_opponent is not None:
            proposed = opponent_weights(params, hp, kinds)
            op = jax.tree.map(lambda new, old: jnp.where(
                done.reshape((b,)+(1,)*(old.ndim-1)), new, old), proposed, old_opponent)
        return states, hp, op

    if capacity <= 0:
        return dense(None)
    k = min(capacity, b)

    def sparse(_):
        selected = jnp.nonzero(done, size=k, fill_value=b)[0]
        # Valid indices are sorted/unique. Every padded OOB index is also unique;
        # clip only reads, and drop writes beyond the lane count.
        indices = jnp.where(selected < b, selected, b+jnp.arange(k))
        reads = jnp.minimum(selected, b-1)
        states = jax.tree.map(lambda p, old: old.at[indices].set(
            p[map_ids[reads]], mode='drop', unique_indices=True, indices_are_sorted=True), pool, final)
        new_history = jax.tree.map(lambda p: p[history_ids[reads]], bank)
        hp = jax.tree.map(lambda new, old: old.at[indices].set(
            new, mode='drop', unique_indices=True, indices_are_sorted=True), new_history, old_history)
        op = None
        if old_opponent is not None:
            proposed = opponent_weights(params, new_history, kinds[reads])
            op = jax.tree.map(lambda new, old: old.at[indices].set(
                new, mode='drop', unique_indices=True, indices_are_sorted=True), proposed, old_opponent)
        return states, hp, op

    return jax.lax.cond(done.any(), lambda _: jax.lax.cond(done.sum() <= k, sparse, dense, None),
                        lambda _: (final, old_history, old_opponent), None)
