"""Accelerator resident batched rollout + clipped PPO, no Python per-game hot loop."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from .environment import (encode, action_mask, decode_action, observations, transition,
                          reset_done, potential, scripted_logits, BOARD_SHAPE, ACTION_COUNT)


class Carry(NamedTuple):
    states: object
    key: jax.Array
    seats: jax.Array
    opponents: jax.Array
    history_ids: jax.Array
    history_params: object
    episodes: jax.Array


def logprob(logits, ids):
    return jnp.take_along_axis(jax.nn.log_softmax(logits), ids[..., None], -1)[..., 0]


def sample(key, logits, method='gumbel'):
    if method == 'gumbel':
        return jax.random.categorical(key, logits).astype(jnp.int32)
    if method != 'inverse_cdf':
        raise ValueError('sampler must be gumbel or inverse_cdf')
    # A single uniform per decision replaces one Gumbel per joint action.
    # The categorical distribution is unchanged in exact arithmetic, but this
    # explicitly changes RNG-to-action mapping and floating accumulation order.
    weights = jnp.exp(logits-jnp.max(logits, axis=-1, keepdims=True))
    cumulative = jnp.cumsum(weights, axis=-1)
    total = cumulative[..., -1]
    threshold = jax.random.uniform(key, total.shape)*total
    # Multiplication can round up to total. Keep the threshold strictly below it
    # so trailing zero-weight (illegal) actions can never be selected.
    threshold = jnp.minimum(threshold, jnp.nextafter(total, -jnp.inf))
    # Parallel floating prefix sums need not be perfectly monotone across long
    # zero-mass plateaus. Select an explicitly positive-mass crossing rather than
    # counting prefixes, which could otherwise land on a masked plateau entry.
    crosses = (weights > 0) & (cumulative > threshold[..., None])
    chosen = jnp.argmax(crosses, axis=-1).astype(jnp.int32)
    return jnp.where(crosses.any(-1), chosen, jnp.argmax(weights, axis=-1).astype(jnp.int32))


def create_train_state(model, key, cfg):
    p = model.init(key, jnp.zeros((1, *BOARD_SHAPE, 14)), jnp.ones((1, ACTION_COUNT), bool))["params"]
    tx = optax.chain(optax.clip_by_global_norm(cfg["max_grad_norm"]), optax.adam(cfg["learning_rate"], eps=1e-5))
    # Keep checkpoint restores and fresh runs on the same non-weak scalar type,
    # avoiding a full CUDA recompile merely because msgpack restored int32.
    return TrainState.create(apply_fn=model.apply, params=p, tx=tx).replace(step=jnp.int32(0))


def initial_carry(pool, key, cfg, bank):
    key, k1, k2, k3 = jax.random.split(key, 4)
    b = cfg["num_envs"]
    states = jax.tree.map(lambda x: x[jax.random.randint(k1, (b,), 0, x.shape[0])], pool)
    # Draw independently per environment, not one shared opponent for the batch.
    kinds = sample(k2, jnp.broadcast_to(jnp.log(jnp.asarray(cfg["opponent_mix"])), (b, 4)))
    hist = jax.random.randint(k3, (b,), 0, cfg["history_size"])
    selected = jax.tree.map(lambda p: p[hist], bank)
    return Carry(states, key, jnp.arange(b) % 2, kinds, hist, selected, jnp.zeros(b, jnp.int32))


def make_collect(model, cfg):
    from .packed import rollout_inputs
    b, t = cfg["num_envs"], cfg["rollout_steps"]
    gamma, shaping = cfg["gamma"], cfg["shaping_scale"]
    rows = jnp.arange(b)
    mix_logits = jnp.broadcast_to(jnp.log(jnp.asarray(cfg["opponent_mix"])), (b, 4))
    draw = lambda key, logits: sample(key, logits, cfg.get('sampler', 'gumbel'))

    @jax.jit
    def collect(params, bank, carry, pool):
        def tick(c, _):
            key, ka, kh, ks, kr, ko, ki = jax.random.split(c.key, 7)
            obs = observations(c.states)
            x, masks = encode(obs), action_mask(obs)
            logits, values = model.apply({"params": params}, x.reshape((-1,)+x.shape[-3:]), masks.reshape((-1, masks.shape[-1])))
            logits, values = logits.reshape((b, 2, masks.shape[-1])), values.reshape((b, 2))
            ids = draw(ka, logits)
            opp_seats = 1-c.seats
            opp_x, opp_mask = x[rows, opp_seats], masks[rows, opp_seats]
            if cfg["opponent_mix"][1] > 0:
                # Cache immutable per-episode weights: ring-buffer slots may be
                # replaced between updates while this game is still in progress.
                hlogits, _ = jax.vmap(lambda p, a, m: model.apply({"params": p}, a[None], m[None]))(c.history_params, opp_x, opp_mask)
                historical = draw(kh, hlogits[:, 0])
            else:
                historical = ids[rows, opp_seats]
            opp_obs = jax.tree.map(lambda a: a[rows, opp_seats], obs)
            scripts = draw(ks, scripted_logits(opp_obs, jnp.maximum(c.opponents-2, 0)))
            opp_ids = jnp.where(c.opponents == 1, historical,
                                jnp.where(c.opponents >= 2, scripts, ids[rows, opp_seats]))
            ids = ids.at[rows, opp_seats].set(opp_ids)
            boundary = transition(c.states, decode_action(ids))
            final_obs = observations(boundary.final_state)
            final_x = encode(final_obs)[rows, c.seats]
            final_mask = action_mask(final_obs)[rows, c.seats]
            _, next_values = model.apply({"params": params}, final_x, final_mask)
            reward = boundary.reward[rows, c.seats]
            phi0, phi1 = potential(obs)[rows, c.seats], potential(final_obs)[rows, c.seats]
            reward += shaping * (gamma * phi1 * (~boundary.terminated) - phi0)
            done = boundary.terminated | boundary.truncated
            new_states = reset_done(boundary.final_state, done, pool, jax.random.randint(kr, (b,), 0, pool.time.shape[0]))
            kinds = sample(ko, mix_logits)
            hist_ids = jax.random.randint(ki, (b,), 0, cfg["history_size"])
            hist_params = jax.tree.map(lambda bank_p, old: jnp.where(done.reshape((b,)+(1,)*(old.ndim-1)),
                                                                    bank_p[hist_ids], old), bank, c.history_params)
            new_c = Carry(new_states, key, jnp.where(done, 1-c.seats, c.seats),
                          jnp.where(done, kinds, c.opponents), jnp.where(done, hist_ids, c.history_ids),
                          hist_params,
                          c.episodes + done)
            chosen = ids[rows, c.seats]
            # All diagnostics returned without host synchronization in the scan.
            out = dict(action=chosen,
                       old_logp=logprob(logits[rows, c.seats], chosen), value=values[rows, c.seats],
                       next_value=next_values, reward=reward, terminated=boundary.terminated, truncated=boundary.truncated,
                       wins=(boundary.reward[rows, c.seats] > 0), losses=(boundary.reward[rows, c.seats] < 0),
                       invalid=~jnp.take_along_axis(masks, ids[..., None], -1)[..., 0],
                       own_land=final_obs.owned_land_count[rows, c.seats])
            out.update(rollout_inputs(obs, x, masks, rows, c.seats, cfg.get('packed_rollout', False)))
            return new_c, out
        return jax.lax.scan(tick, carry, None, length=t)
    return collect


def advantages(data, gamma, lam):
    delta = data["reward"] + gamma * (~data["terminated"]) * data["next_value"] - data["value"]
    done = data["terminated"] | data["truncated"]
    def tick(acc, pair):
        d, end = pair
        acc = d + gamma*lam*(~end)*acc
        return acc, acc
    _, adv = jax.lax.scan(tick, jnp.zeros_like(delta[0]), (delta, done), reverse=True)
    return adv, adv + data["value"]


def make_update(model, cfg):
    from .packed import PACKED_KEYS, unpack_observation
    if cfg.get('learner_convolution'):
        from .model import Policy
        model = Policy(model.width, convolution=cfg['learner_convolution'], compute_dtype=model.compute_dtype)
    n = cfg["num_envs"]*cfg["rollout_steps"]
    minibatches = cfg["minibatches"]
    if n % minibatches:
        raise ValueError("rollout sample count must divide minibatches")

    @jax.jit
    def update(state, data, key):
        adv, ret = advantages(data, cfg["gamma"], cfg["gae_lambda"])
        adv = (adv-adv.mean())/(adv.std()+1e-8)
        inputs = PACKED_KEYS if cfg.get('packed_rollout', False) else ('x', 'mask')
        batch = {k: data[k].reshape((n,)+data[k].shape[2:]) for k in (*inputs, "action", "old_logp", "value")}
        batch.update(adv=adv.reshape(n), ret=ret.reshape(n))

        def loss(params, mb):
            x, mask = unpack_observation(mb) if cfg.get('packed_rollout', False) else (mb['x'], mb['mask'])
            logits, value = model.apply({"params": params}, x, mask)
            lp = logprob(logits, mb["action"])
            ratio = jnp.exp(lp-mb["old_logp"])
            pg = -jnp.mean(jnp.minimum(ratio*mb["adv"], jnp.clip(ratio, 1-cfg["clip_eps"], 1+cfg["clip_eps"])*mb["adv"]))
            clipped_v = mb["value"] + jnp.clip(value-mb["value"], -cfg["clip_eps"], cfg["clip_eps"])
            vl = .5*jnp.mean(jnp.maximum((value-mb["ret"])**2, (clipped_v-mb["ret"])**2))
            probs, lps = jax.nn.softmax(logits), jax.nn.log_softmax(logits)
            entropy = -jnp.mean(jnp.sum(probs*lps, -1))
            kl = jnp.mean((ratio-1)-(lp-mb["old_logp"]))
            total = pg + cfg["value_coef"]*vl - cfg["entropy_coef"]*entropy
            return total, jnp.array([total, pg, vl, entropy, kl])

        def epoch(carry, _):
            st, k = carry
            k, pk = jax.random.split(k)
            perm = jax.random.permutation(pk, n).reshape((minibatches, n//minibatches))
            def minibatch(st, indices):
                mb = jax.tree.map(lambda x: x[indices], batch)
                (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(st.params, mb)
                norm = optax.global_norm(grads)
                st = st.apply_gradients(grads=grads)
                return st, jnp.concatenate([metrics, norm[None]])
            st, metrics = jax.lax.scan(minibatch, st, perm)
            return (st, k), metrics.mean(0)
        (state, _), metrics = jax.lax.scan(epoch, (state, key), None, length=cfg["epochs"])
        return state, metrics.mean(0)
    return update
