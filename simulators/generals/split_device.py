"""Compiled CPU simulation feeding GPU inference/PPO, for matched architecture tests.

No Python per-game loop. Each tick dispatches one whole compiled CPU batch, moves
only packed public observations/compact control arrays, and keeps models,
historical weights, sampled training data and optimizer state on the GPU.
"""
from collections import namedtuple
from functools import partial
import jax
import jax.numpy as jnp
from .environment import (SPATIAL, SCALARS, observations, action_mask, transition,
                          decode_action, reset_done, scripted_logits)
from .packed import pack_observation, unpack_observation
from .learning import sample, logprob, make_update
from .efficiency import metrics_summary
from .resetting import refresh, opponent_weights
from .model import historical_policy

VisibleObservation = namedtuple('VisibleObservation', SPATIAL+SCALARS)


def _raw_observation(packed):
    bits = packed['packed_bits']
    values = [packed['packed_armies']]
    values += [((bits >> i) & 1).astype(bool) for i in range(8)]
    values += [packed['packed_scalars'][..., i] for i in range(5)]
    return VisibleObservation(*values)


def make_collect_split(model, cfg):
    if not cfg.get('packed_rollout') or not cfg.get('compact_opponent_inference'):
        raise ValueError('Split-device comparison requires packed compact inference')
    cpu, gpu = jax.devices('cpu')[0], jax.devices('cuda')[0]
    b, t = cfg['num_envs'], cfg['rollout_steps']
    rows = jnp.arange(b)
    history_model = historical_policy(model, cfg)
    mixture = jnp.broadcast_to(jnp.log(jnp.asarray(cfg['opponent_mix'])), (b, 4))
    draw = lambda key, logits: sample(key, logits, cfg.get('sampler', 'gumbel'))
    cached_weights = jax.jit(opponent_weights, device=gpu)

    @partial(jax.jit, device=gpu)
    def stack_samples(samples):
        return jax.tree.map(lambda *xs: jnp.stack(xs), *samples)

    @partial(jax.jit, device=cpu)
    def prepare(states):
        obs = observations(states)
        return pack_observation(obs, action_mask(obs))

    @partial(jax.jit, device=cpu)
    def step(states, capped, ids, map_ids, pool):
        boundary = transition(states, decode_action(ids))
        done = boundary.terminated | boundary.truncated
        capped = jax.tree.map(lambda new, old: jnp.where(
            boundary.truncated.reshape((b,)+(1,)*(new.ndim-1)), new, old), boundary.final_state, capped)
        stats = dict(reward=boundary.reward, land=boundary.final_state.ownership.sum((2, 3)),
                     terminated=boundary.terminated, truncated=boundary.truncated)
        return reset_done(boundary.final_state, done, pool, map_ids), capped, stats

    @partial(jax.jit, device=gpu)
    def infer(params, c, cached, packed):
        key, ka, kh, ks, kr, ko, ki = jax.random.split(c.key, 7)
        x, masks = unpack_observation(packed)
        opp = 1-c.seats
        ll, lv = model.apply({'params': params}, x[rows, c.seats], masks[rows, c.seats])
        hl, _ = jax.vmap(lambda p, a, m: history_model.apply({'params': p}, a[None], m[None]))(
            cached, x[rows, opp], masks[rows, opp])
        logits = jnp.zeros((b, 2, masks.shape[-1])).at[rows, c.seats].set(ll).at[rows, opp].set(hl[:, 0])
        ids = draw(ka, logits)
        hist = draw(kh, hl[:, 0]) if cfg['opponent_mix'][1] > 0 else ids[rows, opp]
        raw = _raw_observation(packed)
        opp_obs = jax.tree.map(lambda a: a[rows, opp], raw)
        scripts = draw(ks, scripted_logits(opp_obs, jnp.maximum(c.opponents-2, 0)))
        chosen_opp = jnp.where(c.opponents == 1, hist, jnp.where(c.opponents >= 2, scripts, ids[rows, opp]))
        ids = ids.at[rows, opp].set(chosen_opp)
        chosen = ids[rows, c.seats]
        data = dict(action=chosen, old_logp=logprob(ll, chosen), value=lv,
                    invalid=~jnp.take_along_axis(masks, ids[..., None], -1)[..., 0], learner_seat=c.seats,
                    **jax.tree.map(lambda a: a[rows, c.seats], packed))
        control = dict(key=key, kinds=sample(ko, mixture),
                       history_ids=jax.random.randint(ki, (b,), 0, cfg['history_size']))
        map_ids = jax.random.randint(kr, (b,), 0, cfg['train_pool_size'])
        return ids, map_ids, data, control

    @partial(jax.jit, device=gpu)
    def after_step(params, bank, c, cached, data, stats, control):
        done = stats['terminated'] | stats['truncated']
        # Reuse the same exact sparse historical-cache refresh as the resident
        # path. Dummy state outputs disappear from the compiled executable.
        _, hp, cached = refresh(jnp.zeros(b, jnp.int32), done, jnp.zeros(1, jnp.int32),
            jnp.zeros(b, jnp.int32), bank, control['history_ids'], c.history_params,
            params, control['kinds'], cached, cfg.get('sparse_reset_capacity', 128))
        reward = stats['reward'][rows, c.seats]
        land = stats['land'][rows, c.seats]
        area = data['packed_armies'].shape[-2]*data['packed_armies'].shape[-1]
        phi0 = data['packed_scalars'][:, 0].astype(jnp.float32)/area
        shaped = reward+cfg['shaping_scale']*(cfg['gamma']*(land.astype(jnp.float32)/area)*(~stats['terminated'])-phi0)
        data.update(reward=shaped, terminated=stats['terminated'], truncated=stats['truncated'],
                    wins=reward > 0, losses=reward < 0, own_land=land)
        c = c._replace(key=control['key'], seats=jnp.where(done, 1-c.seats, c.seats),
                       opponents=jnp.where(done, control['kinds'], c.opponents),
                       history_ids=jnp.where(done, control['history_ids'], c.history_ids),
                       history_params=hp, episodes=c.episodes+done)
        return c, cached, data

    @partial(jax.jit, device=gpu)
    def finish_values(params, data, final_packed, cap_packed, seats):
        tail_x, tail_mask = unpack_observation(final_packed)
        _, tail = model.apply({'params': params}, tail_x[rows, seats], tail_mask[rows, seats])
        cap_seat = jnp.sum(jnp.where(data['truncated'], data['learner_seat'], 0), 0)
        cap_x, cap_mask = unpack_observation(cap_packed)
        _, cap = model.apply({'params': params}, cap_x[rows, cap_seat], cap_mask[rows, cap_seat])
        shifted = jnp.concatenate([data['value'][1:], tail[None]], 0)
        data['next_value'] = jnp.where(data['terminated'], 0., jnp.where(data['truncated'], cap[None], shifted))
        del data['learner_seat']
        return data

    def collect(params, bank, carry, pool):
        states = jax.tree.map(lambda x: jax.device_put(x, cpu), carry.states)
        pool = jax.tree.map(lambda x: jax.device_put(x, cpu), pool)
        capped = states
        c = carry._replace(states=())
        cached = cached_weights(params, c.history_params, c.opponents)
        samples = []
        for _ in range(t):
            packed = jax.tree.map(lambda x: jax.device_put(x, gpu), prepare(states))
            ids, map_ids, data, control = infer(params, c, cached, packed)
            states, capped, stats = step(states, capped, jax.device_put(ids, cpu), jax.device_put(map_ids, cpu), pool)
            stats = jax.tree.map(lambda x: jax.device_put(x, gpu), stats)
            c, cached, data = after_step(params, bank, c, cached, data, stats, control)
            samples.append(data)
        data = stack_samples(samples)
        final_packed = jax.tree.map(lambda x: jax.device_put(x, gpu), prepare(states))
        cap_packed = jax.tree.map(lambda x: jax.device_put(x, gpu), prepare(capped))
        data = finish_values(params, data, final_packed, cap_packed, c.seats)
        return c._replace(states=states), data
    return collect


def make_iteration_split(model, cfg):
    collect = make_collect_split(model, cfg)
    update = make_update(model, cfg)
    gpu = jax.devices('cuda')[0]
    @partial(jax.jit, device=gpu)
    def learn(train, data, key):
        train, metrics = update(train, data, key)
        return train, metrics_summary(data, metrics)
    def iteration(train, bank, c, pool):
        c, data = collect(train.params, bank, c, pool)
        key, uk = jax.random.split(c.key)
        c = c._replace(key=key)
        train, metrics = learn(train, data, uk)
        return train, c, metrics
    return iteration
