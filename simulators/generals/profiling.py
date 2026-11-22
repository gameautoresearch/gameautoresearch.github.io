"""Synchronised random-action vs complete-learner timing, compile separated."""
import time
import jax
import jax.numpy as jnp
import numpy as np
from .environment import observations, action_mask, decode_action, transition, reset_done
from .learning import sample


def timed(fn, *args):
    t = time.monotonic()
    r = fn(*args)
    jax.block_until_ready(r)
    return r, time.monotonic()-t


def profile(model, train, bank, carry, pool, collect, update, cfg):
    b, t = cfg["num_envs"], cfg["rollout_steps"]
    def tick(c, _):
        states, key = c
        key, ka, kr = jax.random.split(key, 3)
        masks = action_mask(observations(states))
        ids = sample(ka, jnp.where(masks, 0., -1e9))
        boundary = transition(states, decode_action(ids))
        done = boundary.terminated | boundary.truncated
        states = reset_done(boundary.final_state, done, pool, jax.random.randint(kr, (b,), 0, pool.time.shape[0]))
        return (states, key), jnp.sum(~jnp.take_along_axis(masks, ids[..., None], -1)[..., 0])
    # Pool is an explicit argument to avoid embedding all maps into the executable.
    # tick closes over pool only in this local comparison, not in the learner.
    sim_device = carry.states.time.device
    fused = jax.jit(lambda c: jax.lax.scan(tick, c, None, length=t), device=sim_device)
    single = jax.jit(tick, device=sim_device)
    start = (carry.states, jax.device_put(carry.key, sim_device))
    fused_result, fused_compile = timed(fused, start)
    _, tick_compile = timed(single, start, None)
    fused_trials, host_trials = [], []
    for _ in range(3):
        fused_result, seconds = timed(fused, start)
        fused_trials.append(seconds)
        t0 = time.monotonic()
        c = start
        invalid = jax.device_put(jnp.int32(0), sim_device)
        for _ in range(t):
            c, inv = single(c, None)
            invalid += inv
        jax.block_until_ready((c, invalid))
        host_trials.append(time.monotonic()-t0)
        assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(jax.tree.leaves(c), jax.tree.leaves(fused_result[0])))
        assert int(invalid) == int(fused_result[1].sum()) == 0
    (c1, batch), collect_compile = timed(collect, train.params, bank, carry, pool)
    (_, _), update_compile = timed(update, train, batch, jax.random.PRNGKey(77))
    fused_iteration = None
    iteration_compile = None
    if cfg.get('execution') == 'optimized':
        from .efficiency import make_iteration
        if cfg.get('simulation_device') == 'cpu':
            from .split_device import make_iteration_split
            fused_iteration = make_iteration_split(model, cfg)
        else:
            fused_iteration = make_iteration(model, cfg, 'optimized')
        _, iteration_compile = timed(fused_iteration, train, bank, carry, pool)
    trials = []
    for i in range(3):
        t0 = time.monotonic()
        if fused_iteration is not None:
            (trained, c1, metrics), ft = timed(fused_iteration, train, bank, carry, pool)
            ct = ut = None
            invalid = int(metrics[9])
        else:
            (c1, batch), ct = timed(collect, train.params, bank, carry, pool)
            (trained, metrics), ut = timed(update, train, batch, jax.random.PRNGKey(i+78))
            invalid = int(batch["invalid"].sum())
        trials.append(dict(rollout_seconds=ct, ppo_seconds=ut, full_seconds=time.monotonic()-t0,
                           loss=float(metrics[0]), invalid_actions=invalid))
    full = np.median([r["full_seconds"] for r in trials])
    return dict(random_action=dict(compile_scan_seconds=fused_compile, compile_single_tick_seconds=tick_compile,
                    fused_seconds=fused_trials, host_loop_seconds=host_trials,
                    fused_environment_ticks_per_second=b*t/float(np.median(fused_trials)),
                    fusion_speedup=float(np.median(host_trials)/np.median(fused_trials)),
                    exact_state_and_rng_equivalence=True, invalid_actions=0,
                    warning="Random legal joint actions; no neural inference or learning."),
                learner=dict(execution=cfg.get('execution','baseline'), simulation_device=str(sim_device), compile_fused_iteration_seconds=iteration_compile,
                    compile_rollout_seconds=collect_compile, compile_update_seconds=update_compile, trials=trials,
                    learner_samples_per_second=b*t/float(full), environment_ticks_per_second=b*t/float(full),
                    actual_game_player_decisions_per_second=2*b*t/float(full),
                    includes="visible encoding, masks, current and historical neural inference, scripts, JAX simulation, resets, GAE, PPO gradients, device sync",
                    excludes_from_warm_rate="imports, initialization, first compilation, checkpoint IO, development evaluation; all remain in allocated worker budget"),
                optimization="Fused lax.scan avoids per-tick Python dispatch; retained upstream JAX simulation. Joint dense codec avoids argwhere sorting and preserves source/direction/split dependence.")
