"""Paired map/seat evaluation: capture wins, losses, cap draws, never land-adjudicated."""
import time
from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
from generals.agents import HunterAgent
from .environment import (build_pool, observations, encode, action_mask, decode_action,
                          encode_action, transition, scripted_logits, RULESET)
from .learning import sample


@lru_cache(maxsize=16)
def make_evaluator(model, opponents, stochastic=True, sampler='gumbel'):
    if not opponents or any(o not in ("random", "expansion", "hunter", "frozen") for o in opponents):
        raise ValueError("opponents must be random, expansion, hunter or frozen")
    hunter = HunterAgent()

    @jax.jit
    def run(params, frozen, states, seats, kinds, key):
        b = seats.shape[0]
        rows = jnp.arange(b)
        initial = dict(active=jnp.ones(b, bool), wins=jnp.zeros(b, jnp.int32), losses=jnp.zeros(b, jnp.int32),
                       draws=jnp.zeros(b, jnp.int32), invalid=jnp.zeros(b, jnp.int32), length=jnp.zeros(b, jnp.int32),
                       land=jnp.zeros(b, jnp.float32), army=jnp.zeros(b, jnp.float32))
        def tick(carry, _):
            states, key, result = carry
            key, k1, k2 = jax.random.split(key, 3)
            obs = observations(states)
            x, masks = encode(obs), action_mask(obs)
            logits, _ = model.apply({"params": params}, x[rows, seats], masks[rows, seats])
            ids = sample(k1, logits, sampler) if stochastic else jnp.argmax(logits, -1).astype(jnp.int32)
            opp_obs = jax.tree.map(lambda a: a[rows, 1-seats], obs)
            opp_ids = jnp.zeros(b, jnp.int32)
            for i, opponent in enumerate(opponents):
                if opponent in ("random", "expansion"):
                    choice = sample(k2, scripted_logits(opp_obs, int(opponent == "expansion")), sampler)
                elif opponent == "hunter":
                    # Preserve upstream algorithm/provenance; mask and flag unexpected invalid actions.
                    a = jax.vmap(hunter.act)(opp_obs, jax.random.split(k2, b))
                    choice = encode_action(a)
                else:
                    fl, _ = model.apply({"params": frozen}, x[rows, 1-seats], masks[rows, 1-seats])
                    choice = sample(k2, fl, sampler) if stochastic else jnp.argmax(fl, -1).astype(jnp.int32)
                opp_ids = jnp.where(kinds == i, choice, opp_ids)
            joint_ids = jnp.zeros((b, 2), jnp.int32).at[rows, seats].set(ids).at[rows, 1-seats].set(opp_ids)
            invalid = ~jnp.take_along_axis(masks, joint_ids[..., None], -1)[..., 0]
            boundary = transition(states, decode_action(joint_ids))
            active = result["active"]
            done = boundary.terminated | boundary.truncated
            reward = boundary.reward[rows, seats]
            final_obs = observations(boundary.final_state)
            result = dict(active=active & ~done,
                          wins=result["wins"] + (active & (reward > 0)),
                          losses=result["losses"] + (active & (reward < 0)),
                          draws=result["draws"] + (active & boundary.truncated),
                          invalid=result["invalid"] + (active * jnp.sum(invalid, -1)),
                          length=result["length"] + active,
                          land=jnp.where(active, final_obs.owned_land_count[rows, seats], result["land"]),
                          army=jnp.where(active, final_obs.owned_army_count[rows, seats], result["army"]))
            states = jax.tree.map(lambda n, o: jnp.where(active.reshape((b,)+(1,)*(n.ndim-1)), n, o), boundary.final_state, states)
            return (states, key, result), (joint_ids, active)
        (final, _, result), trace = jax.lax.scan(tick, (states, key, initial), None, length=RULESET["truncation"])
        return result, trace, final
    return run


def evaluate(model, params, seeds, opponents=("random", "expansion"), policy_seed=911,
             frozen=None, stochastic=True, replay_path=None, sampler='gumbel'):
    start = time.monotonic()
    pool = build_pool(seeds)
    # Pair each seed with each opponent and both actual engine seats. No relabel shortcut.
    schedule = [(seed, i, seat) for seed in seeds for i in range(len(opponents)) for seat in (0, 1)]
    indices = jnp.repeat(jnp.arange(len(seeds)), len(opponents)*2)
    states = jax.tree.map(lambda x: x[indices], pool)
    seats = jnp.asarray([r[2] for r in schedule])
    kinds = jnp.asarray([r[1] for r in schedule])
    result, trace, final = make_evaluator(model, tuple(opponents), stochastic, sampler)(params, params if frozen is None else frozen,
                                                                      states, seats, kinds, jax.random.PRNGKey(policy_seed))
    result = jax.device_get(result)
    records = [dict(seed=int(s), opponent=opponents[k], seat=int(seat), **{n: np.asarray(v[i]).item() for n, v in result.items()})
               for i, (s, k, seat) in enumerate(schedule)]
    groups = {}
    bootstrap = np.random.default_rng(19991)
    for opponent in opponents:
        for seat in (0, 1, "both"):
            rows = [r for r in records if r["opponent"] == opponent and (seat == "both" or r["seat"] == seat)]
            n = len(rows)
            wins, losses, draws = (sum(r[k] for r in rows) for k in ("wins", "losses", "draws"))
            groups[f"{opponent}/seat-{seat}"] = dict(games=n, wins=wins, losses=losses, cap_draws=draws,
                score=(wins+.5*draws)/n, invalid=sum(r["invalid"] for r in rows),
                mean_land=float(np.mean([r["land"] for r in rows])), mean_length=float(np.mean([r["length"] for r in rows])))
            seed_scores = np.asarray([np.mean([r['wins']+.5*r['draws'] for r in rows if r['seed'] == s]) for s in seeds])
            boot = seed_scores[bootstrap.integers(0,len(seeds),size=(2000,len(seeds)))].mean(-1)
            groups[f"{opponent}/seat-{seat}"].update(seed_count=len(seeds),
                score_ci95_seed_bootstrap=np.quantile(boot,[.025,.975]).tolist())
    replay = None
    if replay_path is not None:
        from pathlib import Path
        from .runtime import digest, sources, write_json
        replay_path = Path(replay_path)
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(replay_path, action_ids=np.asarray(trace[0]), active=np.asarray(trace[1]),
                            seeds=np.asarray([r[0] for r in schedule], np.uint32), seats=np.asarray(seats),
                            kinds=np.asarray(kinds), **{f'final_{k}': np.asarray(v) for k,v in final._asdict().items()})
        replay = dict(path=str(replay_path.resolve()), sha256=digest(replay_path.read_bytes()),
                      source=sources(), opponents=opponents, schedule=schedule)
        write_json(str(replay_path)+'.json', replay)
    return dict(elapsed_seconds=time.monotonic()-start, stochastic=stochastic, policy_seed=policy_seed, sampler=sampler, replay=replay,
                groups=groups, records=records, warning="95% percentile bootstrap resamples seed clusters (both seats together), 2000 draws. One policy seed and one training run do not establish training-seed robustness. Cap draws are not captures.")
