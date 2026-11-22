"""Fresh public viewing games against random/Hunter; not a scored panel."""
from pathlib import Path
import base64
import json
from .bootstrap import activate
activate()
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from generals.core.game import get_observation, step
from generals.core.action import compute_valid_move_mask
from generals.agents import HunterAgent
from networks.common import obs_to_array, augment_obs, init_obs_state, random_action
from networks.transformer import greedy_action_transformer


def packed(a):
    return base64.b64encode(np.packbits(np.asarray(a).reshape(-1), bitorder='little').tobytes()).decode()


def export_recordings(cfg, env, net, output, model_sha, stage):
    output = Path(output); output.mkdir()
    count = 4; rr = jnp.arange(count); seats = jnp.arange(count) % 2
    seeds = np.array([2026110911, 2026110912, 2026110913, 2026110914], np.uint32)
    initial = jax.jit(jax.vmap(env.init_state))(jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds)))
    memory = jax.tree.map(lambda x: jnp.broadcast_to(x, (count,) + x.shape), init_obs_state(cfg.pad_to))
    hunter = HunterAgent()

    @eqx.filter_jit
    def run(net, states, memory):
        def tick(c, _):
            s, m, key, live = c
            obs = jax.vmap(lambda s, p: get_observation(s, p))(s, seats)
            opp = jax.vmap(lambda s, p: get_observation(s, 1-p))(s, seats)
            mask = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs)
            x, m = jax.vmap(augment_obs)(jax.vmap(obs_to_array)(obs), m)
            temporal = jnp.stack([m.opponent_army_history, m.opponent_land_history], 1)
            neural = jax.vmap(greedy_action_transformer, (None, 0, 0, 0))(net, x, mask, temporal)
            key, ak = jax.random.split(key)
            random = jax.vmap(random_action)(jax.random.split(ak, count), opp)
            scripted = jax.vmap(lambda o: hunter.act(o, jax.random.PRNGKey(0)))(opp)
            others = jnp.where((rr < 2)[:, None], random, scripted)
            actions = jnp.zeros((count, 2, 5), jnp.int32).at[rr, seats].set(neural).at[rr, 1-seats].set(others)
            all_obs = jax.vmap(lambda v: jax.vmap(lambda p: get_observation(v, p))(jnp.arange(2)))(s)
            masks = jax.vmap(jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains)))(all_obs)
            legal = actions[..., 0].astype(bool) | masks[rr[:, None], jnp.arange(2)[None, :],
                jnp.clip(actions[..., 1], 0, cfg.pad_to-1), jnp.clip(actions[..., 2], 0, cfg.pad_to-1), actions[..., 3]]
            ns, info = jax.vmap(lambda s, a: step(s, a, general_trade=env.general_trade))(s, actions)
            natural = info.is_done; capped = (ns.time >= cfg.truncation) & ~natural
            ns = jax.tree.map(lambda n, v: jnp.where(live.reshape((count,) + (1,)*(n.ndim-1)), n, v), ns, s)
            return (ns, m, key, live & ~(natural | capped)), (ns, actions, live, natural, capped, live[:, None] & ~legal)
        return jax.lax.scan(tick, (states, memory, jax.random.PRNGKey(2026110910), jnp.ones(count, bool)),
                            None, length=cfg.truncation)

    (final, _, _, live), (history, actions, active, natural, caps, invalid) = jax.device_get(run(net, initial, memory))
    if live.any() or invalid.any():
        raise AssertionError('Unfinished or illegal viewing game')
    # Independent engine-only resimulation of every joint action and exact final state.
    @jax.jit
    def replay(states, actions, active):
        def tick(s, values):
            a, live = values
            ns, _ = jax.vmap(lambda s, a: step(s, a, general_trade=env.general_trade))(s, a)
            return jax.tree.map(lambda n, v: jnp.where(live.reshape((count,) + (1,)*(n.ndim-1)), n, v), ns, s), None
        return jax.lax.scan(tick, states, (actions, active))[0]
    replayed = jax.device_get(replay(initial, actions, active))
    for x, y in zip(jax.tree.leaves(final), jax.tree.leaves(replayed)):
        np.testing.assert_array_equal(x, y)
    np.savez_compressed(output/'actions.npz', seeds=seeds, actions=actions, active=active,
                        **{'initial_'+k: np.asarray(v) for k, v in initial._asdict().items()},
                        **{'final_'+k: np.asarray(v) for k, v in final._asdict().items()})
    states = jax.tree.map(lambda x, y: np.concatenate([np.asarray(x)[None], y], 0), initial, history)
    @jax.jit
    def views(states):
        o = jax.vmap(jax.vmap(lambda s: jax.vmap(lambda p: get_observation(s, p))(jnp.arange(2))))(states)
        return o.fog_cells, o.structures_in_fog
    fog, shadows = jax.device_get(views(states))
    summaries = []
    for i in range(count):
        end = int(np.flatnonzero(active[:, i] & (natural[:, i] | caps[:, i]))[0]) + 1
        opponent = 'Random' if i < 2 else 'Hunter'
        seat = int(i % 2); names = [opponent, opponent]; names[seat] = 'AverageJoe continuation'
        previous = None; frames = []; size = cfg.max_grid_size
        for t in range(end+1):
            s = jax.tree.map(lambda v: np.asarray(v[t, i]), states)
            own = s.ownership[:, :size, :size]
            owner = np.where(own.any(0), own.argmax(0), -1)
            kind = np.where(s.mountains[:size, :size], 1, np.where(s.generals[:size, :size], 3, np.where(s.castles[:size, :size], 2, 0)))
            board = np.stack([kind, owner, s.armies[:size, :size]], -1).reshape(-1, 3).astype(int)
            changed = np.arange(size*size) if previous is None else np.flatnonzero(np.any(board != previous, -1))
            ids = None
            if t:
                order = actions[t-1, i]
                ids = np.where(order[:, 0].astype(bool), 0,
                    1 + (order[:, 1]*24+order[:, 2])*8 + order[:, 3]*2 + order[:, 4]).tolist()
            frames.append({'tick': t, 'changes': [[int(k), *board[k].tolist()] for k in changed],
                           'actions': ids, 'fog': [packed(fog[t, i, p, :size, :size]) for p in range(2)],
                           'shadows': [packed(shadows[t, i, p, :size, :size]) for p in range(2)],
                           'scores': [{'army': int((s.armies*s.ownership[p]).sum()), 'land': int(s.ownership[p].sum())} for p in range(2)]})
            previous = board
        truncated = bool(caps[end-1, i]); winner = None if truncated else int(final.winner[i])
        identifier = f'averagejoe-stage-{stage}-{opponent.lower()}-{i+1}'
        film = {'schemaVersion': 1, 'gameId': 'generals', 'id': identifier,
                'title': f'AverageJoe vs {opponent} · stage {stage}', 'subtitle': f'{size} × {size} · '+('Cap draw' if truncated else ('AI wins' if winner == seat else opponent+' wins')),
                'width': size, 'height': size, 'stepCap': cfg.truncation, 'players': names,
                'policy': {'name': 'AverageJoe continuation', 'checkpoint': model_sha,
                           'description': 'Actual upstream transformer; fresh public viewing game, outside scored evaluation.'},
                'provenance': {'kind': 'fresh-public-demo', 'demoSeed': int(seeds[i]),
                    'ruleset': 'averagejoe-curriculum-stage-'+str(stage), 'engineCommit': '954927f508aa47899019d66317cc313af3e9cc19',
                    'candidateSeat': seat, 'scoredEvaluation': False, 'generalTrade': env.general_trade,
                    'verification': 'All recorded joint actions legal; independent engine-only resimulation matched exact final state.'},
                'frames': frames, 'outcome': {'winner': winner, 'ticks': end, 'terminated': not truncated,
                    'truncated': truncated, 'terminal': 'step-cap' if truncated else 'general-captured'}}
        (output/(identifier+'.json')).write_text(json.dumps(film, separators=(',', ':'))+'\n')
        summaries.append({k: film[k] for k in ('id', 'gameId', 'title', 'subtitle')} |
                         {'url': f'replays/generals/averagejoe-continuation-20261109/{identifier}.json', 'moves': end})
    (output/'index.json').write_text(json.dumps(summaries, indent=2)+'\n')
    return {'games': count, 'illegal_actions': 0, 'exact_engine_replay': True, 'summaries': summaries}
