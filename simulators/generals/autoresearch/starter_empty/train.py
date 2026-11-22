"""Self-play plumbing with no learning. Each run starts from random weights and, as it is, ends with them.

What is here: the training maps are loaded, `num_envs` games are stepped on the device with one policy
sampling the moves of both seats, finished games restart on another training map, and the run writes
`model.json`, `weights.npz`, `progress.jsonl` and `result.json` and stops when its `--seconds` are up.
What is not here: anything that learns. After every `block_steps` joint ticks `learner.update(params, key)`
is called and returns the weights unchanged, and nothing from the games is kept. What to keep from a
step, the learning signal, the loss, the optimizer and the network are yours to write.

Inside `tick` every step has, for both seats of every game (leading shape [num_envs, 2]): `x` the
visible board, `mask` the legal actions, `ids` the actions played, `result.reward` (+1/-1 on a capture,
otherwise 0), `result.terminated` (a general was captured), `result.truncated` (the 1,000-tick cap) and
`result.final_state`, the true board after the step and before any restart.
"""
import time
START = time.monotonic()
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import policy
import learner
from generals.core.game import GameState
from integrations.generals.environment import (observations, encode, action_mask, decode_action, transition,
                                               reset_done, BOARD_SHAPE, ACTION_COUNT)
from integrations.generals_autoresearch.bundle import save_weights
from integrations.generals_autoresearch.flops import check


def load_maps(path='/data/train-maps.npz'):
    with np.load(path, allow_pickle=False) as bank:
        return GameState(**{name: jnp.asarray(bank[name]) for name in GameState._fields})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--seconds', type=float, required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n, steps = cfg['num_envs'], cfg['block_steps']
    maps = load_maps(); count = len(maps.time); model = policy.create(cfg)
    key, init_key, map_key = jax.random.split(jax.random.PRNGKey(a.seed), 3)
    params = model.init(init_key, jnp.zeros((1, *BOARD_SHAPE, 14)), jnp.ones((1, ACTION_COUNT), bool))['params']
    check(model, params)  # compute per move is capped: fail before training, not at grading
    states = jax.tree.map(lambda v: v[jax.random.randint(map_key, (n,), 0, count)], maps)

    @jax.jit
    def block(params, states, key):
        def tick(carry, _):
            states, key = carry
            key, act_key, map_key = jax.random.split(key, 3)
            obs = observations(states); x = encode(obs); mask = action_mask(obs)
            scores = model.apply({'params': params}, x.reshape((2 * n, *BOARD_SHAPE, 14)), mask.reshape((2 * n, ACTION_COUNT)))[0]
            ids = jax.random.categorical(act_key, scores).astype(jnp.int32).reshape((n, 2))
            result = transition(states, decode_action(ids))
            done = result.terminated | result.truncated
            states = reset_done(result.final_state, done, maps, jax.random.randint(map_key, (n,), 0, count))
            return (states, key), jnp.array([result.terminated.sum(), result.truncated.sum()])
        (states, key), counts = jax.lax.scan(tick, (states, key), None, length=steps)
        return states, key, counts.sum(0)

    tmp = out / 'model.json.tmp'; tmp.write_text(json.dumps(cfg, indent=2) + '\n'); tmp.replace(out / 'model.json')
    blocks = 0; totals = np.zeros(2, np.int64); saved = logged = time.monotonic(); row = {}
    while time.monotonic() - START < a.seconds - 8:
        states, key, counts = block(params, states, key)
        key, update_key = jax.random.split(key)
        params = learner.update(params, update_key)
        blocks += 1; totals += np.asarray(counts)
        now = time.monotonic()
        if blocks == 1 or now - logged >= float(cfg.get('progress_seconds', 2)):
            row = dict(block=blocks, joint_ticks=blocks * steps * n, games_captured=int(totals[0]),
                       games_capped=int(totals[1]), elapsed_seconds=now - START)
            with (out / 'progress.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
            logged = now
        if blocks == 1 or now - saved >= float(cfg.get('save_seconds', 30)):
            save_weights(params, out / 'weights.npz'); saved = now
    if blocks == 0: raise RuntimeError('No block of steps completed in the allocation')
    save_weights(params, out / 'weights.npz')
    (out / 'result.json').write_text(json.dumps(dict(row, block=blocks, joint_ticks=blocks * steps * n,
        games_captured=int(totals[0]), games_capped=int(totals[1]), elapsed_seconds=time.monotonic() - START,
        actual_devices=[d.device_kind for d in jax.devices()], fresh_initialization_seed=a.seed), indent=2) + '\n')


if __name__ == '__main__':
    main()
