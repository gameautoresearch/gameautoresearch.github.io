"""Self-play plumbing with no learning. Each run starts from random weights and, as it is, ends with them.

What is here: the training maps are loaded, `num_envs` games are played with the simulator in `simulator/`
(`generals_sim`) and one policy sampling the moves of both seats, finished games restart on another training map,
and the run writes `model.json`, `weights.npz`, `progress.jsonl` and `result.json` and stops when its `--seconds`
are up. What is not here: anything that learns. After every `block_steps` joint ticks `learner.update(params, key)`
is called and returns the weights unchanged, and nothing from the games is kept. What to keep from a step, the
learning signal, the loss, the optimizer and the network are yours to write.

Inside the tick loop every step has, for both seats of every game (leading shape [num_envs, 2]): `x` the visible
board, `mask` the legal actions, `ids` the actions played, and per game `reward` (+1/-1 on a capture, otherwise 0),
`terminated` (a general was captured), `truncated` (the 1,000-tick cap) and the game itself (`games[g]`, the true
board after the step and before any restart).
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
from generals_sim import load_maps, player_inputs, decode_action, BOARD_SHAPE, CHANNELS, ACTION_COUNT
from integrations.generals_autoresearch.bundle import save_weights
from integrations.generals_autoresearch.flops import check


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--seconds', type=float, required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n, steps = cfg['num_envs'], cfg['block_steps']
    maps = load_maps(); model = policy.create(cfg)
    rng = np.random.default_rng(a.seed)
    key, init_key = jax.random.split(jax.random.PRNGKey(a.seed))
    params = model.init(init_key, jnp.zeros((1, *BOARD_SHAPE, CHANNELS)), jnp.ones((1, ACTION_COUNT), bool))['params']
    check(model, params)  # compute per move is capped: fail before training, not at grading
    games = [maps[i].copy() for i in rng.integers(len(maps), size=n)]

    @jax.jit
    def act(params, x, mask, key):
        scores = model.apply({'params': params}, x, mask)[0]
        return jax.random.categorical(key, scores).astype(jnp.int32)

    def block(params, key):
        captured = capped = 0
        for _ in range(steps):
            inputs = [player_inputs(game) for game in games]
            x = np.stack([i[0] for i in inputs]); mask = np.stack([i[1] for i in inputs])
            key, act_key = jax.random.split(key)
            ids = np.asarray(act(params, x.reshape((2 * n, *BOARD_SHAPE, CHANNELS)), mask.reshape((2 * n, ACTION_COUNT)),
                                 act_key)).reshape((n, 2))
            for g in range(n):
                terminated, truncated, reward = games[g].step([decode_action(i) for i in ids[g]])
                captured += terminated; capped += truncated
                if terminated or truncated:
                    games[g] = maps[rng.integers(len(maps))].copy()
        return key, np.array([captured, capped])

    tmp = out / 'model.json.tmp'; tmp.write_text(json.dumps(cfg, indent=2) + '\n'); tmp.replace(out / 'model.json')
    blocks = 0; totals = np.zeros(2, np.int64); saved = logged = time.monotonic(); row = {}
    while time.monotonic() - START < a.seconds - 8:
        key, counts = block(params, key)
        key, update_key = jax.random.split(key)
        params = learner.update(params, update_key)
        blocks += 1; totals += counts
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
