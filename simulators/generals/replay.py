"""Verify a recorded evaluation action stream without loading a policy or hidden inputs."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .environment import build_pool, observations, action_mask, decode_action, transition
from .runtime import digest, sources, write_json


def verify(path):
    path = Path(path)
    metadata = json.loads(Path(str(path)+'.json').read_text())
    if digest(path.read_bytes()) != metadata['sha256']:
        raise ValueError('Replay payload checksum mismatch')
    binding = sources()
    for field in ('commit','tree_sha256','rules_sha256','implementation_sha256'):
        if metadata['source'][field] != binding[field]:
            raise ValueError(f'Replay {field} mismatch')
    with np.load(path, allow_pickle=False) as data:
        seeds = data['seeds'].copy()
        actions = jnp.asarray(data['action_ids'])
        active = jnp.asarray(data['active'])
        expected = {k.removeprefix('final_'): data[k].copy() for k in data.files if k.startswith('final_')}
    pool = build_pool(seeds.tolist())
    def run(states):
        def tick(carry, pair):
            states, live = carry
            ids, recorded_active = pair
            mask = action_mask(observations(states))
            invalid = live * jnp.sum(~jnp.take_along_axis(mask, ids[..., None], -1)[..., 0], -1)
            result = transition(states, decode_action(ids))
            states = jax.tree.map(lambda n, o: jnp.where(live.reshape(live.shape+(1,)*(n.ndim-1)), n, o), result.final_state, states)
            mismatch = jnp.sum(live != recorded_active)
            live = live & ~(result.terminated | result.truncated)
            return (states, live), (jnp.sum(invalid), mismatch)
        return jax.lax.scan(tick, (states, jnp.ones(seeds.shape, bool)), (actions, active))
    (final, live), (invalid, mismatch) = jax.jit(run)(pool)
    final = jax.device_get(final)
    exact = all(np.array_equal(expected[k], getattr(final, k)) for k in expected)
    result = dict(exact_final_state=exact, invalid_actions=int(invalid.sum()),
                  active_boundary_mismatches=int(mismatch.sum()), unfinished_games=int(live.sum()),
                  recorded_games=len(seeds), recorded_active_ticks=int(active.sum()),
                  sha256=metadata['sha256'], backend=jax.default_backend(),
                  device_kinds=[d.device_kind for d in jax.devices()],
                  deterministic_scope=f'Exact on this tested {jax.default_backend()} backend; cross-device bitwise identity unqualified.')
    if not exact or result['invalid_actions'] or result['active_boundary_mismatches'] or result['unfinished_games']:
        raise AssertionError(result)
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('path')
    p.add_argument('--output', required=True)
    a=p.parse_args()
    result=verify(a.path)
    write_json(a.output,result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
