"""Generals tournament worker: uploaded into the grading sandbox and run there as root.

Both entrants run as untrusted `PolicyProcess` children (uid 65534, pipe protocol), exactly as the
evaluator runs a candidate; neither is loaded in-process. For every private map, A plays seat 0 against
B in seat 1, and then the seats are swapped. As in `arena.evaluate`, a part holds one seat assignment and
fresh policy processes, so no process ever sees opposite-seat views of the same private map.

Points (from A's side): capture win 1, capture loss 0, the ruleset's turn cap or simultaneous outcome 0.5. A policy
fault (protocol, deadline or illegal action) forfeits that part's unfinished games to the other entrant; such
rows are marked `outcome: fault` with `fault_by` and are never counted as natural results.

Only the standard library is imported at module level so the host tests can import `schedule`.
"""
import json
import sys
import time


def schedule(map_indices, batch_size=32):
    """Parts of (a_seat, maps). Both seats per map; one seat assignment per part."""
    indices = list(map_indices)
    return [(seat, indices[c:c + batch_size]) for seat in (0, 1) for c in range(0, len(indices), batch_size)]


def points(reward, truncated):
    if truncated or reward == 0:
        return 0.5
    return 1.0 if reward > 0 else 0.0


def main(request_path):
    from concurrent.futures import ThreadPoolExecutor
    import os
    import resource
    import numpy as np
    import jax
    import jax.numpy as jnp
    from integrations.generals_autoresearch.arena import PolicyProcess, PolicyFailure
    from integrations.generals_autoresearch.contract import DEFAULT, write_json
    from integrations.generals_autoresearch.rollout import load_pool
    from integrations.generals.environment import (RULESET, observations, encode as visible, action_mask,
                                                   decode_action, transition)

    request = json.loads(open(request_path).read())
    bundles = {'a': request['a'], 'b': request['b']}
    pool = load_pool(request['pool_path'])
    indices = list(request['map_indices'])
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= len(pool.time):
        raise ValueError('Invalid map selection')
    timeout = DEFAULT['inference_timeout_seconds']
    view = jax.jit(lambda s: (visible(observations(s)), action_mask(observations(s))))
    step = jax.jit(transition)
    rows = []
    faults = []
    started = time.monotonic()
    parts = schedule(indices, request.get('batch_size', 32))
    executor = ThreadPoolExecutor(max_workers=2)
    for number, (a_seat, maps) in enumerate(parts):
        batch = len(maps)
        rr = jnp.arange(batch)
        seat = {'a': jnp.full(batch, a_seat), 'b': jnp.full(batch, 1 - a_seat)}
        states = jax.tree.map(lambda v: v[jnp.asarray(maps)], pool)
        active = np.ones(batch, bool)
        print(json.dumps(dict(part=number + 1, parts=len(parts), a_seat=a_seat, maps=len(maps),
                              games_completed=len(rows), seconds=time.monotonic() - started)), flush=True)
        with PolicyProcess(bundles['a'], False, timeout) as pa, PolicyProcess(bundles['b'], False, timeout) as pb:
            procs = {'a': pa, 'b': pb}
            for tick in range(RULESET['truncation']):
                x, mask = view(states)
                xs = {k: (np.asarray(x[rr, seat[k]]), np.asarray(mask[rr, seat[k]])) for k in procs}
                failed = {}
                chosen = {}
                if tick == 0:
                    # First forwards compile; run them one at a time as in single-candidate grading.
                    for k in procs:
                        try:
                            chosen[k] = procs[k].choose(*xs[k])
                        except PolicyFailure as exc:
                            failed[k] = str(exc)
                else:
                    futures = {k: executor.submit(procs[k].choose, *xs[k]) for k in procs}
                    for k, future in futures.items():
                        try:
                            chosen[k] = future.result()
                        except PolicyFailure as exc:
                            failed[k] = str(exc)
                if failed:
                    for i in np.flatnonzero(active):
                        a_points = 0.5 if len(failed) == 2 else (0.0 if 'a' in failed else 1.0)
                        rows.append(dict(map_index=int(maps[i]), a_seat=a_seat, a_points=a_points, outcome='fault',
                                         fault_by='+'.join(sorted(failed)), length=tick, a_reward=None))
                    faults.append(dict(part=number + 1, tick=tick, errors={k: v[-2000:] for k, v in failed.items()}))
                    active[:] = False
                    break
                joint = (jnp.zeros((batch, 2), jnp.int32).at[rr, seat['a']].set(jnp.asarray(chosen['a']))
                         .at[rr, seat['b']].set(jnp.asarray(chosen['b'])))
                result = step(states, decode_action(joint))
                done = np.asarray(result.terminated | result.truncated)
                reward = np.asarray(result.reward[rr, seat['a']])
                natural = np.asarray(result.terminated)
                cap = np.asarray(result.truncated)
                for i in np.flatnonzero(active & done):
                    rows.append(dict(map_index=int(maps[i]), a_seat=a_seat, a_reward=float(reward[i]),
                                     a_points=points(float(reward[i]), bool(cap[i]) and not bool(natural[i])),
                                     outcome='natural' if natural[i] else 'truncated', fault_by=None, length=tick + 1))
                live = jnp.asarray(active)
                states = jax.tree.map(lambda n, o: jnp.where(live.reshape((batch,) + (1,) * (n.ndim - 1)), n, o),
                                      result.final_state, states)
                active &= ~done
                if not active.any():
                    break
            if active.any():
                raise RuntimeError('Unfinished games after the turn cap')
    executor.shutdown()
    if len(rows) != 2 * len(indices):
        raise RuntimeError('Incomplete tournament schedule')
    write_json('/private/tournament-result.json', dict(
        rows=rows, faults=faults, seconds=time.monotonic() - started, games=len(rows),
        inference_platform=os.environ.get('GENERALS_POLICY_PLATFORM', 'cpu'), unprivileged_policies=True,
        root_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == '__main__':
    main(sys.argv[1])
