"""Offline map and policy artifact audit. Python BFS is diagnostic, never a rollout hot path."""
from collections import deque
import hashlib
import json
from pathlib import Path
import subprocess
import jax
import numpy as np
from .environment import build_pool, map_dimensions, RULESET
from .runtime import HERE, ROOT, sources, write_json


def distances(passable, start, radius=None):
    d = np.full(passable.shape, 9999, np.int32)
    d[start] = 0
    q = deque([start])
    while q:
        i,j = q.popleft()
        if radius is not None and d[i,j] >= radius:
            continue
        for di,dj in ((-1,0),(1,0),(0,-1),(0,1)):
            r,c = i+di,j+dj
            if 0 <= r < d.shape[0] and 0 <= c < d.shape[1] and passable[r,c] and d[r,c] == 9999:
                d[r,c] = d[i,j]+1
                q.append((r,c))
    return d


def main():
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    from .autoresearch.contract import seed_manifest
    seed_data = seed_manifest()
    records = []
    arrays = {}
    for split in ('train','practice','heldout'):
        seeds = seed_data[split]
        states = jax.device_get(build_pool(seeds))
        dimensions = map_dimensions(seeds)
        arrays.update({f'{split}_{k}': np.asarray(v) for k,v in states._asdict().items()})
        for i, seed in enumerate(seeds):
            pos = [tuple(np.argwhere(states.generals[i] & states.ownership[i,p])[0]) for p in (0,1)]
            p0, p1 = pos
            d0 = distances(states.passable[i], p0)
            d1 = distances(states.passable[i], p1)
            dist = int(d0[p1])
            own_cells = states.ownership[i].sum(axis=(1,2)).tolist()
            records.append(dict(split=split, seed=seed, p0=[int(x) for x in p0], p1=[int(x) for x in p1],
                                height=int(dimensions[i,0]), width=int(dimensions[i,1]),
                                spawn_distance=dist, connected=dist < 9999, distinct=p0 != p1,
                                minimum_distance_fallback=dist < RULESET['min_generals_distance'],
                                radius6_room_gap=int(abs(np.sum(d0<=6)-np.sum(d1<=6))),
                                initial_land=own_cells,
                                castles=int(states.castles[i].sum()), mountains=int(states.mountains[i].sum()),
                                board_sha256=hashlib.sha256(b''.join(np.asarray(getattr(states,k)[i]).tobytes() for k in states._fields)).hexdigest()))
    np.savez_compressed(out/'maps.npz', **arrays)
    repo = ROOT/'candidates/generals-bots'
    tracked = subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines()
    weights = [f for f in tracked if Path(f).suffix.lower() in ('.pt','.pth','.ckpt','.pkl','.npz','.safetensors','.msgpack')]
    result = dict(rules=RULESET, source=sources(), maps=records,
                  summary=dict(total=len(records), disconnected=sum(not r['connected'] for r in records),
                               missing_or_colliding_generals=sum(not r['distinct'] or r['initial_land'] != [1,1] for r in records),
                               distance_fallbacks=sum(r['minimum_distance_fallback'] for r in records),
                               min_spawn_distance=min(r['spawn_distance'] for r in records),
                               max_radius6_room_gap=max(r['radius6_room_gap'] for r in records),
                               duplicate_boards=len(records)-len({r['board_sha256'] for r in records})),
                  pretrained_audit=dict(tracked_candidate_weight_files=weights, downloaded_policies=[], policies_used=[],
                    finding='No pretrained policy imported. Upstream experimental PPO/visualization is work in progress; no released weights qualified as a reference.',
                    public_opponents=['integration uniform legal random','integration visible expansion heuristic','upstream MIT HunterAgent'],
                    rules_compatibility='All evaluated via player-visible observations and joint codec on this frozen 1v1 ruleset.'),
                  map_archive_sha256=hashlib.sha256((out/'maps.npz').read_bytes()).hexdigest())
    assert result['summary']['disconnected'] == result['summary']['missing_or_colliding_generals'] == 0
    write_json(out/'audit.json', result)
    print(json.dumps(result['summary'],indent=2))


if __name__ == '__main__':
    main()
