"""Score arbitrary sealed Pokémon bundles against the starter anchors (anchor games only, no round robin).

    PYTHONPATH=$PWD MODAL_PROFILE=duper MODAL_ENVIRONMENT=game-autoresearch-project .venv-pokemon/bin/python \
        -m tools.tournament.pokemon_checkpoints --manifest MANIFEST.json --bank BANK.json \
        (--entrant NAME ... | --group finals | --group research --part K --parts N) --out NEW_DIR \
        [--smoke] [--parallel 48] [--cap-usd 25 --spent-usd X]

MANIFEST is `tools.tournament.pokemon_curves enumerate` output: {release, anchors: {NAME: {bundle, bundle_sha256}},
entrants: {NAME: {bundle, bundle_sha256}}, attempts: [...]}. Every entrant plays every anchor and nothing else: one
pair (entrant = A, anchor = B) per job, on the seeds of BANK (a fresh host-secret bank created on first use and
shared by every part of this job, so all bundles are scored on the same seeds), A in seat 0 and seat 1 of every
seed. The games are played exactly as tools.tournament.pokemon plays a pair (`pokemon.play`: the release's frozen
`runtime/arena.py` on CPU workers, both bundles as isolated untrusted policy processes), with a tournament-local
ledger in NEW_DIR. Each bundle is re-verified with the release's native verifier against the manifest's hash
before any game. Each pair's rows are written as soon as its job settles.

Spend guard: CPU worker seconds are priced at the frozen list rate used by the campaigns' costs.py. Before each job
starts, if --spent-usd + this run's settled spend + (jobs in flight + 1) x the guard estimate per job would exceed
--cap-usd, the job is not launched and is recorded as `budget guard` (no worker created).
"""
import sys

sys.dont_write_bytecode = True

import argparse
import json
import secrets
import threading
import time
from pathlib import Path

from tools.tournament import common, pokemon

CPU_WORKER_USD_S = 4 * 0.00003942 + 16 * 0.00000667   # as runs/local/poke3v3-gpt6-20261115/costs.py
GUARD_JOB_SECONDS = 240.0                              # a 256-game job took 118-233 s in the GPT-6 tournament


def make_bank(path, count):
    path = Path(path)
    if path.exists():
        seeds = json.loads(path.read_text())['seeds']
        if len(seeds) != count:
            raise ValueError(f'{path} has {len(seeds)} seeds, need {count}')
        return seeds
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    seeds = secrets.SystemRandom().sample(range(1_900_000_000, 2_000_000_000), count)
    common.write_json(path, dict(seeds=seeds))
    return seeds


class Guard:
    def __init__(self, cap, spent, clock):
        self.cap, self.spent, self.clock = cap, spent, clock
        self.lock = threading.Lock()
        self.flight = 0

    def enter(self):
        with self.lock:
            projected = self.spent + self.clock.seconds * CPU_WORKER_USD_S + \
                (self.flight + 1) * GUARD_JOB_SECONDS * CPU_WORKER_USD_S
            if projected > self.cap:
                raise RuntimeError(f'budget guard: projected ${projected:.2f} > cap ${self.cap:.2f}; not launched')
            self.flight += 1

    def leave(self):
        with self.lock:
            self.flight -= 1


def select(manifest, args):
    names = list(manifest['entrants'])
    if args.entrant:
        unknown = [n for n in args.entrant if n not in manifest['entrants']]
        if unknown:
            raise ValueError('Not in the manifest: ' + ', '.join(unknown))
        return list(args.entrant)
    if args.group == 'finals':
        return [n for n in names if n.endswith('-final')]
    research = [n for n in names if not n.endswith('-final')]
    size = -(-len(research) // args.parts)
    return research[(args.part - 1) * size: args.part * size]


def main(argv=None, provider=None, poll=5.0):
    p = argparse.ArgumentParser(prog='python -m tools.tournament.pokemon_checkpoints', description=__doc__.split('\n')[0])
    p.add_argument('--manifest', required=True)
    p.add_argument('--bank', required=True, help='private seed bank (created if missing; kept out of results)')
    p.add_argument('--entrant', action='append', default=[])
    p.add_argument('--group', choices=('finals', 'research'))
    p.add_argument('--part', type=int, default=1)
    p.add_argument('--parts', type=int, default=1)
    p.add_argument('--out', required=True)
    p.add_argument('--smoke', action='store_true', help='2 seeds')
    p.add_argument('--parallel', type=int, default=48)
    p.add_argument('--chunk', type=int, default=128, help='seeds per CPU worker (128 = one job per pair)')
    p.add_argument('--cap-usd', type=float, default=25.0)
    p.add_argument('--spent-usd', type=float, default=0.0, help='spend of earlier parts of this job')
    args = p.parse_args(argv)
    if not args.entrant and not args.group:
        p.error('give --entrant or --group')
    started = time.time()
    manifest = json.loads(Path(args.manifest).read_text())
    release = Path(manifest['release']).resolve()
    release_manifest = pokemon.release_manifest(release)
    if release_manifest['release_sha256'] != manifest['release_sha256']:
        raise ValueError('Release changed since the manifest was written')
    chosen = select(manifest, args)
    if not chosen:
        raise ValueError('No entrants selected')
    entrants = {}
    for name, item in [*((n, manifest['entrants'][n]) for n in chosen), *manifest['anchors'].items()]:
        sealed = pokemon.entrant('bundle', item['bundle'], release_manifest)
        if sealed['bundle_sha256'] != item['bundle_sha256']:
            raise ValueError(f'{name}: bundle changed since the manifest ({item["bundle"]})')
        entrants[name] = sealed
    anchors = list(manifest['anchors'])
    out = common.new_out(args.out, [release, *(Path(e['bundle']) for e in entrants.values())])
    count = 2 if args.smoke else pokemon.UNITS[False]
    seeds = make_bank(out / 'private/bank.json' if args.smoke else args.bank, count)
    common.write_json(out / 'request.json', dict(argv=sys.argv if argv is None else argv, seeds=count,
                                                 chunk=args.chunk, entrants=chosen, anchors=anchors,
                                                 bank='private (not in results)', cap_usd=args.cap_usd,
                                                 spent_usd_before=args.spent_usd))
    if provider is None:
        from harness.policy import require_modal
        from harness.games.pokemon.provider import Provider
        require_modal()
        provider = Provider(release, out / 'provider')
    ledger = pokemon.Ledger(out / 'ledger.json')
    clock = common.Clock()
    guard = Guard(args.cap_usd, args.spent_usd, clock)
    pairs = [(a, b) for a in chosen for b in anchors]
    parts = common.chunks(count, args.chunk)
    plan = [(pair, part) for pair in pairs for part in parts]
    progress = dict(done=0, lock=threading.Lock())

    def task(pair, part):
        guard.enter()
        try:
            value = pokemon.play(provider, ledger, clock, release, entrants, pair, part, seeds, poll)
        finally:
            guard.leave()
        a, b = pair
        common.write_json(out / 'pairs' / f'{a}-vs-{b}' / f'rows-{part[0]:04d}.json', value)
        with progress['lock']:
            progress['done'] += 1
            spent = clock.seconds * CPU_WORKER_USD_S
            print(f'{time.strftime("%H:%M:%S")} {progress["done"]}/{len(plan)} {a}-vs-{b} '
                  f'run ${spent:.2f} total ${args.spent_usd + spent:.2f}', flush=True)
        return value

    outcomes = common.run_tasks([lambda pair=pair, part=part: task(pair, part) for pair, part in plan], args.parallel)
    results = []
    for pair in pairs:
        a, b = pair
        rows, errors, chunks = [], [], []
        for (owner, part), (value, error) in zip(plan, outcomes):
            if owner != pair:
                continue
            if error:
                errors.append(dict(seeds=[part[0], part[-1]], error=error))
                continue
            rows.extend(value['rows'])
            errors.extend(value['errors'])
            chunks.append(dict(seeds=[part[0], part[-1]], operation=value['operation'], output=value['output'],
                               status=value['status']))
        rows.sort(key=lambda r: (r['unit'], r['seat']))
        owners = {}
        for row in rows:
            owner = pokemon.fault_owner(row['fault'])
            if owner:
                key = {'a': a, 'b': b}.get(owner, 'other')
                owners[key] = owners.get(key, 0) + 1
        details = out / 'pairs' / f'{a}-vs-{b}'
        common.write_json(details / 'rows.json', rows)
        common.write_json(details / 'chunks.json', chunks)
        results.append(common.pair_summary(a, b, rows, 2 * count, f'pairs/{a}-vs-{b}/', errors, owners))
    value = common.result('pokemon', release, release_manifest['release_sha256'], bool(args.smoke), entrants, results,
                          clock, started, pokemon.POINTS_RULE,
                          harness_release_sha256=common.harness_release_sha256(release), seeds=count,
                          seats='A in seat 0 and seat 1 on every seed', anchors=anchors, schedule='entrants x anchors',
                          evaluator='runtime/arena.py run (frozen release), CPU workers', ledger='ledger.json',
                          cpu_worker_usd_s=CPU_WORKER_USD_S,
                          usd=round(clock.seconds * CPU_WORKER_USD_S, 4))
    common.write_json(out / 'result.json', value)
    print(json.dumps(dict(out=str(out), pairs=len(results), games=sum(r['games'] for r in results),
                          incomplete=[f"{r['a']}-vs-{r['b']}" for r in results if not r['complete']],
                          worker_seconds=value['worker_seconds'], usd=value['usd']), indent=2))
    return value


if __name__ == '__main__':
    main()
