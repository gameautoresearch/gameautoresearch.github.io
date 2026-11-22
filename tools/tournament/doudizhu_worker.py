"""Dou Dizhu tournament worker: uploaded into the CPU grading sandbox and run there as root.

Team scheme. For every private deal the same cards are played twice:
  ordering 0: A is the landlord, B plays both peasants (landlord_down and landlord_up);
  ordering 1: B is the landlord, A plays both peasants.
A side scores 1 when its side wins the deal, else 0, so each deal is zero-sum and both entrants hold every
seat once on identical cards. Each entrant runs as three untrusted `PolicyProcess` children (uid 65534, pipe
protocol), one per role, so no process ever sees two hands of the same deal (a peasant pair is two separate
processes that cannot share hidden cards). A policy fault forfeits the batch's unfinished games to the other
entrant; those rows are `outcome: fault` with `fault_by`, never natural results.

Only the standard library is imported at module level so the host tests can import `schedule`.
"""
import json
import sys
import time

ROLES = ('landlord', 'landlord_down', 'landlord_up')


def schedule(deals, batch_size=32):
    """Batches of (deal_index, ordering); both orderings of a deal share a batch when batch_size is even."""
    games = [(d, o) for d in range(deals) for o in (0, 1)]
    return [games[c:c + batch_size] for c in range(0, len(games), batch_size)]


def controller(ordering, role):
    """Which entrant ('a' or 'b') plays `role` in this ordering."""
    return 'a' if (ordering == 0) == (role == 'landlord') else 'b'


def a_won(ordering, landlord_won):
    return landlord_won == (ordering == 0)


def main(request_path):
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    import resource
    from integrations.doudizhu.autoresearch.arena import PolicyProcess, PolicyFailure
    from integrations.doudizhu.autoresearch.bundle import unpack_bundle
    from integrations.doudizhu.autoresearch.contract import DEFAULT, write_json
    from integrations.doudizhu.environment import deal
    from integrations.doudizhu.workers import VectorGames

    request = json.loads(Path(request_path).read_text())
    bundles = {}
    for team in ('a', 'b'):
        target = request[team]
        unpack_bundle(target + '.tar', target)
        bundles[team] = target
    split, deal_seed, deals = request['split'], int(request['deal_seed']), int(request['deals'])
    if split != 'heldout' or deals <= 0:
        raise ValueError('Invalid tournament schedule')
    timeout = DEFAULT['inference_timeout_seconds']

    def spawn(team, role):
        return PolicyProcess(bundles[team], False, timeout)

    procs = {(team, role): spawn(team, role) for team in ('a', 'b') for role in ROLES}
    rows = []
    faults = []
    started = time.monotonic()
    batches = schedule(deals, request.get('batch_size', 32))
    executor = ThreadPoolExecutor(max_workers=4)
    try:
        with VectorGames(4, engine_backend='native') as vector:
            for number, part in enumerate(batches):
                print(json.dumps(dict(batch=number + 1, batches=len(batches), games_completed=len(rows),
                                      seconds=time.monotonic() - started)), flush=True)
                cards = [deal(split, deal_seed, d) for d, _ in part]
                active = {i: v[0] for i, v in vector.reset(cards).items()}
                while active:
                    groups = {}
                    for i, obs in active.items():
                        groups.setdefault((controller(part[i][1], obs.role), obs.role), []).append(i)
                    futures = {key: executor.submit(procs[key].choose, [active[i] for i in ids])
                               for key, ids in groups.items()}
                    selected = {}
                    failed = {}
                    for key, future in futures.items():
                        try:
                            selected.update(zip(groups[key], future.result()))
                        except PolicyFailure as exc:
                            failed[key] = str(exc)
                    if failed:
                        teams = sorted({team for team, _ in failed})
                        for i in active:
                            d, o = part[i]
                            rows.append(dict(deal_index=d, ordering=o, a_role='landlord' if o == 0 else 'peasants',
                                             a_points=0.5 if len(teams) == 2 else (0.0 if teams == ['a'] else 1.0),
                                             outcome='fault', fault_by='+'.join(teams), landlord_won=None,
                                             scores=None, decisions=None))
                        faults.append(dict(batch=number + 1, errors={t + ':' + r: e[-2000:] for (t, r), e in failed.items()}))
                        for key in failed:
                            procs[key].close()
                            procs[key] = spawn(*key)
                        break
                    result = vector.step(selected)
                    active = {}
                    for i, (obs, outcome, _) in result.items():
                        if outcome is None:
                            active[i] = obs
                            continue
                        d, o = part[i]
                        won = a_won(o, bool(outcome['landlord_won']))
                        rows.append(dict(deal_index=d, ordering=o, a_role='landlord' if o == 0 else 'peasants',
                                         a_points=1.0 if won else 0.0, outcome='natural', fault_by=None,
                                         landlord_won=bool(outcome['landlord_won']),
                                         scores={r: float(outcome['scores'][r]) for r in ROLES},
                                         decisions=outcome.get('decisions')))
    finally:
        executor.shutdown()
        for proc in procs.values():
            proc.close()
    if len(rows) != 2 * deals:
        raise RuntimeError('Incomplete tournament schedule')
    write_json('/private/tournament-result.json', dict(
        rows=rows, faults=faults, seconds=time.monotonic() - started, games=len(rows), unprivileged_policies=True,
        policy_processes_per_entrant=3, root_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == '__main__':
    main(sys.argv[1])
