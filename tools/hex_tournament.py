"""Hex round robin between submitted networks, with optional calibrated panel references, and an Elo fit.

Every pair of players plays the same 2-ply openings from both seats, all games of all
pairs in one lockstep batch. Each submitted network runs in its own unprivileged
policy process (the evaluator's `PolicyProcess`), answering in batches for every game
where it is to move; references (random, heuristic, MoHex levels) are the grading
panel's own players. Ratings are a Bradley-Terry maximum-likelihood fit
(`ladder.fit`); when references are present the scale is shifted so the references'
mean fitted rating equals their mean calibrated panel Elo (MoHex anchor = 0).

It runs on one Modal CPU worker built from the release's own image, in the project
environment, network blocked; nothing here is charged to any research allocation.

    .venv-hex/bin/python tools/hex_tournament.py --release RELEASE \
        --session flash=SESSION_DIR ... --reference-top 5 --openings 64 --out NEW_DIR

`--session` enters a finished session: the network a network session submitted last (the one its final grade
scored) or a recipe session's fresh final network. `--entrant NAME=BUNDLE_DIR` enters a sealed bundle directly.
Every network plays under the contract's per-move inference clock, exactly as in grading.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Runs inside the worker as `python /private/tournament_core.py REQUEST`.
CORE = r'''
import json, sys, time
from itertools import combinations
import numpy as np
import torch
from simulators.hex.autoresearch.arena import (MoHexPool, PolicyProcess, PolicyTimeout, check_panel, opening_bank,
                                               opponent_moves, start_games)
from simulators.hex.autoresearch.contract import DEFAULT, PANEL
# fit, pairwise: prepended from simulators/hex/autoresearch/ladder.py (not in the frozen worker tree)

request = json.load(open(sys.argv[1]))
torch.set_num_threads(1)
entrants = request['entrants']            # {name: bundle dir}
panel = {p['name']: p for p in PANEL}
references = [panel[name] for name in request['references']]
check_panel(references) if references else None
players = list(entrants) + [r['name'] for r in references]
pairs = list(combinations(players, 2))
bank = opening_bank(request['seed'], request['openings'])
jobs = [(a, b, i, a_seat) for a, b in pairs for i in range(request['openings']) for a_seat in (1, 2)]
games, history = start_games([bank[i] for _, _, i, _ in jobs])
n = len(jobs)
rngs = [np.random.default_rng([request['seed'], k]) for k in range(n)]
started = time.monotonic()
procs = {name: PolicyProcess(path, development=False, seed=request['seed'] % 2 ** 31)  # the contract's move clock
         for name, path in entrants.items()}
failures = {}
batch = DEFAULT['inference_batch']
from concurrent.futures import ThreadPoolExecutor
threads = ThreadPoolExecutor(max(1, len(entrants)))
try:
    with MoHexPool(request['workers'], request['seed'] % 2 ** 31) as pool:
        while not bool(games.done.all()):
            running = ~games.done
            legal = games.legal()
            actions = torch.zeros(n, dtype=torch.long)
            mover = {}
            for k in running.nonzero().squeeze(1).tolist():
                a, b, _, a_seat = jobs[k]
                mover[k] = a if int(games.to_play[k]) == a_seat else b
            def answer(name):
                # Each network is its own process, so networks answer concurrently.
                proc, out = procs[name], {}
                ids = [k for k, who in mover.items() if who == name]
                for c in range(0, len(ids), batch):
                    chunk = ids[c:c + batch]
                    try:
                        chosen = proc.choose(boards[chunk], to_play[chunk], legal_np[chunk])
                    except PolicyTimeout:  # the benchmark's move clock: a late call plays random moves, then resumes
                        chosen = [int(np.random.default_rng([request['seed'], k, len(history[k])]).choice(
                            legal_np[k].nonzero()[0])) for k in chunk]
                    except Exception as exc:  # a failing network forfeits: it plays uniformly random moves
                        failures.setdefault(name, str(exc)[:300])
                        chosen = [int(np.random.default_rng([request['seed'], k, len(history[k])]).choice(
                            legal_np[k].nonzero()[0])) for k in chunk]
                    out.update(zip(chunk, chosen))
                return out
            boards, to_play, legal_np = games.board.numpy(), games.to_play.numpy(), legal.numpy()
            for out in threads.map(answer, list(procs)):
                for k, cell in out.items():
                    actions[k] = int(cell)
            refs = [(k, panel[who]) for k, who in mover.items() if who in panel and who not in procs]
            for k, cell in opponent_moves(refs, games, legal, rngs, history, pool).items():
                actions[k] = cell
            games.step(actions, running)
            for k in running.nonzero().squeeze(1).tolist():
                history[k].append(int(actions[k]))
        restarts = list(pool.restarts)
finally:
    for proc in procs.values():
        proc.close()
rows = [{'a': a, 'b': b, 'opening_index': i, 'opening': list(bank[i]), 'a_seat': 'black' if s == 1 else 'white',
         'a_won': int(games.winner[k]) == s, 'plies': len(history[k]), 'moves': history[k]}
        for k, (a, b, i, s) in enumerate(jobs)]
ratings = fit(rows, players)
if references:
    shift = np.mean([panel[r['name']]['elo'] - ratings[r['name']]['elo'] for r in references])
    for value in ratings.values():
        value['elo'] += float(shift)
scores = {p: [0, 0] for p in players}
for r in rows:
    scores[r['a']][0] += r['a_won']; scores[r['a']][1] += 1
    scores[r['b']][0] += 1 - r['a_won']; scores[r['b']][1] += 1
json.dump({'players': players, 'references': [r['name'] for r in references], 'openings': request['openings'],
           'games': len(rows), 'seconds': time.monotonic() - started, 'ratings': ratings,
           'anchored_to_panel': bool(references),
           'scores': {p: {'wins': w, 'games': g, 'rate': w / g} for p, (w, g) in scores.items()},
           'pairwise': pairwise(rows), 'policy_failures': failures,
           'move_timeouts': {name: proc.timeouts for name, proc in procs.items()}, 'mohex_restarts': restarts, 'rows': rows},
          open('/private/tournament.json', 'w'))
print('done', len(rows), 'games', round(time.monotonic() - started), 's')
'''


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--release', required=True, type=Path)
    p.add_argument('--session', action='append', default=[], metavar='NAME=SESSION_DIR', help='a finished session')
    p.add_argument('--entrant', action='append', default=[], metavar='NAME=BUNDLE_DIR', help='a sealed bundle')
    p.add_argument('--reference', action='append', default=[], help='panel opponent name')
    p.add_argument('--reference-top', type=int, default=0, metavar='N', help='also the N strongest panel opponents')
    p.add_argument('--openings', type=int, default=64)
    p.add_argument('--seed', type=int, default=20261113)
    p.add_argument('--cpu', type=int, default=16)
    p.add_argument('--timeout', type=int, default=6 * 3600)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    if a.out.exists():
        p.error('--out must be a new directory')
    import modal
    from harness.policy import ENVIRONMENT, require_modal
    from harness.games.hex.provider import PYTHON, image_for
    from simulators.hex.autoresearch.bundle import archive_bundle, verify
    from simulators.hex.autoresearch.contract import PANEL
    from tools.tournament import common
    require_modal()
    entrants = {}
    for item in a.session:
        name, session = item.split('=', 1)
        session = Path(session).resolve()
        variant = common.session_settings(session).get('variant')
        if variant == 'network':
            submitted = common.network_submission(session)
            entrants[name] = session / 'research/jobs' / submitted['name'] / 'bundle'
            verify(entrants[name], submitted['native']['bundle_sha256'])
        else:
            common.session_release(session)  # refuses anything but a recipe session with a complete fresh final
            expected = json.loads((session / 'final-train.json').read_text())['submission']['bundle_sha256']
            entrants[name] = session / 'final/jobs/final/bundle'
            verify(entrants[name], expected)
    for item in a.entrant:
        name, path = item.split('=', 1)
        if name in entrants:
            p.error('duplicate entrant name ' + name)
        verify(path)
        entrants[name] = Path(path).resolve()
    if len(entrants) < 2:
        p.error('a tournament needs at least two entrants')
    strongest = [x['name'] for x in sorted(PANEL, key=lambda x: -x['elo'])[:a.reference_top]]
    a.reference = list(dict.fromkeys([*a.reference, *strongest]))
    a.out.mkdir(parents=True)
    app = modal.App('game-autoresearch-hex-tournament')
    with app.run(environment_name=ENVIRONMENT):
        sb = modal.Sandbox.create('/bin/sleep', '86400', app=app, image=image_for(a.release), cpu=(a.cpu, a.cpu),
                                  memory=(32768, 32768), timeout=a.timeout, block_network=True,
                                  environment_name=ENVIRONMENT)
        started = time.time()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                for name, path in entrants.items():
                    archive = Path(tmp) / f'{name}.tar'
                    archive_bundle(path, archive)
                    sb.filesystem.copy_from_local(archive, f'/private/{name}.tar')
                    proc = sb.exec('bash', '-c', f'mkdir -p /private/{name} && tar -xf /private/{name}.tar '
                                                 f'-C /private/{name} && ls /private/{name}')
                    proc.wait()
            # The bundle files must be readable by the unprivileged policy child; PolicyProcess copies them.
            request = dict(entrants={n: f'/private/{n}' for n in entrants}, references=a.reference,
                           openings=a.openings, seed=a.seed, workers=a.cpu - 1)
            sb.filesystem.write_text(json.dumps(request), '/private/request.json')
            import inspect
            from simulators.hex.autoresearch import ladder
            core = 'import math\nimport numpy as np\n' + inspect.getsource(ladder.fit) + '\n' + inspect.getsource(ladder.pairwise) + CORE
            sb.filesystem.write_text(core, '/private/tournament_core.py')
            proc = sb.exec('bash', '-c', f'cd /opt/hex && {PYTHON} /private/tournament_core.py /private/request.json',
                           timeout=a.timeout - 60)
            for line in proc.stdout:
                print(line, end='', flush=True)
            code = proc.wait()
            if code != 0:
                raise RuntimeError('Tournament failed: ' + proc.stderr.read()[-3000:])
            result = json.loads(sb.filesystem.read_text('/private/tournament.json'))
        finally:
            sb.terminate()
    result.update(release=str(a.release), entrants={n: str(p) for n, p in entrants.items()},
                  host_seconds=time.time() - started)
    (a.out / 'tournament.json').write_text(json.dumps(result, indent=1) + '\n')
    for name in sorted(result['ratings'], key=lambda n: -result['ratings'][n]['elo']):
        r, s = result['ratings'][name], result['scores'][name]
        print(f"{name:20s} elo {r['elo']:8.1f} ± {r['se']:5.1f}   score {s['wins']}/{s['games']} ({s['rate']:.1%})")


if __name__ == '__main__':
    main()
