"""Builds result.json and summary.md for the local Pokémon 3v3 Foul Play ladder from OUT/rows.jsonl.

Pair format as tools/tournament results: a = network, b = Foul Play rung. Only natural, fault-free games are scored
(win 1, tie 0.5, loss 0 from the network's seat); truncations, faults and unplayed games are counted separately.
"""
import sys

sys.dont_write_bytecode = True

from collections import defaultdict
from pathlib import Path
import argparse
import json
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import pokemon_foulplay_local as ladder  # noqa: E402

RUNG_ORDER = ['foulplay-10ms', 'foulplay-100ms', 'foulplay-1s']


def model_of(name):
    return name.rsplit('-', 1)[0]


def owner(category):
    if not category:
        return None
    return 'a' if category.startswith('candidate') else 'b' if category.startswith('reference') else 'other'


def norm(r):
    clean = bool(r.get('natural')) and r.get('fault') is None
    return dict(unit=r['seed_index'], seat=r['seat'], a_points=r['points'] if clean else None,
                natural=bool(r.get('natural')), truncated=bool(r.get('truncated')),
                fault=r.get('fault_category') or ('fault' if r.get('fault') else None))


def boot(units, replicates=4000, seed=20261111):
    if not units:
        return None
    means = np.array([np.mean(v) for _, v in sorted(units.items())])
    rng = np.random.default_rng(seed)
    b = means[rng.integers(len(means), size=(replicates, len(means)))].mean(1) * 100
    return [float(x) for x in np.quantile(b, [.025, .975])]


def visits(rr):
    v = [r for r in rr if r.get('search_samples')]
    if not v:
        return 'n/a | n/a'
    per = np.median([r['search_visits'] / r['search_samples'] for r in v])
    rate = sum(r['search_visits'] for r in v) / max(1e-9, sum(r['search_seconds'] for r in v))
    return f'{per:,.0f} | {rate / 1e6:.2f} M ({len(v)} games)'


def fmt(score, ci, n):
    if score is None:
        return 'n/a'
    return f'{score:.1f} [{ci[0]:.1f}, {ci[1]:.1f}] (n={n})'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(ladder.DEFAULT_OUT))
    p.add_argument('--jobs', type=int, default=12)
    a = p.parse_args()
    out = Path(a.out)
    rows = [json.loads(line) for line in (out / 'rows.jsonl').read_text().splitlines() if line.strip()]
    run = json.loads((out / 'run-main.json').read_text())
    plan = {item['rung']: item for item in run['plan']}
    entr = ladder.entrants()
    request = json.loads((ladder.RELEASE / 'private/assessment/registry.json').read_text())['evaluator_request']
    release_sha = json.loads((ladder.RELEASE / 'release.json').read_text()).get('release_sha256') \
        if (ladder.RELEASE / 'release.json').exists() else None
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[(r['network'], r['rung'])].append(r)
    pairs = []
    for rung in RUNG_ORDER:
        if rung not in plan:
            continue
        for name in plan[rung]['networks']:
            raw = by_pair.get((name, rung), [])
            nr = [norm(r) for r in raw]
            faults_by = defaultdict(int)
            for r in raw:
                o = owner(r.get('fault_category'))
                if o:
                    faults_by[o] += 1
            s = common.pair_summary(name, rung, nr, 2 * plan[rung]['seeds'], details=str(out / 'rows.jsonl'),
                                    faults_by=dict(faults_by))
            s['seeds_scheduled'] = plan[rung]['seeds']
            s['seeds_played'] = len({r['seed_index'] for r in raw})
            pairs.append(s)
    entrants = {n: dict(bundle=v['bundle'], bundle_sha256=v['bundle_sha256'], kind='network', model=model_of(n))
                for n, v in entr.items()}
    for rung, ms in ladder.RUNGS.items():
        entrants[rung] = dict(kind='search', note=(
            f"Foul Play (pinned source archive sha256 {request['source_archive_sha256']}) via "
            f"strong_bots.foul_process_v4, mode='production', samples={ladder.SAMPLES}, time_ms={ms}, iterations=1000 "
            f"(unused in production mode), per-decision timeout {ladder.foul_timeout(ms):.2f} s, game wall cap "
            f"{ladder.WALL_CAP[rung]} s; local poke-engine 0.0.48 (terastallization), development mode (no UID isolation)"),
            time_ms=ms, samples=ladder.SAMPLES, mode='production')
    seeds_by_rung = {r: plan[r]['seeds'] for r in plan}
    uniform = len(set(seeds_by_rung.values())) == 1
    result = dict(
        game='pokemon', release=str(ladder.RELEASE), release_sha256=release_sha, smoke=False, entrants=entrants,
        pairs=pairs, seeds=next(iter(seeds_by_rung.values())) if uniform else seeds_by_rung,
        seeds_by_rung=seeds_by_rung, seats='network in seat 0 and seat 1 on every seed; same private seeds for every network and rung',
        points_rule=("per game: natural engine outcome from the NETWORK's seat, win 1, tie 0.5, loss 0; administrative "
                     "truncations, faults and unplayed games are counted separately and never scored"),
        evaluator=('tools/tournament/pokemon_foulplay_local.py: EvaluationVector(1, team_size=3) + runtime.policy_process.'
                   'PolicyProcess (bundle-native, verified) + strong_bots.foul_process_v4.FoulPlayProcess, local Mac CPU'),
        machine='Apple M5 Max (6 Super + 12 Performance cores), 64 GB, macOS', concurrency=a.jobs,
        started_at=run.get('started_at'), finished_at=run.get('finished_at'), built_at=time.time(),
        unstarted_games=run.get('unstarted'))
    common.write_json(out / 'result.json', result)

    # ---- summary.md
    rung_rows = defaultdict(list)
    for r in rows:
        rung_rows[r['rung']].append(r)
    models = sorted({model_of(n) for n in entr})
    lines = ['# Pokémon 3v3 networks vs Foul Play, local ladder (2026-11-15)', '',
             'Machine: Apple M5 Max (6 Super + 12 Performance cores, 64 GB), macOS, CPU only, '
             f'{a.jobs} games in parallel, `evidence/local-performance.lock` held for the whole run.', '',
             'Each game: one network (sealed bundle, sha256-verified, runtime binding checked) vs Foul Play, '
             "`mode='production'`, `samples=2`, fresh simulator/network/Foul Play processes per game. "
             'Same 32 private seeds (bank in `private/`) for every network and rung, network in both seats. '
             'Win rate = points per natural, fault-free game (tie 0.5). 95 % CIs: bootstrap over seed clusters '
             '(both seats together); per-model CIs resample (network, seed) clusters, i.e. treat that model\'s networks as fixed.', '']
    lines += ['## Games, faults, timing per rung', '',
              '| rung | networks | seeds | games played | scored | truncations | faults | mean wall s/game | mean CPU s/game | Foul Play decisions/game (search) | MCTS visits per search sample (median) | search visits/s |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for rung in RUNG_ORDER:
        rr = rung_rows.get(rung, [])
        if rung not in plan:
            continue
        scored = sum(1 for r in rr if r.get('natural') and not r.get('fault'))
        lines.append(f"| {rung} | {len(plan[rung]['networks'])} | {plan[rung]['seeds']} | {len(rr)} | {scored} | "
                     f"{sum(bool(r.get('truncated')) for r in rr)} | {sum(bool(r.get('fault')) for r in rr)} | "
                     f"{np.mean([r['wall_seconds'] for r in rr]) if rr else 0:.2f} | "
                     f"{np.mean([r['cpu_seconds'] for r in rr]) if rr else 0:.2f} | "
                     f"{np.mean([r['foul_decisions'] for r in rr]) if rr else 0:.1f} "
                     f"({np.mean([r['decision_counts']['actual_search_actions'] for r in rr]) if rr else 0:.1f}) | "
                     f"{visits(rr)} |")
    lines.append('')
    faults = [r for r in rows if r.get('fault')]
    if faults:
        lines += ['Faults (never scored):', '']
        for r in faults:
            lines.append(f"- {r['rung']} {r['network']} seed {r['seed_index']} seat {r['seat']}: {r['fault_category']}: {r['fault'][:200]}")
        lines.append('')
    lines += ['## Per model: network win rate vs each rung', '',
              '| model | networks | vs 10 ms | vs 100 ms | vs 1 s (tonight\'s a4/a5 + starter-s2 only) |', '|---|---|---|---|---|']
    model_table = {}
    for m in models:
        cells = []
        for rung in RUNG_ORDER:
            units = defaultdict(list)
            for r in rung_rows.get(rung, []):
                if model_of(r['network']) == m and r.get('natural') and not r.get('fault'):
                    units[(r['network'], r['seed_index'])].append(r['points'])
            n = sum(len(v) for v in units.values())
            score = 100 * sum(sum(v) for v in units.values()) / n if n else None
            ci = boot(units)
            model_table[(m, rung)] = (score, ci, n)
            cells.append(fmt(score, ci, n))
        lines.append(f"| {m} | {sum(1 for n in entr if model_of(n) == m)} | " + ' | '.join(cells) + ' |')
    lines += ['', '## Per network', '', '| network | vs 10 ms | vs 100 ms | vs 1 s | truncations | faults |', '|---|---|---|---|---|---|']
    pidx = {(p['a'], p['b']): p for p in pairs}
    for n in sorted(entr):
        cells = []
        tr = fa = 0
        for rung in RUNG_ORDER:
            pr = pidx.get((n, rung))
            if not pr:
                cells.append('—')
                continue
            cells.append(fmt(pr['a_score_0_100'], pr['ci95'], pr['games']))
            tr += pr['truncations']
            fa += pr['faults']
        lines.append(f'| {n} | ' + ' | '.join(cells) + f' | {tr} | {fa} |')
    unstarted = run.get('unstarted')
    lines += ['', '## Cuts and completeness', '',
              f"Planned: 100 ms and 10 ms rungs 38 networks x {plan.get('foulplay-100ms', {}).get('seeds')}/"
              f"{plan.get('foulplay-10ms', {}).get('seeds')} seeds; 1 s rung 15 networks x "
              f"{plan.get('foulplay-1s', {}).get('seeds')} seeds; both seats. Unstarted games at the stop time: {unstarted}.",
              '', '## Caveats', '',
              '- Foul Play\'s search is time-based (`monte_carlo_tree_search(duration_ms=...)`, 1 thread, 2 belief samples searched in '
              'sequence), so a given rung is stronger on a fast CPU. The M5 Max cores are faster than Modal CPU workers, so on an idle Mac '
              'these settings would be stronger than on Modal (including the 10 ms practice-panel opponent); but this run\'s 12-way load roughly halved the search (see calibration below), so whether these rungs are stronger or weaker than Modal\'s is unverified. The M5 Max also mixes '
              '6 Super and 12 Performance cores; with 12 concurrent games the OS scheduler decides which game gets which core type, '
              'so per-decision search effort varies somewhat between games.',
              '- `foulplay-100ms` approximates upstream Foul Play\'s default 100 ms search; upstream also searches 4 x 50 ms samples '
              'before the opponent has revealed moves, which the adapter cannot do (here always 2 samples x time_ms).',
              '- Local differences from the graded arena: no separate-UID isolation (macOS, `development=True`, no deny file); Foul Play '
              'per-decision timeout raised to 3 x samples x time_ms/1000 + 2 s (2.06 / 2.6 / 8 s) and the whole-game cap to '
              '300 / 600 / 900 s; the pinned Foul Play source archive is unpacked once (sha256 checked) and the local `.venv-pokemon` '
              'poke-engine 0.0.48 (built from the pinned source with terastallization) replaces the pinned Linux wheels.',
              '- Smoke games (2 networks x 2 seeds x 2 seats x 3 rungs, `smoke-rows.jsonl`, 16 in parallel) are not counted.']
    lines += ['', '## Run history', '',
              '- 22:46-22:47 smoke (24 games, 16 parallel): all natural, no faults; mean wall ~1.5 s (10 ms), ~4.6 s (100 ms), ~23 s (1 s) per game, CPU ~= wall (one core per game).',
              '- 22:47:39 first full launch at 16 parallel games: the Mac showed 0 % idle CPU (games plus Spotlight/trustd/UI load), i.e. Foul Play was oversubscribed. '
              'Stopped after 469 finished 100 ms games; those rows are kept in `rows-aborted-16jobs.jsonl` (with `main-aborted-16jobs.log`, `run-aborted-16jobs.json`) and are NOT counted.',
              '- 22:49:30 restarted at 12 parallel games (`rows.jsonl`); at ~22:51 stopped once more (282 rows done, kept and counted: same settings) to add MCTS-visit accounting, '
              'then resumed (`main-part1.log`/`run-main-part1.json`, `main.log`/`run-main.json`). The 282 first rows (100 ms rung) have no visit fields. In-flight games at each stop were discarded and replayed.',
              '- 23:44:57 all 5,824 scheduled games finished; nothing cut (the 00:30 stop time was never reached). Wall time of the counted run: ~55 min (22:49:30-23:44:57).',
              '- 23:45-23:48 uncontended calibration: same 2 networks x 2 seeds x 2 seats x 3 rungs with ONE game at a time (`calib-rows.jsonl`, not counted).']
    cal_path = out / 'calib-rows.jsonl'
    if cal_path.exists():
        cal = [json.loads(line) for line in cal_path.read_text().splitlines() if line.strip()]
        lines += ['', '## Search effort under load vs uncontended (calibration)', '',
                  '| rung | median MCTS visits / search sample, main run (12 parallel) | same, 1 game at a time | ratio |', '|---|---|---|---|']
        for rung in RUNG_ORDER:
            a = [r['search_visits'] / r['search_samples'] for r in rung_rows.get(rung, []) if r.get('search_samples')]
            b = [r['search_visits'] / r['search_samples'] for r in cal if r['rung'] == rung and r.get('search_samples')]
            if a and b:
                lines.append(f'| {rung} | {np.median(a):,.0f} | {np.median(b):,.0f} (8 games) | {np.median(a) / np.median(b):.2f} |')
        lines += ['', 'So under the 12-way load Foul Play completed roughly half the MCTS iterations it does on an idle M5 Max at 10/100 ms '
                  '(about two thirds at 1 s): the rungs here are weaker than the same time setting on an idle Mac, and the ratio to a '
                  'Modal CPU worker is unknown (no Modal visit counts were recorded). On the 24 calibration games the outcomes matched '
                  'the main-run outcomes for the same network/seed/seat in all 24 cases (small sample).']
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
