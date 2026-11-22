"""Elo of every checkpoint from the games played so far, per-model research curves, REPORT.md.

Safe to run at any time, also while run.py is playing (it reads only complete lines of games.jsonl). Needs
matplotlib (e.g. .venv-pokemon):

    PYTHONPATH=$PWD .venv-pokemon/bin/python -m tools.tournament.polytopia_elo.report --config CONFIG.json [--draws 200]

Method (as the Pokémon checkpoint-Elo run): Bradley-Terry by MM on pair points (win 1, draw 0.5; a game cut at the
30-round cap counts as the grader adjudicates it, by in-game score), Elo = 400 log10 strength, the anchors (the config's) = 0. Intervals are games-only: every pair's score redrawn binomially from its own game
count and the fit repeated. Failed calls (an illegal action or a grader error) play no games; they are counted.
Outputs (in DIR): elo.json, curves.png, REPORT.md.
"""
import argparse, json, math, time

TITLE = 'polytopia'
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.tournament.polytopia_elo import config as configs
from tools.tournament.polytopia_elo.schedule import fit
PALETTE = ['#2a6fdb', '#d1495b', '#2e9e5b', '#e0a100', '#7b4fd6', '#1a9fb0', '#8a5a2b', '#555555', '#d4569e', '#6b8e23',
           '#ff7f50', '#4b0082', '#00bfff', '#b8860b', '#c71585']
REF_STYLE = {'random': ':', 'starter-20m': '-.', 'starter-65m': '--', 'starter-200m': (0, (6, 2))}


def read_jsonl(path):
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a line being written right now
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--complete-rounds-only', action='store_true')
    ap.add_argument('--draws', type=int, default=200)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    cfg = configs.load(args.config)
    out_dir = Path(cfg['dir'])
    entrants_path = out_dir / 'entrants.json'
    global TITLE
    TITLE = cfg['title']
    entrants = {e['name']: e for e in json.loads(entrants_path.read_text())}
    names = list(entrants)
    kind = {n: entrants[n]['kind'] for n in names}
    npairs = sum(1 for i in range(len(names)) for j in range(i + 1, len(names))
                 if not (kind[names[i]] == 'reference' and kind[names[j]] == 'reference'))
    rows = read_jsonl(out_dir / 'games.jsonl')
    jobs = read_jsonl(out_dir / 'jobs.jsonl')
    per_round = defaultdict(set)
    # Games of a network that is no longer an entrant (its session's record no longer lists it) are left out of the fit.
    dropped = sum(1 for r in rows if r['a'] not in entrants or r['b'] not in entrants)
    if dropped:
        print(f'{dropped} games involve networks that are not entrants; left out of the fit', flush=True)
    rows = [r for r in rows if r['a'] in entrants and r['b'] in entrants]
    for r in rows:
        per_round[r['round']].add(tuple(sorted((r['a'], r['b']))))
    complete = sorted(k for k, v in per_round.items() if len(v) == npairs)
    if args.complete_rounds_only:
        keep = set(complete)
        rows = [r for r in rows if r['round'] in keep]
    agg = defaultdict(lambda: [0.0, 0])
    capped = 0
    for r in rows:
        capped += bool(r.get('capped'))
        key = tuple(sorted((r['a'], r['b'])))
        pts = r['a_points'] if key[0] == r['a'] else 1 - r['a_points']
        agg[key][0] += pts; agg[key][1] += 1
    pairs = [(a, b, v[0], v[1]) for (a, b), v in agg.items() if v[1]]
    if not pairs:
        print('no games yet'); return
    played = sorted({x for p in pairs for x in p[:2]})
    elo = fit(played, pairs)
    idx = {n: i for i, n in enumerate(played)}
    zero = [idx[n] for n in played if kind[n] == 'anchor']
    shift = elo[zero].mean() if zero else elo.mean()
    elo = elo - shift
    rng = np.random.default_rng(20261118)
    draws = []
    for _ in range(args.draws):
        red = [(a, b, float(rng.binomial(n, min(max(s / n, 1e-6), 1 - 1e-6))), n) for a, b, s, n in pairs]
        v = fit(played, red, iterations=1500, tol=1e-8)
        draws.append(v - (v[zero].mean() if zero else v.mean()))
    draws = np.array(draws) if draws else np.zeros((1, len(played)))
    pred = [1 / (1 + 10 ** ((elo[idx[b]] - elo[idx[a]]) / 400)) for a, b, _, _ in pairs]
    obs = [s / n for _, _, s, n in pairs]
    rmse = float(np.sqrt(np.mean((np.array(pred) - np.array(obs)) ** 2)))
    binsd = float(np.mean([math.sqrt(q * (1 - q) / n) for q, (_, _, _, n) in zip(pred, pairs)]))
    games_per = defaultdict(int)
    for a, b, _, n in pairs:
        games_per[a] += n; games_per[b] += n
    nets = []
    for n in played:
        e = entrants[n]
        i = idx[n]
        nets.append(dict(network=n, kind=e['kind'], model=e['model'], label=e.get('label'), run=e.get('run'),
                         elo=round(float(elo[i]), 1),
                         ci95=[round(float(np.percentile(draws[:, i], 2.5)), 1),
                               round(float(np.percentile(draws[:, i], 97.5)), 1)],
                         games=games_per[n], submissions=e.get('submissions', []), practice=e.get('practice', []),
                         grant_seconds=e.get('grant_seconds'), gpu_h_end=e.get('gpu_h_end'),
                         wall_h_end=e.get('wall_h_end'), gpu_h_start=e.get('gpu_h_start'),
                         wall_h_start=e.get('wall_h_start'), continues=e.get('continues')))
    nets.sort(key=lambda x: -x['elo'])
    status = json.loads((out_dir / 'status.json').read_text()) if (out_dir / 'status.json').is_file() else {}
    pair_games = [n for _, _, _, n in pairs]
    failed = sum(1 for j in jobs if j.get('games') != j.get('expected'))
    out = dict(generated_unix=time.time(), games=int(sum(pair_games)), pairs_played=len(pairs), pairs_total=npairs,
               complete_rounds=len(complete), games_per_pair_min=min(pair_games), games_per_pair_max=max(pair_games),
               capped_games=capped, failed_calls=failed, complete_rounds_only=args.complete_rounds_only,
               method='Bradley-Terry (MM) on pair points; Elo = 400 log10 strength; anchors = 0; '
                      f'{args.draws} games-only binomial redraws', fit=dict(pair_score_rmse=rmse, mean_binomial_sd=binsd),
               networks=nets)
    (out_dir / 'elo.json').write_text(json.dumps(out, indent=1) + '\n')
    models = []
    for e in entrants.values():
        if e['kind'] in ('research', 'final') and e['model'] not in models:
            models.append(e['model'])
    labels = {e['model']: e.get('label') or e['model'] for e in entrants.values()}
    parents = {e['model']: e['continues'] for e in entrants.values() if e.get('continues')}
    roots = [m for m in models if m not in parents]
    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(roots)}
    colors.update({m: colors.get(p, PALETTE[0]) for m, p in parents.items()})  # a continuation keeps its model's colour
    # The figure follows the page: only the sessions in "page_sessions" (default all) are drawn; the fit is unchanged.
    shown = cfg.get('page_sessions')
    plot_models = [m for m in models if not shown or m in shown]
    plot(nets, out, plot_models, labels, {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(plot_models)} if shown else colors,
         out_dir)
    write_report(nets, out, status, models, labels, out_dir)
    if not args.quiet:
        print((out_dir / 'REPORT.md').read_text()[:6000])


def plot(nets, out, models, labels, colors, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    by = defaultdict(list)
    for n in nets:
        if n['kind'] == 'research':
            by[n['model']].append(n)
    finals = {n['model']: n for n in nets if n['kind'] == 'final'}
    refs = [n for n in nets if n['kind'] == 'reference']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6), sharey=True)
    for ax, key, label in ((axes[0], 'gpu_h_end', 'research GPU-hours charged when the network was sealed'),
                           (axes[1], 'wall_h_end', 'hours into the session when the network was sealed')):
        ax.axhline(0, color='#999', lw=0.9, ls=':')
        for r in refs:
            ax.axhline(r['elo'], color='#bbb', lw=0.9, ls=REF_STYLE.get(r['network'], '--'))
            ax.annotate(r['network'], (1.0, r['elo']), xycoords=('axes fraction', 'data'), fontsize=7, color='#888',
                        ha='right', va='bottom')
        for m in models:
            pts = sorted((p for p in by.get(m, []) if p[key] is not None), key=lambda x: x[key])
            c = colors[m]
            if pts:
                if pts[0].get('continues'):  # join the earlier session's last network to this one's first
                    before = sorted((p for p in by.get(pts[0]['continues'], []) if p[key] is not None), key=lambda x: x[key])
                    if before:
                        ax.plot([before[-1][key], pts[0][key]], [before[-1]['elo'], pts[0]['elo']], ':', color=c, lw=1)
                ax.plot([p[key] for p in pts], [p['elo'] for p in pts], '-', color=c, lw=1, alpha=0.45)
                ax.errorbar([p[key] for p in pts], [p['elo'] for p in pts],
                            yerr=[[p['elo'] - p['ci95'][0] for p in pts], [p['ci95'][1] - p['elo'] for p in pts]],
                            fmt='o', color=c, ms=4, lw=0.8, capsize=0, label=labels[m])
                best, bx, byy = -1e9, [], []
                for p in pts:
                    best = max(best, p['elo']); bx.append(p[key]); byy.append(best)
                ax.step(bx, byy, where='post', color=c, lw=2.2, alpha=0.9)
                subs = [p for p in pts if p['submissions']]
                ax.scatter([p[key] for p in subs], [p['elo'] for p in subs], marker='*', s=230, color=c,
                           edgecolor='black', linewidth=0.8, zorder=5)
            f = finals.get(m)
            if f and f.get(key) is not None:
                ax.errorbar([f[key]], [f['elo']], yerr=[[f['elo'] - f['ci95'][0]], [f['ci95'][1] - f['elo']]],
                            fmt='D', ms=9, color=c, mec='black', mew=1.0, lw=1.0, zorder=7,
                            label=None if pts else labels[m])
        ax.set_xlabel(label)
        ax.set_xlim(left=0)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('Elo (PPO starter host-trained 900 s = 0)')
    axes[0].legend(loc='lower right', fontsize=8)
    fig.suptitle(f"{TITLE} checkpoints: Elo vs research spent  ({out['games']:,} games, "
                 f"{out['games_per_pair_min']}-{out['games_per_pair_max']} per pair). Thick step = best so far, "
                 "star = formally submitted, diamond = fresh 15-min final of the last submission, "
                 "grey lines = ladder references, dotted = PPO starter 15 min", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / 'curves.png', dpi=130)
    plt.close(fig)


def write_report(nets, out, status, models, labels, out_dir):
    t = time.strftime('%Y-%m-%d %H:%M', time.localtime(out['generated_unix']))
    running = status.get('running')
    L = [f'# {TITLE} checkpoint Elo — {"INTERIM, run in progress" if running else "results"} ({t})', '']
    sched = status.get('schedule', 'full')
    L.append(f"Schedule {sched}: games {out['games']:,} over {out['pairs_played']:,} of {out['pairs_total']:,} possible "
             f"pairs ({out['games_per_pair_min']}–{out['games_per_pair_max']} per played pair); {out['capped_games']:,} games reached the 30-round cap (scored by in-game score); "
             f"{out['failed_calls']} failed calls. Fit: pair-score RMSE {out['fit']['pair_score_rmse']:.3f} vs mean "
             f"binomial SD {out['fit']['mean_binomial_sd']:.3f}.")
    if status:
        L.append(f"Run ({status.get('backend', 'modal-function')}): {status.get('elapsed_h', 0):.2f} h, {status.get('games_per_s')} "
                 f"games/s, est. compute ${status.get('est_compute_usd')}, "
                 f"{'running' if running else status.get('stop_reason')}.")
    L += ['', '![curves](curves.png)', '', '## Per model', '']
    by = defaultdict(list)
    for n in nets:
        by[n['model']].append(n)
    L.append('| Model | Networks | Fresh final | Last submitted network | Best network | Best by 1 / 2 / 3 / 4 GPU-h / end |')
    L.append('|---|---:|---|---|---|---|')
    for m in models:
        pts = sorted((p for p in by.get(m, []) if p['kind'] == 'research'), key=lambda x: x['gpu_h_end'] or 0)
        final = next((p for p in by.get(m, []) if p['kind'] == 'final'), None)
        last = max((p for p in pts if p['submissions']), key=lambda p: max(p['submissions']), default=None)
        best = max(pts, key=lambda p: p['elo']) if pts else None

        def best_by(h):
            v = [p['elo'] for p in pts if (p['gpu_h_end'] or 0) <= h]
            return f'{max(v):+.0f}' if v else '—'
        fmt = lambda p: f"`{p['run']}` {p['elo']:+.0f} [{p['ci95'][0]:+.0f}, {p['ci95'][1]:+.0f}]" if p else '—'
        L.append(f"| {labels[m]} | {len(pts)} | {fmt(final)} | {fmt(last)} | " +
                 (f"`{best['run']}` {best['elo']:+.0f} at {best['gpu_h_end']:.2f} GPU-h" if best else '—') +
                 f" | {best_by(1)} / {best_by(2)} / {best_by(3)} / {best_by(4)} / {best_by(99)} |")
    for m in models:
        pts = sorted(by.get(m, []), key=lambda x: (x['kind'] == 'final', x.get('gpu_h_end') or 0))
        if not pts:
            continue
        L += ['', f'### {labels[m]}', '', '| Network | GPU-h at seal | Session h at seal | Lease grant s | Practice | '
              'Submitted as | Elo [95 %] | Games |', '|---|---:|---:|---:|---:|---|---|---:|']
        for p in pts:
            g = f"{p['gpu_h_end']:.2f}" if p['gpu_h_end'] is not None else '—'
            w = f"{p['wall_h_end']:.2f}" if p['wall_h_end'] is not None else '—'
            pr = ', '.join(f'{x:.1f}' for x in p['practice']) if p['practice'] else '—'
            sub = 'fresh final' if p['kind'] == 'final' else ', '.join(f'#{s}' for s in p['submissions'])
            L.append(f"| `{p['network']}` | {g} | {w} | {p['grant_seconds'] or '—'} | {pr} | {sub} | "
                     f"{p['elo']:+.1f} [{p['ci95'][0]:+.0f}, {p['ci95'][1]:+.0f}] | {p['games']:,} |")
    others = [p for p in nets if p['kind'] in ('anchor', 'reference')]
    if others:
        L += ['', '### Anchor and ladder references', '', '| Network | Kind | Elo [95 %] | Games |', '|---|---|---|---:|']
        for p in sorted(others, key=lambda x: x['elo']):
            L.append(f"| `{p['network']}` | {p['kind']} | {p['elo']:+.1f} [{p['ci95'][0]:+.0f}, {p['ci95'][1]:+.0f}] | "
                     f"{p['games']:,} |")
    pr = [(np.mean(p['practice']), p['elo']) for p in nets if p['kind'] == 'research' and p.get('practice')]
    if len(pr) > 3:
        r = float(np.corrcoef(*zip(*pr))[0, 1])
        L += ['', f'Public practice score vs tournament Elo across {len(pr)} research networks: r = {r:.2f}.']
    L += ['', '## Leaderboard', '', '| # | Network | Elo [95 %] | Games |', '|---:|---|---|---:|']
    for i, p in enumerate(nets, 1):
        L.append(f"| {i} | `{p['network']}` | {p['elo']:+.1f} [{p['ci95'][0]:+.0f}, {p['ci95'][1]:+.0f}] | {p['games']:,} |")
    L += ['', 'GPU-h = research allocation the session had been charged (training-worker hours; a lease charges its '
          'seconds x its shape\'s rate, a running lease pro rata) at the moment the network was sealed (`ar package`). '
          'Session h = hours since the session\'s first `ar` request. A fresh final is placed at the session\'s total '
          'research GPU-h. References never play each other.', '',
          'Played on 1-core Modal Functions with the release\'s own worker image and frozen CPU grader (/opt/v1/grade.py, '
          'each network in its own unprivileged /opt/v1/policy.py process). See README.md.']
    (out_dir / 'REPORT.md').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
