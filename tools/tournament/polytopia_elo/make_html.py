"""Build checkpoints.html (self-contained, no network needed) beside a tournament's config from its elo.json.

    PYTHONPATH=$PWD python3 -m tools.tournament.polytopia_elo.make_html --config CONFIG.json

Optional config keys: "grades" ({session key: path to that session's grade.json}), "subtitle" (HTML shown under the
title), "anchor_label" (default "anchor"), "page_sessions" (the session keys whose networks the page shows; default
all; references and anchors always), "continues" (see config.py: a continued session is drawn as part of its
model's line, with both finals and both grades).
"""
import argparse
import json
from pathlib import Path

from tools.tournament.polytopia_elo import config as configs

HERE = Path(__file__).resolve().parent
PALETTE = ['#2a78d6', '#d23c55', '#e09a00', '#159aa8', '#7a52d6', '#1f9a5a', '#8a5a2b', '#555555', '#d4569e', '#6b8e23',
           '#ff7f50', '#4b0082', '#00bfff', '#b8860b', '#c71585']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    a = p.parse_args()
    cfg = configs.load(a.config)
    out = Path(cfg['dir'])
    elo = json.loads((out / 'elo.json').read_text())
    status = json.loads((out / 'status.json').read_text()) if (out / 'status.json').is_file() else {}
    continues = cfg.get('continues', {})
    grades = {}  # by the final's network name: a model with a continued session has two finals and two grades
    grade_ci = {}
    for key, path in cfg.get('grades', {}).items():
        if key not in set(cfg.get('page_sessions') or cfg['sessions']):
            continue
        path = Path(cfg.get('root') or configs.REPO) / path if not Path(path).is_absolute() else Path(path)
        if path.is_file():
            result = json.loads(path.read_text())['result']
            grades[f'{key}:final'] = result['raw_score_0_100']
            if result.get('ci95_map_cluster'):
                grade_ci[f'{key}:final'] = [round(x, 1) for x in result['ci95_map_cluster']]
    # GPU-hours each shown session actually spent (entrants-meta totals), and optional per-session notes ("flags").
    totals = {}
    if (out / 'entrants-meta.json').is_file():
        totals = json.loads((out / 'entrants-meta.json').read_text()).get('totals', {})
    shown = set(cfg.get('page_sessions') or cfg['sessions'])
    order = [k for k in cfg['sessions'] if k not in continues and k in shown]
    models = [dict(k=k, name=cfg['sessions'][k][0], color=PALETTE[i % len(PALETTE)]) for i, k in enumerate(order)]
    nets = [dict(net=n['network'], kind=n['kind'], model=continues.get(n['model'], n['model']),
                 part=2 if n['model'] in continues else 1, run=n['run'] or n['network'], elo=n['elo'],
                 lo=n['ci95'][0], hi=n['ci95'][1], gpu=n['gpu_h_end'], wall=n['wall_h_end'], subs=n['submissions'],
                 grant=n['grant_seconds'], practice=max(n['practice']) if n['practice'] else None, games=n['games'])
            for n in elo['networks'] if n['kind'] in ('anchor', 'reference') or n['model'] in shown]
    starts = {}  # where each continuation starts on each axis: its first network's lease start, from entrants.json
    entrants = json.loads((out / 'entrants.json').read_text()) if (out / 'entrants.json').is_file() else []
    for e in entrants:
        if e.get('continues') and e.get('gpu_h_offset') is not None and e['model'] in shown:
            starts[e['continues']] = dict(gpu=e['gpu_h_offset'], wall=e['wall_h_offset'])
    scored = [n['elo'] for n in nets if n['kind'] in ('research', 'final')]
    every = [n['elo'] for n in nets]
    lo, hi = min(every), max(every)
    step = lambda span: 50 if span <= 800 else 100 if span <= 1600 else 250 if span <= 4000 else 500
    rnd = lambda v, s, up: (int(v // s) + (1 if up else 0)) * s
    top_lo = max(lo, -450)
    zoom_lo = max(lo, sorted(scored)[max(0, len(scored) // 10)] - 300) if scored else lo
    rng = lambda a, b: [rnd(a, step(b - a), False), rnd(b, step(b - a), True), step(b - a)]
    gpu = [n['gpu'] for n in nets if n['gpu'] is not None]
    wall = [n['wall'] for n in nets if n['wall'] is not None]
    anchor = cfg.get('anchor_label', 'anchor')
    sched = status.get('schedule', 'full')
    meta = dict(games=elo['games'], pairs=elo['pairs_played'], per_pair=elo['games_per_pair_max'], failed=elo['failed_calls'],
                capped=elo['capped_games'], rmse=round(elo['fit']['pair_score_rmse'], 3),
                binsd=round(elo['fit']['mean_binomial_sd'], 3), grades=grades, grade_ci=grade_ci, flags=cfg.get('flags', {}),
                used={k: (totals.get(k) or {}).get('total_gpu_h') for k in cfg['sessions'] if (totals.get(k) or {}).get('total_gpu_h')},
                starts=starts,
                part_label=cfg.get('part_label', 'continued'),
                cost=round(status.get('est_compute_usd') or 0), backend=status.get('backend', 'modal-function'),
                title=cfg['title'], anchor_label=anchor,
                subtitle=cfg.get('subtitle', f"<code>{cfg['name']}</code>: every network each model sealed during research, "
                                             f"its final, the anchors and the references in one tournament. Elo is "
                                             f"relative to {anchor} (= 0)."),
                notes=(f"<p>{elo['games']:,} games over {elo['pairs_played']:,} pairs ({sched} schedule, "
                       f"{elo['games_per_pair_min']}–{elo['games_per_pair_max']} games per played pair; "
                       f"{elo['capped_games']:,} games reached the round cap and were scored by in-game score), the "
                       f"release's frozen grader with uid-isolated policy processes. Elo: Bradley–Terry on pair "
                       f"scores, {anchor} = 0; intervals are games-only (binomial redraws). Fit: pair-score RMSE "
                       f"{round(elo['fit']['pair_score_rmse'], 3)} vs mean binomial SD "
                       f"{round(elo['fit']['mean_binomial_sd'], 3)}.</p>"),
                ytop=rng(top_lo, hi + 30), yzoom=rng(zoom_lo, hi + 30), yfull=rng(lo - 50, hi + 30),
                xmax_gpu=round(max(gpu) * 1.03, 2) if gpu else 1, xmax_wall=round(max(wall) * 1.03, 2) if wall else 1)
    html = (HERE / 'checkpoints.template.html').read_text()
    html = html.replace('/*DATA*/null', json.dumps(dict(nets=nets, meta=meta, models=models), separators=(',', ':')))
    (out / 'checkpoints.html').write_text(html)
    print('wrote', out / 'checkpoints.html', len(html), 'bytes')


if __name__ == '__main__':
    main()
