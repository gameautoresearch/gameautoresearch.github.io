"""Report for tools.tournament.pokemon_curves analyze: lift per research bundle, per-attempt curves and summaries.

Lift of a bundle = 100 x (mean over the three starter anchors of its mean points against that anchor) - 50, as in
tools.tournament.analyze. Its 95 % interval is a bootstrap over the evaluation seeds (one seed = both seats against
all three anchors, resampled together, since every pair uses the same bank); the anchors are fixed, not resampled.
"""
import sys

sys.dont_write_bytecode = True

import csv
import html
import json
import math
from pathlib import Path

import numpy as np

from tools.tournament.pokemon_curves import BATCH, LABELS

STARTER_RECIPE = BATCH / 'pokemon/starter/s2/recipes/001/recipe.json'


def starter_relation(session, run, starter):
    """'starter' = the release's starter recipe files exactly; 'starter-code' = starter code with another train
    config; 'modified' = any code change. From sources/<run>/run.json file hashes."""
    files = json.loads((Path(session) / 'sources' / run / 'run.json').read_text())['files']
    code = all(files.get(k) == v for k, v in starter.items() if k.endswith('.py'))
    if code and files.get('configs/train.json') == starter.get('configs/train.json'):
        return 'starter'
    return 'starter-code' if code else 'modified'

REPLICATES = 4000
SUBSTANTIAL_H = 25 / 60
COLORS = ('#d9822b', '#2a9d8f', '#5b5bd6')  # amber, jade, violet: attempts 1-3 in session order
FINAL_COLOR = '#d1495b'


def load_pairs(results):
    """{(entrant, anchor): {unit: [points]}} from part folders; the first complete pair wins."""
    pairs, sources, spend = {}, [], dict(worker_seconds=0.0, usd=0.0, games=0, faults=0, truncations=0, parts=[])
    for folder in results:
        folder = Path(folder)
        result = json.loads((folder / 'result.json').read_text())
        if result.get('smoke'):
            continue
        spend['worker_seconds'] += result['worker_seconds']
        spend['usd'] += result.get('usd') or 0.0
        spend['parts'].append(dict(folder=str(folder), pairs=len(result['pairs']),
                                   games=sum(p['games'] for p in result['pairs']),
                                   worker_seconds=result['worker_seconds'], usd=result.get('usd')))
        for pair in result['pairs']:
            key = (pair['a'], pair['b'])
            spend['faults'] += pair['faults']
            spend['truncations'] += pair['truncations']
            if key in pairs or not pair['complete']:
                continue
            rows = json.loads((folder / pair['details'] / 'rows.json').read_text())
            units = {}
            for row in rows:
                if row['a_points'] is not None:
                    units.setdefault(row['unit'], []).append(row['a_points'])
            pairs[key] = units
            spend['games'] += pair['games']
    return pairs, spend


def lift(pairs, name, anchors, rng):
    """(lift, [lo, hi], se, games) or None when any anchor pair is missing."""
    if any((name, a) not in pairs for a in anchors):
        return None
    units = sorted(set.intersection(*(set(pairs[(name, a)]) for a in anchors)))
    # matrix[u, k] = mean points on seed u against anchor k
    matrix = np.array([[np.mean(pairs[(name, a)][u]) for a in anchors] for u in units])
    point = 100 * matrix.mean(0).mean() - 50
    boot = matrix[rng.integers(len(units), size=(REPLICATES, len(units)))].mean(1).mean(1) * 100 - 50
    lo, hi = np.quantile(boot, [.025, .975])
    games = sum(len(v) for a in anchors for v in pairs[(name, a)].values())
    return float(point), [float(lo), float(hi)], float(boot.std()), games


def practice_score(job):
    scores = [p['score'] for p in job['practice'] if p['score'] is not None]
    return (max(scores) if scores else None), scores


def run(args):
    out = Path(args.out)
    manifest = json.loads(Path(args.manifest).read_text())
    anchors = list(manifest['anchors'])
    pairs, spend = load_pairs(args.result)
    rng = np.random.default_rng(20261115)
    combined = json.loads(Path(args.combined).read_text())['games']['pokemon']
    combined_lifts = {m['model']: m.get('attempt_lifts') or {} for m in combined['models']}

    starter = json.loads(STARTER_RECIPE.read_text())['files']
    table, attempts, sanity = [], [], []
    for a in manifest['attempts']:
        model, attempt = a['model'], a['attempt']
        rows = []
        for job in a['jobs']:
            value = lift(pairs, job['entrant'], anchors, rng)
            best_practice, scores = practice_score(job)
            row = dict(model=model, label=LABELS[model], attempt=attempt, entrant=job['entrant'], run=job['run'],
                       job_id=job['operation'], requested_seconds=job['requested_seconds'],
                       start_h=round(job['start_h'], 3), end_h=round(job['end_h'], 3),
                       allocation_before_h=round(job['allocation_before_s'] / 3600, 3)
                       if job['allocation_before_s'] is not None else None,
                       lift=value and round(value[0], 2), ci95=value and [round(x, 2) for x in value[1]],
                       se=value and round(value[2], 2), games=value and value[3],
                       practice_score=best_practice, practice_scores=scores,
                       is_final_recipe=job['is_final_recipe'], scored=value is not None,
                       recipe_vs_starter=starter_relation(a['session'], job['run'], starter))
            rows.append(row)
        for s in a['skipped']:
            table.append(dict(model=model, label=LABELS[model], attempt=attempt, entrant=None, run=s['run'],
                              job_id=s['operation'], requested_seconds=s['requested_seconds'],
                              start_h=s['start_h'] and round(s['start_h'], 3), end_h=s['end_h'] and round(s['end_h'], 3),
                              lift=None, ci95=None, practice_score=practice_score(s)[0], practice_scores=[],
                              is_final_recipe=s['is_final_recipe'], scored=False, skipped=s['reason']))
        table.extend(rows)
        final_value = lift(pairs, a['final'].get('entrant'), anchors, rng) if a['final'].get('entrant') else None
        reference = combined_lifts.get(model, {}).get(attempt)
        if final_value is not None and reference is not None:
            # the reference has the same design (3 anchors x 128 seeds x 2 seats) on another bank: same SE assumed
            z = (final_value[0] - reference) / (math.sqrt(2) * final_value[2])
            sanity.append(dict(model=model, attempt=attempt, here=round(final_value[0], 2),
                               ci95=[round(x, 2) for x in final_value[1]], combined=round(reference, 2),
                               diff=round(final_value[0] - reference, 2), z=round(z, 2), within_95=abs(z) < 1.96))
        scored = [r for r in rows if r['scored']]
        scored.sort(key=lambda r: r['end_h'])
        best, curve = -1e9, []
        for r in scored:
            best = max(best, r['lift'])
            curve.append(dict(end_h=r['end_h'], lift=r['lift'], best_so_far=round(best, 2), entrant=r['entrant']))
        first = scored[0] if scored else None
        top = max(scored, key=lambda r: r['lift']) if scored else None
        final_lift = final_value[0] if final_value else None
        recipe_runs = [r for r in rows if r['is_final_recipe']]
        summary = dict(
            model=model, label=LABELS[model], attempt=attempt, research_bundles=len(rows), skipped=len(a['skipped']),
            first_run=first and first['run'], first_run_end_h=first and first['end_h'],
            first_run_requested_s=first and first['requested_seconds'],
            first_run_lift=first and first['lift'], first_run_ci95=first and first['ci95'],
            first_full_run=None, first_full_run_lift=None, first_full_run_end_h=None, share_first_full_of_final=None,
            first_own_run=None, first_own_run_lift=None, first_own_run_end_h=None, share_first_own_of_final=None,
            final_twin_run=None, final_twin_lift=None, final_twin_end_h=None,
            best_research_run=top and top['run'], best_research_end_h=top and top['end_h'],
            best_research_lift=top and top['lift'], best_research_ci95=top and top['ci95'],
            final_lift=final_lift and round(final_lift, 2),
            final_ci95=final_value and [round(x, 2) for x in final_value[1]],
            final_lift_combined=reference and round(reference, 2),
            final_recipe_runs=[r['run'] for r in recipe_runs],
            final_recipe_research_lift=[r['lift'] for r in recipe_runs],
            share_first_of_final=round(first['lift'] / final_lift, 2) if first and final_lift and final_lift > 0 else None,
            share_first_of_final_unstable=bool(final_lift is not None and final_lift < 5),
            first_minus_final=round(first['lift'] - final_lift, 2) if first and final_lift is not None else None,
            best_minus_first=round(top['lift'] - first['lift'], 2) if first else None,
            curve=curve)
        # first run whose worker trained >= 25 min (skips smoke/probe runs); the recipe's research twin is the
        # longest run with exactly the final recipe's files (the fresh final trains for 1 h)
        full = [r for r in scored if r['end_h'] - r['start_h'] >= SUBSTANTIAL_H]
        if full:
            summary.update(first_full_run=full[0]['run'], first_full_run_lift=full[0]['lift'],
                           first_full_run_end_h=full[0]['end_h'],
                           share_first_full_of_final=round(full[0]['lift'] / final_lift, 2)
                           if final_lift and final_lift > 0 else None)
        own = [r for r in full if r['recipe_vs_starter'] != 'starter']
        if own:
            summary.update(first_own_run=own[0]['run'], first_own_run_lift=own[0]['lift'],
                           first_own_run_end_h=own[0]['end_h'],
                           share_first_own_of_final=round(own[0]['lift'] / final_lift, 2)
                           if final_lift and final_lift > 0 else None)
        summary['code_changed_any_run'] = any(r['recipe_vs_starter'] == 'modified' for r in rows)
        twins = sorted([r for r in scored if r['is_final_recipe']], key=lambda r: (-(r['end_h'] - r['start_h']), r['end_h']))
        if twins:
            summary.update(final_twin_run=twins[0]['run'], final_twin_lift=twins[0]['lift'],
                           final_twin_end_h=twins[0]['end_h'])
        attempts.append(summary)

    out.mkdir(parents=True, exist_ok=True)
    fields = ['model', 'label', 'attempt', 'entrant', 'run', 'job_id', 'requested_seconds', 'start_h', 'end_h',
              'allocation_before_h', 'lift', 'ci_lo', 'ci_hi', 'se', 'games', 'practice_score', 'is_final_recipe', 'recipe_vs_starter',
              'scored', 'skipped']
    with (out / 'bundles.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in table:
            writer.writerow(dict(r, ci_lo=r['ci95'] and r['ci95'][0], ci_hi=r['ci95'] and r['ci95'][1]))
    stats = dict(within_95=sum(s['within_95'] for s in sanity), checked=len(sanity),
                 mean_diff=round(float(np.mean([s['diff'] for s in sanity])), 2) if sanity else None,
                 sd_diff=round(float(np.std([s['diff'] for s in sanity], ddof=1)), 2) if len(sanity) > 1 else None,
                 expected_sd_diff=round(float(np.sqrt(2) * np.mean([lift(pairs, a['final']['entrant'], anchors, rng)[2]
                                                                  for a in manifest['attempts']
                                                                  if (a['final']['entrant'], anchors[0]) in pairs])), 2)
                 if sanity else None,
                 correlation=round(float(np.corrcoef([s['here'] for s in sanity],
                                                     [s['combined'] for s in sanity])[0, 1]), 3) if len(sanity) > 2 else None)
    value = dict(method=__doc__, anchors=anchors, spend=spend, sanity=dict(stats, finals=sanity),
                 attempts=attempts, bundles=table)
    (out / 'curves.json').write_text(json.dumps(value, indent=2) + '\n')
    figure(attempts, table, out / 'curves.png')
    page(value, out / 'index.html')
    print(json.dumps(dict(sanity=stats, spend={k: v for k, v in spend.items() if k != 'parts'},
                          attempts=[{k: s[k] for k in ('model', 'attempt', 'first_run_lift', 'best_research_lift',
                                                       'final_lift', 'share_first_of_final')} for s in attempts]),
                     indent=1))


def figure(attempts, table, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = list(dict.fromkeys(a['model'] for a in attempts))
    cols = 4
    fig, axes = plt.subplots(2, cols, figsize=(16, 7.6), sharex=True, sharey=True)
    fig.patch.set_facecolor('#fbf6ec')
    lows = [r['lift'] for r in table if r.get('lift') is not None] + [a['final_lift'] for a in attempts if a['final_lift'] is not None]
    ymin, ymax = min(lows) - 4, max(lows) + 4
    for ax in axes.flat:
        ax.set_facecolor('#fbf6ec')
    for i, model in enumerate(models):
        ax = axes.flat[i]
        ax.axhline(0, color='#888', lw=0.8, ls='--')
        for j, a in enumerate([a for a in attempts if a['model'] == model]):
            color = COLORS[j % 3]
            pts = [r for r in table if r['model'] == model and r['attempt'] == a['attempt'] and r.get('lift') is not None]
            ax.scatter([r['end_h'] for r in pts], [r['lift'] for r in pts], s=14, color=color, alpha=.35, lw=0)
            for r in pts:
                if r['is_final_recipe']:
                    ax.scatter([r['end_h']], [r['lift']], s=60, facecolor='none', edgecolor=color, lw=1.4)
            xs, ys = [], []
            for c in a['curve']:
                xs.append(c['end_h']); ys.append(c['best_so_far'])
            if xs:
                ax.step(xs, ys, where='post', color=color, lw=2, label=f"{a['attempt']}")
            if a['final_lift'] is not None:
                ax.errorbar([5.35 + 0.12 * j], [a['final_lift']], yerr=[[a['final_lift'] - a['final_ci95'][0]],
                            [a['final_ci95'][1] - a['final_lift']]], fmt='D', color=color, ms=6, mec=FINAL_COLOR,
                            mew=1.2, elinewidth=1, capsize=0)
        ax.axvline(5.15, color='#bbb', lw=0.6)
        ax.text(5.45, ymax - 1.2, 'final', ha='center', va='top', fontsize=8, color='#666')
        ax.set_title(LABELS[model], fontsize=11, loc='left', color='#222')
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, 5.75)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8, loc='lower right', frameon=False, title='attempt', title_fontsize=8)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0.02, 0.95,
                       'Lines: best-so-far lift of the attempt\'s finished research\n'
                       'runs vs session hours (run end).\n'
                       'Dots: each research bundle; ring: run whose files\n'
                       'equal the submitted final recipe.\n'
                       'Diamonds right of the rule: lift of the fresh 1 h\n'
                       'final retrain (95 % CI), same seed bank.\n\n'
                       'Lift = 100 x mean points vs 3 starter retrains - 50;\n'
                       '128 private seeds x both seats per anchor.',
                       va='top', fontsize=9, color='#333', transform=axes.flat[-1].transAxes)
    for ax in axes[1]:
        ax.set_xlabel('hours since session start')
    for ax in axes[:, 0]:
        ax.set_ylabel('lift over starter')
    fig.suptitle('Pokémon 3v3: research-network strength over each 4 h session (best so far) and the fresh final',
                 x=0.01, ha='left', fontsize=13, color='#222')
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def fmt(x, digits=1):
    return '—' if x is None else f'{x:+.{digits}f}'


def page(value, path):
    e = html.escape
    rows = []
    for a in value['attempts']:
        rows.append('<tr>' + ''.join(f'<td>{c}</td>' for c in (
            e(a['label']), a['attempt'], a['research_bundles'],
            f"{fmt(a['first_run_lift'])} <small>{e(a['first_run'] or '')} @{a['first_run_end_h']:.1f} h</small>" if a['first_run'] else '—',
            f"{fmt(a['first_own_run_lift'])} <small>{e(a['first_own_run'] or '')} @{a['first_own_run_end_h']:.1f} h</small>" if a['first_own_run'] else '—',
            f"{fmt(a['best_research_lift'])} <small>{e(a['best_research_run'] or '')} @{a['best_research_end_h']:.1f} h</small>" if a['best_research_run'] else '—',
            f"{fmt(a['final_lift'])} <small>[{fmt(a['final_ci95'][0])}, {fmt(a['final_ci95'][1])}]</small>" if a['final_ci95'] else '—',
            fmt(a['final_lift_combined']),
            f"{e(a['final_twin_run'])} {fmt(a['final_twin_lift'])}" if a['final_twin_run'] else '— (no run with identical files)',
            '—' if a['share_first_own_of_final'] is None else f"{a['share_first_own_of_final']:.2f}" + (' *' if a['share_first_of_final_unstable'] else ''),
        )) + '</tr>')
    sanity = value['sanity']
    spend = value['spend']
    body = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pokémon research curves</title>
<style>body{{background:#fbf6ec;color:#222;font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1150px;padding:16px}}
img{{max-width:100%}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{border-bottom:1px solid #e3d9c6;padding:4px 6px;text-align:left}}
small{{color:#777}}.wrap{{overflow-x:auto}}</style></head><body>
<h1>Pokémon 3v3 research curves (2026-11-15)</h1>
<p>Every verified research-run bundle of the 21 counted attempts, plus each attempt's fresh final, scored against the three
starter retrains (s1r, s2, s3r): 128 private seeds × both seats per anchor, one shared seed bank. Lift = 100 × mean
points vs the anchors − 50; intervals bootstrap seeds only.</p>
<p>Games: {spend['games']:,}; worker time {spend['worker_seconds']:,.0f} s; CPU cost ${spend['usd']:.2f};
faults {spend['faults']}, truncations {spend['truncations']}. Finals sanity check vs <code>analysis-combined.json</code>:
{sanity['within_95']}/{sanity['checked']} within the 95 % band, mean difference {fmt(sanity['mean_diff'])},
SD {sanity['sd_diff']} (expected ≈ {sanity['expected_sd_diff']}), correlation {sanity['correlation']}.</p>
<img src="curves.png" alt="Best-so-far lift per attempt over session hours, one panel per model">
<h2>Per attempt</h2><div class="wrap"><table><tr><th>Model</th><th>Attempt</th><th>Bundles</th><th>First run</th><th>First own run ≥ 25 min</th><th>Best research run</th>
<th>Final (this bank)</th><th>Final (combined)</th><th>Research twin of final recipe</th><th>First own / final</th></tr>
{''.join(rows)}</table></div>
<p><small>"Own run": the first research run that trained ≥ 25 min and is not the release's exact starter recipe
(Astra a1/a2, GLM a1–a3, Luna a2 and Inkling b3 began with a starter baseline; Luna never changed the starter code,
only its train config). Best-of-N research lifts are biased upward by selection on noisy estimates (SE ≈ 2 per bundle).
— : final lift ≤ 0, no ratio. * final lift below +5: the ratio is unstable. Data: curves.json, bundles.csv.</small></p>
</body></html>"""
    path.write_text(body)
