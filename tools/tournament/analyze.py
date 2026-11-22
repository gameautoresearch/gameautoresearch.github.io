"""Lift over the starter, with every source of luck in the interval, from finished two-player tournaments.

    python -m tools.tournament.analyze --result TOURNAMENT_DIR [--result ...] --out ANALYSIS.json

Each entrant is tagged from its session binding (`<session>.harness-run.json`):

- model: the binding's `model` (a `harness retrain` records its source session's model, or `starter`);
- attempt: the research session the recipe came from (a retrain of a session counts as that session's
  attempt; every starter retrain is the single starter "attempt");
- seed: the retrain seed, or `final` for a session's own fresh final.

`--tag NAME=MODEL/ATTEMPT/SEED` overrides that for one entrant (tests, bundles).

Primary metric per model and game: **lift** = 100 x (mean points against the starter anchors) - 50,
averaged over the model's attempts, each attempt over its seeds, each seed over the anchors. Its 95 %
interval comes from a hierarchical bootstrap that resamples, in every replicate: each model's attempts,
the seeds within each attempt, the starter anchors, and the evaluation units (maps / deals / seeds, the same
units for every pair). Differences between models use the same replicates, so they are paired.

Also reported: each model's score against the whole field, the per-game variance components (seed,
attempt, evaluation) and, from them, the smallest attempts x seeds design that detects a target gap.
Because a mean hides a model whose attempts sometimes collapse, each model's row also carries the lift of
every attempt and the shape of that distribution: median, best, worst, the expected best of two and of
three attempts, and the share of attempts above the starter (median and share with bootstrap intervals).
Host-only, no cloud, no entrant code.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np

sys.dont_write_bytecode = True

STARTER = 'starter'
UNIT_KEYS = ('unit', 'deal_index', 'map_index', 'seed_index')
Z95, Z80 = 1.959964, 0.841621  # two-sided 95 % test, 80 % power


def tag_from_binding(path):
    binding = Path(path).with_name(Path(path).name + '.harness-run.json')
    settings = json.loads(binding.read_text())['settings']
    model = settings.get('model') or 'unknown'
    if settings.get('kind') == 'retrain':
        source = settings.get('source') or {}
        attempt = STARTER if source.get('kind') == 'starter' else source.get('session')
        model = STARTER if source.get('kind') == 'starter' else model
        seed = str(settings.get('seed'))
    else:
        attempt, seed = str(Path(path).resolve()), 'final'
    return dict(model=model, attempt=str(attempt), seed=seed)


def load_rows(pair_dir):
    """[(unit, a_points)] for one pair; unscored games (faults, unplayed) are dropped."""
    pair_dir = Path(pair_dir)
    if (pair_dir / 'rows.json').is_file():
        raw = json.loads((pair_dir / 'rows.json').read_text())
    elif (pair_dir / 'games.jsonl').is_file():
        raw = [json.loads(line) for line in (pair_dir / 'games.jsonl').read_text().splitlines() if line.strip()]
    else:
        raise FileNotFoundError(f'No per-game rows in {pair_dir}')
    rows = []
    for row in raw:
        if row.get('a_points') is None:
            continue
        key = next((k for k in UNIT_KEYS if k in row), None)
        if key is None:
            raise ValueError(f'Row without an evaluation unit in {pair_dir}')
        rows.append((int(row[key]), float(row['a_points'])))
    return rows


def load_game(result_dir, overrides):
    """One tournament: entrant tags, unit ids, and per-(entrant, opponent) [units] -> (points sum, games)."""
    result_dir = Path(result_dir)
    result = json.loads((result_dir / 'result.json').read_text())
    tags = {}
    for name, entrant in result['entrants'].items():
        if name in overrides:
            tags[name] = overrides[name]
        elif entrant.get('session'):
            tags[name] = tag_from_binding(entrant['session'])
        else:
            raise ValueError(f'Entrant {name} has no session; tag it with --tag')
    names = sorted(tags)
    index = {name: i for i, name in enumerate(names)}
    units = set()
    pairs = []
    for pair in result['pairs']:
        details = Path(pair['details'])
        details = details if details.is_absolute() else result_dir / details
        rows = load_rows(details)
        units.update(u for u, _ in rows)
        pairs.append((pair['a'], pair['b'], rows))
    units = sorted(units)
    position = {u: i for i, u in enumerate(units)}
    # points[i, j, u] = entrant i's points against entrant j on unit u; games[i, j, u] the game count.
    points = np.zeros((len(names), len(names), len(units)))
    games = np.zeros_like(points)
    for a, b, rows in pairs:
        for unit, a_points in rows:
            u = position[unit]
            points[index[a], index[b], u] += a_points
            games[index[a], index[b], u] += 1
            points[index[b], index[a], u] += 1 - a_points
            games[index[b], index[a], u] += 1
    return dict(game=result['game'], release=result.get('release'), names=names, tags=tags, units=units,
                points=points, games=games, smoke=bool(result.get('smoke')))


def add_forfeits(data, forfeits):
    """An attempt whose recipe produced no network (its fresh training failed) loses every game to every anchor.

    forfeits: [(model, attempt)]. Each becomes one synthetic entrant with 0 points against each anchor on every
    evaluation unit, so the attempt counts in its model's mean instead of silently dropping out.
    """
    if not forfeits:
        return data
    names = list(data['names'])
    anchors = [i for i, n in enumerate(names) if data['tags'][n]['model'] == STARTER]
    k = len(names)
    n = k + len(forfeits)
    points = np.zeros((n, n, len(data['units'])))
    games = np.zeros_like(points)
    points[:k, :k], games[:k, :k] = data['points'], data['games']
    tags = dict(data['tags'])
    for j, (model, attempt) in enumerate(forfeits):
        i = k + j
        name = f'forfeit-{model}-{attempt}'[:40]
        names.append(name)
        tags[name] = dict(model=model, attempt=attempt, seed='forfeit')
        for a in anchors:
            games[i, a, :] = games[a, i, :] = 1
            points[a, i, :] = 1  # the anchor wins every game; the forfeit scores 0
    return dict(data, names=names, tags=tags, points=points, games=games)


def structure(data):
    """{model: {attempt: [entrant indices]}} and the starter anchors' indices."""
    models = {}
    for i, name in enumerate(data['names']):
        tag = data['tags'][name]
        models.setdefault(tag['model'], {}).setdefault(tag['attempt'], []).append(i)
    anchors = [i for attempts in models.get(STARTER, {}).values() for i in attempts]
    return models, anchors


def entrant_scores(points, games, unit_weights, opponents):
    """Each entrant's mean points (0-1) against each opponent group, units weighted by bootstrap counts."""
    p = (points[:, opponents, :] * unit_weights).sum(axis=2)
    g = (games[:, opponents, :] * unit_weights).sum(axis=2)
    with np.errstate(invalid='ignore', divide='ignore'):
        return p / g  # NaN where no games (self, or a unit draw with no games)


def model_lifts(data, models, anchors, rng=None):
    """One replicate (or the point estimate when rng is None): {model: lift}, {model: field score} and
    {model: [each attempt's lift]} (attempts as resampled in this replicate)."""
    n_units = len(data['units'])
    if rng is None:
        weights = np.ones(n_units)
        anchor_pick = anchors
    else:
        weights = np.bincount(rng.integers(n_units, size=n_units), minlength=n_units).astype(float)
        anchor_pick = [anchors[k] for k in rng.integers(len(anchors), size=len(anchors))] if anchors else []
    lifts, field, spread = {}, {}, {}
    vs_anchor = entrant_scores(data['points'], data['games'], weights, anchor_pick) if anchor_pick else None
    everyone = list(range(len(data['names'])))
    vs_field_raw = entrant_scores(data['points'], data['games'], weights, everyone)
    for model, attempts in models.items():
        keys = list(attempts)
        if rng is not None:
            keys = [keys[k] for k in rng.integers(len(keys), size=len(keys))]
        attempt_lift, attempt_field = [], []
        for key in keys:
            seeds = attempts[key]
            if rng is not None:
                seeds = [seeds[k] for k in rng.integers(len(seeds), size=len(seeds))]
            if vs_anchor is not None and model != STARTER:  # the anchors are the zero point, not a result
                # An anchor never plays itself: its own column is NaN and drops out of the mean.
                attempt_lift.append(np.nanmean([np.nanmean(vs_anchor[s]) for s in seeds]))
            own = set(i for a in attempts.values() for i in a)
            others = [j for j in everyone if j not in own]
            attempt_field.append(np.nanmean([np.nanmean(vs_field_raw[s][others]) for s in seeds]) if others else np.nan)
        if attempt_lift:
            lifts[model] = 100 * float(np.nanmean(attempt_lift)) - 50
            spread[model] = [100 * float(value) - 50 for value in attempt_lift]
        field[model] = 100 * float(np.nanmean(attempt_field))
    return lifts, field, spread


def best_of(values, k):
    """Expected best lift among k of these attempts drawn without replacement; None with fewer than k."""
    values = sorted(v for v in values if not math.isnan(v))
    n = len(values)
    if n < k:
        return None
    # The i-th smallest value is the maximum of comb(i, k - 1) of the comb(n, k) equally likely draws.
    return float(sum(math.comb(i, k - 1) * value for i, value in enumerate(values)) / math.comb(n, k))


def distribution(values):
    """How one model's attempts are spread: the mean alone hides a model that sometimes collapses."""
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return None
    return dict(median_lift=float(np.median(values)), best_lift=float(max(values)), worst_lift=float(min(values)),
                share_above_starter=float(np.mean(np.array(values) > 0)),
                best_of_2=best_of(values, 2), best_of_3=best_of(values, 3))


def components(data, models, anchors):
    """Variance components of an entrant's score against the anchors, in 0-100 points.

    seed: pooled within-attempt variance across seeds; attempt: between-attempt variance of attempt means
    beyond what seed noise explains; evaluation: mean bootstrap variance of a single entrant's score from the
    evaluation units alone.
    """
    if not anchors:
        return None
    weights = np.ones(len(data['units']))
    score = 100 * np.nanmean(entrant_scores(data['points'], data['games'], weights, anchors), axis=1)
    within, within_df, between = 0.0, 0, []
    for model, attempts in models.items():
        means = []
        for seeds in attempts.values():
            values = score[seeds]
            if len(values) > 1:
                within += float(((values - values.mean()) ** 2).sum())
                within_df += len(values) - 1
            means.append((float(values.mean()), len(values)))
        if model != STARTER and len(means) > 1:
            between.append(means)
    seed_var = within / within_df if within_df else None
    attempt_var = None
    if between:
        estimates = []
        for means in between:
            values = np.array([m for m, _ in means])
            k = np.mean([n for _, n in means])
            estimates.append(float(values.var(ddof=1)) - (seed_var or 0.0) / k)
        attempt_var = max(0.0, float(np.mean(estimates)))
    rng = np.random.default_rng(7)
    n = len(data['units'])
    boot = []
    for _ in range(400):
        w = np.bincount(rng.integers(n, size=n), minlength=n).astype(float)
        boot.append(100 * np.nanmean(entrant_scores(data['points'], data['games'], w, anchors), axis=1))
    eval_var = float(np.nanmean(np.nanvar(np.array(boot), axis=0, ddof=1)))
    return dict(seed_sd=None if seed_var is None else math.sqrt(seed_var), seed_df=within_df,
                attempt_sd=None if attempt_var is None else math.sqrt(attempt_var),
                attempt_models=len(between), evaluation_sd=math.sqrt(eval_var))


def design_table(sd, target_gap, max_attempts=6, max_seeds=3, assume_attempt_sd=None):
    """Detectable difference between two models' lifts (95 % two-sided, 80 % power) per attempts x seeds."""
    seed_sd, eval_sd = sd['seed_sd'] or 0.0, sd['evaluation_sd']
    attempt_sd = sd['attempt_sd'] if sd['attempt_sd'] is not None else assume_attempt_sd
    rows = []
    for attempts in range(1, max_attempts + 1):
        for seeds in range(1, max_seeds + 1):
            var = ((attempt_sd or 0.0) ** 2 + (seed_sd ** 2 + eval_sd ** 2) / seeds) / attempts
            detectable = (Z95 + Z80) * math.sqrt(2 * var)
            rows.append(dict(attempts=attempts, seeds=seeds, retrains=attempts * seeds,
                             detectable_gap=detectable, meets_target=detectable <= target_gap))
    feasible = [r for r in rows if r['meets_target']]
    best = min(feasible, key=lambda r: (r['retrains'], r['attempts'])) if feasible else None
    return dict(target_gap=target_gap, attempt_sd_used=attempt_sd,
                attempt_sd_assumed=sd['attempt_sd'] is None, rows=rows, smallest_design=best)


def analyze(results, overrides=None, replicates=2000, seed=20261112, target_gap=10.0, assume_attempt_sd=None,
            forfeits=None):
    out = dict(method=__doc__.split('\n\n', 2)[2].strip(), replicates=replicates, games={})
    overall = {}
    for result_dir in results:
        data = load_game(result_dir, overrides or {})
        data = add_forfeits(data, [(m, a) for g, m, a in (forfeits or []) if g == data['game']])
        models, anchors = structure(data)
        point_lift, point_field, point_spread = model_lifts(data, models, anchors)
        rng = np.random.default_rng(seed)
        boot_lift = {m: [] for m in point_lift}
        boot_field = {m: [] for m in point_field}
        boot_median = {m: [] for m in point_lift}
        boot_share = {m: [] for m in point_lift}
        diffs = {}
        for _ in range(replicates):
            with warnings.catch_warnings():  # a replicate can draw an anchor set that only contains itself
                warnings.simplefilter('ignore', RuntimeWarning)
                lift, field, spread = model_lifts(data, models, anchors, rng)
            for m, v in lift.items():
                boot_lift[m].append(v)
                boot_median[m].append(float(np.nanmedian(spread[m])))
                boot_share[m].append(float(np.mean(np.array(spread[m]) > 0)))
            for m, v in field.items():
                boot_field[m].append(v)
            ranked = sorted(m for m in lift if m != STARTER)
            for i, a in enumerate(ranked):
                for b in ranked[i + 1:]:
                    diffs.setdefault((a, b), []).append(lift[a] - lift[b])

        def interval(values):
            values = np.array([v for v in values if not math.isnan(v)])
            return [float(x) for x in np.quantile(values, [.025, .975])] if len(values) else None
        table = []
        for model in sorted(point_field, key=lambda m: -(point_lift.get(m, point_field[m]))):
            attempts = models[model]
            row = dict(model=model, attempts=len(attempts), retrains=sum(map(len, attempts.values())),
                       lift=point_lift.get(model), lift_ci95=interval(boot_lift.get(model, [])),
                       field_score=point_field[model], field_score_ci95=interval(boot_field[model]))
            if model in point_spread:
                # Per attempt, in the order the attempts were tagged; then the shape of that distribution.
                row.update(attempt_lifts=dict(zip(attempts, point_spread[model])), **distribution(point_spread[model]),
                           median_lift_ci95=interval(boot_median[model]),
                           share_above_starter_ci95=interval(boot_share[model]))
            table.append(row)
        comparisons = [dict(a=a, b=b, lift_difference=point_lift[a] - point_lift[b], ci95=interval(v),
                            share_a_ahead=float(np.mean(np.array(v) > 0))) for (a, b), v in sorted(diffs.items())]
        sd = components(data, models, anchors)
        game = dict(release=data['release'], smoke=data['smoke'], units=len(data['units']), anchors=len(anchors),
                    entrants={name: data['tags'][name] for name in data['names']}, models=table,
                    comparisons=comparisons, variance_components=sd,
                    design=design_table(sd, target_gap, assume_attempt_sd=assume_attempt_sd) if sd else None)
        out['games'][data['game']] = game
        for row in table:
            if row['lift'] is not None and row['model'] != STARTER:
                overall.setdefault(row['model'], {})[data['game']] = row['lift']
    games = list(out['games'])
    out['overall'] = sorted(
        [dict(model=m, mean_lift=float(np.mean(list(v.values()))), games=v) for m, v in overall.items()
         if set(v) == set(games)], key=lambda r: -r['mean_lift'])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m tools.tournament.analyze', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--result', action='append', required=True, help='a finished tournament directory')
    p.add_argument('--tag', action='append', default=[], metavar='NAME=MODEL/ATTEMPT/SEED')
    p.add_argument('--out', required=True, help='new JSON file')
    p.add_argument('--replicates', type=int, default=2000)
    p.add_argument('--target-gap', type=float, default=10.0, help='lift difference the design table targets')
    p.add_argument('--forfeit', action='append', default=[], metavar='GAME=MODEL::ATTEMPT',
                   help='an attempt whose recipe produced no network: scored 0 against every anchor')
    p.add_argument('--assume-attempt-sd', type=float,
                   help='attempt spread to assume when the tournament has no repeated attempts')
    args = p.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        p.error('--out must not exist')
    overrides = {}
    for item in args.tag:
        name, _, value = item.partition('=')
        model, attempt, seed = value.rsplit('/', 2)  # model names contain slashes
        overrides[name] = dict(model=model, attempt=attempt, seed=seed)
    forfeits = []
    for item in args.forfeit:
        game, _, value = item.partition('=')
        if '::' not in value:
            p.error('--forfeit takes GAME=MODEL::ATTEMPT (model names contain slashes)')
        model, attempt = value.split('::', 1)
        forfeits.append((game, model, attempt))
    value = analyze(args.result, overrides, args.replicates, target_gap=args.target_gap,
                    assume_attempt_sd=args.assume_attempt_sd, forfeits=forfeits)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, indent=2) + '\n')
    for game, item in value['games'].items():
        print(game)
        for row in item['models']:
            ci = row['lift_ci95']
            lift = 'n/a' if row['lift'] is None else f"{row['lift']:+.1f}"
            ci = '' if not ci else f" [{ci[0]:+.1f}, {ci[1]:+.1f}]"
            print(f"  {row['model']:<48} lift {lift}{ci}  field {row['field_score']:.1f}  "
                  f"attempts {row['attempts']} retrains {row['retrains']}")
        print('  variance components:', item['variance_components'])
        if item['design'] and item['design']['smallest_design']:
            print('  smallest design for the target gap:', item['design']['smallest_design'])


if __name__ == '__main__':
    main()
