#!/usr/bin/env python3
"""Analyze host-graded complete paired recipe panels with crossed bootstrap.

Input: JSON array, or {"rows": [...]}. Each row has family, train_seed,
eval_unit (opaque map/deal/lobby-group index), opponent, seat_or_role, score
(0..100), valid:true, and optional replicate (default 0). Eval units must cover
the same full opponent/seat/replicate schedule for both recipes and every seed.
Private evaluation seeds belong in ignored input evidence, never public output.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from statistics import mean


def quantile(values, probability):
    values = sorted(values)
    position = (len(values)-1)*probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper]-values[lower])*(position-lower)


def analyze(rows, baseline='starter', candidate='agent', replicates=4000, bootstrap_seed=92317):
    if baseline == candidate or not rows or replicates < 100:
        raise ValueError('Two families, nonempty rows and at least 100 bootstrap replicates required')
    indexed = {baseline: {}, candidate: {}}
    for row in rows:
        if row.get('valid') is not True:
            raise ValueError('Invalid or unverified grading row; no paired result')
        family = row.get('family')
        if family not in indexed:
            raise ValueError('Unexpected recipe family')
        score = row.get('score')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError('Finite score on 0–100 scale required')
        for field in ('train_seed', 'eval_unit', 'opponent', 'seat_or_role'):
            if field not in row:
                raise ValueError(f'Missing pairing field: {field}')
        key = tuple(str(row[f]) for f in ('train_seed', 'eval_unit', 'opponent', 'seat_or_role')) + (str(row.get('replicate', 0)),)
        if key in indexed[family]:
            raise ValueError('Duplicate evaluation case')
        indexed[family][key] = float(score)
    if set(indexed[baseline]) != set(indexed[candidate]):
        raise ValueError('Incomplete paired schedule; missing recipe/seed/unit/opponent/seat case')
    keys = set(indexed[baseline])
    seeds = sorted({k[0] for k in keys})
    units_by_seed = {s: {k[1] for k in keys if k[0] == s} for s in seeds}
    units = sorted(units_by_seed[seeds[0]])
    if any(u != set(units) for u in units_by_seed.values()):
        raise ValueError('All training seeds must use the same evaluation-unit schedule')
    cells = {(s, u): {k[2:] for k in keys if k[:2] == (s, u)} for s in seeds for u in units}
    schedule = cells[seeds[0], units[0]]
    if any(c != schedule for c in cells.values()):
        raise ValueError('Incomplete or inconsistent opponent/seat/replicate schedule')
    strata = defaultdict(set)
    for opponent, seat, replicate in schedule:
        strata[opponent, seat].add(replicate)
    # Each opponent/seat stratum has equal weight; replicates within it share weight.
    def cell_value(family, seed, unit, opponent=None, seat=None):
        selected = [st for st in strata if (opponent is None or st[0] == opponent) and (seat is None or st[1] == seat)]
        return mean(mean(indexed[family][seed, unit, *st, r] for r in strata[st]) for st in selected)
    matrices = {family: [[cell_value(family, s, u) for u in units] for s in seeds] for family in indexed}
    deltas = [[b-a for a, b in zip(ar, br)] for ar, br in zip(matrices[baseline], matrices[candidate])]
    rng = random.Random(bootstrap_seed)
    boots = []
    for _ in range(replicates):
        ss = [rng.randrange(len(seeds)) for _ in seeds]
        us = [rng.randrange(len(units)) for _ in units]
        # Shared unit draw preserves common map/deal difficulty across seeds.
        # Arms, seats/roles, opponents and replicates remain paired within units.
        boots.append(mean(deltas[s][u] for s in ss for u in us))
    result = {
        'baseline_family': baseline, 'candidate_family': candidate,
        'score_scale': [0, 100], 'training_seeds': len(seeds), 'evaluation_units': len(units),
        'games_per_family': len(keys), 'complete_paired_schedule': True,
        'baseline_mean': mean(mean(r) for r in matrices[baseline]),
        'candidate_mean': mean(mean(r) for r in matrices[candidate]),
        'paired_delta_points': mean(mean(r) for r in deltas),
        'paired_delta_ci95': [quantile(boots, .025), quantile(boots, .975)],
        'bootstrap_positive_fraction': sum(x > 0 for x in boots)/replicates,
        'bootstrap_method': 'crossed paired resampling of training seeds and complete evaluation units; fixed opponent panel',
        'bootstrap_replicates': replicates, 'bootstrap_seed': bootstrap_seed,
        'per_training_seed': [{'train_seed': seed,
            'baseline_mean': mean(matrices[baseline][i]), 'candidate_mean': mean(matrices[candidate][i]),
            'paired_delta_points': mean(deltas[i])} for i, seed in enumerate(seeds)],
        'per_opponent_delta_points': {op: mean(cell_value(candidate,s,u,opponent=op)-cell_value(baseline,s,u,opponent=op) for s in seeds for u in units) for op in sorted({k[0] for k in strata})},
        'per_seat_or_role_delta_points': {seat: mean(cell_value(candidate,s,u,seat=seat)-cell_value(baseline,s,u,seat=seat) for s in seeds for u in units) for seat in sorted({k[1] for k in strata})},
        'reference_normalized_score': None,
        'limitations': ['Conditional on this fixed development panel and compute allowance; not a calibrated model ranking.',
                        'Few training seeds and evaluation units limit bootstrap reliability; positive fraction is not a p-value.']}
    if len(seeds) < 3:
        result['limitations'].append('Fewer than three training seeds: descriptive pilot only.')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path); p.add_argument('--out', type=Path)
    p.add_argument('--baseline', default='starter'); p.add_argument('--candidate', default='agent')
    p.add_argument('--replicates', type=int, default=4000)
    a = p.parse_args()
    data = json.loads(a.input.read_text())
    result = analyze(data['rows'] if isinstance(data, dict) else data, a.baseline, a.candidate, a.replicates)
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(text)
    print(text, end='')


if __name__ == '__main__': main()
