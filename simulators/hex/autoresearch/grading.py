"""Seat-paired scoring against the panel, with Elo against the MoHex anchor.

Without the swap rule every 2-ply opening is a theoretical win for exactly one side, so
each opening is played from both seats. Against a perfect opponent that would cap the
win rate at 50%; the 13x13 anchor is far from perfect, so a candidate can beat it from
both seats of an opening and the anchor's 0 Elo is a reference point, not a ceiling.
"""
import math

import numpy as np

from .contract import ANCHOR, DEFAULT, PANEL

SEATS = ('black', 'white')


def by_opening(rows):
    """{opening_index: mean win over every opponent and both seats}; requires the complete schedule."""
    if not rows:
        raise ValueError('No evaluation games')
    keys = set()
    opponents = {r['opponent'] for r in rows}
    openings = {r['opening_index'] for r in rows}
    result = {}
    for r in rows:
        key = (r['opponent'], r['opening_index'], r['seat'])
        if key in keys or r['seat'] not in SEATS or not r['natural'] or r['truncated'] or r['won'] not in (False, True):
            raise ValueError('Invalid or duplicate evaluation row')
        keys.add(key)
        result.setdefault(r['opening_index'], []).append(float(r['won']))
    if len(keys) != len(opponents) * len(openings) * 2:
        raise ValueError('Incomplete opponent/opening/seat schedule')
    return {i: float(np.mean(v)) for i, v in result.items()}


def elo(p, n):
    """Elo difference implied by win rate p over n games, with p kept 1/(2n) away from 0 and 1."""
    p = min(max(p, 0.5 / n), 1 - 0.5 / n)
    return 400 * math.log10(p / (1 - p))


def summary(rows, replicates=None):
    replicates = replicates or DEFAULT['bootstrap_replicates']
    values = by_opening(rows)
    v = np.array([values[i] for i in sorted(values)])
    rng = np.random.default_rng(83)
    picks = rng.integers(len(v), size=(replicates, len(v)))
    boots = v[picks].mean(1)
    opponents = sorted({r['opponent'] for r in rows})
    per_opponent, per_seat = {}, {}
    for name in opponents:
        mine = [r for r in rows if r['opponent'] == name]
        per_opponent[name] = float(np.mean([r['won'] for r in mine]))
        per_seat[name] = {s: float(np.mean([r['won'] for r in mine if r['seat'] == s])) for s in SEATS}
    result = {
        'raw_score': float(v.mean() * 100), 'score_scale': [0, 100],
        'raw_score_ci95': (np.quantile(boots, [.025, .975]) * 100).tolist(),
        'per_opponent': per_opponent, 'per_seat': per_seat,
        'games': len(rows), 'unique_openings': len(v), 'natural_games': len(rows),
        'truncations': 0, 'illegal_actions': 0, 'simulator_faults': 0,
        'mean_plies': float(np.mean([r['plies'] for r in rows])),
        'aggregation': 'equal weight per opponent; each opening played from both seats; '
                       'bootstrap clustered by opening',
    }
    rated = panel_elo(rows, rng, replicates)
    if rated:
        result['panel_elo'] = rated
    if ANCHOR in per_opponent:
        anchor_rows = [r for r in rows if r['opponent'] == ANCHOR]
        wins = {}
        for r in anchor_rows:
            wins.setdefault(r['opening_index'], []).append(r['won'])
        paired = np.array([np.mean(w) for _, w in sorted(wins.items())])
        n = len(anchor_rows)
        anchor_boots = paired[rng.integers(len(paired), size=(replicates, len(paired)))].mean(1)
        result['anchor'] = {
            'opponent': ANCHOR, 'win_rate': per_opponent[ANCHOR], 'games': n,
            'elo_vs_anchor': elo(per_opponent[ANCHOR], n),
            'elo_vs_anchor_ci95': [elo(float(x), n) for x in np.quantile(anchor_boots, [.025, .975])],
            'openings_won_both_seats': int(sum(sum(w) == 2 for w in wins.values())),
            'openings_won_one_seat': int(sum(sum(w) == 1 for w in wins.values())),
            'openings_won_neither_seat': int(sum(sum(w) == 0 for w in wins.values())),
            'note': 'The anchor is MoHex at 50,000 games per move: 0 Elo by definition, not perfect play. '
                    'Winning an opening from both seats means the anchor erred on it.',
        }
    return result


LOW, HIGH = -3000.0, 1000.0


def _fit_elo(wins, games, ratings, iterations=60):
    """Maximum-likelihood Elo against opponents of known rating, vectorized over rows of (wins, games)."""
    c = np.log(10) / 400
    x = np.full(wins.shape[0], float(np.average(ratings)))
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-c * (x[:, None] - ratings[None, :])))
        gradient = c * (wins - games * p).sum(1)
        information = c * c * (games * p * (1 - p)).sum(1)
        x = np.clip(x + gradient / np.maximum(information, 1e-12), LOW, HIGH)
    return x, information


def panel_elo(rows, rng=None, replicates=None):
    """The candidate's Elo relative to the anchor, fitted against the calibrated panel ratings.

    Opponents without a calibrated rating are left out. The interval resamples whole
    openings, like the score's. At LOW or HIGH the candidate lost or won nearly everything,
    and the value is a bound, not an estimate.
    """
    known = {p['name']: p['elo'] for p in PANEL if 'elo' in p}
    names = sorted({r['opponent'] for r in rows} & set(known))
    if not names:
        return None
    openings = sorted({r['opening_index'] for r in rows})
    column, row = {n: k for k, n in enumerate(names)}, {o: k for k, o in enumerate(openings)}
    wins = np.zeros((len(openings), len(names)))
    games = np.zeros((len(openings), len(names)))
    for r in rows:
        if r['opponent'] in column:
            games[row[r['opening_index']], column[r['opponent']]] += 1
            wins[row[r['opening_index']], column[r['opponent']]] += r['won']
    ratings = np.array([known[n] for n in names], dtype=float)
    estimate, information = _fit_elo(wins.sum(0, keepdims=True), games.sum(0, keepdims=True), ratings)
    rng = rng or np.random.default_rng(83)
    picks = rng.integers(len(openings), size=(replicates or DEFAULT['bootstrap_replicates'], len(openings)))
    boots, _ = _fit_elo(wins[picks].sum(1), games[picks].sum(1), ratings)
    value = float(estimate[0])
    return {'elo': value, 'ci95': np.quantile(boots, [.025, .975]).tolist(), 'se': float(1 / np.sqrt(information[0])),
            'at_bound': value in (LOW, HIGH), 'opponents': names,
            'note': 'Relative to mohex-anchor (0). Random play is about -2600 on 13x13. Panel ratings carry '
                    'about +-50 Elo of their own uncertainty.'}


def grade(rows, *, development=False):
    """The grade is the raw panel score; there is no starter/reference normalization in this contract."""
    return {'status': 'development_only' if development else 'complete', 'reward': None,
            'normalized_quality': None, **summary(rows)}
