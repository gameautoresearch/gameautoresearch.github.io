"""Lift-over-starter analysis on synthetic tournaments with known strengths and noise."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.tournament import analyze


def synthetic(out, entrants, units=400, seed=0, rows_file='rows.json'):
    """entrants: {name: (model, attempt, seed_label, strength)}; P(a wins) = 0.5 + strength_a - strength_b."""
    rng = np.random.default_rng(seed)
    out = Path(out)
    names = list(entrants)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p = min(.98, max(.02, .5 + entrants[a][3] - entrants[b][3]))
            rows = [dict(unit=u, seat=s, a_points=float(rng.random() < p)) for u in range(units) for s in (0, 1)]
            details = out / 'pairs' / f'{a}-vs-{b}'
            details.mkdir(parents=True)
            if rows_file == 'rows.json':
                (details / 'rows.json').write_text(json.dumps(rows))
            else:
                (details / 'games.jsonl').write_text(
                    '\n'.join(json.dumps(dict(deal_index=r['unit'], ordering=r['seat'], a_points=r['a_points']))
                              for r in rows))
            pairs.append(dict(a=a, b=b, details=f'pairs/{a}-vs-{b}'))
    (out / 'result.json').write_text(json.dumps(dict(game='testgame', release='r', smoke=False,
                                                     entrants={n: {} for n in names}, pairs=pairs)))
    return {n: dict(model=m, attempt=a, seed=s) for n, (m, a, s, _) in entrants.items()}


class Analyze(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_recovers_lift_seed_spread_and_ranks_models(self):
        rng = np.random.default_rng(1)
        entrants = {}
        for s in range(3):
            entrants[f'st{s}'] = ('starter', 'starter', str(s), rng.normal(0, .04))
        for model, base in (('good', .15), ('meh', .03)):
            for a in range(3):
                attempt = base + rng.normal(0, .03)
                for s in range(2):
                    entrants[f'{model}{a}{s}'] = (model, f'{model}-a{a}', str(s), attempt + rng.normal(0, .04))
        tags = synthetic(Path(self.temp.name) / 't', entrants)
        out = analyze.analyze([Path(self.temp.name) / 't'], tags, replicates=300)
        game = out['games']['testgame']
        lift = {row['model']: row for row in game['models']}
        self.assertIsNone(lift['starter']['lift'])  # the starter anchors define zero
        self.assertAlmostEqual(lift['starter']['field_score'], 50, delta=8)
        self.assertGreater(lift['good']['lift'], lift['meh']['lift'] + 5)
        low, high = lift['good']['lift_ci95']
        self.assertLess(low, lift['good']['lift'])
        self.assertGreater(high, lift['good']['lift'])
        self.assertGreater(high - low, 3, 'attempt and seed noise must widen the interval')
        comparison = next(c for c in game['comparisons'] if {c['a'], c['b']} == {'good', 'meh'})
        self.assertGreater(abs(comparison['lift_difference']), 5)
        sd = game['variance_components']
        self.assertTrue(1 < sd['seed_sd'] < 9, sd)
        self.assertLess(sd['evaluation_sd'], 3)
        self.assertIsNotNone(game['design']['smallest_design'])
        self.assertEqual([r['model'] for r in out['overall']][0], 'good')
        # The shape of each model's attempts, not only their mean.
        good = lift['good']
        self.assertEqual(sorted(good['attempt_lifts']), ['good-a0', 'good-a1', 'good-a2'])
        self.assertAlmostEqual(float(np.mean(list(good['attempt_lifts'].values()))), good['lift'], places=6)
        self.assertEqual(good['share_above_starter'], 1.0)
        self.assertTrue(good['worst_lift'] <= good['median_lift'] <= good['best_of_2'] <= good['best_lift'])
        self.assertEqual(good['best_of_3'], good['best_lift'])
        self.assertTrue(good['median_lift_ci95'][0] <= good['median_lift'] <= good['median_lift_ci95'][1])
        self.assertNotIn('attempt_lifts', lift['starter'])

    def test_distribution_shows_a_model_that_sometimes_collapses(self):
        steady, erratic = [12.0, 14.0, 13.0, 15.0], [20.0, 18.0, 21.0, -12.0]
        a, b = analyze.distribution(steady), analyze.distribution(erratic)
        self.assertAlmostEqual(float(np.mean(steady)), 13.5)
        self.assertLess(float(np.mean(erratic)), float(np.mean(steady)))  # the mean ranks the erratic model lower
        self.assertGreater(b['median_lift'], a['median_lift'])              # its typical attempt is better
        self.assertEqual((a['share_above_starter'], b['share_above_starter']), (1.0, 0.75))
        self.assertEqual((b['best_lift'], b['worst_lift']), (21.0, -12.0))
        self.assertAlmostEqual(analyze.best_of([1.0, 2.0, 3.0], 2), (2 + 3 + 3) / 3)
        self.assertIsNone(analyze.best_of([1.0], 2))
        self.assertIsNone(analyze.distribution([]))

    def test_a_forfeited_attempt_counts_as_losing_every_anchor_game(self):
        entrants = {'s1': ('starter', 'starter', '1', 0), 's2': ('starter', 'starter', '2', 0),
                    'x1': ('m', 'a1', '1', .1), 'x2': ('m', 'a1', '2', .1)}
        tags = synthetic(Path(self.temp.name) / 't', entrants, units=100)
        plain = analyze.analyze([Path(self.temp.name) / 't'], tags, replicates=50)['games']['testgame']
        lost = analyze.analyze([Path(self.temp.name) / 't'], tags, replicates=50,
                               forfeits=[('testgame', 'm', 'a2')])['games']['testgame']
        before = next(r for r in plain['models'] if r['model'] == 'm')
        after = next(r for r in lost['models'] if r['model'] == 'm')
        self.assertEqual((before['attempts'], after['attempts']), (1, 2))
        self.assertAlmostEqual(after['lift'], (before['lift'] - 50) / 2, delta=0.01)

    def test_reads_games_jsonl_and_tags_from_bindings(self):
        root = Path(self.temp.name)
        entrants = {'s1': ('starter', 'starter', '1', 0), 's2': ('starter', 'starter', '2', 0),
                    'x': ('m', 'x', 'final', .1)}
        synthetic(root / 't', entrants, units=50, rows_file='games.jsonl')
        result = json.loads((root / 't/result.json').read_text())
        for name, settings in (('s1', dict(kind='retrain', seed=1, model='starter', source=dict(kind='starter'))),
                               ('s2', dict(kind='retrain', seed=2, model='starter', source=dict(kind='starter'))),
                               ('x', dict(model='openrouter/m'))):
            session = root / 'sessions' / name
            session.mkdir(parents=True)
            session.with_name(name + '.harness-run.json').write_text(json.dumps(dict(settings=settings)))
            result['entrants'][name] = dict(session=str(session))
        (root / 't/result.json').write_text(json.dumps(result))
        game = analyze.analyze([root / 't'], replicates=50)['games']['testgame']
        self.assertEqual(game['anchors'], 2)
        self.assertEqual(game['entrants']['x']['model'], 'openrouter/m')
        self.assertEqual(game['entrants']['s1'], dict(model='starter', attempt='starter', seed='1'))
        self.assertIsNone(game['variance_components']['attempt_sd'])


if __name__ == '__main__':
    unittest.main()
