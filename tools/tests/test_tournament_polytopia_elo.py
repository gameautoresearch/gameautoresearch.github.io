"""The Polytopia checkpoint-Elo schedule: sparse stages, pair rules and the Bradley-Terry fit (no provider)."""
import random
import types
import unittest

from tools.tournament.polytopia_elo import run, schedule


def field(n=30):
    names = [f'm:{i}' for i in range(n)] + ['anchor0', 'ref-a', 'ref-b']
    kind = {x: 'research' for x in names}
    kind.update({'anchor0': 'anchor', 'ref-a': 'reference', 'ref-b': 'reference'})
    return names, kind


class Schedule(unittest.TestCase):
    def test_references_never_play_each_other(self):
        names, kind = field(5)
        pairs = schedule.all_pairs(names, kind)
        self.assertNotIn(schedule.key('ref-a', 'ref-b'), pairs)
        self.assertEqual(len(pairs), len(names) * (len(names) - 1) // 2 - 1)

    def test_stage_a_covers_every_fixed_opponent_and_random_partners(self):
        names, kind = field(30)
        pairs = set(schedule.stage_a(names, kind, partners=4, seed=1))
        for n in names:
            if kind[n] == 'reference':
                continue
            for f in ('anchor0', 'ref-a', 'ref-b'):
                if f != n:
                    self.assertIn(schedule.key(n, f), pairs)
            others = {b if a == n else a for a, b in pairs if n in (a, b)} - {'anchor0', 'ref-a', 'ref-b'}
            if kind[n] == 'research':
                self.assertGreaterEqual(len(others), 4)
        self.assertNotIn(schedule.key('ref-a', 'ref-b'), pairs)

    def test_neighbours_are_the_nearest_by_provisional_elo(self):
        names, kind = field(20)
        provisional = {n: float(i * 10) for i, n in enumerate(names)}
        pairs = set(schedule.neighbours(names, kind, provisional, k=3))
        self.assertIn(schedule.key('m:5', 'm:6'), pairs)
        self.assertIn(schedule.key('m:5', 'm:4'), pairs)
        self.assertNotIn(schedule.key('m:5', 'm:15'), pairs)

    def test_fit_recovers_order_and_anchor_shift(self):
        rng = random.Random(0)
        true = {'a': 0.0, 'b': 200.0, 'c': -300.0}
        games = []
        for x in true:
            for y in true:
                if x < y:
                    p = 1 / (1 + 10 ** ((true[y] - true[x]) / 400))
                    for _ in range(2000):
                        games.append(dict(a=x, b=y, a_points=1.0 if rng.random() < p else 0.0))
        e = schedule.provisional_elo(list(true), games)
        shifted = {k: v - e['a'] for k, v in e.items()}
        self.assertAlmostEqual(shifted['b'], 200, delta=25)
        self.assertAlmostEqual(shifted['c'], -300, delta=25)

    def test_plan_has_stage_b_only_after_stage_a_games_and_skips_nothing_itself(self):
        names, kind = field(10)
        args = types.SimpleNamespace(smoke=False, schedule='sparse', blocks=3, partners=2, neighbours=4, seed=0,
                                     smoke_pairs=3)
        stages = run.plan(args, names, kind, None)
        self.assertEqual([s for s, _ in stages], ['A'])
        self.assertTrue(all(r == 0 for r, _, _ in stages[0][1]))
        games = [dict(a=a, b=b, a_points=0.5) for _, a, b in stages[0][1]]
        stages = run.plan(args, names, kind, games)
        self.assertEqual([s for s, _ in stages], ['A', 'B'])
        self.assertEqual({r for r, _, _ in stages[1][1]}, {0, 1, 2})
        full = run.plan(types.SimpleNamespace(**dict(vars(args), schedule='full')), names, kind, None)
        self.assertEqual(len(full), 3)


if __name__ == '__main__':
    unittest.main()
