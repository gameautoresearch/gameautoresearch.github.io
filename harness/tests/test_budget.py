"""The two research budget modes: choosing one, prices, limits and the registry's per-session budgets."""
import types
import unittest
from unittest import mock

from harness import budget, registry, session

SPEC = dict(game='fake', venv='.venv-x', constructor='x:y', mode='m', hardware='one L40S', final_train_seconds=3600,
            smoke_final_train_seconds=120, retrain_seed='run', parameter_cap=1, research_seconds=7200,
            smoke_research_seconds=600, research_usd=10.0, smoke_research_usd=0.5, wall_seconds=18000,
            smoke_wall_seconds=4800, budget_mode='seconds')
TRAIN = {'gpu': 1, 'cpu_cores': 4, 'memory_mib': 16384}


class BudgetTest(unittest.TestCase):
    def test_choose(self):
        self.assertEqual(budget.choose(SPEC), 'seconds')
        self.assertEqual(budget.choose(dict(SPEC, budget_mode='usd')), 'usd')
        self.assertEqual(budget.choose(SPEC, research_usd=5), 'usd')
        self.assertEqual(budget.choose(dict(SPEC, budget_mode='usd'), research_seconds=60), 'seconds')
        self.assertEqual(budget.choose(SPEC, 'usd'), 'usd')
        for bad in (dict(budget_mode='minutes'), dict(research_seconds=1, research_usd=1),
                    dict(budget_mode='usd', research_seconds=1), dict(budget_mode='seconds', research_usd=1)):
            with self.assertRaises(ValueError):
                budget.choose(SPEC, **bad)

    def test_prices_and_limits(self):
        self.assertAlmostEqual(budget.rate(TRAIN), 0.0008064)
        self.assertAlmostEqual(budget.rate({'gpu': 1, 'cpu_cores': 16, 'memory_mib': 65536}) * 10800, 17.28, places=2)
        self.assertEqual(budget.seconds_weight(TRAIN, TRAIN), 1.0)
        self.assertLess(budget.seconds_weight({'gpu': 0, 'cpu_cores': 8, 'memory_mib': 16384}, TRAIN), 1.0)
        self.assertEqual(budget.limits('seconds')['max_concurrent_jobs'], 1)
        self.assertEqual(budget.limits('usd')['max_concurrent_jobs'], 4)
        self.assertIn('one at a time', budget.describe('seconds'))
        self.assertIn('up to 4 at once', budget.describe('usd'))

    def test_registry_and_session_follow_the_chosen_mode(self):
        fake = types.SimpleNamespace(SPEC=SPEC)
        with mock.patch.object(registry, 'module', lambda game: fake):
            self.assertEqual(registry.research('fake'), dict(mode='seconds', seconds=7200.0, usd=None))
            self.assertEqual(registry.research('fake', budget_mode='usd'), dict(mode='usd', seconds=None, usd=10.0))
            self.assertEqual(registry.research('fake', smoke=True, budget_mode='usd')['usd'], 0.5)
            self.assertEqual(registry.research('fake', research_usd=3)['usd'], 3.0)
            self.assertEqual(registry.budgets('fake', research_seconds=100), (100.0, 3600.0))
            self.assertEqual(registry.budgets('fake', budget_mode='usd'), (None, 3600.0))
            self.assertEqual(session.wall_limit('fake', research_seconds=7200, budget_mode='seconds'), 2 * 7200 + 3600)
            self.assertEqual(session.wall_limit('fake', budget_mode='usd'), 18000)
            self.assertEqual(session.wall_limit('fake', smoke=True, budget_mode='usd'), 4800)
            self.assertTrue(session.official('fake', dict(smoke=False, budget_mode='seconds', research_seconds=7200,
                                                          variant='network')))
            self.assertFalse(session.official('fake', dict(smoke=False, budget_mode='usd', research_seconds=None,
                                                           research_usd=10.0, variant='network')))
        with mock.patch.object(registry, 'module', lambda game: types.SimpleNamespace(SPEC=dict(SPEC, budget_mode='x'))):
            with self.assertRaises(ValueError):
                registry.spec('fake')


if __name__ == '__main__':
    unittest.main()
