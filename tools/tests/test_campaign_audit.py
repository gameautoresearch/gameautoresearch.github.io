import hashlib
import json
import unittest

from tools.audit_overnight_campaign import audit
from tools.tests import test_overnight_budget as budget_tests


class CampaignAuditTests(unittest.TestCase):
    def setUp(self):
        self.fixture = budget_tests.BudgetTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.budget = self.fixture.a

    def model(self, name, calls, amount):
        operation = self.budget.reserve_model(name, .1, 'model')
        path = self.fixture.root / (name + '.json')
        path.write_text(json.dumps({'closed': True, 'calls': calls}))
        self.budget.settle_model(operation, amount, dict(closed=True,
            ledger_path=str(path), ledger_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def test_provider_and_priced_call_reconcile(self):
        operation = self.budget.reserve_worker('training', 100, 'H100!')
        self.budget.settle_worker(operation, self.fixture.receipt())
        self.model('research', [{'id': 'unique', 'status': 'complete',
                                'usage': {'cost': .01}, 'reserved_usd': .05}], .01)
        result = audit(self.budget)
        self.assertTrue(result['audit_passed'], result['errors'])
        self.assertAlmostEqual(result['settled_model_priced_usd'], .01)

    def test_overrun_is_reported_after_full_charge(self):
        operation = self.budget.reserve_worker('overrun', 100, 'H100!')
        self.budget.settle_worker(operation, self.fixture.receipt(duration=450))
        result = audit(self.budget)
        self.assertFalse(result['audit_passed'])
        self.assertEqual(result['settled']['gpu_seconds'], 450)
        self.assertTrue(any('exceeds current grant' in e for e in result['errors']))

    def test_duplicate_generation_is_not_double_billed_silently(self):
        call = {'id': 'same-provider-generation', 'status': 'priced',
                'usage': {'cost': .01}, 'reserved_usd': .05}
        self.model('first', [call], .01)
        self.model('resumed', [call], .01)
        result = audit(self.budget)
        self.assertFalse(result['audit_passed'])
        self.assertTrue(any('also accounted' in e for e in result['errors']))

    def test_unknown_model_request_remains_liability_not_priced_cost(self):
        self.model('unknown', [{'status': 'unknown', 'reserved_usd': .05}], .05)
        result = audit(self.budget)
        self.assertTrue(result['audit_passed'], result['errors'])
        self.assertEqual(result['settled_model_priced_usd'], 0)
        self.assertEqual(result['settled_model_unknown_liability_usd'], .05)

    def test_changed_closed_ledger_fails_independent_audit(self):
        self.model('closed', [{'id': 'one', 'status': 'complete', 'usage': {'cost': .01}}], .01)
        path = self.fixture.root / 'closed.json'
        data = json.loads(path.read_text())
        data['closed_at'] = 'changed after settlement'
        path.write_text(json.dumps(data))
        result = audit(self.budget)
        self.assertFalse(result['audit_passed'])
        self.assertTrue(any('hash changed' in e for e in result['errors']))


if __name__ == '__main__':
    unittest.main()
