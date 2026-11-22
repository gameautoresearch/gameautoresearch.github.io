import concurrent.futures
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.overnight_budget import CampaignBudget


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.auth = {
            'campaign_id': 'test', 'deadline_utc': '2036-01-01T00:00:00Z',
            'worker_stop_utc': '2035-12-31T23:00:00Z',
            'global_limits': {'gpu_allocated_seconds': 1200, 'resource_charged_or_reserved_usd': 20,
                              'model_charged_or_reserved_usd': 4, 'max_concurrent_gpu_workers': 2},
            'initial_grants': {g: {'gpu_seconds': 400, 'resource_usd': 5, 'model_usd': 1} for g in ('a', 'b')},
            'pricing': {'gpu_per_second': {'H100': .001097, 'L4': .000222},
                        'sandbox_cpu_per_core_second': .00003942,
                        'sandbox_memory_per_gib_second': .00000667,
                        'function_cpu_per_core_second': .0000131,
                        'function_memory_per_gib_second': .00000222, 'reservation_margin': 1.1},
            'researcher_models': {'core': ['model'], 'optional': []}}
        self.auth_path = self.root / 'authorization.json'
        self.auth_path.write_text(json.dumps(self.auth))
        self.db = self.root / 'ledger.sqlite3'
        self.a = CampaignBudget('a', self.db)

    def tearDown(self):
        self.temp.cleanup()

    def receipt(self, identity='sb-test', duration=100):
        return dict(provider_id=identity, started_at=1000., finished_at=1000.+duration, gpu_type='H100!')

    def test_failed_unknown_hold_blocks_relaunch(self):
        op = self.a.reserve_worker('failed', 200, 'H100!', 4, 16)
        with self.assertRaises(ValueError): self.a.reserve_worker('retry', 100, 'H100!', 4, 16)
        with self.assertRaises(ValueError): self.a.settle_worker(op, {'provider_id': 'sb-test'})
        self.assertEqual(self.a.snapshot()['game_usage']['gpu_seconds'], 200)
        self.a.attach(op, 'sb-test')
        self.a.settle_worker(op, self.receipt())
        self.assertEqual(self.a.snapshot()['game_usage']['gpu_seconds'], 100)

    def test_cpu_only_does_not_consume_gpu(self):
        op = self.a.reserve_worker('cpu-grade', 200, None, 4, 16)
        self.a.settle_worker(op, dict(provider_id='cpu', started_at=1, finished_at=101, gpu_type='CPU'))
        snap = self.a.snapshot()
        self.assertEqual(snap['game_usage']['gpu_seconds'], 0)
        self.assertGreater(snap['game_usage']['resource_usd'], 0)

    def test_quotes_include_sandbox_cpu_memory_and_margin(self):
        op = self.a.reserve_worker('gpu', 100, 'H100!', 4, 16)
        row = next(r for r in self.a.snapshot()['operations'] if r['id'] == op)
        self.assertAlmostEqual(row['usd_hold'], 100 * (.001097+4*.00003942+16*.00000667)*1.1)

    def test_overrun_is_not_clipped(self):
        op = self.a.reserve_worker('overrun', 100, 'H100!')
        self.a.settle_worker(op, self.receipt(duration=450))
        self.assertEqual(self.a.snapshot()['game_usage']['gpu_seconds'], 450)
        with self.assertRaises(ValueError): self.a.reserve_worker('again', 1, 'H100!')

    def test_changed_authorization_cannot_reset(self):
        self.auth['initial_grants']['a']['gpu_seconds'] += 100
        self.auth_path.write_text(json.dumps(self.auth))
        with self.assertRaises(ValueError): CampaignBudget('a', self.db)

    def test_parent_reserve_is_bounded(self):
        self.a.parent_grant('a', gpu_seconds=400, resource_usd=10, model_usd=2, reason='test')
        with self.assertRaises(ValueError): self.a.parent_grant('b', gpu_seconds=1)
        with self.assertRaises(ValueError): self.a.parent_grant('b', resource_usd=1)
        with self.assertRaises(ValueError): self.a.parent_grant('b', model_usd=1)

    def test_concurrent_same_game_only_one_gpu_reservation(self):
        def reserve(i):
            try: return CampaignBudget('a', self.db).reserve_worker(str(i), 10, 'H100!')
            except ValueError: return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            values = list(pool.map(reserve, range(8)))
        self.assertEqual(sum(v is not None for v in values), 1)

    def test_wrong_provider_and_repeat_settlement_reject(self):
        op = self.a.reserve_worker('x', 200, 'H100!')
        self.a.attach(op, 'correct')
        with self.assertRaises(ValueError): self.a.settle_worker(op, self.receipt())
        self.a.settle_worker(op, self.receipt('correct'))
        with self.assertRaises(ValueError): self.a.settle_worker(op, self.receipt('correct'))

    def test_no_allocation_requires_recorded_provider_proof(self):
        op = self.a.reserve_worker('x', 100, 'H100!')
        with self.assertRaises(ValueError): self.a.no_allocation(op, {})
        receipt = self.root / 'inventory.json'
        receipt.write_text('{"workers":[]}')
        self.a.no_allocation(op, {'provider_inventory_checked': True, 'worker_count': 0, 'receipt_path': str(receipt)})
        self.assertEqual(self.a.snapshot()['game_usage']['gpu_seconds'], 0)

    def test_model_money_is_separate_and_close_proof_is_required(self):
        op = self.a.reserve_model('llm', .8, 'model')
        with self.assertRaises(ValueError): self.a.reserve_model('other', .3, 'model')
        ledger = self.root / 'model.json'
        ledger.write_text(json.dumps({'closed': True, 'calls': []}))
        proof = dict(closed=True, ledger_path=str(ledger), ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest())
        self.a.settle_model(op, .2, proof)
        self.assertEqual(self.a.snapshot()['game_usage']['model_usd'], .2)
        with self.assertRaises(ValueError): self.a.settle_model(op, .2, proof)
        with self.assertRaises(ValueError): self.a.reserve_model('unverified-model', .1, 'other')

    def test_bad_numeric_inputs_rejected(self):
        for value in (float('nan'), float('inf'), -1, True):
            with self.assertRaises(ValueError): self.a.reserve_worker('x', value, 'H100!')

    def test_parent_transfer_cannot_move_active_holds_or_grow_total(self):
        self.a.reserve_worker('held',300,'H100!')
        with self.assertRaises(ValueError):self.a.parent_transfer('a','b',gpu_seconds=101)
        self.a.parent_transfer('a','b',gpu_seconds=100,resource_usd=1,reason='unused only')
        grants=self.a.snapshot()['grants']
        self.assertEqual(grants['a']['gpu_seconds'],300)
        self.assertEqual(grants['b']['gpu_seconds'],500)
        self.assertEqual(sum(r['gpu_seconds'] for r in grants.values()),800)


if __name__ == '__main__': unittest.main()
