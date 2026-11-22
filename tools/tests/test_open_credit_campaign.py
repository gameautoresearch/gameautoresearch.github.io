import concurrent.futures
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from tools.open_credit_campaign import OpenCreditBudget


class OpenCreditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        now = datetime.now(timezone.utc)
        self.auth = dict(budget_mode='explicit-user-uncapped-modal-time-bounded', campaign_id='new-test',
            started_at_utc=now.isoformat(), deadline_utc=(now+timedelta(hours=5)).isoformat(),
            worker_stop_utc=(now+timedelta(hours=4, minutes=55)).isoformat(),
            selection_close_utc=(now+timedelta(hours=4)).isoformat(), max_worker_seconds=3600,
            global_limits=dict(gpu_allocated_seconds=None, resource_charged_or_reserved_usd=None,
                model_charged_or_reserved_usd=0, max_concurrent_gpu_workers=3,
                max_concurrent_gpu_workers_per_game=2),
            initial_grants={g:dict(gpu_seconds=None, resource_usd=None, model_usd=0) for g in ('a','b')},
            worker_profiles={g:dict(gpu='L4',cpu_cores=4,memory_gib=16) for g in ('a','b')},
            pricing=dict(gpu_per_second=dict(L4=.000222,H100=.001097),
                sandbox_cpu_per_core_second=.00003942,sandbox_memory_per_gib_second=.00000667,
                function_cpu_per_core_second=.0000131,function_memory_per_gib_second=.00000222,
                reservation_margin=1.1), researcher_models=dict(core=['model']))
        self.path = self.root/'ledger.sqlite3'
        self.save()

    def tearDown(self): self.temp.cleanup()

    def save(self): (self.root/'authorization.json').write_text(json.dumps(self.auth))

    def budget(self, game='a'): return OpenCreditBudget(game,self.path)

    def test_uncapped_dollars_are_accounted(self):
        b=self.budget();op=b.reserve_worker('large',10,'L4',extra_usd=100000)
        b.attach(op,'actual');b.settle_worker(op,dict(provider_id='actual',started_at=1,finished_at=21,gpu_type='L4'))
        s=b.snapshot();self.assertEqual(s['game_usage']['gpu_seconds'],20)
        self.assertGreater(s['game_usage']['resource_usd'],100000)

    def test_same_game_concurrency_is_atomic(self):
        self.budget()
        def reserve(i):
            try:return self.budget().reserve_worker(str(i),10,'L4')
            except ValueError:return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool: rows=list(pool.map(reserve,range(8)))
        self.assertEqual(sum(x is not None for x in rows),2)

    def test_global_unknown_holds_block_more_workers(self):
        a=self.budget();a.reserve_worker('1',10,'L4');a.reserve_worker('2',10,'L4')
        b=self.budget('b');b.reserve_worker('3',10,'L4')
        with self.assertRaises(ValueError):b.reserve_worker('4',10,'L4')

    def test_time_and_selection_stops(self):
        now=datetime.now(timezone.utc)
        self.auth['selection_close_utc']=(now+timedelta(seconds=5)).isoformat();self.save()
        b=self.budget()
        with self.assertRaises(ValueError):b.reserve_worker('train',10,'L4')
        b.reserve_worker('grade',10,None,purpose='final')

    def test_absolute_stop_and_expiry(self):
        self.auth['worker_stop_utc']=(datetime.now(timezone.utc)+timedelta(seconds=5)).isoformat();self.save()
        with self.assertRaises(ValueError):self.budget().reserve_worker('cpu',10,None,purpose='final')
        self.auth['deadline_utc']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
        self.auth['started_at_utc']=(datetime.now(timezone.utc)-timedelta(hours=1)).isoformat();self.save()
        with self.assertRaises(ValueError):self.budget().reserve_worker('late',1,None)

    def test_profile_and_stage_bounds(self):
        b=self.budget()
        for kwargs in (dict(gpu='H100!'),dict(gpu='L4',cpu_cores=8),dict(gpu='L4',memory_gib=32)):
            with self.assertRaises(ValueError):b.reserve_worker('wrong',10,**kwargs)
        with self.assertRaises(ValueError):b.reserve_worker('long',3601,'L4')

    def test_authorization_cannot_be_changed_or_inferred(self):
        self.budget();self.auth['max_worker_seconds']=7200;self.save()
        with self.assertRaises(ValueError):self.budget()
        self.auth['budget_mode']='old';self.save()
        with self.assertRaises(ValueError):self.budget()

    def test_model_spending_is_not_uncapped(self):
        with self.assertRaises(ValueError):self.budget().reserve_model('llm',.01,'model')


if __name__=='__main__': unittest.main()
