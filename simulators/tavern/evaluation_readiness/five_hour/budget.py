"""Explicit new null-cap ledger; never instantiate a historical capped guard."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from tools.open_credit_campaign import OpenCreditBudget
CAMPAIGN=ROOT/'evidence/strong-bots-20261109-0516Z'
LEDGER=CAMPAIGN/'campaign.sqlite3'
PREFIX='tavern-evaluator-strong:'
class Budget(OpenCreditBudget):
    def __init__(self):
        super().__init__('tavern',LEDGER)
    def reserve(self,name,seconds,cpu=4,memory=16,sandbox=True,gpu=None,extra=0,purpose='development'):
        if (CAMPAIGN/'tavern/evaluator/stop-record.json').exists():
            raise RuntimeError('Evaluator campaign stopped by explicit user order; no new allocations')
        return self.reserve_worker(PREFIX+name,seconds,gpu,cpu,memory,sandbox=sandbox,extra_usd=extra,purpose=purpose)
    def usage(self):
        ops=[x for x in self.snapshot()['operations'] if x['game']=='tavern' and x['kind'].startswith(PREFIX)]
        return dict(gpu_seconds=sum(x['gpu_charged'] if x['status']=='settled' else x['gpu_hold'] for x in ops),resource_usd=sum(x['usd_charged'] if x['status']=='settled' else x['usd_hold'] for x in ops if x['bucket']=='resource'),model_usd=sum(x['usd_charged'] if x['status']=='settled' else x['usd_hold'] for x in ops if x['bucket']=='model'),operations=ops)
