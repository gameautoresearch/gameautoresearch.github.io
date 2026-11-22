"""Explicit new-campaign ledger plus evaluator role subcaps; never old defaults."""
import fcntl,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from overnight_budget import CampaignBudget
CAMPAIGN=ROOT/'evidence/reference-readiness-20261109-0251Z';LEDGER=CAMPAIGN/'campaign.sqlite3';PREFIX='tavern-evaluator:'

class EvaluatorBudget:
    def __init__(self,ledger=LEDGER):
        self.ledger=Path(ledger).resolve()
        if self.ledger!=LEDGER.resolve():raise ValueError('This launcher is bound to the explicit reference-readiness ledger')
        self.b=CampaignBudget('tavern',path=self.ledger);self.root=self.ledger.parent/'tavern/evaluator';self.root.mkdir(parents=True,exist_ok=True)
    def usage(self):
        rows=[r for r in self.b.snapshot()['operations'] if r['game']=='tavern' and r['kind'].startswith(PREFIX)]
        return dict(gpu_seconds=sum(r['gpu_charged'] if r['status']=='settled' else r['gpu_hold'] for r in rows),resource_usd=sum(r['usd_charged'] if r['status']=='settled' else r['usd_hold'] for r in rows if r['bucket']=='resource'),model_usd=sum(r['usd_charged'] if r['status']=='settled' else r['usd_hold'] for r in rows if r['bucket']=='model'),operations=rows)
    def reserve(self,name,seconds,cpu=4,memory=16,sandbox=True,gpu=None,extra=0):
        rates=self.b.auth['pricing'];flavor='sandbox' if sandbox else 'function';rate=cpu*rates[flavor+'_cpu_per_core_second']+memory*rates[flavor+'_memory_per_gib_second']+(rates['gpu_per_second'][gpu] if gpu else 0);quote=(rate*seconds+extra)*rates['reservation_margin']
        with (self.root/'role-budget.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);u=self.usage()
            if u['gpu_seconds']+(seconds if gpu else 0)>1800 or u['resource_usd']+quote>8:raise ValueError('Evaluator role subcap exhausted')
            return self.b.reserve_worker(PREFIX+name,seconds,gpu,cpu,memory,sandbox=sandbox,extra_usd=extra)
    def attach(self,op,provider):self.b.attach(op,provider)
    def settle(self,op,receipt):self.b.settle_worker(op,receipt)
