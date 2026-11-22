"""Read-only metering for recorded RL apps; unknown image/storage holds stay held."""
import argparse,json,os
from pathlib import Path
from datetime import datetime,timedelta,timezone
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
ROOT=Path(__file__).resolve().parents[4]
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 r=ROOT/'evidence/reference-readiness-20261109-0251Z';ids={json.loads(p.read_text())['app_id'] for p in (r/'tavern/rl').glob('*/provider.json')}
 import modal
 auth=json.loads((r/'authorization.json').read_text());start=datetime.fromisoformat(auth['started_at_utc'].replace('Z','+00:00'));now=datetime.now(timezone.utc);end=now.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1)
 rows=[dict(row.items()) for row in modal.Environment.from_name('staging').billing.report(start=start,end=end,resolution='h') if row.object_id in ids]
 result=dict(source='Modal Environment.billing.report',requested_at_utc=now.isoformat(),owned_app_ids=sorted(ids),rows=rows,observed_app_billing_usd=sum(float(row['cost']) for row in rows),final_invoice=False,coverage_complete=False,holds_released=False,note='Provider metering can lag; image builds/storage/egress may be separately attributed. Keep all unknown holds.')
 a.out.write_text(json.dumps(result,default=str,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['requested_at_utc','observed_app_billing_usd','coverage_complete','holds_released']}))
