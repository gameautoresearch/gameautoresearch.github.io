"""Fetch only this Clash app's Modal provider billing rows; never prints credentials."""
from pathlib import Path
import json,datetime as dt,dataclasses
import modal
OUT=Path(__file__).resolve().parents[2]/'evidence/clash-implementation/cuda-campaign'
now=dt.datetime.now(dt.timezone.utc)
rows=modal.Workspace.from_context().billing.report(start=now-dt.timedelta(days=1),end=now+dt.timedelta(hours=1),resolution='h')
ours=[dataclasses.asdict(r) for r in rows if r.description=='clash-fixed8-cuda-pilot-20261108' and r.environment_name=='staging']
result={'fetched_utc':now.isoformat(),'scope':'provider billing API, provisional until interval settles; includes builds, GPU app load, CPU/RAM and egress','rows':ours,'reported_total_usd':sum(float(r['cost']) for r in ours),
 'l4_billed_resource_seconds_inferred_from_price':sum(float(v) for r in ours for k,v in r['cost_by_resource'].items() if 'L4' in k)/.000222}
(OUT/'provider-billing.json').write_text(json.dumps(result,default=str,indent=2)+'\n');print(json.dumps(result,default=str,indent=2))
