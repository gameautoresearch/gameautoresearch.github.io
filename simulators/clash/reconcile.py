"""Read-only provider billing/inventory reconciliation; never overwrites old receipts."""
from pathlib import Path
import argparse,json,datetime as dt,dataclasses,subprocess
import modal

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out)
    if out.exists():raise ValueError('new receipt path required')
    now=dt.datetime.now(dt.timezone.utc)
    rows=modal.Workspace.from_context().billing.report(start=now-dt.timedelta(days=1),end=now+dt.timedelta(hours=1),resolution='h')
    names={'clash-fixed8-cuda-pilot-20261108','clash-structural-20261108'}
    ours=[dataclasses.asdict(row) for row in rows if row.description in names and row.environment_name=='staging']
    apps=json.loads(subprocess.check_output(['modal','app','list','--env','staging','--json'],text=True))
    apps=[row for row in apps if row.get('description') in names]
    summary={}
    for name in sorted(names):
        selected=[row for row in ours if row['description']==name]
        summary[name]={'provider_reported_usd':sum(float(row['cost']) for row in selected),'l4_resource_seconds_at_list_price':sum(float(value) for row in selected for resource,value in row['cost_by_resource'].items() if 'L4' in resource)/.000222,'apps':len([row for row in apps if row.get('description')==name]),'live_tasks':sum(int(row.get('tasks',0)) for row in apps if row.get('description')==name)}
    result={'fetched_utc':now.isoformat(),'scope':'only the two owned Clash apps; provider billing may lag or have unsettled hourly rows; function time and list-price estimates are separate','price_url':'https://modal.com/pricing','price_verified_2026_11_08':{'L4_per_second':.000222,'physical_cpu_per_second':.0000131,'GiB_per_second':.00000222},'rows':ours,'summary':summary,'apps':apps}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,default=str,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
