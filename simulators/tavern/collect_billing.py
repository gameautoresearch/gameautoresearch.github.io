"""Fetch provider billing filtered strictly to this game's app/image identifiers."""
from datetime import datetime,timezone,timedelta
from pathlib import Path
import json
import re
import modal
import argparse

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'evidence/tavern-implementation'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True,help='new output; historical receipts are immutable')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    ids=set()
    logs=[*EVIDENCE.glob('*-modal.log'),*EVIDENCE.glob('structural-20261108/*/modal.log'),*EVIDENCE.glob('device-port/*/modal.log'),*EVIDENCE.glob('original-baselines-20261108/*/modal.log'),*EVIDENCE.glob('original-baselines-20261108/*diagnostic*.log')]
    for log in logs:
        ids.update(re.findall(r'\b(?:ap|im|fu)-[A-Za-z0-9]+',log.read_text()))
    now=datetime.now(timezone.utc)
    end=now.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1)
    rows=modal.Environment.from_name('staging').billing.report(start=datetime(2026,11,8,tzinfo=timezone.utc),end=end,resolution='h')
    selected=[dict(row.items()) for row in rows if 'autoresearch-tavern-' in row.description or row.object_id in ids]
    result={'requested_at':now.isoformat(),'source':'Modal Environment.billing.report','rows':selected,
      'billed_usd_observed':sum(float(row['cost']) for row in selected),
      'L4_equivalent_billed_seconds':sum(float(v)/.000222 for row in selected for k,v in row['cost_by_resource'].items() if 'L4' in k),
      'CPU_core_second_equivalent':sum(float(v)/.0000131 for row in selected for k,v in row['cost_by_resource'].items() if k=='CPU'),
      'Memory_GiB_second_equivalent':sum(float(v)/.00000222 for row in selected for k,v in row['cost_by_resource'].items() if k=='Memory'),
      'notes':'Provider metering can lag. Historical campaign reservations remain unchanged. New structural campaign is separately authorized without old dollar/hour/attempt gates. Filtered to this game only; CPU/image-build, memory, egress and GPU all retained.'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,default=str,indent=2)+'\n')
    print(json.dumps(result,default=str,indent=2))


if __name__=='__main__':main()
