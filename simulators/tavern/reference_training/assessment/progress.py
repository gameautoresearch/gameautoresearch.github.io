"""Compact read-only provider/Volume progress; no new workers or model calls."""
import argparse,json,os,sys,time
from pathlib import Path
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
ROOT=Path(__file__).resolve().parents[4]
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args();r=ROOT/'evidence/reference-readiness-20261109-0251Z/tavern/rl'/a.attempt;launch=json.loads((r/'provider.json').read_text())
 sys.path.insert(0,str(ROOT/'simulators/tavern/autoresearch'))
 from provider import configure_sandbox_waits,allocation_info
 import modal
 configure_sandbox_waits();info=allocation_info(launch['app_id'],launch['sandbox_id']);v=modal.Volume.from_name(launch['volume_name'],environment_name='staging')
 result=dict(provider=info,checked_unix=time.time())
 for name in ['training/progress.json','training/result.json','lifecycle.json']:
  for read_attempt in range(4):
   try:
    result[name]=json.loads(b''.join(v.read_file('/'+a.attempt+'/'+name)));break
   except json.JSONDecodeError as exc:
    result[name]=dict(unavailable=type(exc).__name__,transient_read_attempts=read_attempt+1)
    if read_attempt<3:time.sleep(.2*(read_attempt+1))
   except Exception as exc:
    result[name]=dict(unavailable=type(exc).__name__);break
 (r/('progress-read-'+str(int(time.time()))+'.json')).write_text(json.dumps(result,indent=2)+'\n')
 row=result['training/progress.json'];counts=row.get('counts',{})
 print(json.dumps(dict(provider_finished=bool(info and info.get('finished_at')),update=row.get('update'),session_seconds=row.get('elapsed'),accepted_this_update=counts.get('accepted_learning_samples'),invalid_lobbies_this_update=counts.get('invalid_lobbies'),rejections_this_update=counts.get('rejected'),status=result['lifecycle.json'].get('status'),result=result['training/result.json']),indent=2))
