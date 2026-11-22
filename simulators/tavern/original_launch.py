"""Freeze both original baselines and current candidate; no historical records are edited."""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'evidence/tavern-implementation/original-baselines-20261108'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    out=BASE/a.run_id;out.mkdir(parents=True,exist_ok=False);payload=out/'payload'
    ignore=shutil.ignore_patterns('target','__pycache__','*.so','*.dylib','*.pyc','tavern.ptx')
    for label in ['upstream','earliest']:shutil.copytree(BASE/label,payload/label,ignore=ignore)
    for part in ['candidates/AutoTavern','integrations/tavern']:
        shutil.copytree(ROOT/part,payload/'candidate'/part,ignore=shutil.ignore_patterns('target','.git','__pycache__','*.so','*.dylib','*.pyc','tavern.ptx'))
    for name in ['upstream-source.json','earliest-source.json']:shutil.copyfile(BASE/name,payload/name)
    record={'run_id':a.run_id,'status':'started','start_unix':time.time(),
        'authorization':'Explicit Nov8 request for pinned original simulator and earliest trustworthy full-training integration on matched hardware; old spending/hour/attempt caps superseded.',
        'resources':{'gpu':'L4','cpu_cores':4,'ram_gib':16,'timeout_seconds':3600,'retries':0,'warm_containers':0}}
    (out/'attempt.json').write_text(json.dumps(record,indent=2)+'\n');start=time.perf_counter()
    with (out/'modal.log').open('x') as stream:
        r=subprocess.run(['modal','run','--env','staging',str(payload/'candidate/integrations/tavern/original_modal.py'),'--run-id',a.run_id],
            env=dict(os.environ,TAVERN_ORIGINAL_PAYLOAD=str(payload)),stdout=stream,stderr=subprocess.STDOUT,timeout=7200)
    record.update(status='succeeded' if not r.returncode else 'failed',returncode=r.returncode,client_build_and_call_seconds=time.perf_counter()-start)
    (out/'attempt.json').write_text(json.dumps(record,indent=2)+'\n')
    (out/'retained').mkdir()
    download=subprocess.run(['modal','volume','get','autoresearch-tavern-original-evidence-20261108',a.run_id,str(out/'retained'),'--env','staging'],capture_output=True,text=True)
    (out/'retrieval.json').write_text(json.dumps({'returncode':download.returncode,'stderr':download.stderr},indent=2)+'\n')
    print(json.dumps(record),flush=True)
    if r.returncode:raise SystemExit(r.returncode)


if __name__=='__main__':main()
