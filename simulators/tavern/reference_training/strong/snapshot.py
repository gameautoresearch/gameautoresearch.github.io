"""Take one immutable public diagnostic inside an already reserved worker.

The trainer atomically replaces latest.pt. A hard link captures one complete old
inode without stopping or editing the trainer. The worker's frozen exporter then
checks its exact source/config binding. This adds no allocation or retry loop.
"""
import argparse,json,os,sys,time
from pathlib import Path
from .common import ROOT,GAME,sha,write

REMOTE = '''
import os,json,hashlib,subprocess
from pathlib import Path
from simulators.tavern.reference_training.strong.export import export
root=Path({root!r});dest=root/'diagnostics'/{label!r}
dest.mkdir(parents=True,exist_ok=False)
os.link(root/'training/latest.pt',dest/'checkpoint.pt')
export(dest/'checkpoint.pt',dest/'export')
subprocess.run(['sync',str(dest)],check=True,timeout=10)
print((dest/'export/summary.json').read_text())
'''

def snapshot(name,label,collect_only=False):
    if not label.replace('-','').isalnum():raise ValueError('Simple unique label required')
    os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
    import modal
    sys.path.insert(0,str(GAME/'autoresearch'))
    from provider import configure_sandbox_waits,allocation_info,command,read_rpc
    configure_sandbox_waits()
    stage=ROOT/'evidence/strong-bots-20261109-0516Z/tavern/rl'/name
    provider=json.loads((stage/'provider.json').read_text())
    info=allocation_info(provider['app_id'],provider['sandbox_id'])
    if not info or not info.get('started_at') or info.get('finished_at'):
        raise ValueError('Snapshot requires the recorded live reserved worker')
    dest=stage/'diagnostics'/label
    if not collect_only:
        dest.mkdir(parents=True,exist_ok=False)
        code=REMOTE.format(root='/retained/'+name,label=label)
        (dest/'command.py').write_text(code)
        write(dest/'request.json',dict(provider=info,helper_sha256=sha(__file__),command_sha256=sha(dest/'command.py'),requested_unix=time.time(),new_paid_allocation=False))
    elif not (dest/'remote-summary.json').exists():raise ValueError('No successful immutable remote snapshot to collect')
    sandbox=modal.Sandbox.from_id(provider['sandbox_id'])
    try:
        if not collect_only:
            result=command(sandbox,['python','-u','-c',code],timeout=90)
            write(dest/'remote-summary.json',json.loads(result))
        # Live v2-volume writes may not yet be visible through the host Volume
        # API. Read the completed immutable files from the existing worker.
        for f in ['policy.py','learner.py','train.py','identity.json','summary.json','weights.npz']:
            target=dest/'export'/f;target.parent.mkdir(exist_ok=True)
            if target.exists():continue
            temp=target.with_suffix('.download')
            temp.write_bytes(read_rpc(lambda:sandbox.filesystem.read_bytes('/retained/'+name+'/diagnostics/'+label+'/export/'+f)))
            temp.replace(target)
    finally:sandbox.detach()
    s=json.loads((dest/'export/summary.json').read_text())
    assert sha(dest/'export/identity.json')==s['identity_sha256']
    assert sha(dest/'export/weights.npz')==s['weights_sha256']
    write(dest/'collected.json',dict(complete=True,summary=s,checkpoint_remote='/'+name+'/diagnostics/'+label+'/checkpoint.pt'))
    print(json.dumps(dict(name=name,label=label,**s)),flush=True)
    return dest/'export'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--label',required=True);p.add_argument('--collect-only',action='store_true');a=p.parse_args();snapshot(a.name,a.label,a.collect_only)
