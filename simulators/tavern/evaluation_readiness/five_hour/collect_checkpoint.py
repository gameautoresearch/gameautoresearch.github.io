"""Read a published immutable RL snapshot; never create/exec a paid worker."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
from .budget import CAMPAIGN
from simulators.tavern.autoresearch.evaluation_v2.release import sha,write


def collect(stage,snapshot,out):
    if any(not value.replace('-','').isalnum() for value in (stage,snapshot)):raise ValueError('Exact published stage/snapshot names required')
    root=CAMPAIGN/'tavern/rl'/stage;published=root/'collected' if snapshot=='endpoint' else root/'diagnostics'/snapshot;identity=json.loads((published/'export/identity.json').read_text());launch=json.loads((root/'provider.json').read_text());expected=identity['reference_lineage']['checkpoint_sha256']
    local=published/'training/latest.pt' if snapshot=='endpoint' else published/'checkpoint.pt'
    if snapshot=='endpoint':
        allocation=json.loads((root/'allocation.json').read_text())
        if not allocation['provider'].get('finished_at'):raise ValueError('Endpoint collection requires authoritative provider completion')
    if local.exists() and sha(local)==expected:return dict(checkpoint=str(local.resolve()),sha256=expected,read_only_existing=True,new_paid_allocation=False)
    out=Path(out);out.mkdir(parents=True,mode=0o700,exist_ok=False);temp=out/'checkpoint.download';target=out/'checkpoint.pt'
    import modal
    volume=modal.Volume.from_name(launch['volume_name'],environment_name='staging');remote='/'+stage+'/training/latest.pt' if snapshot=='endpoint' else '/'+stage+'/diagnostics/'+snapshot+'/checkpoint.pt';digest=hashlib.sha256()
    with temp.open('wb') as f:
        for chunk in volume.read_file(remote):f.write(chunk);digest.update(chunk)
    if digest.hexdigest()!=expected:raise ValueError('Downloaded checkpoint does not match published export; partial evidence retained')
    temp.replace(target);target.chmod(0o600);receipt=dict(stage=stage,snapshot=snapshot,checkpoint=str(target.resolve()),sha256=expected,export_identity_sha256=sha(published/'export/identity.json'),volume_name=launch['volume_name'],remote_path=remote,new_paid_allocation=False,can_create_worker=False)
    write(out/'collection.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(collect(a.stage,a.snapshot,a.out)))
