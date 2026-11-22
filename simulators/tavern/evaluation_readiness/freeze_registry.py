"""Append a strictly admitted export without changing the frozen mixed profiles."""
import argparse,json,shutil
from pathlib import Path
from .policies import admit_bundle
from .admission import audit_export
from contract import sha,write

def append_export(parent,out,bundle,policy_id,source,checkpoint,role):
    parent,out,bundle=map(Path,(parent,out,bundle))
    if not policy_id.replace('-','').isalnum():raise ValueError('Invalid policy ID')
    manifest=json.loads((parent/'manifest.json').read_text())
    if any(sha(parent/f)!=h for f,h in manifest.items()):raise ValueError('Parent registry drift')
    audit=audit_export(bundle,source,out.parent/('admission-'+policy_id+'.json'),checkpoint)
    registry=json.loads((parent/'registry.json').read_text())
    if policy_id in {p['policy_id'] for p in registry['opponents']}:raise ValueError('Duplicate identity')
    out.mkdir(exist_ok=False)
    for f in manifest:
        target=out/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/f,target);target.chmod(0o644)
    dest=out/'learned'/policy_id;dest.mkdir(parents=True)
    files=('policy.py','learner.py','train.py','identity.json','summary.json','weights.npz')
    for f in files:shutil.copy2(bundle/f,dest/f)
    ident,summary=admit_bundle(dest)
    registry['opponents'].append(dict(policy_id=policy_id,kind='learned',path='learned/'+policy_id,source_origin=str(bundle.resolve()),files={f:sha(dest/f) for f in files},reference_lineage=ident['reference_lineage'],role=role,admission_qualified=True,strong_qualified=False,visibility='Exact reviewed feedforward policy; numeric/card observations and legal mask only'))
    registry['parent_registry_manifest_sha256']=sha(parent/'manifest.json')
    write(out/'registry.json',registry)
    write(out/'manifest.json',{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()})
    for f in out.rglob('*'):
        if f.is_file():f.chmod(0o444)
    for d in sorted([f for f in out.rglob('*') if f.is_dir()],key=lambda p:len(p.parts),reverse=True):d.chmod(0o555)
    out.chmod(0o555)
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('parent','out','bundle','source','checkpoint'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--policy-id',required=True);p.add_argument('--role',required=True);a=p.parse_args()
    print(json.dumps(append_export(a.parent,a.out,a.bundle,a.policy_id,a.source,a.checkpoint,a.role)))
