"""Append an entrant without importing its code or changing any opponent/profile."""
import json,shutil
from pathlib import Path
import numpy as np
from .release import sha,write

def append(parent,bundle,out,identity):
    parent,bundle,out=map(Path,(parent,bundle,out));manifest=json.loads((parent/'manifest.json').read_text());registry=json.loads((parent/'registry.json').read_text());ident=json.loads((bundle/'identity.json').read_text());summary=json.loads((bundle/'summary.json').read_text())
    if any(sha(parent/f)!=h for f,h in manifest.items()):raise ValueError('Frozen reference registry drift')
    if ident['binding']!=registry['binding'] or sha(bundle/'identity.json')!=summary['identity_sha256'] or sha(bundle/'weights.npz')!=summary['weights_sha256']:raise ValueError('Entrant identity/weights/task mismatch')
    if any(sha(bundle/f)!=ident['source'][f] for f in ('policy.py','learner.py','train.py')):raise ValueError('Entrant source mutation')
    if (bundle/'weights.npz').stat().st_size>32000000:raise ValueError('Entrant weights exceed limit')
    with np.load(bundle/'weights.npz',allow_pickle=False) as z:
        if sum(x.size for x in z.values())>2000000 or any(not np.isfinite(x).all() for x in z.values()):raise ValueError('Nonfinite/oversized entrant state')
    if identity in {x['policy_id'] for x in registry['opponents']}:raise ValueError('Candidate ID collides with reference')
    out.mkdir(parents=True,exist_ok=False)
    for rel in manifest:
        p=out/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/rel,p);p.chmod(0o644)
    dest=out/'learned'/identity;dest.mkdir(parents=True)
    for name in ('policy.py','learner.py','train.py','weights.npz','identity.json','summary.json'):shutil.copy2(bundle/name,dest/name)
    registry['opponents'].append(dict(policy_id=identity,kind='learned',path='learned/'+identity,role='isolated-entrant',admission='Exact original source/config/task/NPZ; host never imports entrant source'))
    write(out/'registry.json',registry);write(out/'manifest.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    for p in out.rglob('*'):
        if p.is_file():p.chmod(0o444)
    for p in sorted([p for p in out.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):p.chmod(0o555)
    out.chmod(0o555);return out
