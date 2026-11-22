"""Content-addressed policy packages; loading a package never executes its code."""
from pathlib import Path
import json
import shutil
import math
import tarfile

from .contract import DEFAULT,EDITABLE,digest,sha,write_json,frozen_manifest

FILES=(*EDITABLE,'model.json','weights.pt')

def archive_bundle(directory,destination):
    verify(directory,check_tensors=False)
    with tarfile.open(destination,'w') as archive:
        for name in (*FILES,'manifest.json'):archive.add(Path(directory)/name,arcname=name,recursive=False)

def unpack_bundle(archive_path,destination):
    limits={**{n:DEFAULT['source_bytes_cap'] for n in EDITABLE},'model.json':65536,
        'weights.pt':DEFAULT['artifact_bytes_cap'],'manifest.json':1048576}
    if Path(archive_path).stat().st_size>sum(limits.values())+20480:raise ValueError('Oversized bundle archive')
    with tarfile.open(archive_path,'r:') as archive:
        members=archive.getmembers()
        if len(members)!=len(limits) or {m.name for m in members}!=set(limits):raise ValueError('Unexpected archive members')
        if any(not m.isfile() or m.size>limits[m.name] for m in members):raise ValueError('Invalid archive member')
        root=Path(destination);root.mkdir(parents=True,exist_ok=False)
        for member in members:
            with archive.extractfile(member) as source,(root/member.name).open('xb') as target:
                shutil.copyfileobj(source,target)
    verify(destination,check_tensors=False)

def source_payload(directory):
    root=Path(directory)
    result={}
    for name in EDITABLE:
        p=root/name
        if p.is_symlink() or not p.is_file():raise ValueError('Missing or symlinked editable file: '+name)
        result[name]=p.read_text()
    if sum(len(v.encode()) for v in result.values())>DEFAULT['source_bytes_cap']:raise ValueError('Editable source exceeds transport cap')
    return result

# Any bundle the arena loads, retained panel policies included. A researcher's own export is held to the
# contract's smaller `policy_tensor_cap` when it is sealed.
BUNDLE_TENSOR_BOUND=10_000_000

def validate_weights(path,cap=BUNDLE_TENSOR_BOUND):
    import torch
    p=Path(path)
    if p.is_symlink() or p.stat().st_size>DEFAULT['artifact_bytes_cap']:raise ValueError('Oversized or symlinked policy weights')
    state=torch.load(p,map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not state:raise ValueError('Weights must be a nonempty tensor state_dict')
    if any(not isinstance(k,str) or not torch.is_tensor(v) for k,v in state.items()):raise ValueError('Policy state must contain tensors only')
    count=sum(v.numel() for v in state.values())
    if count>cap:raise ValueError('Policy tensor cap exceeded')
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):raise ValueError('Non-finite policy weights')
    return count

def seal(source,destination,receipt=None,cap=None):
    source=Path(source);dest=Path(destination)
    if dest.exists():raise FileExistsError('Immutable bundle already exists')
    for name in FILES:
        if (source/name).is_symlink() or not (source/name).is_file():raise ValueError('Bundle requires regular '+name)
    source_payload(source)
    if (source/'model.json').stat().st_size>65536:raise ValueError('Model metadata too large')
    model=json.loads((source/'model.json').read_text())
    tensors=validate_weights(source/'weights.pt',DEFAULT['policy_tensor_cap'] if cap is None else cap)
    manifest={'format':1,'contract':DEFAULT['version'],'files':{n:sha(source/n) for n in FILES},
        'tensors':tensors,'engine':digest(frozen_manifest()),'model':model,'provenance':receipt or {}}
    manifest['bundle_sha256']=digest(manifest)
    dest.mkdir(parents=True)
    for name in FILES:shutil.copy2(source/name,dest/name)
    write_json(dest/'manifest.json',manifest,exclusive=True)
    verify(dest)
    return manifest['bundle_sha256']

def verify(directory,expected=None,check_tensors=True):
    root=Path(directory)
    if root.is_symlink():raise ValueError('Symlinked policy package')
    m=json.loads((root/'manifest.json').read_text())
    identity=m['bundle_sha256']
    if expected and identity!=expected:raise ValueError('Submitted bundle identity changed')
    if digest({k:v for k,v in m.items() if k!='bundle_sha256'})!=identity:raise ValueError('Bundle manifest drift')
    if m.get('contract')!=DEFAULT['version'] or m.get('engine')!=digest(frozen_manifest()):raise ValueError('Bundle frozen engine mismatch')
    if set(m['files'])!=set(FILES):raise ValueError('Unexpected bundle members')
    for name,value in m['files'].items():
        if (root/name).is_symlink() or sha(root/name)!=value:raise ValueError('Bundle file drift: '+name)
    if check_tensors and validate_weights(root/'weights.pt')!=m['tensors']:raise ValueError('Tensor count changed')
    return m

def export_existing(checkpoint,destination):
    """Trusted conversion still goes through the original source-bound loader."""
    import torch
    from integrations.doudizhu.artifacts import load_checkpoint
    policy,data=load_checkpoint(checkpoint,'cpu')
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)
        policy_code=(Path(__file__).parent/'starter/policy.py').read_text()
        (p/'policy.py').write_text(policy_code)
        for name in ('learner.py','train.py'):shutil.copy2(Path(__file__).parent/'starter'/name,p/name)
        write_json(p/'model.json',data['architecture'])
        torch.save(policy.state_dict(),p/'weights.pt')
        return seal(p,destination,{'source_checkpoint_sha256':sha(checkpoint),'historical_config':data['config'],
            'trained_games':data['state'].get('episodes'),'status':'development_bot_not_a_calibrated_anchor'},
            cap=BUNDLE_TENSOR_BOUND)
