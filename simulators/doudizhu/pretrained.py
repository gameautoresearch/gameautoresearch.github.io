"""Strict, local-only import of the official DouZero demo's three role networks.

External role weights are not integration training checkpoints. They get a
separate content-addressed manifest; existing checkpoint admission stays intact.
"""
from pathlib import Path
import json
import numpy as np
import torch

from .environment import ROLES, PIN, verify_checkout, digest
from .model import Policy
from .artifacts import HERE, sha256, write_json

ARCHITECTURE = {'hidden':512, 'depth':5, 'history':128}
FORMAT = 'douzero-official-demo-local-v1'


def contract_pins():
    verify_checkout()
    return {name:sha256(HERE/name) for name in ('source.lock.json','rules.lock.json','public-data-seeds.json')}


def load_roles(directory, expected=None):
    directory=Path(directory);policy=Policy(ARCHITECTURE).cpu().eval();records={}
    for role in ROLES:
        path=directory/(role+'.ckpt')
        if path.is_symlink() or not path.is_file() or path.stat().st_size>32*1024*1024:raise ValueError('Invalid role weight file: '+role)
        checksum=sha256(path)
        if expected and checksum!=expected[role]['sha256']:raise ValueError('Pretrained checkpoint hash changed: '+role)
        state=torch.load(path,map_location='cpu',weights_only=True)
        target=policy.roles[role].state_dict()
        if not isinstance(state,dict) or set(state)!=set(target):raise ValueError('Unexpected pretrained tensor keys: '+role)
        for name,tensor in state.items():
            if not torch.is_tensor(tensor) or tensor.shape!=target[name].shape or tensor.dtype!=target[name].dtype or not torch.isfinite(tensor).all():raise ValueError('Invalid pretrained tensor: '+role+'/'+name)
        policy.roles[role].load_state_dict(state,strict=True)
        records[role]={'file':path.name,'sha256':checksum,'bytes':path.stat().st_size,'tensor_elements':sum(t.numel() for t in state.values())}
    return policy,records


def import_demo(directory, source):
    """Seal provenance without modifying the downloaded role files."""
    directory=Path(directory);_,roles=load_roles(directory)
    manifest={'format':FORMAT,'source_commit':PIN,'contract_pins':contract_pins(),'architecture':ARCHITECTURE,
        'roles':roles,'source':source,'local_training_performed':False,
        'qualification':'Strict tensor/source compatibility; move-score and replay receipts kept separately. Not automatically admitted to frozen benchmark panels.'}
    manifest['identity']=digest(manifest)
    path=directory/'pretrained.json'
    if path.exists():raise FileExistsError('Pretrained import already sealed')
    write_json(path,manifest)
    return manifest


class PretrainedDouZero:
    def __init__(self,directory):
        self.directory=Path(directory).resolve();manifest=json.loads((self.directory/'pretrained.json').read_text())
        if manifest.get('format')!=FORMAT or manifest['source_commit']!=PIN or manifest['architecture']!=ARCHITECTURE:raise ValueError('Unsupported pretrained import')
        if manifest['contract_pins']!=contract_pins() or digest({k:v for k,v in manifest.items() if k!='identity'})!=manifest['identity']:raise ValueError('Pretrained manifest/source drift')
        self.policy,_=load_roles(self.directory,manifest['roles']);self.identity=manifest['identity'];self.manifest=manifest

    def values(self,observations):
        # Preserve the original duplicated-history FP32 forward arithmetic.
        return self.policy.values_many(observations,max_candidates=16384,factored=False,encoding_backend='direct')

    def choose(self,observations):
        return [int(np.argmax(q)) for q in self.values(observations)]
