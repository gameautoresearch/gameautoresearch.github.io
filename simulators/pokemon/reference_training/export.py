"""Export trusted campaign-generated tensors after exact worker/source receipt checks."""
import hashlib,json
from pathlib import Path
import torch
from .remote import BASE
from .trainer import atomic,state_hash
from ..network import Policy
from ..provenance import digest

def export(label):
    job=BASE/'jobs'/label;result=json.loads((job/'collected/worker-result.json').read_text());allocation=json.loads((job/'allocation.json').read_text())
    path=job/'collected/training/checkpoint.pt';metadata=json.loads((Path(json.loads((job/'reservation.json').read_text())['source_release'])/'release.json').read_text())
    if result['status']!='complete' or not result['training']['source_admitted'] or not allocation['finished_at']:raise ValueError('Completed source-admitted training/allocation required')
    if digest(path)!=result['training']['checkpoint_sha256']:raise ValueError('Export transport hash changed')
    cp=torch.load(path,map_location='cpu',weights_only=False)
    if cp['reference_source_signature']!=metadata['reference_source_signature'] or cp['executable_signature']!=metadata['canonical_executable_signature']:raise ValueError('Immutable source lineage mismatch')
    if state_hash(cp['model'])!=cp['model_sha256']:raise ValueError('Tensor identity mismatch')
    model=Policy(**cp['network']);model.load_state_dict(cp['model'],strict=True)
    out=BASE/'exports'/label;out.mkdir(parents=True,exist_ok=False);torch.save(cp['model'],out/'weights.pt')
    # Re-import through tensor-only loader and strict neural architecture.
    replica=Policy(**cp['network']);replica.load_state_dict(torch.load(out/'weights.pt',map_location='cpu',weights_only=True),strict=True)
    if state_hash(replica.state_dict())!=cp['model_sha256']:raise ValueError('Export tensor parity failed')
    record={'schema':'pokemon-reference-inference-v1','network':cp['network'],'weights_sha256':digest(out/'weights.pt'),'model_sha256':cp['model_sha256'],'checkpoint_sha256':digest(path),'source_release':metadata,'recipe':cp['config'],'training_iteration':cp['iteration'],'format':'gen9randombattle','observation_values':700,'legal_actions':14,'initialization':'random; no teacher/imported weights','inference':'Policy only, greedy masked neural logits; no search','resume_artifact':str(path),'qualification':'learned diagnostic baseline; independent evaluator-v2 and strength admission remain separate'}
    atomic(out/'policy.json',record);return {'path':str(out),'weights_sha256':record['weights_sha256'],'model_sha256':record['model_sha256']}
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('label');a=p.parse_args();print(json.dumps(export(a.label),indent=2))
