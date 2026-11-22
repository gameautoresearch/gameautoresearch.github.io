"""Audit an RL-owner export against its exact archived training source/lineage."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from .budget import ROOT
from .policies import admit_bundle
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'autoresearch'))
from contract import sha,write,validate

def audit_export(bundle,runtime_source,out,checkpoint=None):
    bundle=Path(bundle);runtime=Path(runtime_source);ident,summary=admit_bundle(bundle);validate(ident['config']);lineage=ident['reference_lineage'];rb=lineage['reference_binding']
    if rb['native_source_hash']!='217577b83fc4525e' or rb['observations']!=[3227,124,245]:raise ValueError('Reference native/interface differs')
    for name,digest in rb['files'].items():
        path=runtime/name if not name.startswith('vendor/') else ROOT/name
        if not path.is_file() or sha(path)!=digest:raise ValueError('Archived reference source drift: '+name)
    source_map={'policy.py':'simulators/tavern/model.py','learner.py':'simulators/tavern/run.py','train.py':'simulators/tavern/reference_training/train.py'}
    if any(sha(bundle/name)!=rb['files'][source] for name,source in source_map.items()):raise ValueError('Export did not preserve exact training source modules')
    checkpoint_verified=False
    if checkpoint:
        checkpoint=Path(checkpoint)
        if sha(checkpoint)!=lineage['checkpoint_sha256']:raise ValueError('Export checkpoint digest mismatch')
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if ck['binding']!=rb or ck['update']!=lineage['update']:raise ValueError('Checkpoint lineage mismatch')
        with np.load(bundle/'weights.npz',allow_pickle=False) as arrays:
            if set(arrays.files)!=set(ck['policy']) or any(not np.array_equal(arrays[k],ck['policy'][k].cpu().numpy()) for k in arrays.files):raise ValueError('Export transformed selected weights')
        checkpoint_verified=True
    result=dict(admission_qualified=True,native_fingerprint=rb['native_source_hash'],exact_inference_source_preserved=True,full_training_source_archive_verified=True,checkpoint_digest_and_weights_verified=checkpoint_verified,source_bound_checkpoint_migration=False,export=str(bundle.resolve()),runtime_source=str(runtime.resolve()),identity_sha256=sha(bundle/'identity.json'),weights_sha256=sha(bundle/'weights.npz'),update=lineage['update'],samples=summary['samples'],parameters=summary['parameters'],finite_learning_metrics=summary['finite_losses'],changed_weights=summary['changed_weights'],strong_panel_qualified=False,superhuman_evidence=False)
    write(out,result);return result

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path);a=p.parse_args();print(json.dumps(audit_export(a.bundle,a.source,a.out,a.checkpoint)))
