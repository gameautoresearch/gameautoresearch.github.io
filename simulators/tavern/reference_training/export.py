"""Finite inference-only artifact with exact admission and training lineage."""
import argparse,json,shutil,sys
from pathlib import Path
from .bootstrap import GAME,HERE,ROOT,binding,sha,write
import numpy as np
import torch

def export(checkpoint,out):
 sys.path.insert(0,str(GAME/'autoresearch'))
 from contract import binding as admission_binding
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
 if ck['binding']!=binding():raise ValueError('Reference source identity mismatch')
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 arrays={k:v.detach().cpu().numpy() for k,v in ck['policy'].items()}
 if sum(v.size for v in arrays.values())>2000000 or not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Invalid finite/size export')
 for source,name in [(GAME/'model.py','policy.py'),(GAME/'run.py','learner.py'),(HERE/'train.py','train.py')]:shutil.copy2(source,out/name)
 np.savez(out/'weights.npz',**arrays)
 identity=dict(binding=admission_binding(),source={f:sha(out/f) for f in ['policy.py','learner.py','train.py']},config=ck['config'],reference_lineage=dict(checkpoint_sha256=sha(checkpoint),update=ck['update'],reference_binding=ck['binding'],population=[{k:v for k,v in e.items() if k!='policy'} for e in ck['population']['entries']],inference='feedforward masked neural policy; no search or test-time LLM'))
 write(out/'identity.json',identity)
 write(out/'summary.json',dict(identity_sha256=sha(out/'identity.json'),weights_sha256=sha(out/'weights.npz'),updates=ck['update'],samples=sum(r['learning']['training_samples'] for r in ck['history']),parameters=sum(v.size for v in arrays.values()),finite_losses=all(np.isfinite(r['learning']['loss']) for r in ck['history']),changed_weights=any(r['learning']['parameter_delta_l2']>0 for r in ck['history']),entrant_size_compatible=True,strong_qualified=False))
 # Read through the isolated candidate's strict architecture/NPZ path.
 from model import Policy
 m=Policy(**ck['architecture']);m.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in arrays.items()},strict=True)
 for p in out.iterdir():p.chmod(0o444)
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(export(a.checkpoint,a.out))
