"""Strict admitted neural inference bundle; explicit warmstart lineage is retained."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
from .common import HERE,admission,binding,sha,write

def export(checkpoint,out):
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
 if ck['binding']!=binding():raise ValueError('Frozen reference source differs')
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 arrays={k:v.detach().cpu().numpy() for k,v in ck['policy'].items()}
 if sum(v.size for v in arrays.values())>2000000 or not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Invalid state')
 for source,name in [(HERE/'policy.py','policy.py'),(HERE/'learner.py','learner.py'),(HERE/'train.py','train.py')]:shutil.copy2(source,out/name)
 np.savez(out/'weights.npz',**arrays)
 identity=dict(binding=admission(),source={f:sha(out/f) for f in ['policy.py','learner.py','train.py']},config=ck['config'],reference_lineage=dict(schema='tavern-strong-population-v1',checkpoint_sha256=sha(checkpoint),update=ck['update'],reference_binding=ck['binding'],parent=ck['lineage'],population=[{k:v for k,v in e.items() if k!='policy'} for e in ck['population']['entries']],inference='visible-only feedforward masked neural policy; no search or LLM'))
 write(out/'identity.json',identity)
 write(out/'summary.json',dict(identity_sha256=sha(out/'identity.json'),weights_sha256=sha(out/'weights.npz'),updates=ck['update'],samples=sum(h['learning']['samples'] for h in ck['history']),parameters=sum(v.size for v in arrays.values()),finite_losses=all(np.isfinite(h['learning']['loss']) for h in ck['history']),changed_weights=any(h['learning']['parameter_delta_l2']>0 for h in ck['history']),entrant_size_compatible=True,strong_qualified=False,parent_update=ck['lineage']['parent_update']))
 for f in out.iterdir():f.chmod(0o444)
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(export(a.checkpoint,a.out))
