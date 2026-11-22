"""Exact uninterrupted/resumed optimizer, RNG, population and admission proof."""
import argparse,json
from pathlib import Path
from .bootstrap import write
from .train import run
import numpy as np
import torch

def equal(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def validate(config,root,seconds=120):
 root=Path(root);root.mkdir(parents=True,exist_ok=False)
 # Identical config on both paths; limit_updates is execution scheduling only.
 full=run(config,root/'full',seconds,limit_updates=4)
 half=run(config,root/'half',seconds,limit_updates=2)
 resumed=run(config,root/'resumed',seconds,resume=half,limit_updates=2)
 a=torch.load(full,weights_only=False,map_location='cpu');b=torch.load(resumed,weights_only=False,map_location='cpu')
 keys=['policy','optimizer','rng','population','history','env_states','next_cohort_seed','update','binding','config']
 result={k:equal(a[k],b[k]) for k in keys}
 assert all(result.values()),result
 bad=root/'wrong-identity.pt';b['binding']['native_source_hash']='old-identity';torch.save(b,bad)
 try:run(config,root/'must-reject',seconds,resume=bad,limit_updates=1)
 except ValueError as exc:result['old_identity_rejected']='Source/interface/recipe differs' in str(exc)
 else:raise AssertionError('Old identity admitted')
 result['all_passed']=all(result.values());write(root/'validation.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(validate(json.loads(a.config.read_text()),a.out))
