"""Unprivileged policy process. Only the wire's public tensors enter this process."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

def main():
    from .wire import read_frame,decode,reply
    import torch
    import numpy as np
    import random
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);a=p.parse_args()
    root=Path(a.bundle)
    sys.path.insert(0,str(root))
    torch.set_num_threads(1)
    torch.manual_seed(0);np.random.seed(0);random.seed(0)
    config=json.loads((root/'model.json').read_text())
    spec=importlib.util.spec_from_file_location('entrant_policy',root/'policy.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    policy=module.create(config)
    if not isinstance(policy,torch.nn.Module):raise ValueError('create() must return a torch.nn.Module')
    if sum(p.numel() for p in policy.parameters())+sum(p.numel() for p in policy.buffers())>10_000_000:
        raise ValueError('Constructed policy exceeds tensor cap')
    policy.load_state_dict(torch.load(root/'weights.pt',map_location='cpu',weights_only=True),strict=True)
    policy.eval()
    while (raw:=read_frame(sys.stdin.buffer)) is not None:
        role,x,z,actions,counts=decode(raw)
        with torch.inference_mode():indices=module.act(policy,role,x,z,actions,counts)
        values=list(indices)
        if len(values)!=len(counts) or any(isinstance(i,bool) or int(i)!=i or not 0<=int(i)<int(n) for i,n in zip(values,counts)):
            raise ValueError('Illegal policy response')
        sys.stdout.buffer.write(reply(values));sys.stdout.buffer.flush()

if __name__=='__main__':main()
