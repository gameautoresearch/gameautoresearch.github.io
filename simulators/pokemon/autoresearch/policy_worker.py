"""Unprivileged candidate process: public observations in, legal actions out."""
import argparse
import importlib.util
import json
from pathlib import Path
import struct
import sys

def frame_read(stream):
    header=stream.read(4)
    if not header: return None
    if len(header)!=4: raise ValueError('Truncated frame')
    n=struct.unpack('>I',header)[0]
    if n>32*1024*1024: raise ValueError('Oversized request')
    value=stream.read(n)
    if len(value)!=n: raise ValueError('Truncated request')
    return json.loads(value)

def main():
    protocol=sys.stdout.buffer; sys.stdout=sys.stderr
    import torch
    from integrations.pokemon.network import batch
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);a=p.parse_args();root=Path(a.bundle)
    torch.set_num_threads(1)
    config=json.loads((root/'config.json').read_text())
    sys.path.insert(0,str(root))
    spec=importlib.util.spec_from_file_location('submitted_policy',root/'policy.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.create(config['network'])
    model.load_state_dict(torch.load(root/'weights.pt',map_location='cpu',weights_only=True));model.eval()
    while (observations:=frame_read(sys.stdin.buffer)) is not None:
        x=batch(observations,'cpu')
        with torch.no_grad():logits,_=model(x)
        if logits.shape!=x['mask'].shape or not torch.isfinite(logits[x['mask']]).all():raise ValueError('Invalid logits')
        ids=logits.masked_fill(~x['mask'],-torch.inf).argmax(-1).tolist()
        raw=json.dumps(ids).encode();protocol.write(struct.pack('>I',len(raw))+raw);protocol.flush()

if __name__=='__main__':main()
