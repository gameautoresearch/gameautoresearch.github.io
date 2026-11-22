"""Frozen tensor-graph inference process; submitted Python is never imported."""
import argparse
import json
from pathlib import Path
import struct
import sys


def main():
    protocol=sys.stdout.buffer;sys.stdout=sys.stderr
    import torch
    from ..policy_worker import frame_read
    from ...network import batch
    from .graph_export import load_graph,GraphPolicy
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);a=p.parse_args();root=Path(a.bundle)
    torch.set_num_threads(1)
    weights=torch.load(root/'weights.pt',map_location='cpu',weights_only=True)
    model=GraphPolicy(load_graph(root/'graph.json'),weights)
    while(observations:=frame_read(sys.stdin.buffer))is not None:
        actions=[]
        for obs in observations:
            x=batch([obs],'cpu')
            with torch.no_grad():logits,_=model(x)
            actions.append(int(logits.masked_fill(~x['mask'],-torch.inf).argmax(-1).item()))
        raw=json.dumps(actions).encode();protocol.write(struct.pack('>I',len(raw))+raw);protocol.flush()


if __name__=='__main__':main()
