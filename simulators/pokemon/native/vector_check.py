"""Ordinary Vector, rollout and tensor-export differential integration checks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..efficient import unpack_tensor
from ..env import Vector, WorkerError
from ..network import Policy
from .bridge import build
from .vector import NativeVector


def compare(a,b):
    for key in ('seed','seeds','steps','turn','terminated','truncated','reason','winner','returns','ready','potential','rejections'):
        if a[key]!=b[key]:raise AssertionError((key,a[key],b[key]))
    for side in range(2):
        actual=b['observations'][side].flat
        obs=a['observations'][side]
        expected=np.concatenate([np.asarray(obs[k]).ravel()for k in ('tokens','mons','global','global_tokens','action_tokens','actions','mask')])
        if not np.array_equal(actual,expected):
            where=np.flatnonzero(actual!=expected);raise AssertionError(('observation',side,[(int(i),float(actual[i]),float(expected[i]))for i in where[:12]]))


def run(output,generators,constructors,chart,checkpoint,library_path=None,team_size=6):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);library=Path(library_path)if library_path else build(output/'build')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=Policy(hidden=64,embedding=8);model.load_state_dict(cp['model']);model.eval()
    weights=output/'weights.pt';torch.save(model.state_dict(),weights)
    replica=Policy(hidden=64,embedding=8);replica.load_state_dict(torch.load(weights,weights_only=True));replica.eval()
    rows=[]
    for limit in (16,4096):
        with Vector(3,8,packed=True,binary=True,team_size=team_size)as oracle,NativeVector(library,generators,constructors,chart,workers=3,per_worker=8,developer_scope='Ordinary vector/tensor export differential contract; production admission closed',team_size=team_size)as native:
            seeds=list(range(1150000,1150024));swapped=[bool(i%2)for i in range(24)]
            expected=oracle.reset(seeds,limit,swapped);actual=native.reset(seeds,limit,swapped);phases=0
            while True:
                for a,b in zip(expected,actual,strict=True):compare(a,b)
                if all(s['terminated']or s['truncated']for s in expected):break
                flat=np.asarray([obs.flat for s in actual for obs in s['observations']],dtype=np.float32)
                with torch.no_grad():
                    x=unpack_tensor(torch.from_numpy(flat));logits,value=model(x);r_logits,r_value=replica(x)
                    if not torch.equal(logits,r_logits)or not torch.equal(value,r_value):raise AssertionError('Tensor-only export changed neural inference')
                    actions=logits.argmax(-1).reshape(24,2).tolist()
                choices=[[int(action)if state['ready'][side]else None for side,action in enumerate(pair)]for state,pair in zip(expected,actions,strict=True)]
                expected=oracle.step(choices);actual=native.step(choices);phases+=1
                if phases>4096:raise AssertionError('Ordinary vector stalled beyond natural bound')
            snapshot=native.snapshot();native.restore(snapshot)
            for a,b in zip(expected,native.states,strict=True):compare(a,b)
            for vector in(oracle,native):
                changed=vector.reset_slots([(1150100+i,bool(i%2))for i in range(24)],limit)
                if vector is oracle:replaced=changed
                else:
                    for a,b in zip(replaced,changed,strict=True):compare(a,b)
            rows.append(dict(limit=limit,phases=phases,natural=sum(s['terminated']for s in expected),truncations=sum(s['truncated']for s in expected),exact_all_700=True,exact_complete_vector_restore=True,exact_slot_resets=True))
    result=dict(full_gen9_admission=False,library_sha256=native.arena.binary_sha256,source_signature=native.arena.source_signature,**({}if team_size==6 else {'team_size':team_size}),scope='Ordinary Vector API and tensor-only neural export; no production source admission',exact_export_logits_and_values=True,checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),rows=rows)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--generators',required=True);p.add_argument('--constructors',required=True);p.add_argument('--chart',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--library');p.add_argument('--team-size',type=int,default=6);a=p.parse_args();run(a.output,a.generators,a.constructors,a.chart,a.checkpoint,a.library,a.team_size)
