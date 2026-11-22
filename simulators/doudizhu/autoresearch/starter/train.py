"""Editable, deliberately plain self-play starter. No history pool/research switches."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import policy as network
import learner
from integrations.doudizhu.environment import ROLES,deal
from integrations.doudizhu.rollout import Policies,collect
from integrations.doudizhu.workers import VectorGames

class Deadline:
    def __init__(self,seconds):self.end=time.monotonic()+seconds
    def expired(self):return time.monotonic()>=self.end

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--seconds',type=float,required=True)
    a=p.parse_args();c=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(a.seed);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    rng=np.random.default_rng(a.seed)
    policy=network.create(c['architecture']).to('cuda:0');policy.compute_dtype='bfloat16'
    optimizer=learner.make_optimizer(policy,c)
    policies=Policies(policy,'cuda:0',max_candidates=16384);policies.actor_rng='splitmix-counter-v1'
    deadline=Deadline(max(0.,a.seconds-2));games=steps=waves=0
    (out/'model.json').write_text(json.dumps(c['architecture']))
    def save():
        path=out/'weights.pt.tmp';torch.save(policy.state_dict(),path);path.replace(out/'weights.pt')
    save()
    initial_sha=hashlib.sha256((out/'weights.pt').read_bytes()).hexdigest()
    with VectorGames(0,engine_backend='cuda') as env:
        while not deadline.expired():
            n=c['vector_games'];cards=[deal('train',a.seed,games+i) for i in range(n)]
            assignments=[{r:{'kind':'learner'} for r in ROLES} for _ in cards]
            batch=collect(env,policies,cards,assignments,rng,epsilon=c['epsilon'],budget=deadline,training=True)
            if batch['aborted']:break
            metrics=learner.learn(policy,optimizer,batch['samples'],rng,c)
            policy.eval();games+=n;steps+=batch['decisions'];waves+=1
            save()
            row={'games':games,'decisions':steps,'waves':waves,'mse':metrics['mse'],'updates':metrics['updates'],'weights_changed':hashlib.sha256((out/'weights.pt').read_bytes()).hexdigest()!=initial_sha,'initial_weights_sha256':initial_sha,'weights_sha256':hashlib.sha256((out/'weights.pt').read_bytes()).hexdigest(),'natural_games':games,'truncations':0,'faults':0}
            with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    save()

if __name__=='__main__':main()
