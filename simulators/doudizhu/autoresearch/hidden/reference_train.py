"""Host-private reference: stronger network, annealing and a small in-memory league."""
import argparse
import copy
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
    def __init__(self,seconds):self.start=time.monotonic();self.seconds=seconds
    def expired(self):return time.monotonic()-self.start>=self.seconds
    def fraction(self):return min(1.,(time.monotonic()-self.start)/max(.01,self.seconds))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--seconds',type=float,required=True);a=p.parse_args()
    c=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(a.seed);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    rng=np.random.default_rng(a.seed);policy=network.create(c['architecture']).to('cuda:0');policy.compute_dtype='bfloat16'
    optimizer=learner.make_optimizer(policy,c);policies=Policies(policy,'cuda:0',max_candidates=16384);policies.actor_rng='splitmix-counter-v1'
    deadline=Deadline(max(.01,a.seconds-2));games=steps=waves=0;pool=[]
    (out/'model.json').write_text(json.dumps(c['architecture']))
    def save():
        path=out/'weights.pt.tmp';torch.save(policy.state_dict(),path);path.replace(out/'weights.pt')
    save()
    with VectorGames(0,engine_backend='cuda') as env:
        while not deadline.expired():
            n=c['vector_games'];cards=[deal('train',a.seed,games+i) for i in range(n)];assignments=[]
            for i in range(n):
                if not pool or rng.random()<.7:assignments.append({r:{'kind':'learner'} for r in ROLES})
                else:
                    selected=pool[int(rng.integers(len(pool)))];role=ROLES[(games+i)%3]
                    assignments.append({r:{'kind':'learner'} if r==role else selected for r in ROLES})
            frac=deadline.fraction()
            for optimizer_role in optimizer.values():
                for group in optimizer_role.param_groups:group['lr']=c['learning_rate']*(1-.95*frac)
            batch=collect(env,policies,cards,assignments,rng,epsilon=.2-.18*frac,budget=deadline,training=True)
            if batch['aborted']:break
            metrics=learner.learn(policy,optimizer,batch['samples'],rng,c);policy.eval()
            games+=n;steps+=batch['decisions'];waves+=1;save()
            if waves==1 or waves%400==0:
                descriptor={'kind':'checkpoint','path':f'league-{waves}','sha256':f'{waves:064x}'}
                policies.cache[(descriptor['path'],descriptor['sha256'])]=copy.deepcopy(policy).eval();pool.append(descriptor)
                if len(pool)>4:
                    old=pool.pop(0);policies.cache.pop((old['path'],old['sha256']))
            row={'games':games,'decisions':steps,'waves':waves,'mse':metrics['mse'],'natural_games':games,'truncations':0,'faults':0}
            with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    save()

if __name__=='__main__':main()
