"""Device-resident simulator, ragged inference, legal sampling and DMC trajectory rings."""
from collections import defaultdict
import time
import numpy as np
import torch

from ..environment import ROLES,Game,digest,action_bits
from ..native_rollout import ArraySamples
from .backend import CudaBatch
from .actor_rng import actor_seeds


class DeviceSamples(ArraySamples):
    def export_cpu(self):
        return ArraySamples(self.x.cpu().numpy(),self.z.cpu().numpy(),self.y.cpu().numpy())


def collect_device(vector,policies,cards,assignments,rng,objective,epsilon,budget,record,training):
    timing=defaultdict(float)
    started=time.perf_counter()
    n=len(cards)
    t=time.perf_counter()
    if vector.cuda is None or vector.cuda.n!=n:
        if vector.cuda is not None:vector.cuda.close()
        vector.cuda=CudaBatch(n)
    batch=vector.cuda
    batch.reset(cards)
    device=batch.device
    # Immutable assignment groups and domain-separated per-game actor streams.
    groups={}
    kinds={}
    learners={}
    pure_learner={}
    for role in ROLES:
        lookup={}
        tags=np.zeros(n,dtype=np.int8)
        for i,assignment in enumerate(assignments):
            descriptor=assignment[role]
            policy=policies.resolve(descriptor)
            kind=descriptor['kind']
            tags[i]={'learner':1,'random':2,'heuristic':3,'checkpoint':4}[kind]
            if kind in ('learner','checkpoint'):
                lookup.setdefault(id(policy),(policy,[]))[1].append(i)
        groups[role]=[(policy,torch.as_tensor(np.isin(np.arange(n),ids),device=device)) for policy,ids in lookup.values()]
        kinds[role]=torch.as_tensor(tags,device=device)
        learners[role]=(kinds[role]==1).to(torch.uint8)
        pure_learner[role]=bool(np.all(tags==1))
    seeds=actor_seeds(rng,n)
    seeds=torch.as_tensor(seeds,device=device)
    if training or record:
        xring=torch.zeros((162,n,484),dtype=torch.int8,device=device)
        zring=torch.zeros((162,n,5,162),dtype=torch.int8,device=device)
        valid=torch.zeros((162,n),dtype=torch.uint8,device=device)
        choices_log=torch.zeros((162,n),dtype=torch.int64,device=device)
    timing['reset_simulation_ipc_seconds']+=time.perf_counter()-t
    decisions=candidates=max_legal=0
    role_decisions={r:0 for r in ROLES}
    turns=0
    for turn in range(163):
        if budget and budget.expired():
            return {'aborted':True,'decisions':decisions,'candidates':candidates,'timing':dict(timing),
                'samples':{r:[] for r in ROLES},'outcomes':{},'traces':{},'partial_completed_games':0}
        t=time.perf_counter()
        total,maximum,active=batch.generate()
        timing['simulation_observation_ipc_seconds']+=time.perf_counter()-t
        if total==0:break
        if turn>=162:raise RuntimeError('CUDA finite shedding-game guard; no outcomes scored')
        role=ROLES[turn%3]
        width=319 if role=='landlord' else 430
        t=time.perf_counter()
        owners,local,actions,starts=batch.compact(total)
        q=torch.zeros(total,device=device)
        for policy,mask in groups[role]:
            if len(groups[role])==1 and pure_learner[role]:
                selected=None
                score_actions,score_owners=actions,owners
            else:
                selected=torch.nonzero(mask[owners],as_tuple=True)[0]
                if not len(selected):continue
                score_actions,score_owners=actions[selected],owners[selected]
            values=policy.values_device(role,batch.features[:,:width],batch.features[:,width:width+810].reshape(n,5,162),
                                       score_actions,score_owners,policies.max_candidates)
            if selected is None:q=values
            else:q[selected]=values
        greedy=batch.argmax(q,starts)
        choices=batch.choose(greedy,kinds[role],seeds,epsilon)
        timing['encoding_transfer_inference_action_seconds']+=time.perf_counter()-t
        t=time.perf_counter()
        if training or record:
            batch.store(choices,xring[turn],zring[turn],valid[turn])
            choices_log[turn].copy_(choices)
            if training:valid[turn].bitwise_and_(learners[role])
        timing['training_sample_encoding_seconds']+=time.perf_counter()-t
        t=time.perf_counter()
        batch.step(choices)
        timing['simulation_observation_ipc_seconds']+=time.perf_counter()-t
        decisions+=active
        role_decisions[role]+=active
        candidates+=total
        max_legal=max(max_legal,maximum)
        turns=turn+1
    flags=batch.outcomes()
    if not np.all(flags[:,0]==1):raise RuntimeError('CUDA incomplete natural wave')
    outcomes={}
    for i,(_,won,bombs,steps) in enumerate(flags):
        won,bombs,steps=bool(won),int(bombs),int(steps)
        sign=1 if won else -1
        unit=2**bombs
        target={'wp':1,'adp':unit,'logadp':bombs+1}[objective]
        outcomes[i]={'landlord_won':won,'bombs':bombs,'decisions':steps,
            'returns':{r:float(sign*target*(1 if r=='landlord' else -1)) for r in ROLES},
            'scores':{r:sign*unit*(2 if r=='landlord' else -1) for r in ROLES},
            'terminal':'natural_empty_hand','truncated':False}
    samples={r:[] for r in ROLES}
    if training:
        sign=torch.where(batch.flags[:,1].bool(),1.,-1.)
        if objective=='wp':magnitude=torch.ones(n,device=device)
        elif objective=='adp':magnitude=torch.bitwise_left_shift(torch.ones(n,dtype=torch.int64,device=device),batch.flags[:,2]).float()
        else:magnitude=batch.flags[:,2].float()+1.
        for r,role in enumerate(ROLES):
            width=373 if r==0 else 484
            mask=valid[r:turns:3].T.bool()
            game,row=torch.nonzero(mask,as_tuple=True)
            if not len(game):continue
            steps=row*3+r
            x=xring[steps,game,:width].clone()
            z=zring[steps,game].clone()
            returns=sign*magnitude*(1. if r==0 else -1.)
            samples[role]=DeviceSamples(x,z,returns[game])
    traces={}
    if record:
        limit=vector.record_limit if vector.record_limit is not None else n
        take=min(n,limit)
        xx=xring[:turns,:take].cpu().numpy()
        zz=zring[:turns,:take].cpu().numpy()
        selected=choices_log[:turns,:take].cpu().numpy()
        for i in range(take):
            game=Game(cards[i],objective,record=True,engine_backend='upstream')
            for turn in range(outcomes[i]['decisions']):
                obs=game.observe()
                index=int(selected[turn,i])
                x,z,act=obs.encode(backend='upstream')
                np.testing.assert_array_equal(xx[turn,i,:len(x)],x)
                np.testing.assert_array_equal(xx[turn,i,len(x):len(x)+54],act[index])
                np.testing.assert_array_equal(zz[turn,i],z)
                game.step(index)
            if game.outcome()!=outcomes[i]:raise AssertionError('GPU/CPU replay outcome mismatch')
            traces[i]=game.trajectory()
    timing['rollout_total_seconds']=time.perf_counter()-started
    return {'aborted':False,'samples':samples,'outcomes':outcomes,'traces':traces,'decisions':decisions,
        'candidates':candidates,'role_decisions':role_decisions,'max_legal_candidates':max_legal,
        'timing':dict(timing),'deal_sha256':[digest(c) for c in cards],
        'device_resident':True,'actor_rng':'splitmix-counter-v1',
        'explicit_turn_control_bytes':32*(turns+1)}
