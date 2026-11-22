"""Vectorized public-feature DMC rollout: native batch -> ragged head -> array samples.

One Python control iteration per whole batch turn. All roles remain independent,
all actions remain legal, every selected action receives its natural terminal return.
The policy receives only public x/z/candidate arrays, never the native batch handle.
"""
from collections import defaultdict
import time
import numpy as np
from .environment import ROLES,digest


class ArraySamples:
    def __init__(self,x,z,y):
        self.x,self.z,self.y=x,z,y

    def __len__(self):
        return len(self.y)

    def __iter__(self):
        return iter(zip(self.x,self.z,self.y))


def collect_native(vector,policies,cards,assignments,rng,objective,epsilon,budget,record,training):
    native=vector.native
    timing=defaultdict(float)
    started=time.perf_counter()
    t=time.perf_counter()
    native.reset(cards,objective,record)
    ids=np.arange(len(cards),dtype=np.int32)
    counter=getattr(policies,'actor_rng','numpy-pcg64-legacy')=='splitmix-counter-v1'
    turn=0
    if counter:
        from .cuda.actor_rng import actor_seeds,explore
        seeds=actor_seeds(rng,len(cards))
    timing['reset_simulation_ipc_seconds']+=time.perf_counter()-t
    pending={r:[] for r in ROLES}
    frame=native.arrays(ids)
    outcomes={}
    decisions=candidates=max_legal=0
    role_decisions={r:0 for r in ROLES}
    while len(ids):
        if budget and budget.expired():
            return {'aborted':True,'decisions':decisions,'candidates':candidates,'timing':dict(timing),
                    'samples':{r:[] for r in ROLES},'outcomes':{},'traces':{},'partial_completed_games':len(outcomes)}
        t=time.perf_counter()
        meta,features=frame['meta'],frame['features']
        roles=meta[:,0]
        if not np.all(roles==roles[0]):
            raise AssertionError('Native shedding wave lost synchronous role boundaries')
        role=ROLES[int(roles[0])]
        width=319 if role=='landlord' else 430
        counts=meta[:,3].astype(np.int64)
        starts=meta[:,2].astype(np.int64)
        q=policies.learner.values_arrays(role,features[:,:width],features[:,width:width+810].reshape(-1,5,162),
                                        frame['actions'],counts,policies.max_candidates)
        best=np.maximum.reduceat(q,starts)
        indices=np.arange(len(q),dtype=np.int64)
        winners=np.minimum.reduceat(np.where(q==np.repeat(best,counts),indices,len(q)),starts)-starts
        choices=winners.astype(np.int32)
        # Preserve the old epsilon stream and draw order exactly. Only the small RNG
        # scalar calls remain per game; simulator, features, scoring and samples batch.
        if counter:
            choices=explore(seeds[ids],turn,counts,choices,epsilon)
        elif epsilon:
            for j,index in enumerate(ids):
                local=rng[int(index)] if isinstance(rng,dict) else rng
                if local.random()<epsilon:
                    choices[j]=int(local.integers(counts[j]))
        timing['encoding_transfer_inference_action_seconds']+=time.perf_counter()-t
        n=len(ids)
        decisions+=n
        candidates+=int(counts.sum())
        role_decisions[role]+=n
        max_legal=max(max_legal,int(counts.max()))
        t=time.perf_counter()
        if training:
            x=np.concatenate((features[:,:width],frame['actions'][starts+choices]),axis=1)
            z=features[:,width:width+810].reshape(-1,5,162).copy()
            pending[role].append((ids.copy(),x,z))
        timing['training_sample_encoding_seconds']+=time.perf_counter()-t
        t=time.perf_counter()
        after=native.advance_arrays(ids,choices)
        terminal=after['meta'][:,6].astype(bool)
        for j in np.flatnonzero(terminal):
            i=int(ids[j])
            _,_,_,_,bombs,steps,_,won,_=map(int,after['meta'][j])
            sign=1 if won else -1
            unit=2**bombs
            target={'wp':1,'adp':unit,'logadp':bombs+1}[objective]
            outcomes[i]={'landlord_won':bool(won),'bombs':bombs,'decisions':steps,
                'returns':{r:float(sign*target*(1 if r=='landlord' else -1)) for r in ROLES},
                'scores':{r:sign*unit*(2 if r=='landlord' else -1) for r in ROLES},
                'terminal':'natural_empty_hand','truncated':False}
            if i in native.audit:
                assert outcomes[i]==native.audit[i].outcome()
        ids=ids[~terminal]
        turn+=1
        frame=dict(after,ids=ids,meta=after['meta'][~terminal],features=after['features'][~terminal])
        timing['simulation_observation_ipc_seconds']+=time.perf_counter()-t
    samples={r:[] for r in ROLES}
    if training:
        for role,parts in pending.items():
            if not parts:
                continue
            game_ids=np.concatenate([a[0] for a in parts])
            order=np.argsort(game_ids,kind='stable')
            x=np.concatenate([a[1] for a in parts])[order]
            z=np.concatenate([a[2] for a in parts])[order]
            y=np.array([outcomes[int(i)]['returns'][role] for i in game_ids[order]],dtype=np.float32)
            samples[role]=ArraySamples(x,z,y)
    traces={i:game.trajectory() for i,game in native.audit.items()}
    timing['rollout_total_seconds']=time.perf_counter()-started
    return {'aborted':False,'samples':samples,'outcomes':outcomes,'traces':traces,'decisions':decisions,
            'candidates':candidates,'role_decisions':role_decisions,'max_legal_candidates':max_legal,
            'timing':dict(timing),'deal_sha256':[digest(c) for c in cards]}
