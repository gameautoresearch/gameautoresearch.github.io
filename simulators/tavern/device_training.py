"""Experimental complete-rule GPU simulation + CUDA PPO qualification runner.

Game state, views, masks, actions and retained learning rows remain on CUDA. Only
small readiness/terminal metadata and explicit JSON checkpoints cross to the host.
A CPU transport backend uses the SAME collector/learner as a measured comparator.
No device fallback calls the CPU engine. Production gating requires retained parity.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import taverngym as tg
from device_env import DeviceEnv
from model import Policy, graph_action
from learner_graph import backward_graph, loss_terms

RULES={'num_players':8,'content':'Implemented','turn_action_caps':[30],
       'max_actions_per_turn':30,'max_turns':100,'record_log':False}
PPO={'learning_rate':.0003,'gamma':1.,'gae_lambda':.95,'ppo_epochs':3,
     'minibatch_size':1024,'clip':.2,'entropy_coef':.01,'value_coef':.5,'max_grad_norm':.5}


class CpuTransport:
    """Strong native reference; explicitly measured feature/action transfers."""
    def __init__(self,n_games,seed,controlled,config,**_):
        self.inner=tg.BatchEnv(n_games,seed=seed,controlled=controlled,config=config,threads=4)
        self.n_games=n_games;self.k=len(controlled);self.n_slots=n_games*self.k
        self.active=torch.ones(n_games,dtype=torch.uint8,device='cuda')
        self.kernels=DeviceEnv(1,kernels_only=True)
    def launch(self,*args):return self.kernels.launch(*args)
    def observe(self):
        self.active_host=self.active.cpu().numpy().astype(bool)
        o=self.inner.observe_ready(copy=False,active_games=self.active_host)
        slots=torch.from_numpy(o['slots'].copy()).cuda()
        ready=torch.zeros(self.n_slots,dtype=torch.bool,device='cuda');ready[slots]=True
        return {'features':torch.from_numpy(o['features'].copy()).cuda(),
                'cards':torch.from_numpy(o['cards'].copy()).cuda(),
                'action_mask':torch.from_numpy(o['action_mask'].copy()).cuda(),
                'ready':ready,'slots':slots}
    def step(self,actions):
        r,d,i=self.inner.step(actions.cpu().numpy(),active_games=self.active_host)
        return torch.from_numpy(r).cuda(),torch.from_numpy(d).cuda(),{k:torch.from_numpy(v).cuda() for k,v in i.items()}
    def save(self):return [self.inner.save(g) for g in range(self.n_games)]
    def close(self):pass


def synchronize():torch.cuda.synchronize()


def collect(model,n_games,seed,seats=(0,),backend='device',deterministic=False,max_steps=6000):
    """One complete natural episode per controlled seat, original slot/decision order."""
    cls=DeviceEnv if backend=='device' else CpuTransport
    start=time.perf_counter();env=cls(n_games,seed,controlled=seats,config=RULES)
    n=env.n_slots;finished=torch.zeros(n,dtype=torch.bool,device='cuda')
    invalid=torch.zeros(n_games,dtype=torch.bool,device='cuda')
    rewards=torch.zeros(n,device='cuda');rows=[];results=[];stats=Counter();phases=Counter()
    try:
        for step in range(max_steps):
            begin=time.perf_counter()
            env.active.copy_((~finished.reshape(n_games,len(seats)).all(1)).to(torch.uint8))
            o=env.observe();ids=(o['slots'] if 'slots' in o else
                torch.nonzero(o['ready']&~finished,as_tuple=False).flatten())
            # Dynamic compact row allocation synchronizes a row count; large buffers do not transfer.
            inputs=((o['features'],o['cards'].long(),o['action_mask'].bool()) if 'slots' in o else
                (o['features'].index_select(0,ids),o['cards'].index_select(0,ids).long(),
                 o['action_mask'].index_select(0,ids).bool()))
            synchronize();phases['observation_compaction']+=time.perf_counter()-begin
            begin=time.perf_counter()
            with torch.no_grad():
                if len(ids):
                    output=graph_action(model,inputs,deterministic)
                    if output is None:
                        dist,value=model.distribution(*inputs)
                        a=dist.logits.argmax(-1) if deterministic else dist.sample()
                        output=torch.stack((a.float(),dist.log_prob(a),value),1)
                    # Capture output is reused; retain independent rows for later PPO.
                    output=output.clone()
                else:output=torch.empty((0,3),device='cuda')
                actions=torch.zeros(n,dtype=torch.int64,device='cuda')
                actions.index_copy_(0,ids,output[:,0].long())
            rows.append((ids,inputs,output))
            synchronize();phases['actor']+=time.perf_counter()-begin
            begin=time.perf_counter()
            r,d,info=env.step(actions)
            if bool((info['rejected']&o['ready']).any()):raise RuntimeError('masked action rejected')
            rewards+=r*(~finished)
            new=d&~finished
            # Metadata only: placements, natural/truncated/faults stay distinct.
            records=torch.stack((torch.arange(n,device='cuda'),info['placement'].long(),
                                 info['stop_reason'].long(),new.long()),1).cpu().numpy()
            for slot,place,reason,ended in records:
                if ended:
                    results.append({'slot':int(slot),'game':int(slot//len(seats)),
                                    'seat':int(seats[slot%len(seats)]),'placement':int(place),'stop_reason':int(reason)})
            invalid|=(new&(info['stop_reason']!=1)).reshape(n_games,len(seats)).any(1)
            finished|=d
            stats['batch_steps']+=1;stats['decisions']+=len(ids)
            synchronize();phases['simulation_and_metadata']+=time.perf_counter()-begin
            if bool(finished.all()):break
        else:raise RuntimeError('incomplete cohort at external guard; no invented terminals')
        begin=time.perf_counter()
        slots=torch.cat([r[0] for r in rows]);order=torch.argsort(slots,stable=True)
        keep=~invalid.index_select(0,slots//len(seats));order=order[keep.index_select(0,order)]
        f,c,m=(torch.cat([r[1][j] for r in rows]).index_select(0,order) for j in range(3))
        outputs=torch.cat([r[2] for r in rows]).index_select(0,order)
        counts=torch.bincount(slots.index_select(0,order),minlength=n)
        offsets=torch.cat((torch.zeros(1,dtype=torch.int64,device='cuda'),counts.cumsum(0)))
        rr=torch.zeros(len(order),device='cuda')
        nonempty=counts>0;last=offsets[1:][nonempty]-1
        rr.index_copy_(0,last,rewards[nonempty])
        adv=torch.empty_like(rr);ret=torch.empty_like(rr)
        # One thread per trajectory; deterministic reverse f32 GAE, no CPU rewards/values.
        import ctypes as ct
        env.launch('tavern_gae',n,[outputs[:,2].contiguous(),rr,offsets,ct.c_uint32(n),
                                  ct.c_double(PPO['gamma']),ct.c_double(PPO['gae_lambda']),adv,ret])
        synchronize();phases['pack_and_gae']+=time.perf_counter()-begin
        states=env.save()
        phases['collection_wall_seconds']=time.perf_counter()-start
        if backend=='device':phases['allocator_statistics']=env.allocator_stats()
        stats.update(samples=len(order),natural_episodes=sum(r['stop_reason']==1 for r in results),
                     invalid_lobbies=int(invalid.sum().item()))
        return (f,c,m,outputs[:,0].long(),outputs[:,1],adv,ret),results,dict(stats),states,dict(phases)
    finally:
        env.close();phases['collection_wall_seconds']=time.perf_counter()-start


def update(model,optimizer,data,config=PPO):
    f,c,m,a,lp,adv,ret=data
    if len(f)<2:raise RuntimeError('insufficient valid natural samples')
    f=model.encode_numeric(f);adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    total=torch.zeros(6,device='cuda');batches=0
    before=torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
    for epoch in range(config['ppo_epochs']):
        for ids in torch.randperm(len(f),device='cuda').split(config['minibatch_size']):
            inputs=tuple(x[ids] for x in (f,c,m,a,lp,adv,ret))
            if len(ids)==config['minibatch_size']:
                terms=backward_graph(model,inputs,config,True)
            else:
                optimizer.zero_grad(set_to_none=False);terms=loss_terms(model,inputs,config,True)
                terms[0].backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'])
                terms=(*terms,grad)
            if not bool(torch.isfinite(terms[0])):raise RuntimeError('nonfinite PPO loss')
            optimizer.step();total+=torch.stack([t.detach() for t in terms]);batches+=1
    delta=(torch.cat([p.detach().flatten() for p in model.parameters()])-before).norm()
    return dict(zip(('loss','policy_loss','value_loss','entropy','approx_kl','grad_norm'),(total/batches).cpu().tolist()),
                parameter_delta_l2=float(delta),samples=len(f))


def source_hash():
    root=Path(__file__).parent;h=hashlib.sha256()
    paths=[*sorted((root/'device').rglob('*.rs')),*sorted((root/'device').rglob('*.json')),*sorted((root/'device').rglob('Cargo.toml')),root/'device/Cargo.lock',
           root/'device_env.py',root/'rollout.py',root/'device_training.py',root/'model.py',root/'learner_graph.py']
    for p in paths:
        if any(part=='target' or part.startswith('target-') for part in p.parts):continue
        h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    for p in sorted((root/'device').glob('tavern*.ptx'))+sorted((root/'device').glob('tavern*.cubin')):
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()


def runtime_options():
    return {'stack_bytes':int(os.environ.get('TAVERN_DEVICE_STACK_BYTES','65536')),
        'heap_bytes':2<<30,'arena_bytes':1<<20,'allocator':os.environ.get('TAVERN_DEVICE_ARENA','0'),
        'game_arena':os.environ.get('TAVERN_DEVICE_GAME_ARENA','1')!='0',
        'block_threads':int(os.environ.get('TAVERN_DEVICE_BLOCK_THREADS','32'))}


def save(path,model,optimizer,update_index,seed,backend,n_games,stats,states,accounted):
    obj={'schema':'tavern-device-ppo-v1','runtime_options':runtime_options(),'source_sha256':source_hash(),'rules':RULES,
         'backend':backend,'n_games':n_games,'seed':seed,'update':update_index,'policy':model.state_dict(),
         'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
         'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),
         'python_rng':random.getstate(),'stats':stats,'env_states':states,'accounted_seconds':accounted}
    temp=path.with_suffix('.tmp');torch.save(obj,temp);temp.replace(path)


def train(out,n_games=128,updates=4,seed=12345,backend='device',resume=None):
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=Policy().cuda();model.fast_inference=True;model.actor_graph_max_shapes=32;model.inference_metrics=Counter()
    optimizer=torch.optim.Adam(model.parameters(),lr=PPO['learning_rate'],eps=1e-5)
    first=0;prior=0.;history=[]
    if resume:
        ck=torch.load(resume,map_location='cuda',weights_only=False)
        if ck['runtime_options']!=runtime_options() or ck['source_sha256']!=source_hash() or ck['rules']!=RULES or ck['n_games']!=n_games or ck['backend']!=backend or ck['seed']!=seed:
            raise ValueError('source/PTX/rules/worker trajectory mismatch; checkpoint cannot be resumed')
        model.load_state_dict(ck['policy']);optimizer.load_state_dict(ck['optimizer']);first=ck['update'];prior=ck['accounted_seconds']
        torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
        np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng']);history=list(ck['stats'])
    states=[]
    for u in range(first,updates):
        begin=time.perf_counter()
        data,results,stats,states,phases=collect(model,n_games,(seed+u*1000003)%2**64,backend=backend)
        learned=update(model,optimizer,data);synchronize()
        history.append({'update':u+1,'seconds':time.perf_counter()-begin,'collection':stats,'phases':phases,'learning':learned,'episodes':results})
        save(out/'latest.pt',model,optimizer,u+1,seed,backend,n_games,history,states,prior+time.perf_counter()-start)
        (out/'training.json').write_text(json.dumps({'backend':backend,'n_games':n_games,'source_sha256':source_hash(),
            'worker_seconds':time.perf_counter()-start,'history':history},indent=2)+'\n')
    return history


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--backend',choices=['device','cpu'],default='device');p.add_argument('--n-games',type=int,default=128)
    p.add_argument('--updates',type=int,default=4);p.add_argument('--seed',type=int,default=12345);p.add_argument('--resume',type=Path)
    a=p.parse_args();train(a.out,a.n_games,a.updates,a.seed,a.backend,a.resume)


if __name__=='__main__':main()
