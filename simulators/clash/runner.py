"""Game-local CLI: masked PPO, paired evaluation, and complete-loop profiling."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,random,time,os
from collections import Counter,defaultdict
from contextlib import contextmanager
from pathlib import Path
_WORKER_START=time.perf_counter()
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from game import HERE,ROOT,WorkerClock,Session,canonical,dump,load_config,make_vector,new_run,resolve,sha,status,ScriptedBot
from model import Policy
from actor import sample_actions,predict_values,evaluation_probabilities

class Timer:
    def __init__(self,device,synchronize=True):self.device=device;self.synchronize=synchronize;self.seconds=defaultdict(float);self.calls=Counter()
    def sync(self):
        if not self.synchronize:return
        if self.device=='cuda':torch.cuda.synchronize()
        elif self.device=='mps':torch.mps.synchronize()
    @contextmanager
    def section(self,name):
        self.sync();t=time.perf_counter()
        try:yield
        finally:self.sync();self.seconds[name]+=time.perf_counter()-t;self.calls[name]+=1
    def report(self):return {k:{'seconds':v,'calls':self.calls[k]} for k,v in self.seconds.items()}

def setup(c):
    torch.set_num_threads(c['torch_threads']);torch.manual_seed(c['seed']);random.seed(c['seed']);np.random.seed(c['seed'])
    if c['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested, unavailable; refusing silent CPU fallback')
    if c['hardware_allocation']['gpu_count']:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('declared allocation requires exactly one visible CUDA GPU')
        if torch.cuda.get_device_name()!=c['hardware_allocation']['gpu_model']:raise RuntimeError('visible GPU differs from specified allocation')
    if c['device']=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS requested, unavailable')
    torch.use_deterministic_algorithms(c.get('deterministic',True))
    return np.random.default_rng(c['seed'])

def network(v,c):
    model=Policy(v.schema,v.vocabulary_size,len(v.buff_names),c['width'],c['embedding']).to(c['device'])
    model.entity_bucket=c.get('entity_bucket',0);model.pinned_transfer=c.get('pinned_transfer',False)
    model.device_indices=c.get('device_indices',False);model.graph_inference=c.get('graph_inference',False)
    model.ragged_entities=c.get('ragged_entities',False);model.sparse_policy_head=c.get('sparse_policy_head',False)
    model.inference_graph_cache_size=c.get('inference_graph_cache_size',2);model.graph_sampling=c.get('graph_sampling',False)
    model.transfer_cache_limit_bytes=c.get('transfer_cache_limit_bytes',256*1024*1024)
    if (model.ragged_entities or model.sparse_policy_head) and c['device']!='cpu':
        raise ValueError('ragged_entities/sparse_policy_head are qualified for CPU only; CUDA uses static graph inference')
    if c.get('compile_forward',False):model.forward=torch.compile(model.forward,fullgraph=True,dynamic=False)
    return model

def sample(model,obs,masks,c,timer,greedy=False):
    chunk=c.get('actor_chunk_size',0)
    if chunk and len(obs)>chunk:
        # Independent rows have no batch normalization or cross-game attention.
        # Chunking bounds graph/input memory; the changed sampler call partition
        # is recorded in the recipe and is a separate numerical/RNG backend.
        pieces=[sample(model,obs[start:start+chunk],masks[start:start+chunk],dict(c,actor_chunk_size=0),timer,greedy) for start in range(0,len(obs),chunk)]
        return tuple(np.concatenate([part[i] for part in pieces]) for i in range(3))
    with timer.section('pack_and_transfer'):batch=model.pack(obs,masks,c['device'],c['trim_entities'],bucket_override=c.get('inference_entity_bucket'))
    with torch.no_grad(),timer.section('policy_inference'):
        if c.get('graph_sampling',False):a,logp,value=model.sample_batch(batch,greedy)
        else:
            d,value=model.distribution(batch);a=d.logits.argmax(-1) if greedy else d.sample()
            logp=d.log_prob(a)
    with timer.section('actions_to_cpu'):
        if c.get('combined_readback',False):
            result=torch.stack((a.float(),logp,value),1).cpu().numpy()
            return result[:,0].astype(np.int32),result[:,1],result[:,2]
        return a.cpu().numpy(),logp.cpu().numpy(),value.cpu().numpy()

def potential(obs,s):
    o=np.asarray(obs);total=np.zeros(o.shape[:-1],np.float32)
    for side,sign in (('own',1),('enemy',-1)):
        for tower in ('king','left','right'):
            hp=o[...,s.header[f'{side}_{tower}_hp'][0]].astype(np.float32)
            maximum=o[...,s.header[f'{side}_{tower}_max_hp'][0]].astype(np.float32)
            total+=sign*hp/np.maximum(maximum,1)/3
    return total


def advantages(rewards,values,dones,trunc_values,last_values,valid,gamma,lam):
    """Natural terminal: V'=0. Time limit: bootstrap terminal observation, stop GAE.

    Fault/unsupported transitions have no loss and break the GAE chain. A rollout boundary
    bootstraps its last observation without pretending the episode ended.
    """
    out=np.zeros_like(rewards);carry=np.zeros_like(last_values)
    for t in reversed(range(len(rewards))):
        nv=last_values if t==len(rewards)-1 else values[t+1]
        nv=np.where(dones[t],trunc_values[t],nv)
        delta=rewards[t]+gamma*nv-values[t]
        carry=(delta+gamma*lam*(~dones[t])*carry)*valid[t]
        out[t]=carry
    return out,out+values

class Opponents:
    def __init__(self,v,c):
        self.v,self.c=v,c;self.pool=[];self.models={};self.scripted=ScriptedBot().bind(v)
    def add(self,model,session=None):
        self.pool.append({k:x.detach().cpu().clone() for k,x in model.state_dict().items()})
        # Keep a bounded recent population plus any snapshots assigned to ongoing matches.
        active=set(session.opponent_ids.tolist()) if session else set()
        for i in range(max(0,len(self.pool)-self.c['pool_size'])):
            if i not in active:self.pool[i]=None
        self.models={}
    def select(self,session,indices):
        for i in indices:
            if self.c['opponent_mode']=='random':session.opponent_ids[i]=-2
            elif self.c['opponent_mode']=='wait':session.opponent_ids[i]=-3
            elif self.c['opponent_mode']=='scripted' or not self.pool:session.opponent_ids[i]=-1
            elif self.c['opponent_mode']=='population':
                available=[j for j,state in enumerate(self.pool) if state is not None]
                session.opponent_ids[i]=-1 if session.rng.random()<self.c['scripted_fraction'] else int(session.rng.choice(available))
            else:session.opponent_ids[i]=len(self.pool)-1
    def act(self,session,actions,timer):
        v=self.v;c=self.c;seats=1-session.episode_seat
        for key in np.unique(session.opponent_ids):
            ids=np.flatnonzero(session.opponent_ids==key)
            if key>=0:
                if key not in self.models:
                    # Constructing a frozen opponent must not consume the learner's sampling RNG.
                    mps_rng=torch.mps.get_rng_state() if c['device']=='mps' else None
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if c['device']=='cuda' else []):
                        m=network(v,c);m.load_state_dict(self.pool[key]);m.eval();self.models[key]=m
                    if mps_rng is not None:torch.mps.set_rng_state(mps_rng)
                actions[ids,seats[ids]]=sample(self.models[key],v.observations[ids,seats[ids]],v.masks[ids,seats[ids]],c,timer)[0]
            else:
                with timer.section('scripted_or_random_opponent'):
                    for i in ids:
                        seat=seats[i];mask=v.masks[i,seat]
                        if key==-1:a=self.scripted(v.observations[i,seat],mask,session.rng)
                        elif key==-2:a=int(session.rng.choice(np.flatnonzero(mask)))
                        else:a=0
                        actions[i,seat]=a


def collect(model,session,opponents,c,timer,clock):
    v=session.v;s=v.schema;n=v.count;both=c['opponent_mode']=='selfplay'
    rows=n*2 if both else n
    data={'obs':[],'mask':[],'action':[],'logp':[],'value':[],'reward':[],'done':[],'trunc_value':[],'valid':[]}
    occupancy=[];late=0
    preallocated=c.get('preallocated_rollout',False)
    if preallocated:
        obs_store=np.empty((c['rollout_steps'],rows,s.seat_ints),dtype=v.observations.dtype)
        mask_store=np.empty((c['rollout_steps'],rows,s.actions),dtype=v.masks.dtype)
    for _ in range(c['rollout_steps']):
        if clock.exhausted():break
        ids=np.arange(n);seats=session.episode_seat.copy()
        if preallocated:
            t=len(data['obs']);obs=obs_store[t];mask=mask_store[t]
            obs[:]=v.observations.reshape(-1,s.seat_ints) if both else v.observations[ids,seats]
            mask[:]=v.masks.reshape(-1,s.actions) if both else v.masks[ids,seats]
        else:
            obs=v.observations.reshape(-1,s.seat_ints).copy() if both else v.observations[ids,seats].copy()
            mask=v.masks.reshape(-1,s.actions).copy() if both else v.masks[ids,seats].copy()
        occupancy.extend(obs[:,s.header['entity_count'][0]].tolist())
        late+=int((obs[:,s.header['tick'][0]]>=2400).sum())
        if c.get('deferred_values',False):
            a,logp=sample_actions(model,obs,mask,c,timer)
            value=np.zeros(rows,np.float32)
        else:a,logp,value=sample(model,obs,mask,c,timer)
        actions=np.zeros((n,2),np.int32)
        if both:actions[:]=a.reshape(n,2)
        else:
            actions[ids,seats]=a;opponents.act(session,actions,timer)
        if not np.all(v.masks[np.arange(n)[:,None],np.arange(2)[None,:],actions]):raise RuntimeError('policy selected masked action')
        with timer.section('environment_step'):st=session.step(actions)
        reward=v.rewards.reshape(-1).copy() if both else v.rewards[ids,seats].copy()
        nextobs=v.observations.reshape(-1,s.seat_ints) if both else v.observations[ids,seats]
        done=np.repeat(v.dones.astype(bool),2) if both else v.dones.astype(bool).copy()
        bad=(st['fault']>0)|(st['unsupported']>0)
        valid=~(np.repeat(bad,2) if both else bad)
        trunc=(st['truncated']>0)&(st['fault']==0)&(st['unsupported']==0)
        trunc=np.repeat(trunc,2) if both else trunc
        tv=np.zeros(rows,np.float32)
        if trunc.any():
            # A terminal mask may be all zero; value is independent of the policy mask.
            m=np.ones((int(trunc.sum()),s.actions),bool)
            with torch.no_grad(),timer.section('truncation_bootstrap'):
                tv[trunc]=model(model.pack(nextobs[trunc],m,c['device'],c['trim_entities']))[1].cpu().numpy()
        if c['reward_shaping']:
            phi=potential(obs,s);nextphi=potential(nextobs,s)
            nextphi=np.where(done&~trunc,0,nextphi)
            reward+=c['reward_shaping']*(c['gamma']*nextphi-phi)
        for key,x in zip(data,(obs,mask,a,logp,value,reward,done,tv,valid)):
            data[key].append(x)
        session.counters['learner_decisions']+=rows
        with timer.section('resets'):
            reset=session.reset_done();opponents.select(session,reset)
        if not valid.all():
            raise RuntimeError('fault/unsupported action: see episodes.jsonl; task run failed closed, no score')
    if occupancy:
        session.counters['entity_count_sum']+=sum(occupancy);session.counters['entity_observations']+=len(occupancy)
        session.counters['entity_count_max']=max(session.counters['entity_count_max'],max(occupancy))
        session.counters['late_game_player_decisions']+=late
    if not data['obs']:return None
    obs=v.observations.reshape(-1,s.seat_ints) if both else v.observations[np.arange(n),session.episode_seat]
    mask=v.masks.reshape(-1,s.actions) if both else v.masks[np.arange(n),session.episode_seat]
    arrays={k:np.stack(x) for k,x in data.items() if not preallocated or k not in ('obs','mask')}
    if preallocated:arrays.update(obs=obs_store[:len(data['obs'])],mask=mask_store[:len(data['mask'])])
    if c.get('deferred_values',False):
        # Preserve the original bootstrap call's RNG consumption; no fake action is
        # submitted. Values for every state use the same pre-update parameters.
        sample_actions(model,obs,mask,c,timer)
        snapshots=arrays['obs'].reshape(-1,s.seat_ints)
        arrays['value']=predict_values(model,snapshots,c,timer).reshape(-1,rows)
        last=predict_values(model,obs,c,timer)
    else:last=sample(model,obs,mask,c,timer)[2]
    adv,ret=advantages(arrays['reward'],arrays['value'],arrays['done'],arrays['trunc_value'],last,arrays['valid'],c['gamma'],c['gae_lambda'])
    arrays['advantage']=adv;arrays['return']=ret
    return {k:x.reshape((-1,)+x.shape[2:]) for k,x in arrays.items()}


def ppo_update(model,opt,data,c,rng,timer,clock):
    inds=np.flatnonzero(data['valid']);active=data['mask'].sum(1)>1
    norm_inds=inds[active[inds]] if c.get('active_policy_loss',False) else inds
    if not len(norm_inds):norm_inds=inds
    adv=data['advantage'][norm_inds]
    data['advantage'][inds]=(data['advantage'][inds]-adv.mean())/(adv.std()+1e-8)
    cached=None;fields=None
    if c.get('cache_rollout_device',False):
        with timer.section('rollout_device_cache'):
            cached=model.pack(data['obs'],data['mask'],c['device'],c['trim_entities'])
            # No pack occurs until this PPO call returns when the cache is used.
            # Borrowing is therefore safe for this bounded lifetime, and avoids a
            # second whole-rollout GPU copy. Keep the clone comparator available.
            if not c.get('borrow_rollout_cache',False):cached={k:v.clone() for k,v in cached.items()}
            if c.get('cache_rollout_fields',False):
                fields={key:torch.tensor(data[key],device=c['device']) for key in ('action','logp','advantage','return')}
                fields['active']=torch.tensor(active,device=c['device'])
    losses=[];steps=0
    for _ in range(c['epochs']):
        rng.shuffle(inds)
        for start in range(0,len(inds),c['minibatch']):
            if clock.exhausted():break
            idx=inds[start:start+c['minibatch']]
            with timer.section('update_pack_and_transfer'):
                if cached is None:batch=model.pack(data['obs'][idx],data['mask'][idx],c['device'],c['trim_entities'])
                else:
                    ix=torch.tensor(idx,device=c['device']);batch={k:v.index_select(0,ix) for k,v in cached.items()}
                if fields is not None:
                    a,old,advantage,target=(fields[key].index_select(0,ix) for key in ('action','logp','advantage','return'))
                else:
                    a=torch.tensor(data['action'][idx],device=c['device'])
                    old=torch.tensor(data['logp'][idx],device=c['device'])
                    advantage=torch.tensor(data['advantage'][idx],device=c['device'])
                    target=torch.tensor(data['return'][idx],device=c['device'])
            with timer.section('ppo_update'):
                if c.get('sparse_ppo',False):new,entropy,value=model.action_terms(batch,a)
                else:
                    d,value=model.distribution(batch);new=d.log_prob(a);entropy=d.entropy()
                ratio=(new-old).exp()
                weight=(fields['active'].index_select(0,ix) if fields is not None else torch.tensor(active[idx],device=c['device'])).float() if c.get('active_policy_loss',False) else torch.ones_like(advantage)
                pg=-(torch.minimum(ratio*advantage,ratio.clamp(1-c['clip'],1+c['clip'])*advantage)*weight).sum()/weight.sum().clamp_min(1)
                vl=(value-target).square().mean();ent=(entropy*weight).sum()/weight.sum().clamp_min(1)
                loss=pg+c['value_coef']*vl-c['entropy_coef']*ent
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),c['max_grad_norm']);opt.step()
                with torch.no_grad():kl=((ratio-1)-(new-old)).mean()
                metrics=torch.stack((pg.detach(),vl.detach(),ent.detach(),kl.detach(),grad.detach())).cpu().tolist() if c.get('combined_learner_metrics',False) else [float(pg.detach()),float(vl.detach()),float(ent.detach()),float(kl),float(grad)]
                losses.append(metrics);steps+=1
            if metrics[3]>c['target_kl']:break
        if clock.exhausted() or (losses and losses[-1][3]>c['target_kl']):break
    names=('policy_loss','value_loss','entropy','approx_kl','gradient_norm')
    return dict(zip(names,np.mean(losses,axis=0).tolist()))|{'optimizer_steps':steps,'samples':len(inds),'active_policy_samples':int(active[inds].sum()),
            'reward_mean':float(data['reward'].mean()),'advantage_raw_std':float(adv.std()),'return_mean':float(data['return'].mean())} if losses else {'optimizer_steps':0}


def save_checkpoint(out,model,opt,session,opponents,c,manifest,clock,update,checkpoint_name='checkpoint.pt'):
    state={'format':1,'config':c,'metadata':session.v.metadata,'manifest_sha256':manifest['manifest_sha256'],
        'rules_sha256':canonical(manifest['rules_lock']),'source_sha256':manifest['integration_source_sha256'],
        'model':model.state_dict(),'optimizer':opt.state_dict(),'session':session.state(),
        'opponent_pool':opponents.pool,'update':update,'clock':clock.report(),
        'allocation_ledger':str(clock.ledger_path),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if c['device']=='cuda' else None,
        'mps_rng':torch.mps.get_rng_state() if c['device']=='mps' else None,'python_rng':random.getstate()}
    temp=out/(checkpoint_name+'.tmp');torch.save(state,temp);temp.replace(out/checkpoint_name)



def warmstart(model,opt,session,opponents,c,timer,clock,out):
    """Optional teacher initialization, charged to the same worker budget. No strong weights shipped."""
    if not c['warmstart_steps']:return
    v=session.v;s=v.schema;observations=[];masks=[];targets=[]
    with timer.section('teacher_collection'):
        for _ in range(c['warmstart_steps']):
            if clock.exhausted():break
            actions=np.zeros((v.count,2),np.int32)
            for i in range(v.count):
                for seat in (0,1):
                    o=v.observations[i,seat];mask=v.masks[i,seat]
                    actions[i,seat]=opponents.scripted(o,mask,session.rng)
                    if mask.sum()>1:
                        observations.append(o.copy());masks.append(mask.copy());targets.append(int(actions[i,seat]))
            st=session.step(actions)
            session.counters['teacher_player_decisions']+=2*v.count
            if st['fault'].any() or st['unsupported'].any():raise RuntimeError('teacher collection met unsupported/faulted task')
            session.reset_done()
    if not targets:return
    observations=np.array(observations);masks=np.array(masks);targets=np.array(targets)
    if c.get('retain_teacher_dataset',False):np.savez_compressed(out/'teacher_dataset.npz',obs=observations,mask=masks,action=targets)
    optimizer=torch.optim.Adam(model.parameters(),lr=c['warmstart_lr'])
    def nll():
        sums=0.
        with torch.no_grad():
            for start in range(0,len(targets),c['minibatch']):
                b=model.pack(observations[start:start+c['minibatch']],masks[start:start+c['minibatch']],c['device'],c['trim_entities'])
                a=torch.tensor(targets[start:start+c['minibatch']],device=c['device'])
                sums+=float(-model.distribution(b)[0].log_prob(a).sum())
        return sums/len(targets)
    before=nll();updates=0
    with timer.section('teacher_updates'):
        for epoch in range(c['warmstart_epochs']):
            order=session.rng.permutation(len(targets))
            for start in range(0,len(targets),c['minibatch']):
                if clock.exhausted():break
                ids=order[start:start+c['minibatch']]
                b=model.pack(observations[ids],masks[ids],c['device'],c['trim_entities'])
                a=torch.tensor(targets[ids],device=c['device']);loss=-model.distribution(b)[0].log_prob(a).mean()
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c['max_grad_norm']);optimizer.step();updates+=1
            if clock.exhausted():break
    row={'examples':len(targets),'optimizer_steps':updates,'teacher_nll_before':before,'teacher_nll_after':nll(),
         'dataset_sha256':hashlib.sha256(observations.tobytes()+masks.tobytes()+targets.tobytes()).hexdigest(),
         'scope':'teacher initialization on real engine states; not held-out performance',**clock.report()}
    session.teacher_counts=dict(session.counters)
    row['counts']=session.teacher_counts
    dump(out/'warmstart.json',row)
    opponents.add(model,session);opponents.select(session,range(v.count))


def train(args,clock,c,rng):
    timer=Timer(c['device'],c.get('synchronize_timing',True))
    if c['device']=='cuda':torch.cuda.reset_peak_memory_stats()
    with timer.section('initialization'):v=make_vector(c)
    with v:
        out,manifest=new_run(args.out,c,v,'profile' if args.command=='profile' else 'train',clock)
        model=network(v,c)
        if getattr(args,'init_checkpoint',None):
            saved_init=torch.load(resolve(args.init_checkpoint),map_location='cpu',weights_only=False)
            v.check_metadata(saved_init['metadata'])
            if saved_init['rules_sha256']!=canonical(manifest['rules_lock']):raise ValueError('initial weights rules differ')
            model.load_state_dict(saved_init['model'])
        opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],eps=1e-5,fused=c.get('optimizer_fused',False))
        session=Session(v,c,rng,out);opponents=Opponents(v,c);start=0
        if args.resume:
            saved=torch.load(resolve(args.resume),map_location='cpu',weights_only=False)
            # Budget is total across resumes. Only increasing the allowed ceiling is accepted.
            old=dict(saved['config']);now=dict(c)
            old.pop('budget_seconds');now.pop('budget_seconds');old.pop('updates');now.pop('updates')
            if old!=now:raise ValueError('resume recipe differs beyond budget/update ceiling')
            if saved['source_sha256']!=manifest['integration_source_sha256']:raise ValueError('resume source differs')
            if saved['rules_sha256']!=canonical(manifest['rules_lock']):raise ValueError('resume rules differ')
            v.check_metadata(saved['metadata']);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
            clock.prior=max(clock.prior,saved['clock']['allocated_worker_seconds'])
            with timer.section('checkpoint_engine_replay'):session.restore(saved['session'])
            opponents.pool=saved['opponent_pool'];start=saved['update']
            torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng'])
            if c['device']=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
            if c['device']=='mps':torch.mps.set_rng_state(saved['mps_rng'])
        else:
            opponents.add(model);opponents.select(session,range(v.count))
            with timer.section('checkpoint'):save_checkpoint(out,model,opt,session,opponents,c,manifest,clock,0,'initial.pt')
            warmstart(model,opt,session,opponents,c,timer,clock,out)
            if c['warmstart_steps']:
                with timer.section('checkpoint'):save_checkpoint(out,model,opt,session,opponents,c,manifest,clock,0,'post-warmstart.pt')
        initial={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
        checkpoint_time=0.;last_update=start;error=None
        try:
            for update in range(start+1,c['updates']+1):
                if clock.exhausted():break
                iteration_start=time.perf_counter();phase_before=dict(timer.seconds)
                data=collect(model,session,opponents,c,timer,clock)
                if data is None:break
                metrics=ppo_update(model,opt,data,c,rng,timer,clock);last_update=update
                if update%c['snapshot_interval']==0:opponents.add(model,session)
                row={'update':update,**metrics,**clock.report(),'iteration_seconds':time.perf_counter()-iteration_start,
                     'iteration_timing':{key:value-phase_before.get(key,0.) for key,value in timer.seconds.items()},'counts':dict(session.counters)}
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
                if update%c['checkpoint_interval']==0:
                    with timer.section('checkpoint'):save_checkpoint(out,model,opt,session,opponents,c,manifest,clock,update)
            with timer.section('checkpoint'):save_checkpoint(out,model,opt,session,opponents,c,manifest,clock,last_update)
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
            dump(out/'failure.json',{'error':error,**clock.report(),'counts':dict(session.counters)})
            raise
        finally:
            delta=math.sqrt(sum(float((x.detach().cpu()-initial[k]).square().sum()) for k,x in model.state_dict().items() if x.dtype.is_floating_point))
            result={'status':'failed' if error else 'budget_exhausted' if clock.exhausted() else 'completed','updates':last_update,'parameter_l2_change':delta,
                'timing_scope':'synchronized phase wall clocks' if timer.synchronize else 'host enqueue/wait clocks; GPU operations may complete in later sections; use full process wall time for throughput','counts':dict(session.counters),'ppo_counts':dict(session.counters-Counter(session.teacher_counts)),'fault_signatures':v.fault_counts(),'timing':timer.report(),
                **clock.report(),'gpu_measurement_executed':c['device']=='cuda',
                'throughput':{k:session.counters[k]/max(clock.elapsed,1e-9) for k in ('ticks','environment_decisions','player_decisions','learner_decisions')},
                'checkpoint_sha256':sha(out/'checkpoint.pt') if (out/'checkpoint.pt').exists() else None,
                'pinned_staging_cache_peak_bytes':getattr(model,'staging_cache_peak_bytes',0),
                'cuda_peak_memory_allocated':torch.cuda.max_memory_allocated() if c['device']=='cuda' else None,
                'cuda_peak_memory_reserved':torch.cuda.max_memory_reserved() if c['device']=='cuda' else None,
                'qualification':'learning smoke or runnable recipe; not a trained strong reference'}
            dump(out/'result.json',result)
        return out


def eval_policy(kind,obs,masks,model,bot,rng,c,timer):
    if kind=='neural':
        if c.get('eval_skip_forced',False) and not c.get('eval_greedy',False):probabilities=evaluation_probabilities(model,obs,masks,c,timer)
        else:
            with timer.section('pack_and_transfer'):batch=model.pack(obs,masks,c['device'],c['trim_entities'])
            with torch.no_grad(),timer.section('policy_inference'):
                d,_=model.distribution(batch);probabilities=d.probs.cpu().numpy().astype(np.float64)
        output=[];s=model.s
        for j,prob in enumerate(probabilities):
            if c.get('eval_greedy',False):
                # Deterministic hierarchical decoding: choose slot by marginal mass, then tile.
                tile=prob[1:s.first_ability].reshape(s.hand_slots,-1)
                groups=np.concatenate((prob[:1],tile.sum(1),prob[s.first_ability:]))
                group=int(groups.argmax())
                a=0 if group==0 else 1+(group-1)*s.cols*s.rows+int(tile[group-1].argmax()) if group<=s.hand_slots else s.first_ability+group-s.hand_slots-1
            else:
                stream=rng[j] if isinstance(rng,list) else rng
                a=int(stream.choice(len(prob),p=prob/prob.sum()))
            output.append(a)
        return np.array(output,np.int32)
    if kind=='wait':return np.zeros(len(obs),np.int32)
    with timer.section('scripted_or_random_opponent'):
        return np.array([bot(o,m,rng) if kind=='scripted' else rng.choice(np.flatnonzero(m)) for o,m in zip(obs,masks)],np.int32)


def evaluate(args,clock,c,rng):
    spec=json.loads(Path(args.schedule).read_text());seeds=spec['engine_seeds'];opps=spec['opponents']
    # The schedule is intentionally visible practice/holdout-demo, never represented as a secret grader.
    if any(c['train_seed_range'][0]<=seed<=c['train_seed_range'][1] for seed in seeds):
        raise ValueError('evaluation seeds overlap training range')
    jobs=[(op,seed,seat) for op in opps for seed in seeds for seat in (0,1)]
    timer=Timer(c['device'],c.get('synchronize_timing',True));v=make_vector(c,count=min(c['envs'],len(jobs)))
    with v:
        out,manifest=new_run(args.out,c,v,'evaluate',clock);dump(out/'schedule.json',spec)
        model=network(v,c)
        if args.checkpoint:
            cp=torch.load(resolve(args.checkpoint),map_location='cpu',weights_only=False)
            v.check_metadata(cp['metadata'])
            if cp['rules_sha256']!=canonical(manifest['rules_lock']):raise ValueError('checkpoint rules differ')
            model.load_state_dict(cp['model'])
        model.eval();bot=ScriptedBot().bind(v);panel={};counts=Counter();episodes=[]
        for op in opps:
            if op.startswith('checkpoint:'):
                cp=torch.load(resolve(op.split(':',1)[1]),map_location='cpu',weights_only=False)
                v.check_metadata(cp['metadata'])
                if cp['rules_sha256']!=canonical(manifest['rules_lock']):raise ValueError('opponent rules differ')
                m=network(v,c);m.load_state_dict(cp['model']);m.eval();panel[op]=m
            elif op not in ('scripted','random','wait'):raise ValueError('unknown opponent '+op)
        cursor=0;active={};lane_rng={};bad=np.zeros(v.count,bool)
        def assign(i):
            nonlocal cursor
            if cursor>=len(jobs):
                active.pop(i,None);v.reset_env(i,seeds[0]);return
            op,seed,seat=jobs[cursor];cursor+=1;active[i]=(op,seed,seat)
            lane_rng[i]=[np.random.default_rng([spec['policy_seed'],seed,seat,opps.index(op),j]) for j in (0,1)]
            v.reset_env(i,seed);bad[i]=False
        deck=c['rules']['deck'];v.reset([seeds[0]]*v.count,deck,deck)
        for i in range(v.count):assign(i)
        while active and not clock.exhausted():
            ids=np.array(sorted(active));seats=np.array([active[i][2] for i in ids]);actions=np.zeros((v.count,2),np.int32)
            # Each row uses its own deterministic RNG stream. Neural sampling uses seeded torch RNG.
            if args.policy=='neural':
                actions[ids,seats]=eval_policy(args.policy,v.observations[ids,seats],v.masks[ids,seats],model,bot,[lane_rng[i][seat] for i,seat in zip(ids,seats)],c,timer)
            else:
                for i,seat in zip(ids,seats):
                    actions[i,seat]=eval_policy(args.policy,v.observations[i,seat][None,:],v.masks[i,seat][None,:],model,bot,lane_rng[i][seat],c,timer)[0]
            for op in opps:
                group=np.array([i for i in ids if active[i][0]==op],dtype=int)
                if not len(group):continue
                other=np.array([1-active[i][2] for i in group])
                if op in panel:
                    actions[group,other]=eval_policy('neural',v.observations[group,other],v.masks[group,other],panel[op],bot,[lane_rng[i][seat] for i,seat in zip(group,other)],c,timer)
                else:
                    for i,seat in zip(group,other):actions[i,seat]=eval_policy(op,v.observations[i,seat][None,:],v.masks[i,seat][None,:],model,bot,lane_rng[i][seat],c,timer)[0]
            before=v.infos[:,v.schema.info['tick']].copy()
            with timer.section('environment_step'):v.step(actions)
            st=status(v);bad|=(st['fault']>0)|(st['unsupported']>0)
            counts['environment_decisions']+=v.count;counts['player_decisions']+=2*len(ids)
            counts['padding_environment_decisions']+=v.count-len(ids)
            counts['padding_wait_decisions']+=2*(v.count-len(ids))
            counts['ticks']+=int(np.maximum(0,v.infos[:,v.schema.info['tick']]-before).sum())
            counts['illegal_actions']+=int(st['illegal'][ids].sum());counts['unsupported_actions']+=int(st['unsupported'][ids].sum())
            for i in ids:
                if not v.dones[i]:continue
                op,seed,seat=active[i];winner=int(st['winner'][i]);fault=int(st['fault'][i])
                reason='fault' if fault else 'unsupported' if bad[i] else 'truncation' if st['truncated'][i] or winner==v.schema.winner_none else 'win' if winner==seat else 'loss' if winner==1-seat else 'draw'
                counts[reason]+=1
                row={'opponent':op,'engine_seed':seed,'seat':seat,'outcome':reason,
                    'ticks':int(v.infos[i,v.schema.info['tick']]),'fault':v.fault(i).__dict__ if fault else None}
                episodes.append(row)
                with (out/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                assign(i)
            for i in np.flatnonzero(v.dones):
                if i not in active:v.reset_env(int(i),seeds[0])
        valid=counts['win']+counts['loss']+counts['draw'];invalid=counts['fault']+counts['unsupported']+counts['truncation']
        result={'status':'completed' if not active else 'budget_exhausted','planned_episodes':len(jobs),'completed_episodes':len(episodes),
                'counts':dict(counts),'valid_episodes':valid,'invalid_episodes':invalid,
                'score':(counts['win']+.5*counts['draw'])/valid if valid and invalid==0 and counts['illegal_actions']==0 and not active else None,
                'valid_only_diagnostic_score':(counts['win']+.5*counts['draw'])/valid if valid else None,
                'fault_signatures':v.fault_counts(),'timing_scope':'synchronized phase wall clocks' if timer.synchronize else 'host enqueue/wait clocks; not additive GPU phase durations','timing':timer.report(),**clock.report(),
                'policy':args.policy,'checkpoint_sha256':sha(resolve(args.checkpoint)) if args.checkpoint else None,
                'decoding':'hierarchical slot/tile greedy' if c.get('eval_greedy',False) else 'stochastic with independent per-job per-seat RNG',
                'opponent_checkpoint_sha256':{op:sha(resolve(op.split(':',1)[1])) for op in opps if op.startswith('checkpoint:')},
                'schedule_sha256':sha(out/'schedule.json'),'visibility':spec['visibility'],
                'by_seat':{str(seat):dict(Counter(r['outcome'] for r in episodes if r['seat']==seat)) for seat in (0,1)},
                'by_opponent':{op:dict(Counter(r['outcome'] for r in episodes if r['opponent']==op)) for op in opps}}
        dump(out/'result.json',result);print(json.dumps(result),flush=True)
        return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('train','evaluate','profile'))
    parser.add_argument('--config',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--init-checkpoint',help='initialize weights only; new optimizer/engine/RNG, explicit new training run')
    parser.add_argument('--resume');parser.add_argument('--checkpoint');parser.add_argument('--schedule')
    parser.add_argument('--budget-seconds',type=float,help='override total cumulative elapsed worker budget')
    parser.add_argument('--updates',type=int,help='override maximum PPO update count')
    parser.add_argument('--charge-to',help='shared elapsed-allocation ledger; checkpoint evaluation/resume uses its parent ledger by default')
    parser.add_argument('--policy',choices=('neural','scripted','random','wait'),default='neural')
    args=parser.parse_args()
    # The worker clock begins before config validation, framework setup, rules hashing, and JVM initialization.
    clock=WorkerClock(float('inf'));clock.start=_WORKER_START;c=load_config(args.config)
    if args.budget_seconds is not None:
        if args.budget_seconds<=0:parser.error('positive budget required')
        c['budget_seconds']=args.budget_seconds
    if args.updates is not None:
        if args.updates<0:parser.error('nonnegative update count required')
        c['updates']=args.updates
    clock.limit=c['budget_seconds'];clock.gpu_allocated=c['hardware_allocation']['gpu_count']>0;rng=setup(c)
    # Serial worker ledger: share this path across training, profiling and development evaluations.
    # It is mutable accounting alongside immutable source/rules manifests. Concurrent workers must
    # use an external allocation controller; this local CLI refuses a concurrently locked ledger.
    origin=args.resume or args.checkpoint
    origin_state=torch.load(resolve(origin),map_location='cpu',weights_only=False) if origin else None
    inherited=origin_state.get('allocation_ledger',str(resolve(origin).parent/'allocation-ledger.json')) if origin_state else None
    ledger=resolve(args.charge_to) if args.charge_to else resolve(inherited) if inherited else resolve(args.out)/'allocation-ledger.json'
    clock.ledger_path=ledger
    if origin_state:clock.prior=origin_state['clock']['allocated_worker_seconds']
    ledger.parent.mkdir(parents=True,exist_ok=True) if args.charge_to or origin else None
    import fcntl,os
    # A separate lock lives beside the ledger; acquire only after the run output can be created.
    lock_path=ledger.with_suffix('.lock')
    lock_parent_exists=lock_path.parent.exists()
    lockfile=lock_path.open('a') if lock_parent_exists else None
    if lockfile:fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if ledger.exists():
        previous=json.loads(ledger.read_text())
        if previous['hardware_allocation']!=c['hardware_allocation']:raise ValueError('allocation ledger hardware differs')
        clock.prior=max(clock.prior,previous['allocated_worker_seconds'])
        if clock.gpu_allocated:
            # A GPU allocation stays charged between sequential development commands as well.
            # The external scheduler owns allocation release; this CLI never allocates services.
            if 'recorded_monotonic' in previous:
                clock.prior+=max(0.,_WORKER_START-previous['recorded_monotonic'])
    try:
        if args.command=='evaluate':
            if not args.schedule:parser.error('evaluate requires --schedule')
            if args.policy=='neural' and not args.checkpoint:parser.error('neural evaluation requires --checkpoint')
            evaluate(args,clock,c,rng)
        else:train(args,clock,c,rng)
    except Exception as exc:
        if resolve(args.out).exists():dump(resolve(args.out)/'failure.json',{'error':f'{type(exc).__name__}: {exc}',**clock.report()})
        raise
    finally:
        if ledger.parent.exists():
            dump(ledger,{**clock.report(),'hardware_allocation':c['hardware_allocation'],'last_run':str(resolve(args.out)),'recorded_monotonic':time.perf_counter()})
        if lockfile:lockfile.close()

if __name__=='__main__':main()
