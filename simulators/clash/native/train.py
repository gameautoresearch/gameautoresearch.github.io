"""Device-resident native fixed8 self-play PPO with source-bound exact resume.

The CUDA backend keeps the complete simulator, public observations, legal masks,
policy samples, rollout, GAE and learner on one GPU. Native CPU is a reference and
measured transport comparator. No Java or oracle executes in this training loop.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,os,random,sys,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
_PROCESS_START=time.perf_counter()
HERE=Path(__file__).resolve().parent;INTEGRATION=HERE.parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(INTEGRATION))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(data_path):
    files={str(p.relative_to(INTEGRATION)):sha(p) for p in sorted(INTEGRATION.rglob('*')) if p.is_file() and p.suffix in ('.py','.hpp','.cpp','.cu','.cuh','.json') and not any(x.startswith('generated') or x in ('.build','__pycache__') for x in p.parts)}
    return {'source':files,'task_data_sha256':sha(data_path),'rules_lock_sha256':sha(INTEGRATION/'rules.lock.json')}
def schema(data):return SimpleNamespace(**(data['metadata']['schema']|{k:data['schema'][k] for k in ('header','entity','info','flags','kinds')}))
def recipe(c):return {k:v for k,v in c.items() if k not in ('updates','budget_seconds')}

def archive_owned_source(out,source_map):
    import io,tarfile
    target=Path(out)/'source.tar.gz';additional={}
    with tarfile.open(target,'w:gz',compresslevel=1) as archive:
        files=dict(source_map)
        for name in ('native/LICENSE','native/NOTICE','native/README.md','README.md'):
            if (INTEGRATION/name).is_file():files[name]=sha(INTEGRATION/name);additional[name]=files[name]
        for name,want in sorted(files.items()):
            payload=(INTEGRATION/name).read_bytes()
            if hashlib.sha256(payload).hexdigest()!=want:raise ValueError('source changed while preserving native run: '+name)
            member=tarfile.TarInfo('integrations/clash/'+name);member.size=len(payload);member.mode=0o444;member.mtime=0;archive.addfile(member,io.BytesIO(payload))
    target.chmod(0o444)
    return {'source_archive_sha256':sha(target),'source_archive_additional_files':additional}

def device_gae(reward,value,done,trunc_value,last,gamma,lam):
    import torch
    carry=torch.zeros_like(last);adv=torch.empty_like(reward)
    for t in reversed(range(reward.shape[0])):
        nxt=last if t==reward.shape[0]-1 else value[t+1]
        nxt=torch.where(done[t],trunc_value[t],nxt)
        carry=reward[t]+gamma*nxt-value[t]+gamma*lam*(~done[t])*carry
        adv[t]=carry
    return adv,adv+value

def device_potential(observations,s):
    import torch
    total=torch.zeros(observations.shape[:-1],device=observations.device,dtype=torch.float32)
    for side,sign in (('own',1),('enemy',-1)):
        for tower in ('king','left','right'):
            hp=observations[...,s.header[f'{side}_{tower}_hp'][0]].float();maximum=observations[...,s.header[f'{side}_{tower}_max_hp'][0]].float()
            total=total+sign*hp/maximum.clamp_min(1)/3
    return total

class Trainer:
    def __init__(self,c,data_path,resume=None):
        import numpy as np,torch
        from model import Policy
        from runtime import NativeCUDA
        from cpu_batch import NativeCPUBatch
        from gpu_features import DeviceBatchEncoder
        self.torch=torch;c=dict(c);self.c=c
        if c.get('backend')=='native-cuda':
            c.setdefault('cuda_launch_threads',int(os.getenv('CLASH_CUDA_LAUNCH_THREADS','32')))
            c.setdefault('cuda_render_mode',int(os.getenv('CLASH_CUDA_RENDER_MODE','0')))
        c.setdefault('checkpoint_codec','zlib1')
        c.setdefault('precision','fp32');c.setdefault('compile_loss',False);c.setdefault('packed_head',False)
        if c['precision'] not in ('fp32','tf32','bf16'):raise ValueError('unsupported neural precision')
        if type(c['compile_loss']) is not bool:raise ValueError('compile_loss must be boolean')
        if type(c['packed_head']) is not bool:raise ValueError('packed_head must be boolean')
        if c['packed_head'] and (c['backend']=='native-cpu' or not c.get('bucket_learner',False) or c['compile_loss'] or c['precision']!='fp32'):
            raise ValueError('experimental packed_head requires CUDA learner, bucket_learner, strict FP32 and compile_loss=False')
        if c['checkpoint_codec'] not in ('zlib1','raw'):raise ValueError('unsupported native checkpoint codec')
        self._initial_recipe=json.loads(json.dumps(recipe(c)))
        if c['backend'] not in ('native-cpu','native-cpu-cuda','native-cuda'):raise ValueError('unknown native backend')
        if (c.get('bucket_actor',False) or c.get('bucket_learner',False)) and c.get('rollout_graph',False):raise ValueError('device-selected neural buckets require outer rollout_graph=false')
        from data_binding import validate_data
        verified=validate_data();self.data_path=Path(data_path)
        if sha(self.data_path)!=verified['task_data_sha256']:raise ValueError('requested native data differs from compiled simulator data')
        self.data=json.loads(self.data_path.read_text());self.binding=binding(self.data_path);self.schema=schema(self.data)
        # The match form (native/match_forms.py) sets the decision cadence and timeline; `full` is the pinned form.
        import match_forms
        self.form=c.get('match_form','full');form_rules=dict(self.data['rules'],decision_ticks=match_forms.variant(self.form)['decision_ticks'])
        if c.get('task',self.data['task'])!=self.data['task'] or c.get('rules',form_rules)!=form_rules:raise ValueError('native trainer supports only the pinned fixed8 task/rules, with decision_ticks set by its match_form; artificial limits use top-level max_ticks')
        if c.get('opponent_mode','selfplay')!='selfplay' or c.get('warmstart_steps',0):raise ValueError('native trainer currently supports selfplay PPO without teacher initialization')
        if c.get('train_seed_range',[0,999999])!=[0,999999]:raise ValueError('native training seeds must stay within the frozen 0..999999 range')
        saved=torch.load(resume,map_location='cpu',weights_only=False) if resume else None
        if saved is not None:
            if saved['format']!='clash-native-ppo-v2' or saved['binding']!=self.binding:raise ValueError('native checkpoint source/data/rules binding differs')
            if recipe(saved['config'])!=recipe(c):raise ValueError('native resume recipe differs beyond update/budget ceiling')
        self.device=torch.device('cuda' if c['backend']!='native-cpu' else 'cpu');self.gpu=self.device.type=='cuda'
        if self.gpu:
            if torch.cuda.device_count()!=1:raise ValueError('native training requires exactly one visible GPU')
            allocation=c.get('hardware_allocation')
            if allocation is not None and (allocation.get('gpu_count')!=1 or allocation.get('gpu_model')!=torch.cuda.get_device_name(self.device)):raise ValueError('native actual GPU differs from declared fixed allocation')
        from build_receipt import backend_build_binding
        if c['backend']=='native-cuda':
            from runtime import build_cuda
            library=build_cuda(self.form)
        else:
            from build import build
            library=build(self.form)
        self.backend_build=backend_build_binding(library)
        self.auxiliary_builds={}
        if c.get('bucket_actor',False) or c.get('bucket_learner',False):
            if not self.gpu:raise ValueError('device-selected neural buckets require CUDA')
            from device_branch import build as build_branch
            self.auxiliary_builds['device_branch']=backend_build_binding(build_branch())
        if c['packed_head']:
            from packed_head_branch import build as build_packed_head
            self.auxiliary_builds['packed_head_branch']=backend_build_binding(build_packed_head())
        self.execution_runtime={'torch':torch.__version__,'numpy':np.__version__,'python':list(sys.version_info[:3]),'cuda':torch.version.cuda if self.gpu else None,'gpu':torch.cuda.get_device_name(self.device) if self.gpu else None}
        if c['backend']=='native-cuda':self.execution_runtime['engine_policy']={k:c[k] for k in ('cuda_launch_threads','cuda_render_mode')}
        torch.set_num_threads(c.get('torch_threads',1));torch.use_deterministic_algorithms(True);random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
        if self.gpu:torch.cuda.manual_seed_all(c['seed'])
        self.model=Policy(self.schema,len(self.data['vocabulary']),len(self.data['buff_names']),c['width'],c['embedding']).to(self.device)
        if c.get('compact_embedding_grad',False):
            from compact_embeddings import install
            self.embedding_support=install(self.model,self.data)
        from precision import configure_precision
        self.execution_runtime['precision']=configure_precision(self.model,c['precision'],self.device)
        if c['packed_head']:
            self.execution_runtime['packed_head']={'version':1,'entity_bounds':[32,64,96,200],'head_bounds':[32,64,128,'fullN'],'policy_terms':'static active-prefix head, full-N scalar scatter and unchanged PPO denominators','sampling':'unchanged actor'}
        if c['compile_loss']:
            from compiled_loss import compile_recipe
            self.execution_runtime['compiled_loss']=compile_recipe()
        if self.gpu and (c['compile_loss'] or c.get('gae_kernel')=='triton'):
            import triton
            self.execution_runtime['triton']=triton.__version__
        if saved is not None and (saved.get('backend_build')!=self.backend_build or saved.get('auxiliary_builds',{})!=self.auxiliary_builds or saved.get('execution_runtime')!=self.execution_runtime):raise ValueError('native checkpoint backend build/runtime binding differs')
        self.model.device_indices=True;self.model.graph_sampling=c.get('graph_sampling',True) and self.gpu;self.model.inference_graph_cache_size=2
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=c['learning_rate'],eps=1e-5,fused=self.gpu and c.get('optimizer_fused',True),capturable=self.gpu and (c.get('learner_graph',False) or c.get('bucket_learner',False)))
        seeds=torch.randint(0,1000000,(c['envs'],),device=self.device,dtype=torch.int32)
        self.env=NativeCUDA(seeds,self.device,launch_threads=c['cuda_launch_threads'],render_mode=c['cuda_render_mode'],form=self.form) if c['backend']=='native-cuda' else NativeCPUBatch(seeds.cpu().tolist(),threads=c.get('threads',1),form=self.form)
        self._host_output_tensors=[];names=('observations','masks','infos','rewards')
        if self.gpu and c['backend']!='native-cuda':
            for name in names:
                original=getattr(self.env,name);host=torch.empty(original.shape,dtype=torch.from_numpy(original).dtype,pin_memory=True);setattr(self.env,name,host.numpy());self._host_output_tensors.append(host)
            self.env.read();self._output_tensors=tuple(torch.empty_like(x,device=self.device) for x in self._host_output_tensors);self.stage_outputs()
        else:self._output_tensors=tuple(getattr(self.env,name) if isinstance(getattr(self.env,name),torch.Tensor) else torch.from_numpy(getattr(self.env,name)) for name in names)
        self.encoder=DeviceBatchEncoder(self.schema,copy=False);self.actor=None;self.update=0;self.samples=0;self.prior_seconds=0.;self.counters=torch.zeros(11,device=self.device,dtype=torch.int64);self.bucket_usage=torch.zeros(8,device=self.device,dtype=torch.int64)
        self.refresh_actor(create=True)
        if resume:
            self.model.load_state_dict(saved['model']);self.optimizer.load_state_dict(saved['optimizer'])
            from snapshots import decode_engine_state
            width=int(self.env.states.shape[1]) if c['backend']=='native-cuda' else self.env.world_bytes
            restored=decode_engine_state(saved['engine_state'],as_tensor=c['backend']=='native-cuda',expected_count=c['envs'],expected_world_bytes=width,workers=max(1,c.get('threads',1)),max_total_bytes=c['envs']*width)
            self.env.load(restored)
            for name in ('observations','masks','infos','rewards'):
                target=getattr(self.env,name);source=saved['engine_outputs'][name]
                if isinstance(target,torch.Tensor):target.copy_(source.to(self.device))
                else:target[:]=source.numpy()
            self.update=saved['update'];self.samples=saved['samples'];self.prior_seconds=saved['allocated_seconds'];self.counters.copy_(saved['counters'].to(self.device));self.stage_outputs();self.refresh_actor()
            self.bucket_usage.copy_(saved['bucket_usage'].to(self.device))
            random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
            if self.gpu:torch.cuda.set_rng_state_all(saved['cuda_rng'])
        n=c['envs']*2;t=c['rollout_steps'];s=self.schema
        self.store={'obs':torch.empty((t,n,s.seat_ints),dtype=torch.int32,device=self.device),'mask':torch.empty((t,n,s.actions),dtype=torch.uint8,device=self.device)}
        for name in ('logp','value','reward','trunc_value'):self.store[name]=torch.empty((t,n),dtype=torch.float32,device=self.device)
        self.store['action']=torch.empty((t,n),dtype=torch.int64,device=self.device);self.store['done']=torch.empty((t,n),dtype=torch.bool,device=self.device)
    def outputs(self):return self._output_tensors
    def stage_outputs(self):
        for host,gpu in zip(self._host_output_tensors,self._output_tensors):gpu.copy_(host,non_blocking=True)
    def refresh_actor(self,create=False):
        obs,mask,_,_=self.outputs();packed=self.encoder(obs,mask)
        if create:self.actor={k:v.clone() for k,v in packed.items()}
        else:
            for k,v in packed.items():self.actor[k].copy_(v)
    def collect(self):
        self.validate_engine_runtime()
        if self.c.get('rollout_graph',False):
            if not hasattr(self,'_rollout_graph'):
                from cuda_graph import RolloutGraph
                self._rollout_graph=RolloutGraph(self)
            return self._rollout_graph()
        return self._collect_eager()
    def validate_engine_runtime(self):
        from precision import matmul_runtime_flags
        if recipe(self.c)!=self._initial_recipe:raise ValueError('native training recipe changed after runtime binding')
        if matmul_runtime_flags()!=self.execution_runtime['precision']['matmul_runtime_flags']:raise ValueError('native matmul flags changed after precision binding')
        if bool(self.torch.backends.cuda.matmul.allow_tf32)!=(self.c['precision']=='tf32') or getattr(self.model,'_precision_metadata',None)!=self.execution_runtime['precision']:raise ValueError('native neural precision changed after runtime binding')
        if self.c['backend']=='native-cuda' and self.env.runtime_policy()!=self.execution_runtime['engine_policy']:raise ValueError('native CUDA scheduling changed after checkpoint/runtime binding')
    def _collect_eager(self):
        torch=self.torch;c=self.c;s=self.schema;buf=self.store
        with torch.no_grad():
            for t in range(c['rollout_steps']):
                obs,mask,info,_=self.outputs();buf['obs'][t].copy_(obs.reshape(-1,s.seat_ints));buf['mask'][t].copy_(mask.reshape(-1,s.actions))
                if c.get('bucket_actor',False):
                    if not hasattr(self,'_bucket_actor'):
                        from bucket_actor import BucketActor
                        self._bucket_actor=BucketActor(self.model,c['envs']*2,graph_sampler=c.get('graph_sampling',True))
                    action,logp,value=self._bucket_actor.sample(obs,mask)
                    self.bucket_usage[:4].add_(torch.nn.functional.one_hot(self._bucket_actor._branch.selected_bucket.long(),4))
                else:self.refresh_actor();action,logp,value=self.model.sample_batch(self.actor)
                buf['action'][t].copy_(action);buf['logp'][t].copy_(logp);buf['value'][t].copy_(value)
                before=info[:,s.info['tick']].clone()
                if c['backend']=='native-cuda':self.env.step(action.reshape(-1,2),c.get('max_ticks',0))
                else:self.env.step(action.reshape(-1,2).cpu().numpy(),c.get('max_ticks',0));self.stage_outputs()
                obs,mask,info,reward=self.outputs();bad=(info[:,s.info['fault']]!=0)|(info[:,s.info['not_modelled']]!=0)|(info[:,s.info['illegal_0']]!=0)|(info[:,s.info['illegal_1']]!=0)
                torch._assert_async((~bad).all(),'native simulator fault/unsupported/illegal transition; training fails closed')
                natural=info[:,s.info['winner']]>=0;truncated=info[:,s.info['truncated']]!=0;done=natural|truncated
                buf['reward'][t].copy_(reward.flatten());buf['done'][t].copy_(done[:,None].expand(-1,2).reshape(-1))
                if c.get('reward_shaping',0):
                    phi=device_potential(buf['obs'][t],s);nxt=device_potential(obs.reshape(-1,s.seat_ints),s);nxt=torch.where(natural[:,None].expand(-1,2).reshape(-1),0.,nxt);buf['reward'][t].add_(c['reward_shaping']*(c['gamma']*nxt-phi))
                if c.get('max_ticks',0):
                    final_value=self.model.values(self.encoder(obs,None));buf['trunc_value'][t].copy_(torch.where(truncated[:,None].expand(-1,2).reshape(-1),final_value,0.))
                else:buf['trunc_value'][t].zero_()
                active=(buf['mask'][t].sum(-1)>1).sum();late=(buf['obs'][t,:,s.header['tick'][0]]>=2400).sum();maximum=buf['obs'][t,:,s.header['entity_count'][0]].max()
                self.counters[:10].add_(torch.stack((natural.sum(),(info[:,s.info['winner']]==0).sum(),(info[:,s.info['winner']]==1).sum(),(info[:,s.info['winner']]==2).sum(),truncated.sum(),bad.sum(),(info[:,s.info['tick']]-before).sum(),active,late,torch.zeros((),device=self.device,dtype=torch.int64))))
                self.counters[10].copy_(torch.maximum(self.counters[10],maximum))
                seeds=torch.randint(0,1000000,(c['envs'],),device=self.device,dtype=torch.int32)
                if c['backend']=='native-cuda':self.env.reset(seeds,done)
                else:
                    flags=done.cpu().numpy()
                    if flags.any():self.env.reset(seeds.cpu().numpy(),flags);self.stage_outputs()
            self.refresh_actor();last=self.model.values(self.actor)
            gae=device_gae
            if c.get('gae_kernel','torch')=='triton':
                from triton_kernels import gae
            advantage,returns=gae(buf['reward'],buf['value'],buf['done'],buf['trunc_value'],last,c['gamma'],c['gae_lambda'])
        data={k:v.flatten(0,1) for k,v in buf.items()};data['advantage']=advantage.flatten();data['return']=returns.flatten()
        return data
    def learn(self,data):
        self.validate_engine_runtime()
        torch=self.torch;c=self.c;active=(data['mask'].sum(-1)>1);weight=active.float() if c.get('active_policy_loss',True) else torch.ones_like(data['advantage'])
        norm=torch.where(weight.sum()>0,weight,torch.ones_like(weight));count=norm.sum();mean=(data['advantage']*norm).sum()/count;std=torch.sqrt(((data['advantage']-mean).square()*norm).sum()/count)
        adv=(data['advantage']-mean)/(std+1e-8);total=data['action'].numel();metrics=[];steps=0
        for epoch in range(c['epochs']):
            order=torch.randperm(total,device=self.device)
            for start in range(0,total,c['minibatch']):
                ix=order[start:start+c['minibatch']]
                if c.get('learner_graph',False) or c.get('bucket_learner',False):
                    from learner_graph import LearnerGraph
                    if c.get('bucket_learner',False):
                        from bucket_learner import BucketLearnerGraph as LearnerGraph
                    if c['packed_head']:
                        from packed_head_graph import PackedHeadLearnerGraph as LearnerGraph
                    fields={k:data[k][ix] for k in ('obs','mask','action','logp','return')};fields.update(advantage=adv[ix],weight=weight[ix])
                    if not hasattr(self,'_learner_graphs'):self._learner_graphs={}
                    key=ix.numel()
                    if key not in self._learner_graphs:self._learner_graphs[key]=LearnerGraph(self,fields)
                    metrics.append(self._learner_graphs[key](fields).clone());kl=metrics[-1][3];steps+=1
                    if c.get('bucket_learner',False):self.bucket_usage[4:].add_(torch.nn.functional.one_hot(self._learner_graphs[key].branch.selected_bucket.long(),4))
                elif c['compile_loss']:
                    from learner_graph import eager_minibatch
                    fields={k:data[k][ix] for k in ('obs','mask','action','logp','return')};fields.update(advantage=adv[ix],weight=weight[ix])
                    metrics.append(eager_minibatch(self,fields));kl=metrics[-1][3];steps+=1
                else:
                    batch=self.encoder(data['obs'][ix],data['mask'][ix]);d,value=self.model.distribution(batch);new=d.log_prob(data['action'][ix]);old=data['logp'][ix];ratio=(new-old).exp();w=weight[ix];a=adv[ix]
                    pg=-(torch.minimum(ratio*a,ratio.clamp(1-c['clip'],1+c['clip'])*a)*w).sum()/w.sum().clamp_min(1);vl=(value-data['return'][ix]).square().mean();ent=(d.entropy()*w).sum()/w.sum().clamp_min(1);loss=pg+c['value_coef']*vl-c['entropy_coef']*ent
                    torch._assert_async(torch.isfinite(loss),'nonfinite native PPO loss');self.optimizer.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(self.model.parameters(),c['max_grad_norm']);self.optimizer.step();steps+=1
                    with torch.no_grad():kl=((ratio-1)-(new-old)).mean();metrics.append(torch.stack((pg.detach(),vl.detach(),ent.detach(),kl,grad.detach())))
                if c.get('target_kl',0)>0 and float(kl)>c['target_kl']:break
            if c.get('target_kl',0)>0 and float(kl)>c['target_kl']:break
        self.update+=1;self.samples+=total
        return torch.stack(metrics).mean(0),steps
    def checkpoint(self,path,elapsed):
        torch=self.torch
        self.validate_engine_runtime();started=time.perf_counter()
        # Copy the paused live CUDA image directly to CPU: no second full GPU
        # allocation is needed. Encoding is lossless over every ABI byte.
        state=self.env.states.detach().cpu() if self.c['backend']=='native-cuda' else self.env.state()
        copied=time.perf_counter()
        if self.c['checkpoint_codec']=='zlib1':
            from snapshots import encode_engine_state,snapshot_sizes
            state=encode_engine_state(state,workers=max(1,self.c.get('threads',1)));sizes=snapshot_sizes(state)
        else:
            sizes={'uncompressed_bytes':state.numel() if isinstance(state,torch.Tensor) else sum(map(len,state))}
        encoded=time.perf_counter()
        outputs={name:(getattr(self.env,name).cpu().clone() if isinstance(getattr(self.env,name),torch.Tensor) else torch.from_numpy(getattr(self.env,name).copy())) for name in ('observations','masks','infos','rewards')}
        saved={'format':'clash-native-ppo-v2','config':self.c,'binding':self.binding,'backend_build':self.backend_build,'auxiliary_builds':self.auxiliary_builds,'execution_runtime':self.execution_runtime,'metadata':self.data['metadata'],'allocation_ledger':getattr(self,'allocation_ledger',None),'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'engine_state':state,'engine_outputs':outputs,'update':self.update,'samples':self.samples,'allocated_seconds':self.prior_seconds+elapsed,'counters':self.counters.cpu(),'bucket_usage':self.bucket_usage.cpu(),'python_rng':random.getstate(),'numpy_rng':__import__('numpy').random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if self.gpu else None}
        path=Path(path);temp=path.with_suffix('.tmp');torch.save(saved,temp);temp.replace(path)
        self.last_checkpoint_diagnostic={'codec':self.c['checkpoint_codec'],'state_copy_seconds':copied-started,'encode_seconds':encoded-copied,'serialize_seconds':time.perf_counter()-encoded,'total_seconds':time.perf_counter()-started,'file_bytes':path.stat().st_size,**sizes}
        return self.last_checkpoint_diagnostic
    def close(self):
        if self.gpu:self.torch.cuda.synchronize(self.device)
        if hasattr(self,'_bucket_actor'):
            self._bucket_actor.close();del self._bucket_actor
        for graph in getattr(self,'_learner_graphs',{}).values():
            if hasattr(graph,'close'):graph.close()
            else:graph.graph.reset()
        self._learner_graphs={}
        if hasattr(self,'_rollout_graph'):
            self._rollout_graph.graph.reset();del self._rollout_graph
        self.env.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,default=HERE/'generated-v2/task-data.json');p.add_argument('--resume',type=Path)
    p.add_argument('--charge-to',type=Path,help='existing cumulative allocated-worker account')
    p.add_argument('--allow-moved-ledger',action='store_true',help='explicitly preserve a moved checkpoint lower bound when its original ledger is unavailable; unknown prior tail remains labelled')
    a=p.parse_args();started=_PROCESS_START;a.out.mkdir(parents=True,exist_ok=False);c=json.loads(a.config.read_text())
    import torch
    from allocation import AllocationLedger
    origin=torch.load(a.resume,map_location='cpu',weights_only=False) if a.resume else None
    inherited=(origin.get('allocation_ledger') or str(a.resume.resolve().parent/'allocation-ledger.json')) if origin is not None else None
    lower_bound=origin['allocated_seconds'] if origin is not None else 0.
    del origin
    ledger_path=a.charge_to or (None if inherited and Path(inherited).exists() else a.out/'allocation-ledger.json')
    hardware=dict(c.get('hardware_allocation') or {'gpu_count':int(c['backend']!='native-cpu'),'gpu_model':torch.cuda.get_device_name() if c['backend']!='native-cpu' else None,'cpu_threads':c.get('threads',1),'ram_gib':None,'qualification':'local declared backend; external allocation receipt remains authoritative'})
    with AllocationLedger.start(ledger_path,budget_seconds=c['budget_seconds'],hardware=hardware,process_start=started,resume_lower_bound=lower_bound,inherited_path=inherited,moved_checkpoint_floor=a.allow_moved_ledger) as account:
        trainer=None
        try:
            if c['backend']!='native-cpu' and torch.cuda.device_count()!=1:raise ValueError('exactly one GPU required')
            trainer=Trainer(c,a.data,a.resume);trainer.prior_seconds=account.prior_seconds;trainer.allocation_ledger=str(account.path)
            manifest={'format':2,'config':trainer.c,'binding':trainer.binding,'backend_build':trainer.backend_build,'auxiliary_builds':trainer.auxiliary_builds,'execution_runtime':trainer.execution_runtime,'allocation_ledger':str(account.path),'initial_allocated_seconds':account.elapsed(),'gpu':torch.cuda.get_device_name() if trainer.gpu else None,'scope':'native fixed8 selfplay PPO; whole-engine qualification remains separately recorded; all setup/compile/checkpoint time charged'}
            manifest.update(archive_owned_source(a.out,trainer.binding['source']));(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            initial={k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()};heartbeat_time=time.perf_counter()
            with (a.out/'metrics.jsonl').open('w') as log:
                while trainer.update<c['updates'] and not account.exhausted():
                    start=time.perf_counter();data=trainer.collect();metrics,steps=trainer.learn(data);numbers=metrics.cpu().tolist()
                    row={'update':trainer.update,'samples':trainer.samples,'optimizer_steps':steps,'losses':numbers,'counters':trainer.counters.cpu().tolist(),'iteration_seconds':time.perf_counter()-start,'allocated_seconds':account.elapsed()}
                    log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                    if time.perf_counter()-heartbeat_time>=1.:account.heartbeat();heartbeat_time=time.perf_counter()
            trainer.checkpoint(a.out/'checkpoint.pt',time.perf_counter()-started)
            delta=max(float((value.detach().cpu()-initial[name]).abs().max()) for name,value in trainer.model.state_dict().items())
            result={'status':'budget_exhausted' if account.exhausted() else 'completed','updates':trainer.update,'learner_samples':trainer.samples,'model_max_change':delta,'counters':trainer.counters.cpu().tolist(),'checkpoint':trainer.last_checkpoint_diagnostic,'peak_cuda_bytes':torch.cuda.max_memory_allocated() if trainer.gpu else 0,'allocation_ledger':str(account.path),'scope':'full process includes native setup/compilation, selfplay rollouts, GAE, PPO, checkpoint and cleanup; provider full allocation receipt authoritative; strength gain requires independent evaluation'}
            trainer.close();trainer=None
            result.update(allocated_seconds=account.elapsed(),process_seconds=time.perf_counter()-started)
            (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');account.record(result['status'],details={'output':str(a.out.resolve()),'updates':result['updates'],'learner_samples':result['learner_samples'],'includes_final_checkpoint_and_cleanup':True});print(json.dumps(result),flush=True)
        except BaseException as exc:
            (a.out/'failure.json').write_text(json.dumps({'error':f'{type(exc).__name__}: {exc}','allocated_seconds':account.elapsed(),'allocation_ledger':str(account.path)},indent=2)+'\n')
            raise
        finally:
            if trainer is not None:trainer.close()
if __name__=='__main__':main()
