"""CUDA/native parity and real PPO on a declared component slice, NOT full Gen 9."""
from pathlib import Path
import ctypes as C,hashlib,json,subprocess,time,copy
import numpy as np
import torch
from .bridge import Native,Move,State,build,HERE
from .validate import fill

def run(out,fixture):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
 cpu=build(out/'cpu-build');cuda=out/'libpokemon_cuda.so'
 command=['nvcc','-std=c++17','-O3','--fmad=false','-arch=sm_89','--shared','-Xcompiler','-fPIC',str(HERE/'kernels.cu'),'-o',str(cuda),'-Xlinker','-rpath','-Xlinker','/usr/local/cuda/lib64']
 compile_started=time.perf_counter();subprocess.run(command,check=True);compile_seconds=time.perf_counter()-compile_started
 lib=C.CDLL(str(cuda));stream=torch.cuda.current_stream().cuda_stream
 for name,argc in [('pkn_step_device',8),('pkn_masks_device',4),('pkn_component_views_device',6),('pkn_reset_done_device',4)]:
  getattr(lib,name).argtypes=[C.c_void_p]*argc;getattr(lib,name).restype=C.c_int
 lib.pkn_step_device.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p]
 lib.pkn_masks_device.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_void_p]
 lib.pkn_component_views_device.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_void_p]
 lib.pkn_reset_done_device.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_void_p]
 d=json.loads(Path(fixture).read_text());initial=d['initial'];n=1024
 with Native(cpu,n,3) as arena:
  template=Native(cpu,1,0);fill(template,initial);raw=template.snapshot();template.close()
  for i in range(n):C.memmove(C.addressof(arena.records[i]),raw,len(raw));arena.records[i].rng=i+1
  packed=arena.snapshot();host=np.frombuffer(packed,dtype=np.uint8).copy()
  state=torch.from_numpy(host).cuda();initial_state=state.clone()
  moves=(Move*1)();m=moves[0];m.attack_stat=-1;m.defense_stat=-1;m.id=0;m.power=40;m.accuracy=100;m.priority=0;m.type=0;m.category=0;m.maxpp=56
  move_data=torch.from_numpy(np.frombuffer(C.string_at(C.addressof(moves),C.sizeof(moves)),dtype=np.uint8).copy()).cuda()
  chart=torch.zeros((18,18),dtype=torch.int8,device='cuda');actions=torch.zeros((n,2),dtype=torch.int32,device='cuda')
  masks=torch.empty((n,2,14),dtype=torch.uint8,device='cuda');obs=torch.empty((n,2,16),device='cuda');rewards=torch.empty((n,2),device='cuda');flags=torch.empty((n,3),dtype=torch.int32,device='cuda')
  def check(code):
   if code:raise RuntimeError(f'CUDA kernel launch failed {code}')
  def tick():check(lib.pkn_step_device(state.data_ptr(),actions.data_ptr(),n,move_data.data_ptr(),1,chart.data_ptr(),stream))
  def observe():
   check(lib.pkn_masks_device(state.data_ptr(),masks.data_ptr(),n,stream));check(lib.pkn_component_views_device(state.data_ptr(),obs.data_ptr(),rewards.data_ptr(),flags.data_ptr(),n,stream))
  def reset_done():check(lib.pkn_reset_done_device(state.data_ptr(),initial_state.data_ptr(),n,stream))
  compared=0
  for step in range(32):
   arena.step(np.zeros((n,2),dtype=np.int32),moves,np.zeros((18,18),dtype=np.int8));tick();observe();torch.cuda.synchronize()
   if state.cpu().numpy().tobytes()!=arena.snapshot():raise AssertionError(f'CPU/CUDA state parity failed step {step}')
   np.testing.assert_array_equal(masks.cpu().numpy(),arena.masks());compared+=n
  # Real masked PPO with device-resident states/masks/observations. This is a
  # constructed mechanics/collector canary, not the full benchmark or strength.
  torch.manual_seed(731);torch.cuda.manual_seed_all(731)
  model=torch.nn.Sequential(torch.nn.Linear(16,64),torch.nn.Tanh(),torch.nn.Linear(64,15)).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
  state.copy_(initial_state);observe();before={k:v.clone() for k,v in model.state_dict().items()}
  def rollout_update():
   storage=[]
   for t in range(16):
    with torch.no_grad():
     x=obs.clone();outcome=model(x);logits=outcome[...,:14].masked_fill(~masks.bool(),-1e9);dist=torch.distributions.Categorical(logits=logits,validate_args=False)
     a=dist.sample();v=outcome[...,14];lp=dist.log_prob(a)
     actions.copy_(a);tick();observe();boundary=model(obs)[...,14]
     done=flags[:,0].bool();truncated=flags[:,1].bool();bootstrap=torch.where(done[:,None],0,boundary)
     storage.append((x,a,lp,v,rewards.clone(),bootstrap,(done|truncated).clone()))
     reset_done();observe()
   if flags[:,2].any().item():raise RuntimeError('Native component simulator fault before learner update')
   advantage=torch.zeros((n,2),device='cuda');samples=[]
   for x,a,lp,v,r,boot,end in reversed(storage):
    advantage=r+.99*boot-v+.99*.95*(~end[:,None])*advantage
    samples.append((x,a,lp,advantage.clone(),(advantage+v).clone()))
   xx=torch.cat([x[0].reshape(-1,16) for x in samples]);aa=torch.cat([x[1].flatten() for x in samples]);old=torch.cat([x[2].flatten() for x in samples]);adv=torch.cat([x[3].flatten() for x in samples]);target=torch.cat([x[4].flatten() for x in samples])
   # All component decisions have one ordinary or Tera move. Rebuild masks
   # from the observer's public own Tera-used field at each retained state.
   mm=torch.zeros((len(xx),14),dtype=torch.bool,device='cuda');mm[:,0]=True;mm[:,4]=xx[:,5]==0
   adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
   for _ in range(2):
    for idx in torch.randperm(len(xx),device='cuda').split(4096):
     p=model(xx[idx]);dist=torch.distributions.Categorical(logits=p[:,:14].masked_fill(~mm[idx],-1e9),validate_args=False);ratio=(dist.log_prob(aa[idx])-old[idx]).exp();loss=-torch.minimum(ratio*adv[idx],ratio.clamp(.8,1.2)*adv[idx]).mean()+.25*(p[:,14]-target[idx]).square().mean()-.01*dist.entropy().mean()
     optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
   return len(xx)
  warm=time.perf_counter();count=rollout_update();torch.cuda.synchronize();first=time.perf_counter()-warm
  cp=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),state=state.clone(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
  cp['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in ('.cpp','.hpp','.cu','.py','.cjs')}
  cp['binary_sha256']=hashlib.sha256(cuda.read_bytes()).hexdigest()
  torch.save(cp,out/'component-checkpoint.pt')
  count+=rollout_update();expected={k:v.clone() for k,v in model.state_dict().items()};expected_state=state.clone();expected_rng=torch.cuda.get_rng_state()
  cp=torch.load(out/'component-checkpoint.pt',map_location='cuda',weights_only=False)
  if cp['source_sha256']!={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in ('.cpp','.hpp','.cu','.py','.cjs')}:raise ValueError('Native CUDA checkpoint source changed')
  if cp['binary_sha256']!=hashlib.sha256(cuda.read_bytes()).hexdigest():raise ValueError('Native CUDA checkpoint executable changed')
  cp['torch_rng']=cp['torch_rng'].cpu();cp['cuda_rng']=cp['cuda_rng'].cpu()
  model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);state.copy_(cp['state']);observe();torch.set_rng_state(cp['torch_rng']);torch.cuda.set_rng_state(cp['cuda_rng']);rollout_update()
  exact=all(torch.equal(v,expected[k]) for k,v in model.state_dict().items()) and torch.equal(state,expected_state) and torch.equal(torch.cuda.get_rng_state(),expected_rng)
  if not exact:raise AssertionError('Native component CUDA resume changed state/model/RNG')
  changed=sum(float((v-before[k]).square().sum()) for k,v in model.state_dict().items())**.5
  result=dict(full_gen9_admission=False,scope='constructed Mew/Tackle component slice only',cpu_cuda_state_mask_comparisons=compared,exact_component_cuda_resume=exact,
   parameter_change_l2=changed,canary_policy_samples=count,first_update_seconds=first,compile_seconds=compile_seconds,allocated_seconds=time.perf_counter()-started,
   gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,peak_cuda_bytes=torch.cuda.max_memory_allocated(),fixture_sha256=hashlib.sha256(Path(fixture).read_bytes()).hexdigest(),native_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in ('.cpp','.hpp','.cu','.py')})
  (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');return result
