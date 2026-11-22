"""Lossless observation transport and one-copy policy staging; optional CUDA graphs.

Only player-visible encoder output crosses this buffer. Float64 wire values retain
JS Number semantics, including scripted-opponent tie breaking. Policy features are
cast once to float32 exactly as in the baseline network batch().
"""
from __future__ import annotations
import base64,time
from collections import OrderedDict
from collections.abc import Mapping
import numpy as np
import torch
from torch.distributions import Categorical
from .network import KEYS

SHAPES={'tokens':(12,12),'mons':(12,24),'global':(40,),'global_tokens':(4,),
        'action_tokens':(14,3),'actions':(14,12),'mask':(14,)}
SLICES={};cursor=0
for key in KEYS:
    n=int(np.prod(SHAPES[key]));SLICES[key]=slice(cursor,cursor+n);cursor+=n
WIDTH=cursor

class PackedStorage:
    pass

class PackedObservation(dict,PackedStorage):
    """Immutable wire storage with keyed views and a contiguous fast-copy view."""
    def __init__(self, flat):
        if flat.size != WIDTH:raise ValueError(f'Packed observation width {flat.size} != {WIDTH}')
        self.flat=flat
        super().__init__({k:flat[SLICES[k]].reshape(SHAPES[k]) for k in KEYS})

class LazyPackedObservation(Mapping,PackedStorage):
    """Create keyed views on demand; bulk actor/learner paths need only flat."""
    __slots__=('flat','_views')
    def __init__(self,flat):
        if flat.size!=WIDTH:raise ValueError('Packed observation width mismatch')
        self.flat=flat;self._views={}
    def __getitem__(self,key):
        if key not in self._views:self._views[key]=self.flat[SLICES[key]].reshape(SHAPES[key])
        return self._views[key]
    def __iter__(self):return iter(KEYS)
    def __len__(self):return len(KEYS)

def decode_observation(obs,lazy=False):
    if 'packed_f64le' not in obs:return obs
    flat=np.frombuffer(base64.b64decode(obs['packed_f64le'],validate=True),dtype='<f8')
    if flat.size!=WIDTH:raise ValueError(f'Packed observation width {flat.size} != {WIDTH}')
    return (LazyPackedObservation if lazy else PackedObservation)(flat)

def unpack_tensor(flat):
    result={k:flat[:,SLICES[k]].reshape(flat.shape[0],*SHAPES[k]) for k in KEYS}
    for key in ('tokens','global_tokens','action_tokens'):result[key]=result[key].long()
    result['mask']=result['mask'].bool()
    return result

class EfficientInference:
    """One pinned H2D staging buffer and one combined action/logp/value D2H copy."""
    def __init__(self,device,timings,graph=False,flat_copy=False,bulk_stack=False):
        self.bulk_stack=bulk_stack
        self.flat_copy=flat_copy
        self.device,self.timings=device,timings
        self.use_graph=graph and str(device).startswith('cuda')
        self.buffers={};self.graphs=OrderedDict();self.graph_setup_seconds=0.
        self.events=[torch.cuda.Event(enable_timing=True) for _ in range(4)] if str(device).startswith("cuda") and getattr(timings,"cuda_events",False) else None
    def cpu_buffer(self,n):
        if n not in self.buffers:
            self.buffers[n]=torch.zeros((n,WIDTH),dtype=torch.float32,
                                        pin_memory=str(self.device).startswith('cuda'))
        return self.buffers[n]
    @staticmethod
    def compute(model,x,deterministic):
        logits,values=model(x)
        dist=Categorical(logits=logits,validate_args=False)
        actions=dist.logits.argmax(-1) if deterministic else dist.sample()
        return torch.stack((actions.float(),dist.log_prob(actions),values),dim=1)
    @torch.no_grad()
    def __call__(self,model,observations,deterministic=False):
        started=time.perf_counter();n=len(observations)
        capacity=(1<<(n-1).bit_length()) if self.use_graph and deterministic else n
        # Validate on the CPU, before transfer, avoiding a per-call device-wide sync.
        if not self.bulk_stack and any(not np.any(o['mask']) for o in observations):
            raise ValueError('Policy sampling requested on a waiting/terminal seat')
        cpu=self.cpu_buffer(capacity);arr=cpu.numpy()
        if self.bulk_stack:
            arr[:n]=np.stack([o.flat for o in observations])
            if not arr[:n,SLICES['mask']].any(-1).all():raise ValueError('Policy sampling requested on a waiting/terminal seat')
        for i,obs in enumerate(() if self.bulk_stack else observations):
            if self.flat_copy and isinstance(obs,PackedStorage):arr[i]=obs.flat
            else:
                for key in KEYS:arr[i,SLICES[key]]=np.asarray(obs[key]).reshape(-1)
        self.timings.add('cpu_feature_pack',time.perf_counter()-started,n)
        if self.use_graph:
            key=(id(model),capacity,deterministic)
            entry=self.graphs.get(key)
            if entry is None:
                setup=time.perf_counter()
                # Warm/capture consumes no research-policy RNG: restore its state.
                rng=torch.cuda.get_rng_state(self.device)
                dev=cpu.to(self.device,non_blocking=True)
                stream=torch.cuda.Stream(device=self.device);stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):self.compute(model,unpack_tensor(dev),deterministic)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize(self.device)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output=self.compute(model,unpack_tensor(dev),deterministic)
                torch.cuda.set_rng_state(rng,self.device)
                # Hold model references while captured storage exists; bounded LRU.
                entry=(model,dev,output,graph);self.graphs[key]=entry
                while len(self.graphs)>16:self.graphs.popitem(last=False)
                elapsed=time.perf_counter()-setup;self.graph_setup_seconds+=elapsed
                self.timings.add('cuda_graph_cold_setup',elapsed)
            _,dev,output,graph=entry
            start=time.perf_counter()
            if self.events:self.events[0].record()
            dev.copy_(cpu,non_blocking=True)
            if self.events:self.events[1].record()
            graph.replay()
            if self.events:self.events[2].record()
            returning=time.perf_counter();result=output.cpu().numpy()
            if self.events:
                self.events[3].record();self.events[3].synchronize()
                self.timings.add('h2d_stream_elapsed',self.events[0].elapsed_time(self.events[1])/1000,n)
                self.timings.add('cuda_graph_device_execution',self.events[1].elapsed_time(self.events[2])/1000,n)
            self.timings.add('d2h_and_host_synchronization',time.perf_counter()-returning,n)
            self.timings.add('packed_transfer_inference_return',time.perf_counter()-start,n)
        else:
            start=time.perf_counter()
            if self.events:self.events[0].record()
            dev=cpu.to(self.device,non_blocking=True)
            if self.events:self.events[1].record()
            output=self.compute(model,unpack_tensor(dev),deterministic)
            if self.events:self.events[2].record()
            returning=time.perf_counter();result=output.cpu().numpy()
            if self.events:
                self.events[3].record();self.events[3].synchronize()
                self.timings.add('h2d_stream_elapsed',self.events[0].elapsed_time(self.events[1])/1000,n)
                self.timings.add('policy_device_stream_elapsed',self.events[1].elapsed_time(self.events[2])/1000,n)
            self.timings.add('d2h_and_host_synchronization',time.perf_counter()-returning,n)
            self.timings.add('packed_transfer_inference_return',time.perf_counter()-start,n)
        return result[:n,0].astype(np.int64),result[:n,1],result[:n,2]

class HybridInference:
    """CPU batched actors, CUDA learner; GPU allocation remains fully charged.

Small networks can be dominated by per-turn CUDA launch/round-trip latency.
Synchronize actor parameters once per rollout instead of transferring every state.
This mode has CPU-vs-CUDA sampling streams, so learning parity is statistical.
"""
    def __init__(self,device,timings):
        self.device,self.timings=device,timings
        self.models=OrderedDict();self.synced=set()
        self.cpu_inference=EfficientInference('cpu',timings)
    def begin_rollout(self):self.synced.clear()
    @torch.no_grad()
    def __call__(self,model,observations,deterministic=False):
        key=id(model);entry=self.models.get(key)
        if entry is None:
            from .network import Policy
            # Constructing an actor is not a policy-randomness event.
            with torch.random.fork_rng(devices=[]):actor=Policy(model.hidden,model.embedding_size)
            entry=(model,actor);self.models[key]=entry
            while len(self.models)>8:self.models.popitem(last=False)
        _,actor=entry
        if key not in self.synced:
            start=time.perf_counter()
            actor.load_state_dict({k:v.detach().cpu() for k,v in model.state_dict().items()});actor.eval()
            self.timings.add('actor_weight_gpu_to_cpu_sync',time.perf_counter()-start)
            self.synced.add(key)
        return self.cpu_inference(actor,observations,deterministic)


def batch_flat(observations,device):
    """One bulk cast/H2D for packed observations, preserving the original casts."""
    if not all(isinstance(o,PackedStorage) for o in observations):
        from .network import batch
        return batch(observations,device)
    flat=np.stack([o.flat for o in observations]).astype(np.float32)
    return unpack_tensor(torch.as_tensor(flat,device=device))
