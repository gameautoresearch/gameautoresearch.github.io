"""GPU-owned complete-game backend. Kernel transitions never call the CPU oracle.

This backend remains experimental until CPU/device differential and learning gates pass.
All tensors and actions are CUDA resident. JSON export/import is explicit checkpoint IO.
"""
import ctypes as c
import hashlib
import json
import os
import time
from pathlib import Path
import torch

OBS_FLOATS=3227
OBS_CARDS=124
N_ACTIONS=245


class DeviceEnv:
    _module_cache={}
    _arena_bindings={}
    def __init__(self,n_games,seed=0,controlled=(0,),config=None,ptx=None,heap_bytes=2<<30,stack_bytes=65536,kernels_only=False,allocator=None,arena_bytes=1<<20,persistent_bytes=64<<20,block_threads=None):
        stack_bytes=int(os.environ.get('TAVERN_DEVICE_STACK_BYTES',stack_bytes))
        if stack_bytes<4096:raise ValueError('device stack must be at least 4096 bytes')
        if n_games<1 or n_games>16384:raise ValueError('device batch must be 1..16384')
        self.block_threads=int(os.environ.get('TAVERN_DEVICE_BLOCK_THREADS','32')) if block_threads is None else block_threads
        if self.block_threads not in [1,2,4,8,16,32]:raise ValueError('device block width must be 1,2,4,8,16 or 32')
        if not controlled or len(set(controlled))!=len(controlled) or any(p not in range(8) for p in controlled):
            raise ValueError('distinct controlled seats 0..7 required')
        if not torch.cuda.is_available():raise RuntimeError('actual CUDA required')
        self.n_games=n_games;self.controlled=tuple(controlled);self.k=len(controlled);self.n_slots=n_games*self.k
        self.device=torch.device('cuda:0');self.closed=False
        self.arenas=None;self.allocator=allocator or ('arena' if os.environ.get('TAVERN_DEVICE_ARENA')=='1' else 'cuda')
        self.game_arena=os.environ.get('TAVERN_DEVICE_GAME_ARENA','1')!='0'
        if self.allocator not in ['arena','cuda']:raise ValueError('explicit cuda or arena allocator required')
        def trace(stage):
            if os.environ.get('TAVERN_DEVICE_TRACE_INIT')=='1':
                print(json.dumps({'device_init_stage':stage,'unix':time.time(),'n_games':n_games}),flush=True)
        trace('CUDA context start')
        torch.empty(1,device=self.device)
        trace('CUDA context ready')
        self.driver=c.CDLL('libcuda.so.1')
        self.check(self.driver.cuInit(0))
        self.rules=dict(config or {})
        if self.rules.get('num_players',8)!=8:raise ValueError('frozen eight-player task')
        raw=json.dumps(self.rules,separators=(',',':')).encode()
        self.cfg=torch.tensor(list(raw),dtype=torch.uint8,device=self.device)
        self.seats=torch.tensor(controlled,dtype=torch.int32,device=self.device)
        ptx=Path(ptx or '/workspace/integrations/tavern/device/tavern.ptx')
        self.ptx_sha256=hashlib.sha256(ptx.read_bytes()).hexdigest()
        state_ptx=ptx.with_name('tavern-state.ptx')
        self.split=state_ptx.exists()
        paths={'main':ptx}
        if self.split:paths['state']=state_ptx
        self.module_receipts={};modules={}
        for family,path in paths.items():
            actual=path.with_suffix('.cubin') if path.with_suffix('.cubin').exists() else path
            data=actual.read_bytes()
            self.module_receipts[family]={'ptx_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'module_sha256':hashlib.sha256(data).hexdigest(),'format':actual.suffix}
            modules[family]=data
        self.module_sha256=hashlib.sha256(json.dumps(self.module_receipts,sort_keys=True).encode()).hexdigest()
        self.module_format=self.module_receipts['main']['format']
        key=(self.module_sha256,torch.cuda.current_device());self.module_key=key
        stream=torch.cuda.current_stream().cuda_stream
        if key not in self._module_cache:
            if self._module_cache:raise RuntimeError('one source-bound device module set per process; start fresh for another build')
            self.check(self.driver.cuCtxSetLimit(2,c.c_size_t(heap_bytes)))
            self.check(self.driver.cuCtxSetLimit(0,c.c_size_t(stack_bytes)))
            loaded={}
            for family,data in modules.items():
                module=c.c_void_p();error=c.create_string_buffer(8192);info=c.create_string_buffer(8192)
                options=(c.c_int*4)(5,6,3,4)
                values=(c.c_void_p*4)(c.cast(error,c.c_void_p),c.c_void_p(len(error)),c.cast(info,c.c_void_p),c.c_void_p(len(info)))
                trace('module load start '+family)
                status=self.driver.cuModuleLoadDataEx(c.byref(module),c.c_char_p(data),4,options,values)
                if status:raise RuntimeError(f'CUDA {family} module load {status}: {error.value.decode(errors="replace")}')
                trace('module load ready '+family)
                loaded[family]=module
            self._module_cache[key]=(loaded,{},heap_bytes,stack_bytes,stream,(self.allocator,self.game_arena),persistent_bytes,None)
        self.modules,self.functions,heap,stack,self.stream,cached_allocator,cached_persistent_bytes,self.persistent=self._module_cache[key]
        self.module=self.modules['main']
        if cached_allocator!=(self.allocator,self.game_arena) or cached_persistent_bytes!=persistent_bytes:
            raise RuntimeError('shared device modules require identical allocation modes and persistent storage size')
        if (heap,stack,self.stream)!=(heap_bytes,stack_bytes,stream):
            raise RuntimeError('shared device modules require identical heap/stack and one ordered stream per process')
        if self.allocator=='arena' and self.persistent is None:
            if persistent_bytes<4096 or persistent_bytes%256:raise ValueError('persistent storage must be a positive multiple of 256 bytes')
            self.persistent=torch.empty(persistent_bytes,dtype=torch.uint8,device=self.device)
            self.launch('tavern_bind_persistent',1,[self.persistent,c.c_uint64(persistent_bytes),c.c_uint32(1)])
            if self.split:self.launch('tavern_state_bind_persistent',1,[self.persistent,c.c_uint64(persistent_bytes),c.c_uint32(0)])
            self._module_cache[key]=(self.modules,self.functions,heap,stack,self.stream,cached_allocator,cached_persistent_bytes,self.persistent)
        if self.split:
            trace('shared layout check start')
            main_layout=torch.empty(8,dtype=torch.int64,device=self.device)
            state_layout=torch.empty_like(main_layout)
            self.launch('tavern_layout',1,[main_layout]);self.launch('tavern_state_layout',1,[state_layout])
            if not torch.equal(main_layout,state_layout):raise RuntimeError('simulation/checkpoint device ABI layout mismatch')
            trace('shared layout check ready')
        trace('rule contract start')
        contract=torch.empty(1,dtype=torch.int64,device=self.device)
        self.launch('tavern_contract',1,[contract]);contract=int(contract.item())
        if contract not in range(4):raise RuntimeError('unsupported device table/rule contract')
        self.frozen_rules=bool(contract&1);self.ordered_tables=bool(contract&2)
        if self.frozen_rules:
            frozen={'num_players':8,'content':'Implemented','turn_action_caps':[30],
                'max_actions_per_turn':30,'max_turns':100,'record_log':False}
            if any(key not in frozen or value!=frozen[key] for key,value in self.rules.items()):
                raise ValueError('this module implements the frozen eight-player benchmark rule configuration')
            self.rules=frozen;raw=json.dumps(frozen,separators=(',',':')).encode()
            self.cfg=torch.tensor(list(raw),dtype=torch.uint8,device=self.device)
        trace('rule contract ready')
        self.limits={}
        for name,which in [('stack_bytes',0),('malloc_heap_bytes',2)]:
            value=c.c_size_t();self.check(self.driver.cuCtxGetLimit(c.byref(value),which));self.limits[name]=value.value
        if kernels_only:
            self.closed=True
            return
        if self.allocator=='arena' and self.game_arena:
            if arena_bytes<4096 or arena_bytes%256:raise ValueError('per-game arena must be a multiple of 256 bytes, at least 4096')
            self.arena_bytes=arena_bytes
            self.arenas=torch.empty((n_games,arena_bytes),dtype=torch.uint8,device=self.device)
            self.launch('tavern_bind_arenas',n_games,[self.arenas,c.c_uint32(n_games),c.c_uint64(arena_bytes),c.c_uint32(1)])
            self._arena_bindings[(key,'main')]=id(self)
        self.handles=torch.zeros(n_games,dtype=torch.int64,device=self.device)
        self.active=torch.ones(n_games,dtype=torch.uint8,device=self.device)
        self.features=torch.empty((self.n_slots,OBS_FLOATS),dtype=torch.float32,device=self.device)
        self.cards=torch.empty((self.n_slots,OBS_CARDS),dtype=torch.int32,device=self.device)
        self.masks=torch.empty((self.n_slots,N_ACTIONS),dtype=torch.uint8,device=self.device)
        self.ready=torch.empty(self.n_slots,dtype=torch.uint8,device=self.device)
        self.rewards=torch.empty(self.n_slots,dtype=torch.float32,device=self.device)
        self.dones=torch.empty(self.n_slots,dtype=torch.uint8,device=self.device)
        self.places=torch.empty_like(self.dones);self.rejected=torch.empty_like(self.dones);self.stop=torch.empty_like(self.dones)
        trace('database/registry prewarm start')
        self.launch('tavern_prewarm',1,[])
        if self.ordered_tables and self.split:self.launch('tavern_state_prewarm',1,[])
        if os.environ.get('TAVERN_DEVICE_TRACE_INIT')=='1':torch.cuda.synchronize()
        trace('database/registry prewarm enqueued or ready')
        trace('game initialization start')
        self.launch('tavern_init',n_games,[self.handles,c.c_uint64(seed),c.c_uint32(n_games),self.seats,
            c.c_uint32(self.k),self.cfg,c.c_uint32(len(raw))])
        torch.cuda.synchronize()
        trace('game initialization ready')

    def check(self,status):
        if status:
            name=c.c_char_p()
            self.driver.cuGetErrorName(status,c.byref(name))
            raise RuntimeError(f'CUDA driver {status}: {name.value.decode() if name.value else "unknown"}')

    def launch(self,name,n,args):
        if torch.cuda.current_stream().cuda_stream!=self.stream:
            raise RuntimeError('complete-rule kernels require their owning ordered CUDA stream')
        family='state' if name.startswith('tavern_state_') else 'main'
        binding=(self.module_key,family)
        bind_name='tavern_state_bind_arenas' if family=='state' else 'tavern_bind_arenas'
        if self.arenas is not None and name!=bind_name and self._arena_bindings.get(binding)!=id(self):
            self.launch(bind_name,self.n_games,[self.arenas,c.c_uint32(self.n_games),c.c_uint64(self.arena_bytes),c.c_uint32(0)])
            self._arena_bindings[binding]=id(self)
        if name not in self.functions:
            f=c.c_void_p();self.check(self.driver.cuModuleGetFunction(c.byref(f),self.modules[family],name.encode()));self.functions[name]=f
        values=[c.c_void_p(x.data_ptr()) if isinstance(x,torch.Tensor) else x for x in args]
        pointers=(c.c_void_p*len(values))(*[c.cast(c.byref(x),c.c_void_p) for x in values]) if values else None
        stream=c.c_void_p(torch.cuda.current_stream().cuda_stream)
        # One logical thread still owns each complete lobby. Sparse warp blocks
        # can reduce control-flow divergence between independent lobbies. Packing
        # kernels retain full warps for contiguous checkpoint transfers.
        block=32 if name in ['tavern_pack_states','tavern_state_pack'] else self.block_threads
        self.check(self.driver.cuLaunchKernel(self.functions[name],(n+block-1)//block,1,1,block,1,1,0,stream,pointers,None))
        if os.environ.get('TAVERN_DEVICE_TRACE_KERNELS')=='1':
            try:self.check(self.driver.cuStreamSynchronize(stream))
            except RuntimeError as exc:raise RuntimeError(f'{name}: {exc}') from exc

    def kernel_resources(self):
        """Read compiler resource attributes without launching timed kernels."""
        result={}
        for name in ['tavern_init','tavern_observe','tavern_step','tavern_state_export']:
            if name.startswith('tavern_state_') and not self.split:continue
            family='state' if name.startswith('tavern_state_') else 'main'
            if name not in self.functions:
                function=c.c_void_p()
                self.check(self.driver.cuModuleGetFunction(c.byref(function),self.modules[family],name.encode()))
                self.functions[name]=function
            row={}
            for label,attribute in [('maximum_threads_per_block',0),('shared_bytes',1),('local_bytes_per_thread',3),('registers_per_thread',4)]:
                value=c.c_int()
                self.check(self.driver.cuFuncGetAttribute(c.byref(value),attribute,self.functions[name]))
                row[label]=value.value
            result[name]=row
        return result

    def observe(self):
        self.launch('tavern_observe',self.n_games,[self.handles,c.c_uint32(self.n_games),self.seats,c.c_uint32(self.k),
            self.active,self.features,self.cards,self.masks,self.ready])
        return {'features':self.features,'cards':self.cards,'action_mask':self.masks.view(torch.bool),'ready':self.ready.view(torch.bool)}

    def step(self,actions):
        if actions.device!=self.device or actions.dtype!=torch.int64 or actions.numel()!=self.n_slots:
            raise ValueError('one CUDA int64 action per full slot required')
        self.launch('tavern_step',self.n_games,[self.handles,actions,c.c_uint32(self.n_games),self.seats,c.c_uint32(self.k),
            self.active,self.rewards,self.dones,self.places,self.rejected,self.stop])
        return self.rewards,self.dones.view(torch.bool),{'placement':self.places,'rejected':self.rejected.view(torch.bool),
            'stop_reason':self.stop,'terminated':self.stop==1,'truncated':self.stop>=2}

    def save(self,indices=None):
        handles=self.handles;n=self.n_games
        if indices is not None:
            ids=list(indices)
            if not ids or len(set(ids))!=len(ids) or any(not isinstance(i,int) or i<0 or i>=self.n_games for i in ids):
                raise ValueError('distinct valid checkpoint game indices required')
            handles=self.handles.index_select(0,torch.tensor(ids,dtype=torch.int64,device=self.device));n=len(ids)
        capacity=262144
        while True:
            output=torch.empty((n,capacity),dtype=torch.uint8,device=self.device)
            lengths=torch.empty(n,dtype=torch.int32,device=self.device)
            self.launch('tavern_state_export' if self.split else 'tavern_export',n,[handles,c.c_uint32(n),output,c.c_uint32(capacity),lengths])
            sizes=lengths.cpu().tolist();largest=max(sizes)
            if largest<=capacity:
                import numpy as np
                offsets=np.concatenate((np.zeros(1,np.int64),np.cumsum(sizes,dtype=np.int64)))
                positions=torch.from_numpy(offsets[:-1]).to(self.device)
                packed=torch.empty(int(offsets[-1]),dtype=torch.uint8,device=self.device)
                self.launch('tavern_state_pack' if self.split else 'tavern_pack_states',n*32,
                    [output,packed,lengths,positions,c.c_uint32(n),c.c_uint32(capacity)])
                data=packed.cpu().numpy()
                return [bytes(data[offsets[i]:offsets[i+1]]).decode() for i in range(n)]
            capacity=1<<(largest-1).bit_length()

    def load(self,states):
        if len(states)!=self.n_games:raise ValueError('one exact episode state per game')
        for state in states:
            item=json.loads(state)
            if item.get('k')!=self.k or tuple(item.get('controlled',[])[:self.k])!=self.controlled:
                raise ValueError('saved controlled seats differ')
            if len(item['game']['players'])!=8 or item['game']['cfg'].get('num_players')!=8:
                raise ValueError('saved state is outside the frozen eight-player task')
            if any(item['game']['cfg'].get(key,value)!=value for key,value in self.rules.items()):
                raise ValueError('saved explicit rule configuration differs')
        raw=[s.encode() for s in states];capacity=max(map(len,raw))
        import numpy as np
        arr=np.zeros((len(raw),capacity),np.uint8)
        for i,s in enumerate(raw):arr[i,:len(s)]=np.frombuffer(s,np.uint8)
        data=torch.from_numpy(arr).to(self.device)
        lengths=torch.tensor([len(x) for x in raw],dtype=torch.int32,device=self.device)
        status=torch.empty(self.n_games,dtype=torch.int32,device=self.device)
        if self.split:
            parsed=torch.zeros(self.n_games,dtype=torch.int64,device=self.device)
            self.launch('tavern_state_parse',self.n_games,[parsed,c.c_uint32(self.n_games),data,c.c_uint32(capacity),lengths,status])
            self.launch('tavern_validate_import',self.n_games,[self.handles,parsed,c.c_uint32(self.n_games),status,self.seats,c.c_uint32(self.k)])
            if (status.cpu()!=0).any():
                self.launch('tavern_state_destroy',self.n_games,[parsed,c.c_uint32(self.n_games)])
                torch.cuda.synchronize()
                raise ValueError('device state/rules/card-table validation failed; running states preserved')
            self.launch('tavern_commit_import',self.n_games,[self.handles,parsed,c.c_uint32(self.n_games)])
            torch.cuda.synchronize()
        else:
            self.launch('tavern_import',self.n_games,[self.handles,c.c_uint32(self.n_games),data,c.c_uint32(capacity),lengths,status,self.seats,c.c_uint32(self.k)])
            if (status.cpu()!=0).any():raise ValueError('device state/rules/card-table validation failed')

    def close(self):
        if not self.closed:
            self.launch('tavern_destroy',self.n_games,[self.handles,c.c_uint32(self.n_games)])
            torch.cuda.synchronize();self.closed=True;self.arenas=None
            for family in self.modules:
                binding=(self.module_key,family)
                if self._arena_bindings.get(binding)==id(self):self._arena_bindings.pop(binding)
    def allocator_stats(self):
        if self.allocator!='arena':return {'allocator':'CUDA device malloc/free'}
        if self.arenas is not None:
            result=torch.empty((self.n_games,4),dtype=torch.int64,device=self.device)
            self.launch('tavern_arena_stats',self.n_games,[c.c_uint32(self.n_games),result])
            values=result.cpu().numpy()
            game={'allocator':'per-game size classes with CUDA spill fallback','reserved_bytes_per_game':self.arena_bytes,
                'maximum_bump_bytes':int(values[:,0].max()),'maximum_live_peak_bytes':int(values[:,1].max()),
                'arena_allocations':int(values[:,2].sum()),'CUDA_spills':int(values[:,3].sum())}
        else:game={'allocator':'individual CUDA malloc/free; persistent tables retain shared arena','reserved_bytes_per_game':0}
        persistent=torch.empty(4,dtype=torch.int64,device=self.device)
        self.launch('tavern_persistent_stats',1,[persistent]);persistent_values=persistent.cpu().tolist()
        return {**game,'persistent_reserved_bytes':self.persistent.numel(),
            'persistent_bump_bytes':persistent_values[0],'persistent_peak_bytes':persistent_values[1],
            'persistent_allocations':persistent_values[2],'persistent_CUDA_spills':persistent_values[3]}
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
