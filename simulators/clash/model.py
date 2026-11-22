"""Entity/card-aware PPO policy. No game mechanics are reimplemented here."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

CARD_FIELDS = ('hand_card','next_card','deck','enemy_seen','pending_card',
               'ability_champion','enemy_play_card','enemy_play_cast','queue')
GLOBAL_FIELDS = ('tick','elixir','elixir_reserved','crowns','crowns_enemy','hand_refill_ms',
                 'hand_cost','hand_pending','own_king_hp','own_king_max_hp','own_left_hp',
                 'own_left_max_hp','own_right_hp','own_right_max_hp','enemy_king_hp',
                 'enemy_king_max_hp','enemy_left_hp','enemy_left_max_hp','enemy_right_hp',
                 'enemy_right_max_hp','entity_count','entity_overflow','pending_count','time_ms',
                 'section_left_ms','full_bar_ms','overtime','play_open_in_ticks','pending_x',
                 'pending_y','pending_ticks','hand_form','ability_cooldown_ms','ability_charges',
                 'ability_command_ticks','enemy_play_count','enemy_elixir','enemy_play_ticks',
                 'enemy_play_x','enemy_play_y','enemy_play_point_ticks','deck_level')
ENTITY_NUM = ('x','y','hp','max_hp','shield','level','dx','dy','target_x','target_y',
              'deploy_ms','attack_ms','load_ms','speed_percent','hit_speed_percent',
              'buff_count','buff_ms_0','buff_ms_1','buff_ms_2','aim_x','aim_y',
              'time_left_ms','radius','ability_ticks')

def scale(name):
    if name in ('elixir','elixir_reserved','enemy_elixir'): return 100000.
    if 'hp' in name or name == 'shield': return 5000.
    if name.endswith('_x') or name in ('x','dx'): return 18000.
    if name.endswith('_y') or name in ('y','dy'): return 32000.
    if 'ms' in name: return 10000.
    if 'tick' in name: return 1200.
    if 'percent' in name: return 100.
    return 10.

class Policy(nn.Module):
    def __init__(self, schema, vocab_size, buffs, width=64, embedding=12):
        super().__init__()
        self.s, self.width, self.embedding = schema, width, embedding
        self.card_idx = [i for f in CARD_FIELDS for i in schema.header[f]]
        self.global_idx = [i for f in GLOBAL_FIELDS for i in schema.header[f]]
        self.entity_num_idx = [schema.entity[f] for f in ENTITY_NUM]
        self.entity_cat_idx = [schema.entity[f] for f in ('kind','side','state')]
        self.buff_idx = [schema.entity[f'buff_{i}'] for i in range(schema.buff_slots)]
        for name,indices in (('card_indices',self.card_idx),('global_indices',self.global_idx),('entity_num_indices',self.entity_num_idx),('buff_indices',self.buff_idx),('hand_indices',list(schema.header['hand_card'])),('cost_indices',list(schema.header['hand_cost']))):
            self.register_buffer(name,torch.tensor(indices,dtype=torch.long),persistent=False)
        self.register_buffer('global_scale',torch.tensor([scale(f) for f in GLOBAL_FIELDS for _ in schema.header[f]]))
        self.register_buffer('entity_scale',torch.tensor([scale(f) for f in ENTITY_NUM]))
        self.register_buffer('flag_bits',2**torch.arange(len(schema.flag_names)))
        self.cards = nn.Embedding(vocab_size+1,embedding,padding_idx=0)
        self.buffs = nn.Embedding(buffs+1,4,padding_idx=0)
        self.kind = nn.Embedding(8,4)
        self.side = nn.Embedding(4,3)
        self.state = nn.Embedding(len(schema.state_names)+1,5)
        ein = embedding+4+3+5+4*schema.buff_slots+len(ENTITY_NUM)+len(schema.flag_names)
        self.entity_net=nn.Sequential(nn.Linear(ein,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh())
        self.global_net=nn.Sequential(nn.Linear(len(self.card_idx)*embedding+len(self.global_idx)+2*width,width*2),nn.Tanh(),nn.Linear(width*2,width),nn.Tanh())
        self.value=nn.Linear(width,1)
        self.slot=nn.Linear(width,1+schema.hand_slots+schema.ability_slots)
        self.hand_query=nn.Sequential(nn.Linear(width+embedding+1,width),nn.Tanh())
        self.tiles=nn.Parameter(torch.randn(schema.cols*schema.rows,width)*.02)
        nn.init.zeros_(self.slot.weight); nn.init.zeros_(self.slot.bias)
        nn.init.zeros_(self.value.bias)

    def pack(self, observations, masks=None, device='cpu', trim=True, bucket_override=None):
        """Copy only live entity rows to the learner; the original JVM buffers stay shared.

        CUDA pinned mode returns borrowed reusable buffers, valid until a later pack of that shape.
        The learner clones its full-rollout cache before reuse; staging is bounded to 8 shapes.
        Dense mode is retained as a correctness/performance control. Per-batch maximum rather
        than a hard entity cap preserves every entity the upstream observation exposes.
        """
        o=np.asarray(observations).reshape(-1,self.s.seat_ints)
        n=max(1,int(o[:,self.s.header['entity_count'][0]].max())) if trim else self.s.max_entities
        bucket=getattr(self,'entity_bucket',0) if bucket_override is None else bucket_override
        if bucket:n=((n+bucket-1)//bucket)*bucket
        n=min(n,self.s.max_entities)
        rows=o[:,self.s.header_ints:].reshape(-1,self.s.max_entities,self.s.entity_ints)[:,:n]
        if getattr(self,'pinned_transfer',False) and str(device).startswith('cuda'):
            arrays={'header':o[:,:self.s.header_ints], 'entities':rows}
            if masks is not None:arrays['mask']=np.asarray(masks).reshape(-1,self.s.actions).astype(bool)
            key=tuple((name,a.shape,str(a.dtype)) for name,a in arrays.items())
            if not hasattr(self,'_transfer_cache'):self._transfer_cache={}
            if key in self._transfer_cache:
                self._transfer_cache[key]=self._transfer_cache.pop(key)
            if key not in self._transfer_cache:
                limit=getattr(self,'transfer_cache_limit_bytes',256*1024*1024)
                wanted=sum(a.nbytes for a in arrays.values())
                def cache_bytes():return sum(t.numel()*t.element_size() for host,_ in self._transfer_cache.values() for t in host.values())
                # Large one-off rollout/value tensors must not evict reusable actor
                # buffers or accumulate eight full-rollout copies in pinned RAM.
                if wanted<=limit:
                    while self._transfer_cache and (len(self._transfer_cache)>=8 or cache_bytes()+wanted>limit):self._transfer_cache.pop(next(iter(self._transfer_cache)))
                host={name:torch.empty(a.shape,dtype=torch.from_numpy(np.empty((),a.dtype)).dtype,pin_memory=True) for name,a in arrays.items()}
                gpu={name:torch.empty_like(value,device=device) for name,value in host.items()}
                if wanted<=limit:self._transfer_cache[key]=(host,gpu)
                self.staging_cache_peak_bytes=max(getattr(self,'staging_cache_peak_bytes',0),cache_bytes()+(wanted if wanted>limit else 0))
            else:host,gpu=self._transfer_cache[key]
            for name,a in arrays.items():
                host[name].copy_(torch.from_numpy(a));gpu[name].copy_(host[name],non_blocking=True)
            return gpu
        # torch.tensor intentionally owns a snapshot: the next vector step overwrites the views.
        out={'header':torch.tensor(o[:,:self.s.header_ints].copy(),device=device),
             'entities':torch.tensor(rows.copy(),device=device)}
        if masks is not None: out['mask']=torch.tensor(np.asarray(masks).reshape(-1,self.s.actions).copy(),device=device,dtype=torch.bool)
        return out

    def encode(self, batch):
        h,e=batch['header'].long(),batch['entities'].long()
        alive=torch.arange(e.shape[1],device=e.device)[None,:] < h[:,self.s.header['entity_count'][0],None]
        # Encode live entities once, then restore the original padded reduction layout.
        # Keeping sum/max on the same layout preserves first-index max subgradients,
        # including ties; segment max would silently change that learning behavior.
        ragged=getattr(self,'ragged_entities',False)
        if ragged:
            layout=e.shape[:2]
            e=e[alive]
        card=self.cards((h[:,self.card_indices if getattr(self,'device_indices',False) else self.card_idx]+1).clamp(0,self.cards.num_embeddings-1)).flatten(1)
        g=(h[:,self.global_indices if getattr(self,'device_indices',False) else self.global_idx].float()/self.global_scale).clamp(-10,10)
        ec=self.cards((e[...,self.s.entity['row']]+1).clamp(0,self.cards.num_embeddings-1))
        ef=[ec,self.kind(e[...,self.entity_cat_idx[0]].clamp(0,7)),
            self.side(e[...,self.entity_cat_idx[1]].clamp(0,3)),
            self.state((e[...,self.entity_cat_idx[2]]+1).clamp(0,self.state.num_embeddings-1)),
            self.buffs((e[...,self.buff_indices if getattr(self,'device_indices',False) else self.buff_idx]+1).clamp(0,self.buffs.num_embeddings-1)).flatten(-2),
            (e[...,self.entity_num_indices if getattr(self,'device_indices',False) else self.entity_num_idx].float()/self.entity_scale).clamp(-10,10),
            ((e[...,self.s.entity['flags'],None]&self.flag_bits)!=0).float()]
        z=self.entity_net(torch.cat(ef,-1))
        if ragged:
            padded=z.new_zeros((*layout,self.width))
            padded[alive]=z
            z=padded
        mean=(z*alive[:,:,None]).sum(1)/alive.sum(1).clamp_min(1)[:,None]
        maximum=z.masked_fill(~alive[:,:,None],-1e4).max(1).values
        maximum=torch.where(alive.any(1)[:,None],maximum,torch.zeros_like(maximum))
        context=self.global_net(torch.cat((card,g,mean,maximum),-1))
        return context,h

    def values(self,batch):
        context,_=self.encode(batch)
        return self.value(context).squeeze(-1)

    def forward(self,batch):
        context,h=self.encode(batch)
        value=self.value(context).squeeze(-1)
        mask=batch['mask']
        if getattr(self,'sparse_policy_head',False):
            # A wait-only row has a point-mass policy but still needs its value.
            # Compute even an empty head so Adam receives zero gradients (not None)
            # for policy parameters when a whole minibatch is forced.
            active=mask[:,1:].any(1)|~mask[:,0]
            logits=context.new_full(mask.shape,-1e9)
            logits[:,0]=0
            logits[active]=self.policy_logits(context[active],h[active],mask[active])
        else:
            logits=self.policy_logits(context,h,mask)
        return logits,value

    def policy_logits(self,context,h,mask):
        hands=self.cards((h[:,self.hand_indices if getattr(self,'device_indices',False) else list(self.s.header['hand_card'])]+1).clamp(0,self.cards.num_embeddings-1))
        costs=h[:,self.cost_indices if getattr(self,'device_indices',False) else list(self.s.header['hand_cost'])].float().unsqueeze(-1)/10
        query=self.hand_query(torch.cat((context[:,None,:].expand(-1,self.s.hand_slots,-1),hands,costs),-1))
        spatial=query@self.tiles.T
        smask=mask[:,1:self.s.first_ability].reshape(-1,self.s.hand_slots,self.s.cols*self.s.rows)
        spatial=spatial.masked_fill(~smask,-1e9)
        # Normalize placements within a slot, avoiding a 576:1 initial play-vs-wait prior.
        spatial=spatial-torch.logsumexp(spatial,-1,keepdim=True)
        slots=self.slot(context)
        plays=(spatial+slots[:,1:1+self.s.hand_slots,None]).flatten(1)
        logits=torch.cat((slots[:,:1],plays,slots[:,1+self.s.hand_slots:]),-1).masked_fill(~mask,-1e9)
        return logits

    def action_terms(self,batch,actions):
        """All-state values and exact point-mass terms for forced-wait rows.

        The active-row categorical is unchanged. Zero-size head operations keep
        zero gradients present for Adam even on an entirely forced minibatch.
        """
        context,h=self.encode(batch)
        mask=batch['mask'];active=mask[:,1:].any(1)|~mask[:,0]
        d=Categorical(logits=self.policy_logits(context[active],h[active],mask[active]),validate_args=False)
        logp=context.new_zeros(len(h));entropy=context.new_zeros(len(h))
        logp[active]=d.log_prob(actions[active]);entropy[active]=d.entropy()
        return logp,entropy,self.value(context).squeeze(-1)

    def distribution(self,batch):
        if getattr(self,'graph_inference',False) and batch['header'].is_cuda and not torch.is_grad_enabled():
            if not hasattr(self,'_inference_graphs'):self._inference_graphs={}
            key=tuple((k,v.data_ptr(),tuple(v.shape)) for k,v in batch.items())
            if key not in self._inference_graphs:
                if len(self._inference_graphs)>=getattr(self,'inference_graph_cache_size',2):self._inference_graphs.pop(next(iter(self._inference_graphs)))
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):self(batch)
                torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):outputs=self(batch)
                self._inference_graphs[key]=(graph,batch,outputs)
            graph,_,(logits,value)=self._inference_graphs[key];graph.replay()
        else:logits,value=self(batch)
        return Categorical(logits=logits,validate_args=False),value

    def sample_batch(self,batch,greedy=False):
        """Capture the complete actor, including categorical sampling and log-probability.

        Capture/warmup cannot consume the research policy's RNG stream. Replays use
        PyTorch's graph-aware default CUDA generator, which checkpoints already save.
        Returned tensors are borrowed until the next replay of this graph.
        """
        def act():
            logits,value=self(batch)
            d=Categorical(logits=logits,validate_args=False)
            action=d.logits.argmax(-1) if greedy else d.sample()
            return action,d.log_prob(action),value
        if not batch['header'].is_cuda or not getattr(self,'graph_sampling',False):return act()
        if torch.is_grad_enabled():raise RuntimeError('actor graph is inference-only')
        if not hasattr(self,'_action_graphs'):self._action_graphs={}
        key=(greedy,tuple((k,v.data_ptr(),tuple(v.shape)) for k,v in batch.items()))
        if key not in self._action_graphs:
            if len(self._action_graphs)>=getattr(self,'inference_graph_cache_size',2):self._action_graphs.pop(next(iter(self._action_graphs)))
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            rng=torch.cuda.get_rng_state()
            try:
                with torch.cuda.stream(stream):
                    for _ in range(2):act()
                torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):outputs=act()
            finally:torch.cuda.set_rng_state(rng)
            self._action_graphs[key]=(graph,batch,outputs)
        graph,_,outputs=self._action_graphs[key];graph.replay()
        return outputs
