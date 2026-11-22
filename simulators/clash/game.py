"""Pinned local crforge task, manifests, and direct-buffer vector lifecycle."""
from __future__ import annotations
import hashlib,json,os,platform,subprocess,sys,tarfile,time
from pathlib import Path
from collections import Counter
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
ENGINE=ROOT/'candidates/crforge'
PIN='f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de'
JAVA=Path('/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home')
if 'JAVA_HOME' not in os.environ and JAVA.exists(): os.environ['JAVA_HOME']=str(JAVA)
from crforge_env import VectorBattle,LadderConfig,ScriptedBot
import crforge_env


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def dump(path,x): Path(path).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def hashes(path): return {str(p.relative_to(path)):sha(p) for p in sorted(Path(path).rglob('*')) if p.is_file()}
def resolve(p):
    p=Path(p).expanduser()
    return p if p.is_absolute() else ROOT/p

class WorkerClock:
    """Elapsed allocated worker time, charged even when GPU is idle. Not utilization."""
    def __init__(self,limit,prior=0.):
        self.start=time.perf_counter(); self.prior=prior; self.limit=limit; self.gpu_allocated=False
    @property
    def elapsed(self): return self.prior+time.perf_counter()-self.start
    def exhausted(self): return self.elapsed>=self.limit
    def report(self): return {'allocated_worker_seconds':self.elapsed,'charged_worker_hours':self.elapsed/3600,'allocated_gpu_hours':self.elapsed/3600 if self.gpu_allocated else None,
                              'gpu_allocation_executed':self.gpu_allocated,
                              'limit_seconds':self.limit,'budget_overshoot_seconds':max(0.,self.elapsed-self.limit),'accounting':'elapsed allocated worker; includes CPU simulation, initialization, transfers, inference, updates, checkpointing and development evaluation'}


def load_config(path):
    c=json.loads(Path(path).read_text())
    if c['task']!='clash-fixed8-v1': raise ValueError('unknown task')
    if c['device'] not in ('cpu','cuda','mps'): raise ValueError('explicit cpu/cuda/mps device required')
    for k in ('envs','threads','rollout_steps','minibatch','epochs','torch_threads'):
        if c[k]<1: raise ValueError(k+' must be positive')
    if c['budget_seconds']<=0: raise ValueError('positive elapsed worker budget required')
    allocation=c['hardware_allocation']
    if max(c['threads'],c['torch_threads'])>allocation['cpu_threads']:raise ValueError('thread counts exceed declared CPU allocation')
    if c['device']=='cuda' and (allocation['gpu_count']!=1 or allocation['ram_gib'] is None or allocation['gpu_model'].startswith('SET')):raise ValueError('CUDA scoring requires one specified GPU, CPU and RAM allocation')
    for key in ('value_batch_size','inference_graph_cache_size','transfer_cache_limit_bytes'):
        if key in c and (not isinstance(c[key],int) or c[key]<1):raise ValueError(key+' must be a positive integer')
    for key in ('actor_batch_bucket','actor_chunk_size','entity_bucket','inference_entity_bucket'):
        if c.get(key) is not None and (not isinstance(c[key],int) or c[key]<0):raise ValueError(key+' must be a nonnegative integer')
    if c.get('deferred_values',False) and c.get('graph_sampling',False):raise ValueError('deferred_values and graph_sampling are separate actor backends')
    if not 0<=c.get('source_compresslevel',9)<=9:raise ValueError('source_compresslevel must be 0..9')
    return c


def make_vector(c,count=None):
    lock=json.loads((HERE/'rules.lock.json').read_text())
    tables=resolve(c['tables']); install=resolve(c['install'])
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ENGINE,text=True).strip()
    if commit!=PIN: raise ValueError('engine commit differs from task pin')
    if hashes(tables)!=lock['table_sha256']: raise ValueError('game tables differ from pinned task hashes')
    # Detect changed mechanics and stale Java distributions, not just changed git HEAD.
    for relative,want in lock['engine_source_sha256'].items():
        if sha(ENGINE/relative)!=want: raise ValueError('engine source changed: '+relative)
    for p in (ENGINE/'python/crforge_env').glob('*.py'):
        installed=Path(crforge_env.__file__).parent/p.name
        if sha(installed)!=sha(p):raise ValueError('installed Python wrapper differs from pinned source: '+p.name)
    if hashes(install/'lib')!=lock['jar_sha256']: raise ValueError('built distribution differs from pinned task')
    if c['rules']!=lock['rules']: raise ValueError('rules differ: create a separately versioned task')
    v=VectorBattle(count or c['envs'],c['threads'],install=install,tables=tables,
        jvm_args=[f"-XX:ActiveProcessorCount={c['hardware_allocation']['cpu_threads']}"]+([f"-Xmx{c['hardware_allocation']['ram_gib']//2}g"] if c['hardware_allocation']['ram_gib'] else []),
        config=LadderConfig(**{k:c['rules'][k] for k in ('decision_ticks','level','tower_level','max_ticks','mask_version')}))
    v.set_autoreset(False);v.set_autoreset_seeds(*c['train_seed_range'])
    return v


def new_run(out,c,v,mode,clock):
    out=resolve(out);out.mkdir(parents=True,exist_ok=False)
    files=[p for p in HERE.rglob('*') if p.is_file() and (p.suffix in ('.py','.json','.md','.txt','.h','.hpp','.c','.cc','.cpp','.cu') or p.name in ('LICENSE','NOTICE')) and not any(part in ('__pycache__','.build') or part.startswith('generated') for part in p.parts)]
    sources={str(p.relative_to(HERE)):sha(p) for p in sorted(files)}
    with tarfile.open(out/'source.tar.gz','w:gz',compresslevel=c.get('source_compresslevel',9)) as tar:
        for p in sorted(files): tar.add(p,arcname='integration/'+str(p.relative_to(HERE)))
        tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ENGINE).decode().split('\0')
        for name in tracked:
            if name and (ENGINE/name).is_file(): tar.add(ENGINE/name,arcname='crforge/'+name)
    import torch
    manifest={'format':1,'mode':mode,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'config':c,'engine_commit':PIN,'rules_lock':json.loads((HERE/'rules.lock.json').read_text()),
        'metadata':v.metadata,'integration_source_sha256':sources,'source_archive_sha256':sha(out/'source.tar.gz'),
        'hardware':{'platform':platform.platform(),'machine':platform.machine(),'cpu':platform.processor(),
            'logical_cpus':os.cpu_count(),'torch_threads':torch.get_num_threads(),'device':c['device'],
            'cuda_device':torch.cuda.get_device_name() if c['device']=='cuda' else None,
            'allocation':c['hardware_allocation']},
        'runtime':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
                   'float32_matmul_precision':torch.get_float32_matmul_precision(),'cuda_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                   'java':subprocess.check_output([str(Path(os.environ['JAVA_HOME'])/'bin/java'),'-version'],stderr=subprocess.STDOUT,text=True)},
        'initial_clock':clock.report(),'allocation_ledger':str(clock.ledger_path),'runtime_artifact':{'file':'evidence/clash-implementation/pinned-env-dist.tar.gz','sha256':sha(ROOT/'evidence/clash-implementation/pinned-env-dist.tar.gz')},'seed_contract':{'training_range':c['train_seed_range'],
                   'engine_seeds':'explicit int32 per reset; saved in checkpoint and episodes.jsonl',
                   'evaluation':'paired same engine seed, policy switches seats; explicit schedule'},
        'fidelity':'fixed-deck emulator task; private upstream reference corpus absent; commercial-game parity unverified'}
    manifest['manifest_sha256']=canonical(manifest)
    dump(out/'manifest.json',manifest)
    (out/'manifest.json').chmod(0o444);(out/'source.tar.gz').chmod(0o444)
    return out,manifest


def status(v):
    s=v.schema;i=v.infos
    return {'fault':i[:,s.info['fault']].copy(),'truncated':i[:,s.info['truncated']].copy(),
            'unsupported':i[:,s.info['not_modelled']].copy(),
            'illegal':i[:,[s.info['illegal_0'],s.info['illegal_1']]].copy(),
            'winner':i[:,s.info['winner']].copy()}

class Session:
    def __init__(self,v,c,rng,out=None):
        self.v,self.c,self.rng,self.out=v,c,rng,out
        lo,hi=c['train_seed_range'];self.seeds=rng.integers(lo,hi+1,v.count).tolist()
        self.histories=[[] for _ in range(v.count)]
        self.episode_bad=np.zeros(v.count,dtype=bool)
        self.episode_seat=np.arange(v.count)%2
        self.opponent_ids=np.full(v.count,-1,dtype=np.int64)
        self.counters=Counter();self.teacher_counts={};self.reward_sum=0.
        self.reset_all()
    def reset_all(self):
        deck=self.c['rules']['deck'];self.v.reset(self.seeds,deck,deck)
        if self.v.dones.any(): raise RuntimeError('task faulted at reset')
    def step(self,actions):
        v=self.v;s=v.schema
        before=v.infos[:,s.info['tick']].copy()
        # Actions are owned by Python in history, while the live action/observation arrays remain direct views.
        if self.c.get('vector_bookkeeping',False):
            # One C-level conversion of the action buffer avoids per-seat NumPy
            # scalars. Histories retain every action and the original lane order.
            for history,row in zip(self.histories,np.asarray(actions).tolist()):history.append(row)
            action_array=np.asarray(actions)
            env,seat=np.nonzero((action_array>0)&(action_array<s.first_ability))
            if len(env):
                slots=(action_array[env,seat]-1)//(s.cols*s.rows)
                card=v.observations[env,seat,np.asarray(s.header['hand_card'])[slots]]
                unique,counts=np.unique(card,return_counts=True)
                for card,count in zip(unique,counts):
                    name=v.vocabulary[card] if 0<=card<len(v.vocabulary) else 'unknown'
                    self.counters['card_play_requests_'+name]+=int(count)
        else:
            for i,a in enumerate(actions):
                self.histories[i].append([int(a[0]),int(a[1])])
                for seat,action in enumerate(a):
                    if 0<int(action)<s.first_ability:
                        slot,_,_=s.decode(int(action));card=int(v.observations[i,seat,s.header['hand_card'][slot]])
                        name=v.vocabulary[card] if 0<=card<len(v.vocabulary) else 'unknown'
                        self.counters['card_play_requests_'+name]+=1
        v.step(actions)
        self.counters['environment_decisions']+=v.count
        self.counters['player_decisions']+=v.count*2
        self.counters['ticks']+=int(np.maximum(0,v.infos[:,s.info['tick']]-before).sum())
        st=status(v)
        self.counters['illegal_actions']+=int(st['illegal'].sum())
        self.counters['unsupported_actions']+=int(st['unsupported'].sum())
        self.counters['entity_overflow_observations']+=int((v.observations[:,:,s.header['entity_overflow'][0]]>0).sum())
        self.episode_bad|=(st['unsupported']>0)|(st['fault']>0)
        for i in np.flatnonzero(v.dones):
            fault=int(st['fault'][i]);trunc=bool(st['truncated'][i]);winner=int(st['winner'][i])
            if fault: reason='fault';self.counters['fault_'+s.fault_names[fault]]+=1
            elif self.episode_bad[i]: reason='unsupported_episode'
            elif trunc or winner==s.winner_none: reason='truncation'
            else: reason='terminal';self.counters['natural_terminals']+=1
            if reason=='terminal':
                self.counters['draws' if winner==s.winner_draw else 'wins' if winner==self.episode_seat[i] else 'losses']+=1
            self.counters['episodes']+=1
            self.counters[reason+'_episodes']+=1
            row={'engine_seed':self.seeds[i],'env':int(i),'learner_seat':int(self.episode_seat[i]),
                 'reason':reason,'winner':winner if reason=='terminal' else None,
                 'ticks':int(v.infos[i,s.info['tick']]),'opponent_id':int(self.opponent_ids[i]),
                 'fault':v.fault(i).__dict__ if fault else None}
            if self.out:
                with (self.out/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return st
    def reset_done(self):
        done=np.flatnonzero(self.v.dones).tolist()
        for i in done:
            self.seeds[i]=int(self.rng.integers(self.c['train_seed_range'][0],self.c['train_seed_range'][1]+1))
            self.histories[i]=[];self.episode_bad[i]=False;self.episode_seat[i]=1-self.episode_seat[i]
            self.v.reset_env(i,self.seeds[i])
        return done
    def state(self):
        return {'seeds':self.seeds,'histories':self.histories,'episode_bad':self.episode_bad,
                'episode_seat':self.episode_seat,'opponent_ids':self.opponent_ids,
                'rng':self.rng.bit_generator.state,'counters':dict(self.counters),'teacher_counts':self.teacher_counts,
                'observation_sha256':hashlib.sha256(self.v.observations.tobytes()+self.v.masks.tobytes()+self.v.infos.tobytes()).hexdigest()}
    def restore(self,state):
        self.seeds=state['seeds'];self.histories=state['histories'];self.reset_all()
        lengths=[len(h) for h in self.histories];n=max(lengths)
        # Right-align independent histories; a late reset discards each lane's padding steps.
        for t in range(n):
            a=np.zeros((self.v.count,2),np.int32)
            for i,h in enumerate(self.histories):
                offset=n-lengths[i]
                if t==offset or (t<offset and self.v.dones[i]):self.v.reset_env(i,self.seeds[i])
                if t>=offset:a[i]=h[t-offset]
            self.v.step(a)
        for i,length in enumerate(lengths):
            if length==0:self.v.reset_env(i,self.seeds[i])
        got=hashlib.sha256(self.v.observations.tobytes()+self.v.masks.tobytes()+self.v.infos.tobytes()).hexdigest()
        if got!=state['observation_sha256']: raise RuntimeError('checkpoint engine replay digest differs')
        for key in ('episode_bad','episode_seat','opponent_ids'):setattr(self,key,np.array(state[key]))
        self.rng.bit_generator.state=state['rng'];self.counters=Counter(state['counters']);self.teacher_counts=state.get('teacher_counts',{})
