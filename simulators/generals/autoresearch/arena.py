"""Trusted native evaluator; submitted policy sees only public tensors over a pipe."""
from pathlib import Path
import ctypes
import json
import os
import resource
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

from .bundle import FILES,verify
from .contract import DEFAULT,PANEL_TENSOR_CAP,ROOT,sha,unprivileged_paths
from .wire import encode

class PolicyFailure(RuntimeError):pass

# How long a policy process may take to load (imports, weights, the host's compute and learned checks) before its
# first request. This is the host's allowance and not part of the per-answer deadline.
STARTUP_SECONDS=180

# Panel opponents with their own source run under another unprivileged uid than a submitted policy, so
# neither can signal, trace or read the other.
PANEL_UID=65533

from .policy_worker import limit as _limit_policy

def _drop_privileges():
    os.setgroups([]);os.setgid(65534);os.setuid(65534)
    _limit_policy()

class PolicyProcess:
    def __init__(self,bundle,development=False,timeout=10,tensor_cap=None,uid=65534):
        # tensor_cap and uid are the evaluator's own choices for a sealed panel opponent (see PANEL_UID);
        # a submitted policy has the contract's parameter cap and uid 65534.
        verify(bundle,cap=tensor_cap)
        if not development and (sys.platform!='linux' or os.geteuid()!=0):
            raise RuntimeError('Official grading requires a root-owned Linux evaluator with an unprivileged policy child')
        self.tmp=tempfile.TemporaryDirectory(prefix='generals-policy-')
        root=Path(self.tmp.name);root.chmod(0o755)
        target=root/'bundle';target.mkdir(mode=0o755)
        for name in FILES:shutil.copyfile(Path(bundle)/name,target/name);(target/name).chmod(0o444)
        scratch=root/'scratch';scratch.mkdir(mode=0o700)
        if not development:
            # Only this process's own uid (and root) can enter its folder: a submitted policy and a panel
            # opponent run side by side and must not read each other's package.
            os.chown(scratch,uid,uid);os.chown(root,uid,uid);root.chmod(0o500)
        self.timeout=timeout
        platform=os.environ.get('GENERALS_POLICY_PLATFORM','cpu')
        if platform not in ('cpu','cuda'):raise ValueError('Unsupported fixed inference platform')
        self.err=(root/'stderr.log').open('wb')
        code=f"import sys;sys.path[:0]={unprivileged_paths()!r};"
        code+='import faulthandler;faulthandler.enable();'
        if not development:code+='from integrations.generals_autoresearch.policy_worker import limit;limit();'
        code+='from integrations.generals_autoresearch.policy_worker import main;main()'
        env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'HOME':str(scratch),'TMPDIR':str(scratch),
             'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'0' if platform=='cuda' else '',
             'JAX_PLATFORMS':platform,'XLA_PYTHON_CLIENT_PREALLOCATE':'false','XLA_FLAGS':'--xla_cpu_multi_thread_eigen=false'}
        profile=os.environ.get('GENERALS_CPU_INFERENCE_PROFILE','default')
        if profile not in ('default','avx2'):raise ValueError('Unsupported host CPU inference profile')
        if platform=='cpu' and profile=='avx2':env['XLA_FLAGS']+=' --xla_cpu_max_isa=AVX2'
        self.process=subprocess.Popen([sys.executable,'-I','-c',code,'--bundle',str(target),
            *(['--tensor-cap',str(tensor_cap)] if tensor_cap else [])],cwd=scratch,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,env=env,start_new_session=True,
            **({} if development else {'user':uid,'group':uid,'extra_groups':[]}))
        self.unprivileged=not development
        if not development:
            status=Path(f'/proc/{self.process.pid}/status').read_text()
            observed=next(v for v in status.splitlines() if v.startswith('Uid:')).split()[1:]
            if any(int(v)!=uid for v in observed):self.close();raise RuntimeError('Policy process did not drop privileges')
        try:
            import struct
            try:size=struct.unpack('>I',self._read(4,time.monotonic()+STARTUP_SECONDS))[0]
            except PolicyFailure as exc:
                if 'deadline' in str(exc):raise PolicyFailure(f'Policy did not finish loading within {STARTUP_SECONDS} seconds') from exc
                raise
            if size>16 or json.loads(self._read(size,time.monotonic()+10))!=[]:raise PolicyFailure('Policy process broke its start-up protocol')
        except BaseException:
            self.close();raise

    def _read(self,n,deadline):
        chunks=[]
        while n:
            ready,_,_=select.select([self.process.stdout],[],[],max(0.,deadline-time.monotonic()))
            if not ready:raise PolicyFailure('Policy inference deadline exceeded')
            value=os.read(self.process.stdout.fileno(),n)
            if not value:
                tail=Path(self.err.name).read_text(errors='replace')[-2000:]
                diagnostics={}
                for name in ('memory.current','memory.max','memory.events','pids.current','pids.max'):
                    try:diagnostics[name]=Path('/sys/fs/cgroup',name).read_text().strip()
                    except OSError:pass
                raise PolicyFailure('Policy process exited or broke its response protocol; returncode='+str(self.process.poll())+
                    '; root_max_rss_kib='+str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)+
                    '; resource_diagnostics='+json.dumps(diagnostics)+'; stderr='+tail)
            chunks.append(value);n-=len(value)
        return b''.join(chunks)

    def choose(self,x,mask):
        import struct
        raw=encode(x,mask);deadline=time.monotonic()+self.timeout
        try:
            fd=self.process.stdin.fileno();view=memoryview(raw)
            chunk=os.fpathconf(fd,'PC_PIPE_BUF')
            while view:
                _,ready,_=select.select([],[fd],[],max(0.,deadline-time.monotonic()))
                if not ready:raise PolicyFailure('Policy request deadline exceeded')
                sent=os.write(fd,view[:chunk]);view=view[sent:]
            size=struct.unpack('>I',self._read(4,deadline))[0]
            if size>65536:raise PolicyFailure('Oversized policy response')
            ids=json.loads(self._read(size,deadline))
            if not isinstance(ids,list) or len(ids)!=len(x) or any(type(i)!=int or not 0<=i<mask.shape[1] for i in ids):raise PolicyFailure('Wrong policy response')
            if not mask[np.arange(len(x)),ids].all():raise PolicyFailure('Illegal policy response')
            return np.asarray(ids,np.int32)
        except (BrokenPipeError,EOFError,ValueError,OSError) as exc:
            # A policy process that ended between two requests (the learned check fails after an answer is out)
            # breaks the pipe on the next one: say why it ended, from its own error output.
            detail=''
            if isinstance(exc,(BrokenPipeError,OSError)):
                try:
                    self.process.wait(timeout=2)
                    detail='; returncode='+str(self.process.poll())+'; stderr='+Path(self.err.name).read_text(errors='replace')[-2000:]
                except Exception:pass
            raise PolicyFailure(type(exc).__name__+': '+str(exc)+detail) from exc

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid,signal.SIGTERM)
            try:self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:os.killpg(self.process.pid,signal.SIGKILL);self.process.wait()
        self.err.close();self.tmp.cleanup()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

class TrustedPanelPolicy:
    def __init__(self,bundle):
        import jax
        from .starter.policy import create
        from .bundle import load_weights
        root=Path(bundle);verify(root)
        if sha(root/'policy.py')!=sha(Path(__file__).parent/'starter/policy.py'):
            raise ValueError('Hidden panel requires the trusted built-in architecture')
        self.model=create(json.loads((root/'model.json').read_text()));self.params=load_weights(root/'weights.npz')
        self.act=jax.jit(lambda x,m:jax.numpy.argmax(self.model.apply({'params':self.params},x,m)[0],-1).astype(jax.numpy.int32))
    def choose(self,x,mask):return self.act(x,mask)

def evaluate_unfused(bundle,panel,*,pool_path,map_indices=None,split='heldout',batch_size=32,development=False):
    import jax
    import jax.numpy as jnp
    from generals.agents import HunterAgent
    from integrations.generals.environment import observations,encode as visible,action_mask,encode_action,decode_action,transition,scripted_logits,RULESET
    from .rollout import load_pool
    if split not in ('practice','heldout','calibration') or not panel:raise ValueError('Invalid grading schedule')
    if len({p['name'] for p in panel})!=len(panel):raise ValueError('Duplicate panel names')
    pool=load_pool(pool_path);indices=list(range(len(pool.time))) if map_indices is None else list(map_indices)
    if not indices or len(indices)!=len(set(indices)) or min(indices)<0 or max(indices)>=len(pool.time):raise ValueError('Invalid map selection')
    trusted={p['name']:TrustedPanelPolicy(p['bundle']) for p in panel if p['kind']=='checkpoint'}
    jobs=[(p,i,seat) for p in panel for seat in (0,1) for i in indices]
    # No process/batch ever receives opposite-seat views of the same private map.
    parts=[[(p,i,seat) for i in indices[c:c+batch_size]] for p in panel for seat in (0,1)
           for c in range(0,len(indices),batch_size)]
    # Practice maps are already supplied in the public workspace. Batch their
    # complete schedule together; private schedules retain the isolation above.
    if split=='practice':parts=[jobs[c:c+batch_size] for c in range(0,len(jobs),batch_size)]
    hunter=HunterAgent();rows=[];started=time.monotonic()
    view=jax.jit(lambda s:(observations(s),visible(observations(s)),action_mask(observations(s))))
    step=jax.jit(transition)
    hunting=jax.jit(jax.vmap(lambda o:hunter.act(o,jax.random.PRNGKey(0))))
    script=jax.jit(lambda o,keys,kind:jax.vmap(lambda k,l:jax.random.categorical(k,l))(keys,scripted_logits(o,kind)).astype(jnp.int32))
    for part in parts:
        with PolicyProcess(bundle,development,DEFAULT['inference_timeout_seconds']) as candidate:
            real=len(part);batch_size=real;padded=part
            seats=jnp.asarray([v[2] for v in padded]);rr=jnp.arange(batch_size)
            states=jax.tree.map(lambda a:a[jnp.asarray([v[1] for v in padded])],pool)
            active=np.arange(batch_size)<real
            groups={p['name']:np.asarray([i for i,v in enumerate(padded) if v[0]['name']==p['name']]) for p in panel}
            keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(83027),i))(jnp.asarray([v[1] for v in padded]))
            for tick in range(RULESET['truncation']):
                obs,x,mask=view(states)
                ids=jnp.asarray(candidate.choose(np.asarray(x[rr,seats]),np.asarray(mask[rr,seats])))
                opp=jnp.zeros(batch_size,jnp.int32)
                for p in panel:
                    idx=groups[p['name']]
                    if len(idx)==0:continue
                    oo=jax.tree.map(lambda a:a[idx,1-seats[idx]],obs)
                    xx,mm=x[idx,1-seats[idx]],mask[idx,1-seats[idx]]
                    if p['kind']=='checkpoint':chosen=trusted[p['name']].choose(xx,mm)
                    elif p['kind']=='hunter':chosen=encode_action(hunting(oo))
                    elif p['kind'] in ('random','expansion'):
                        kk=jax.vmap(lambda k:jax.random.fold_in(k,tick))(keys[idx])
                        chosen=script(oo,kk,int(p['kind']=='expansion'))
                    else:raise ValueError('Unknown trusted opponent')
                    if not np.asarray(jnp.take_along_axis(mm,chosen[:,None],-1)).all():raise RuntimeError('Trusted panel emitted an illegal action')
                    opp=opp.at[idx].set(chosen)
                joint=jnp.zeros((batch_size,2),jnp.int32).at[rr,seats].set(ids).at[rr,1-seats].set(opp)
                result=step(states,decode_action(joint));done=np.asarray(result.terminated|result.truncated)
                reward=np.asarray(result.reward[rr,seats]);natural=np.asarray(result.terminated);cap=np.asarray(result.truncated)
                for i in np.flatnonzero(active&done):
                    p,index,seat=part[i]
                    rows.append(dict(opponent=p['name'],map_index=index,seat=seat,score=float((reward[i]+1)/2),
                        natural=bool(natural[i]),truncated=bool(cap[i]),capture_win=bool(reward[i]>0),
                        capture_loss=bool(reward[i]<0),length=tick+1))
                live=jnp.asarray(active)
                states=jax.tree.map(lambda n,o:jnp.where(live.reshape((batch_size,)+(1,)*(n.ndim-1)),n,o),result.final_state,states)
                active &= ~done
                if not active.any():break
            if active.any():raise RuntimeError('Unfinished games; no partial grade awarded')
    if len(rows)!=len(jobs):raise RuntimeError('Incomplete evaluation')
    return dict(rows=rows,seconds=time.monotonic()-started,games=len(rows),illegal_actions=0,simulator_faults=0,
        inference_platform=os.environ.get('GENERALS_POLICY_PLATFORM','cpu'),
        development_only=development,unprivileged_policy_observed=not development,
        information='Submitted process receives x and mask only; private maps and opponents remain root-only')



def batch_steps(panel,trusted,hunter,part):
    """Fuse trusted CPU preparation/advance; the submitted forward stays isolated."""
    import jax
    import jax.numpy as jnp
    from integrations.generals.environment import observations,encode as visible,action_mask,encode_action,decode_action,transition,scripted_logits
    batch=len(part);rr=jnp.arange(batch);seats=jnp.asarray([v[2] for v in part])
    groups=[(p,np.asarray([i for i,v in enumerate(part) if v[0]['name']==p['name']])) for p in panel]
    # A `policy` opponent chooses outside this step, in its own process: its rows of `opp` stay 0 here and
    # the opponent-side views are returned for it (see `_play`).
    external=any(p['kind']=='policy' and len(idx) for p,idx in groups)
    @jax.jit
    def prepare(states,keys,tick):
        obs=observations(states);x=visible(obs);mask=action_mask(obs)
        opp=jnp.zeros(batch,jnp.int32);legal=jnp.ones(batch,bool)
        for p,idx in groups:
            if not len(idx) or p['kind']=='policy':continue
            oo=jax.tree.map(lambda a:a[idx,1-seats[idx]],obs)
            xx,mm=x[idx,1-seats[idx]],mask[idx,1-seats[idx]]
            if p['kind']=='checkpoint':
                bot=trusted[p['name']]
                chosen=jnp.argmax(bot.model.apply({'params':bot.params},xx,mm)[0],-1).astype(jnp.int32)
            elif p['kind']=='hunter':chosen=encode_action(jax.vmap(lambda o:hunter.act(o,jax.random.PRNGKey(0)))(oo))
            elif p['kind'] in ('random','expansion'):
                kk=jax.vmap(lambda k:jax.random.fold_in(k,tick))(keys[idx])
                logits=scripted_logits(oo,int(p['kind']=='expansion'))
                chosen=jax.vmap(jax.random.categorical)(kk,logits).astype(jnp.int32)
            else:raise ValueError('Unknown trusted opponent')
            legal=legal.at[idx].set(jnp.take_along_axis(mm,chosen[:,None],-1)[:,0])
            opp=opp.at[idx].set(chosen)
        return x[rr,seats],mask[rr,seats],opp,legal,((x[rr,1-seats],mask[rr,1-seats]) if external else None)
    @jax.jit
    def advance(states,ids,opp,active):
        joint=jnp.zeros((batch,2),jnp.int32).at[rr,seats].set(ids).at[rr,1-seats].set(opp)
        result=transition(states,decode_action(joint))
        current=jax.tree.map(lambda n,o:jnp.where(active.reshape((batch,)+(1,)*(n.ndim-1)),n,o),result.final_state,states)
        return current,result.terminated|result.truncated,result.reward[rr,seats],result.terminated,result.truncated
    return prepare,advance

def evaluate(bundle,panel,*,pool_path,map_indices=None,split='heldout',batch_size=32,development=False):
    import jax
    import jax.numpy as jnp
    from generals.agents import HunterAgent
    from integrations.generals.environment import RULESET
    from .rollout import load_pool
    if split not in ('practice','heldout','calibration') or not panel:raise ValueError('Invalid grading schedule')
    if len({p['name'] for p in panel})!=len(panel):raise ValueError('Duplicate panel names')
    pool=load_pool(pool_path);indices=list(range(len(pool.time))) if map_indices is None else list(map_indices)
    if not indices or len(indices)!=len(set(indices)) or min(indices)<0 or max(indices)>=len(pool.time):raise ValueError('Invalid map selection')
    trusted={p['name']:TrustedPanelPolicy(p['bundle']) for p in panel if p['kind']=='checkpoint'}
    jobs=[(p,i,seat) for p in panel for seat in (0,1) for i in indices]
    parts=[[(p,i,seat) for i in indices[c:c+batch_size]] for p in panel for seat in (0,1) for c in range(0,len(indices),batch_size)]
    if split=='practice':parts=[jobs[c:c+batch_size] for c in range(0,len(jobs),batch_size)]
    rows=[];hunter=HunterAgent();cached={};started=time.monotonic()
    # A private schedule starts a fresh policy process for every part, so no process ever sees both seats of
    # a private map. Public practice keeps one process for its whole schedule: its maps are public, and one
    # start and one compile instead of one per part keeps a 64-map practice short.
    shared=None;kept={}
    try:
        if split=='practice':
            shared=PolicyProcess(bundle,development,DEFAULT['inference_timeout_seconds'])
            kept=_panel_processes(panel,None,development)
        rows=_play(parts,panel,trusted,hunter,pool,bundle,development,shared,cached,kept)
    finally:
        for process in (shared,*kept.values()):
            if process is not None:process.close()
    if len(rows)!=len(jobs):raise RuntimeError('Incomplete evaluation')
    return dict(rows=rows,seconds=time.monotonic()-started,games=len(rows),illegal_actions=0,simulator_faults=0,
        inference_platform=os.environ.get('GENERALS_POLICY_PLATFORM','cpu'),development_only=development,unprivileged_policy_observed=not development,
        information='Submitted process receives x and mask only; private maps and opponents remain root-only')

class _Kept:
    """A policy process that outlives one part: entering and leaving it does not end it."""
    def __init__(self,process):self.process=process
    def __enter__(self):return self.process
    def __exit__(self,*args):pass

def _panel_processes(panel,part,development):
    """One isolated process per `policy` opponent (of this part, or of the whole panel when part is None)."""
    names=None if part is None else {p['name'] for p,_,_ in part};started={}
    try:
        for p in panel:
            if p['kind']=='policy' and (names is None or p['name'] in names):
                started[p['name']]=PolicyProcess(p['bundle'],development,DEFAULT['inference_timeout_seconds'],PANEL_TENSOR_CAP,PANEL_UID)
    except BaseException:
        for process in started.values():process.close()
        raise
    return started

def _panel_choice(process,name,x,mask):
    # A panel opponent that fails is the evaluator's fault, never the candidate's: grading aborts with no score.
    try:return process.choose(x,mask)
    except PolicyFailure as exc:raise RuntimeError('Panel policy '+name+' failed: '+str(exc)[-1500:]) from exc

def _play(parts,panel,trusted,hunter,pool,bundle,development,shared,cached,kept=None):
    from concurrent.futures import ThreadPoolExecutor
    rows=[];executor=ThreadPoolExecutor(max_workers=4)
    try:
        for number,part in enumerate(parts):
            signature=tuple((p['name'],seat) for p,_,seat in part)
            if signature not in cached:cached[signature]=batch_steps(panel,trusted,hunter,part)
            progress={'part':number+1,'parts':len(parts),'opponent':part[0][0]['name'],'games_completed':len(rows)}
            rows+=_play_part(part,panel,pool,bundle,development,shared,cached[signature],kept,executor,progress)
    finally:executor.shutdown(wait=False,cancel_futures=True)
    return rows

def _play_part(part,panel,pool,bundle,development,shared,steps,kept,executor,progress):
    import jax
    import jax.numpy as jnp
    from contextlib import ExitStack
    from integrations.generals.environment import RULESET
    prepare,advance=steps;rows=[]
    states=jax.tree.map(lambda a:a[jnp.asarray([v[1] for v in part])],pool)
    keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(83027),i))(jnp.asarray([v[1] for v in part]))
    active=np.ones(len(part),bool)
    groups={p['name']:np.asarray([i for i,v in enumerate(part) if v[0]['name']==p['name']]) for p in panel if p['kind']=='policy'}
    with ExitStack() as stack:
        candidate=stack.enter_context(_Kept(shared) if shared is not None else PolicyProcess(bundle,development,DEFAULT['inference_timeout_seconds']))
        # A private schedule starts fresh panel processes for every part, as it does for the candidate;
        # practice keeps one per opponent for its whole schedule.
        if shared is not None:rivals={n:kept[n] for n,idx in groups.items() if len(idx)}
        else:rivals={n:stack.enter_context(v) for n,v in _panel_processes(panel,part,development).items()}
        for tick in range(RULESET['truncation']):
            if tick%128==0 and not development:
                from .contract import write_json
                write_json('/private/evaluation-progress.json',dict(progress,tick=tick,games_completed=progress['games_completed']+len(rows),
                    root_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            x,mask,opp,legal,views=prepare(states,keys,jnp.int32(tick))
            if not np.asarray(legal).all():raise RuntimeError('Trusted panel emitted an illegal action')
            asked={}
            if rivals:
                ox,omask=map(np.asarray,views);opp=np.array(opp)
                for n,process in rivals.items():
                    # Each process compiles on its first forward, one at a time; after that the panel's
                    # processes answer while the candidate is asked.
                    if tick==0:opp[groups[n]]=_panel_choice(process,n,ox[groups[n]],omask[groups[n]])
                    else:asked[n]=executor.submit(_panel_choice,process,n,ox[groups[n]],omask[groups[n]])
            try:ids=candidate.choose(np.asarray(x),np.asarray(mask))
            finally:
                # A candidate fault still waits for the panel's answers: no process is closed mid-request.
                for future in asked.values():future.exception()
            for n,future in asked.items():opp[groups[n]]=future.result()
            states,done,reward,natural,cap=advance(states,ids,opp,active)
            done,reward,natural,cap=map(np.asarray,(done,reward,natural,cap))
            for i in np.flatnonzero(active&done):
                p,index,seat=part[i]
                rows.append(dict(opponent=p['name'],map_index=index,seat=seat,score=float((reward[i]+1)/2),natural=bool(natural[i]),truncated=bool(cap[i]),capture_win=bool(reward[i]>0),capture_loss=bool(reward[i]<0),length=tick+1))
            active &= ~done
            if not active.any():break
        if active.any():raise RuntimeError('Unfinished games; no partial grade awarded')
    return rows
