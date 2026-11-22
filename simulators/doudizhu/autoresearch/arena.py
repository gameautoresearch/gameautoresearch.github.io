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
from .contract import DEFAULT,ROOT,sha
from .wire import encode

class PolicyFailure(RuntimeError):pass

def _drop_privileges():
    os.setgroups([]);os.setgid(65534);os.setuid(65534)
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(38,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_NO_NEW_PRIVS failed')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
    resource.setrlimit(resource.RLIMIT_NPROC,(64,64))

class PolicyProcess:
    def __init__(self,bundle,development=False,timeout=10):
        verify(bundle)
        if not development and (sys.platform!='linux' or os.geteuid()!=0):
            raise RuntimeError('Official grading requires a root-owned Linux evaluator with an unprivileged policy child')
        self.tmp=tempfile.TemporaryDirectory(prefix='dd-policy-')
        root=Path(self.tmp.name);root.chmod(0o755)
        target=root/'bundle';target.mkdir(mode=0o755)
        for name in FILES:shutil.copyfile(Path(bundle)/name,target/name);(target/name).chmod(0o444)
        scratch=root/'scratch';scratch.mkdir(mode=0o700)
        if not development:os.chown(scratch,65534,65534)
        self.timeout=timeout
        self.err=(root/'stderr.log').open('wb')
        code=f"import sys;sys.path.insert(0,{str(ROOT)!r});from integrations.doudizhu.autoresearch.policy_worker import main;main()"
        env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'HOME':str(scratch),'TMPDIR':str(scratch),
             'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
        self.process=subprocess.Popen([sys.executable,'-I','-c',code,'--bundle',str(target)],cwd=scratch,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,env=env,start_new_session=True,
            preexec_fn=None if development else _drop_privileges)

    def _read(self,n,deadline):
        chunks=[]
        while n:
            ready,_,_=select.select([self.process.stdout],[],[],max(0.,deadline-time.monotonic()))
            if not ready:raise PolicyFailure('Policy inference deadline exceeded')
            value=os.read(self.process.stdout.fileno(),n)
            if not value:raise PolicyFailure('Policy process exited or broke its response protocol')
            chunks.append(value);n-=len(value)
        return b''.join(chunks)

    def choose(self,observations):
        import struct
        role=observations[0].role
        values=[o.encode(backend='direct') for o in observations]
        counts=np.asarray([len(o.legal) for o in observations],dtype=np.int32)
        raw=encode(role,np.stack([v[0] for v in values]),np.stack([v[1] for v in values]),np.concatenate([v[2] for v in values]),counts)
        deadline=time.monotonic()+self.timeout
        try:
            # Payloads are bounded public batches. Nonblocking writes enforce the
            # same deadline if a malicious policy stops reading stdin.
            fd=self.process.stdin.fileno();os.set_blocking(fd,False)
            view=memoryview(raw)
            while view:
                _,ready,_=select.select([],[fd],[],max(0.,deadline-time.monotonic()))
                if not ready:raise PolicyFailure('Policy request write deadline exceeded')
                sent=os.write(fd,view);view=view[sent:]
            size=struct.unpack('>I',self._read(4,deadline))[0]
            if size>65536:raise PolicyFailure('Oversized policy response')
            indices=json.loads(self._read(size,deadline))
            if not isinstance(indices,list) or len(indices)!=len(counts):raise PolicyFailure('Wrong policy response length')
            if any(type(i)!=int or not 0<=i<int(n) for i,n in zip(indices,counts)):raise PolicyFailure('Illegal policy action')
            return indices
        except (BrokenPipeError,EOFError,ValueError,OSError) as exc:raise PolicyFailure(type(exc).__name__+': '+str(exc)) from exc

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
        import torch
        from integrations.doudizhu.model import Policy
        root=Path(bundle);verify(root)
        # Hidden opponents are ours. Never import arbitrary sidecar code in the
        # privileged evaluator process, even from a purported panel package.
        if sha(root/'policy.py')!=sha(Path(__file__).parent/'starter/policy.py'):
            raise ValueError('Panel source is not the trusted built-in policy adapter')
        self.policy=Policy(json.loads((root/'model.json').read_text()))
        self.policy.load_state_dict(torch.load(root/'weights.pt',map_location='cpu',weights_only=True),strict=True)
        self.policy.eval();torch.set_num_threads(1)
    def choose(self,observations):
        q=self.policy.values_many(observations,16384)
        return [int(np.argmax(v)) for v in q]

def evaluate(bundle,panel,*,deal_seed,policy_seed,deals,split='heldout',batch_size=32,development=False):
    from integrations.doudizhu.environment import ROLES,deal,digest,heuristic,seed_value
    from integrations.doudizhu.workers import VectorGames
    if split not in ('practice','heldout') or deals<=0:raise ValueError('Invalid evaluation schedule')
    if not panel or len({x['name'] for x in panel})!=len(panel):raise ValueError('Panel must be nonempty with unique names')
    trusted={p['name']:TrustedPanelPolicy(p['bundle']) for p in panel if p['kind']=='checkpoint'}
    jobs=[(p,i,r) for p in panel for i in range(deals) for r in ROLES]
    rows=[];started=time.monotonic()
    with PolicyProcess(bundle,development,DEFAULT['inference_timeout_seconds']) as candidate,VectorGames(4,engine_backend='native') as vector:
        for cursor in range(0,len(jobs),batch_size):
            part=jobs[cursor:cursor+batch_size]
            cards=[deal(split,deal_seed,i) for _,i,_ in part]
            active={i:v[0] for i,v in vector.reset(cards).items()}
            rngs={i:np.random.default_rng(seed_value(split,policy_seed,index)%2**64 ^ int(digest([p['name'],role])[:16],16)) for i,(p,index,role) in enumerate(part)}
            while active:
                groups={};selected={}
                for i,obs in active.items():
                    p,_,role=part[i]
                    if obs.role==role:groups.setdefault(('candidate',obs.role),[]).append(i)
                    elif p['kind']=='random':selected[i]=int(rngs[i].integers(len(obs.legal)))
                    elif p['kind']=='heuristic':selected[i]=heuristic(obs)
                    elif p['kind']=='checkpoint':groups.setdefault((p['name'],obs.role),[]).append(i)
                    else:raise ValueError('Unknown panel policy kind')
                for (name,_),ids in groups.items():
                    chosen=(candidate if name=='candidate' else trusted[name]).choose([active[i] for i in ids])
                    selected.update(zip(ids,chosen))
                result=vector.step(selected);active={}
                for i,(obs,outcome,_) in result.items():
                    if outcome is None:active[i]=obs;continue
                    p,index,role=part[i]
                    rows.append({'opponent':p['name'],'deal_index':index,'role':role,
                        'won':outcome['landlord_won']==(role=='landlord'),'economic_score':outcome['scores'][role],
                        'decisions':outcome['decisions'],'natural':True,'truncated':False})
    if len(rows)!=len(jobs):raise RuntimeError('Incomplete evaluation; no partial grade awarded')
    return {'rows':rows,'seconds':time.monotonic()-started,'games':len(rows),
        'faults':0,'illegal_actions':0,'truncations':0,'development_only':development}
