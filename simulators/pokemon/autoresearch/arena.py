"""Root-owned Showdown grader; private state never crosses the candidate pipe."""
from pathlib import Path
import ctypes
import json
import math
import os
import resource
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import numpy as np
from .bundle import verify, FILES
from .contract import ROOT, write

def drop_privileges():
    os.setgroups([]);os.setgid(65534);os.setuid(65534)
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(38,1,0,0,0):raise OSError('Could not set no-new-privileges')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
    resource.setrlimit(resource.RLIMIT_NPROC,(64,64))

class PolicyProcess:
    def __init__(self,bundle,development=False,timeout=15):
        verify(bundle)
        if not development and (sys.platform!='linux' or os.geteuid()!=0):
            raise RuntimeError('Official grader requires root Linux and unprivileged inference child')
        self.tmp=tempfile.TemporaryDirectory(prefix='pk-policy-');root=Path(self.tmp.name);root.chmod(0o755)
        target=root/'bundle';target.mkdir(mode=0o755)
        for name in FILES:shutil.copyfile(Path(bundle)/name,target/name);(target/name).chmod(0o444)
        scratch=root/'scratch';scratch.mkdir(mode=0o700)
        if not development:os.chown(scratch,65534,65534)
        self.error=(root/'stderr.log').open('wb');self.timeout=timeout
        code=f"import sys;sys.path.insert(0,{str(ROOT)!r});from integrations.pokemon.autoresearch.policy_worker import main;main()"
        env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'HOME':str(scratch),'TMPDIR':str(scratch),
            'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
        self.process=subprocess.Popen([sys.executable,'-I','-c',code,'--bundle',str(target)],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=self.error,cwd=scratch,env=env,start_new_session=True,
            preexec_fn=None if development else drop_privileges)

    def read(self,n,deadline):
        result=b''
        while len(result)<n:
            if not select.select([self.process.stdout],[],[],max(0,deadline-time.monotonic()))[0]:raise TimeoutError('Candidate inference timeout')
            chunk=os.read(self.process.stdout.fileno(),n-len(result))
            if not chunk:raise RuntimeError('Candidate broke inference protocol')
            result+=chunk
        return result

    def choose(self,observations):
        # Explicit schema-only projection: no seed, engine object or opponent choice.
        from integrations.pokemon.network import KEYS
        obs=[{k:np.asarray(o[k]).tolist() for k in KEYS} for o in observations]
        raw=json.dumps(obs,separators=(',',':')).encode();raw=struct.pack('>I',len(raw))+raw
        if len(raw)>32*1024*1024:raise ValueError('Oversized inference batch')
        deadline=time.monotonic()+self.timeout;fd=self.process.stdin.fileno();os.set_blocking(fd,False)
        view=memoryview(raw)
        while view:
            if not select.select([],[fd],[],max(0,deadline-time.monotonic()))[1]:raise TimeoutError('Candidate request timeout')
            written=os.write(fd,view);view=view[written:]
        size=struct.unpack('>I',self.read(4,deadline))[0]
        if size>65536:raise ValueError('Oversized policy reply')
        actions=json.loads(self.read(size,deadline))
        if not isinstance(actions,list) or len(actions)!=len(obs):raise ValueError('Policy reply length')
        if any(type(a) is not int or not 0<=a<14 or not o['mask'][a] for o,a in zip(obs,actions)):
            raise ValueError('Illegal candidate action')
        return actions

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid,signal.SIGKILL);self.process.wait(timeout=5)
        self.process.stdin.close();self.process.stdout.close()
        self.error.close();self.tmp.cleanup()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def summarize(rows,scheduled,private=False):
    natural=[r for r in rows if r.get('natural')]
    complete=len(rows)==scheduled and len(natural)==scheduled and all(r.get('fault') is None for r in rows)
    score=100*float(np.mean([r['points'] for r in natural])) if complete else None
    breakdown={}
    for key in sorted({(r['opponent'],r['seat']) for r in rows}):
        group=[r for r in rows if (r['opponent'],r['seat'])==key]
        breakdown[f'{key[0]}/seat{key[1]}']={'games':len(group),'natural':sum(r['natural'] for r in group),
            'raw_score_0_100':100*float(np.mean([r['points'] for r in group])) if all(r['natural'] for r in group) else None}
    interval=None
    if complete:
        # Resample paired seeds, retaining both seats and every opponent together.
        clusters=sorted({r['seed_index'] for r in rows})
        means=np.array([np.mean([r['points'] for r in rows if r['seed_index']==i]) for i in clusters])
        rng=np.random.default_rng(20261108)
        samples=means[rng.integers(len(means),size=(2000,len(means)))].mean(1)*100
        interval=[float(v) for v in np.quantile(samples,[.025,.975])]
    return {'status':'complete' if complete else 'incomplete','scheduled_games':scheduled,'attempted_games':len(rows),
        'natural_games':len(natural),'truncations':sum(r.get('truncated',False) for r in rows),
        'faults':sum(r.get('fault') is not None for r in rows),'raw_score_0_100':score,
        'normalized_reference_score':None,'calibration_qualified':False,'by_opponent_and_seat':breakdown,
        'seed_cluster_bootstrap_95_percent':interval,'interval_caution':'Descriptive; tiny smoke panels do not establish strength',
        **({'rows':rows} if private else {})}

def evaluate(bundle,panel,seeds,limit=512,development=False):
    from .contract import runtime_signature
    binding=json.loads((Path(bundle)/'binding.json').read_text())
    if binding.get('runtime_sha256')!=runtime_signature():raise ValueError('Grader runtime differs from source-bound checkpoint')
    import torch
    from integrations.pokemon.env import Vector
    from integrations.pokemon.network import Policy,batch,scripted
    torch.set_num_threads(1)
    opponents={}
    for entry in panel:
        if entry['kind']=='checkpoint':
            root=Path(entry['path']);meta=json.loads((root/'opponent.json').read_text())
            from .contract import sha
            if sha(root/'weights.pt')!=meta['weights_sha256']:raise ValueError('Opponent hash changed')
            model=Policy(**meta['network']);model.load_state_dict(torch.load(root/'weights.pt',weights_only=True,map_location='cpu'));model.eval()
            opponents[entry['name']]=model
    jobs=[(p,index,seat) for index in range(len(seeds)) for p in panel for seat in (0,1)]
    rows=[];started=time.monotonic()
    with PolicyProcess(bundle,development=development) as candidate,Vector(1,min(12,len(jobs)),packed=True,binary=True) as vector:
        for start in range(0,len(jobs),vector.size):
            real=jobs[start:start+vector.size];group=real+[real[-1]]*(vector.size-len(real))
            states=vector.reset([seeds[i] for _,i,_ in group],limit)
            rngs=[[np.random.default_rng(np.random.SeedSequence([seeds[i],s,8675309])) for s in (0,1)] for _,i,_ in group]
            while any(not(s['terminated'] or s['truncated']) for s in states):
                actions=[[None,None] for _ in states];candidates=[];groups={}
                for i,(state,(opponent,_,seat)) in enumerate(zip(states,group)):
                    for s,ready in enumerate(state['ready']):
                        if not ready:continue
                        obs=state['observations'][s]
                        if s==seat:candidates.append((i,s,obs))
                        elif opponent['kind']=='checkpoint':groups.setdefault(opponent['name'],[]).append((i,s,obs))
                        else:actions[i][s]=scripted(obs,rngs[i][s],opponent['kind'])
                if candidates:
                    chosen=candidate.choose([r[2] for r in candidates])
                    for (i,s,_),action in zip(candidates,chosen):actions[i][s]=action
                with torch.no_grad():
                    for name,items in groups.items():
                        x=batch([r[2] for r in items],'cpu');logits,_=opponents[name](x);chosen=logits.argmax(-1).tolist()
                        for (i,s,_),action in zip(items,chosen):actions[i][s]=action
                states=vector.step(actions)
            for (opponent,index,seat),state in zip(real,states):
                rows.append({'opponent':opponent['name'],'seed_index':index,'seat':seat,'natural':state['terminated'],
                    'truncated':state['truncated'],'fault':None,'points':(state['returns'][seat]+1)/2 if state['terminated'] else None,
                    'decision_phases':state['steps'],'visible_choice_rejections':sum(state['rejections'])})
    result=summarize(rows,len(jobs),private=True);result['seconds']=time.monotonic()-started
    return result

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    spec=json.loads(Path(a.request).read_text())
    try:result=evaluate(spec['bundle'],spec['panel'],spec['seeds'],spec.get('limit',512))
    except Exception as exc:
        from integrations.pokemon.env import WorkerError
        if isinstance(exc,WorkerError):category=exc.category
        elif isinstance(exc,TimeoutError):category='policy_inference_timeout'
        elif isinstance(exc,ValueError):category='policy_illegal_action' if 'Illegal candidate action' in str(exc) else 'policy_invalid_output'
        elif isinstance(exc,OSError):category='infrastructure_failure'
        else:category='policy_protocol_or_adapter_fault'
        result={'status':'failed','raw_score_0_100':None,'normalized_reference_score':None,
            'failure_category':category,'fault':type(exc).__name__+': '+str(exc)}
        write(a.output,result);raise
    write(a.output,result)

if __name__=='__main__':main()
