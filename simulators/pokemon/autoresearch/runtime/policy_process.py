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
from .sdk import write
ROOT=Path(__file__).resolve().parents[4]

def drop_privileges():
    os.setgroups([]);os.setgid(65534);os.setuid(65534)
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(38,1,0,0,0):raise OSError('Could not set no-new-privileges')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
    resource.setrlimit(resource.RLIMIT_NPROC,(64,64))

class PolicyProcess:
    def __init__(self,bundle,development=False,timeout=15,uid=65534,original=False):
        if original:
            from ..bundle import verify as original_verify
            from ..contract import runtime_signature
            original_verify(bundle)
            if json.loads((Path(bundle)/'binding.json').read_bytes())['runtime_sha256']!=runtime_signature():raise ValueError('Original opponent runtime differs')
        else:verify(bundle)
        if not development and (sys.platform!='linux' or os.geteuid()!=0):
            raise RuntimeError('Official grader requires root Linux and unprivileged inference child')
        self.tmp=tempfile.TemporaryDirectory(prefix='pk-policy-');root=Path(self.tmp.name)
        if not development:os.chown(root,0,uid)
        root.chmod(0o750 if not development else 0o755)
        target=root/'bundle';target.mkdir(mode=0o755)
        if original:
            from ..bundle import FILES as source_files
        else:source_files=FILES
        for name in source_files:shutil.copyfile(Path(bundle)/name,target/name);(target/name).chmod(0o444)
        scratch=root/'scratch';scratch.mkdir(mode=0o700)
        if not development:os.chown(scratch,uid,uid)
        self.error=(root/'stderr.log').open('wb');self.timeout=timeout
        module='integrations.pokemon.autoresearch.policy_worker'if original else'integrations.pokemon.autoresearch.runtime.graph_worker'
        public_root=os.environ.get('POKEMON_PILOT_PUBLIC_PYTHONPATH',str(ROOT))
        code=f"import sys;sys.path.insert(0,{public_root!r});from {module} import main;main()"
        env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'HOME':str(scratch),'TMPDIR':str(scratch),
            'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
        def drop():
            os.setgroups([]);os.setgid(uid);os.setuid(uid)
            libc=ctypes.CDLL(None,use_errno=True)
            if libc.prctl(38,1,0,0,0):raise OSError('Could not set no-new-privileges')
            resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
            resource.setrlimit(resource.RLIMIT_NPROC,(64,64))
        self.process=subprocess.Popen([sys.executable,'-I','-c',code,'--bundle',str(target)],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=self.error,cwd=scratch,env=env,start_new_session=True,
            preexec_fn=None if development else drop)

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
