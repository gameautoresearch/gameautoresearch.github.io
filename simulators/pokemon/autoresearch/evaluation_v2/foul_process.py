"""Separate-UID public-information opponent with bounded framed inference and no fallback."""
import ctypes,json,os,resource,select,shutil,signal,struct,subprocess,sys,tempfile,time
from pathlib import Path
from ..arena import PolicyProcess
from ..contract import ROOT


def drop_reference_privileges():
    os.setgroups([]);os.setgid(65533);os.setuid(65533)
    if ctypes.CDLL(None,use_errno=True).prctl(38,1,0,0,0):raise OSError('No-new-privileges failed')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
    resource.setrlimit(resource.RLIMIT_NPROC,(64,64))


class FoulPlayProcess:
    read=PolicyProcess.read
    def __init__(self,source,sets,*,mode='production',iterations=1000,time_ms=10,samples=2,timeout=1.,development=False,deny_file=None):
        if not development and (sys.platform!='linux' or os.geteuid()!=0):raise RuntimeError('Official reference requires root Linux isolation')
        self.tmp=tempfile.TemporaryDirectory(prefix='pk-public-search-');root=Path(self.tmp.name);root.chmod(0o755)
        shutil.copytree(source,root/'source')
        (root/'source').chmod(0o755)
        # Upstream creates this directory at import. Data is injected from our
        # pinned public JSON, so no network fetch or mutable set cache is needed.
        (root/'source/fp/data/pkmn_sets_cache').mkdir(exist_ok=True)
        (root/'source/fp/data/smogon_stats_cache').mkdir(exist_ok=True)
        for p in (root/'source').rglob('*'):p.chmod(0o755 if p.is_dir() else 0o444)
        shutil.copyfile(sets,root/'sets.json');(root/'sets.json').chmod(0o444)
        scratch=root/'scratch';scratch.mkdir(mode=0o700)
        if not development:os.chown(scratch,65533,65533)
        self.error=(root/'stderr.log').open('wb');self.timeout=timeout
        code=f"import sys;sys.path.insert(0,{str(ROOT)!r});from integrations.pokemon.autoresearch.evaluation_v2.foul_process import serve;serve()"
        env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'HOME':str(scratch),'TMPDIR':str(scratch),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
        args=[sys.executable,'-I','-c',code,'--source',str(root/'source'),'--sets',str(root/'sets.json'),'--mode',mode,'--iterations',str(iterations),'--time-ms',str(time_ms),'--samples',str(samples)]
        if deny_file:args+=['--deny-file',str(deny_file)]
        self.process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.error,cwd=scratch,env=env,start_new_session=True,
            preexec_fn=None if development else drop_reference_privileges)
        try:
            self.startup=self.response(time.monotonic()+20)
            if not self.startup.get('ready'):raise RuntimeError('Reference startup failed')
        except Exception as exc:
            self.error.flush()
            detail=(root/'stderr.log').read_text(errors='replace')[-10000:]
            self.close()
            raise RuntimeError('Reference startup failed: '+detail) from exc

    def response(self,deadline):
        size=struct.unpack('>I',self.read(4,deadline))[0]
        if size>1024*1024:raise ValueError('Oversized reference reply')
        return json.loads(self.read(size,deadline))

    def choose(self,visible):
        raw=json.dumps(visible,separators=(',',':')).encode();raw=struct.pack('>I',len(raw))+raw
        if len(raw)>16*1024*1024:raise ValueError('Oversized reference request')
        deadline=time.monotonic()+self.timeout;fd=self.process.stdin.fileno();os.set_blocking(fd,False);view=memoryview(raw)
        while view:
            if not select.select([],[fd],[],max(0,deadline-time.monotonic()))[1]:raise TimeoutError('Reference request timeout')
            view=view[os.write(fd,view):]
        result=self.response(deadline)
        if not result.get('ok'):raise ValueError('Reference adapter fault: '+str(result.get('error')))
        action=result['action']
        if type(action) is not int or not 0<=action<14 or not visible['mask'][action]:raise ValueError('Reference illegal action')
        return action,result['metadata']

    def close(self):
        if self.process.poll() is None:os.killpg(self.process.pid,signal.SIGKILL);self.process.wait(timeout=5)
        self.process.stdin.close();self.process.stdout.close();self.error.close();self.tmp.cleanup()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def serve():
    import argparse
    from .foul_play import FoulPlay
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--sets',required=True);p.add_argument('--mode');p.add_argument('--iterations',type=int);p.add_argument('--time-ms',type=int);p.add_argument('--samples',type=int);p.add_argument('--deny-file');a=p.parse_args()
    isolated=None
    if a.deny_file:
        try:Path(a.deny_file).read_bytes();isolated=False
        except PermissionError:isolated=True
        if not isolated:raise RuntimeError('Private input isolation failed')
    bot=FoulPlay(a.source,a.sets,mode=a.mode,iterations=a.iterations,time_ms=a.time_ms,samples=a.samples)
    def send(value):
        data=json.dumps(value,separators=(',',':')).encode();sys.stdout.buffer.write(struct.pack('>I',len(data))+data);sys.stdout.buffer.flush()
    send({'ready':True,'uid':os.getuid(),'private_input_denied':isolated})
    while True:
        header=sys.stdin.buffer.read(4)
        if not header:return
        size=struct.unpack('>I',header)[0]
        if size>16*1024*1024:raise ValueError('Oversized player input')
        raw=sys.stdin.buffer.read(size)
        try:
            action,metadata=bot.choose(json.loads(raw));send({'ok':True,'action':action,'metadata':metadata})
        except Exception as exc:send({'ok':False,'error':type(exc).__name__+': '+str(exc)})
