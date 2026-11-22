"""Separate per-player processes; private maps/opponent assets stay inaccessible."""
from pathlib import Path
import ctypes,json,os,resource,select,shutil,signal,struct,subprocess,sys,tempfile,time
from ..contract import ROOT
from .contract import CLASSES
from .registry import verify_entry


class InferenceFault(RuntimeError):pass


def limit(kind):
 c=CLASSES[kind]
 if ctypes.CDLL(None).prctl(38,1,0,0,0)!=0:raise RuntimeError('no-new-privileges failed')
 resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_NPROC,(c['process_limit'],)*2);resource.setrlimit(resource.RLIMIT_NOFILE,(c['file_limit'],)*2)


class Client:
 def __init__(self,entry,uid=65534,development=False):
  self.development=development;self.meta=verify_entry(entry);self.kind=self.meta['inference_class'];self.calls=0;self.uid=uid
  if not development and (sys.platform!='linux' or os.geteuid()!=0):raise RuntimeError('Official isolation requires Linux root')
  self.tmp=tempfile.TemporaryDirectory(prefix='generals-v2-player-');base=Path(self.tmp.name)
  source=base/'entry';shutil.copytree(entry,source);scratch=base/'scratch';scratch.mkdir()
  if not development:
   os.chown(base,0,uid);base.chmod(0o710);os.chown(scratch,uid,uid);scratch.chmod(0o700)
  else:base.chmod(0o700)
  for p in source.rglob('*'):
   if p.is_file():p.chmod(0o444)
   elif p.is_dir():p.chmod(0o555)
  source.chmod(0o555)
  self.err=(base/'stderr.log').open('wb')
  code=f"import sys;sys.path[:0]=[{str(ROOT)!r},{str(ROOT/'vendor/engines/generals-bots')!r}];import faulthandler;faulthandler.enable();"
  if not development:code+=f"from simulators.generals.autoresearch.strong_eval.client import limit;limit({self.kind!r});"
  code+='from simulators.generals.autoresearch.strong_eval.worker import main;main()'
  env={'PATH':os.environ['PATH'],'HOME':str(scratch),'TMPDIR':str(scratch),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','JAX_PLATFORMS':os.environ.get('GENERALS_POLICY_PLATFORM','cuda'),'JAX_DEFAULT_MATMUL_PRECISION':'highest','XLA_PYTHON_CLIENT_PREALLOCATE':'false','XLA_FLAGS':'--xla_cpu_multi_thread_eigen=false'}
  self.proc=subprocess.Popen([sys.executable,'-I','-c',code,'--entry',str(source)],cwd=scratch,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,start_new_session=True,**({} if development else dict(user=uid,group=uid,extra_groups=[])))
  if not development:
   status=Path(f'/proc/{self.proc.pid}/status').read_text();ids=next(v for v in status.splitlines() if v.startswith('Uid:')).split()[1:]
   if any(int(v)!=uid for v in ids):self.close();raise RuntimeError('UID drop not observed')

 def _read(self,n,deadline):
  chunks=[]
  while n:
   ready,_,_=select.select([self.proc.stdout],[],[],max(0,deadline-time.monotonic()))
   if not ready:raise InferenceFault('Neural response deadline')
   data=os.read(self.proc.stdout.fileno(),n)
   if not data:raise InferenceFault('Neural process closed: '+str(self.proc.poll())+';'+Path(self.err.name).read_text(errors='replace')[-4000:])
   chunks.append(data);n-=len(data)
  return b''.join(chunks)

 def choose(self,raw,directions):
  import numpy as np
  from .codec import encode,entrant_mask
  c=CLASSES[self.kind];deadline=time.monotonic()+c['first_response_seconds' if self.calls==0 else 'response_seconds'];data=memoryview(encode(raw,directions));fd=self.proc.stdin.fileno()
  while data:
   _,ready,_=select.select([],[fd],[],max(0,deadline-time.monotonic()))
   if not ready:raise InferenceFault('Neural request deadline')
   sent=os.write(fd,data[:os.fpathconf(fd,'PC_PIPE_BUF')]);data=data[sent:]
  size=struct.unpack('>I',self._read(4,deadline))[0]
  if size>4096:raise InferenceFault('Oversized response')
  ids=json.loads(self._read(size,deadline));mask=np.asarray(entrant_mask(directions))
  if not isinstance(ids,list) or len(ids)!=len(raw) or any(type(i)!=int or not 0<=i<4609 for i in ids):raise InferenceFault('Bad response IDs')
  if not mask[np.arange(len(raw)),ids].all():raise InferenceFault('Illegal neural action')
  if self.calls==0 and sys.platform=='linux' and not self.development:
   status=Path(f'/proc/{self.proc.pid}/status').read_text();limits=Path(f'/proc/{self.proc.pid}/limits').read_text()
   if 'NoNewPrivs:\t1' not in status:raise InferenceFault('NoNewPrivs not observed')
   self.observed_isolation=dict(uid=self.uid,no_new_privileges=True,private_asset_parent_mode=oct(Path(self.tmp.name).stat().st_mode&0o777),private_asset_parent_owner=Path(self.tmp.name).stat().st_uid,private_map_probe_denied='PRIVATE_MAP_READ_DENIED' in Path(self.err.name).read_text(errors='replace'),hidden_fixture_probe_denied='HIDDEN_FIXTURE_READ_DENIED' in Path(self.err.name).read_text(errors='replace'),limits=limits)
  self.calls+=1;return np.asarray(ids,np.int32)

 def close(self):
  if getattr(self,'closed',False):return
  self.closed=True
  if self.proc.poll() is None:
   os.killpg(self.proc.pid,signal.SIGTERM)
   try:self.proc.wait(timeout=2)
   except subprocess.TimeoutExpired:os.killpg(self.proc.pid,signal.SIGKILL);self.proc.wait()
  self.err.close();self.tmp.cleanup()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
