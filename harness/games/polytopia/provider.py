"""Polytopia Modal workers: training runs, leases, the CPU parameter check, practice and private grading.

One image serves all of them. A training run is the fixed L40S worker (`train`); a lease is a worker of a shape the
researcher chooses that runs the researcher's own commands as an unprivileged user, with the session's scratch Volume
at /scratch (`hold`). Both hold their grant on the session ledger and settle on the provider's own start/finish record.
Practice and grading run the trusted evaluator on separate CPU workers outside the budget (`evaluate`).
"""
import json,time,math,threading,shutil,os,io,tarfile,hashlib,gzip,zlib,shlex,uuid
from pathlib import Path
from harness.policy import ENVIRONMENT
from harness import hardware,workers,handoff
from harness.workspace import TRAINING
from simulators.polytopia.autoresearch.contract import DEFAULT,TRAIN_SHAPE
from simulators.polytopia.autoresearch.v1.core import HERE,BENCHMARK,write,verify,seal,digest,EDITABLE
from .transport import configure_sandbox_waits,allocation_info,read_rpc,wait_process,reset_rpc_channels

STOPPED_QUEUED='Stopped while queued for a GPU: the provider never started this worker; it was terminated and charged 0 seconds'
LATE_START='The worker started only after its GPU queue wait was abandoned; it was terminated and charged from its provider start/finish receipt'
STOPPED='Stopped by request'
NO_EXPORT='the trainer had not written a complete policy export, so there is no network to keep'
# The file the host creates in a training worker to ask its trainer to end; the trainer is told the path in
# $POLYTOPIA_STOP_REQUEST and its loop ends at the next step (starter rl/gpu_resident.py).
STOP_REQUEST='/workspace/stop-request'
# How a grant is divided, counted back from its end (measured on L40S workers, 2026-11-13: the training loop begins
# 16-32 s after the worker starts and the trainer exits about 3 s after its loop; an archive stream moves 0.4-0.7 MB
# a second, and a one-hour run's two archives are about 5 MB each):
#   TEARDOWN_SECONDS        the worker's own timeout and the host's timer end it this long before the grant does;
#   COLLECT_SECONDS         before that, the window kept for handing the run's files back: the in-worker export and
#                           publish to the hand-off Volume (seconds), and room for the live read it falls back to
#                           when the publish fails. The trainer command's time limit is where it begins, so a
#                           trainer cut off there still has all of it;
#   TRAINER_MARGIN_SECONDS  the trainer is told to stop its loop this long before the worker's end, counted from the
#                           moment its command is started: 50 s for starting up and exiting, then the collection
#                           window. A trainer that ends by itself therefore leaves more than the window.
# Every stream from a worker ends silently at its command's own time limit (the pinned SDK returns what had arrived
# and reports exit code -1), so nothing below reads a stream it does not need, and each read has its own limit.
TEARDOWN_SECONDS,COLLECT_SECONDS,TRAINER_MARGIN_SECONDS=15,65,115
COLLECT_SECONDS_MAX=240   # the longest one archive stream is waited for, when the grant has that much left
LOGS_SECONDS=20           # the longest the trainer's log tails are waited for after a failure
# The trainer's output goes to files in the worker, never through its command's stream: only their tails are fetched,
# inside the archive. (Reading the whole stream after a long run took the collection's time: see `collect`.)
CPU_REQUEST,MEMORY_REQUEST_MIB=1.0,2048   # what a CPU-only worker reserves (its cap stays the 4-core / 16 GiB shape)
TRAINER_STDOUT,TRAINER_STDERR='/output/trainer.stdout','/output/trainer.stderr'
STDOUT_TAIL_BYTES,STDERR_TAIL_BYTES=262144,65536
ALWAYS=('initial.npz','initial.json','protocol.json','training.jsonl')
NETWORK=('final.npz','final.json','final-critic.npz','final-critic.json')
OPTIONAL=('training-complete.json','export.json')
LOG_TAILS=('stdout.tail','stderr.tail')
NO_EXPORT_MARK='no complete policy export'
# Runs in the training worker and writes one gzip-compressed tar of a part of a run to its stdout.
#   network  what a kept run needs: the initial policy, protocol, training log, the network, and the tails of the
#            trainer's output. The network is the trainer's own final one when it reached its end
#            (training-complete.json); otherwise, when asked for (`early`), the newest whole periodic export
#            (exports/update-N, renamed into place by the trainer only once complete). A checkpoint counts only when
#            it matches the digest in its own record. `extras.json` lists what the second part would hold.
#   extras   the clock-* snapshots: collected afterwards if there is time, and never needed for the run to count.
#   logs     only the tails of the trainer's output (after a failure).
# Every file is read once, so a file still being appended to cannot fail its checksum. Reasons go to stderr.
EXPORT_SCRIPT='''import gzip,hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1]);early=sys.argv[2]=='early';part=sys.argv[3];logs=(('stdout.tail',sys.argv[4],int(sys.argv[6])),('stderr.tail',sys.argv[5],int(sys.argv[7])))
NETWORK=('final.npz','final.json','final-critic.npz','final-critic.json')
def whole(folder):
 try:
  data={name:(folder/name).read_bytes() for name in NETWORK}
  for stem in ('final','final-critic'):
   if json.loads(data[stem+'.json'])['sha256']!=hashlib.sha256(data[stem+'.npz']).hexdigest():return None
  return data
 except (OSError,ValueError,KeyError,TypeError):return None
def tails():
 found={}
 for name,path,limit in logs:
  try:
   with open(path,'rb') as stream:
    stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-limit));found[name]=stream.read()
  except OSError:pass
 return found
clocks=sorted(p for p in root.glob('clock-*') if p.is_file())
if part=='logs':files={}
elif part=='extras':files={p.name:p.read_bytes() for p in clocks}
else:
 try:files={name:(root/name).read_bytes() for name in ('initial.npz','initial.json','protocol.json','training.jsonl')}
 except OSError as exc:sys.exit('the run folder lacks '+str(getattr(exc,'filename',exc)))
 network=None
 if (root/'training-complete.json').is_file():
  network=whole(root)
  if network is not None:network['training-complete.json']=(root/'training-complete.json').read_bytes()
  elif not early:sys.exit('training-complete.json is there, but final.npz / final-critic.npz and their records are missing or do not match their digests')
 if network is None and early:
  for folder in sorted((p for p in (root/'exports').glob('update-*') if p.is_dir()),key=lambda p:(len(p.name),p.name),reverse=True):
   network=whole(folder)
   try:
    if network is not None:network['export.json']=json.dumps(json.loads((folder/'export.json').read_text())).encode();break
   except (OSError,ValueError):network=None
 if network is None:sys.exit('no complete policy export'+('' if early else ': the trainer ended without training-complete.json'))
 files.update(network)
files['export-checksums.json']=json.dumps({name:hashlib.sha256(data).hexdigest() for name,data in files.items()}).encode()
if part!='extras':
 files.update(tails())
 files['extras.json']=json.dumps({p.name:p.stat().st_size for p in clocks}).encode()
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w') as archive:
 for name,data in files.items():
  info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
sys.stdout.buffer.write(gzip.compress(buffer.getvalue(),3));sys.stdout.buffer.flush()
'''
# A training run's result travels to the host through the session's private hand-off Volume (harness.handoff), never
# through the host's link inside the worker's paid time: the trusted step below writes the parts of the run that
# EXPORT_SCRIPT makes into files in the worker and publishes them, as root, before the worker ends. The host then
# ends the worker and reads the Volume with retries and no deadline. Reading the archives live out of the worker
# (`Provider._collect`) is only the fallback for a publish that failed while the worker is still alive.
HANDOFF_STAGING='/output/.handoff'   # in the worker, beside (never inside) the run folder EXPORT_SCRIPT reads
HANDOFF_FILES=('network.tgz','extras.tgz','logs.tgz','status.json')
PUBLISH_SECONDS_MAX=60     # the longest the host waits for the in-worker publish (in-datacenter: seconds)
VISIBLE_ATTEMPTS,VISIBLE_WAIT=3,2.0     # the manifest is looked for this often while the worker is still alive ...
COLLECT_ATTEMPTS,COLLECT_WAIT=12,10.0   # ... and the files read this often once it has ended (no paid time)
# Runs in the training worker as root: python -c HANDOFF_SCRIPT RUN MODE KIND STDOUT STDERR OUT_LIMIT ERR_LIMIT STAGING PUBLISH...
# KIND `network` exports the run (MODE as EXPORT_SCRIPT takes it), then the clock snapshots (`extras`), or the log tails
# (`logs`) when the run could not be exported; KIND `logs` exports only the tails (a failed trainer). Each part is
# EXPORT_SCRIPT's own archive, written to STAGING/<part>.tgz, and status.json records how each export ended. Then
# PUBLISH (harness.handoff) copies the files into the Volume and commits it. A job is published once: a second call
# (the trainer's own wrapper and the host may both ask) prints the first one's record. Prints that record as JSON.
HANDOFF_SCRIPT=r'''import fcntl,json,os,subprocess,sys
export=%r
run,mode,kind,out_log,err_log,out_limit,err_limit,staging=sys.argv[1:9];publish=sys.argv[9:]
os.makedirs(staging,exist_ok=True);os.chmod(staging,0o700)
with open(os.path.join(staging,'.lock'),'w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);done=os.path.join(staging,'done.json')
 if os.path.isfile(done):
  with open(done) as stream:record=json.load(stream)
  record['reused']=True;print(json.dumps(record));sys.exit(0)
 status={'kind':kind,'mode':mode,'parts':{}}
 try:
  with open(os.path.join(staging,'prepare-error.txt')) as stream:status['prepare_error']=stream.read()[-2000:]
 except OSError:pass
 def part(name):
  path=os.path.join(staging,name+'.tgz')
  with open(path,'wb') as stream:p=subprocess.run([sys.executable,'-c',export,run,mode,name,out_log,err_log,out_limit,err_limit],stdout=stream,stderr=subprocess.PIPE)
  status['parts'][name]={'returncode':p.returncode,'stderr':p.stderr.decode(errors='replace')[-2000:],'bytes':os.path.getsize(path)}
  return p.returncode
 if kind=='logs':part('logs')
 elif part('network')==0:part('extras')
 else:part('logs')
 for name in ('network','extras','logs'):open(os.path.join(staging,name+'.tgz'),'ab').close()
 with open(os.path.join(staging,'status.json'),'w') as stream:json.dump(status,stream)
 manifest=json.loads(subprocess.run(publish,check=True,stdout=subprocess.PIPE).stdout.decode().strip().splitlines()[-1])
 record={'status':status,'manifest':manifest}
 with open(done+'.partial','w') as stream:json.dump(record,stream)
 os.replace(done+'.partial',done);print(json.dumps(record))
''' % EXPORT_SCRIPT
# Runs the trainer in its worker as root, in place of the shell that only redirected its output: first the hand-off
# Volume is made root-only (handoff.prepare_command; a failure is recorded, never fatal: the publish then fails and the
# host falls back to a live read), then the trainer, its output to two files. A trainer that ends by itself with
# success is published right there (HANDOFF_SCRIPT, `network`, `final`), so a finished run reaches the Volume even
# when the host is unreachable at that moment. Exits with the trainer's own code.
TRAIN_WRAPPER=r'''import json,os,subprocess,sys
plan=json.loads(sys.argv[1]);command=sys.argv[2:]
try:subprocess.run(plan['prepare'],check=True,timeout=60,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
except Exception as exc:
 os.makedirs(plan['staging'],exist_ok=True)
 with open(os.path.join(plan['staging'],'prepare-error.txt'),'w') as stream:stream.write(repr(exc)[-2000:])
os.makedirs(os.path.dirname(plan['stdout']),exist_ok=True)
with open(plan['stdout'],'wb') as out,open(plan['stderr'],'wb') as err:code=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err).returncode
if code==0:
 try:subprocess.run(plan['handoff'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=plan['handoff_seconds'])
 except Exception:pass
sys.exit(code if code>=0 else 128-code)
'''

def schedule(started_at,grant,now,smoke=False):
 """When a training worker's phases end: its own end (`deadline`), the trainer command's time limit (`limit`), and
 the training time the trainer is given now (`seconds`). See the constants above."""
 deadline=started_at+grant-TEARDOWN_SECONDS
 return {'deadline':deadline,'limit':deadline-COLLECT_SECONDS,'seconds':min(180 if smoke else grant,max(10,deadline-now-TRAINER_MARGIN_SECONDS))}

def collect_seconds(deadline,now,most=COLLECT_SECONDS_MAX):
 """The time limit of one archive stream started now: what the worker has left, less 5 s, and at most `most`."""
 return max(1,min(most,int(deadline-now-5)))

# The most one file of a result archive may hold. A training run's training.jsonl grows by about 7 MB an hour
# (99 MB at 14 hours of the starter); the 50 MiB this used to be failed every starter final of 8 hours or more
# after it had trained in full (poly-frontier-20261116).
MEMBER_BYTES_MAX=1024**3

def unpack(payload):
 """The files of one archive stream, {name: bytes}. A stream that was cut, or whose files do not match the checksums
 made in the worker, raises ValueError saying which."""
 if len(payload)>200*1024**2:raise ValueError('the archive exceeds the 200 MiB cap')
 try:raw=gzip.decompress(payload)
 except (OSError,EOFError,zlib.error) as exc:raise ValueError('the archive stream is incomplete or damaged (%d bytes arrived): %s'%(len(payload),exc)) from None
 files={}
 with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
  for member in archive.getmembers():
   if not member.isfile() or '/' in member.name:raise ValueError('invalid archive member '+member.name[:80])
   if member.size>MEMBER_BYTES_MAX:raise ValueError('archive member %s is %d bytes, over the %d MiB cap'%(member.name[:80],member.size,MEMBER_BYTES_MAX//1024**2))
   files[member.name]=archive.extractfile(member).read()
 if 'export-checksums.json' not in files:raise ValueError('the archive has no checksums')
 for name,expected in json.loads(files['export-checksums.json']).items():
  if name not in files or hashlib.sha256(files[name]).hexdigest()!=expected:raise ValueError('checksum mismatch for '+name[:80])
 return files

def why(record):
 """One sentence on why a collection failed, from its record."""
 size='%d bytes had arrived after %.0f s'%(record['bytes'],record['seconds'])
 if record.get('reached_time_limit'):return 'the collection was cut off at its %d-second time limit (%s)'%(record['timeout_seconds'],size)
 if record.get('returncode'):return 'the collection command ended with code %s: %s'%(record['returncode'],(record.get('stderr') or 'it wrote nothing to stderr').strip()[-600:])
 return record.get('unpack_error') or record.get('error') or 'no reason was recorded (%s)'%size

def job_name(name,sandbox_id):
 """A run's hand-off job: one per worker, so nothing published by an earlier worker can stand in for this one's."""
 return handoff.check_name(name+'.'+sandbox_id)

def handoff_command(job,name,kind,mode,python='python'):
 """The worker command that publishes run NAME's KIND (HANDOFF_SCRIPT) as hand-off JOB."""
 staging=HANDOFF_STAGING+'/'+job
 return [python,'-c',HANDOFF_SCRIPT,'/output/'+name,mode,kind,TRAINER_STDOUT,TRAINER_STDERR,str(STDOUT_TAIL_BYTES),str(STDERR_TAIL_BYTES),staging,*handoff.publish_command(job,{f:staging+'/'+f for f in HANDOFF_FILES},python=python)]

class Received:
 """What a training worker handed back, one part at a time: `part(name)` is ({file: bytes} or None, a record of it).
 The parts come from the hand-off Volume (`route['handoff']=='volume'`), or, when the publish failed while the worker
 was still alive, are read live out of it as before (`'live'`, with `route['fallback']` saying why)."""
 def __init__(self,provider,sb,name,mode,deadline,route,parts=None):
  self.provider,self.sb,self.name,self.mode,self.deadline,self.route,self.parts=provider,sb,name,mode,deadline,route,parts
 def part(self,part):
  if self.parts is not None:
   if part in self.parts:return self.parts[part]
   return None,{'part':part,'mode':self.mode,'bytes':0,'seconds':0.,'returncode':None,'stderr':'','handoff':'volume','error':self.route.get('failure') or 'the worker did not export this part'}
  if part=='logs' and self.deadline-time.time()<10:return None,{'part':'logs','skipped':'less than 10 s of the worker were left'}
  if part=='extras' and self.deadline-time.time()<20:return None,{'part':'extras','skipped':'less than 20 s of the worker were left'}
  files,record=self.provider._collect(self.sb,self.name,'final' if part=='logs' else self.mode,part,self.deadline,LOGS_SECONDS if part=='logs' else COLLECT_SECONDS_MAX)
  record['handoff']='live';return files,record

def receipts(app_id,sandbox_id):
 """A read of one worker's provider record; a failed read is no record, never a start. `.last` keeps the last read."""
 def info():
  try:value=allocation_info(app_id,sandbox_id) or {}
  except Exception:value={}
  info.last=value;return value
 info.last={}
 return info

# ---- leases ---------------------------------------------------------------------------------------------------------
SCRATCH='/scratch'
WORK='/work'
# Researcher commands on a lease: unprivileged, no way back to root, in their own session.
UNPRIVILEGED='setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs'
LEASE_TEARDOWN_SECONDS=20  # before the worker's end: stop commands, commit /scratch, let Modal record the end
OUTPUT_CHARS=20000
PACKAGE_SECONDS=120
# What `ar package` reads from a directory on the lease (a run folder or one of its exports), and the most of each.
PACKAGE_FILES={'final.npz':DEFAULT['artifact_bytes_cap'],'final.json':DEFAULT['metadata_bytes_cap']}
# A package is copied through this root-only folder: researcher processes cannot plant or swap files there.
PACKAGE_STAGE='/private/ar-package'
# researcher file on a lease -> the trainer path it stands in for. The package under /work/src is the worker's own,
# copied, with these four paths linked to the pushed files, so commands import exactly what was pushed.
LEASE_LINKS=dict(TRAINING['polytopia'])

def package_bundle(bundle,checkpoint,files,release_sha256,origin):
 """Seal a checkpoint a lease produced, with these trainer files, as the bundle a training run leaves.

 Nothing submitted is imported here: the checkpoint is checked as data (its own record's digest, the policy
 parameter cap, finite values), as a training run's is."""
 import numpy as np
 try:meta=json.loads(checkpoint['final.json'])
 except ValueError:raise ValueError('final.json is not JSON') from None
 if not isinstance(meta,dict) or meta.get('sha256')!=hashlib.sha256(checkpoint['final.npz']).hexdigest():
  raise ValueError('final.json does not name the digest of final.npz; write both with the trainer\'s own checkpoint writer (rl/runner.py save_checkpoint)')
 try:
  with np.load(io.BytesIO(checkpoint['final.npz']),allow_pickle=False) as weights:
   parameters=sum(weights[k].size for k in weights.files if k.startswith('p'))
   finite=all(bool(np.isfinite(weights[k]).all()) for k in weights.files)
 except Exception as exc:raise ValueError('final.npz is not a readable checkpoint: '+type(exc).__name__+': '+str(exc)[:200]) from None
 if not parameters:raise ValueError('final.npz holds no policy parameters (p0, p1, ...)')
 if parameters>DEFAULT['policy_parameter_cap']:raise ValueError('Policy parameter cap exceeded: %d of %d'%(parameters,DEFAULT['policy_parameter_cap']))
 if not finite:raise ValueError('final.npz holds values that are not finite')
 bundle=Path(bundle);bundle.mkdir()
 for name,text in files.items():
  target=bundle/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
 for name,data in checkpoint.items():(bundle/name).write_bytes(data)
 (bundle/'network.py').write_text(files[EDITABLE[0]]);(bundle/'ar_arch.json').write_text(files['arch.json'])
 write(bundle/'binding.json',{'benchmark_id':BENCHMARK,'release_sha256':release_sha256,'recipe_sha256':hashlib.sha256(files['recipe.json'].encode()).hexdigest(),'packaged_from':origin,'policy_parameters':int(parameters)})
 return seal(bundle)

class Lease:
 """One running lease: the sandbox, its commands and the host's record of them. Callers hold `lock`."""
 def __init__(self,sb,target,shape):
  self.sb=sb;self.target=Path(target);self.shape=shape;self.lock=threading.Lock()
  self.commands={}  # exec id -> {'process','command','started'}
  (self.target/'exec').mkdir(parents=True,exist_ok=True)
 def _root(self,script,*args,timeout=60):
  process=self.sb.exec('/bin/sh','-c',script,*args,timeout=timeout)
  stdout=process.stdout.read();stderr=process.stderr.read()
  return wait_process(process),stdout,stderr
 def setup(self):
  """/work: a writable copy of the worker's package and launch script, owned by the researcher's user."""
  code,_,err=self._root(f'mkdir -p {WORK}/.ar && cp -a /workspace/src /workspace/experiments {WORK}/ && chmod -R u+w {WORK} && chown -R 65534:65534 {WORK} && chmod 1777 {SCRATCH} && {UNPRIVILEGED} sh -c "touch {SCRATCH}/.writable && rm {SCRATCH}/.writable"',timeout=120)
  if code!=0:raise RuntimeError('Lease setup failed: '+err[-500:])
 def push(self,files):
  """Copy the editable files to /work and link the trainer paths to them."""
  folders=sorted({str(Path(WORK,path).parent) for path in files})
  code,_,err=self._root('mkdir -p '+' '.join(map(shlex.quote,folders)))
  if code!=0:raise RuntimeError('Lease push failed: '+err[-300:])
  for path,text in files.items():self.sb.filesystem.write_text(text,f'{WORK}/{path}')
  links=' && '.join(f'ln -sfn {WORK}/{path} {WORK}/{native}' for path,native in LEASE_LINKS.items() if path in files)
  code,_,err=self._root((links+' && ' if links else '')+f'chown -hR 65534:65534 {WORK}')
  if code!=0:raise RuntimeError('Lease push failed: '+err[-300:])
  return {'pushed':sorted(files),'to':WORK,'linked':{path:f'{WORK}/{native}' for path,native in LEASE_LINKS.items() if path in files}}
 def running(self):
  return [i for i,c in self.commands.items() if read_rpc(c['process'].poll) is None]
 def start(self,command):
  if len(self.running())>=DEFAULT['lease_running_commands']:raise ValueError(f'At most {DEFAULT["lease_running_commands"]} commands run on a lease at once')
  exec_id=f'{len(self.commands)+1:04d}';log=f'{WORK}/.ar/{exec_id}.log';cores=self.shape['cpu_cores']
  # The image default keeps JAX on the CPU for the host's own processes; a GPU lease's commands get the GPU.
  environment=(f'HOME={WORK} TMPDIR=/tmp VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:/usr/local/bin:/usr/local/cuda/bin:/usr/sbin:/usr/bin:/sbin:/bin '
   f'PYTHONPATH={WORK}/src:{WORK}/experiments PYTHONUNBUFFERED=1 POLYTOPIA_PC_BUILD=/ar/engine-build PC_CUDA_WARP_TICK=1 PC_CUDA_STACK_KB=64 '
   f'OMP_NUM_THREADS={cores} OPENBLAS_NUM_THREADS={cores} JAX_PLATFORMS={"cuda" if self.shape["gpu"] else "cpu"} XLA_PYTHON_CLIENT_PREALLOCATE=false')
  # $0 is the researcher's command, passed as an argument so nothing of it is interpreted as root. `--wait`: in
  # this image the command is a process group leader, so setsid forks; without it the command would be reported
  # as exited (code 0) the moment it started (seen live 2026-11-14).
  redirect = shlex.quote('exec bash -c "$1" > "$2" 2>&1 < /dev/null')
  script=f'cd {WORK} && exec {UNPRIVILEGED} env {environment} setsid --wait bash -c {redirect} lease "$0" {log}'
  process=self.sb.exec('/bin/sh','-c',script,command)
  self.commands[exec_id]={'process':process,'command':command,'started':time.time()}
  write(self.target/'exec'/(exec_id+'.json'),{'exec':exec_id,'command':command,'started':self.commands[exec_id]['started'],'state':'running','log':log})
  return {'exec':exec_id,'state':'running','log':log}
 def poll(self,exec_id,tail=None):
  entry=self.commands.get(exec_id)
  if entry is None:raise ValueError('Unknown command id for this lease')
  code=read_rpc(entry['process'].poll);chars=min(int(tail or OUTPUT_CHARS),OUTPUT_CHARS)
  _,output,_=self._root(f'{UNPRIVILEGED} tail -c {chars} -- {WORK}/.ar/{exec_id}.log 2>/dev/null',timeout=30)
  record={'exec':exec_id,'command':entry['command'],'started':entry['started'],'state':'running' if code is None else 'exited','exit_code':code,'output':output,'log':f'{WORK}/.ar/{exec_id}.log'}
  if code is not None:write(self.target/'exec'/(exec_id+'.json'),record)
  return record
 def sync(self,reload=False):
  """Commit /scratch so other leases can see it; `reload` also brings in what other leases committed."""
  code,_,err=self._root(f'sync {SCRATCH}',timeout=120)
  if code!=0:raise RuntimeError('Committing /scratch failed: '+err[-300:])
  if reload:
   try:self.sb.reload_volumes()
   except Exception as exc:raise ValueError('Reload failed; close files and leave directories under /scratch first: '+str(exc)[:300]) from None
  return {'committed':SCRATCH,'reloaded':bool(reload)}
 def fetch(self,folder,path):
  """The checkpoint files of a directory on the lease, {name: bytes}, read as the unprivileged user."""
  if not path.startswith('/') or '\0' in path:raise ValueError('--path is an absolute directory on the lease, such as /scratch/runs/r1')
  source=shlex.quote(path.rstrip('/') or '/')
  copies=' && '.join(f'{UNPRIVILEGED} head -c {cap+1} -- {source}/{name} > {PACKAGE_STAGE}/{name}' for name,cap in PACKAGE_FILES.items())
  code,_,err=self._root(f'rm -rf {PACKAGE_STAGE} && mkdir -p {PACKAGE_STAGE} && '+copies,timeout=PACKAGE_SECONDS)
  if code!=0:raise ValueError(f'{path} needs a readable final.npz and final.json: '+err[-300:].strip())
  folder=Path(folder);folder.mkdir(parents=True);found={}
  for name,cap in PACKAGE_FILES.items():
   read_rpc(lambda:self.sb.filesystem.copy_to_local(f'{PACKAGE_STAGE}/{name}',folder/name))
   found[name]=(folder/name).read_bytes()
   if len(found[name])>cap:raise ValueError(f'{name} is larger than the {cap}-byte limit')
  return found
 def close(self):
  """End every researcher process, keep the last output of each command, commit /scratch."""
  try:
   self._root('pkill -KILL -u 65534 || true',timeout=10)
   for exec_id in list(self.commands):
    try:
     record=self.poll(exec_id)
     if record['state']=='running':write(self.target/'exec'/(exec_id+'.json'),dict(record,state='ended with the lease'))
    except Exception:pass
   self._root(f'sync {SCRATCH}',timeout=120)
  except Exception as exc:  # the worker may already be gone; Modal commits the Volume when it terminates
   (self.target/'lease-close-error.log').write_text(type(exc).__name__+': '+str(exc))

class Provider:
 # How a created worker's start is awaited and an abandoned one is settled (harness.workers rules).
 start_patience=None;start_poll=2.0;abandon_poll=1.0
 # `ar stop` on a training run: how long its trainer may take to end by itself, and how often the request is looked for.
 stop_grace=180.0;stop_poll=0.5
 # Host-only (`--backend-option training_updates=N`): end training after N updates, for checkpoints at exact sample counts.
 training_updates=None
 # How long the hand-off Volume is waited for between attempts (VISIBLE_WAIT, COLLECT_WAIT; tests shorten them).
 visible_wait=VISIBLE_WAIT;collect_wait=COLLECT_WAIT
 def __init__(self,release,app,account=None,base_image_id=None,cpu_timeout=300,reuse_image_id=None,training_updates=None):
  self.account=account;self.cpu_timeout=cpu_timeout;self.cpu_cores=4;self.cpu_memory_gib=16;self._tokens={};self._training={}
  self.live={};self.lock=threading.Lock()  # lease name -> Lease, while its worker is serving commands
  if training_updates is not None:
   if isinstance(training_updates,bool) or not isinstance(training_updates,int) or training_updates<1:raise ValueError('training_updates is a positive whole number of updates')
   self.training_updates=training_updates
  import modal
  self.release=Path(release);verify(release);self.app=app;configure_sandbox_waits()
  if reuse_image_id:
   self.image=modal.Image.from_id(reuse_image_id);return
  base=modal.Image.from_id(base_image_id) if base_image_id else modal.Image.from_dockerfile(str(self.release/'base.Dockerfile'),context_dir=str(self.release/'base-context'))
  self.image=(base
   .add_local_dir(self.release/'runtime/src','/workspace/src',copy=True,ignore=['**/__pycache__/**'])
   .add_local_dir(self.release/'runtime/experiments','/workspace/experiments',copy=True)
   .add_local_dir(self.release/'maps','/results/maps',copy=True)
   .add_local_dir(self.release/'v1','/opt/v1',copy=True,ignore=['**/__pycache__/**'])
   .env({'PYTHONPATH':'/workspace/src','JAX_PLATFORMS':'cpu','XLA_PYTHON_CLIENT_PREALLOCATE':'false','POLYTOPIA_PC_BUILD':'/ar/engine-build','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'})
   .run_commands('chmod o+x /root /root/.local /root/.local/share /root/.local/share/uv /root/.local/share/uv/python','chmod -R a-w /workspace/src /workspace/experiments /results/maps /opt/v1','mkdir -p /private /candidate /opponent; chmod 700 /private').workdir('/workspace'))
  build_token=self.account.reserve_build(verify(self.release)) if self.account else None
  self.image.build(app)
  if self.account:self.account.hold_build(build_token,self.image.object_id)
 def _create(self,role,seconds,gpu=False,shape=None,volumes=None):
  import modal
  # A training worker is the contract's training shape with the GPU; a lease names its own shape (and mounts
  # the scratch Volume); every other worker is the 4-core CPU shape.
  if shape:gpu,cores,memory=bool(shape['gpu']),shape['cpu_cores'],shape['memory_mib']//1024
  else:cores,memory=(TRAIN_SHAPE['cpu_cores'],TRAIN_SHAPE['memory_mib']//1024) if gpu else (self.cpu_cores,self.cpu_memory_gib)
  token=self.account.reserve(role,seconds,hardware.GPU if gpu else None,cores,memory) if self.account else None
  if token:self.account.note(token,app_id=self.app.app_id)
  # Modal bills max(request, use). A training worker or a lease is charged to the research ledger by its shape and
  # reserves it whole; a CPU-only worker (preflight, parameter check, practice, grading: one game at a time, two
  # single-threaded policy processes) reserves Modal's 1 CPU / 2 GiB under the unchanged 4-core / 16 GiB cap.
  lean=not gpu and shape is None
  cpu=(min(CPU_REQUEST,cores),cores) if lean else (cores,cores)
  mem=(min(MEMORY_REQUEST_MIB,memory*1024),memory*1024) if lean else (memory*1024,memory*1024)
  sb=modal.Sandbox.create('/bin/sleep','86400',environment_name=ENVIRONMENT,app=self.app,image=self.image,gpu=hardware.GPU if gpu else None,cpu=cpu,memory=mem,timeout=max(1,int(seconds-15)),block_network=True,include_oidc_identity_token=False,secrets=[],volumes=volumes or {},tags={'polytopia_v1_role':role,'reservation':token or 'historical'})
  if token:self._tokens[sb.object_id]=token;self.account.attach(token,sb)
  return sb
 def _stop(self,sb):
  termination_error=None
  try:sb.terminate(wait=True)
  except Exception as e:termination_error=type(e).__name__+': '+str(e)
  receipt=None
  for _ in range(40):
   receipt=allocation_info(self.app.app_id,sb.object_id)
   if receipt and receipt.get('finished_at'):break
   time.sleep(.25)
  token=self._tokens.get(sb.object_id)
  if token:self.account.finish(token,receipt)
  return receipt
 def parameter_counts(self,workspace):
  import modal
  workspace=Path(workspace);sb=None
  try:
   sb=self._create('cpu-parameter-check',120)
   for rel in EDITABLE:sb.filesystem.write_text((workspace/rel).read_text(),'/workspace/'+rel)
   cfg=json.loads((workspace/'recipe.json').read_text());cfg.pop('trainer',None);cfg.pop('_doc',None)
   sb.filesystem.write_text(json.dumps(cfg),'/tmp/config.json');sb.filesystem.copy_from_local(workspace/'arch.json','/tmp/arch.json')
   code="import json,jax;from polytopia_bot.rl.gpu_resident import LearnerConfig,network_spec_of,policy_template,critic_template;c=LearnerConfig(**json.load(open('/tmp/config.json')));c.validate();s=network_spec_of(c,json.load(open('/tmp/arch.json')));n=lambda p:int(sum(x.size for x in jax.tree.leaves(p)));print(json.dumps({'policy':n(policy_template(s)),'critic':n(critic_template(s))}))"
   wrapper="import subprocess,sys,os;p=subprocess.run(['/opt/venv/bin/python','-c',sys.argv[1]],user=65534,group=65534,extra_groups=[],cwd='/tmp',env=dict(os.environ,PYTHONPATH='/workspace/src',JAX_PLATFORMS='cpu'),capture_output=True,text=True,timeout=85);print(p.stdout);print(p.stderr,file=sys.stderr);sys.exit(p.returncode)"
   proc=sb.exec('python','-c',wrapper,code,timeout=95);stdout=proc.stdout.read();stderr=proc.stderr.read();wait_process(proc)
   if proc.returncode:raise ValueError('CPU parameter preflight failed: '+stderr[-700:])
   result=json.loads(stdout.strip().splitlines()[-1]);result['policy_cap']=500000;result['critic_cap']=500000
   result['errors']=[key+' parameter cap exceeded' for key,cap in [('policy',500000),('critic',500000)] if result[key]>cap]
   return result
  finally:
   if sb:self._stop(sb)
 def preflight(self,out):
  out=Path(out).with_name('cpu-preflight-'+str(time.time_ns())+'.json')
  import modal
  sb=None
  try:
   sb=self._create('cpu-preflight',min(300,self.cpu_timeout))
   sb.filesystem.copy_from_local(self.release/'runtime/src/polytopia_bot/rl/network.py','/candidate/network.py')
   sb.filesystem.write_text('private marker','/private/marker')
   p=sb.exec('python','/opt/v1/preflight.py',timeout=max(1,min(260,self.cpu_timeout-25)));stdout=p.stdout.read();stderr=p.stderr.read();wait_process(p)
   write(out,{'returncode':p.returncode,'stdout':stdout,'stderr':stderr,'gpu_seconds':0,'sandbox_id':sb.object_id})
   if p.returncode:
    q=sb.exec('/bin/sh','-c','tail -c 3000 /private/policy-*.stderr',timeout=10);extra=q.stdout.read();wait_process(q);Path(out).with_suffix('.policy-error.log').write_text(extra)
    raise RuntimeError('CPU preflight failed; no GPU launched')
  finally:
   if sb:self._stop(sb)
 def stop(self,name):
  """Terminate a training worker at once (a session being closed): nothing of the run is kept. It still settles on
  its provider receipt, exactly like the deadline timer. `ar stop` does not come here: see `train`."""
  sb=self._training.get(name)
  if sb is not None:sb.terminate()
  return sb is not None
 def abandon(self,app_id,sandbox_id):
  """Terminate a recorded worker this session no longer drives and return its final provider record (harness.workers)."""
  import modal
  return workers.abandon(lambda:modal.Sandbox.from_id(sandbox_id).terminate(),receipts(app_id,sandbox_id),poll=self.abandon_poll)
 def lookup(self,app_id,sandbox_id):
  """Read-only provider receipt for an allocation this session recorded; never starts or restarts work."""
  return allocation_info(app_id,sandbox_id)
 def _collect(self,sb,name,mode,part,deadline,most=COLLECT_SECONDS_MAX):
  """One archive stream of a part of a run (EXPORT_SCRIPT) from its worker: ({name: bytes} or None, a record of it).

  Never raises: the record says how long it took, how many bytes arrived, how the command ended, what it wrote to
  stderr and, if the archive was unusable, why. The stream has its own time limit, and reaching it is told apart
  from a command that failed: the pinned SDK ends a stream silently at that limit and reports code -1."""
  import modal
  began=time.time();timeout=collect_seconds(deadline,began,most)
  record={'part':part,'mode':mode,'timeout_seconds':timeout,'started_seconds_before_worker_end':deadline-began,'bytes':0,'seconds':0.,'returncode':None,'stderr':''}
  files=None
  try:
   reset_rpc_channels();worker=modal.Sandbox.from_id(sb.object_id)
   proc=worker.exec('python','-c',EXPORT_SCRIPT,'/output/'+name,mode,part,TRAINER_STDOUT,TRAINER_STDERR,str(STDOUT_TAIL_BYTES),str(STDERR_TAIL_BYTES),text=False,timeout=timeout)
   payload=proc.stdout.read();record['bytes']=len(payload)
   stderr=proc.stderr.read();record['stderr']=(stderr.decode(errors='replace') if isinstance(stderr,bytes) else str(stderr))[-2000:]
   code=wait_process(proc);record['returncode']=code
   record['seconds']=time.time()-began
   record['reached_time_limit']=bool(code==-1 or (code and record['seconds']>=timeout-1))
   if not code:
    try:files=unpack(payload)
    except ValueError as exc:record['unpack_error']=str(exc)
  except Exception as exc:
   record['seconds']=time.time()-began;record['error']=type(exc).__name__+': '+str(exc)[:600]
  return files,record
 def _keep_logs(self,files,out):
  """Write the tails of the trainer's output that an archive carried as the job's stdout.log and stderr.log."""
  for member,target in zip(LOG_TAILS,('stdout.log','stderr.log')):
   if files and member in files:(Path(out)/target).write_bytes(files[member])
   elif not (Path(out)/target).exists():(Path(out)/target).write_text('')
 def _volume(self,out):
  """The hand-off Volume of the backend root a job folder belongs to (the controller's <root>/jobs/<name>): a research
  session's and its fresh final's are separate."""
  return handoff.SessionVolume('polytopia',Path(out).resolve().parent.parent)
 def _publish(self,sb,name,job,kind,mode,deadline):
  """Run the hand-off step in the worker (HANDOFF_SCRIPT) and wait only for it: a copy into the Volume inside the
  datacenter, whose reply is one short JSON line. Never raises: the record says how it ended."""
  import modal
  began=time.time();timeout=collect_seconds(deadline,began,PUBLISH_SECONDS_MAX)
  record={'kind':kind,'mode':mode,'timeout_seconds':timeout,'started_seconds_before_worker_end':deadline-began,'returncode':None,'seconds':0.,'stderr':''}
  try:
   reset_rpc_channels();worker=modal.Sandbox.from_id(sb.object_id)
   proc=worker.exec(*handoff_command(job,name,kind,mode),timeout=timeout)
   stdout=proc.stdout.read();record['stderr']=str(proc.stderr.read())[-2000:]
   code=wait_process(proc);record['returncode']=code;record['seconds']=time.time()-began
   record['reached_time_limit']=bool(code==-1 or (code and record['seconds']>=timeout-1))
   if not code:
    reply=json.loads(stdout.strip().splitlines()[-1]);record['manifest']=reply['manifest'];record['reused']=bool(reply.get('reused'))
  except Exception as exc:
   record['seconds']=time.time()-began;record['error']=type(exc).__name__+': '+str(exc)[:600]
  return record
 def _receive(self,sb,volume,name,job,kind,mode,deadline,out):
  """Bring an ended trainer's result to the host. The worker publishes it (`_publish`); once its manifest shows in the
  Volume the worker is ended, and only then is the Volume read, with retries and no deadline, so the host's link never
  costs paid time or a finished run. Every file is checked against the manifest (harness.handoff) and each archive
  against its own checksums (`unpack`), as strictly as a live read. A publish that failed, or that the Volume does not
  show while the worker is still alive, falls back to reading the archives live out of the worker, as before."""
  route={'handoff':'volume','job':job,'volume':volume.name}
  publish=self._publish(sb,name,job,kind,mode,deadline);route['publish']=publish;failure=None
  if publish['returncode']==0 or 'error' in publish:  # published, or its reply was lost: look for it in the Volume
   began=time.time()
   for attempt in range(VISIBLE_ATTEMPTS):
    try:
     seen=volume.manifest(job)
     if 'manifest' in publish and seen!=publish['manifest']:raise handoff.NotPublished('the Volume does not show the manifest the worker published')
     failure=None;break
    except Exception as exc:
     failure='the published run did not show in the hand-off Volume: '+type(exc).__name__+': '+str(exc)[:600]
     if attempt+1<VISIBLE_ATTEMPTS:time.sleep(self.visible_wait)
   route['visible_seconds']=time.time()-began
  elif publish.get('reached_time_limit'):failure='the publish step was cut off at its %d-second time limit'%publish['timeout_seconds']
  else:failure='the publish step ended with code %s: %s'%(publish['returncode'],(publish.get('stderr') or 'it wrote nothing to stderr').strip()[-600:])
  if failure is not None:
   route.update(handoff='live',fallback=failure)
   return Received(self,sb,name,mode,deadline,route)
  # The worker's part is done: end it before anything crosses the host's link (the finally of `train` settles it).
  try:sb.terminate();route['worker_ended_before_collection']=True
  except Exception as exc:route['worker_ended_before_collection']=type(exc).__name__+': '+str(exc)[:300]
  began=time.time();parts={}
  try:
   got=volume.collect(job,Path(out)/'handoff',attempts=COLLECT_ATTEMPTS,wait=self.collect_wait)
   route.update(collect_seconds=time.time()-began,bytes=sum(entry['bytes'] for entry in got.values()))
   status=json.loads(Path(got['status.json']['path']).read_text());route['prepare_error']=status.get('prepare_error')
   for part,entry in status['parts'].items():
    data=Path(got[part+'.tgz']['path']).read_bytes();files=None
    record={'part':part,'mode':status['mode'],'bytes':len(data),'seconds':0.,'returncode':entry['returncode'],'stderr':str(entry['stderr'])[-2000:],'handoff':'volume'}
    if not entry['returncode']:
     try:files=unpack(data)
     except ValueError as exc:record['unpack_error']=str(exc)
    parts[part]=(files,record)
  except Exception as exc:
   route.update(collect_seconds=time.time()-began,failure='the run was published to the hand-off Volume %s as %s, but could not be read from it: %s: %s'%(volume.name,job,type(exc).__name__,str(exc)[:600]))
   return Received(self,sb,name,mode,deadline,route,{})  # the worker has ended; the published files stay in the Volume
  # An archive the host could not unpack is the only copy of a finished run: it stays in the Volume and beside the
  # job (handoff/*.tgz), for a host that can read it. The worker has ended, so nothing is left to protect by removing it.
  refused=sorted(part for part,(files,record) in parts.items() if 'unpack_error' in record)
  if refused:
   route.update(removed_from_volume=False,kept_unread=refused)
   return Received(self,sb,name,mode,deadline,route,parts)
  # What the archives held is kept, checked, in the bundle and the job's logs; nothing stays for later workers to read.
  for entry in got.values():
   if entry['path'].endswith('.tgz'):os.unlink(entry['path'])
  try:volume.volume().remove_file(job,recursive=True);route['removed_from_volume']=True
  except Exception as exc:route['removed_from_volume']=type(exc).__name__+': '+str(exc)[:300]
  return Received(self,sb,name,mode,deadline,route,parts)
 def train(self,ledger,name,grant,workspace,out,smoke=True,stop=None):
  import modal
  out=Path(out);out.mkdir(parents=True,exist_ok=False);workspace=Path(workspace);verify(self.release);shutil.copytree(workspace,out/'input');workspace=out/'input';counts=self.parameter_counts(workspace);write(out/'parameter-counts.json',counts)
  if counts['errors']:raise ValueError('; '.join(counts['errors']))
  if stop and stop():raise RuntimeError('Stopped before any GPU allocation')
  # The session's private hand-off Volume (created if needed, before anything is reserved) is mounted into this
  # training worker only, never into a CPU worker; the wrapper makes it root-only before the trainer starts.
  volume=self._volume(out);mounts=volume.mounts()
  ledger.reserve(name,grant);sb=None;receipt=None;error=None;timer=None;unstarted=None
  # SandboxCreate timeout is shorter than reservation; no automatic retries/warm pool.
  try:
   sb=self._create('training-'+name,grant,True,volumes=mounts);self._training[name]=sb;job=job_name(name,sb.object_id)
   write(out/'provider.json',{'sandbox_id':sb.object_id,'app_id':self.app.app_id,'ledger_job_name':getattr(ledger,'prefix','')+name,'handoff_volume':volume.name,'handoff_job':job})
   # A created worker may sit in the provider's queue for a GPU: wait for its start (queued time is not charged),
   # ending the wait early on a stop or when it finished without ever starting.
   info=receipts(self.app.app_id,sb.object_id)
   receipt=workers.await_start(info,patience=self.start_patience,poll=self.start_poll,stop=lambda:bool(stop and stop()) or bool(info.last.get('finished_at')))
   if receipt is None:
    stopped=bool(stop and stop())
    receipt=workers.abandon(lambda:sb.terminate(),info,poll=self.abandon_poll)
    write(out/'abandoned.json',{'receipt':receipt,'stop_requested':stopped,'patience_seconds':workers.START_PATIENCE_SECONDS if self.start_patience is None else self.start_patience})
    if receipt.get('provider_id') and workers.never_started(receipt):
     reason=STOPPED_QUEUED if stopped else workers.NEVER_STARTED
     ledger.settle_unstarted(name,receipt,stopped,reason);unstarted=workers.NeverStarted(receipt,reason)
     raise unstarted
    # It started while being abandoned: the failure path below settles it from its start/finish receipt.
    raise RuntimeError('Stopped by request' if stopped else LATE_START)
   ledger.start(name,sb.object_id,receipt['started_at'])
   if not hardware.matches(receipt['gpu_type']):raise RuntimeError('Wrong provider GPU SKU')
   deadline=receipt['started_at']+grant-TEARDOWN_SECONDS
   timer=threading.Timer(max(1,deadline-time.time()),lambda:sb.terminate());timer.daemon=True;timer.start()
   for rel in EDITABLE:sb.filesystem.write_text((Path(workspace)/rel).read_text(),'/workspace/'+rel)
   cfg=json.loads((Path(workspace)/'recipe.json').read_text());trainer=cfg.pop('trainer');cfg.pop('_doc',None)
   sb.filesystem.write_text(json.dumps(cfg),'/workspace/config.json');sb.filesystem.write_text((Path(workspace)/'arch.json').read_text(),'/workspace/arch.json')
   if stop and stop():raise RuntimeError(STOPPED)  # nothing has trained yet: the worker is simply ended
   plan=schedule(receipt['started_at'],grant,time.time(),smoke);seconds,limit=plan['seconds'],plan['limit']
   args=['python','/opt/v1/train.py','--config','/workspace/config.json','--arch','/workspace/arch.json','--run-id',name,'--run-root','/output','--seconds',str(seconds)]
   for k in ('games','sets','ring'):args+=['--'+k,str(trainer[k])]
   if self.training_updates:args+=['--updates',str(self.training_updates)]
   write(out/'schedule.json',dict(plan,grant=grant,started_at=receipt['started_at'],command_started_at=time.time(),teardown_seconds=TEARDOWN_SECONDS,collect_seconds=COLLECT_SECONDS,trainer_margin_seconds=TRAINER_MARGIN_SECONDS))
   # The trainer writes its output to two files in the worker; nothing is read from this command's own stream. Its
   # wrapper makes /handoff root-only before it starts and publishes the run once it has ended by itself (TRAIN_WRAPPER).
   plan={'prepare':handoff.prepare_command('python'),'staging':HANDOFF_STAGING+'/'+job,'stdout':TRAINER_STDOUT,'stderr':TRAINER_STDERR,'handoff':handoff_command(job,name,'network','final'),'handoff_seconds':PUBLISH_SECONDS_MAX}
   p=sb.exec('python','-c',TRAIN_WRAPPER,json.dumps(plan),*args,timeout=max(1,int(limit-time.time())),env={'JAX_PLATFORMS':'cuda','POLYTOPIA_ALLOCATION_STARTED':str(receipt['started_at']),'POLYTOPIA_STOP_REQUEST':STOP_REQUEST})
   # While the trainer runs, look for a stop. `ar stop` asks the trainer to end at its next step by creating
   # STOP_REQUEST; it then finishes as it does at its time limit and is collected like any finished run. A trainer
   # that has not ended within the grace is no longer waited for: its newest whole export is collected instead.
   # A session being closed (`stop()` is 'now'), or a request that could not be delivered, ends the worker at once.
   asked=None;waited_out=False
   while read_rpc(p.poll) is None and time.time()<limit+3:
    request=stop() if stop else None
    if request=='now':raise RuntimeError(STOPPED)
    if request and asked is None:
     asked=time.time()
     try:read_rpc(lambda:sb.filesystem.write_text('stop\n',STOP_REQUEST))
     except Exception:raise RuntimeError(STOPPED+'; the request could not be delivered to the trainer') from None
    if asked is not None and time.time()-asked>=self.stop_grace:waited_out=True;break
    time.sleep(self.stop_poll)
   # A trainer that did not end by itself with success keeps the newest whole export it wrote (a run with none
   # fails) when the researcher stopped it or when it was cut off at its time limit (its command reached `limit`;
   # the SDK then reports code -1). A trainer that fails by itself, unasked and before its limit, stays a failure.
   code=None if waited_out else read_rpc(p.poll);ended_at=time.time()
   unfinished=code is None or bool(code)
   write(out/'trainer.json',{'returncode':code,'ended_seconds_before_worker_end':deadline-ended_at,'ended_seconds_before_command_limit':limit-ended_at,'stop_asked':asked is not None,'waited_out':waited_out})
   # The run is handed back the moment the trainer has ended: published in the worker, then read from the Volume
   # (`_receive`). A trainer that failed by itself hands back only the tails of its output.
   alone=unfinished and asked is None and ended_at<limit-2
   mode='early' if unfinished else 'final'
   got=self._receive(sb,volume,name,job,'logs' if alone else 'network',mode,deadline,out);write(out/'handoff.json',got.route)
   if alone:
    self._keep_logs(got.part('logs')[0],out)
    raise RuntimeError('Training failed; see retained stderr')
   files,record=got.part('network');records=[record]
   self._keep_logs(files,out)
   if files is None or not set(ALWAYS+NETWORK)<=set(files) or not set(OPTIONAL)&set(files):
    reason=why(record) if files is None else 'the archive lacks '+', '.join(sorted(set(ALWAYS+NETWORK)-set(files)) or OPTIONAL)
    write(out/'export-error.json',{'reason':reason,'collections':records,'trainer_returncode':code,'mode':mode,'handoff':got.route['handoff'],'handoff_fallback':got.route.get('fallback')})
    if files is None:self._keep_logs(got.part('logs')[0],out)
    if unfinished and NO_EXPORT_MARK in (record.get('stderr') or ''):raise RuntimeError((STOPPED+', and ' if asked is not None else 'Training was cut off at its time limit, and ')+NO_EXPORT)
    if unfinished:raise RuntimeError((STOPPED if asked is not None else 'Training was cut off at its time limit')+', and its last export could not be collected: '+reason)
    raise RuntimeError('Bounded archive export failed: '+reason)
   bundle=out/'bundle';bundle.mkdir();shutil.copytree(Path(workspace),bundle/'source')
   kept=[n for n in files if n in ALWAYS+NETWORK+OPTIONAL+('export-checksums.json',)]
   for n in kept:(bundle/n).write_bytes(files[n])
   # The clock snapshots are a second archive (read live only if the worker has time left); the run counts without them.
   extras=json.loads(files.get('extras.json',b'{}'));collected_extras=[]
   if extras:
    more,record=got.part('extras');records.append(record)
    if more is not None and all(n.startswith('clock-') or n=='export-checksums.json' for n in more):
     sums=json.loads((bundle/'export-checksums.json').read_text());sums.update(json.loads(more.pop('export-checksums.json')))
     for n,data in more.items():(bundle/n).write_bytes(data)
     write(bundle/'export-checksums.json',sums);collected_extras=sorted(more)
    elif 'skipped' not in record:record['not_kept']=why(record) if more is None else 'unexpected members'
   checksums=json.loads((bundle/'export-checksums.json').read_text())
   if any(digest(bundle/f)!=h for f,h in checksums.items()):raise ValueError('Export checksum mismatch')
   names=set(kept)|set(collected_extras)
   # What was kept: the trainer's own final network, or the newest whole export of a trainer that did not get there.
   last_export=json.loads((bundle/'export.json').read_text()) if 'export.json' in names else None
   if last_export is not None:ended='stopped_early' if asked is not None else 'cut_short'
   else:ended='stopped_early' if asked is not None and json.loads((bundle/'training-complete.json').read_text()).get('stopped') is not False else 'complete'
   transport=('hand-off Volume: the run and its clock snapshots as gzip-tar archives, published in the worker and read after it ended' if got.route['handoff']=='volume' else 'gzip-tar streams read live from the worker: the run, then the clock snapshots')
   write(out/'export.json',{'transport':transport,'handoff':got.route['handoff'],'handoff_fallback':got.route.get('fallback'),'bytes':sum(r.get('bytes',0) for r in records),'seconds':sum(r.get('seconds',0) for r in records),'files':sorted(names),'fresh_rpc_channels':True,'ended':ended,'network':'last complete export' if last_export else 'final','last_export':last_export,'clock_snapshots':{'in_worker':sorted(extras),'collected':collected_extras},'collections':records})
   shutil.copy2(Path(workspace)/EDITABLE[0],bundle/'network.py');shutil.copy2(Path(workspace)/'arch.json',bundle/'ar_arch.json')
   write(bundle/'binding.json',{'benchmark_id':BENCHMARK,'release_sha256':verify(self.release),'recipe_sha256':digest(Path(workspace)/'recipe.json')})
   seal(bundle)
   # Check deltas without importing submitted code on the host.
   import numpy as np
   with np.load(bundle/'initial.npz',allow_pickle=False) as a,np.load(bundle/'final.npz',allow_pickle=False) as b:
    changed=sum(not np.array_equal(a[k],b[k]) for k in a.files if k.startswith('p'))
    if sum(b[k].size for k in b.files if k.startswith('p'))>500000:raise ValueError('Policy parameter cap exceeded')
    finite=all(np.isfinite(b[k]).all() for k in b.files)
   with np.load(bundle/'final-critic.npz',allow_pickle=False) as critic:
    if sum(critic[k].size for k in critic.files if k.startswith('p'))>500000:raise ValueError('Critic parameter cap exceeded')
   lines=[l for l in (bundle/'training.jsonl').read_text().splitlines() if l.strip()];rows=[]
   for i,l in enumerate(lines):
    try:rows.append(json.loads(l))
    except ValueError:
     if last_export is None or i!=len(lines)-1:raise  # only the line an ended trainer was writing may be cut
   # A kept export is judged on the updates it holds, not on later ones the trainer logged but never exported.
   if last_export is not None:rows=[r for r in rows if not isinstance(r,dict) or r.get('update',0)<=last_export.get('update',0)]
   if not rows or not changed or not finite or any(not math.isfinite(float(r[k])) for r in rows for k in ('policy_loss','value_loss') if k in r):raise RuntimeError('Learning evidence failed')
   write(out/'learning.json',{'updates':len(rows),'changed_policy_tensors':changed,'finite_weights':finite,'last':rows[-1],'ended':ended})
  except BaseException as e:
   import traceback
   error=type(e).__name__+': '+str(e);(out/'error.txt').write_text(traceback.format_exc())
  finally:
   if timer:timer.cancel()
   self._training.pop(name,None)
   if sb and unstarted is None:
    try:self._stop(sb)
    except Exception as e:error=(error or '')+'; termination '+str(e)
    for _ in range(30):
     receipt=allocation_info(self.app.app_id,sb.object_id)
     if receipt and receipt.get('finished_at'):break
     time.sleep(.5)
   token=self._tokens.get(sb.object_id) if sb else None
   if token and receipt and receipt.get('finished_at') and self.account.operations[token]['status']!='settled':self.account.finish(token,receipt)
   if unstarted is None:ledger.settle(name,receipt,'failed' if error else 'complete')
   write(out/'allocation.json',{'provider':receipt,'error':error,'all_worker_phases_charged':True,'charged':DEFAULT['allocation_rule']})
  if unstarted is not None:raise unstarted
  if error:raise RuntimeError(error)
  return {'bundle':str(bundle),'learning':json.loads((out/'learning.json').read_text()),'budget':ledger.public(),'ended':json.loads((out/'export.json').read_text())['ended']}
 # ---- leases ---------------------------------------------------------------------------------------------------
 lease_tick=1.0  # how often a held lease looks for its end or a stop
 def scratch(self,root):
  """The session's scratch Volume, created with its first lease and recorded in the session directory.

  It is never deleted here: it holds the session's checkpoints and is collected for the run archive."""
  import modal
  path=Path(root)/'scratch-volume.json'
  with self.lock:
   if not path.exists():write(path,{'volume':'polytopia-scratch-'+uuid.uuid4().hex,'environment':ENVIRONMENT,'mount':SCRATCH,'created_unix':time.time()})
   name=json.loads(path.read_text())['volume']
  return modal.Volume.from_name(name,environment_name=ENVIRONMENT,create_if_missing=True,version=2)
 def hold(self,ledger,name,grant,shape,files,out,stop=None):
  """A lease: one worker of `shape`, held for the researcher's own commands until its grant runs out or it is stopped.

  The grant, the start wait, the abandon rules and the settlement on the provider's receipt are a training run's
  (`train`). Researcher commands run as an unprivileged user in /work with /scratch mounted; nothing of the host's
  is on the worker (the practice opponents and private schedules never enter this image)."""
  out=Path(out);out.mkdir(parents=True,exist_ok=False);verify(self.release)
  write(out/'request.json',{'kind':'lease','shape':shape,'grant':grant,'files':sorted(files)})
  if stop and stop():raise RuntimeError('Stopped before any allocation')
  ledger.reserve(name,grant);sb=None;receipt=None;error=None;timer=None;unstarted=None
  try:
   sb=self._create('lease-'+name,grant,shape=shape,volumes={SCRATCH:self.scratch(out.parent.parent)})
   write(out/'provider.json',{'sandbox_id':sb.object_id,'app_id':self.app.app_id,'ledger_job_name':name})
   info=receipts(self.app.app_id,sb.object_id)
   receipt=workers.await_start(info,patience=self.start_patience,poll=self.start_poll,stop=lambda:bool(stop and stop()) or bool(info.last.get('finished_at')))
   if receipt is None:
    stopped=bool(stop and stop())
    receipt=workers.abandon(lambda:sb.terminate(),info,poll=self.abandon_poll)
    write(out/'abandoned.json',{'receipt':receipt,'stop_requested':stopped,'patience_seconds':workers.START_PATIENCE_SECONDS if self.start_patience is None else self.start_patience})
    if receipt.get('provider_id') and workers.never_started(receipt):
     reason=STOPPED_QUEUED if stopped else workers.NEVER_STARTED
     ledger.settle_unstarted(name,receipt,stopped,reason);unstarted=workers.NeverStarted(receipt,reason)
     raise unstarted
    raise RuntimeError('Stopped by request' if stopped else LATE_START)
   ledger.start(name,sb.object_id,receipt['started_at'])
   if shape['gpu'] and not hardware.matches(receipt['gpu_type']):raise RuntimeError('Wrong provider GPU SKU')
   deadline=receipt['started_at']+grant-TEARDOWN_SECONDS
   timer=threading.Timer(max(1,deadline-time.time()),lambda:sb.terminate());timer.daemon=True;timer.start()
   lease=Lease(sb,out,shape);lease.setup();lease.push(files)
   self.live[name]=lease
   write(out/'lease.json',{'shape':shape,'scratch':SCRATCH,'work':WORK,'ready_unix':time.time(),'serves_until_unix':deadline-LEASE_TEARDOWN_SECONDS})
   try:
    while time.time()<deadline-LEASE_TEARDOWN_SECONDS and not (stop and stop()):time.sleep(self.lease_tick)
   finally:
    self.live.pop(name,None)
    with lease.lock:lease.close()  # after any command or package in flight
  except BaseException as e:
   import traceback
   error=type(e).__name__+': '+str(e);(out/'error.txt').write_text(traceback.format_exc())
  finally:
   if timer:timer.cancel()
   if sb and unstarted is None:
    try:self._stop(sb)
    except Exception as e:error=(error or '')+'; termination '+str(e)
    for _ in range(30):
     receipt=allocation_info(self.app.app_id,sb.object_id)
     if receipt and receipt.get('finished_at'):break
     time.sleep(.5)
   if unstarted is None:ledger.settle(name,receipt,'failed' if error else 'stopped' if stop and stop() else 'complete')
   write(out/'allocation.json',{'provider':receipt,'error':error,'shape':shape,'all_worker_phases_charged':True,'charged':DEFAULT['allocation_rule']})
  if unstarted is not None:raise unstarted
  if error:raise RuntimeError(error)
 def lease_live(self,name):
  return name in self.live
 def _live(self,name):
  lease=self.live.get(name)
  if lease is None:raise ValueError('The lease is not serving commands (starting up or ending)')
  return lease
 def lease_exec(self,name,command):
  lease=self._live(name)
  with lease.lock:return lease.start(command)
 def lease_poll(self,name,exec_id,tail=None):
  lease=self._live(name)
  with lease.lock:return lease.poll(exec_id,tail)
 def lease_push(self,name,files):
  lease=self._live(name)
  with lease.lock:return lease.push(files)
 def lease_sync(self,name,reload=False):
  lease=self._live(name)
  with lease.lock:return lease.sync(reload)
 def lease_package(self,target,name,path,files):
  """Seal final.npz and final.json from a directory on the lease, with these trainer files, as a run's bundle.

  The researcher's files are read as the unprivileged user and checked on the host as data (package_bundle), so
  the policy parameter cap holds for a packaged network as for a trained one. Returns the bundle's folder."""
  lease=self._live(name);target=Path(target)
  with lease.lock:
   target.mkdir(parents=True,exist_ok=False)
   try:
    checkpoint=lease.fetch(target/'fetched',path)
    package_bundle(target/'bundle',checkpoint,files,verify(self.release),{'lease':name,'path':path})
    return target/'bundle'
   except BaseException as exc:
    (target/'package-error.log').write_text(type(exc).__name__+': '+str(exc))
    target.rename(target.with_name(target.name+'.failed-'+uuid.uuid4().hex[:8]))
    raise
 def evaluate(self,bundle,seeds,opponent_seed,out,opponent_bundle=None):
  import modal
  verify(bundle);sb=None;start=time.time()
  try:
   sb=self._create('cpu-grade',self.cpu_timeout)
   write(Path(out).with_suffix('.worker.json'),{'sandbox_id':sb.object_id,'app_id':self.app.app_id,'gpu':None})
   for f in ('final.npz','final.json','network.py','ar_arch.json'):sb.filesystem.copy_from_local(Path(bundle)/f,'/candidate/'+f)
   sb.filesystem.copy_from_local(self.release/'runtime/src/polytopia_bot/rl/network.py','/opponent/network.py')
   if opponent_bundle:
    for f in ('final.npz','final.json','network.py','ar_arch.json'):
     if (Path(opponent_bundle)/f).exists():sb.filesystem.copy_from_local(Path(opponent_bundle)/f,'/opponent/'+f)
   sb.filesystem.write_text(json.dumps({'seeds':seeds,'opponent_seed':opponent_seed,'opponent_name':Path(opponent_bundle).name if opponent_bundle else 'independent_random_minimal'}),'/private/request.json')
   proc=sb.exec('python','/opt/v1/grade.py','/private/request.json',timeout=max(1,self.cpu_timeout-25))
   stdout=proc.stdout.read();stderr=proc.stderr.read();wait_process(proc)
   Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).with_suffix('.stdout.log').write_text(stdout);Path(out).with_suffix('.stderr.log').write_text(stderr)
   if proc.returncode:
    p=sb.exec('/bin/sh','-c','tail -c 3000 /private/policy-*.stderr',timeout=10);extra=p.stdout.read();wait_process(p);Path(out).with_suffix('.policy-error.log').write_text(extra)
    raise RuntimeError('CPU evaluation failed')
   result=json.loads(sb.filesystem.read_text('/private/result.json'));write(out,result);return {k:v for k,v in result.items() if k!='details'}
  finally:
   if sb:self._stop(sb)
   write(Path(out).with_suffix('.cleanup.json'),{'owned_worker_stopped':sb is not None,'wall_seconds':time.time()-start,'gpu_seconds':0})
