"""Tavern Modal workers: the training and evaluation sandboxes and their provider receipts."""
from harness.policy import ENVIRONMENT
from harness import hardware
from pathlib import Path
import time,asyncio,json,threading
from simulators.tavern.autoresearch.contract import sha
async def _poll_wait(process):
    """Pinned SDK workaround: short status RPCs instead of TaskExecWait."""
    while (code:=await process.poll()) is None:await asyncio.sleep(.05)
    return code

def configure_sandbox_waits():
    import modal
    if modal.__version__!='1.5.2':raise RuntimeError('Sandbox transport adapter requires pinned Modal 1.5.2')
    from modal.container_process import _ContainerProcess
    # SDK filesystem helpers await this internal method after mkdir/read/write.
    # No command is re-executed: poll observes the same exec ID and deadline.
    _ContainerProcess.wait=_poll_wait

def wait_process(process):
    while (code:=read_rpc(process.poll)) is None:time.sleep(.05)
    return code

def reset_rpc_channels():
    """Discard stale transports on the SDK event loop; retain client identity."""
    async def reset():
        from modal.client import _Client
        client=await _Client.from_env()
        client._connection_manager.close()
    from modal._utils.async_utils import synchronizer
    synchronizer.create_blocking(reset)()

def read_rpc(call):
    """Retry the observed closed-channel race, without starting another allocation.

    Only read-only/idempotent status operations use this wrapper. The original
    provider deadline and every failed allocation remain charged.
    """
    from grpclib.exceptions import StreamTerminatedError
    for attempt in range(4):
        try:return call()
        except (AttributeError,StreamTerminatedError) as exc:
            if (isinstance(exc,AttributeError) and "'Connection' object has no attribute '_transport'" not in str(exc)) or attempt==3:raise
            reset_rpc_channels()
            time.sleep(.2*(attempt+1))

def allocation_info(app_id,sandbox_id):
    """Pinned Modal 1.5.2 provider metadata; no guest-reported clock is trusted."""
    async def fetch():
        from modal.client import _Client
        from modal_proto import api_pb2
        client=await _Client.from_env()
        before=None
        while True:
            request=api_pb2.SandboxListRequest(app_id=app_id,environment_name=ENVIRONMENT,include_finished=True,
                **({'before_timestamp':before} if before is not None else {}))
            response=await client.stub.SandboxList(request)
            for item in response.sandboxes:
                if item.id==sandbox_id:
                    task=item.task_info
                    return {'provider_id':sandbox_id,'task_id':task.id,'created_at':item.created_at,
                        'started_at':task.started_at,'finished_at':task.finished_at,'gpu_type':task.gpu_type}
            if not response.sandboxes:return None
            before=response.sandboxes[-1].created_at
    # Use the SDK's own event loop: asyncio.run() would reuse a cached client and
    # gRPC channel belonging to the synchronizer's different loop.
    from modal._utils.async_utils import synchronizer
    return read_rpc(synchronizer.create_blocking(fetch))


def image_for(release):
 import modal
 release=Path(release)
 expected=json.loads((release/'release.json').read_text())['frozen_files']
 actual={str(p.relative_to(release/'frozen')) for p in (release/'frozen').rglob('*') if p.is_file()}
 if actual!=set(expected) or any(sha(release/'frozen'/f)!=h for f,h in expected.items()):raise ValueError('Immutable release source snapshot mutated')
 return (modal.Image.debian_slim(python_version='3.12').apt_install('curl','build-essential','git','ca-certificates')
  .pip_install('torch==2.9.1','numpy==2.3.5','gymnasium==1.2.2','maturin==1.10.2','pytest==9.0.2')
  .run_commands('curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.0')
  .env({'PATH':'/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin','PYTHONPATH':'/opt/task','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
  .add_local_dir(release/'frozen/vendor/engines/AutoTavern','/opt/task/vendor/engines/AutoTavern',copy=True)
  .run_commands('cd /opt/task/vendor/engines/AutoTavern && python -m pip install --no-build-isolation --editable .')
  .add_local_dir(release/'frozen/simulators/tavern','/opt/task/simulators/tavern',copy=True)
  .run_commands('chmod -R a-w /opt/task','mkdir -p /job /private','chmod 700 /private'))


def command(sb,args,timeout=220):
 p=sb.exec(*args,timeout=timeout);stdout=p.stdout.read();stderr=p.stderr.read();code=wait_process(p)
 if code:raise RuntimeError('Remote command failed: '+stdout[-2000:]+stderr[-4000:])
 return stdout

def upload(sb,source,dest):
 sb.filesystem.make_directory(dest)
 for p in Path(source).rglob('*'):
  if p.is_file():
   target=dest+'/'+str(p.relative_to(source));sb.filesystem.make_directory(str(Path(target).parent));sb.filesystem.copy_from_local(p,target)

def download(sb,source,dest,names):
 dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
 for name in names:(dest/name).write_bytes(sb.filesystem.read_bytes(source+'/'+name))


# Ends the trainer process (and anything wrapping it) inside a worker; prints how many processes it ended.
KILL_TRAINER='''import os,signal
ended=0
for entry in os.listdir('/proc'):
 if entry.isdigit() and int(entry)!=os.getpid():
  try:
   if any(arg.endswith(b'/train_entry.py') for arg in open('/proc/'+entry+'/cmdline','rb').read().split(b'\\0')):
    os.kill(int(entry),signal.SIGKILL);ended+=1
  except OSError:pass
print(ended)'''


class ModalProvider:
 """Every cloud call the controller makes. Hardware and isolation are the mixed-v2 contract's:
 one L40S with four CPU cores and 16 GiB for training, the same CPU/RAM without a GPU for evaluation,
 no network, no secrets. The only volume is the session's private hand-off Volume (harness.handoff) at
 /handoff, made root-only before any researcher or candidate code runs; workers publish their results there."""
 def __init__(self,release,app):
  configure_sandbox_waits()
  self.release=Path(release);self.app=app;self.app_id=app.app_id;self.image=None;self.lock=threading.Lock()
 def ready(self):
  """Build the frozen-source image once. This is not worker time and happens before any reservation."""
  with self.lock:
   if self.image is None:
    image=image_for(self.release);image.build(self.app);self.image=image
 def volume(self,root):
  """The private hand-off Volume bound to `root`, created now: before any reservation, never inside worker time."""
  from harness.handoff import SessionVolume
  session=SessionVolume('tavern',root);session.volume(create=True).hydrate()
  return session
 def create(self,gpu,timeout,volume=None):
  import modal
  self.ready()
  options=dict(app=self.app,image=self.image,cpu=(4,4),memory=(16384,16384),timeout=int(timeout),block_network=True,secrets=[],
   volumes=volume.mounts() if volume is not None else {},include_oidc_identity_token=False,environment_name=ENVIRONMENT)
  if gpu:return modal.Sandbox.create('/bin/sleep','10000',gpu=hardware.GPU,**options)
  return modal.Sandbox.create('/bin/sleep','86400',**options)
 def identity(self,sb):return sb.object_id
 def command(self,sb,args,timeout=220):return command(sb,args,timeout)
 def upload(self,sb,source,dest):upload(sb,source,dest)
 def copy(self,sb,local,remote):sb.filesystem.copy_from_local(local,remote)
 def download(self,sb,source,dest,names):download(sb,source,dest,names)
 def read(self,sb,remote):return sb.filesystem.read_bytes(remote)
 def kill_trainer(self,sb):
  """End the trainer process and keep the worker and what the trainer exported. Returns how many processes ended."""
  return int(command(sb,['python','-c',KILL_TRAINER],timeout=20).strip() or 0)
 def terminate(self,sb):sb.terminate()
 def terminate_id(self,sandbox_id):
  """End a recorded worker by its identity alone (after a host restart no sandbox object is left)."""
  import modal
  modal.Sandbox.from_id(sandbox_id).terminate()
 def allocation(self,app_id,sandbox_id):return allocation_info(app_id,sandbox_id)
