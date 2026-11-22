"""Dou Dizhu Modal workers: training, practice and private grading sandboxes."""
from harness.policy import ENVIRONMENT
from harness import handoff
from harness import hardware
from harness import workers
from pathlib import Path
import hashlib
import json
import math
import re
import tempfile
import threading
import time
import traceback

from simulators.doudizhu.autoresearch.contract import HERE,EDITABLE,DEFAULT,write_json
from simulators.doudizhu.autoresearch.bundle import verify,archive_bundle,unpack_bundle

PYTHON='/opt/exact-python/cpython-3.12.9-linux-x86_64-gnu/bin/python3.12'
STAGE='/job/handoff-stage'  # root-only, made by CHAIN after every unprivileged process has ended
KILL_RESEARCHER=('/bin/sh','-c','pkill -KILL -u 65534 || true')
# Parallel jobs (dollar mode) share only the session's hand-off Volume, each under its own job name; its binding
# is created once, so two first runs starting together cannot bind the session to two different Volumes.
VOLUME_BINDING=threading.Lock()

# A training worker's whole trusted sequence, run once as root inside the worker: python -c CHAIN SPEC.
# It trains (the supervisor drops privileges for the researcher's trainer), ends every unprivileged process,
# exports, keeps bounded tails of the logs (regular files only, no symlink following) and publishes all of it
# to the session's private hand-off Volume. It runs in the worker itself, so a finished run is published even
# when the host is slow or unreachable at that moment; the host only waits for it, terminates the worker and
# collects from the Volume afterwards. The last stdout line is the publish manifest. Every step is bounded by
# the chain's own budget, keeping `publish_seconds` for the logs and the publish whatever the export does.
CHAIN=r'''import json,os,stat,subprocess,sys,time
spec=json.loads(sys.argv[1]);begun=time.monotonic();stage=spec['stage']
left=lambda:spec['budget']-(time.monotonic()-begun)
def run(argv,limit,**options):
    if limit<1:return 'no time'
    try:return subprocess.run(argv,timeout=limit,**options).returncode
    except subprocess.TimeoutExpired:return 'timeout'
    except OSError as exc:return 'error: '+str(exc)
def tail(path,limit,whole_lines=True):
    try:fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    except OSError:return None
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):return None
        stream.seek(max(0,info.st_size-limit));data=stream.read(limit)
    return data[data.find(b'\n')+1:] if whole_lines and info.st_size>limit else data
def keep(name,data):
    with open(os.path.join(stage,name),'wb') as stream:stream.write(data)
    files[name]=os.path.join(stage,name)
os.makedirs(stage,mode=0o700,exist_ok=True);os.chmod(stage,0o700);files={}
record={'train':run(spec['train'],min(spec['train_timeout'],left()-spec['publish_seconds']))}
record['kill']=run(spec['kill'],min(5,left()-spec['publish_seconds']))
with open(os.path.join(stage,'export.out'),'wb') as out,open(os.path.join(stage,'export.err'),'wb') as err:
    record['export']=run(spec['export'],left()-spec['publish_seconds'],stdout=out,stderr=err)
for name,path in spec['logs'].items():
    data=tail(path,spec['tail_bytes'])
    if data is not None:keep(name,data)
if record['export']==0:files['bundle.tar']=spec['archive']
else:keep('export-error.log',tail(os.path.join(stage,'export.err'),spec['error_bytes'],False) or b'')
record['elapsed_seconds']=round(time.monotonic()-begun,3);keep('chain.json',json.dumps(record,sort_keys=True).encode())
published=subprocess.run(spec['publish']+[name+'='+path for name,path in files.items()],timeout=max(1,left()),stdout=subprocess.PIPE)
sys.stdout.write(published.stdout.decode(errors='replace'));sys.stdout.flush()
raise SystemExit(published.returncode)'''


def image_for(release):
    import modal
    release=Path(release)
    from simulators.doudizhu.autoresearch.prepare import audit_release
    audit_release(release)
    return (modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu24.04',add_python='3.12')
        .apt_install('git','g++','util-linux','procps')
        .pip_install('uv==0.9.25')
        .add_local_file(HERE.parent/'requirements-universal-lock.txt','/tmp/doudizhu-requirements.txt',copy=True)
        .run_commands('uv python install 3.12.9 --install-dir /opt/exact-python',
            f'uv pip install --python {PYTHON} --require-hashes -r /tmp/doudizhu-requirements.txt')
        .add_local_dir(release/'frozen','/opt/dd',copy=True,ignore=['**/__pycache__/**'])
        .env({'PYTHONPATH':'/opt/dd','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
        .run_commands('mkdir -p /job/input /job/output /private /opt/dd/evidence',
            'chmod 700 /private', 'git config --system --add safe.directory /opt/dd/candidates/DouZero',
            PYTHON+' -m integrations.doudizhu.autoresearch.build_cuda',
            'chmod -R a-w /opt/dd/integrations /opt/dd/candidates')
        .workdir('/job'))

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
    return synchronizer.create_blocking(fetch)()

def terminate_sandbox(sandbox_id):
    import modal
    modal.Sandbox.from_id(sandbox_id).terminate(wait=True)

def handoff_job(name):
    """A run's job name in the hand-off Volume: readable, unique per run name and always a valid hand-off name."""
    clean=re.sub(r'[^A-Za-z0-9._-]+','-',name).strip('.-_')[:48]
    return 'train-'+(clean+'-' if clean else '')+hashlib.sha256(name.encode()).hexdigest()[:12]

def note(notes,where,exc):
    if len(notes)<20:notes.append(where+': '+type(exc).__name__+': '+str(exc)[:500])

class ModalProvider:
    # Seconds kept after a training run's learner is stopped for the root-side export and its hand-off. A 1-hour
    # final with a CPU-busy recipe ran out of a flat 15 s (its export never started), and another final lost its
    # export to the sandbox's own timeout, so the reserve now grows with the grant: 15 s for short research runs,
    # up to 60 s for long ones. The hand-off is the in-worker publish to the session's Volume (CHAIN), no longer a
    # host download inside the allocation; its last PUBLISH_SECONDS are kept for the log tails and the publish.
    MIN_EXPORT_RESERVE=15
    EXPORT_RESERVE_FRACTION=0.05
    PUBLISH_SECONDS=5
    PROGRESS_POLL_SECONDS=5.0
    TAIL_BYTES=1000000
    TAIL_TIMEOUT=30
    ERROR_BYTES=10000
    STOP_REPEAT_SECONDS=3.0  # a stop re-ends unprivileged processes until the chain has moved on
    PUBLISHED_CHECKS=3       # quick manifest reads while the worker lives, when the chain's publish reply is missing
    PUBLISHED_WAIT=2.0
    COLLECT_ROUNDS=20        # `SessionVolume.collect` calls (each with its own retries) after the worker has ended
    COLLECT_ATTEMPTS=6
    COLLECT_WAIT=5.0

    def __init__(self,release,app,export_reserve_seconds=60):
        self.release=Path(release);self.app=app;self.image=image_for(release)
        self.export_reserve_seconds=export_reserve_seconds

    def export_reserve(self,grant):
        """Export reserve for a run granted `grant` seconds: 5 % of it, at least 15 s, at most the configured cap."""
        cap=max(self.MIN_EXPORT_RESERVE,float(self.export_reserve_seconds))
        return min(cap,max(self.MIN_EXPORT_RESERVE,self.EXPORT_RESERVE_FRACTION*float(grant)))

    def execute(self,ledger,job,stop=None):
        import modal
        name=job['name'];payload=json.loads(job['payload']);grant=job['grant']
        sb=None;started=None;artifact=None;identity=None;error=None;status='failed';provider_receipt={}
        unstarted=False;never=None;halted=False;volume=None;relay=None;termination=None
        target=ledger.root/'jobs'/name;target.mkdir(parents=True,exist_ok=False)
        write_json(target/'request.json',payload,exclusive=True)
        try:
            if payload.get('kind')!='practice':
                # A training run's results come back through this backend root's private Volume (root-only /handoff).
                with VOLUME_BINDING:volume=handoff.SessionVolume('doudizhu',ledger.root);mounts=volume.mounts()
                relay=dict(handoff=None,volume=volume.name,job=handoff_job(name),reason=None,chain=False,manifest=None,
                    chain_error=None,files=None,host_errors=[])
                write_json(target/'handoff.json',{'volume':volume.name,'job':relay['job'],'mount':handoff.MOUNT},exclusive=True)
            sb=modal.Sandbox.create('/bin/sleep','86400',app=self.app,image=self.image,gpu=hardware.GPU,
                cpu=(4,4),memory=(16384,16384),timeout=math.ceil(grant),block_network=True,
                include_oidc_identity_token=False,secrets=[],volumes=mounts if volume else {},
                environment_name=ENVIRONMENT,tags={'dd_job':name})
            write_json(target/'provider.json',{'sandbox_id':sb.object_id,'app_id':self.app.app_id},exclusive=True)
            # Queued time (created, no GPU yet) is never charged; wait for the provider's start or a stop.
            info=workers.await_start(lambda:allocation_info(self.app.app_id,sb.object_id),stop=stop)
            if info is None:
                unstarted=True
                raise RuntimeError(('Stopped while' if stop and stop() else 'Gave up after '+str(int(workers.START_PATIENCE_SECONDS))+' s')
                    +' waiting in the provider queue for a GPU')
            started=info['started_at'];ledger.start(name,sb.object_id,started)
            if not hardware.matches(info['gpu_type']):raise RuntimeError('Requested '+hardware.GPU+' but provider reported '+info['gpu_type'])
            deadline=started+grant
            if payload.get('kind')=='practice':
                # Practice returns one small score file over exec; it carries no artifact and mounts no Volume.
                self.stage_evaluation(sb,payload['bundle'],payload['panel'],payload['seeds'],{},True)
                remaining=deadline-time.time()-5
                if remaining<1:raise RuntimeError('Practice setup exhausted allocation')
                proc=sb.exec(PYTHON,'-m','integrations.doudizhu.autoresearch.evaluate_entry','--request','/private/evaluation.json',timeout=max(1,int(remaining)))
                while proc.poll() is None and time.time()<deadline-3:
                    ledger.milestones()
                    if stop and stop():break
                    time.sleep(.5)
                if proc.poll() is None:raise RuntimeError('Practice allocation deadline; no partial score awarded')
                if proc.returncode!=0:raise RuntimeError('Practice evaluation failed: '+proc.stderr.read()[-1000:])
                write_json(target/'practice.json',json.loads(sb.filesystem.read_text('/job/evaluation-public.json')))
                status='complete'
                return {'practice_complete':True}  # finally still charges and closes the allocation
            for filename in EDITABLE:sb.filesystem.write_text(payload['code'][filename],'/job/input/'+filename)
            sb.filesystem.write_text(json.dumps(payload['config']),'/job/input/config.json')
            sb.filesystem.write_text(json.dumps({'seed':payload['seed']}),'/job/input/request.json')
            prepare=sb.exec(*handoff.prepare_command(PYTHON),timeout=30)
            if prepare.wait()!=0:raise RuntimeError('Could not make the hand-off mount root-only: '+prepare.stderr.read()[-1000:])
            reserve=self.export_reserve(grant)
            remaining=deadline-time.time()-reserve
            if remaining<2:raise RuntimeError('Allocation consumed by setup before the learner could start')
            # Train, end every unprivileged process, export and publish: one root sequence inside the worker.
            spec=self.chain_spec(relay['job'],remaining,deadline-time.time()-3)
            proc=sb.exec(PYTHON,'-c',CHAIN,json.dumps(spec),timeout=max(1,int(deadline-time.time()-2)))
            relay['chain']=True
            # Progress is mirrored by its own thread for live status only; the run's final logs come from the Volume.
            mirroring=threading.Event()
            threading.Thread(target=self.mirror_progress,args=(sb,target,mirroring),daemon=True).start()
            try:halted=self.await_chain(proc,sb,deadline,ledger,stop,relay['host_errors'])
            finally:mirroring.set()
            relay['manifest']=self.chain_manifest(proc,relay)
            if relay['manifest'] is None:
                # No publish reply: ask the Volume; if nothing was published, read the worker while it still lives.
                relay['manifest'],reason=self.published(volume,relay['job'])
                if relay['manifest'] is None:
                    relay.update(handoff='live',reason=reason)
                    self.live_read(sb,target)
        except BaseException as exc:
            status='failed'
            error=type(exc).__name__+': '+str(exc)
            (target/'host-exception.log').write_text(traceback.format_exc())
        finally:
            host_error=error
            if sb is not None:
                failures=[]
                def terminate():
                    try:sb.terminate(wait=True)
                    except Exception as exc:failures.append(exc);raise
                final=workers.abandon(terminate,lambda:allocation_info(self.app.app_id,sb.object_id))
                if failures:termination='termination: '+str(failures[0]);error=(error or '')+'; '+termination
                if final.get('started_at') and final.get('finished_at'):provider_receipt=final
                elif started is None and final.get('provider_id') and workers.never_started(final):
                    # Created but never started: no GPU was allocated, so nothing is charged and nothing stays held.
                    stopped=bool(unstarted and stop and stop())
                    reason=workers.NEVER_STARTED+'. '+(error or 'Worker setup failed before the provider started it')
                    ledger.abandon_unstarted(name,final,reason,stopped=stopped)
                    status='stopped' if stopped else 'failed';provider_receipt=final
                    never=workers.NeverStarted(final,reason)
            if relay is not None and relay['chain'] and never is None:
                # The worker has ended: nothing below is charged, so the Volume is read without a deadline.
                try:
                    artifact,identity=self.settle_training(volume,relay,target)
                    if host_error:relay['host_errors'].append('host: '+host_error)  # the worker finished on its own
                    status='stopped' if halted else 'complete';error=('; '+termination) if termination else None
                except BaseException as exc:
                    artifact=identity=None;status='failed'
                    if not host_error:error=type(exc).__name__+': '+str(exc)+(('; '+termination) if termination else '')
                    with (target/'host-exception.log').open('a') as stream:stream.write(traceback.format_exc())
            if never is None:
                if provider_receipt:
                    if started is None:ledger.start(name,sb.object_id,provider_receipt['started_at'])
                    ledger.finish(name,provider_receipt['finished_at'],status,artifact,identity,error)
                else:ledger.uncertain(name,error or 'Provider allocation end is unavailable; reservation retained')
            write_json(target/'allocation.json',{'provider':provider_receipt,'status':status,'error':error,
                'budget_scope':DEFAULT['allocation_rule'],'artifact_sha256':identity,'guest_clock_trusted':False,
                'never_started':never is not None,'handoff':relay and relay['handoff'],
                'handoff_detail':relay and {k:v for k,v in relay.items() if k not in ('handoff','chain','manifest')}})
        if never is not None:raise never
        return next(x for x in ledger.jobs() if x['name']==name)

    def chain_spec(self,job,seconds,budget):
        """CHAIN's arguments: train for `seconds`, then export and publish `job`, all within `budget` seconds."""
        return {'train':[PYTHON,'-m','integrations.doudizhu.autoresearch.train_entry','--seconds',str(seconds)],
            'train_timeout':seconds+2,'kill':list(KILL_RESEARCHER),
            'export':[PYTHON,'-m','integrations.doudizhu.autoresearch.export_entry'],'archive':'/job/export.tar',
            'logs':{'trainer.log':'/job/trainer.log','progress.jsonl':'/job/output/progress.jsonl','supervisor.json':'/job/supervisor.json'},
            'stage':STAGE,'budget':budget,'publish_seconds':self.PUBLISH_SECONDS,'tail_bytes':self.TAIL_BYTES,
            'error_bytes':self.ERROR_BYTES,'publish':handoff.publish_command(job,{},PYTHON)}

    def await_chain(self,proc,sb,deadline,ledger,stop,notes):
        """Wait for the worker's chain until the allocation deadline; True when a stop was requested.

        A failed poll (a slow or broken host link) is noted and retried: the worker trains, exports and publishes
        on its own. A stop ends every unprivileged process, repeated until the chain has moved on (a trainer may
        start just after the first kill); the chain then exports the last network and publishes it.
        """
        halted=False;killed=None
        while time.time()<deadline-1:
            try:
                if proc.poll() is not None:break
            except Exception as exc:note(notes,'poll',exc)
            ledger.milestones()
            if stop and stop():
                halted=True
                if killed is None or time.time()-killed>=self.STOP_REPEAT_SECONDS:
                    killed=time.time()
                    try:sb.exec(*KILL_RESEARCHER,timeout=5).wait()
                    except Exception as exc:note(notes,'stop',exc)
            time.sleep(.5)
        return halted

    @staticmethod
    def chain_manifest(proc,relay):
        """The publish manifest the chain printed last; None while it runs, when it failed or its reply was lost."""
        try:
            if proc.poll() is None:return None
            if proc.returncode!=0:
                relay['chain_error']='exit '+str(proc.returncode)+': '+proc.stderr.read()[-2000:]
                return None
            lines=[line for line in proc.stdout.read().splitlines() if line.strip()]
            manifest=json.loads(lines[-1]) if lines else None
        except Exception as exc:
            note(relay['host_errors'],'publish reply',exc);return None
        return manifest if isinstance(manifest,dict) and manifest else None

    def published(self,volume,job):
        """(manifest, None) when the Volume holds the job, else (None, why): a few quick reads while the worker lives."""
        reason=None
        for attempt in range(self.PUBLISHED_CHECKS):
            try:return volume.manifest(job),None
            except Exception as exc:reason=type(exc).__name__+': '+str(exc)
            if attempt+1<self.PUBLISHED_CHECKS:time.sleep(self.PUBLISHED_WAIT)
        return None,'No publish reply from the worker and nothing in the hand-off Volume ('+reason+')'

    def collect_published(self,volume,job,target,manifest):
        """Fetch a published run from the Volume into its job folder, every byte checked against its manifest.

        The worker has ended, so this has no deadline: a broken read is retried for COLLECT_ROUNDS collects. An
        unpublished job is retried only when the worker's manifest is in hand (a commit still landing); a
        verification failure is final. The fetched files must also match the manifest the worker reported.
        """
        for attempt in range(self.COLLECT_ROUNDS):
            try:
                files=volume.collect(job,target,attempts=self.COLLECT_ATTEMPTS,wait=self.COLLECT_WAIT);break
            except ValueError:raise
            except handoff.NotPublished:
                if manifest is None or attempt+1>=self.COLLECT_ROUNDS:raise
            except Exception:
                if attempt+1>=self.COLLECT_ROUNDS:raise
        if manifest is not None and {k:(int(v['bytes']),v['sha256']) for k,v in files.items()}!={k:(int(v['bytes']),v['sha256']) for k,v in manifest.items()}:
            raise ValueError('Hand-off Volume differs from the manifest the worker published')
        return files

    def settle_training(self,volume,relay,target):
        """After the worker has ended: the run's bundle from the Volume (or the live read), verified as strictly as ever."""
        if not (relay['handoff']=='live' and (target/'bundle.tar').is_file()):
            files=self.collect_published(volume,relay['job'],target,relay['manifest'])
            relay['files']={k:{'bytes':v['bytes'],'sha256':v['sha256']} for k,v in files.items()}
            if relay['handoff']=='live':relay['reason']+='; the live read failed and the Volume had the run after the worker ended'
            relay['handoff']='volume'
        if not (target/'bundle.tar').is_file():
            raise RuntimeError('Trainer did not leave a valid policy package; export diagnostics retained')
        artifact=target/'bundle';unpack_bundle(target/'bundle.tar',artifact)
        return artifact,verify(artifact,check_tensors=False)['bundle_sha256']

    def live_read(self,sb,target):
        """The former path, used only when nothing was published while the worker lives: read its export and logs."""
        try:
            try:record=json.loads(sb.filesystem.read_text(STAGE+'/chain.json'))
            except Exception:record={}
            if record.get('export',0)!=0:
                read=sb.exec('tail','-c',str(self.ERROR_BYTES),STAGE+'/export.err',timeout=self.TAIL_TIMEOUT)
                (target/'export-error.log').write_text(read.stdout.read())
                raise RuntimeError('Trainer did not leave a valid policy package; export diagnostics retained')
            sb.filesystem.copy_to_local('/job/export.tar',target/'bundle.tar')
        finally:
            self.collect_diagnostics(sb,target)

    @classmethod
    def tail(cls,sb,remote):
        """The newest TAIL_BYTES of a worker file (whole lines), read by a bounded `tail` with its own exec timeout.

        Whole-file reads of a 3-hour run's logs (14-21 MB) stalled for minutes; a tail stays as small as the
        1-hour logs that always read promptly. None when the read fails or the file is absent.
        """
        read=sb.exec('tail','-c',str(cls.TAIL_BYTES),remote,timeout=cls.TAIL_TIMEOUT)
        text=read.stdout.read()
        if read.wait()!=0:return None
        return text[text.find('\n')+1:] if len(text.encode())>=cls.TAIL_BYTES else text

    def mirror_progress(self,sb,target,done):
        """Copy the tail of the trainer's progress log every poll until `done` is set; a failed read skips a copy."""
        while not done.wait(self.PROGRESS_POLL_SECONDS):
            try:
                text=self.tail(sb,'/job/output/progress.jsonl')
                if text is not None and not done.is_set():(target/'progress.jsonl').write_text(text)
            except Exception:pass

    @classmethod
    def collect_diagnostics(cls,sb,target):
        """Best effort, after the export: a sandbox at its timeout or gone must not cost the trained bundle."""
        try:(target/'supervisor.json').write_text(sb.filesystem.read_text('/job/supervisor.json'))
        except Exception:pass
        for remote,local in (('/job/trainer.log','trainer.log'),('/job/output/progress.jsonl','progress.jsonl')):
            try:
                text=cls.tail(sb,remote)
                if text is not None:(target/local).write_text(text)
            except Exception:pass

    def recover_jobs(self,ledger):
        """Recover only from external metadata; a disconnected worker keeps its hold.

        A lost host cannot promise that root-side export completed. Such allocations
        are conservatively failed, even if a learner checkpoint may exist remotely.
        Existing valid submissions are never replaced by a recovered unsubmitted run.
        """
        recovered=[]
        for job in ledger.jobs():
            if job['status'] not in ('queued','running','uncertain'):continue
            if self.recover(ledger,job):recovered.append(job['name'])
        return recovered

    def recover(self,ledger,job,stopped=False):
        """Settle one unresolved lease from its recorded worker; True when it is no longer active.

        `running` keeps the existing rule: reconciled once the provider reports its end, otherwise held as
        uncertain. A queued or uncertain job has no thread driving its worker, so the worker is terminated
        now: never started -> failed (stopped when `stopped`) with 0 charged; started -> the exact
        start/finish lease is charged.
        """
        path=ledger.root/'jobs'/job['name']/'provider.json'
        if not path.exists():
            ledger.uncertain(job['name'],'No authoritative creation receipt after controller interruption')
            return False
        meta=json.loads(path.read_text())
        read=lambda:allocation_info(meta['app_id'],meta['sandbox_id'])
        info=read()
        if info and job['status'] in ('queued','uncertain') and not info.get('finished_at'):
            info=workers.abandon(lambda:terminate_sandbox(meta['sandbox_id']),read)
            if not job.get('started') and info.get('provider_id') and workers.never_started(info):
                reason=workers.NEVER_STARTED+(' (stopped while unresolved)' if stopped else '')
                ledger.abandon_unstarted(job['name'],info,reason,stopped=stopped)
                write_json(path.parent/('recovered-'+str(time.time_ns())+'.json'),dict(info,never_started=True,charged_seconds=0),exclusive=True)
                return True
        if info and info.get('started_at') and info.get('finished_at'):
            ledger.reconcile(job['name'],meta['sandbox_id'],info['started_at'],info['finished_at'],
                'Controller recovery: charged exact completed provider lease; unexported work is failed')
            write_json(path.parent/('recovered-'+str(time.time_ns())+'.json'),info,exclusive=True)
            return True
        ledger.uncertain(job['name'],'Controller recovery awaits authoritative provider end; hold retained')
        return False

    def stage_evaluation(self,sb,bundle,panel,seeds,anchors,development):
        for label,path in [('candidate',bundle)]+[(p['name'],p['bundle']) for p in panel if p['kind']=='checkpoint']:
            remote='/private/'+label
            with tempfile.TemporaryDirectory(prefix='dd-transport-') as temporary:
                archive=Path(temporary)/'bundle.tar';archive_bundle(path,archive)
                sb.filesystem.copy_from_local(archive,remote+'.tar')
        remote_panel=[dict(p,bundle='/private/'+p['name']) if p['kind']=='checkpoint' else p for p in panel]
        request=dict(seeds,candidate='/private/candidate',panel=remote_panel,anchors=anchors or {},development=development)
        sb.filesystem.write_text(json.dumps(request),'/private/evaluation.json')

    def evaluate(self,bundle,panel,seeds,anchors=None,development=True,timeout=3600):
        """Separate host grading allocation, never mounted into a researcher box."""
        import modal
        sb=None;result=None;host_start=time.time()
        try:
            sb=modal.Sandbox.create('/bin/sleep','86400',app=self.app,image=self.image,cpu=(4,4),memory=(16384,16384),
                timeout=timeout,block_network=True,secrets=[],volumes={},include_oidc_identity_token=False,environment_name=ENVIRONMENT)
            self.stage_evaluation(sb,bundle,panel,seeds,anchors or {},development)
            # /private is root-only. Candidate code runs as uid65534 and receives
            # only its own three public bundle files in its scratch directory.
            proc=sb.exec(PYTHON,'-m','integrations.doudizhu.autoresearch.evaluate_entry','--request','/private/evaluation.json',timeout=timeout-10)
            if proc.wait()!=0:raise RuntimeError('Isolated evaluation failed: '+proc.stderr.read()[-1500:])
            result={'public':json.loads(sb.filesystem.read_text('/job/evaluation-public.json')),
                'private':json.loads(sb.filesystem.read_text('/private/evaluation-private.json'))}
            return result
        finally:
            if sb is not None:
                sb.terminate(wait=True)
                info=allocation_info(self.app.app_id,sb.object_id)
                if result is not None:result['allocation']={'gpu_count':0,'cpu_cores':4,'memory_gib':16,
                    'provider':info,'host_elapsed_seconds':time.time()-host_start,'separate_from_gpu_budget':True}
