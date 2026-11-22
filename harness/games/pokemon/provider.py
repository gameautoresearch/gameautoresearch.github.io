"""Host side of the native workers; a full-lease reservation precedes every creation.

Each job is one finite, network-blocked sandbox on the retained runtime image.
Inputs and results travel through the session's private Volume, so provider IDs
and artifacts survive a host disconnect. Charges come from the provider's own
start/stop timestamps, never from the host clock. Jobs that run at once (dollar
mode) share only the session's app, Volume and its verified read-only `_shared`
inputs; each has its own operation folder on the host and on the Volume.
"""
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import threading
import time
import uuid

from harness.policy import ENVIRONMENT
from harness import hardware, workers
from simulators.pokemon.autoresearch.contract import sha, identity
from simulators.pokemon.autoresearch.runtime.bootstrap import ENTRY, IMAGE
from simulators.pokemon.autoresearch.runtime.worker import STOP_REQUEST

MAX_ARTIFACT_BYTES=512*1024*1024
MAX_ARTIFACT_TOTAL_BYTES=2*1024*1024*1024
MAX_ARTIFACT_MEMBERS=4096
MAX_ARCHIVE_BYTES=MAX_ARTIFACT_TOTAL_BYTES+16*1024*1024
# A running trainer's log is followed only while it stays small enough to stream on each look.
MAX_LIVE_LOG_BYTES=64*1024*1024
# The native contract's fixed worker profile.
GPU=hardware.GPU
CPU_CORES=4
MEMORY_MIB=16384
SHAPE=dict(gpu=1,cpu_cores=CPU_CORES,memory_mib=MEMORY_MIB)  # every training run (research or final)
# CPU-only jobs (public practice, private grading, tournaments) play games on the CPU: one arena uses ~1.4 cores and
# < 1 GiB (measured 2026-11-17). They reserve this much and keep the contract shape as their hard cap; Modal bills
# max(reservation, use). Practice is charged to the research ledger per worker second exactly as before.
CPU_JOB_REQUEST_CORES=1
CPU_JOB_REQUEST_MIB=2048
SCOPE='researcher'  # binds a run's sealed environment state to this session's own runs
MOUNT='/campaign'   # where a worker sees the session Volume; the bootstrap makes it and each job folder root-only


# The session's app and Volume are created on first use; jobs launched at once must not race to create them.
CREATING=threading.Lock()


STOPPED_QUEUED='Stopped while queued for a GPU: the provider never started this worker; it was terminated and charged 0 seconds'


class Stopped(RuntimeError):
    """The researcher or the host asked for this job to end before a worker was created."""


def private_write(path,value):
    """Atomic owner-only host journal entry, durable before the next side effect."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w')as stream:
        json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    descriptor=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def allocation_info(app_id,sandbox_id,environment=ENVIRONMENT):
    from modal.client import _Client
    from modal_proto import api_pb2
    from modal._utils.async_utils import synchronizer
    async def fetch():
        client=await _Client.from_env();before=None
        while True:
            response=await client.stub.SandboxList(api_pb2.SandboxListRequest(app_id=app_id,
                environment_name=environment,include_finished=True,
                **({'before_timestamp':before}if before is not None else{})))
            for item in response.sandboxes:
                if item.id==sandbox_id:
                    task=item.task_info
                    return dict(provider_id=item.id,task_id=task.id,created_at=item.created_at,
                        started_at=task.started_at,finished_at=task.finished_at,gpu_type=task.gpu_type,
                        environment=environment,initialization_failure_status=task.result.status,
                        initialization_failure=task.result.exception,
                        cpu_milli=item.resource_info.milli_cpu.value,
                        cpu_milli_max=item.resource_info.milli_cpu_max,
                        memory_mib=item.resource_info.memory_mb.value,
                        memory_mib_max=item.resource_info.memory_mb_max,
                        resource_gpu_type=getattr(item.resource_info,'gpu_type',None),
                        provider_timeout_seconds=item.timeout_secs)
            if not response.sandboxes:return None
            after=response.sandboxes[-1].created_at
            if before is not None and after>=before:raise RuntimeError('Provider inventory pagination did not advance')
            before=after
    return synchronizer.create_blocking(fetch)()


def typed_initialization_failure(receipt):
    """The provider's typed result for a worker that failed to initialize: its lifetime stays charged."""
    try:from modal_proto import api_pb2
    except ImportError:return False
    return (receipt or {}).get('initialization_failure_status')==api_pb2.GenericResult.GENERIC_STATUS_INIT_FAILURE


def shared_file(name):
    """Only trusted public/reference inputs are cacheable; candidate state is not."""
    rel=PurePosixPath(name)
    if not name or rel.is_absolute()or '..'in rel.parts or str(rel)!=name or '\\'in name:
        raise ValueError('Canonical input file name required')
    return (name in ('foul-play.tar.gz','public-sets.json','native-admission.json')
        or rel.parts[0]in ('wheels','runtime') or rel.parts[0].startswith('peer-'))


def runtime_files(release):
    """What a training worker needs to build and run the admitted native engine."""
    release=Path(release);runtime=release/'frozen/runtime'
    key=release/'private/env-state-key.bin'
    if key.is_symlink()or not key.is_file()or key.stat().st_size!=32:
        raise ValueError('Private persistent environment state key required')
    return [(runtime/'native-admission.json','native-admission.json'),(key,'env-state-key.bin'),
            *((runtime/(name+'.json'),'runtime/'+name+'.json')for name in('generators','constructors','chart'))]


def assessment_files(release):
    """The frozen evaluator request and its inputs, as paths inside a worker."""
    from simulators.pokemon.autoresearch.bundle import FILES
    root=Path(release)/'private/assessment'
    registry=json.loads((root/'registry.json').read_bytes());request=dict(registry['evaluator_request']);files=[];panel=[]
    for index,value in enumerate(registry['panel']):
        item=dict(value)
        if 'path'in item:
            # A frozen learned opponent is a checkpoint (opponent.json + weights.pt) or a sealed original bundle.
            names=('opponent.json','weights.pt')if item['kind']=='checkpoint'else(*FILES,'manifest.json')
            for name in names:files.append((root/item['path']/name,f'peer-{index}/'+name))
            item['path']=f'/pilotjob/input/peer-{index}'
        panel.append(item)
    for key,filename in(('source_archive','foul-play.tar.gz'),('public_sets','public-sets.json')):
        files.append((root/filename,filename));request[key]='/pilotjob/input/'+filename
    for name in request['wheel_sha256']:files.append((root/'wheels'/name,'wheels/'+name))
    request.update(wheels='/pilotjob/input/wheels',panel=panel,limit=4096,public_development=True)
    return request,files


def listed_size(volume,folder,name):
    """The size the Volume lists for folder/name, or None where it cannot say (no listing, or no such entry)."""
    listdir=getattr(volume,'listdir',None)
    if listdir is None:return None
    try:entries=list(listdir(folder))
    except Exception:return None
    for entry in entries:
        if PurePosixPath(str(entry.path)).name==name and isinstance(getattr(entry,'size',None),int):return entry.size
    return None


def unpack_results(archive,output):
    """Preflight every member before any extraction; never follow archive links."""
    archive=Path(archive);output=Path(output)
    if archive.stat().st_size>MAX_ARCHIVE_BYTES:raise ValueError('Oversized worker archive')
    with tarfile.open(archive,mode='r:')as tar:
        members=[];names=set();total=0
        for member in tar:
            path=Path(member.name)
            if (len(members)>=MAX_ARTIFACT_MEMBERS or not member.isfile()or path.is_absolute()
                    or not path.parts or '..'in path.parts or str(path) in names
                    or member.size<0 or member.size>MAX_ARTIFACT_BYTES):
                raise ValueError('Unsafe bounded worker artifact')
            names.add(str(path));members.append(member);total+=member.size
            if total>MAX_ARTIFACT_TOTAL_BYTES:raise ValueError('Worker artifacts exceed aggregate byte ceiling')
        output.mkdir(exist_ok=True)
        if output.is_symlink():raise ValueError('Symlinked worker output')
        for member in members:
            dest=output/Path(member.name)
            for parent in dest.parents:
                if parent==output.parent:break
                if parent.is_symlink():raise ValueError('Symlinked artifact parent')
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.is_symlink():raise ValueError('Symlinked worker artifact')
            incoming=tar.extractfile(member)
            temporary=dest.with_name('.'+dest.name+'.collect-partial')
            try:
                with incoming,temporary.open('wb')as stream:shutil.copyfileobj(incoming,stream,1024*1024)
                if temporary.stat().st_size!=member.size:raise ValueError('Incomplete worker artifact')
                if dest.exists():
                    if sha(dest)!=sha(temporary):raise ValueError('Artifact changed after collection')
                else:os.replace(temporary,dest)
            finally:
                if temporary.exists():temporary.unlink()


class Provider:
    # How a created worker's start is awaited and an abandoned one is settled (harness.workers rules).
    start_patience=None        # seconds; None is workers.START_PATIENCE_SECONDS
    start_poll=2.0
    abandon_poll=1.0

    def __init__(self,release,root):
        from simulators.pokemon.autoresearch.prepare import audit
        from simulators.pokemon.autoresearch.runtime import sdk
        self.release=Path(release).resolve();self.root=Path(root).resolve()
        release_manifest=audit(self.release)
        self.manifest=json.loads((self.release/'frozen/source-manifest.json').read_bytes())
        if self.manifest['image_id']!=IMAGE:raise ValueError('Release was prepared for a different runtime image')
        # The host's bootstrap and sealing checks must be the ones this release's workers were frozen with.
        if release_manifest['protected_runtime_sha256']!=sdk.runtime_signature():
            raise ValueError('Runtime source changed since this release was prepared; prepare a new release')
        binding=self.root/'provider-binding.json'
        if binding.exists():self.binding=json.loads(binding.read_bytes())
        else:
            # One private app and Volume per session root: nothing is shared between sessions.
            label=''.join(c if c.isalnum()else'-'for c in (self.root.parent.name+'-'+self.root.name).lower())[:36].strip('-')
            name='pokemon-ar-'+label+'-'+identity(str(self.root))[:8]
            self.binding=dict(app=name,volume=name+'-private',environment=ENVIRONMENT,image_id=IMAGE,
                source_sha256=self.manifest['source_sha256'])
            private_write(binding,self.binding)
        if (self.binding['environment'],self.binding['image_id'],self.binding['source_sha256'])!=(ENVIRONMENT,IMAGE,self.manifest['source_sha256']):
            raise ValueError('Session was created with a different environment, image or release source')

    def job(self,operation):
        return self.root/'jobs'/operation

    def volume(self,create=False):
        import modal
        return modal.Volume.from_name(self.binding['volume'],create_if_missing=create,
            **(dict(version=2)if create else{}),environment_name=ENVIRONMENT)

    def stage_shared(self,volume,files):
        """Commit the release's source pack and shared assets once per session Volume.

        Receipts are tied to this exact volume/environment/path/hash, and every
        worker independently hashes cached objects before consuming them.
        """
        pack=self.release/'frozen/source-pack.tar'
        source=dict(path='_shared/source/'+self.manifest['source_sha256']+'/'+sha(pack)+'.tar',
            sha256=sha(pack),bytes=pack.stat().st_size)
        assets={};objects={source['path']:(pack,source)}
        for src,name in files:
            if not shared_file(name):continue
            src=Path(src)
            if src.is_symlink()or not src.is_file():raise ValueError('Regular shared input required')
            digest=sha(src);item=dict(path='_shared/assets/'+digest,sha256=digest,bytes=src.stat().st_size)
            assets[name]=item;objects[item['path']]=(src,item)
        cache=self.root/'transport/shared-ready';cache.mkdir(parents=True,exist_ok=True)
        with(self.root/'transport/shared-upload.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            pending=[];where=dict(volume=self.binding['volume'],environment=ENVIRONMENT)
            for path,(local,item)in objects.items():
                receipt=cache/(identity(dict(where,**item))+'.json')
                if receipt.exists():
                    if json.loads(receipt.read_bytes())!=dict(where,**item,ready=True):
                        raise ValueError('Shared upload receipt changed')
                else:pending.append((path,local,item,receipt))
            if pending:
                with volume.batch_upload()as upload:
                    for path,local,item,receipt in pending:upload.put_file(local,path)
                for path,local,item,receipt in pending:private_write(receipt,dict(where,**item,ready=True))
        return dict(source_pack=source,manifest_sha256=sha(self.release/'frozen/source-manifest.json'),shared_files=assets)

    def launch(self,ledger,operation,kind,spec,files,seconds,stopped=None):
        """Publish one reserved operation's inputs and create its worker.

        Until the provider is asked to create, every failure settles the
        reservation at a provable zero. From that point an interrupted creation
        is an unknown liability that keeps its full hold.
        """
        import modal
        target=self.job(operation);target.mkdir(parents=True,exist_ok=True)
        action=spec['action']
        try:
            with CREATING:volume=self.volume(create=True)
            layout=self.stage_shared(volume,files)
            spec=dict(spec,arm=SCOPE,allocation_seconds=seconds,allocation_margin_seconds=5 if action=='assess'else 30,
                original_campaign_deadline_unix=None,absolute_stop_unix=time.time()+seconds)
            private_write(target/'request.json',spec);private_write(target/'input-layout.json',layout)
            shutil.copyfile(self.release/'frozen/source-manifest.json',target/'source-manifest.json')
            private_write(target/'reservation.json',dict(operation=operation,kind=kind,seconds=seconds))
            with volume.batch_upload()as upload:
                for name in('request.json','source-manifest.json','input-layout.json'):
                    upload.put_file(target/name,operation+'/input/'+name)
                for src,name in files:
                    if not shared_file(name):upload.put_file(src,operation+'/input/'+name)
            with CREATING:app=modal.App.lookup(self.binding['app'],create_if_missing=True,environment_name=ENVIRONMENT)
            if stopped and stopped():raise Stopped('Stopped before a worker was created')
        except BaseException as exc:
            ledger.cancel_unattached(operation,'no provider create attempted: '+type(exc).__name__)
            private_write(target/'preparation-failure.json',dict(error_type=type(exc).__name__,provider_created=False))
            raise
        gpu=GPU if kind=='research'else None
        private_write(target/'create-attempt.json',dict(operation=operation,at_unix=time.time(),provider_creation_attempted=True))
        try:
            sandbox=modal.Sandbox.create('python','-c',ENTRY,app=app,image=modal.Image.from_id(IMAGE),
                gpu=gpu,cpu=(CPU_CORES if gpu else CPU_JOB_REQUEST_CORES,CPU_CORES),
                memory=(MEMORY_MIB if gpu else CPU_JOB_REQUEST_MIB,MEMORY_MIB),timeout=int(seconds)-(2 if action=='assess'else 20),
                block_network=True,secrets=[],include_oidc_identity_token=False,environment_name=ENVIRONMENT,
                env={'POKEMON_OPERATION_ID':operation},volumes={MOUNT:volume},
                tags={'role':'pokemon-research','operation':operation,'kind':kind})
            private_write(target/'provider.json',dict(operation=operation,sandbox_id=sandbox.object_id,app_id=app.app_id,
                gpu=gpu,cpu=CPU_CORES,memory_mib=MEMORY_MIB,image_id=IMAGE,environment=ENVIRONMENT))
            ledger.attach(operation,sandbox.object_id)
        except BaseException:
            ledger.unknown(operation,dict(reason='Provider create interrupted/rejected; reconcile exact operation, no retry'))
            raise
        # A created worker may sit in the provider's queue for a GPU. Wait for its start (queued time is not
        # charged), end the wait early on a stop or when it finished without ever starting.
        info=self.receipts(app.app_id,sandbox.object_id)
        if workers.await_start(info,patience=self.start_patience,poll=self.start_poll,
                stop=lambda:bool(stopped and stopped())or bool(info.last.get('finished_at')))is None:
            stop_requested=bool(stopped and stopped())
            receipt=workers.abandon(sandbox.terminate,info,poll=self.abandon_poll)
            private_write(target/'abandoned.json',dict(receipt=receipt,stop_requested=stop_requested,at_unix=time.time(),
                patience_seconds=workers.START_PATIENCE_SECONDS))
            self.settle_abandoned(ledger,operation,receipt,STOPPED_QUEUED if stop_requested else None)
            # It started while being abandoned (or failed to initialize): `collect` charges it from its receipt.
            return dict(operation=operation,provider=sandbox.object_id,seconds=seconds,abandoned=True)
        return dict(operation=operation,provider=sandbox.object_id,seconds=seconds)

    @staticmethod
    def receipts(app_id,sandbox_id):
        """A read of one worker's provider record; a failed read is no record, never a start. `.last` keeps the last read."""
        def info():
            try:value=allocation_info(app_id,sandbox_id)or{}
            except Exception:value={}
            info.last=value
            return value
        info.last={}
        return info

    def settle_abandoned(self,ledger,operation,receipt,reason=None):
        """After `workers.abandon`: a provably never-started worker settles at 0 and raises `workers.NeverStarted`.

        Returns quietly when it started (or failed to initialize), for the receipt-based settlement in `collect`.
        A receipt that cannot be read proves nothing: the full hold stays as an unknown allocation.
        """
        if not receipt.get('provider_id'):
            ledger.unknown(operation,dict(reason='Worker abandoned while queued; provider receipt unreadable'))
            raise RuntimeError('Worker abandoned while queued for a GPU; its provider record could not be read, so its hold stays until recovery')
        if workers.never_started(receipt)and not typed_initialization_failure(receipt):
            settled=ledger.settle_unstarted(operation,receipt,reason or workers.NEVER_STARTED)
            private_write(self.job(operation)/'provider-stopped.json',dict(settled,provider_verified_stopped=bool(receipt.get('finished_at'))))
            raise workers.NeverStarted(settled,reason or workers.NEVER_STARTED)

    def resolve(self,ledger,operation):
        """Resolve an unresolved allocation now from its recorded worker, which is terminated first.

        Never started: settled at 0 (`never_started`). Started: settled from its start/finish receipt by `collect`,
        without waiting out the grant. Without a recorded worker nothing can be proven and the hold stays.
        """
        import modal
        record=self.job(operation)/'provider.json'
        if not record.is_file():return dict(status='unknown',operation=operation,reason='No recorded worker')
        provider=json.loads(record.read_bytes())
        receipt=workers.abandon(lambda:modal.Sandbox.from_id(provider['sandbox_id']).terminate(),
            self.receipts(provider['app_id'],provider['sandbox_id']),poll=self.abandon_poll)
        private_write(self.job(operation)/'resolved.json',dict(receipt=receipt,at_unix=time.time()))
        try:self.settle_abandoned(ledger,operation,receipt)
        except workers.NeverStarted as exc:
            return dict(status='settled',operation=operation,successful=False,never_started=True,provider=exc.receipt)
        return self.collect(ledger,operation)

    def request_stop(self,operation):
        """Ask this session's own running trainer to end and be packaged; True once the request is in the worker.

        The request is one file in the job's root-only folder, where the trusted supervisor looks every few
        seconds: it ends the trainer and seals the last complete policy it wrote, and the worker then publishes
        and finishes by itself like any other. Nothing is terminated here; a False leaves that to the caller.
        """
        import modal
        record=self.job(operation)/'provider.json'
        if not record.is_file():return False
        sandbox=json.loads(record.read_bytes())['sandbox_id'];path=MOUNT+'/'+operation+'/'+STOP_REQUEST;error=None
        for attempt in range(3):
            try:
                modal.Sandbox.from_id(sandbox).filesystem.write_text('stop\n',path)
                private_write(self.job(operation)/'stop-request.json',dict(path=path,at_unix=time.time(),attempts=attempt+1))
                return True
            except Exception as exc:
                error=exc;time.sleep(.2*(attempt+1))
        private_write(self.job(operation)/'stop-request-error.json',dict(path=path,at_unix=time.time(),
            error_type=type(error).__name__,message=str(error)[:500]))
        return False

    def stop(self,operation):
        """End this session's own worker now; `collect` then settles it from the provider receipt."""
        import modal
        record=self.job(operation)/'provider.json'
        if record.is_file():modal.Sandbox.from_id(json.loads(record.read_bytes())['sandbox_id']).terminate()

    def peek(self,operation,limit=64*1024):
        """The end of a running trainer's own output, as last committed to the session Volume. Read-only.

        The supervisor writes the trainer's stdout and stderr straight to the
        operation's output mount, so this needs nothing from the worker. The
        Volume shows a running worker's writes only after its periodic
        background commit, so the text trails the trainer.
        """
        tail=bytearray();total=0
        for chunk in self.volume().read_file(operation+'/output/stage/trainer.log'):
            total+=len(chunk)
            if total>MAX_LIVE_LOG_BYTES:raise ValueError('The trainer has printed too much to follow while it runs')
            tail.extend(chunk);del tail[:-limit]
        return dict(bytes=total,cut=total>len(tail),text=bytes(tail).decode('utf-8',errors='replace'))

    def collect(self,ledger,operation):
        """One reconciliation pass: settle a stopped worker at its measured interval and fetch its archive."""
        target=self.job(operation);provider=json.loads((target/'provider.json').read_bytes())
        receipt=allocation_info(provider['app_id'],provider['sandbox_id'])
        if not receipt or not receipt.get('finished_at'):return dict(status='active-or-unknown',operation=operation,provider=receipt)
        row=ledger.job(operation)
        # A worker the provider finished without ever starting it (terminated while queued for a GPU) had no
        # allocation: harness.workers settles it at 0. A typed initialization failure stays charged below.
        if (not receipt.get('started_at')and not typed_initialization_failure(receipt)
                and(row['status']!='settled'or json.loads(row['receipt']or'{}').get('provider_never_started'))):
            settled=ledger.settle_unstarted(operation,receipt,workers.NEVER_STARTED)
            private_write(target/'provider-stopped.json',dict(settled,provider_verified_stopped=True))
            return dict(status='settled',operation=operation,successful=False,never_started=True,provider=receipt)
        if row['status']!='settled':
            if not receipt.get('started_at'):
                # Failed initialization has no task execution start. Only the
                # provider's typed result bounds the lifecycle by creation-to-finish.
                receipt.update(provider_verified_stopped=True,allocation_upper_bound_seconds=receipt['finished_at']-receipt['created_at'])
                ledger.settle(operation,receipt['allocation_upper_bound_seconds'],receipt)
                private_write(target/'provider-stopped.json',receipt)
                private_write(target/'initialization-failure.json',receipt)
                return dict(status='settled',operation=operation,successful=False,
                    initialization_failure=receipt['initialization_failure'],provider=receipt)
            if self.expects_gpu(operation,row)and GPU not in receipt['gpu_type'].upper():
                ledger.unknown(operation,dict(reason='Hardware differs',receipt=receipt));raise ValueError('Expected exact L40S worker')
            ledger.settle(operation,receipt['finished_at']-receipt['started_at'],dict(receipt,provider_verified_stopped=True,
                pricing_status='Public resource rate plus5% retained; invoice not yet reconciled'))
        private_write(target/'provider-stopped.json',dict(receipt,provider_verified_stopped=True))
        volume=self.volume()
        archive=target/'results.tar';partial=archive.with_suffix('.partial');collection_error=None
        try:
            # A read over a congested link can end early without an error (seen 2026-11-15): compare it with the
            # size the Volume lists and read again, up to three times, before calling the collection failed.
            listed=listed_size(volume,operation+'/output','results.tar')
            for attempt in range(3):
                total=0
                try:
                    with partial.open('wb')as stream:
                        for chunk in volume.read_file(operation+'/output/results.tar'):
                            total+=len(chunk)
                            if total>MAX_ARCHIVE_BYTES:raise OverflowError('Worker archive exceeds aggregate byte ceiling')
                            stream.write(chunk)
                except OverflowError:raise
                except Exception as exc:  # a read that breaks off partway is read again, like a silently short one
                    if attempt==2 or isinstance(exc,FileNotFoundError) or type(exc).__name__=='NotFoundError':raise
                    time.sleep(5*(attempt+1));continue
                if listed is None or total==listed:break
                if attempt==2:raise ValueError(f'Worker archive read {total} of {listed} bytes three times')
                time.sleep(5*(attempt+1))
            if archive.exists():
                if sha(archive)!=sha(partial):raise ValueError('Collected archive changed')
            else:os.replace(partial,archive)
        except Exception as exc:
            collection_error=type(exc).__name__
        finally:
            if partial.exists():partial.unlink()
        output=target/'output'
        if collection_error is None:
            try:unpack_results(archive,output)
            except Exception as exc:collection_error=type(exc).__name__
        if collection_error is not None:
            output.mkdir(exist_ok=True)
            from simulators.pokemon.autoresearch.runtime.recovery import collect_durable_checkpoint
            try:recovered=collect_durable_checkpoint(volume,operation,output,json.loads((target/'request.json').read_bytes()))
            except Exception as exc:recovered=dict(available=False,error_type=type(exc).__name__)
            private_write(target/'collection-failure.json',dict(error_type=collection_error,successful=False,recovery=recovered))
            return dict(status='settled',operation=operation,output=str(output),provider=receipt,successful=False,
                collection_failure=collection_error,checkpoint_recovery=recovered)
        if (output/'bootstrap-failure.json').exists():
            private_write(self.root/'ALLOCATION-BLOCKED.json',dict(operation=operation,
                reason='shared-input-admission-failed',candidate_executed=False,
                evidence=str(output/'bootstrap-error.log'),at_unix=time.time()))
            return dict(status='settled',operation=operation,output=str(output),provider=receipt,
                successful=False,shared_input_admission_failed=True)
        return dict(status='settled',operation=operation,output=str(output),provider=receipt,
            successful=(output/'completed.json').exists())

    def expects_gpu(self,operation,row):
        """Only a training run (launched as kind `research`) holds a GPU; the ledger kind of a practice is also
        `research` (it is charged to the research allocation), so the launch's own reservation decides."""
        path=self.job(operation)/'reservation.json'
        kind=json.loads(path.read_bytes()).get('kind')if path.is_file()else row['kind']
        return kind=='research'

    def verify_profile(self,operation,gpu):
        """Read-only check that a stopped worker really had the contract's resources: a GPU job's fixed shape, or a
        CPU job's reservation under the same cap."""
        target=self.job(operation);provider=json.loads((target/'provider.json').read_bytes())
        value=allocation_info(provider['app_id'],provider['sandbox_id'])
        if value is None:raise RuntimeError('Actual provider resource receipt absent')
        actual=str(value['resource_gpu_type']if value['resource_gpu_type']is not None else value['gpu_type']).upper()
        if (gpu is None and actual not in('','CPU','NONE'))or(gpu and gpu not in actual):raise ValueError('Provider GPU profile differs')
        cores,mib=(CPU_CORES,MEMORY_MIB)if gpu else(CPU_JOB_REQUEST_CORES,CPU_JOB_REQUEST_MIB)
        if (value['cpu_milli']!=cores*1000 or value['cpu_milli_max']!=CPU_CORES*1000
                or value['memory_mib']!=mib or value['memory_mib_max']!=MEMORY_MIB):
            raise ValueError('Provider fixed CPU/RAM differs from the contract profile')
        if not value['started_at']or not value['finished_at']:raise ValueError('Provider profile interval incomplete')
        private_write(target/'fixed-resource-profile.json',dict(passed=True,**value))
        return value
