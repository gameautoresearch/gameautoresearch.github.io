"""Trusted isolated trainer supervisor: deadline headroom and failed-stage preservation.

Editable Python never executes as the sealer. A trainer this supervisor ends
itself (at the hard stage deadline, or on a stop request) is packaged from the
last complete policy it wrote, by the same validation and sealing as a trainer
that finished; one that wrote none, or exits non-zero by itself, stays failed.
"""
import hashlib
import math
import os,shutil,signal,stat,subprocess,sys,tempfile,time,uuid
from contextlib import contextmanager
from pathlib import Path
from ..arena import drop_privileges
from .sdk import EDITABLE,write,sha,source_signature,runtime_signature,identity,validate_recipe
from .bundle import seal

MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
FINALIZATION_RESERVE_SECONDS = 180
CHECKPOINT_POLL_SECONDS = 2
RECOVERY_LIMITS = {'checkpoint.pt': MAX_CHECKPOINT_BYTES,
                   'progress.jsonl': 128 * 1024 * 1024,
                   'result.json': 128 * 1024 * 1024,
                   'graph.json': 2 * 1024 * 1024}


class HardDeadline(TimeoutError):
    """Own stage deadline, distinct from an outer worker alarm."""


class StopRequested(RuntimeError):
    """The host asked for this trainer to end now."""


# How a trainer this supervisor ended is flagged when what it had written seals: receipt key by termination reason.
EARLY = {'hard_deadline': 'cut_short', 'stop_requested': 'stopped_early'}
# What a trainer must leave, complete, for its policy to be validated and sealed.
STAGE_FILES = ('checkpoint.pt', 'result.json', 'progress.jsonl', 'graph.json')


def durable_write(path, value):
    write(path, value)
    with Path(path).open('rb') as stream:os.fsync(stream.fileno())
    descriptor=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def publish_checkpoint(output, inputs, out, config, deadline=None):
    """Mirror an atomic candidate save to the trusted durable output mount.

    Opaque bytes only; validation remains in the ordinary isolated resume path.
    An index commits last, so an interrupted copy leaves its predecessor usable.
    Keep two complete generations to bound publication/storage costs.
    """
    directory=None;descriptor=None;temporary=None
    root=out/'durable-checkpoints'
    try:
        directory=os.open(output/'train',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        descriptor=os.open('checkpoint.pt',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
        before=os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode)or before.st_nlink!=1 or not 0<before.st_size<=MAX_CHECKPOINT_BYTES:
            raise ValueError('Invalid periodic checkpoint')
        root.mkdir(mode=0o700,exist_ok=True)
        index_path=root/'latest.json'
        index=__import__('json').loads(index_path.read_bytes())if index_path.exists()else{}
        signature=[before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns]
        previous=index.get('generations',[])
        if previous and index.get('source_file_signature')==signature:
            return dict(checkpoint_sha256=previous[0],new_generation=False)
        temporary=root/('.pending-'+uuid.uuid4().hex);temporary.mkdir(mode=0o700)
        digest=hashlib.sha256();count=0
        with os.fdopen(descriptor,'rb')as incoming,(temporary/'checkpoint.pt').open('xb')as stream:
            descriptor=None
            while chunk:=incoming.read(1024*1024):
                if deadline is not None and time.monotonic()>=deadline:raise HardDeadline('Periodic copy deadline')
                count+=len(chunk)
                if count>before.st_size:raise ValueError('Checkpoint grew during periodic copy')
                stream.write(chunk);digest.update(chunk)
            after=os.fstat(incoming.fileno())
            # Atomic replacement unlinks the old inode while our descriptor
            # remains a complete valid save; unlink changes ctime, not bytes.
            if (count!=before.st_size or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns)
                    or(after.st_nlink!=0 and before.st_ctime_ns!=after.st_ctime_ns)):
                raise ValueError('Checkpoint changed during periodic copy')
            stream.flush();os.fsync(stream.fileno())
        key=digest.hexdigest()
        if previous and previous[0]==key:return dict(checkpoint_sha256=key,new_generation=False)
        provenance=temporary/'source';provenance.mkdir(mode=0o700)
        for name in (*EDITABLE,'config.json'):
            shutil.copyfile(inputs/name,provenance/name)
            with (provenance/name).open('rb')as stream:os.fsync(stream.fileno())
        fd=os.open(provenance,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        record=dict(schema='pokemon-failed-training-artifacts-v1',status='retained_unvalidated',
            eligible_for_submission=False,original_stage_successful=False,checkpoint_retained=True,
            files={'checkpoint.pt':dict(size_bytes=count,sha256=key)},errors={},
            source_sha256=source_signature(inputs),config_sha256=identity(config),runtime_sha256=runtime_signature(),
            capture_kind='durable_periodic',published_at_unix=time.time())
        durable_write(temporary/'manifest.json',record)
        if (root/key).exists():shutil.rmtree(temporary);temporary=None
        else:os.replace(temporary,root/key);temporary=None
        keep=[key]+[v for v in previous if v!=key][:1]
        durable_write(index_path,dict(schema='pokemon-durable-checkpoint-index-v1',generations=keep,source_file_signature=signature))
        for old in previous:
            if old not in keep and len(old)==64 and all(c in '0123456789abcdef'for c in old):shutil.rmtree(root/old)
        return dict(checkpoint_sha256=key,new_generation=True,published_at_unix=record['published_at_unix'])
    except FileNotFoundError:return None
    finally:
        if descriptor is not None:os.close(descriptor)
        if directory is not None:os.close(directory)
        if temporary is not None and temporary.exists():shutil.rmtree(temporary)


def child_budget(deadline, reserve, now=None):
    remaining = deadline - (time.monotonic() if now is None else now)
    if remaining <= reserve:
        raise TimeoutError('Setup exhausted training budget before finalization reserve')
    return remaining - reserve, remaining


def retain_failed_outputs(output, inputs, out, config):
    """Retain opaque, bounded evidence after all candidate processes are dead.

    Never unpickle or seal failed output. No symlinks, hardlinks, FIFOs, partial
    checkpoint.tmp files or arbitrary candidate filenames enter the archive.
    """
    destination = out / 'recovery'
    destination.mkdir(mode=0o700)
    record = dict(schema='pokemon-failed-training-artifacts-v1',
                  status='retained_unvalidated', eligible_for_submission=False,
                  original_stage_successful=False, files={}, errors={},
                  source_sha256=source_signature(inputs),
                  config_sha256=identity(config), runtime_sha256=runtime_signature())
    directory = None
    try:
        directory = os.open(output / 'train', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for name, limit in RECOVERY_LIMITS.items():
            descriptor = None
            target = destination / name
            try:
                descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_size > limit:
                    raise ValueError('Invalid bounded regular recovery artifact')
                digest = hashlib.sha256()
                count = 0
                with os.fdopen(descriptor, 'rb') as incoming, target.open('xb') as stream:
                    descriptor = None
                    while chunk := incoming.read(1024 * 1024):
                        count += len(chunk)
                        if count > limit or count > metadata.st_size:
                            raise ValueError('Recovery artifact grew after process cleanup')
                        stream.write(chunk); digest.update(chunk)
                    if count != metadata.st_size:
                        raise ValueError('Incomplete recovery artifact')
                target.chmod(0o600)
                record['files'][name] = dict(size_bytes=count, sha256=digest.hexdigest())
            except FileNotFoundError:
                pass
            except Exception as exc:
                record['errors'][name] = type(exc).__name__
                if target.exists():target.unlink()
            finally:
                if descriptor is not None:os.close(descriptor)
    except FileNotFoundError:
        record['errors']['train_directory'] = 'not_created'
    except Exception as exc:
        record['errors']['train_directory'] = type(exc).__name__
    finally:
        if directory is not None:os.close(directory)
    source = destination / 'source'; source.mkdir(mode=0o700)
    for name in (*EDITABLE, 'config.json'):
        shutil.copyfile(inputs / name, source / name)
        (source / name).chmod(0o600)
    record['checkpoint_retained'] = 'checkpoint.pt' in record['files']
    if not record['checkpoint_retained']:
        index=out/'durable-checkpoints/latest.json'
        if index.is_file():
            for key in __import__('json').loads(index.read_bytes()).get('generations',[]):
                if len(key)!=64 or any(c not in '0123456789abcdef'for c in key):continue
                candidate=out/'durable-checkpoints'/key
                try:
                    saved=__import__('json').loads((candidate/'manifest.json').read_bytes())
                    cp=candidate/'checkpoint.pt';meta=saved['files']['checkpoint.pt']
                    if (cp.is_symlink()or cp.stat().st_size!=meta['size_bytes']or sha(cp)!=key
                            or any(saved[n]!=record[n]for n in ('source_sha256','config_sha256','runtime_sha256'))):continue
                    shutil.copyfile(cp,destination/'checkpoint.pt')
                    record['files']['checkpoint.pt']=meta;record['checkpoint_retained']=True
                    record['checkpoint_origin']='durable_periodic';break
                except (OSError,ValueError,KeyError):continue
    write(destination / 'manifest.json', record)
    return record


def bounded_regular(path,limit,label):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>limit:
        raise ValueError('Invalid bounded '+label)
    return path


@contextmanager
def native_service(root,inputs,out,seconds):
    """Privileged native state stays in a separate process and private source tree."""
    path=root/'env.sock'
    environment={k:v for k,v in os.environ.items()if k in (
        'PATH','POKEMON_PILOT_RUNTIME','POKEMON_PILOT_ENV_KEY','POKEMON_PILOT_ENV_SCOPE')}
    environment.update(PYTHONPATH=str(Path(__file__).resolve().parents[4]),CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        POKEMON_PILOT_RELEASE_SHA256=runtime_signature(),
        POKEMON_PILOT_ENV_SOURCE_SHA256=source_signature(inputs))
    with(out/'env-service.log').open('wb')as log:
        process=subprocess.Popen([sys.executable,'-m','integrations.pokemon.autoresearch.runtime.env_service',
            '--socket',str(path),'--config',str(inputs/'config.json')],env=environment,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+min(60,seconds)
            while not path.exists():
                if process.poll()is not None:raise RuntimeError('Trusted environment service startup failed; see env-service.log')
                if time.monotonic()>=deadline:raise TimeoutError('Trusted environment startup deadline reached')
                time.sleep(.05)
            yield str(path)
        finally:
            if process.poll()is None:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait(timeout=5)


def train(source,config,out,seconds,device='cpu',resume=None,updates=None,development=False,fork=None,stop_request=None):
    """Run the candidate trainer unprivileged, then validate and seal what it wrote.

    `stop_request` is a path only the trusted side can create; the wait loop ends the trainer when it appears.
    """
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<seconds<=3600:
        raise ValueError('Finite training stage of at most3600 seconds required')
    if not development and (sys.platform!='linux' or os.geteuid()!=0):
        raise RuntimeError('Official training requires root Linux supervisor and isolated uid65534')
    if resume and fork:raise ValueError('Resume and explicit fork are distinct operations')
    if updates is not None and (type(updates)is not int or updates<1):raise ValueError('Positive update count required')
    validate_recipe(config);source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    started=time.time();clock_start=time.monotonic();deadline=clock_start+seconds
    reserve=min(FINALIZATION_RESERVE_SECONDS,seconds*.25)
    code=None;reason='setup_failed';phase='setup';success=False;ended=None
    receipt=dict(started_at_unix=started,requested_seconds=seconds,device=device,
        development=development,uid=os.getuid()if development else 65534,
        full_provider_allocation_required_separately=True,
        resume_parent_sha256=sha(resume)if resume else None,
        source_fork_parent_sha256=sha(fork)if fork else None,
        hard_deadline_monotonic=deadline,finalization_reserve_seconds=reserve,
        training_deadline_monotonic=deadline-reserve)
    with tempfile.TemporaryDirectory(prefix='pokemon-training-vnext-')as temporary:
        root=Path(temporary);root.chmod(0o755);inputs=root/'input';inputs.mkdir(mode=0o755)
        output=root/'output'
        try:
            for name in EDITABLE:
                path=source/name
                if path.is_symlink()or not path.is_file()or path.stat().st_size>1024*1024:
                    raise ValueError('Invalid bounded editable source '+name)
                shutil.copyfile(path,inputs/name);(inputs/name).chmod(0o444)
            write(inputs/'config.json',config);(inputs/'config.json').chmod(0o444)
            parent=resume or fork
            if parent:
                path=bounded_regular(parent,MAX_CHECKPOINT_BYTES,'resume checkpoint')
                shutil.copyfile(path,inputs/'resume.pt');(inputs/'resume.pt').chmod(0o444)
            output.mkdir(mode=0o700)
            if not development:os.chown(output,65534,65534)
            public=os.environ.get('POKEMON_PILOT_PUBLIC_PYTHONPATH')
            if not development and(not public or not Path(public).is_dir()):
                raise RuntimeError('A separately staged public import closure is required')
            environment={k:v for k,v in os.environ.items()if k in[
                'PATH','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}
            environment.update(HOME=str(output),TMPDIR=str(output),
                PYTHONPATH=str(Path(__file__).resolve().parents[4])if development else public,
                POKEMON_PILOT_RELEASE_SHA256=runtime_signature(),
                POKEMON_TRAINING_DEADLINE_MONOTONIC=str(deadline-reserve),
                POKEMON_TRAINING_HARD_DEADLINE_MONOTONIC=str(deadline),
                POKEMON_TRAINING_FINALIZATION_RESERVE_SECONDS=str(reserve),
                CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            child_budget(deadline,reserve)
            with native_service(root,inputs,out,max(.001,deadline-time.monotonic()))as socket_path,(out/'trainer.log').open('wb')as log:
                training_seconds,remaining=child_budget(deadline,reserve)
                cmd=[sys.executable,str(inputs/'train.py'),'--config',str(inputs/'config.json'),
                     '--out',str(output/'train'),'--seconds',str(training_seconds),'--device',device]
                if resume:cmd+=['--resume',str(inputs/'resume.pt')]
                if fork:cmd+=['--fork',str(inputs/'resume.pt')]
                if updates is not None:cmd+=['--updates',str(updates)]
                environment['POKEMON_PILOT_ENV_SOCKET']=socket_path
                receipt.update(child_training_seconds=training_seconds,child_hard_limit_seconds=remaining)
                phase='running';reason='trainer_failed'
                child=subprocess.Popen(cmd,cwd=inputs,stdout=log,stderr=subprocess.STDOUT,env=environment,
                    start_new_session=True,preexec_fn=None if development else drop_privileges)
                try:
                    try:
                        while True:
                            if stop_request is not None and os.path.lexists(stop_request):
                                raise StopRequested('Trainer ended on a stop request')
                            remaining=deadline-time.monotonic()
                            if remaining<=0:raise HardDeadline('Trainer exceeded hard stage deadline')
                            try:code=child.wait(timeout=min(CHECKPOINT_POLL_SECONDS,remaining));break
                            except subprocess.TimeoutExpired:
                                if time.monotonic()>=deadline:raise HardDeadline('Trainer exceeded hard stage deadline')
                                try:
                                    saved=publish_checkpoint(output,inputs,out,config,deadline)
                                    if saved:receipt['last_durable_checkpoint']=saved
                                except TimeoutError:raise
                                except Exception as exc:receipt['periodic_checkpoint_error_type']=type(exc).__name__
                        reason='completed'if code==0 else'signal_'+str(-code)if code<0 else'trainer_error'
                    except (HardDeadline,StopRequested)as exc:
                        # Not raised yet: the trainer is killed below, then what it wrote is validated and sealed.
                        reason='stop_requested'if isinstance(exc,StopRequested)else'hard_deadline';ended=exc
                    except BaseException as exc:
                        reason='supervisor_interrupted_'+type(exc).__name__;raise
                finally:
                    if child.poll()is None:
                        try:os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        child.wait(timeout=5)
                    code=child.returncode
                    if not development:
                        subprocess.run(['pkill','-KILL','-u','65534'],check=False,timeout=5,
                            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            if ended is None and code!=0:raise RuntimeError('Trainer failed; retained recovery artifacts are not a successful bundle')
            packaging=time.monotonic()
            try:
                if ended is not None:
                    # Every candidate process is dead. The flag is in the receipt before sealing binds it.
                    receipt.update({EARLY[reason]:True},trainer_ended_at_unix=time.time())
                else:phase='validating';reason='artifact_validation_failed'
                if (output/'train').is_symlink() or not (output/'train').is_dir():
                    raise ValueError('Regular trainer output directory required')
                for name in STAGE_FILES:
                    path=output/'train'/name
                    bounded_regular(path,RECOVERY_LIMITS[name],'trainer output '+name)
                    shutil.copyfile(path,out/name)
                if ended is None:phase='sealing';reason='sealing_failed'
                manifest=seal(out/'bundle',inputs,config,out/'checkpoint.pt',receipt)
            except Exception as exc:
                if ended is None:raise
                # No complete policy: the stage fails as it always did, with its retained recovery outputs.
                receipt.pop(EARLY[reason]);receipt['early_packaging_error_type']=type(exc).__name__
                for name in STAGE_FILES:(out/name).unlink(missing_ok=True)
                raise ended from exc
            receipt.update(bundle_sha256=manifest['bundle_sha256'],runtime_sha256=runtime_signature(),
                packaging_seconds=time.monotonic()-packaging,
                source_sha256=source_signature(inputs),config_sha256=identity(config))
            success=True;phase='complete'
            if ended is None:reason='completed'
            return receipt
        except BaseException as exc:
            receipt['error_type']=type(exc).__name__
            raise
        finally:
            # This runs while the private temporary directory still exists.
            # A failed run stays failed; preservation never calls seal().
            if not success:
                if (out/'bundle').exists():shutil.rmtree(out/'bundle')
                try:
                    receipt['recovery']=retain_failed_outputs(output,inputs,out,config)
                    # Sealing may have failed after normal artifacts were copied.
                    # Keep one verified copy instead of duplicating up to512MiB
                    # and overflowing the worker's bounded publication archive.
                    for name,metadata in receipt['recovery']['files'].items():
                        duplicate=out/name
                        if duplicate.is_file()and not duplicate.is_symlink()and sha(duplicate)==metadata['sha256']:
                            duplicate.unlink()
                except Exception as exc:receipt['recovery_error_type']=type(exc).__name__
            receipt.update(returncode=code,termination_reason=reason,phase=phase,successful=success,
                finished_at_unix=time.time(),process_wall_seconds=time.monotonic()-clock_start)
            write(out/'supervisor.json',receipt)
