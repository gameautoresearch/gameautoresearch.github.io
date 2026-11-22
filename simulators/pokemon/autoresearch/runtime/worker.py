"""Trusted finite sandbox entry; no provider/model credentials enter this worker."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import tarfile
import time
import traceback

MAX_ARTIFACT_BYTES=512*1024*1024
MAX_ARTIFACT_TOTAL_BYTES=2*1024*1024*1024
MAX_ARTIFACT_MEMBERS=4096
PUBLICATION_RESERVE_SECONDS=30
ASSESSMENT_PUBLICATION_RESERVE_SECONDS=15
# A training stage must end this long before the worker's absolute stop: the first half is for ending the trainer,
# validating and sealing what it wrote (also when the stage is cut off at its own deadline), the second is the
# publication reserve above.
STAGE_END_RESERVE_SECONDS=60
# The host asks a running trainer to end by creating this file in the job's root-only directory; the supervisor
# then seals the last complete policy the trainer wrote. Nothing unprivileged can reach the directory.
STOP_REQUEST='stop-request'


def timing_reserves(action):
    """Short public diagnostics retain real publication/cleanup headroom."""
    if action=='assess':
        return dict(publication=ASSESSMENT_PUBLICATION_RESERVE_SECONDS,start_guard=20)
    return dict(publication=PUBLICATION_RESERVE_SECONDS,start_guard=60)


def _deadline(_signum,_frame):
    raise TimeoutError('Trusted worker absolute deadline reached')


def publish_results(out,job,deadline):
    """Bounded, atomic archive publication; incomplete archives are never exposed."""
    out=Path(out);job=Path(job);members=[];total=0
    archive=out/'results.tar';temporary=out/'results.tar.partial'
    try:
        for path in sorted(out.rglob('*')):
            if time.time()>=deadline:raise TimeoutError('Worker publication deadline reached')
            relative=path.relative_to(out)
            if any(x in relative.parts for x in ('build','public-source','__pycache__'))or path in (archive,temporary):continue
            metadata=path.lstat()
            if stat.S_ISLNK(metadata.st_mode):raise ValueError('Linked worker artifact')
            if stat.S_ISDIR(metadata.st_mode):continue
            if (not stat.S_ISREG(metadata.st_mode)or metadata.st_size>MAX_ARTIFACT_BYTES
                    or len(members)>=MAX_ARTIFACT_MEMBERS):raise ValueError('Invalid bounded worker publication member')
            total+=metadata.st_size
            if total>MAX_ARTIFACT_TOTAL_BYTES:raise ValueError('Worker publication aggregate byte ceiling')
            members.append((path,str(relative)))
        with tarfile.open(temporary,'w')as tar:
            for path,name in members:
                if time.time()>=deadline:raise TimeoutError('Worker publication deadline reached')
                tar.add(path,arcname=name,recursive=False)
        remaining=deadline-time.time()
        if remaining<=0:raise TimeoutError('Worker publication deadline reached')
        # Publish the complete tar before syncing the durable operation mount.
        os.replace(temporary,archive)
        subprocess.run(['sync',str(job)],timeout=min(10,remaining),check=True)
        return dict(files=len(members),uncompressed_bytes=total,archive_bytes=archive.stat().st_size)
    finally:
        if temporary.exists():temporary.unlink()


def runtime(inputs):
    from ...native.bridge import build
    from .sdk import sha,write
    root=Path('/runtime');root.mkdir(mode=0o700)
    for key in ('generators','constructors','chart'):
        shutil.copyfile(inputs/'runtime'/f'{key}.json',root/f'{key}.json');(root/f'{key}.json').chmod(0o444)
    library=build(root/'build',admission=inputs/'native-admission.json')
    paths=dict(library=library,generators=root/'generators.json',constructors=root/'constructors.json',chart=root/'chart.json')
    # The certificate names the one team size it qualifies; the environment plays exactly that format.
    team_size=json.loads((inputs/'native-admission.json').read_bytes()).get('team_size',6)
    write(root/'runtime.json',dict(full_gen9_admission=True,team_size=team_size,sha256={k:sha(v)for k,v in paths.items()},
        **{k:str(v.relative_to(root))for k,v in paths.items()}))
    for p in root.rglob('*'):p.chmod(0o755 if p.is_dir() else 0o444)
    os.environ['POKEMON_PILOT_RUNTIME']=str(root/'runtime.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    job=Path(a.job);inputs=job/'input';out=job/'output';out.mkdir(mode=0o700,exist_ok=True)
    request_file=inputs/'effective-request.json'if(inputs/'effective-request.json').exists()else inputs/'request.json'
    spec=json.loads(request_file.read_bytes());started=time.time()
    deadline=float(spec['absolute_stop_unix'])
    if not __import__('math').isfinite(deadline):raise ValueError('Finite absolute worker deadline required')
    reserves=timing_reserves(spec['action'])
    previous=signal.signal(signal.SIGALRM,_deadline)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()-reserves['publication']))
    try:
        if time.time()>spec['absolute_stop_unix']-reserves['start_guard']:raise RuntimeError('Finite absolute deadline reached before work')
        if spec['action']in('train','qualification'):
            key=inputs/'env-state-key.bin';key.chmod(0o400)
            os.environ['POKEMON_PILOT_ENV_KEY']=str(key)
            os.environ['POKEMON_PILOT_ENV_SCOPE']=spec['arm']
        if spec['action'] in ('train','qualification'):runtime(inputs)
        if spec['action']=='qualification':
            from .qualification import run
            run(spec,out,inputs)
        elif spec['action']=='qualification_cpu':
            from .qualification import run_cpu
            run_cpu(spec,out,inputs)
        elif spec['action']=='train':
            from .supervisor import train
            from .sdk import sha,source_signature
            source=inputs/'candidate'
            if spec.get('fresh_final'):
                if spec.get('resume') or spec.get('fork') or any((inputs/n).exists() for n in ('resume.pt','fork.pt')):raise ValueError('Final worker forbids research state')
                if sorted(p.name for p in source.iterdir())!=sorted(['policy.py','learner.py','train.py','config.json']):raise ValueError('Final source-only payload')
            if source_signature(source)!=spec['source_sha256']:raise ValueError('Candidate staged source changed')
            practice_seconds=spec.get('public_practice',{}).get('seconds',0)
            if spec.get('fresh_final') and practice_seconds:raise ValueError('Final training forbids public-practice followup')
            seconds=min(spec['training_seconds'],spec['absolute_stop_unix']-time.time()-STAGE_END_RESERVE_SECONDS-practice_seconds)
            train(source,spec['config'],out/'stage',seconds,'cuda:0',inputs/'resume.pt'if spec.get('resume')else None,
                updates=spec.get('updates'),fork=inputs/'fork.pt'if spec.get('fork')else None,stop_request=job/STOP_REQUEST)
            if spec.get('public_practice'):
                from .practice import run_practice
                try:run_practice(spec,out,inputs,deadline)
                except Exception as exc:
                    (out/'practice-failure.json').write_text(json.dumps(dict(error_type=type(exc).__name__,score_valid=False)))
        elif spec['action']=='assess':
            from .arena import run
            # Private request stays under root-owned700 job mount; candidates
            # receive only schema-projected public observations through pipes.
            request=spec['assessment'];run(request,out,str(inputs/'request.json'))
        else:raise ValueError('Unknown trusted worker action')
        (out/'completed.json').write_text(json.dumps(dict(status='complete',body_seconds=time.time()-started)))
    except BaseException:
        (out/'worker-error.log').write_text(traceback.format_exc());raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
        try:
            publish_results(out,job,deadline)
        except BaseException:(out/'archive-error.log').write_text(traceback.format_exc())
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':main()
