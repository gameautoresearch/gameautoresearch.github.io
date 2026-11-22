"""Generals Modal workers: training sandboxes, leases, practice and private grading sandboxes.

One image serves all of them: the pinned engine and the control modules at /opt/gr and the public training
maps at /data. Opponent checkpoints and private maps are never in the image; the host copies them into the
root-only /private of an evaluation worker.

A lease is a worker of a shape the researcher chooses that runs the researcher's own commands as an
unprivileged user, with the session's scratch Volume at /scratch.

Results reach the host through the backend root's private hand-off Volume (`harness.handoff`), mounted root-only
at /handoff in training workers and leases: the worker publishes them as root before it ends, and the host
collects them afterwards, outside the worker's paid time and with no deadline. A worker that could not publish
falls back to the old live read while it is still alive.
"""
from harness.policy import ENVIRONMENT
from harness import hardware
from harness import workers
from harness.handoff import SessionVolume, NotPublished, publish_command, prepare_command
from simulators.generals.autoresearch.provider import configure_sandbox_waits,wait_process,reset_rpc_channels,read_rpc,allocation_info
from pathlib import Path
import asyncio
import json
import math
import shlex
import shutil
import tempfile
import threading
import time
import uuid

from simulators.generals.autoresearch.contract import ROOT,HERE,EDITABLE,DEFAULT,BUNDLED,PRACTICE_SHAPE,TRAIN_SHAPE,digest,write_json
from simulators.generals.autoresearch.bundle import verify,unpack_bundle,FILES,ARCHIVE

PYTHON='python'
PYTHONPATH='/opt/gr:/opt/gr/candidates/generals-bots'

SCRATCH = '/scratch'
WORK = '/work'


def unprivileged_pythonpath(release):
    """What a researcher's command imports from. A release with an empty starter stages `frozen/unprivileged` (the
    engine's rule modules and the wrapper without its scripted players) and its image hides the rest of /opt/gr from
    that user; other releases import the frozen tree itself."""
    return '/opt/gr/unprivileged' if (Path(release) / 'frozen/unprivileged').is_dir() else PYTHONPATH
# Researcher commands on a lease: unprivileged, no way back to root, in their own session.
UNPRIVILEGED = 'setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs'
LEASE_TEARDOWN_SECONDS = 20  # before the grant ends: stop commands, commit /scratch, let Modal record the end
OUTPUT_CHARS = 20000
PACKAGE_SECONDS = 300
# What `ar package` reads from a directory on the lease, and the most it reads of each.
PACKAGE_FILES = {'weights.npz': DEFAULT['artifact_bytes_cap'], 'model.json': 65536}
# The trainer's console goes to these root-owned worker files (not the exec's streams), so the trusted end of a
# run can hand it over with everything else.
TRAINER_LOGS = {'stdout': '/job/trainer-stdout.log', 'stderr': '/job/trainer-stderr.log'}
LOG_BYTES = 16000
# Where the trusted end of a run stages its bounded copies inside the worker (root-only).
STAGING = '/job/handoff-out'

# Runs in the worker as root after every unprivileged process has been killed: python -c FINISH SPEC.
# The trusted export, bounded no-follow copies of the logs the host keeps, a summary, then the publish to the
# session's hand-off Volume (harness.handoff.PUBLISH through `publish_command`), all in one exec: once it has
# started, nothing in it waits on the host's link. Prints the summary as one JSON line; exit 0 = published.
FINISH = r'''import json,os,stat,subprocess,sys,time
began=time.time();spec=json.loads(sys.argv[1]);out=spec['folder']
os.makedirs(out,mode=0o700,exist_ok=True);os.chmod(out,0o700)
summary={}
try:
    done=subprocess.run([sys.executable,'-m','integrations.generals_autoresearch.export_entry'],capture_output=True,text=True,timeout=spec['export_seconds'])
    summary.update(export_returncode=done.returncode,export_stdout=done.stdout[-2000:],export_stderr=done.stderr[-2000:])
except subprocess.TimeoutExpired:summary.update(export_returncode=None,export_error='export timed out')
settle=time.time()+spec['settle_seconds']
while not os.path.exists('/job/supervisor.json') and time.time()<settle:time.sleep(.1)
def copy(source,name,limit):
    try:fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    except OSError:return
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):return
        os.lseek(fd,max(0,info.st_size-limit),os.SEEK_SET);data=bytearray()
        while len(data)<limit:
            chunk=os.read(fd,min(1<<20,limit-len(data)))
            if not chunk:break
            data+=chunk
    finally:os.close(fd)
    with open(os.path.join(out,name),'wb') as stream:stream.write(data)
for source,name,limit in spec['copies']:copy(source,name,limit)
summary['seconds']=round(time.time()-began,3)
with open(os.path.join(out,'finish.json'),'w') as stream:json.dump(summary,stream,sort_keys=True)
command=spec['publish'];present=[]
for item in command[4:]:
    path=item.partition('=')[2]
    if os.path.isfile(path) and not os.path.islink(path):present.append(item)
try:
    done=subprocess.run(command[:4]+present,capture_output=True,text=True,timeout=max(1.,spec['seconds']-(time.time()-began)))
    published=done.returncode==0;detail=done.stderr[-2000:]
except subprocess.TimeoutExpired:published=False;detail='publish timed out'
print(json.dumps(dict(summary,published=published,published_files=sorted(i.partition('=')[0] for i in present),
    publish_error=None if published else detail),sort_keys=True),flush=True)
raise SystemExit(0 if published else 3)'''


def capture_training_diagnostics(sandbox, process, target, *, checkpoint_valid, handed=None):
    """Host-only evidence; never turns a failed trainer into a clean completion.

    `handed` is the worker's handed-over copies, {'folder': local folder with supervisor.json and the trainer
    logs, 'parent_rc': the training exec's return code polled before the worker ended, 'parent_error'}; nothing
    is then read from the worker. Without it the supervisor and the exec's streams are read live.
    """
    target = Path(target)
    errors = {}
    supervisor = parent_rc = None
    if handed is not None:
        folder = Path(handed['folder'])
        parent_rc = handed.get('parent_rc')
        if handed.get('parent_error'):
            errors['parent_exec_rc'] = handed['parent_error']
        try:
            supervisor = json.loads((folder / 'supervisor.json').read_text())
            write_json(target / 'supervisor.json', supervisor)
        except Exception as exc:
            errors['supervisor'] = type(exc).__name__ + ': ' + str(exc)[:500]
        for stream in TRAINER_LOGS:
            try:
                text = (folder / ('trainer-' + stream + '.log')).read_bytes()[-LOG_BYTES:].decode('utf-8', 'replace')
                (target / ('trainer-' + stream + '.log')).write_text(text)
            except Exception as exc:
                errors[stream] = type(exc).__name__ + ': ' + str(exc)[:500]
    else:
        try:
            supervisor=json.loads(read_rpc(lambda:sandbox.filesystem.read_text('/job/supervisor.json')))
            write_json(target/'supervisor.json',supervisor)
        except Exception as exc:errors['supervisor']=type(exc).__name__+': '+str(exc)[:500]
        try:parent_rc=read_rpc(process.poll)
        except Exception as exc:errors['parent_exec_rc']=type(exc).__name__+': '+str(exc)[:500]
        if parent_rc is not None:
            for stream in ('stdout','stderr'):
                try:(target/('trainer-'+stream+'.log')).write_text(getattr(process,stream).read()[-LOG_BYTES:])
                except Exception as exc:errors[stream]=type(exc).__name__+': '+str(exc)[:500]
    child_rc=supervisor.get('returncode') if supervisor else None
    clean=bool(supervisor and supervisor.get('status')=='completed' and child_rc==0 and parent_rc==0)
    known_failure=bool((child_rc is not None and child_rc!=0) or (supervisor and supervisor.get('status')=='deadline') or (parent_rc is not None and parent_rc!=0))
    failure_scope=('child' if child_rc is not None and child_rc!=0 else
        'deadline' if supervisor and supervisor.get('status')=='deadline' else
        'parent_exec' if parent_rc is not None and parent_rc!=0 else None)
    lifecycle=('clean_trainer_completion' if checkpoint_valid and clean else
        'salvaged_checkpoint_after_process_failure' if checkpoint_valid and known_failure else
        'valid_checkpoint_training_exit_unverified' if checkpoint_valid else 'no_valid_exported_checkpoint')
    record=dict(parent_exec_returncode=parent_rc,child_returncode=child_rc,supervisor_status=supervisor.get('status') if supervisor else None,
        checkpoint_valid=bool(checkpoint_valid),training_completion_verified=clean,lifecycle=lifecycle,
        failure_scope=failure_scope,
        diagnostic_errors=errors,admission_note='Checkpoint validity and training completion are separate. This host-only sidecar does not retroactively change prior C eligibility, reset charges, select another policy or supply researcher feedback.')
    write_json(target/'training-lifecycle.json',record)
    return record


def image_for(release):
    import modal
    from simulators.generals.autoresearch.prepare import audit_release
    audit_release(release);release=Path(release)
    # util-linux (setpriv, setsid) and procps (pkill) run and end the unprivileged processes of a worker.
    return (modal.Image.debian_slim(python_version='3.12').apt_install('git','util-linux','procps')
        .pip_install('jax[cuda12]==0.12.0.1','flax==0.12.12','optax==0.2.8','numpy==2.5.4','pygame==2.6.1','python-socketio[client]==5.18.1','pytest==9.2.0','psutil==7.2.2')
        .add_local_dir(release/'frozen','/opt/gr',copy=True,ignore=['**/__pycache__/**'])
        .add_local_file(release/'public/train-maps.npz','/data/train-maps.npz',copy=True)
        .env({'PYTHONPATH':PYTHONPATH,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
              'JAX_PLATFORMS':'cpu','JAX_DEFAULT_MATMUL_PRECISION':'highest','XLA_PYTHON_CLIENT_PREALLOCATE':'false'})
        .run_commands('mkdir -p /job/input /job/output /private', 'chmod 700 /private',
            'git config --system --add safe.directory /opt/gr/candidates/generals-bots',
            'chmod -R a-w /opt/gr /data',
            # An empty-starter release: only root reads the full tree (reference policy, scripted players, the
            # upstream reward helpers and example trainer, the engine's history); everyone reads `unprivileged`.
            *(['chmod 700 /opt/gr/integrations /opt/gr/candidates'] if (release/'frozen/unprivileged').is_dir() else []))
        .workdir('/job'))


class Lease:
    """One running lease: the sandbox, its commands and the host's record of them. Callers hold `lock`."""

    def __init__(self, sb, target, shape, pythonpath=PYTHONPATH):
        self.sb = sb
        self.pythonpath = pythonpath
        self.target = Path(target)
        self.shape = shape
        self.lock = threading.Lock()
        self.commands = {}  # exec id -> {'process', 'command', 'started'}
        self.pushes = 0
        (self.target / 'exec').mkdir(parents=True, exist_ok=True)

    def _root(self, script, *args, timeout=60):
        process = self.sb.exec('/bin/sh', '-c', script, *args, timeout=timeout)
        code = process.wait()
        return code, process.stdout.read(), process.stderr.read()

    def push(self, files):
        """Copy the editable files to /work, owned by the researcher's user."""
        folders = sorted({str(Path(WORK, path).parent) for path in files})
        code, _, err = self._root('mkdir -p ' + ' '.join(map(shlex.quote, folders)))
        for path, text in files.items():
            self.sb.filesystem.write_text(text, f'{WORK}/{path}')
        code, _, err = self._root(f'chown -R 65534:65534 {WORK}') if code == 0 else (code, '', err)
        if code != 0:
            raise RuntimeError('Lease push failed: ' + err[-300:])
        self.pushes += 1
        return {'pushed': sorted(files), 'to': WORK}

    def running(self):
        return [i for i, c in self.commands.items() if c['process'].poll() is None]

    def start(self, command):
        if len(self.running()) >= DEFAULT['lease_running_commands']:
            raise ValueError(f'At most {DEFAULT["lease_running_commands"]} commands run on a lease at once')
        exec_id = f'{len(self.commands) + 1:04d}'
        log = f'{WORK}/.ar/{exec_id}.log'
        cores = self.shape['cpu_cores']
        # The image default keeps JAX on the CPU for the host's own processes; a GPU lease's commands get the GPU.
        environment = (f'HOME={WORK} TMPDIR=/tmp PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin '
                       f'PYTHONPATH={self.pythonpath}:{WORK}/training:{WORK}/simulator OMP_NUM_THREADS={cores} MKL_NUM_THREADS={cores} '
                       f'OPENBLAS_NUM_THREADS={cores} JAX_PLATFORMS={"cuda" if self.shape["gpu"] else "cpu"} '
                       f'XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_COMPILATION_CACHE_DIR={WORK}/.jax-cache')
        # $0 is the researcher's command, passed as an argument so nothing of it is interpreted as root.
        # `--wait`: when the command is a process group leader setsid forks, and without it the command would be
        # reported as exited (code 0) the moment it started (seen live in Polytopia, 2026-11-14). `training/launch.py`
        # packages when its training command exits, so it depends on this.
        redirect = shlex.quote('exec bash -c "$1" > "$2" 2>&1 < /dev/null')
        script = f'cd {WORK} && exec {UNPRIVILEGED} env {environment} setsid --wait bash -c {redirect} lease "$0" {log}'
        process = self.sb.exec('/bin/sh', '-c', script, command)
        self.commands[exec_id] = {'process': process, 'command': command, 'started': time.time()}
        write_json(self.target / 'exec' / (exec_id + '.json'),
                   {'exec': exec_id, 'command': command, 'started': self.commands[exec_id]['started'],
                    'state': 'running', 'log': log})
        return {'exec': exec_id, 'state': 'running', 'log': log}

    def poll(self, exec_id, tail=None):
        entry = self.commands.get(exec_id)
        if entry is None:
            raise ValueError('Unknown command id for this lease')
        code = entry['process'].poll()
        chars = min(int(tail or OUTPUT_CHARS), OUTPUT_CHARS)
        _, output, _ = self._root(f'{UNPRIVILEGED} tail -c {chars} -- {WORK}/.ar/{exec_id}.log 2>/dev/null', timeout=30)
        record = {'exec': exec_id, 'command': entry['command'], 'started': entry['started'],
                  'state': 'running' if code is None else 'exited', 'exit_code': code, 'output': output,
                  'log': f'{WORK}/.ar/{exec_id}.log'}
        if code is not None:
            write_json(self.target / 'exec' / (exec_id + '.json'), record)
        return record

    def sync(self, reload=False):
        """Commit /scratch so other leases can see it; `reload` also brings in what other leases committed."""
        code, _, err = self._root(f'sync {SCRATCH}', timeout=120)
        if code != 0:
            raise RuntimeError('Committing /scratch failed: ' + err[-300:])
        if reload:
            try:
                self.sb.reload_volumes()
            except Exception as exc:
                raise ValueError('Reload failed; close files and leave directories under /scratch first: '
                                 + str(exc)[:300]) from None
        return {'committed': SCRATCH, 'reloaded': bool(reload)}

    def package(self, target, path, code, job):
        """Seal the network at `path` with `code` as root and publish the archive to the hand-off Volume as `job`.

        Returns True when the worker confirmed the publish; the host then collects it from the Volume.
        """
        if not path.startswith('/') or '\0' in path:
            raise ValueError('--path is an absolute directory on the lease, such as /scratch/run1')
        source = shlex.quote(path.rstrip('/') or '/')
        copies = ' && '.join(f'{UNPRIVILEGED} head -c {cap + 1} -- {source}/{name} > /job/output/{name}'
                             for name, cap in PACKAGE_FILES.items())
        status, _, err = self._root('rm -rf /job/input /job/output /job/export /job/trusted-export && '
                                    'mkdir -p /job/input /job/output && ' + copies, timeout=PACKAGE_SECONDS)
        if status != 0:
            raise ValueError(f'{path} needs a readable weights.npz and model.json: ' + err[-300:].strip())
        for name in EDITABLE:
            self.sb.filesystem.write_text(code[name], '/job/input/' + name)
        status, _, err = self._root(f'{PYTHON} -m integrations.generals_autoresearch.export_entry',
                                    timeout=PACKAGE_SECONDS)
        if status != 0:
            (target / 'export-error.log').write_text(err[-10000:])
            raise ValueError('Not a valid policy package: ' + err[-600:].strip())
        publish = self.sb.exec(*publish_command(job, {ARCHIVE: '/job/' + ARCHIVE}, python=PYTHON),
                               timeout=PACKAGE_SECONDS)
        status = publish.wait()
        if status != 0:
            (target / 'publish-error.log').write_text(publish.stderr.read()[-10000:])
        return status == 0

    def fetch(self, local):
        """The fallback when the package was not published: the archive in one live transfer, as before."""
        self.sb.filesystem.copy_to_local('/job/' + ARCHIVE, local)

    def close(self):
        """End every researcher process, keep the last output of each command, commit /scratch."""
        try:
            self._root('pkill -KILL -u 65534 || true', timeout=10)
            for exec_id in list(self.commands):
                try:
                    record = self.poll(exec_id)
                    if record['state'] == 'running':
                        write_json(self.target / 'exec' / (exec_id + '.json'), dict(record, state='ended with the lease'))
                except Exception:
                    pass
            self._root(f'sync {SCRATCH}', timeout=120)
        except Exception as exc:  # the worker may already be gone; Modal commits the Volume when it terminates
            (self.target / 'lease-close-error.log').write_text(type(exc).__name__ + ': ' + str(exc))


class ModalProvider:
    """Start/abandon semantics are the shared ones in `harness.workers`: queued time is never charged, a worker
    Modal never starts is terminated and settled at 0, and no backend leaves such a job `uncertain`."""

    # Seconds of a training run's grant kept for ending the trainer, the root-side export and fetching the bundle:
    # 5 % of the grant, at least 40 s, at most 60 s. The sandbox's own timeout stays the grant, so a worker can
    # never be charged past it; the slack the export needs is therefore taken inside the grant.
    # Sizing (evidence/small-20261113, live L40S workers): from the host's stop to the last bundle file on the
    # host took 8-14 s on a good link and 21-40 s on a slow one (126-325 KB of weights at about 10 KB/s). With a
    # 15 s reserve, four of six 120 s runs lost their network to the sandbox timeout. The bundle now comes back
    # as one archive in one transfer, and the reserve covers the slowest transfer seen.
    # Since 2026-11-14 the bundle and logs reach the host through the session's hand-off Volume after the worker
    # has ended, so the normal path no longer streams anything inside the reserve. The reserve is kept unchanged
    # (the trainer's time is not silently lengthened): it still holds the export and the in-datacenter publish,
    # and the live read that is the fallback when a worker could not publish.
    MIN_EXPORT_RESERVE=40
    MAX_EXPORT_RESERVE=60
    EXPORT_RESERVE_FRACTION=0.05
    # The worker's supervisor ends an overrunning trainer when the reserve begins; the host waits this much
    # longer before ending it itself.
    STOP_GRACE=3
    # A transfer still unanswered this long after the worker's own timeout is given up on.
    TRANSFER_GRACE=5
    PROGRESS_POLL_SECONDS=5.0
    TAIL_BYTES=1000000
    TAIL_TIMEOUT=30
    # Hand-off: of the finish exec's time, at most this much is kept for the publish after the export; the
    # finish waits at most SETTLE_SECONDS for the supervisor's record of a trainer it has just seen killed.
    PUBLISH_SECONDS = 15
    SETTLE_SECONDS = 3
    # Reading the Volume. After the worker confirmed its publish (and, for a training run, has been ended): long
    # retries, no worker deadline. While the worker is alive and did not confirm: a short probe before the live
    # fallback. A package (the researcher waits for its reply) and recovery of an interrupted controller: shorter.
    COLLECT_ATTEMPTS = 120
    COLLECT_WAIT = 5.0
    PROBE_ATTEMPTS = 2
    PROBE_WAIT = 2.0
    PACKAGE_ATTEMPTS = 24
    RECOVER_ATTEMPTS = 6

    @classmethod
    def export_reserve(cls,grant):
        """Export reserve for a run granted `grant` seconds: 5 % of it, at least 40 s, at most 60 s."""
        return min(cls.MAX_EXPORT_RESERVE,max(cls.MIN_EXPORT_RESERVE,cls.EXPORT_RESERVE_FRACTION*float(grant)))

    @classmethod
    def tail(cls,sb,remote,timeout=None):
        """The newest TAIL_BYTES of a worker file (whole lines), read by a bounded `tail` with its own exec timeout.

        A whole-file read of a long run's progress log stalled for longer than the time left to export; a
        tail has a fixed size and a deadline. None when the file is absent or the read fails. No channel
        reset is attempted here: this also runs beside the loop that stops the trainer and must not disturb it.
        """
        read=sb.exec('tail','-c',str(cls.TAIL_BYTES),remote,timeout=max(1,int(cls.TAIL_TIMEOUT if timeout is None else timeout)))
        text=read.stdout.read()
        while (code:=read.poll()) is None:time.sleep(.05)
        if code!=0:return None
        return text[text.find('\n')+1:] if len(text.encode())>=cls.TAIL_BYTES else text

    def mirror_progress(self,sb,target,done,writing):
        """Copy the tail of the trainer's progress log every poll until `done` is set; a failed read skips a copy.

        Runs in its own thread. Once `done` is set (under `writing`) nothing more is written, so a read that
        returns late cannot replace the final copy taken after the export.
        """
        while not done.wait(self.PROGRESS_POLL_SECONDS):
            try:
                text=self.tail(sb,'/job/output/progress.jsonl')
                with writing:
                    if text is not None and not done.is_set():(target/'progress.jsonl').write_text(text)
            except Exception:pass

    @staticmethod
    def fetch(sb,remote,local,seconds):
        """Copy one worker file to the host, giving up after `seconds`: a transfer cannot hold the host for longer."""
        failure=[]
        def work():
            try:read_rpc(lambda:sb.filesystem.copy_to_local(remote,local))
            except BaseException as exc:failure.append(exc)
        transfer=threading.Thread(target=work,daemon=True);transfer.start();transfer.join(max(1.,seconds))
        if transfer.is_alive():raise RuntimeError('Bundle transfer did not finish within '+str(int(max(1.,seconds)))+' s')
        if failure:raise failure[0]

    # What the trusted end of a run copies for the host: (worker file, handed-over name, newest bytes kept).
    COPIES = (('/job/output/progress.jsonl', 'progress.jsonl', TAIL_BYTES),
              ('/job/output/result.json', 'result.json', TAIL_BYTES),
              ('/job/supervisor.json', 'supervisor.json', 1 << 20),
              (TRAINER_LOGS['stdout'], 'trainer-stdout.log', LOG_BYTES),
              (TRAINER_LOGS['stderr'], 'trainer-stderr.log', LOG_BYTES))

    @staticmethod
    def handoff(root):
        """The backend root's private hand-off Volume: a research session's and its fresh final's are separate."""
        return SessionVolume('generals', root)

    def finish(self, sb, name, deadline):
        """The trusted end of a training run, in one root exec inside the worker (FINISH): export, copy, publish.

        The host waits only for this short in-datacenter exec; it returns the worker's one-line summary
        ({'published': bool, 'export_returncode', ...}), or raises when the exec or its reply was lost.
        """
        seconds = max(1, int(deadline - time.time() - 2))
        files = {ARCHIVE: '/job/' + ARCHIVE, 'finish.json': STAGING + '/finish.json',
                 **{local: STAGING + '/' + local for _, local, _ in self.COPIES}}
        spec = dict(seconds=seconds, export_seconds=max(1., seconds - min(self.PUBLISH_SECONDS, seconds / 2)),
                    settle_seconds=self.SETTLE_SECONDS, folder=STAGING, copies=[list(c) for c in self.COPIES],
                    publish=publish_command(name, files, python=PYTHON))
        run = sb.exec(PYTHON, '-c', FINISH, json.dumps(spec), timeout=seconds)
        code = wait_process(run)
        lines = run.stdout.read().strip().splitlines()
        reply = json.loads(lines[-1]) if lines else {}
        if not isinstance(reply, dict):
            raise RuntimeError('Unreadable finish reply')
        if code != 0:
            reply['published'] = False
        return dict(reply, finish_returncode=code)

    def read_live(self, sb, folder, deadline, finish=None, bundle=True):
        """The fallback when a still-running worker did not publish, and the old way: read the results live.

        The archive comes in one transfer that is given up on once the worker must be gone (its failure fails
        the run, never leaving a partial bundle), then bounded tails of the small files. Writes the same names
        into `folder` as a Volume collection would.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if bundle:
                if not finish or 'export_returncode' not in finish:
                    text = self.tail(sb, STAGING + '/finish.json', timeout=min(self.TAIL_TIMEOUT, deadline - time.time()))
                    finish = json.loads(text) if text else {}
                write_json(folder / 'finish.json', {k: v for k, v in finish.items() if k.startswith('export_')})
                if finish.get('export_returncode') == 0:
                    self.fetch(sb, '/job/' + ARCHIVE, folder / ARCHIVE, deadline - time.time() + self.TRANSFER_GRACE)
        finally:
            # Best effort, after the bundle: final progress, the result, the supervisor's record and the trainer's
            # console. Every read is a bounded tail that ends by the sandbox's own timeout.
            for remote, local, _ in self.COPIES:
                try:
                    text = self.tail(sb, remote, timeout=min(self.TAIL_TIMEOUT, deadline - time.time()))
                    if text is not None:
                        (folder / local).write_text(text)
                except Exception:
                    pass

    def receive(self, folder, target, status, cut):
        """The run's outcome from its handed-over files, with the same checks whichever way they came.

        Returns (status, early, bundle identity, bundle directory) or raises when there is no valid package.
        """
        folder = Path(folder)
        for name in ('progress.jsonl', 'result.json'):
            if (folder / name).is_file():
                text = (folder / name).read_bytes().decode('utf-8', 'replace')
                (target / name).write_text(text[text.find('\n') + 1:] if len(text.encode()) >= self.TAIL_BYTES else text)
        finish = json.loads((folder / 'finish.json').read_text()) if (folder / 'finish.json').is_file() else {}
        if finish.get('export_returncode') != 0:
            raise RuntimeError('Trainer did not leave a valid policy package')
        if not (folder / ARCHIVE).is_file():
            raise RuntimeError('The exported policy package was not handed over')
        # The package directory appears only when every member arrived and verified: never a partial bundle.
        identity = unpack_bundle(folder / ARCHIVE, target / 'bundle')
        (folder / ARCHIVE).unlink()
        # The trainer saves its network as it goes, so a stopped or cut-off run whose last save sealed into a
        # valid package is complete and says how it ended. Without one the checks above raised: the run stays
        # stopped (or failed) and has nothing to practise or submit.
        early = 'stopped_early' if status == 'stopped' else 'cut_short' if cut else None
        # The worker's own supervisor ends a trainer that overruns its command limit and records it.
        try:
            ended = json.loads((folder / 'supervisor.json').read_text()).get('status')
        except Exception:
            ended = None
        if early is None and ended == 'deadline':
            early = 'cut_short'
        return 'complete', early, identity, target / 'bundle'

    def __init__(self,release,app,cash=None,allocation_budget=None,accounting=None,cpu_inference_profile='default'):
        configure_sandbox_waits()
        self.live={}  # lease name -> Lease, while its worker is serving commands
        self.lock=threading.Lock()
        self.release=Path(release);self.app=app;self.image=image_for(release);self.cash=cash
        self.allocation_budget=allocation_budget
        self.accounting=accounting
        if cpu_inference_profile not in ('default','avx2'):raise ValueError('Unsupported fixed CPU inference profile')
        self.cpu_inference_profile=cpu_inference_profile
        if cash and not any(r['name']=='image-and-egress-margin' for r in cash.snapshot()['reservations']):
            cash.reserve('image-and-egress-margin',.5,'infrastructure margin')
        # Resolve immutable image uploads before any SandboxCreate request.
        # Retrying this content-addressed build cannot duplicate an allocation.
        self.build_reservation=accounting.reserve_build('generals-source-image-build') if accounting else None
        read_rpc(lambda:self.image.build(app))

    def recover(self,ledger,job,stopped=False):
        """Reconcile an interrupted controller without creating a GPU worker.

        A `running` job keeps its provider deadline across a host restart. An `uncertain` job is one the backend
        abandoned: its recorded worker is terminated now; never started -> settled failed (stopped when
        `stopped`) with 0 charged, started -> reconciled from the provider's start/finish receipt.
        """
        import modal
        target=ledger.root/'jobs'/job['name'];path=target/'provider.json'
        if not path.exists():
            ledger.uncertain(job['name'],'No provider identity available; original reservation retained')
            return
        identity=json.loads(path.read_text())
        read=lambda:allocation_info(identity['app_id'],identity['sandbox_id'])
        info=read()
        if info and job['status']=='uncertain' and not info.get('finished_at'):
            info=workers.abandon(lambda:modal.Sandbox.from_id(identity['sandbox_id']).terminate(wait=True),read)
            if not job.get('started') and info.get('provider_id') and workers.never_started(info):
                reason=workers.NEVER_STARTED+(' (stopped while unresolved)' if stopped else '')
                ledger.abandon_unstarted(job['name'],info,reason,stopped=stopped)
                write_json(target/'recovery.json',dict(provider=info,status='stopped' if stopped else 'failed',
                    never_started=True,charged_usd=0,created_workers=0))
                return
        if not info or not info.get('started_at'):
            ledger.uncertain(job['name'],'Provider allocation metadata still unavailable; no rerun')
            return
        if not info.get('finished_at'):
            # The provider timeout remains authoritative across host interruption.
            # Terminate only this recorded worker when its original deadline passes.
            if time.time()<info['started_at']+job['grant']:return
            modal.Sandbox.from_id(identity['sandbox_id']).terminate(wait=True)
            for _ in range(30):
                info=read()
                if info and info.get('finished_at'):break
                time.sleep(.5)
        if not info or not info.get('finished_at'):
            ledger.uncertain(job['name'],'Provider end still unknown; original hold retained')
            return
        artifact=target/'bundle';bundle_sha=None;status='failed'
        early = handed = None
        try:
            if artifact.exists():bundle_sha=verify(artifact)['bundle_sha256'];status='complete'
        except (ValueError,OSError,KeyError):artifact=None
        if not bundle_sha:artifact=None
        # A package copied out before the interruption came from a trainer that was ended; a stop request says by whom.
        if bundle_sha and stopped:
            early = 'stopped_early'
        if not bundle_sha and json.loads(job.get('payload') or '{}').get('kind', 'train') == 'train':
            # A worker that published its result before the controller was interrupted handed it to the Volume.
            try:
                self.handoff(ledger.root).collect(job['name'], target / 'handoff', attempts=self.RECOVER_ATTEMPTS,
                                                  wait=self.COLLECT_WAIT)
                handed = 'volume'
            except NotPublished:
                pass
            except ValueError as exc:  # a published file that differs from its manifest: never used
                write_json(target / 'recovery-handoff-error.json', dict(type=type(exc).__name__, message=str(exc)[:1000]))
            except Exception as exc:
                ledger.uncertain(job['name'], 'Hand-off Volume unreadable during recovery (' + type(exc).__name__
                                 + '); original hold retained')
                return
            if handed:
                try:
                    status, early, bundle_sha, artifact = self.receive(target / 'handoff', target,
                                                                       'stopped' if stopped else 'failed', False)
                except (ValueError, RuntimeError, OSError) as exc:
                    write_json(target / 'recovery-handoff-error.json',
                               dict(type=type(exc).__name__, message=str(exc)[:1000]))
        ledger.reconcile(job['name'],info,status,artifact,bundle_sha,
            None if bundle_sha else 'Controller interruption: the lease ended with its worker; allocation charged'
            if json.loads(job['payload'] or '{}').get('kind')=='lease' else
            'Controller interruption: no complete validated exported artifact; allocation charged',early=early)
        write_json(target/'recovery.json',dict(provider=info,status=status,artifact_sha256=bundle_sha,created_workers=0,
                                               handoff=handed,**({early:True} if early else {})))
        if self.accounting and identity.get('accounting_ticket'):
            self.accounting.settle_worker(identity['accounting_ticket'],info)
        if self.allocation_budget and identity.get('allocation_budget_ticket'):
            ticket=identity['allocation_budget_ticket']
            if not any(r['id']==ticket and r['status']=='settled' for r in self.allocation_budget.snapshot()['allocations']):
                self.allocation_budget.settle(ticket,info)

    def scratch(self, root):
        """The session's scratch Volume, created with its first lease and recorded in the session directory.

        It is never deleted here: it holds the session's checkpoints and is collected for the run archive.
        """
        import modal
        path = Path(root) / 'scratch-volume.json'
        with self.lock:
            if not path.exists():
                write_json(path, {'volume': 'generals-scratch-' + uuid.uuid4().hex, 'environment': ENVIRONMENT,
                                  'mount': SCRATCH, 'created_unix': time.time()}, exclusive=True)
            name = json.loads(path.read_text())['volume']
        return modal.Volume.from_name(name, environment_name=ENVIRONMENT, create_if_missing=True, version=2)

    def _sandbox(self, name, grant, kind, shape, root):
        """A worker of `shape`. A training run mounts the hand-off Volume; a lease mounts it beside /scratch (it
        hands packages over); practice mounts nothing (its replies are small). Each mount is made root-only
        before researcher code runs."""
        import modal
        resources = dict(cpu=(shape['cpu_cores'],) * 2, memory=(shape['memory_mib'],) * 2)
        if shape['gpu']:
            resources['gpu'] = hardware.GPU
        volumes = {}
        if kind == 'lease':
            volumes[SCRATCH] = self.scratch(root)
        if kind in ('train', 'lease'):
            volumes.update(self.handoff(root).mounts())
        return modal.Sandbox.create('/bin/sleep', '86400', app=self.app, image=self.image, timeout=math.ceil(grant),
                                    block_network=True, include_oidc_identity_token=False, secrets=[],
                                    volumes=volumes, environment_name=ENVIRONMENT, tags={'generals_ar_job': name},
                                    **resources)

    def execute(self, ledger, job, stop=None):
        name = job['name']
        payload = json.loads(job['payload'])
        grant = job['grant']
        kind = payload.get('kind', 'train')
        shape = payload.get('shape') or (PRACTICE_SHAPE if kind == 'practice' else TRAIN_SHAPE)
        gpu = hardware.GPU if shape['gpu'] else None
        sb = proc = started = artifact = identity = error = never = early = None
        status = 'failed'
        # Hand-off: which way the result came ('volume' | 'live'), whether it is collected after the worker has
        # ended, why the worker's publish was not confirmed, the local copies, the exec's reply and return code.
        route = unconfirmed = handed = reply = manifest = None
        pending = cut = False
        polled = {}
        provider_receipt = {}
        unstarted = False
        target = ledger.root / 'jobs' / name
        target.mkdir(parents=True, exist_ok=False)
        write_json(target / 'request.json', payload, exclusive=True)
        campaign_hold = self.allocation_budget.reserve(kind, grant) if self.allocation_budget else None
        resource_ticket = self.accounting.reserve_worker('generals-' + kind, grant + 30) if self.accounting else None
        try:
            sb = self._sandbox(name, grant, kind, shape, ledger.root)
            if resource_ticket:
                self.accounting.attach(resource_ticket, sb.object_id)
            write_json(target / 'provider.json', {'sandbox_id': sb.object_id, 'app_id': self.app.app_id,
                                                  'accounting_ticket': resource_ticket,
                                                  'allocation_budget_ticket': campaign_hold}, exclusive=True)
            # Queued time (created, no worker yet) is never charged; wait for the provider's start or a stop.
            info = workers.await_start(lambda: allocation_info(self.app.app_id, sb.object_id), stop=stop)
            if info is None:
                unstarted = True
                raise RuntimeError(('Stopped while' if stop and stop() else
                                    'Gave up after ' + str(int(workers.START_PATIENCE_SECONDS)) + ' s')
                                   + ' waiting in the provider queue for a GPU')
            started = info['started_at']
            ledger.start(name, sb.object_id, started)
            if gpu and not hardware.matches(info['gpu_type']):
                raise RuntimeError('Requested ' + hardware.GPU + ' but provider reported ' + info['gpu_type'])
            deadline = started + grant
            if kind == 'lease':
                self._serve_lease(sb, name, payload, shape, target, deadline, stop, ledger)
                status = 'stopped' if stop and stop() else 'complete'
            elif kind == 'practice':
                self.stage_evaluation(sb, payload['bundle'], payload['panel'], payload['seeds'], {}, True)
                remaining = deadline - time.time() - 12
                if remaining < 1:
                    raise RuntimeError('Practice setup exhausted allocation')
                # Game state and trusted opponents stay on CPU. Only the
                # unprivileged submitted forward gets the allocated GPU.
                proc = sb.exec(PYTHON, '-m', 'integrations.generals_autoresearch.evaluate_entry', '--request',
                               '/private/evaluation.json', timeout=max(1, int(remaining)),
                               env={'JAX_PLATFORMS': 'cpu', 'GENERALS_POLICY_PLATFORM': 'cuda'})
                while read_rpc(proc.poll) is None and time.time() < deadline - 3:
                    ledger.milestones()
                    if stop and stop():break
                    time.sleep(.5)
                if read_rpc(proc.poll) is None:raise RuntimeError('Practice allocation deadline; no partial score awarded')
                if proc.returncode!=0:raise RuntimeError('Practice evaluation failed (exit '+str(proc.returncode)+'): '+proc.stderr.read()[-1000:])
                write_json(target/'practice.json',json.loads(read_rpc(lambda:sb.filesystem.read_text('/job/evaluation-public.json'))))
                status='complete'
            else:
                # The hand-off mount is made root-only before any researcher code runs.
                if wait_process(sb.exec(*prepare_command(PYTHON), timeout=30)) != 0:
                    raise RuntimeError('Could not make the hand-off mount private')
                for filename in EDITABLE:sb.filesystem.write_text(payload['code'][filename],'/job/input/'+filename)
                sb.filesystem.write_text(json.dumps(payload['config']),'/job/input/config.json')
                sb.filesystem.write_text(json.dumps({'seed':payload['seed']}),'/job/input/request.json')
                reserve=self.export_reserve(grant)
                remaining=deadline-time.time()-reserve
                if remaining<5:raise RuntimeError('Allocation consumed by setup before the learner could start')
                # The same supervisor process as before (`exec`), its console in root-owned files the end of the
                # run hands over.
                proc = sb.exec('/bin/sh', '-c', 'exec "$@" >' + TRAINER_LOGS['stdout'] + ' 2>' + TRAINER_LOGS['stderr'],
                               'sh', PYTHON, '-m', 'integrations.generals_autoresearch.train_entry',
                               '--seconds', str(remaining), timeout=max(1, int(remaining + 2)))
                # Progress is mirrored by its own thread. This loop only polls the trainer, so it reaches the
                # stop point and the export on time whatever a read does.
                mirrored=threading.Event();writing=threading.Lock()
                threading.Thread(target=self.mirror_progress,args=(sb,target,mirrored,writing),daemon=True).start()
                try:
                    while read_rpc(proc.poll) is None and time.time()<deadline-(reserve-self.STOP_GRACE):
                        ledger.milestones()
                        if stop and stop():
                            status='stopped';break
                        time.sleep(.5)
                finally:
                    with writing:mirrored.set()
                # A trainer still running here was stopped by the researcher or has reached the host's time limit.
                cut=status!='stopped' and read_rpc(proc.poll) is None
                # End every unprivileged process, then the trusted end of the run at once: it owns the reserve.
                kill=sb.exec('/bin/sh','-c','pkill -KILL -u 65534 || true',timeout=2);wait_process(kill)
                if time.time() > deadline - 2:
                    raise RuntimeError('No time remains for validated artifact export')
                # Export, bounded copies of the logs and the publish to the session's Volume, in one exec inside the
                # worker. The host waits for that short exec only and collects once the worker has ended.
                handed = target / 'handoff'
                try:
                    reply = self.finish(sb, name, deadline)
                except Exception as exc:
                    reply = {'published': False, 'finish_error': type(exc).__name__ + ': ' + str(exc)[:500]}
                try:
                    polled = dict(parent_rc=read_rpc(proc.poll))
                except Exception as exc:
                    polled = dict(parent_error=type(exc).__name__ + ': ' + str(exc)[:500])
                route = 'volume'
                if reply.get('published'):
                    pending = True
                else:
                    unconfirmed = (reply.get('finish_error') or reply.get('publish_error')
                                   or 'The worker did not confirm a publish (exit ' + str(reply.get('finish_returncode')) + ')')
                    # The worker may have published although its reply was lost: a short look while it is alive.
                    try:
                        manifest = self.handoff(ledger.root).collect(name, handed, attempts=self.PROBE_ATTEMPTS,
                                                                     wait=self.PROBE_WAIT)
                    except NotPublished:
                        # Nothing published: read the results live from the still-running worker, as before.
                        route = 'live'
                        self.read_live(sb, handed, deadline, reply)
                    except ValueError:
                        raise
                    except Exception as exc:
                        # The Volume cannot be read from here right now: collect again once the worker has ended.
                        unconfirmed += '; Volume probe: ' + type(exc).__name__ + ': ' + str(exc)[:300]
                        pending = True
                    if not pending:
                        status, early, identity, artifact = self.receive(handed, target, status, cut)
        except BaseException as exc:
            # A training run the researcher stopped that left no valid package stays stopped; anything else failed.
            status = 'stopped' if kind == 'train' and status == 'stopped' else 'failed'
            error = type(exc).__name__ + ': ' + str(exc)
            import traceback
            (target / 'host-error-traceback.txt').write_text(traceback.format_exc())
        finally:
            if sb is not None:
                training = kind == 'train' and proc is not None
                if training and route is None:
                    # The run failed before its end: its record and console, read live and bounded, as before.
                    handed = target / 'handoff'
                    try:
                        self.read_live(sb, handed, deadline, bundle=False)
                    except Exception:
                        pass
                    try:
                        polled = dict(parent_rc=read_rpc(proc.poll))
                    except Exception as exc:
                        polled = dict(parent_error=type(exc).__name__ + ': ' + str(exc)[:500])
                failures = []

                def terminate():
                    try:
                        sb.terminate(wait=True)
                    except Exception as exc:
                        failures.append(exc)
                        raise
                final = workers.abandon(terminate, lambda: allocation_info(self.app.app_id, sb.object_id))
                if failures:
                    error = (error or '') + '; termination: ' + str(failures[0])
                if pending:
                    # Outside the worker's paid time: fetch the published result, retrying with no worker deadline.
                    try:
                        manifest = self.handoff(ledger.root).collect(name, handed, attempts=self.COLLECT_ATTEMPTS,
                                                                     wait=self.COLLECT_WAIT)
                        status, early, identity, artifact = self.receive(handed, target, status, cut)
                    except BaseException as exc:
                        error = (error + '; ' if error else '') + type(exc).__name__ + ': ' + str(exc)
                        import traceback
                        with (target / 'host-error-traceback.txt').open('a') as stream:
                            stream.write(traceback.format_exc())
                if training:
                    try:
                        capture_training_diagnostics(sb, proc, target, checkpoint_valid=identity is not None,
                                                     handed=dict(folder=handed, **polled))
                    except Exception as exc:
                        write_json(target / 'diagnostic-capture-error.json',
                                   dict(type=type(exc).__name__, message=str(exc)[:1000]))
                if route:
                    write_json(target / 'handoff.json', dict(
                        handoff=route, volume=self.handoff(ledger.root).name, job=name, unconfirmed_publish=unconfirmed,
                        finish={k: v for k, v in (reply or {}).items() if k not in ('export_stdout', 'export_stderr')},
                        manifest={k: dict(bytes=v['bytes'], sha256=v['sha256']) for k, v in manifest.items()}
                        if manifest else None))
                if final.get('started_at') and final.get('finished_at'):
                    provider_receipt = final
                elif started is None and final.get('provider_id') and workers.never_started(final):
                    # Created but never started: nothing was allocated, so nothing is charged and nothing stays held.
                    stopped = bool(unstarted and stop and stop())
                    reason = workers.NEVER_STARTED + '. ' + (error or 'Worker setup failed before the provider started it')
                    ledger.abandon_unstarted(name, final, reason, stopped=stopped)
                    status = 'stopped' if stopped else 'failed'
                    provider_receipt = final
                    never = workers.NeverStarted(final, reason)
            if never is None:
                if provider_receipt:
                    if resource_ticket:
                        self.accounting.settle_worker(resource_ticket, provider_receipt)
                    if campaign_hold:
                        self.allocation_budget.settle(campaign_hold, provider_receipt)
                    if started is None:
                        ledger.start(name, sb.object_id, provider_receipt['started_at'])
                    ledger.finish(name, provider_receipt['finished_at'], status, artifact if identity else None, identity,
                                  error, early=early if status == 'complete' and identity else None)
                else:
                    ledger.uncertain(name, error or 'Provider allocation end is unavailable; reservation retained')
            write_json(target / 'allocation.json', {'provider': provider_receipt, 'status': status, 'error': error,
                                                    **({early: True} if early and status == 'complete' and identity else {}),
                                                    'budget_scope': DEFAULT['allocation_rule'], 'shape': shape,
                                                    'artifact_sha256': identity, 'guest_clock_trusted': False,
                                                    'never_started': never is not None,
                                                    **({'handoff': route} if route else {}),
                                                    **({'handoff_fallback': unconfirmed} if route == 'live' else {})})
        if never is not None:
            raise never
        return next(x for x in ledger.jobs() if x['name'] == name)

    # --- leases -----------------------------------------------------------------------------------------

    def _serve_lease(self, sb, name, payload, shape, target, deadline, stop, ledger):
        """Hold the worker for the researcher's commands until the grant runs out or the lease is stopped."""
        # The hand-off mount is root-only before any researcher command runs (they run as 65534 via setpriv).
        if wait_process(sb.exec(*prepare_command(PYTHON), timeout=30)) != 0:
            raise RuntimeError('Could not make the hand-off mount private')
        setup = sb.exec('/bin/sh', '-c', f'mkdir -p {WORK}/.ar && chmod 1777 {SCRATCH} && chown -R 65534:65534 {WORK} '
                        f'&& {UNPRIVILEGED} sh -c "touch {SCRATCH}/.writable && rm {SCRATCH}/.writable"', timeout=60)
        if wait_process(setup) != 0:
            raise RuntimeError('Lease setup failed: ' + setup.stderr.read()[-500:])
        lease = Lease(sb, target, shape, unprivileged_pythonpath(self.release))
        lease.push(payload['files'])
        self.live[name] = lease
        write_json(target / 'lease.json', {'shape': shape, 'scratch': SCRATCH, 'work': WORK, 'ready_unix': time.time()})
        try:
            while time.time() < deadline - LEASE_TEARDOWN_SECONDS and not (stop and stop()):
                ledger.milestones()
                time.sleep(1)
        finally:
            self.live.pop(name, None)
            with lease.lock:  # after any command or export in flight
                lease.close()

    def lease_live(self, name):
        return name in self.live

    def _live(self, name):
        lease = self.live.get(name)
        if lease is None:
            raise ValueError('The lease is not serving commands (starting up or ending)')
        return lease

    def lease_exec(self, name, command):
        lease = self._live(name)
        with lease.lock:
            return lease.start(command)

    def lease_poll(self, name, exec_id, tail=None):
        lease = self._live(name)
        with lease.lock:
            return lease.poll(exec_id, tail)

    def lease_push(self, name, files):
        lease = self._live(name)
        with lease.lock:
            return lease.push(files)

    def lease_sync(self, name, reload=False):
        lease = self._live(name)
        with lease.lock:
            return lease.sync(reload)

    def lease_package(self, target, name, path, code, config):
        """Seal weights.npz and model.json from a directory on the lease, with this source, as a policy package.

        The researcher's files are read as the unprivileged user; the sealing is the same root-side export a
        training run gets, so the parameter and size caps are enforced here too. The lease publishes the sealed
        archive to the hand-off Volume as root; the host collects it from there after releasing the lease, or
        copies it live when the lease could not publish.
        """
        lease = self._live(name)
        target = Path(target)
        volume = self.handoff(lease.target.parent.parent)  # the lease's backend root
        job = target.name + '-' + uuid.uuid4().hex[:8]  # a retried package name never meets an earlier attempt
        handed = target / 'handoff'
        route = unconfirmed = None
        with lease.lock:
            target.mkdir(parents=True, exist_ok=False)
        try:
            with lease.lock:
                if not lease.package(target, path, code, job):
                    unconfirmed = 'The lease did not confirm its publish'
                    try:
                        volume.collect(job, handed, attempts=self.PROBE_ATTEMPTS, wait=self.PROBE_WAIT)
                        route = 'volume'
                    except NotPublished:
                        route = 'live'
                        handed.mkdir(parents=True, exist_ok=True)
                        lease.fetch(handed / ARCHIVE)
            if route is None:
                volume.collect(job, handed, attempts=self.PACKAGE_ATTEMPTS, wait=self.COLLECT_WAIT)
                route = 'volume'
            # The directory appears only once every member arrived and verified.
            identity = unpack_bundle(handed / ARCHIVE, target / 'bundle')
            (handed / ARCHIVE).unlink()
            write_json(target / 'request.json', {'kind': 'package', 'path': path, 'config': config, 'code': code})
            write_json(target / 'handoff.json', {'handoff': route, 'volume': volume.name, 'job': job,
                                                 'unconfirmed_publish': unconfirmed})
            return target / 'bundle', identity
        except BaseException as exc:
            (target / 'package-error.log').write_text(type(exc).__name__ + ': ' + str(exc))
            target.rename(target.with_name(target.name + '.failed-' + uuid.uuid4().hex[:8]))
            raise

    def stage_evaluation(self,sb,bundle,panel,seeds,anchors,development):
        for label,path in [('candidate',bundle)]+[(p['name'],p['bundle']) for p in panel if p['kind'] in BUNDLED]:
            remote='/private/'+label;sb.filesystem.make_directory(remote)
            for filename in (*FILES,'manifest.json'):
                sb.filesystem.copy_from_local(Path(path)/filename,remote+'/'+filename)
        sb.filesystem.copy_from_local(seeds['pool_path'],'/private/maps.npz')
        remote_panel=[dict(p,bundle='/private/'+p['name']) if p['kind'] in BUNDLED else p for p in panel]
        request=dict(seeds,pool_path='/private/maps.npz',candidate='/private/candidate',panel=remote_panel,
            anchors=anchors or {},development=development)
        sb.filesystem.write_text(json.dumps(request),'/private/evaluation.json')

    def evaluate(self,bundle,panel,seeds,anchors=None,development=True,timeout=600,inference_gpu=True):
        """GPU inference, CPU private engine/panel; separate UID and contexts."""
        import modal,uuid
        reservation='grade-'+uuid.uuid4().hex
        sb=None;info=None;receipt_path=None;abandoned=None
        campaign_hold=self.allocation_budget.reserve('hidden-grading' if not development else 'host-assessment',timeout) if self.allocation_budget and inference_gpu else None
        resource_ticket=self.accounting.reserve_worker('generals-private-grading',timeout+30,hardware.GPU if inference_gpu else None) if self.accounting else None
        # Keep failed CPU evaluations discoverable as well as successful ones.
        if self.cash:
            receipt_path=self.cash.path.parent/'provider-evaluations'/(reservation+'.json')
        elif self.allocation_budget:
            receipt_path=self.allocation_budget.path.parent/'provider-evaluations'/(reservation+'.json')
        elif self.accounting:
            receipt_path=self.accounting.root/'provider-evaluations'/(reservation+'.json')
        try:
            gpu=hardware.GPU if inference_gpu else None
            sb=modal.Sandbox.create('/bin/sleep','86400',app=self.app,image=self.image,cpu=(4,4),memory=(16384,16384),
                timeout=timeout,block_network=True,secrets=[],volumes={},include_oidc_identity_token=False,environment_name=ENVIRONMENT,**({'gpu':gpu} if gpu else {}))
            if resource_ticket:self.accounting.attach(resource_ticket,sb.object_id)
            if receipt_path:write_json(receipt_path,{'app_id':self.app.app_id,'sandbox_id':sb.object_id,'state':'allocated','gpu':gpu,'accounting_ticket':resource_ticket})
            # Queued time is not charged. A worker that never starts is terminated and recorded as uncharged.
            if workers.await_start(lambda:allocation_info(self.app.app_id,sb.object_id)) is None:
                abandoned=workers.abandon(lambda:sb.terminate(wait=True),lambda:allocation_info(self.app.app_id,sb.object_id))
                if abandoned.get('provider_id') and workers.never_started(abandoned):raise workers.NeverStarted(abandoned)
                raise RuntimeError('Evaluation worker did not start within '+str(int(workers.START_PATIENCE_SECONDS))+' s; abandoned and charged from its receipt')
            self.stage_evaluation(sb,bundle,panel,seeds,anchors or {},development)
            proc=sb.exec(PYTHON,'-m','integrations.generals_autoresearch.evaluate_entry',timeout=timeout-10,
                env={'JAX_PLATFORMS':'cpu','GENERALS_POLICY_PLATFORM':'cuda' if inference_gpu else 'cpu',
                     'GENERALS_CPU_INFERENCE_PROFILE':getattr(self,'cpu_inference_profile','default')})
            next_progress=0.
            while (code:=read_rpc(proc.poll)) is None:
                if receipt_path and time.time()>=next_progress:
                    try:
                        value=read_rpc(lambda:sb.filesystem.read_text('/private/evaluation-progress.json'))
                        receipt_path.with_suffix('.progress.json').write_text(value)
                    except Exception:pass
                    next_progress=time.time()+5
                time.sleep(.1)
            if code!=0:
                progress=None
                try:progress=read_rpc(lambda:sb.filesystem.read_text('/private/evaluation-progress.json'))
                except Exception:pass
                if receipt_path:write_json(receipt_path.with_suffix('.failure.json'),{'returncode':code,'private_progress':progress})
                stderr=''
                try:stderr=proc.stderr.read()[-1500:]
                except Exception:pass
                raise RuntimeError('Isolated evaluation failed (exit '+str(code)+'): '+stderr)
            return {'public':json.loads(read_rpc(lambda:sb.filesystem.read_text('/job/evaluation-public.json'))),
                'private':json.loads(read_rpc(lambda:sb.filesystem.read_text('/private/evaluation-result.json'))),
                'provider':{'app_id':self.app.app_id,'sandbox_id':sb.object_id,'gpu':gpu}}
        finally:
            if sb is not None:
                termination_error=None
                try:
                    if abandoned is None:sb.terminate(wait=True)
                except Exception as exc:termination_error=exc
                finally:
                    if abandoned is not None:info=abandoned
                    elif receipt_path:
                        try:
                            for _ in range(30):
                                info=allocation_info(self.app.app_id,sb.object_id)
                                if info and info.get('finished_at'):break
                                time.sleep(.5)
                        except Exception as exc:info={'unresolved':True,'error':type(exc).__name__+': '+str(exc)}
                    if receipt_path:
                        write_json(receipt_path,{'app_id':self.app.app_id,'sandbox_id':sb.object_id,'gpu':gpu,'allocation':info,
                            'never_started':bool(abandoned is not None and abandoned.get('provider_id') and workers.never_started(abandoned))})
                    if campaign_hold and (info or {}).get('started_at') and (info or {}).get('finished_at'):
                        self.allocation_budget.settle(campaign_hold,info)
                    if resource_ticket and (info or {}).get('started_at') and (info or {}).get('finished_at'):
                        self.accounting.settle_worker(resource_ticket,info)
                if termination_error and not (info or {}).get('finished_at'):raise termination_error
