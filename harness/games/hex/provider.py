"""Hex Modal workers: training sandboxes, leases, CPU practice and private grading sandboxes.

Two images share their base layers (CUDA PyTorch from the hash lock):

- the research image, for training runs and leases, where the researcher's code runs. It holds the frozen
  engine at /opt/hex and nothing of MoHex: neither its source nor a binary.
- the evaluation image, for practice and grading. It adds MoHex compiled from the release's own benzene
  source, readable by root only, so the unprivileged policy process under evaluation cannot run or read it.

A lease is a worker of a shape the researcher chooses that runs the researcher's own commands as an
unprivileged user, with the session's scratch Volume at /scratch.

Results travel back through the backend root's private hand-off Volume (`harness.handoff`), mounted root-only at
/handoff in training workers and leases: the worker's trusted export step publishes the sealed archive there
before the worker ends, and the host reads it afterwards, outside the worker's paid time, with no deadline. A
slow or interrupted host link therefore never costs a finished run. Only when the worker reports that its publish
failed, and is still up, does the host fall back to reading the archive out of it live, as it used to.
"""
import json
import math
from pathlib import Path
import shlex
import tempfile
import threading
import time
import traceback
import uuid

from harness import handoff, workers
from harness.policy import ENVIRONMENT
from simulators.hex.autoresearch.bundle import archive_bundle, unpack_bundle, verify
from simulators.hex.autoresearch.contract import DEFAULT, EDITABLE, GAME, PRACTICE_SHAPE, TRAIN_SHAPE, write_json

PYTHON = '/opt/exact-python/cpython-3.12.9-linux-x86_64-gnu/bin/python3.12'
MOHEX = '/opt/mohex-build/src/mohex/mohex'
GPU = 'L40S'


SCRATCH = '/scratch'
WORK = '/work'
# Researcher commands on a lease: unprivileged, no way back to root, in their own session.
UNPRIVILEGED = 'setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs'
LEASE_TEARDOWN_SECONDS = 20  # before the grant ends: stop commands, commit /scratch, let Modal record the end
OUTPUT_CHARS = 20000
PACKAGE_SECONDS = 300
STAGE = '/job/handoff-stage'  # root-only: what a training worker publishes besides its sealed archive
LOG_BYTES = 1000000
# The largest sealed archive: `unpack_bundle`'s own member limits plus tar headers.
ARCHIVE_BYTES = (DEFAULT['artifact_bytes_cap'] + len(EDITABLE) * DEFAULT['source_bytes_cap'] + 65536 + 1048576
                 + 20480)

# A training worker's whole root-side sequence, run as one exec so that it completes without the host: train,
# end every researcher process, export, then publish the archive and the run's logs to the hand-off Volume.
# A stop (`pkill` from the host) only ends the trainer early; the export and publish still follow. Logs are
# copied by size-capped tails; the researcher-owned progress file is read as the researcher, with a time limit.
TRAIN_SCRIPT = '''{python} -m simulators.hex.autoresearch.train_entry --seconds {seconds}
pkill -KILL -u 65534
mkdir -p -m 700 {stage}
{python} -m simulators.hex.autoresearch.export_entry > {stage}/export.out 2> {stage}/export.err
code=$?
printf '{{"exit_code": %d}}\\n' "$code" > {stage}/export.json
if [ "$code" -ne 0 ]; then tail -c 10000 {stage}/export.err > {stage}/export-error.log; fi
tail -c {logs} /job/trainer.log > {stage}/trainer.log 2> /dev/null
head -c 65536 /job/supervisor.json > {stage}/supervisor.json 2> /dev/null
timeout -s KILL 10 {unprivileged} tail -c {logs} -- /job/output/progress.jsonl > {stage}/progress.jsonl 2> /dev/null
set --
for name in export.json export-error.log trainer.log supervisor.json progress.jsonl; do
  if [ -s {stage}/$name ]; then set -- "$@" "$name={stage}/$name"; fi
done
if [ "$code" -eq 0 ]; then set -- "$@" bundle.tar=/job/export.tar; fi
exec {publish} "$@"
'''


class NoPolicyPackage(RuntimeError):
    """The worker published its logs but its export left no sealed archive."""


def handoff_job(name, worker):
    """A result's folder in the hand-off Volume: the job and the worker (or attempt) that produced it, so a
    published folder is never mistaken for another attempt's."""
    return handoff.check_name(f'{name}.{worker}')


def train_script(seconds, job):
    return TRAIN_SCRIPT.format(python=PYTHON, seconds=seconds, stage=STAGE, logs=LOG_BYTES, unprivileged=UNPRIVILEGED,
                               publish=shlex.join(handoff.publish_command(job, {}, python=PYTHON)))


def prepare(sb):
    """Make /handoff root-only before any researcher code runs in this worker, and check the researcher's user
    cannot see into it."""
    script = (shlex.join(handoff.prepare_command(PYTHON)) +
              f' && ! {UNPRIVILEGED} test -r {handoff.MOUNT} -o -w {handoff.MOUNT} -o -x {handoff.MOUNT}')
    process = sb.exec('/bin/sh', '-c', script, timeout=60)
    if process.wait() != 0:
        raise RuntimeError('Hand-off mount setup failed: ' + process.stderr.read()[-500:])


def collect_published(volume, job, target, attempts=6, wait=5.0):
    """A training run's published result in `target`: every file checked against the worker's manifest, then the
    archive unpacked and verified as a policy package exactly as a live copy is. Returns (bundle, sha, files)."""
    files = volume.collect(job, target, limit=ARCHIVE_BYTES, attempts=attempts, wait=wait)
    if 'bundle.tar' not in files:
        raise NoPolicyPackage('Trainer did not leave a valid policy package; export diagnostics retained')
    artifact = target / 'bundle'
    if artifact.exists():  # an unpack an interrupted host left behind: kept aside, never trusted
        artifact.rename(target / ('bundle.partial-' + uuid.uuid4().hex[:8]))
    unpack_bundle(target / 'bundle.tar', artifact)
    return artifact, verify(artifact, check_tensors=False)['bundle_sha256'], sorted(files)


def recover_published(volume, ledger, job, sandbox_id, info, target, stopped):
    """Settle a training run whose host was lost, from what its worker published. False when it published nothing
    (the run then fails as before); a Volume that cannot be read now keeps the job's hold for a later attempt."""
    name = job['name']
    key = handoff_job(name, sandbox_id)
    try:
        volume.manifest(key)
    except handoff.NotPublished:
        return False
    except Exception as exc:
        if type(exc).__name__ == 'NotFoundError':  # Modal's: the session has no hand-off Volume (an older session)
            return False
        ledger.uncertain(name, 'Controller recovery could not read the hand-off Volume (' + type(exc).__name__ +
                         '); hold retained')
        return True
    record = {'handoff': 'volume', 'job': key, 'volume': volume.name}
    artifact = identity = None
    try:
        artifact, identity, record['files'] = collect_published(volume, key, target)
        status = 'stopped' if stopped else 'complete'
        error = 'Controller recovery: result collected from the hand-off Volume; charged exact provider lease'
    except (ValueError, NoPolicyPackage) as exc:
        status, record['error'] = 'failed', type(exc).__name__ + ': ' + str(exc)
        error = 'Controller recovery: charged exact completed provider lease; ' + record['error']
    except Exception as exc:
        ledger.uncertain(name, 'Controller recovery could not read the hand-off Volume (' + type(exc).__name__ +
                         '); hold retained')
        return True
    ledger.reconcile(name, sandbox_id, info['started_at'], info['finished_at'], error, status=status,
                     artifact=artifact, artifact_sha=identity)
    write_json(target / ('recovered-' + str(time.time_ns()) + '.json'), dict(provider=info, handoff=record),
               exclusive=True)
    return True


def _base(release):
    import modal
    from simulators.hex.autoresearch.prepare import audit_release
    audit_release(release)
    return (modal.Image.from_registry('nvidia/cuda:13.0.2-devel-ubuntu24.04', add_python='3.12')
            .apt_install('util-linux', 'procps')
            .pip_install('uv==0.9.25')
            .add_local_file(GAME / 'requirements-universal-lock.txt', '/tmp/hex-requirements.txt', copy=True)
            .run_commands('uv python install 3.12.9 --install-dir /opt/exact-python',
                          f'uv pip install --python {PYTHON} --require-hashes -r /tmp/hex-requirements.txt'))


ENV = {'PYTHONPATH': '/opt/hex', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
       'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}


def research_image(release):
    """Where researcher code runs (training and leases): the engine, no MoHex source and no MoHex binary."""
    release = Path(release)
    return (_base(release)
            .add_local_dir(release / 'frozen', '/opt/hex', copy=True, ignore=['**/__pycache__/**', 'vendor/**'])
            .env(ENV)
            .run_commands('mkdir -p /job/input /job/output /private', 'chmod 700 /private',
                          'test ! -e /opt/hex/vendor', 'chmod -R a-w /opt/hex')
            .workdir('/job'))


def image_for(release):
    """The evaluation image (practice and grading): the engine plus MoHex, which only root can read or run."""
    release = Path(release)
    return (_base(release)
            .apt_install('cmake', 'g++', 'make', 'libboost-all-dev', 'libdb-dev')
            .add_local_dir(release / 'frozen', '/opt/hex', copy=True, ignore=['**/__pycache__/**'])
            .env(dict(ENV, HEX_MOHEX=MOHEX))
            .run_commands('mkdir -p /job/input /job/output /private', 'chmod 700 /private',
                          'HEX_BUILD_JOBS=8 sh /opt/hex/simulators/hex/build_mohex.sh '
                          '/opt/hex/vendor/engines/benzene /opt/mohex-build',
                          f'test -x {MOHEX}', 'chmod -R a-w /opt/hex /opt/mohex-build',
                          # The evaluator runs as root; the policy it evaluates runs as uid 65534.
                          'chmod -R go-rwx /opt/mohex-build /opt/hex/vendor')
            .workdir('/job'))


def allocation_info(app_id, sandbox_id):
    """Pinned Modal 1.5.2 provider metadata; no guest-reported clock is trusted."""
    async def fetch():
        from modal.client import _Client
        from modal_proto import api_pb2
        client = await _Client.from_env()
        before = None
        while True:
            request = api_pb2.SandboxListRequest(app_id=app_id, environment_name=ENVIRONMENT, include_finished=True,
                                                 **({'before_timestamp': before} if before is not None else {}))
            response = await client.stub.SandboxList(request)
            for item in response.sandboxes:
                if item.id == sandbox_id:
                    task = item.task_info
                    return {'provider_id': sandbox_id, 'task_id': task.id, 'created_at': item.created_at,
                            'started_at': task.started_at, 'finished_at': task.finished_at, 'gpu_type': task.gpu_type}
            if not response.sandboxes:
                return None
            before = response.sandboxes[-1].created_at
    # The SDK's own event loop: asyncio.run() would reuse a cached client bound to another loop.
    from modal._utils.async_utils import synchronizer
    return synchronizer.create_blocking(fetch)()


def unstarted(info):
    """Modal's own record of this sandbox exists and shows no start: nothing was allocated, nothing is charged."""
    return bool(info and info.get('provider_id')) and workers.never_started(info)


class Lease:
    """One running lease: the sandbox, its commands and the host's record of them. Callers hold `lock`."""

    def __init__(self, sb, target, shape, volume=None):
        self.sb = sb
        self.target = Path(target)
        self.shape = shape
        self.volume = volume  # the session's hand-off Volume, mounted root-only at /handoff
        self.closed = False
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
        python = str(Path(PYTHON).parent)
        environment = (f'HOME={WORK} PATH={python}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin '
                       f'PYTHONPATH=/opt/hex:{WORK}/training OMP_NUM_THREADS={self.shape["cpu_cores"]} '
                       f'MKL_NUM_THREADS={self.shape["cpu_cores"]} OPENBLAS_NUM_THREADS={self.shape["cpu_cores"]}')
        # $0 is the researcher's command, passed as an argument so nothing of it is interpreted as root.
        # `--wait`: when the command is a process group leader setsid forks, and without it the command would be
        # reported as exited (code 0) the moment it started (seen live in Polytopia, 2026-11-14).
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

    def seal(self, target, path, code, config, job):
        """Seal the researcher's network as a policy package and publish it to the hand-off Volume as `job`.

        Runs on the worker, under the lock. Returns how the archive reaches the host: 'volume' once the worker
        has published and committed it, or 'live' (already copied out of the worker) when the publish failed.
        """
        if not path.startswith('/') or '\0' in path:
            raise ValueError('--path is an absolute directory on the lease, such as /scratch/run1')
        source = shlex.quote(path.rstrip('/') or '/')
        caps = {'weights.pt': DEFAULT['artifact_bytes_cap'], 'model.json': 65536}
        copies = ' && '.join(f'{UNPRIVILEGED} head -c {cap + 1} -- {source}/{name} > /job/output/{name}'
                             for name, cap in caps.items())
        status, _, err = self._root('rm -rf /job/input /job/output /job/export /job/export.tar /job/trusted-export && '
                                    'mkdir -p /job/input /job/output && ' + copies, timeout=PACKAGE_SECONDS)
        if status != 0:
            raise ValueError(f'{path} needs a readable weights.pt and model.json: ' + err[-300:].strip())
        for name in EDITABLE:
            self.sb.filesystem.write_text(code[name], '/job/input/' + name)
        self.sb.filesystem.write_text(json.dumps(config), '/job/input/config.json')
        self.sb.filesystem.write_text(json.dumps({'lease_path': path}), '/job/input/request.json')
        status, out, err = self._root(f'{PYTHON} -m simulators.hex.autoresearch.export_entry', timeout=PACKAGE_SECONDS)
        if status != 0:
            (target / 'export-error.log').write_text(err[-10000:])
            raise ValueError('Not a valid policy package: ' + err[-600:].strip())
        try:  # the identity the export printed: whatever reaches the host must be this package
            identity = json.loads(out.strip().splitlines()[-1])['bundle_sha256']
        except Exception:
            identity = None
        write_json(target / 'request.json', {'kind': 'package', 'path': path, 'config': config, 'code': code})
        record = {'job': job, 'volume': self.volume.name, 'identity': identity}
        process = self.sb.exec(*handoff.publish_command(job, {'bundle.tar': '/job/export.tar'}, python=PYTHON),
                               timeout=PACKAGE_SECONDS)
        published = process.wait()
        if published == 0:
            return dict(record, handoff='volume')
        reason = f'In-worker publish exited {published}: ' + process.stderr.read()[-300:].strip()
        self.sb.filesystem.copy_to_local('/job/export.tar', target / 'bundle.tar')
        return dict(record, handoff='live', fallback_reason=reason)

    def collect(self, target, sealed, attempts=6, wait=5.0):
        """Bring a sealed package to the host and verify it. Without the lock: a Volume read on a slow host link
        holds no lease command (or the lease's own teardown) back. Only when the package is not in the Volume and
        the lease still serves is it read out of the worker, as before."""
        record = dict(sealed)
        if record['handoff'] == 'volume':
            try:
                self.volume.collect(record['job'], target, names=['bundle.tar'], limit=ARCHIVE_BYTES,
                                    attempts=attempts, wait=wait)
            except handoff.NotPublished as exc:
                with self.lock:
                    if self.closed:
                        raise RuntimeError('The package is not in the hand-off Volume and the lease has ended') from exc
                    self.sb.filesystem.copy_to_local('/job/export.tar', target / 'bundle.tar')
                record.update(handoff='live', fallback_reason='NotPublished: ' + str(exc))
        unpack_bundle(target / 'bundle.tar', target / 'bundle')
        identity = verify(target / 'bundle', record['identity'], check_tensors=False)['bundle_sha256']
        write_json(target / 'handoff.json', record)
        return target / 'bundle', identity

    def close(self):
        """End every researcher process, keep the last output of each command, commit /scratch."""
        self.closed = True
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

    handoff_wait = 5.0  # between reads of a hand-off Volume that does not show a commit yet

    def __init__(self, release, app, export_reserve_seconds=15):
        self.release = Path(release)
        self.app = app
        self.image = image_for(release)
        self.research = research_image(release)
        self.export_reserve_seconds = export_reserve_seconds
        self.live = {}  # lease name -> Lease, while its worker is serving commands
        self.lock = threading.Lock()

    def scratch(self, root):
        """The session's scratch Volume, created with its first lease and recorded in the session directory.

        It is never deleted here: it holds the session's checkpoints and is collected for the run archive.
        """
        import modal
        path = Path(root) / 'scratch-volume.json'
        with self.lock:
            if not path.exists():
                write_json(path, {'volume': 'hex-scratch-' + uuid.uuid4().hex, 'environment': ENVIRONMENT,
                                  'mount': SCRATCH, 'created_unix': time.time()}, exclusive=True)
            name = json.loads(path.read_text())['volume']
        return modal.Volume.from_name(name, environment_name=ENVIRONMENT, create_if_missing=True, version=2)

    def handoff_volume(self, root):
        """The backend root's private hand-off Volume: a research session's `research/` and its fresh final's
        `final/` each have their own. Unlike /scratch, only root inside a worker can reach it."""
        return handoff.SessionVolume('hex', root)

    def _sandbox(self, name, grant, kind, shape, root):
        import modal
        resources = dict(cpu=(shape['cpu_cores'],) * 2, memory=(shape['memory_mib'],) * 2)
        if shape['gpu']:
            resources['gpu'] = GPU
        volumes = {SCRATCH: self.scratch(root)} if kind == 'lease' else {}
        if kind != 'practice':  # training runs and leases hand their results back through it
            volumes.update(self.handoff_volume(root).mounts())
        return modal.Sandbox.create('/bin/sleep', '86400', app=self.app,
                                    image=self.image if kind == 'practice' else self.research,
                                    timeout=math.ceil(grant), block_network=True, include_oidc_identity_token=False,
                                    secrets=[], volumes=volumes,
                                    environment_name=ENVIRONMENT, tags={'hex_job': name}, **resources)

    def execute(self, ledger, job, stop=None):
        name = job['name']
        payload = json.loads(job['payload'])
        grant = job['grant']
        kind = payload.get('kind')
        practice = kind == 'practice'
        shape = payload.get('shape') or (PRACTICE_SHAPE if practice else TRAIN_SHAPE)
        sb = started = artifact = identity = error = never = None
        status = 'failed'
        provider_receipt = {}
        queued_out = False  # gave up (or was stopped) while Modal had not started the worker
        published = None  # a training run's hand-off folder, once its worker may publish there on its own
        stopping = False  # the researcher stopped a training run: it ends `stopped`, keeping its last network
        transfer = {}  # how the result reached the host: handoff 'volume' or 'live', and why a fallback happened
        target = ledger.root / 'jobs' / name
        target.mkdir(parents=True, exist_ok=False)
        write_json(target / 'request.json', payload, exclusive=True)
        try:
            sb = self._sandbox(name, grant, kind, shape, ledger.root)
            write_json(target / 'provider.json', {'sandbox_id': sb.object_id, 'app_id': self.app.app_id}, exclusive=True)
            # Queued time (created, no GPU yet) is never charged; wait for Modal's start or a stop.
            info = workers.await_start(lambda: allocation_info(self.app.app_id, sb.object_id), stop=stop,
                                       sleep=time.sleep, clock=time.time)
            if info is None:
                queued_out = True
                raise RuntimeError(('Stopped while' if stop and stop() else
                                    f'Gave up after {workers.START_PATIENCE_SECONDS:.0f} s')
                                   + ' waiting in the provider queue for a GPU')
            started = info['started_at']
            ledger.start(name, sb.object_id, started)
            if shape['gpu'] and GPU not in info['gpu_type'].upper():
                raise RuntimeError(f'Requested {GPU} but provider reported {info["gpu_type"]}')
            deadline = started + grant
            if kind == 'lease':
                self._serve_lease(sb, name, payload, shape, target, deadline, stop, ledger)
                status = 'stopped' if stop and stop() else 'complete'
                return {'lease_ended': True}  # finally still charges and closes the allocation
            if practice:
                self.stage_evaluation(sb, payload['bundle'], payload['panel'], payload['seeds'], True)
                remaining = deadline - time.time() - 5
                if remaining < 1:
                    raise RuntimeError('Practice setup exhausted allocation')
                limit = max(1, int(remaining))
                # Modal kills the evaluator at `limit` with no stderr; that is the grant running out, not a crash.
                killed_at = time.time() + limit - 1
                proc = sb.exec(PYTHON, '-m', 'simulators.hex.autoresearch.evaluate_entry',
                               '--request', '/private/evaluation.json', timeout=limit)
                while proc.poll() is None and time.time() < deadline - 3:
                    ledger.milestones()
                    if stop and stop():
                        break
                    time.sleep(.5)
                timed_out = (f'Practice did not finish within its grant ({grant:.0f} s); no partial score awarded. '
                             'Request a longer practice grant.')
                if proc.poll() is None:
                    if stop and stop():
                        raise RuntimeError('Practice stopped before it finished; no partial score awarded')
                    raise RuntimeError(timed_out)
                if proc.returncode != 0:
                    detail = proc.stderr.read()[-1000:].strip()
                    if time.time() >= killed_at and not detail:
                        raise RuntimeError(timed_out)
                    raise RuntimeError(f'Practice evaluation failed (exit {proc.returncode}): '
                                       + (detail or 'no error output'))
                write_json(target / 'practice.json', json.loads(sb.filesystem.read_text('/job/evaluation-public.json')))
                status = 'complete'
                return {'practice_complete': True}  # finally still charges and closes the allocation
            prepare(sb)  # /handoff is root-only before the researcher's trainer starts
            for filename in EDITABLE:
                sb.filesystem.write_text(payload['code'][filename], '/job/input/' + filename)
            sb.filesystem.write_text(json.dumps(payload['config']), '/job/input/config.json')
            sb.filesystem.write_text(json.dumps({'seed': payload['seed']}), '/job/input/request.json')
            remaining = deadline - time.time() - self.export_reserve_seconds
            if remaining < 2:
                raise RuntimeError('Allocation consumed by setup before the learner could start')
            key = handoff_job(name, sb.object_id)
            # Train, export and publish in one exec: the worker finishes the sequence even if the host is gone.
            proc = sb.exec('/bin/sh', '-c', train_script(remaining, key), timeout=max(1, int(deadline - time.time() - 2)))
            published = key
            transfer = {'job': key, 'volume': self.handoff_volume(ledger.root).name}
            code, killed = None, False
            while True:
                try:
                    code = proc.poll()
                except Exception as exc:  # a slow or broken host link: the worker carries on without the host
                    transfer['link_error'] = type(exc).__name__ + ': ' + str(exc)[:300]
                if code is not None or time.time() >= deadline:
                    break
                ledger.milestones()
                for remote, local, cap in (('/job/output/progress.jsonl', 'progress.jsonl', 1000000),
                                           ('/job/trainer.log', 'trainer.log', 1000000)):
                    try:
                        text = sb.filesystem.read_text(remote)
                        (target / local).write_text(text if len(text) <= cap else text[text.find('\n', len(text) - cap) + 1:])
                    except Exception:
                        pass
                if stop and stop() and not killed:
                    stopping = True
                    try:  # end the researcher's processes; the worker's own sequence then exports and publishes
                        sb.exec('/bin/sh', '-c', 'pkill -KILL -u 65534 || true', timeout=2).wait()
                        killed = True
                    except Exception:
                        pass  # asked again on the next pass
                time.sleep(2)
            if code not in (None, 0):
                # The worker says its publish failed and it is still up: read the result out of it, as before.
                try:
                    detail = proc.stderr.read()[-300:].strip()
                except Exception:
                    detail = ''
                published = None
                transfer.update(handoff='live', fallback_reason=f'In-worker publish exited {code}: {detail}')
                artifact, identity = self._live_result(sb, target, deadline)
            # Otherwise the result is read from the hand-off Volume once the worker has ended (below).
            status = 'stopped' if stopping else 'complete'
        except BaseException as exc:
            status = 'failed'
            error = type(exc).__name__ + ': ' + str(exc)
            (target / 'host-exception.log').write_text(traceback.format_exc())
        finally:
            final = {}
            if sb is not None:
                failures = []

                def terminate():
                    try:
                        sb.terminate(wait=True)
                    except Exception as exc:
                        failures.append(exc)
                        raise
                # Settles as soon as Modal's record is final: finished (it ran) or still unstarted after termination.
                final = workers.abandon(terminate, lambda: allocation_info(self.app.app_id, sb.object_id),
                                        sleep=time.sleep, clock=time.time)
                if failures:
                    error = (error or '') + '; termination: ' + str(failures[0])
            if final.get('started_at') and final.get('finished_at'):
                provider_receipt = final
                if started is None:  # it started while being abandoned: charged from Modal's receipt
                    ledger.start(name, sb.object_id, final['started_at'])
                if published and artifact is None:
                    status, artifact, identity, error = self._settle_published(
                        ledger, name, sb.object_id, final, published, target, 'stopped' if stopping else 'complete',
                        error, transfer)
                else:
                    ledger.finish(name, final['finished_at'], status, artifact, identity, error)
            elif started is None and unstarted(final):
                # Created but never started: no GPU was allocated, so nothing is charged and nothing stays held.
                stopped = bool(queued_out and stop and stop())
                status = 'stopped' if stopped else 'failed'
                error = workers.NEVER_STARTED + '. ' + (error or 'Worker setup failed before the provider started it')
                provider_receipt = final
                ledger.abandon_unstarted(name, final, error, stopped=stopped)
                never = workers.NeverStarted(final, error)
            else:
                provider_receipt = final
                if published and artifact is None:
                    transfer.update(handoff='volume', pending='recovery collects the result from the hand-off Volume')
                ledger.uncertain(name, error or 'Provider allocation end is unavailable; reservation retained')
            write_json(target / 'allocation.json', {'provider': provider_receipt, 'status': status, 'error': error,
                                                    'budget_scope': ledger.config.get('allocation_rule',
                                                                                      DEFAULT['allocation_rule']),
                                                    'artifact_sha256': identity, 'guest_clock_trusted': False,
                                                    'never_started': never is not None,
                                                    'handoff': transfer.get('handoff'), 'transfer': transfer})
        if never is not None:
            raise never
        return next(x for x in ledger.jobs() if x['name'] == name)

    def _settle_published(self, ledger, name, sandbox_id, receipt, key, target, status, error, transfer):
        """After the worker has ended: read its published result from the hand-off Volume, outside its paid time
        and with no deadline, then settle the job from the provider receipt as before.

        While the host reads, the job holds its grant (as `uncertain`) rather than a running interval that would
        grow with a slow link; a host lost meanwhile leaves it to `recover`, which reads the same folder.
        """
        ledger.uncertain(name, 'The worker has ended; collecting its result from the session hand-off Volume')
        artifact = identity = None
        transfer['handoff'] = 'volume'
        try:
            artifact, identity, transfer['files'] = collect_published(self.handoff_volume(ledger.root), key, target,
                                                                      wait=self.handoff_wait)
        except BaseException as exc:
            status = 'failed'
            transfer['error'] = type(exc).__name__ + ': ' + str(exc)
            error = '; '.join(filter(None, (error, transfer['error'])))
            (target / 'handoff-exception.log').write_text(traceback.format_exc())
        ledger.reconcile(name, sandbox_id, receipt['started_at'], receipt['finished_at'], error, status=status,
                         artifact=artifact, artifact_sha=identity)
        return status, artifact, identity, error

    def _live_result(self, sb, target, deadline):
        """The fallback when the worker could not publish: read the result out of the still running worker,
        inside its paid time, as the host used to. Runs the export itself if the worker's sequence did not."""
        for remote, local in (('/job/supervisor.json', 'supervisor.json'), ('/job/trainer.log', 'trainer.log'),
                              ('/job/output/progress.jsonl', 'progress.jsonl')):
            try:
                (target / local).write_text(sb.filesystem.read_text(remote)[-1000000:])
            except Exception:
                pass
        try:
            exported = json.loads(sb.filesystem.read_text(STAGE + '/export.json'))['exit_code']
        except Exception:
            exported = None
        if exported is None:
            # End every unprivileged process before exporting.
            sb.exec('/bin/sh', '-c', 'pkill -KILL -u 65534 || true', timeout=2).wait()
            if time.time() > deadline - 2:
                raise RuntimeError('No time remains for validated artifact export')
            export = sb.exec(PYTHON, '-m', 'simulators.hex.autoresearch.export_entry',
                             timeout=max(1, int(deadline - time.time() - 2)))
            exported = export.wait()
            if exported != 0:
                (target / 'export-error.log').write_text(export.stderr.read()[-10000:])
        elif exported != 0:
            try:
                (target / 'export-error.log').write_text(sb.filesystem.read_text(STAGE + '/export-error.log'))
            except Exception:
                pass
        if exported != 0:
            raise RuntimeError('Trainer did not leave a valid policy package; export diagnostics retained')
        artifact = target / 'bundle'
        sb.filesystem.copy_to_local('/job/export.tar', target / 'bundle.tar')
        unpack_bundle(target / 'bundle.tar', artifact)
        return artifact, verify(artifact, check_tensors=False)['bundle_sha256']

    # --- leases -----------------------------------------------------------------------------------------

    def _serve_lease(self, sb, name, payload, shape, target, deadline, stop, ledger):
        """Hold the worker for the researcher's commands until the grant runs out or the lease is stopped."""
        prepare(sb)  # /handoff is root-only before any researcher command
        setup = sb.exec('/bin/sh', '-c', f'mkdir -p {WORK}/.ar && chmod 1777 {SCRATCH} && chown 65534:65534 {WORK} '
                        f'&& {UNPRIVILEGED} sh -c "touch {SCRATCH}/.writable && rm {SCRATCH}/.writable"', timeout=60)
        if setup.wait() != 0:
            raise RuntimeError('Lease setup failed: ' + setup.stderr.read()[-500:])
        lease = Lease(sb, target, shape, self.handoff_volume(ledger.root))
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
        """Seal weights.pt and model.json from a directory on the lease, with this source, as a policy package.

        The researcher's files are read as the unprivileged user; the sealing is the same root-side export a
        training run gets, so the tensor and size caps are enforced here too. The sealed archive travels through
        the hand-off Volume; each attempt publishes to its own folder.
        """
        lease = self._live(name)
        target = Path(target)

        def failed(exc):
            (target / 'package-error.log').write_text(type(exc).__name__ + ': ' + str(exc))
            target.rename(target.with_name(target.name + '.failed-' + uuid.uuid4().hex[:8]))
        with lease.lock:
            target.mkdir(parents=True, exist_ok=False)
            try:
                sealed = lease.seal(target, path, code, config, handoff_job(target.name, uuid.uuid4().hex[:12]))
            except BaseException as exc:
                failed(exc)
                raise
        try:
            return lease.collect(target, sealed, wait=self.handoff_wait)
        except BaseException as exc:
            failed(exc)
            raise

    def recover(self, ledger, job, stopped=False):
        """Settle one unresolved job (`queued` with a recorded worker, `running` after a lost host, or `uncertain`)
        from Modal's record alone, without creating a worker.

        Its recorded sandbox is terminated. A training run whose worker published its result to the hand-off
        Volume is settled with that result (verified as any run's is); any other worker that ran is failed. Both
        are charged their exact start->finish lease; one Modal never started is settled at 0 (`stopped` when the
        researcher stopped it). Without a usable record the job stays `uncertain` and keeps its hold. Existing
        submissions are never replaced.
        """
        name = job['name']
        target = ledger.root / 'jobs' / name
        path = target / 'provider.json'
        if not path.exists():
            if job['status'] != 'uncertain':
                ledger.uncertain(name, 'No authoritative creation receipt after controller interruption')
            return
        meta = json.loads(path.read_text())
        read = lambda: allocation_info(meta['app_id'], meta['sandbox_id'])  # noqa: E731
        info = read()
        if not info:
            ledger.uncertain(name, 'Controller recovery awaits authoritative provider metadata; hold retained')
            return
        if not info.get('finished_at'):
            # Make sure it can no longer start (or keep running) behind the ledger's back.
            info = workers.abandon(lambda: ModalProvider.terminate(meta['sandbox_id']), read,
                                   sleep=time.sleep, clock=time.time)
        if not job.get('started') and unstarted(info):
            reason = workers.NEVER_STARTED + ('. Stopped while unresolved' if stopped else '. Controller recovery')
            ledger.abandon_unstarted(name, info, reason, stopped=stopped)
            write_json(target / ('recovered-' + str(time.time_ns()) + '.json'),
                       {'provider': info, 'never_started': True, 'charged_' + ledger.unit: 0, 'created_workers': 0},
                       exclusive=True)
            return
        if info.get('started_at') and info.get('finished_at'):
            if (json.loads(job.get('payload') or '{}').get('kind') or 'train') == 'train' and recover_published(
                    self.handoff_volume(ledger.root), ledger, job, meta['sandbox_id'], info, target, stopped):
                return
            ledger.reconcile(name, meta['sandbox_id'], info['started_at'], info['finished_at'],
                             'Controller recovery: charged exact completed provider lease; unexported work is failed')
            write_json(target / ('recovered-' + str(time.time_ns()) + '.json'), info, exclusive=True)
            return
        ledger.uncertain(name, 'Controller recovery awaits authoritative provider end; hold retained')

    def recover_jobs(self, ledger):
        """Reopening a session: settle every unresolved job from Modal's records. Returns the names settled."""
        recovered = []
        for job in ledger.jobs():
            if job['status'] not in ('queued', 'running', 'uncertain'):
                continue
            self.recover(ledger, job)
            if next(j for j in ledger.jobs() if j['name'] == job['name'])['status'] not in ('queued', 'running', 'uncertain'):
                recovered.append(job['name'])
        return recovered

    @staticmethod
    def terminate(sandbox_id):
        import modal
        modal.Sandbox.from_id(sandbox_id).terminate(wait=True)

    def stage_evaluation(self, sb, bundle, panel, seeds, development):
        with tempfile.TemporaryDirectory(prefix='hex-transport-') as temporary:
            archive = Path(temporary) / 'bundle.tar'
            archive_bundle(bundle, archive)
            sb.filesystem.copy_from_local(archive, '/private/candidate.tar')
        request = dict(seeds, candidate='/private/candidate', panel=panel, development=development,
                       workers=DEFAULT['evaluation_cpu_cores'] - 1)
        sb.filesystem.write_text(json.dumps(request), '/private/evaluation.json')

    def evaluate(self, bundle, panel, seeds, development=True, timeout=7200):
        """A separate host grading allocation, never mounted into a researcher box."""
        import modal
        sb = result = None
        host_start = time.time()
        try:
            sb = modal.Sandbox.create('/bin/sleep', '86400', app=self.app, image=self.image,
                                      cpu=(DEFAULT['evaluation_cpu_cores'],) * 2,
                                      memory=(DEFAULT['evaluation_memory_mib'],) * 2, timeout=timeout,
                                      block_network=True, secrets=[], volumes={}, include_oidc_identity_token=False,
                                      environment_name=ENVIRONMENT)
            self.stage_evaluation(sb, bundle, panel, seeds, development)
            # /private is root-only. Candidate code runs as uid 65534 with only its own bundle files.
            proc = sb.exec(PYTHON, '-m', 'simulators.hex.autoresearch.evaluate_entry', '--request',
                           '/private/evaluation.json', timeout=timeout - 10)
            if proc.wait() != 0:
                raise RuntimeError('Isolated evaluation failed: ' + proc.stderr.read()[-1500:])
            result = {'public': json.loads(sb.filesystem.read_text('/job/evaluation-public.json')),
                      'private': json.loads(sb.filesystem.read_text('/private/evaluation-private.json'))}
            return result
        finally:
            if sb is not None:
                sb.terminate(wait=True)
                info = allocation_info(self.app.app_id, sb.object_id)
                if result is not None:
                    result['allocation'] = {'gpu_count': 0, 'cpu_cores': DEFAULT['evaluation_cpu_cores'],
                                            'memory_gib': DEFAULT['evaluation_memory_mib'] / 1024, 'provider': info,
                                            'host_elapsed_seconds': time.time() - host_start,
                                            'separate_from_gpu_budget': True}
