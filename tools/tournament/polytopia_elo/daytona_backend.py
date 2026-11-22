"""Daytona backend: a pool of CPU sandboxes, one call at a time each, paid from the Daytona organization's credits.

Resources it creates, all labelled `polytopia-elo=<tournament name>`:
  snapshot  `polytopia-elo-<release sha[:12]>`: Ubuntu 22.04, the release's Python 3.12 venv layout (/opt/venv with
            numpy, CPU JAX 0.11.2, zstandard and the release's Rust engine wheel), the release's runtime/src, v1 and
            maps, job.py and daytona_worker.py. Built once per release (minutes: it compiles the engine), then reused.
            Only what the frozen CPU grader imports; no CUDA, no torch.
  volume    `polytopia-elo-<tournament name>`: the entrants' play files (final.npz, final.json, network.py,
            ar_arch.json), uploaded once, mounted at /bundles in every worker. Deleted at the end.
  sandboxes `--workers` of them from the snapshot (its resources: 2 vCPU, 3 GiB, 10 GiB disk by default), network
            blocked, ephemeral, auto-stop disabled (Daytona's auto-stop fires even while a process runs) and a TTL as a
            backstop; each runs one call at a time as root (job.py drops each network to its own uid). Deleted at the
            end, and any left over from an earlier run of the same tournament are deleted first.
Daytona bills a sandbox's whole allocation while it exists (list $0.0504 / vCPU-h, $0.0162 / GiB-h): about $0.10 per
1,000 games against about $0.06 on a Modal Function, so it is worth it while organization credits pay for it. Games are
identical to the Modal backend's (README: parity).

Credentials: the Daytona CLI's own login (its config file), never copied anywhere; refresh it with
`daytona organization list` (or `daytona login`) when it expires. The organization is the CLI's active one.
"""
import concurrent.futures as cf
import json
import queue
import shutil
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLI_CONFIG = Path.home() / 'Library/Application Support/daytona/config.json'
PLAY_FILES = ('final.npz', 'final.json', 'network.py', 'ar_arch.json')
VCPU_HOUR, GIB_HOUR = 0.0504, 0.0162
JAX = 'jax==0.11.2'      # the release worker image's JAX (base.Dockerfile), CPU build


def client():
    from daytona import Daytona, DaytonaConfig
    c = json.loads(CLI_CONFIG.read_text())
    p = next(x for x in c['profiles'] if x['id'] == c['activeProfile'])
    return Daytona(DaytonaConfig(jwt_token=p['api']['token']['accessToken'], organization_id=p['activeOrganizationId'],
                                 api_url=p['api']['url'])), p['activeOrganizationId']


def snapshot_image(release):
    from daytona import Image
    release = Path(release)
    return (Image.base('ubuntu:22.04')
            .env({'DEBIAN_FRONTEND': 'noninteractive'})
            .run_commands(
                'apt-get update && apt-get install -y --no-install-recommends build-essential curl pkg-config '
                'ca-certificates procps && rm -rf /var/lib/apt/lists/*',
                'curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh',
                'uv python install 3.12 && uv venv /opt/venv --python 3.12',
                'curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable')
            .env({'VIRTUAL_ENV': '/opt/venv',
                  'PATH': '/opt/venv/bin:/root/.cargo/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'})
            .run_commands(f"uv pip install 'numpy>=2.1' 'maturin>=1.7,<2' zstandard '{JAX}'")
            .add_local_dir(release / 'runtime/engine-rs', '/build/engine-rs')
            .run_commands('cd /build/engine-rs && maturin build --release -o /build/wheels && uv pip install /build/wheels/*.whl',
                          'rm -rf /build/engine-rs/target')
            .add_local_dir(release / 'runtime/src', '/workspace/src')
            .add_local_dir(release / 'v1', '/opt/v1')
            .add_local_dir(release / 'maps', '/results/maps')
            .add_local_file(HERE / 'job.py', '/opt/elo/job.py')
            .add_local_file(HERE / 'daytona_worker.py', '/opt/elo/worker.py')
            .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cpu', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
                  'PYTHONUNBUFFERED': '1'})
            .run_commands('find /workspace/src /opt/v1 -name __pycache__ -prune -exec rm -rf {} +',
                          'chmod o+x /root /root/.local /root/.local/share /root/.local/share/uv '
                          '/root/.local/share/uv/python || true',
                          'chmod -R a+rX /opt/venv /workspace /opt/v1 /opt/elo /results',
                          'chmod -R a-w /workspace/src /results/maps /opt/v1',
                          'mkdir -p /private /candidate /opponent /bundles; chmod 700 /private')
            .workdir('/workspace'))


class DaytonaBackend:
    name = 'daytona'

    def __init__(self, cfg, entrants, workers=120, cpu=2, memory_gib=3, disk_gib=10, ttl_hours=6.0, log=print):
        self.cfg, self.entrants = cfg, entrants
        self.workers, self.cpu, self.memory_gib, self.disk_gib = workers, cpu, memory_gib, disk_gib
        self.ttl_minutes = int(ttl_hours * 60) + 30
        self.log = log
        self.label = {'polytopia-elo': self.cfg['name'][:60]}
        self.sandboxes = []
        self.started = None

    # ---- resources
    def ensure_snapshot(self):
        from daytona import CreateSnapshotParams
        from daytona.common.sandbox import Resources
        from simulators.polytopia.autoresearch.v1 import core
        sha = core.verify(self.cfg['release'])
        name = f'polytopia-elo-{sha[:12]}-c{self.cpu}m{self.memory_gib}'
        def state():
            try:
                return str(self.daytona.snapshot.get(name).state).lower().rsplit('.', 1)[-1]
            except Exception:
                return None
        current = state()
        if current in ('error', 'build_failed'):
            self.log(f'snapshot {name} is {current}; rebuilding')
            self.daytona.snapshot.delete(self.daytona.snapshot.get(name))
            current = None
        if current is None:
            self.log(f'building snapshot {name} (compiles the engine; several minutes)')
            try:
                # No log streaming: the build-log proxy refuses the CLI's token (403); the build continues server-side.
                self.daytona.snapshot.create(CreateSnapshotParams(
                    name=name, image=snapshot_image(self.cfg['release']),
                    resources=Resources(cpu=self.cpu, memory=self.memory_gib, disk=self.disk_gib)), timeout=0)
            except Exception as exc:
                if state() is None:
                    raise
                self.log(f'snapshot create returned {type(exc).__name__}; polling the build')
        deadline = time.time() + 3600
        while time.time() < deadline:
            current = state()
            if current == 'active':
                return name
            if current in ('error', 'build_failed'):
                raise RuntimeError(f'snapshot {name} build failed')
            time.sleep(20)
        raise TimeoutError(f'snapshot {name} not active after an hour ({current})')

    def upload_bundles(self, snapshot):
        """Entrants' play files onto the tournament volume, through one short-lived uploader sandbox."""
        from daytona import CreateSandboxFromSnapshotParams
        from daytona.common.filesystem import FileUpload
        from daytona.common.volume import VolumeMount
        vol = self.daytona.volume.get(f"polytopia-elo-{self.cfg['name']}"[:60], create=True)
        for _ in range(60):
            if str(getattr(vol, 'state', 'ready')).lower().endswith('ready'):
                break
            time.sleep(5)
            vol = self.daytona.volume.get(vol.name)
        self.volume = vol
        up = self.daytona.create(CreateSandboxFromSnapshotParams(
            snapshot=snapshot, labels=dict(self.label, role='uploader'), os_user='root', network_block_all=True,
            ephemeral=True, auto_stop_interval=0, ttl_minutes=90,
            volumes=[VolumeMount(volume_id=vol.id, mount_path='/bundles')]), timeout=600)
        try:
            files = []
            for e in self.entrants:
                folder = '/bundles/' + e['name'].replace(':', '_')
                for f in PLAY_FILES:
                    files.append(FileUpload(source=str(Path(e['bundle']) / f), destination=f'{folder}/{f}'))
            for i in range(0, len(files), 200):
                up.fs.upload_files(files[i:i + 200])
            check = up.process.exec('ls /bundles | wc -l', timeout=120)
            if int(check.result.strip().splitlines()[-1]) != len(self.entrants):
                raise RuntimeError(f'volume holds {check.result.strip()} bundles, expected {len(self.entrants)}')
        finally:
            up.delete()

    def sweep(self):
        from daytona import ListSandboxesQuery
        left = list(self.daytona.list(ListSandboxesQuery(labels=self.label)))
        for s in left:
            try:
                s.delete()
            except Exception:
                pass
        return len(left)

    def __enter__(self):
        from daytona import CreateSandboxFromSnapshotParams
        from daytona.common.volume import VolumeMount
        self.daytona, self.organization = client()
        swept = self.sweep()
        if swept:
            self.log(f'deleted {swept} leftover sandboxes of this tournament')
        snap = self.ensure_snapshot()
        self.upload_bundles(snap)
        self.started = time.time()

        def make(i):
            return self.daytona.create(CreateSandboxFromSnapshotParams(
                snapshot=snap, labels=dict(self.label, role='worker'), os_user='root', network_block_all=True,
                ephemeral=True, auto_stop_interval=0, ttl_minutes=self.ttl_minutes,
                volumes=[VolumeMount(volume_id=self.volume.id, mount_path='/bundles')]), timeout=900)
        with cf.ThreadPoolExecutor(16) as pool:
            for f in cf.as_completed([pool.submit(make, i) for i in range(self.workers)]):
                try:
                    self.sandboxes.append(f.result())
                except Exception as exc:
                    self.log(f'worker create failed: {type(exc).__name__}: {str(exc)[:200]}')
        if not self.sandboxes:
            raise RuntimeError('no Daytona worker could be created')
        self.log(f'{len(self.sandboxes)} Daytona workers up ({self.cpu} vCPU, {self.memory_gib} GiB each)')
        return self

    # ---- work
    def _call(self, sandbox, job):
        spec = json.dumps(dict(a=job['a'], b=job['b'], seeds=job['seeds']))
        if time.time() > job['hard_deadline']:
            return dict(job=job, rows=[], skipped='past hard deadline')
        r = sandbox.process.exec("/opt/venv/bin/python /opt/elo/worker.py '" + spec.replace("'", "'\\''") + "'",
                                 timeout=1800)
        out = [l for l in (r.result or '').splitlines() if l.startswith('RESULT ')]
        if r.exit_code != 0 or not out:
            raise RuntimeError(f'worker exit {r.exit_code}: {(r.result or "")[-1500:]}')
        res = json.loads(out[-1][len('RESULT '):])
        res.update(job=job, container=sandbox.id)
        return res

    def map(self, jobs):
        """Results as they finish; one call at a time per sandbox; a failed call is retried twice on another worker."""
        todo = queue.Queue()
        for j in jobs:
            todo.put((j, 0))
        results = queue.Queue()
        remaining = [len(jobs)]
        lock = threading.Lock()

        def worker(sandbox):
            while True:
                with lock:
                    if remaining[0] == 0:
                        return
                try:
                    job, attempt = todo.get(timeout=2)
                except queue.Empty:
                    continue
                try:
                    results.put(self._call(sandbox, job))
                except Exception as exc:
                    if attempt < 2:
                        todo.put((job, attempt + 1))
                        continue
                    results.put(exc)
                with lock:
                    remaining[0] -= 1
        threads = [threading.Thread(target=worker, args=(s,), daemon=True) for s in self.sandboxes]
        for t in threads:
            t.start()
        for _ in range(len(jobs)):
            yield results.get()

    def cost(self, call_seconds, cpu_seconds, wall_seconds):
        """Allocation cost: every worker sandbox for the whole time it existed."""
        alive = time.time() - (self.started or time.time())
        return len(self.sandboxes) * alive / 3600 * (self.cpu * VCPU_HOUR + self.memory_gib * GIB_HOUR)

    def info(self):
        return dict(backend=self.name, organization=getattr(self, 'organization', None), workers=len(self.sandboxes),
                    cpu=self.cpu, memory_gib=self.memory_gib, label=self.label)

    def __exit__(self, *exc):
        with cf.ThreadPoolExecutor(16) as pool:
            list(pool.map(lambda s: s.delete(), self.sandboxes))
        self.sweep()
        try:
            self.daytona.volume.delete(self.volume)
        except Exception as e:
            self.log(f'volume delete failed: {e}')
        return False
