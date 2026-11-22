"""Modal backend: each call is one 1-core Function container (billed max(request, usage); never a Sandbox).

Images: `cpu` (default) holds only what the frozen CPU grader imports (a call peaks at ~1.7 GB; 2 GiB request);
`release` is the release's own CUDA worker image built exactly as harness/games/polytopia/provider.py builds it
(~8.5 GB summed per call; 4 GiB request; what polytopia-empty15-20261118 ran on). Games are identical on both
(README: parity). Both add the entrants' play files under /bundles and job.py. Project environment only
(game-autoresearch-project, duper).
"""
import json
import os
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = 'game-autoresearch-project'
PLAY_FILES = ('final.npz', 'final.json', 'network.py', 'ar_arch.json')
CPU_PRICE, MEMORY_PRICE = 1.31e-5, 2.22e-6          # Function list prices per core-second and GiB-second


def safe(name):
    return name.replace(':', '_')


def stage_bundles(entrants, stage):
    """Byte copies of the entrants' play files keyed by entrant name (rebuilt only when the entrant set changes)."""
    stage = Path(stage)
    marker = stage / 'READY'
    want = sorted((e['name'], e['bundle_sha256']) for e in entrants)
    if marker.is_file() and json.loads(marker.read_text()) == want:
        return stage / 'bundles'
    if stage.exists():
        shutil.rmtree(stage)
    for e in entrants:
        target = stage / 'bundles' / safe(e['name'])
        target.mkdir(parents=True)
        for f in PLAY_FILES:
            shutil.copyfile(Path(e['bundle']) / f, target / f)
    marker.write_text(json.dumps(want))
    return stage / 'bundles'


class ModalBackend:
    name = 'modal-function'

    def __init__(self, cfg, entrants, cpu=1.0, memory_mib=2048, max_containers=600, image='cpu'):
        self.cfg, self.entrants = cfg, entrants
        self.cpu, self.memory_mib, self.max_containers = cpu, memory_mib, max_containers
        self.image_kind = image
        self.app_id = None

    def cpu_image(self, modal, release, bundles):
        """Only what the frozen CPU grader imports, as daytona_backend.snapshot_image: Ubuntu 22.04, /opt/venv with
        numpy, CPU JAX 0.11.2 and the release's Rust engine wheel; no CUDA, no torch. A call peaks at ~1.4 GB instead
        of the release worker image's 4-5 GB (its CUDA JAX build is loaded by every process even on CPU)."""
        from tools.tournament.polytopia_elo.daytona_backend import JAX
        return (modal.Image.from_registry('ubuntu:22.04')
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
                .add_local_dir(release / 'runtime/engine-rs', '/build/engine-rs', copy=True, ignore=['target/**'])
                .run_commands('cd /build/engine-rs && maturin build --release -o /build/wheels '
                              '&& uv pip install /build/wheels/*.whl && rm -rf /build/engine-rs/target')
                .add_local_dir(release / 'runtime/src', '/workspace/src', copy=True, ignore=['**/__pycache__/**'])
                .add_local_dir(release / 'maps', '/results/maps', copy=True)
                .add_local_dir(release / 'v1', '/opt/v1', copy=True, ignore=['**/__pycache__/**'])
                .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cpu', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
                      'PYTHONUNBUFFERED': '1'})
                .run_commands('chmod o+x /root /root/.local /root/.local/share /root/.local/share/uv '
                              '/root/.local/share/uv/python || true',
                              'chmod -R a+rX /opt/venv /workspace /opt/v1 /results',
                              'chmod -R a-w /workspace/src /results/maps /opt/v1',
                              'mkdir -p /private /candidate /opponent; chmod 700 /private')
                .workdir('/workspace')
                .add_local_dir(str(bundles), '/bundles', copy=True)
                .add_local_file(str(HERE / 'job.py'), '/opt/elo/job.py', copy=True)
                .run_commands('chmod -R a+rX /bundles /opt/elo'))

    def __enter__(self):
        import modal
        if os.environ.get('MODAL_DISABLE_API_PROXY'):
            raise SystemExit('unset MODAL_DISABLE_API_PROXY (ephemeral apps miss heartbeats with it)')
        release = Path(self.cfg['release'])
        bundles = stage_bundles(self.entrants, Path(self.cfg['dir']) / 'stage')
        if self.image_kind == 'cpu':
            image = self.cpu_image(modal, release, bundles)
            return self._start(modal, image)
        base = modal.Image.from_dockerfile(str(release / 'base.Dockerfile'), context_dir=str(release / 'base-context'))
        image = (base
                 .add_local_dir(release / 'runtime/src', '/workspace/src', copy=True, ignore=['**/__pycache__/**'])
                 .add_local_dir(release / 'runtime/experiments', '/workspace/experiments', copy=True)
                 .add_local_dir(release / 'maps', '/results/maps', copy=True)
                 .add_local_dir(release / 'v1', '/opt/v1', copy=True, ignore=['**/__pycache__/**'])
                 .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cpu', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
                       'POLYTOPIA_PC_BUILD': '/ar/engine-build', 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
                       'PYTHONUNBUFFERED': '1'})
                 .run_commands('chmod o+x /root /root/.local /root/.local/share /root/.local/share/uv '
                               '/root/.local/share/uv/python',
                               'chmod -R a-w /workspace/src /workspace/experiments /results/maps /opt/v1',
                               'mkdir -p /private /candidate /opponent; chmod 700 /private')
                 .workdir('/workspace')
                 .add_local_dir(str(bundles), '/bundles', copy=True)
                 .add_local_file(str(HERE / 'job.py'), '/opt/elo/job.py', copy=True)
                 .run_commands('chmod -R a+rX /bundles /opt/elo'))
        return self._start(modal, image)

    def _start(self, modal, image):
        self.app = modal.App(f"polytopia-elo-{self.cfg['name']}"[:60], image=image)

        @self.app.function(cpu=self.cpu, memory=self.memory_mib, timeout=1800, max_containers=self.max_containers,
                           block_network=True, serialized=True,
                           retries=modal.Retries(max_retries=2, initial_delay=2.0, backoff_coefficient=1.0))
        def play(job):
            import os as _os, sys as _sys, time as _time
            _sys.path.insert(0, '/opt/elo')
            import job as runner
            if _time.time() > job['hard_deadline']:
                return dict(job=job, rows=[], skipped='past hard deadline')
            res = runner.play('/bundles/' + job['a'].replace(':', '_'), '/bundles/' + job['b'].replace(':', '_'),
                              job['seeds'], job['b'])
            res.update(job=job, container=_os.environ.get('MODAL_TASK_ID'))
            return res

        self.play = play
        self._outputs = modal.enable_output()
        self._outputs.__enter__()
        self._run = self.app.run(environment_name=ENV)
        self.app_id = self._run.__enter__().app_id
        return self

    def map(self, jobs):
        """Results as they finish (an exception object for a call that failed after its retries)."""
        if not jobs:  # Function.map([]) never returns
            return iter(())
        return self.play.map(jobs, order_outputs=False, return_exceptions=True)

    def cost(self, call_seconds, cpu_seconds, wall_seconds):
        return (max(call_seconds * self.cpu, cpu_seconds) * CPU_PRICE
                + call_seconds * self.memory_mib / 1024 * MEMORY_PRICE)

    def info(self):
        return dict(backend=self.name, image=self.image_kind, environment=ENV, app_id=self.app_id, cpu=self.cpu,
                    memory_mib=self.memory_mib,
                    max_containers=self.max_containers)

    def __exit__(self, *exc):
        try:
            self._run.__exit__(*exc)
        finally:
            self._outputs.__exit__(*exc)
        return False
