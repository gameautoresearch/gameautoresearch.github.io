"""One tournament call inside the Function container: network A vs network B on a few maps, both seats, through the
release's frozen CPU grader exactly as `Provider.evaluate` runs it in its Sandbox: A's sealed files in /candidate
(policy process uid 65534), B's in /opponent (uid 65533, mode 700, chowned by grade.py), the request in /private
(mode 700), `python /opt/v1/grade.py /private/request.json`, the result read back from /private/result.json.
Submitted Python is never imported here: each network.py runs only inside its own unprivileged /opt/v1/policy.py.

Differences from the Sandbox, none of which reach a game's moves: the container is a 1-core Function instead of a
4-core Sandbox, so OMP / OpenBLAS / XLA are pinned to one thread; files come from the image (/bundles) instead of
`copy_from_local`. Both policies are deterministic (greedy), so a game's result does not depend on timing.
"""
import json, os, resource, shutil, subprocess, time
from pathlib import Path

PLAY_FILES = ('final.npz', 'final.json', 'network.py', 'ar_arch.json')


def _fill(folder, source):
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    for child in folder.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    for name in PLAY_FILES:
        shutil.copyfile(Path(source) / name, folder / name)
        (folder / name).chmod(0o644)


class _Sampler:
    """Peak of the summed resident memory of every process in the container (the grader and both policies), from
    /proc, sampled every 0.25 s while a call runs (the cgroup's own peak is not readable in these containers)."""

    def __init__(self):
        import threading
        self.peak, self.stop = 0, threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        page = os.sysconf('SC_PAGE_SIZE')
        while not self.stop.wait(0.25):
            total = 0
            for proc in Path('/proc').iterdir():
                if proc.name.isdigit():
                    try:
                        total += int((proc / 'statm').read_text().split()[1]) * page
                    except (OSError, ValueError, IndexError):
                        pass
            self.peak = max(self.peak, total)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        return self.peak


def play(a_dir, b_dir, seeds, b_name, timeout=1700):
    """Rows of the grader's details (seed, seat = A's seat, reward from A's side, capped, turn, steps, scores)."""
    _fill('/candidate', a_dir)
    os.chown('/opponent', 0, 0)
    os.chmod('/opponent', 0o755)
    _fill('/opponent', b_dir)
    os.chmod('/candidate', 0o755)
    private = Path('/private')
    private.mkdir(exist_ok=True)
    for child in private.iterdir():
        child.unlink()
    os.chmod(private, 0o700)
    (private / 'request.json').write_text(json.dumps(dict(seeds=list(seeds), opponent_seed=0, opponent_name=b_name)))
    env = dict(os.environ, PYTHONPATH='/workspace/src', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    sampler = _Sampler()
    t = time.time()
    proc = subprocess.run(['/opt/venv/bin/python', '/opt/v1/grade.py', '/private/request.json'], cwd='/workspace',
                          env=env, capture_output=True, timeout=timeout)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    rss_peak = sampler.close()
    result = json.loads((private / 'result.json').read_text()) if (private / 'result.json').is_file() else None
    stderr = proc.stderr.decode(errors='replace')[-2500:]
    policy_err = {p.name: p.read_text(errors='replace')[-1500:] for p in private.glob('policy-*.stderr')}
    peak = None
    for path in ('/sys/fs/cgroup/memory.peak', '/sys/fs/cgroup/memory/memory.max_usage_in_bytes'):
        try:
            peak = int(Path(path).read_text().split()[0])
            break
        except (OSError, ValueError):
            pass
    return dict(exit=proc.returncode, rows=(result or {}).get('details', []), seconds=time.time() - t,
                cpu_seconds=(after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime),
                child_maxrss_kib=after.ru_maxrss, cgroup_memory_peak=peak, total_rss_peak=rss_peak,
                stderr=stderr if proc.returncode else '', policy_stderr=policy_err if proc.returncode else {})
