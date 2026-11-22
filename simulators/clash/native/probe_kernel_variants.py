"""Prepare/benchmark private baseline/A/B/C/BC native CUDA source capsules.

Run only in an already authorized single-L4/16CPU/64GiB worker. This command
does not provision compute. Every variant uses its retained ORIGINAL native
runtime/build/build_receipt and benchmark_cuda.py in a separate interpreter.
No loader/checkpoint monkeypatch, Java, policy or model participates. The only
added source is an IDENTICAL host-only resource-query C ABI suffix in each
cuda_engine.cu; original device code and B's scratch state are preserved.

--prepare-only verifies/extracts/combines/archives private sources and validates
frozen data using stdlib subprocesses, without importing Torch or compiling CUDA.
The 1600-second total guard includes preparation, canonical builds, parity,
warmups and timing. All outcomes/partial failures remain in the new output dir;
the external worker's complete allocation receipt remains authoritative.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import argparse
import hashlib
import io
import json
import os
import signal
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

ARCHIVE_SHA256 = '05b5a075bc9fa8c4707f688f80452185477954eae8fee4854156e83c88bd5364'
DEFAULT_SOURCE_ROOT = Path('/artifacts/native-registers-compile-1')
TAPE = Path('/workspace/evidence/clash-implementation/native-performance-20261108/cpu-baseline-1/generation/tape.npz')
WORKSPACE_EVIDENCE = Path('/workspace/evidence')
VARIANTS = ('baseline', 'A', 'B', 'C', 'BC')
PREFIX = 'integrations/clash/native/'
A_CHANGED = {PREFIX + 'holder.hpp'}
B_CHANGED = {PREFIX + name for name in ('holder.hpp', 'world.hpp', 'world_impl.hpp')}
C_CHANGED = {PREFIX + name for name in ('cuda_engine.cu', 'pathfinding/grid.hpp', 'pathfinding/movement.hpp', 'pathfinding/world_queries.hpp')}
RESOURCE_FIELDS = ('version', 'registers_per_thread', 'local_bytes_per_thread', 'static_shared_bytes',
                   'constant_bytes', 'kernel_max_threads_per_block', 'binary_version', 'ptx_version',
                   'device_index', 'multiprocessors', 'warp_size', 'max_threads_per_multiprocessor',
                   'registers_per_multiprocessor', 'registers_per_block', 'device_max_threads_per_block',
                   'max_blocks_per_multiprocessor', 'shared_bytes_per_multiprocessor', 'shared_bytes_per_block',
                   'compute_major', 'compute_minor', 'max_dynamic_shared_bytes', 'preferred_shared_carveout',
                   'active_blocks_at_1', 'active_blocks_at_4', 'active_blocks_at_8', 'active_blocks_at_16', 'active_blocks_at_32')
RESOURCE_SUFFIX = r'''
// BEGIN identical host-only kernel resource query; no device behavior changes.
extern "C" int cn_probe_step_resources(long long* out,int capacity){
 if(!out||capacity<27)return int(cudaErrorInvalidValue);
 cudaFuncAttributes a{};cudaDeviceProp p{};int device=0;
 auto code=cudaGetDevice(&device);if(code!=cudaSuccess)return int(code);
 code=cudaGetDeviceProperties(&p,device);if(code!=cudaSuccess)return int(code);
 code=cudaFuncGetAttributes(&a,step_worlds);if(code!=cudaSuccess)return int(code);
 out[0]=1;out[1]=a.numRegs;out[2]=a.localSizeBytes;out[3]=a.sharedSizeBytes;
 out[4]=a.constSizeBytes;out[5]=a.maxThreadsPerBlock;out[6]=a.binaryVersion;out[7]=a.ptxVersion;
 out[8]=device;out[9]=p.multiProcessorCount;out[10]=p.warpSize;
 out[11]=p.maxThreadsPerMultiProcessor;out[12]=p.regsPerMultiprocessor;out[13]=p.regsPerBlock;
 out[14]=p.maxThreadsPerBlock;out[15]=p.maxBlocksPerMultiProcessor;
 out[16]=p.sharedMemPerMultiprocessor;out[17]=p.sharedMemPerBlock;
 out[18]=p.major;out[19]=p.minor;out[20]=a.maxDynamicSharedSizeBytes;out[21]=a.preferredShmemCarveout;
 const int blocks[5]={1,4,8,16,32};
 for(int i=0;i<5;++i){int active=0;
  code=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,step_worlds,blocks[i],0);
  if(code!=cudaSuccess)return int(code);out[22+i]=active;}
 return int(cudaSuccess);
}
// END identical host-only kernel resource query.
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


class Guard:
    def __init__(self, seconds=1600):
        self.start = time.perf_counter()
        self.deadline = self.start + seconds
        self.active = None
        self.commands = []

    def run(self, command, *, cwd, log, env=None):
        remaining = self.deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError('1600-second total process guard exhausted')
        entry = {'command': list(map(str, command)), 'cwd': str(cwd), 'log': str(log), 'status': 'running'}
        self.commands.append(entry)
        with Path(log).open('wb') as handle:
            process = subprocess.Popen(entry['command'], cwd=cwd, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            self.active = process
            try:
                code = process.wait(timeout=remaining)
                entry.update(status='passed' if code == 0 else 'failed', returncode=code)
                if code:
                    raise RuntimeError('capsule subprocess failed; see ' + str(log))
            except subprocess.TimeoutExpired as exc:
                entry['status'] = 'total_guard_timeout'
                self.stop_active()
                raise TimeoutError('1600-second total process guard exhausted; process group terminated') from exc
            finally:
                self.active = None

    def stop_active(self):
        process = self.active
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        # The immediate Python child can exit before nvcc/ptxas grandchildren.
        # Always finish the entire process group, even if its leader exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            raise RuntimeError('failed to reap terminated capsule process group')


def source_environment(native):
    env = dict(os.environ)
    env.update(PYTHONPATH=str(native), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return env


def _safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts or '\\' in name:
        raise ValueError('unsafe capsule archive path: ' + name)
    return path


def extract_verified(source_root, out):
    archive = source_root / 'source-capsules.tar.gz'
    external = source_root / 'source-manifest.json'
    if sha(archive) != ARCHIVE_SHA256:
        raise ValueError('retained source archive SHA256 differs')
    manifest_bytes = external.read_bytes()
    manifests = json.loads(manifest_bytes)
    if set(manifests) != {'baseline', 'A', 'B', 'C'}:
        raise ValueError('retained variant manifest set differs')
    expected = {}
    for variant, files in manifests.items():
        if not isinstance(files, dict) or not files:
            raise ValueError('retained source file manifest is empty')
        for name, checksum in files.items():
            _safe_name(name)
            if not (name.startswith(PREFIX) or name == 'integrations/clash/rules.lock.json') or '.build' in PurePosixPath(name).parts:
                raise ValueError('unexpected capsule compile context: ' + name)
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise ValueError('malformed capsule SHA256')
            expected['capsules/' + variant + '/' + name] = checksum
    extracted = out / 'verified-input'
    extracted.mkdir()
    seen, files_seen, total = set(), set(), 0
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            relative = _safe_name(member.name)
            if member.name in seen:
                raise ValueError('duplicate archive member: ' + member.name)
            seen.add(member.name)
            if member.isdir():
                (extracted / str(relative)).mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.size < 0:
                raise ValueError('capsule symlink/special archive member is forbidden')
            total += member.size
            if total > 512 * 1024 ** 2:
                raise ValueError('capsule uncompressed archive exceeds explicit 512MiB bound')
            handle = tar.extractfile(member)
            if handle is None:
                raise ValueError('missing archive payload')
            payload = handle.read(member.size + 1)
            if len(payload) != member.size:
                raise ValueError('truncated archive member')
            if member.name == 'source-manifest.json':
                if payload != manifest_bytes:
                    raise ValueError('external source manifest differs from SHA-pinned archive')
            elif member.name in expected:
                if hashlib.sha256(payload).hexdigest() != expected[member.name]:
                    raise ValueError('retained source file checksum differs: ' + member.name)
                files_seen.add(member.name)
            else:
                raise ValueError('unexpected unmanifested archive file: ' + member.name)
            target = extracted / str(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    if files_seen != set(expected) or 'source-manifest.json' not in seen:
        raise ValueError('retained archive omits manifest files')
    shutil.copyfile(external, out / 'retained-source-manifest.json')
    return extracted / 'capsules', manifests


def private_archive(out, capsules, manifests):
    target = out / 'prepared-capsules-private.tar.gz'
    with tarfile.open(target, 'w:gz', compresslevel=1) as archive:
        for variant in VARIANTS:
            for name, checksum in sorted(manifests[variant].items()):
                payload = (capsules / variant / name).read_bytes()
                if hashlib.sha256(payload).hexdigest() != checksum:
                    raise ValueError('prepared capsule changed while archiving: ' + name)
                member = tarfile.TarInfo('capsules/' + variant + '/' + name)
                member.size = len(payload); member.mode = 0o444; member.mtime = 0
                archive.addfile(member, io.BytesIO(payload))
        payload = json.dumps(manifests, indent=2, sort_keys=True).encode()
        member = tarfile.TarInfo('prepared-source-manifests.json'); member.size = len(payload); member.mtime = 0
        archive.addfile(member, io.BytesIO(payload))
    return {'path': str(target), 'sha256': sha(target), 'includes_private_generated_data': True, 'publish': False}


def prepare(source_root, out, guard, evidence_root=WORKSPACE_EVIDENCE):
    verified, retained = extract_verified(source_root, out)
    baseline = retained['baseline']
    if any(set(files) != set(baseline) for files in retained.values()):
        raise ValueError('retained variants have different source file sets')
    changed_b = {name for name in baseline if retained['B'][name] != baseline[name]}
    changed_c = {name for name in baseline if retained['C'][name] != baseline[name]}
    changed_a = {name for name in baseline if retained['A'][name] != baseline[name]}
    if changed_a != A_CHANGED or changed_b != B_CHANGED or changed_c != C_CHANGED or changed_b & changed_c:
        raise ValueError('B/C patch files overlap or differ from authorized exact changes')
    capsules = out / 'capsules'; capsules.mkdir()
    for variant in ('baseline', 'A', 'B', 'C'):
        shutil.copytree(verified / variant, capsules / variant)
    shutil.copytree(verified / 'B', capsules / 'BC')
    for name in changed_c:
        if sha(capsules / 'BC' / name) != baseline[name]:
            raise ValueError('BC conflicts with C base file: ' + name)
        shutil.copyfile(verified / 'C' / name, capsules / 'BC' / name)
    combination = dict(retained['B']) | {name: retained['C'][name] for name in changed_c}
    before = {name: dict(retained[name]) for name in ('baseline', 'A', 'B', 'C')} | {'BC': combination}
    prepared = {}
    bindings = {}
    for variant in VARIANTS:
        root = capsules / variant
        for name, checksum in before[variant].items():
            if sha(root / name) != checksum:
                raise ValueError('fresh capsule combination checksum differs: ' + variant + '/' + name)
        if variant == 'BC' and any(sha(root / name) != retained['B'][name] for name in B_CHANGED):
            raise ValueError('BC modified B holder/world scratch')
        engine = root / PREFIX / 'cuda_engine.cu'
        original = engine.read_bytes()
        if b'cn_probe_step_resources' in original:
            raise ValueError('resource suffix already present in retained engine')
        engine.write_bytes(original + RESOURCE_SUFFIX.encode())
        prepared[variant] = dict(before[variant]) | {PREFIX + 'cuda_engine.cu': sha(engine)}
        if not evidence_root.is_absolute():
            raise ValueError('shared evidence root must be absolute')
        (root / 'evidence').symlink_to(evidence_root, target_is_directory=True)
        native = root / PREFIX
        command = [sys.executable, '-c', 'import json;from data_binding import validate_data;print(json.dumps(validate_data(),sort_keys=True))']
        log = out / (variant + '-data-binding.log')
        guard.run(command, cwd=native, env=source_environment(native), log=log)
        bindings[variant] = json.loads(log.read_text())
    if any(value != bindings['baseline'] for value in bindings.values()):
        raise ValueError('variant data/rules binding differs')
    dump(out / 'prepared-source-manifests.json', prepared)
    suffix = out / 'identical-resource-query-suffix.cu'; suffix.write_text(RESOURCE_SUFFIX)
    archive = private_archive(out, capsules, prepared)
    result = {'status': 'prepared', 'source_archive_sha256': ARCHIVE_SHA256,
              'source_manifest_sha256': sha(source_root / 'source-manifest.json'),
              'A_changed_files': sorted(changed_a),
              'B_changed_files': sorted(changed_b), 'C_changed_files': sorted(changed_c),
              'BC_conflict_checks': 'C files matched baseline before replacement; B scratch files retain exact B bytes',
              'resource_suffix_sha256': sha(suffix), 'resource_suffix_identical_all_variants': True,
              'capsules': str(capsules), 'prepared_manifests_sha256': sha(out / 'prepared-source-manifests.json'),
              'data_bindings': bindings, 'shared_lock_path': str(evidence_root / 'local-performance.lock'),
              'private_source_archive': archive}
    dump(out / 'preparation.json', result)
    return result


def resource_child(native, out):
    # Normal capsule import/build/receipt validation, not a replaced loader.
    sys.path.insert(0, str(native))
    import ctypes as C
    import torch
    from runtime import build_cuda
    from build_receipt import backend_build_binding
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA L4':
        raise ValueError('probe requires the authorized single NVIDIA L4 allocation')
    library = build_cuda()
    build = backend_build_binding(library)
    function = C.CDLL(str(library)).cn_probe_step_resources
    function.argtypes = [C.POINTER(C.c_longlong), C.c_int]; function.restype = C.c_int
    values = (C.c_longlong * len(RESOURCE_FIELDS))()
    code = function(values, len(values))
    if code:
        raise RuntimeError('CUDA host resource query failed with code ' + str(code))
    resources = dict(zip(RESOURCE_FIELDS, map(int, values)))
    if resources['version'] != 1 or resources['compute_major'] != 8 or resources['compute_minor'] != 9:
        raise ValueError('unexpected resource ABI/architecture')
    occupancy = [{'threads_per_block': threads, 'active_blocks_per_multiprocessor': resources['active_blocks_at_' + str(threads)],
                  'resident_threads_per_multiprocessor': threads * resources['active_blocks_at_' + str(threads)]}
                 for threads in (1, 4, 8, 16, 32)]
    result = {'status': 'passed', 'library': str(library), 'library_sha256': sha(library),
              'backend_build': build, 'resources': resources, 'occupancy': occupancy,
              'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'cpu_count': os.cpu_count(), 'query_scope': 'host cudaFuncGetAttributes + cudaOccupancyMaxActiveBlocksPerMultiprocessor; no kernel launch'}
    dump(out, result)
    print(json.dumps({'status': 'passed', 'library_sha256': result['library_sha256'], 'resources': resources}), flush=True)


def main(args):
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    guard = Guard(1600)
    result = {'status': 'running', 'variants': {}, 'total_guard_seconds': 1600,
              'complete_allocation_authority': 'external provider receipt; all preparation/compile/parity/warmup/timing/idle charged',
              'orchestrator_sha256': sha(__file__), 'prepare_only': args.prepare_only}
    interrupted = []
    def on_signal(number, frame):
        interrupted.append(number); guard.stop_active()
        if hasattr(signal, 'SIGALRM') and number == signal.SIGALRM:
            raise TimeoutError('1600-second total orchestration guard exhausted')
        raise KeyboardInterrupt('orchestrator received signal ' + str(number))
    signals = (signal.SIGTERM, signal.SIGINT) + ((signal.SIGALRM,) if hasattr(signal, 'SIGALRM') else ())
    original_handlers = {sig: signal.signal(sig, on_signal) for sig in signals}
    if hasattr(signal, 'setitimer'):
        signal.setitimer(signal.ITIMER_REAL, 1600)
    try:
        result['preparation'] = prepare(args.source_root.resolve(), out, guard)
        dump(out / 'result.json', result)
        if args.prepare_only:
            result['status'] = 'prepared_only_no_cuda_execution'
        else:
            if not TAPE.is_file() or not WORKSPACE_EVIDENCE.is_dir():
                raise ValueError('retained trained tape or shared workspace evidence is missing')
            result['tape_sha256'] = sha(TAPE)
            for variant in VARIANTS:
                remaining = guard.deadline - time.perf_counter()
                completed_durations = [entry['variant_elapsed_seconds'] for entry in result['variants'].values() if entry.get('status') == 'passed']
                estimated_required = max(150., max(completed_durations, default=0.) * 1.15)
                if remaining < estimated_required:
                    result['variants'][variant] = {'status': 'not_started_insufficient_total_guard',
                                                   'remaining_seconds': remaining,
                                                   'estimated_required_seconds': estimated_required}
                    raise TimeoutError('remaining total guard cannot fit the next full unchanged variant benchmark')
                variant_start = time.perf_counter()
                root = out / 'capsules' / variant; native = root / PREFIX
                entry = {'status': 'resource_query_running'}; result['variants'][variant] = entry
                dump(out / 'result.json', result)
                resources = out / (variant + '-resources.json')
                command = [sys.executable, str(Path(__file__).resolve()), '--resource-child', str(native), str(resources)]
                guard.run(command, cwd=root, env=source_environment(native), log=out / (variant + '-resources.log'))
                entry['resources'] = json.loads(resources.read_text()); entry['status'] = 'benchmark_running'
                dump(out / 'result.json', result)
                benchmark_out = out / (variant + '-benchmark')
                command = [sys.executable, str(native / 'benchmark_cuda.py'), '--out', str(benchmark_out), '--tape', str(TAPE),
                           '--threads', '1', '--render-modes', '4', '--replicas', '8', '--repeats', '3', '--warmups', '1']
                guard.run(command, cwd=root, env=source_environment(native), log=out / (variant + '-benchmark.log'))
                benchmark = json.loads((benchmark_out / 'result.json').read_text())
                if benchmark.get('status') != 'passed' or benchmark['cuda_library_sha256'] != entry['resources']['library_sha256']:
                    raise ValueError('benchmark result/binary differs from its canonical resource-query build')
                if benchmark['cuda_build_receipt']['identity_sha256'] != entry['resources']['backend_build']['identity_sha256']:
                    raise ValueError('benchmark and resource build receipts differ')
                if Path(benchmark['lock_path']).resolve() != (WORKSPACE_EVIDENCE / 'local-performance.lock').resolve():
                    raise ValueError('benchmark did not use the shared workspace performance lock')
                entry.update(status='passed', benchmark_result=str(benchmark_out / 'result.json'),
                             benchmark_sha256=sha(benchmark_out / 'result.json'), medians=benchmark['medians'],
                             validation=benchmark['validation'], variant_elapsed_seconds=time.perf_counter() - variant_start)
                dump(out / 'result.json', result)
            result['status'] = 'passed'
    except BaseException as exc:
        result.update(status='partial_or_failed', failure={'type': type(exc).__name__, 'message': str(exc)}, traceback=traceback.format_exc())
    finally:
        guard.stop_active()
        if hasattr(signal, 'setitimer'):
            signal.setitimer(signal.ITIMER_REAL, 0)
        result.update(commands=guard.commands, process_signals=interrupted,
                      elapsed_orchestration_seconds=time.perf_counter() - guard.start)
        dump(out / 'result.json', result)
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)
    print(json.dumps({'status': result['status'], 'variants': {name: value['status'] for name, value in result['variants'].items()},
                      'failure': result.get('failure')}, indent=2), flush=True)
    return 0 if result['status'] in ('passed', 'prepared_only_no_cuda_execution') else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resource-child', type=Path, nargs=2, metavar=('NATIVE', 'OUT'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.resource_child:
        resource_child(*args.resource_child)
    else:
        if args.out is None:
            parser.error('--out is required')
        raise SystemExit(main(args))
