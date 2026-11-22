"""Lossless complete native engine snapshots, without simulator/RNG changes.

The payload stores EVERY byte of EVERY NativeWorld lane, including unused
capacity and padding. This is storage encoding, never a state projection.
The native layout/data/source binding remains the checkpoint owner's duty;
legacy raw decoding is provided only for read-only historical analyses.

CUDA callers may pass ``env.states.cpu()`` directly, avoiding a full additional
GPU clone. Passing a CUDA uint8 matrix here also copies it directly to CPU.
Compression uses builtin zlib level 1, independently per lane, and bounded
threads within the current process's CPU affinity/quota. No random APIs run.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import os
import zlib

FORMAT = 'clash-native-engine-zlib-v1'
ENCODING = 'zlib'
LEVEL = 1
DEFAULT_MAX_TOTAL_BYTES = 16 * 1024 ** 3


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return value


def available_cpu_workers():
    """Bound workers to attached cores, including Linux fractional CPU quotas."""
    limits = [os.cpu_count() or 1]
    if hasattr(os, 'process_cpu_count'):
        limits.append(os.process_cpu_count() or 1)
    if hasattr(os, 'sched_getaffinity'):
        try:
            limits.append(len(os.sched_getaffinity(0)))
        except OSError:
            pass
    # cgroup v2, then the usual v1 mount. Missing/unreadable quota files do
    # not expand the affinity bound. Floor a fractional quota to whole cores.
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            limits.append(max(1, int(quota) // int(period)))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    for folder in ('/sys/fs/cgroup/cpu', '/sys/fs/cgroup/cpu,cpuacct'):
        try:
            quota = int((Path(folder) / 'cpu.cfs_quota_us').read_text())
            period = int((Path(folder) / 'cpu.cfs_period_us').read_text())
            if quota > 0:
                limits.append(max(1, quota // period))
        except (OSError, ValueError, ZeroDivisionError):
            pass
    return max(1, min(limits))


def _workers(requested, count):
    limit = available_cpu_workers()
    if requested is None:
        return min(limit, count)
    requested = _integer(requested, 'workers')
    if requested > limit:
        raise ValueError('snapshot workers exceed the attached CPU affinity/quota')
    return min(requested, count)


def _ordered_map(function, count, workers):
    """Bound submitted work as well as threads; yield in deterministic order."""
    if workers == 1:
        for index in range(count):
            yield function(index)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='native-snapshot') as pool:
        # Executor.map eagerly submits all lanes on older Python versions.
        # At most one lane per worker is outstanding here, bounding temporary
        # decompressed/encoded storage independently of the total lane count.
        for base in range(0, count, workers):
            pending = [pool.submit(function, i) for i in range(base, min(base + workers, count))]
            for future in pending:
                yield future.result()


def _raw_input(state):
    """Return count, exact lane bytes and a lane reader, retaining its owner."""
    import torch
    if isinstance(state, torch.Tensor):
        if state.dtype != torch.uint8 or state.ndim != 2 or state.layout != torch.strided:
            raise ValueError('raw engine tensor must be a strided uint8 [count,world_bytes] matrix')
        count, width = state.shape
        _integer(count, 'count'); _integer(width, 'world_bytes')
        # .cpu() never creates an intermediate CUDA clone. Contiguity is
        # established on CPU so a strided legacy image also decodes exactly.
        host = state.detach().cpu().contiguous()
        return count, width, lambda index: memoryview(host[index].numpy())
    if not isinstance(state, (list, tuple)) or not state:
        raise ValueError('raw engine state must be a nonempty byte list or uint8 matrix')
    if any(not isinstance(lane, (bytes, bytearray, memoryview)) for lane in state):
        raise ValueError('raw engine byte list contains a non-byte lane')
    # Snapshot mutable byte-like inputs once. NativeCPU already returns bytes,
    # so its ordinary path needs no extra full-world copy.
    lanes = tuple(lane if isinstance(lane, bytes) else bytes(lane) for lane in state)
    width = len(lanes[0]); _integer(width, 'world_bytes')
    if any(len(lane) != width for lane in lanes):
        raise ValueError('raw engine lanes have inconsistent world_bytes')
    return len(lanes), width, lambda index: lanes[index]


def encode_engine_state(state, *, workers=None):
    """Encode complete CPU byte lanes or a CPU/CUDA uint8 matrix with zlib1.

    The caller must not mutate a tensor input until this synchronous operation
    returns. Results are deterministic for identical bytes at every thread
    count. Worker count is deliberately absent from the wire representation.
    """
    count, width, lane_at = _raw_input(state)
    threads = _workers(workers, count)

    def encode(index):
        lane = lane_at(index)
        return {'uncompressed_bytes': width, 'sha256': hashlib.sha256(lane).hexdigest(),
                'payload': zlib.compress(lane, level=LEVEL)}

    lanes = list(_ordered_map(encode, count, threads))
    return {'format': FORMAT, 'encoding': ENCODING, 'level': LEVEL,
            'count': count, 'world_bytes': width, 'lanes': lanes}


def _shape(count, width, expected_count, expected_world_bytes, max_total_bytes):
    _integer(count, 'count'); _integer(width, 'world_bytes')
    _integer(max_total_bytes, 'max_total_bytes')
    if expected_count is not None and count != _integer(expected_count, 'expected_count'):
        raise ValueError('snapshot count differs from requested native batch')
    if expected_world_bytes is not None and width != _integer(expected_world_bytes, 'expected_world_bytes'):
        raise ValueError('snapshot world_bytes differs from requested native layout')
    if count * width > max_total_bytes:
        raise ValueError('snapshot uncompressed bytes exceed explicit decode bound')


def _materialize(lanes, count, width, as_tensor):
    if not isinstance(as_tensor, bool):
        raise ValueError('as_tensor must be boolean')
    if not as_tensor:
        return list(lanes)
    import torch
    result = torch.empty((count, width), dtype=torch.uint8, device='cpu')
    for index, lane in enumerate(lanes):
        # bytearray owns a writable temporary; avoid PyTorch's non-writable
        # bytes warning and never give it an alias to immutable input storage.
        source = torch.frombuffer(bytearray(lane), dtype=torch.uint8)
        result[index].copy_(source)
    return result


def decode_engine_state(state, *, as_tensor=False, expected_count=None,
                        expected_world_bytes=None, workers=None,
                        max_total_bytes=DEFAULT_MAX_TOTAL_BYTES):
    """Validate/decode an exact image; optionally return CPU uint8 [N,bytes].

    Supply expected_count/world_bytes from the live native ABI for production
    restore. The byte bound is checked BEFORE output allocation/decompression;
    each zlib stream is capped at its exact expected length plus one byte.
    Legacy raw matrices/lists are accepted for read-only analysis, without
    overriding any checkpoint owner's independent source/data binding checks.
    """
    if not isinstance(as_tensor, bool):
        raise ValueError('as_tensor must be boolean')
    if not isinstance(state, dict):
        count, width, lane_at = _raw_input(state)
        _shape(count, width, expected_count, expected_world_bytes, max_total_bytes)
        _workers(workers, count)
        return _materialize((bytes(lane_at(i)) for i in range(count)), count, width, as_tensor)
    if set(state) != {'format', 'encoding', 'level', 'count', 'world_bytes', 'lanes'}:
        raise ValueError('snapshot encoding metadata fields differ')
    if state['format'] != FORMAT or state['encoding'] != ENCODING or type(state['level']) is not int or state['level'] != LEVEL:
        raise ValueError('snapshot encoding/format/level is unsupported')
    count, width = state['count'], state['world_bytes']
    _shape(count, width, expected_count, expected_world_bytes, max_total_bytes)
    lanes = state['lanes']
    if not isinstance(lanes, list) or len(lanes) != count:
        raise ValueError('snapshot lane count differs from metadata')
    # Validate ALL metadata before starting a single decompress job.
    for lane in lanes:
        if not isinstance(lane, dict) or set(lane) != {'uncompressed_bytes', 'sha256', 'payload'}:
            raise ValueError('snapshot lane metadata fields differ')
        if type(lane['uncompressed_bytes']) is not int or lane['uncompressed_bytes'] != width:
            raise ValueError('snapshot lane uncompressed length/shape differs')
        checksum = lane['sha256']
        if not isinstance(checksum, str) or len(checksum) != 64 or any(ch not in '0123456789abcdef' for ch in checksum):
            raise ValueError('snapshot lane SHA256 is malformed')
        if not isinstance(lane['payload'], bytes) or not lane['payload']:
            raise ValueError('snapshot lane payload must be nonempty bytes')
    threads = _workers(workers, count)

    def decode(index):
        lane = lanes[index]
        decompressor = zlib.decompressobj()
        try:
            raw = decompressor.decompress(lane['payload'], width + 1)
        except zlib.error as exc:
            raise ValueError('snapshot zlib stream is corrupt in lane ' + str(index)) from exc
        if len(raw) != width:
            raise ValueError('snapshot decoded lane length differs in lane ' + str(index))
        if not decompressor.eof:
            raise ValueError('snapshot zlib stream is truncated/oversized in lane ' + str(index))
        if decompressor.unused_data or decompressor.unconsumed_tail:
            raise ValueError('snapshot zlib stream has trailing data in lane ' + str(index))
        if hashlib.sha256(raw).hexdigest() != lane['sha256']:
            raise ValueError('snapshot decoded lane checksum differs in lane ' + str(index))
        return raw

    return _materialize(_ordered_map(decode, count, threads), count, width, as_tensor)


def snapshot_sizes(encoded):
    """Storage diagnostic only; no simulator fields or encoded bytes omitted."""
    if not isinstance(encoded, dict) or encoded.get('format') != FORMAT:
        raise ValueError('snapshot size report requires an encoded snapshot')
    raw = encoded['count'] * encoded['world_bytes']
    compressed = sum(len(lane['payload']) for lane in encoded['lanes'])
    return {'lanes': encoded['count'], 'world_bytes': encoded['world_bytes'],
            'uncompressed_bytes': raw, 'compressed_payload_bytes': compressed,
            'compressed_over_uncompressed': compressed / raw,
            'uncompressed_over_compressed': raw / compressed}
