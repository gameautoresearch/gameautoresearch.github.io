"""Lossless full-world snapshot encoding correctness, with no Java or GPU.

Phase durations and compression ratios are storage diagnostics only, not a
timing comparison/benchmark. Every encoded lane includes all native bytes.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import pickle
import random
import sys
import time
import zlib

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def qualify(out):
    import jpype
    import numpy as np
    import torch
    from runtime import NativeCPU
    from snapshots import (available_cpu_workers, decode_engine_state,
                           encode_engine_state, snapshot_sizes)
    from train import binding
    data_path = HERE / 'generated-v2/task-data.json'
    initial_binding = binding(data_path)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    assert not jpype.isJVMStarted()
    torch.set_num_threads(1)
    seeds = [1000000, 1000001, 1000002, 1000003]
    env = NativeCPU(seeds)
    restored = None
    rng = np.random.default_rng(8110809)
    maximum_entities = 0
    phase_times = {}
    data = json.loads(data_path.read_text())
    count_index = data['schema']['header']['entity_count'][0]
    try:
        # Dense legal random play exercises routes, targets, projectiles,
        # queues, command history, PRNG state and first-sight bookkeeping.
        for step in range(192):
            actions = [[int(rng.choice(np.flatnonzero(env.masks[lane, side]))) for side in (0, 1)] for lane in range(4)]
            env.step(actions)
            maximum_entities = max(maximum_entities, int(env.observations[..., count_index].max()))
        assert maximum_entities > 12
        start = time.monotonic(); raw = env.state(); phase_times['save_full_native_bytes'] = time.monotonic() - start
        width = len(raw[0]); assert all(len(lane) == width for lane in raw)
        # Observe all three ordinary RNG families before EVERY codec variant.
        python_state = random.getstate()
        numpy_state = pickle.dumps(np.random.get_state())
        torch_state = torch.get_rng_state().clone()
        local_state = copy.deepcopy(rng.bit_generator.state)
        def unchanged_rng():
            assert random.getstate() == python_state
            assert pickle.dumps(np.random.get_state()) == numpy_state
            assert torch.equal(torch.get_rng_state(), torch_state)
            assert rng.bit_generator.state == local_state
        workers = min(4, available_cpu_workers())
        start = time.monotonic(); encoded = encode_engine_state(raw, workers=workers); phase_times['encode_full_bytes'] = time.monotonic() - start
        unchanged_rng()
        # Exact encoded payload bytes do not depend on worker scheduling.
        assert encode_engine_state(raw, workers=1) == encoded
        start = time.monotonic(); decoded = decode_engine_state(encoded, expected_count=4, expected_world_bytes=width, workers=workers); phase_times['decode_byte_lanes'] = time.monotonic() - start
        assert decoded == raw
        start = time.monotonic(); tensor = decode_engine_state(encoded, as_tensor=True, expected_count=4, expected_world_bytes=width, workers=workers); phase_times['decode_cpu_uint8_tensor'] = time.monotonic() - start
        assert tensor.dtype == torch.uint8 and tensor.device.type == 'cpu' and tensor.shape == (4, width)
        assert [row.numpy().tobytes() for row in tensor] == raw
        assert encode_engine_state(tensor, workers=workers) == encoded
        assert decode_engine_state(raw, expected_count=4, expected_world_bytes=width) == raw
        assert decode_engine_state(tensor, expected_count=4, expected_world_bytes=width) == raw
        assert torch.equal(decode_engine_state(raw, as_tensor=True), tensor)
        assert torch.equal(decode_engine_state(tensor, as_tensor=True), tensor)
        # Noncontiguous raw matrix preserves exact logical lane bytes too.
        strided = torch.empty((4, width * 2), dtype=torch.uint8)
        strided[:, ::2].copy_(tensor)
        assert encode_engine_state(strided[:, ::2], workers=workers) == encoded
        assert decode_engine_state(strided[:, ::2]) == raw
        unchanged_rng()
        # Persist using the same torch.save container the trainer will use.
        start = time.monotonic(); torch.save(encoded, out / 'encoded-full-world.pt'); phase_times['torch_save_encoded_container'] = time.monotonic() - start
        loaded = torch.load(out / 'encoded-full-world.pt', weights_only=True)
        assert decode_engine_state(loaded) == raw
        unchanged_rng()
        rejected = []
        def reject(label, value, **kwargs):
            try:
                decode_engine_state(value, **kwargs)
            except (ValueError, TypeError):
                rejected.append(label)
            else:
                raise AssertionError('corrupt/invalid snapshot accepted: ' + label)
        bad = copy.deepcopy(encoded); bad['lanes'][0]['sha256'] = '0' * 64; reject('checksum', bad)
        bad = copy.deepcopy(encoded); payload = bytearray(bad['lanes'][0]['payload']); payload[len(payload) // 2] ^= 1; bad['lanes'][0]['payload'] = bytes(payload); reject('compressed-byteflip', bad)
        changed_raw = bytearray(raw[0]); changed_raw[width // 2] ^= 1
        bad = copy.deepcopy(encoded); bad['lanes'][0]['payload'] = zlib.compress(changed_raw, 1); reject('raw-byteflip-recompressed', bad)
        bad = copy.deepcopy(encoded); bad['count'] += 1; reject('metadata-count', bad)
        bad = copy.deepcopy(encoded); bad['lanes'].pop(); reject('missing-lane', bad)
        bad = copy.deepcopy(encoded); bad['world_bytes'] += 1; reject('metadata-shape', bad)
        bad = copy.deepcopy(encoded); bad['lanes'][0]['uncompressed_bytes'] += 1; reject('lane-shape', bad)
        bad = copy.deepcopy(encoded); bad['lanes'][0]['payload'] = bad['lanes'][0]['payload'][:-1]; reject('truncated-stream', bad)
        bad = copy.deepcopy(encoded); bad['lanes'][0]['payload'] += b'not part of the zlib stream'; reject('trailing-data', bad)
        bad = copy.deepcopy(encoded); bad['lanes'][0]['payload'] += zlib.compress(raw[0], 1); reject('trailing-second-stream', bad)
        bad = copy.deepcopy(encoded); bad['lanes'][0]['payload'] = zlib.compress(raw[0] + b'X', 1); reject('oversized-decode', bad)
        bad = copy.deepcopy(encoded); bad['lanes'][0]['payload'] = zlib.compress(raw[0][:-1], 1); reject('undersized-decode', bad)
        bad = copy.deepcopy(encoded); bad['count'] = True; reject('boolean-count', bad)
        bad = copy.deepcopy(encoded); bad['lanes'][0]['sha256'] = 'X' * 64; reject('malformed-checksum', bad)
        bad = copy.deepcopy(encoded); bad['extra'] = 1; reject('unknown-metadata', bad)
        bad = copy.deepcopy(encoded); bad['level'] = 9; reject('wrong-encoding-level', bad)
        reject('expected-count', encoded, expected_count=3)
        reject('expected-width', encoded, expected_world_bytes=width + 1)
        reject('total-byte-bound', encoded, max_total_bytes=4 * width - 1)
        reject('raw-list-shape', [b'a', b'ab'])
        reject('raw-empty-list', [])
        reject('raw-empty-width', [b''])
        reject('raw-tensor-dtype', torch.zeros((1, 4), dtype=torch.int32))
        reject('raw-tensor-rank', torch.zeros((4,), dtype=torch.uint8))
        reject('raw-expected-shape', raw, expected_world_bytes=width - 1)
        reject('raw-byte-bound', raw, max_total_bytes=4 * width - 1)
        reject('workers-over-allocation', encoded, workers=available_cpu_workers() + 1)
        unchanged_rng()
        # Restore decoded bytes into a separate native world. The ABI rebinds
        # pointers itself; storage decoding never omits/normalizes those bytes.
        restored = NativeCPU(seeds)
        restored.load(decoded); restored.read()
        for key in ('observations', 'masks', 'infos', 'rewards'):
            np.testing.assert_array_equal(getattr(restored, key), getattr(env, key))
        for step in range(32):
            actions = [[int(rng.choice(np.flatnonzero(env.masks[lane, side]))) for side in (0, 1)] for lane in range(4)]
            env.step(actions); restored.step(actions)
            for key in ('observations', 'masks', 'infos', 'rewards'):
                np.testing.assert_array_equal(getattr(restored, key), getattr(env, key))
        assert not jpype.isJVMStarted()
        assert binding(data_path) == initial_binding, 'source changed during snapshot qualification'
        result = {'status': 'passed', 'jvm_started': False, 'complete_bytes_preserved': True,
                  'busy_decisions': 192, 'maximum_entities': maximum_entities,
                  'exact_continuation_lanes': 4, 'exact_continuation_decisions': 32,
                  'thread_count_payload_independent': True, 'rng_unchanged': True,
                  'corruption_rejections': rejected,
                  'lane_uncompressed_sha256': [hashlib.sha256(lane).hexdigest() for lane in raw],
                  'sizes': snapshot_sizes(encoded), 'container_bytes': (out / 'encoded-full-world.pt').stat().st_size,
                  'phase_seconds_diagnostic_only': phase_times,
                  'timing_scope': 'single correctness run storage phase diagnostics; no timing comparison or performance claim',
                  'binding': initial_binding}
        (out / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        return result
    finally:
        if restored is not None:
            restored.close()
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.out)
    print(json.dumps({k: result[k] for k in ('status', 'sizes', 'rng_unchanged')}, indent=2))
