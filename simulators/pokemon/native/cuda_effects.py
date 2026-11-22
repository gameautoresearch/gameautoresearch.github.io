"""Compare CUDA to CPU and retained oracle fixtures for scalar effect families."""
import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .bridge import Native, State
from .effects_check import compare, load_fixture
from .rules import compile_moves


def run(output, fixtures, cpu_library, cuda_library):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fixture_path = Path(fixtures)
    cases = [json.loads(line) for line in fixture_path.read_text().splitlines()]
    records = {move['id']: move for case in cases for move in case['oracle']['moves']}
    names = sorted(records)
    moves = compile_moves([records[name] for name in names])
    ids = {name: index for index, name in enumerate(names)}
    chart_host = np.asarray(cases[0]['oracle']['chart'], dtype=np.int8)
    if not all(case['oracle']['chart'] == cases[0]['oracle']['chart'] for case in cases):
        raise ValueError('Oracle type chart changed within component corpus')
    library = C.CDLL(str(cuda_library))
    library.pkn_step_device.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p,
                                      C.c_int, C.c_void_p, C.c_void_p]
    library.pkn_masks_device.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p]
    n = len(cases)
    with Native(cpu_library, n, 0) as arena:
        for i, case in enumerate(cases):
            with Native(cpu_library) as single:
                load_fixture(single, case['oracle'])
                for side, move in enumerate(case['oracle']['moves']):
                    single.records[0].mons[side][0].move_ids[0] = ids[move['id']]
                C.memmove(C.addressof(arena.records[i]), single.snapshot(), C.sizeof(State))
        state = torch.from_numpy(np.frombuffer(arena.snapshot(), dtype=np.uint8).copy()).cuda()
        move_data = torch.from_numpy(np.frombuffer(C.string_at(C.addressof(moves), C.sizeof(moves)), dtype=np.uint8).copy()).cuda()
        chart = torch.from_numpy(chart_host).cuda()
        actions = torch.zeros((n, 2), dtype=torch.int32, device='cuda')
        masks = torch.empty((n, 2, 14), dtype=torch.uint8, device='cuda')
        stream = torch.cuda.current_stream().cuda_stream
        comparisons = 0
        oracle_frames = 0
        for frame in range(max(len(case['oracle']['frames']) for case in cases)):
            arena.step(np.zeros((n, 2), dtype=np.int32), moves, chart_host)
            code = library.pkn_step_device(state.data_ptr(), actions.data_ptr(), n,
                                          move_data.data_ptr(), len(moves), chart.data_ptr(), stream)
            if code:
                raise RuntimeError(f'Effect CUDA stepping failed: {code}')
            code = library.pkn_masks_device(state.data_ptr(), masks.data_ptr(), n, stream)
            if code:
                raise RuntimeError(f'Effect CUDA masks failed: {code}')
            torch.cuda.synchronize()
            if state.cpu().numpy().tobytes() != arena.snapshot():
                raise AssertionError(f'Effect CPU/CUDA state mismatch at frame {frame}')
            np.testing.assert_array_equal(masks.cpu().numpy(), arena.masks())
            comparisons += n
            for i, case in enumerate(cases):
                frames = case['oracle']['frames']
                if frame >= len(frames):
                    continue
                mismatch = compare(arena.records[i], frames[frame]['state'])
                if mismatch:
                    raise AssertionError(f'Effect oracle mismatch case {i} frame {frame}: {mismatch}')
                oracle_frames += 1
    result = dict(full_gen9_admission=False, scope='constructed No Ability/itemless effect corpus only',
                  cases=n, cpu_cuda_full_state_and_mask_comparisons=comparisons,
                  oracle_transition_comparisons=oracle_frames,
                  fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                  cpu_binary_sha256=hashlib.sha256(Path(cpu_library).read_bytes()).hexdigest(),
                  cuda_binary_sha256=hashlib.sha256(Path(cuda_library).read_bytes()).hexdigest())
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
