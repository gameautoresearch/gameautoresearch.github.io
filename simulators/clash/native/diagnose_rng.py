"""Small CPU/CUDA RNG/deck/Match.init diagnostic. Never provisions compute or runs Java.

Native source provenance: crforge BattleRandom.java and battle/match/{DeckShuffle,Hand,
LadderMatch}.java @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de.
Worker: python integrations/clash/native/diagnose_rng.py NEW_EVIDENCE_DIR
Local CPU-only: add --cpu-only. Optional --world-state reads a saved raw native GPU state.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PIN = "f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de"
STRIDE = 192
DECK_MODES = ("loop-constructors", "unrolled-constructors", "noinline-per-side", "assignment-reuse", "inline-per-side", "explicit-index-zero")
MATCH_MODES = ("source-Match.init", "stack-zero-playerWords", "unrolled-Match.init")


def _ptr(tensor):
    return C.c_void_p(tensor.data_ptr())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _compile(command, log):
    with Path(log).open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def _decks(row):
    return {
        "seed": row[0], "player_words": row[1:3], "warm_draws": row[3],
        "random_after_warm": row[4], "shuffle_draws": [row[5], row[7]],
        "shuffle_seeds": row[9:11], "random_after_two": row[8],
        "battle_draws": row[16:48:2], "battle_states": row[17:48:2],
        "sides": [{"initial_index": row[64+s*32], "seeded_words": row[65+s*32:73+s*32],
                   "order": row[73+s*32:81+s*32], "final_index": row[81+s*32],
                   "twisted_words": row[82+s*32:90+s*32]} for s in range(2)],
    }


def _match(row):
    return {"seed": row[0], "player_words": row[1:3], "warm_draws": row[3],
            "random_after_init": row[4], "shuffle_draws": row[5:7], "tick": row[7],
            "clock_ms": row[8], "starting_elixir": row[9], "match_bytes": row[10], "rules_bytes": row[11],
            "sides": [{"hand": row[64+s*32:68+s*32], "queue": row[68+s*32:76+s*32],
                       "queue_count": row[76+s*32], "cooldown_ms": row[77+s*32], "elixir": row[78+s*32]}
                      for s in range(2)]}


def diagnose_world_metadata(states, cuda_library, layout):
    """Read cached old-world bytes using CPU offsetof metadata; no full-world CUDA linking.

    This assumes the cached binary uses the same POD layout. The caller must compare its
    recorded world_bytes to layout['world_bytes']; a mismatch is refused before kernel launch.
    """
    import torch
    if states.dtype != torch.uint8 or states.device.type != "cuda" or states.ndim != 2:
        raise ValueError("world states must be a CUDA uint8 [N,world_bytes] tensor")
    if states.shape[1] != layout["world_bytes"]:
        raise ValueError("cached world byte size differs from current CPU layout; refusing offset read")
    if not states.is_contiguous():
        raise ValueError("raw cached state must be contiguous")
    output = torch.empty((len(states), 32), dtype=torch.int32, device=states.device)
    f = cuda_library.rng_cuda_metadata
    f.argtypes = [C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_void_p, C.c_int, C.c_void_p]
    f.restype = C.c_int
    stream = C.c_void_p(torch.cuda.current_stream(states.device).cuda_stream)
    code = f(_ptr(states), states.shape[1], layout["match_offset"], layout["random_offset"], _ptr(output), len(states), stream)
    if code:
        raise RuntimeError(cuda_library.rng_cuda_error(code).decode())
    torch.cuda.synchronize(states.device)
    return output.cpu().tolist()


def _cached_world_probe(cache_library, diagnostic_library, layout, out):
    """Reset one explicitly owned old cached library without invoking current build/runtime."""
    import torch
    cache_value = os.environ.get("CLASH_NATIVE_BUILD_CACHE")
    if not cache_value:
        raise RuntimeError("--cached-world-probe requires the owned CLASH_NATIVE_BUILD_CACHE directory")
    cache = Path(cache_value).resolve()
    architecture = "sm_" + "".join(map(str, torch.cuda.get_device_capability()))
    if cache_library:
        candidates = [Path(cache_library).resolve()]
    else:
        candidates = sorted(cache.glob("libclash-cuda-*.so"))
    eligible = []
    for library in candidates:
        if library.parent.resolve() != cache:
            raise ValueError("cached library must be a direct child of the owned build cache")
        receipt_path = library.with_suffix(".json")
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("architecture") != architecture:
            continue
        if receipt.get("library_sha256") != _sha(library):
            raise ValueError(f"cached library SHA differs from its receipt: {library}")
        eligible.append((library, receipt_path, receipt))
    if len(eligible) != 1:
        raise ValueError("require exactly one receipted legacy library on the current architecture; "
                         f"found {[str(v[0]) for v in eligible]}; select --cached-library explicitly")
    library, receipt_path, receipt = eligible[0]
    legacy = C.CDLL(str(library))
    legacy.cn_cuda_world_bytes.restype = C.c_size_t
    legacy.cn_cuda_create.restype = C.c_void_p
    legacy.cn_cuda_free.argtypes = [C.c_void_p]
    legacy.cn_cuda_reset.argtypes = [C.c_void_p] * 4 + [C.c_int, C.c_void_p]
    legacy.cn_cuda_reset.restype = C.c_int
    legacy.cn_cuda_read.argtypes = [C.c_void_p] * 5 + [C.c_int, C.c_void_p]
    legacy.cn_cuda_read.restype = C.c_int
    legacy.cn_cuda_error_string.argtypes = [C.c_int]
    legacy.cn_cuda_error_string.restype = C.c_char_p
    stride = int(legacy.cn_cuda_world_bytes())
    # This is intentionally an OLD-ABI diagnostic, not source-compatible restore/qualification.
    # Match/random prefix offsets were unchanged in GPU2; only the later World tail gained queues.
    old_layout = dict(layout, world_bytes=stride, match_offset=13880, random_offset=15712)
    if old_layout["random_offset"] + 4 > stride:
        raise ValueError("legacy stride cannot contain the explicitly recorded GPU2 prefix")
    states = torch.zeros((4, stride), dtype=torch.uint8, device="cuda")
    seeds = torch.tensor([0, 1, -1, 771], dtype=torch.int32, device="cuda")
    obs = torch.empty((4, 2, 7024), dtype=torch.int32, device="cuda")
    masks = torch.empty((4, 2, 2307), dtype=torch.uint8, device="cuda")
    infos = torch.empty((4, 12), dtype=torch.int32, device="cuda")
    rewards = torch.empty((4, 2), dtype=torch.float32, device="cuda")
    stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
    handle = legacy.cn_cuda_create()
    if not handle:
        raise RuntimeError("legacy cn_cuda_create failed")
    try:
        code = legacy.cn_cuda_reset(handle, _ptr(states), _ptr(seeds), None, 4, stream)
        if code:
            raise RuntimeError(legacy.cn_cuda_error_string(code).decode())
        torch.cuda.synchronize()
        metadata_before = diagnose_world_metadata(states, diagnostic_library, old_layout)
        code = legacy.cn_cuda_read(_ptr(states), _ptr(obs), _ptr(masks), _ptr(infos), _ptr(rewards), 4, stream)
        if code:
            raise RuntimeError(legacy.cn_cuda_error_string(code).decode())
        torch.cuda.synchronize()
        metadata_after = diagnose_world_metadata(states, diagnostic_library, old_layout)
        # Preserve the complete diagnostic state and ABI outputs privately for byte-level audit.
        torch.save({"states": states.cpu(), "seeds": seeds.cpu(), "observations": obs.cpu(),
                    "masks": masks.cpu(), "infos": infos.cpu(), "rewards": rewards.cpu(),
                    "layout": old_layout, "receipt": receipt}, out / "cached-world-reset.pt")
        # Same global member addresses/layout, initialized only by the tiny independent kernel.
        member_states = torch.zeros_like(states)
        member_inputs = torch.tensor([[seed, 0, 0, 2] for seed in (0, 1, -1, 771)], dtype=torch.int32, device="cuda")
        member_output = torch.empty((4, STRIDE), dtype=torch.int32, device="cuda")
        f = diagnostic_library.rng_cuda_match_member
        f.argtypes = [C.c_void_p, C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_void_p, C.c_int, C.c_void_p]
        f.restype = C.c_int
        code = f(_ptr(member_inputs), _ptr(member_states), stride, old_layout["match_offset"], old_layout["random_offset"], _ptr(member_output), 4, stream)
        if code:
            raise RuntimeError(diagnostic_library.rng_cuda_error(code).decode())
        torch.cuda.synchronize()
        member_rows = member_output.cpu().tolist()
        torch.save({"output": member_output.cpu(), "states": member_states.cpu()}, out / "isolated-global-member.pt")
    finally:
        legacy.cn_cuda_free(handle)
    return {"scope": "diagnostic only: old cached GPU2 ABI prefix; no current full-world parity claim",
            "library": str(library), "receipt": str(receipt_path), "verified_library_sha256": receipt["library_sha256"],
            "architecture": architecture, "current_cpu_layout": layout, "legacy_offset_override": old_layout,
            "metadata_before_render": metadata_before, "metadata_after_render": metadata_after,
            "isolated_global_member_match": [_match(row) for row in member_rows]}


def run(output_dir, *, cpu_only=False, world_states=None, cached_world_probe=False, cached_library=None):
    import torch
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = HERE / "diagnose_rng.cu"
    compiler = shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
    if not compiler:
        raise RuntimeError("C++ compiler required for independent CPU reference")
    cpu_path = out / ("rng-cpu.dylib" if sys.platform == "darwin" else "rng-cpu.so")
    cpu_command = [compiler, "-std=c++20", "-O3", "-x", "c++", "-fPIC", "-dynamiclib" if sys.platform == "darwin" else "-shared", str(source), "-o", str(cpu_path)]
    _compile(cpu_command, out / "cpu-build.log")
    cpu = C.CDLL(str(cpu_path))
    cpu.rng_diag_stride.restype = C.c_int
    assert cpu.rng_diag_stride() == STRIDE
    cpu.rng_cpu_world_layout.argtypes = [C.POINTER(C.c_uint64)]
    values = (C.c_uint64 * 8)()
    cpu.rng_cpu_world_layout(values)
    layout = dict(zip(("world_bytes", "match_offset", "random_offset", "match_bytes", "shuffle_draws_offset", "sides_offset", "side_bytes", "hand_bytes"), map(int, values)))
    # First four cases reproduce the native-cuda-2 qualification reset seeds exactly. Extra
    # player words, signed-boundary seeds and warmup counts distinguish seeding from lifecycle.
    cases = [[seed, 0, 0, 2] for seed in (0, 1, -1, 771)]
    cases += [[seed, words[0], words[1], warm] for seed in (2, 2147483647, -2147483648, 123456789)
              for words in ((0, 0), (17, -31)) for warm in (0, 1, 2, 3)]
    inputs = torch.tensor(cases, dtype=torch.int32)
    cpu_outputs = {}
    result = {"oracle_commit": PIN, "status": "running", "cases": cases, "cpu_command": cpu_command,
              "cpu_compiler": subprocess.check_output([compiler, "--version"], text=True),
              "source_sha256": _sha(source), "driver_sha256": _sha(__file__), "cpu_layout": layout,
              "timing_comparison": False, "cuda_executed": not cpu_only, "comparisons": [], "differences": []}
    for kind, modes in (("decks", DECK_MODES), ("match", MATCH_MODES)):
        f = getattr(cpu, "rng_cpu_" + kind)
        f.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int]
        for mode, name in enumerate(modes):
            output = torch.empty((len(cases), STRIDE), dtype=torch.int32)
            f(_ptr(inputs), _ptr(output), len(cases), mode)
            cpu_outputs[(kind, mode)] = output
    result["cpu_reset_decks"] = [_decks(row) for row in cpu_outputs[("decks", 0)][:4].tolist()]
    result["cpu_reset_match"] = [_match(row) for row in cpu_outputs[("match", 0)][:4].tolist()]
    if cpu_only:
        result["status"] = "cpu-reference-ready"
        torch.save({"inputs": inputs, "outputs": {f"{kind}:{mode}": value for (kind, mode), value in cpu_outputs.items()}}, out / "cpu-reference.pt")
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    if not torch.cuda.is_available():
        raise RuntimeError("actual CUDA required; --cpu-only is the explicit local reference mode")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("nvcc required for standalone diagnostic kernel")
    capability = torch.cuda.get_device_capability()
    arch = "sm_" + "".join(map(str, capability))
    cuda_path = out / "rng-cuda.so"
    cuda_command = [nvcc, "-std=c++20", "-O3", "-arch=" + arch, "--ptxas-options=-O1,-v", "--expt-relaxed-constexpr", "-Xcompiler=-fPIC", "-shared", str(source), "-o", str(cuda_path)]
    _compile(cuda_command, out / "cuda-build.log")
    gpu = C.CDLL(str(cuda_path))
    gpu.rng_cuda_error.argtypes = [C.c_int]
    gpu.rng_cuda_error.restype = C.c_char_p
    gpu.rng_cuda_set_stack.argtypes = [C.c_int]
    code = gpu.rng_cuda_set_stack(65536)
    if code:
        raise RuntimeError(gpu.rng_cuda_error(code).decode())
    result.update(cuda_command=cuda_command, cuda_name=torch.cuda.get_device_name(), architecture=arch,
                  cuda_compiler=subprocess.check_output([nvcc, "--version"], text=True),
                  cuda_library_sha256=_sha(cuda_path), cpu_library_sha256=_sha(cpu_path))
    device_inputs = inputs.cuda()
    stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
    gpu_outputs = {}
    for kind, modes in (("decks", DECK_MODES), ("match", MATCH_MODES)):
        f = getattr(gpu, "rng_cuda_" + kind)
        f.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p]
        f.restype = C.c_int
        for mode, name in enumerate(modes):
            output = torch.empty((len(cases), STRIDE), dtype=torch.int32, device="cuda")
            code = f(_ptr(device_inputs), _ptr(output), len(cases), mode, stream)
            if code:
                raise RuntimeError(gpu.rng_cuda_error(code).decode())
            torch.cuda.synchronize()
            actual = output.cpu()
            expected = cpu_outputs[(kind, mode)]
            gpu_outputs[(kind, mode)] = actual
            unequal = actual != expected
            coordinates = unequal.nonzero().tolist()
            result["comparisons"].append({"kind": kind, "mode": mode, "name": name,
                                          "exact": not coordinates, "integer_fields": actual.numel(), "different_fields": len(coordinates)})
            if coordinates:
                result["differences"].append({"kind": kind, "mode": mode, "name": name,
                    "fields": [{"case": case, "field": field, "cpu": int(expected[case, field]), "cuda": int(actual[case, field])} for case, field in coordinates[:100]]})
    result["cuda_reset_decks"] = [_decks(row) for row in gpu_outputs[("decks", 0)][:4].tolist()]
    result["cuda_reset_match"] = [_match(row) for row in gpu_outputs[("match", 0)][:4].tolist()]
    if world_states is not None:
        result["cached_world_metadata"] = diagnose_world_metadata(world_states, gpu, layout)
    if cached_world_probe:
        result["cached_world_probe"] = _cached_world_probe(cached_library, gpu, layout, out)
    result["status"] = "passed" if not result["differences"] else "diverged"
    torch.save({"inputs": inputs,
                "cpu": {f"{k}:{m}": v for (k, m), v in cpu_outputs.items()},
                "cuda": {f"{k}:{m}": v for (k, m), v in gpu_outputs.items()}}, out / "all-probes.pt")
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--world-state", help="optional raw uint8 native CUDA snapshot .pt")
    parser.add_argument("--cached-world-probe", action="store_true", help="reset the single receipted owned old cached CUDA engine without a rebuild")
    parser.add_argument("--cached-library", help="exact old .so in CLASH_NATIVE_BUILD_CACHE if multiple eligible receipts exist")
    args = parser.parse_args()
    world = None
    if args.world_state:
        import torch
        world = torch.load(args.world_state, map_location="cuda", weights_only=True)
        if isinstance(world, dict):
            world = world.get("engine_state", world.get("states"))
    if args.cached_library and not args.cached_world_probe:
        parser.error("--cached-library requires --cached-world-probe")
    if args.cpu_only and args.cached_world_probe:
        parser.error("--cached-world-probe requires actual CUDA")
    summary = run(args.output_dir, cpu_only=args.cpu_only, world_states=world,
                  cached_world_probe=args.cached_world_probe, cached_library=args.cached_library)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("cpu_reset_decks", "cuda_reset_decks", "cpu_reset_match", "cuda_reset_match", "cpu_compiler", "cuda_compiler", "cases")}, indent=2), flush=True)
    if summary["status"] == "diverged":
        raise SystemExit(1)
