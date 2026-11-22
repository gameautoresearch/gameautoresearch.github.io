"""Immutable local source/rules/seed manifests and elapsed allocated-worker ledger."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENGINE = ROOT / "candidates" / "AutoTavern"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def prepare(out, config, command, argv, started):
    import taverngym as tg
    import torch
    if tg.build_is_current(ENGINE) is not True:
        raise RuntimeError("native extension is stale; rebuild with --build")
    if config.get("sparse_observations", False):
        view_pin=json.loads((HERE/'observation-provenance.json').read_text())
        if sha(ROOT/view_pin['oracle'])!=view_pin['oracle_sha256']:
            raise RuntimeError("compact observation oracle pin changed; requalify field/visibility parity")
    out.mkdir(parents=True, exist_ok=False)
    files = sorted(set([*(ENGINE / "crates").rglob("*.rs"),
        *(ENGINE / "python").rglob("*.py"), ENGINE / "crates/taverngym/data/cards.json",
        ENGINE / "Cargo.toml", ENGINE / "Cargo.lock", ENGINE / "pyproject.toml",
        ENGINE / "LICENSE", ENGINE / "NOTICE", ENGINE / "rust-toolchain.toml", *HERE.glob("*.py"), *HERE.glob("*.txt"),
        *HERE.glob("*.md"), *HERE.glob("*provenance.json"), *(HERE / "recipes").glob("*.json"),
        *(HERE / "tests").glob("*.py")]))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files if p.is_file()}
    with tarfile.open(out / "source.tar.gz", "w:gz") as archive:
        for name in hashes:
            archive.add(ROOT / name, arcname=name, recursive=False)
    native = Path(tg.build_info()["file"])
    write_json(out / "recipe.json", config)
    manifest = {"schema": "tavern-run-v1", "command": command, "argv": argv,
        "started_unix": time.time() - (time.perf_counter() - started),
        "upstream": {"url": "https://github.com/MCG-NJU/AutoTavern.git",
                     "commit": (subprocess.check_output(["git", "-C", str(ENGINE), "rev-parse", "HEAD"], text=True).strip() if (ENGINE / ".git").exists() else "1c2086a997afde46d68c6c5992cab194ddcfce88")},
        "source_sha256": hashes, "source_archive_sha256": sha(out / "source.tar.gz"),
        "native": {**tg.build_info(), "sha256": sha(native)},
        "rules": json.loads(tg.default_config_json()) | config["rules"],
        "game_version": tg.game_version(), "server_rules_version": tg.server_rules_version(),
        "observation": {"features": tg.OBS_FLOATS, "cards": tg.OBS_CARDS,
                        "card_vocabulary": tg.N_CARDS, "actions": tg.N_ACTIONS},
        "seeding": {"training_cohort": "(seed + update_index * 1000003) modulo 2^64",
                    "engine": "splitmix64(splitmix64(cohort_seed ^ splitmix64(game_index)) ^ episode_index)",
                    "eval": config.get("evaluation"), "python_torch_seed": config["seed"]},
        "hardware": {"configured_allocation": config["machine"],
                     "local_allocation_enforced": bool(os.environ.get("MODAL_TASK_ID")),
                     "modal_task_id": os.environ.get("MODAL_TASK_ID"),
                     "platform": platform.platform(), "cpu": platform.processor(),
                     "logical_cpus": os.cpu_count(), "device": config["device"],
                     "cuda_available": torch.cuda.is_available(),
                     "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None},
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": __import__("numpy").__version__},
        "budget_contract": "Charge elapsed allocated-worker wall time including CPU simulation, transfers, compilation, inference, updates, evaluation and IO; GPU idle time counts. No utilization discount.",
        "fidelity": "Pinned emulator 37.0.2 conventions; commercial-game parity unverified."}
    write_json(out / "manifest.json", manifest)
    for name in ("manifest.json", "recipe.json", "source.tar.gz"):
        (out / name).chmod(0o444)
    return manifest


def ledger(out, started, prior_seconds, status, error=None, measurement_kind="local_validation"):
    seconds = time.perf_counter() - started
    result = {"status": status, "session_allocated_worker_seconds": seconds,
              "prior_allocated_worker_seconds": prior_seconds,
              "total_allocated_worker_seconds": prior_seconds + seconds,
              "total_allocated_worker_hours": (prior_seconds + seconds) / 3600,
              "error": error, "accounting": "wall time, including GPU idle time",
              "measurement_kind": measurement_kind}
    write_json(out / "accounting.json", result)
    return result
