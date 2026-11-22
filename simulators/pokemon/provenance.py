"""Immutable source/rules/seed manifests and elapsed allocated-worker ledger."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ENGINE = ROOT / "candidates/pokemon-showdown"
PINS = {"pokemon-showdown": "567c1cedddeaa2f35576005d3bfd526cf4f8a13e",
        "poke-env": "dd35fb672bb1cb3f00ee67769e3a9d86688d8d54"}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"))
def write_json(path, x, immutable=False):
    path = Path(path)
    with path.open("x" if immutable else "w") as f:
        json.dump(x, f, indent=2, sort_keys=True)
        f.write("\n")
    if immutable:
        path.chmod(0o444)

def manifest(out, config, command, schedule=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    repos = {}
    for name, pin in PINS.items():
        repo = ROOT / "candidates" / name
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if commit != pin:
            raise RuntimeError(f"Rules pin changed: {name} {commit} != {pin}")
        files = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z"]).decode().split("\0")
        hashes = {p: digest(repo/p) for p in files if p and (repo/p).is_file()}
        dirty = subprocess.check_output(["git", "-C", str(repo), "diff", "--name-only", "HEAD"], text=True).strip()
        if dirty:
            raise RuntimeError(f"Tracked upstream rules/source changed: {name}: {dirty}")
        repos[name] = {"commit": commit, "tracked_sha256": hashes}
    # Include actual executed builds and transitive Node dependencies, not just TS source pins.
    runtime_hashes = {}
    for directory in (ENGINE/"dist", ENGINE/"node_modules"):
        for p in sorted(directory.rglob("*")):
            if p.is_file() and p.suffix in (".js", ".cjs", ".json", ".node"):
                runtime_hashes[str(p.relative_to(ENGINE))] = digest(p)
    own = {}
    for p in sorted(HERE.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".cjs", ".json", ".md", ".txt", ".cpp", ".hpp", ".cu", ".h", ".c") and "__pycache__" not in p.parts:
            rel = p.relative_to(HERE)
            target = out / "source" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
            target.chmod(0o444)
            own[str(rel)] = digest(p)
    import torch
    import numpy
    result = {"schema": 1, "command": command, "config": config, "seed_schedule": {"training": {"master_start":config["training_seed_start"],"requested_count":config["iterations"]*config.get("episodes_per_update",config["hardware"]["workers"]*config["hardware"]["environments_per_worker"]),"stride":1,"policy_seed":config["seed"],"opponent_rng":"NumPy SeedSequence([master,seat,8675309])"}, "evaluation_and_inputs":schedule},
              "format": "gen9randombattle", "encoder": "pokemon-visible-v1", "action_codec": "moves0:4,tera4:8,switches8:14",
              "trace_normalization":"Only |t:|<wallclock> presentation timestamps replaced with |t:|0",
              "seed_derivation": "SHA256(pokemon-v1:<integer>:battle|team1|team2), first 8 bytes, uint16 LE x4",
              "budget_unit": "elapsed allocated training-worker seconds, GPU idle time included",
              "allocation": config["hardware"], "source": own, "repos": repos, "runtime": runtime_hashes,
              "host": {"platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
                       "python": sys.version, "torch": torch.__version__, "numpy": numpy.__version__,
                       "node": subprocess.check_output(["node", "--version"], text=True).strip(),
                       "cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda},
              "created_unix": time.time()}
    write_json(out/"manifest.json", result, immutable=True)
    return digest(out/"manifest.json")

class Ledger:
    def __init__(self, start, prior=0.0):
        self.start = start
        self.prior = prior
    @property
    def elapsed(self):
        return self.prior + time.perf_counter() - self.start
    def result(self):
        return {"allocated_worker_seconds": self.elapsed, "allocated_worker_hours": self.elapsed/3600,
                "accounting": "wall-clock allocation including startup, CPU simulation, copies, inference, updates, checkpointing and development evaluation; no utilization discount"}
