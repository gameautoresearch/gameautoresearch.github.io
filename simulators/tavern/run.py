#!/usr/bin/env python3
"""Game-local train/evaluate/profile command surface. No cloud launch or service calls."""
import time
STARTED = time.perf_counter()  # Includes imports, optional native compilation and setup.
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from provenance import ENGINE, prepare, write_json, sha, ledger


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate", "profile"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="new, immutable run directory")
    parser.add_argument("--checkpoint", type=Path, help="policy for evaluate/profile")
    parser.add_argument("--resume", type=Path, help="trusted local full training checkpoint")
    parser.add_argument("--build", action="store_true", help="compile release extension inside accounting window")
    return parser.parse_args()


def validate(c):
    if c["rules"].get("num_players", 8) != 8:
        raise ValueError("benchmark is eight-player Battlegrounds")
    if c["opponent"] not in ["scripted", "selfplay", "frozen"]:
        raise ValueError("opponent must be scripted, selfplay or frozen")
    if c["opponent"] == "frozen" and not c.get("opponent_checkpoint"):
        raise ValueError("frozen opponent requires opponent_checkpoint")
    if not c["learning_seats"] or len(set(c["learning_seats"])) != len(c["learning_seats"]) or any(s not in range(8) for s in c["learning_seats"]):
        raise ValueError("learning_seats must be distinct 0..7")
    if not c["rules"].get("turn_action_caps"):
        raise ValueError("explicit turn_action_caps required to bound policy loops")
    if c["native_threads"] > c["machine"]["attached_cpu_threads"]:
        raise ValueError("native_threads exceeds configured attached CPU allocation")
    if not c["evaluation"]["seeds"] or c["evaluation"]["games_per_seat"] < 1:
        raise ValueError("evaluation requires a nonempty seed schedule and positive games_per_seat")
    if c["n_games"] < 1 or c["updates"] < 1 or c["minibatch_size"] < 1:
        raise ValueError("positive n_games, updates and minibatch_size required")
    if c.get("sparse_observations", False) and (not c.get("skip_finished_lobbies", False) or not c.get("vector_bookkeeping", True)):
        raise ValueError("sparse_observations requires finished-lobby freezing and vector_bookkeeping")
    if c.get("allow_profile_eliminated",False) and c.get("start_turn",1)<=1:
        raise ValueError("allow_profile_eliminated applies only to labelled continuation profiles")


def checkpoint(path, model, optimizer, update, config, started, prior, manifest, env_states, stats):
    import numpy as np
    import torch
    obj = {"schema": "tavern-ppo-v1", "policy": model.state_dict(),
        "architecture": model.arch, "optimizer": optimizer.state_dict(), "update": update,
        "config": config, "manifest_source_sha256": manifest["source_sha256"],
        "native_source_hash": manifest["native"]["source_hash"], "env_states": env_states,
        "accounted_seconds": prior + time.perf_counter() - started,
        "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(), "stats": stats,
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps_rng": torch.mps.get_rng_state() if str(config["device"]).startswith("mps") else None,
        "environment_cohort_seed": (config["seed"] + (update-1) * 1000003) % 2**64,
        "next_cohort_seed": (config["seed"] + update * 1000003) % 2**64}
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def make_optimizer(model, config):
    import torch
    options = {"fused": True} if config.get("fused_adam", False) else {}
    if options and next(model.parameters()).device.type != "cuda":
        raise ValueError("fused_adam requires CUDA")
    return torch.optim.Adam(model.parameters(), lr=config["learning_rate"], eps=1e-5, **options)


def ppo_update(model, optimizer, episodes, config, device):
    import numpy as np
    import torch
    from rollout import gae
    rows, advantages, returns = [], [], []
    for ep in episodes:
        if not ep:
            continue
        adv, ret = gae(ep, config["gamma"], config["gae_lambda"])
        rows.extend(ep)
        advantages.extend(adv)
        returns.extend(ret)
    if len(rows) < 2:
        raise RuntimeError("insufficient valid completed episode samples")
    # Stage once per PPO update, not once per optimization epoch. This includes transfers.
    if hasattr(episodes, "pack"):
        f, c, m = episodes.pack(rows, device)
    else:
        f = torch.from_numpy(np.stack([r["features"] for r in rows])).to(device)
        c = torch.from_numpy(np.stack([r["cards"] for r in rows])).to(device=device, dtype=torch.long)
        m = torch.from_numpy(np.stack([r["mask"] for r in rows])).to(device)
    actions = torch.tensor([r["action"] for r in rows], device=device)
    oldlp = torch.tensor([r["logp"] for r in rows], device=device)
    adv = torch.tensor(np.array(advantages), device=device)
    ret = torch.tensor(np.array(returns), device=device)
    encoded = config.get("preencode_learner", False)
    if encoded:
        # Features are constant rollout data; this elementwise transform has no
        # trainable parameters. Embeddings remain inside every trainable forward.
        f = model.encode_numeric(f)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    total, batches = Counter(), 0
    deferred = config.get("defer_learner_metrics", False)
    device_totals = torch.zeros(6, device=device) if deferred else None
    before = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
    graph_learner = config.get("graph_learner", False)
    if graph_learner and device.type != "cuda":
        raise ValueError("graph_learner requires CUDA")
    with torch.no_grad():
        value_mse_before = float(((model(f, c, numeric_encoded=encoded)[1] - ret)**2).mean())
    for _ in range(config["ppo_epochs"]):
        for ids in torch.randperm(len(rows), device=device).split(config["minibatch_size"]):
            if graph_learner and len(ids) == config["minibatch_size"]:
                from learner_graph import backward_graph
                loss, policy_loss, value_loss, entropy, kl, grad = backward_graph(model,
                    (f[ids], c[ids], m[ids], actions[ids], oldlp[ids], adv[ids], ret[ids]), config, encoded)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite PPO loss")
            else:
                dist, value = model.distribution(f[ids], c[ids], m[ids], numeric_encoded=encoded)
                logp = dist.log_prob(actions[ids])
                ratio = (logp - oldlp[ids]).exp()
                policy_loss = -torch.minimum(ratio * adv[ids],
                    ratio.clamp(1-config["clip"], 1+config["clip"]) * adv[ids]).mean()
                value_loss = ((value - ret[ids])**2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + config["value_coef"] * value_loss - config["entropy_coef"] * entropy
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite PPO loss")
                optimizer.zero_grad(set_to_none=not graph_learner)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                kl = ((ratio - 1) - (logp - oldlp[ids])).mean().detach()
            optimizer.step()
            if deferred:
                device_totals += torch.stack((loss.detach(), policy_loss.detach(), value_loss.detach(),
                    entropy.detach(), kl, grad.detach()))
            else:
                total.update({"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
                              "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach()),
                              "approx_kl": float(kl), "grad_norm": float(grad)})
            batches += 1
    if deferred:
        total.update(dict(zip(["loss","policy_loss","value_loss","entropy","approx_kl","grad_norm"],device_totals.cpu().tolist())))
    after = torch.cat([p.detach().flatten() for p in model.parameters()])
    with torch.no_grad():
        value_mse_after = float(((model(f, c, numeric_encoded=encoded)[1] - ret)**2).mean())
    return {**{k: v/batches for k, v in total.items()}, "training_samples": len(rows),
            "parameter_delta_l2": float((after-before).norm()),
            "value_mse_before": value_mse_before, "value_mse_after": value_mse_after}


def evaluation(model, config, device, timer, deadline, opponent):
    import numpy as np
    from rollout import collect
    rows, stats = [], Counter()
    # One evaluated policy seat vs seven opponents; rotate through all eight positions.
    # Eight copies playing each other would force mean placement 4.5 and be uninformative.
    for seed in config["evaluation"]["seeds"]:
        for seat in range(8):
            ec = copy.deepcopy(config)
            ec["learning_seats"] = [seat]
            ec["n_games"] = config["evaluation"]["games_per_seat"]
            ec["opponent"] = config["evaluation"]["opponent"]
            if ec["opponent"] == "selfplay":
                raise ValueError("placement evaluation requires scripted or frozen opposition")
            _, records, counts, _ = collect(model, ec, seed, device, timer, deadline, opponent,
                deterministic=config["evaluation"].get("deterministic", False),
                copy_observations=config.get("copy_observations", False))
            rows.extend(records)
            stats.update(counts)
    valid = [r for r in rows if r["valid"] and r["stop_reason"] == "natural"]
    p = np.array([r["placement"] for r in valid], dtype=float)
    return {"episodes": rows, "counts": dict(stats), "valid_episodes": len(valid),
            "invalid_episodes": len(rows)-len(valid), "complete_valid_schedule": len(valid) == len(rows),
            "mean_placement": float(p.mean()) if len(p) else None,
            "normalized_placement_return": float((1 - 2*(p-1)/7).mean()) if len(p) else None,
            "top4_rate": float((p <= 4).mean()) if len(p) else None,
            "per_seat_mean_placement": {str(s): float(np.mean([r["placement"] for r in valid if r["seat"] == s]))
                for s in range(8) if any(r["seat"] == s for r in valid)}}


def actor_copy(model, device, timer, cached=None):
    """Copy one fixed policy per cohort, preserving all caller RNG streams."""
    import torch
    from model import Policy
    if next(model.parameters()).device == device:
        return model
    with timer.measure("actor_weight_transfer"):
        if cached is None:
            with torch.random.fork_rng(devices=[]):
                cached = Policy(**model.arch).to(device)
        cached.load_state_dict({k:v.detach().to(device) for k,v in model.state_dict().items()})
        cached.eval()
    return cached


def execute(args, config):
    # Imported after the worker timer and optional build. Compilation is charged.
    import numpy as np
    import torch
    import taverngym as tg
    from model import Policy, load_policy
    from rollout import Timer, collect, BudgetExpired
    validate(config)
    if config.get("allow_profile_eliminated",False) and args.command!='profile':
        raise ValueError("preprofile elimination filtering is profile-only, never training or evaluation")
    device = torch.device(config["device"])
    actor_device = torch.device(config.get("actor_device", str(device)))
    if actor_device != device and actor_device.type != "cpu":
        raise ValueError("actor_device must be cpu or the learner device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback or fabricated GPU result")
    if device.type == "cuda" and (config.get("validated_sampler", False) or config.get("graph_inference", False) or config.get("graph_learner", False)) and torch.__version__.split("+")[0] != "2.9.1":
        raise RuntimeError("CUDA sampler/graphs are qualified on PyTorch 2.9.1; disable them or rerun parity on this version")
    if config.get("graph_inference", False) and (device.type != "cuda" or not config.get("fast_inference", False)):
        raise ValueError("graph_inference requires CUDA and fast_inference")
    if config.get("graph_inference", False) and actor_device.type != "cuda":
        raise ValueError("graph_inference requires a CUDA actor")
    if config.get("graph_learner", False) and not config.get("fast_inference", False):
        raise ValueError("graph_learner requires CPU-validated fast_inference")
    torch.set_num_threads(config.get("torch_threads", config["machine"]["attached_cpu_threads"]))
    torch.set_num_interop_threads(1)
    random.seed(config["seed"])
    np.random.seed(config["seed"] % 2**32)
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    manifest = prepare(args.out, config, args.command, sys.argv, STARTED)
    prior, start_update, env_states = 0.0, 0, []
    status, error = "succeeded", None
    try:
        model = Policy(**config["network"]).to(device)
        optimizer = make_optimizer(model, config)
        if args.checkpoint:
            model, loaded = load_policy(args.checkpoint, device)
            optimizer = make_optimizer(model, config)
            if loaded["native_source_hash"] != manifest["native"]["source_hash"] or loaded["config"]["rules"] != config["rules"]:
                raise ValueError("checkpoint rules/native source differs from evaluation")
            write_json(args.out / "input_checkpoint.json", {"path": str(args.checkpoint.resolve()), "sha256": sha(args.checkpoint)})
        if args.resume:
            model, loaded = load_policy(args.resume, device)
            if loaded["config"] != config:
                raise ValueError("resume requires exact recipe; use a larger updates target in the original recipe")
            if loaded["manifest_source_sha256"] != manifest["source_sha256"]:
                raise ValueError("resume source snapshot differs; refuse untracked recipe drift")
            optimizer = make_optimizer(model, config)
            optimizer.load_state_dict(loaded["optimizer"])
            start_update, prior = loaded["update"], loaded["accounted_seconds"]
            # Charge all actually executed prior-session time, even when resuming an
            # earlier snapshot. The session ledger includes final checkpoint IO/cleanup.
            parent_ledger = args.resume.parent / "accounting.json"
            if parent_ledger.exists():
                prior = max(prior, json.loads(parent_ledger.read_text())["total_allocated_worker_seconds"])
            torch.set_rng_state(loaded["torch_rng"].cpu())
            random.setstate(loaded["python_rng"])
            np.random.set_state(loaded["numpy_rng"])
            if loaded.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in loaded["cuda_rng"]])
            if loaded.get("mps_rng") is not None:
                torch.mps.set_rng_state(loaded["mps_rng"].cpu())
            env_states = loaded["env_states"]
            # Verify every exact engine/RNG state can be restored. Next cohort starts at the
            # saved deterministic seed boundary; no partial rollout is silently replayed.
            controlled = config["learning_seats"] if config["opponent"] == "scripted" else list(range(8))
            restored = tg.BatchEnv(config["n_games"], seed=loaded.get("environment_cohort_seed",
                (config["seed"] + (start_update-1) * 1000003) % 2**64), controlled=controlled, config=config["rules"])
            for i, state in enumerate(env_states):
                restored.load(i, state)
                if json.loads(restored.save(i)) != json.loads(state):
                    raise RuntimeError("engine/RNG checkpoint restore mismatch")
            write_json(args.out / "resume.json", {"checkpoint": str(args.resume.resolve()), "sha256": sha(args.resume), "start_update": start_update, "prior_seconds": prior})
        if args.command == "train" and args.checkpoint:
            raise ValueError("use --resume for training; --checkpoint is evaluation/profile only")
        opponent = None
        if config.get("opponent_checkpoint"):
            opponent, opponent_state = load_policy(Path(config["opponent_checkpoint"]), actor_device)
            if opponent_state["native_source_hash"] != manifest["native"]["source_hash"] or opponent_state["config"]["rules"] != config["rules"]:
                raise ValueError("frozen opponent rules/native source differs from this run")
            write_json(args.out / "opponent.json", {"sha256": sha(config["opponent_checkpoint"]), "path": config["opponent_checkpoint"]})
        if config["evaluation"]["opponent"] == "frozen" and opponent is None:
            raise ValueError("frozen evaluation requires opponent_checkpoint")
        cap = config.get("allocated_worker_hours")
        deadline = STARTED + cap * 3600 - prior if cap is not None else float("inf")
        timer = Timer(device)
        if config.get("save_initial", False) and args.command == "train" and not args.resume:
            checkpoint(args.out / "initial.pt", model, optimizer, 0, config, STARTED, prior, manifest, [], {})
        if args.command == "evaluate":
            model.eval()
            actor = actor_copy(model, actor_device, timer)
            result = evaluation(actor, config, actor_device, timer, deadline, opponent)
            write_json(args.out / "evaluation.json", result)
            print(json.dumps({k:v for k,v in result.items() if k not in ["episodes"]}), flush=True)
        else:
            # profile performs real rollouts + real optimizer updates. No synthetic step-only proxy.
            updates = config["updates"] if args.command == "train" else config["profile_updates"]
            totals, last = Counter(loaded["stats"]) if args.resume else Counter(), start_update
            actor = None
            for update in range(start_update, updates):
                iteration_started = time.perf_counter()
                prior_stages = dict(timer.seconds)
                if update == start_update:
                    startup_seconds = iteration_started - STARTED
                if time.perf_counter() >= deadline:
                    raise BudgetExpired("allocated-worker deadline reached before next cohort")
                model.eval()
                # The learner also consumes CPU-validated rollout masks when the
                # actor is a separate CPU replica. Do not accidentally re-enable
                # per-minibatch CUDA distribution validation on that path.
                model.fast_inference = config.get("fast_inference", False)
                # Device changes have explicit numerical/RNG semantics and receive
                # separate learning measurements. Fixed-policy cohorts are unchanged.
                actor = actor_copy(model, actor_device, timer, actor)
                cohort_seed = (config["seed"] + update * 1000003) % 2**64
                episodes, records, stats, env_states = collect(actor, config, cohort_seed, actor_device,
                    timer, deadline, opponent, copy_observations=config.get("copy_observations", False))
                with (args.out / "rollouts.jsonl").open("a") as raw:
                    raw.write(json.dumps({"update": update+1, "cohort_seed": cohort_seed,
                        "counts": stats, "episodes": records}) + "\n")
                model.train()
                with timer.measure("ppo_update_transfer"):
                    losses = ppo_update(model, optimizer, episodes, config, device)
                totals.update(stats)
                last = update + 1
                row = {"update": last, "cohort_seed": cohort_seed, "counts": stats,
                       "learning": losses, "episodes": records,
                       "allocated_worker_seconds": prior+time.perf_counter()-STARTED,
                       "iteration_wall_seconds_excluding_checkpoint": time.perf_counter()-iteration_started,
                       "stage_seconds_this_update": {k:v-prior_stages.get(k,0) for k,v in timer.seconds.items()},
                       "phase": "first_update_cold" if update == start_update else "warmed_update"}
                with (args.out / "metrics.jsonl").open("a") as log:
                    log.write(json.dumps(row) + "\n")
                print(json.dumps({k:v for k,v in row.items() if k != "episodes"}), flush=True)
                with timer.measure("checkpoint_io"):
                    checkpoint(args.out / "latest.pt", model, optimizer, last, config, STARTED,
                               prior, manifest, env_states, dict(totals))
                if config.get("snapshot_every", 0) and last % config["snapshot_every"] == 0:
                    checkpoint(args.out / f"opponent-{last:06d}.pt", model, optimizer, last,
                               config, STARTED, prior, manifest, env_states, dict(totals))
            write_json(args.out / "training.json", {"completed_updates": last, "counts": dict(totals),
                        "strong_reference_trained": False, "validation_scope": "local learning/profiling recipe"})
        import resource
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(args.out / "profile.json", {"stage_seconds": dict(timer.seconds),
                   "diagnostic_seconds": dict(timer.diagnostics), "workload_context": timer.context,
                   "diagnostic_counts": dict(timer.counters),
                   "startup_before_first_update_seconds": locals().get("startup_seconds"),
                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                   "peak_process_rss_bytes": peak_rss if sys.platform == "darwin" else peak_rss * 1024,
                   "actor_graph_shapes": len(getattr(locals().get("actor", model), "_actor_graphs", {})),
                   "learner_graph_build_seconds": getattr(model, "learner_graph_build_seconds", 0.0),
                   "synchronized_device_timing": True, "device": str(device),
                   "notes": "Complete cohort loop including simulator, observation, transfers, sampling, storage, updates and checkpointing. Startup/import/build/source archive in accounting.json."})
    except BudgetExpired as exc:
        status, error = "budget_exhausted", str(exc)
        if args.command != "evaluate" and 'totals' in locals():
            write_json(args.out / "training.json", {"completed_updates": last,
                "counts": dict(totals), "strong_reference_trained": False,
                "status": status, "partial_cohort_discarded": "during rollout" in str(exc),
                "validation_scope": "last complete checkpoint; incomplete cohort is never scored"})
        if 'timer' in locals():
            write_json(args.out / "profile.json", {"stage_seconds": dict(timer.seconds),
                "diagnostic_seconds": dict(timer.diagnostics), "device": str(device),
                "diagnostic_counts": dict(timer.counters),
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                "notes": "Includes partial-cohort work, which is charged but excluded from training samples and placements."})
    except BaseException as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        result = ledger(args.out, STARTED, prior, status, error, measurement_kind="cuda_validation" if device.type == "cuda" else "cpu_validation")
        print(json.dumps(result), flush=True)


def main():
    args = arguments()
    config = json.loads(args.config.read_text())
    os.environ["OMP_NUM_THREADS"] = str(config["machine"]["attached_cpu_threads"])
    os.environ["MKL_NUM_THREADS"] = str(config["machine"]["attached_cpu_threads"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TAVERNGYM_WORKERS"] = config.get("worker_backend", "pooled")
    os.environ["TAVERNGYM_SCHEDULER"] = config.get("native_scheduler", "static")
    os.environ["TAVERNGYM_GUARD_DIR"] = str(args.out.resolve())
    if args.out.exists():
        raise FileExistsError("choose a new run directory")
    try:
        if args.build:
            subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--reinstall", "--editable", str(ENGINE)], check=True)
        execute(args, config)
    except BaseException as exc:
        if not args.out.exists():
            args.out.mkdir(parents=True)
        if not (args.out / "accounting.json").exists():
            ledger(args.out, STARTED, 0.0, "failed_preflight", repr(exc))
        raise


if __name__ == "__main__":
    main()
