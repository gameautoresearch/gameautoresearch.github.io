"""python -m integrations.generals {train,evaluate,profile,manifest} ..."""
# Start the allocation clock BEFORE importing JAX/Flax or probing the engine.
import time
START = time.monotonic()
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("command", choices=("train", "evaluate", "profile", "manifest"))
    p.add_argument("--recipe", default="integrations/generals/recipes/full-smoke.json")
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--execution", choices=("auto", "baseline", "optimized"), default=None,
                   help="auto uses the measured baseline on CPU and fused shifted-value collector on CUDA")
    p.add_argument("--checkpoint")
    p.add_argument("--resume")
    p.add_argument("--opponent-checkpoint")
    p.add_argument("--split", choices=("practice", "heldout"), default="practice")
    p.add_argument("--seed-file", help="external evaluation seed JSON list; must be disjoint from training")
    p.add_argument("--opponents", default="random,expansion,hunter")
    p.add_argument("--max-updates", type=int, help="stop at this absolute update for resumability tests; does not change recipe")
    p.add_argument("--greedy", action="store_true")
    a = p.parse_args()
    cfg = json.loads(Path(a.recipe).read_text())
    # JAX 0.12 expands 'gpu' to CUDA and ROCm; explicitly choose CUDA on the
    # NVIDIA-only worker so a missing ROCm backend cannot abort initialization.
    os.environ["JAX_PLATFORMS"] = "cuda" if a.device == "gpu" else "cpu"
    if cfg.get('simulation_device') == 'cpu' and a.command in ('train', 'profile'):
        if a.device != 'gpu':
            raise ValueError('CPU-producer/GPU-learner recipes require --device gpu; omit simulation_device for all-CPU execution')
        os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
    if a.device == "gpu":
        graph = cfg.get('gpu_command_buffer', 'default')
        if graph not in ('default', 'while', 'unroll'):
            raise ValueError('gpu_command_buffer must be default, while or unroll')
        if graph != 'default':
            if 'xla_gpu_enable_command_buffer' in os.environ.get('XLA_FLAGS', ''):
                raise ValueError('Specify command buffer mode in recipe or XLA_FLAGS, not both')
            flags = '--xla_gpu_enable_command_buffer=+WHILE'
            if graph == 'unroll':
                flags += ' --xla_gpu_command_buffer_unroll_loops=true'
            os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '')+' '+flags
        # Reuse executables across train/resume/evaluate/profile processes in
        # this allocated worker. Fresh containers still pay their cold build.
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/generals-xla-cache")
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    # Fixed precision for deterministic CPU and consistent accelerator computation.
    os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .environment import RULESET, build_pool
    from .learning import create_train_state, initial_carry, make_collect, make_update
    from .model import Policy
    from .runtime import (HERE, Ledger, digest, canonical, hardware, sources, write_json,
                          save_checkpoint, read_checkpoint)
    from .evaluation import evaluate
    cfg["execution"] = a.execution or cfg.get("execution", "auto")
    if cfg["execution"] == "auto":
        cfg["execution"] = "optimized" if jax.default_backend() == "gpu" else "baseline"
    cfg["frozen_opponent_sha256"] = digest(Path(a.opponent_checkpoint).read_bytes()) if a.opponent_checkpoint else None
    from .autoresearch.contract import seed_manifest as load_seed_manifest
    seed_manifest = load_seed_manifest()
    if cfg.get('simulation_device') == 'cpu' and a.command in ('train', 'profile'):
        cfg['train_pool_size'] = len(seed_manifest['train'])
        if cfg['execution'] != 'optimized' or not cfg.get('packed_rollout') or not cfg.get('compact_opponent_inference'):
            raise ValueError('CPU producer requires optimized execution, packed_rollout and compact_opponent_inference')
    validate_config(cfg, seed_manifest)
    out = Path(a.output)
    if out.resolve().is_relative_to(HERE):
        raise ValueError("Keep generated outputs outside integrations/generals, which is source-bound")
    out.mkdir(parents=True, exist_ok=True)
    if a.command == "train" and not a.resume and (out/"train.jsonl").exists():
        raise ValueError("Existing training output: choose a new directory or resume explicitly")
    binding = sources()
    hw = hardware(cfg, a.device)
    ledger = Ledger(START, cfg["allocated_worker_seconds"])
    write_json(out/"source-manifest.json", binding)
    write_json(out/"run-manifest.json", dict(recipe=cfg, hardware=hw, seeds=seed_manifest,
                                             command=vars(a), rules=RULESET, seed_manifest_sha256=digest(canonical(seed_manifest))))
    if a.command == "manifest":
        pool = build_pool(seed_manifest["practice"]+seed_manifest["heldout"])
        np.savez_compressed(out/"evaluation-maps.npz", **{k: np.asarray(v) for k, v in pool._asdict().items()})
        write_json(out/"map-manifest.json", dict(seed_order=seed_manifest["practice"]+seed_manifest["heldout"],
                    file="evaluation-maps.npz", sha256=digest((out/"evaluation-maps.npz").read_bytes()), rules=RULESET))
        write_json(out/"budget.json", ledger.report())
        return
    key = jax.random.PRNGKey(cfg["seed"])
    key, initkey = jax.random.split(key)
    model = Policy(cfg["network_width"], convolution=cfg.get('convolution', 'direct'), compute_dtype=cfg.get('compute_dtype', 'float32'))
    train = create_train_state(model, initkey, cfg)
    if a.command == "evaluate":
        if a.checkpoint:
            meta, data = read_checkpoint(a.checkpoint, binding=binding)
            if meta["config"]["network_width"] != cfg["network_width"]:
                raise ValueError("Evaluation network_width must match checkpoint")
            train = train.replace(params=data["params"])
        seeds = json.loads(Path(a.seed_file).read_text()) if a.seed_file else seed_manifest[a.split]
        if not seeds:
            raise SystemExit("No local held-out seeds. Supply --seed-file for a new evaluation.")
        if not seeds or len(seeds) != len(set(seeds)) or any(not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds):
            raise ValueError("Evaluation seeds must be a nonempty unique uint32 list")
        if set(seeds) & set(seed_manifest["train"]):
            raise ValueError("Evaluation seeds overlap training seeds")
        if a.seed_file and set(seeds) & set(seed_manifest["practice"]+seed_manifest["heldout"]):
            raise ValueError("External private evaluation seeds must be disjoint from all public fixtures")
        opponents = tuple(a.opponents.split(","))
        frozen = None
        if "frozen" in opponents:
            if not a.opponent_checkpoint:
                raise ValueError("frozen evaluation requires --opponent-checkpoint")
            fm, fd = read_checkpoint(a.opponent_checkpoint, binding=binding)
            if fm["config"]["network_width"] != cfg["network_width"]:
                raise ValueError("Frozen policy network must match evaluation network")
            frozen = fd["params"]
        result = evaluate(model, train.params, seeds, opponents, frozen=frozen, stochastic=not a.greedy,
                          replay_path=out/"replay.npz", sampler=cfg.get('sampler', 'gumbel'))
        result.update(split="external" if a.seed_file else a.split, checkpoint=a.checkpoint,
                      checkpoint_sha256=digest(Path(a.checkpoint).read_bytes()) if a.checkpoint else None,
                      budget=ledger.report(), hardware=hw,
                      simulation_backend_for_evaluation=hw['backend'],
                      note_backend='Evaluation uses --device for the canonical simulator and policy; simulation_device selects only the training/profile producer')
        write_json(out/"evaluation.json", result)
        print(json.dumps(result["groups"], indent=2), flush=True)
        return

    setup_start = time.monotonic()
    pool = build_pool(seed_manifest["train"])
    bank = jax.tree.map(lambda x: jnp.repeat(x[None], cfg["history_size"], axis=0), train.params)
    frozen_history_slot = False
    if a.opponent_checkpoint:
        fm, fd = read_checkpoint(a.opponent_checkpoint, binding=binding)
        if fm["config"]["network_width"] != cfg["network_width"] or cfg["history_size"] < 2:
            raise ValueError("Frozen training opponent needs matching network width and history_size >= 2")
        bank = jax.tree.map(lambda b, p: b.at[0].set(p), bank, fd["params"])
        frozen_history_slot = True
        write_json(out/"frozen-opponent.json", dict(path=str(Path(a.opponent_checkpoint).resolve()),
                   sha256=digest(Path(a.opponent_checkpoint).read_bytes()), metadata=fm))
    carry = initial_carry(pool, key, cfg, bank)
    update_number, bank_cursor = 0, 0
    if a.resume:
        meta, (train, bank, carry, pool) = read_checkpoint(a.resume, train, cfg, binding)
        ledger.previous = meta["budget"]["allocated_worker_seconds"]
        update_number, bank_cursor = meta["update"], meta["bank_cursor"]
    if cfg.get('simulation_device') == 'cpu':
        cpu = jax.devices('cpu')[0]
        pool = jax.tree.map(lambda x: jax.device_put(x, cpu), pool)
        carry = carry._replace(states=jax.tree.map(lambda x: jax.device_put(x, cpu), carry.states))
    jax.block_until_ready((train, carry, pool, bank))
    setup_seconds = time.monotonic()-setup_start
    from .efficiency import make_collect_optimized, make_iteration
    collect = make_collect_optimized(model, cfg) if cfg["execution"] == "optimized" else make_collect(model, cfg)
    update = make_update(model, cfg)
    iteration = make_iteration(model, cfg, cfg["execution"])
    if cfg.get('simulation_device') == 'cpu':
        from .split_device import make_collect_split, make_iteration_split
        collect = make_collect_split(model, cfg)
        iteration = make_iteration_split(model, cfg)
    if a.command == "profile":
        from .profiling import profile
        result = profile(model, train, bank, carry, pool, collect, update, cfg)
        result.update(setup_seconds=setup_seconds, hardware=hw, budget=ledger.report())
        write_json(out/"profile.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return

    initial_params = jax.tree.map(lambda p: p.copy(), train.params)
    evaluation_rows = []
    # Development evaluation is practice only, included in the same allocated clock.
    if cfg["evaluate_before_after"] and not a.resume and not ledger.expired():
        evaluation_rows.append(dict(stage="before", update=0, **evaluate(model, train.params,
                            seed_manifest["practice"][:cfg["development_eval_seeds"]], ("random", "expansion"), sampler=cfg.get('sampler', 'gumbel'))))
    timings, log_rows = [], []
    upper = min(cfg["max_updates"], a.max_updates if a.max_updates is not None else cfg["max_updates"])
    latest = out/"latest.msgpack"
    stop_reason = "max_updates"
    while update_number < upper:
        # Leave one measured update of reserve after the first iteration. JIT is included;
        # a scheduler must kill the worker at the hard deadline if compilation hangs.
        reserve = max(timings[-3:], default=0.)
        evaluation_reserve = cfg["evaluation_reserve_seconds"] if cfg["evaluate_before_after"] else 0
        if ledger.expired(reserve=reserve+cfg["checkpoint_reserve_seconds"]+evaluation_reserve):
            stop_reason = "allocated_worker_budget"
            break
        iteration_start = time.monotonic()
        t0 = time.monotonic()
        if cfg["execution"] == "optimized":
            train, carry, compact = iteration(train, bank, carry, pool)
            jax.block_until_ready((train, carry, compact))
            telemetry = np.asarray(compact)
            m = telemetry[:6]
            stats = dict(mean_training_reward=float(telemetry[11]), mean_land=float(telemetry[10]),
                         captures_won=int(telemetry[6]), captures_lost=int(telemetry[7]),
                         truncations=int(telemetry[8]), invalid_actions=int(telemetry[9]))
            # A single fused executable has no honest split wall-time attribution.
            rollout_seconds = update_seconds = None
        else:
            carry, batch = collect(train.params, bank, carry, pool)
            jax.block_until_ready((carry, batch))
            rollout_seconds = time.monotonic()-t0
            # Keep the RNG stream in the carry so save/resume is exact.
            key, uk = jax.random.split(carry.key)
            carry = carry._replace(key=key)
            t0 = time.monotonic()
            train, metrics = update(train, batch, uk)
            jax.block_until_ready(train)
            update_seconds = time.monotonic()-t0
            m = np.asarray(metrics)
            stats = dict(mean_training_reward=float(batch["reward"].mean()), mean_land=float(batch["own_land"].mean()),
                         captures_won=int(batch["wins"].sum()), captures_lost=int(batch["losses"].sum()),
                         truncations=int(batch["truncated"].sum()), invalid_actions=int(batch["invalid"].sum()))
        iteration_seconds = time.monotonic()-iteration_start
        update_number += 1
        if update_number % cfg["history_interval"] == 0:
            slot = (bank_cursor % (cfg["history_size"]-1) + 1) if frozen_history_slot else bank_cursor % cfg["history_size"]
            bank = jax.tree.map(lambda b, p: b.at[slot].set(p), bank, train.params)
            bank_cursor += 1
        if not np.all(np.isfinite(m)):
            raise FloatingPointError(f"nonfinite PPO metrics at update {update_number}: {m}")
        row = dict(update=update_number, learner_samples=cfg["num_envs"]*cfg["rollout_steps"],
                   environment_ticks=cfg["num_envs"]*cfg["rollout_steps"], policy_decisions=2*cfg["num_envs"]*cfg["rollout_steps"],
                   rollout_seconds=rollout_seconds, update_seconds=update_seconds, iteration_seconds=iteration_seconds,
                   execution=cfg["execution"],
                   loss=float(m[0]), policy_loss=float(m[1]), value_loss=float(m[2]), entropy=float(m[3]),
                   approximate_kl=float(m[4]), gradient_norm=float(m[5]), **stats, allocation=ledger.report())
        if row["invalid_actions"]:
            raise RuntimeError("Illegal action emitted; refusing to train through codec fault")
        with (out/"train.jsonl").open("a") as f:
            f.write(json.dumps(row)+"\n")
        log_rows.append(row)
        timings.append(time.monotonic()-iteration_start)
        print(json.dumps({k: row[k] for k in ("update", "loss", "mean_land", "captures_won", "captures_lost", "truncations", "rollout_seconds", "update_seconds")}), flush=True)
        if update_number % cfg["checkpoint_interval"] == 0:
            save_checkpoint(latest, train, bank, carry, pool, cfg, binding, update_number, ledger, bank_cursor)
    final_requested = a.max_updates is None or update_number >= cfg["max_updates"]
    if cfg["evaluate_before_after"] and final_requested and not ledger.expired(reserve=cfg["checkpoint_reserve_seconds"]):
        evaluation_rows.append(dict(stage="after", update=update_number, **evaluate(model, train.params,
                            seed_manifest["practice"][:cfg["development_eval_seeds"]], ("random", "expansion"), sampler=cfg.get('sampler', 'gumbel'))))
    if evaluation_rows:
        write_json(out/"development-evaluation.json", evaluation_rows)
    jax.block_until_ready((train, bank, carry))
    norm = np.sqrt(sum(float(jnp.sum((p-q)**2)) for p, q in zip(jax.tree.leaves(train.params), jax.tree.leaves(initial_params))))
    save_checkpoint(latest, train, bank, carry, pool, cfg, binding, update_number, ledger, bank_cursor)
    result = dict(update=update_number, stop_reason=stop_reason, setup_seconds=setup_seconds,
                  parameter_l2_change_this_invocation=float(norm), optimizer_steps=int(train.step),
                  episodes=int(carry.episodes.sum()), history_snapshots_written=bank_cursor,
                  learner_samples_total=update_number*cfg["num_envs"]*cfg["rollout_steps"],
                  budget=ledger.report(), hardware=hw, checkpoint=str(latest.resolve()),
                  learning_claim="Bounded local PPO smoke only; reference recipe is untrained until separately executed and validated.")
    write_json(out/"result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def validate_config(cfg, seeds):
    from .environment import RULESET
    if cfg.get('environment_id') != RULESET['id']:
        raise ValueError('Use a full-v2 recipe with environment_id='+RULESET['id']+
                         '; archived8x8 recipes/checkpoints require their exact archived source')
    if cfg.get('simulation_device', 'gpu') not in ('gpu', 'cpu'):
        raise ValueError('simulation_device must be cpu or gpu')
    if cfg.get('compute_dtype', 'float32') not in ('float32', 'bfloat16'):
        raise ValueError('compute_dtype must be float32 or bfloat16')
    if cfg.get('sampler', 'gumbel') not in ('gumbel', 'inverse_cdf'):
        raise ValueError('sampler must be gumbel or inverse_cdf')
    if cfg.get('convolution', 'direct') not in ('direct', 'patch', 'slice') or cfg.get('learner_convolution', 'direct') not in ('direct', 'patch', 'slice'):
        raise ValueError('Unknown convolution implementation')
    if cfg.get('historical_convolution', 'direct') not in ('direct', 'patch', 'slice', 'implicit'):
        raise ValueError('Unknown historical convolution implementation')
    if cfg.get('historical_convolution') == 'implicit' and (
            cfg['network_width'] != 16 or cfg.get('compute_dtype') != 'bfloat16'):
        raise ValueError('Implicit historical inference requires width16 BF16')
    if cfg.get('rollout_unroll', 1) not in (1, 2, 4, 8, 16, 32):
        raise ValueError('rollout_unroll must be a bounded static factor')
    if cfg["num_envs"] < 2 or cfg["num_envs"] % 2:
        raise ValueError("Use an even number of environments for paired initial seats")
    if cfg["history_size"] < 1 or cfg["history_interval"] < 1 or cfg["rollout_steps"] < 1:
        raise ValueError("Invalid rollout/history dimensions")
    if len(cfg["opponent_mix"]) != 4 or any(x < 0 for x in cfg["opponent_mix"]) or abs(sum(cfg["opponent_mix"])-1)>1e-6:
        raise ValueError("opponent_mix is [current, history, random, expansion] probabilities")
    for k in ("train", "practice", "heldout"):
        if (k != "heldout" and not seeds[k]) or len(seeds[k]) != len(set(seeds[k])):
            raise ValueError(f"Empty/duplicate seed split {k}")
    if any(set(seeds[a]) & set(seeds[b]) for a, b in (("train", "practice"), ("train", "heldout"), ("practice", "heldout"))):
        raise ValueError("Seed splits overlap")


if __name__ == "__main__":
    main()
