"""python -m integrations.pokemon.cli train|evaluate|profile|test"""
from __future__ import annotations
import time
PROCESS_START = time.perf_counter()
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import torch
from .env import Vector, HERE, vector_for, evaluation_vector_for
from .network import Policy
from .provenance import Ledger, canonical, digest, manifest, write_json
from .rl import Inference, Population, Timings, rollout, update, inference_for, rollout_queue


def config_load(path):
    cfg = json.loads(Path(path).read_text())
    h = cfg["hardware"]
    if cfg.get('optimization','baseline') not in ('baseline','packed','graph','hybrid','stream','shared'):
        raise ValueError('Unknown optimization mode')
    if cfg.get('rollout_scheduler','static') not in ('static','queue'):raise ValueError('Unknown rollout scheduler')
    if cfg.get('episodes_per_update',h['workers']*h['environments_per_worker']) < h['workers']*h['environments_per_worker']:
        raise ValueError('episodes_per_update must cover every vector slot')
    if cfg.get('worker_timeout_seconds',60)<=0:raise ValueError('Positive worker timeout required')
    torch.set_num_threads(h["cpu_threads"])
    if h["cpu_threads"]+h["workers"] > h["attached_cpu_count"]:
        raise ValueError("Torch threads plus simulator workers exceed attached_cpu_count")
    if h["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA recipe requested but CUDA is unavailable; explicitly select CPU for local validation")
    if h["device"] != "cpu" and not h.get("attached_ram_gb"):
        raise ValueError("Set fixed attached_ram_gb for GPU allocation recipes")
    if h.get("cpu_affinity") is not None:
        if not hasattr(__import__("os"), "sched_setaffinity"):
            raise ValueError("cpu_affinity requires Linux")
        __import__("os").sched_setaffinity(0, h["cpu_affinity"])
    if cfg["iterations"] < 1 or cfg["max_decision_phases"] < 1 or cfg["history_size"] < 1:
        raise ValueError("Positive iterations, max_decision_phases and history_size required")
    if "SET_TO_" in h.get("machine", ""):
        raise ValueError("Set hardware.machine to the specified allocated machine before running")
    if cfg["allocation_seconds"] <= 0:
        raise ValueError("allocation_seconds must be positive")
    for split in ("practice", "heldout"):
        schedule = json.loads((HERE/"recipes"/f"{split}.json").read_text())
        held_seeds = {g["seed"] for g in schedule["games"]}
        n = cfg.get("episodes_per_update",h["workers"]*h["environments_per_worker"])*cfg["iterations"]
        if any(cfg["training_seed_start"] <= s < cfg["training_seed_start"]+n for s in held_seeds):
            raise ValueError(f"Training seed schedule overlaps {split}")
    return cfg

def initialize(cfg):
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    return Policy(**cfg["network"]).to(cfg["hardware"]["device"])

def signature(cfg):
    return hashlib.sha256(canonical({k:v for k,v in cfg.items() if k not in ("iterations", "allocation_seconds")}).encode()).hexdigest()

def load_policy(path, cfg):
    cp = torch.load(path, map_location=cfg["hardware"]["device"], weights_only=False)
    model = Policy(**cp["network"]).to(cfg["hardware"]["device"])
    model.load_state_dict(cp["model"])
    model.eval()
    return model, cp

def executable_signature():
    files = sorted([*HERE.glob("*.py"), *HERE.glob("*.cjs"), HERE/"requirements.txt"])
    return hashlib.sha256(canonical({p.name:digest(p) for p in files}).encode()).hexdigest()

def checkpoint(path, model, optimizer, cfg, iteration, rng, population, ledger, manifest_hash):
    data = {"version":1,"executable_signature":executable_signature(),"model":model.state_dict(),"optimizer":optimizer.state_dict(),"network":cfg["network"],
            "config_signature":signature(cfg),"iteration":iteration,"next_battle_index":iteration*cfg.get("episodes_per_update",cfg["hardware"]["workers"]*cfg["hardware"]["environments_per_worker"]),
            "numpy_rng":rng.bit_generator.state,"torch_rng":torch.get_rng_state(),"python_rng":random.getstate(),
            "cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "history":population.states(),"allocated_worker_seconds":ledger.elapsed,"manifest_sha256":manifest_hash,
            "resume_boundary":"between complete vector batches; no simulator state in flight"}
    temp = path.with_suffix(".tmp")
    torch.save(data, temp)
    temp.replace(path)


def evaluation(vector, model, schedule, cfg, timings, frozen=None, deadline=None):
    inference = inference_for(cfg,timings)
    scores, results = [], []
    started = time.perf_counter()
    games = schedule["games"]
    rng = np.random.default_rng(0)
    ecfg = dict(cfg, max_decision_phases=schedule["max_decision_phases"], faint_reward=0)
    for start in range(0, len(games), vector.size):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        group = games[start:start+vector.size]
        real_count = len(group)
        group = group + [group[-1]]*(vector.size-len(group))
        assignments = []
        for spec in group:
            opponent = frozen if spec["opponent"] == "checkpoint" else spec["opponent"]
            if opponent is None:
                raise ValueError("Checkpoint opponent needs --opponent-checkpoint")
            pair = [opponent,opponent]
            pair[spec["learner_seat"]] = model
            assignments.append(pair)
        _, report = rollout(vector, model, assignments, [s["seed"] for s in group], ecfg, rng, inference,
                            timings, deadline, deterministic=schedule.get("deterministic_policy",True))
        for spec, result in zip(group[:real_count], report["results"][:real_count], strict=True):
            natural = result["terminated"]
            score = (result["returns"][spec["learner_seat"]]+1)/2 if natural else None
            results.append(dict(result, learner_seat=spec["learner_seat"],opponent=spec["opponent"],score=score))
            if natural:
                scores.append(score)
    truncated = sum(r["truncated"] for r in results)
    # A truncated schedule is explicitly incomplete; no conditioning-away of difficult games.
    complete = len(results)==len(games) and truncated==0
    summary = {"split":schedule["split"],"scheduled_games":len(games),"attempted_games":len(results),
               "natural_terminals":len(scores),"truncations":truncated,"valid_complete_schedule":complete,
               "score":float(np.mean(scores)) if complete else None,
               "completed_only_score_diagnostic":float(np.mean(scores)) if scores else None,
               "wins":sum(s==1 for s in scores),"draws":sum(s==.5 for s in scores),"losses":sum(s==0 for s in scores),
               "rejected_visible_choices":sum(sum(r["rejections"]) for r in results),
               "seconds":time.perf_counter()-started,"games":results}
    return summary


def train(args, cfg, out, ledger, manifest_hash):
    model = initialize(cfg)
    device = cfg["hardware"]["device"]
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], eps=1e-5)
    rng = np.random.default_rng(cfg["seed"])
    iteration = 0
    population = Population(cfg, model, device)
    if args.resume:
        cp = torch.load(args.resume, map_location=device, weights_only=False)
        if cp.get("executable_signature") != executable_signature():
            raise ValueError("Resume executable source changed; use the recorded source snapshot/environment")
        if cp["config_signature"] != signature(cfg):
            raise ValueError("Resume configuration changed; only iterations/allocation_seconds may change")
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        population = Population(cfg, model, device, cp["history"])
        rng.bit_generator.state = cp["numpy_rng"]
        torch.set_rng_state(cp["torch_rng"].cpu())
        random.setstate(cp["python_rng"])
        if cp["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in cp["cuda_rng"]])
        iteration = cp["iteration"]
        prior = cp["allocated_worker_seconds"]
        accounting = Path(args.resume).parent/"accounting.json"
        if accounting.is_file():
            prior = max(prior, json.loads(accounting.read_text())["allocated_worker_seconds"])
        ledger.prior = prior
    timings = Timings()
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    reports = []
    h = cfg["hardware"]
    deadline = ledger.start+cfg["allocation_seconds"]-ledger.prior
    episodes=cfg.get('episodes_per_update',h['workers']*h['environments_per_worker'])
    collect=rollout_queue if cfg.get('rollout_scheduler','static')=='queue' else rollout
    if collect is rollout and episodes!=h['workers']*h['environments_per_worker']:
        raise ValueError('Static rollout episodes_per_update must equal vector size')
    with vector_for(cfg) as vector:
        inference = inference_for(cfg,timings)
        while iteration < cfg["iterations"] and time.perf_counter() < deadline:
            seeds = list(range(cfg["training_seed_start"]+iteration*episodes, cfg["training_seed_start"]+(iteration+1)*episodes))
            assignments = population.assignments(episodes, iteration*episodes, rng, model)
            samples, report = collect(vector,model,assignments,seeds,cfg,rng,inference,timings,deadline)
            # Do not begin an optimizer phase after allocation expiry. Partial batches are
            # recorded and discarded; untrained samples never count as PPO updates.
            stats = update(model,optimizer,samples,cfg,device,timings) if time.perf_counter()<deadline else {"samples":0,"skipped":"allocation_expired"}
            iteration += 1
            if iteration % cfg["snapshot_every"] == 0:
                population.add(model)
            dev = cfg.get("development_evaluation")
            if dev and iteration % dev["every"] == 0 and time.perf_counter()<deadline:
                schedule = json.loads((HERE/"recipes"/dev["schedule"]).read_text())
                if schedule["split"] != "practice":
                    raise ValueError("Training may evaluate only the practice split")
                if vector.size>len(schedule['games']):
                    with evaluation_vector_for(cfg,len(schedule['games'])) as ev:
                        report["development_evaluation"]=evaluation(ev,model,schedule,cfg,timings,deadline=deadline)
                else:report["development_evaluation"] = evaluation(vector,model,schedule,cfg,timings,deadline=deadline)
            report.update(iteration=iteration,ppo=stats,allocated_worker_seconds=ledger.elapsed)
            reports.append(report)
            with (out/"metrics.jsonl").open("a") as f:
                f.write(canonical(report)+"\n")
            start = time.perf_counter()
            checkpoint(out/"checkpoint.pt",model,optimizer,cfg,iteration,rng,population,ledger,manifest_hash)
            timings.add("checkpoint",time.perf_counter()-start)
            print(canonical({"iteration":iteration,"samples":len(samples),"terminals":report["natural_terminals"],
                             "truncations":report["truncations"],"ppo":stats,"allocated_worker_seconds":ledger.elapsed}),flush=True)
    change = sum(float((v.detach().cpu()-initial[k]).square().sum()) for k,v in model.state_dict().items())**.5
    result = {"command":"train","iterations_completed":iteration,"iterations_this_process":len(reports),
              "policy_samples":sum(r["policy_samples"] for r in reports),"trained_samples":sum(r["ppo"]["samples"] for r in reports),
              "parameter_change_l2":change,"natural_terminals":sum(r["natural_terminals"] for r in reports),
              "truncations":sum(r["truncations"] for r in reports),"timings":timings.result(),
              "manifest_sha256":manifest_hash,"gpu_executed":device != "cpu",**ledger.result()}
    write_json(out/"results.json",result)
    return result


def evaluate(args,cfg,out,ledger,manifest_hash,schedule):
    model = load_policy(args.checkpoint,cfg)[0] if args.checkpoint else initialize(cfg)
    model.eval()
    frozen = load_policy(args.opponent_checkpoint,cfg)[0] if args.opponent_checkpoint else None
    timings = Timings()
    h = cfg["hardware"]
    with evaluation_vector_for(cfg,len(schedule['games'])) as vector:
        evaluation_topology={'workers':len(vector.workers),'environments_per_worker':vector.per_worker}
        result = evaluation(vector,model,schedule,cfg,timings,frozen,ledger.start+cfg["allocation_seconds"])
    result.update(command="evaluate",evaluation_topology=evaluation_topology,checkpoint_sha256=digest(args.checkpoint) if args.checkpoint else None,
                  opponent_checkpoint_sha256=digest(args.opponent_checkpoint) if args.opponent_checkpoint else None,
                  manifest_sha256=manifest_hash,timings=timings.result(),**ledger.result())
    write_json(out/"results.json",result)
    return {k:v for k,v in result.items() if k not in ("games","timings")}


def profile(args,cfg,out,ledger,manifest_hash):
    h = cfg["hardware"]
    seeds = list(range(cfg["training_seed_start"],cfg["training_seed_start"]+h["workers"]*h["environments_per_worker"]))
    rows = []
    deadline = ledger.start+cfg["allocation_seconds"]
    variants = [("scalar_uncached",False,False,False,False),("batch_uncached",True,False,False,False),("batch_cached",True,True,False,False),("batch_packed",True,True,True,False),("batch_stream",True,True,True,True)]
    for trial in range(args.repeats):
        # Alternate order to reduce first-run/temperature bias.
        for name,batched,optimized,packed,stream in (variants if trial%2==0 else list(reversed(variants))):
            if time.perf_counter() >= deadline:
                raise RuntimeError("Profile allocation exhausted; increase local allocation_seconds or reduce repeats")
            start = time.perf_counter()
            model = initialize(cfg)
            if args.checkpoint:
                model = load_policy(args.checkpoint,cfg)[0]
            optimizer = torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"],eps=1e-5)
            timings = Timings()
            with Vector(h["workers"],h["environments_per_worker"],optimized=optimized,packed=packed,binary=stream,token_cache=stream,mon_cache=stream and cfg.get("mon_cache",False),fast_callbacks=stream and cfg.get("fast_callbacks",False)) as vector:
                pcfg=dict(cfg,optimization='stream' if stream else 'packed' if packed else 'baseline')
                policy_inference=inference_for(pcfg,timings) if packed else Inference(h['device'],timings,batched)
                samples, report = rollout(vector,model,[[model,model] for _ in seeds],seeds,pcfg,np.random.default_rng(0),
                                          policy_inference,timings,deadline,deterministic=True)
                traces = [x["sha256"] for x in vector.trace()]
                stats = update(model,optimizer,samples,pcfg,h["device"],timings)
                # Checkpoint serialization is included in the end-to-end profile.
                torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict()},out/"profile-checkpoint.pt")
            seconds = time.perf_counter()-start
            rows.append({"trial":trial,"variant":name,"batched_inference":batched,"cached_encoding":optimized,"binary_transport":stream,"elapsed_seconds":seconds,"policy_decisions":len(samples),
                         "decisions_per_second":len(samples)/seconds,"traces":traces,"timings":timings.result(),"ppo":stats,
                         "natural_terminals":report["natural_terminals"],"truncations":report["truncations"]})
    for trial in range(args.repeats):
        pair = [r for r in rows if r["trial"]==trial]
        if any(r["traces"] != pair[0]["traces"] for r in pair):
            raise AssertionError("Batched/scalar policies changed battle traces; comparison is not controlled")
    scalar = statistics.median(r["elapsed_seconds"] for r in rows if r["variant"]=="scalar_uncached")
    batched = statistics.median(r["elapsed_seconds"] for r in rows if r["variant"]=="batch_stream")
    result = {"command":"profile","same_trace":True,"scope":"persistent CPU workers, observation encoding/IPC, policy transfer/inference, PPO updates, checkpoint, worker startup/teardown",
              "scalar_median_seconds":scalar,"batched_median_seconds":batched,"end_to_end_speedup":scalar/batched,
              "variant_median_seconds":{name:statistics.median(r["elapsed_seconds"] for r in rows if r["variant"]==name) for name,_,_,_,_ in variants},
              "rows":rows,"manifest_sha256":manifest_hash,"gpu_executed":h["device"] != "cpu",**ledger.result()}
    write_json(out/"results.json",result)
    return {k:v for k,v in result.items() if k != "rows"}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command",choices=("train","evaluate","profile"))
    parser.add_argument("--config",default=str(HERE/"recipes/smoke.json"))
    parser.add_argument("--output",required=True)
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    parser.add_argument("--schedule",default=str(HERE/"recipes/practice.json"))
    parser.add_argument("--opponent-checkpoint")
    parser.add_argument("--iterations",type=int)
    parser.add_argument("--repeats",type=int,default=2)
    args = parser.parse_args()
    cfg = config_load(args.config)
    if args.iterations is not None:
        cfg["iterations"] = args.iterations
        # Recheck split disjointness after overrides.
        for name in ("practice","heldout"):
            protected = json.loads((HERE/"recipes"/(name+".json")).read_text())
            end = cfg["training_seed_start"]+cfg["iterations"]*cfg["hardware"]["workers"]*cfg["hardware"]["environments_per_worker"]
            if any(cfg["training_seed_start"]<=g["seed"]<end for g in protected["games"]):
                raise ValueError("Overridden training seeds overlap evaluation")
    if args.command == "train" and cfg.get("development_evaluation"):
        dev_schedule = json.loads((HERE/"recipes"/cfg["development_evaluation"]["schedule"]).read_text())
        if dev_schedule["split"] != "practice":
            raise ValueError("Training may evaluate only the practice split")
    out = Path(args.output).resolve()
    schedule = json.loads(Path(args.schedule).read_text()) if args.command == "evaluate" else None
    if schedule and args.opponent_checkpoint:
        schedule = copy.deepcopy(schedule)
        schedule["games"] = [dict(g,opponent="checkpoint") for g in schedule["games"]]
        schedule["opponent_sha256"] = digest(args.opponent_checkpoint)
    ledger = Ledger(PROCESS_START)
    input_artifacts = {name: {"path":str(Path(value).resolve()),"sha256":digest(value)} for name,value in (("resume",args.resume),("checkpoint",args.checkpoint),("opponent_checkpoint",args.opponent_checkpoint)) if value}
    manifest_hash = manifest(out,cfg,args.command,{"evaluation":schedule,"inputs":input_artifacts} if schedule or input_artifacts else None)
    try:
        if args.command == "train":
            result = train(args,cfg,out,ledger,manifest_hash)
        elif args.command == "evaluate":
            result = evaluate(args,cfg,out,ledger,manifest_hash,schedule)
        else:
            result = profile(args,cfg,out,ledger,manifest_hash)
        print(json.dumps(result,indent=2,sort_keys=True),flush=True)
    except BaseException as exc:
        write_json(out/"failure.json",{"type":type(exc).__name__,"category":getattr(exc,"category","training_failure"),"error":str(exc),**ledger.result()})
        raise
    finally:
        write_json(out/"accounting.json",ledger.result())

if __name__ == "__main__":
    main()
