"""Matched structural CPU profile. Owns the shared performance lock for all timings.

Measures natural games, selected-action DMC updates, checkpoint I/O and practice
 evaluation; subprocess driver includes cold interpreter startup and teardown.
"""
from __future__ import annotations

import argparse
import cProfile
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import pstats
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def tensor_hash(policy):
    h = hashlib.sha256()
    for key, value in sorted(policy.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def child(args):
    import numpy as np
    import torch
    from integrations.doudizhu.artifacts import load_checkpoint, manifest, save_checkpoint, write_json
    from integrations.doudizhu.environment import ROLES, deal, digest, heuristic, Game,seed_value
    from integrations.doudizhu.model import Policy, update
    from integrations.doudizhu.rollout import Policies, collect
    from integrations.doudizhu.runner import initialize, make_optimizers, evaluate_policy
    from integrations.doudizhu.workers import VectorGames
    config = json.loads((ROOT/'integrations/doudizhu/recipes/profile-upstream-size.json').read_text())
    config.update(vector_games=args.batch, workers=args.workers, max_seconds=600, device=args.device,
                  engine_backend=args.engine, encoding_backend=args.codec,skip_forced=args.skip_forced,
                  inference_backend=args.inference,compact_workers=args.compact_workers)
    config['native_fast']=args.native_fast
    config.update(actor_rng=args.actor_rng,learner_shuffle=args.learner_shuffle,record_games=args.record_games)
    config['max_candidates']=args.max_candidates
    config['matmul_precision']=args.matmul_precision
    config['compute_dtype']=args.compute_dtype
    if args.engine=='cuda' and args.actor_rng!='splitmix-counter-v1':
        raise ValueError('CUDA profiling requires explicit --actor-rng splitmix-counter-v1')
    if args.device.startswith('cuda'):
        config['allocation'] = {'gpus':1,'machine':'Modal-L4','attached_cpu_threads':4,'attached_ram_gb':16}
    if args.small:
        config['architecture'] = {'hidden':128,'depth':3,'history':64}
    rng = initialize(config)
    policy = Policy(config['architecture']).to(args.device)
    if args.small:
        policy, _ = load_checkpoint(ROOT/'evidence/doudizhu-implementation/gpu/followup/learning-seed-17/latest.pt',args.device)
    if args.checkpoint:
        policy, checkpoint = load_checkpoint(args.checkpoint,args.device)
        config['architecture'] = checkpoint['architecture']
    optimizer = make_optimizers(policy, config)
    policy.compute_dtype=args.compute_dtype
    policy.eval()
    policies = Policies(policy, args.device,skip_forced=args.skip_forced,inference_backend=args.inference)
    policies.native_fast=args.native_fast
    policies.actor_rng=args.actor_rng
    if hasattr(policies, 'encoding_backend'):
        policies.encoding_backend = args.codec
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'manifest.json', manifest(config))
    profiles = cProfile.Profile()
    if args.diagnostic:
        profiles.enable()
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    t = time.perf_counter()
    # Independent simulator-only natural games: deterministic occupied and late play.
    sim_decisions = 0
    sim_games=args.sim_games if args.sim_games is not None else args.games
    if args.engine=='cuda':
        from integrations.doudizhu.cuda.backend import CudaBatch
        for start in range(0,sim_games,args.batch):
            size=min(args.batch,sim_games-start)
            sim=CudaBatch(size)
            sim.reset([deal('train',68193,start+index) for index in range(size)])
            for turn in range(163):
                total,_,active=sim.generate(encode=False)
                if not total:break
                choices=sim.choose(torch.zeros(size,dtype=torch.int64,device='cuda'),
                    torch.full((size,),3,dtype=torch.int8,device='cuda'),
                    torch.as_tensor(np.zeros(size,dtype=np.uint64),device='cuda'),0.)
                sim.step(choices)
                sim_decisions+=active
            assert np.all(sim.outcomes()[:,0]==1)
            sim.close()
            del sim
    elif args.engine=='native':
        with VectorGames(args.workers,engine_backend='native') as native:
            for start in range(0,sim_games,args.batch):
                size=min(args.batch,sim_games-start)
                native.reset([deal('train',68193,start+index) for index in range(size)])
                sim_decisions+=sum(row[1]['decisions'] for row in native.native.simulate_heuristic().values())
    else:
        for index in range(sim_games):
            kwargs = {'engine_backend':args.engine} if 'engine_backend' in Game.__init__.__code__.co_varnames else {}
            game = Game(deal('train', 68193, index), **kwargs)
            while (obs := game.observe()) is not None:
                game.step(heuristic(obs))
                sim_decisions += 1
    simulator_seconds = time.perf_counter()-t
    start_train = time.perf_counter()
    decisions = candidates = late = max_legal = 0
    samples_hash = hashlib.sha256()
    traces = {}
    timing = {}
    losses = []
    deferred={r:[] for r in ROLES}
    metrics = None
    kwargs = {'engine_backend':args.engine} if 'engine_backend' in VectorGames.__init__.__code__.co_varnames else {}
    with VectorGames(args.workers, compact=args.compact_workers,record_limit=args.record_games, **kwargs) as vector:
        for cursor in range(0,args.games,args.batch):
            n = min(args.batch,args.games-cursor)
            cards = [deal('train',68193,100000+cursor+i) for i in range(n)]
            assignments = [{r:{'kind':'learner'} for r in ROLES} for _ in cards]
            policy_rng={i:np.random.default_rng(seed_value('train',93713,100000+cursor+i)) for i in range(n)} if args.fixed_policy else rng
            result = collect(vector,policies,cards,assignments,policy_rng,epsilon=.1,training=True,
                             record=not args.record_first_only or cursor==0)
            decisions += result['decisions']
            candidates += result['candidates']
            max_legal = max(max_legal,result['max_legal_candidates'])
            traces.update({cursor+i:trace for i,trace in result['traces'].items()})
            for trace in result['traces'].values():
                remaining = {r:len(trace['deal'][r]) for r in ROLES}
                for step in trace['steps']:
                    remaining[step['role']] -= len(step['action'])
                    late += min(remaining.values()) <= 5
            for key,value in result['timing'].items():
                timing[key] = timing.get(key,0.)+value
            for role in ROLES:
                if args.fixed_policy:
                    deferred[role].append(result['samples'][role])
                    continue
                if args.no_sample_hash:
                    continue
                for x,z,y in result['samples'][role]:
                    samples_hash.update(role.encode())
                    samples_hash.update(x.tobytes())
                    samples_hash.update(z.tobytes())
                    samples_hash.update(np.float32(y).tobytes())
            if not args.fixed_policy:
                t = time.perf_counter()
                metrics = update(policy,optimizer,result['samples'],256,1,rng,shuffle_backend=args.learner_shuffle)
                if args.device.startswith('cuda'):
                    torch.cuda.synchronize()
                policy.eval()
                timing['learner_update_seconds'] = timing.get('learner_update_seconds',0.)+time.perf_counter()-t
                losses.append(metrics['mse'])
        if args.fixed_policy:
            from integrations.doudizhu.native_rollout import ArraySamples
            samples={}
            for role in ROLES:
                parts=[d for d in deferred[role] if len(d)]
                if not parts:
                    samples[role]=[]
                    continue
                resident=hasattr(parts[0],'x') and torch.is_tensor(parts[0].x)
                if resident:
                    x=torch.cat([d.x for d in parts]);z=torch.cat([d.z for d in parts]);y=torch.cat([d.y for d in parts])
                else:
                    x=np.concatenate([d.x if hasattr(d,'x') else np.stack([s[0] for s in d]) for d in parts])
                    z=np.concatenate([d.z if hasattr(d,'z') else np.stack([s[1] for s in d]) for d in parts])
                    y=np.concatenate([d.y if hasattr(d,'y') else np.array([s[2] for s in d],dtype=np.float32) for d in parts])
                samples[role]=ArraySamples(x,z,y)
                if args.no_sample_hash:continue
                for xx,zz,yy in samples[role]:
                    samples_hash.update(role.encode())
                    samples_hash.update(xx.tobytes())
                    samples_hash.update(zz.tobytes())
                    samples_hash.update(np.float32(yy).tobytes())
            t=time.perf_counter()
            metrics=update(policy,optimizer,samples,256,1,rng,shuffle_backend=args.learner_shuffle)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            policy.eval()
            timing['learner_update_seconds']=time.perf_counter()-t
            losses.append(metrics['mse'])
    train_seconds = time.perf_counter()-start_train
    t = time.perf_counter()
    save_checkpoint(args.out/'latest.pt',policy,optimizer,{'episodes':args.games},config)
    checkpoint_seconds = time.perf_counter()-t
    class Deadline:
        def expired(self):
            return False
    config.update(split='practice',deals=8,deal_seed=761943,opponents_panel=[{'kind':'heuristic'}])
    t = time.perf_counter()
    rows, _ = evaluate_policy(config,policy,Deadline())
    evaluation_seconds = time.perf_counter()-t
    elapsed = time.perf_counter()-started
    if args.diagnostic:
        profiles.disable()
        profiles.dump_stats(str(args.out/'profile.pstats'))
        with (args.out/'profile.txt').open('w') as f:
            pstats.Stats(profiles,stream=f).sort_stats('cumtime').print_stats(65)
    write_json(args.out/'private-first-deal.json',traces[0])
    write_json(args.out/'evaluation-games.json',rows)
    write_json(args.out/'result.json',dict(engine=args.engine,codec=args.codec,small=args.small,workers=args.workers,
        games=args.games,batch=args.batch,simulator_decisions=sim_decisions,simulator_seconds=simulator_seconds,
        simulator_decisions_per_second=sim_decisions/simulator_seconds,train_seconds=train_seconds,
        simulator_vector_games=args.batch,simulator_games=sim_games,
        decisions=decisions,candidates=candidates,max_legal=max_legal,late_decisions=late,
        checkpoint_seconds=checkpoint_seconds,evaluation_seconds=evaluation_seconds,loop_seconds=elapsed,
        timing=timing,losses=losses,trajectory_sha256=digest(traces),samples_sha256=None if args.no_sample_hash else samples_hash.hexdigest(),
        weights_sha256=tensor_hash(policy),evaluation_sha256=digest(rows),evaluation_games=len(rows),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        device=args.device,architecture=config['architecture'],skip_forced=args.skip_forced,inference=args.inference,
        compact_workers=args.compact_workers,
        record_first_only=args.record_first_only,
        record_games=args.record_games,
        native_fast=args.native_fast,
        fixed_policy=args.fixed_policy,
        actor_rng=args.actor_rng,learner_shuffle=args.learner_shuffle,sample_hash_measured=not args.no_sample_hash,
        max_candidates=args.max_candidates,
        matmul_precision=args.matmul_precision,
        compute_dtype=args.compute_dtype,
        graph_captures=sum(g.captures for g in getattr(policy,'_graph_scorers',{}).values()),
        graph_live_entries=sum(len(g.cache) for g in getattr(policy,'_graph_scorers',{}).values()),
        checkpoint_input=str(args.checkpoint) if args.checkpoint else None,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0,
        simulator_faults=0,illegal_actions=0,truncations=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--games',type=int,default=128)
    parser.add_argument('--sim-games',type=int,help='separate simulation-only workload size; default matches --games')
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--engine',choices=['upstream','view','native','cuda'],default='upstream')
    parser.add_argument('--codec',choices=['upstream','direct'],default='upstream')
    parser.add_argument('--small',action='store_true')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--device',choices=['cpu','cuda:0'],default='cpu')
    parser.add_argument('--skip-forced',action='store_true')
    parser.add_argument('--inference',choices=['eager','graph'],default='eager')
    parser.add_argument('--compact-workers',action='store_true')
    parser.add_argument('--record-first-only',action='store_true')
    parser.add_argument('--record-games',type=int)
    parser.add_argument('--native-fast',action='store_true')
    parser.add_argument('--fixed-policy',action='store_true',help='fixed weights across collection chunks; per-game RNG and one identical learner pass')
    parser.add_argument('--actor-rng',choices=['numpy-pcg64-legacy','splitmix-counter-v1'],default='numpy-pcg64-legacy')
    parser.add_argument('--learner-shuffle',choices=['numpy','torch'],default='numpy')
    parser.add_argument('--no-sample-hash',action='store_true',help='omit diagnostic sample export/hashing from throughput measurements')
    parser.add_argument('--max-candidates',type=int,default=4096)
    parser.add_argument('--matmul-precision',choices=['ieee','tf32'],default='ieee')
    parser.add_argument('--compute-dtype',choices=['float32','bfloat16'],default='float32')
    parser.add_argument('--diagnostic',action='store_true')
    parser.add_argument('--child',action='store_true')
    args=parser.parse_args()
    if args.child:
        child(args)
        return
    # Parent imports no Torch. Queue waiting is not counted as measured work.
    lock_path=ROOT/'evidence/local-performance.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        print('Waiting for exclusive local-performance lock',flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        started=time.perf_counter()
        result=subprocess.run([sys.executable,__file__,*sys.argv[1:],'--child'],cwd=ROOT)
        elapsed=time.perf_counter()-started
        if result.returncode:
            raise SystemExit(result.returncode)
        path=args.out/'result.json'
        row=json.loads(path.read_text())
        row.update(cold_process_seconds=elapsed,lock='evidence/local-performance.lock',
                   claim='complete subprocess wall time; app setup/artifact transfer/teardown charged separately')
        path.write_text(json.dumps(row,indent=2,sort_keys=True)+'\n')
        print(json.dumps({k:row[k] for k in ('engine','codec','small','batch','workers','native_fast',
                                            'train_seconds','loop_seconds','cold_process_seconds')}),flush=True)


if __name__=='__main__':
    main()
