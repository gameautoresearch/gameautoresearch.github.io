"""Bounded neural efficiency experiments; CPU and CUDA run the identical workload.

Invoke via Modal launcher, or locally under the shared advisory flock. The legacy
trained snapshot is explicitly migrated only after checking rules/engine pins;
paired variants must then pass trajectory/GAE parity on the current device.
"""
import time
START=time.monotonic()
import argparse
import json
from pathlib import Path
import os
import sys
import subprocess
import threading
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from .model import Policy
from .learning import create_train_state, initial_carry, make_collect, make_update, Carry, sample
from .environment import (build_pool, observations, encode, action_mask, transition, reset_done, decode_action)
from .runtime import HERE, sources, write_json, read_checkpoint, Ledger, save_checkpoint, digest


def timed(fn,*args):
    start=time.perf_counter()
    value=fn(*args)
    jax.block_until_ready(value)
    return value,time.perf_counter()-start


def summary_trials(fn,args,repeats=5):
    _,cold=timed(fn,*args)
    trials=[]
    for _ in range(repeats):
        _,elapsed=timed(fn,*args)
        trials.append(elapsed)
    return dict(first_call_seconds=cold,warm_seconds=trials,median_seconds=float(np.median(trials)))


def cfg_for(b=64,seed=42):
    cfg=json.loads((HERE/'recipes/smoke.json').read_text())
    cfg.update(num_envs=b,rollout_steps=32,minibatches=4,seed=seed,max_updates=96,
               allocated_worker_seconds=800,hardware=dict(gpu_model='NVIDIA L4',gpu_count=1,attached_cpu_cores=4,attached_ram_gib=16))
    return cfg


def trained_state(cfg,model,snapshot):
    train=create_train_state(model,jax.random.PRNGKey(1),cfg)
    meta,(train,bank,old,pool)=read_checkpoint(snapshot,train)
    current=sources()
    for field in ('commit','tree_sha256','rules_sha256'):
        if current[field]!=meta['source_binding'][field]:raise ValueError(f'Legacy snapshot {field} mismatch')
    indices=jnp.arange(cfg['num_envs']) % old.seats.shape[0]
    c=Carry(jax.tree.map(lambda p:p[indices],old.states),old.key,old.seats[indices],old.opponents[indices],
            old.history_ids[indices],jax.tree.map(lambda p:p[indices],old.history_params),old.episodes[indices])
    return train,bank,c,pool,dict(payload_sha256=meta['payload_sha256'],original_source=meta['source_binding'],
                migration='Same frozen rules and engine; performance implementation migration requires paired trajectory and GAE checks.')


def phase_profile(model,train,bank,c,pool,cfg,collect,update):
    (after,data),collect_cold=timed(collect,train.params,bank,c,pool)
    _,update_cold=timed(update,train,data,jax.random.PRNGKey(79))
    trials=[]
    for i in range(5):
        start=time.perf_counter()
        (after,data),ct=timed(collect,train.params,bank,c,pool)
        (_,metrics),ut=timed(update,train,data,jax.random.PRNGKey(80+i))
        trials.append(dict(rollout_seconds=ct,update_seconds=ut,full_seconds=time.perf_counter()-start,
                           completed_games=int(data['terminated'].sum()+data['truncated'].sum()),
                           captures=int(data['terminated'].sum()),caps=int(data['truncated'].sum()),
                           invalid=int(data['invalid'].sum()),loss=float(metrics[0])))
    obs=observations(c.states)
    x,m=encode(obs),action_mask(obs)
    rows=jnp.arange(cfg['num_envs']);opp=1-c.seats
    actor=jax.jit(lambda p,x,m:model.apply({'params':p},x.reshape((-1,8,8,14)),m.reshape((-1,513))))
    historical=jax.jit(jax.vmap(lambda p,x,m:model.apply({'params':p},x[None],m[None])))
    ids=sample(jax.random.PRNGKey(3),jnp.where(m,0.,-1e9))
    actions=decode_action(ids)
    boundary=transition(c.states,actions)
    indices=jnp.arange(cfg['num_envs']) % pool.time.shape[0]
    phases=dict(observation=summary_trials(jax.jit(observations),(c.states,)),
        encoding_masks=summary_trials(jax.jit(lambda o:(encode(o),action_mask(o))),(obs,)),
        current_inference=summary_trials(actor,(train.params,x,m)),
        historical_inference=summary_trials(historical,(c.history_params,x[rows,opp],m[rows,opp])),
        simulation=summary_trials(jax.jit(transition),(c.states,actions)),
        reset=summary_trials(jax.jit(reset_done),(boundary.final_state,jnp.ones_like(c.seats,bool),pool,indices)),
        learner_update=summary_trials(update,(train,data,jax.random.PRNGKey(80))))
    host=np.asarray(jax.device_get(x)); metrics=jnp.zeros((24,),jnp.float32)
    phases['encoded_observation_h2d']=summary_trials(lambda a:jax.device_put(a),(host,))
    phases['compact_metrics_d2h']=summary_trials(jax.device_get,(metrics,))
    total=float(np.median([r['full_seconds'] for r in trials]));n=cfg['num_envs']*cfg['rollout_steps']
    return dict(collect_first_seconds=collect_cold,update_first_seconds=update_cold,trials=trials,phases=phases,
                learner_samples_per_second=n/total,legal_player_decisions_per_second=2*n/total,updates_per_second=1/total,
                game_tick_definition='both engine players choose actions; one learner sample per game tick',
                workload=dict(batch=cfg['num_envs'],horizon=cfg['rollout_steps'],network_width=cfg['network_width'],
                 state_times=np.asarray(c.states.time).tolist(),total_army=np.asarray(c.states.armies.sum((1,2))).tolist(),
                 learner_land=np.asarray(c.states.ownership.sum((2,3))).tolist()),
                note='Isolated component timings are diagnostic and do not sum to fused pipeline time. Transfers are measured probes; rollout states/observations/masks/weights stay device-resident.')


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--stage',choices=['probe','compare','learning','cli'],default='probe')
    p.add_argument('--output',required=True)
    p.add_argument('--snapshot',required=True)
    p.add_argument('--batches',default='16,64,128')
    p.add_argument('--steps',type=int,default=96)
    p.add_argument('--seeds',default='42,43,44')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    jax.config.update('jax_compilation_cache_dir','/tmp/generals-xla-cache')
    jax.config.update('jax_persistent_cache_min_compile_time_secs',0)
    smi=None
    try:
        smi=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw',
            '--format=csv,noheader,nounits','-l','1'],stdout=(out/'gpu-utilization.csv').open('w'),stderr=subprocess.DEVNULL)
    except FileNotFoundError:pass
    try:
        versions=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout
        (out/'packages.txt').write_text(versions)
        info=dict(backend=jax.default_backend(),devices=[str(d) for d in jax.devices()],device_kinds=[d.device_kind for d in jax.devices()],
                  jax=jax.__version__,source=sources(),clock_includes='imports, setup/JIT, probes, transfers, training, evaluation and artifact IO')
        if jax.default_backend()=='gpu':
            info['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True)
        write_json(out/'manifest.json',info)
        import tarfile
        with tarfile.open(out/'executed-integration-source.tgz','w:gz') as tar:
            for file in HERE.rglob('*'):
                if "autoresearch" in file.relative_to(HERE).parts:
                    continue
                if file.is_file() and '__pycache__' not in file.parts and '.pytest_cache' not in file.parts:
                    tar.add(file,arcname=str(Path('integrations/generals')/file.relative_to(HERE)))
        if a.stage=='probe':
            results=[]
            for b in map(int,a.batches.split(',')):
                cfg=cfg_for(b);model=Policy(cfg['network_width'])
                train,bank,c,pool,provenance=trained_state(cfg,model,a.snapshot)
                jax.block_until_ready((train,bank,c,pool))
                result=phase_profile(model,train,bank,c,pool,cfg,make_collect(model,cfg),make_update(model,cfg))
                results.append(result)
                write_json(out/'probe.json',dict(results=results,provenance=provenance,elapsed_seconds=time.monotonic()-START))
                print(json.dumps(dict(batch=b,samples_per_second=result['learner_samples_per_second'],phases={k:v['median_seconds'] for k,v in result['phases'].items()})),flush=True)
        elif a.stage=='cli':
            from .efficiency import run_cli_checks
            run_cli_checks(out)
        else:
            from .efficiency import run_comparison,run_learning
            fn=run_comparison if a.stage=='compare' else run_learning
            fn(a,out)
        try:info['device_memory_stats']=jax.devices()[0].memory_stats()
        except Exception:info['device_memory_stats']=None
        info['elapsed_seconds']=time.monotonic()-START
        write_json(out/'receipt.json',info)
    finally:
        if smi is not None:smi.terminate();smi.wait()


if __name__=='__main__':main()
