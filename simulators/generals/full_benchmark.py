"""Full-v2 neural training workload, paired warm profiles, exact simulator replay."""
import time
START=time.monotonic()
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS','0')
import jax
import jax.numpy as jnp
import numpy as np
from .environment import build_pool,map_dimensions,observations,encode,action_mask,RULESET
from .runtime import HERE,write_json,sources,save_checkpoint,Ledger
from .model import Policy
from .learning import create_train_state,initial_carry
from .efficiency import make_iteration,make_collect_optimized
from .structural_experiment import neural_action_simulator_profile


def timed(fn,*args):
    start=time.monotonic();result=fn(*args);jax.block_until_ready(result)
    return result,time.monotonic()-start


def profile(out,recipe,batch,repeats=7):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);binding=sources()
    cfg=json.loads(Path(recipe).read_text());cfg.update(num_envs=batch,
        minibatches=max(4,batch*32//4096),compact_checkpoint=True,allocated_worker_seconds=1800)
    from .autoresearch.contract import seed_manifest
    seed_data = seed_manifest()
    pool=build_pool(seed_data['train']);model=Policy(16,compute_dtype='bfloat16')
    train=create_train_state(model,jax.random.PRNGKey(81),cfg)
    bank=jax.tree.map(lambda p:jnp.repeat(p[None],4,0),train.params)
    carry=initial_carry(pool,jax.random.PRNGKey(82),cfg,bank)
    fn=make_iteration(model,cfg,'optimized');pre=[]
    # Actual PPO-driven games, naturally crossing2048; no artificial clock or
    # hand-injected armies are used as a performance workload.
    for u in range(1,69):
        train,carry,m=fn(train,bank,carry,pool);jax.block_until_ready(m)
        host=np.asarray(m)
        if not np.isfinite(host).all() or host[9]:raise AssertionError('Invalid/nonfinite preconditioning')
        pre.append(dict(update=u,captures=int(host[6]+host[7]),caps=int(host[8])))
        if u%4==0:bank=jax.tree.map(lambda b,p:b.at[(u//4-1)%4].set(p),bank,train.params)
    save_checkpoint(out/'starting.msgpack',train,bank,carry,pool,cfg,binding,68,Ledger(START,1800),17)
    variants={}
    if batch==128:
        variants['legacy-f32']=(Policy(16),dict(cfg,compute_dtype='float32',execution='baseline',
            packed_rollout=False,historical_convolution='direct',compact_opponent_inference=False,sparse_reset_capacity=0),'baseline')
        variants['optimized-f32']=(Policy(16),dict(cfg,compute_dtype='float32',historical_convolution='direct'),'optimized')
    variants['slice-bf16']=(model,dict(cfg,historical_convolution='slice'),'optimized')
    variants['implicit-bf16']=(model,cfg,'optimized')
    funcs={};rows={}
    for name,(net,config,execution) in variants.items():
        fn=make_iteration(net,config,execution)
        (tr,nc,m),cold=timed(fn,train,bank,carry,pool)
        host=np.asarray(m)
        if not np.isfinite(host).all() or host[9]:raise AssertionError('Invalid/nonfinite profile')
        funcs[name]=fn;rows[name]=dict(config=config,cold_seconds=cold,trials=[])
    for repeat in range(repeats):
        for name in (list(funcs) if repeat%2==0 else list(funcs)[::-1]):
            (tr,nc,m),seconds=timed(funcs[name],train,bank,carry,pool)
            host=np.asarray(m)
            if not np.isfinite(host).all() or host[9]:raise AssertionError('Invalid/nonfinite profile')
            rows[name]['trials'].append(dict(seconds=seconds,metrics=host.tolist()))
    for row in rows.values():
        row['median_seconds']=float(np.median([r['seconds'] for r in row['trials']]))
        row['learner_samples_per_second']=batch*32/row['median_seconds']
    tc=dict(cfg,trace_actions=True)
    final,data=make_collect_optimized(model,tc)(train.params,bank,carry,pool)
    phases=neural_action_simulator_profile(carry,final,pool,data,tc,repeats)
    result=dict(rules=RULESET,source=binding,hardware=[d.device_kind for d in jax.devices()],
        batch=batch,rollout_steps=32,model_width=16,preconditioning=pre,variants=rows,
        neural_action_simulator=phases,train_dimension_counts={str(tuple(d)):int(n) for d,n in zip(*np.unique(map_dimensions(seed_data['train']),axis=0,return_counts=True))},
        full_child_seconds=time.monotonic()-START,
        accounting='Warm medians exclude cold/setup; complete parent worker/provider allocation is separately retained. One learner sample and two player actions per joint tick.')
    write_json(out/'profile.json',result)
    print(json.dumps(dict(batch=batch,rates={k:v['learner_samples_per_second'] for k,v in rows.items()})),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True)
    p.add_argument('--recipe',required=True);p.add_argument('--batch',type=int,default=128)
    p.add_argument('--mode',choices=('profile','validate'),default='profile')
    a=p.parse_args()
    if a.mode=='validate':
        from .full_validation import validate
        validate(a.output,a.recipe,cpu_parity=True)
    else:profile(a.output,a.recipe,a.batch)


if __name__=='__main__':main()
