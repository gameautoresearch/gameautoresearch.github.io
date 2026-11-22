"""Full-v2 map/rule oracle, optimized rollout and source-bound resume checks."""
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .environment import (RULESET,build_pool,map_dimensions,make_env,observations,
    encode,action_mask,decode_action,transition,reset_done,scripted_logits)
from .runtime import HERE,write_json,sources,save_checkpoint,read_checkpoint,Ledger
from .model import Policy
from .learning import create_train_state,initial_carry,sample,make_collect
from .efficiency import make_collect_optimized,make_iteration


def identical(a,b):
    aa,ta=jax.tree.flatten(a);bb,tb=jax.tree.flatten(b)
    return ta==tb and all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(aa,bb))


def map_oracle(out, cpu_parity=False):
    """Every declared rectangle, exact standalone upstream generation and BFS."""
    from .audit import distances
    dims=np.array([(h,w) for h in range(17,24) for w in range(17,24)])
    seeds=np.arange(1000000,1000049,dtype=np.uint32)
    start=time.monotonic();pool=build_pool(seeds,dimensions=dims);jax.block_until_ready(pool)
    env=make_env();records=[]
    for i,(h,w) in enumerate(dims):
        raw=env._make_single_state_fixed(jax.random.PRNGKey(seeds[i]),int(h),int(w))
        if not identical(jax.tree.map(lambda a:a[i],pool),raw):raise AssertionError('Map differs from pinned upstream')
        s=jax.device_get(raw);pos=s.general_positions
        distance=int(distances(s.passable,tuple(pos[0]))[tuple(pos[1])])
        if distance<17 or distance>=9999:raise AssertionError('Invalid full-distance spawn')
        if s.passable[h:,:].any() or s.passable[:,w:].any():raise AssertionError('Padding is playable')
        if not (9<=s.castles.sum()<=11):raise AssertionError('Castle range differs from pinned randint semantics')
        armies=s.armies[s.castles]
        if not ((40<=armies)&(armies<51)).all():raise AssertionError('Castle armies differ')
        records.append(dict(seed=int(seeds[i]),height=int(h),width=int(w),spawn_distance=distance,
            castles=int(s.castles.sum()),exact_upstream=True))
    if cpu_parity and jax.default_backend()=='gpu':
        with jax.default_device(jax.devices('cpu')[0]):
            cpu=build_pool(seeds,dimensions=dims)
        if not identical(pool,cpu):raise AssertionError('CPU/CUDA map source differs')
    result=dict(records=records,all49_rectangles=True,exact_cpu_cuda=cpu_parity,
        elapsed_seconds=time.monotonic()-start,rules=RULESET)
    write_json(Path(out)/'map-oracle.json',result)
    return pool


def rule_oracle(pool,out):
    """Compare every state field/observation/reward/boundary on every tick.

    The original public Env.step controls one path. Only pool_idx metadata is
    adapted to replay the external reset schedule, exactly as in the old study.
    """
    b=pool.time.shape[0];env=make_env();env.pool_size=b
    rawstep=jax.vmap(env.step,in_axes=(0,0,None))
    @jax.jit
    def run(states,key,pool):
        def tick(c,_):
            s,key=c;key,ka,kr=jax.random.split(key,3)
            obs=observations(s);mask=action_mask(obs)
            ids=sample(ka,scripted_logits(obs,(jnp.arange(b)%2)[:,None]))
            indices=jax.random.randint(kr,(b,),0,b)
            r=transition(s,decode_action(ids));done=r.terminated|r.truncated
            current=reset_done(r.final_state,done,pool,indices)
            ts,original=rawstep(s._replace(pool_idx=indices),decode_action(ids),pool)
            original=original._replace(pool_idx=jnp.where(done,pool.pool_idx[indices],s.pool_idx))
            raw_final=ts.last_state._replace(pool_idx=r.final_state.pool_idx)
            def mismatches(a,b):
                return sum(jnp.sum(x!=y) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)))
            observed=observations(current)
            errors=mismatches(current,original)+mismatches(r.final_state,raw_final)
            errors+=mismatches(observed,ts.observation)
            errors+=jnp.sum(r.reward!=ts.reward)+jnp.sum(r.terminated!=ts.terminated)+jnp.sum(r.truncated!=ts.truncated)
            errors+=jnp.sum(action_mask(observed)!=action_mask(ts.observation))
            invalid=(~jnp.take_along_axis(mask,ids[...,None],-1)[...,0]).sum()
            return (current,key),jnp.array([errors,invalid,r.terminated.sum(),r.truncated.sum()])
        return jax.lax.scan(tick,(states,key),None,length=2048)
    start=time.monotonic();(final,key),counts=run(pool,jax.random.PRNGKey(813),pool);jax.block_until_ready(counts)
    total=np.asarray(counts).sum(0)
    if total[0] or total[1] or total[3]==0:raise AssertionError(f'Full rule oracle failed:{total}')
    result=dict(joint_ticks=b*2048,per_tick_all_state_fields_observations_masks_rewards_boundaries_exact=True,
        invalid_actions=int(total[1]),natural_captures=int(total[2]),truncations=int(total[3]),
        elapsed_seconds=time.monotonic()-start,metadata_adapter='External map reset schedule; pool_idx only')
    write_json(Path(out)/'rule-oracle.json',result)
    return result


def validate(out,recipe,cpu_parity=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    binding=sources();write_json(out/'source-manifest.json',binding)
    pool49=map_oracle(out,cpu_parity);rule_oracle(pool49,out)
    cfg=json.loads(Path(recipe).read_text());cfg.update(num_envs=128,minibatches=4,allocated_worker_seconds=1800)
    # Exact original/optimized float32 parity, including a real cap cohort.
    fcfg=dict(cfg,compute_dtype='float32',historical_convolution='direct',packed_rollout=False)
    model=Policy(16);tr=create_train_state(model,jax.random.PRNGKey(81),fcfg)
    bank=jax.tree.map(lambda p:jnp.repeat(p[None],4,0),tr.params)
    carry=initial_carry(pool49,jax.random.PRNGKey(82),fcfg,bank)
    carry=carry._replace(states=carry.states._replace(time=jnp.full(128,2040,jnp.int32)))
    original=make_collect(model,fcfg)(tr.params,bank,carry,pool49)
    optimized=make_collect_optimized(model,fcfg)(tr.params,bank,carry,pool49)
    if not identical(original[0],optimized[0]):raise AssertionError('Original/optimized carry differs')
    for k in ('action','mask','reward','terminated','truncated','invalid'):
        if not identical(original[1][k],optimized[1][k]):raise AssertionError('Original rollout differs:'+k)
    for k in ('value','old_logp','next_value'):
        a,b=original[1][k],optimized[1][k]
        if k=='next_value':a=jnp.where(original[1]['terminated'],0.,a)
        np.testing.assert_allclose(a,b,atol=5e-5,rtol=5e-5)
    # Current BF16/Pallas path: train through a2048 boundary, maintain history.
    model=Policy(16,compute_dtype='bfloat16');tr=create_train_state(model,jax.random.PRNGKey(91),cfg)
    bank=jax.tree.map(lambda p:jnp.repeat(p[None],4,0),tr.params)
    carry=initial_carry(pool49,jax.random.PRNGKey(92),cfg,bank)
    carry=carry._replace(states=carry.states._replace(time=jnp.full(128,2040,jnp.int32)))
    fn=make_iteration(model,cfg,'optimized');logs=[]
    for u in range(1,9):
        tr,carry,m=fn(tr,bank,carry,pool49);jax.block_until_ready(m)
        host=np.asarray(m)
        if not np.isfinite(host).all() or host[9]:raise AssertionError('Nonfinite/illegal full-size update')
        logs.append(host.tolist())
        if u%4==0:bank=jax.tree.map(lambda b,p:b.at[(u//4-1)%4].set(p),bank,tr.params)
    checkpoint=out/'latest.msgpack'
    meta=save_checkpoint(checkpoint,tr,bank,carry,pool49,cfg,binding,8,Ledger(start,1800),2)
    _,restored=read_checkpoint(checkpoint,tr,cfg,binding)
    if not identical((tr,bank,carry,pool49),restored):raise AssertionError('Checkpoint arrays differ')
    expected=fn(tr,bank,carry,pool49);actual=fn(*restored)
    jax.block_until_ready((expected,actual))
    if not identical(expected[1],actual[1]):raise AssertionError('Resumed simulator/RNG/history differs')
    err=0.
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
        if jnp.issubdtype(a.dtype,jnp.inexact):
            err=max(err,float(jnp.max(jnp.abs(a-b))));np.testing.assert_allclose(a,b,atol=5e-5,rtol=5e-5)
    # Same BF16 actor/params and states with slice vs the tiled Pallas backend.
    a=make_collect_optimized(model,dict(cfg,historical_convolution='slice'))(tr.params,bank,carry,pool49)
    b=make_collect_optimized(model,cfg)(tr.params,bank,carry,pool49)
    kernel_exact=identical(a,b)
    if not kernel_exact:raise AssertionError('Tiled implicit inference changes the recorded BF16 trajectory')
    result=dict(rules=RULESET,config=cfg,source=binding,logs=logs,
        checkpoint_arrays_exact=True,resumed_simulator_rng_history_exact=True,
        resumed_update_max_abs_error=err,implicit_slice_trajectory_exact=kernel_exact,
        checkpoint_bytes=checkpoint.stat().st_size,elapsed_seconds=time.monotonic()-start)
    write_json(out/'validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source','logs','config','rules')}),flush=True)
