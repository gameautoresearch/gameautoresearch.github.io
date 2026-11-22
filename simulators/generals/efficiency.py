"""Measured efficiency variants preserving the frozen rules and visible interface.

The optimized collector reuses V(s[t+1]) from the following tick, with explicit
cached final observations for truncations. It performs two batched critic calls
per rollout instead of one per tick. A horizon <= cap guarantees at most one cap
per environment within the rollout. Terminations still bootstrap zero.
"""
import time
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .environment import (observations,encode,action_mask,decode_action,transition,reset_done,potential,scripted_logits,RULESET,build_pool)
from .learning import Carry,sample,logprob,make_collect,make_update,create_train_state,initial_carry,advantages
from .model import Policy
from .runtime import HERE,write_json,sources,save_checkpoint,read_checkpoint,Ledger


def make_collect_optimized(model,cfg):
    from .packed import rollout_inputs
    from .model import historical_policy
    from .resetting import refresh,opponent_weights
    historical_model=historical_policy(model,cfg)
    b,t=cfg['num_envs'],cfg['rollout_steps']
    if t>RULESET['truncation']:raise ValueError('Optimized rollout horizon must not exceed the episode cap')
    rows=jnp.arange(b);gamma=cfg['gamma'];shaping=cfg['shaping_scale']
    mix=jnp.broadcast_to(jnp.log(jnp.asarray(cfg['opponent_mix'])),(b,4))
    draw=lambda key,logits:sample(key,logits,cfg.get('sampler','gumbel'))
    @jax.jit
    def collect(params,bank,carry,pool):
        def tick(scan_c,_):
            c,capped,cached_opponent=scan_c
            key,ka,kh,ks,kr,ko,ki=jax.random.split(c.key,7)
            obs=observations(c.states);x,masks=encode(obs),action_mask(obs)
            opp=1-c.seats
            if cfg.get('compact_opponent_inference',False):
                # One common learner batch plus one per-episode opponent batch,
                # rather than 2B current and B historical forwards. Never index
                # the mutable bank for a continuing historical episode.
                ll,lv=model.apply({'params':params},x[rows,c.seats],masks[rows,c.seats])
                hl,hv=jax.vmap(lambda p,a,m:historical_model.apply({'params':p},a[None],m[None]))(cached_opponent,x[rows,opp],masks[rows,opp])
                logits=jnp.zeros((b,2,masks.shape[-1])).at[rows,c.seats].set(ll).at[rows,opp].set(hl[:,0])
                values=jnp.zeros((b,2)).at[rows,c.seats].set(lv).at[rows,opp].set(hv[:,0])
            else:
                logits,values=model.apply({'params':params},x.reshape((-1,)+x.shape[-3:]),masks.reshape((-1,masks.shape[-1])))
                logits,values=logits.reshape((b,2,masks.shape[-1])),values.reshape((b,2))
            ids=draw(ka,logits);opp=1-c.seats
            if cfg['opponent_mix'][1]>0:
                if not cfg.get('compact_opponent_inference',False):
                    hl,_=jax.vmap(lambda p,a,m:historical_model.apply({'params':p},a[None],m[None]))(c.history_params,x[rows,opp],masks[rows,opp])
                hist=draw(kh,hl[:,0])
            else:hist=ids[rows,opp]
            opp_obs=jax.tree.map(lambda a:a[rows,opp],obs)
            script=draw(ks,scripted_logits(opp_obs,jnp.maximum(c.opponents-2,0)))
            oi=jnp.where(c.opponents==1,hist,jnp.where(c.opponents>=2,script,ids[rows,opp]))
            ids=ids.at[rows,opp].set(oi)
            boundary=transition(c.states,decode_action(ids))
            done=boundary.terminated|boundary.truncated
            # Only PUBLIC own land is used for shaping/diagnostics. Hidden state
            # does not enter actor/critic; cached final states pass through observe.
            own_land=boundary.final_state.ownership.sum((2,3))[rows,c.seats]
            reward=boundary.reward[rows,c.seats]+shaping*(gamma*(own_land.astype(jnp.float32)/np.prod(x.shape[-3:-1]))*(~boundary.terminated)-potential(obs)[rows,c.seats])
            capped=jax.tree.map(lambda n,o:jnp.where(boundary.truncated.reshape((b,)+(1,)*(n.ndim-1)),n,o),boundary.final_state,capped)
            # Remember the pre-reset seat for the final capped observation.
            kinds=sample(ko,mix);hist_ids=jax.random.randint(ki,(b,),0,cfg['history_size'])
            traced_map_ids=jax.random.randint(kr,(b,),0,pool.time.shape[0]) if cfg.get('trace_actions',False) else None
            def resets(_):
                map_ids=traced_map_ids if cfg.get('trace_actions',False) else jax.random.randint(kr,(b,),0,pool.time.shape[0])
                return refresh(boundary.final_state,done,pool,map_ids,
                               bank,hist_ids,c.history_params,params,kinds,cached_opponent,
                               cfg.get('sparse_reset_capacity',0))
            if cfg.get('guard_resets',False):
                new_states,hp,new_opponent=jax.lax.cond(done.any(),resets,lambda _:(boundary.final_state,c.history_params,cached_opponent),None)
            else:new_states,hp,new_opponent=resets(None)
            nc=Carry(new_states,key,jnp.where(done,1-c.seats,c.seats),jnp.where(done,kinds,c.opponents),
                     jnp.where(done,hist_ids,c.history_ids),hp,c.episodes+done)
            chosen=ids[rows,c.seats]
            out=dict(action=chosen,
                     old_logp=logprob(logits[rows,c.seats],chosen),value=values[rows,c.seats],reward=reward,
                     terminated=boundary.terminated,truncated=boundary.truncated,
                     wins=boundary.reward[rows,c.seats]>0,losses=boundary.reward[rows,c.seats]<0,
                     invalid=~jnp.take_along_axis(masks,ids[...,None],-1)[...,0],own_land=own_land,
                     learner_seat=c.seats)
            out.update(rollout_inputs(obs,x,masks,rows,c.seats,cfg.get('packed_rollout',False)))
            if cfg.get('trace_actions',False):
                out.update(joint_action=ids,reset_map_ids=traced_map_ids)
            return (nc,capped,new_opponent),out
        cached=opponent_weights(params,carry.history_params,carry.opponents) if cfg.get('compact_opponent_inference',False) else None
        (final,capped,_),data=jax.lax.scan(tick,(carry,carry.states,cached),None,length=t,unroll=cfg.get('rollout_unroll',1))
        # Values of reset states are ignored on truncations/terminations. Every
        # non-ending transition keeps its learner seat, so the shifted value is V(s').
        obs=observations(final.states)
        _,tail=model.apply({'params':params},encode(obs)[rows,final.seats],action_mask(obs)[rows,final.seats])
        # One cap maximum per lane; select the seat at that cap, not after reset.
        cap_seat=jnp.sum(jnp.where(data['truncated'],data['learner_seat'],0),axis=0)
        def cap_critic(_):
            cap_obs=observations(capped)
            _,value=model.apply({'params':params},encode(cap_obs)[rows,cap_seat],action_mask(cap_obs)[rows,cap_seat])
            return value
        if cfg.get('guard_resets',False):
            cap_value=jax.lax.cond(data['truncated'].any(),cap_critic,lambda _:jnp.zeros((b,),jnp.float32),None)
        else:cap_value=cap_critic(None)
        shifted=jnp.concatenate([data['value'][1:],tail[None]],axis=0)
        data['next_value']=jnp.where(data['truncated'],cap_value[None],shifted)
        data['next_value']=jnp.where(data['terminated'],0.,data['next_value'])
        del data['learner_seat']
        return final,data
    return collect


def metrics_summary(data,metrics):
    return jnp.concatenate([metrics,jnp.array([data['wins'].sum(),data['losses'].sum(),data['truncated'].sum(),
        data['invalid'].sum(),data['own_land'].mean(),data['reward'].mean()],jnp.float32)])


def make_iteration(model,cfg,variant):
    collect=make_collect_optimized(model,cfg) if variant=='optimized' else make_collect(model,cfg)
    update=make_update(model,cfg)
    if variant=='optimized':
        @jax.jit
        def iteration(train,bank,c,pool):
            c,data=collect(train.params,bank,c,pool)
            key,uk=jax.random.split(c.key);c=c._replace(key=key)
            train,metrics=update(train,data,uk)
            return train,c,metrics_summary(data,metrics)
    else:
        def iteration(train,bank,c,pool):
            c,data=collect(train.params,bank,c,pool);jax.block_until_ready((c,data))
            key,uk=jax.random.split(c.key);c=c._replace(key=key)
            train,metrics=update(train,data,uk);jax.block_until_ready(train)
            return train,c,metrics_summary(data,metrics)
    return iteration


def parity(model,cfg,train,bank,c,pool):
    from .cuda_experiment import timed
    (ca,a),cold_a=timed(make_collect(model,cfg),train.params,bank,c,pool)
    (cb,b),cold_b=timed(make_collect_optimized(model,cfg),train.params,bank,c,pool)
    exact=[];errors={}
    for name in ('action','mask','terminated','truncated','wins','losses','invalid','own_land','reward'):
        if not np.array_equal(np.asarray(a[name]),np.asarray(b[name])):raise AssertionError(f'Trajectory parity failed: {name}')
        exact.append(name)
    if not all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(jax.tree.leaves(ca),jax.tree.leaves(cb))):
        raise AssertionError('Final board/RNG/opponent cache differs')
    for name in ('x','value','old_logp','next_value'):
        aa=np.asarray(a[name]);bb=np.asarray(b[name])
        if name=='next_value':aa=np.where(np.asarray(a['terminated']),0.,aa)
        errors[name]=float(np.max(np.abs(aa-bb)))
        np.testing.assert_allclose(aa,bb,rtol=3e-5,atol=3e-5)
    aa=advantages(a,cfg['gamma'],cfg['gae_lambda']);bb=advantages(b,cfg['gamma'],cfg['gae_lambda'])
    adv_err=max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(aa,bb))
    for x,y in zip(aa,bb):np.testing.assert_allclose(x,y,rtol=5e-5,atol=5e-5)
    return dict(exact_fields=exact,exact_final_state_rng_opponent_cache=True,max_abs_errors=errors,
                gae_max_abs_error=adv_err,baseline_compile_seconds=cold_a,optimized_compile_seconds=cold_b,
                captures=int(a['terminated'].sum()),caps=int(a['truncated'].sum()),invalid=int(a['invalid'].sum()))


def run_comparison(args,out):
    from .cuda_experiment import cfg_for,trained_state,timed
    results=[]
    for b in map(int,args.batches.split(',')):
        cfg=cfg_for(b);model=Policy(cfg['network_width'])
        train,bank,c,pool,provenance=trained_state(cfg,model,args.snapshot)
        baseline=make_collect(model,cfg)
        # Real trained policy actions, not hand-injected armies/shortened caps.
        for _ in range(3):c,_=baseline(train.params,bank,c,pool)
        jax.block_until_ready(c)
        row=dict(batch=b,config=cfg,starting_times=np.asarray(c.states.time).tolist(),
                 starting_army=np.asarray(c.states.armies.sum((1,2))).tolist(),parity=parity(model,cfg,train,bank,c,pool))
        fns={v:make_iteration(model,cfg,v) for v in ('baseline','optimized')}
        row['cold_seconds']={};row['trials']={v:[] for v in fns}
        for v,fn in fns.items():_,row['cold_seconds'][v]=timed(fn,train,bank,c,pool)
        # Alternate order; every pair gets the identical original trained weights,
        # game states/RNG and workload, never a faster shortened/faulted game.
        for repeat in range(7):
            for v in (('baseline','optimized') if repeat%2==0 else ('optimized','baseline')):
                (tr,nc,m),elapsed=timed(fns[v],train,bank,c,pool)
                host=np.asarray(m)
                if host[9]!=0 or not np.isfinite(host).all():raise AssertionError('Fault/illegal/nonfinite trial')
                row['trials'][v].append(dict(seconds=elapsed,wins=int(host[6]),losses=int(host[7]),caps=int(host[8]),invalid=int(host[9]),loss=float(host[0])))
        med={v:float(np.median([r['seconds'] for r in rs])) for v,rs in row['trials'].items()}
        n=b*cfg['rollout_steps'];row.update(median_seconds=med,speedup=med['baseline']/med['optimized'],
            samples_per_allocated_gpu_hour_warm={v:n/s*3600 for v,s in med.items()},
            legal_player_decisions_per_second={v:2*n/s for v,s in med.items()},updates_per_second={v:1/s for v,s in med.items()})
        results.append(row);write_json(out/'comparison.json',dict(results=results,source=sources(),provenance=provenance))
        print(json.dumps(dict(batch=b,median_seconds=med,speedup=row['speedup'],parity=row['parity'])),flush=True)
        if b==int(args.batches.split(',')[-1]) and jax.default_backend()=='gpu':
            try:
                with jax.profiler.trace(str(out/'trace'),create_perfetto_trace=True):
                    for v in ('baseline','optimized'):
                        with jax.profiler.TraceAnnotation(v):timed(fns[v],train,bank,c,pool)
                write_json(out/'trace-status.json',dict(recorded=True))
            except Exception as exc:write_json(out/'trace-status.json',dict(recorded=False,error=str(exc)))


def run_learning(args,out):
    from .cuda_experiment import cfg_for,timed
    from .evaluation import evaluate
    from .runtime import HERE
    from .autoresearch.contract import seed_manifest
    seed_data = seed_manifest()
    b=int(args.batches.split(',')[0]);results=[]
    _,legacy_data=read_checkpoint(args.snapshot)
    generated=build_pool(seed_data['train'])
    pool_parity={name:bool(np.array_equal(np.asarray(getattr(generated,name)),np.asarray(value)))
                 for name,value in legacy_data['pool'].items()}
    write_json(out/'cpu-cuda-map-parity.json',dict(exact_fields=pool_parity,all_exact=all(pool_parity.values())))
    for seed in map(int,args.seeds.split(',')):
        cfg=cfg_for(b,seed);cfg['max_updates']=args.steps
        model=Policy(cfg['network_width'])
        seed_start=time.monotonic()
        key,ik=jax.random.split(jax.random.PRNGKey(seed))
        initial=create_train_state(model,ik,cfg)
        pool=build_pool(seed_data['train'])
        bank0=jax.tree.map(lambda p:jnp.repeat(p[None],cfg['history_size'],0),initial.params)
        c0=initial_carry(pool,key,cfg,bank0)
        before=evaluate(model,initial.params,seed_data['practice'][:4],('random','expansion'))
        shared_setup_seconds=time.monotonic()-seed_start
        for variant in ('baseline','optimized'):
            folder=out/f'seed-{seed}-{variant}';folder.mkdir(exist_ok=True)
            train,c,bank=initial,c0,bank0
            fn=make_iteration(model,cfg,variant);cold_start=time.monotonic()
            # Compile without consuming the actual training trajectory.
            _,cold=timed(fn,train,bank,c,pool)
            logs=[];start=time.monotonic();cursor=0
            for u in range(1,args.steps+1):
                (train,c,m),elapsed=timed(fn,train,bank,c,pool)
                host=np.asarray(m)
                if not np.isfinite(host).all() or host[9]:raise AssertionError('Learning fault/illegal/nonfinite')
                if u%cfg['history_interval']==0:
                    bank=jax.tree.map(lambda bank,p:bank.at[cursor%cfg['history_size']].set(p),bank,train.params);cursor+=1
                row=dict(update=u,seconds=elapsed,metrics=host.tolist())
                logs.append(row)
                with (folder/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if u%32==0:print(json.dumps(dict(seed=seed,variant=variant,update=u,completed=sum(int(r['metrics'][6]+r['metrics'][7]+r['metrics'][8]) for r in logs))),flush=True)
            training_wall=time.monotonic()-start
            after=evaluate(model,train.params,seed_data['practice'][:4],('random','expansion'))
            heldout=evaluate(model,train.params,seed_data['heldout'][:8],('random','expansion','hunter'))
            ledger=Ledger(cold_start,800)
            save_checkpoint(folder/'final.msgpack',train,bank,c,pool,cfg,sources(),args.steps,ledger,cursor)
            resume_start=time.monotonic()
            _,(rt,rb,rc,rp)=read_checkpoint(folder/'final.msgpack',train,cfg,sources())
            restored_exact=all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in
                zip(jax.tree.leaves((train,bank,c,pool)),jax.tree.leaves((rt,rb,rc,rp))))
            if not restored_exact:raise AssertionError('Checkpoint changed saved arrays')
            expected,_=timed(fn,train,bank,c,pool)
            actual,_=timed(fn,rt,rb,rc,rp)
            resume_exact=all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)))
            # cuDNN backward reductions are not bitwise deterministic by default.
            # Keep exact simulator/RNG/opponent checks; qualify floating updates
            # numerically rather than mistaking reduction noise for lost state.
            state_exact=all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in
                zip(jax.tree.leaves(expected[1]),jax.tree.leaves(actual[1])))
            if not state_exact:raise AssertionError('Resumed simulation/RNG differs')
            max_error=0.
            for x,y in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
                xx,yy=np.asarray(x),np.asarray(y)
                if np.issubdtype(xx.dtype,np.inexact):
                    max_error=max(max_error,float(np.max(np.abs(xx-yy))))
                    np.testing.assert_allclose(xx,yy,rtol=5e-5,atol=5e-5)
                elif not np.array_equal(xx,yy):raise AssertionError('Resumed integer state differs')
            resume_seconds=time.monotonic()-resume_start
            delta=float(np.sqrt(sum(float(jnp.sum((p-q)**2)) for p,q in zip(jax.tree.leaves(initial.params),jax.tree.leaves(train.params)))))
            total=time.monotonic()-cold_start
            result=dict(seed=seed,variant=variant,config=cfg,compile_seconds=cold,training_wall_seconds=training_wall,
                variant_wall_including_compile_eval_checkpoint=total,learner_samples=b*cfg['rollout_steps']*args.steps,
                completed_games=int(c.episodes.sum()),wins=sum(int(r['metrics'][6]) for r in logs),
                losses=sum(int(r['metrics'][7]) for r in logs),caps=sum(int(r['metrics'][8]) for r in logs),invalid=0,
                parameter_l2_change=delta,before=before,after=after,heldout=heldout,
                checkpoint_resume_exact=resume_exact,checkpoint_resume_verification_seconds=resume_seconds,
                checkpoint_restored_arrays_exact=restored_exact,resumed_simulator_rng_exact=state_exact,
                resumed_update_max_abs_error=max_error,
                learner_samples_per_allocated_gpu_hour_including_compile_eval=b*cfg['rollout_steps']*args.steps/(total+shared_setup_seconds/2)*3600,
                allocated_seconds_with_half_shared_setup=total+shared_setup_seconds/2,
                median_update_seconds=float(np.median([r['seconds'] for r in logs])),
                shared_seed_setup_and_initial_eval_seconds=shared_setup_seconds)
            write_json(folder/'result.json',result);results.append(result)
            write_json(out/'learning.json',dict(results=results,source=sources(),note='Same seeds/model/rules/workload. Float reduction differences may diverge stochastic learning trajectories; score comparisons are paired by training seed/map/seat.'))
    run_cli_checks(out)
    return results


def run_cli_checks(out):
    # Exercise the public command surface on this actual CUDA allocation too.
    # Imports, profile, save/resume, evaluation and replay all remain charged.
    import os,sys,subprocess
    env=dict(os.environ,GENERALS_ALLOCATION_CONFIRMED='1',
             GENERALS_ALLOCATION_START_UNIX=os.environ.get('GENERALS_ALLOCATION_START_UNIX',str(time.time())))
    folder=out/'cli';folder.mkdir(exist_ok=True)
    recipe=str(HERE/'recipes/l4-small.json')
    train_out=str(folder/'train')
    commands=[['train','--max-updates','16','--output',train_out],
              ['train','--max-updates','24','--resume',str(folder/'train/latest.msgpack'),'--output',train_out],
              ['evaluate','--checkpoint',str(folder/'train/latest.msgpack'),'--split','heldout','--output',str(folder/'evaluation')],
              ['profile','--output',str(folder/'profile')]]
    cli=[]
    for i,command in enumerate(commands):
        started=time.monotonic()
        with (folder/f'command-{i}.log').open('w') as log:
            result=subprocess.run([sys.executable,'-m','integrations.generals',command[0],'--device','gpu','--recipe',recipe]+command[1:],
                env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        cli.append(dict(command=command,seconds=time.monotonic()-started,exit_code=result.returncode))
        write_json(folder/'commands.json',cli)
        if result.returncode:raise RuntimeError(f'CUDA CLI failed: {command}')
    with (folder/'replay.log').open('w') as log:
        subprocess.run([sys.executable,'-m','integrations.generals.replay',str(folder/'evaluation/replay.npz'),
            '--output',str(folder/'replay-verification.json')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)

