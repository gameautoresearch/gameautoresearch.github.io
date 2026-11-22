"""Bounded, resumable single-GPU AverageJoe reference pilot.

Model, memory, rewards, action distribution, curriculum thresholds, PPO, schedule
arithmetic and 200k-map pool are upstream. Single-device orchestration, cached
CPU map setup, evaluation early stopping, checks and IO are project-owned.
"""
from pathlib import Path
import argparse,json,os,time,signal,hashlib,traceback
START=time.monotonic()
from .bootstrap import activate,binding,sha
activate()
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from .framework import recipe,initial,make_env,memories,make_kernels,next_stage
from .evaluation import make_evaluator
from generals.core.game import GameState


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def load_bank(path):
    with np.load(path,allow_pickle=False) as a:return GameState(**{k:jnp.asarray(a[k]) for k in GameState._fields})


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=550);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=recipe();source=binding();meta=json.loads((a.inputs/'maps.json').read_text())
    old=meta['binding']['sources'];current=source['sources']
    for field in ('averagejoe_commit','averagejoe_files','engine_files','recipe_sha256'):
        if old[field]!=current[field]:raise ValueError('Prepared input source binding changed: '+field)
    # Adding trainer/evaluator modules after CPU preparation does not change the
    # generator. Every source file already present during preparation stays bound.
    for name,value in old['framework_files'].items():
        if current['framework_files'].get(name)!=value:raise ValueError('Map preparation code drift: '+name)
    for n,h in meta['files'].items():
        if sha(a.inputs/n)!=h:raise ValueError('Prepared input checksum changed')
    if meta['config']!=cfg.to_dict():raise ValueError('Smoke must retain the actual published configuration')
    if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise RuntimeError('This qualification requires exactly one GPU')
    jax.config.update('jax_default_matmul_precision','tensorfloat32')
    write(a.output/'source.json',source);write(a.output/'config.json',cfg.to_dict())
    import importlib.metadata
    write(a.output/'runtime.json',{'packages':{n:importlib.metadata.version(n) for n in ['jax','jaxlib','equinox','optax','numpy','ruamel.yaml']},'devices':[d.device_kind for d in jax.devices()]})
    from ruamel.yaml import YAML
    with (a.output/'config.yaml').open('w') as f:YAML().dump(cfg.to_dict(),f)
    log=a.output/'progress.jsonl';events=a.output/'events.jsonl'
    def event(kind,**kw):
        row={'kind':kind,'worker_process_seconds':time.monotonic()-START,**kw}
        with events.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
    event('initializing',model='upstream HistoryTransformer',curriculum=[(s.min_generals_distance,s.max_generals_distance) for s in cfg.curriculum_stages])
    net,opt,opt_state,key=initial(cfg);ema=net
    parameters=sum(x.size for x in jax.tree.leaves(eqx.filter(net,eqx.is_array)))
    initial_probe=np.asarray(net.policy_head.weight).copy();initial_sha=hashlib.sha256(initial_probe.tobytes()).hexdigest()
    pool=load_bank(a.inputs/'stage0-pool.npz');states=load_bank(a.inputs/'stage0-start.npz');starts=load_bank(a.inputs/'stage0-eval.npz')
    m0=memories(cfg);m1=memories(cfg);key=jnp.asarray(meta['initial_key'],jnp.uint32)
    env=make_env(cfg);env.pool_size=meta['actual_pool_size'];stage=0;iteration=0;totals={'captures':0,'caps':0,'nonzero_rewards':0};last_eval=None
    rollout,update,ema_step=make_kernels(cfg,env,net);evaluate=make_evaluator(cfg)
    event('ready',parameters=parameters,pool_size=env.pool_size,all_training_shapes=49)
    stop=False
    def handle(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,handle)
    checkpoint_value=None
    def checkpoint(label):
        nonlocal checkpoint_value
        target=(net,opt_state,ema,key,states,m0,m1)
        path=a.output/(label+'.eqx');tmp=path.with_suffix('.tmp');eqx.tree_serialise_leaves(tmp,target);tmp.replace(path)
        # Strict read-back checks all tensors, optimizer counters, memory, RNG and env.
        restored=eqx.tree_deserialise_leaves(path,target)
        exact=all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(jax.tree.leaves(target),jax.tree.leaves(restored)) if eqx.is_array(x))
        if not exact:raise AssertionError('Checkpoint readback mismatch')
        checkpoint_value={'iteration':iteration,'stage':stage,'last_evaluation':last_eval,'sha256':sha(path),'binding_sha256':source['sha256'],'pool_size':env.pool_size,'pool_sha256':meta['files']['stage0-pool.npz'] if stage==0 else None,'exact_tensor_readback':exact,'restores': 'network, optimizer, EMA, PRNG, game state/reset cursor and both observation memories; stage0 map input remains required'}
        write(a.output/(label+'.json'),checkpoint_value)
        return checkpoint_value
    before,ek=evaluate(net,starts,jax.random.PRNGKey(2026110900));before={k:int(v) for k,v in jax.device_get(before).items()}
    if before['illegal_actions'] or before['finished']!=before['games']:raise AssertionError('Initial eval incomplete or illegal')
    event('baseline_evaluation',**before)
    last_eval=before;write(a.output/'initial.json',{'policy_head_sha256':initial_sha,'parameters':parameters,'evaluation':before})
    iteration_cost=0.
    # Reserve 100 seconds for final evaluation, saving, validation and return.
    train_deadline=a.seconds-100
    while not stop and time.monotonic()-START+max(iteration_cost,10)<train_deadline:
        t=time.monotonic()
        if iteration and iteration%cfg.reset_pool_every==0:
            key,pk=jax.random.split(key);pool,_=env.reset(pk);jax.block_until_ready(pool);event('pool_refresh',iteration=iteration,size=env.pool_size)
        states,data,key,m0,m1=rollout(net,states,key,m0,m1,pool)
        net,opt_state,key,metrics=update(net,opt_state,key,data,jnp.int32(iteration));ema=ema_step(ema,net)
        jax.block_until_ready(metrics);values={k:float(v) for k,v in jax.device_get(metrics).items()}
        if not all(np.isfinite(list(values.values()))) or values['invalid_actions']:raise RuntimeError('Nonfinite learning metrics or illegal action')
        iteration+=1;iteration_cost=time.monotonic()-t
        totals['captures']+=int(values['terminated']);totals['caps']+=int(values['caps']);totals['nonzero_rewards']+=int(values['nonzero_rewards'])
        row={'iteration':iteration,'stage':stage,'stage_distance':[cfg.curriculum_stages[stage].min_generals_distance,cfg.curriculum_stages[stage].max_generals_distance], 'joint_ticks':iteration*cfg.num_envs*cfg.num_steps,'player_transitions':iteration*2*cfg.num_envs*cfg.num_steps,'iteration_seconds':iteration_cost,'worker_process_seconds':time.monotonic()-START,'player_transitions_per_iteration_second':2*cfg.num_envs*cfg.num_steps/iteration_cost,**values, 'totals':dict(totals)}
        with log.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
        del data
        if iteration%cfg.eval_every==0:
            evaluation,key=evaluate(net,starts,key);last_eval={k:int(v) for k,v in jax.device_get(evaluation).items()}
            if last_eval['illegal_actions'] or last_eval['finished']!=last_eval['games']:raise RuntimeError('Curriculum eval invalid')
            event('curriculum_evaluation',iteration=iteration,stage=stage,**last_eval)
            ns=next_stage(cfg,stage,last_eval['wins']/last_eval['games'],evaluation_complete=True)
            if ns!=stage:
                stage=ns;env=make_env(cfg,stage);key,pk=jax.random.split(key);pool,_=env.reset(pk)
                key,ik=jax.random.split(key);states=jax.jit(jax.vmap(env.init_state))(jax.random.split(ik,cfg.num_envs));starts=jax.tree.map(lambda x:x[:cfg.eval_games//2],states)
                m0=memories(cfg);m1=memories(cfg);rollout,update,ema_step=make_kernels(cfg,env,net)
                event('curriculum_advanced',iteration=iteration,stage=stage,distance=[env.min_generals_distance,env.max_generals_distance])
        if iteration==1:checkpoint('after-first-update')
    checkpoint('final-state')
    eqx.tree_serialise_leaves(a.output/'final.eqx',net);eqx.tree_serialise_leaves(a.output/'final-ema.eqx',ema)
    after,key=evaluate(net,starts,jax.random.PRNGKey(2026110900));after={k:int(v) for k,v in jax.device_get(after).items()};event('final_evaluation',**after)
    if not iteration or after['illegal_actions'] or after['finished']!=after['games']:raise RuntimeError('No trained, valid completed policy assessment')
    change=float(np.linalg.norm(np.asarray(net.policy_head.weight)-initial_probe))
    result={'status':'complete','reference_strength':'unqualified ten-minute smoke','parameters':parameters,'iterations':iteration,'stage':stage,'total_player_transitions':iteration*2*cfg.num_envs*cfg.num_steps,'total_joint_ticks':iteration*cfg.num_envs*cfg.num_steps,'initial_policy_head_sha256':initial_sha,'policy_head_l2_change':change,'baseline':before,'final':after,'outcomes':totals,'checkpoint':checkpoint_value,'worker_process_seconds':time.monotonic()-START,'initial_setup_cpu_seconds':meta['cpu_preparation_seconds'],'accounting':'GPU setup/loading/compile/simulation/PPO/evaluation/save/idle all count; CPU preprocessing separately recorded. Final provider allocation authoritative.', 'limitations':['One fresh seed, short pilot; no claim of superhuman or full-distance strength.','Source model and PPO/GAE/truncation equations retained exactly, including upstream truncation masking.','Initial public inputs cached on CPU; subsequent 20-iteration refreshes generated inside worker.']}
    write(a.output/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
