"""Large-batch fixed-resource and resource-scaling CUDA experiments."""
from pathlib import Path
import copy,json,time,statistics,resource,hashlib
import numpy as np
import torch
from . import cuda_suite as suite
from .cli import initialize,evaluation,checkpoint
from .env import vector_for,Vector
from .rl import Timings,Population,inference_for,rollout,update
from .provenance import manifest,write_json,Ledger,canonical

def cfg_for(cpus,envs,mode):
    c=suite.config('cuda:0',cpus-1,envs)
    c.update(optimization=mode,iterations=100000,allocation_seconds=180,development_evaluation=None)
    c['hardware'].update(attached_cpu_count=cpus,attached_ram_gb=8 if cpus==4 else 32,machine=f'Modal-L4-{cpus}physical')
    return c

def timed_learning(out,cfg,seed,seconds=180):
    out=Path(out);c=copy.deepcopy(cfg);c.update(seed=seed,training_seed_start=1000000+seed*10000)
    started=time.perf_counter();ledger=Ledger(started);mh=manifest(out,c,'wide-timed-learning',{'seed':seed,'training_limit_s':seconds,'final_eval_reserve_s':15})
    m=initialize(c);initial_weights={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    opt=torch.optim.Adam(m.parameters(),lr=c['learning_rate'],eps=1e-5)
    population=Population(c,m,'cuda:0');rng=np.random.default_rng(seed);t=Timings();inf=inference_for(c,t)
    practice=json.loads((suite.ROOT/'integrations/pokemon/recipes/practice.json').read_text())
    def evaluate():
        with Vector(2,16,packed=True,binary=True,shared=True) as ev:
            return evaluation(ev,m,practice,dict(c,optimization='shared'),t)
    first=evaluate();reports=[];next_index=0;cutoff=started+seconds-15
    with vector_for(c) as v:
        iteration=0
        while time.perf_counter()<cutoff:
            seeds=list(range(c['training_seed_start']+next_index,c['training_seed_start']+next_index+v.size))
            assignments=population.assignments(v.size,next_index,rng,m)
            samples,r=rollout(v,m,assignments,seeds,c,rng,inf,t,deadline=cutoff)
            stats=update(m,opt,samples,c,'cuda:0',t) if time.perf_counter()<cutoff else {'samples':0,'skipped':'allocation_expired'}
            next_index+=v.size;iteration+=1
            if iteration%c['snapshot_every']==0:population.add(m)
            r.update(iteration=iteration,ppo=stats,seconds=ledger.elapsed);reports.append(r)
            with (out/'metrics.jsonl').open('a') as f:f.write(canonical(r)+'\n')
            checkpoint(out/'checkpoint.pt',m,opt,c,iteration,rng,population,ledger,mh)
            print(f'SCALE cpus={c["hardware"]["attached_cpu_count"]} envs={v.size} mode={c["optimization"]} seed={seed} update={iteration} samples={sum(x["ppo"]["samples"] for x in reports)} allocated={ledger.elapsed:.2f}',flush=True)
        metrics=v.worker_metrics
    final=evaluate()
    trained=sum(r['ppo']['samples'] for r in reports)
    result=dict(config=c,seed=seed,initial_evaluation=first,final_evaluation=final,updates=len(reports),
        policy_samples=sum(r['policy_samples'] for r in reports),trained_samples=trained,
        natural_terminals=sum(r['natural_terminals'] for r in reports),truncations=sum(r['truncations'] for r in reports),
        late_policy_samples_turn50=sum(r['late_policy_samples_turn50'] for r in reports),
        parameter_change_l2=sum(float((v.detach().cpu()-initial_weights[k]).square().sum()) for k,v in m.state_dict().items())**.5,
        faults=0,timings=t.result(),worker_metrics=metrics,next_battle_index=next_index,
        trained_samples_per_allocated_hour=trained/ledger.elapsed*3600,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),rss_max_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**ledger.result())
    write_json(out/'results.json',result);write_json(out/'accounting.json',ledger.result());suite.PERSIST();return result

def run(out,cpus=4,fast=False,frontier=False):
    out=Path(out);c=cfg_for(cpus,8 if cpus==4 else 32,'shared')
    manifest(out,c,'massive-cuda',{'resources':cpus,'private_panel_used':False,'Polytopia_reference':'read-only CUDA per-game engine and device ring; no Polytopia jobs run'})
    result=dict(environment=suite.environment(),cpuinfo=Path('/proc/cpuinfo').read_text().split('\n\n')[0],profiles={},learning=[])
    plans=[(8,'packed',False,256),(8,'shared',False,256),(64,'packed',False,256),(64,'shared',False,256)] if cpus==4 else [(8,'shared',False,256),(32,'shared',False,256)]
    if fast:plans=[(8,'shared',False,256),(8,'shared',True,256),(32,'shared',True,256),(32,'shared',True,4096)]
    if frontier:plans=[(8,'shared',False,256),(8,'shared',True,256),(32,'shared',True,4096),(128,'shared',True,4096)]
    for envs,mode,lazy,mini in plans:
        cfg=cfg_for(cpus,envs,mode);cfg.update(lazy_observations=lazy,minibatch=mini,batched_scripted=frontier and lazy);cfg['profile_opponents']='mixed-scripted'
        label=f'e{envs}-{mode}'+(f'-lazy{lazy}-m{mini}' if fast or frontier else '')
        rows=suite.profile_case(out/f'profile-{label}',cfg,variant=mode,repeats=4,checkpoint=Path('/workspace/input-policy.pt'))
        result['profiles'][label]={'rows':rows,'median_seconds':statistics.median(r['seconds'] for r in rows[1:])}
        write_json(out/'partial.json',result);suite.PERSIST()
    if cpus==4:
        for envs in (8,64):
            packed=result['profiles'][f'e{envs}-packed']['rows'][0]['traces']
            assert all(r['traces']==packed for mode in ('packed','shared') for r in result['profiles'][f'e{envs}-{mode}']['rows'])
    for seed in (1729,2718,3141):
        plans=[(8,'packed'),(64,'shared')] if cpus==4 else [(32,'shared')]
        for envs,mode in plans:
            cfg=cfg_for(cpus,envs,mode)
            if fast or frontier:cfg.update(lazy_observations=True,minibatch=4096,batched_scripted=frontier)
            result['learning'].append(timed_learning(out/f'learn-{seed}-e{envs}-{mode}',cfg,seed))
            write_json(out/'partial.json',result);suite.PERSIST()
    write_json(out/'results.json',result);suite.PERSIST();return result


def cli_validation(out,cpus=32):
    import subprocess,sys
    out=Path(out);cfg=cfg_for(cpus,32,'shared');cfg.update(lazy_observations=True,batched_scripted=True,minibatch=4096,iterations=1,allocation_seconds=600)
    manifest(out,cfg,'massive-public-cli-resume',{'evaluation':'practice only'})
    path=out/'config.json';write_json(path,cfg);commands=[]
    def command(args,label):
        start=time.perf_counter()
        with (out/(label+'.log')).open('w') as log:
            r=subprocess.run([sys.executable,'-m','integrations.pokemon.cli',*args],cwd=suite.ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=360)
        if r.returncode:raise RuntimeError(f'CLI {label} failed {r.returncode}')
        commands.append(dict(label=label,seconds=time.perf_counter()-start))
    base=out/'base';resume=out/'resume';full=out/'continuous'
    command(['train','--config',str(path),'--output',str(base)],'base')
    command(['train','--config',str(path),'--iterations','2','--resume',str(base/'checkpoint.pt'),'--output',str(resume)],'resume')
    command(['train','--config',str(path),'--iterations','2','--output',str(full)],'continuous')
    a=torch.load(resume/'checkpoint.pt',map_location='cpu',weights_only=False);b=torch.load(full/'checkpoint.pt',map_location='cpu',weights_only=False)
    error=max(float((a['model'][k]-v).abs().max()) for k,v in b['model'].items())
    identity=error==0 and torch.equal(a['torch_rng'],b['torch_rng']) and all(torch.equal(x,y) for x,y in zip(a['cuda_rng'],b['cuda_rng'])) and a['numpy_rng']==b['numpy_rng']
    if not identity:raise AssertionError(f'Massive CLI resume identity failed {error}')
    command(['evaluate','--config',str(path),'--checkpoint',str(full/'checkpoint.pt'),'--output',str(out/'practice')],'practice')
    result=dict(exact_cuda_model_and_rng_resume=True,max_parameter_error=error,commands=commands,
        evaluation=json.loads((out/'practice/results.json').read_text()),config=cfg)
    write_json(out/'results.json',result);suite.PERSIST();return result
