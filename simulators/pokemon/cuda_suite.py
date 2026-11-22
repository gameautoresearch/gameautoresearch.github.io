"""Small actual-model CUDA experiments; imported remotely after allocation begins."""
from __future__ import annotations
import copy,json,time,subprocess,resource,hashlib,threading,random,statistics
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from .cli import initialize,load_policy,checkpoint,evaluation,train as train_command
from .network import Policy
from .env import Vector,vector_for
from .rl import Timings,Inference,rollout,update,sync,Population,inference_for
from .provenance import manifest,write_json,Ledger,canonical,digest

ROOT=Path(__file__).resolve().parents[2]
PERSIST=lambda:None

def config(device='cuda:0',workers=2,per_worker=4):
    cfg=json.loads((ROOT/'integrations/pokemon/recipes/learning-smoke.json').read_text())
    cfg.update(iterations=80,allocation_seconds=800,max_decision_phases=512,history_size=2,snapshot_every=40,development_evaluation=None)
    cfg['hardware'].update(device=device,workers=workers,environments_per_worker=per_worker,cpu_threads=1,
                           machine='Modal-L4-4-physical-cores-8GiB',attached_ram_gb=8,attached_cpu_count=4)
    return cfg

def environment():
    return {'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
            'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip() if torch.cuda.is_available() else None}

def profile_case(out,cfg,*,variant='baseline',repeats=2,checkpoint=None):
    """Fixed complete-episode workload, including reset/IPC/learning/checkpointing."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    device=cfg['hardware']['device'];torch.set_num_threads(cfg['hardware']['cpu_threads'])
    rows=[]
    cfg=dict(cfg,optimization=variant if variant in ('baseline','packed','stream','shared') else 'packed')
    cold=time.perf_counter();model=initialize(cfg)
    if checkpoint:model=load_policy(checkpoint,cfg)[0]
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    timings=Timings();inference=Inference(device,timings)
    if variant!='baseline':
        from .efficient import EfficientInference
        inference=EfficientInference(device,timings,graph=variant=='graph',flat_copy=variant in ('stream','shared'),bulk_stack=variant=='shared')
        if variant=='hybrid':
            from .efficient import HybridInference
            inference=HybridInference(device,timings)
    cold_model_seconds=time.perf_counter()-cold
    for repeat in range(repeats):
        model.load_state_dict(initial);torch.manual_seed(cfg['seed'])
        if device!='cpu':torch.cuda.manual_seed_all(cfg['seed']);torch.cuda.reset_peak_memory_stats()
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],eps=1e-5)
        timings.seconds.clear();timings.counts.clear()
        timings.cuda_events=(device!='cpu' and repeat==0)
        # First repetition is an explicitly instrumented cold diagnostic. The
        # following repetitions measure normal production inference overhead.
        if hasattr(inference,'events'):
            inference.events=[torch.cuda.Event(enable_timing=True) for _ in range(4)] if timings.cuda_events else None
        start=time.perf_counter()
        with Vector(cfg['hardware']['workers'],cfg['hardware']['environments_per_worker'],packed=variant!='baseline',binary=variant in ('stream','shared'),shared=variant=='shared',token_cache=variant=='stream',mon_cache=cfg.get('mon_cache',False),fast_callbacks=cfg.get('fast_callbacks',False),lazy=cfg.get('lazy_observations',False)) as vector:
            assignments=[[model,model] for _ in range(vector.size)]
            if cfg.get('profile_opponents')=='mixed-scripted':
                assignments=[]
                for i in range(vector.size):
                    kind='power' if i%2==0 else 'random';pair=[kind,kind];pair[(i//2)%2]=model;assignments.append(pair)
            samples,report=rollout(vector,model,assignments,
                list(range(20000,20000+vector.size)),cfg,np.random.default_rng(cfg['seed']),inference,timings,deterministic=True)
            hashes=vector.hashes();trace=[t['sha256'] for t in hashes]
            stats=update(model,optimizer,samples,cfg,device,timings)
            sync(device)
            torch.save({'network':cfg['network'],'model':model.state_dict(),'optimizer':optimizer.state_dict()},out/f'checkpoint-{repeat}.pt')
            worker_metrics=getattr(vector,'worker_metrics',{})
        elapsed=time.perf_counter()-start
        rows.append({'repeat':repeat,'instrumented_cuda_events':timings.cuda_events,'variant':variant,'seconds':elapsed,'legal_current_policy_decisions':len(samples),
          'decisions_per_second':(len(samples)-report.get('current_policy_rejections',report['rejected_visible_choices']))/elapsed,'learner_samples_per_second':len(samples)/elapsed,
          'updates_per_second':stats.get('minibatch_updates',0)/elapsed,'completed_games':report['natural_terminals'],
          'truncations':report['truncations'],'faults':0,'rejections':report['rejected_visible_choices'],
          'traces':trace,'games':report['results'],'timings':timings.result(),'worker_metrics':worker_metrics,
          'late_hp_change_events':sum(g.get('activity',{}).get('late_damage',0)+g.get('activity',{}).get('late_heals',0) for g in report['results']),
          'late_policy_samples_turn50':report.get('late_policy_samples_turn50',0),'max_turn':max(h['turn'] for h in hashes),
          'accepted_current_policy_decisions':len(samples)-report.get('current_policy_rejections',report['rejected_visible_choices']),
          'cold_model_seconds':cold_model_seconds if repeat==0 else 0.,
          'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device!='cpu' else 0,
          'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved() if device!='cpu' else 0,
          'rss_max_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'ppo':stats})
    write_json(out/'results.json',{'rows':rows,'config':cfg,'environment':environment()});PERSIST()
    return rows

def run_probe(out):
    cfg=config();out=Path(out)
    manifest(out,cfg,'cuda-probe',{'workload':'trained policy, fixed seeds 20000..20007, max512 phases, selfplay'})
    checkpoint=ROOT/'input-policy.pt'
    rows=profile_case(out/'baseline',cfg,checkpoint=checkpoint)
    result={'environment':environment(),'rows':rows,'gpu_executed':True}
    write_json(out/'probe.json',result)
    return result

class GPUMonitor:
    def __init__(self):self.samples=[];self.stop=threading.Event();self.thread=None
    def start(self):
        def sample():
            while not self.stop.is_set():
                try:
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=3).strip()
                    self.samples.append({'unix':time.time(),'gpu_percent,memory_percent,memory_mib,power_w':raw})
                except Exception as exc:self.samples.append({'error':str(exc)})
                self.stop.wait(1)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start()
    def close(self):self.stop.set();self.thread.join(timeout=4)

def learning_case(out,cfg,variant,seed,iterations=100):
    out=Path(out);cfg=copy.deepcopy(cfg);cfg.update(seed=seed,optimization=variant,iterations=iterations,
        max_decision_phases=512,training_seed_start=1000000+seed*10000)
    start=time.perf_counter();ledger=Ledger(start)
    mh=manifest(out,cfg,'cuda-learning',{'paired_implementation':variant,'training_seed':seed})
    device=cfg['hardware']['device'];model=initialize(cfg)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],eps=1e-5)
    population=Population(cfg,model,device);rng=np.random.default_rng(seed);timings=Timings()
    inference=inference_for(cfg,timings);reports=[];traces=[]
    schedule=json.loads((ROOT/'integrations/pokemon/recipes/practice.json').read_text())
    h=cfg['hardware']
    with vector_for(cfg) as vector:
        ev=time.perf_counter();initial=evaluation(vector,model,schedule,cfg,timings);initial_seconds=time.perf_counter()-ev
        checkpoint(out/'initial.pt',model,optimizer,cfg,0,rng,population,ledger,mh)
        train_started=time.perf_counter()
        for it in range(iterations):
            seeds=list(range(cfg['training_seed_start']+it*vector.size,cfg['training_seed_start']+(it+1)*vector.size))
            assignments=population.assignments(vector.size,it*vector.size,rng,model)
            samples,report=rollout(vector,model,assignments,seeds,cfg,rng,inference,timings)
            hashes=vector.hashes();traces.append([g['sha256'] for g in hashes])
            stats=update(model,optimizer,samples,cfg,device,timings)
            if (it+1)%cfg['snapshot_every']==0:population.add(model)
            report.update(iteration=it+1,ppo=stats,late_samples=report.get('late_policy_samples_turn50',0))
            reports.append(report)
            with (out/'metrics.jsonl').open('a') as f:f.write(canonical(report)+'\n')
            if (it+1)==iterations//2 and variant in ('packed','hybrid','stream','graph','shared'):
                # Exercise CUDA RNG and optimizer resume during the paired learning run.
                checkpoint(out/'midpoint.pt',model,optimizer,cfg,it+1,rng,population,ledger,mh)
                cp=torch.load(out/'midpoint.pt',map_location=device,weights_only=False)
                restored=Policy(**cfg['network']).to(device);restored.load_state_dict(cp['model'])
                opt=torch.optim.Adam(restored.parameters(),lr=cfg['learning_rate'],eps=1e-5);opt.load_state_dict(cp['optimizer'])
                population=Population(cfg,restored,device,cp['history'])
                rng.bit_generator.state=cp['numpy_rng'];torch.set_rng_state(cp['torch_rng'].cpu())
                torch.cuda.set_rng_state_all([x.cpu() for x in cp['cuda_rng']]);random.setstate(cp['python_rng'])
                model,optimizer=restored,opt;inference=inference_for(cfg,timings);PERSIST()
            if (it+1)%20==0:print(f'LEARN {variant} seed={seed} iteration={it+1} allocated={ledger.elapsed:.2f}s',flush=True)
        train_seconds=time.perf_counter()-train_started
        ev=time.perf_counter();final=evaluation(vector,model,schedule,cfg,timings);final_seconds=time.perf_counter()-ev
        checkpoint(out/'final.pt',model,optimizer,cfg,iterations,rng,population,ledger,mh)
        worker_metrics=vector.worker_metrics
    samples=sum(r['policy_samples'] for r in reports)
    updates=sum(r['ppo']['minibatch_updates'] for r in reports)
    ready=sum(r.get('total_ready_player_decisions',0) for r in reports)
    rejections=sum(r['rejected_visible_choices'] for r in reports)
    result={'variant':variant,'seed':seed,'iterations':iterations,'initial_evaluation':initial,'final_evaluation':final,
      'train_seconds':train_seconds,'initial_eval_seconds':initial_seconds,'final_eval_seconds':final_seconds,
      'policy_samples':samples,'accepted_all_player_decisions':ready-rejections,'samples_per_train_second':samples/train_seconds,
      'samples_per_allocated_gpu_hour':samples/ledger.elapsed*3600,'updates':updates,'updates_per_second':updates/train_seconds,
      'completed_games':sum(r['natural_terminals'] for r in reports),'truncations':sum(r['truncations'] for r in reports),
      'late_policy_samples_turn50':sum(r['late_samples'] for r in reports),'rejections':rejections,'faults':0,
      'timings':timings.result(),'worker_metrics':worker_metrics,'training_trajectory_sha256':hashlib.sha256(canonical(traces).encode()).hexdigest(),
      'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
      'manifest_sha256':mh,**ledger.result()}
    write_json(out/'results.json',result)
    write_json(out/'accounting.json',ledger.result());PERSIST()
    return result

def numerical_parity(cfg):
    from .network import batch
    from .efficient import EfficientInference
    model=initialize(cfg);timings=Timings()
    with Vector(1,8) as v:
        states=v.reset(list(range(77,85)),512)
    obs=[g['observations'][seat] for g in states for seat in (0,1)]
    baseline=Inference(cfg['hardware']['device'],timings);fast=EfficientInference(cfg['hardware']['device'],timings)
    rng=torch.cuda.get_rng_state()
    a=baseline(model,obs,True);torch.cuda.set_rng_state(rng)
    b=fast(model,obs,True)
    max_logp=float(np.max(np.abs(a[1]-b[1])));max_value=float(np.max(np.abs(a[2]-b[2])))
    if not np.array_equal(a[0],b[0]) or max(max_logp,max_value)>1e-6:raise AssertionError('Packed CUDA inference changed policy')
    graph=EfficientInference(cfg['hardware']['device'],timings,graph=True)
    c=graph(model,obs,True)
    if not np.array_equal(a[0],c[0]):raise AssertionError('Graph argmax changed')
    return {'actions_equal':True,'max_logp_error':max_logp,'max_value_error':max_value,'graph_actions_equal':True}

def run_matrix(out):
    out=Path(out);cfg=config(workers=3,per_worker=8);monitor=GPUMonitor();monitor.start()
    manifest(out,cfg,'cuda-matrix',{'profile_seeds':'20000..20023','training_seeds':[1729,2718],
                                    'profile':'same trained checkpoint and full episode bounds per variant'})
    result={'environment':environment(),'numerical_parity':numerical_parity(cfg),'profiles':{},'learning':[], 'learning_parity':[]}
    try:
        checkpoint_path=ROOT/'input-policy.pt'
        for device,workers,per_worker in [('cpu',2,4),('cuda:0',2,4),('cuda:0',3,8)]:
            c=config(device,workers,per_worker);key=f'{device.replace(":","-")}-w{workers}-e{per_worker}'
            group={}
            for variant in (['baseline','packed'] if device=='cpu' else ['baseline','packed','graph']):
                print(f'PROFILE {key} {variant}',flush=True)
                group[variant]=profile_case(out/key/variant,c,variant=variant,repeats=3,checkpoint=checkpoint_path)
            reference=group['baseline'][0]['traces']
            parity={k:all(row['traces']==reference for row in rows) for k,rows in group.items()}
            if not parity['baseline'] or not parity['packed']:raise AssertionError(f'Packed trajectory parity failed {key}')
            result['profiles'][key]={'rows':group,'same_trajectory_by_variant':parity,'same_trajectory':all(parity.values()),'median_seconds':{k:statistics.median(r['seconds'] for r in rows) for k,rows in group.items()}}
            write_json(out/'partial.json',result)
        for seed in (1729,2718):
            pair=[]
            for variant in ('baseline','packed'):
                print(f'TRAIN {variant} seed={seed}',flush=True)
                run=learning_case(out/f'learn-{seed}-{variant}',cfg,variant,seed,iterations=100)
                pair.append(run);result['learning'].append(run);write_json(out/'partial.json',result)
            a=torch.load(out/f'learn-{seed}-baseline/final.pt',map_location='cpu',weights_only=False)
            b=torch.load(out/f'learn-{seed}-packed/final.pt',map_location='cpu',weights_only=False)
            delta=max(float((a['model'][k]-v).abs().max()) for k,v in b['model'].items())
            parity={'seed':seed,'max_parameter_error':delta,'bitwise_model_equal':delta==0,
                    'training_traces_equal':pair[0]['training_trajectory_sha256']==pair[1]['training_trajectory_sha256'],
                    'final_evaluation_scores':[r['final_evaluation']['score'] for r in pair]}
            if not parity['training_traces_equal'] or delta>1e-6:raise AssertionError(f'Learning/resume parity failed: {parity}')
            result['learning_parity'].append(parity)
        result['gpu_executed']=True;write_json(out/'results.json',result)
    finally:
        monitor.close();write_json(out/'gpu-utilization.json',monitor.samples)
    return result

def run_refine(out):
    out=Path(out);cfg=config(workers=3,per_worker=8);monitor=GPUMonitor();monitor.start()
    manifest(out,cfg,'cuda-refinement',{'changes':['bulk typed-array wire','bucketed deterministic CUDA graphs','CPU actors with GPU learner'],'learning_seeds':[1729,2718]})
    result={'environment':environment(),'profiles':{},'learning':[],'cli_cuda_resume':None}
    try:
        result['learner_numeric_parity']=gpu_update_parity(cfg)
        for device,workers,per_worker in [('cpu',2,4),('cuda:0',3,8)]:
            c=config(device,workers,per_worker);c['profile_opponents']='mixed-scripted';key=f'{device.replace(":","-")}-w{workers}-e{per_worker}';groups={}
            for variant in (['baseline','packed'] if device=='cpu' else ['baseline','packed','graph','hybrid']):
                print(f'REFINE PROFILE {key} {variant}',flush=True)
                groups[variant]=profile_case(out/key/variant,c,variant=variant,repeats=4,checkpoint=ROOT/'input-policy.pt')
            reference=groups['baseline'][0]['traces']
            parity={k:all(r['traces']==reference for r in rows) for k,rows in groups.items()}
            if not all(parity[k] for k in ('baseline','packed')):raise AssertionError('Exact CUDA/CPU packed trajectory parity changed')
            result['profiles'][key]={'rows':groups,'same_trajectory_by_variant':parity,
              'median_seconds':{k:statistics.median(r['seconds'] for r in rows if not r.get('instrumented_cuda_events')) for k,rows in groups.items()}}
            write_json(out/'partial.json',result)
        for seed in (1729,2718):
            for variant in ('packed','hybrid','stream','graph','shared'):
                print(f'REFINE TRAIN {variant} {seed}',flush=True)
                case_started=time.perf_counter()
                case=learning_case(out/f'learn-{seed}-{variant}',cfg,variant,seed,iterations=60)
                case['inclusive_allocated_seconds']=time.perf_counter()-case_started
                case['inclusive_samples_per_allocated_gpu_hour']=case['policy_samples']/case['inclusive_allocated_seconds']*3600
                result['learning'].append(case)
                write_json(out/'partial.json',result)
        # Exercise actual public CLI subprocesses on CUDA, including fresh-process
        # save/resume, evaluation and the four-variant profile command.
        c=copy.deepcopy(cfg);c.update(iterations=2,optimization='packed',max_decision_phases=128,snapshot_every=1,
            opponents=[{'kind':'power','weight':.25},{'kind':'selfplay','weight':.25},{'kind':'historical','weight':.5}])
        config_path=out/'cli-config.json';write_json(config_path,c)
        base=out/'cli-base';resumed=out/'cli-resumed';continuous=out/'cli-continuous'
        def command(arguments,label):
            import sys
            argv=[sys.executable,'-m','integrations.pokemon.cli',*arguments]
            began=time.perf_counter()
            with (out/(label+'.log')).open('w') as log:
                process=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            if process.returncode:raise RuntimeError(f'Public CLI {label} failed with {process.returncode}; see log')
            return {'argv':argv,'seconds':time.perf_counter()-began,'returncode':0}
        commands=[]
        commands.append(command(['train','--config',str(config_path),'--output',str(base)],'cli-base'))
        commands.append(command(['train','--config',str(config_path),'--iterations','4','--resume',str(base/'checkpoint.pt'),'--output',str(resumed)],'cli-resumed'))
        commands.append(command(['train','--config',str(config_path),'--iterations','4','--output',str(continuous)],'cli-continuous'))
        commands.append(command(['evaluate','--config',str(config_path),'--checkpoint',str(continuous/'checkpoint.pt'),'--output',str(out/'cli-evaluation')],'cli-evaluation'))
        pcfg=copy.deepcopy(c);pcfg['hardware'].update(workers=1,environments_per_worker=2)
        profile_config=out/'cli-profile-config.json';write_json(profile_config,pcfg)
        commands.append(command(['profile','--config',str(profile_config),'--checkpoint',str(continuous/'checkpoint.pt'),'--repeats','1','--output',str(out/'cli-profile')],'cli-profile'))
        for stage in ('initial','final'):
            commands.append(command(['evaluate','--config',str(config_path),'--checkpoint',str(out/f'learn-1729-packed/{stage}.pt'),
                '--schedule',str(ROOT/'integrations/pokemon/recipes/heldout.json'),'--output',str(out/f'heldout-{stage}')],f'heldout-{stage}'))
        result['public_cuda_commands']=commands
        a=torch.load(resumed/'checkpoint.pt',map_location='cpu',weights_only=False);b=torch.load(continuous/'checkpoint.pt',map_location='cpu',weights_only=False)
        max_delta=max(float((a['model'][k]-v).abs().max()) for k,v in b['model'].items())
        if max_delta!=0:raise AssertionError(f'CUDA CLI resume changed model weights {max_delta}')
        result['cli_cuda_resume']={'bitwise_model_equal':True,'max_parameter_error':max_delta,'iterations':4,
          'torch_rng_equal':torch.equal(a['torch_rng'],b['torch_rng']),
          'cuda_rng_equal':all(torch.equal(x,y) for x,y in zip(a['cuda_rng'],b['cuda_rng']))}
        result['gpu_executed']=True;write_json(out/'results.json',result)
    finally:
        monitor.close();write_json(out/'gpu-utilization.json',monitor.samples)
    return result

def gpu_update_parity(cfg):
    """Real battle experience, then identical CUDA PPO updates with/without scalar syncs."""
    c=copy.deepcopy(cfg);c['max_decision_phases']=128
    m=initialize(c);t=Timings()
    with Vector(1,4) as v:
        samples,report=rollout(v,m,[[m,'power'],['power',m],[m,'random'],['random',m]],
            [712,713,714,715],c,np.random.default_rng(1),Inference(c['hardware']['device'],t),t)
    a=Policy(**c['network']).to(c['hardware']['device']);a.load_state_dict(m.state_dict())
    b=Policy(**c['network']).to(c['hardware']['device']);b.load_state_dict(m.state_dict())
    oa=torch.optim.Adam(a.parameters(),lr=c['learning_rate'],eps=1e-5)
    ob=torch.optim.Adam(b.parameters(),lr=c['learning_rate'],eps=1e-5)
    state=torch.cuda.get_rng_state()
    baseline=update(a,oa,samples,dict(c,optimization='baseline'),c['hardware']['device'],Timings())
    torch.cuda.set_rng_state(state)
    optimized=update(b,ob,samples,dict(c,optimization='packed'),c['hardware']['device'],Timings())
    delta=max(float((a.state_dict()[k]-v).abs().max().cpu()) for k,v in b.state_dict().items())
    if delta>1e-6:raise AssertionError(f'Asynchronous CUDA learner changed parameters {delta}')
    return {'real_battle_samples':len(samples),'completed_games':report['natural_terminals'],
            'truncations':report['truncations'],'max_parameter_error':delta,'bitwise_parameters_equal':delta==0,
            'baseline':baseline,'optimized':optimized}
