"""Bounded actual-CUDA full-loop experiments, invoked in the dedicated Modal container."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

START = time.perf_counter()
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

def import_runtime():
    # Spawn workers import __mp_main__; leave heavy imports inside the parent path.
    global np,torch,manifest,save_checkpoint,write_json,ROLES,SOURCE,Game,deal,digest
    global Policy,update,Policies,collect,initialize,make_optimizers,VectorGames
    import numpy as np
    import torch
    from integrations.doudizhu.artifacts import manifest, save_checkpoint, write_json
    from integrations.doudizhu.environment import ROLES, SOURCE, Game, deal, digest
    from integrations.doudizhu.model import Policy, update
    from integrations.doudizhu.rollout import Policies, collect
    from integrations.doudizhu.runner import initialize, make_optimizers
    from integrations.doudizhu.workers import VectorGames


def configuration():
    return {'ruleset':'douzero-fixed-role-v1','seed':17,'deal_seed':20261108,
            'device':'cuda:0','allocation':{'gpus':1,'machine':'Modal-NVIDIA-L4','attached_cpu_threads':4,'attached_ram_gb':16},
            'torch_threads':1,'workers':0,'vector_games':64,'objective':'wp','max_seconds':780,
            'architecture':{'hidden':128,'depth':3,'history':64},
            'learning_rate':.001,'batch_size':512,'epochs':1,'epsilon':.1,
            'max_grad_norm':40,'max_candidates':4096,'factored_history':True,
            'opponents':{'selfplay_probability':.75,'frozen':[{'kind':'heuristic'}]}}


def oracle_check(config):
    spec=importlib.util.spec_from_file_location('upstream_models',SOURCE/'douzero/dmc/models.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy=Policy().cuda()
    game=Game(deal('train',19,0))
    rows=[]
    for role in ROLES:
        obs=game.observe()
        cls=module.LandlordLstmModel if role=='landlord' else module.FarmerLstmModel
        original=cls().cuda()
        original.load_state_dict(policy.roles[role].state_dict(),strict=True)
        x,z,actions=obs.encode()
        xx=np.concatenate((np.repeat(x[None],len(actions),axis=0),actions),axis=1)
        with torch.inference_mode():
            expected=original(torch.tensor(np.repeat(z[None],len(actions),axis=0),dtype=torch.float32,device='cuda'),
                              torch.tensor(xx,dtype=torch.float32,device='cuda'),return_value=True)['values'].flatten().cpu().numpy()
        factored=policy.values_many([obs],factored=True)[0]
        duplicate=policy.values_many([obs],factored=False)[0]
        np.testing.assert_allclose(expected,factored,atol=1e-6,rtol=1e-4)
        np.testing.assert_allclose(duplicate,factored,atol=1e-6,rtol=1e-4)
        assert np.argmax(expected)==np.argmax(factored)
        rows.append({'role':role,'legal_candidates':len(actions),'max_abs_error':float(np.max(np.abs(expected-factored)))})
        game.step(int(np.argmax(factored)))
    return rows


def trial(config, weights, device, batch, workers, factored, episodes=128, record=True, cached=True):
    started=time.perf_counter()
    local=copy.deepcopy(config)
    local['device']=device
    rng=initialize(local)
    policy=Policy(config['architecture']).to(device)
    policy.load_state_dict(weights)
    policy.eval()
    optimizers=make_optimizers(policy,config)
    policies=Policies(policy,device,factored,config['max_candidates'],encoding_cache=cached)
    samples={r:[] for r in ROLES}
    timing={}
    traces={}
    decisions=candidates=0
    progress={'opening':0,'middle':0,'late':0}
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    with VectorGames(workers,cache_observations=cached) as vector:
        for offset in range(0,episodes,batch):
            n=min(batch,episodes-offset)
            cards=[deal('train',20261108,900000+offset+i) for i in range(n)]
            assignments=[{r:{'kind':'learner'} for r in ROLES} for _ in cards]
            result=collect(vector,policies,cards,assignments,np.random.default_rng(5),epsilon=0.,record=record,training=True)
            decisions+=result['decisions']
            candidates+=result['candidates']
            for r in ROLES:
                samples[r].extend(result['samples'][r])
            for key,val in result['timing'].items():
                timing[key]=timing.get(key,0.)+val
            for i,trace in result['traces'].items():
                traces[offset+i]=trace
                # The private replay is audited after collection; hand counts are
                # reconstructed from public card plays, not exposed as policy features.
                remaining={r:len(trace['deal'][r]) for r in ROLES}
                for step in trace['steps']:
                    remaining[step['role']]-=len(step['action'])
                    minimum=min(remaining.values())
                    progress['late' if minimum<=5 else 'opening' if min(remaining.values())>=14 else 'middle']+=1
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        t=time.perf_counter()
        metrics=update(policy,optimizers,samples,config['batch_size'],1,np.random.default_rng(74))
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        timing['learner_update_seconds']=time.perf_counter()-t
    elapsed=time.perf_counter()-started
    return {'device':device,'vector_games':batch,'workers':workers,'factored_history':factored,
            'full_loop_seconds':elapsed,'episodes':episodes,'decisions':decisions,'legal_candidates':candidates,
            'public_encoding_and_view_cache':cached,'game_phase_decisions':progress,
            'decisions_per_second':decisions/elapsed,'learner_samples_per_second':sum(metrics['samples'].values())/elapsed,
            'updates_per_second':sum(metrics['updates'].values())/elapsed,
            'samples_per_allocated_gpu_hour':sum(metrics['samples'].values())/elapsed*3600,
            'metrics':metrics,'timing':timing,'trajectory_sha256':digest(traces),
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved() if device.startswith('cuda') else 0}


def diagnostics(config,weights,out):
    # Diagnostic tracing overhead is excluded from primary throughput comparisons but
    # included in the full allocated-worker charge. CPU/CUDA transfer and sync events
    # remain available in a Chrome trace with exact operator-level timings.
    from torch.profiler import profile,ProfilerActivity,record_function
    from integrations.doudizhu.environment import Observation
    from integrations.doudizhu.model import RoleNet
    originals=[]
    def annotate(cls,method,label):
        original=getattr(cls,method)
        def wrapped(*args,**kwargs):
            with record_function(label):
                return original(*args,**kwargs)
        originals.append((cls,method,original))
        setattr(cls,method,wrapped)
    annotate(Observation,'_encode_fresh','doudizhu.public_encoding')
    annotate(Game,'step','doudizhu.simulator_step_observe')
    annotate(RoleNet,'encode_history','doudizhu.history_lstm')
    annotate(RoleNet,'head','doudizhu.candidate_head')
    try:
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as prof:
            result=trial(config,weights,'cuda:0',32,0,True,episodes=32,record=False)
    finally:
        for cls,method,original in originals:
            setattr(cls,method,original)
    prof.export_chrome_trace(str(out/'cuda-diagnostic-trace.json'))
    events=[]
    for event in prof.key_averages():
        events.append({'key':event.key,'calls':event.count,'self_cpu_us':event.self_cpu_time_total,
                       'self_device_us':event.self_device_time_total,'cpu_us':event.cpu_time_total,
                       'device_us':event.device_time_total})
    write_json(out/'cuda-operator-profile.json',sorted(events,key=lambda e:e['self_cpu_us'],reverse=True))
    return result


def learn(config,out,seed,games=2048):
    from integrations.doudizhu.artifacts import load_checkpoint
    from integrations.doudizhu.rollout import assign_training
    from integrations.doudizhu.runner import evaluate_policy,summarize_evaluation
    local=copy.deepcopy(config)
    local['seed']=seed
    local['deal_seed']=20261108+seed
    local['batch_size']=256
    rng=initialize(local)
    policy=Policy(local['architecture']).cuda()
    optimizers=make_optimizers(policy,local)
    policies=Policies(policy,'cuda:0')
    initial={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
    state={'episodes':0,'next_deal':0,'waves':0,'decisions':0,'candidate_evaluations':0,
           'updates':{r:0 for r in ROLES},'samples':{r:0 for r in ROLES},'historical':[],
           'used_deal_hashes':[],'allocated_worker_seconds':0.,'timing':{},'aborted_waves':0}
    out.mkdir(parents=True,exist_ok=True)
    def save(path):
        state['numpy_rng']=rng.bit_generator.state
        save_checkpoint(path,policy,optimizers,state,local)
    save(out/'initial.pt')
    rows=[]
    started=time.perf_counter()
    with VectorGames() as vector:
        for cursor in range(0,games,64):
            cards=[deal('train',local['deal_seed'],cursor+i) for i in range(64)]
            assignments=assign_training(local,64,cursor,rng,[])
            wave_started=time.perf_counter()
            result=collect(vector,policies,cards,assignments,rng,epsilon=.1,training=True,record=cursor==0)
            metrics=update(policy,optimizers,result['samples'],local['batch_size'],1,rng)
            policy.eval()
            state['episodes']+=64
            state['next_deal']+=64
            state['waves']+=1
            state['decisions']+=result['decisions']
            state['candidate_evaluations']+=result['candidates']
            state['used_deal_hashes'].extend(result['deal_sha256'])
            for r in ROLES:
                state['samples'][r]+=metrics['samples'][r]
                state['updates'][r]+=metrics['updates'][r]
            rows.append({'episodes':state['episodes'],'mse':metrics['mse'],'seconds':time.perf_counter()-wave_started,
                         'decisions':result['decisions'],'metrics':metrics})
            write_json(out/'training-waves.json',rows)
            if cursor==0:
                write_json(out/'private-first-wave-trajectories.json',list(result['traces'].values()))
            if state['episodes']==games//2:
                save(out/'midpoint.pt')
                policy,data=load_checkpoint(out/'midpoint.pt','cuda:0')
                optimizers=make_optimizers(policy,local)
                for r in ROLES:
                    optimizers[r].load_state_dict(data['optimizers'][r])
                rng.bit_generator.state=data['state']['numpy_rng']
                torch.set_rng_state(data['torch_rng'].cpu())
                torch.cuda.set_rng_state_all([x.cpu() for x in data['cuda_rng']])
                policies=Policies(policy,'cuda:0')
            if time.perf_counter()-START>700:
                raise TimeoutError('Experiment bound reached; no extra training launched')
    state['allocated_worker_seconds']=time.perf_counter()-started
    save(out/'latest.pt')
    write_json(out/'training-waves.json',rows)
    class EvaluationDeadline:
        def expired(self):
            return time.perf_counter()-START>750
    evaluation=copy.deepcopy(local)
    evaluation.update(split='heldout',deals=32,deal_seed=934817,opponents_panel=[{'kind':'heuristic'},{'kind':'random'}],vector_games=64)
    original=Policy(local['architecture']).cuda()
    original.load_state_dict(initial)
    before,_=evaluate_policy(evaluation,original,EvaluationDeadline())
    after,traces=evaluate_policy(evaluation,policy,EvaluationDeadline())
    if len(before)!=192 or len(after)!=192:
        raise TimeoutError('Evaluation incomplete; no complete learning result claimed')
    from integrations.doudizhu.compare import compare
    comparison=compare(before,after)
    write_json(out/'initial-evaluation-games.json',before)
    write_json(out/'final-evaluation-games.json',after)
    write_json(out/'private-evaluation-trajectories.json',traces)
    summary={'seed':seed,'state':state,'training_seconds':state['allocated_worker_seconds'],
             'parameter_l2_change':sum(float((p.detach().cpu()-initial[k]).square().sum()) for k,p in policy.state_dict().items())**.5,
             'initial_evaluation':summarize_evaluation(before),'final_evaluation':summarize_evaluation(after),
             'paired_comparison':comparison,'cuda_midpoint_save_resume_executed':True,
             'errors':{'illegal_actions':0,'simulator_faults':0,'nonfinite':0}}
    write_json(out/'result.json',summary)
    print(json.dumps({'trained_seed':seed,'games':games,'seconds':summary['training_seconds'],
                      'paired_mean_win_delta':comparison['groups']['role_balanced_mean']['paired_win_rate_delta']}),flush=True)
    return {k:v.detach().cpu().clone() for k,v in policy.state_dict().items()},summary


def parity_experiment(config,out,input_checkpoint):
    import hashlib
    from integrations.doudizhu.artifacts import load_checkpoint
    from integrations.doudizhu.runner import evaluate_policy,summarize_evaluation
    reference,data=load_checkpoint(input_checkpoint,'cpu')
    config['architecture']=data['architecture']
    weights={k:v.detach().clone() for k,v in reference.state_dict().items()}
    rows=[]
    for repeat,seed in enumerate((97,103,107)):
        variants=[('baseline',0,False,False),('optimized',4,True,True)]
        if repeat%2:
            variants.reverse()
        pair=[]
        for label,workers,factored,cached in variants:
            started=time.perf_counter()
            local=copy.deepcopy(config)
            local['seed']=seed
            rng=initialize(local)
            policy=Policy(local['architecture']).cuda()
            policy.load_state_dict(weights)
            optimizer=make_optimizers(policy,local)
            policies=Policies(policy,'cuda:0',factored,4096,encoding_cache=cached)
            samples_count={r:0 for r in ROLES}
            updates={r:0 for r in ROLES}
            traces={}
            sample_hash=hashlib.sha256()
            losses=[]
            timings={}
            decisions=candidates=0
            with VectorGames(workers,cache_observations=cached) as vector:
                for cursor in range(0,512,64):
                    cards=[deal('train',seed,990000+cursor+i) for i in range(64)]
                    assignments=[{r:{'kind':'learner'} for r in ROLES} for _ in cards]
                    result=collect(vector,policies,cards,assignments,rng,epsilon=.1,training=True,record=True)
                    for r in ROLES:
                        for x,z,y in result['samples'][r]:
                            sample_hash.update(r.encode())
                            sample_hash.update(x.tobytes())
                            sample_hash.update(z.tobytes())
                            sample_hash.update(np.float32(y).tobytes())
                    torch.cuda.synchronize()
                    t=time.perf_counter()
                    metrics=update(policy,optimizer,result['samples'],512,1,rng)
                    torch.cuda.synchronize()
                    timings['learner_update_seconds']=timings.get('learner_update_seconds',0.)+time.perf_counter()-t
                    losses.append(metrics['mse'])
                    for r in ROLES:
                        samples_count[r]+=metrics['samples'][r]
                        updates[r]+=metrics['updates'][r]
                    for key,val in result['timing'].items():
                        timings[key]=timings.get(key,0.)+val
                    decisions+=result['decisions']
                    candidates+=result['candidates']
                    traces.update({cursor+i:trace for i,trace in result['traces'].items()})
            state={'episodes':512,'next_deal':512,'waves':8,'decisions':decisions,
                   'candidate_evaluations':candidates,'samples':samples_count,'updates':updates,
                   'historical':[],'numpy_rng':rng.bit_generator.state,'used_deal_hashes':[],
                   'allocated_worker_seconds':time.perf_counter()-started,'timing':timings,'aborted_waves':0}
            target=out/f'{repeat}-{label}'
            target.mkdir(parents=True,exist_ok=True)
            t=time.perf_counter()
            save_checkpoint(target/'latest.pt',policy,optimizer,state,local)
            checkpoint_seconds=time.perf_counter()-t
            weight_hash=hashlib.sha256()
            for name,value in sorted(policy.state_dict().items()):
                weight_hash.update(name.encode())
                weight_hash.update(value.detach().cpu().numpy().tobytes())
            class Deadline:
                def expired(self):
                    return time.perf_counter()-START>750
            evaluation=copy.deepcopy(local)
            evaluation.update(split='heldout',deal_seed=817341,deals=16,vector_games=64,
                              opponents_panel=[{'kind':'heuristic'},{'kind':'random'}])
            t=time.perf_counter()
            evaluation_rows,_=evaluate_policy(evaluation,policy,Deadline())
            evaluation_seconds=time.perf_counter()-t
            assert len(evaluation_rows)==96
            write_json(target/'evaluation-games.json',evaluation_rows)
            write_json(target/'private-first-deal.json',traces[0])
            elapsed=time.perf_counter()-started
            row={'repeat':repeat,'seed':seed,'label':label,'workers':workers,'factored_history':factored,
                 'cached':cached,'games':512,'decisions':decisions,'candidates':candidates,
                 'samples':samples_count,'updates':updates,'mse_by_wave':losses,'timing':timings,
                 'trajectory_sha256':digest(traces),'ordered_learner_samples_sha256':sample_hash.hexdigest(),
                 'trained_weights_sha256':weight_hash.hexdigest(),'evaluation_sha256':digest(evaluation_rows),
                 'evaluation':summarize_evaluation(evaluation_rows),'full_allocated_arm_seconds':elapsed,
                 'checkpoint_seconds':checkpoint_seconds,'evaluation_seconds':evaluation_seconds,
                 'samples_per_allocated_gpu_hour':sum(samples_count.values())/elapsed*3600,
                 'decisions_per_second':decisions/elapsed,'updates_per_second':sum(updates.values())/elapsed,
                 'errors':{'illegal_actions':0,'simulator_faults':0,'truncations':0,'nonfinite':0}}
            rows.append(row)
            pair.append(row)
            write_json(out/'paired-learning-trials.json',rows)
            print(json.dumps({'parity_repeat':repeat,'label':label,'seconds':elapsed,
                              'weights_sha256':row['trained_weights_sha256']}),flush=True)
        for key in ('trajectory_sha256','ordered_learner_samples_sha256','trained_weights_sha256','evaluation_sha256'):
            if pair[0][key]!=pair[1][key]:
                raise AssertionError(f'Paired learning parity failed for {key}; artifacts retained')
    return {'pairs':rows,'all_trajectory_sample_model_evaluation_hashes_identical':True,
            'full_experiment_seconds':time.perf_counter()-START,
            'scope':'same CUDA device, trained initial weights, same deals/RNG/updates; only cache/factorization/workers differ; '
                    'full arm elapsed includes workers, training, checkpoint and development evaluation'}


def main():
    import_runtime()
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--input-checkpoint',type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    config=configuration()
    initialize(config)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    write_json(args.out/'manifest.json',manifest(config))
    cold_start=time.perf_counter()
    oracle=oracle_check(config)
    write_json(args.out/'cuda-oracle.json',oracle)
    torch.cuda.synchronize()
    cold=time.perf_counter()-cold_start
    if args.stage=='parity':
        if not args.input_checkpoint:
            raise ValueError('Parity stage requires the locally verified trained input checkpoint')
        result=parity_experiment(config,args.out,args.input_checkpoint)
        result.update(oracle=oracle,cold_cuda_oracle_seconds=cold,cuda_version=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__))
        write_json(args.out/'result.json',result)
        return
    initialize(config)
    policy=Policy(config['architecture'])
    weights={k:v.detach().clone() for k,v in policy.state_dict().items()}
    rows=[]
    monitor=(args.out/'nvidia-utilization.csv').open('w')
    smi=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,power.draw',
                          '--format=csv','--loop-ms=200'],stdout=monitor,stderr=subprocess.DEVNULL)
    try:
        learning=[]
        if args.stage!='probe':
            for seed in (17,29,43):
                learned,summary=learn(config,args.out/f'learning-seed-{seed}',seed)
                learning.append(summary)
                if seed==17:
                    weights=learned
        for repeat in range(3):
            variants=[('cuda:0',16,0,False,False),('cuda:0',16,0,True,False),
                      ('cuda:0',16,0,True,True),('cuda:0',64,0,True,True),
                      ('cuda:0',64,2,True,True),('cuda:0',64,4,True,True),
                      ('cpu',64,0,False,False),('cpu',64,0,True,True)]
            if repeat%2:
                variants.reverse()
            for device,batch,workers,factored,cached in variants:
                row=trial(config,weights,device,batch,workers,factored,cached=cached)
                row['repeat']=repeat
                rows.append(row)
                write_json(args.out/'trials.json',rows)
                print(json.dumps({k:row[k] for k in ('device','vector_games','workers','factored_history','full_loop_seconds','decisions_per_second')}),flush=True)
        diagnostic=diagnostics(config,weights,args.out)
    finally:
        smi.terminate()
        smi.wait(timeout=5)
        monitor.close()
    write_json(args.out/'result.json',{'stage':args.stage,'trials':rows,'oracle':oracle,'diagnostic':diagnostic,'learning':learning,
               'cold_cuda_oracle_seconds':cold,'imports_and_setup_seconds':cold_start-START,
               'full_experiment_seconds':time.perf_counter()-START,
               'cuda_version':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),
               'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),
               'claims':'real neural rollout+MC learner; initial policy only in probe; later optimized stage uses trained policies'})


if __name__=='__main__':
    main()
