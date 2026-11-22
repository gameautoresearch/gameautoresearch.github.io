"""Actual CUDA validation and full-collector component performance development."""
import copy
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch

from ..network import Policy
from .bridge import HERE, build
from .resident import ResidentBattles
from .resident_ppo import ResidentPPO
from .teams_check import team_comparison


def assert_exact_tree(actual, expected, path='checkpoint'):
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual,torch.Tensor) or not torch.equal(actual,expected):
            raise AssertionError(f'{path} tensor changed')
    elif isinstance(expected, dict):
        if actual.keys()!=expected.keys():raise AssertionError(f'{path} keys changed')
        for key in expected:assert_exact_tree(actual[key],expected[key],f'{path}.{key}')
    elif isinstance(expected,(list,tuple)):
        if type(actual)!=type(expected) or len(actual)!=len(expected):raise AssertionError(f'{path} sequence changed')
        for key in range(len(expected)):assert_exact_tree(actual[key],expected[key],f'{path}[{key}]')
    elif isinstance(expected,np.ndarray):
        if not isinstance(actual,np.ndarray):raise AssertionError(f'{path} array type changed')
        np.testing.assert_array_equal(actual,expected)
    elif actual!=expected:raise AssertionError(f'{path} changed')


def flat_observations(state):
    keys=('tokens','mons','global','global_tokens','action_tokens','actions','mask')
    return np.asarray([np.concatenate([np.asarray(obs[key],dtype=np.float64).ravel() for key in keys])
                       for obs in state['observations']])


def compare_corpus(cpu,cuda,fixtures):
    env=ResidentBattles(cpu,cuda,fixtures,len(fixtures),developer_scope='retained constructed-team oracle corpus')
    masks=torch.empty((len(fixtures),2,14),dtype=torch.uint8,device='cuda')
    state_comparisons=0;oracle_frames=0;feature_values=0
    for frame in range(-1,max(len(data['frames']) for data in fixtures)):
        if frame>=0:
            actions=np.asarray([data['frames'][frame]['actions'] if frame<len(data['frames']) else [-1,-1]
                                for data in fixtures],dtype=np.int32)
            env.arena.step(actions,env.moves,env.chart)
            env.actions.copy_(torch.from_numpy(actions).cuda())
            code=env.lib.pkn_step_device(env.state.data_ptr(),env.actions.data_ptr(),env.n,env.move_data.data_ptr(),
                                       len(env.moves),env.chart_data.data_ptr(),env.stream)
            if code:raise RuntimeError(f'Corpus stepping CUDA error: {code}')
            env.refresh_observation()
        code=env.lib.pkn_masks_device(env.state.data_ptr(),masks.data_ptr(),env.n,env.stream)
        if code:raise RuntimeError(f'Corpus mask CUDA error: {code}')
        torch.cuda.synchronize()
        if env.state.cpu().numpy().tobytes()!=env.arena.snapshot():
            raise AssertionError(f'Complete CPU/CUDA state mismatch at phase {frame}')
        cpu_masks=env.arena.masks()
        np.testing.assert_array_equal(masks.cpu().numpy(),cpu_masks)
        cpu_obs=env.tables.observe(env.arena);gpu_obs=env.obs.cpu().numpy()
        np.testing.assert_array_equal(gpu_obs,cpu_obs.astype(np.float32))
        state_comparisons+=env.n
        for i,data in enumerate(fixtures):
            if frame>=len(data['frames']):continue
            reference=data['initial'] if frame<0 else data['frames'][frame]['state']
            mismatch=team_comparison(env.arena,reference,i,cpu_masks)
            if mismatch:raise AssertionError(f'Oracle mechanics mismatch case {i} phase {frame}: {mismatch}')
            target=flat_observations(reference)
            np.testing.assert_array_equal(cpu_obs[i],target)
            feature_values+=target.size
            oracle_frames+=int(frame>=0)
    env.close()
    return dict(cases=len(fixtures),complete_state_mask_comparisons=state_comparisons,
                oracle_transitions=oracle_frames,exact_oracle_feature_values=feature_values)


def run(output,fixture_path,generator_path=None,constructor_path=None,reset_path=None,task_path=None):
    started=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cpu=build(output/'cpu-build');cuda=output/'libpokemon_cuda.so'
    command=['nvcc','-std=c++17','-O3','--fmad=false','-arch=sm_89','--shared','-Xcompiler','-fPIC',
             '--ptxas-options=-v',str(Path(cpu).parent/'source'/'kernels.cu'),'-o',str(cuda),'-Xlinker','-rpath','-Xlinker','/usr/local/cuda/lib64']
    with (output/'cuda-compile.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    result=dict(full_gen9_admission=False,scope='constructed team/effect corpus; native random teams and complete callback catalog unfinished',
                gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,profiles=[],learning=[])
    if generator_path:
        from .teamgen_cuda import run as generator_check
        result['generator']=generator_check(cpu,cuda,generator_path,constructor_path)
        (output/'partial.json').write_text(json.dumps(result,indent=2)+'\n')
        result['scope']='Complete native team generation plus constructed battle corpus; full battle callback catalog remains unfinished'
    if reset_path:
        from .startup_cuda import run as reset_check
        result['random_reset']=reset_check(cpu,cuda,generator_path,constructor_path,reset_path)
        (output/'partial.json').write_text(json.dumps(result,indent=2)+'\n')
    if task_path:
        from .startup_cuda import run as reset_check
        result['task_domain']=reset_check(cpu,cuda,generator_path,constructor_path,task_path)
        (output/'partial.json').write_text(json.dumps(result,indent=2)+'\n')
        from .random_resident_check import run as random_collector_check
        result['random_collector']=random_collector_check(cpu,cuda,generator_path,constructor_path,task_path,output)
    # Stream retained corpora in bounded shards. Parsing the entire expanded
    # JSON corpus would exceed the worker's fixed 8 GiB attached RAM.
    fixtures=[];batch=[];totals={};fixture_digest=hashlib.sha256();fixture_count=0
    def qualify_batch():
        row=compare_corpus(cpu,cuda,batch)
        for key,value in row.items():totals[key]=totals.get(key,0)+value
        (output/'corpus-partial.json').write_text(json.dumps(totals,indent=2)+'\n')
        batch.clear()
    with Path(fixture_path).open('rb')as retained:
        for raw in retained:
            fixture_digest.update(raw);data=json.loads(raw);batch.append(data)
            if fixture_count%16==0 and len(fixtures)<256:fixtures.append(data)
            fixture_count+=1
            if len(batch)==256:qualify_batch()
    if batch:qualify_batch()
    result['corpus']=dict(**totals,shard_size=256,profile_templates=len(fixtures),profile_selection='every 16th retained fixture, maximum 256')
    (output/'partial.json').write_text(json.dumps(result,indent=2)+'\n')
    identities=[]
    for backend in ('native_cpu','cuda'):
        torch.manual_seed(731);torch.cuda.manual_seed_all(731)
        env=ResidentBattles(cpu,cuda,fixtures,256,backend=backend,cpu_threads=3,phase_limit=64,
                            developer_scope='constructed-team neural collector and checkpoint canary',diversify=True)
        model=Policy(hidden=64,embedding=8).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001,eps=1e-5)
        trainer=ResidentPPO(env,model,optimizer,steps=32,minibatch=4096,epochs=2)
        initial={k:v.clone() for k,v in model.state_dict().items()}
        first=trainer.collect_update();checkpoint=trainer.checkpoint();torch.save(checkpoint,output/f'{backend}-checkpoint.pt')
        second=trainer.collect_update();expected=trainer.checkpoint()
        trainer.restore(torch.load(output/f'{backend}-checkpoint.pt',weights_only=False))
        repeated=trainer.collect_update();actual=trainer.checkpoint()
        assert_exact_tree(actual,expected)
        if second!=repeated:raise AssertionError('Resident state/trajectory resume changed')
        changed=sum(float((value-initial[name]).square().sum()) for name,value in model.state_dict().items())**.5
        raw_state=env.state.cpu().numpy().tobytes() if backend=='cuda' else env.arena.snapshot()
        identities.append((copy.deepcopy(model.state_dict()),raw_state,torch.cuda.get_rng_state().clone()))
        result['learning'].append(dict(backend=backend,first=first,second=second,parameter_change_l2=changed,
                                       exact_resume=True,exact_optimizer_resume=True,final_state_sha256=hashlib.sha256(raw_state).hexdigest()))
        env.close();del trainer,model,optimizer,env
        (output/'partial.json').write_text(json.dumps(result,indent=2)+'\n')
    result['exact_cpu_cuda_learning']=all(torch.equal(identities[0][0][k],v) for k,v in identities[1][0].items()) and identities[0][1]==identities[1][1] and torch.equal(identities[0][2],identities[1][2])
    if not result['exact_cpu_cuda_learning']:raise AssertionError('Native CPU / resident CUDA learning diverged')
    del identities
    for n in (512,2048):
        entries={}
        for backend in ('native_cpu','cuda'):
            torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
            env=ResidentBattles(cpu,cuda,fixtures,n,backend=backend,cpu_threads=3,phase_limit=128,
                                developer_scope='component full-loop profile; not full Gen 9 benchmark',diversify=True)
            model=Policy(hidden=64,embedding=8).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001,eps=1e-5)
            trainer=ResidentPPO(env,model,optimizer,steps=32,minibatch=4096,epochs=2)
            warm=trainer.collect_update()
            entries[backend]=dict(env=env,trainer=trainer,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),warm=warm,rows=[])
        for repeat in range(5):
            order=('native_cpu','cuda') if repeat%2==0 else ('cuda','native_cpu')
            for backend in order:
                entry=entries[backend]
                torch.set_rng_state(entry['torch_rng']);torch.cuda.set_rng_state(entry['cuda_rng'])
                torch.cuda.synchronize();tick=time.perf_counter();row=entry['trainer'].collect_update();torch.cuda.synchronize()
                row.update(repeat=repeat,seconds=time.perf_counter()-tick);row['samples_per_second']=row['samples']/row['seconds']
                entry['rows'].append(row);entry['torch_rng']=torch.get_rng_state();entry['cuda_rng']=torch.cuda.get_rng_state()
                print(f'RESIDENT n={n} backend={backend} repeat={repeat} samples={row["samples"]} s={row["seconds"]:.4f}',flush=True)
            for key in ('samples','natural','truncations','faults'):
                if entries['native_cpu']['rows'][-1][key]!=entries['cuda']['rows'][-1][key]:raise AssertionError('Profile workloads diverged')
        for backend,entry in entries.items():
            rows=entry['rows']
            result['profiles'].append(dict(n=n,backend=backend,steps=32,epochs=2,minibatch=4096,warmup=entry['warm'],rows=rows,
                                           order='alternating CPU/GPU, separate exact RNG streams',
                                           median_samples_per_second=statistics.median(row['samples_per_second'] for row in rows)))
            entry['env'].close()
        del entries,entry,env,trainer,model,optimizer
        (output/'partial.json').write_text(json.dumps(result,indent=2)+'\n')
    result.update(peak_cuda_bytes=torch.cuda.max_memory_allocated(),allocated_seconds=time.perf_counter()-started,
                  fixture_sha256=fixture_digest.hexdigest())
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
