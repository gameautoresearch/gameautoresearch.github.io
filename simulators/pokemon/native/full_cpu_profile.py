"""One bounded L4 validation: native CPU simulation, GPU neural learning only.

No CUDA simulator is built or loaded. Input is an immutable local runtime and
unfiltered retained complete oracle games. This continuation reserves 240 L4
seconds, with no retry. Invoke this FILE through Modal, never with -m.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import modal

ROOT=Path(__file__).resolve().parents[3]if modal.is_local()else Path('/workspace')
OUT=ROOT/'evidence/pokemon-implementation/native-cpu-completion-20261109'
UPLOAD=None
if modal.is_local()and os.environ.get('POKEMON_NATIVE_RUNTIME'):
    runtime_path=Path(os.environ['POKEMON_NATIVE_RUNTIME']);runtime=json.loads(runtime_path.read_bytes())
    temporary=tempfile.TemporaryDirectory(prefix='pokemon-native-full-cpu-');UPLOAD=Path(temporary.name)
    library=runtime_path.parent/runtime['library'];shutil.copytree(library.parent/'source',UPLOAD/'native')
    for key in('generators','constructors','chart'):shutil.copyfile(runtime_path.parent/runtime[key],UPLOAD/f'{key}.json')
    for name in('network.py','efficient.py','env.py'):shutil.copyfile(ROOT/'simulators/pokemon'/name,UPLOAD/name)
    # Only neural tensors are retained; no old environment or optimizer resumes.
    import torch
    checkpoint=OUT/'learning-unfiltered-2/checkpoint.pt';cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    torch.save(cp['model'],UPLOAD/'weights.pt')
    policy=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),updates=cp['updates'],integration_signature=cp['integration_signature'],weights_sha256=hashlib.sha256((UPLOAD/'weights.pt').read_bytes()).hexdigest())
    (UPLOAD/'policy.json').write_text(json.dumps(policy))
    reference=OUT/'unfiltered-learned-2/oracle-fixtures.jsonl';cases=[]
    with reference.open()as stream:
        for _ in range(96):cases.append(json.loads(next(stream)))
    if not all(c['frames'][-1]['state']['ended']and not c.get('schedule_violations')for c in cases):raise ValueError('Complete unfiltered learned-policy oracle games required')
    (UPLOAD/'cases.json.gz').write_bytes(gzip.compress(json.dumps(cases).encode(),mtime=0))
    (UPLOAD/'candidate.json').write_text(json.dumps(dict(runtime=runtime,build=json.loads((library.parent/'build.json').read_bytes()))))

app=modal.App('pokemon-native-cpu-full-20261109')
volume=modal.Volume.from_name('pokemon-native-cpu-full-20261109',environment_name='staging',create_if_missing=True)
image=(modal.Image.from_registry('nvidia/cuda:13.0.2-devel-ubuntu24.04',add_python='3.12')
       .pip_install('numpy==2.5.4','torch==2.15.0')
       .env({'PYTHONPATH':'/workspace','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
       .add_local_dir(UPLOAD/'native','/workspace/integrations/pokemon/native')
       .add_local_file(UPLOAD/'network.py','/workspace/integrations/pokemon/network.py')
       .add_local_file(UPLOAD/'efficient.py','/workspace/integrations/pokemon/efficient.py')
       .add_local_file(UPLOAD/'env.py','/workspace/integrations/pokemon/env.py')
       .add_local_file(UPLOAD/'generators.json','/workspace/generators.json')
       .add_local_file(UPLOAD/'constructors.json','/workspace/constructors.json')
       .add_local_file(UPLOAD/'chart.json','/workspace/chart.json')
       .add_local_file(UPLOAD/'weights.pt','/workspace/weights.pt')
       .add_local_file(UPLOAD/'policy.json','/workspace/policy.json')
       .add_local_file(UPLOAD/'cases.json.gz','/workspace/cases.json.gz')
       .add_local_file(UPLOAD/'candidate.json','/workspace/candidate.json'))if UPLOAD else modal.Image.debian_slim()


@app.function(image=image,gpu='L4',cpu=(4,4),memory=(8192,8192),timeout=110,startup_timeout=120,
              max_containers=1,min_containers=0,buffer_containers=0,retries=0,single_use_containers=True,
              scaledown_window=2,volumes={'/receipts':volume})
def profile(label):
    started=time.perf_counter();result=dict(full_gen9_admission=False,simulation_backend='native C++ CPU; GPU neural learning only',resources=dict(gpu='L4',cpus=4,RAM_GiB=8),rows=[])
    try:
        import ctypes as C
        import statistics
        import numpy as np
        import torch
        from integrations.pokemon.native.bridge import build,Native
        from integrations.pokemon.native.cpu_env import CPURandomBattles
        from integrations.pokemon.native.random_resident import task_seeds
        from integrations.pokemon.native.resident_ppo import ResidentPPO
        from integrations.pokemon.native.resident_suite import assert_exact_tree
        from integrations.pokemon.native.teams_check import team_comparison
        from integrations.pokemon.native.resident_suite import flat_observations
        from integrations.pokemon.network import Policy
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
        tick=time.perf_counter();library=build('/workspace/cpu-build');result['compile_seconds']=time.perf_counter()-tick
        candidate=json.loads(Path('/workspace/candidate.json').read_bytes());compiled=json.loads(Path('/workspace/cpu-build/build.json').read_bytes())
        if compiled['source_sha256']!=candidate['build']['source_sha256']:raise ValueError('Remote CPU source differs from qualified candidate')
        result.update(remote_library_sha256=compiled['binary_sha256'],qualified_library_sha256=candidate['build']['binary_sha256'],exact_qualified_sources=True,
                      torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),policy=json.loads(Path('/workspace/policy.json').read_bytes()))
        chart=json.loads(Path('/workspace/chart.json').read_bytes())['chart'];cases=json.loads(gzip.decompress(Path('/workspace/cases.json.gz').read_bytes()))
        # Independent complete-game cross-platform audit. This is timed separately
        # from accepted learner collection and contributes zero learner rows.
        audit=CPURandomBattles(library,'/workspace/generators.json','/workspace/constructors.json',chart,
            [[task_seeds(c['spec']['master'],4096,c['spec']['swapped'])]for c in cases],threads=3,developer_scope='Cross-platform complete unfiltered natural-game oracle audit')
        simulator_seconds=observation_seconds=0.;features=0
        for phase in range(-1,max(len(c['frames'])for c in cases)):
            if phase>=0:
                actions=[c['frames'][phase]['actions']if phase<len(c['frames'])else[-1,-1]for c in cases]
                tick=time.perf_counter();audit.arena.step(actions,audit.moves,audit.chart);simulator_seconds+=time.perf_counter()-tick
            tick=time.perf_counter();actual=audit.tables.observe(audit.arena);observation_seconds+=time.perf_counter()-tick
            masks=audit.arena.masks()
            for i,c in enumerate(cases):
                state=c['initial']if phase<0 else c['frames'][min(phase,len(c['frames'])-1)]['state']
                mismatch=team_comparison(audit.arena,state,index=i,masks=masks)
                if mismatch is not None:raise AssertionError(f'Cross-platform oracle state mismatch {i}/{phase}: {mismatch}')
                if not np.array_equal(actual[i],flat_observations(state)):raise AssertionError(f'Cross-platform observation mismatch {i}/{phase}')
                features+=1400
        if not all(s.terminated and not s.truncated and not s.fault for s in audit.arena.records):raise AssertionError('Complete natural audit failed')
        result['cross_platform_audit']=dict(games=len(cases),exact_public_values=features,simulator_seconds=simulator_seconds,observation_seconds=observation_seconds)
        audit.close();del audit,cases

        class GPUCPU:
            backend='native_cpu_gpu_learning'
            def __init__(self,n):
                schedule=[[task_seeds(1160000+(i//2)*256+j,4096,bool(i%2))for j in range(256)]for i in range(n)]
                self.cpu=CPURandomBattles(library,'/workspace/generators.json','/workspace/constructors.json',chart,schedule,threads=3,developer_scope='Bounded full-format CPU simulation with GPU neural validation')
                self.n=n;self.obs=self.cpu.obs.cuda();self.host_actions=torch.empty((n,2),dtype=torch.int32,pin_memory=True)
                pinned=torch.empty_like(self.cpu.obs,pin_memory=True);pinned.copy_(self.cpu.obs);self.cpu.obs=pinned
                for name in('source_signature','binary_sha256','rules_sha256','initial_sha256','integration_signature'):setattr(self,name,getattr(self.cpu,name))
                self.profiling=False;self.timers=dict(transfers=0.,simulator_and_observation=0.)
            def allocate_rollout(self,steps):
                self.cpu.allocate_rollout(steps)
                for name in('flags','rewards','final_obs'):
                    original=getattr(self.cpu,name);setattr(self.cpu,name,torch.empty_like(original,pin_memory=True))
                self.flags=torch.empty_like(self.cpu.flags,device='cuda');self.rewards=torch.empty_like(self.cpu.rewards,device='cuda')
            def step(self,actions,index,faint_reward=0):
                if self.profiling:torch.cuda.synchronize()
                tick=time.perf_counter();self.host_actions.copy_(actions,non_blocking=True);torch.cuda.synchronize()
                if self.profiling:self.timers['transfers']+=time.perf_counter()-tick
                tick=time.perf_counter();self.cpu.step(self.host_actions,index,faint_reward)
                if self.profiling:self.timers['simulator_and_observation']+=time.perf_counter()-tick
                tick=time.perf_counter()
                for target,source in((self.obs,self.cpu.obs),(self.flags[index],self.cpu.flags[index]),(self.rewards[index],self.cpu.rewards[index])):target.copy_(source,non_blocking=True)
                if self.profiling:torch.cuda.synchronize();self.timers['transfers']+=time.perf_counter()-tick
            def truncated_views(self):
                selected=self.flags[...,1].bool()[...,None].expand(-1,-1,2).reshape(-1)
                cpu_selected=self.cpu.flags[...,1].bool()[...,None].expand(-1,-1,2).reshape(-1)
                return self.cpu.final_obs.reshape(-1,700)[cpu_selected].cuda(),selected
            def snapshot(self):return self.cpu.snapshot()
            def restore(self,state):self.cpu.restore(state);self.obs.copy_(self.cpu.obs,non_blocking=True)
            def close(self):self.cpu.close()
        tick=time.perf_counter();env=GPUCPU(2048);model=Policy(hidden=64,embedding=8).cuda();model.load_state_dict(torch.load('/workspace/weights.pt',map_location='cuda',weights_only=True))
        optimizer=torch.optim.Adam(model.parameters(),lr=.0003,eps=1e-5);trainer=ResidentPPO(env,model,optimizer,steps=32,minibatch=4096,epochs=2,faint_reward=0,value_coef=.5)
        torch.cuda.synchronize();result['learner_setup_seconds']=time.perf_counter()-tick
        result['warmup']=trainer.collect_update();torch.cuda.synchronize()
        for repeat in range(80):
            if time.perf_counter()-started>=85:break
            tick=time.perf_counter();row=trainer.collect_update();torch.cuda.synchronize();row.update(seconds=time.perf_counter()-tick,repeat=repeat);row['samples_per_second']=row['samples']/row['seconds'];result['rows'].append(row)
            if row['faults']or row['truncations']:raise AssertionError('Full-format performance collection faulted or capped')
        if len(result['rows'])<3:raise AssertionError('Insufficient complete full-loop measurements')
        result['median_accepted_new_rows_per_second']=statistics.median(r['samples_per_second']for r in result['rows'])
        # Instrument one additional update. Forward/GAE/PPO wall phases include
        # explicit GPU synchronization and are not substituted for warm-loop rates.
        phases=dict(inference=0.,GAE=0.);first_grad=[None];forward=model.forward
        def measured_forward(x):
            torch.cuda.synchronize();tick=time.perf_counter();grad=torch.is_grad_enabled()
            if grad and first_grad[0]is None:first_grad[0]=tick
            answer=forward(x)
            if not grad:torch.cuda.synchronize();phases['inference']+=time.perf_counter()-tick
            return answer
        import integrations.pokemon.native.resident_ppo as ppo_module
        gae=ppo_module.decision_gae
        def measured_gae(*args,**kwargs):
            torch.cuda.synchronize();tick=time.perf_counter();answer=gae(*args,**kwargs);torch.cuda.synchronize();phases['GAE']+=time.perf_counter()-tick;return answer
        model.forward=measured_forward;ppo_module.decision_gae=measured_gae;env.profiling=True
        tick=time.perf_counter();instrumented=trainer.collect_update();torch.cuda.synchronize();finished=time.perf_counter();phases['PPO']=finished-first_grad[0];phases['instrumented_loop']=finished-tick
        phases.update(env.timers);model.forward=forward;ppo_module.decision_gae=gae;env.profiling=False
        phases['opponent']=0.;phases['simulator']=simulator_seconds;phases['observation']=observation_seconds
        # A complete GPU-neural/CPU-environment resume, excluding the duplicate
        # verification rollout from accepted-new-row totals.
        cp=trainer.checkpoint();tick=time.perf_counter();torch.save(cp,'/receipts/'+label+'-checkpoint.pt');phases['checkpoint']=time.perf_counter()-tick
        expected_row=trainer.collect_update();torch.cuda.synchronize();expected=trainer.checkpoint();trainer.restore(cp);actual_row=trainer.collect_update();torch.cuda.synchronize();assert_exact_tree(trainer.checkpoint(),expected)
        if expected_row!=actual_row:raise AssertionError('Full GPU neural/CPU environment checkpoint continuation changed')
        result.update(instrumented=instrumented,phase_seconds=phases,exact_full_resume=True,accepted_new_both_seat_rows=result['warmup']['samples']+sum(r['samples']for r in result['rows'])+instrumented['samples']+expected_row['samples'],
                      natural_games=result['warmup']['natural']+sum(r['natural']for r in result['rows'])+instrumented['natural']+expected_row['natural'],
                      truncations=0,faults=0,duplicate_resume_samples_excluded=actual_row['samples'],ppo_epoch_reuse_excluded=True,opponent_rows=0,
                      count_both_seats=True,peak_cuda_bytes=torch.cuda.max_memory_allocated(),source_signature=env.source_signature)
        env.close()
    except BaseException:
        import traceback
        result['error']=traceback.format_exc()
    result['body_before_receipt_seconds']=time.perf_counter()-started
    target=Path('/receipts')/label;target.mkdir(exist_ok=False);(target/'results.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    result['body_seconds']=time.perf_counter()-started
    return result


@app.local_entrypoint()
def main(label:str='full-cpu-1'):
    from modal.config import config
    if config.get('environment')!='staging'or UPLOAD is None:raise RuntimeError('Staging and POKEMON_NATIVE_RUNTIME are required')
    reserved=sum(json.loads(p.read_bytes()).get('reserved_L4_seconds',0)for p in OUT.glob('*/allocation.json'))
    if reserved+240>1200:raise RuntimeError('Separate 1200-L4-second continuation ceiling exhausted')
    target=OUT/label;target.mkdir(parents=True,exist_ok=False)
    allocation=dict(reserved_L4_seconds=240,prior_reserved_L4_seconds=reserved,body_timeout_seconds=110,startup_timeout_seconds=120,teardown_margin_seconds=10,
                    retries=0,workers=1,resources=dict(gpu='L4',cpus=4,RAM_GiB=8),status='reserved')
    (target/'allocation.json').write_text(json.dumps(allocation,indent=2)+'\n');started=time.perf_counter()
    try:
        allocation['app_id']=app.app_id;result=profile.remote(label);allocation['client_call_seconds']=time.perf_counter()-started
        (target/'results.json').write_text(json.dumps(result,indent=2)+'\n');allocation['status']='returned'
        if 'error'in result:raise RuntimeError(result['error'])
    except BaseException as error:
        allocation.update(status='failed',error=str(error),client_call_seconds=time.perf_counter()-started);raise
    finally:(target/'allocation.json').write_text(json.dumps(allocation,indent=2)+'\n')
