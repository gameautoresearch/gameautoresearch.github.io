"""Real entrant entrypoint using canonical native rollouts, learner and snapshots."""
from pathlib import Path
import json,shutil,sys,time,os
from .common import DATA,EDITABLE,MATCH_FORM,STOP_REQUEST,setup_native,load_module,dump,sha

def runtime_binding():
    from .common import GAME,HERE
    from .graph_bridge import FILES
    return {str(p.relative_to(GAME)):sha(p) for p in (GAME/'model.py',HERE/'fast_policy.py',HERE/'fast_adapter.py',HERE/'fast_entry.py',HERE/'train_entry.py',HERE/'checkpoint_writer.py',*(HERE/n for n in FILES))}

def run(source,out,seconds,resume=None,updates=None,deadline_at=None,worker_started_at=None,milestone_seconds=(),milestone_offset_seconds=0):
    started=time.perf_counter();setup_native()
    # Before any entrant code: the native graph bridges are built and loaded in their copied-body form.
    from .graph_bridge import install
    install()
    # Entrant code runs under a distinct UID; never inherit a root-only Triton cache.
    os.environ.setdefault('TRITON_CACHE_DIR',f'/tmp/clash-triton-{os.getuid()}')
    import numpy as np,torch
    source=Path(source).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    c=json.loads((source/'config.json').read_text())
    if c.get('max_ticks',0)!=0 or c.get('rules',{}).get('max_ticks',0)!=0:raise ValueError('benchmark training uses natural match horizon')
    if c.get('match_form')!=MATCH_FORM:raise ValueError(f'benchmark training uses match_form {MATCH_FORM!r}')
    if not 1<=c['envs']<=2048 or not 1<=c['rollout_steps']<=256:raise ValueError('shape cap')
    if c.get('warmstart_steps',0) or c.get('reward_shaping',0) or c.get('opponent_mode')!='selfplay':raise ValueError('starting recipe requires unassisted sparse selfplay')
    expected={'source':{n:sha(source/n) for n in (*EDITABLE,'config.json')},'trusted_runtime':runtime_binding()}
    origin=torch.load(resume,map_location='cpu',weights_only=False) if resume else None
    if origin is not None and origin.get('entrant_binding')!=expected:raise ValueError('entrant resume binding differs')
    # Each entrypoint is a separate worker process. Clear only the editable module
    # names so local same-process correctness checks also select their exact source.
    sys.path.insert(0,str(source))
    for n in ('policy','learner'):sys.modules.pop(n,None)
    module=load_module(source/'train.py','entrant_train')
    trainer=module.Trainer(c,DATA,resume)
    initial={k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()}
    initial_samples=trainer.samples;state=(origin or {}).get('entrant_state',{})
    steps=state.get('optimizer_steps',0);losses=list(state.get('losses',[]));milestones_done=[]
    phases={'initialization':time.perf_counter()-started,'collect_gae':0.,'ppo':0.,'checkpoint_export':0.,'cleanup':0.}
    limit=c['updates'] if updates is None else updates
    try:
        from simulators.clash.native.train import archive_owned_source
        archive=archive_owned_source(out,trainer.binding['source'])
        dump(out/'manifest.json',{'entrant_binding':expected,'native_binding':trainer.binding,'config':trainer.c,'execution_runtime':trainer.execution_runtime,**archive})
        with (out/'progress.jsonl').open('w') as log:
            while trainer.update<limit and time.perf_counter()-started<seconds and (deadline_at is None or time.time()<deadline_at) and not os.path.exists(STOP_REQUEST):
                mark=time.perf_counter();data=trainer.collect()
                if trainer.gpu:torch.cuda.synchronize()
                phases['collect_gae']+=time.perf_counter()-mark
                mark=time.perf_counter();metrics,count=trainer.learn(data);numbers=metrics.detach().cpu().tolist()
                phases['ppo']+=time.perf_counter()-mark
                if not all(np.isfinite(numbers)):raise ValueError('nonfinite loss or gradient')
                steps+=count;losses.append(numbers)
                row={'updates':trainer.update,'samples':trainer.samples,'natural_games':int(trainer.counters[0]),'optimizer_steps':steps,'losses':numbers,'counters':trainer.counters.cpu().tolist(),'process_seconds':time.perf_counter()-started}
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if worker_started_at is not None:
                    worker_elapsed=time.time()-worker_started_at
                    for threshold in milestone_seconds:
                        if threshold not in milestones_done and worker_elapsed+milestone_offset_seconds>=threshold:
                            folder=out/'milestones'/str(int(threshold));folder.mkdir(parents=True)
                            export(trainer,source,folder)
                            dump(folder/'result.json',{'target_cumulative_worker_seconds':threshold,'observed_worker_seconds':worker_elapsed,'prior_failed_worker_seconds':milestone_offset_seconds,'observed_cumulative_worker_seconds':worker_elapsed+milestone_offset_seconds,'updates':trainer.update,'samples':trainer.samples,'natural_games':int(trainer.counters[0]),'kind':'weights-only intermediate; not an optimizer/world resume checkpoint'})
                            milestones_done.append(threshold)
        mark=time.perf_counter()
        checkpoint_diagnostic=trainer.checkpoint(out/'checkpoint.pt',time.perf_counter()-started,entrant_binding=expected,entrant_state={'optimizer_steps':steps,'losses':losses})
        final_update=trainer.update;final_samples=trainer.samples
        export(trainer,source,out)
        changed=sum(float((v.detach().cpu()-initial[k]).abs().sum()) for k,v in trainer.model.state_dict().items())
        phases['checkpoint_export']+=time.perf_counter()-mark
        counters=trainer.counters.cpu().tolist();buckets=trainer.bucket_usage.cpu().tolist()
        peak=torch.cuda.max_memory_allocated() if trainer.gpu else 0
        mark=time.perf_counter();trainer.close();trainer=None;phases['cleanup']=time.perf_counter()-mark
        result={'worker_uid':os.getuid(),'training_route':'native-fast-v1','updates':final_update,'samples':final_samples,'new_learner_decisions':final_samples-initial_samples,'natural_games':counters[0],'optimizer_steps':steps,'finite_losses':bool(losses) and bool(np.isfinite(losses).all()),'finite_gradients':bool(losses) and all(np.isfinite(v[4]) for v in losses),'loss_summary':{'last':losses[-1] if losses else None},'weight_l1_change':changed,'seconds':time.perf_counter()-started,'phases_seconds':phases,'counters':counters,'bucket_usage':buckets,'peak_cuda_bytes':peak,'checkpoint':'clash-native-ppo-v2','checkpoint_diagnostic':checkpoint_diagnostic,'binding':expected,'milestone_seconds':milestones_done,'counting':'new accepted selfplay learner seat decisions; excludes PPO epochs, ticks and inherited rows'}
        dump(out/'result.json',result);return result
    finally:
        if trainer is not None:trainer.close()

def export(trainer,source,out):
    import numpy as np
    temporary=out/'weights.tmp.npz'
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,**{k:v.detach().cpu().numpy().astype(np.float32) for k,v in trainer.model.state_dict().items()})
            stream.flush();os.fsync(stream.fileno())
        temporary.replace(out/'weights.npz')
    finally:
        if temporary.exists():temporary.unlink()
    for n in (*EDITABLE,'config.json'):shutil.copyfile(source/n,out/n)
