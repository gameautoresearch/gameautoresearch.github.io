"""Trained-policy action tape, matched simulator replay and optional Java Flight Recorder.

The tape contains public training-domain states only. Replays validate every retained
digest, masks, faults and resets. JFR instrumentation runs outside the timed trials.
"""
from pathlib import Path
import argparse,json,time,hashlib,resource
import numpy as np
import torch
from game import ROOT,HERE,make_vector,Session,dump,sha,ScriptedBot
from runner import setup,network,sample,Timer

def digest(v):return hashlib.sha256(v.observations.tobytes()+v.masks.tobytes()+v.infos.tobytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--envs',type=int,default=32);p.add_argument('--decisions',type=int,default=4096);p.add_argument('--repeats',type=int,default=3);p.add_argument('--jfr',action='store_true');p.add_argument('--threads',type=int,default=4);p.add_argument('--policy',choices=('trained','scripted'),default='trained')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    c=json.loads((HERE/'configs/profile-trimmed.json').read_text());c.update(envs=args.envs,threads=args.threads);c['hardware_allocation']['cpu_threads']=max(args.threads,c['hardware_allocation']['cpu_threads'])
    rng=setup(c);started=time.perf_counter()
    with make_vector(c) as v:
        session=Session(v,c,rng);initial=list(session.seeds)
        cp=ROOT/'evidence/clash-implementation/cuda-campaign/learning2/train/checkpoint.pt'
        saved=torch.load(cp,weights_only=False,map_location='cpu');v.check_metadata(saved['metadata'])
        model=network(v,c);model.load_state_dict(saved['model']);model.eval();timer=Timer('cpu');bot=ScriptedBot().bind(v)
        tape=[];resets=[];digests={};occupancy=[];late=0;active=0
        for t in range(args.decisions):
            occupancy.extend(v.observations[:,:,v.schema.header['entity_count'][0]].reshape(-1).tolist())
            late+=int((v.observations[:,:,v.schema.header['tick'][0]]>=2400).sum());active+=int((v.masks.sum(-1)>1).sum())
            actions=sample(model,v.observations.reshape(-1,v.schema.seat_ints),v.masks.reshape(-1,v.schema.actions),c,timer)[0].reshape(v.count,2) if args.policy=='trained' else np.array([[bot(o,m,rng) for o,m in zip(row,masks)] for row,masks in zip(v.observations,v.masks)],np.int32)
            if not v.masks[np.arange(v.count)[:,None],np.arange(2)[None,:],actions].all():raise RuntimeError('masked tape action')
            tape.append(actions.copy());st=session.step(actions)
            if st['fault'].any() or st['unsupported'].any() or st['illegal'].any():raise RuntimeError('invalid tape')
            done=session.reset_done();reset=np.full(v.count,-1,np.int32)
            for i in done:reset[i]=session.seeds[i]
            resets.append(reset)
            if (t+1)%128==0 or t+1==args.decisions:digests[str(t)]=digest(v)
        tape=np.array(tape,np.int32);resets=np.array(resets,np.int32)
        np.savez_compressed(out/'tape.npz',actions=tape,reset_seeds=resets,initial_seeds=np.array(initial))
        cold=time.perf_counter()-started;generation_counts=dict(session.counters)
        def replay(bookkeeping=False,recording=False):
            session.seeds=list(initial);session.histories=[[] for _ in initial];session.reset_all()
            engine=reset_time=0.;begin=time.perf_counter()
            for t,actions in enumerate(tape):
                before=time.perf_counter()
                if bookkeeping:session.step(actions)
                else:v.step(actions)
                engine+=time.perf_counter()-before
                before=time.perf_counter()
                for i in np.flatnonzero(resets[t]>=0):
                    v.reset_env(int(i),int(resets[t,i]));session.seeds[i]=int(resets[t,i]);session.histories[i]=[]
                reset_time+=time.perf_counter()-before
                if str(t) in digests:
                    if digest(v)!=digests[str(t)]:raise AssertionError(f'tape replay drift at {t}')
            elapsed=time.perf_counter()-begin
            return {'seconds':elapsed,'step_seconds':engine,'reset_seconds':reset_time,'environment_decisions_per_second':len(tape)*v.count/elapsed,'bookkeeping':bookkeeping,'instrumented':recording,'digest':digest(v)}
        warmup=replay();trials=[]
        for trial in range(args.repeats):
            for bookkeeping in ((False,True) if trial%2==0 else (True,False)):
                trials.append(dict(trial=trial,**replay(bookkeeping)))
        if args.jfr:
            import jpype
            recording=jpype.JClass('jdk.jfr.Recording')(jpype.JClass('jdk.jfr.Configuration').getConfiguration('profile'))
            recording.enable('jdk.ExecutionSample').withPeriod(jpype.JClass('java.time.Duration').ofMillis(2)).withStackTrace()
            recording.start();jfr_trial=replay(recording=True);recording.stop()
            recording.dump(jpype.JClass('java.nio.file.Paths').get(str((out/'simulator.jfr').resolve())));recording.close()
        else:jfr_trial=None
        dump(out/'result.json',{'config':c,'tape_sha256':sha(out/'tape.npz'),'checkpoint_sha256':sha(cp),'metadata':v.metadata,'decisions':args.decisions,'envs':v.count,'training_domain':c['train_seed_range'],'cold_generation_seconds':cold,'occupancy':{'mean':float(np.mean(occupancy)),'max':max(occupancy),'p95':float(np.percentile(occupancy,95)),'late_player_decisions':late,'active_player_decisions':active},'counts':generation_counts,'policy':args.policy,'warmup':warmup,'trials':trials,'jfr_trial':jfr_trial,'maxrss_platform_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'faults':v.fault_counts(),'digests':digests})

if __name__=='__main__':main()
