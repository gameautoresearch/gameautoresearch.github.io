"""CPU-only dense-vs-live-row/selected-action neural qualification.

Uses real native PUBLIC observations from the retained public development tape;
synthetic boundary/tie fixtures are neural inputs only, never game transitions.
No trainer/model edits, GPU/cloud launch, teacher, sampler or rule changes.
--timing additionally runs a small neural-only matched local CPU comparison
under evidence/local-performance.lock. This is not H100 or full-PPO evidence.
"""
from pathlib import Path
import argparse
import copy
import fcntl
import hashlib
import json
import os
import random
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def collect_public(out,data):
    import jpype
    import numpy as np
    import torch
    from cpu_batch import NativeCPUBatch
    from build_receipt import backend_build_binding
    assert not jpype.isJVMStarted()
    tape_path=ROOT/'evidence/clash-implementation/native-performance-20261108/cpu-baseline-1/generation/tape.npz'
    tape=np.load(tape_path);metadata=json.loads(tape_path.with_name('result.json').read_text())
    assert sha(tape_path)==metadata['tape_sha256']
    env=NativeCPUBatch(tape['initial_seeds'],threads=1)
    raw=[];masks=[];stun=None;busiest=None;maximum=-1;s=data['schema'];buff=data['buff_names'].index('ZapFreeze')
    try:
        for step,actions in enumerate(tape['actions']):
            env.step(actions)
            flat=env.observations.reshape(-1,7024);mask=env.masks.reshape(-1,2307)
            counts=flat[:,s['header']['entity_count'][0]]
            if int(counts.max())>maximum:
                maximum=int(counts.max());index=int(counts.argmax());busiest=(flat[index].copy(),mask[index].copy())
            if stun is None:
                entities=flat[:,s['header_ints']:].reshape(-1,s['max_entities'],s['entity_ints'])
                hit=(entities[...,[s['entity'][f'buff_{i}'] for i in range(data['metadata']['schema']['buff_slots'])]]==buff).any(-1).any(-1)
                if hit.any():
                    index=int(np.flatnonzero(hit)[0]);stun=(flat[index].copy(),mask[index].copy())
            if step%128==0:raw.append(flat.copy());masks.append(mask.copy())
            selected=tape['reset_seeds'][step]>=0
            if selected.any():env.reset(np.where(selected,tape['reset_seeds'][step],0),selected)
        assert stun is not None and busiest is not None
        raw=torch.from_numpy(np.concatenate(raw));masks=torch.from_numpy(np.concatenate(masks))
        value={'obs':raw,'mask':masks,'stun_obs':torch.from_numpy(stun[0]),'stun_mask':torch.from_numpy(stun[1]),
               'busiest_obs':torch.from_numpy(busiest[0]),'busiest_mask':torch.from_numpy(busiest[1]),
               'tape_sha256':sha(tape_path),'engine_build':backend_build_binding(env.library),'max_public_entities':maximum}
        torch.save(value,out/'public-fixture.pt');return value
    finally:env.close()


def fixtures(public,s):
    import torch
    raw,masks=public['obs'],public['mask'];active=(masks.sum(-1)>1)
    a=torch.nonzero(active).flatten();f=torch.nonzero(~active).flatten();assert a.numel() and f.numel()
    ix=torch.cat((a[:8],f[:8]));base=raw[ix].clone();mask=masks[ix].clone()
    yield 'real',base,mask,False
    for label,population in [('empty',0),('full200',200),('repeated',17),('tied_max',11)]:
        obs=base.clone();rows=obs[:,s.header_ints:].reshape(-1,s.max_entities,s.entity_ints)
        template=public['busiest_obs'][s.header_ints:].reshape(s.max_entities,s.entity_ints)[0].clone()
        rows.zero_()
        if population:rows[:,:population].copy_(template.expand(len(rows),population,-1))
        # Tied outputs with distinct first-row inputs make first-index max
        # backpropagation observable after zeroing entity_net's first weights.
        if label=='tied_max':
            rows[:,:population,s.entity['x']]=torch.arange(population,dtype=torch.int32)*37
            rows[:,:population,s.entity['hp']]=torch.arange(population,dtype=torch.int32)*23+100
        obs[:,s.header['entity_count'][0]]=population
        yield label,obs,mask,label=='tied_max'
    stun=public['stun_obs'].expand(8,-1).clone();sm=public['stun_mask'].expand(8,-1).clone()
    yield 'real_stun',stun,sm,False
    forced=mask.clone().zero_();forced[:,0]=1
    yield 'forced_after_momentum',base,forced,False
    for n in (1,3,7):yield 'tail_'+str(n),base[:n],mask[:n],False


def qualify(out,public,data,args):
    import numpy as np
    import torch
    from train import schema
    from model import Policy
    from gpu_features import DeviceBatchEncoder
    from cpu_neural import CpuNeuralAdapter,normalize_advantages,ppo_objective
    s=schema(data);encoder=DeviceBatchEncoder(s,copy=True)
    # Dense/ragged GEMM batch shapes and trim sum layouts may change FP32 ulps.
    # These bounds qualify numerical equivalence, never claim bitwise dense identity.
    limits={'output_atol':2e-6,'output_rtol':2e-5,'gradient_atol':3e-6,'gradient_rtol':3e-5,
            'parameter_atol':3e-6,'parameter_rtol':3e-5,'adam_atol':3e-6,'adam_rtol':3e-5}
    worst={};reports=[];numeric_failures=[];resume_reports=[]
    def check(a,b,group,name):
        error=float((a.detach()-b.detach()).abs().max()) if a.numel() else 0.
        worst[group]=max(worst.get(group,0.),error)
        try:torch.testing.assert_close(a,b,atol=limits[group+'_atol'],rtol=limits[group+'_rtol'],msg=name)
        except AssertionError:
            failure={'name':name,'group':group,'maximum_absolute_error':error,
                     'relative_l2_error':float(torch.linalg.vector_norm((a.detach()-b.detach()).float())/torch.linalg.vector_norm(b.detach().float()).clamp_min(1e-20))}
            if a.numel()<=8:failure.update(actual=a.detach().tolist(),expected=b.detach().tolist())
            numeric_failures.append(failure);dump(out/'numeric-failures.json',numeric_failures)
            if not args.audit_only:raise
    torch.manual_seed(8110831)
    initial=Policy(s,len(data['vocabulary']),len(data['buff_names']),args.width,args.embedding)
    initial.device_indices=True
    # Optimizers exist before adapter construction; object identity is preserved.
    for active_weight in (False,True):
        dense=copy.deepcopy(initial);candidate=copy.deepcopy(initial)
        od=torch.optim.Adam(dense.parameters(),lr=.0003,eps=1e-5,foreach=False)
        oc=torch.optim.Adam(candidate.parameters(),lr=.0003,eps=1e-5,foreach=False)
        ids=tuple(map(id,candidate.parameters()));keys=tuple(candidate.state_dict());optimizer_id=id(oc)
        numpy_state=np.random.get_state();python_state=random.getstate();torch_state=torch.get_rng_state().clone()
        adapter=CpuNeuralAdapter(candidate,trim_entities=not args.no_trim)
        assert tuple(map(id,candidate.parameters()))==ids and tuple(candidate.state_dict())==keys and id(oc)==optimizer_id
        assert torch.equal(torch.get_rng_state(),torch_state) and random.getstate()==python_state
        assert np.array_equal(np.random.get_state()[1],numpy_state[1])
        for index,(name,raw,mask,tied) in enumerate(fixtures(public,s)):
            if args.same_start:
                # Independent one-update equivalence from IDENTICAL weights
                # and existing moments. Copy values into existing objects;
                # never replace optimizer/parameter/state tensor identities.
                with torch.no_grad():
                    for pd,pc in zip(dense.parameters(),candidate.parameters()):
                        pc.copy_(pd)
                        for key,value in od.state.get(pd,{}).items():
                            if isinstance(value,torch.Tensor):oc.state[pc][key].copy_(value)
                            else:oc.state[pc][key]=value
            if tied:
                with torch.no_grad():dense.entity_net[0].weight.zero_();candidate.entity_net[0].weight.zero_()
            batch=encoder(raw,mask);snapshot={key:value.clone() for key,value in batch.items()}
            with torch.no_grad():
                dd,vd=dense.distribution(batch);dc,vc=adapter.distribution(batch)
                # Use the original policy sampler, so legal play/tile gradients
                # are exercised rather than mostly initial-prior wait argmaxes.
                actions=dd.sample()
                check(dc.log_prob(actions),dd.log_prob(actions),'output',name+' actor logp')
                check(dc.entropy(),dd.entropy(),'output',name+' actor entropy');check(vc,vd,'output',name+' actor value')
                # Same full-N categorical draw, RNG state and seeded replay.
                rng=torch.get_rng_state();sd=dense.sample_batch(batch);after=torch.get_rng_state();torch.set_rng_state(rng)
                sc=adapter.sample_batch(batch);assert torch.equal(sd[0],sc[0]),name+' actor actions'
                assert torch.equal(after,torch.get_rng_state()),name+' actor RNG'
                check(sd[1],sc[1],'output',name+' sample logp');check(sd[2],sc[2],'output',name+' sample values')
                old=dd.log_prob(actions)+torch.linspace(-.4,.4,len(actions))
                returns=vd+torch.linspace(-.5,.5,len(actions))
            weight=(mask.sum(-1)>1).float() if active_weight else torch.ones(len(actions))
            advantage=normalize_advantages(torch.linspace(-1.,1.,len(actions)),weight)
            od.zero_grad(set_to_none=True);oc.zero_grad(set_to_none=True)
            distribution,value=dense.distribution(batch)
            ld,md=ppo_objective(distribution.log_prob(actions),distribution.entropy(),value,actions,old,advantage,returns,weight)
            lc,mc=adapter.ppo_loss(batch,actions,old,advantage,returns,weight)
            check(lc,ld,'output',name+' loss')
            for key in ('policy_loss','value_loss','entropy','kl','ratio','logp','values','policy_denominator'):
                check(mc[key],md[key],'output',name+' '+key)
            assert mc['value_rows']==mc['kl_rows']==len(actions)
            ld.backward();lc.backward();head_zero=True
            for (key,pd),(other,pc) in zip(dense.named_parameters(),candidate.named_parameters()):
                assert key==other and pd.grad is not None and pc.grad is not None,name+' missing gradient '+key
                check(pc.grad,pd.grad,'gradient',name+' '+key)
                if name=='forced_after_momentum' and (key=='tiles' or key.startswith(('slot.','hand_query.'))):
                    assert torch.count_nonzero(pc.grad)==0,name+' forced head gradient';head_zero &= pc.grad is not None
            gd=torch.nn.utils.clip_grad_norm_(dense.parameters(),.5);gc=torch.nn.utils.clip_grad_norm_(candidate.parameters(),.5)
            check(gc,gd,'gradient',name+' clipped norm')
            before=candidate.tiles.detach().clone()
            if name=='forced_after_momentum':assert torch.count_nonzero(oc.state[candidate.tiles]['exp_avg'])>0
            od.step();oc.step()
            if name=='forced_after_momentum':assert not torch.equal(before,candidate.tiles),'Adam momentum was skipped'
            for (key,pd),(other,pc) in zip(dense.named_parameters(),candidate.named_parameters()):
                check(pc,pd,'parameter',name+' updated '+key)
                for state_key in od.state[pd]:
                    actual,expected=oc.state[pc][state_key],od.state[pd][state_key]
                    if isinstance(actual,torch.Tensor):check(actual,expected,'adam',name+' '+key+' '+state_key)
                    else:assert actual==expected
            for key,value in snapshot.items():assert torch.equal(batch[key],value),'public batch was mutated'
            assert tuple(map(id,candidate.parameters()))==ids and tuple(candidate.state_dict())==keys and id(oc)==optimizer_id
            prepared=adapter.prepare(batch)
            expected_rows=s.max_entities if args.no_trim else max(1,int(raw[:,s.header['entity_count'][0]].max()))
            assert prepared['entities'].shape[1]==expected_rows
            reports.append({'fixture':name,'active_policy_weight':active_weight,'rows':len(raw),
                            'entity_rows_retained':prepared['entities'].shape[1],
                            'active_rows':int((mask.sum(-1)>1).sum()),'selected_play_actions':int((actions>0).sum()),
                            'zero_not_None_head_gradient':head_zero})
        # Corrupt weights must fail even when empty entities/all heads are skipped.
        guard_batch={key:value.clone() for key,value in batch.items()}
        guard_batch['header'][:,s.header['entity_count'][0]]=0
        guard_batch['entities'].zero_();guard_batch['mask'].zero_();guard_batch['mask'][:,0]=True
        guard_actions=torch.zeros(len(guard_batch['header']),dtype=torch.long)
        with torch.no_grad():saved=candidate.tiles[0,0].clone();candidate.tiles[0,0]=float('nan')
        try:adapter.action_terms(guard_batch,guard_actions)
        except ValueError as exc:assert 'nonfinite' in str(exc)
        else:raise AssertionError('corrupt unused policy parameter was concealed')
        with torch.no_grad():candidate.tiles[0,0]=saved
        with torch.no_grad():saved=candidate.entity_net[0].weight[0,0].clone();candidate.entity_net[0].weight[0,0]=float('nan')
        try:adapter.action_terms(guard_batch,guard_actions)
        except ValueError as exc:assert 'nonfinite' in str(exc)
        else:raise AssertionError('corrupt unused entity parameter was concealed')
        with torch.no_grad():candidate.entity_net[0].weight[0,0]=saved
        for count in (-1,201):
            bad={key:value.clone() for key,value in batch.items()};bad['header'][:,s.header['entity_count'][0]]=count
            try:adapter.prepare(bad)
            except ValueError:pass
            else:raise AssertionError('invalid entity capacity was concealed')
        invalid=guard_actions.clone();invalid[0]=1
        try:adapter.action_terms(guard_batch,invalid)
        except ValueError:pass
        else:raise AssertionError('illegal forced-row action was concealed')
        # Exact checkpoint replay INSIDE this mode, including the original
        # full-N actor sample, every clipped gradient, Adam and all parameters.
        live=torch.nonzero(public['mask'].sum(-1)>1).flatten()[:8]
        forced=torch.nonzero(public['mask'].sum(-1)==1).flatten()[:8]
        resume_ix=torch.cat((live,forced));raw,mask=public['obs'][resume_ix],public['mask'][resume_ix]
        replay_batch=encoder(raw,mask)
        from train import binding
        checkpoint={'format':'clash-cpu-neural-qualification-v1','binding':binding(HERE/'generated-v2/task-data.json'),
                    'width':args.width,'embedding':args.embedding,'model':copy.deepcopy(candidate.state_dict()),
                    'optimizer':copy.deepcopy(oc.state_dict()),'torch_rng':torch.get_rng_state().clone(),
                    'adapter':adapter.metadata()}
        torch.save(checkpoint,out/('within-mode-'+('active' if active_weight else 'allrow')+'.pt'))
        assert checkpoint['binding']==binding(HERE/'generated-v2/task-data.json')
        restored=copy.deepcopy(candidate);restored.load_state_dict(checkpoint['model'])
        restored_optimizer=torch.optim.Adam(restored.parameters(),lr=.0003,eps=1e-5,foreach=False)
        restored_optimizer.load_state_dict(checkpoint['optimizer'])
        restored_adapter=CpuNeuralAdapter(restored,trim_entities=not args.no_trim)
        def advance(model,optimizer,api):
            with torch.no_grad():sample=tuple(value.clone() for value in api.sample_batch(replay_batch))
            w=(mask.sum(-1)>1).float() if active_weight else torch.ones(len(mask))
            a=normalize_advantages(torch.linspace(-1,1,len(mask)),w)
            optimizer.zero_grad(set_to_none=True)
            loss,_=api.ppo_loss(replay_batch,sample[0],sample[1],a,sample[2]+.1,w)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
            gradients=[p.grad.clone() for p in model.parameters()];optimizer.step()
            return sample,gradients
        expected=advance(candidate,oc,adapter);expected_rng=torch.get_rng_state().clone()
        torch.set_rng_state(checkpoint['torch_rng']);actual=advance(restored,restored_optimizer,restored_adapter)
        assert torch.equal(torch.get_rng_state(),expected_rng)
        for aa,bb in zip(actual[0],expected[0]):assert torch.equal(aa,bb),'within-mode sampled outputs differ'
        for aa,bb in zip(actual[1],expected[1]):assert torch.equal(aa,bb),'within-mode gradients differ'
        for (name,aa),(other,bb) in zip(candidate.state_dict().items(),restored.state_dict().items()):
            assert name==other and torch.equal(aa,bb),'within-mode model state differs'
        for pd,pc in zip(candidate.parameters(),restored.parameters()):
            for key in oc.state[pd]:
                aa,bb=oc.state[pd][key],restored_optimizer.state[pc][key]
                assert torch.equal(aa,bb) if isinstance(aa,torch.Tensor) else aa==bb
        restored_adapter.close();resume_reports.append({'active_policy_weight':active_weight,'status':'passed',
                'actor_RNG_outputs_gradients_model_and_Adam_bitwise_exact':True})
        adapter.close();assert 'ragged_entities' not in candidate.__dict__ and 'sparse_policy_head' not in candidate.__dict__
    assert any(row['fixture']=='real' and row['selected_play_actions']>0 for row in reports)
    result={'status':'numeric_audit_only' if numeric_failures else 'passed','fixtures':reports,'limits':limits,'maximum_absolute_errors':worst,
            'numeric_failures':numeric_failures,'trim_entities':not args.no_trim,
            'same_start_per_fixture':args.same_start,'within_mode_exact_resume':resume_reports,
            'full_N_values_policy_denominators_and_KL':True,'zero_not_None_gradients_and_existing_Adam_momentum':True,
            'parameter_optimizer_and_state_dict_identity_preserved':True,'actor_actions_and_RNG_seeded_replay_exact':True,
            'no_public_input_mutation':True,'nonfinite_and_overflow_rejected':True,
            'dense_bitwise_equivalence_claimed':False,'scope':'FP32 numerical dense-reference equivalence; exact resume in-mode separately tested below'}
    dump(out/'qualification.json',result);return result,initial


def timing(out,public,data,initial,args):
    import torch
    from cpu_neural import CpuNeuralAdapter,normalize_advantages,ppo_objective
    from gpu_features import DeviceBatchEncoder
    from train import schema
    s=schema(data);ix=torch.linspace(0,len(public['obs'])-1,args.batch_size).long()
    batch=DeviceBatchEncoder(s,copy=True)(public['obs'][ix],public['mask'][ix])
    weight=(batch['mask'].sum(-1)>1).float()
    advantage=normalize_advantages(torch.linspace(-1,1,len(ix)),weight)
    with torch.no_grad():d,v=initial.distribution(batch);actions=d.sample();old=d.log_prob(actions);returns=v+.1
    trials=[];lock_path=ROOT/'evidence/local-performance.lock'
    actor_rng=torch.get_rng_state().clone()
    with lock_path.open('a') as lock:
        print('waiting for shared performance lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
        print('CPU neural-only timing lock acquired',flush=True)
        for repeat in range(3):
            for mode in (('dense','candidate') if repeat%2==0 else ('candidate','dense')):
                model=copy.deepcopy(initial);optimizer=torch.optim.Adam(model.parameters(),lr=.0003,eps=1e-5,foreach=False)
                adapter=CpuNeuralAdapter(model) if mode=='candidate' else None
                torch.set_rng_state(actor_rng)
                def actor():
                    with torch.no_grad():
                        return (adapter.sample_batch(batch) if adapter else model.sample_batch(batch))
                def update():
                    optimizer.zero_grad(set_to_none=True)
                    if adapter:loss,_=adapter.ppo_loss(batch,actions,old,advantage,returns,weight)
                    else:
                        d,value=model.distribution(batch)
                        loss,_=ppo_objective(d.log_prob(actions),d.entropy(),value,actions,old,advantage,returns,weight)
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
                for _ in range(3):actor();update()
                start=time.perf_counter()
                for _ in range(10):actor()
                actor_seconds=(time.perf_counter()-start)/10
                start=time.perf_counter()
                for _ in range(10):update()
                learner_seconds=(time.perf_counter()-start)/10
                trials.append({'repeat':repeat+1,'mode':mode,'actor_seconds':actor_seconds,'PPO_update_seconds':learner_seconds})
                if adapter:adapter.close()
        medians={mode:{key:statistics.median(t[key] for t in trials if t['mode']==mode) for key in ('actor_seconds','PPO_update_seconds')} for mode in ('dense','candidate')}
    result={'status':'passed','scope':'local Mac CPU neural component only; no simulator/GAE/full trainer/GPU/H100/allocation efficiency claim',
            'torch_threads':torch.get_num_threads(),'batch_size':len(ix),'width':args.width,'embedding':args.embedding,
            'mean_live_entities':float(batch['header'][:,s.header['entity_count'][0]].float().mean()),
            'max_live_entities':int(batch['header'][:,s.header['entity_count'][0]].max()),'active_fraction':float(weight.mean()),
            'iterations':10,'warmups':3,'reverse_order_repeats':3,'trials':trials,'medians':medians,
            'actor_speedup':medians['dense']['actor_seconds']/medians['candidate']['actor_seconds'],
            'PPO_update_speedup':medians['dense']['PPO_update_seconds']/medians['candidate']['PPO_update_seconds'],
            'finite_guard_included':True,'lock_path':str(lock_path),'OS_affinity_enforced':False}
    dump(out/'timing.json',result);return result


def main(args):
    import jpype
    import torch
    from train import binding
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(args.threads)
    if torch.cuda.is_initialized():raise RuntimeError('CPU qualification requires no initialized CUDA context')
    assert not jpype.isJVMStarted()
    data_path=HERE/'generated-v2/task-data.json';data=json.loads(data_path.read_text());before=binding(data_path)
    public=collect_public(args.out,data)
    qualification,initial=qualify(args.out,public,data,args)
    result={'status':qualification['status'],'qualification':qualification,'public_fixture_sha256':sha(args.out/'public-fixture.pt'),
            'maximum_public_entities_collected':public['max_public_entities'],'source_binding':before,
            'new_files_only':True,'train_model_edited':False,'gpu_executed':False,'JVM_started':False,
            'full_actual_resource_CPU_vs_H100_comparison_required':True}
    if args.timing:
        if qualification['status']!='passed':raise ValueError('timing requires passing numeric qualification')
        result['timing']=timing(args.out,public,data,initial,args)
    assert binding(data_path)==before,'source changed during CPU neural qualification'
    assert not jpype.isJVMStarted() and not torch.cuda.is_initialized()
    dump(args.out/'result.json',result)
    print(json.dumps({'status':result['status'],'maximum_absolute_errors':qualification['maximum_absolute_errors'],
                      'timing':result.get('timing')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--timing',action='store_true');parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--width',type=int,default=64);parser.add_argument('--embedding',type=int,default=12)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--no-trim',action='store_true',help='retain all200 padded rows; diagnose trim reduction rounding separately')
    parser.add_argument('--audit-only',action='store_true',help='record numeric failures without treating them as qualification')
    parser.add_argument('--same-start',action='store_true',help='one-update checks from identical weights/moments; sequential drift remains a separate audit')
    args=parser.parse_args();main(args)
