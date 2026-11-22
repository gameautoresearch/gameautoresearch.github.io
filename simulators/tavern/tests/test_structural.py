"""Structural changes preserve gradients, legal actions, seed domains and RNG state."""
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import torch
import taverngym as tg

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import Policy, act
from rollout import Timer, collect
from run import ppo_update


def test_preencoded_learner_exact_weights_optimizer_and_metrics():
    torch.set_num_threads(1)
    c=json.loads((Path(__file__).resolve().parents[1]/'recipes/starter-local.json').read_text())
    c.update(n_games=2,native_threads=1,fast_inference=True,defer_learner_metrics=True)
    torch.manual_seed(512)
    policy=Policy(hidden=16,embedding=2)
    eps,*_=collect(policy,c,12345,torch.device('cpu'),Timer('cpu'),time.perf_counter()+30)
    outputs=[]
    for encoded in [False,True]:
        model=copy.deepcopy(policy)
        optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],eps=1e-5)
        torch.manual_seed(111)
        losses=ppo_update(model,optimizer,eps,dict(c,preencode_learner=encoded),torch.device('cpu'))
        outputs.append((copy.deepcopy(model.state_dict()),optimizer.state_dict(),losses))
    assert all(torch.equal(outputs[0][0][k],outputs[1][0][k]) for k in outputs[0][0])
    assert outputs[0][2]==outputs[1][2]
    for k,v in outputs[0][1]['state'].items():
        assert all(torch.equal(x,outputs[1][1]['state'][k][n]) for n,x in v.items())


def test_dynamic_scheduler_replay_and_faults(monkeypatch):
    envs=[]
    for scheduler in ['static','dynamic']:
        monkeypatch.setenv('TAVERNGYM_SCHEDULER',scheduler)
        envs.append(tg.BatchEnv(9,seed=721,controlled=[0,3],threads=4))
    rng=np.random.default_rng(45)
    for step in range(220):
        active=np.array([step%5!=i%5 for i in range(9)])
        x,y=[e.observe(active_games=active) for e in envs]
        assert all(np.array_equal(x[k],y[k]) for k in x)
        actions=x['action_mask'].argmax(-1)
        for i in np.flatnonzero(x['ready']):
            legal=np.flatnonzero(x['action_mask'][i])
            actions[i]=0 if 0 in legal and rng.random()<.4 else rng.choice(legal)
        a,b=[e.step(actions,active_games=active) for e in envs]
        assert all(np.array_equal(u,v) for u,v in zip(a[:2],b[:2]))
        assert all(np.array_equal(a[2][k],b[2][k]) for k in a[2])
    assert all(envs[0].save(i)==envs[1].save(i) for i in range(9))
    for e in envs:
        state=json.loads(e.save(3));state['game']['truncated_combats']=1
        e.load(3,json.dumps(state))
        rw,dn,info=e.step(np.zeros(18,np.int64))
        assert dn[6:8].all() and (info['stop_reason'][6:8]==5).all()
        assert not rw[6:8].any() and not info['placement'][6:8].any()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA required')
def test_graph_sampling_exact_outputs_rng_and_weight_updates():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(31);policy=Policy(hidden=32,embedding=2).cuda().eval()
    policy.fast_inference=True;policy.validated_sampler=True;policy.inference_metrics={}
    from collections import Counter
    policy.inference_metrics=Counter()
    obs=tg.BatchEnv(7,seed=921).observe()
    for deterministic in [False,True]:
        for n in [7,3,1,7,3]:
            # Reuse captured graphs after actual weights change.
            with torch.no_grad(): policy.critic.bias.add_(.01)
            results=[]
            for graph in [False,True]:
                policy.graph_inference=graph
                torch.cuda.manual_seed_all(131+n)
                r=act(policy,obs,np.arange(n),torch.device('cuda:0'),deterministic)
                results.append((r,torch.cuda.get_rng_state()))
            assert all(np.array_equal(a,b) for a,b in zip(results[0][0],results[1][0]))
            assert torch.equal(results[0][1],results[1][1])
    assert len(policy._actor_graphs)==6


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA required')
def test_graph_learner_exact_weights_optimizer_and_remainder():
    torch.set_num_threads(1)
    c=json.loads((Path(__file__).resolve().parents[1]/'recipes/starter-local.json').read_text())
    c.update(n_games=2,native_threads=1,fast_inference=True,defer_learner_metrics=True,
             preencode_learner=True,minibatch_size=64,ppo_epochs=3)
    torch.manual_seed(512)
    policy=Policy(hidden=32,embedding=2)
    eps,*_=collect(policy,c,12345,torch.device('cpu'),Timer('cpu'),time.perf_counter()+30)
    outputs=[]
    for graph in [False,True]:
        model=copy.deepcopy(policy).cuda();model.fast_inference=True
        optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],eps=1e-5)
        for update in range(2):
            torch.cuda.manual_seed_all(111+update)
            losses=ppo_update(model,optimizer,eps,dict(c,graph_learner=graph),torch.device('cuda:0'))
        outputs.append((copy.deepcopy(model.state_dict()),optimizer.state_dict(),losses))
    assert all(torch.equal(outputs[0][0][k],outputs[1][0][k]) for k in outputs[0][0])
    assert outputs[0][2]==outputs[1][2]
    for k,v in outputs[0][1]['state'].items():
        assert all(torch.equal(x,outputs[1][1]['state'][k][n]) for n,x in v.items())


def test_actor_copy_preserves_cpu_rng_and_reuses_policy():
    from run import actor_copy
    # Meta is deliberately unsupported by the CLI; CPU no-copy path is also exact.
    torch.manual_seed(422)
    model=Policy(hidden=16,embedding=2)
    rng=torch.get_rng_state().clone()
    assert actor_copy(model,torch.device('cpu'),Timer('cpu')) is model
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA required')
def test_cpu_actor_copy_preserves_rng_and_weights():
    from run import actor_copy
    torch.manual_seed(422)
    model=Policy(hidden=16,embedding=2).cuda()
    cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
    actor=actor_copy(model,torch.device('cpu'),Timer('cuda:0'))
    assert all(torch.equal(v.cpu(),actor.state_dict()[k]) for k,v in model.state_dict().items())
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
    assert actor_copy(model,torch.device('cpu'),Timer('cuda:0'),actor) is actor


@pytest.mark.skipif(not Path('/proc/self/task').exists(),reason='Linux native thread inventory')
def test_idle_native_pool_cache_is_bounded():
    import subprocess
    code="""
import gc, os
from pathlib import Path
import taverngym as tg
os.environ['TAVERNGYM_WORKERS']='pooled'
os.environ['TAVERNGYM_SCHEDULER']='dynamic'
def count(): return len(list(Path('/proc/self/task').iterdir()))
before=count()
for n in [2,3,4,5,6,7,8]:
    e=tg.BatchEnv(2,seed=12,threads=n)
    del e
    gc.collect()
# Exactly the last-size idle pool may stay resident, rather than 2+...+8 workers.
import time
time.sleep(.1)
assert count() <= before+8, (before,count())
"""
    subprocess.run([sys.executable,'-c',code],check=True)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA required')
def test_graph_cache_limit_falls_back_without_changing_rng():
    from collections import Counter
    torch.manual_seed(19);model=Policy(hidden=32,embedding=2).cuda().eval()
    model.fast_inference=True;model.validated_sampler=True
    model.inference_metrics=Counter();model.actor_graph_max_shapes=1
    obs=tg.BatchEnv(7,seed=921).observe()
    for n in [7,3,1,7]:
        values=[]
        for graph in [False,True]:
            model.graph_inference=graph;torch.cuda.manual_seed_all(555)
            values.append((act(model,obs,np.arange(n),torch.device('cuda:0')),torch.cuda.get_rng_state()))
        assert all(np.array_equal(x,y) for x,y in zip(values[0][0],values[1][0]))
        assert torch.equal(values[0][1],values[1][1])
    assert len(model._actor_graphs)==1


def test_migration_refuses_dependency_drift(tmp_path,monkeypatch):
    import migrate_performance_checkpoint as migration
    # A current-source unit fixture keeps this guard test independent of private
    # historical policies, which must remain bound to their original engine.
    source=tmp_path/'current-source-fixture.pt'
    engine=migration.ENGINE
    files=[*engine.joinpath('crates').rglob('*.rs'),engine/'crates/taverngym/data/cards.json',
           engine/'Cargo.toml',engine/'Cargo.lock',engine/'rust-toolchain.toml',engine/'pyproject.toml']
    manifest={str(p.relative_to(migration.ROOT)):migration.sha(p) for p in files}
    fixture={'native_source_hash':tg.build_info()['source_hash'],'manifest_source_sha256':manifest,
             'policy':{'test_weight':torch.tensor([1.,2.])}}
    torch.save(fixture,source)
    original=migration.sha
    migration.migrate(source,tmp_path/'verified.pt')
    copied=torch.load(tmp_path/'verified.pt',map_location='cpu',weights_only=False)
    assert torch.equal(copied['policy']['test_weight'],fixture['policy']['test_weight'])
    monkeypatch.setattr(migration,'sha',lambda p:'changed' if str(p).endswith('/Cargo.lock') else original(p))
    with pytest.raises(RuntimeError,match='mechanics/data drift:.*Cargo.lock'):
        migration.migrate(source,tmp_path/'must-not-exist.pt')
    assert not (tmp_path/'must-not-exist.pt').exists()
    monkeypatch.setattr(migration,'sha',original)
    key='candidates/AutoTavern/crates/taverngym/src/actions.rs'
    fixture['manifest_source_sha256'][key]='different historical mechanics'
    torch.save(fixture,tmp_path/'old-source-fixture.pt')
    with pytest.raises(RuntimeError,match='mechanics/data drift:.*actions.rs'):
        migration.migrate(tmp_path/'old-source-fixture.pt',tmp_path/'refused-old-source.pt')
    assert not (tmp_path/'refused-old-source.pt').exists()


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_device_rollouts_exact_collection_and_ppo(device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('actual CUDA required')
    c=json.loads((Path(__file__).resolve().parents[1]/'recipes/starter-local.json').read_text())
    c.update(n_games=3,native_threads=2,torch_threads=1,fast_inference=True,
        skip_finished_lobbies=True,defer_learner_metrics=True,preencode_learner=True,
        learning_seats=[0,3],opponent='selfplay')
    outputs=[]
    for retained in [False,True]:
        torch.manual_seed(223)
        if device.startswith('cuda'):torch.cuda.manual_seed_all(223)
        policy=Policy(hidden=32,embedding=2).to(device)
        cc=dict(c,device_rollouts=retained)
        eps,rows,counts,states=collect(policy,cc,775,torch.device(device),Timer(device),time.perf_counter()+60)
        optimizer=torch.optim.Adam(policy.parameters(),lr=c['learning_rate'],eps=1e-5)
        loss=ppo_update(policy,optimizer,eps,cc,torch.device(device))
        outputs.append((copy.deepcopy(policy.state_dict()),rows,counts,states,loss))
    assert all(torch.equal(outputs[0][0][k],outputs[1][0][k]) for k in outputs[0][0])
    assert outputs[0][1:]==outputs[1][1:]


def test_device_rollout_gather_excludes_faulted_rows_and_preserves_order():
    from device_rollout import DeviceEpisodes
    chunks=[(torch.tensor([[1.],[2.]]),torch.tensor([[10],[20]]),torch.tensor([[True],[False]])),
            (torch.tensor([[3.],[4.]]),torch.tensor([[30],[40]]),torch.tensor([[False],[True]]))]
    # Row 1 belongs to a faulted lobby. Slot order gathers 2,0,3, not actor-call order.
    rows=[{'_device_row':i} for i in [2,0,3]]
    episodes=DeviceEpisodes([rows],chunks)
    f,c,m=episodes.pack(rows,torch.device('cpu'))
    assert f[:,0].tolist()==[3.,1.,4.] and c[:,0].tolist()==[30,10,40]
    assert m[:,0].tolist()==[False,True,True] and not episodes.chunks


def test_native_warmstart_exact_scripted_driver(monkeypatch):
    monkeypatch.setenv('TAVERNGYM_SCHEDULER','dynamic')
    a=tg.BatchEnv(5,seed=12345,controlled=[0],threads=4)
    b=tg.BatchEnv(5,seed=12345,controlled=[0],threads=4)
    a.warm_start(8)
    for i in range(5):
        game=b.game(i)
        for _ in range(1000):
            if game.turn>=8 or game.is_over():break
            for p in range(8):
                if game.needs_action(p):game.bot_play(p)
                if game.turn>=8 or game.is_over():break
        else:pytest.fail('scripted driver guard')
        b.load(i,json.dumps({'game':json.loads(game.state_json()),'index':0,'finished':[False]}))
        assert a.save(i)==b.save(i)
    with pytest.raises((ValueError,RuntimeError)):a.warm_start(0)


def test_vector_bookkeeping_preserves_complete_trajectory_and_ppo():
    torch.set_num_threads(1)
    c=json.loads((Path(__file__).resolve().parents[1]/'recipes/starter-local.json').read_text())
    c.update(n_games=4,native_threads=2,fast_inference=True,skip_finished_lobbies=True,
             learning_seats=[0,3],opponent='selfplay')
    outputs=[]
    for vector in [False,True]:
        torch.manual_seed(999);model=Policy(hidden=16,embedding=2)
        cc=dict(c,vector_bookkeeping=vector)
        eps,records,stats,states=collect(model,cc,542,torch.device('cpu'),Timer('cpu'),time.perf_counter()+30)
        optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],eps=1e-5)
        losses=ppo_update(model,optimizer,eps,cc,torch.device('cpu'))
        outputs.append((copy.deepcopy(model.state_dict()),records,stats,states,losses))
    assert all(torch.equal(outputs[0][0][k],outputs[1][0][k]) for k in outputs[0][0])
    assert outputs[0][1:]==outputs[1][1:]


@pytest.mark.parametrize('warm',[1,8])
def test_compact_native_observer_matches_oracle_through_busy_games(warm):
    env=tg.BatchEnv(4,seed=771,controlled=range(8),threads=4,
        config={'turn_action_caps':[30],'max_actions_per_turn':30,'max_turns':100})
    if warm>1:env.warm_start(warm)
    rng=np.random.default_rng(832)
    saved=None
    for step in range(420):
        active=np.array([step%7!=g for g in range(4)])
        dense=env.observe(copy=False,active_games=active)
        compact=env.observe_ready(copy=False,active_games=active)
        slots=np.flatnonzero(dense['ready'])
        assert np.array_equal(compact['slots'],slots)
        for key in ['features','cards','action_mask']:
            assert np.array_equal(compact[key],dense[key][slots]), (warm,step,key)
            assert not compact[key].flags.writeable
        if saved is not None:
            assert all(np.array_equal(saved[0][k],saved[1][k]) for k in ['features','cards','action_mask','slots'])
        saved=(compact,{k:v.copy() for k,v in compact.items()})
        actions=dense['action_mask'].argmax(-1)
        for s in slots:
            legal=np.flatnonzero(dense['action_mask'][s])
            actions[s]=0 if 0 in legal and rng.random()<.4 else rng.choice(legal)
        env.step(actions,active_games=active)
    empty=env.observe_ready(active_games=[False]*4)
    assert empty['features'].shape==(0,tg.OBS_FLOATS) and empty['slots'].size==0
    state=json.loads(env.save(1));state['game']['truncated_combats']=1;state['finished']=[False]*8;env.load(1,json.dumps(state))
    assert not np.isin(env.observe_ready()['slots'],np.arange(8,16)).any()
    rw,dn,info=env.step(np.zeros(32,np.int64))
    assert dn[8:16].all() and (info['stop_reason'][8:16]==5).all() and not rw[8:16].any()


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_sparse_training_exact_ppo_and_rng(device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('actual CUDA required')
    c=json.loads((Path(__file__).resolve().parents[1]/'recipes/starter-local.json').read_text())
    c.update(n_games=3,native_threads=2,torch_threads=1,fast_inference=True,
        skip_finished_lobbies=True,defer_learner_metrics=True,preencode_learner=True,
        device_rollouts=True,learning_seats=[0,3],opponent='selfplay')
    values=[]
    for compact in [False,True]:
        torch.manual_seed(741)
        if device.startswith('cuda'):torch.cuda.manual_seed_all(741)
        model=Policy(hidden=32,embedding=2).to(device)
        cc=dict(c,sparse_observations=compact)
        eps,records,stats,states=collect(model,cc,921,torch.device(device),Timer(device),time.perf_counter()+60)
        optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],eps=1e-5)
        losses=ppo_update(model,optimizer,eps,cc,torch.device(device))
        values.append((copy.deepcopy(model.state_dict()),records,stats,states,losses,
            torch.get_rng_state(),torch.cuda.get_rng_state() if device.startswith('cuda') else None))
    assert all(torch.equal(values[0][0][k],values[1][0][k]) for k in values[0][0])
    assert values[0][1:5]==values[1][1:5]
    assert torch.equal(values[0][5],values[1][5])
    if device.startswith('cuda'):assert torch.equal(values[0][6],values[1][6])


def test_continuation_preprofile_eliminations_never_count_as_policy_episodes():
    torch.set_num_threads(1)
    c=json.loads((Path(__file__).resolve().parents[1]/'recipes/gpu-small.json').read_text())
    c.update(device='cpu',start_turn=8,native_warmstart=True,allow_profile_eliminated=True,
        sparse_observations=True,copy_observations=False,device_rollouts=True)
    torch.manual_seed(41);timer=Timer('cpu')
    eps,records,stats,_=collect(Policy(hidden=16,embedding=2),c,5012360,
        torch.device('cpu'),timer,time.perf_counter()+30)
    omitted=timer.context[0]['excluded_preprofile_eliminated_seats']
    assert omitted>0
    assert len(eps)==len(records)==stats['completed_learning_episodes']==32-omitted
    assert all(len(ep)>0 for ep in eps)
    assert all(r['valid'] and r['stop_reason']=='natural' for r in records)
    assert stats['invalid_lobbies']==0 and stats['rejected']==0


def test_compact_view_pin_refuses_changed_oracle(tmp_path,monkeypatch):
    import provenance
    original=provenance.sha
    monkeypatch.setattr(provenance,'sha',lambda p:'changed' if Path(p).name=='obs.rs' else original(p))
    c=json.loads((Path(__file__).resolve().parents[1]/'recipes/starter-local.json').read_text())
    c['sparse_observations']=True
    with pytest.raises(RuntimeError,match='oracle pin changed'):
        provenance.prepare(tmp_path/'refused',c,'profile',[],time.perf_counter())
    assert not (tmp_path/'refused').exists()
