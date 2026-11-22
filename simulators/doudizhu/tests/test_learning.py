from copy import deepcopy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from integrations.doudizhu.artifacts import HERE, Budget, load_checkpoint, sha256, verify_source
from integrations.doudizhu.environment import ROOT, ROLES, Game, deal, digest
from integrations.doudizhu.model import Policy, update
from integrations.doudizhu.rollout import Policies, assign_training, collect
from integrations.doudizhu.runner import make_optimizers
from integrations.doudizhu.workers import VectorGames


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)
    torch.manual_seed(4)


def settings():
    config = json.loads((HERE/'recipes/smoke.json').read_text())
    config.update(vector_games=4, architecture={'hidden':16,'depth':1,'history':8},
                  max_episodes=8, max_seconds=90, batch_size=128)
    return config


def test_pin():
    assert verify_source()['commit'].startswith('718a5c9')


def test_factored_q_matches_duplicated_and_upstream_network():
    # Import the pinned model file directly: upstream dmc/__init__.py eagerly
    # imports its legacy logger and GitPython even when only a network is needed.
    import importlib.util
    spec = importlib.util.spec_from_file_location('douzero_oracle_models', ROOT/'candidates/DouZero/douzero/dmc/models.py')
    oracle_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle_module)
    LandlordLstmModel, FarmerLstmModel = oracle_module.LandlordLstmModel, oracle_module.FarmerLstmModel
    # Full upstream model dimensions, exact weight key compatibility, all roles.
    policy = Policy()
    game = Game(deal('train', 6, 3))
    for role, reference_type in zip(ROLES, (LandlordLstmModel, FarmerLstmModel, FarmerLstmModel)):
        obs = game.observe()
        assert obs.role == role
        reference = reference_type()
        reference.load_state_dict(policy.roles[role].state_dict(), strict=True)
        x, z, actions = obs.encode()
        xx = np.concatenate((np.repeat(x[None], len(actions), axis=0), actions), axis=1)
        with torch.no_grad():
            expected = reference(torch.tensor(np.repeat(z[None], len(actions), axis=0), dtype=torch.float32),
                                 torch.tensor(xx,dtype=torch.float32), return_value=True)['values'].squeeze(-1).numpy()
        optimized = policy.values_many([obs], max_candidates=7)[0]
        duplicate = policy.values_many([obs], max_candidates=7, factored=False)[0]
        np.testing.assert_allclose(optimized, expected, atol=1e-7, rtol=1e-5)
        np.testing.assert_allclose(optimized, duplicate, atol=1e-7, rtol=1e-5)
        game.step(int(np.argmax(optimized)))


def test_variable_legal_batch_and_all_role_gradient_updates():
    config = settings()
    policy = Policy(config['architecture'])
    optimizers = make_optimizers(policy, config)
    policies = Policies(policy, 'cpu')
    cards = [deal('train', 73, i) for i in range(4)]
    assignment = [{r:{'kind':'learner'} for r in ROLES} for _ in cards]
    with VectorGames() as vector:
        result = collect(vector, policies, cards, assignment, np.random.default_rng(21), training=True, record=True)
    before = {r: torch.cat([p.detach().flatten() for p in policy.roles[r].parameters()]).clone() for r in ROLES}
    training_modes=[]
    hooks=[net.register_forward_pre_hook(lambda net,args:training_modes.append(net.training)) for net in policy.roles.values()]
    policy.eval()
    metrics = update(policy, optimizers, result['samples'],128,1,np.random.default_rng(4))
    for hook in hooks:
        hook.remove()
    assert training_modes and all(training_modes)
    assert not policy.training
    assert metrics['mse'] > 0
    for r in ROLES:
        after = torch.cat([p.detach().flatten() for p in policy.roles[r].parameters()])
        assert metrics['updates'][r] > 0 and metrics['samples'][r] > 0
        assert torch.isfinite(after).all() and not torch.equal(before[r], after)


def test_process_workers_identical_public_trajectories_and_exit():
    cards = [deal('train', 5, i) for i in range(4)]
    assignments = [{r:{'kind':'random'} for r in ROLES} for _ in cards]
    results = []
    for workers in (0, 2):
        with VectorGames(workers) as vector:
            results.append(collect(vector, Policies(None,'cpu'), cards, assignments,
                                   np.random.default_rng(4), record=True))
        assert all(not p.is_alive() for p in vector.processes)
    assert results[0]['traces'] == results[1]['traces']
    assert results[0]['decisions'] == results[1]['decisions']


def test_learner_sample_and_update_order_independent_of_worker_shards():
    config=settings()
    cards=[deal('train',16,i) for i in range(4)]
    assignments=[{r:{'kind':'learner'} for r in ROLES} for _ in cards]
    policies=[]
    results=[]
    for workers in (0,2):
        torch.manual_seed(13)
        policy=Policy(config['architecture'])
        with VectorGames(workers) as vector:
            result=collect(vector,Policies(policy,'cpu'),cards,assignments,
                           np.random.default_rng(4),training=True,epsilon=.1)
        optimizer=make_optimizers(policy,config)
        update(policy,optimizer,result['samples'],64,1,np.random.default_rng(4))
        policies.append(policy)
        results.append(result)
    for role in ROLES:
        for a,b in zip(results[0]['samples'][role],results[1]['samples'][role]):
            for x,y in zip(a,b):
                np.testing.assert_array_equal(x,y)
    for key,a in policies[0].state_dict().items():
        torch.testing.assert_close(a,policies[1].state_dict()[key],rtol=0,atol=0)


def test_opponent_assignment_preserves_team_and_rotates_learner_role():
    config = settings()
    config['opponents']['selfplay_probability'] = 0
    config['opponents']['frozen'] = [{'kind':'random'}]
    assignments = assign_training(config,6,0,np.random.default_rng(4),[])
    for i, assignment in enumerate(assignments):
        assert assignment[ROLES[i%3]] == {'kind':'learner'}
        assert sum(d['kind']=='learner' for d in assignment.values()) == 1
        assert all(d['kind']=='random' for r,d in assignment.items() if r != ROLES[i%3])


def run_train(tmp_path, name, config, resume=None):
    path = tmp_path / f'{name}.json'
    path.write_text(json.dumps(config))
    out = tmp_path/name
    cmd = [sys.executable,str(HERE/'launch.py'),'train','--recipe',str(path),'--out',str(out)]
    if resume:
        cmd += ['--resume',str(resume)]
    result = subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr
    return out


def test_exact_checkpoint_resume_with_historical_opponents(tmp_path):
    config = settings()
    config['opponents'].update(selfplay_probability=.4,historical_every_waves=1,historical_size=2)
    full = run_train(tmp_path,'full',config)
    partial_config = deepcopy(config)
    partial_config['max_episodes'] = 4
    partial = run_train(tmp_path,'partial',partial_config)
    resumed = run_train(tmp_path,'resumed',config,partial/'latest.pt')
    a, data_a = load_checkpoint(full/'latest.pt')
    b, data_b = load_checkpoint(resumed/'latest.pt')
    for key in a.state_dict():
        torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)
    for key in ('episodes','next_deal','decisions','updates','samples','used_deal_hashes','numpy_rng'):
        assert data_a['state'][key] == data_b['state'][key]
    assert torch.equal(data_a['torch_rng'], data_b['torch_rng'])
    assert (resumed/'allocation.json').exists()
    assert json.loads((resumed/'allocation.json').read_text())['cumulative_allocated_worker_seconds'] > json.loads((partial/'allocation.json').read_text())['cumulative_allocated_worker_seconds']
    descriptor = data_b['state']['historical'][0]
    # No silent opponent substitution when a frozen file changes.
    with pytest.raises(ValueError):
        load_checkpoint(descriptor['path'],expected_sha256='0'*64)


def test_budget_accounts_cpu_idle_and_prior_time():
    import time
    config = settings()
    config['allocation']['gpus']=1
    config['max_seconds']=100
    b = Budget(config,time.monotonic()-2,time.time(),prior=8)
    assert b.report()['allocated_worker_seconds'] >= 10
    assert b.report()['allocated_gpu_hours'] >= 10/3600
    assert not b.expired()
    config['max_seconds']=1
    b = Budget(config,time.monotonic()-2,time.time())
    assert b.expired()


def test_worker_fault_is_raised_not_scored():
    with VectorGames(1) as vector:
        vector.reset([deal('train',5,0)])
        with pytest.raises(RuntimeError,match='Illegal action index'):
            vector.step({0:-1})


def test_aborted_budget_produces_no_training_targets():
    class Expired:
        def expired(self):
            return True
    with VectorGames() as vector:
        result = collect(vector,Policies(None,'cpu'),[deal('train',8,0)],
                         [{r:{'kind':'random'} for r in ROLES}],np.random.default_rng(4),
                         budget=Expired(),training=True)
    assert result['aborted'] and not result['outcomes']
    assert all(not data for data in result['samples'].values())


def test_private_evaluator_seed_manifest_and_paired_schedule(tmp_path):
    from integrations.doudizhu.compare import compare
    config = settings()
    config['max_episodes']=0
    run = run_train(tmp_path,'initial',config)
    config.update(split='heldout',deals=100,opponents_panel=[{'kind':'heuristic'}])
    recipe = tmp_path/'eval.json'
    recipe.write_text(json.dumps(config))
    secret = tmp_path/'private-seeds.json'
    secret.write_text(json.dumps({'split':'heldout','deal_seed':7819283711,'policy_seed':81312,'deals':3}))
    out=tmp_path/'eval'
    cmd=[sys.executable,str(HERE/'launch.py'),'evaluate','--recipe',str(recipe),'--out',str(out),
         '--checkpoint',str(run/'initial.pt'),'--seed-file',str(secret)]
    result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['config']['deal_seed']=='evaluator-private'
    assert manifest['config']['seed']=='evaluator-private'
    assert manifest['config']['private_seed_manifest_sha256']==sha256(secret)
    rows=json.loads((out/'games.json').read_text())
    assert len(rows)==9 and len({r['deal_sha256'] for r in rows})==3
    assert {r['candidate_role'] for r in rows}==set(ROLES)
    assert compare(rows,rows)['groups']['role_balanced_mean']['paired_win_rate_delta']==0
    with pytest.raises(ValueError):
        compare(rows,rows[:-1])


def test_launcher_enforces_full_process_deadline(tmp_path):
    config=settings()
    config['max_seconds']=.05
    recipe=tmp_path/'timeout.json'
    recipe.write_text(json.dumps(config))
    out=tmp_path/'timeout'
    cmd=[sys.executable,str(HERE/'launch.py'),'train','--recipe',str(recipe),'--out',str(out)]
    result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=10)
    assert result.returncode != 0
    allocation=json.loads((out/'allocation.json').read_text())
    assert allocation['hard_deadline_hit']
    assert allocation['invocation_allocated_worker_seconds']>=.05
    assert allocation['cumulative_allocated_gpu_hours']==0


def test_launcher_cancellation_stops_worker_and_records_elapsed(tmp_path):
    import os
    import signal
    import time
    config=settings();config.update(max_seconds=120,max_episodes=100000)
    recipe=tmp_path/'cancel.json';recipe.write_text(json.dumps(config))
    out=tmp_path/'cancel'
    command=[sys.executable,str(HERE/'launch.py'),'train','--recipe',str(recipe),'--out',str(out)]
    launcher=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
    try:
        # The lock is acquired before spawn. Allow enough time for the launcher to
        # register handlers; the signal still arrives during normal child work.
        deadline=time.monotonic()+10
        while not (out/'manifest.json').exists():
            if launcher.poll() is not None:raise AssertionError(launcher.stderr.read())
            if time.monotonic()>deadline:raise AssertionError('worker did not initialize')
            time.sleep(.02)
        launcher.send_signal(signal.SIGTERM)
        launcher.wait(timeout=5)
        assert launcher.returncode==128+signal.SIGTERM
        allocation=json.loads((out/'allocation.json').read_text())
        assert allocation['interrupted_signal']==signal.SIGTERM
        assert allocation['invocation_allocated_worker_seconds']>0
        assert not allocation['hard_deadline_hit']
        with pytest.raises(ProcessLookupError):os.kill(allocation['worker_pid'],0)
    finally:
        if launcher.poll() is None:launcher.kill();launcher.wait()


def test_forced_moves_preserve_rng_trajectory_samples_and_weights():
    config=settings()
    cards=[deal('train',91,i) for i in range(12)]
    assignments=[{r:{'kind':'learner'} for r in ROLES} for _ in cards]
    results=[]
    weights=[]
    rng_states=[]
    for forced in (False,True):
        torch.manual_seed(18)
        policy=Policy(config['architecture'])
        rng=np.random.default_rng(93)
        with VectorGames() as vector:
            result=collect(vector,Policies(policy,'cpu',skip_forced=forced),cards,assignments,rng,
                           epsilon=.37,training=True,record=True)
        update(policy,make_optimizers(policy,config),result['samples'],128,1,rng)
        results.append(result)
        weights.append({k:v.clone() for k,v in policy.state_dict().items()})
        rng_states.append(rng.bit_generator.state)
    assert results[0]['traces']==results[1]['traces']
    assert results[0]['candidates']==results[1]['candidates']
    assert rng_states[0]==rng_states[1]
    for role in ROLES:
        assert len(results[0]['samples'][role])==len(results[1]['samples'][role])
        for a,b in zip(results[0]['samples'][role],results[1]['samples'][role]):
            for x,y in zip(a,b):
                np.testing.assert_array_equal(x,y)
    for key in weights[0]:
        torch.testing.assert_close(weights[0][key],weights[1][key],atol=0,rtol=0)


def test_worker_empty_shard_reset_closed_and_timeout_cleanup():
    with VectorGames(2) as vector:
        vector.reset([deal('train',5,0)])
        assert set(vector.step({0:0}))=={0}
        assert set(vector.reset([deal('train',5,1)]))=={0}
    vector.close()  # idempotent
    assert all(not p.is_alive() for p in vector.processes)
    with pytest.raises(RuntimeError,match='closed'):
        vector.reset([])
    with pytest.raises(RuntimeError,match='closed'):
        vector.step({})
    # A stopped worker cannot multiply timeout by worker count or leak on exit.
    import os
    import signal
    import time
    vector=VectorGames(1,timeout=10)
    try:
        vector.reset([deal('train',6,0)])
        vector.timeout=.1
        os.kill(vector.processes[0].pid,signal.SIGSTOP)
        with pytest.raises(TimeoutError):
            vector.step({0:0})
    finally:
        vector.close()
    assert not vector.processes[0].is_alive()


def test_nonfinite_update_restores_caller_mode():
    config=settings()
    policy=Policy(config['architecture'])
    policy.eval()
    obs=Game(deal('train',4,0)).observe()
    x,z,actions=obs.encode()
    samples={r:[] for r in ROLES}
    samples['landlord']=[(np.concatenate((x,actions[0])),z,float('nan'))]
    with pytest.raises(FloatingPointError):
        update(policy,make_optimizers(policy,config),samples,128,1,np.random.default_rng(4))
    assert not policy.training


def test_new_checkpoint_rejects_contract_drift(tmp_path,monkeypatch):
    import integrations.doudizhu.artifacts as artifacts
    config=settings()
    policy=Policy(config['architecture'])
    path=tmp_path/'contract.pt'
    artifacts.save_checkpoint(path,policy,make_optimizers(policy,config),{},config)
    artifacts.load_checkpoint(path)
    monkeypatch.setattr(artifacts,'contract_hash',lambda:'0'*64)
    with pytest.raises(ValueError,match='contract mismatch'):
        artifacts.load_checkpoint(path)


def test_compact_worker_features_samples_and_heuristics_match():
    from integrations.doudizhu.environment import CompactObservation, heuristic
    import pickle
    obs=Game(deal('train',73,0)).observe()
    compact=pickle.loads(pickle.dumps(CompactObservation.from_observation(obs)))
    assert heuristic(compact)==heuristic(obs)
    for a,b in zip(obs.encode(),compact.encode()):
        np.testing.assert_array_equal(a,b)
        assert not b.flags.writeable
    config=settings()
    cards=[deal('train',16,i) for i in range(8)]
    assignments=[{r:{'kind':'learner'} for r in ROLES} for _ in cards]
    rows=[]
    for workers,packed in ((0,False),(2,False),(2,True)):
        torch.manual_seed(13)
        policy=Policy(config['architecture'])
        with VectorGames(workers,compact=packed) as vector:
            row=collect(vector,Policies(policy,'cpu'),cards,assignments,np.random.default_rng(4),
                        training=True,epsilon=.1,record=True)
        rows.append(row)
    for row in rows[1:]:
        assert row['traces']==rows[0]['traces']
        for role in ROLES:
            for a,b in zip(row['samples'][role],rows[0]['samples'][role]):
                for x,y in zip(a,b):
                    np.testing.assert_array_equal(x,y)


def test_failed_wave_reservation_is_not_reused_on_resume(tmp_path,monkeypatch):
    import time
    from types import SimpleNamespace
    import integrations.doudizhu.runner as runner
    config=settings()
    config.update(max_episodes=4)
    out=tmp_path/'failed'
    args=SimpleNamespace(out=str(out),resume=None,prior_worker_seconds=0.,allocated_start_unix=None)
    real=runner.collect
    def fail(*a,**k):raise RuntimeError('injected simulator fault')
    monkeypatch.setattr(runner,'collect',fail)
    with pytest.raises(RuntimeError,match='injected'):
        runner.train(config,args,time.monotonic(),time.time())
    assert json.loads((out/'deal-reservation.json').read_text())['next_deal']==4
    monkeypatch.setattr(runner,'collect',real)
    resumed=run_train(tmp_path,'after-fault',config,out/'initial.pt')
    _,data=load_checkpoint(resumed/'latest.pt')
    assert data['state']['next_deal']==8
    assert data['state']['aborted_waves']==1
    assert data['state']['used_deal_hashes']==[digest(deal('train',config['deal_seed'],i)) for i in range(4,8)]
