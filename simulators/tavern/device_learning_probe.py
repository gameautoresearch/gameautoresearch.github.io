"""CUDA neural, CPU/device collection parity and source-bound exact resume gates."""
from collections import Counter
import copy
import json
from pathlib import Path
import random
import numpy as np
import torch
from device_env import DeviceEnv
from device_training import collect,train,update,PPO
from model import Policy


def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def qualify_gae():
    # Execute the original retained reference function without importing its
    # unrelated environment adapter or running any simulator on the host.
    import ast,ctypes as ct
    source=Path(__file__).with_name('rollout.py')
    tree=ast.parse(source.read_text())
    body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='gae']
    if len(body)!=1:raise RuntimeError('original GAE reference missing')
    scope={'np':np};exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),scope)
    rng=np.random.default_rng(816);values=rng.normal(size=21).astype(np.float32)
    rewards=rng.normal(size=21).astype(np.float32);offsets=[0,1,8,8,21];checks=[]
    env=DeviceEnv(1,kernels_only=True)
    for gamma,lam in [(1.,.95),(.99,.95),(.9,0.)]:
        expected=[scope['gae']([{'reward':float(rewards[j]),'value':float(values[j])}
            for j in range(start,end)],gamma,lam) for start,end in zip(offsets,offsets[1:])]
        adv=torch.empty(21,device='cuda');ret=torch.empty_like(adv)
        vv=torch.tensor(values,device='cuda');rr=torch.tensor(rewards,device='cuda')
        oo=torch.tensor(offsets,dtype=torch.int64,device='cuda')
        env.launch('tavern_gae',4,[vv,rr,oo,ct.c_uint32(4),ct.c_double(gamma),ct.c_double(lam),adv,ret])
        checks.append(np.array_equal(adv.cpu().numpy(),np.concatenate([x[0] for x in expected])) and
            np.array_equal(ret.cpu().numpy(),np.concatenate([x[1] for x in expected])))
    if not all(checks):raise AssertionError(('GPU GAE differs from original reference',checks))
    return {'original_cpu_reference':True,'bitwise':checks,'empty_and_variable_length_trajectories':True}


def qualify(out):
    gae_check=qualify_gae()
    (out/'gae-original-reference.json').write_text(json.dumps(gae_check,indent=2)+'\n')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.manual_seed(721);torch.cuda.manual_seed_all(721)
    base=Policy().cuda();base.fast_inference=True;base.actor_graph_max_shapes=32;base.inference_metrics=Counter()
    policy=copy.deepcopy(base.state_dict());rng=torch.cuda.get_rng_state().clone();arms=[]
    for backend in ['cpu','device']:
        model=Policy().cuda();model.load_state_dict(policy);model.fast_inference=True
        model.actor_graph_max_shapes=32;model.inference_metrics=Counter()
        torch.cuda.set_rng_state(rng)
        data,episodes,stats,states,phases=collect(model,8,12345,backend=backend)
        optimizer=torch.optim.Adam(model.parameters(),lr=PPO['learning_rate'],eps=1e-5)
        loss=update(model,optimizer,data)
        arms.append({'backend':backend,'data':[x.cpu() for x in data],'episodes':episodes,'stats':stats,
                     'policy':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
                     'cuda_rng':torch.cuda.get_rng_state().clone(),'loss':loss,'phases':phases})
    checks={k:equal(arms[0][k],arms[1][k]) for k in ['data','episodes','stats','policy','optimizer','cuda_rng','loss']}
    (out/'neural-collection-parity.json').write_text(json.dumps({'checks':checks,
        'rows':[{'backend':r['backend'],'stats':r['stats'],'phases':r['phases'],'loss':r['loss']} for r in arms]},indent=2)+'\n')
    if not all(checks.values()):raise AssertionError(checks)
    train(out/'full',n_games=8,updates=2,seed=45678)
    train(out/'split',n_games=8,updates=1,seed=45678)
    train(out/'resumed',n_games=8,updates=2,seed=45678,resume=out/'split/latest.pt')
    x=torch.load(out/'full/latest.pt',map_location='cpu',weights_only=False)
    y=torch.load(out/'resumed/latest.pt',map_location='cpu',weights_only=False)
    resume={k:equal(x[k],y[k]) for k in ['source_sha256','rules','seed','n_games','backend','update',
                'policy','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng','env_states']}
    (out/'neural-resume.json').write_text(json.dumps(resume,indent=2)+'\n')
    if not all(resume.values()):raise AssertionError(resume)
    if not all(row['learning']['parameter_delta_l2']>0 for row in x['stats']):raise AssertionError('no neural update')
    return {'gae_vs_original':gae_check,'collection_learner_parity':checks,'exact_resume':resume,'updates':2,'complete_neural_learning':True}
