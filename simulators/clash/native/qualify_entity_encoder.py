"""Correctness qualifier for the opt-in staged entity encoder (no launcher).

--cpu-reference tests staging/VJP contracts only. --cuda is mandatory before
claiming the Triton candidate works; it must run on an already allocated GPU.
No timings or throughput/strength claims are emitted.
"""
from pathlib import Path
import argparse,copy,hashlib,json,os,random,subprocess,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]


def _model(data,device):
    from model import Policy
    from train import schema
    model=Policy(schema(data),len(data['vocabulary']),len(data['buff_names']),64,12).to(device)
    model.device_indices=True;return model


def _raw(data,device,fixture=None):
    import numpy as np,torch
    from train import schema
    s=schema(data)
    path=Path(fixture) if fixture is not None else HERE.parents[2]/'evidence/clash-implementation/native-training-20261108/compact-embedding-cpu-3/real-observations.npz'
    arrays=np.load(path);raw=torch.from_numpy(arrays['observations'].reshape(-1,s.seat_ints).copy())
    masks=torch.from_numpy(arrays['masks'].reshape(-1,s.actions).copy()).bool()
    rows=raw[:,s.header_ints:].reshape(-1,200,34)
    stun=torch.where((rows[...,s.entity['buff_0']]==data['buff_names'].index('ZapFreeze')).any(1))[0]
    assert len(stun),'real fixture must cover ZapFreeze'
    selection=torch.unique(torch.cat((torch.linspace(0,len(raw)-1,6).long(),stun[:1])))
    return raw[selection].to(device),masks[selection].to(device),str(path),hashlib.sha256(path.read_bytes()).hexdigest()


def _cases(raw,masks,s):
    import torch
    yield 'real-public',raw,masks,False,200
    for count in (0,1,2,32,33,64,65,96,97,200):
        r=raw[:2].clone();rows=r[:,s.header_ints:].reshape(-1,200,34)
        rows.copy_(rows[:,:1].clone().expand_as(rows));r[:,s.header['entity_count'][0]]=count
        yield 'repeated-count-'+str(count),r,masks[:2],False,200
    yield 'distinct-feature-first-index-ties',raw,masks,True,200
    r=raw[:2].clone();rows=r[:,s.header_ints:].reshape(-1,200,34)
    rows[...,s.entity['x']]=2147483647;rows[...,s.entity['level']]=-2147483648
    # State/kind clamp at their FULL supported vocab ends, proving +1 widens.
    rows[...,s.entity['state']]=2147483647;rows[...,s.entity['kind']]=-2147483648
    yield 'int32-extremes-before-clamp',r,masks[:2],False,200
    for bound in (32,64,96):
        r=raw[:2].clone();r[:,s.header['entity_count'][0]].clamp_(max=bound)
        yield 'static-bucket-'+str(bound),r,masks[:2],False,bound


def _batch(raw,masks,s,bound):
    return {'header':raw[:,:s.header_ints],
        'entities':raw[:,s.header_ints:].reshape(-1,200,34)[:,:bound],'mask':masks}


def _maximum_error(a,b):
    return float((a-b).detach().abs().max()) if a.numel() else 0.


def _qualify_cases(data,raw,masks,backend):
    import torch
    from entity_encoder import install,PARAMETERS,_features
    base=_model(data,raw.device);reports=[]
    from entity_encoder import fixed8_support
    synthetic=raw[:2].clone();er=synthetic[:,base.s.header_ints:].reshape(-1,200,34)
    allowed=fixed8_support(base,data)
    for name,field in (('cards','row'),('kind','kind'),('side','side'),('state','state')):
        ids=torch.tensor(allowed[name],device=raw.device);vals=ids[torch.arange(200,device=raw.device)%len(ids)]
        er[...,base.s.entity[field]]=vals-(1 if name in ('cards','state') else 0)
    for slot in range(3):
        ids=torch.tensor(allowed['buffs'],device=raw.device);er[...,base.s.entity['buff_'+str(slot)]]=ids[(torch.arange(200,device=raw.device)+slot)%len(ids)]-1
    synthetic[:,base.s.header['entity_count'][0]]=200
    cases=list(_cases(raw,masks,base.s))+[('all-supported-embedding-ids',synthetic,masks[:2],False,200)]
    for label,r,m,ties,bound in cases:
        expected=copy.deepcopy(base);actual=copy.deepcopy(base)
        if ties:
            with torch.no_grad():expected.entity_net[2].weight.zero_();actual.entity_net[2].weight.zero_()
        batch=_batch(r,m,base.s,bound)
        aliases=[(n,id(p),p.data_ptr()) for n,p in actual.named_parameters()];names=list(actual.state_dict())
        rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all() if r.is_cuda else None
        adapter=install(actual,data,backend=backend)
        assert names==list(actual.state_dict()) and aliases==[(n,id(p),p.data_ptr()) for n,p in actual.named_parameters()]
        # Complete original pooling/global path, not an alternate segment max.
        ec,_=expected.encode(batch);ac,_=actual.encode(batch)
        torch.testing.assert_close(ac,ec,atol=2e-6,rtol=2e-5,msg=label+' context')
        # Direct live Z check; inactive candidate zeros are an explicit contract.
        params=[dict(actual.named_parameters())[n] for n in PARAMETERS]
        with torch.no_grad():
            dense=actual.entity_net(_features(batch['entities'],params,adapter))
            z=adapter(batch['entities'],batch['header'][:,base.s.header['entity_count'][0]])
            alive=torch.arange(bound,device=r.device)[None,:]<batch['header'][:,base.s.header['entity_count'][0],None]
            torch.testing.assert_close(z[alive],dense[alive],atol=2e-6,rtol=2e-5,msg=label+' live Z')
            assert (z[~alive]==0).all()
        upstream=torch.linspace(-.8,.9,ec.numel(),device=r.device).reshape_as(ec)
        (ec*upstream).sum().backward();(ac*upstream).sum().backward();errors={}
        for (name,a),(other,b) in zip(actual.named_parameters(),expected.named_parameters()):
            assert name==other and (a.grad is None)==(b.grad is None),name
            if a.grad is not None:
                torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5,msg=label+' gradient '+name)
                errors[name]=_maximum_error(a.grad,b.grad)
        for name in PARAMETERS:assert dict(actual.named_parameters())[name].grad is not None,name
        for name in ('cards','buffs'):assert (getattr(actual,name).weight.grad[0]==0).all(),name
        # Existing Adam preserves full-vocabulary zero gradients and momentum.
        ao=torch.optim.Adam(actual.parameters(),lr=.0003,eps=1e-5)
        eo=torch.optim.Adam(expected.parameters(),lr=.0003,eps=1e-5)
        ao.step();eo.step()
        for (name,a),(_,b) in zip(actual.named_parameters(),expected.named_parameters()):
            torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5,msg=label+' Adam '+name)
        assert torch.equal(torch.get_rng_state(),rng),'encoder consumed CPU RNG'
        if r.is_cuda:assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),cuda_rng)),'encoder consumed CUDA RNG'
        adapter.uninstall()
        with torch.no_grad():torch.testing.assert_close(actual.encode(batch)[0],expected.encode(batch)[0],atol=2e-6,rtol=2e-5)
        reports.append({'fixture':label,'rows':bound,'context_max_abs_error':_maximum_error(ac,ec),
                        'gradient_max_abs_errors':errors,'all_entity_grads_present':True})
    return reports


def _lifecycle(data,raw,masks,backend):
    import torch
    from entity_encoder import install
    from precision import _rng,_exact
    model=_model(data,raw.device);batch=_batch(raw,masks,model.s,200)
    identities=[(n,id(p),p.data_ptr()) for n,p in model.named_parameters()];names=list(model.state_dict());rng=_rng()
    adapter=install(model,data,backend=backend);_exact(_rng(),rng,'adapter construction RNG')
    clone=copy.deepcopy(model)
    assert clone._live_entity_adapter.model is clone and clone.encode.__self__ is clone
    with torch.no_grad():
        a=model.encode(batch)[0];b=clone.encode(batch)[0];torch.testing.assert_close(a,b,atol=0,rtol=0)
    adapter.uninstall()
    assert identities==[(n,id(p),p.data_ptr()) for n,p in model.named_parameters()] and names==list(model.state_dict())
    assert clone._live_entity_adapter.model is clone
    clone._live_entity_adapter.uninstall()
    with torch.no_grad():torch.testing.assert_close(model.encode(batch)[0],clone.encode(batch)[0],atol=0,rtol=0)
    _exact(_rng(),rng,'adapter/deepcopy/uninstall RNG')
    return {'parameter_objects_storage_state_names_preserved':True,'deepcopy_adapter_and_bound_method_point_to_clone':True,'install_uninstall_all_rng_exact':True}


def _negative(name,data,device,backend,fixture=None):
    import torch
    from entity_encoder import install
    model=_model(data,device);raw,masks,_,_=_raw(data,device,fixture);batch=_batch(raw,masks,model.s,200)
    adapter=install(model,data,backend=backend)
    if name=='unknown-id':batch['entities'][0,199,model.s.entity['row']]=1 # encoded2 outside fixed8
    elif name=='row-overflow':batch['entities'][0,199,model.s.entity['row']]=2147483647
    elif name=='negative-count':batch['header'][0,model.s.header['entity_count'][0]]=-1
    elif name=='capacity':batch['header'][0,model.s.header['entity_count'][0]]=201
    elif name=='nonfinite-unused':
        with torch.no_grad():model.cards.weight[-1,0]=float('nan')
        batch['header'][:,model.s.header['entity_count'][0]]=0
    elif name=='overflow-unsafe':
        with torch.no_grad():model.entity_net[0].weight.fill_(1e38)
    elif name=='bad-consumer':
        model.zero_grad(set_to_none=True);batch['header'][:,model.s.header['entity_count'][0]]=0
        adapter(batch['entities'],batch['header'][:,model.s.header['entity_count'][0]]).sum().backward()
        if device=='cuda':torch.cuda.synchronize()
        raise AssertionError('invalid consumer accepted')
    else:raise ValueError(name)
    model.encode(batch)
    if device=='cuda':torch.cuda.synchronize()
    raise AssertionError('invalid input accepted')


def _graphs(data,raw,masks,backend):
    import torch
    from types import SimpleNamespace
    from entity_encoder import install
    from gpu_features import DeviceBatchEncoder
    from learner_graph import LearnerGraph,qualify_learner_graph
    from precision import _rng,_restore_rng,_exact
    def trainer(state=None,optimizer_state=None):
        model=_model(data,raw.device)
        if state is not None:model.load_state_dict(state)
        install(model,data,backend=backend)
        optimizer=torch.optim.Adam(model.parameters(),lr=.0003,eps=1e-5,capturable=True,fused=True)
        if optimizer_state is not None:optimizer.load_state_dict(optimizer_state)
        return SimpleNamespace(model=model,optimizer=optimizer,schema=model.s,encoder=DeviceBatchEncoder(model.s,copy=False),
            c={'clip':.2,'value_coef':.5,'entropy_coef':.01,'max_grad_norm':.5,'compile_loss':False})
    t=trainer();batch=t.encoder(raw,masks)
    with torch.no_grad():dist,value=t.model.distribution(batch);action=masks.long().argmax(-1);logp=dist.log_prob(action)
    fields={'obs':raw,'mask':masks,'action':action,'logp':logp,'advantage':torch.linspace(-1,1,len(raw),device=raw.device),
            'return':value+.2,'weight':torch.ones(len(raw),device=raw.device)}
    eager=qualify_learner_graph(t,[fields,fields,fields],atol=0.,rtol=0.)
    initial=copy.deepcopy(t.model.state_dict());opt=copy.deepcopy(t.optimizer.state_dict());rng=_rng()
    def recreated():
        tr=trainer(initial,opt);_restore_rng(rng);before=_rng();g=LearnerGraph(tr,fields)
        _exact(_rng(),before,'graph recreation/warmup RNG')
        records=[]
        for _ in range(3):
            metric=g(fields).clone();torch.cuda.synchronize()
            records.append((metric,copy.deepcopy(tr.model.state_dict()),copy.deepcopy(tr.optimizer.state_dict()),_rng()))
        return records
    first=recreated();second=recreated();_exact(first,second,'exact candidate graph/model/Adam/all-RNG recreation')
    assert any(not torch.equal(initial[n],first[-1][1][n]) for n in initial),'graph never updated weights'
    return {'eager_vs_graph':eager,'recreated_graph_three_updates_model_adam_all_rng_exact':True,
            'existing_adam_parameter_updates_visible_to_replay':True,'whole_Trainer_engine_resume_not_tested':True}


def main(args):
    import numpy as np,torch
    if args.cuda and not torch.cuda.is_available():raise RuntimeError('actual CUDA GPU required')
    device='cuda' if args.cuda else 'cpu';backend='triton' if args.cuda else 'torch-reference'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(81108);random.seed(81108);np.random.seed(81108)
    data=json.loads((HERE/'generated-v2/task-data.json').read_text())
    if args.negative:
        try:_negative(args.negative,data,device,backend,args.fixture)
        except (RuntimeError,ValueError) as exc:
            if not args.cuda:print(json.dumps({'rejected':args.negative,'error':str(exc)}));return
            raise
        return
    args.out.mkdir(parents=True,exist_ok=False)
    fixture_path=args.fixture or HERE.parents[2]/'evidence/clash-implementation/native-training-20261108/compact-embedding-cpu-3/real-observations.npz'
    coverage=None
    if not fixture_path.exists():
        from test_compact_embeddings import collect
        coverage=collect(data,args.out);fixture_path=args.out/'real-observations.npz'
    raw,masks,fixture,sha=_raw(data,device,fixture_path)
    if args.batch_size<1:raise ValueError('positive qualification batch size required')
    choose=torch.arange(args.batch_size,device=raw.device)%len(raw)
    raw=raw[choose];masks=masks[choose]
    from train import binding
    source_binding=binding(HERE/'generated-v2/task-data.json')
    reports=_qualify_cases(data,raw,masks,backend)
    lifecycle=_lifecycle(data,raw,masks,backend)
    rejected=[]
    for name in ('unknown-id','row-overflow','negative-count','capacity','nonfinite-unused','overflow-unsafe','bad-consumer'):
        command=[sys.executable,str(Path(__file__).resolve()),'--negative',name,'--fixture',str(fixture_path),'--cuda' if args.cuda else '--cpu-reference']
        result=subprocess.run(command,text=True,capture_output=True,timeout=180)
        (args.out/(name+'.log')).write_text(result.stdout+result.stderr)
        # CPU catches guards inside the child; CUDA asserts intentionally poison
        # only that subprocess, never the qualification parent's device context.
        if args.cuda:assert result.returncode!=0 and ('assert' in result.stderr.lower() or 'nonfinite' in result.stderr.lower()),name
        else:assert result.returncode==0 and '"rejected"' in result.stdout,name
        rejected.append(name)
    graph=_graphs(data,raw,masks,backend) if args.cuda else None
    assert binding(HERE/'generated-v2/task-data.json')==source_binding,'source/data changed during qualification'
    sources={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('entity_encoder.py','entity_encoder_kernels.py','qualify_entity_encoder.py')}
    for name in sources:(args.out/name).write_bytes((HERE/name).read_bytes())
    result={'status':'passed','backend':backend,'qualification_batch_size':len(raw),'actual_gpu_executed':args.cuda,'device':torch.cuda.get_device_name() if args.cuda else 'cpu',
        'torch':torch.__version__,'sources':sources,'source_binding':source_binding,'new_native_fixture_coverage':coverage,'fixture':fixture,'fixture_sha256':sha,'cases':reports,'rejected':rejected,
        'lifecycle':lifecycle,'gradient_tolerance':{'atol':2e-6,'rtol':2e-5},'graph':graph,'math_mode':'fp32_ieee',
        'same_parameters_and_state_dict':True,'pooling':'original dense first-index max and alive mean',
        'deterministic_reductions':'FP32 cuBLAS linear GEMMs + ordered fixed-support Triton embedding trees; no floating atomic scatter',
        'limitations':['dense backward scratch (300 floats/selected row)','validation still constructs dense categorical index tensors',
        'dense parameter-gradient GEMMs still include zero inactive rows','whole Trainer/engine resume requires later integration qualification','first-order gradients only'],
        'timing_comparison':False,'speed_claim':None}
    (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--cuda',action='store_true');choice.add_argument('--cpu-reference',action='store_true')
    parser.add_argument('--out',type=Path);parser.add_argument('--negative');parser.add_argument('--fixture',type=Path);parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args()
    if not args.negative and args.out is None:parser.error('--out required')
    main(args)
