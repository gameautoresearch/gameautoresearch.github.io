"""Correctness qualification for OPTIONAL compact embedding reduction (no benchmark).

CPU: .venv-clash/bin/python integrations/clash/native/test_compact_embeddings.py --out NEW
GPU: same command --cuda on an already allocated worker; never allocates a worker.
"""
from pathlib import Path
import argparse,copy,json,os,subprocess,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))


def collect(data,out):
    import numpy as np
    from runtime import NativeCPU
    from train import schema
    s=schema(data);vocab=data['vocabulary'];env=NativeCPU([301,409,811,1207]);counts={name:0 for name in data['cards']};saved=[];stun=0;rows=set();buffs=set();max_entities=0
    history=[];rng=np.random.default_rng(81108)
    try:
        for decision in range(850):
            obs=env.observations.copy();mask=env.masks.copy();entities=obs[...,s.header_ints:].reshape(4,2,s.max_entities,s.entity_ints)
            n=obs[...,s.header['entity_count'][0]];max_entities=max(max_entities,int(n.max()))
            live=np.arange(s.max_entities)[None,None,:]<n[...,None]
            rows.update(int(x) for x in entities[...,s.entity['row']][live] if x>=0)
            b=entities[..., [s.entity[f'buff_{i}'] for i in range(s.buff_slots)]]
            buffs.update(int(x) for x in b[live].reshape(-1) if x>=0)
            has_stun=bool((b==244).any());stun+=int(has_stun)
            # Retain bounded public frames including every observed stun window.
            if decision%23==0 or has_stun and stun<=8:saved.append((obs,mask))
            actions=np.zeros((4,2),np.int32)
            for lane in range(4):
                for seat in range(2):
                    choices=[]
                    for slot,index in enumerate(s.header['hand_card']):
                        card=int(obs[lane,seat,index]);legal=np.flatnonzero(mask[lane,seat,1+slot*576:1+(slot+1)*576])
                        if card>=0 and len(legal):choices.append((counts[vocab[card]],float(rng.random()),slot,card,legal))
                    if not choices:continue
                    _,_,slot,card,legal=min(choices,key=lambda x:x[:2]);name=vocab[card]
                    # Public seat coordinates always show own half below row16.
                    # Zap at the enemy Princess tower guarantees a reachable stun;
                    # moving units at the bridge additionally exercise combat buffs.
                    if name in ('Zap','Fireball','Arrows'):target=25*18+3
                    else:target=14*18+3
                    tile=int(legal[np.argmin(abs(legal-target))]);actions[lane,seat]=1+slot*576+tile;counts[name]+=1
            history.append(actions.tolist());env.step(actions)
            assert not env.infos[:,s.info['fault']].any(),'native simulator fault'
            assert not env.infos[:,[s.info['illegal_0'],s.info['illegal_1']]].any(),'illegal test action'
            done=(env.infos[:,s.info['winner']]>=0)|(env.infos[:,s.info['truncated']]!=0)
            if done.any():env.reset(rng.integers(0,1000000,4,dtype=np.int32),done)
            if decision>=300 and all(counts.values()) and stun>=2:break
        assert all(counts.values()),counts
        assert 244 in buffs,'real public observations did not expose ZapFreeze'
        np.savez_compressed(out/'real-observations.npz',observations=np.concatenate([x for x,_ in saved]),masks=np.concatenate([x for _,x in saved]))
        (out/'actions.json').write_text(json.dumps(history)+'\n')
        return dict(status='passed',decisions=len(history),lanes=4,card_plays=counts,stun_observation_frames=stun,buff_ids=sorted(buffs),entity_row_names=[vocab[x] for x in sorted(rows)],max_entities=max_entities,retained_frames=len(saved)*4)
    finally:env.close()


def compare_gradient(actual,expected,label,rtol=2e-5,atol=2e-7):
    import torch
    torch.testing.assert_close(actual,expected,rtol=rtol,atol=atol,msg=label)
    delta=(actual-expected).abs();return dict(max_absolute=float(delta.max()) if delta.numel() else 0.,max_relative=float((delta/expected.abs().clamp_min(atol)).max()) if delta.numel() else 0.,rtol=rtol,atol=atol)


def qualify(args):
    import numpy as np,torch
    from model import Policy
    from train import schema,binding
    from compact_embeddings import install,CompactEmbedding
    from gpu_features import DeviceBatchEncoder
    out=args.out;out.mkdir(parents=True,exist_ok=False);data=json.loads(args.data.read_text());s=schema(data)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(81108)
    device=torch.device('cuda' if args.cuda else 'cpu')
    if args.cuda:
        assert torch.cuda.device_count()==1,'requires one already allocated GPU'
        torch.cuda.manual_seed_all(81108);torch.backends.cuda.matmul.allow_tf32=False
    source=binding(args.data);coverage=collect(data,out)
    baseline=Policy(s,len(data['vocabulary']),len(data['buff_names']),64,12).to(device)
    torch.nn.init.normal_(baseline.slot.weight,std=.01)
    compact=copy.deepcopy(baseline);supports=install(compact,data);supports={k:list(map(int,v)) for k,v in supports.items()}
    assert supports['buffs']==[0,1,245],supports['buffs']
    npz=np.load(out/'real-observations.npz');obs=npz['observations'];masks=npz['masks']
    encoder=DeviceBatchEncoder(s,copy=True);samples=np.linspace(0,len(obs)-1,min(48,len(obs)),dtype=int)
    batch=encoder(torch.as_tensor(obs[samples],device=device),torch.as_tensor(masks[samples],device=device))
    records={};synthetic={}
    # All support IDs, repeated IDs, and many padding IDs. Gradients are externally
    # specified, so this checks embedding backward independently of Policy.encode.
    for name,ids in supports.items():
        a=getattr(baseline,name);b=getattr(compact,name);a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
        values=torch.tensor(([0]*240+ids*3)*16,device=device,dtype=torch.long).reshape(16,-1)
        aout=a(values);bout=b(values);assert torch.equal(aout,bout),name+' forward'
        upstream=torch.randn(aout.shape,device=device)/256
        aout.backward(upstream);bout.backward(upstream)
        errors=compare_gradient(b.weight.grad,a.weight.grad,name,rtol=2e-5,atol=2e-6)
        if a.padding_idx is not None:
            assert torch.count_nonzero(a.weight.grad[a.padding_idx])==0
            assert torch.count_nonzero(b.weight.grad[a.padding_idx])==0
        outside=sorted(set(range(a.num_embeddings))-set(ids))
        if outside:assert torch.count_nonzero(b.weight.grad[outside])==0
        synthetic[name]=dict(status='passed',support_ids=ids,lookups=values.numel(),forward_exact=True,padding_gradient_exact_zero=a.padding_idx is not None,unsupported_gradients_exact_zero=True,gradient_error=errors)
    def loss(model,packed):
        logits,value=model(packed);actions=packed['mask'].long().argmax(-1)
        logp=torch.log_softmax(logits,dim=-1).gather(1,actions[:,None]).squeeze(1)
        return (value.square()+.3*value-.2*logp).mean(),(logits,value)
    baseline.zero_grad(set_to_none=True);compact.zero_grad(set_to_none=True)
    la,fa=loss(baseline,batch);lb,fb=loss(compact,batch)
    for x,y in zip(fa,fb):assert torch.equal(x,y),'real model forward'
    la.backward();lb.backward()
    for (name,a),(other,b) in zip(baseline.named_parameters(),compact.named_parameters()):
        assert name==other;records[name]=compare_gradient(b.grad,a.grad,name)
    # ID1 is looked up by zero-filled inactive entity rows, then excluded by
    # Policy's alive mask. It is distinct from nn.Embedding's padding_idx=0.
    for name in ('cards','buffs'):
        for model in (baseline,compact):
            assert torch.count_nonzero(getattr(model,name).weight.grad[:2])==0,name+' explicit/masked padding gradient'
    # Same data/actions/Adam settings, explicitly measure accumulated reduction
    # roundoff; this is tolerance parity and not a claim of identical trajectories.
    oa=torch.optim.Adam(baseline.parameters(),lr=.0003,eps=1e-5,fused=args.cuda)
    ob=torch.optim.Adam(compact.parameters(),lr=.0003,eps=1e-5,fused=args.cuda)
    trajectory=[]
    for step in range(8):
        for model,opt in [(baseline,oa),(compact,ob)]:
            opt.zero_grad(set_to_none=True);value,_=loss(model,batch);value.backward();opt.step()
        maximum=max(float((a-b).detach().abs().max()) for a,b in zip(baseline.parameters(),compact.parameters()))
        adam=max(float((oa.state[a][key]-ob.state[b][key]).abs().max()) for a,b in zip(baseline.parameters(),compact.parameters()) for key in ('exp_avg','exp_avg_sq','step'))
        trajectory.append(dict(update=step+1,model_max_absolute_difference=maximum,adam_state_max_absolute_difference=adam))
    assert trajectory[-1]['model_max_absolute_difference']<3e-5,trajectory
    # Test fail-closed support validation on CPU so a CUDA device assert cannot
    # poison the allocated worker. The GPU variant repeats these in child processes.
    rejected=[]
    for name in ('cards','buffs'):
        module=getattr(compact,name);unknown=next(x for x in range(module.num_embeddings) if x not in supports[name])
        original=torch.nn.Embedding(module.num_embeddings,module.embedding_dim,padding_idx=0)
        bad=CompactEmbedding(original,supports[name])
        for grad in (True,False):
            try:
                with torch.set_grad_enabled(grad):bad(torch.tensor([unknown]))
            except (RuntimeError,ValueError,IndexError) as exc:
                assert 'frozen fixed8 support' in str(exc),str(exc)
            else:raise AssertionError('unknown ID accepted '+name+' grad='+str(grad))
            rejected.append(dict(module=name,id=unknown,grad_enabled=grad,device='cpu'))
            if args.cuda:
                cmd=[sys.executable,str(Path(__file__).resolve()),'--unknown-worker',name,'--grad',str(int(grad)),'--data',str(args.data)]
                proc=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                (out/f'reject-cuda-{name}-{int(grad)}.log').write_text(proc.stdout)
                assert proc.returncode!=0 and ('frozen fixed8 support' in proc.stdout or 'device-side assert' in proc.stdout),proc.stdout
                rejected.append(dict(module=name,id=unknown,grad_enabled=grad,device='cuda',isolated_process=True))
    assert source==binding(args.data),'source changed during qualification'
    result=dict(status='passed',device=str(device),gpu_name=torch.cuda.get_device_name() if args.cuda else None,real_native_observation_coverage=coverage,synthetic=synthetic,real_policy_forward_exact=True,real_policy_parameter_gradients=records,explicit_and_masked_padding_gradients_exact_zero=True,optimizer_trajectory=trajectory,optimizer_trajectory_contract='Roundoff tolerance only versus standard nn.Embedding. Exact resume must be checked within the enabled compact mode; bitwise cross-mode optimizer trajectories are not promised.',unknown_id_rejections=rejected,source_binding=source,timing_comparison=False)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_binding','real_policy_parameter_gradients','synthetic')}))


def unknown_worker(args):
    import torch
    from model import Policy
    from train import schema
    from compact_embeddings import install
    data=json.loads(args.data.read_text());m=Policy(schema(data),len(data['vocabulary']),len(data['buff_names'])).cuda();support=install(m,data)
    item=getattr(m,args.unknown_worker);unknown=next(x for x in range(item.num_embeddings) if x not in support[args.unknown_worker])
    with torch.set_grad_enabled(bool(args.grad)):item(torch.tensor([unknown],device='cuda'))
    torch.cuda.synchronize()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--data',type=Path,default=HERE/'generated-v2/task-data.json');p.add_argument('--cuda',action='store_true');p.add_argument('--unknown-worker',choices=('cards','buffs'));p.add_argument('--grad',type=int,default=1);a=p.parse_args()
    if a.unknown_worker:unknown_worker(a)
    else:
        if a.out is None:p.error('--out required')
        qualify(a)
