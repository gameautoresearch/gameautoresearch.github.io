"""Actual-device eager/CUDA-graph Adam qualification from native real rollouts.

No timing comparison, cloud launch, alternate optimizer, or checkpoint resume.
The optional compact embedding path is selected explicitly and compared to its
OWN eager execution; this does not claim exactness against dense embedding math.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time,os
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=HERE.parent/'configs/cuda-learning.json')
    p.add_argument('--data',type=Path,default=HERE/'generated-v2/task-data.json')
    p.add_argument('--compact-embedding-grad',action='store_true')
    p.add_argument('--bucket',action='store_true')
    args=p.parse_args()
    import torch
    if not torch.cuda.is_available():raise RuntimeError('actual CUDA hardware required; this qualifier never launches a cloud job')
    from train import Trainer
    from learner_graph import eager_minibatch,qualify_learner_graph,_optimizer_signature
    if args.bucket:
        from bucket_learner import qualify_bucket_learner
        qualify_learner_graph=qualify_bucket_learner
    args.out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    config=json.loads(args.config.read_text())
    config.update(backend='native-cuda',device='cuda',envs=4,width=32,minibatch=32,
                  rollout_steps=96,epochs=1,target_kl=0.,max_ticks=0,
                  graph_sampling=False,rollout_graph=False,learner_graph=True,
                  optimizer_fused=True,optimizer_capturable=True,
                  compact_embedding_grad=args.compact_embedding_grad,opponent_mode='selfplay',warmstart_steps=0)
    if 'hardware_allocation' in config:config['hardware_allocation'].update(gpu_model=torch.cuda.get_device_name(),gpu_count=1,cpu_threads=int(os.getenv('CLASH_CPU_CORES','4')),ram_gib=int(os.getenv('CLASH_RAM_GIB','16')))
    (args.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    trainer=Trainer(config,args.data)
    # This changes only the existing optimizer's capture permission. Fused Adam
    # already owns device-side step tensors; no parameter/state is replaced.
    for group in trainer.optimizer.param_groups:group['capturable']=True
    optimizer_id=id(trainer.optimizer);parameter_ids=tuple(id(p) for p in trainer.model.parameters())
    try:
        data=trainer.collect()
        weight=(data['mask'].sum(-1)>1).float() if config.get('active_policy_loss',True) else torch.ones_like(data['advantage'])
        norm=torch.where(weight.sum()>0,weight,torch.ones_like(weight));count=norm.sum()
        mean=(data['advantage']*norm).sum()/count
        std=torch.sqrt(((data['advantage']-mean).square()*norm).sum()/count)
        advantage=(data['advantage']-mean)/(std+1e-8)
        active=torch.nonzero(data['mask'].sum(-1)>1,as_tuple=False).flatten()
        forced=torch.nonzero(data['mask'].sum(-1)==1,as_tuple=False).flatten()
        if active.numel()<1 or forced.numel()<32:raise AssertionError('real rollout lacks both active and forced-wait qualification samples')
        def select(indices,n,offset=0):
            return indices[(torch.arange(n,device=indices.device)+offset)%indices.numel()]
        definitions=[('forced',32),('mixed',32),('mixed',16),('mixed',32),('forced',16),('mixed',16)]
        minibatches=[];selection=[]
        for j,(kind,n) in enumerate(definitions):
            if kind=='forced':ix=select(forced,n,j*7)
            else:
                na=min(n//2,active.numel());ix=torch.cat((select(active,na,j*3),select(forced,n-na,j*7)))
            fields={name:data[key][ix] for name,key in (('obs','obs'),('mask','mask'),('action','action'),('logp','logp'),('return','return'))}
            fields.update(advantage=advantage[ix],weight=weight[ix]);minibatches.append(fields)
            selection.append({'kind':kind,'size':n,'active_samples':int((fields['mask'].sum(-1)>1).sum().cpu()),'source_indices':ix.cpu().tolist()})
        if args.bucket:
            synthetic=[]
            for population in (0,32,33,64,65,96,97,200):
                fields={k:v.clone() for k,v in minibatches[1].items()}
                entities=fields['obs'][:,trainer.schema.header_ints:].reshape(-1,200,trainer.schema.entity_ints)
                template=entities[:,:1].clone();entities.zero_()
                if population:entities[:,:population].copy_(template.expand(-1,population,-1))
                fields['obs'][:,trainer.schema.header['entity_count'][0]]=population
                synthetic.append(fields)
            minibatches.extend(synthetic)
        assert not trainer.optimizer.state,'fresh qualification must start before any real Adam update'
        tolerance=dict(atol=2e-6,rtol=2e-5) if args.bucket else dict(atol=0.,rtol=0.)
        fresh=qualify_learner_graph(trainer,minibatches,**tolerance)
        for state in trainer.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):assert torch.count_nonzero(value)==0,'warmup/qualification consumed fresh Adam state'
        # Seed existing moments with TWO real updates, using the same optimizer.
        eager_minibatch(trainer,minibatches[1]);eager_minibatch(trainer,minibatches[2])
        torch.cuda.synchronize()
        state_before={p:{k:v.detach().clone() for k,v in state.items() if torch.is_tensor(v)} for p,state in trainer.optimizer.state.items()}
        pointers={(id(p),name):value.data_ptr() for p,state in trainer.optimizer.state.items() for name,value in state.items() if torch.is_tensor(value)}
        signature=_optimizer_signature(trainer.optimizer)
        existing=qualify_learner_graph(trainer,minibatches,**tolerance)
        assert id(trainer.optimizer)==optimizer_id
        assert tuple(id(p) for p in trainer.model.parameters())==parameter_ids
        assert _optimizer_signature(trainer.optimizer)==signature
        for parameter,state in state_before.items():
            for name,want in state.items():
                actual=trainer.optimizer.state[parameter][name]
                assert actual.data_ptr()==pointers[(id(parameter),name)]
                torch.testing.assert_close(actual,want,atol=0,rtol=0)
        result={'status':'passed','cuda_executed':True,'gpu':torch.cuda.get_device_name(),
                'fresh_adam':fresh,'existing_adam':existing,'same_optimizer_and_parameter_objects':True,
                'optimizer_state_addresses_preserved':True,'minibatches':selection,
                'real_rollout_samples':data['action'].numel(),'active_rollout_samples':active.numel(),
                'forced_rollout_samples':forced.numel(),'compact_embedding_grad':args.compact_embedding_grad,'bucket':args.bucket,
                'optimizer_fused':True,'optimizer_capturable':True,'timing_claims':False,
                'binding':trainer.binding,'learner_graph_sha256':hashlib.sha256((HERE/'learner_graph.py').read_bytes()).hexdigest(),
                'process_setup_compile_qualification_seconds':time.perf_counter()-started}
        (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='binding'}),flush=True)
    finally:trainer.env.close()
if __name__=='__main__':main()
