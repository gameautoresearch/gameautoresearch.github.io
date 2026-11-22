"""CUDA parity, complete actor graph RNG/live weights, and sparse gradient evidence."""
from pathlib import Path
import sys,json,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from runner import setup,network,Timer,sample
from actor import sample_actions,predict_values
from game import ROOT,HERE,Session,make_vector,dump

def main():
    out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
    checkpoint=ROOT/'evidence/clash-implementation/cuda-campaign/learning2/train/checkpoint.pt'
    saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
    c=dict(saved['config']);c.update(device='cuda',device_indices=True,pinned_transfer=True,entity_bucket=0,inference_entity_bucket=200,graph_inference=False)
    c['hardware_allocation']['gpu_model']=torch.cuda.get_device_name();setup(c)
    with make_vector(c) as v:
        session=Session(v,c,np.random.default_rng(13));session.restore(saved['session'])
        obs=v.observations.reshape(-1,v.schema.seat_ints).copy();masks=v.masks.reshape(-1,v.schema.actions).copy()
        m=network(v,c);m.load_state_dict(saved['model']);timer=Timer('cuda')
        result={'cuda':torch.version.cuda,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'states':len(obs),'active_rows':int((masks.sum(1)>1).sum()),'max_entities':int(obs[:,v.schema.header['entity_count'][0]].max())}
        torch.cuda.manual_seed_all(34);a,lp,values=sample(m,obs,masks,c,timer);state=torch.cuda.get_rng_state()
        torch.cuda.manual_seed_all(34);aa,ll=sample_actions(m,obs,masks,c,timer)
        np.testing.assert_array_equal(a,aa);assert torch.equal(state,torch.cuda.get_rng_state())
        np.testing.assert_allclose(lp,ll,rtol=2e-5,atol=3e-6)
        deferred=predict_values(m,obs,dict(c,value_batch_size=17),timer)
        np.testing.assert_allclose(values,deferred,rtol=2e-5,atol=3e-6)
        result['deferred_max_value_error']=float(np.max(np.abs(values-deferred)));result['deferred_actor_rng_equal']=True
        batch=m.pack(obs,masks,'cuda',True)
        a_model=network(v,c);a_model.load_state_dict(saved['model']);b_model=network(v,c);b_model.load_state_dict(saved['model'])
        actions=torch.tensor(a,device='cuda')
        d,va=a_model.distribution(batch);la=d.log_prob(actions);ea=d.entropy()
        lb,eb,vb=b_model.action_terms(batch,actions)
        for x,y in ((la,lb),(ea,eb),(va,vb)):torch.testing.assert_close(x,y,rtol=3e-5,atol=5e-6)
        (la.mean()+ea.mean()*.01+va.square().mean()).backward();(lb.mean()+eb.mean()*.01+vb.square().mean()).backward()
        errors={}
        for name,x in a_model.named_parameters():
            y=dict(b_model.named_parameters())[name]
            assert x.grad is not None and y.grad is not None,name
            torch.testing.assert_close(x.grad,y.grad,rtol=1e-3,atol=1e-5,msg=name)
            errors[name]=float((x.grad-y.grad).abs().max())
        result['sparse_max_gradient_error']=max(errors.values())
        graph=network(v,dict(c,graph_sampling=True));graph.load_state_dict(saved['model'])
        gc=dict(c,graph_sampling=True)
        for trial in range(3):
            torch.cuda.manual_seed_all(250+trial)
            expected=sample(m,obs,masks,c,timer);expected_rng=torch.cuda.get_rng_state()
            torch.cuda.manual_seed_all(250+trial)
            actual=sample(graph,obs,masks,gc,timer)
            for x,y in zip(expected,actual):np.testing.assert_array_equal(x,y)
            assert torch.equal(expected_rng,torch.cuda.get_rng_state()),'graph sampler RNG drift'
        with torch.no_grad():m.slot.bias[0].add_(.25);graph.slot.bias[0].add_(.25)
        torch.cuda.manual_seed_all(777);expected=sample(m,obs,masks,c,timer)
        torch.cuda.manual_seed_all(777);actual=sample(graph,obs,masks,gc,timer)
        for x,y in zip(expected,actual):np.testing.assert_array_equal(x,y)
        result['complete_actor_graph']={'actions_logp_values':'bit-identical for three RNG seeds','rng':'bit-identical to eager including first capture','live_parameter_update':'bit-identical'}
        dump(out/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
