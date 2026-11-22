"""Compare outer CUDA rollout capture against its unfused device execution."""
from pathlib import Path
import json,sys,time,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train import Trainer

def main(config,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);c=json.loads(Path(config).read_text());c.update(envs=4,rollout_steps=8,minibatch=32,epochs=1,target_kl=0,graph_sampling=False,rollout_graph=False)
    eager=Trainer(c,Path(__file__).parent/'generated-v2/task-data.json');graph=Trainer(dict(c,rollout_graph=True),Path(__file__).parent/'generated-v2/task-data.json');rows=[];started=time.perf_counter()
    try:
        for i in range(4):
            rng=torch.cuda.get_rng_state();a=eager.collect();ma,sa=eager.learn(a);after=torch.cuda.get_rng_state();torch.cuda.set_rng_state(rng);b=graph.collect();mb,sb=graph.learn(b)
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            torch.testing.assert_close(ma,mb,rtol=0,atol=0);assert sa==sb
            for x,y in zip(eager.model.parameters(),graph.model.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
            for x,y in zip(eager.outputs(),graph.outputs()):torch.testing.assert_close(x,y,rtol=0,atol=0)
            torch.testing.assert_close(after,torch.cuda.get_rng_state(),rtol=0,atol=0);torch.testing.assert_close(eager.counters,graph.counters,rtol=0,atol=0)
            rows.append({'update':i+1,'samples':eager.samples,'exact':['rollout_every_field','losses','model_parameters','public_engine_outputs','counters','cuda_rng']})
        result={'status':'passed','cuda_executed':True,'gpu':torch.cuda.get_device_name(),'cases':rows,'process_seconds':time.perf_counter()-started};(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    finally:eager.env.close();graph.env.close()
if __name__=='__main__':main(*sys.argv[1:])
