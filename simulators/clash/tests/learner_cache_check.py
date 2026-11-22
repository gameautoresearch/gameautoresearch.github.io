"""CPU identity check for borrowing and whole-rollout field caching."""
from pathlib import Path
import sys,json,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from runner import setup,network,Session,Opponents,collect,Timer,WorkerClock,ppo_update
from game import HERE,make_vector,dump

def main():
    out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
    c=json.loads((HERE/'configs/profile-trimmed.json').read_text());c.update(envs=4,rollout_steps=16,cache_rollout_device=True)
    rng=setup(c)
    with make_vector(c) as v:
        m=network(v,c);session=Session(v,c,rng);opponents=Opponents(v,c);opponents.select(session,range(v.count))
        data=collect(m,session,opponents,c,Timer('cpu'),WorkerClock(100))
        a=copy.deepcopy(m);b=copy.deepcopy(m)
        oa=torch.optim.Adam(a.parameters(),lr=c['learning_rate'],eps=1e-5);ob=torch.optim.Adam(b.parameters(),lr=c['learning_rate'],eps=1e-5)
        ma=ppo_update(a,oa,copy.deepcopy(data),c,np.random.default_rng(11),Timer('cpu'),WorkerClock(100))
        mb=ppo_update(b,ob,copy.deepcopy(data),dict(c,borrow_rollout_cache=True,cache_rollout_fields=True,combined_learner_metrics=True),np.random.default_rng(11),Timer('cpu'),WorkerClock(100))
        for k in a.state_dict():assert torch.equal(a.state_dict()[k],b.state_dict()[k]),k
        for key in oa.state_dict()['state']:
            for name in oa.state_dict()['state'][key]:assert torch.equal(oa.state_dict()['state'][key][name],ob.state_dict()['state'][key][name]),(key,name)
        assert ma==mb,(ma,mb)
        result={'status':'passed','scope':'bit-identical model/Adam/loss metrics for one full PPO update, cached full rollout vs borrowed buffers and cached learner fields'}
        dump(out/'result.json',result);print(json.dumps(result))

if __name__=='__main__':main()
