"""Public standard-versus-pressure trajectory diagnostics, never a gameplay score."""
from pathlib import Path
import argparse,json,fcntl
import numpy as np,torch
from simulators.clash.autoresearch.common import setup_native,load_module,dump

def run(original,out):
 setup_native();from cpu_batch import NativeCPUBatch
 from evaluate import scripted_bot
 from simulators.clash.autoresearch.common import DATA
 torch.set_num_threads(1);source=Path(original);module=load_module(source/'policy.py','standard_agreement_policy');model=module.create(json.loads((source/'config.json').read_text()))
 with np.load(source/'weights.npz',allow_pickle=False) as weights:model.load_state_dict({k:torch.from_numpy(weights[k].copy()) for k in weights.files})
 data=json.loads(DATA.read_text());expert=scripted_bot(data);pressure=scripted_bot(data)
 for k,v in {'attack_elixir':40000,'support_elixir':30000,'back_row':8,'spell_targets':2,'spell_worth':1.0}.items():setattr(pressure,k,v)
 env=NativeCPUBatch([2200201,2200202,2200203,2200204],threads=1);active=np.ones(4,bool);counts={k:0 for k in ('nontrivial','teacher_deploy','teacher_wait','deploy_when_deploy','same_slot','same_action','wait_when_wait')};nll={k:[] for k in ('deploy','wait')};games=0
 try:
  with torch.inference_mode():
   for _ in range(1600):
    if not active.any():break
    obs=env.observations.copy();masks=env.masks.copy();seats=np.arange(4)%2;x=obs[np.arange(4),seats];m=masks[np.arange(4),seats]
    target=np.array([expert(a,b,np.random.default_rng(0)) for a,b in zip(x,m)],np.int64);other=np.array([pressure(obs[i,1-seats[i]],masks[i,1-seats[i]],np.random.default_rng(0)) for i in range(4)],np.int64)
    logits,_=model(torch.from_numpy(x),torch.from_numpy(m));chosen=module.act(model,torch.from_numpy(x),torch.from_numpy(m)).numpy();loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(target),reduction='none').numpy()
    live=active&(m.sum(-1)>1);deploy=live&(target>0);wait=live&(target==0)
    for key,value in [('nontrivial',live),('teacher_deploy',deploy),('teacher_wait',wait),('deploy_when_deploy',deploy&(chosen>0)),('same_slot',deploy&(chosen>0)&((chosen-1)//576==(target-1)//576)),('same_action',deploy&(chosen==target)),('wait_when_wait',wait&(chosen==0))]:counts[key]+=int(value.sum())
    for key,mask in [('deploy',deploy),('wait',wait)]:nll[key].extend(loss[mask].tolist())
    actions=np.zeros((4,2),np.int32);actions[np.arange(4),seats]=target;actions[np.arange(4),1-seats]=other;actions[~active]=0;env.step(actions)
    if env.observations[:,:,49].any() or env.infos[:,[5,7,8,9,11]].any():raise ValueError('Invalid public diagnostic trajectory')
    done=active&(env.infos[:,2]>=0);games+=int(done.sum());active[done]=False
  value={'scope':'public standard-versus-pressure trajectory imitation agreement only; not candidate gameplay','natural_teacher_games':games,'counts':counts,'deployment_recall':counts['deploy_when_deploy']/max(1,counts['teacher_deploy']),'same_card_on_teacher_deploy':counts['same_slot']/max(1,counts['teacher_deploy']),'same_action_on_teacher_deploy':counts['same_action']/max(1,counts['teacher_deploy']),'wait_accuracy':counts['wait_when_wait']/max(1,counts['teacher_wait']),'nll':{k:float(np.mean(v)) if v else None for k,v in nll.items()}};dump(out,value);return value
 finally:env.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 with open('evidence/local-performance.lock','a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);print(json.dumps(run(a.original,a.out),indent=2))
