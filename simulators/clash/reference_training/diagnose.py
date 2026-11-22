"""Teacher agreement on public expert trajectories; diagnostic, never a score."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from simulators.clash.autoresearch.common import setup_native,load_module,dump

def run(original,out):
 setup_native()
 from cpu_batch import NativeCPUBatch
 from simulators.clash.autoresearch.visible_teacher import PublicTeacher
 original=Path(original);c=json.loads((original/'config.json').read_text());module=load_module(original/'policy.py','diagnostic_policy');torch.set_num_threads(1);model=module.create(c)
 with np.load(original/'weights.npz',allow_pickle=False) as data:model.load_state_dict({k:torch.from_numpy(data[k].copy()) for k in data.files})
 teacher=PublicTeacher('teacher');env=NativeCPUBatch([1000201,1000202,1000203,1000204],threads=1);counts={'rows':0,'nontrivial':0,'teacher_deploy':0,'teacher_wait':0,'deploy_when_teacher_deploy':0,'same_slot_when_teacher_deploy':0,'same_action_when_teacher_deploy':0,'wait_when_teacher_wait':0};nll={k:[] for k in ('deploy','wait')};active=np.ones(4,dtype=bool);games=0
 try:
  with torch.inference_mode():
   for step in range(1600):
    if not active.any():break
    o=env.observations.copy();m=env.masks.copy();x=o.reshape(-1,7024);mask=m.reshape(-1,2307)
    expected=np.asarray([teacher(a,b) for a,b in zip(x,mask)],np.int64);logits,_=model(torch.from_numpy(x),torch.from_numpy(mask));chosen=module.act(model,torch.from_numpy(x),torch.from_numpy(mask)).numpy();loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(expected),reduction='none').numpy()
    live=np.repeat(active,2)&(mask.sum(-1)>1);deploy=live&(expected>0);wait=live&(expected==0)
    counts['rows']+=int(np.repeat(active,2).sum());counts['nontrivial']+=int(live.sum());counts['teacher_deploy']+=int(deploy.sum());counts['teacher_wait']+=int(wait.sum());counts['deploy_when_teacher_deploy']+=int((deploy&(chosen>0)).sum());counts['same_slot_when_teacher_deploy']+=int((deploy&(chosen>0)&((chosen-1)//576==(expected-1)//576)).sum());counts['same_action_when_teacher_deploy']+=int((deploy&(chosen==expected)).sum());counts['wait_when_teacher_wait']+=int((wait&(chosen==0)).sum())
    nll['deploy'].extend(loss[deploy].tolist());nll['wait'].extend(loss[wait].tolist());actions=expected.reshape(4,2);actions[~active]=0;env.step(actions)
    if env.observations[:,:,49].any() or env.infos[:,[5,7,8,9,11]].any():raise ValueError('invalid public diagnostic trajectory')
    done=active&(env.infos[:,2]>=0);games+=int(done.sum());active[done]=False
  result={'scope':'public teacher trajectory agreement; no candidate gameplay score','natural_teacher_games':games,'counts':counts,'teacher_nonwait_share':counts['teacher_deploy']/max(1,counts['nontrivial']),'deploy_recall':counts['deploy_when_teacher_deploy']/max(1,counts['teacher_deploy']),'slot_accuracy_given_teacher_deploy':counts['same_slot_when_teacher_deploy']/max(1,counts['teacher_deploy']),'exact_accuracy_given_teacher_deploy':counts['same_action_when_teacher_deploy']/max(1,counts['teacher_deploy']),'wait_accuracy':counts['wait_when_teacher_wait']/max(1,counts['teacher_wait']),'NLL':{k:float(np.mean(v)) if v else None for k,v in nll.items()}}
  dump(out,result);return result
 finally:env.close()
def main():
 p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(run(a.original,a.out),indent=2))
if __name__=='__main__':main()
