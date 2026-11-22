"""Explicit supervised control; no PPO/value/entropy gradient is applied."""
import torch

def calibrated_loss(logits,mask,target):
 active=mask.sum(-1)>1;deploy=(target>0)&active
 tile=logits[:,1:2305].reshape(-1,4,576);marginal=tile.logsumexp(-1)
 gate=marginal.logsumexp(-1)-logits[:,0]
 gate_loss=(torch.nn.functional.binary_cross_entropy_with_logits(gate,deploy.float(),reduction='none')*active).sum()/active.sum().clamp_min(1)
 slot=((target-1).clamp_min(0)//576).long();position=((target-1).clamp_min(0)%576).long()
 card_loss=(torch.nn.functional.cross_entropy(marginal,slot,reduction='none')*deploy).sum()/deploy.sum().clamp_min(1)
 selected=tile[torch.arange(len(tile),device=tile.device),slot]
 tile_loss=(torch.nn.functional.cross_entropy(selected,position,reduction='none')*deploy).sum()/deploy.sum().clamp_min(1)
 return gate_loss+.5*(card_loss+tile_loss)

def update(model,opt,s,last,c,teacher_coef):
 if c['ppo_coef'] or c['value_coef'] or c['entropy_coef'] or not teacher_coef:raise ValueError('This source is an explicitly supervised-only control')
 flat={k:v.flatten(0,1) for k,v in s.items()};rows=[]
 for _ in range(c['epochs']):
  for ix in torch.randperm(len(flat['obs']),device=flat['obs'].device).split(c['minibatch']):
   logits,_=model(flat['obs'][ix],flat['mask'][ix]);demo,_=model(flat['demo_obs'][ix],flat['demo_mask'][ix])
   own=calibrated_loss(logits,flat['mask'][ix],flat['teacher'][ix])
   expert=calibrated_loss(demo,flat['demo_mask'][ix],flat['demo_teacher'][ix])
   mix=c['learner_state_teacher_mix'];bc=mix*own+(1-mix)*expert;loss=teacher_coef*bc
   if not torch.isfinite(loss):raise ValueError('nonfinite supervised loss')
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c['max_grad_norm']);opt.step()
   rows.append(torch.stack((loss.detach(),own.detach(),expert.detach())))
 avg=torch.stack(rows).mean(0).cpu().tolist()
 return dict(loss=avg[0],policy_loss=None,teacher_nll=avg[0]/teacher_coef,learner_state_supervised_loss=avg[1],expert_state_supervised_loss=avg[2],teacher_coef=teacher_coef,optimizer_steps=len(rows),supervised_only=True,ppo_gradient_applied=False)
