"""One learner seat per lane; immutable opponents remain assigned through episodes."""
import torch
from . import policy,learner
from .league import visible_teacher,pfsp,tensor_identity

class Trainer:
 def __init__(self,c,env,device):
  self.c=c;self.env=env;self.device=device;torch.manual_seed(c['seed'])
  self.model=policy.create(c).to(device);self.optimizer=torch.optim.Adam(self.model.parameters(),lr=c['learning_rate'],eps=1e-5)
  self.updates=0;self.samples=0;self.natural_games=0;self.steps=0;self.losses=[]
  from simulators.clash.autoresearch.common import HERE,sha
  if sha(HERE/'visible_teacher.py')!=c['public_teacher_sha256']:raise ValueError('public teacher source pin differs')
  from simulators.clash.autoresearch.visible_teacher import PublicTeacher
  self.expert=PublicTeacher('teacher');self.base=3
  self.visible_refusal_nonzero_decisions=0
  self.league=[];self.identities=[];self.add_history()
  # 0/1 scripts, 2 untrained/history. All enter PFSP, none silently removed.
  self.wins=torch.zeros(4,device=device);self.games=torch.zeros(4,device=device)
  self.assignment=torch.arange(c['envs'],device=device)%4
  self.seat=torch.arange(c['envs'],device=device)%2;self.rows=torch.arange(c['envs'],device=device)
  self.neural_opponent_decisions=0;self.learned_neural_opponent_decisions=0;self.script_opponent_decisions=0
 def outputs(self):return tuple(torch.as_tensor(getattr(self.env,k),device=self.device) for k in ('observations','masks','infos','rewards'))
 def add_history(self):
  model=policy.create(self.c).to(self.device);model.load_state_dict(self.model.state_dict());model.eval()
  for p in model.parameters():p.requires_grad_(False)
  self.league.append(model);self.identities.append(tensor_identity(model.state_dict()))
 def expert_actions(self,o,m):
  import numpy as np
  x=o.detach().cpu().numpy();mask=m.detach().cpu().numpy()
  return torch.as_tensor(np.asarray([self.expert(a,b) for a,b in zip(x,mask)],dtype=np.int64),device=self.device)
 def step(self):
  c=self.c;s={k:[] for k in ('obs','mask','action','logp','value','reward','done','bootstrap','teacher')}
  for _ in range(c['rollout_steps']):
   o,m,info,_=self.outputs();x=o[self.rows,self.seat];mask=m[self.rows,self.seat]
   with torch.no_grad():
    logits,v=self.model(x,mask);dist=torch.distributions.Categorical(logits=logits);a=dist.sample();logp=dist.log_prob(a)
    ox=o[self.rows,1-self.seat];om=m[self.rows,1-self.seat]
    opponent=visible_teacher(ox,om);opponent=torch.where(self.assignment==1,visible_teacher(ox,om,True),opponent)
    expert_ids=torch.where(self.assignment==2)[0]
    if len(expert_ids):opponent[expert_ids]=self.expert_actions(ox[expert_ids],om[expert_ids])
    # Frozen policies select greedily, exactly as the exported inference contract.
    for k,model in enumerate(self.league):
     ids=torch.where(self.assignment==k+self.base)[0]
     if len(ids):
      opponent[ids]=policy.act(model,ox[ids],om[ids]);self.neural_opponent_decisions+=len(ids)
      if k>0:self.learned_neural_opponent_decisions+=len(ids)
    self.script_opponent_decisions+=int((self.assignment<self.base).sum())
    teacher=self.expert_actions(x,mask)
   for k,vv in (('obs',x),('mask',mask),('action',a),('logp',logp),('value',v),('teacher',teacher)):s[k].append(vv.clone())
   actions=torch.zeros((len(x),2),device=self.device,dtype=torch.int64);actions[self.rows,self.seat]=a;actions[self.rows,1-self.seat]=opponent
   self.env.step(actions if self.device=='cuda' else actions.numpy(),0)
   no,nm,ni,r=self.outputs()
   if (no[:,:,49]!=0).any() or (ni[:,[5,7,8,11]]!=0).any():raise ValueError('overflow/simulator fault/illegal/unsupported transition')
   self.visible_refusal_nonzero_decisions+=int((no[:,:,62]!=0).sum())
   natural=ni[:,2]>=0;truncated=ni[:,9]!=0;done=natural|truncated
   # Natural horizon only; truncation is always reported, never counted as a win.
   if truncated.any():raise ValueError('unexpected administrative truncation in fixed8 training')
   rewards=r[self.rows,self.seat]
   # Potential shaping from visible tower HP; zero terminal potential.
   def potential(z):return (z[:,36]/z[:,37].clamp_min(1)+z[:,38]/z[:,39].clamp_min(1)+z[:,40]/z[:,41].clamp_min(1)-z[:,42]/z[:,43].clamp_min(1)-z[:,44]/z[:,45].clamp_min(1)-z[:,46]/z[:,47].clamp_min(1))/3
   shaped=rewards+c['reward_shaping']*(c['gamma']*torch.where(natural,0.,potential(no[self.rows,self.seat]))-potential(x))
   for k,vv in (('reward',shaped),('done',done),('bootstrap',torch.zeros_like(shaped))):s[k].append(vv.clone())
   for k in range(len(self.games)):
    hit=natural&(self.assignment==k);self.games[k]+=hit.sum();self.wins[k]+=torch.where(ni[:,2]==2,.5,(ni[:,2]==self.seat).float())[hit].sum()
   self.natural_games+=int(natural.sum());ids=torch.where(done)[0]
   if len(ids):
    self.seat[ids]=1-self.seat[ids];self.assignment[ids]=torch.multinomial(pfsp(self.wins,self.games),len(ids),replacement=True)
   seeds=torch.randint(0,1000000,(c['envs'],),device=self.device,dtype=torch.int32)
   if self.device=='cuda':self.env.reset(seeds,done)
   elif len(ids):self.env.reset(seeds.numpy(),done.numpy())
   self.samples+=c['envs']
  o,m,_,_=self.outputs()
  with torch.no_grad():last=self.model(o[self.rows,self.seat],m[self.rows,self.seat])[1]
  coef=c['teacher_coef'] if self.updates<c['teacher_updates'] else 0.
  result=learner.update(self.model,self.optimizer,{k:torch.stack(v) for k,v in s.items()},last,c,coef)
  self.updates+=1;self.steps+=result['optimizer_steps'];self.losses.append(result['loss'])
  if self.updates%c['history_every']==0 and len(self.league)<c['history_limit']:
   # Construction must not perturb the sampling RNG stream.
   rng=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all() if self.device=='cuda' else []
   self.add_history();torch.set_rng_state(rng)
   if gpu:torch.cuda.set_rng_state_all(gpu)
   self.wins=torch.cat((self.wins,self.wins.new_zeros(1)));self.games=torch.cat((self.games,self.games.new_zeros(1)))
  return {'updates':self.updates,'samples':self.samples,'natural_games':self.natural_games,'neural_opponent_decisions':self.neural_opponent_decisions,'script_opponent_decisions':self.script_opponent_decisions,'learned_neural_opponent_decisions':self.learned_neural_opponent_decisions,'league_identities':self.identities,'visible_refusal_nonzero_decisions':self.visible_refusal_nonzero_decisions,'league_games':self.games.tolist(),'league_points':self.wins.tolist(),**result}
