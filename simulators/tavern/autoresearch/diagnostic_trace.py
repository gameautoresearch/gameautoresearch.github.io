"""Opt-in bounded action traces without changing native rules, RNG or observations.

Only failing lobbies are written. New diagnostic releases use this equally for
both training arms. Native/source checkpoint identity remains unchanged.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import taverngym as tg
from contract import write,sha
MASK64=(1<<64)-1

def splitmix64(z):
 z=(int(z)+0x9E3779B97F4A7C15)&MASK64;z=((z^(z>>30))*0xBF58476D1CE4E5B9)&MASK64;z=((z^(z>>27))*0x94D049BB133111EB)&MASK64;return z^(z>>31)

def observation_hash(obs):
 h=hashlib.sha256()
 for name in ('features','cards','action_mask','ready'):
  value=np.ascontiguousarray(obs[name]);h.update(name.encode());h.update(str((value.shape,value.dtype.str)).encode());h.update(value.tobytes())
 return h.hexdigest()

def install(out,max_steps=6000):
 original=tg.BatchEnv;out=Path(out)
 class RecordingBatch:
  def __init__(self,n_games,seed=0,controlled=(0,),config=None,threads=1):
   self.env=original(n_games,seed=seed,controlled=controlled,config=config,threads=threads);self.seed=int(seed);self.config=config;self.controlled_seats=list(controlled);self.game_count=int(n_games);self.k=len(controlled);self.actions=[];self.active=[];self.last_obs=None
  def __getattr__(self,name):return getattr(self.env,name)
  def observe(self,*args,**kwargs):self.last_obs=self.env.observe(*args,**kwargs);return self.last_obs
  def observe_ready(self,*args,**kwargs):self.last_obs=self.env.observe_ready(*args,**kwargs);return self.last_obs
  def step(self,actions,active_games=None):
   if len(self.actions)>=max_steps:raise RuntimeError('Diagnostic trace bound reached')
   self.actions.append(np.asarray(actions,dtype=np.int64).copy());self.active.append(np.ones(self.game_count,bool) if active_games is None else np.asarray(active_games,bool).copy())
   reward,done,info=self.env.step(actions,active_games=active_games)
   rejected=np.asarray(info.get('rejected',np.zeros_like(done)))
   bad=sorted(set((np.flatnonzero((done & (info['stop_reason']!=1)) | rejected.astype(bool))//self.k).tolist()))
   for game in bad:
    stem=f'cohort-{self.seed:016x}-game-{game}-step-{len(self.actions)}';folder=out/stem;folder.mkdir(parents=True,exist_ok=False)
    slots=self.last_obs.get('slots',np.arange(self.game_count*self.k));indices=np.flatnonzero(slots//self.k==game);pre={key:self.last_obs[key][indices].copy() for key in ['features','cards','action_mask','ready']}
    np.savez_compressed(folder/'actions.npz',actions=np.stack(self.actions)[:,game*self.k:(game+1)*self.k],active=np.stack(self.active)[:,game],**pre)
    meta=dict(schema='tavern-bounded-native-action-trace-v2',cohort_seed=self.seed,game_index=game,mapped_single_game_seed=(self.seed^splitmix64(game)^splitmix64(0))&MASK64,controlled=self.controlled_seats,config=self.config,steps=len(self.actions),max_steps=max_steps,native_source_hash=tg.build_info()['source_hash'],pre_step_observation_sha256=observation_hash(pre),post_step_state_sha256=hashlib.sha256(self.env.save(game).encode()).hexdigest(),stop_reason=info['stop_reason'][game*self.k:(game+1)*self.k].tolist(),placement=info['placement'][game*self.k:(game+1)*self.k].tolist(),done=done[game*self.k:(game+1)*self.k].tolist(),rejected=rejected[game*self.k:(game+1)*self.k].tolist(),observation_slots=(slots[indices]-game*self.k).tolist(),trace_sha256=sha(folder/'actions.npz'))
    write(folder/'trace.json',meta)
   return reward,done,info
 tg.BatchEnv=RecordingBatch
 return lambda:setattr(tg,'BatchEnv',original)

def replay(folder,out):
 folder=Path(folder);out=Path(out);out.mkdir(parents=True,exist_ok=False);meta=json.loads((folder/'trace.json').read_text())
 if meta['native_source_hash']!=tg.build_info()['source_hash'] or sha(folder/'actions.npz')!=meta['trace_sha256']:raise ValueError('Native source/trace identity differs')
 with np.load(folder/'actions.npz',allow_pickle=False) as z:actions=z['actions'];active=z['active']
 env=tg.BatchEnv(1,seed=meta['mapped_single_game_seed'],controlled=meta['controlled'],config=meta['config'],threads=1);pre_hash=None
 for index,(a,on) in enumerate(zip(actions,active)):
  obs=env.observe(copy=True,active_games=np.array([on]))
  if index==len(actions)-1:
   pre_hash=observation_hash(obs);(out/'before-final-native-step.json').write_text(env.save(0))
  reward,done,info=env.step(a,active_games=np.array([on]))
 post_hash=hashlib.sha256(env.save(0).encode()).hexdigest();rejection_matches='rejected' not in meta or info['rejected'].tolist()==meta['rejected'];result=dict(reproduced=pre_hash==meta['pre_step_observation_sha256'] and info['stop_reason'].tolist()==meta['stop_reason'] and info['placement'].tolist()==meta['placement'] and post_hash==meta['post_step_state_sha256'] and rejection_matches,pre_step_observation_matches=pre_hash==meta['pre_step_observation_sha256'],post_step_state_matches=post_hash==meta['post_step_state_sha256'],expected_stop_reason=meta['stop_reason'],observed_stop_reason=info['stop_reason'].tolist(),rejection_matches=rejection_matches,expected_rejected=meta.get('rejected'),observed_rejected=info['rejected'].tolist(),steps=len(actions),native_source_unchanged=True,scope='Exact pre-native-step boundary plus actions, not internal mid-step crash memory')
 write(out/'result.json',result)
 if not result['reproduced']:raise ValueError('Deterministic trace replay mismatch')
 return result
