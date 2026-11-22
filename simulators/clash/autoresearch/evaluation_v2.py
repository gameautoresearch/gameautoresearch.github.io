"""Visible-opponent qualification with separate natural/policy/engine/infra outcomes."""
import argparse,itertools,json,time,contextlib,hashlib
from pathlib import Path
import numpy as np
from .common import DATA,MATCH_FORM,ROOT,dump,digest,setup_native
from .arena import PolicyProcess
from .visible_teacher import PublicTeacher
from .readiness_contract import admit,INTERFACE

PUBLIC_SEEDS=list(range(2026110901,2026110909))

def classify(exc):
 text=str(exc)
 if text.startswith('entrant'):return 'entrant_admission_failure'
 if any(word in text for word in ('bundle','parameter cap','tensor cap','weight','greedy','stateful','cross-row','policy import','engine/source binding')):return 'entrant_admission_failure'
 if 'administrative' in text or 'horizon watchdog' in text:return 'administrative_cap'
 if 'deadline' in text or 'latency budget' in text:return 'policy_timeout'
 if 'illegal' in text:return 'illegal_action'
 if 'process exited' in text:return 'policy_process_failure'
 if 'simulator fault' in text or 'overflow' in text:return 'simulator_fault'
 return 'infrastructure_fault'

class Actor:
 def __init__(self,spec,data,seed,development,uid=65520):
  self.spec=spec;self.latencies=[];self.decisions=0;self.process=None;self.reasons={};self.game_bots={}
  if spec['kind']=='checkpoint':
   self.admission=admit(spec['bundle']);self.process=PolicyProcess(spec['bundle'],development,uid=uid,diagnostics=True)
  elif spec['kind'] in ('teacher','intermediate'):self.bot=PublicTeacher(spec['kind'],data)
  else:
   from evaluate import PublicActor
   def make(game_seed):
    bot=PublicActor(spec['kind'],game_seed,data=data)
    if spec['name']=='pressure':
     for k,v in {'attack_elixir':40000,'support_elixir':30000,'back_row':8,'spell_targets':2,'spell_worth':1.0}.items():setattr(bot.bot,k,v)
    return bot
   self.make_bot=make;self.bot=None
 def choose(self,x,m,contexts=None):
  if self.process:
   start=time.perf_counter();result=self.process.choose(x,m);self.latencies.append((time.perf_counter()-start)*1000);self.decisions+=len(x);return result
  result=[]
  for j,(o,mask) in enumerate(zip(x,m)):
   if self.bot is not None:bot=self.bot
   else:
    context=contexts[j] if contexts is not None else (j,0)
    if context not in self.game_bots:self.game_bots[context]=self.make_bot(int(digest([20261109,self.spec['name'],context[0]])[:8],16))
    bot=self.game_bots[context]
   start=time.perf_counter();a=int(bot(o,mask));elapsed=time.perf_counter()-start;self.latencies.append(elapsed*1000);self.decisions+=1
   if elapsed>INTERFACE['trusted_opponent_inference']['per_decision_deadline_seconds']:raise RuntimeError('trusted opponent latency budget')
   if not 0<=a<2307 or not mask[a]:raise ValueError('illegal trusted opponent action')
   reason=getattr(bot,'last_reason','scripted');self.reasons[reason]=self.reasons.get(reason,0)+1;result.append(a)
  return result
 def close(self):
  if self.process:self.process.close()
 def summary(self):
  return {'decisions':self.decisions,'calls':len(self.latencies),'latency_measurement':'warm batch wall ms after admission probes; cold startup is included in full worker life' if self.process else 'single-seat wall ms','latency_ms':{'mean':float(np.mean(self.latencies)),'p95':float(np.quantile(self.latencies,.95)),'max':float(max(self.latencies))} if self.latencies else None,'decision_reasons':self.reasons,'boundary':self.process.boundary if self.process else {'trusted_source':True,'inputs':'single-seat public arrays only','search':False}}

def probe(actor,obs,masks):
 if not actor.process:return None
 p=actor.process;a=p.choose(obs,masks);d={k:np.asarray(v) for k,v in p.last_diagnostics.items()};b=p.choose(obs,masks)
 if a!=b or any(not np.allclose(v,np.asarray(p.last_diagnostics[k]),rtol=1e-5,atol=1e-6) for k,v in d.items()):raise ValueError('entrant repeated-input statefulness')
 order=np.arange(len(obs))[::-1];c=p.choose(obs[order],masks[order])
 if list(reversed(c))!=a:raise ValueError('entrant batch-permutation disagreement')
 for i in range(len(obs)):
  solo=p.choose(obs[i:i+1],masks[i:i+1])
  if solo!=a[i:i+1]:raise ValueError('entrant single-row action disagreement')
  for k,v in d.items():
   if not np.allclose(np.asarray(p.last_diagnostics[k]),v[i:i+1],rtol=1e-5,atol=1e-6):raise ValueError('entrant cross-row or stateful probability output')
 return {'repeated_actions':True,'repeated_probability_probes':True,'batch_permutation':True,'single_row_agreement':True,'source_review':actor.admission['source_review'],'arbitrary_Python_purity_proven':False}

def screen(specs,out,seeds=PUBLIC_SEEDS,development=False,max_seconds=900,diagnostics=False,interface=None,pair_names=None,simulation_threads=4,uid_base=65520,trajectory_digest=False):
 setup_native()
 from cpu_batch import NativeCPUBatch
 data=json.loads(DATA.read_text());out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();rows=[];actors={};latencies={};probes={};errors=[];failure_context={}
 interface=INTERFACE if interface is None else interface
 pairs_to_run=[(l,r) for l,r in itertools.combinations(specs,2) if pair_names is None or (l['name'],r['name']) in pair_names]
 if not pairs_to_run or len({s['name'] for s in specs})!=len(specs) or not seeds or len(set(seeds))!=len(seeds):raise ValueError('invalid unique panel/seeds/pairs')
 dump(out/'plan.json',{'public_engine_seeds':seeds,'opponents':specs,'interface':interface,'pair_games':2*len(seeds),'max_seconds':max_seconds,'development_only':development,'execution_diagnostics':diagnostics})
 try:
  for i,spec in enumerate(specs):actors[spec['name']]=Actor(spec,data,20261109,development,uid=uid_base-i)
  for left,right in pairs_to_run:
   failure_context={'pair':[left['name'],right['name']]}
   for actor in actors.values():actor.game_bots.clear()
   jobs=[(i,seed,seat) for i,seed in enumerate(seeds) for seat in (0,1)];env=NativeCPUBatch([v[1] for v in jobs],threads=simulation_threads,form=MATCH_FORM);active=set(range(len(jobs)));counts=np.zeros(len(jobs),np.int32);reject=np.zeros((len(jobs),2),np.int64)
   if trajectory_digest:
    trajectories=[hashlib.sha256() for _ in jobs]
    def hash_public(i):
     for value in (env.observations[i],env.masks[i],env.infos[i],env.rewards[i]):trajectories[i].update(np.ascontiguousarray(value).tobytes())
    for i in range(len(jobs)):hash_public(i)
   if diagnostics:
    from .refusal_diagnostics import DecisionCounters,visible_own_plays
    observers=[[DecisionCounters(),DecisionCounters()] for _ in jobs]
   try:
    for spec in (left,right):
     actor=actors[spec['name']]
     if actor.process and spec['name'] not in probes:
      probes[spec['name']]=probe(actor,env.observations[:4,0].copy(),env.masks[:4,0].copy())
    while active:
     if time.monotonic()-started>max_seconds:raise TimeoutError('qualification controller administrative deadline')
     ids=sorted(active);actions=np.zeros((len(jobs),2),np.int32)
     for spec,flip in ((left,False),(right,True)):
      sides=[1-jobs[i][2] if flip else jobs[i][2] for i in ids]
      x=np.stack([env.observations[i,s] for i,s in zip(ids,sides)]);m=np.stack([env.masks[i,s] for i,s in zip(ids,sides)])
      failure_context={'pair':[left['name'],right['name']],'actor':spec['name'],'public_seed_indices':[jobs[i][0] for i in ids],'actor_seats':sides}
      chosen=actors[spec['name']].choose(x,m,[(jobs[i][1],s) for i,s in zip(ids,sides)])
      for i,s,a in zip(ids,sides,chosen):actions[i,s]=a
     if diagnostics:before_obs=env.observations.copy();before_masks=env.masks.copy()
     failure_context={'pair':[left['name'],right['name']],'phase':'engine-advance','active_public_seed_indices':[jobs[i][0] for i in ids]};env.step(actions)
     for i in ids:
      counts[i]+=1;info=env.infos[i];reject[i]+=env.observations[i,:,62]!=0
      if trajectory_digest:trajectories[i].update(actions[i].tobytes());hash_public(i)
      if diagnostics:
       for side in (0,1):
        old,new,cards=visible_own_plays(before_obs[i,1-side],env.observations[i,1-side]);observers[i][side].observe(before_obs[i,side],before_masks[i,side],int(actions[i,side]),env.observations[i,side],own_placed_before=old,own_placed_after=new,own_placed_cards=cards)
      if env.observations[i,:,49].any():raise ValueError('entity overflow')
      if info[5] or info[11]:raise ValueError('simulator fault or unsupported transition')
      if info[7] or info[8]:raise ValueError('illegal engine action')
      if info[9] or counts[i]>3000:raise TimeoutError('administrative match cap')
      if info[2]>=0:
       idx,_,side=jobs[i];winner=int(info[2]);points=.5 if winner==2 else float(winner==side)
       row={'left':left['name'],'right':right['name'],'public_seed_index':idx,'left_seat':side,'natural':True,'winner':winner,'left_points':points,'decisions':int(counts[i]),'visible_refusal_nonzero_decisions_by_seat':reject[i].tolist(),'administrative_cap':False,'illegal_action':False,'simulator_fault':False,'infrastructure_fault':False}
       if diagnostics:row['execution_diagnostics']={left['name']:observers[i][side].summary(),right['name']:observers[i][1-side].summary()}
       if trajectory_digest:row['public_trajectory_sha256']=trajectories[i].hexdigest()
       rows.append(row);active.remove(i)
    dump(out/'progress.json',{'completed_games':len(rows),'last_pair':[left['name'],right['name']]})
   finally:env.close()
 except Exception as exc:
  failure='administrative_cap' if isinstance(exc,TimeoutError) else classify(exc);errors.append({'kind':failure,'error':type(exc).__name__+': '+str(exc),'context':failure_context});dump(out/'failure.json',errors[-1])
 finally:
  for name,actor in actors.items():latencies[name]=actor.summary();actor.close()
 expected=len(pairs_to_run)*2*len(seeds);pairs=[]
 for left,right in pairs_to_run:
  records=[r for r in rows if r['left']==left['name'] and r['right']==right['name']];valid=len(records)==2*len(seeds)
  per_unit=[np.mean([r['left_points'] for r in records if r['public_seed_index']==i]) for i in range(len(seeds))] if valid else []
  rng=np.random.default_rng(20261109);boot=[float(np.mean(rng.choice(per_unit,len(per_unit),replace=True)))*100 for _ in range(2000)] if valid else []
  pairs.append({'left':left['name'],'right':right['name'],'games':len(records),'natural_wins':sum(r['left_points']==1 for r in records),'natural_ties':sum(r['left_points']==.5 for r in records),'natural_losses':sum(r['left_points']==0 for r in records),'left_points_0_100':float(np.mean([r['left_points'] for r in records])*100) if valid else None,'paired_seed_interval95':np.quantile(boot,[.025,.975]).tolist() if valid else None,'by_left_seat_0_100':{str(s):float(np.mean([r['left_points'] for r in records if r['left_seat']==s])*100) for s in (0,1)} if valid else {}})
 result={'complete':not errors and len(rows)==expected,'games':len(rows),'expected_games':expected,'natural_games':len(rows),'failures':errors,'pairs':pairs,'latencies':latencies,'entrant_probes':probes,'rows':rows,'seconds':time.monotonic()-started,'development_only':development,'interface_sha256':digest(interface),'execution_diagnostics':diagnostics,'harness_qualified':not errors and len(rows)==expected,'learned_reference_qualified':False,'strong_panel_qualified':False,'superhuman_evidence':False,'reference_normalized_score':None}
 dump(out/'result.json',result);return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--specs',required=True);p.add_argument('--out',required=True);p.add_argument('--development',action='store_true');p.add_argument('--max-seconds',type=int,default=900);a=p.parse_args()
 lock=open(ROOT/'evidence/local-performance.lock','a+') if a.development else contextlib.nullcontext()
 with lock:
  if a.development:
   import fcntl
   fcntl.flock(lock,fcntl.LOCK_EX)
  r=screen(json.loads(Path(a.specs).read_text()),a.out,development=a.development,max_seconds=a.max_seconds)
 print(json.dumps({k:v for k,v in r.items() if k not in ('rows','latencies')}))
if __name__=='__main__':main()
