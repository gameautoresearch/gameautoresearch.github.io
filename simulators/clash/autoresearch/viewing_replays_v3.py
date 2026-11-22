"""Fresh public viewing movie and exact action/public-output/frame replay receipt."""
import argparse,json,fcntl
from pathlib import Path
import numpy as np
from .common import MATCH_FORM,ROOT,dump,sha,digest,engine_binding
from .bundle import verify
from .evaluation_v2 import Actor
from .refusal_diagnostics import DecisionCounters,visible_own_plays
from .release_v3 import PUBLIC_DEMO_SEEDS

def generate(bundle,out,seed,seat=0,opponent='teacher',label='Selected learned reference'):
 from simulators.clash.export_demo_replays import Recorder,NativeCPU,DATA,no_jvm,public_digest,public_outputs
 if seed not in PUBLIC_DEMO_SEEDS or seat not in (0,1) or opponent not in ('standard','pressure','teacher','intermediate'):raise ValueError('explicit public demo seed/seat/opponent required')
 out=Path(out);out.mkdir(parents=True,exist_ok=False);identity=verify(bundle)['identity'];data=json.loads(DATA.read_text());rec=Recorder(data);actors=[None,None];actors[seat]=Actor({'name':'candidate','kind':'checkpoint','bundle':str(bundle)},data,seed,True);actors[1-seat]=Actor({'name':opponent,'kind':'scripted' if opponent in ('standard','pressure') else opponent},data,seed,True);env=NativeCPU([seed]);frames=[];hashes=[];actions=[];observers=[DecisionCounters(),DecisionCounters()]
 try:
  frames.append(rec.frame(env,[]));hashes.append(public_digest(public_outputs(env)))
  for _ in range(3000):
   obs=env.observations[0].copy();mask=env.masks[0].copy();tick=int(env.infos[0,0]);a=[actors[p].choose(obs[p:p+1],mask[p:p+1],[(seed,p)])[0] for p in (0,1)]
   if any(not mask[p,a[p]] for p in (0,1)):raise ValueError('illegal viewing action')
   commands=[c for p in (0,1) if (c:=rec.command(a[p],obs[p],p,tick)) is not None];env.step([a]);actions.append(a);frame=rec.frame(env,commands);frames.append(frame);hashes.append(public_digest(public_outputs(env)))
   for p in (0,1):
    old,new,cards=visible_own_plays(obs[1-p],env.observations[0,1-p]);observers[p].observe(obs[p],mask[p],a[p],env.observations[0,p],own_placed_before=old,own_placed_after=new,own_placed_cards=cards)
   if any(int(env.infos[0,rec.s.info[k]]) for k in ('fault','not_modelled','illegal_0','illegal_1','truncated')):raise ValueError('invalid viewing episode')
   if frame['ended']:break
  else:raise ValueError('viewing game not natural')
 finally:
  env.close()
  for actor in actors:actor.close()
 env=NativeCPU([seed],form=MATCH_FORM)
 try:
  assert rec.frame(env,[])==frames[0] and public_digest(public_outputs(env))==hashes[0]
  for i,a in enumerate(actions,1):
   assert all(env.masks[0,p,a[p]] for p in (0,1));env.step([a]);assert public_digest(public_outputs(env))==hashes[i];assert rec.frame(env,frames[i]['commands'])==frames[i]
 finally:env.close()
 no_jvm();verify(bundle,identity);last=frames[-1];identifier=f'clash-strong-{seed}-seat{seat}';players=[{'name':opponent,'kind':'scripted'} for _ in (0,1)];players[seat]={'name':label,'kind':'neural'}
 movie={'schemaVersion':1,'gameId':'clash','id':identifier,'title':f'{label} vs {opponent}','subtitle':'Fresh public seed; greedy learned inference; independently replayed','players':players,'arena':{'width':18000,'height':32000,'decisionTicks':5,'tickMs':50,'deck':data['rules']['deck'],'towers':[e for e in frames[0]['entities'] if e['kind']=='tower']},'provenance':{'kind':'fresh-public-demo','engineCommit':'f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de','ruleset':data['task'],'dataVersion':'16.635.8','demoSeed':seed,'candidateSeat':seat,'submissionIdentity':identity,'verification':'Every complete public output and visual frame independently replayed; natural finish; no private evaluation schedule; no policy receives display/debug world state','verifiedFrames':len(frames),'trajectorySha256':digest(actions),'finalPublicSha256':hashes[-1],'policy':'Source-bound stateless neural greedy actor; display-only recorder is outside policy process; expert strength unqualified'},'frames':frames,'outcome':{'kind':'natural','winner':last['winner'],'draw':last['winner'] is None,'tick':last['tick'],'decisions':len(actions),'crowns':last['crowns']}}
 dump(out/(identifier+'.json'),movie);dump(out/'actions.json',actions);receipt={'id':identifier,'identity':identity,'seed':seed,'candidateSeat':seat,'opponent':opponent,'frames':len(frames),'actions':len(actions),'movie_sha256':sha(out/(identifier+'.json')),'trajectory_sha256':digest(actions),'all_public_outputs_and_visual_frames_replayed':True,'engine_binding_sha256':digest(engine_binding()),'execution_diagnostics_by_seat':[o.summary() for o in observers],'outcome':movie['outcome'],'private_schedule_used':False,'display_debug_state_given_to_actor':False};dump(out/'verification.json',receipt);return receipt

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--seat',type=int,default=0);p.add_argument('--opponent',default='teacher');p.add_argument('--label',default='Selected learned reference');a=p.parse_args()
 with (ROOT/'evidence/local-performance.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);print(json.dumps(generate(a.bundle,a.out,a.seed,a.seat,a.opponent,a.label),indent=2))
