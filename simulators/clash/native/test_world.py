"""Full native C ABI vs pinned Java differential qualification, never a timing benchmark.

Every read compares all observation/mask/info/reward entries. First divergence retains
complete arrays, action history, native debug state and raw Java world objects. No engine
state or encoded observation is copied into native transitions.
"""
from __future__ import annotations
from pathlib import Path
import argparse,ctypes,hashlib,json,sys,subprocess
import numpy as np
import jpype
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from game import make_vector,resolve
from crforge_env import VectorBattle,LadderConfig,ScriptedBot
from build import build

class Native:
 def __init__(self,library):
  self.lib=ctypes.CDLL(str(library));self.lib.cn_world_create.argtypes=[ctypes.c_int];self.lib.cn_world_create.restype=ctypes.c_void_p;self.lib.cn_world_free.argtypes=[ctypes.c_void_p];self.lib.cn_world_reset.argtypes=[ctypes.c_void_p,ctypes.c_int];self.lib.cn_world_step.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]
  ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');bp=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
  self.lib.cn_world_read.argtypes=[ctypes.c_void_p,ip,bp,ip,fp];self.lib.cn_world_debug.argtypes=[ctypes.c_void_p,ip,ctypes.c_int]
  self.p=self.lib.cn_world_create(0);assert self.p,'native allocation failed'
  self.observations=np.empty((2,7024),np.int32);self.masks=np.empty((2,2307),np.uint8);self.infos=np.empty(12,np.int32);self.rewards=np.empty(2,np.float32)
 def reset(self,seed):assert self.lib.cn_world_reset(self.p,int(seed))==0
 def step(self,actions,limit):return int(self.lib.cn_world_step(self.p,int(actions[0]),int(actions[1]),int(limit)))
 def read(self):return int(self.lib.cn_world_read(self.p,self.observations,self.masks,self.infos,self.rewards))
 def close(self):self.lib.cn_world_free(self.p)
 def debug(self):
  values=np.empty(100000,np.int32);n=int(self.lib.cn_world_debug(self.p,values,len(values)));v=list(map(int,values[:min(n,len(values))]));header_names=['tick','clock_ms','random','fault','live_count','timeline_ms','end_timer_ms','tiebreak_ms','king_wait_listed','king_activation_listed'];out=dict(zip(header_names,v[:10]));out['entities']=[];cursor=10
  for _ in range(out.get('live_count',0)):
   if cursor+2>len(v):break
   id,kind=v[cursor:cursor+2];cursor+=2;e={'id':id,'kind':kind};names=['row','side','x','y','state','deploy_countdown','delay','hp','reference','attack_ms','load_ms','dir_x','dir_y','flags','pending_flags','pending_damage','pending_duration','active'] if kind==5 else ['row','side','x','y','z','target','aim_x','aim_y','delay'] if kind==4 else ['countdown'];e.update(zip(names,v[cursor:cursor+len(names)]));cursor+=len(names);out['entities'].append(e)
  out['raw']=v;return out

def field(obj,name):
 cls=obj.getClass()
 while cls is not None:
  try:f=cls.getDeclaredField(name);f.setAccessible(True);return f.get(obj)
  except Exception:cls=cls.getSuperclass()
 raise KeyError(name)
def call(obj,name,default=None):
 try:return getattr(obj,name)()
 except Exception:return default

def java_debug(v):
 env=field(v._env,'envs')[0];standard=env.battle();world=standard.getWorld();battle=standard.getBattle();match=standard.getMatch();rows=[]
 J=jpype.JClass;World=J('org.crforge.core.battle.unit.WorldEntity');Character=J('org.crforge.core.battle.unit.CharacterEntity');Projectile=J('org.crforge.core.battle.projectile.ProjectileEntity');Area=J('org.crforge.core.battle.unit.AreaEffectEntity')
 for e in battle.getHolder().entities():
  row={'id':int(e.getId()),'class':str(e.getClass().getSimpleName()),'name':str(e.name())}
  if isinstance(e,World):
   view=e.getView();hp=e.getHitPoints();target=e.getTargeting();ref=target.getReference();row.update(row=str(e.getData().name()),side=int(e.side()),x=int(view.getX()),y=int(view.getY()),z=int(view.getZ()),state=int(view.getState()),alive=bool(view.isAlive()),flags=int(view.getFlags()),hp=None if hp is None else int(hp.getHitPoints()),maximum=None if hp is None else int(hp.getMaximum()),shield=None if hp is None else int(hp.getShield()),deploy_countdown=int(view.getDeployCountdown()),delay=call(view,'getDelay'),dir_x=int(view.getDirX()),dir_y=int(view.getDirY()),reference=None if ref is None else int(ref.getEntity().getId()),attack_ms=int(target.getAttackTimerMs()),load_ms=int(target.getLoadTimerMs()),pending_damage=call(view,'getPendingDamageAmount'),pending_duration=call(view,'getPendingDamageDurationMs'),target_lost_ms=call(target,'getTargetLostTimerMs'),hit_started=bool(call(target,'isHitStarted',False)),active=sum((1<<slot) for slot in range(4) if e.isActive(slot)))
   if isinstance(e,Character):
    movement=e.getUnit().movement();route=movement.getRoute();row['movement']={name:call(movement,name) for name in ['getTargetX','getTargetY','getBlockCountdown','getRouteLeadsAway','getRouteDirX','getRouteDirY','getPushbackBudget','getPushbackInFlight','getPushX','getPushY','getPushCount']};row['movement']['route']=str(route)
  elif isinstance(e,Projectile):row.update(row=str(e.getData().name()),side=int(e.side()),x=int(e.getX()),y=int(e.getY()),z=int(e.getZ()),aim_x=int(e.getAimX()),aim_y=int(e.getAimY()),aim_z=int(e.getAimZ()),target=None if e.getTarget() is None else int(e.getTarget().getId()),delay=int(e.getDelayMs()),released=bool(e.isReleased()),deploy_index=int(e.getDeployIndex()))
  elif isinstance(e,Area):row.update(row=str(e.getData().name()),side=int(e.side()),x=int(e.getX()),y=int(e.getY()),countdown=int(e.getCountdown()),radius=int(e.radius()))
  rows.append(row)
 public_plays=[]
 for side,plays in enumerate(field(env,'playsPlaced')):
  for play in plays:
   public_plays.append({'side':side,'card':int(field(play,'card')),'tick':int(field(play,'tick')),'point_tick':int(field(play,'pointTick')),'point_shown':bool(field(play,'pointShown')),'projectiles':[{'id':int(p.getId()),'released':bool(p.isReleased())} for p in field(play,'projectiles')]})
 return {'public_plays':public_plays,'tick':int(battle.getTick()),'clock_ms':int(match.getTimeline().getTimeMs()),'random':int(world.getRandom().getState()),'entities':rows,'own_elixir':[int(match.side(s).getElixir()) for s in range(2)]}

def mismatch_fields(native,v,schema):
 out=[];headers={int(index):name for name,indices in schema['header'].items() for index in indices};entity_fields={index:name for name,index in schema['entity'].items()};info_fields={index:name for name,index in schema['info'].items()}
 for kind,actual,expected in [('observation',native.observations,v.observations[0]),('mask',native.masks,v.masks[0]),('info',native.infos,v.infos[0]),('reward',native.rewards,v.rewards[0])]:
  where=np.argwhere(actual!=expected)
  for p in where[:60]:
   idx=tuple(map(int,p));item={'array':kind,'index':list(idx),'native':actual[idx].item(),'java':expected[idx].item()}
   if kind=='observation':
    seat,col=idx;item['seat']=seat
    if col<224:item['field']=headers.get(col,f'reserved_{col}')
    else:row,offset=divmod(col-224,34);item.update(entity_row=row,field=entity_fields[offset],native_entity_id=int(actual[seat,224+row*34]),java_entity_id=int(expected[seat,224+row*34]),native_entity_row_vocab=int(actual[seat,224+row*34+3]),java_entity_row_vocab=int(expected[seat,224+row*34+3]))
   elif kind=='info':item['field']=info_fields[idx[0]]
   elif kind=='mask':
    seat,action=idx;item['seat']=seat
    if 0<action<2305:slot,tile=divmod(action-1,576);r,c=divmod(tile,18);item.update(slot=slot,col=c,row=r)
   out.append(item)
  if len(where)>60:out.append({'array':kind,'additional_mismatches':int(len(where)-60)})
 return out

class Diverged(Exception):pass

def oracle_vector(config,limit,decision_ticks=5):
 if limit==0 and decision_ticks==5:return make_vector(config)
 # make_vector first verifies all frozen source/data/jar pins. A second oracle changes only
 # its documented artificial max_ticks for the explicit truncation fixture, or the decision
 # cadence for a match form that changes only that (match_forms.py `cadence10`).
 checked=make_vector(config);checked.close()
 v=VectorBattle(1,1,install=resolve(config['install']),tables=resolve(config['tables']),config=LadderConfig(decision_ticks=decision_ticks,level=11,tower_level=11,max_ticks=limit,mask_version=2));v.set_autoreset(False);return v

def trace_case(args,library,config):
 payload=json.loads(args.trace_history.read_text());seed=int(payload['seed']);history=payload['history']
 metadata=json.loads(library.with_suffix('.json').read_text());drift=[name for name,want in metadata['source_sha256'].items() if Path(name).suffix in ('.hpp','.h') and (not (HERE/name).is_file() or hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=want)]
 if drift:raise RuntimeError('diagnostic bridge requires headers matching the library; rebuild first: '+str(drift))
 native=Native(library)
 # Diagnostic wrapper splits the exact advance() decision framing from its existing per-tick ABI.
 # It is emitted under new evidence and never edits/replaces production source or transitions.
 bridge=args.out/'trace_bridge.cpp';bridge.write_text('#include "'+str(HERE/'world_impl.hpp')+'"\nusing namespace clash_native;\nextern "C" void cn_trace_begin(void*p,int a0,int a1){auto&w=*static_cast<NativeWorld*>(p);w.rendered=false;w.illegalNow[0]=w.illegalNow[1]=0;w.lastRunRefusal[0]=w.lastRunRefusal[1]=0;w.firstSide=w.episodeStep&1;w.apply_action(w.firstSide,w.firstSide?a1:a0);w.apply_action(1-w.firstSide,w.firstSide?a0:a1); }\nextern "C" void cn_trace_finish(void*p){auto&w=*static_cast<NativeWorld*>(p);++w.episodeStep;w.justReset=false;}\nextern "C" int cn_trace_extra(void*p,int*out){auto&w=*static_cast<NativeWorld*>(p);int n=0;for(int j=0;j<w.holder.live_count;++j){int i=w.holder.live[j];if(!w.is_unit(i))continue;out[n++]=w.holder.ids[i];out[n++]=w.targets[i].targetLostTimerMs;out[n++]=w.targets[i].hitStarted;out[n++]=int(w.entities[i].flags>>32);out[n++]=int(w.entities[i].pendingFlags>>32);}return n;}\n')
 bridge.write_text(bridge.read_text()+'extern "C" int cn_trace_plays(void*p,int*out){auto&w=*static_cast<NativeWorld*>(p);int n=0;for(int side=0;side<2;++side){int count=w.observer.play_count[side];for(int i=(count>2?count-2:0);i<count;++i){const auto&p=w.observer.plays[side][i];out[n++]=side;out[n++]=p.play.card;out[n++]=p.play.tick;out[n++]=p.point_tick;out[n++]=p.point_shown;out[n++]=p.play.deploy_index;out[n++]=p.play.projectile_count;}}return n;}\n')
 trace_lib=args.out/'trace_bridge.dylib';subprocess.run(['clang++','-std=c++20','-O3','-fPIC','-fno-fast-math','-ffp-contract=off','-dynamiclib',str(bridge),'-o',str(trace_lib)],check=True)
 t=ctypes.CDLL(str(trace_lib));t.cn_trace_begin.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int];t.cn_trace_finish.argtypes=[ctypes.c_void_p];t.cn_trace_extra.argtypes=[ctypes.c_void_p,np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')];t.cn_trace_plays.argtypes=[ctypes.c_void_p,np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')];native.lib.cn_world_tick.argtypes=[ctypes.c_void_p]
 snapshots=[]
 with make_vector(config) as v:
  v.reset([seed],config['rules']['deck'],config['rules']['deck']);native.reset(seed);native.read();env=field(v._env,'envs')[0];methods={}
  def invoke(name,*values):
   if name not in methods:
    method=env.getClass().getDeclaredMethod(name,*[jpype.JClass('java.lang.Integer').TYPE for _ in values]);method.setAccessible(True);methods[name]=method
   return methods[name].invoke(env,*[jpype.JInt(x) for x in values])
  def snapshot():
   raw=native.debug();extra=np.empty(10000,np.int32);n=t.cn_trace_extra(native.p,extra);extras={int(extra[i]):{'target_lost_ms':int(extra[i+1]),'hit_started':bool(extra[i+2]),'flags_high':int(extra[i+3]),'pending_flags_high':int(extra[i+4])} for i in range(0,n,5)}
   for e in raw['entities']:e.update(extras.get(e['id'],{}))
   play_rows=np.empty(128,np.int32);pn=t.cn_trace_plays(native.p,play_rows);raw['public_plays']=[dict(zip(['side','card','tick','point_tick','point_shown','deploy_index','projectile_count'],map(int,play_rows[i:i+7]))) for i in range(0,pn,7)]
   snapshots.append({'tick':int(raw['tick']),'native':raw,'java':java_debug(v)})
  for decision,actions in enumerate(history):
   tick=int(v.infos[0,0])
   if tick+5<=args.trace_start:
    native.step(actions,0);v.step(np.array([actions],np.int32));native.read();continue
   snapshot();invoke('beginDecision');first=decision%2;invoke('apply',first,actions[first]);invoke('apply',1-first,actions[1-first]);t.cn_trace_begin(native.p,*map(int,actions))
   for k in range(5):
    native.lib.cn_world_tick(native.p);invoke('advance');snapshot()
    if snapshots[-1]['tick']>=args.trace_end:break
   t.cn_trace_finish(native.p);invoke('endDecision');native.read()
   if snapshots[-1]['tick']>=args.trace_end:break
 native.close();summary={'status':'diagnostic_trace','seed':seed,'trace_start':args.trace_start,'trace_end':args.trace_end,'library':str(library),'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'snapshots':snapshots};(args.out/'tick-trace.json').write_text(json.dumps(summary,indent=2,default=lambda v:int(v))+'\n');print(json.dumps({k:summary[k] for k in ['status','seed','trace_start','trace_end','library']}));return 0

def main():
 ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--stage',choices=['reset','wait','cards','scripted','selfplay','random','limit','illegal','full'],default='full');ap.add_argument('--seeds',default='0,1,-1,771');ap.add_argument('--max-decisions',type=int,default=1400);ap.add_argument('--limit',type=int,default=103);ap.add_argument('--library',type=Path);ap.add_argument('--policy-seed',type=int,default=78917);ap.add_argument('--trace-history',type=Path);ap.add_argument('--trace-start',type=int,default=895);ap.add_argument('--trace-end',type=int,default=900);ap.add_argument('--form',default='full',help='native match form (match_forms.py); the Java oracle takes a cadence but no timeline override');args=ap.parse_args()
 import match_forms
 form=match_forms.variant(args.form)
 if form['timeline_override']:raise SystemExit(f'the pinned Java oracle has no timeline override: {args.form} cannot be compared here (use cadence10 for its decision cadence)')
 if args.library and args.form!='full':raise SystemExit('--library is the pinned form only')
 args.out.mkdir(parents=True,exist_ok=False)
 library=args.library or build(args.form);config=json.loads((HERE.parent/'configs/profile-trimmed.json').read_text());config.update(envs=1,threads=1)
 if args.trace_history:return trace_case(args,library,config)
 native=Native(library);schema=json.loads((HERE/'generated-v2/task-data.json').read_text())['schema'];seeds=list(map(int,args.seeds.split(',')));results=[];all_history=[];rng=np.random.default_rng(args.policy_seed);total_frames=0;natural=0;truncations=0;card_counts={};max_entities=0;overflow_observations=0
 stages=['reset','wait','cards','scripted','selfplay','random','limit','illegal'] if args.stage=='full' else [args.stage]
 manifest={'format':1,'match_form':args.form,'decision_ticks':form['decision_ticks'],'library':str(library),'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'stage':args.stage,'seeds':seeds,'policy_seed':args.policy_seed,'engine_pin':'f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de','test_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'timing_claims':False};(args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 def check(v,stage,seed,decision,history):
  nonlocal total_frames,max_entities,overflow_observations
  code=native.read();first_read=[native.observations.copy(),native.masks.copy(),native.infos.copy(),native.rewards.copy()];second_code=native.read();read_stable=code==second_code and all(np.array_equal(before,after) for before,after in zip(first_read,[native.observations,native.masks,native.infos,native.rewards]));diff=mismatch_fields(native,v,schema)
  if code or diff or not read_stable:
   np.savez(args.out/'first-divergence.npz',native_obs=native.observations,java_obs=v.observations[0],native_mask=native.masks,java_mask=v.masks[0],native_info=native.infos,java_info=v.infos[0],native_reward=native.rewards,java_reward=v.rewards[0])
   report={'status':'diverged','stage':stage,'seed':seed,'decision':decision,'tick_java':int(v.infos[0,0]),'tick_native':int(native.infos[0]),'fault_code':code,'repeated_read_stable':bool(read_stable),'differences':diff,'history':history,'native_raw':native.debug(),'java_raw':java_debug(v)};(args.out/'first-divergence.json').write_text(json.dumps(report,indent=2,default=lambda x:int(x))+'\n');print(json.dumps({k:report[k] for k in ['status','stage','seed','decision','tick_java','tick_native','fault_code','differences']}));raise Diverged
  total_frames+=1;max_entities=max(max_entities,int((native.observations[:,48]+native.observations[:,49]).max()));overflow_observations+=int(np.any(native.observations[:,49]>0))
 try:
  for stage in stages:
   limit=args.limit if stage=='limit' else 0
   with oracle_vector(config,limit,form['decision_ticks']) as v:
    bot=ScriptedBot().bind(v)
    cases=[(seed,None) for seed in seeds]
    if stage=='cards':
     cases=[]
     for card in config['rules']['deck']:
      for seed in range(1000,1100):
       v.reset([seed],config['rules']['deck'],config['rules']['deck']);hand=[v.vocabulary[int(i)] for i in v.observations[0,0,7:11]]
       if card in hand:cases.append((seed,card));break
      else:raise AssertionError('card never appeared in opening hand')
    for seed,card in cases:
     v.reset([seed],config['rules']['deck'],config['rules']['deck']);native.reset(seed);history=[];check(v,stage,seed,0,history);played=False
     count=0 if stage=='reset' else 6 if stage=='illegal' else min(args.max_decisions,360) if stage=='cards' else args.max_decisions
     for decision in range(count):
      if v.dones[0]:break
      actions=[0,0]
      if stage=='illegal':actions=[[-1,0],[2307,0],[2305,0],[1,0],[0,-1],[0,2306]][decision]
      elif stage in ['scripted','selfplay']:actions=[bot(v.observations[0,s],v.masks[0,s],rng) if s==0 or stage=='selfplay' else 0 for s in range(2)]
      elif stage=='random':actions=[int(rng.choice(np.flatnonzero(v.masks[0,s]))) for s in range(2)]
      elif stage=='cards' and not played and int(v.infos[0,0])>=90:
       for slot,idx in enumerate(v.observations[0,0,7:11]):
        if int(idx)>=0 and v.vocabulary[int(idx)]==card:
         legal=np.flatnonzero(v.masks[0,0,1+slot*576:1+(slot+1)*576]);target=(25 if card in ['Fireball','Arrows','Zap'] else 14)*18+3
         if len(legal):actions[0]=int(1+slot*576+legal[np.argmin(abs(legal-target))]);played=True
         break
      for seat,action in enumerate(actions):
       if 0<action<2305 and int(v.observations[0,seat,7+(action-1)//576])>=0:
        name=v.vocabulary[int(v.observations[0,seat,7+(action-1)//576])];card_counts[name]=card_counts.get(name,0)+1
      history.append(actions);native.step(actions,limit);v.step(np.array([actions],np.int32));check(v,stage,seed,decision+1,history)
     done=bool(v.dones[0]);is_trunc=bool(v.infos[0,9]);is_natural=done and not is_trunc and int(v.infos[0,5])==0
     natural+=is_natural;truncations+=is_trunc
     if stage in ['wait','scripted','selfplay','limit'] and not done:raise AssertionError(f'{stage} did not reach required boundary in {count} decisions')
     if stage=='cards' and (not played or int(v.observations[0,1,114])!=1):raise AssertionError(f'{card} was not placed exactly once')
     results.append({'stage':stage,'seed':seed,'card':card,'decisions':len(history),'terminal':is_natural,'truncated':is_trunc,'winner':int(v.infos[0,2]),'final_tick':int(v.infos[0,0])});all_history.append({'stage':stage,'seed':seed,'card':card,'actions':history});(args.out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
  summary={'status':'passed','frames':total_frames,'compared_observation_ints':total_frames*2*7024,'compared_mask_bytes':total_frames*2*2307,'natural_terminals':int(natural),'truncations':int(truncations),'card_requests':card_counts,'max_visible_entities':max_entities,'entity_overflow_observations':overflow_observations,'cases':results,'full_task_qualified':args.stage=='full','runtime_java_dependency':False,'repeated_reads_checked':True,'timing_claims':False}
 except Diverged:summary={'status':'diverged','frames_before_divergence':total_frames,'cases_completed':results,'full_task_qualified':False}
 finally:native.close()
 (args.out/'histories.json').write_text(json.dumps(all_history)+'\n');(args.out/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary));return 0 if summary['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
