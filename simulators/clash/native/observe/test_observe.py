"""Extract raw oracle game objects into independent native observer PODs, compare all outputs.
Java is used only for differential fixtures, never by the production observer.
"""
from pathlib import Path
import ctypes,json,sys,subprocess,hashlib
import numpy as np
import jpype
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from game import make_vector
from crforge_env import ScriptedBot

class Oracle:
 def __init__(self,v):
  self.v=v;self.J=jpype.JClass;self.cls=self.J('org.crforge.env.LadderEnv');self.fields={};self.methods={}
  f=v._env.getClass().getDeclaredField('envs');f.setAccessible(True);self.env=f.get(v._env)[0]
  self.standard=self.env.battle();self.world=self.standard.getWorld();self.battle=self.standard.getBattle();self.match=self.standard.getMatch();self.vocab=v._vocabulary
  self.World=self.J('org.crforge.core.battle.unit.WorldEntity');self.Char=self.J('org.crforge.core.battle.unit.CharacterEntity');self.Tower=self.J('org.crforge.core.battle.unit.TowerEntity');self.Projectile=self.J('org.crforge.core.battle.projectile.ProjectileEntity');self.Area=self.J('org.crforge.core.battle.unit.AreaEffectEntity')
  self.tracked=0;self.side_plays=[0,0];self.in_flight={};self.play_counts={};self.SpawnState=self.J('org.crforge.core.pathfinding.GridEntityState');self.saw=set()
 def field(self,name):
  if name not in self.fields:
   f=self.cls.class_.getDeclaredField(name);f.setAccessible(True);self.fields[name]=f
  return self.fields[name].get(self.env)
 def invoke(self,name,*args):
  if name not in self.methods:
   types=[self.J('java.lang.Integer').TYPE for _ in args];m=self.cls.class_.getDeclaredMethod(name,*types);m.setAccessible(True);self.methods[name]=m
  return self.methods[name].invoke(self.env,*[jpype.JInt(x) for x in args])
 def frame(self,reset=False):
  match=self.match;timeline=match.getTimeline();tick=int(self.battle.getTick())
  clock=np.array([tick,self.field('episodeStep'),timeline.getTimeMs(),timeline.getSection(),timeline.getSectionType(),timeline.sectionLeftMs(),timeline.getFullBarMs(),match.getStartCooldownTicks(),5,20,match.getWinner(),self.field('firstSide'),match.crowns(0),match.crowns(1),*list(self.field('illegalNow')),0,self.field('notModelled'),self.battle.getMode().isOver(),timeline.overtime(),self.field('truncated'),reset],np.int32)
  own=np.zeros((2,96),np.int32);towers=self.field('towers');waiting=self.field('waiting')
  for side in range(2):
   o=own[side];player=match.side(side);hand=player.getHand();slots=list(hand.slots());deck=player.deck();queue=list(hand.queue());o[0]=player.getElixir();o[1]=self.env.commands().promised(side)*10000;o[2]=hand.getCooldownMs()
   for k,index in enumerate(slots):
    o[3+k]=-1 if index<0 else self.vocab.id(deck.get(index).name());o[7+k]=0 if index<0 else deck.get(index).cost();o[11+k]=index>=0 and self.env.commands().isPending(side,index);o[15+k]=0 if index<0 else player.item(index).field()
   for k in range(8):o[19+k]=self.vocab.id(deck.get(k).name());o[27+k]=player.slotFlags(k);o[35+k]=-1 if player.evolutionCountUnknown(k) else player.evolutionCount(k);o[43+k]=11
   for k in range(4):o[51+k]=self.vocab.id(deck.get(queue[k]).name()) if k<len(queue) else -1
   o[55]=o[51]
   for k,tower in enumerate(towers[side]):o[56+k]=tower.getHitPoints().getHitPoints() if tower.getView().isAlive() else 0;o[62+k]=tower.getHitPoints().getMaximum()
   for at,name in [(68,'lastRefusal'),(69,'lastRunRefusal'),(70,'lastAbilityRunRefusal'),(71,'illegalTotal')]:o[at]=self.field(name)[side]
   count=0
   for pending in waiting:
    if pending.side()!=side or count==4:continue
    o[73+4*count:77+4*count]=[pending.card(),pending.x(),pending.y(),pending.runTick()];count+=1
   o[72]=count
  rows=[]
  for e in self.battle.getHolder().entities():
   r=np.zeros(64,np.int32);r[0]=e.getId();r[11]=-1;r[17]=-1;r[22]=-1;flags=1|8192|16384
   if isinstance(e,self.World):
    view=e.getView();d=e.getData();points=e.getHitPoints();t=e.getTargeting();reference=t.getReference();character=isinstance(e,self.Char);r[1]=1 if character else 2;r[2]=e.side();r[3]=self.vocab.id(d.name());r[4]=view.getX();r[5]=view.getY();r[6]=0 if points is None else points.getHitPoints();r[7]=0 if points is None else points.getMaximum();r[8]=0 if points is None else points.getShield();r[9]=view.getState();r[10]=e.level();r[11]=-1 if reference is None else self.world.entityOf(reference.getEntity()).getId();r[12]=view.getDeployCountdown() if character else 0;r[13]=t.getAttackTimerMs();r[14]=t.getLoadTimerMs();r[15]=e.getBuffs().speed(100);r[16]=e.getBuffs().hitSpeed(100)
    bits=view.getFlagBits();tags=int(view.getFlags());vals=[view.isAlive(),d.air(),d.building(),d.king(),e.isClone() if character else False,not e.isInactive() if isinstance(e,self.Tower) else False,bool(tags&bits.noMove()),bool(tags&bits.noAttack()),bool(tags&bits.charging()),e.hidden() if character else False,e.invisible() if character else False,e.underground() if character else False,view.getState()==self.SpawnState.WAITING_TO_DEPLOY,points is not None,True,False,False,character and d.flyingHeight()==0]
    flags=sum((1<<i) for i,val in enumerate(vals) if val);buffs=list(e.getBuffs().items());r[24]=len(buffs)
    for j,b in enumerate(buffs):r[25+4*j:29+4*j]=[self.vocab.buffId(b.getBuff().name()),b.getRemaining(),b.getBuff().speedMultiplier(),b.getBuff().hitSpeedMultiplier()]
   elif isinstance(e,self.Projectile):
    self.saw.add(str(e.getData().name()));r[1]=3;r[2]=e.side();r[3]=self.vocab.id(e.getData().name());r[4]=e.getX();r[5]=e.getY();r[10]=e.level();r[11]=-1 if e.getTarget() is None else e.getTarget().getId();r[18]=e.getAimX();r[19]=e.getAimY();r[22]=e.getDeployIndex()
   elif isinstance(e,self.Area):
    self.saw.add(str(e.getData().name()));r[1]=4;r[2]=e.side();r[3]=self.vocab.id(e.getData().name());r[4]=e.getX();r[5]=e.getY();r[10]=self.J('org.crforge.core.pathfinding.combat.PackedLevel').level(e.getPackedLevel());r[20]=e.getCountdown();r[21]=e.radius();flags|=16 if e.isClone() else 0
   else:raise AssertionError(type(e))
   r[23]=flags;rows.append(r)
  return clock,own,np.array(rows,np.int32).reshape(-1,64)
 def tick_events(self):
  plays=self.standard.getPlays();out=[]
  for i in range(self.tracked,len(plays)):
   p=plays.get(i)
   if p.result() is None or not p.result().placed():continue
   side=int(p.side());item=p.evolution();assert item is not None
   index=int(item.index());card=self.match.side(side).deck().get(index);cast=self.world.getRecords().card(item.spell().name());deploy=self.side_plays[side];self.side_plays[side]+=1
   projectiles=[e for e in self.battle.getHolder().entities() if isinstance(e,self.Projectile) and e.side()==side and e.getDeployIndex()==deploy]
   kind=1 if not cast.spell() else 3 if cast.projectile() is not None else 2
   row=[side,self.vocab.id(card.name()),self.vocab.id(cast.name()),item.field(),p.tick(),p.result().x(),p.result().y(),kind,item.cost(),item.spell().elixirProductionStopTimeMs(),deploy,len(projectiles),0,0];out.append(row)
   self.play_counts[str(card.name())]=self.play_counts.get(str(card.name()),0)+1
   if kind==3 and projectiles:self.in_flight[(side,deploy)]=projectiles
  self.tracked=len(plays);released=[]
  for key,projectiles in list(self.in_flight.items()):
   if any(p.isReleased() for p in projectiles):released.append([*key,self.battle.getTick()-1]);del self.in_flight[key]
  return np.array(out,np.int32).reshape(-1,14),np.array(released,np.int32).reshape(-1,3)

def main():
 out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);lib=out/'observe-tests.dylib';subprocess.run(['clang++','-std=c++20','-O2','-Wall','-Wextra','-pedantic','-fPIC','-shared',str(HERE/'test_bridge.cpp'),'-o',str(lib)],check=True)
 native=ctypes.CDLL(str(lib));ptr=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fptr=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
 native.cn_observer_create.argtypes=[ctypes.c_int];native.cn_observer_create.restype=ctypes.c_void_p;native.cn_observer_destroy.argtypes=[ctypes.c_void_p];native.cn_observer_tick.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int];native.cn_observer_frame.argtypes=[ctypes.c_void_p,ptr,ptr,ptr,ctypes.c_int,ptr,ptr,fptr]
 config=json.loads((HERE.parents[1]/'configs/profile-trimmed.json').read_text());config.update(envs=1,threads=1);rng=np.random.default_rng(8817);frames=0;terminal=0;cards={};saw=set();max_entities=0
 with make_vector(config) as v:
  bot=ScriptedBot().bind(v)
  for seed in [771,772]:
   v.reset([seed],config['rules']['deck'],config['rules']['deck']);oracle=Oracle(v);state=native.cn_observer_create(60000)
   try:
    for decision in range(1600):
     clock,own,entities=oracle.frame();result=np.empty((2,7024),np.int32);info=np.empty(12,np.int32);rewards=np.empty(2,np.float32)
     error=native.cn_observer_frame(state,clock,own,entities,len(entities),result,info,rewards);assert error==0,error
     mismatch=np.argwhere(result!=v.observations[0])
     if len(mismatch):
      np.savez(out/'mismatch.npz',actual=result,expected=v.observations[0],clock=clock,own=own,entities=entities)
      raise AssertionError((seed,decision,[(tuple(p),int(result[tuple(p)]),int(v.observations[0][tuple(p)])) for p in mismatch[:20]]))
     np.testing.assert_array_equal(info,v.infos[0]);np.testing.assert_array_equal(rewards,v.rewards[0]);frames+=1;max_entities=max(max_entities,len(entities))
     if oracle.env.done():terminal+=1;break
     actions=[bot(v.observations[0,seat],v.masks[0,seat],rng) if seat==0 else 0 for seat in range(2)]
     # Deliberately cast all three spells when affordable to exercise aim secrecy, Zap and history.
     if decision%7==0:
      for seat in range(2):
       for slot,card in enumerate(v.observations[0,seat,7:11]):
        if int(card)>=0 and (v.vocabulary[int(card)] in ['Fireball','Arrows','Zap'] or (v.vocabulary[int(card)] in ['Knight','Archer'] and oracle.play_counts.get(v.vocabulary[int(card)],0)<2)):
         legal=np.flatnonzero(v.masks[0,seat,1+slot*576:1+(slot+1)*576]);
         if len(legal):actions[seat]=int(1+slot*576+legal[min(len(legal)-1,14*18+3)]);break
     oracle.invoke('beginDecision');first=int(oracle.field('episodeStep'))%2;oracle.invoke('apply',first,actions[first]);oracle.invoke('apply',1-first,actions[1-first])
     for tick in range(5):
      if bool(oracle.invoke('over')):break
      oracle.invoke('advance');plays,released=oracle.tick_events();rate=oracle.match.getTimeline().getFullBarMs() if oracle.match.isLastTicked() else 0;native.cn_observer_tick(state,plays,len(plays),released,len(released),int(rate),10)
      if any(isinstance(e,oracle.Area) or isinstance(e,oracle.World) and len(e.getBuffs().items()) for e in oracle.battle.getHolder().entities()):
       # Additional observer calls on the same active game tick exercise stun buffs
       # without advancing or modifying the game's transitions.
       oracle.invoke('observe');clk,own_tick,ent_tick=oracle.frame();mid=np.empty((2,7024),np.int32);mid_info=np.empty(12,np.int32);mid_rewards=np.empty(2,np.float32)
       err=native.cn_observer_frame(state,clk,own_tick,ent_tick,len(ent_tick),mid,mid_info,mid_rewards);assert err==0,err
       np.testing.assert_array_equal(mid,v.observations[0]);np.testing.assert_array_equal(mid_info,v.infos[0]);np.testing.assert_array_equal(mid_rewards,v.rewards[0]);frames+=1;max_entities=max(max_entities,len(ent_tick))
     oracle.invoke('endDecision')
    else:raise AssertionError('no terminal')
   finally:native.cn_observer_destroy(state)
   for card,n in oracle.play_counts.items():cards[card]=cards.get(card,0)+n
   saw|=oracle.saw
 executable=out/'source-scenarios';subprocess.run(['clang++','-std=c++20','-O2','-Wall','-Wextra','-pedantic',str(HERE/'test_scenarios.cpp'),'-o',str(executable)],check=True)
 scenario=subprocess.run([str(executable)],check=True,capture_output=True,text=True);(out/'source-scenarios.log').write_text(scenario.stdout)
 summary={'source_scenarios':'passed','status':'passed','engine_commit':'f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de','complete_two_seat_frames':frames,'compared_ints':frames*2*7024,'natural_terminals':terminal,'max_entities':max_entities,'played_cards':cards,'other_rows_seen':sorted(saw),'runtime_jvm_dependency':False,'observer_header_sha256':hashlib.sha256((HERE/'observe.hpp').read_bytes()).hexdigest(),'scope':'native observer vs raw pinned Java object extraction; native whole-world transitions not exercised'}
 (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
