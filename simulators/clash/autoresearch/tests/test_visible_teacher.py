import json,time,subprocess,sys
import numpy as np
import pytest
from simulators.clash.autoresearch.common import DATA,setup_native
from simulators.clash.autoresearch.visible_teacher import PublicTeacher

@pytest.fixture
def scene():
 setup_native()
 from cpu_batch import NativeCPUBatch
 env=NativeCPUBatch([2026110901],threads=1);o=env.observations[0,0].copy();env.close()
 teacher=PublicTeacher();s=teacher.schema;h=s.header;e=s.entity
 def make(hand,entities=(),elixir=9,tick=0,overtime=0):
  x=o.copy();x[h['elixir'][0]]=int(elixir*10000);x[h['elixir_reserved'][0]]=0;x[h['pending_count'][0]]=0;x[h['tick'][0]]=tick;x[h['overtime'][0]]=overtime
  mask=np.zeros(2307,dtype=np.uint8);mask[0]=1
  for slot,name in enumerate(hand):x[h['hand_card'][slot]]=teacher.ids[name];mask[1+576*slot:1+576*(slot+1)]=1
  count=int(x[h['entity_count'][0]])
  for name,side,px,py,hp,flags in entities:
   row=x[224+count*34:224+(count+1)*34];row[:]=0
   for k,v in {'kind':1,'side':side,'row':teacher.units[teacher.ids[name]].unit,'x':int(px*1000),'y':int(py*1000),'hp':hp,'max_hp':hp,'flags':1|flags}.items():row[e[k]]=v
   count+=1
  x[h['entity_count'][0]]=count
  return x,mask
 return teacher,make

def test_conserves_elixir_without_visible_reason(scene):
 t,make=scene;o,m=make(['Knight','Archer'],elixir=3)
 assert t(o,m)==0

def test_defends_tank_and_respects_mask(scene):
 t,make=scene;o,m=make(['Knight','MiniPekka','Musketeer'],[('Giant',1,3.5,11,4000,0)])
 a=t(o,m);assert t.last_reason=='defense' and (a-1)//576==1 and m[a]
 m[577:1153]=0;a=t(o,m);assert (a-1)//576!=1 and m[a]

def test_air_threat_uses_ranged_defender(scene):
 t,make=scene;o,m=make(['Knight','MiniPekka','Musketeer'],[('Archer',1,14.5,10,800,16)])
 a=t(o,m);assert t.last_reason=='defense' and (a-1)//576==2

def test_supports_visible_tank(scene):
 t,make=scene;o,m=make(['Musketeer','Archer'],[('Giant',0,3.5,13,4000,0)],elixir=5)
 a=t(o,m);assert t.last_reason=='tank-support' and (a-1)//576==0
 assert ((a-1)%576)//18<13

def test_spell_values_cluster_not_isolated_tank(scene):
 t,make=scene;o,m=make(['Fireball','Knight'],[('Musketeer',1,3.5+i*.5,23,688,0) for i in range(3)],elixir=6)
 assert (t(o,m)-1)//576==0 and t.last_reason=='spell-value'
 o,m=make(['Fireball'],[('Giant',1,3.5,23,4000,0)],elixir=6);assert t(o,m)==0

def test_overtime_pressure_is_on_legal_own_half(scene):
 t,make=scene;o,m=make(['Giant'],overtime=1)
 a=t(o,m);assert t.last_reason=='tank-pressure' and ((a-1)%576)//18==13
 # Boundary masks are authoritative even when the desired cell is closed.
 m[:]=0;m[0]=1;m[1]=1;assert t(o,m)==1

def test_intermediate_has_declared_cadence(scene):
 _,make=scene;t=PublicTeacher('intermediate');o,m=make(['Giant'],elixir=10,tick=5);assert t(o,m)==0
 o[t.schema.header['tick'][0]]=20;assert t(o,m)!=0

def test_real_both_seats_and_natural_overtime():
 setup_native()
 from cpu_batch import NativeCPUBatch
 env=NativeCPUBatch([2026110901,2026110902],threads=1);bots=[PublicTeacher(),PublicTeacher()]
 try:
  for _ in range(725):env.step(np.zeros((2,2),np.int32))
  assert env.observations[:,:,60].all()  # actual regulation-to-overtime transition
  for _ in range(30):
   actions=np.array([[bots[seat](env.observations[i,seat],env.masks[i,seat]) for seat in (0,1)] for i in range(2)],np.int32)
   assert all(env.masks[i,s,actions[i,s]] for i in range(2) for s in (0,1));env.step(actions)
   assert not env.infos[:,[5,7,8,9,11]].any()
 finally:env.close()

def test_policy_pipe_timeout_is_not_a_natural_outcome():
 from simulators.clash.autoresearch.arena import PolicyProcess
 p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(2)'],stdout=subprocess.PIPE)
 actor=PolicyProcess.__new__(PolicyProcess);actor.process=p
 try:
  with pytest.raises(RuntimeError,match='deadline'):actor.read(4,time.monotonic()+.03)
 finally:p.kill();p.wait()
