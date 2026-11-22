import numpy as np
from simulators.clash.autoresearch.refusal_diagnostics import DecisionCounters,visible_own_plays

def view(tick,step):
 o=np.zeros(7024,np.int32);o[0]=tick;o[54]=step;o[7]=123;return o

def test_request_and_run_refusals_are_separate_and_advance_gated():
 before=view(100,10);after=view(105,11);after[62]=6;after[63]=13
 mask=np.zeros(2307,np.uint8);mask[0]=mask[1]=1;c=DecisionCounters()
 c.observe(before,mask,1,after,own_placed_before=0,own_placed_after=0)
 d=c.summary();assert d['request_refusal_decisions']==d['run_refusal_advancing_decisions']==1;assert d['request_refusal_codes']=={'6':1} and d['run_refusal_codes']=={'13':1};assert d['actual_publicly_placed_cards']==0
 stale=view(105,12);stale[63]=13;c.observe(after,mask,0,stale,own_placed_before=0,own_placed_after=0)
 assert c.summary()['run_refusal_advancing_decisions']==1 and c.summary()['stale_run_refusal_nonadvancing_samples']==1
 c.observe(stale,mask,0,stale,own_placed_before=0,own_placed_after=0);assert c.summary()['nonadvancing_observation_pairs_skipped']==1

def test_visible_placed_history_whitelists_count_and_cards():
 a=view(0,0);b=view(5,1);a[114]=3;b[114]=5;b[116]=30;b[125]=452;b[7:11]=999
 assert visible_own_plays(a,b)==(3,5,[30,452])

def test_observer_preserves_actions_and_every_public_output():
 from simulators.clash.autoresearch.common import setup_native
 from simulators.clash.autoresearch.visible_teacher import PublicTeacher
 setup_native()
 from cpu_batch import NativeCPUBatch
 baseline=NativeCPUBatch([2026110951],threads=1);observed=NativeCPUBatch([2026110951],threads=1);actors=[[PublicTeacher(),PublicTeacher()] for _ in range(2)];counters=[DecisionCounters(),DecisionCounters()]
 try:
  for _ in range(180):
   old=observed.observations.copy();masks=observed.masks.copy();actions=[]
   for env,bots in zip((baseline,observed),actors):actions.append(np.array([[bots[s](env.observations[0,s],env.masks[0,s]) for s in (0,1)]],np.int32))
   assert np.array_equal(*actions);baseline.step(actions[0]);observed.step(actions[1])
   for s in (0,1):
    old_count,new_count,cards=visible_own_plays(old[0,1-s],observed.observations[0,1-s]);counters[s].observe(old[0,s],masks[0,s],int(actions[1][0,s]),observed.observations[0,s],own_placed_before=old_count,own_placed_after=new_count,own_placed_cards=cards)
   for name in ('observations','masks','infos','rewards'):assert getattr(baseline,name).tobytes()==getattr(observed,name).tobytes()
  for s,c in enumerate(counters):assert c.summary()['actual_publicly_placed_cards']==int(observed.observations[0,1-s,114])>0
 finally:baseline.close();observed.close()
