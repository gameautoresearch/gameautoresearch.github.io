"""Exhaustive selected-action predicate vs independent full-mask and admission controls.

Local correctness only; no timing claims, Java in this harness, or cloud launch. Source rules
come from the pinned crforge files attributed in rules/selected_action.hpp.
"""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess
HERE=Path(__file__).resolve().parent
SOURCE=r'''
#include "world_impl.hpp"
#include "rules/observation_world.hpp"
#include "generated-v2/config.hpp"
#include <cstring>
#include <memory>
#include <cstdio>
using namespace clash_native;
// Frozen original World.apply_action, using full independent legal() for its actor gate.
void original_action(NativeWorld&w,int side,int action){
 w.lastRefusal[side]=0;if(action==0)return;uint8_t open[2307];w.legal(side,open);
 if(action<1||action>=2307||!open[action]){++w.illegalNow[side];++w.illegalTotal[side];w.lastRefusal[side]=100;return;}
 int encoded=action-1,slot=encoded/576,tile=encoded%576,x=(tile%18)*1000+500,y=(tile/18)*1000+500;
 if(side==0)x=18000-x;else y=32000-y;MaskEntity mask[WORLD_CAPACITY];int n=w.mask_entities(mask);
 auto given=w.commands.give(w.match,w,w.spec->cards,side,slot,x,y,w.arena_view(),mask,n);
 w.lastRefusal[side]=int(given.refusal);if(!given.given()){++w.illegalNow[side];++w.illegalTotal[side];
 if(given.refusal==GivingRefusal::UnsupportedPromiseExpiry)w.fault(216);}
}
bool same_commands(const PlayerCommands&a,const PlayerCommands&b){
 if(a.count!=b.count)return false;for(int i=0;i<a.count;i++){const auto&x=a.queued[i];const auto&y=b.queued[i];
 if(x.side!=y.side||x.card!=y.card||x.givenTick!=y.givenTick||x.runTick!=y.runTick||x.expiresTick!=y.expiresTick||
 x.cost!=y.cost||x.x!=y.x||x.y!=y.y||x.code!=y.code||x.ran!=y.ran||x.placed!=y.placed)return false;}return true;
}
int main(){
 static const auto spec=make_fixed8_spec();auto w=std::make_unique<NativeWorld>();
 auto old=std::make_unique<NativeWorld>();auto selected=std::make_unique<NativeWorld>();
 long long bytes=0;int frames=0,natural=0,maxEntities=0,admissionCases=0,syntheticCases=0;bool cards[8]{};
 auto check_predicate=[&](){
  MaskEntity seen[WORLD_CAPACITY];int count=w->mask_entities(seen);uint8_t legal[2307];
  bool done=w->match.is_over(*w)||w->match.truncated||w->match.fault;
  for(int side=0;side<2;side++){w->legal(side,legal);
   for(int action=0;action<2307;action++){
    bool actual=selected_action_legal(w->match,*w,w->commands,spec.cards,w->arena_view(),seen,count,side,action,done);
    if(actual!=bool(legal[action])){std::fprintf(stderr,"predicate mismatch frame%d side%d action%d\n",frames,side,action);return false;}++bytes;}
   for(int action:{-1,INT32_MIN,2307,INT32_MAX})if(selected_action_legal(w->match,*w,w->commands,spec.cards,w->arena_view(),seen,count,side,action,done))return false;
  }return true;
 };
 auto check_admission=[&](int side,int action){
  std::memcpy(old.get(),w.get(),sizeof(NativeWorld));old->bind(&spec);
  std::memcpy(selected.get(),w.get(),sizeof(NativeWorld));selected->bind(&spec);
  original_action(*old,side,action);selected->apply_action(side,action);++admissionCases;
  return old->lastRefusal[side]==selected->lastRefusal[side]&&old->illegalNow[side]==selected->illegalNow[side]&&
   old->illegalTotal[side]==selected->illegalTotal[side]&&old->faultCode==selected->faultCode&&
   !std::memcmp(&old->match,&selected->match,sizeof(old->match))&&same_commands(old->commands,selected->commands);
 };
 uint32_t rng=4451;auto next=[&](){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;};
 int obs[14048],info[12];uint8_t masks[4614];float rewards[2];
 for(int seed:{0,1,-1,771}){w->init(&spec,seed);
  for(int step=0;step<1500;step++){
   w->render(obs,masks,info,rewards);if(!check_predicate())return 10;
   ++frames;maxEntities=std::max(maxEntities,std::max(obs[48],obs[7072]));
   if(w->match.is_over(*w)){++natural;w->init(&spec,int(next()%1000000));continue;}
   int actions[2]{};for(int side=0;side<2;side++)if(step>18&&next()%5==0){
    int legal[2307],count=0;for(int i=1;i<2305;i++)if(masks[side*2307+i])legal[count++]=i;
    if(count){actions[side]=legal[next()%count];int card=w->match.sides[side].hand.slots[(actions[side]-1)/576];if(card>=0&&card<8)cards[card]=true;}}
   if(step%97==0)for(int side=0;side<2;side++)if(!check_admission(side,actions[side]))return 11;
   w->advance(actions[0],actions[1]);
  }
 }
 // Queue-capacity is intentionally NOT a mask predicate: a legal action can reach give() and
 // set match.fault when compacted queue capacity is exhausted. Keep that source behavior.
 for(int scenario=0;scenario<10;scenario++){
  w->init(&spec,1001);w->match.tick=w->match.config.startCooldownTicks;
  if(scenario==1)w->match.tick=0;
  if(scenario==2)w->match.sides[0].elixir=0;
  if(scenario==3)w->match.sides[0].hand.slots[0]=-1;
  if(scenario==4){w->commands.count=1;w->commands.queued[0]={0,w->match.sides[0].hand.slots[0],1,w->match.tick+20,w->match.tick+60,2,1000,1000,0,false,false};}
  if(scenario==5){int card=w->match.sides[0].hand.slots[0];w->match.sides[0].elixir=spec.cards[card].cost*10000;
   w->commands.count=1;w->commands.queued[0]={0,-99,1,w->match.tick-1,w->match.tick+1,1,1000,1000,0,true,false};}
  if(scenario==6){w->commands.count=PlayerCommands::CAPACITY;for(int i=0;i<w->commands.count;i++)
   w->commands.queued[i]={i&1,-99,1,w->match.tick+20,w->match.tick+60,0,1000,1000,0,false,false};}
  if(scenario==7)w->match.truncated=true;
  if(scenario==8)w->fault(999);
  if(scenario==9)w->health[w->kingSlots[1]].hp=0;
  if(!check_predicate())return 12;
  for(int side=0;side<2;side++){
   uint8_t legal[2307];w->legal(side,legal);int action=0;for(int a=1;a<2305;a++)if(legal[a]){action=a;break;}
   for(int trial:{action,0,-1,2305,2306,2307,INT32_MAX,INT32_MIN})if(!check_admission(side,trial))return 13;
   if(scenario==6){if(action==0)return 14;selected->apply_action(side,action);
    if(!selected->match.fault)return 15;}
  }++syntheticCases;
 }
 int covered=0;for(bool card:cards)covered+=card;
 std::printf("{\"status\":\"passed\",\"real_world_frames\":%d,\"selected_mask_bytes_compared\":%lld,\"all2307_actions_both_seats\":true,\"natural_games\":%d,\"max_entities\":%d,\"cards_requested\":%d,\"admission_control_cases\":%d,\"synthetic_gate_scenarios\":%d,\"queue_capacity_fault_preserved\":true}\n",frames,bytes,natural,maxEntities,covered,admissionCases,syntheticCases);
 return covered==8&&natural>0?0:16;
}
'''
def main(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);source=out/'selected-action-test.cpp';source.write_text(SOURCE)
 compiler=shutil.which('clang++') or shutil.which('g++');exe=out/'selected-action-test'
 command=[compiler,'-std=c++20','-O2','-fno-fast-math','-ffp-contract=off','-I'+str(HERE),str(source),'-o',str(exe)]
 subprocess.run(command,check=True);result=json.loads(subprocess.check_output([str(exe)],text=True))
 result.update(cuda_executed=False,timing_claims=False,command=command,
  source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['rules/selected_action.hpp','rules/actions.hpp','rules/placement.hpp','rules/match.hpp','world.hpp']})
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
