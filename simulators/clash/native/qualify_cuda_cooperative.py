"""CPU controls for cooperative observation clear and per-action legal materialization.

No timing claims or cloud allocation. The CUDA matrix lives in qualify_cuda_render.py after
this independent proof; original NativeWorld.render and legal_actions remain the controls.
"""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess
HERE=Path(__file__).resolve().parent
CPU_SOURCE=r'''
#include "cuda_render_cooperative.hpp"
#include "generated-v2/config.hpp"
#include <cstring>
#include <memory>
#include <cstdio>
using namespace clash_native;
bool equivalent(NativeWorld&a,NativeWorld&b){
 return !std::memcmp(a.renderedObs,b.renderedObs,sizeof(a.renderedObs))&&
 !std::memcmp(a.renderedMasks,b.renderedMasks,sizeof(a.renderedMasks))&&
 !std::memcmp(a.renderedInfo,b.renderedInfo,sizeof(a.renderedInfo))&&
 !std::memcmp(a.renderedRewards,b.renderedRewards,sizeof(a.renderedRewards))&&
 !std::memcmp(&a.observer,&b.observer,sizeof(a.observer))&&a.rendered==b.rendered&&
 a.faultCode==b.faultCode&&a.match.fault==b.match.fault;
}
int main(){
 static const auto spec=make_fixed8_spec();auto a=std::make_unique<NativeWorld>();
 auto b=std::make_unique<NativeWorld>();auto c=std::make_unique<NativeWorld>();
 auto scratch=std::make_unique<LegalMaskScratch<WORLD_CAPACITY>>();
 int expected[14048],info[12];uint8_t masks[4614],got[4614];float rewards[2];
 constexpr int widths[]={1,4,8,16,32,64,128,256};int frames=0,natural=0,maxEntities=0;bool cards[8]{};
 uint32_t rng=4451;auto next=[&](){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;};
 for(int seed:{0,1,-1,771}){a->init(&spec,seed);
  for(int step=0;step<1500;step++){
   std::memcpy(b.get(),a.get(),sizeof(NativeWorld));b->bind(&spec);
   std::memcpy(c.get(),a.get(),sizeof(NativeWorld));c->bind(&spec);
   a->render(expected,masks,info,rewards);
   // Separate legality proof: original serial masks vs each complete per-action partition.
   prepare_parallel_legal(*a,*scratch);
   for(int lanes:widths){std::memset(got,0xa5,sizeof(got));
    for(int lane=0;lane<lanes;lane++)parallel_legal_slice(*a,*scratch,got,lane,lanes);
    if(std::memcmp(got,masks,sizeof(got))){std::fprintf(stderr,"legal mismatch seed%d step%d lanes%d\n",seed,step,lanes);return 20;}}
   int lanes=widths[step%8];materialize_cooperative_render_cache_cpu<false>(*b,*scratch,lanes);
   materialize_cooperative_render_cache_cpu<true>(*c,*scratch,lanes);
   if(!equivalent(*a,*b)||!equivalent(*a,*c)){std::fprintf(stderr,"cache mismatch seed%d step%d\n",seed,step);return 21;}
   int serial=b->observer.serial;materialize_cooperative_render_cache_cpu<false>(*b,*scratch,lanes);
   materialize_cooperative_render_cache_cpu<true>(*c,*scratch,lanes);
   if(b->observer.serial!=serial||!equivalent(*a,*b)||!equivalent(*a,*c))return 22;
   ++frames;maxEntities=std::max(maxEntities,std::max(expected[48],expected[7072]));
   if(a->match.is_over(*a)){++natural;a->init(&spec,int(next()%1000000));continue;}
   int actions[2]{};for(int side=0;side<2;side++)if(step>18&&next()%5==0){
    int legal[2307],n=0;for(int action=1;action<2305;action++)if(masks[side*2307+action])legal[n++]=action;
    if(n){actions[side]=legal[next()%n];int card=a->match.sides[side].hand.slots[(actions[side]-1)/576];if(card>=0&&card<8)cards[card]=true;}}
   a->advance(actions[0],actions[1]);
  }
  a->fault(999);a->rendered=false;std::memcpy(b.get(),a.get(),sizeof(NativeWorld));b->bind(&spec);
  std::memcpy(c.get(),a.get(),sizeof(NativeWorld));c->bind(&spec);a->render(expected,masks,info,rewards);
  materialize_cooperative_render_cache_cpu<false>(*b,*scratch,256);materialize_cooperative_render_cache_cpu<true>(*c,*scratch,256);
  if(!equivalent(*a,*b)||!equivalent(*a,*c)||info[obs_schema::I_FAULT]!=999||info[obs_schema::I_WINNER]!=-1||rewards[0]!=0||rewards[1]!=0)return 23;
 }
 int covered=0;for(bool card:cards)covered+=card;
 std::printf("{\"status\":\"passed\",\"frames\":%d,\"legal_mask_bytes_compared\":%lld,\"partition_widths\":[1,4,8,16,32,64,128,256],\"natural_games\":%d,\"max_entities\":%d,\"cards_requested\":%d,\"observer_exact\":true,\"repeated_reads\":true,\"fault_override\":true}\n",frames,1LL*frames*4614*8,natural,maxEntities,covered);
 return covered==8&&natural>0?0:24;
}
'''
def main(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);source=out/'cooperative-test.cpp';source.write_text(CPU_SOURCE)
 exe=out/'cooperative-test';compiler=shutil.which('clang++') or shutil.which('g++')
 command=[compiler,'-std=c++20','-O2','-fno-fast-math','-ffp-contract=off','-I'+str(HERE),str(source),'-o',str(exe)]
 subprocess.run(command,check=True);result=json.loads(subprocess.check_output([str(exe)],text=True))
 result.update(cuda_executed=False,timing_claims=False,command=command,
 source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['parallel_legal.hpp','cuda_render_cooperative.hpp','observe/observe.hpp','world.hpp']})
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
