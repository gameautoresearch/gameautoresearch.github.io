"""Qualify split CUDA render semantics; never performs timing or cloud launch.

CPU mode checks the exact materialization helper and all strided-copy tails.
CUDA mode additionally compares every render/launch policy on actual device code.
"""
from pathlib import Path
import argparse,ctypes as C,hashlib,json,shutil,subprocess,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
CPU_SOURCE=r'''
#include "cuda_render_parallel.hpp"
#include "generated-v2/config.hpp"
#include <cstring>
#include <memory>
#include <cstdio>
using namespace clash_native;
int main(){
    static const auto spec=make_fixed8_spec();
    auto a=std::make_unique<NativeWorld>();auto b=std::make_unique<NativeWorld>();
    int expected[14048],actual[14048],info[12],gotInfo[12];
    uint8_t masks[4614],gotMasks[4614];float rewards[2],gotRewards[2];
    int frames=0,naturals=0,maxEntities=0;bool cards[8]{};
    uint32_t rng=4451;auto next=[&](){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;};
    for(int seed: {0,1,-1,771}){
        a->init(&spec,seed);
        for(int step=0;step<1500;++step){
            std::memcpy(b.get(),a.get(),sizeof(NativeWorld));b->bind(&spec);
            a->render(expected,masks,info,rewards);materialize_render_cache(*b);
            if(std::memcmp(&a->observer,&b->observer,sizeof(a->observer)))return 10;
            for(int lanes:{1,4,8,16,32,64,128,256}){
                std::memset(actual,0xa5,sizeof(actual));std::memset(gotMasks,0xa5,sizeof(gotMasks));
                std::memset(gotInfo,0xa5,sizeof(gotInfo));std::memset(gotRewards,0xa5,sizeof(gotRewards));
                for(int lane=0;lane<lanes;++lane)copy_render_cache(*b,actual,gotMasks,gotInfo,gotRewards,lane,lanes);
                if(std::memcmp(actual,expected,sizeof(actual))||std::memcmp(gotMasks,masks,sizeof(masks))||std::memcmp(gotInfo,info,sizeof(info))||std::memcmp(gotRewards,rewards,sizeof(rewards)))return 11;
            }
            // A repeated read must neither change first-sight IDs nor movement.
            int serial=b->observer.serial;materialize_render_cache(*b);
            if(serial!=b->observer.serial)return 12;
            ++frames;maxEntities=std::max(maxEntities,std::max(expected[48],expected[7024+48]));
            if(a->match.is_over(*a)){++naturals;a->init(&spec,int(next()%1000000));continue;}
            int actions[2]{};
            for(int seat=0;seat<2;++seat)if(step>18&&next()%5==0){
                int legal[2307],count=0;for(int i=1;i<2305;++i)if(masks[seat*2307+i])legal[count++]=i;
                if(count){actions[seat]=legal[next()%count];int slot=(actions[seat]-1)/576;int card=a->match.sides[seat].hand.slots[slot];if(card>=0&&card<8)cards[card]=true;}
            }
            a->advance(actions[0],actions[1]);
        }
        // Cache fault override must stay exact, including zero rewards.
        a->fault(999);a->rendered=false;std::memcpy(b.get(),a.get(),sizeof(NativeWorld));b->bind(&spec);
        a->render(expected,masks,info,rewards);materialize_render_cache(*b);
        copy_render_cache(*b,actual,gotMasks,gotInfo,gotRewards,0,1);
        if(std::memcmp(actual,expected,sizeof(actual))||std::memcmp(gotMasks,masks,sizeof(masks))||std::memcmp(gotInfo,info,sizeof(info))||std::memcmp(gotRewards,rewards,sizeof(rewards)))return 13;
        if(info[obs_schema::I_FAULT]!=999||info[obs_schema::I_WINNER]!=-1||rewards[0]!=0||rewards[1]!=0)return 14;
    }
    int covered=0;for(bool v:cards)covered+=v;
    std::printf("{\"status\":\"passed\",\"frames\":%d,\"partition_widths\":[1,4,8,16,32,64,128,256],\"natural_games\":%d,\"max_entities\":%d,\"cards_requested\":%d,\"repeated_read_and_fault_override\":true}\n",frames,naturals,maxEntities,covered);
    return covered==8&&naturals>0?0:15;
}
'''
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cpu-only',action='store_true')
    p.add_argument('--steps',type=int,default=700)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    source=args.out/'render-cache-test.cpp';source.write_text(CPU_SOURCE)
    exe=args.out/'render-cache-test';compiler=shutil.which('clang++') or shutil.which('g++')
    command=[compiler,'-std=c++20','-O2','-fno-fast-math','-ffp-contract=off','-I'+str(HERE),str(source),'-o',str(exe)]
    subprocess.run(command,check=True)
    cpu=json.loads(subprocess.check_output([str(exe)],text=True))
    from qualify_cuda_cooperative import CPU_SOURCE as COOPERATIVE_CPU_SOURCE
    cooperative_source=args.out/'cooperative-cache-test.cpp';cooperative_source.write_text(COOPERATIVE_CPU_SOURCE)
    cooperative_exe=args.out/'cooperative-cache-test'
    cooperative_command=[compiler,'-std=c++20','-O2','-fno-fast-math','-ffp-contract=off','-I'+str(HERE),str(cooperative_source),'-o',str(cooperative_exe)]
    subprocess.run(cooperative_command,check=True)
    cooperative_cpu=json.loads(subprocess.check_output([str(cooperative_exe)],text=True))
    result={'status':'passed','cpu_materialize_and_copy':cpu,'cpu_cooperative_controls':cooperative_cpu,'cpu_command':command,'cpu_cooperative_command':cooperative_command,'timing_claims':False,'source_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['cuda_engine.cu','cuda_render_parallel.hpp','cuda_render_cooperative.hpp','parallel_legal.hpp','world.hpp','observe/observe.hpp','pathfinding/spatial.hpp','pathfinding/world_queries.hpp']}}
    if not args.cpu_only:
        import torch
        from runtime import NativeCPU,NativeCUDA
        seeds=[0,1,-1,771,103];cpu_env=NativeCPU(seeds);gpu=NativeCUDA(seeds)
        lib=gpu.lib
        for name in ('cn_cuda_set_launch_threads','cn_cuda_set_render_mode'):
            fn=getattr(lib,name);fn.argtypes=[C.c_int];fn.restype=C.c_int
        configs=[]
        try:
            # Includes partial simulation blocks, selected reset holes, repeated
            # reads, illegal actions and a deliberate decision-boundary limit.
            for threads in [1,4,8,16,32]:
                for mode in [0,1,2,3,4]:
                    gpu.check(lib.cn_cuda_set_launch_threads(threads));gpu.check(lib.cn_cuda_set_render_mode(mode))
                    cpu_env.reset(seeds);gpu.reset(seeds);frames=0;rng=np.random.default_rng(4451)
                    def compare(label):
                        nonlocal frames
                        for name in ('observations','masks','infos','rewards'):
                            a=getattr(gpu,name).cpu().numpy();b=getattr(cpu_env,name)
                            if not np.array_equal(a,b):
                                np.savez_compressed(args.out/'divergence.npz',native=b,cuda=a)
                                raise AssertionError(f'{threads=} {mode=} {label} {name} differs at {np.argwhere(a!=b)[:10].tolist()}')
                        frames+=len(seeds)
                    compare('initial')
                    for step in range(args.steps):
                        actions=np.zeros((len(seeds),2),np.int32)
                        if step>=18:
                            for lane in range(len(seeds)):
                                for seat in range(2):
                                    if rng.random()<.2:actions[lane,seat]=rng.choice(np.flatnonzero(cpu_env.masks[lane,seat]))
                        cpu_env.step(actions);gpu.step(actions);compare(str(step))
                        if step%31==0:cpu_env.read();gpu.read();compare('repeated')
                        selected=cpu_env.infos[:,2]>=0
                        if selected.any():
                            nxt=rng.integers(0,1000000,len(seeds),dtype=np.int32);cpu_env.reset(nxt,selected);gpu.reset(nxt,selected);compare('natural reset')
                        if step==37:
                            selected=np.array([1,0,1,0,1],np.int32);nxt=[11,22,33,44,55]
                            cpu_env.reset(nxt,selected);gpu.reset(nxt,selected);compare('reset holes')
                    actions=np.full((len(seeds),2),999999,np.int32)
                    cpu_env.step(actions,tick_limit=1);gpu.step(actions,tick_limit=1);compare('illegal+limit')
                    configs.append({'threads':threads,'render_mode':mode,'frames':frames})
            assert lib.cn_cuda_set_launch_threads(3)!=0 and lib.cn_cuda_set_render_mode(5)!=0
            result['cuda_configs']=configs;result['cuda_library_sha256']=hashlib.sha256(gpu.library.read_bytes()).hexdigest();result['cuda_name']=torch.cuda.get_device_name()
        finally:
            lib.cn_cuda_set_launch_threads(32);lib.cn_cuda_set_render_mode(0);gpu.close();cpu_env.close()
    (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
