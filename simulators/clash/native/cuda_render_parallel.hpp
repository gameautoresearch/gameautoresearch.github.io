// SPDX-License-Identifier: Apache-2.0
// CUDA render scheduling only. Native output semantics come from world.hpp and
// observe/observe.hpp, ported from candidates/crforge/env/src/main/java/org/
// crforge/env/LadderEnv.java at
// f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de. No game transition is replaced.
#pragma once
#include "world_impl.hpp"
#include "rules/observation_world.hpp"
namespace clash_native {
// Keep this equivalent to NativeWorld::render's !rendered branch. A separate
// helper avoids the original one-thread copy of ~61 KB after materialization.
// The CPU differential qualifier compares cache bytes and observer mutations.
CN_HD inline CN_WORLD_NOINLINE void materialize_render_cache(NativeWorld& w){
    if(w.rendered)return;
    w.extract_observation();
    observe(w.observer,w,w.renderedObs,w.renderedInfo,w.renderedRewards);
    w.legal(0,w.renderedMasks);w.legal(1,w.renderedMasks+2307);
    if(w.faultCode){
        w.renderedInfo[obs_schema::I_FAULT]=w.faultCode;
        w.renderedInfo[obs_schema::I_WINNER]=-1;
        w.renderedRewards[0]=w.renderedRewards[1]=0;
    }
    w.rendered=true;
}
// Each CUDA block owns one world; adjacent lanes access adjacent output fields.
// Also callable on CPU to qualify every partition width and tail exactly.
CN_HD inline void copy_render_cache(const NativeWorld& w,int* observations,
        uint8_t* masks,int* info,float* rewards,int lane,int lanes){
    for(int i=lane;i<14048;i+=lanes)observations[i]=w.renderedObs[i];
    for(int i=lane;i<4614;i+=lanes)masks[i]=w.renderedMasks[i];
    for(int i=lane;i<12;i+=lanes)info[i]=w.renderedInfo[i];
    for(int i=lane;i<2;i+=lanes)rewards[i]=w.renderedRewards[i];
}
#ifdef __CUDACC__
namespace cuda_parallel_render {
constexpr int COPY_THREADS=128;
__global__ void prepare(NativeWorld* worlds,const int* selected,int count){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count&&(!selected||selected[i]))materialize_render_cache(worlds[i]);
}
__global__ void reset_prepare(NativeWorld* worlds,const NativeTaskSpec* spec,
        const int* seeds,const int* selected,int count){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count&&(!selected||selected[i])){
        worlds[i].init(spec,seeds[i]);materialize_render_cache(worlds[i]);
    }
}
__global__ void copy(const NativeWorld* worlds,const int* selected,int* observations,
        uint8_t* masks,int* info,float* rewards,int count){
    int i=blockIdx.x;
    if(i<count&&(!selected||selected[i]))
        copy_render_cache(worlds[i],observations+i*14048,masks+i*4614,
            info+i*12,rewards+i*2,threadIdx.x,blockDim.x);
}
__global__ void prepare_copy(NativeWorld* worlds,const NativeTaskSpec* spec,
        const int* seeds,const int* selected,int* observations,uint8_t* masks,
        int* info,float* rewards,int count){
    int i=blockIdx.x;
    // Selection is block-uniform, including the barrier. Unselected output
    // buffers and observer serial remain untouched, exactly like reset_read.
    if(i>=count||(selected&&!selected[i]))return;
    if(threadIdx.x==0){
        if(seeds)worlds[i].init(spec,seeds[i]);
        materialize_render_cache(worlds[i]);
    }
    __syncthreads();
    copy_render_cache(worlds[i],observations+i*14048,masks+i*4614,
        info+i*12,rewards+i*2,threadIdx.x,blockDim.x);
}
} // namespace cuda_parallel_render
#endif
} // namespace clash_native
