// SPDX-License-Identifier: Apache-2.0
// Cooperative scheduling of the observer/mask port of crforge env/src/main/java/org/crforge/env/
// {LadderEnv,SeatIds,SeatView,Schema}.java @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de.
// https://github.com/voonhous/crforge. Original serial materialization remains the control.
#pragma once
#include "cuda_render_parallel.hpp"
#include "parallel_legal.hpp"
namespace clash_native {
CN_HD inline void clear_render_observations(NativeWorld&w,int lane,int lanes){
 for(int k=lane;k<14048;k+=lanes)w.renderedObs[k]=0;
}
// Caller must clear all observation entries first and synchronize before this single-thread body.
// It neither computes legal masks nor marks the cache complete; both follow in source order.
CN_HD inline CN_WORLD_NOINLINE void materialize_precleared_observations(NativeWorld&w){
 w.extract_observation();observe<false>(w.observer,w,w.renderedObs,w.renderedInfo,w.renderedRewards);
}
CN_HD inline void finish_cooperative_render_cache(NativeWorld&w){
 if(w.faultCode){w.renderedInfo[obs_schema::I_FAULT]=w.faultCode;w.renderedInfo[obs_schema::I_WINNER]=-1;
 w.renderedRewards[0]=w.renderedRewards[1]=0;}w.rendered=true;
}
// CPU scheduling control for every partition width, including repeated-read/observer semantics.
template<bool ParallelLegal> CN_HD inline void materialize_cooperative_render_cache_cpu(NativeWorld&w,
 LegalMaskScratch<WORLD_CAPACITY>&scratch,int lanes){
 if(w.rendered)return;
 for(int lane=0;lane<lanes;lane++)clear_render_observations(w,lane,lanes);
 materialize_precleared_observations(w);
 if constexpr(ParallelLegal){prepare_parallel_legal(w,scratch);
 for(int lane=0;lane<lanes;lane++)parallel_legal_slice(w,scratch,w.renderedMasks,lane,lanes);}
 else{w.legal(0,w.renderedMasks);w.legal(1,w.renderedMasks+2307);}
 finish_cooperative_render_cache(w);
}
#ifdef __CUDACC__
namespace cuda_cooperative_render {
constexpr int THREADS=256;
template<bool ParallelLegal> __global__ void prepare_copy(NativeWorld*worlds,const NativeTaskSpec*spec,
 const int*seeds,const int*selected,int*observations,uint8_t*masks,int*info,float*rewards,int count){
 int i=blockIdx.x;if(i>=count||(selected&&!selected[i]))return;
 __shared__ int needed;
 // MaskEntity has default member initializers. Aligned unsigned-byte shared storage avoids
 // unsupported dynamic shared constructors; assignment populates every field before use.
 __shared__ __align__(16) unsigned char legalStorage[ParallelLegal?sizeof(LegalMaskScratch<WORLD_CAPACITY>):1];
 NativeWorld&w=worlds[i];
 if(threadIdx.x==0){if(seeds)w.init(spec,seeds[i]);needed=!w.rendered;}
 __syncthreads();
 if(needed){
  clear_render_observations(w,threadIdx.x,blockDim.x);__syncthreads();
  if(threadIdx.x==0){materialize_precleared_observations(w);
   if constexpr(ParallelLegal){auto&scratch=*reinterpret_cast<LegalMaskScratch<WORLD_CAPACITY>*>(legalStorage);prepare_parallel_legal(w,scratch);}
   else{w.legal(0,w.renderedMasks);w.legal(1,w.renderedMasks+2307);}
  }
  __syncthreads();
  if constexpr(ParallelLegal){auto&scratch=*reinterpret_cast<LegalMaskScratch<WORLD_CAPACITY>*>(legalStorage);
   parallel_legal_slice(w,scratch,w.renderedMasks,threadIdx.x,blockDim.x);__syncthreads();}
  if(threadIdx.x==0)finish_cooperative_render_cache(w);
 }
 __syncthreads();
 copy_render_cache(w,observations+i*14048,masks+i*4614,info+i*12,rewards+i*2,threadIdx.x,blockDim.x);
}
}
#endif
}
