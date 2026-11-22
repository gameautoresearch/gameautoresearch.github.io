// SPDX-License-Identifier: Apache-2.0
// CUDA device-resident fixed8 transitions/observations. Uses exactly the native
// CPU rules headers; no CPU simulation, Python, JVM or oracle calls in kernels.
// Explicit fixed8 device specialization; generic/CPU builds stay dynamic.
#define CN_CUDA_FIXED8_GRID 1
#include "world_impl.hpp"
#include "rules/observation_world.hpp"
#include "task_spec.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include "cuda_render_parallel.hpp"
#include "cuda_render_cooperative.hpp"
using namespace clash_native;
struct CudaClash {NativeTaskSpec* spec=nullptr;};
// Host-side launch policy is intentionally separate from saved simulation state.
// Set before graph capture; already captured graphs retain their launch policy.
static int launch_threads=32;
static int render_mode=0;
static int launch_blocks(int count){return (count+launch_threads-1)/launch_threads;}
static int render_outputs(NativeWorld* worlds,const NativeTaskSpec* spec,
        const int* seeds,const int* selected,int* obs,uint8_t* masks,int* info,
        float* rewards,int count,cudaStream_t stream);

__global__ void reset_worlds(NativeWorld* worlds,const NativeTaskSpec* spec,const int* seeds,const int* selected,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count&&(!selected||selected[i]))worlds[i].init(spec,seeds[i]);}
__global__ void reset_render_worlds(NativeWorld* worlds,const NativeTaskSpec* spec,const int* seeds,const int* selected,int* observations,uint8_t* masks,int* info,float* rewards,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count&&(!selected||selected[i])){NativeWorld&w=worlds[i];w.init(spec,seeds[i]);w.render(observations+i*14048,masks+i*4614,info+i*12,rewards+i*2);}}
__global__ void rebind_worlds(NativeWorld* worlds,const NativeTaskSpec* spec,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)worlds[i].bind(spec);}
__global__ void step_worlds(NativeWorld* worlds,const int* actions,int count,int tick_limit){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)worlds[i].advance(actions[i*2],actions[i*2+1],tick_limit);}
__global__ void read_worlds(NativeWorld* worlds,int* observations,uint8_t* masks,int* info,float* rewards,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){NativeWorld&w=worlds[i];w.render(observations+i*14048,masks+i*4614,info+i*12,rewards+i*2);}}
static int render_outputs(NativeWorld* worlds,const NativeTaskSpec* spec,
        const int* seeds,const int* selected,int* obs,uint8_t* masks,int* info,
        float* rewards,int count,cudaStream_t stream){
    if(count<=0)return int(cudaSuccess);
    using namespace cuda_parallel_render;
    if(render_mode==0){
        if(seeds)reset_render_worlds<<<launch_blocks(count),launch_threads,0,stream>>>(worlds,spec,seeds,selected,obs,masks,info,rewards,count);
        else read_worlds<<<launch_blocks(count),launch_threads,0,stream>>>(worlds,obs,masks,info,rewards,count);
    }else if(render_mode==1){
        if(seeds)reset_prepare<<<launch_blocks(count),launch_threads,0,stream>>>(worlds,spec,seeds,selected,count);
        else prepare<<<launch_blocks(count),launch_threads,0,stream>>>(worlds,selected,count);
        auto code=cudaGetLastError();if(code!=cudaSuccess)return int(code);
        copy<<<count,COPY_THREADS,0,stream>>>(worlds,selected,obs,masks,info,rewards,count);
    }else if(render_mode==2){
        prepare_copy<<<count,COPY_THREADS,0,stream>>>(worlds,spec,seeds,selected,obs,masks,info,rewards,count);
    }else if(render_mode==3){
        cuda_cooperative_render::prepare_copy<false><<<count,cuda_cooperative_render::THREADS,0,stream>>>(worlds,spec,seeds,selected,obs,masks,info,rewards,count);
    }else{
        cuda_cooperative_render::prepare_copy<true><<<count,cuda_cooperative_render::THREADS,0,stream>>>(worlds,spec,seeds,selected,obs,masks,info,rewards,count);
    }
    return int(cudaGetLastError());
}
extern "C" {
// Existing APIs and defaults stay unchanged. Invalid knobs never mutate policy.
int cn_cuda_set_launch_threads(int value){if(value!=1&&value!=4&&value!=8&&value!=16&&value!=32)return int(cudaErrorInvalidValue);launch_threads=value;return int(cudaSuccess);}
int cn_cuda_get_launch_threads(){return launch_threads;}
int cn_cuda_set_render_mode(int value){if(value<0||value>4)return int(cudaErrorInvalidValue);render_mode=value;return int(cudaSuccess);}
int cn_cuda_get_render_mode(){return render_mode;}

size_t cn_cuda_world_bytes(){return sizeof(NativeWorld);}
// The cadence and timeline this binary runs; native/match_forms.py checks it against its form.
int cn_cuda_match_form(int*out,int cap){static const NativeTaskSpec host=make_task_spec();return report_match_form(host.rules,out,cap);}
int cn_cuda_last_error(){return int(cudaGetLastError());}
const char*cn_cuda_error_string(int code){return cudaGetErrorString(cudaError_t(code));}
// Source-bound PTX receipts after caller-owned route/spatial/placement scratch:
// step 13,104 B, maximum reachable reset/render frame 21,664 B. Keep headroom
// without reserving the former 64-KiB stack for every resident CUDA thread.
void*cn_cuda_create(){if(cudaDeviceSetLimit(cudaLimitStackSize,32768)!=cudaSuccess)return nullptr;auto*h=new CudaClash;auto host=make_task_spec();auto err=cudaMalloc(&h->spec,sizeof(host));if(err!=cudaSuccess){delete h;return nullptr;}err=cudaMemcpy(h->spec,&host,sizeof(host),cudaMemcpyHostToDevice);if(err!=cudaSuccess){cudaFree(h->spec);delete h;return nullptr;}return h;}
void cn_cuda_free(void*p){auto*h=static_cast<CudaClash*>(p);if(h){cudaFree(h->spec);delete h;}}
int cn_cuda_reset(void*p,void*states,const int*seeds,const int*selected,int count,void*stream){auto*h=static_cast<CudaClash*>(p);reset_worlds<<<launch_blocks(count),launch_threads,0,static_cast<cudaStream_t>(stream)>>>(static_cast<NativeWorld*>(states),h->spec,seeds,selected,count);return int(cudaGetLastError());}
int cn_cuda_reset_read(void*p,void*states,const int*seeds,const int*selected,int*obs,uint8_t*masks,int*info,float*rewards,int count,void*stream){auto*h=static_cast<CudaClash*>(p);return render_outputs(static_cast<NativeWorld*>(states),h->spec,seeds,selected,obs,masks,info,rewards,count,static_cast<cudaStream_t>(stream));}
int cn_cuda_rebind(void*p,void*states,int count,void*stream){auto*h=static_cast<CudaClash*>(p);rebind_worlds<<<launch_blocks(count),launch_threads,0,static_cast<cudaStream_t>(stream)>>>(static_cast<NativeWorld*>(states),h->spec,count);return int(cudaGetLastError());}
int cn_cuda_step(void*states,const int*actions,int count,int tick_limit,void*stream){step_worlds<<<launch_blocks(count),launch_threads,0,static_cast<cudaStream_t>(stream)>>>(static_cast<NativeWorld*>(states),actions,count,tick_limit);return int(cudaGetLastError());}
int cn_cuda_read(void*states,int*obs,uint8_t*masks,int*info,float*rewards,int count,void*stream){return render_outputs(static_cast<NativeWorld*>(states),nullptr,nullptr,nullptr,obs,masks,info,rewards,count,static_cast<cudaStream_t>(stream));}
}
