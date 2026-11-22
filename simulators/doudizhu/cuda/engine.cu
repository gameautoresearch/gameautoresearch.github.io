// Complete device-resident fixed-role DouZero simulator. Shared rules.h CPU oracle.
#include "rules.h"
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <string>
namespace {
thread_local std::string error;
int checked(cudaError_t e){if(e!=cudaSuccess){error=cudaGetErrorString(e);return-1;}return 0;}
__global__ void reset_kernel(dd::State*s,const int8_t*hands,const int8_t*bottom,int n){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=n)return;
 for(int r=0;r<3;++r)for(int k=0;k<16;++k)s[g].hand[r][k]=hands[(g*3+r)*16+k];
 for(int k=0;k<16;++k)s[g].bottom[k]=bottom[g*16+k];
}
__global__ void generate_kernel(const dd::State*states,dd::Action*all,int32_t*counts,int32_t*fault,int n){
 int g=blockIdx.x;if(g>=n)return;dd::Action*a=all+(size_t)g*dd::MAX_ACTIONS;
 __shared__ int raw,power;
 if(threadIdx.x==0){dd::Generator gen(states[g],a);gen.generate();raw=gen.n;power=1;while(power<raw)power*=2;
  if(gen.overflow||(!states[g].done&&raw==0)){atomicExch(fault,1);raw=0;}counts[g]=0;}
 __syncthreads();if(raw==0)return;
 for(int i=raw+threadIdx.x;i<power;i+=blockDim.x){a[i]=dd::Action{};a[i].hi=LLONG_MAX;a[i].lo=ULLONG_MAX;}
 __syncthreads();
 for(int size=2;size<=power;size*=2)for(int jump=size/2;jump>0;jump/=2){
  for(int i=threadIdx.x;i<power;i+=blockDim.x){int other=i^jump;if(other>i){dd::Action x=a[i],y=a[other];
   bool swap=(i&size)==0?dd::less(y,x):dd::less(x,y);if(swap){a[i]=y;a[other]=x;}}}
  __syncthreads();}
 if(threadIdx.x==0){int unique=0;for(int i=0;i<raw;++i){dd::Action value=a[i];if(unique==0||!dd::equal(value,a[unique-1]))a[unique++]=value;}counts[g]=unique;}
}
__global__ void encode_kernel(const dd::State*s,int8_t*features,int role,int32_t*fault,int n){
 int g=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;if(g>=n)return;
 if(!s[g].done&&s[g].role!=role){if(lane==0)atomicExch(fault,1);return;}
 int row=s[g].steps-1;if(row>0&&!dd::total(s[g].history[row]))--row;
 const uint8_t*previous=row>=0?s[g].history[row]:nullptr;
 int r0=dd::remaining(s[g],0),r1=dd::remaining(s[g],1),r2=dd::remaining(s[g],2);
 for(int i=lane;i<1240;i+=32)features[(size_t)g*1240+i]=dd::encode_byte(s[g],role,previous,r0,r1,r2,i);
}
__global__ void compact_kernel(const dd::Action*actions,const int64_t*owners,const int64_t*local,int8_t*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const auto&a=actions[(size_t)owners[i]*dd::MAX_ACTIONS+local[i]];dd::bits(a.c,out+(size_t)i*54);
}
__global__ void argmax_kernel(const float*q,const int32_t*counts,const int64_t*starts,int64_t*out,int32_t*fault,int n){
 int g=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;if(g>=n)return;
 float best=-INFINITY;int index=4096;
 for(int i=lane;i<counts[g];i+=32){float value=q[starts[g]+i];if(!isfinite(value))atomicExch(fault,1);
  if(value>best||(value==best&&i<index)){best=value;index=i;}}
 for(int jump=16;jump>0;jump/=2){float value=__shfl_down_sync(0xffffffff,best,jump);int other=__shfl_down_sync(0xffffffff,index,jump);
  if(value>best||(value==best&&other<index)){best=value;index=other;}}
 if(lane==0)out[g]=counts[g]?index:0;
}
__global__ void step_kernel(dd::State*s,const dd::Action*actions,const int32_t*counts,const int64_t*choice,int32_t*fault,int n){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=n||s[g].done)return;
 int index=choice[g];if(index<0||index>=counts[g]){atomicExch(fault,1);return;}
 if(dd::step(s[g],actions[(size_t)g*dd::MAX_ACTIONS+index]))atomicExch(fault,1);
}
__global__ void flags_kernel(const dd::State*s,int32_t*out,int n){int g=blockIdx.x*blockDim.x+threadIdx.x;if(g<n){
 out[g*4]=s[g].done;out[g*4+1]=s[g].won;out[g*4+2]=s[g].bombs;out[g*4+3]=s[g].steps;}}
__global__ void choose_kernel(const dd::State*s,const dd::Action*actions,const int32_t*counts,const int64_t*greedy,
 const int8_t*kinds,const uint64_t*seeds,int turn,uint64_t epsilon,int64_t*choice,int32_t*fault,int n){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=n)return;if(s[g].done){choice[g]=0;return;}
 if(counts[g]<=0){atomicExch(fault,1);return;}
 int index=dd::choose(s[g],actions+(size_t)g*dd::MAX_ACTIONS,counts[g],greedy[g],kinds[g],seeds[g],turn,epsilon);
 if(index<0||index>=counts[g])atomicExch(fault,1);choice[g]=index;
}
__global__ void store_kernel(const dd::State*s,const dd::Action*actions,const int32_t*counts,const int64_t*choice,const int8_t*features,
 int8_t*x,int8_t*z,uint8_t*valid,int32_t*fault,int role,int n){
 int g=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;if(g>=n)return;if(lane==0)valid[g]=!s[g].done;if(s[g].done)return;
 if(choice[g]<0||choice[g]>=counts[g]){if(lane==0){valid[g]=0;atomicExch(fault,1);}return;}
 int width=role==0?319:430;const int8_t*f=features+(size_t)g*1240;
 for(int i=lane;i<width;i+=32)x[(size_t)g*484+i]=f[i];
 for(int i=lane;i<54;i+=32)x[(size_t)g*484+width+i]=dd::bit(actions[(size_t)g*dd::MAX_ACTIONS+choice[g]].c,i);
 for(int i=lane;i<810;i+=32)z[(size_t)g*810+i]=f[width+i];
}
}
extern "C" {
const char* dg_error(){return error.c_str();}
int dg_state_bytes(){return sizeof(dd::State);}int dg_action_bytes(){return sizeof(dd::Action);}
int dg_reset(void*state,void*hands,void*bottom,int n,void*stream){
 auto st=(cudaStream_t)stream;auto e=cudaMemsetAsync(state,0,(size_t)n*sizeof(dd::State),st);if(checked(e))return-1;
 reset_kernel<<<(n+127)/128,128,0,st>>>((dd::State*)state,(int8_t*)hands,(int8_t*)bottom,n);return checked(cudaGetLastError());}
int dg_generate(void*state,void*actions,void*counts,void*fault,int n,void*stream){
 generate_kernel<<<n,128,0,(cudaStream_t)stream>>>((dd::State*)state,(dd::Action*)actions,(int32_t*)counts,(int32_t*)fault,n);return checked(cudaGetLastError());}
int dg_encode(void*state,void*features,int role,void*fault,int n,void*stream){
 encode_kernel<<<(n+7)/8,256,0,(cudaStream_t)stream>>>((dd::State*)state,(int8_t*)features,role,(int32_t*)fault,n);return checked(cudaGetLastError());}
int dg_compact(void*actions,void*owners,void*local,void*out,int total,void*stream){
 compact_kernel<<<(total+255)/256,256,0,(cudaStream_t)stream>>>((dd::Action*)actions,(int64_t*)owners,(int64_t*)local,(int8_t*)out,total);return checked(cudaGetLastError());}
int dg_argmax(void*q,void*counts,void*starts,void*out,void*fault,int n,void*stream){
 argmax_kernel<<<(n+7)/8,256,0,(cudaStream_t)stream>>>((float*)q,(int32_t*)counts,(int64_t*)starts,(int64_t*)out,(int32_t*)fault,n);return checked(cudaGetLastError());}
int dg_step(void*state,void*actions,void*counts,void*choice,void*fault,int n,void*stream){
 step_kernel<<<(n+127)/128,128,0,(cudaStream_t)stream>>>((dd::State*)state,(dd::Action*)actions,(int32_t*)counts,(int64_t*)choice,(int32_t*)fault,n);return checked(cudaGetLastError());}
int dg_flags(void*state,void*out,int n,void*stream){
 flags_kernel<<<(n+127)/128,128,0,(cudaStream_t)stream>>>((dd::State*)state,(int32_t*)out,n);return checked(cudaGetLastError());}
int dg_choose(void*state,void*actions,void*counts,void*greedy,void*kinds,void*seeds,int turn,uint64_t epsilon,
 void*choice,void*fault,int n,void*stream){
 choose_kernel<<<(n+127)/128,128,0,(cudaStream_t)stream>>>((dd::State*)state,(dd::Action*)actions,(int32_t*)counts,
 (int64_t*)greedy,(int8_t*)kinds,(uint64_t*)seeds,turn,epsilon,(int64_t*)choice,(int32_t*)fault,n);return checked(cudaGetLastError());}
int dg_store(void*state,void*actions,void*counts,void*choice,void*features,void*x,void*z,void*valid,void*fault,int role,int n,void*stream){
 store_kernel<<<(n+7)/8,256,0,(cudaStream_t)stream>>>((dd::State*)state,(dd::Action*)actions,(int32_t*)counts,(int64_t*)choice,(int8_t*)features,(int8_t*)x,(int8_t*)z,(uint8_t*)valid,(int32_t*)fault,role,n);return checked(cudaGetLastError());}
}
