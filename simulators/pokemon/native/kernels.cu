// SPDX-License-Identifier: MIT
#include "sim.hpp"
#include "observe.hpp"
#include "env_core.hpp"
#include "teamgen.hpp"
#include "reset.hpp"
#include "startup.hpp"
#include "random_env.hpp"
#include <cuda_runtime.h>
__global__ void pkn_tick(pkn::State* states,const int32_t* actions,int n,pkn::Rules rules){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)pkn::step(states[i],actions[i*2],actions[i*2+1],rules);}
__global__ void pkn_mask_kernel(const pkn::State* states,uint8_t* masks,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){pkn::mask(states[i],0,masks+i*28);pkn::mask(states[i],1,masks+i*28+14);}}
extern "C" int pkn_step_device(void* states,const int32_t* actions,int n,const pkn::Move* moves,int count,const int8_t* chart,void* stream){pkn_tick<<<(n+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(static_cast<pkn::State*>(states),actions,n,{moves,count,chart});return int(cudaGetLastError());}
extern "C" int pkn_masks_device(const void* states,uint8_t* masks,int n,void* stream){pkn_mask_kernel<<<(n+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const pkn::State*>(states),masks,n);return int(cudaGetLastError());}
__global__ void pkn_visible_kernel(const pkn::State* states,float* output,int n,pkn::PublicRules rules){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){pkn::observe(states[i],0,output+i*1400,rules);pkn::observe(states[i],1,output+i*1400+700,rules);}}
extern "C" int pkn_observe_device(const void* states,float* output,int n,const pkn::PublicRules* rules,void* stream){pkn_visible_kernel<<<(n+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const pkn::State*>(states),output,n,*rules);return int(cudaGetLastError());}
__global__ void pkn_env_kernel(pkn::State* states,const pkn::State* initial,const int32_t* actions,int n,pkn::Rules rules,pkn::PublicRules visible,float* obs,float* final_obs,float* rewards,int32_t* flags,double faint_reward){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)pkn::env_step(states[i],initial[i],actions[i*2],actions[i*2+1],rules,visible,obs+i*1400,final_obs+i*1400,rewards+i*2,flags+i*3,faint_reward);}
extern "C" int pkn_env_step_device(void* states,const void* initial,const int32_t* actions,int n,const pkn::Move* moves,int count,const int8_t* chart,const pkn::PublicRules* visible,float* obs,float* final_obs,float* rewards,int32_t* flags,double faint_reward,void* stream){pkn_env_kernel<<<(n+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(static_cast<pkn::State*>(states),static_cast<const pkn::State*>(initial),actions,n,{moves,count,chart},*visible,obs,final_obs,rewards,flags,faint_reward);return int(cudaGetLastError());}
// Development observer for constructed-duel component validation. Full player
// view encoding and full-task admission are separate, unfinished obligations.
__global__ void pkn_component_views(const pkn::State* states,float* obs,float* rewards,int32_t* flags,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const auto& s=states[i];
 flags[i*3]=s.terminated;flags[i*3+1]=s.truncated;flags[i*3+2]=s.fault;
 for(int side=0;side<2;side++){
  const auto& own=pkn::active(s,side);const auto& foe=pkn::active(s,1-side);float* row=obs+(i*2+side)*16;
  for(int k=0;k<16;k++)row[k]=0;
  row[0]=float(own.hp)/pkn::maximum(1,own.maxhp);row[1]=float(pkn::public_hp(foe))/100;
  row[2]=own.terastallized;row[3]=foe.terastallized;row[4]=float(s.turn)/100;
  row[5]=s.tera_used[side];row[6]=s.tera_used[1-side];row[7]=float(own.pp[0])/pkn::maximum(1,own.maxpp[0]);
  for(int k=0;k<5;k++)row[8+k]=float(own.stats[k])/300;
  rewards[i*2+side]=!s.terminated||s.winner<0?0:s.winner==side?1:-1;
 }
}
__global__ void pkn_reset_done(pkn::State* states,const pkn::State* initial,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&(states[i].terminated||states[i].truncated)){
  uint64_t next_seed=states[i].rng;states[i]=initial[i];states[i].rng=next_seed;
 }
}
extern "C" int pkn_component_views_device(const void* states,float* obs,float* rewards,int32_t* flags,int n,void* stream){pkn_component_views<<<(n+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const pkn::State*>(states),obs,rewards,flags,n);return int(cudaGetLastError());}
extern "C" int pkn_reset_done_device(void* states,const void* initial,int n,void* stream){pkn_reset_done<<<(n+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(static_cast<pkn::State*>(states),static_cast<const pkn::State*>(initial),n);return int(cudaGetLastError());}
__global__ void pkn_gen_sets_kernel(pkn::GenRules rules,const int32_t* species,const uint32_t* teams,const int32_t* leads,uint64_t* seeds,pkn::GenSet* output,int32_t* ok,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)ok[i]=pkn::gen_random_set(rules,species[i],teams[i],leads[i],seeds[i],output[i]);}
__global__ void pkn_gen_teams_kernel(pkn::GenRules rules,uint64_t* seeds,pkn::GenTeam* output,int32_t* ok,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)ok[i]=pkn::gen_random_team(rules,seeds[i],output[i]);}
extern "C" int pkn_gen_sets_device(const pkn::GenRules* rules,const int32_t* species,const uint32_t* teams,const int32_t* leads,uint64_t* seeds,void* output,int32_t* ok,int n,void* stream){pkn_gen_sets_kernel<<<(n+63)/64,64,0,static_cast<cudaStream_t>(stream)>>>(*rules,species,teams,leads,seeds,static_cast<pkn::GenSet*>(output),ok,n);return int(cudaGetLastError());}
extern "C" int pkn_gen_teams_device(const pkn::GenRules* rules,uint64_t* seeds,void* output,int32_t* ok,int n,void* stream){pkn_gen_teams_kernel<<<(n+63)/64,64,0,static_cast<cudaStream_t>(stream)>>>(*rules,seeds,static_cast<pkn::GenTeam*>(output),ok,n);return int(cudaGetLastError());}

__global__ void pkn_materialize_sets_kernel(pkn::ResetRules rules,const pkn::GenSet* source,pkn::Mon* result,int32_t* ok,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)ok[i]=pkn::materialize_gen_set(rules,source[i],result[i]);}
extern "C" int pkn_materialize_sets_device(const pkn::ResetRules* rules,const void* source,void* result,int32_t* ok,int n,void* stream){pkn_materialize_sets_kernel<<<(n+63)/64,64,0,static_cast<cudaStream_t>(stream)>>>(*rules,static_cast<const pkn::GenSet*>(source),static_cast<pkn::Mon*>(result),ok,n);return int(cudaGetLastError());}
__global__ void pkn_materialize_teams_kernel(pkn::ResetRules rules,const pkn::GenTeam* source,pkn::Mon* result,int32_t* ok,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)ok[i]=pkn::materialize_gen_team(rules,source[i],result+i*6);}
extern "C" int pkn_materialize_teams_device(const pkn::ResetRules* rules,const void* source,void* result,int32_t* ok,int n,void* stream){pkn_materialize_teams_kernel<<<(n+63)/64,64,0,static_cast<cudaStream_t>(stream)>>>(*rules,static_cast<const pkn::GenTeam*>(source),static_cast<pkn::Mon*>(result),ok,n);return int(cudaGetLastError());}

__global__ void pkn_random_reset_kernel(pkn::GenRules generator,pkn::ResetRules constructor,const pkn::ResetSeeds* seeds,pkn::State* result,int32_t* ok,int n,pkn::Rules rules){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)ok[i]=pkn::reset_battle(generator,constructor,seeds[i],result[i],rules);}
extern "C" int pkn_random_reset_device(const pkn::GenRules* generator,const pkn::ResetRules* constructor,const void* seeds,void* result,int32_t* ok,int n,const pkn::Move* moves,int count,const int8_t* chart,void* stream){pkn_random_reset_kernel<<<(n+31)/32,32,0,static_cast<cudaStream_t>(stream)>>>(*generator,*constructor,static_cast<const pkn::ResetSeeds*>(seeds),static_cast<pkn::State*>(result),ok,n,{moves,count,chart});return int(cudaGetLastError());}

__global__ void pkn_random_env_kernel(pkn::State* states,const pkn::ResetSeeds* schedule,int length,int32_t* episodes,const int32_t* actions,int n,pkn::GenRules generator,pkn::ResetRules constructor,pkn::Rules rules,pkn::PublicRules visible,float* obs,float* final_obs,float* rewards,int32_t* flags,double faint_reward){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)pkn::random_env_step(states[i],schedule+i*length,length,episodes[i],actions[i*2],actions[i*2+1],generator,constructor,rules,visible,obs+i*1400,final_obs+i*1400,rewards+i*2,flags+i*3,faint_reward);}
extern "C" int pkn_random_env_step_device(void* states,const void* schedule,int length,int32_t* episodes,const int32_t* actions,int n,const pkn::GenRules* generator,const pkn::ResetRules* constructor,const pkn::Move* moves,int count,const int8_t* chart,const pkn::PublicRules* visible,float* obs,float* final_obs,float* rewards,int32_t* flags,double faint_reward,void* stream){pkn_random_env_kernel<<<(n+31)/32,32,0,static_cast<cudaStream_t>(stream)>>>(static_cast<pkn::State*>(states),static_cast<const pkn::ResetSeeds*>(schedule),length,episodes,actions,n,*generator,*constructor,{moves,count,chart},*visible,obs,final_obs,rewards,flags,faint_reward);return int(cudaGetLastError());}
extern "C" int pkn_cuda_prepare(size_t minimum_stack,size_t* actual_stack,size_t* free_bytes,size_t* total_bytes){
  size_t current=0;cudaError_t status=cudaDeviceGetLimit(&current,cudaLimitStackSize);if(status!=cudaSuccess)return int(status);
  if(current<minimum_stack){status=cudaDeviceSetLimit(cudaLimitStackSize,minimum_stack);if(status!=cudaSuccess)return int(status);}
  status=cudaDeviceGetLimit(actual_stack,cudaLimitStackSize);if(status!=cudaSuccess)return int(status);
  return int(cudaMemGetInfo(free_bytes,total_bytes));
}
