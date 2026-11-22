// SPDX-License-Identifier: MIT
#pragma once
#include "observe.hpp"
namespace pkn {
// Shared CPU/CUDA environment transition. The current reset template bank is a
// component-development interface, not the unfinished native random-team reset.
PK_HD inline void env_advance(State& s,int a0,int a1,const Rules& rules,
                          const PublicRules& visible,float* final_obs,
                          float* rewards,int32_t* flags,double faint_reward){
  if(s.fault){flags[0]=0;flags[1]=0;flags[2]=s.fault;rewards[0]=0;rewards[1]=0;return;}
  double old_potential[2];
  for(int side=0;side<2;side++)old_potential[side]=double(public_dead(s,1-side)-potential_dead(s,side))/6;
  step(s,a0,a1,rules);
  flags[0]=s.terminated;flags[1]=s.truncated;flags[2]=s.fault;
  for(int side=0;side<2;side++){
    double potential=double(public_dead(s,1-side)-potential_dead(s,side))/6;
    double reward=faint_reward*(potential-old_potential[side]);
    if(s.terminated&&s.winner>=0)reward+=s.winner==side?1.0:-1.0;
    rewards[side]=float(reward);
    // Retain actual boundary views before reset for truncation bootstrapping.
    if(s.truncated)observe(s,side,final_obs+side*700,visible);
  }
}
PK_HD inline void env_step(State& s,const State& initial,int a0,int a1,const Rules& rules,
                          const PublicRules& visible,float* next_obs,float* final_obs,
                          float* rewards,int32_t* flags,double faint_reward){
  env_advance(s,a0,a1,rules,visible,final_obs,rewards,flags,faint_reward);
  if((s.terminated||s.truncated)&&!s.fault){uint64_t rng=s.rng;s=initial;s.rng=rng;}
  for(int side=0;side<2;side++)observe(s,side,next_obs+side*700,visible);
}
}
