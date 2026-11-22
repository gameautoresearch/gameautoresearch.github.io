// SPDX-License-Identifier: MIT
#pragma once
#include "env_core.hpp"
#include "startup.hpp"
namespace pkn {
// A finite, source-bound schedule is consumed exactly once. No resampling,
// filtering, oracle fallback or template substitution is allowed on failure.
PK_HD inline void random_env_step(State& state,const ResetSeeds* schedule,int length,int32_t& episode,int a0,int a1,
    const GenRules& generator,const ResetRules& constructor,const Rules& rules,const PublicRules& visible,
    float* next_obs,float* final_obs,float* rewards,int32_t* flags,double faint_reward){
  env_advance(state,a0,a1,rules,visible,final_obs,rewards,flags,faint_reward);
  if((state.terminated||state.truncated)&&!state.fault){
    ++episode;
    if(episode>=length)state.fault=SEED_EXHAUSTED;
    else reset_battle(generator,constructor,schedule[episode],state,rules);
    // Boundary outcome flags describe the completed episode. A reset failure
    // is separately reported in the fault channel and must stop collection.
    flags[2]=state.fault;
  }
  for(int side=0;side<2;side++)observe(state,side,next_obs+side*700,visible);
}
}
