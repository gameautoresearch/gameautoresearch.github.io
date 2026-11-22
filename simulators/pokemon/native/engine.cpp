// SPDX-License-Identifier: MIT
#include "sim.hpp"
#include "observe.hpp"
#include "env_core.hpp"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include "teamgen.hpp"
#include "reset.hpp"
#include "startup.hpp"
#include "random_env.hpp"
#ifndef PKN_CPU_CHUNK
#define PKN_CPU_CHUNK 4
#endif
struct Arena {
  struct EnvIO {const pkn::State* initial;pkn::PublicRules visible;float *obs,*final_obs,*rewards;int32_t* flags;double faint_reward;const pkn::GenRules* generator=nullptr;const pkn::ResetRules* constructor=nullptr;const pkn::ResetSeeds* schedule=nullptr;int schedule_length=0;int32_t* episodes=nullptr;};
  std::vector<pkn::State> state;
  std::vector<std::thread> workers;
  std::mutex mutex;std::condition_variable ready,finished;
  uint64_t generation=0;bool stop=false;int pending=0;const int chunk_size;
  std::atomic<int> next{0};const int32_t* actions=nullptr;pkn::Rules rules{};
  const EnvIO* env=nullptr;
  void run_one(int j){if(env&&env->generator)pkn::random_env_step(state[j],env->schedule+j*env->schedule_length,env->schedule_length,env->episodes[j],actions[j*2],actions[j*2+1],*env->generator,*env->constructor,rules,env->visible,env->obs+j*1400,env->final_obs+j*1400,env->rewards+j*2,env->flags+j*3,env->faint_reward);else if(env)pkn::env_step(state[j],env->initial[j],actions[j*2],actions[j*2+1],rules,env->visible,env->obs+j*1400,env->final_obs+j*1400,env->rewards+j*2,env->flags+j*3,env->faint_reward);else pkn::step(state[j],actions[j*2],actions[j*2+1],rules);}
  Arena(int n,int count):state(n),chunk_size(PKN_CPU_CHUNK>0?PKN_CPU_CHUNK:pkn::maximum(1,pkn::minimum(32,(n+pkn::maximum(1,count*4)-1)/pkn::maximum(1,count*4)))) {
    for(int i=0;i<count;i++)workers.emplace_back([this]{
      uint64_t seen=0;
      for(;;){
        {std::unique_lock<std::mutex> lock(mutex);ready.wait(lock,[&]{return stop||generation!=seen;});if(stop)return;seen=generation;}
        for(;;){int begin=next.fetch_add(chunk_size,std::memory_order_relaxed);if(begin>=int(state.size()))break;int end=pkn::minimum(begin+chunk_size,int(state.size()));for(int j=begin;j<end;j++)run_one(j);}
        {std::lock_guard<std::mutex> lock(mutex);if(--pending==0)finished.notify_one();}
      }
    });
  }
  ~Arena(){{std::lock_guard<std::mutex> lock(mutex);stop=true;}ready.notify_all();for(auto& t:workers)t.join();}
  void run(const int32_t* input,pkn::Rules r,const EnvIO* io=nullptr){
    if(workers.empty()){actions=input;rules=r;env=io;for(size_t j=0;j<state.size();j++)run_one(j);return;}
    std::unique_lock<std::mutex> lock(mutex);actions=input;rules=r;env=io;next=0;pending=int(workers.size());generation++;ready.notify_all();finished.wait(lock,[&]{return pending==0;});
  }
};
extern "C" {
void* pkn_create(int n,int threads){if(n<=0||threads<0)return nullptr;try{return new Arena(n,threads);}catch(...){return nullptr;}}
void pkn_destroy(void* ptr){delete static_cast<Arena*>(ptr);}
void* pkn_states(void* ptr){return static_cast<Arena*>(ptr)->state.data();}
size_t pkn_state_size(){return sizeof(pkn::State);}
size_t pkn_mon_size(){return sizeof(pkn::Mon);}
size_t pkn_move_size(){return sizeof(pkn::Move);}
void pkn_step(void* ptr,const int32_t* actions,const pkn::Move* moves,int count,const int8_t* chart){static_cast<Arena*>(ptr)->run(actions,{moves,count,chart});}
void pkn_random(uint64_t* states,uint32_t* output,int n,int frames){for(int i=0;i<n;i++){pkn::State s{};s.rng=states[i];for(int f=0;f<frames;f++)output[i*frames+f]=pkn::next(s);states[i]=s.rng;}}
void pkn_masks(void* ptr,uint8_t* output){auto* a=static_cast<Arena*>(ptr);for(size_t i=0;i<a->state.size();i++)for(int side=0;side<2;side++)pkn::mask(a->state[i],side,output+i*28+side*14);}
void pkn_init_public(void* ptr){auto* a=static_cast<Arena*>(ptr);for(auto& s:a->state)pkn::initialize_public(s);}
void pkn_observe(void* ptr,double* output,const pkn::PublicRules* rules){auto* a=static_cast<Arena*>(ptr);for(size_t i=0;i<a->state.size();i++)for(int side=0;side<2;side++)pkn::observe(a->state[i],side,output+(i*2+side)*700,*rules);}
void pkn_observe_f32(void* ptr,float* output,const pkn::PublicRules* rules){auto* a=static_cast<Arena*>(ptr);for(size_t i=0;i<a->state.size();i++)for(int side=0;side<2;side++)pkn::observe(a->state[i],side,output+(i*2+side)*700,*rules);}
void pkn_env_step(void* ptr,const int32_t* actions,const pkn::Move* moves,int count,const int8_t* chart,const void* initial,const pkn::PublicRules* visible,float* obs,float* final_obs,float* rewards,int32_t* flags,double faint_reward){Arena::EnvIO io{static_cast<const pkn::State*>(initial),*visible,obs,final_obs,rewards,flags,faint_reward};static_cast<Arena*>(ptr)->run(actions,{moves,count,chart},&io);}
void pkn_random_env_step(void* ptr,const int32_t* actions,const pkn::Move* moves,int count,const int8_t* chart,const pkn::GenRules* generator,const pkn::ResetRules* constructor,const pkn::ResetSeeds* schedule,int length,int32_t* episodes,const pkn::PublicRules* visible,float* obs,float* final_obs,float* rewards,int32_t* flags,double faint_reward){Arena::EnvIO io{nullptr,*visible,obs,final_obs,rewards,flags,faint_reward,generator,constructor,schedule,length,episodes};static_cast<Arena*>(ptr)->run(actions,{moves,count,chart},&io);}
size_t pkn_public_rules_size(){return sizeof(pkn::PublicRules);}
size_t pkn_gen_species_size(){return sizeof(pkn::GenSpecies);}
size_t pkn_gen_rules_size(){return sizeof(pkn::GenRules);}
size_t pkn_gen_pool_size(){return sizeof(pkn::GenPool);}
size_t pkn_gen_counter_size(){return sizeof(pkn::GenCounter);}
size_t pkn_gen_move_size(){return sizeof(pkn::GenMove);}
size_t pkn_gen_move_list_size(){return sizeof(pkn::GenMoveList);}
size_t pkn_gen_set_spec_size(){return sizeof(pkn::GenSetSpec);}
size_t pkn_gen_set_size(){return sizeof(pkn::GenSet);}
size_t pkn_gen_team_size(){return sizeof(pkn::GenTeam);}
int pkn_gen_pool(const pkn::GenRules* rules,pkn::GenPool* pool,int type,const uint8_t* excluded){return pkn::gen_initialize_pool(*rules,*pool,type,excluded);}
int pkn_gen_forme(const pkn::GenRules* rules,int species,uint64_t* rng){return pkn::gen_forme(*rules,species,*rng);}
int pkn_gen_compatible(const pkn::GenRules* rules,int candidate,const int32_t* team,int length){return pkn::gen_compatible(*rules,candidate,team,length);}
int pkn_gen_query_moves(const pkn::GenRules* rules,int species,const int32_t* moves,int count,uint64_t abilities,int tera,pkn::GenCounter* result){return pkn::gen_query_moves(*rules,species,moves,count,abilities,tera,*result);}
int pkn_gen_ability(const pkn::GenRules* rules,int species,const int32_t* abilities,int count,const pkn::GenCounter* counter,uint32_t team,uint32_t moves,int tera,uint64_t* rng){return pkn::gen_ability(*rules,species,abilities,count,*counter,team,moves,tera,*rng);}
int pkn_gen_cull_moves(const pkn::GenRules* rules,int species,const pkn::GenMoveList* chosen,pkn::GenMoveList* pool,uint64_t abilities,uint32_t team,int tera){if(species<0||species>=rules->species_count||chosen->count<0||chosen->count>4||pool->count<0||pool->count>64||tera<0||tera>18)return 0;pkn::gen_cull_move_pool(*rules,species,*chosen,*pool,abilities,team,tera);return !pool->fault;}
int pkn_gen_moveset(const pkn::GenRules* rules,int species,pkn::GenMoveList* pool,uint64_t abilities,uint32_t team,int tera,int role,uint64_t* rng,pkn::GenMoveList* result){return pkn::gen_random_moveset(*rules,species,*pool,abilities,team,tera,role,*rng,*result);}
int pkn_gen_set_stage(const pkn::GenRules* rules,int species,pkn::GenMoveList* pool,const int32_t* options,int count,uint64_t abilities,uint32_t team,int tera,int role,int lead,uint64_t* rng,pkn::GenMoveList* result,int32_t* selected){
  if(!pkn::gen_random_moveset(*rules,species,*pool,abilities,team,tera,role,*rng,*result))return 0;
  pkn::GenCounter counter{};if(!pkn::gen_query_moves(*rules,species,result->values,result->count,abilities,tera,counter))return 0;
  selected[0]=pkn::gen_ability(*rules,species,options,count,counter,team,(pkn::gen_named_chosen(*rules,*result,"defog")?1:0)|(pkn::gen_named_chosen(*rules,*result,"flipturn")?2:0)|(pkn::gen_named_chosen(*rules,*result,"snowscape")?4:0),tera,*rng);
  if(selected[0]<0)return 0;selected[1]=pkn::gen_choose_item(*rules,species,selected[0],*result,counter,team,lead,role,*rng);return selected[1]>=0;
}
int pkn_gen_random_set(const pkn::GenRules* rules,int species,uint32_t team,int lead,uint64_t* rng,pkn::GenSet* result){return pkn::gen_random_set(*rules,species,team,lead,*rng,*result);}
int pkn_gen_random_team(const pkn::GenRules* rules,uint64_t* rng,pkn::GenTeam* result){return pkn::gen_random_team(*rules,*rng,*result);}
int pkn_gen_random_team_sized(const pkn::GenRules* rules,uint64_t* rng,pkn::GenTeam* result,int team_size){return pkn::gen_random_team(*rules,*rng,*result,team_size);}
void pkn_gen_sample_pool(int32_t* pool,int count,uint64_t* rng,int32_t* output){while(count){int index=count;output[index-1]=pkn::gen_sample_no_replace(*rng,pool,count);}}
size_t pkn_reset_species_size(){return sizeof(pkn::ResetSpecies);}
size_t pkn_reset_move_size(){return sizeof(pkn::ResetMove);}
size_t pkn_reset_rules_size(){return sizeof(pkn::ResetRules);}
int pkn_materialize_set(const pkn::ResetRules* rules,const pkn::GenSet* source,pkn::Mon* result){return pkn::materialize_gen_set(*rules,*source,*result);}
int pkn_materialize_team(const pkn::ResetRules* rules,const pkn::GenTeam* source,pkn::Mon* result){return pkn::materialize_gen_team(*rules,*source,result);}
int pkn_materialize_team_sized(const pkn::ResetRules* rules,const pkn::GenTeam* source,pkn::Mon* result,int team_size){return pkn::materialize_gen_team(*rules,*source,result,team_size);}
size_t pkn_reset_seeds_size(){return sizeof(pkn::ResetSeeds);}
int pkn_start_battle(const pkn::Mon* first,const pkn::Mon* second,uint64_t seed,int limit,pkn::State* state,const pkn::Move* moves,int count,const int8_t* chart){return pkn::start_battle(*state,first,second,seed,limit,{moves,count,chart});}
int pkn_start_battle_sized(const pkn::Mon* first,const pkn::Mon* second,uint64_t seed,int limit,pkn::State* state,const pkn::Move* moves,int count,const int8_t* chart,int team_size){return pkn::start_battle(*state,first,second,seed,limit,{moves,count,chart},team_size);}
int pkn_reset_battle(const pkn::GenRules* generator,const pkn::ResetRules* constructor,const pkn::ResetSeeds* seeds,pkn::State* state,const pkn::Move* moves,int count,const int8_t* chart){return pkn::reset_battle(*generator,*constructor,*seeds,*state,{moves,count,chart});}
// This is intentionally an admission barrier, not a claim of full task support.
int pkn_gen9_complete(){
#ifdef PKN_FULL_GEN9_ADMITTED
return 1;
#else
return 0;
#endif
}
}
