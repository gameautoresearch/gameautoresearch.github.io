// SPDX-License-Identifier: MIT
#pragma once
#include "reset.hpp"
namespace pkn {
// team_size occupies what was padding: 0 (every schedule written before it existed) means 6.
struct ResetSeeds {uint64_t battle,teams[2];int32_t limit,team_size;};
PK_HD inline int seeds_team_size(const ResetSeeds& seeds){return seeds.team_size?seeds.team_size:6;}
// This entry point never filters/resamples unsupported teams. It preserves all
// generated records and raises the explicit development fault before callbacks.
// Slots below team_size must hold a Pokémon; slots at or beyond it must be empty (maxhp 0).
PK_HD inline bool start_battle(State& state,const Mon* first,const Mon* second,uint64_t seed,int limit,const Rules& rules,int team_size=6){
  for(uint32_t i=0;i<sizeof(State);i++)reinterpret_cast<uint8_t*>(&state)[i]=0;
  state.rng=seed;state.limit=limit;state.winner=-1;state.last_faint_side=-1;
  if(team_size<1||team_size>6){state.fault=UNSUPPORTED;return false;}
  for(int side=0;side<2;side++)for(int i=0;i<6;i++){
    state.mons[side][i]=side?second[i]:first[i];state.order[side][i]=i;
    if(i<team_size?(state.mons[side][i].maxhp<=0||state.mons[side][i].unsupported):state.mons[side][i].maxhp!=0)state.fault=UNSUPPORTED;
  }
  if(state.fault)return false;
  initialize_public(state);
  // Constructor caches are raw. The first switch is inserted before the other
  // side becomes active; only the second insertion can see both ability holders.
  for(int side=0;side<2;side++)for(int i=0;i<6;i++)state.mons[side][i].speed_cache=state.mons[side][i].stats[4];
  int second_ability=active(state,1).ability;active(state,1).ability=NO_ABILITY;
  active(state,0).speed_cache=calculate_speed(active(state,0));
  append_action(state,make_action(state,RUN_SWITCH,0,-1,101));
  active(state,1).ability=second_ability;
  boost_event_order(state);active(state,1).speed_cache=calculate_speed(active(state,1));
  insert_action(state,make_action(state,RUN_SWITCH,1,-1,101));
  // Start itself does not run the ordinary action-finished Update event.
  QueuedAction first_switch=pop_action(state);run_switches(state,first_switch.side,rules);
  if(state.fault||state.terminated)return false;
  each_event(state);
  state.turn=1;state.mid_turn=0;
  for(int side=0;side<2;side++){active(state,side).active_turns=1;active(state,side).newly_switched=0;}
  update_trapping(state);refresh_requests(state);return true;
}
PK_HD inline bool reset_battle(const GenRules& generator,const ResetRules& constructor,const ResetSeeds& seeds,State& state,const Rules& rules){
  GenTeam teams[2]{};Mon pokemon[2][6]{};uint64_t rng[2]={seeds.teams[0],seeds.teams[1]};const int team_size=seeds_team_size(seeds);
  for(int side=0;side<2;side++)if(!gen_random_team(generator,rng[side],teams[side],team_size)||!materialize_gen_team(constructor,teams[side],pokemon[side],team_size)){
    for(uint32_t i=0;i<sizeof(State);i++)reinterpret_cast<uint8_t*>(&state)[i]=0;
    state.rng=seeds.battle;state.fault=UNSUPPORTED;return false;
  }
  for(int side=0;side<2;side++)for(int i=0;i<team_size;i++)if(!prepare_gen_start(constructor,teams[side].pokemon[i],pokemon[side][i])){state.fault=UNSUPPORTED;return false;}
  bool ok=start_battle(state,pokemon[0],pokemon[1],seeds.battle,seeds.limit,rules,team_size);
  state.team_rng[0]=rng[0];state.team_rng[1]=rng[1];return ok;
}
}
