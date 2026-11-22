// SPDX-License-Identifier: Apache-2.0
// Test-only POD bridge; observer port attribution is in observe.hpp.
#include "observe.hpp"
#include <cstdlib>
#include <vector>
using namespace clash_native;
struct TestObserveWorld {
 ObservationClock c;ObservationOwn own_sides[2];std::vector<ObservationEntity> entities;int error=0;
 int entity_count(){return int(entities.size());}const ObservationEntity& entity_at(int i){return entities[i];}
 const ObservationEntity* find_entity(int id){for(auto& e:entities)if(e.id==id)return &e;return nullptr;}
 const ObservationOwn& own(int seat){return own_sides[seat];}const ObservationClock& clock(){return c;}void fault(int code){error=code;}
};
extern "C" {
void* cn_observer_create(int elixir){auto* s=new ObserverState<>;s->reset(elixir);return s;}
void cn_observer_destroy(void* p){delete static_cast<ObserverState<>*>(p);}
void cn_observer_reset(void* p,int elixir){static_cast<ObserverState<>*>(p)->reset(elixir);}
void cn_observer_tick(void* p,const int* plays,int n,const int* released,int nr,int rate,int max_mana){auto& s=*static_cast<ObserverState<>*>(p);for(int k=0;k<n;++k){const int* r=plays+k*14;PublicPlay v;v.side=r[0];v.card=r[1];v.cast=r[2];v.form=r[3];v.tick=r[4];v.x=r[5];v.y=r[6];v.kind=r[7];v.cost=r[8];v.production_stop_ms=r[9];v.deploy_index=r[10];v.projectile_count=r[11];v.mirror=r[12];s.on_play(v);}for(int k=0;k<nr;++k)s.projectile_released(released[k*3],released[k*3+1],released[k*3+2]);s.tick(rate,max_mana);}
int cn_observer_frame(void* p,const int* clock,const int* own,const int* entities,int n,int* output,int* info,float* rewards){
 TestObserveWorld w;auto& c=w.c;c.tick=clock[0];c.episode_step=clock[1];c.time_ms=clock[2];c.section=clock[3];c.section_type=clock[4];c.section_left_ms=clock[5];c.full_bar_ms=clock[6];c.start_cooldown_ticks=clock[7];c.decision_ticks=clock[8];c.play_delay_ticks=clock[9];c.winner=clock[10];c.first_side=clock[11];c.crowns[0]=clock[12];c.crowns[1]=clock[13];c.illegal_now[0]=clock[14];c.illegal_now[1]=clock[15];c.fault=clock[16];c.not_modelled=clock[17];c.over=clock[18];c.overtime=clock[19];c.truncated=clock[20];c.reset=clock[21];
 for(int seat=0;seat<2;++seat){auto& o=w.own_sides[seat];const int* r=own+seat*96;o.elixir=r[0];o.reserved=r[1];o.hand_refill_ms=r[2];for(int k=0;k<4;++k){o.hand_card[k]=r[3+k];o.hand_cost[k]=r[7+k];o.hand_pending[k]=r[11+k];o.hand_form[k]=r[15+k];o.queue[k]=r[51+k];}for(int k=0;k<8;++k){o.deck[k]=r[19+k];o.deck_slots[k]=r[27+k];o.deck_evolution[k]=r[35+k];o.deck_level[k]=r[43+k];}o.next_card=r[55];for(int k=0;k<6;++k){o.tower_hp[k]=r[56+k];o.tower_max_hp[k]=r[62+k];}o.last_refusal=r[68];o.last_run_refusal=r[69];o.last_ability_run_refusal=r[70];o.illegal_total=r[71];o.pending_count=r[72];for(int k=0;k<4;++k)o.pending[k]={r[73+k*4],r[74+k*4],r[75+k*4],r[76+k*4]};}
 for(int k=0;k<n;++k){const int* r=entities+k*64;ObservationEntity e;e.id=r[0];e.kind=r[1];e.side=r[2];e.row=r[3];e.x=r[4];e.y=r[5];e.hp=r[6];e.max_hp=r[7];e.shield=r[8];e.state=r[9];e.level=r[10];e.target_id=r[11];e.deploy_ms=r[12];e.attack_ms=r[13];e.load_ms=r[14];e.speed_percent=r[15];e.hit_speed_percent=r[16];e.ability_ran_tick=r[17];e.aim_x=r[18];e.aim_y=r[19];e.time_left_ms=r[20];e.radius=r[21];e.deploy_index=r[22];int f=r[23];e.alive=f&1;e.air=f&2;e.building=f&4;e.king=f&8;e.clone=f&16;e.awake=f&32;e.no_move=f&64;e.no_attack=f&128;e.charged=f&256;e.hidden=f&512;e.invisible=f&1024;e.underground=f&2048;e.waiting=f&4096;e.has_hp=f&8192;e.bar_always=f&16384;e.bar_damaged=f&32768;e.bar_rolling=f&65536;e.ground_trail=f&131072;e.buff_count=r[24];for(int j=0;j<e.buff_count&&j<8;++j)e.buffs[j]={r[25+4*j],r[26+4*j],r[27+4*j],r[28+4*j]};w.entities.push_back(e);}
 observe(*static_cast<ObserverState<>*>(p),w,output,info,rewards);return w.error;
}
int cn_observer_battle_id(void* p,int seat,int id){return static_cast<ObserverState<>*>(p)->battle_id(seat,id);}
}
