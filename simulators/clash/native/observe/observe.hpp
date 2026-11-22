// SPDX-License-Identifier: Apache-2.0
// Ported from crforge env/src/main/java/org/crforge/env/{LadderEnv,SeatIds,SeatView,ElixirCount,BuffSlots,Schema}.java
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge).
// See ../LICENSE and ../NOTICE. Fixed8 has no private-cost tunnels or hero/evolution slots.
#pragma once
#include "../arith.hpp"
#include "schema.hpp"
namespace clash_native {
CN_HD inline int obs_min(int x,int y){return x<y?x:y;}
CN_HD inline int obs_max(int x,int y){return x>y?x:y;}
CN_HD inline int observation_x(int seat,int x){return seat==0?sub32(obs_schema::WIDTH,x):x;}
CN_HD inline int observation_y(int seat,int y){return seat==0?y:sub32(obs_schema::HEIGHT,y);}
struct ObservationBuff {int id=-1,remaining=0,speed=0,hit_speed=0;};
struct ObservationEntity {
    int id=0,kind=0,side=0,row=-1,x=0,y=0,hp=0,max_hp=0,shield=0,state=0,level=11,target_id=-1;
    int deploy_ms=0,attack_ms=0,load_ms=0,speed_percent=100,hit_speed_percent=100,ability_ran_tick=-1;
    int aim_x=0,aim_y=0,time_left_ms=0,radius=0,deploy_index=-1;
    bool alive=true,air=false,building=false,king=false,clone=false,awake=false,no_move=false,no_attack=false,charged=false;
    bool hidden=false,invisible=false,underground=false,waiting=false,has_hp=true;
    // For SeatView.Bars: frozen fixed8 rows use ALWAYS. A different row needs its own compiled bar rule.
    bool bar_always=true,bar_damaged=false,bar_rolling=false,ground_trail=false;
    // Name the two alignment bytes so POD copies and opaque checkpoints have a
    // deterministic object representation under both host and CUDA compilers.
    uint16_t reservedPadding=0;
    int buff_count=0;ObservationBuff buffs[8]={};
};
struct ObservationPending {int card=-1,x=0,y=0,run_tick=0;};
struct ObservationOwn {
    int elixir=0,reserved=0,crowns=0,hand_refill_ms=0;
    int hand_card[4]={-1,-1,-1,-1},hand_cost[4]={},hand_pending[4]={},hand_form[4]={};
    int deck[8]={-1,-1,-1,-1,-1,-1,-1,-1},deck_slots[8]={},deck_evolution[8]={},deck_level[8]={11,11,11,11,11,11,11,11};
    int queue[4]={-1,-1,-1,-1},next_card=-1;
    int tower_hp[6]={},tower_max_hp[6]={}; // own King/left/right then enemy King/left/right, screen left/right
    int last_refusal=0,last_run_refusal=0,last_ability_run_refusal=0,illegal_total=0,pending_count=0;
    ObservationPending pending[4]={};
};
struct ObservationClock {
    int tick=0,episode_step=0,time_ms=0,section=0,section_type=0,section_left_ms=0,full_bar_ms=0;
    int start_cooldown_ticks=0,decision_ticks=5,play_delay_ticks=20,winner=-1,first_side=0;
    int crowns[2]={},illegal_now[2]={},fault=0,not_modelled=0;
    bool over=false,overtime=false,truncated=false,reset=false;
};
struct PublicPlay {
    int side=0,card=-1,cast=-1,form=0,tick=0,x=0,y=0,kind=obs_schema::PLAY_TROOP,cost=0,production_stop_ms=0,deploy_index=0;
    int projectile_count=0; // Number of admitted projectiles actually found at trackPlays, not planned count.
    bool mirror=false;
};
struct ObservedPlay {PublicPlay play;int point_tick=-1;bool card_shown=false,point_shown=false;};
struct PublicElixir {
    int amount=0,stop_ms=0;
    CN_HD void start(int value){amount=value;stop_ms=0;}
    CN_HD void spend(int value){amount=sub32(amount,obs_min(value,amount));}
    CN_HD void play(int cost,int stop){spend(cost>0?mul32(cost,10000):0);if(stop>=1)stop_ms=stop;}
    CN_HD void gain(int value,int maximum){if(value>=1)amount=obs_min(add32(amount,value),mul32(maximum,10000));}
    CN_HD void regenerate(int full_bar_ms,int maximum){if(full_bar_ms==0)return;if(stop_ms>=1){stop_ms=obs_max(stop_ms,50)-50;return;}int step=div32(mul32(maximum,500000),full_bar_ms);if(step>=1)amount=obs_min(add32(amount,step),mul32(maximum,10000));}
};
struct ObservationIdEntry {int battle_id=-1,seat_id=-1,shown_at=-2,x=0,y=0;};
// No deletion: first-sight IDs never recycle within an episode. Capacity exhaustion is an
// explicit simulator fault. World can choose a larger template capacity without schema change.
template<int IdCapacity=8192,int PlayCapacity=2048> struct ObserverState {
    ObservationIdEntry ids[2][IdCapacity];int given[2]={},serial=0,fault=0;
    ObservedPlay plays[2][PlayCapacity];int play_count[2]={},seen[2][8]={{-1,-1,-1,-1,-1,-1,-1,-1},{-1,-1,-1,-1,-1,-1,-1,-1}},seen_count[2]={};
    PublicElixir public_elixir[2];
    CN_HD void reset(int starting_elixir){for(int side=0;side<2;++side){for(int i=0;i<IdCapacity;++i)ids[side][i]=ObservationIdEntry{};given[side]=play_count[side]=seen_count[side]=0;for(int k=0;k<8;++k)seen[side][k]=-1;public_elixir[side].start(starting_elixir);}serial=0;fault=0;}
    CN_HD ObservationIdEntry* entry(int seat,int battle_id){if(battle_id<0){fault=301;return nullptr;}unsigned start=(uint32_t(battle_id)*2654435761u)%IdCapacity;for(int probe=0;probe<IdCapacity;++probe){auto& e=ids[seat][(start+probe)%IdCapacity];if(e.battle_id==battle_id)return &e;if(e.battle_id<0){e.battle_id=battle_id;e.seat_id=given[seat]++;return &e;}}fault=302;return nullptr;}
    CN_HD int id(int seat,int battle_id){auto e=entry(seat,battle_id);return e?e->seat_id:-1;}
    CN_HD int battle_id(int seat,int seat_id)const{if(seat_id<0||seat_id>=given[seat])return -1;for(int i=0;i<IdCapacity;++i)if(ids[seat][i].seat_id==seat_id)return ids[seat][i].battle_id;return -1;}
    CN_HD void see(int seat,int card){for(int i=0;i<seen_count[seat];++i)if(seen[seat][i]==card)return;if(seen_count[seat]<8)seen[seat][seen_count[seat]++]=card;}
    CN_HD bool on_play(const PublicPlay& play){int side=play.side&1;if(play_count[side]==PlayCapacity){fault=303;return false;}if(play.kind==obs_schema::PLAY_TUNNEL||play.form!=0||play.mirror){fault=304;return false;} // unsupported by frozen task; do not reveal private costs
        ObservedPlay& p=plays[side][play_count[side]++];p.play=play;p.card_shown=true;p.point_shown=play.kind!=obs_schema::PLAY_LAUNCHED||play.projectile_count==0;p.point_tick=p.point_shown?play.tick:-1;see(1-side,play.card);public_elixir[side].play(play.cost,play.production_stop_ms);return true;}
    CN_HD void projectile_released(int side,int deploy_index,int step){side&=1;for(int i=play_count[side]-1;i>=0;--i){ObservedPlay& p=plays[side][i];if(p.play.deploy_index==deploy_index&&p.play.kind==obs_schema::PLAY_LAUNCHED){if(!p.point_shown){p.point_shown=true;p.point_tick=step;}return;}}}
    CN_HD bool aim_hidden(int side,int deploy_index)const{if(deploy_index<0)return false;side&=1;for(int i=play_count[side]-1;i>=0;--i){const ObservedPlay& p=plays[side][i];if(p.play.deploy_index==deploy_index)return p.play.kind==obs_schema::PLAY_LAUNCHED&&!p.point_shown;}return false;}
    // Call after tracking placed plays and first projectile landings in EVERY completed tick.
    // A tick that did not update entities supplies full_bar_ms=0. Costs already entered on_play.
    CN_HD void tick(int full_bar_ms,int maximum,int spend0=0,int spend1=0,int gain0=0,int gain1=0){for(int side=0;side<2;++side){public_elixir[side].spend(side?spend1:spend0);public_elixir[side].gain(side?gain1:gain0,maximum);public_elixir[side].regenerate(full_bar_ms,maximum);}}
};
CN_HD inline bool observation_sees(int seat,const ObservationEntity& e){return e.side==seat||e.kind!=obs_schema::KIND_CHARACTER||!e.waiting;}
CN_HD inline int observation_relative(int seat,int side){return side==seat?0:side==1-seat?1:2;}
CN_HD inline int observation_buff_rank(const ObservationBuff& b){return b.speed<=-100||b.hit_speed<=-100?0:b.speed<0||b.hit_speed<0?1:b.remaining==-1?3:2;}
template<class State> CN_HD void observation_moved(State& s,int seat,int* row,const ObservationEntity& e,int x,int y){auto entry=s.entry(seat,e.id);if(!entry)return;if(entry->shown_at==s.serial-1){row[obs_schema::E_DX]=sub32(x,entry->x);row[obs_schema::E_DY]=sub32(y,entry->y);}entry->shown_at=s.serial;entry->x=x;entry->y=y;}
// World query API: entity_count(), entity_at(i)->const ObservationEntity& (holder order),
// find_entity(id)->const ObservationEntity*, own(seat)->const ObservationOwn&,
// clock()->const ObservationClock&, fault(code). No enemy hand/deck/elixir is ever queried.
template<class State,class World> CN_HD void observation_target(State& s,World& w,int seat,int* row,int id){const auto* t=w.find_entity(id);if(!t||!observation_sees(seat,*t)){row[obs_schema::E_TARGET]=-1;return;}row[obs_schema::E_TARGET]=s.id(seat,t->id);row[obs_schema::E_TARGET_X]=observation_x(seat,t->x);row[obs_schema::E_TARGET_Y]=observation_y(seat,t->y);}
template<class State,class World> CN_HD void observation_entity(State& s,World& w,int seat,int* row,const ObservationEntity& e){
    using namespace obs_schema;int x=observation_x(seat,e.x),y=observation_y(seat,e.y);bool character=e.kind==KIND_CHARACTER,world=character||e.kind==KIND_TOWER;
    if(world){row[E_ID]=s.id(seat,e.id);row[E_KIND]=e.kind;row[E_SIDE]=observation_relative(seat,e.side);row[E_ROW]=e.row;row[E_X]=x;row[E_Y]=y;
        bool trail=e.side!=seat&&character&&e.underground&&e.ground_trail;
        if(trail){row[E_ROW]=row[E_HP]=row[E_MAX_HP]=row[E_SHIELD]=row[E_LEVEL]=-1;row[E_STATE]=e.state;row[E_FLAGS]=(e.alive?F_ALIVE:0)|F_HIDDEN|F_UNDERGROUND;observation_moved(s,seat,row,e,x,y);row[E_TARGET]=-1;for(int b=0;b<3;++b)row[E_BUFF_0+b*2]=-1;row[E_ABILITY_TICKS]=-1;return;}
        bool bar=e.side==seat||!character||(!e.underground&&(e.bar_always||(e.bar_damaged&&!e.bar_rolling)));
        if(e.has_hp){row[E_HP]=bar?e.hp:-1;row[E_MAX_HP]=e.max_hp;row[E_SHIELD]=bar?e.shield:-1;}
        row[E_STATE]=e.state;row[E_FLAGS]=(e.alive?F_ALIVE:0)|(e.air?F_AIR:0)|(e.building?F_BUILDING:0)|(e.king?F_KING:0)|(e.no_move?F_NO_MOVE:0)|(e.no_attack?F_NO_ATTACK:0)|(e.charged?F_CHARGED:0);
        if(character)row[E_FLAGS]|=(e.hidden?F_HIDDEN:0)|(e.invisible?F_INVISIBLE:0)|(e.underground?F_UNDERGROUND:0)|(e.clone?F_CLONE:0);else row[E_FLAGS]|=e.awake?F_AWAKE:0;
        row[E_LEVEL]=e.level;observation_moved(s,seat,row,e,x,y);observation_target(s,w,seat,row,e.target_id);
        row[E_DEPLOY_MS]=character?e.deploy_ms:0;row[E_ATTACK_MS]=e.attack_ms;row[E_LOAD_MS]=e.load_ms;row[E_SPEED_PERCENT]=e.speed_percent;row[E_HIT_SPEED_PERCENT]=e.hit_speed_percent;row[E_BUFF_COUNT]=e.buff_count;
        if(e.buff_count>8){w.fault(305);return;}int filled=0;for(int rank=0;rank<4&&filled<3;++rank)for(int k=0;k<e.buff_count&&filled<3;++k)if(observation_buff_rank(e.buffs[k])==rank){row[E_BUFF_0+2*filled]=e.buffs[k].id;row[E_BUFF_MS_0+2*filled++]=e.buffs[k].remaining;}
        for(int k=filled;k<3;++k)row[E_BUFF_0+2*k]=-1;row[E_ABILITY_TICKS]=e.ability_ran_tick<0?-1:sub32(w.clock().tick,e.ability_ran_tick);return;
    }
    // Original writeOther names a target BEFORE giving the projectile's own first-sight id.
    if(e.kind==KIND_PROJECTILE){row[E_LEVEL]=e.level;if(e.side!=seat&&s.aim_hidden(e.side,e.deploy_index))row[E_AIM_X]=row[E_AIM_Y]=-1;else{row[E_AIM_X]=observation_x(seat,e.aim_x);row[E_AIM_Y]=observation_y(seat,e.aim_y);}observation_target(s,w,seat,row,e.target_id);}
    else if(e.kind==KIND_AREA){row[E_LEVEL]=e.level;row[E_TIME_LEFT_MS]=e.time_left_ms;row[E_RADIUS]=e.radius;row[E_TARGET]=-1;}
    else{w.fault(306);return;}
    row[E_ID]=s.id(seat,e.id);row[E_KIND]=e.kind;row[E_SIDE]=observation_relative(seat,e.side);row[E_ROW]=e.row;row[E_X]=x;row[E_Y]=y;row[E_STATE]=-1;row[E_FLAGS]=F_ALIVE|((e.kind==KIND_AREA&&e.clone)?F_CLONE:0);observation_moved(s,seat,row,e,x,y);for(int b=0;b<3;++b)row[E_BUFF_0+b*2]=-1;row[E_ABILITY_TICKS]=-1;
}
template<bool Clear,class State,class World> CN_HD void observe_seat(State& s,World& w,int seat,int* o){
    using namespace obs_schema;if constexpr(Clear)for(int k=0;k<SEAT_INTS;++k)o[k]=0;const auto& own=w.own(seat);const auto& c=w.clock();int enemy=1-seat;
    o[H_TICK]=c.tick;o[H_RESULT]=!c.over?0:c.winner==seat?1:c.winner==enemy?2:3;o[H_ELIXIR]=own.elixir;o[H_ELIXIR_RESERVED]=own.reserved;o[H_CROWNS]=c.crowns[seat];o[H_CROWNS_ENEMY]=c.crowns[enemy];o[H_HAND_REFILL_MS]=own.hand_refill_ms;
    for(int k=0;k<4;++k){o[H_HAND_CARD+k]=own.hand_card[k];if(own.hand_card[k]>=0){o[H_HAND_COST+k]=own.hand_cost[k];o[H_HAND_PENDING+k]=own.hand_pending[k];}o[H_HAND_FORM+k]=own.hand_form[k];o[H_QUEUE+k]=own.queue[k];}
    o[H_NEXT_CARD]=own.next_card;for(int k=0;k<8;++k){o[H_DECK+k]=own.deck[k];o[H_ENEMY_SEEN+k]=s.seen[seat][k];o[H_DECK_SLOT+k]=own.deck_slots[k];o[H_DECK_EVOLUTION+k]=own.deck_evolution[k];o[H_DECK_LEVEL+k]=own.deck_level[k];o[H_ENEMY_SEEN_EVOLUTION+k]=-1;}
    for(int k=0;k<6;++k){o[H_OWN_KING_HP+2*k]=own.tower_hp[k];o[H_OWN_KING_MAX_HP+2*k]=own.tower_max_hp[k];}
    o[H_DECISION_TICKS]=c.decision_ticks;o[H_PLAY_DELAY_TICKS]=c.play_delay_ticks;o[H_SEAT]=seat;o[H_EPISODE_STEP]=c.episode_step;o[H_TIME_MS]=c.time_ms;o[H_SECTION]=c.section;o[H_SECTION_TYPE]=c.section_type;o[H_SECTION_LEFT_MS]=c.section_left_ms;o[H_FULL_BAR_MS]=c.full_bar_ms;o[H_OVERTIME]=c.overtime;o[H_PLAY_OPEN_IN_TICKS]=obs_max(c.start_cooldown_ticks-c.tick,0);o[H_LAST_REFUSAL]=own.last_refusal;o[H_LAST_RUN_REFUSAL]=own.last_run_refusal;o[H_ILLEGAL]=own.illegal_total;o[H_LAST_ABILITY_RUN_REFUSAL]=own.last_ability_run_refusal;
    int pending=obs_min(own.pending_count,4);o[H_PENDING_COUNT]=pending;for(int k=0;k<4;++k){int at=H_PENDING_CARD+k*4;if(k<pending){o[at]=own.pending[k].card;o[at+1]=observation_x(seat,own.pending[k].x);o[at+2]=observation_y(seat,own.pending[k].y);o[at+3]=obs_max(own.pending[k].run_tick-c.tick,0);}else o[at]=-1;}
    for(int k=0;k<2;++k){o[H_ABILITY_CHAMPION+6*k]=o[H_ABILITY_UNIT+6*k]=-1;o[H_ABILITY_CHARGES+6*k]=-1;}
    o[H_ENEMY_PLAY_COUNT]=s.play_count[enemy];o[H_ENEMY_ELIXIR]=s.public_elixir[enemy].amount;
    for(int k=0;k<8;++k){int at=H_ENEMY_PLAY_CARD+k*9,idx=s.play_count[enemy]-1-k;if(idx<0){o[at]=o[at+1]=-1;continue;}const auto& p=s.plays[enemy][idx];o[at]=p.card_shown?p.play.card:-1;o[at+1]=p.card_shown?p.play.cast:-1;o[at+2]=p.card_shown?p.play.form:-1;o[at+3]=sub32(c.tick,p.play.tick);o[at+4]=p.point_shown?observation_x(seat,p.play.x):-1;o[at+5]=p.point_shown?observation_y(seat,p.play.y):-1;o[at+6]=p.play.kind;o[at+7]=(p.card_shown?1:0)|(p.point_shown?2:0)|(p.card_shown&&p.play.mirror?4:0);o[at+8]=p.point_shown?sub32(c.tick,p.point_tick):-1;}
    int rows=0,overflow=0;for(int k=0;k<w.entity_count();++k){const auto& e=w.entity_at(k);if(e.kind!=KIND_CHARACTER&&e.kind!=KIND_TOWER)continue;if(!observation_sees(seat,e))continue;if(rows==MAX_ENTITIES){++overflow;continue;}observation_entity(s,w,seat,o+HEADER_INTS+rows++*ENTITY_INTS,e);}
    for(int k=w.entity_count()-1;k>=0;--k){const auto& e=w.entity_at(k);if(e.kind==KIND_CHARACTER||e.kind==KIND_TOWER)continue;if(rows==MAX_ENTITIES){++overflow;continue;}observation_entity(s,w,seat,o+HEADER_INTS+rows++*ENTITY_INTS,e);}
    o[H_ENTITY_COUNT]=rows;o[H_ENTITY_OVERFLOW]=overflow;if(s.fault)w.fault(s.fault);
}
// Default control retains its original template signature and clear-on-call semantics.
template<class State,class World> CN_HD void observe_seat(State&s,World&w,int seat,int*o){observe_seat<true>(s,w,seat,o);}
template<bool Clear,class State,class World> CN_HD void observe(State& s,World& w,int* two_seats,int* info,float* rewards){
    using namespace obs_schema;++s.serial;observe_seat<Clear>(s,w,0,two_seats);observe_seat<Clear>(s,w,1,two_seats+SEAT_INTS);const auto& c=w.clock();for(int k=0;k<12;++k)info[k]=0;info[I_TICK]=c.tick;info[I_EPISODE_STEP]=c.episode_step;info[I_WINNER]=!c.over?-1:c.winner==0||c.winner==1?c.winner:2;info[I_CROWNS_0]=c.crowns[0];info[I_CROWNS_1]=c.crowns[1];info[I_FAULT]=c.fault;info[I_RESET]=c.reset;info[I_ILLEGAL_0]=c.illegal_now[0];info[I_ILLEGAL_1]=c.illegal_now[1];info[I_TRUNCATED]=c.truncated;info[I_FIRST_SIDE]=c.first_side;info[I_NOT_MODELLED]=c.not_modelled;for(int seat=0;seat<2;++seat)rewards[seat]=!c.over?0.f:c.winner==seat?1.f:c.winner==1-seat?-1.f:0.f;
}
template<class State,class World> CN_HD void observe(State&s,World&w,int*two_seats,int*info,float*rewards){observe<true>(s,w,two_seats,info,rewards);}
} // namespace clash_native
