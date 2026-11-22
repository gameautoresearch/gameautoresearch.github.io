// SPDX-License-Identifier: Apache-2.0
// Ported from crforge core/src/main/java/org/crforge/core/battle/match/LadderMatch.java,
// core/src/main/java/org/crforge/core/battle/Battle.java, and
// core/src/main/java/org/crforge/core/battle/unit/PlayerCommands.java and Standard1v1Battle.java
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge).
// Native/LICENSE and native/NOTICE apply. No heap allocation or Java runtime dependency.
#pragma once
#include "timeline.hpp"
#include "match_form.hpp"
#include "placement.hpp"
namespace clash_native {
struct LadderMatch {
    RulesConfig config{};Timeline timeline{};MatchSide sides[2]{};
    int shuffleDraws[2]={},tick=0,clockMs=0,endTimerMs=0,winner=0,circleMs=0,tiebreakMs=0;
    bool ended=false,lastTicked=false,tiebreakDone=false,truncated=false,fault=false;
    CN_HD CN_NOINLINE void init(const RulesConfig& c,BattleRandom& random,const int playerWords[2]){
        *this=LadderMatch{};config=c;
        for(int s=0;s<2;++s){shuffleDraws[s]=random.next(0x0fffffff);DeckShuffle shuffle(add32(shuffleDraws[s],playerWords[s]));int32_t order[8];shuffle.fixed_eight(order);sides[s].hand.deal(order);sides[s].elixir=c.startingElixir*10000;}
    }
    template<class World> CN_HD int crowns(const World&w,int side)const {int other=1-side;if(!w.king_alive(other))return 3;int n=w.princess_count(other);return 2-(n<2?n:2);}
    template<class World> CN_HD bool decided(const World&w)const {return endTimerMs>0||!w.king_alive(0)||!w.king_alive(1)||(timeline.overtime(config)&&crowns(w,0)!=crowns(w,1))||timeline.time_up(config);}
    template<class World> CN_HD bool is_over(const World&w)const{return decided(w)&&endTimerMs>=config.endDelayMs;}
    template<class World> CN_HD bool tiebreak_armed(const World&w)const{return crowns(w,0)==crowns(w,1)&&!tiebreakDone&&!ended;}
    template<class World> CN_HD int gate(const World&w,int side,int card,int cost,int atTick)const {
        if(decided(w))return 4;if(!w.king_alive(side))return 8;if(sides[side].slot_of(card)<0)return 9;
        if(atTick<config.startCooldownTicks)return 22;if(sides[side].whole_elixir()<cost)return 13;return 0;
    }
    template<class World> CN_HD void end(World&w){if(ended)return;ended=true;w.end_match();timeline.freeze();endTimerMs=1;int a=crowns(w,0),b=crowns(w,1);winner=a<b?1:a>b?0:-1;}
    CN_HD static int drain_step(int lowest){return lowest<21?1:lowest<200?10:lowest<500?20:lowest<1000?40:50;}
    template<class World> CN_HD bool update(World&w){
        timeline.crownsEqual=crowns(w,0)==crowns(w,1);timeline.advance(config,tick);
        for(int side=0;side<2;++side)if(!w.king_alive(side)){circleMs+=50;int duration=config.endDelayMs/2,elapsed=circleMs<duration?circleMs:duration;int radius=w.arena_length();if(duration>=1)radius=radius*elapsed/duration;w.circle(side,radius);}
        if(endTimerMs>=1){endTimerMs+=50;if(decided(w)&&endTimerMs>=config.endDelayMs)return false;}return true;
    }
    template<class World> CN_HD bool run_update(World&w){bool ticked=update(w);if(ticked)w.tick_entities(tick);else w.cleanup();return ticked;}
    template<class World> CN_HD void after_update(World&w,bool ticked){lastTicked=ticked;if(ticked)timeline.crownsEqual=crowns(w,0)==crowns(w,1);}
    template<class World> CN_HD void tiebreak_step(World&w){
        int before=tiebreakMs;tiebreakMs+=50;w.hold_hits(true);lastTicked=false;
        if(before<=1450){if(w.clear_field()>=1)after_update(w,run_update(w));}
        else if(before>=3250){int l0=w.lowest_tower_hp(0),l1=w.lowest_tower_hp(1),step=drain_step(l0<l1?l0:l1);w.drain_towers(0,step);w.drain_towers(1,step);
            l0=w.lowest_tower_hp(0);l1=w.lowest_tower_hp(1);if(l0<1||l1<1||l0==l1){w.cleanup();tiebreakDone=true;}}
        w.tiebreak_step_end();w.execute_due_commands(tick);
    }
    // World.tick_entities must call sides[side].visit(config,timeline) in each king's
    // post-hook, after combat but before its post-hook followers, as the oracle does.
    template<class World> CN_HD bool step(World&w){
        if(is_over(w)||truncated||fault)return false;clockMs+=50;
        if(decided(w)&&!tiebreak_armed(w))end(w);
        if(decided(w)&&tiebreak_armed(w))tiebreak_step(w);
        else {w.execute_due_commands(tick);bool ticked=run_update(w);after_update(w,ticked);if(decided(w)&&!tiebreak_armed(w))end(w);}
        ++tick;return true;
    }
    // Environment cadence: up to DECISION_TICKS ticks (five in the pinned form, see
    // match_form.hpp), stopping only at natural oracle over or the caller's versioned
    // time limit. Truncation is separate.
    template<class World> CN_HD int decision(World&w,int tickLimit=0){int n=0;for(;n<DECISION_TICKS;++n){if(tickLimit>0&&tick>=tickLimit&&!is_over(w)){truncated=true;break;}if(!step(w))break;}return n;}
};
// Giving refusal matches PlayerCommands.Refusal's ordinal for supported kinds.
enum class GivingRefusal:int {None=0,MatchOver=1,KingDead=2,NotInHand=3,StartCooldown=4,CardPending=5,NotEnoughElixir=6,NoTile=7,CapacityFault=100,UnsupportedPromiseExpiry=101};
struct Giving {GivingRefusal refusal=GivingRefusal::None;int code=0,givenTick=0,runTick=0,cost=0,x=-1,y=-1;CN_HD bool given()const{return refusal==GivingRefusal::None;}};
struct QueuedPlayerCommand {int side=0,card=-1,givenTick=0,runTick=0,expiresTick=0,cost=0,x=0,y=0,code=0;bool ran=false,placed=false;};
struct PlayerCommands {
    static constexpr int CAPACITY=32;
    QueuedPlayerCommand queued[CAPACITY]{};int count=0;
    CN_HD int standing(const QueuedPlayerCommand&c,int tick)const {if(c.runTick>=tick||!c.ran)return 0;if(c.placed)return 3;if(tick<=c.expiresTick-2)return 1;return tick<=c.expiresTick+2?2:3;}
    CN_HD int promised(int side,int tick)const{int total=0;for(int i=0;i<count;++i)if(queued[i].side==side&&standing(queued[i],tick)!=3)total+=queued[i].cost;return total;}
    CN_HD bool pending(int side,int card,int tick)const{for(int i=0;i<count;++i)if(queued[i].side==side&&queued[i].card==card&&standing(queued[i],tick)==0)return true;return false;}
    CN_HD void compact(int tick){int size=0;for(int i=0;i<count;++i)if(standing(queued[i],tick)!=3)queued[size++]=queued[i];count=size;}
    CN_HD static GivingRefusal refusal_for(int gate){return gate==4?GivingRefusal::MatchOver:gate==8?GivingRefusal::KingDead:gate==9?GivingRefusal::NotInHand:gate==22?GivingRefusal::StartCooldown:GivingRefusal::NotEnoughElixir;}
    template<class World> CN_HD Giving try_card(const LadderMatch&m,const World&w,const CardRule cards[8],int side,int slot)const {
        int stamp=m.tick>1?m.tick:1;Giving out{};out.givenTick=stamp;out.runTick=stamp+20;
        int card=slot>=0&&slot<4?m.sides[side].hand.slots[slot]:-1;
        if(card<0){out.refusal=GivingRefusal::NotInHand;out.code=9;return out;}
        if(pending(side,card,m.tick)){out.refusal=GivingRefusal::CardPending;out.code=-1;return out;}
        int gate=m.gate(w,side,card,0,m.tick);if(gate){out.refusal=refusal_for(gate);out.code=gate;return out;}
        out.cost=cards[card].cost;int sure=0,dropping=0;
        for(int i=0;i<count;++i)if(queued[i].side==side){int s=standing(queued[i],m.tick);if(s==0||s==1)sure+=queued[i].cost;else if(s==2)dropping+=queued[i].cost;}
        int free=m.sides[side].whole_elixir()-sure;bool covered=(free>0?free:0)>=out.cost;
        int all=free-dropping;if(dropping>0&&((all>0?all:0)>=out.cost)!=covered){out.refusal=GivingRefusal::UnsupportedPromiseExpiry;out.code=-2;return out;}
        if(!covered){out.refusal=GivingRefusal::NotEnoughElixir;out.code=13;}return out;
    }
    template<class World> CN_HD Giving give(LadderMatch&m,const World&w,const CardRule cards[8],int side,int slot,int x,int y,const ArenaView&a,const MaskEntity*entities,int n){
        Giving out=try_card(m,w,cards,side,slot);if(!out.given())return out;
        int card=m.sides[side].hand.slots[slot];uint8_t mask[RULES_MAX_TILES];deploy_mask(a,cards[card],side,entities,n,mask);
        Point p=placement_find(a,cards[card],x,y,side,mask,m.config.symmetricalSnap);if(!p.found){out.refusal=GivingRefusal::NoTile;out.code=-1;return out;}
        compact(m.tick);if(count>=CAPACITY){m.fault=true;out.refusal=GivingRefusal::CapacityFault;out.code=-3;return out;}
        out.x=p.x;out.y=p.y;queued[count++]={side,card,out.givenTick,out.runTick,m.tick+60,out.cost,p.x,p.y,0,false,false};return out;
    }
    // Resolve in insertion order BEFORE match.update. World.deploy_card receives
    // complete placement, side/card/deployCounter BEFORE its increment as oracle
    // construction records it. The elixir spend/cycle precede all entity creation.
    template<class World> CN_HD void execute_due(LadderMatch&m,World&w,const CardRule cards[8],const ArenaView&a,const MaskEntity*entities,int n){
        for(int i=0;i<count;++i){auto&c=queued[i];if(c.ran||c.runTick>m.tick)continue;c.ran=true;c.code=m.gate(w,c.side,c.card,c.cost,m.tick);if(c.code)continue;
            Placement p=place_card(a,cards[c.card],c.x,c.y,c.side,entities,n,m.config.symmetricalSnap,m.config.laneSequence);c.code=p.code;if(p.code){if(p.code<0)m.fault=true;continue;}
            int deployIndex=m.sides[c.side].deployCounter;m.sides[c.side].play(c.card,c.cost,cards[c.card].productionStopMs);c.placed=true;w.deploy_card(c.side,c.card,deployIndex,p);
        }
    }
};
}
