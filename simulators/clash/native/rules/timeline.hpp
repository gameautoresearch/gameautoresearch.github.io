// SPDX-License-Identifier: Apache-2.0
// Ported from crforge core/src/main/java/org/crforge/core/battle/match/Timeline.java,
// BattleTimeline.java and MatchSide.java @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de
// https://github.com/voonhous/crforge ; native/LICENSE and native/NOTICE apply.
#pragma once
#include "../deck.hpp"
namespace clash_native {
struct RulesConfig {
    int startingElixir=0,maxMana=0,startCooldownTicks=0,endDelayMs=0;
    int sectionCount=0,rateCount=0,cooldownCount=0,eventCount=0;
    int sectionLengths[8]={},sectionTypes[8]={},rateLengths[8]={},fullBarMs[8]={};
    int cooldownLengths[8]={},cooldownMs[8]={},eventTimes[16]={}; // All durations in ms.
    bool symmetricalSnap=true,laneSequence=true;
};
struct Timeline {
    int lastTick=-1,timeMs=0,frozenTimeMs=0,section=0,sectionElapsedMs=0,rate=0,rateElapsedMs=0;
    int cooldown=0,cooldownElapsedMs=0,eventsPassed=0,forcedRateSteps=0;
    bool firstAdvance=false,frozen=false,crownsEqual=true;
    CN_HD void advance(const RulesConfig& c,int tick) {
        firstAdvance=false;if(lastTick==tick)return;firstAdvance=lastTick<0;
        int ms=mul32(tick,50),delta=sub32(ms,timeMs);lastTick=tick;timeMs=ms;if(frozen)delta=0;
        sectionElapsedMs=add32(sectionElapsedMs,delta);
        if(sectionElapsedMs>=c.sectionLengths[section]&&c.sectionLengths[section]>=1)
            while(section+1<c.sectionCount) {
                int next=c.sectionTypes[section+1];if(next==2||(next==1&&!crownsEqual))break;
                sectionElapsedMs-=c.sectionLengths[section];++section;
                if(sectionElapsedMs<c.sectionLengths[section]||c.sectionLengths[section]<=0)break;
            }
        rateElapsedMs=add32(rateElapsedMs,delta);
        if(rateElapsedMs>=c.rateLengths[rate]||forcedRateSteps>=1)
            while(rate+1<c.rateCount) {
                if(c.rateLengths[rate]<=0&&forcedRateSteps<1)break;
                rateElapsedMs-=c.rateLengths[rate];if(forcedRateSteps>0){rateElapsedMs=0;--forcedRateSteps;}
                ++rate;if(rateElapsedMs<c.rateLengths[rate]&&forcedRateSteps<=0)break;
            }
        cooldownElapsedMs=add32(cooldownElapsedMs,delta);
        if(cooldownElapsedMs>=c.cooldownLengths[cooldown]&&c.cooldownLengths[cooldown]>=1)
            while(cooldown+1<c.cooldownCount) {
                cooldownElapsedMs-=c.cooldownLengths[cooldown];++cooldown;
                if(cooldownElapsedMs<c.cooldownLengths[cooldown]||c.cooldownLengths[cooldown]<=0)break;
            }
        while(eventsPassed<c.eventCount&&c.eventTimes[eventsPassed]<=ms)++eventsPassed;
    }
    CN_HD void freeze(){if(!frozen){frozen=true;frozenTimeMs=timeMs;}}
    CN_HD int section_left_ms(const RulesConfig& c)const {int d=c.sectionLengths[section]-sectionElapsedMs;return d>0?d:0;}
    CN_HD bool overtime(const RulesConfig& c)const{return c.sectionTypes[section]==1;}
    CN_HD bool time_up(const RulesConfig& c)const {
        if(c.sectionLengths[section]<1||sectionElapsedMs<c.sectionLengths[section])return false;
        if(section+1>=c.sectionCount)return true;
        int n=c.sectionTypes[section+1];return n==2||(n==1&&!crownsEqual);
    }
};
struct MatchSide {
    Hand hand{};int elixir=0,productionStopMs=0,spent=0,wasted=0,deployCounter=0,lastPlayed=-1,lastPlayedCopy=-1;
    CN_HD int whole_elixir()const{return elixir/10000;}
    CN_HD int slot_of(int card)const{for(int i=0;i<4;++i)if(hand.slots[i]==card)return i;return -1;}
    CN_HD void add(int amount,int maxMana){int cap=maxMana*10000,total=add32(elixir,amount),over=sub32(total,cap);if(over>=1)wasted=add32(wasted,over);elixir=total>0?(total<cap?total:cap):0;}
    CN_HD void spend(int amount){int take=amount<elixir?amount:elixir;spent=add32(spent,take);elixir-=take;}
    CN_HD void play(int card,int cost,int stopMs=0){if(cost>0)spend(cost*10000);if(stopMs>=1)productionStopMs=stopMs;hand.remove(slot_of(card));++deployCounter;lastPlayed=card;}
    CN_HD int visit(const RulesConfig& c,const Timeline& timeline){
        int filled=hand.refill(c.cooldownMs[timeline.cooldown]),full=c.fullBarMs[timeline.rate];
        if(full!=0){if(productionStopMs>=1)productionStopMs=productionStopMs>50?productionStopMs-50:0;else{int step=div32(mul32(c.maxMana,500000),full);if(step>=1)add(step,c.maxMana);}}
        lastPlayedCopy=lastPlayed;return filled;
    }
};
}
