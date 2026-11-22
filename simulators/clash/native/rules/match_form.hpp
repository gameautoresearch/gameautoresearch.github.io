// SPDX-License-Identifier: Apache-2.0
// Benchmark match form over the frozen fixed8 rules: decision cadence and an optional
// timeline override. Values come only from explicit compile flags written by
// native/match_forms.py; with none, the pinned form (5 ticks per decision, the
// generated Ladder timeline) is unchanged. The pinned generated data is never edited.
#pragma once
#include "timeline.hpp"
#ifndef CN_DECISION_TICKS
#define CN_DECISION_TICKS 5
#endif
namespace clash_native {
constexpr int DECISION_TICKS=CN_DECISION_TICKS;
static_assert(DECISION_TICKS>=1&&DECISION_TICKS<=50,"decision cadence outside 1..50 ticks");
// Applied to the pinned RulesConfig before any world copies it (LadderMatch::init).
inline void apply_match_form(RulesConfig& c){
#ifdef CN_TIMELINE_OVERRIDE
    // One define per value: nvcc splits a -D value at commas.
    constexpr int sections[2]={CN_TIMELINE_S0,CN_TIMELINE_S1},rates[3]={CN_TIMELINE_R0,CN_TIMELINE_R1,CN_TIMELINE_R2},
        cooldowns[3]={CN_TIMELINE_C0,CN_TIMELINE_C1,CN_TIMELINE_C2};
    if(c.sectionCount!=2||c.rateCount!=3||c.cooldownCount!=3){c.sectionCount=0;return;} // fails the form report check
    for(int i=0;i<2;++i)c.sectionLengths[i]=sections[i];
    for(int i=0;i<3;++i){c.rateLengths[i]=rates[i];c.cooldownLengths[i]=cooldowns[i];}
#else
    (void)c;
#endif
}
// Report: ticks, counts, then section lengths/types, rate lengths/full bars, cooldown lengths/ms.
inline int report_match_form(const RulesConfig& c,int* out,int cap){
    int n=0;auto put=[&](int v){if(n<cap)out[n]=v;++n;};
    put(DECISION_TICKS);put(c.sectionCount);put(c.rateCount);put(c.cooldownCount);
    for(int i=0;i<c.sectionCount;++i)put(c.sectionLengths[i]);for(int i=0;i<c.sectionCount;++i)put(c.sectionTypes[i]);
    for(int i=0;i<c.rateCount;++i)put(c.rateLengths[i]);for(int i=0;i<c.rateCount;++i)put(c.fullBarMs[i]);
    for(int i=0;i<c.cooldownCount;++i)put(c.cooldownLengths[i]);for(int i=0;i<c.cooldownCount;++i)put(c.cooldownMs[i]);
    return n;
}
}
