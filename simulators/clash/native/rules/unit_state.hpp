// SPDX-License-Identifier: Apache-2.0
// Ported from crforge core/src/main/java/org/crforge/core/pathfinding/state/EntityStateVisit.java,
// ResumeHelper.java, StateTimers.java; core/src/main/java/org/crforge/core/pathfinding/GridStateSetter.java;
// core/src/main/java/org/crforge/core/battle/unit/CombatGate.java; and
// core/src/main/java/org/crforge/core/battle/action/WaitToActivate.java, WithDuration.java, ActionHolder.java
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge).
// Only states and action rows reachable in the frozen plain fixed8 task are accepted.
#pragma once
#include "../pathfinding/types.hpp"
namespace clash_native {
struct StateTimers {
    bool spawnImmune=false,removalRequested=false;int spawnImmuneElapsedMs=0,dashImmunityRemainingMs=0;
};
struct StateVisitConfig {
    int deployTimeMs=0,attackFinishTimeMs=250;bool kingTowerMiddle=false,deployTimeAffectedByCharacterSpeed=false;
};
CN_HD inline bool fixed8_state(int s){return s==Standing||s==Moving||s==Attacking||s==Deploying||s==WaitingToDeploy;}
// Q.prepare_route applies the existing native routing chain. Q.combat_gate is
// allowed to recurse into a state change (as a stun clears the reference).
// Returns -1 for unsupported state, 0 for a guard/no-op, 1 for an applied change.
template<class Q> CN_HD inline int state_set(PfEntity&e,PfMovement*movement,PfTargeting*targeting,const StateVisitConfig&c,int next,Q&q){
    if(!fixed8_state(next)||!fixed8_state(e.state))return -1;
    int old=e.state;if(old==next||e.deployCountdown>=1)return 0;
    if(old==Attacking){if(targeting&&targeting->targetingComponentActive)targeting->targetLostTimerMs=0;q.attacking_exit();}
    if(old==Deploying)e.deployCountdown=0;
    e.state=next;
    if(next==Standing||next==Attacking){if(movement){movement->route.clear();movement->routeLeadsAway=0;}}
    else if(next==Moving){if(movement)q.prepare_route();}
    else if(next==Deploying){if(movement&&e.movementComponent)e.movementActive=true;if(c.deployTimeMs>=0)e.deployCountdown=e.deployCountdown>c.deployTimeMs?e.deployCountdown:c.deployTimeMs;q.deploying_entry();}
    q.combat_gate();return 1;
}
// has_hit_points is the existence of the entity's HP object, NOT its alive test.
template<class Q> CN_HD inline int state_resume(PfEntity&e,PfMovement*m,PfTargeting*t,const StateVisitConfig&c,Q&q){
    if(e.state==WaitingToDeploy)return 0;if(!fixed8_state(e.state))return -1;
    if(q.may_hold_route()){bool flag=q.grid_route_flag();if(q.grid_allows_route(flag))return state_set(e,m,t,c,Moving,q);}
    if(q.has_hit_points())return state_set(e,m,t,c,Standing,q);
    // The fixed8 task has HP objects on every unit. Removal announcements from
    // ResumeHelper are represented by -2 for a caller that supplies another row.
    return c.kingTowerMiddle?0:-2;
}
// Returns 1 when the visit reaches visit_tail, 0 for an early deployment/stagger
// return, -1 for an unsupported state. The caller runs the tail combat gate ONLY
// on 1; a state change meanwhile runs its own gate even on an early return.
template<class Q> CN_HD inline int state_visit(PfEntity&e,StateTimers&timers,PfMovement*m,PfTargeting*t,const StateVisitConfig&c,Q&q){
    if(!fixed8_state(e.state))return -1;
    if(!pf_in_mask(e.state,0x850))e.delay=add32(e.delay,50);
    if(timers.spawnImmune){timers.spawnImmuneElapsedMs=add32(timers.spawnImmuneElapsedMs,50);if(timers.spawnImmuneElapsedMs>c.attackFinishTimeMs)timers.spawnImmune=false;}
    if(e.state==WaitingToDeploy){e.delay=e.delay>50?e.delay-50:0;if(e.delay==0){int changed=state_set(e,m,t,c,Deploying,q);if(changed<0)return changed;}return 0;}
    // Fixed8 has no not-attacking-section buff, kamikaze, dash, ability or hide row.
    e.pendingDamageDurationMs=e.pendingDamageDurationMs>50?e.pendingDamageDurationMs-50:0;
    if(!q.has_hit_points()&&!c.kingTowerMiddle&&e.deployCountdown==0)timers.removalRequested=true;
    timers.dashImmunityRemainingMs=sub32(timers.dashImmunityRemainingMs,50);
    if(e.deployCountdown>=1){int remaining=e.deployCountdown-(c.deployTimeAffectedByCharacterSpeed?q.scaled_deploy_step_ms():50);e.deployCountdown=remaining>0?remaining:0;if(remaining>0)return 0;int resumed=state_resume(e,m,t,c,q);if(resumed<0)return resumed;}
    return 1;
}
// Q.clear_reference invokes native TargetingVisit's null-reference path and its
// route request; it must preserve the same immediate route preparation semantics.
template<class Q> CN_HD inline bool combat_gate(PfEntity&e,PfTargeting&t,bool targetingOn,bool alive,int scaledHitStep,bool rowHasHitPoints,uint64_t keepsTargetingOff,Q&q){
    if((e.flags&keepsTargetingOff)!=0||e.state==WaitingToDeploy)return false;
    if(alive&&e.deployCountdown<=0&&scaledHitStep!=0)return true;
    if((targetingOn||scaledHitStep==0)&&t.reference>=0)q.clear_reference();
    return !rowHasHitPoints&&targetingOn;
}
// The exact two mechanically relevant king actions, with ActionHolder's retained
// finished runs. pending_pass is called in phases 1,2,3; run_pass once after the
// component slots. Fold tags at the entity pre-hook, BEFORE phase1 pending starts.
struct KingActivation {
    bool waitQueued=true,waitListed=false,waitFinished=false;
    bool activationQueued=false,activationListed=false,activationFinished=false;
    // Fill the existing +6..7 and +20..23 ABI padding. Aggregate assignment then
    // cannot copy uninitialized stack bytes into a retained raw World image.
    uint16_t reservedBeforeCounter=0;
    int64_t activationCounter=0;int durationMs=3300;uint32_t reservedTail=0;
    CN_HD void pending_pass(){if(waitQueued){waitQueued=false;waitListed=true;}if(activationQueued){activationQueued=false;activationListed=true;activationCounter=0;}}
    CN_HD void run_pass(bool condition){
        if(waitListed){if(waitFinished)waitListed=false;else if(condition){waitFinished=true;activationQueued=true;}}
        if(activationListed){if(activationFinished)activationListed=false;else if(activationCounter>=int64_t(durationMs)*1000)activationFinished=true;else activationCounter+=50000;}
    }
    CN_HD uint64_t tags(uint64_t inactiveTag,uint64_t activatingTag)const{return (waitListed?inactiveTag:0)|(activationListed?activatingTag:0);}
    CN_HD bool active()const{return !waitQueued&&!waitListed&&!activationQueued&&!activationListed;}
};
}
