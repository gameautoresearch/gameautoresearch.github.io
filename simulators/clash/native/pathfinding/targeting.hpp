// SPDX-License-Identifier: Apache-2.0
// Ported from crforge core/src/main/java/org/crforge/core/pathfinding/target/{AttackRange,
// RangeTest,SightRange,ReferenceSetter,ReferenceValidator,CandidateSelector,TargetPriority,
// DefaultTargetSelection,AttackTimerAdvance,TargetingVisit}.java; final acceptance from
// core/src/main/java/org/crforge/core/battle/unit/CharacterEntity.java accepts()/hidden().
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge)
#pragma once
#include "types.hpp"
#include "spatial.hpp"
namespace clash_native {
CN_HD inline int attack_range(const PfTargetConfig&c,const PfEntity*target=nullptr){
 return target&&target->type==3?250:add32(c.range,c.collisionRadius);}
CN_HD inline int minimum_range(const PfTargetConfig&c){return c.minimumRange>=1?add32(c.minimumRange,c.collisionRadius):c.minimumRange;}
CN_HD inline bool range_test(const PfEntity&target,int x,int y,int range,int minimum,bool innerOnly=false){
 int bound=add32(target.collisionRadius,range),outer=mul32(bound,bound),d=squared_distance(target.x,target.y,x,y);
 if(minimum>=1){int inner=add32(target.collisionRadius,minimum);bool beyond=d>=mul32(inner,inner);
 return innerOnly?beyond:d<=outer&&beyond;}return d<=outer||innerOnly;
}
template<class Q> CN_HD inline bool reference_in_range(const PfTargeting&t,const PfEntity&e,
 const PfTargetConfig&c,Q&q,int extension=0){return t.reference>=0&&range_test(q.entity(t.reference),e.x,e.y,
 add32(attack_range(c,&q.entity(t.reference)),extension),minimum_range(c));}
template<class Q> CN_HD inline void store_reference_position(PfTargeting&t,Q&q,bool z=false){
 if(t.reference>=0){const PfEntity&r=q.entity(t.reference);t.lastReferenceX=r.x;t.lastReferenceY=r.y;if(z)t.lastReferenceZ=r.zTotal();}}
// Selection's suppressWindUpReload is true; explicit clears/stuns use false.
template<class Q> CN_HD inline void set_reference(PfTargeting&t,PfEntity&e,const PfTargetConfig&c,int reference,
 Q&q,PfTargetOutcome&out,bool forceRefresh=false,bool suppressWindUpReload=false,bool skipRecheck=false){
 if(t.reference==reference&&!forceRefresh){store_reference_position(t,q);return;}
 t.previousReference=t.reference;t.reference=reference;t.keptByPendingDamageCheck=false;t.hitStarted=false;
 if(reference>=0){bool in=reference_in_range(t,e,c,q);
 if(in){if(c.loadAfterRetarget)t.clearAttack();else if(t.attackTimerMs>=1&&c.hasOnStartingAttackAction)q.onStartingAttack();}
 else{bool windup=c.loadFirstHit||c.loadAfterRetarget;
 if(windup&&t.attackTimerMs!=0){int into=c.hitSpeed==0?t.attackTimerMs:
 sub32(t.attackTimerMs,mul32(div32(t.attackTimerMs,c.hitSpeed),c.hitSpeed));
 t.loadTimerMs=sub32(c.loadTime,into);t.attackTimerMs=0;t.hitInProgress=t.hitInProgressWithoutReference=false;}
 else if(!windup)t.clearAttack();}
 if(!skipRecheck)q.validate(reference,0);
 }else{if(c.loadFirstHit||c.loadAfterRetarget){if(!suppressWindUpReload){t.clearAttack();t.loadTimerMs=c.loadTime;}}
 else if(c.resetHitTimerWhenNoTarget)t.clearAttack();}
 t.specialLoadPending=false;t.specialLoadTimerMs=0;
 if(t.targetLostTimerMs==0&&t.movementComponentActive){out.routePreparationRequested=true;
 // Real SelectionChain's outcome runs this synchronously at the store, before the next query.
 // Pure helper fixtures can omit it; the supplied World adapter always implements it.
 if constexpr(requires {q.prepareRoute();})q.prepareRoute();}
 store_reference_position(t,q);
}
// Final acceptance is CharacterEntity.accepts/hidden, beyond ReferenceValidator's column gates.
// WAITING_TO_DEPLOY can be indexed with flags==0, so the hidden tag alone is insufficient.
// q.acceptsAttacker optionally supplies world-owned spawn immunity/invisibility/parent checks;
// q.pendingDamageKeeps implements the owning world's projectile reservation lethality test.
template<class Q> CN_HD inline bool validate_reference(const PfTargeting&t,const PfEntity&e,
 const PfTargetConfig&c,int index,int mode,Q&q){
 if(index<0)return false;const PfEntity&r=q.entity(index);
 if(!t.aliveCheckBypass&&!r.alive)return false;
 if(r.id==e.id||pf_team(r)==pf_team(e)||(r.flags&r.flagBits.untargetable))return false;
 if(c.targetOnlyBuildings&&!r.building&&!r.towerFlag&&!r.buildingTarget)return false;
 if(c.targetOnlyTroops&&r.building)return false;
 if(c.doNotTargetTowers&&(r.crownTower||r.towerFlag))return false;
 if(c.targetOnlyTowers&&!r.towerFlag&&!r.crownTower)return false;
 if(c.targetOnlyKingTower&&!r.towerFlag)return false;
 if(r.air&&!c.attacksAir)return false;if(!r.air&&!c.attacksGround&&c.attacksAir)return false;
 if(r.presenceFlag&&(r.state==IngamePathfind||(r.state==Jumping&&!c.attacksAir)))return false;
 if(c.hasProjectile&&(mode&1)&&r.pendingDamageAmount!=0&&r.pendingDamageDurationMs<=600&&
 !q.pendingDamageKeeps(index,r.pendingDamageAmount,r.pendingDamageDurationMs))return false;
 if(!r.crownTower&&(r.state==WaitingToDeploy||r.state==SpawnPathfind||(r.flags&r.flagBits.hidden)))return false;
 if constexpr(requires {q.acceptsAttacker(index);})return r.hpPresent&&q.acceptsAttacker(index);
 return r.hpPresent;
}
template<class Q> CN_HD inline int default_target(const PfEntity&e,int seed,const int*candidates,int count,Q&q){
 int threshold=seed<0?INT32_MAX:squared_distance(e.x,e.y,q.entity(seed).x,q.entity(seed).y);
 int chosen=-1,smallest=INT32_MAX;
 for(int i=0;i<count;i++){int idx=candidates[i];const PfEntity&r=q.entity(idx);
 if(r.lane!=e.lane&&(e.delay<500||count==1))continue;int dx=abs32(sub32(r.x,e.x));
 if(dx<smallest){chosen=idx;smallest=dx;}}
 if(chosen>=0){const PfEntity&r=q.entity(chosen);int d=approximate_distance(sub32(r.x,e.x),sub32(r.y,e.y));
 if(mul32(d,d)<threshold&&q.validate(chosen,1))return chosen;}return seed;
}
template<class Q> CN_HD inline int candidate_select(PfTargeting&t,PfEntity&e,const PfTargetConfig&c,
 Q&q,PfTargetOutcome&out,bool useAll=false,bool buildingsOnly=false){
 if((t.reference>=0&&t.targetLockingBuffs>0)||e.state==Dashing||e.state==Jumping||(e.flags&e.flagBits.lockTarget))return t.reference;
 int sight=add32(c.sightRange,c.collisionRadius),minimum=minimum_range(c);
 auto candidates=q.candidates(e.x,e.y,add32(2000,sight),useAll);
 int best=-1,bestDistance=INT32_MAX,bestPriority=0,bestHp=INT32_MAX;
 for(int i=0;i<candidates.count;i++){int idx=candidates.indices[i];const PfEntity&r=q.entity(idx);
 if(!q.validate(idx,1)||!r.alive||(buildingsOnly&&!r.building)||r.hiddenCountdownMs>0)continue;
 if(minimum>=1&&!range_test(r,e.x,e.y,0,minimum,true))continue;
 int square=squared_distance(e.x,e.y,r.x,r.y),reach=add32(r.collisionRadius,sight);bool towerLike=false;
 if(!c.isBuilding&&r.presenceFlag&&r.sightRangeForCharacters>=1){reach=add32(pf_max(r.sightRangeForCharacters,c.range),r.collisionRadius);towerLike=true;}
 int total=add32(r.crownTower?2000:0,reach),reduced=pf_max(sub32(square,r.squaredDistanceReduction),0);
 if(uint32_t(reduced)>uint32_t(mul32(total,total))&&!useAll)continue;
 if(!(c.sightClip<1||useAll||towerLike||e.building)){int behind=pf_max(sub32(total,c.sightClip),0),dy=sub32(r.y,e.y);
 if(pf_team(e)==0?dy<sub32(0,behind):dy>behind)continue;}
 if(!(towerLike||useAll||c.sightClipSide<1)){int side=pf_max(sub32(total,c.sightClipSide),0);
 if(uint32_t(abs32(sub32(r.x,e.x)))>uint32_t(side))continue;}
 int priority=e.type==5&&c.targetOnlyBuildings&&(r.building||(r.presenceFlag&&r.buildingTarget))?1:0;
 if(c.targetLowestHp){int hp=r.hpPresent?r.hp:INT32_MAX;if(hp<bestHp){bestHp=hp;best=idx;}continue;}
 if(priority>bestPriority||(reduced<bestDistance&&priority>=bestPriority)){best=idx;bestDistance=reduced;bestPriority=priority;}}
 if(best>=0){set_reference(t,e,c,best,q,out,false,true);return t.reference;}
 int chosen=-1;if(t.reference>=0&&t.movementComponentActive&&t.routeLeadsAway&&q.validate(t.reference,1))chosen=t.reference;
 if(chosen<0)chosen=q.defaultTarget();if(q.validate(chosen,1))set_reference(t,e,c,chosen,q,out,false,true);
 return t.reference;
}
template<class Q> CN_HD inline void attack_timer_advance(PfTargeting&t,const PfTargetConfig&c,Q&q){
 if(q.attackTimersHeld()){t.attackTimerMs=0;t.hitInProgressWithoutReference=false;t.attackTimeRoundedUp=false;return;}
 int base=t.attackSequenceIndex!=-1?div32(c.hitSpeedMultiplier,2):50,step=q.scaleTimeStep(base),hitSpeed=c.hitSpeed;
 if(step<1){t.hitInProgressWithoutReference=false;return;}
 int attackTime=t.attackTimerMs;bool roundUp=t.chargeStrike||t.instantHit;
 if(attackTime==0&&!roundUp&&t.attackStartDelayMs>=1){t.attackTimerMs=0;t.hitInProgressWithoutReference=false;
 t.attackStartDelayMs=pf_max(t.attackStartDelayMs,50)-50;return;}
 if(attackTime==0&&!roundUp){int loadTime=c.loadTime,countdown=t.loadTimerMs;
 if(loadTime<=hitSpeed){attackTime=sub32(loadTime,countdown);t.attackTimerMs=attackTime;t.loadTimerMs=loadTime;
 int into=sub32(attackTime,mul32(div_or_zero(attackTime,hitSpeed),hitSpeed));
 if(into>=loadTime){if(t.reference>=0)t.hitInProgressWithoutReference=true;}else t.hitInProgressWithoutReference=false;
 }else if(countdown<=hitSpeed){if(t.reference>=0)t.hitInProgressWithoutReference=true;t.loadTimerMs=0;}
 else{t.attackTimerMs=0;t.hitInProgressWithoutReference=false;return;}}
 if(roundUp){t.chargeStrike=t.instantHit=false;t.attackTimeRoundedUp=true;
 int whole=mul32(div_or_zero(attackTime,hitSpeed),hitSpeed);
 t.attackTimerMs=add32(add32(attackTime,hitSpeed),sub32(whole,attackTime));return;}
 // Fixed8 has no burst rows; preserve timer if caller carries it for diagnostics.
 if(t.burstProgressMs>=1)t.burstProgressMs=add32(t.burstProgressMs,step);
 t.attackTimerMs=add32(attackTime,step);t.attackTimeRoundedUp=false;
}
CN_HD inline void attack_reset(PfTargeting&t,PfTargetOutcome&out){if(t.burstDelayTimerMs)return;t.clearAttack();out.resumeRequested=true;}
template<class Q> CN_HD inline void target_tail(PfTargeting&t,PfEntity&e,const PfTargetConfig&c,
 Q&q,PfTargetOutcome&out,bool justAttacked){
 if(justAttacked&&c.dropsReferenceAfterAttack)set_reference(t,e,c,-1,q,out);
 if(t.reference<0&&t.movementComponentActive&&t.specialLoadPending){t.specialLoadPending=false;t.specialLoadTimerMs=0;}
}
// Q owns side effects: setState(e,state), turnToward(index), onStartingAttack(), hit(index)
// and selection/validation. hit returns true iff the hit was discarded. All calls are synchronous.
template<class Q> CN_HD inline void targeting_visit(PfTargeting&t,PfEntity&e,PfMovement*movement,
 const PfTargetConfig&c,Q&q,PfTargetOutcome&out){
 bool skip=t.skipSelectionNextVisit;t.skipSelectionNextVisit=false;
 t.loadTimerMs=pf_max(t.loadTimerMs,50)-50;t.retargetCooldownMs=pf_max(t.retargetCooldownMs,50)-50;
 int state=e.state;if(state==SpawnPathfind||state==IngamePathfind||state==Morphing)return;
 store_reference_position(t,q,true);
 // No fixed8 row can dash, jump, burst or load a special attack.
 assert(state!=Dashing&&state!=Jumping&&!t.specialLoadPending&&!t.burstProgressMs);
 bool pushback=movement&&(movement->pushbackInFlight||movement->blockCountdown>0),
 attackPushback=movement&&movement->attackPushback;
 int block=t.attackBlockTimerMs;if(block>=1){t.attackBlockTimerMs=pf_max(block,50)-50;if(!t.attackBlockTimerMs)t.clearAttack();}
 if(t.resumeDelayMs>=1){t.resumeDelayElapsedMs=add32(t.resumeDelayElapsedMs,50);
 if(t.resumeDelayElapsedMs<t.resumeDelayMs)return;t.resumeDelayElapsedMs=t.resumeDelayMs=0;
 if(t.visitSuspended)return;t.attackTimerMs=1;t.attackResumeFlag=true;}else if(t.visitSuspended)return;
 if(t.visitHoldMs>=1){t.visitHoldMs=pf_max(t.visitHoldMs,50)-50;return;}
 bool inRange=false;if(t.reference>=0){const PfEntity&r=q.entity(t.reference);
 int extension=c.isBuilding||(e.flags&e.flagBits.ignoreRangeExtensionToKeepTarget)?0:25;
 inRange=reference_in_range(t,e,c,q,extension);
 if(!inRange&&c.hasProjectile&&reference_in_range(t,e,c,q,500)&&c.hitSpeed!=0)
 inRange=t.attackTimerMs%c.hitSpeed>50;
 (void)r;}
 if(c.reselectsEveryVisit&&t.retargetCooldownMs==0)set_reference(t,e,c,-1,q,out);
 t.keptByPendingDamageCheck=false;bool keep=false;
 if(t.reference>=0&&inRange){keep=q.validate(t.reference,1);
 if(!keep&&t.hitStarted&&c.keepTargetWithPendingDamage){keep=q.validate(t.reference,0);t.keptByPendingDamageCheck=keep;}}
 if(!keep&&t.retargetCooldownMs==0){bool idle=!(block>0||t.targetLostTimerMs!=0||t.burstProgressMs>=1);
 if(!skip&&(idle||c.reselectsEveryVisit))t.reference=q.select(out);
 if(t.reference>=0&&(q.entity(t.reference).state==IngamePathfind||
 (q.entity(t.reference).flags&q.entity(t.reference).flagBits.untargetable)))set_reference(t,e,c,-1,q,out);}
 if(block>0){target_tail(t,e,c,q,out,false);return;}
 if(t.reference<0){bool idle=t.hitInProgressWithoutReference||t.attackBlockTimerMs<0||t.burstProgressMs>=1;
 if(!(idle&&!q.stopsWithoutTarget())){
 if(t.targetLostTimerMs>=1){t.targetLostTimerMs=add32(t.targetLostTimerMs,50);
 int limit=c.overrideAttackFinishTime?c.attackFinishTime:250;
 if(t.targetLostTimerMs>=limit)t.clearAttack();target_tail(t,e,c,q,out,false);return;}
 if(!c.isBuilding||q.buildingKeepsAttacking())attack_reset(t,out);
 target_tail(t,e,c,q,out,false);return;}}
 int hitSpeed=c.hitSpeed;if(!hitSpeed)return;int entry=t.attackTimerMs;
 bool plain=reference_in_range(t,e,c,q),noAttack=(e.flags&e.flagBits.noAttack)!=0;
 if(noAttack){t.clearAttack();t.specialChargeTimerMs=0;}
 bool timing=t.hitInProgress,proceed=!pushback&&(noAttack?timing:plain||timing);
 if(proceed&&e.state==Casting&&!q.abilityInProgress())proceed=false;
 if(!proceed){bool ready=c.loadFirstHit?pushback&&timing:pushback&&timing&&c.loadAfterRetarget;
 if(!ready||attackPushback)attack_reset(t,out);else out.resumeRequested=true;
 target_tail(t,e,c,q,out,false);return;}
 if(e.state!=Attacking&&t.attackTimerMs==0)t.attackStartDelayMs=c.attackStartDelay;
 q.setState(e,Attacking);if(t.reference>=0)q.turnToward(t.reference);
 int before=div32(entry,hitSpeed);attack_timer_advance(t,c,q);
 t.hitInProgress=t.attackTimerMs%hitSpeed>50;
 if(entry==0||div32(t.attackTimerMs,hitSpeed)>before)q.onStartingAttack();
 bool pending=div32(t.attackTimerMs,hitSpeed)>before,
 hit=pending&&(t.reference>=0||(t.hitInProgressWithoutReference&&!q.stopsWithoutTarget()))&&t.specialLoadTimerMs==0;
 // attackDashTime is zero on the qualified deck: this is the same crossed-hit boundary.
 if(pending&&(t.reference>=0||(t.hitInProgressWithoutReference&&!q.stopsWithoutTarget())))q.turnToward(t.reference);
 if(!hit){target_tail(t,e,c,q,out,false);return;}
 t.hitInProgress=t.hitInProgressWithoutReference=false;bool discarded=q.hit(t.reference);
 if(c.loadFirstHit){if(discarded){t.attackTimerMs=0;t.loadTimerMs=0;}else{
 t.hitInProgress=t.hitInProgressWithoutReference=false;t.burstProgressMs=0;t.attackTimerMs=0;t.loadTimerMs=c.loadTime;
 t.attackResumeFlag=false;t.targetLostTimerMs=t.resumeDelayMs=t.resumeDelayElapsedMs=0;}}
 target_tail(t,e,c,q,out,true);
}
}
