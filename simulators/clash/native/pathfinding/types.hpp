// SPDX-License-Identifier: Apache-2.0
// Ported from crforge core/src/main/java/org/crforge/core/pathfinding/{GridEntity.java,
// GridEntityState.java,EntityFlags.java,move/MovementState.java,target/TargetingState.java,
// target/TargetingConfig.java} @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de
// https://github.com/voonhous/crforge ; see ../LICENSE and ../NOTICE.
#pragma once
#include "../arith.hpp"
#include <cassert>
namespace clash_native {
#ifndef CN_ROUTE_CAPACITY
#define CN_ROUTE_CAPACITY 4096
#endif
#ifndef CN_ROUTE_NODE_TYPE
#define CN_ROUTE_NODE_TYPE int
#endif
using RouteNode = CN_ROUTE_NODE_TYPE;
template<class T> CN_HD inline T pf_min(T a,T b){return a<b?a:b;}
template<class T> CN_HD inline T pf_max(T a,T b){return a>b?a:b;}
CN_HD inline bool pf_in_mask(int state,int mask){return state>=0&&state<=13&&((mask>>state)&1)!=0;}
enum PfState : int {Standing=0,Moving=1,Attacking=2,Dashing=3,Deploying=4,Jumping=5,
 SpawnPathfind=6,IngamePathfind=7,CloneSetup=8,Morphing=9,Casting=10,
 WaitingToDeploy=11,FollowingRemoved=12,FollowingRemovedBuilding=13,
 ComponentsDisabled=14,RouteFollowingAlternate=15,AbilityFollowUp=16};
// World adapters populate masks from the pinned engine tag registry, never from enum ordinals.
struct PfFlagBits {
 uint64_t noMove=0,noMoveAllowAttract=0,noAttack=0,noCheckCollisions=0,noCheckAvoidance=0,
 disablePhysical=0,avoidanceAsObstacle=0,noPushedByEnemy=0,noPushedByAlly=0,noPushback=0,
 untargetable=0,lockTarget=0,ignoreRangeExtensionToKeepTarget=0,inactive=0,activating=0,
 combatDisabled=0,unkillable=0,hidden=0;
};
struct PfEntity {
 int id=0,side=0,type=5,lane=0,x=0,y=0,z=0,heightOffset=0,dirX=0,dirY=0,
 prevX=0,prevY=0,prevZ=0,state=0,collisionRadius=0,mass=0,deployCountdown=0,
 delay=0,blockCountdownMs=0,kingCandidate=0,targetable=0,squaredDistanceReduction=0,
 pendingDamageAmount=0,pendingDamageDurationMs=0,hp=0,hiddenCountdownMs=0;
 uint64_t flags=0,pendingFlags=0;
 PfFlagBits flagBits{};
 bool crownTower=false,building=false,air=false,occludes=false,occluder=false,
 movedSinceCopy=false,alive=true,movementActive=false,movementComponent=false,
 attached=false,clone=false,besideBuilding=false,presenceFlag=true,
 hpPresent=true,towerFlag=false,buildingTarget=false;
 int sightRangeForCharacters=0;
 CN_HD int zTotal()const{return add32(z,heightOffset);}
 CN_HD bool occludesCells()const{return occludes||(occluder&&(!movementComponent||!movedSinceCopy));}
};
// Fixed storage preserves all route nodes; overflow is a fault, never silent truncation.
// Standard arena is 36x64=2304 cells. Capacity 4096 exceeds every simple standard route.
template<int CAP=CN_ROUTE_CAPACITY> struct Route {
 RouteNode nodes[CAP]{}; int count=0;
 CN_HD bool empty()const{return count==0;}
 CN_HD int size()const{return count;}
 CN_HD int get(int i)const{assert(i>=0&&i<count);return nodes[i];}
 CN_HD int last()const{assert(count>0);return nodes[count-1];}
 CN_HD void add(int n){
  assert(count<CAP);
  // World may select uint16_t: every cell ID still has exactly the original int value.
  // Preserve full route capacity and refuse negative/out-of-range values before narrowing.
  assert(n>=0);RouteNode stored=static_cast<RouteNode>(n);assert(static_cast<int>(stored)==n);
  nodes[count++]=stored;
 }
 CN_HD int pop(){assert(count>0);return nodes[--count];}
 CN_HD void clear(){count=0;}
};
struct PfMovement {
 int targetX=0,targetY=0,pushbackBudget=0,blockCountdown=0,pushbackInFlight=0,
 attackPushback=0,explicitX=-1,explicitY=-1,routeLeadsAway=0,chargeProgress=-1,
 moveTimeMs=0,moveTimeMsCopy=0,avoidanceBlend=0,dashStopsInRange=1,waypointReached=0,
 jumpTotalDistance=0,dashTimeMs=0,pushX=0,pushY=0,pushCount=0,workX=0,workY=0,
 pushStuck=0,pushUnclamped=0,routeDirX=0,routeDirY=0;
 Route<> route{};
 CN_HD void clearPush(){pushX=pushY=pushCount=pushStuck=pushUnclamped=0;}
};
struct PfMovementConfig {
 int speed=0,jumpSpeed=0,spawnPathfindSpeed=0,ingamePathfindSpeed=0,
 chargeSpeedMultiplier=0,stopMovementAfterMs=0,waitMs=0,flyingHeight=0;
 bool hovering=false,entersWaterWhileSpawnPathfinding=false,ingamePathfindVisible=false,
 flyDirectPaths=false;
};
struct PfTargetConfig {
 int range=0,minimumRange=0,collisionRadius=0,sightRange=0,sightRangeForCharacters=0,
 sightClip=1000,sightClipSide=0,hitSpeed=0,loadTime=0,attackStartDelay=0,
 hitSpeedMultiplier=100,attackSequenceIndex=-1,attackFinishTime=250,stopTimeAfterAttack=0;
 bool attacksGround=true,attacksAir=false,isBuilding=false,targetOnlyBuildings=false,
 targetOnlyTroops=false,targetOnlyTowers=false,targetOnlyKingTower=false,doNotTargetTowers=false,
 hasProjectile=false,loadFirstHit=false,loadAfterRetarget=false,resetHitTimerWhenNoTarget=false,
 reselectsEveryVisit=false,dropsReferenceAfterAttack=false,targetLowestHp=false,
 keepTargetWithPendingDamage=false,overrideAttackFinishTime=false,hasOnStartingAttackAction=false;
};
struct PfTargeting {
 int reference=-1,previousReference=-1,targetLockingBuffs=0,lastReferenceX=0,
 lastReferenceY=0,lastReferenceZ=0,retargetCooldownMs=0,attackSequenceIndex=0,
 attackTimerMs=0,loadTimerMs=0,attackBlockTimerMs=0,resumeDelayElapsedMs=0,
 resumeDelayMs=0,visitHoldMs=0,targetLostTimerMs=0,specialLoadTimerMs=0,
 specialChargeTimerMs=0,dashWindupMs=0,burstProgressMs=0,burstDelayTimerMs=0,
 attackStartDelayMs=0;
 bool targetingComponentActive=true,movementComponentActive=false,routeLeadsAway=false,
 keptByPendingDamageCheck=false,hitStarted=false,chargeStrike=false,instantHit=false,
 attackTimeRoundedUp=false,attackResumeFlag=false,hitInProgress=false,
 hitInProgressWithoutReference=false,visitSuspended=false,specialLoadPending=false,
 aliveCheckBypass=false,skipSelectionNextVisit=false;
 CN_HD void clearAttack(){attackTimerMs=0;hitInProgress=hitInProgressWithoutReference=false;
 burstProgressMs=0;attackResumeFlag=false;targetLostTimerMs=resumeDelayElapsedMs=resumeDelayMs=0;}
};
struct PfTargetOutcome {bool resumeRequested=false,routePreparationRequested=false;};
struct PfPoint {int x,y;};
}
