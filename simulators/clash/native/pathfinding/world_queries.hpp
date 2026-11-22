// SPDX-License-Identifier: Apache-2.0
// Ported from crforge core/src/main/java/org/crforge/core/pathfinding/{GridMovementQueries,
// target/SelectionChain,target/ReferenceValidator,target/DefaultTargetSelection,
// target/RemovalNotice}.java and core/src/main/java/org/crforge/core/battle/unit/{WorldEntity,
// CharacterEntity,TowerEntity,CombatGate,BattleValidatorQueries}.java
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge)
#pragma once
#include "pathfinding.hpp"
#ifndef CN_PF_VISIT_BOUNDARY
#ifdef __CUDACC__
#define CN_PF_VISIT_BOUNDARY __noinline__
#else
#define CN_PF_VISIT_BOUNDARY
#endif
#endif
namespace clash_native {
// Minimal World contract (slots are stable until closing cleanup):
// entity(slot), movement(slot) -> PfMovement* (nullable for towers), targeting(slot) -> PfTargeting&;
// targetConfig(slot), movementConfig(slot) -> corresponding const POD config reference;
// grid() -> GridView&, searchScratch() -> RouteSearchBuffers<arena-cell-capacity>&;
// spatialQuery(PfSpatialQuery) -> PfNeighbours<CAP> in exact snapshot bucket order;
// enemyKing(slot) -> king slot or -1; enemyPrincessTowers(slot) -> PfNeighbours<2>, in
// creation order, removed towers omitted ONLY when holder cleanup unregisters them;
// speedPercents(slot) -> const int*, speedPercentCount(slot) -> count (buff modifiers);
// scaleHitTime(slot,baseMs) -> BuffComponent.hitSpeed(baseMs); matchEnded() -> bool;
// setUnitState(slot,state) -> full GridStateSetter semantics (including route on entry);
// prepareRoute(slot) -> immediate MovementChain.prepareRoute using fresh movement query;
// resumeUnit(slot) -> ResumeHelper semantics; applyAttack(attackerSlot,targetSlot) -> discarded.
// An absent target is -1. Every slot returned by spatial/default queries must remain readable
// until cleanup, including an entity killed earlier in this tick. No observation-row cap applies.

template<class World> struct Fixed8TargetQueries {
 World&w;int slot;bool selectionDone=false;int selectionResult=-1;
 CN_HD PfEntity&entity(int i){return w.entity(i);}
 CN_HD PfTargeting&state(){return w.targeting(slot);}
 CN_HD const PfTargetConfig&config(){return w.target_config(slot);}
 CN_HD bool validate(int index,int mode){return validate_reference(state(),entity(slot),config(),index,mode,*this);}
 CN_HD bool acceptsAttacker(int index){const PfEntity&e=entity(index);
  // A tower uses WorldEntity's unconditional acceptance. Fixed8 characters have no invisibility
  // or hide-time row, but waiting, underground, tag-hidden and parent/immunity gates still apply.
  if(e.crownTower)return true;
  if(e.attached||e.state==WaitingToDeploy||e.state==SpawnPathfind||e.state==IngamePathfind||
     (e.flags&e.flagBits.hidden))return false;
  if constexpr(requires {w.spawn_immune(index);})if(w.spawn_immune(index))return false;
  return true;
 }
 CN_HD bool pendingDamageKeeps(int index,int amount,int duration){
  const PfEntity&e=entity(index);if(duration>600||amount==0||!e.hpPresent)return true;
  // No fixed8 row has shields, healing, dash immunity, reduction or finite unit lifetime.
  // UNKILLABLE is retained as a tag gate even though the fixed8 source closure sets none.
  if(e.flags&e.flagBits.unkillable)return true;
  return pf_max(amount,1)<e.hp;
 }
 CN_HD auto candidates(int x,int y,int radius,bool useAll){
  // useAll is unreachable in the ordinary visitor, but target selector diagnostics may request
  // it. World can provide listQuery(true) when supporting this optional extension.
  if constexpr(requires {w.spatial_query_view(PfSpatialQuery{});}){
   if constexpr(requires {w.spatial_list_view(true);}){if(useAll)return w.spatial_list_view(true);}
   else assert(!useAll);
   return w.spatial_query_view(PfSpatialQuery{x,y,radius,0,true,false,0,-1});
  }else{
   if constexpr(requires {w.spatial_list(true);}){if(useAll)return w.spatial_list(true);}
   else assert(!useAll);
   return w.spatial_query(PfSpatialQuery{x,y,radius,0,true,false,0,-1});
  }
 }
 CN_HD int defaultTarget(){auto towers=w.enemy_princess_towers(slot);
  return default_target(entity(slot),w.enemy_king(slot),towers.indices,towers.count,*this);}
 CN_HD int select(PfTargetOutcome&out){if(selectionDone)return selectionResult;
  selectionResult=candidate_select(state(),entity(slot),config(),*this,out);
  selectionDone=true;return selectionResult;}
 CN_HD bool attackTimersHeld(){return w.match_ended();}
 CN_HD int scaleTimeStep(int base){return w.scale_hit_time(slot,base);}
 CN_HD bool stopsWithoutTarget(){return true;}
 CN_HD bool buildingKeepsAttacking(){return true;}
 CN_HD bool abilityInProgress(){return false;}
 CN_HD void setState(PfEntity&e,int value){assert(&e==&entity(slot));w.set_unit_state(slot,value);}
 CN_HD void turnToward(int index){
  // CharacterEntity installs its turn callback; TowerEntity deliberately leaves it absent.
  if(index<0||config().isBuilding)return;PfEntity&e=entity(slot);const PfEntity&r=entity(index);
  int x=sub32(r.x,e.x),y=sub32(r.y,e.y);normalize(x,y,256);e.dirX=x;e.dirY=y;
 }
 CN_HD void onStartingAttack(){assert(!config().hasOnStartingAttackAction);}
 CN_HD bool hit(int index){state().hitStarted=true;return w.apply_attack(slot,index);}
 CN_HD void prepareRoute(){w.prepare_route(slot);}
};

// WorldEntity constructor wiring; call after POD/config initialization, once per new entity.
CN_HD inline void initialize_fixed8_targeting(PfTargeting&t,const PfEntity&e,const PfTargetConfig&c){
 t.aliveCheckBypass=true;t.movementComponentActive=e.movementActive;
 if(c.loadFirstHit)t.loadTimerMs=c.loadTime;
}

// Called only when this entity's combat component is active in the phase snapshot. The active
// component mask and the combat gate are distinct: the holder owns their same-tick phase snapshots.
// Keep target selection and movement as separate CUDA call frames. Inlining both complete
// visitors into component_visit combines their large neighbour-query live ranges and spills
// registers across the component dispatch. Host/generic algorithms are unchanged. The macro
// may be overridden to empty for a source-bound compiler/throughput control.
template<class World> CN_HD CN_PF_VISIT_BOUNDARY void run_fixed8_targeting(World&w,int slot){
 PfEntity&e=w.entity(slot);PfTargeting&t=w.targeting(slot);const PfTargetConfig&c=w.target_config(slot);
 if(!c.isBuilding&&(e.state==Deploying||e.state==WaitingToDeploy))return;
 if(!c.isBuilding)t.routeLeadsAway=w.movement(slot)->routeLeadsAway!=0;
 Fixed8TargetQueries<World>q{w,slot};PfTargetOutcome out{};
 t.movementComponentActive=e.movementActive;targeting_visit(t,e,e.movementActive?w.movement(slot):nullptr,c,q,out);
 if(out.resumeRequested)w.resume_unit(slot);
}

// RemovalNotice precedes SelectionChain.unregister: run this for all survivors first, then drop
// default registration/seed references. Do NOT clear via set_reference: removal's retarget load
// differs from a stun's null path, including its 1ms lost-target countdown.
CN_HD inline void target_removed(PfTargeting&t,const PfTargetConfig&c,int removed,int replacement=-1){
 if(t.reference>=0&&t.reference==removed){bool load=true;
  if(replacement>=0){t.reference=replacement;t.keptByPendingDamageCheck=false;}
  else{load=!t.keptByPendingDamageCheck;t.reference=-1;t.keptByPendingDamageCheck=false;}
  if(load){if(c.loadAfterRetarget){if(t.attackTimerMs>=1){t.clearAttack();t.loadTimerMs=c.loadTime;}}
  else if(c.loadFirstHit){if(t.attackTimerMs>=1){int credit=sub32(c.loadTime,t.attackTimerMs);t.clearAttack();t.loadTimerMs=pf_max(credit,0);}}
  else if(t.attackTimerMs>=1&&c.hitSpeed>=2&&!t.visitSuspended&&t.burstProgressMs==0&&!c.overrideAttackFinishTime)t.targetLostTimerMs=1;}
 }
 if(t.previousReference==removed)t.previousReference=-1;
}

// Mirrors CombatGate.targetingOn. The returned boolean is the GATED_SLOT, not TARGETING_SLOT.
// Caller stores that boolean separately and raises COMBAT_DISABLED in the next flag word.
template<class World> CN_HD inline bool fixed8_targeting_gate(World&w,int slot,bool targetingOn){
 PfEntity&e=w.entity(slot);PfTargeting&t=w.targeting(slot);
 if(e.flags&(e.flagBits.inactive|e.flagBits.activating))return false;
 if(e.state==WaitingToDeploy)return false;
 assert(e.state!=Casting&&e.state!=CloneSetup&&e.state!=FollowingRemoved&&e.state!=AbilityFollowUp);
 int hitSpeed=w.scale_hit_time(slot,50);
 if(e.alive&&e.deployCountdown<=0&&hitSpeed!=0)return true;
 if((targetingOn||hitSpeed==0)&&t.reference>=0){Fixed8TargetQueries<World>q{w,slot};PfTargetOutcome out{};
  set_reference(t,e,w.target_config(slot),-1,q,out);
  if(out.resumeRequested)w.resume_unit(slot);}
 // All fixed8 rows have hit points: the nonacting entity's gate is always off.
 return false;
}

template<class World> struct Fixed8MovementQueries {
 World&w;int slot;bool targetingOn=true,hasReference=false;PfPoint reference{};
 // GridMovementQueries snapshots its reference coordinate when the movement visit's query is
 // created. This remains frozen even if a synchronous state/target action later changes it.
 CN_HD Fixed8MovementQueries(World&world,int index,bool on=true):w(world),slot(index),targetingOn(on){
  int ref=w.targeting(slot).reference;if(on&&ref>=0){const PfEntity&r=w.entity(ref);reference={r.x,r.y};hasReference=true;}}
 CN_HD PfEntity&entity(int i){return w.entity(i);}
 CN_HD PfMovement&movement(int i){PfMovement*p=w.movement(i);assert(p);return *p;}
 CN_HD PfTargeting&targeting(int i){return w.targeting(i);}
 CN_HD const PfMovementConfig&config(){return w.movement_config(slot);}
 CN_HD auto near(int x,int y,int radius){
  if constexpr(requires {w.spatial_query_view(PfSpatialQuery{});})return w.spatial_query_view(PfSpatialQuery{x,y,radius,0,false,false,0,-1});
  else return w.spatial_query(PfSpatialQuery{x,y,radius,0,false,false,0,-1});}
 CN_HD bool facingUpdateSuppressed(){return false;}
 CN_HD bool referenceAvailable(){return hasReference;}
 CN_HD PfPoint referencePoint(){return reference;}
 CN_HD int endpoint(int col,int row,int range){if(!hasReference)return -1;PfEntity&e=entity(slot);
  auto ds=[&](int x,int y){return squared_distance(reference.x,reference.y,x,y);};
  return select_endpoint(w.grid(),e.x,e.y,col,row,range,e.air,ds);}
 CN_HD int attackRange(){PfTargeting&t=targeting(slot);if(!t.targetingComponentActive)return 0;
  return attack_range(w.target_config(slot),t.reference<0?nullptr:&entity(t.reference));}
 CN_HD int gameModeGoal(){return -1;}
 CN_HD Route<> search(int sc,int sr,int gc,int gr,int adjust){PfEntity&e=entity(slot);
  GridCostLookup lookup{w.grid(),CellCosts{},e.state,e.lane,config().hovering,false};
  auto result=grid_search(w.grid().columns(),w.grid().rows(),lookup,sc,sr,gc,gr,adjust,w.search_scratch());
  Route<> route;for(int i=0;i<result.route.size();i++)route.add(result.route.nodes[i]);return route;}
 CN_HD RouteSearchStats search_into(int sc,int sr,int gc,int gr,int adjust,Route<>&route){PfEntity&e=entity(slot);
  GridCostLookup lookup{w.grid(),CellCosts{},e.state,e.lane,config().hovering,false};
  return grid_search_into(w.grid().columns(),w.grid().rows(),lookup,sc,sr,gc,gr,adjust,w.search_scratch(),route);}
 CN_HD Route<>&saved_route_scratch() requires requires(World&world){world.saved_route_scratch();}{
  return w.saved_route_scratch();}
 CN_HD bool referenceInRange(){return reference_in_range(targeting(slot),entity(slot),w.target_config(slot),*this);}
 CN_HD PfPoint specialWaypoint(){int x=sub32(entity(slot).x,reference.x),y=sub32(entity(slot).y,reference.y);
  if(distance(x,y)==0)y=1;normalize(x,y,attackRange());return {add32(reference.x,x),add32(reference.y,y)};}
 CN_HD void requestState(PfEntity&e,int state){assert(&e==&entity(slot));w.set_unit_state(slot,state);}
 CN_HD PfTargeting speedTargeting(){PfTargeting t=targeting(slot);t.targetingComponentActive=targetingOn;return t;}
 CN_HD int avoidanceGate(){return avoidance_gate(entity(slot),speedTargeting());}
 CN_HD int pushGate(){return push_gate(entity(slot),speedTargeting(),config());}
 CN_HD int speedBudget(){return speed_budget(entity(slot),movement(slot),speedTargeting(),config(),w.speed_percents(slot),w.speed_percent_count(slot));}
 CN_HD bool hasModifierComponent(){return true;}
 // GridMovementQueries.withBuffs receives BuffComponent.hitSpeed(100); follower divides by 2.
 CN_HD int scaledTimeStep(){return w.scale_hit_time(slot,100);}
 CN_HD int facingGate(){return facing_gate(entity(slot),speedTargeting());}
 CN_HD int routeRequest(){int s=entity(slot).state;return s==Moving||s==SpawnPathfind||s==IngamePathfind;}
};
template<class World> CN_HD CN_PF_VISIT_BOUNDARY void run_fixed8_movement(World&w,int slot,bool targetingOn=true){
 Fixed8MovementQueries<World>q{w,slot,targetingOn};PfEntity&e=w.entity(slot);
 assert(w.movement(slot));movement_visit(*w.movement(slot),e,w.movement_config(slot),w.grid(),q,(e.flags&e.flagBits.noCheckCollisions)!=0);
}
template<class World> CN_HD inline void prepare_fixed8_route(World&w,int slot){
 Fixed8MovementQueries<World>q{w,slot};if(!w.movement(slot))return;prepare_route(*w.movement(slot),w.entity(slot),w.grid(),q);
}
}
