// SPDX-License-Identifier: Apache-2.0
// Adapter for ports of crforge core/src/main/java/org/crforge/core/pathfinding/GridStateSetter.java,
// core/src/main/java/org/crforge/core/pathfinding/state/EntityStateVisit.java, ResumeHelper.java,
// and core/src/main/java/org/crforge/core/battle/unit/CharacterEntity.java, TowerEntity.java,
// CombatGate.java @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de.
#pragma once
#include "unit_state.hpp"
namespace clash_native {
// World interface (slot is the owner's stable native index):
//   entity(slot) -> PfEntity&
//   movement(slot) -> PfMovement* (nullptr for a crown tower)
//   targeting(slot) -> PfTargeting&
//   state_config(slot) -> StateVisitConfig (or const reference)
//   state_timers(slot) -> StateTimers&
//   targeting_active(slot) -> bool; set_targeting_active(slot,bool)
//   scaled_hit_step(slot), scaled_deploy_step(slot) -> int
//   keeps_targeting_off_mask() -> uint64_t (registry INACTIVE|ACTIVATING)
//   prepare_route(slot), clear_reference(slot), fault(code) -> void
// clear_reference runs the native reference setter including immediate route
// preparation. All five fixed8 troop rows have neither a buff when not attacking
// nor spawn attachments/deploy pushes: their entry/exit action hooks are no-ops.
template<class World> struct Fixed8StateQueries {
    World& world;int slot;
    CN_HD PfEntity& entity(){return world.entity(slot);}
    CN_HD void prepare_route(){world.prepare_route(slot);}
    CN_HD void clear_reference(){world.clear_reference(slot);}
    CN_HD void attacking_exit(){}
    CN_HD void deploying_entry(){}
    CN_HD bool may_hold_route(){return entity().movementComponent;}
    CN_HD bool grid_route_flag(){return false;}
    CN_HD bool grid_allows_route(bool){return true;}
    CN_HD bool has_hit_points(){return entity().hpPresent;}
    CN_HD int scaled_deploy_step_ms(){return world.scaled_deploy_step(slot);}
    CN_HD void apply_gate(bool targetingOn){
        PfEntity&e=entity();PfTargeting&t=world.targeting(slot);
        bool on=clash_native::combat_gate(e,t,targetingOn,e.alive,world.scaled_hit_step(slot),e.hpPresent,world.keeps_targeting_off_mask(),*this);
        t.targetingComponentActive=on;world.set_targeting_active(slot,on);
    }
    // Used by every state change: read the component's switch directly.
    CN_HD void combat_gate(){apply_gate(world.targeting_active(slot));}
    // CharacterEntity's post-hook tail gates its current switch by deployment;
    // TowerEntity's post-hook passes the switch directly, then applies holdingFire
    // outside this adapter (the qualified task's towers always attack).
    CN_HD void tail_gate(){PfEntity&e=entity();bool on=world.targeting_active(slot);if(e.movementComponent)on=on&&e.deployCountdown<=0&&e.state!=WaitingToDeploy;apply_gate(on);}
};
template<class World> CN_HD inline int fixed8_set_state(World&w,int slot,int next){
    Fixed8StateQueries<World>q{w,slot};auto&t=w.targeting(slot);t.targetingComponentActive=w.targeting_active(slot);
    int code=state_set(w.entity(slot),w.movement(slot),&t,w.state_config(slot),next,q);if(code<0)w.fault(231);return code;
}
template<class World> CN_HD inline int fixed8_resume(World&w,int slot){
    Fixed8StateQueries<World>q{w,slot};auto&t=w.targeting(slot);t.targetingComponentActive=w.targeting_active(slot);
    int code=state_resume(w.entity(slot),w.movement(slot),&t,w.state_config(slot),q);if(code<0)w.fault(232);return code;
}
template<class World> CN_HD inline int fixed8_post_state(World&w,int slot){
    Fixed8StateQueries<World>q{w,slot};auto&t=w.targeting(slot);t.targetingComponentActive=w.targeting_active(slot);
    int tail=state_visit(w.entity(slot),w.state_timers(slot),w.movement(slot),&t,w.state_config(slot),q);
    if(tail==1)q.tail_gate();else if(tail<0)w.fault(233);return tail;
}
}
