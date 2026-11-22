// SPDX-License-Identifier: Apache-2.0
// Fixed8 world adapters for crforge source paths documented in world.hpp,
// at f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de.
#pragma once
#include "world.hpp"
#include "pathfinding/world_queries.hpp"
namespace clash_native {
CN_HD inline CN_WORLD_NOINLINE void NativeWorld::prepare_route(int i){if(unit_config(i).hasMovement)prepare_fixed8_route(*this,i);}
CN_HD inline CN_WORLD_NOINLINE void NativeWorld::clear_reference(int i){Fixed8TargetQueries<NativeWorld> q{*this,i};PfTargetOutcome out{};set_reference(targets[i],entities[i],target_config(i),-1,q,out);if(out.resumeRequested)fixed8_resume(*this,i);}
CN_HD inline CN_WORLD_NOINLINE void NativeWorld::component_visit(int i,int component){if(component==0)run_fixed8_targeting(*this,i);else if(component==1){if(entities[i].movementActive)run_fixed8_movement(*this,i,targeting_active(i));}else if(component==2){/* fixed8 has no lifetime decay or shield */}else if(component==3)tick_combat_buffs(buffs[i]);}
CN_HD inline void NativeWorld::entity_removed(int i,int gone){if(is_unit(i)&&is_unit(gone))target_removed(targets[i],target_config(i),gone);else if(holder.kinds[i]==4&&is_unit(gone)&&projectiles[i].target_id==holder.ids[gone]){if(spec->projectiles[projectileRows[i]].homing){projectiles[i].aim_x=entities[gone].x;projectiles[i].aim_y=entities[gone].y;projectiles[i].aim_z=entities[gone].z;}projectiles[i].target_id=0;projectiles[i].pending_registered=false;}}
CN_HD inline CN_WORLD_NOINLINE int NativeWorld::launch_unit_shot(int i,int target){const auto&u=unit_config(i);int row=u.projectile;if(row<0||target<0)return -1;int slot=holder.reserve(*this,4,0,0);if(slot<0)return -1;removeRequested[slot]=false;projectileRows[slot]=row;projectiles[slot]=ProjectileState{};auto&p=projectiles[slot];auto&e=entities[i];auto&t=targets[i];const auto&c=spec->projectiles[row];launch_unit_projectile(p,c,e.id,entities[target].id,e.side,e.x,e.y,e.z,e.heightOffset,u.projectileStartRadius,u.projectileStartZ,u.projectileYOffset,t.lastReferenceX,t.lastReferenceY);p.id=holder.ids[slot];
 holder.hand_over(*this,slot);return slot;}
CN_HD inline CN_WORLD_NOINLINE bool NativeWorld::deliver_hit(int i,int target){auto&e=entities[i];auto&t=targets[i];const auto&c=unit_config(i);t.hitStarted=true;if(e.flags&e.flagBits.noAttack)return true;
 bool missed=target>=0&&!c.building&&!range_test(entities[target],e.x,e.y,add32(attack_range(c.targeting,&entities[target]),1500),0);
 t.loadTimerMs=c.targeting.loadTime;++t.specialChargeTimerMs;
 if(missed){if(c.projectile<0)next_hit_id();return true;}
 if(target<0){if(c.projectile<0)next_hit_id();return false;}
 // HitApplication's default single-entry attack records the attack on its owner
 // before handing direct delivery to the queue or launching the projectile.
 e.pendingFlags|=spec->tags.attacking;t.hitStarted=true;
 if(c.projectile>=0){launch_unit_shot(i,target);return false;}
 return !queue_direct_hit(*this,e.id,e.side,entities[target].id,c.damage,c.crownPercent,e.x,e.y,t.lastReferenceX,t.lastReferenceY);
}
CN_HD inline CN_WORLD_NOINLINE void NativeWorld::deploy_card(int side,int card,int deployIndex,const Placement& place){const auto&c=spec->cards[card];int made=0;
 if(c.unitType>=0){for(int k=0;k<place.count;++k){const auto&p=place.units[k];if(add_unit(c.unitType,side,p.x,p.y,p.lane,p.waitMs)>=0)++made;}}
 else if(spec->cardProjectile[card]>=0){int row=spec->cardProjectile[card],firstChain=-1;const auto&spell=spec->spells[card];int count=pf_max(spell.count,1)*pf_max(spell.waves,1);if(count>30){fault(221);return;}
  if(spell.ring){for(int i=0;i+spell.waves<=WORLD_CHAINS;++i){bool free=true;for(int j=0;j<spell.waves;++j)free=free&&chainUse[i+j]==0;if(free){firstChain=i;break;}}if(firstChain<0){fault(222);return;}}
  ProjectileState temporary[30];ProjectileChain temporaryChains[3];int king=kingSlots[side];const auto&e=entities[king];int actual=cast_spell_projectiles(temporary,30,temporaryChains,3,firstChain,spell,spec->projectiles[row],e.id,side,e.x,e.y,e.collisionRadius,place.x,place.y,spec->cardVocab[card],deployIndex,random);if(actual<0){fault(223);return;}
  for(int k=0;k<spell.waves&&spell.ring;++k){chains[firstChain+k]=temporaryChains[k];chainUse[firstChain+k]=0;}
  for(int k=0;k<actual;++k){int slot=holder.reserve(*this,4,0,0);if(slot<0)return;removeRequested[slot]=false;projectileRows[slot]=row;projectiles[slot]=temporary[k];projectiles[slot].id=holder.ids[slot];if(projectiles[slot].chain_index>=0)++chainUse[projectiles[slot].chain_index];holder.hand_over(*this,slot);++made;}
 }else if(spec->cardArea[card]>=0){int slot=holder.reserve(*this,3,0,0);if(slot<0)return;removeRequested[slot]=false;areaCard[slot]=card;areas[slot]={holder.ids[slot],side,place.x,place.y,spec->zap.life_ms};holder.hand_over(*this,slot);++made;}
 else{fault(224);return;}
 PublicPlay play;play.side=side;play.card=play.cast=spec->cardVocab[card];play.tick=match.tick;play.x=place.x;play.y=place.y;play.kind=c.unitType>=0?obs_schema::PLAY_TROOP:spec->cardProjectile[card]>=0?obs_schema::PLAY_LAUNCHED:obs_schema::PLAY_AREA;play.cost=c.cost;play.production_stop_ms=c.productionStopMs;play.deploy_index=deployIndex;play.projectile_count=spec->cardProjectile[card]>=0?made:0;if(publicPendingCount>=PlayerCommands::CAPACITY)fault(225);else publicPending[publicPendingCount++]=play;
}
}
