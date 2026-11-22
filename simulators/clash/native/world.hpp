// SPDX-License-Identifier: Apache-2.0
// Independent fixed8 world orchestration ported from crforge
// core/src/main/java/org/crforge/core/battle/{EntityHolder,BattleEntity}.java,
// core/src/main/java/org/crforge/core/battle/unit/{BattleWorld,WorldEntity,
// CharacterEntity,TowerEntity,Standard1v1Battle,HitQueue,EntityDeaths}.java and
// core/src/main/java/org/crforge/core/battle/projectile/ProjectileEntity.java
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge).
// Header-only, integer, CN_HD; Java remains an external qualification oracle.
#pragma once
#ifndef CN_ROUTE_CAPACITY
#define CN_ROUTE_CAPACITY 2304
#endif
#ifndef CN_ROUTE_NODE_TYPE
#define CN_ROUTE_NODE_TYPE uint16_t
#endif
#include "spec.hpp"
#include "rules/selected_action.hpp"
#include "holder.hpp"
#include "rules/world_state_adapter.hpp"
#include "observe/observe.hpp"
#ifdef __CUDACC__
#define CN_WORLD_NOINLINE __noinline__
#else
#define CN_WORLD_NOINLINE
#endif
namespace clash_native {
constexpr int WORLD_CAPACITY=512,WORLD_CHAINS=128,WORLD_HITS=2048;
struct WorldHit {CombatHit hit{};bool kill=false;};
struct NativeWorld {
 const NativeTaskSpec* spec=nullptr;
 NativeHolder<WORLD_CAPACITY> holder{};LadderMatch match{};PlayerCommands commands{};BattleRandom random{};
 // Explicit ABI-reserved bytes make raw snapshots deterministic across heap/
 // aggregate initialization. They occupy existing padding; no rule reads them.
 uint32_t reservedBeforeEntities=0;
 PfEntity entities[WORLD_CAPACITY]{};PfMovement movements[WORLD_CAPACITY]{};
 PfTargeting targets[WORLD_CAPACITY]{};StateTimers timers[WORLD_CAPACITY]{};
 CombatHealth health[WORLD_CAPACITY]{};CombatBuffs buffs[WORLD_CAPACITY]{};
 ProjectileState projectiles[WORLD_CAPACITY]{};CombatAreaEffect areas[WORLD_CAPACITY]{};
 CombatTarget combatTargets[WORLD_CAPACITY]{};
 int rows[WORLD_CAPACITY]{},projectileRows[WORLD_CAPACITY]{},areaCard[WORLD_CAPACITY]{},chainUse[WORLD_CHAINS]{};
 bool removeRequested[WORLD_CAPACITY]{};
 ProjectileChain chains[WORLD_CHAINS]{};WorldHit hits[WORLD_HITS]{};int hitCount=0,hitSerial=0;
 int present[WORLD_CAPACITY]{},presentCount=0,overlay[2][2304]{},gridCurrent=0;
 GridView arenaGrid{};CellCosts costs{};RouteSearchBuffers<2304> routeScratch{};Route<> savedRouteScratch{};PfSpatialIndex<WORLD_CAPACITY> spatial{};
 uint32_t reservedBeforeKings=0;
 KingActivation kings[2]{};int kingSlots[2]={-1,-1},lastRefusal[2]{},lastRunRefusal[2]{},illegalTotal[2]{},illegalNow[2]{};
 PublicPlay publicPending[PlayerCommands::CAPACITY]{};int publicPendingCount=0;
 int episodeStep=0,firstSide=0,faultCode=0;bool matchEnded=false,hitsHeld=false,justReset=true;
 uint8_t reservedBeforeObserver=0;
 ObserverState<> observer{};
 // These observation caches contain extracted PODs only, never engine inputs.
 ObservationEntity observationEntities[WORLD_CAPACITY]{};ObservationOwn observationOwn[2]{};ObservationClock observationClock{};int observationCount=0;
 int renderedObs[14048]{},renderedInfo[12]{};uint8_t renderedMasks[4614]{};uint16_t reservedBeforeRewards=0;float renderedRewards[2]{};bool rendered=false;
 uint8_t reservedAfterRendered[3]{};
 // Caller-owned spatial query scratch, private to this game. Fixed8 consumers finish each
 // traversal before the next query (PushPass's aligned query follows its completed near loop).
 PfNeighbours<WORLD_CAPACITY> spatialResultScratch{};int spatialKeysScratch[WORLD_CAPACITY]{};
 // Placement consumers are not reentrant in fixed8: give() only queues, and execute_due's
 // deploy_card path creates entities/projectiles and state gates without querying masks.
 // mask_entities assigns every field in [0,n) before use; unused entries are never read.
 MaskEntity placementScratch[WORLD_CAPACITY]{};uint32_t reservedTail=0;
 CN_HD void fault(int code){if(!faultCode)faultCode=code;match.fault=true;}
 CN_HD PfEntity& entity(int i){return entities[i];}
 CN_HD const PfEntity& entity(int i)const{return entities[i];}
 CN_HD PfTargeting& targeting(int i){return targets[i];}
 CN_HD PfMovement* movement(int i){return spec->units[rows[i]].hasMovement?&movements[i]:nullptr;}
 CN_HD const NativeUnitConfig& unit_config(int i)const{return spec->units[rows[i]];}
 CN_HD const PfTargetConfig& target_config(int i)const{return unit_config(i).targeting;}
 CN_HD const PfMovementConfig& movement_config(int i)const{return unit_config(i).movement;}
 CN_HD StateVisitConfig state_config(int i)const{return unit_config(i).state;}
 CN_HD StateTimers& state_timers(int i){return timers[i];}
 CN_HD bool targeting_active(int i)const{return (holder.active[i]&1)!=0;}
 CN_HD void set_targeting_active(int i,bool on){if(on)holder.active[i]|=1;else{holder.active[i]&=uint8_t(~1);entities[i].pendingFlags|=spec->tags.path.combatDisabled;targets[i].chargeStrike=false;}targets[i].targetingComponentActive=on;}
 CN_HD uint64_t keeps_targeting_off_mask()const{return spec->tags.keepsTargetingOff;}
 CN_HD int scaled_hit_step(int i){auto lookup=[&](int)->const CombatBuffConfig&{return spec->stun;};return combat_buff_speed(buffs[i],50,true,lookup);}
 CN_HD int scaled_deploy_step(int i){auto lookup=[&](int)->const CombatBuffConfig&{return spec->stun;};return combat_buff_speed(buffs[i],50,false,lookup);}
 CN_HD int speed_percent(int i,bool attack=false){auto lookup=[&](int)->const CombatBuffConfig&{return spec->stun;};return combat_buff_speed(buffs[i],100,attack,lookup);}
 CN_HD ArenaView arena_view()const{return {36,64,spec->arenaCells};}
 CN_HD void bind(const NativeTaskSpec* value){spec=value;arenaGrid.width=36;arenaGrid.height=64;arenaGrid.tiles=spec->arenaCells;arenaGrid.current=overlay[gridCurrent];arenaGrid.previous=overlay[1-gridCurrent];arenaGrid.buildingCost=costs.building;}
 CN_HD void refresh_health(int i){auto&e=entities[i];e.hp=health[i].hp;e.alive=health[i].hp>0;}
 CN_HD int add_unit(int row,int side,int x,int y,int lane=-1,int wait=-1){
  const auto& c=spec->units[row];int bits=c.hasMovement?15:13;
  int slot=holder.reserve(*this,5,bits,bits);if(slot<0)return -1;
  rows[slot]=row;removeRequested[slot]=false;entities[slot]=PfEntity{};targets[slot]=PfTargeting{};timers[slot]=StateTimers{};health[slot]=CombatHealth{};buffs[slot]=CombatBuffs{};movements[slot]=PfMovement{};
  auto&e=entities[slot];e.id=holder.ids[slot];e.side=side;e.x=x;e.y=y;e.collisionRadius=c.collisionRadius;e.mass=c.mass;e.flagBits=spec->tags.path;e.lane=lane<0?lane_assignment(arenaGrid,x,y):lane;e.building=c.building;e.occludes=c.building;e.occluder=c.occluder;e.crownTower=c.king||c.princess;e.kingCandidate=c.king;e.towerFlag=c.king;e.buildingTarget=c.building;e.targetable=1;e.presenceFlag=true;e.movementComponent=c.hasMovement;e.movementActive=c.hasMovement;e.hpPresent=true;e.sightRangeForCharacters=c.targeting.sightRangeForCharacters;
  health[slot].hp=health[slot].maximum=c.maximumHp;refresh_health(slot);
  auto&t=targets[slot];t.aliveCheckBypass=true;t.movementComponentActive=c.hasMovement;t.attackSequenceIndex=0;if(c.targeting.loadFirstHit)t.loadTimerMs=c.targeting.loadTime;
  if(c.king||c.princess){e.state=Standing;if(c.king)kingSlots[side]=slot;}
  else{movements[slot].targetX=x;movements[slot].targetY=y;movements[slot].routeDirY=side?-256:256;e.dirY=side?-256:256;e.state=Deploying;e.deployCountdown=c.state.deployTimeMs;if(wait>=0){e.state=WaitingToDeploy;e.deployCountdown=0;e.delay=wait;e.movementActive=false;}}
  if(!c.king&&!c.princess){Fixed8StateQueries<NativeWorld> q{*this,slot};q.apply_gate(targeting_active(slot));}
  holder.hand_over(*this,slot);return slot;
 }
 CN_HD CN_WORLD_NOINLINE void init(const NativeTaskSpec* value,int seed){
  // Reset mutable state in place. CPU and CUDA use this same operation; there is
  // no hidden JVM state and no transition lookup table.
  spec=value;reservedBeforeEntities=reservedBeforeKings=reservedTail=0;reservedBeforeObserver=0;reservedBeforeRewards=0;for(int i=0;i<3;++i)reservedAfterRendered[i]=0;
  costs=CellCosts{};holder=NativeHolder<WORLD_CAPACITY>{};commands=PlayerCommands{};random=BattleRandom(seed);hitCount=hitSerial=presentCount=gridCurrent=0;arenaGrid=GridView{};savedRouteScratch.count=0;faultCode=episodeStep=firstSide=0;matchEnded=hitsHeld=false;justReset=true;
  for(int i=0;i<2;++i){kings[i]=KingActivation{};kingSlots[i]=-1;lastRefusal[i]=lastRunRefusal[i]=illegalTotal[i]=illegalNow[i]=0;for(int j=0;j<2304;++j)overlay[i][j]=0;}
  for(int i=0;i<WORLD_CHAINS;++i)chainUse[i]=0;
  bind(value);random.next(1);random.next(1);int words[2]={0,0};match.init(spec->rules,random,words);
  for(int i=0;i<6;++i)add_unit(spec->towerRow[i],i/3,spec->towerX[i],spec->towerY[i]);holder.cleanup(*this);
  observer.reset(spec->rules.startingElixir*10000);publicPendingCount=0;rendered=false;spatialResultScratch.count=0;
 }
 CN_HD bool hidden(int i)const{return (entities[i].flags&spec->tags.hidden)!=0||(!unit_config(i).king&&!unit_config(i).princess&&entities[i].state==WaitingToDeploy);}
 CN_HD bool is_unit(int i)const{return holder.kinds[i]==5;}
 CN_HD bool removable(int i)const{if(removeRequested[i])return true;if(is_unit(i))return !unit_config(i).king&&health[i].hp<1;return holder.kinds[i]==4?projectiles[i].released:areas[i].countdown<0;}
 CN_HD void add_tag_fold(int i){if(!is_unit(i))return;auto&e=entities[i];uint64_t action=unit_config(i).king?kings[e.side].tags(spec->tags.inactive,spec->tags.activating):0;e.flags=e.pendingFlags|unit_config(i).rowTags|action;e.pendingFlags=0;}
 CN_HD void before_registration(int){}
 CN_HD void on_registered(int i){if(is_unit(i))entities[i].id=holder.ids[i];else if(holder.kinds[i]==4){auto&p=projectiles[i];const auto&c=spec->projectiles[projectileRows[i]];int target=holder.find(p.target_id);if(c.homing&&target>=0&&is_unit(target)){auto&r=entities[target];int flight=div_or_zero(mul32(distance(sub32(p.x,r.x),sub32(p.y,r.y)),50),c.speed),rounded=div32(add32(flight,49),50)*50;r.pendingDamageAmount=add32(r.pendingDamageAmount,r.crownTower?crown_tower_damage(c.crown_percent,c.damage):c.damage);r.pendingDamageDurationMs=pf_min(pf_max(r.pendingDamageDurationMs,rounded),1000);p.pending_registered=true;}}}
 CN_HD void pre_hook(int i){add_tag_fold(i);}
 CN_HD void pending_actions(int i,int){if(is_unit(i)&&unit_config(i).king)kings[entities[i].side].pending_pass();}
 CN_HD void running_actions(int i,int){if(is_unit(i)&&unit_config(i).king){int side=entities[i].side;kings[side].run_pass(health[i].hp<health[i].maximum||princess_count(side)<2);}}
 CN_HD void action_countdown(int){}
 CN_HD CN_WORLD_NOINLINE void pre_pass(int,const int* list,int count){presentCount=0;for(int j=0;j<count;++j){int i=list[j];if(!is_unit(i))continue;present[presentCount++]=i;auto&e=entities[i];e.prevX=e.x;e.prevY=e.y;e.prevZ=e.z;}
  struct View {NativeWorld&w;CN_HD int entityCount(){return w.presentCount;}CN_HD const PfEntity&entity(int i){return w.entities[w.present[i]];}} q{*this};spatial.rebuild(36,64,q);build_overlay(arenaGrid,q);for(int s=0;s<2;++s)arenaGrid.changeFlags[s]=arenaGrid.changed[s];}
 CN_HD void component_refresh(int i,int component){if(component==1){auto&e=entities[i];e.movedSinceCopy=e.x!=e.prevX||e.y!=e.prevY;}}
 CN_HD CN_WORLD_NOINLINE void component_visit(int i,int component);
 CN_HD CN_WORLD_NOINLINE void post_hook(int i){if(is_unit(i)){if(unit_config(i).king)match.sides[entities[i].side].visit(spec->rules,match.timeline);fixed8_post_state(*this,i);}else if(holder.kinds[i]==4){tick_projectile(projectiles[i],spec->projectiles[projectileRows[i]],*this);if(projectiles[i].released)observer.projectile_released(projectiles[i].side,projectiles[i].deploy_index,match.tick);}else tick_combat_area(areas[i],spec->zap,*this);}
 CN_HD void after_post_hooks(){drain_hits();}
 CN_HD void post_pass(int){arenaGrid.swap();gridCurrent=arenaGrid.current==overlay[0]?0:1;spatial.clear();}
 CN_HD void leaving(int i){if(holder.kinds[i]==4){auto&p=projectiles[i];if(p.pending_registered){return_pending(p.target_id,spec->projectiles[projectileRows[i]].damage);p.pending_registered=false;}if(p.chain_index>=0&&--chainUse[p.chain_index]==0)chains[p.chain_index]=ProjectileChain{};}}
 CN_HD void entity_removed(int i,int gone);
 CN_HD void actions_object_left(int,int){}
 CN_HD void actions_instigator_left(int,int){}
 CN_HD void parent_removed(int,int){}
 CN_HD void holder_entity_removed(int){}
 CN_HD void actions_released(int){}
 CN_HD void released_at_fold(int){}
 CN_HD bool king_alive(int side)const{return kingSlots[side]>=0&&health[kingSlots[side]].hp>0;}
 CN_HD int princess_count(int side)const{int n=0;for(int j=0;j<holder.live_count;++j){int i=holder.live[j];if(is_unit(i)&&unit_config(i).princess&&entities[i].side==side)++n;}return n;}
 CN_HD int lowest_tower_hp(int side)const{int n=INT32_MAX;for(int j=0;j<holder.live_count;++j){int i=holder.live[j];if(is_unit(i)&&(unit_config(i).king||unit_config(i).princess)&&entities[i].side==side)n=pf_min(n,health[i].hp);}return n;}
 CN_HD int arena_length()const{return 32000;}
 CN_HD void hold_hits(bool held){hitsHeld=held;}
 CN_HD void end_match(){matchEnded=true;}
 CN_HD void cleanup(){holder.cleanup(*this);}
 CN_HD void tick_entities(int tick){holder.tick(*this,tick);}
 CN_HD void tiebreak_step_end(){drain_hits();cleanup();}
 CN_HD void drain_towers(int side,int amount){for(int j=0;j<holder.live_count;++j){int i=holder.live[j];if(is_unit(i)&&(unit_config(i).king||unit_config(i).princess)&&entities[i].side==side){DamageContext q;auto r=drain_health(health[i],amount,q);if(r.died)death(i);refresh_health(i);}}}
 CN_HD void remove_at_once(int i){leaving(i);int at=-1;for(int j=0;j<holder.live_count;++j)if(holder.live[j]==i){at=j;break;}if(at<0)return;for(int j=at;j+1<holder.live_count;++j)holder.live[j]=holder.live[j+1];--holder.live_count;holder.notices(*this,i,false);actions_released(i);holder_entity_removed(i);holder.free_slot(i);}
 CN_HD int clear_field(){int count=0;for(int j=0;j<holder.live_count;++j){int i=holder.live[j];if(is_unit(i)){if(unit_config(i).king||unit_config(i).princess)continue;fixed8_resume(*this,i);queue_kill(i);++count;}else{remove_at_once(i);++count;}}return count;}
 CN_HD void circle(int side,int radius){const auto&k=entities[kingSlots[side]];for(int j=0;j<holder.live_count;++j){int i=holder.live[j];if(is_unit(i)&&entities[i].side==side&&entities[i].alive&&squared_distance(k.x,k.y,entities[i].x,entities[i].y)<=mul32(radius,radius))queue_kill(i);}}
 CN_HD void death(int i){entities[i].movementActive=false;targets[i].movementComponentActive=false;holder.active[i]&=uint8_t(~2);}
 CN_HD void queue_kill(int i){CombatHit h;h.target_id=holder.ids[i];h.amount=health[i].hp;queue_hit(h);if(hitCount)hits[hitCount-1].kill=true;}
 CN_HD void queue_hit(const CombatHit& hit){if(hitCount==WORLD_HITS){fault(211);return;}hits[hitCount++]={hit,false};}
 CN_HD int next_hit_id(){return ++hitSerial;}
 CN_HD ProjectileChain&chain(int i){if(i<0||i>=WORLD_CHAINS){fault(212);return chains[0];}return chains[i];}
 CN_HD int target_count()const{return presentCount;}
 CN_HD CombatTarget& target_at(int n){return sync_target(present[n]);}
 CN_HD CombatTarget* find_target(int id){int i=holder.find(id);return i>=0&&is_unit(i)?&sync_target(i):nullptr;}
 CN_HD CombatTarget& sync_target(int i){auto&e=entities[i];auto&t=combatTargets[i];t.id=e.id;t.side=e.side;t.kind=5;t.x=e.x;t.y=e.y;t.z=e.zTotal();t.radius=e.collisionRadius;t.hp=health[i].hp;t.present=true;t.has_hp=e.hpPresent;t.building=e.building;t.crown=e.crownTower;t.tower_slot=e.kingCandidate!=0;t.air=e.air;t.untargetable=(e.flags&e.flagBits.untargetable)!=0;t.hidden=hidden(i);t.untouchable=timers[i].spawnImmune;return t;}
 CN_HD void return_pending(int id,int amount){int i=holder.find(id);if(i>=0&&is_unit(i)){if(entities[i].crownTower){/* fixed8 unit-shot tower percent is zero */}entities[i].pendingDamageAmount=sub32(entities[i].pendingDamageAmount,amount);}}
 CN_HD void apply_buff(int id,int buff,int ms,int source,int,int packed){int i=holder.find(id);if(i<0||!is_unit(i))return;if(buff!=spec->stun.id){fault(213);return;}if(entities[i].flags&spec->tags.noBuffs)return;if(!apply_combat_buff(buffs[i],spec->stun,ms,source,packed))fault(214);}
 CN_HD void push(int id,int x,int y,int distance){int i=holder.find(id);if(i<0||!is_unit(i)||!unit_config(i).hasMovement)return;auto&e=entities[i];auto&m=movements[i];if(m.pushbackInFlight||unit_config(i).ignorePushback||!e.alive||hidden(i)||(e.flags&e.flagBits.noPushback))return;set_pushback(m,e,x,y,distance);}
 CN_HD CN_WORLD_NOINLINE void drain_hits(){for(int h=0;h<hitCount;++h){auto&hit=hits[h];int i=holder.find(hit.hit.target_id);if(i<0||!is_unit(i))continue;DamageContext q;q.tick=match.tick;q.held=hitsHeld;q.ended=matchEnded;q.forbidden=(entities[i].flags&spec->tags.noDamage)!=0;q.hidden=hidden(i);q.unkillable=(entities[i].flags&spec->tags.unkillable)!=0;q.untouchable=timers[i].spawnImmune;q.crown=entities[i].crownTower;
   auto r=hit.kill?drain_health(health[i],hit.hit.amount,q):hit.hit.typed_area?apply_typed_area_damage(health[i],hit.hit.amount,hit.hit.dx,hit.hit.dy,q):damage_with_heading(health[i],hit.hit.amount,hit.hit.dedupe_id,hit.hit.dx,hit.hit.dy,q);if(r.overflow)fault(215);if(r.died)death(i);refresh_health(i);if(r.accepted&&hit.hit.buff_id>=0)apply_buff(entities[i].id,hit.hit.buff_id,hit.hit.buff_ms,hit.hit.source_id,hit.hit.side,hit.hit.packed_level);
  }hitCount=0;}
 CN_HD int mask_entities(MaskEntity*out)const{int count=0;for(int stage=0;stage<2;++stage){int n=stage?holder.pending_count:holder.live_count;for(int j=0;j<n;++j){int i=stage?holder.pending[j]:holder.live[j];if(!is_unit(i))continue;const auto&e=entities[i];const auto&u=unit_config(i);out[count++]={e.alive,e.side,e.x,e.y,u.princess,u.king,u.building,u.noDeploySizeW,u.noDeploySizeH,u.tileSizeOverride>0?u.tileSizeOverride:(u.collisionRadius+499)/500+1};}}return count;}
 CN_HD CN_WORLD_NOINLINE void execute_due_commands(int){bool ran[PlayerCommands::CAPACITY];for(int i=0;i<commands.count;++i)ran[i]=commands.queued[i].ran;MaskEntity*mask=placementScratch;int n=mask_entities(mask);commands.execute_due(match,*this,spec->cards,arena_view(),mask,n);for(int i=0;i<commands.count;++i)if(!ran[i]&&commands.queued[i].ran&&commands.queued[i].code)lastRunRefusal[commands.queued[i].side]=commands.queued[i].code<15?commands.queued[i].code:1000;}
 CN_HD CN_WORLD_NOINLINE void deploy_card(int side,int card,int deployIndex,const Placement&place);
 CN_HD void apply_action(int side,int action){lastRefusal[side]=0;if(action==0)return;MaskEntity*mask=placementScratch;int n=mask_entities(mask);bool done=match.is_over(*this)||match.truncated||match.fault;if(!selected_action_legal(match,*this,commands,spec->cards,arena_view(),mask,n,side,action,done)){++illegalNow[side];++illegalTotal[side];lastRefusal[side]=100;return;}int encoded=action-1,slot=encoded/576,tile=encoded%576,x=(tile%18)*1000+500,y=(tile/18)*1000+500;if(side==0)x=18000-x;else y=32000-y;auto given=commands.give(match,*this,spec->cards,side,slot,x,y,arena_view(),mask,n);lastRefusal[side]=int(given.refusal);if(!given.given()){++illegalNow[side];++illegalTotal[side];if(given.refusal==GivingRefusal::UnsupportedPromiseExpiry)fault(216);}}
 CN_HD CN_WORLD_NOINLINE int advance(int action0,int action1,int tickLimit=0){rendered=false;illegalNow[0]=illegalNow[1]=0;lastRunRefusal[0]=lastRunRefusal[1]=0;firstSide=episodeStep&1;apply_action(firstSide,firstSide?action1:action0);apply_action(1-firstSide,firstSide?action0:action1);int advanced=0;for(;advanced<DECISION_TICKS;++advanced){if(!match.step(*this))break;finish_public_tick();}if(tickLimit>0&&match.tick>=tickLimit&&!match.is_over(*this))match.truncated=true;++episodeStep;justReset=false;return advanced;}
 CN_HD void finish_public_tick(){for(int p=0;p<publicPendingCount;++p){auto play=publicPending[p];if(play.kind==obs_schema::PLAY_LAUNCHED){play.projectile_count=0;for(int j=0;j<holder.live_count;++j){int i=holder.live[j];if(holder.kinds[i]==4&&projectiles[i].side==play.side&&projectiles[i].deploy_index==play.deploy_index)++play.projectile_count;}}if(!observer.on_play(play))fault(observer.fault);}publicPendingCount=0;observer.tick(match.lastTicked?spec->rules.fullBarMs[match.timeline.rate]:0,spec->rules.maxMana);}
 CN_HD void legal(int side,uint8_t*out){MaskEntity*mask=placementScratch;int n=mask_entities(mask);legal_actions(match,*this,commands,spec->cards,arena_view(),mask,n,side,match.is_over(*this)||match.truncated||match.fault,out);}
 CN_HD PfNeighbours<WORLD_CAPACITY> spatial_query(const PfSpatialQuery&q){struct View{NativeWorld&w;CN_HD const PfEntity&entity(int i){return w.entities[w.present[i]];}} view{*this};auto found=spatial.query(q,view);for(int i=0;i<found.count;++i)found.indices[i]=present[found.indices[i]];return found;}
 CN_HD PfNeighbourView spatial_query_view(const PfSpatialQuery&q){
  struct View{NativeWorld&w;CN_HD const PfEntity&entity(int i){return w.entities[w.present[i]];}}view{*this};
  spatial.query_into(q,view,spatialResultScratch,spatialKeysScratch);
  for(int i=0;i<spatialResultScratch.count;i++)spatialResultScratch.indices[i]=present[spatialResultScratch.indices[i]];
  return {spatialResultScratch.indices,spatialResultScratch.count};
 }
 CN_HD GridView&grid(){return arenaGrid;}
 CN_HD RouteSearchBuffers<2304>&search_scratch(){return routeScratch;}
 CN_HD Route<>&saved_route_scratch(){return savedRouteScratch;}
 CN_HD int enemy_king(int i)const{return kingSlots[1-entities[i].side];}
 CN_HD PfNeighbours<2> enemy_princess_towers(int i)const{PfNeighbours<2> out;int enemy=1-entities[i].side;for(int j=0;j<holder.live_count;++j){int slot=holder.live[j];if(is_unit(slot)&&unit_config(slot).princess&&entities[slot].side==enemy)out.add(slot);}return out;}
 CN_HD const int*speed_percents(int i)const{return buffs[i].count?&spec->stun.speed_percent:nullptr;}
 CN_HD int speed_percent_count(int i)const{return buffs[i].count?1:0;}
 CN_HD int scale_hit_time(int i,int base){auto lookup=[&](int)->const CombatBuffConfig&{return spec->stun;};return combat_buff_speed(buffs[i],base,true,lookup);}
 CN_HD bool match_ended()const{return matchEnded;}
 CN_HD bool spawn_immune(int i)const{return timers[i].spawnImmune;}
 CN_HD void set_unit_state(int i,int state){fixed8_set_state(*this,i,state);}
 CN_HD void resume_unit(int i){fixed8_resume(*this,i);}
 CN_HD bool apply_attack(int i,int target){return deliver_hit(i,target);}
 CN_HD CN_WORLD_NOINLINE void prepare_route(int i);
 CN_HD CN_WORLD_NOINLINE void clear_reference(int i);
 CN_HD CN_WORLD_NOINLINE bool deliver_hit(int i,int target);
 CN_HD CN_WORLD_NOINLINE int launch_unit_shot(int i,int target);
 CN_HD void extract_observation();
 CN_HD CN_WORLD_NOINLINE void render(int*out,uint8_t*mask,int*info,float*reward){if(!rendered){extract_observation();observe(observer,*this,renderedObs,renderedInfo,renderedRewards);legal(0,renderedMasks);legal(1,renderedMasks+2307);if(faultCode){renderedInfo[obs_schema::I_FAULT]=faultCode;renderedInfo[obs_schema::I_WINNER]=-1;renderedRewards[0]=renderedRewards[1]=0;}rendered=true;}for(int i=0;i<14048;++i)out[i]=renderedObs[i];for(int i=0;i<4614;++i)mask[i]=renderedMasks[i];for(int i=0;i<12;++i)info[i]=renderedInfo[i];for(int i=0;i<2;++i)reward[i]=renderedRewards[i];}
 CN_HD int entity_count()const{return observationCount;}
 CN_HD const ObservationEntity&entity_at(int i)const{return observationEntities[i];}
 CN_HD const ObservationEntity*find_entity(int id)const{for(int i=0;i<observationCount;++i)if(observationEntities[i].id==id)return &observationEntities[i];return nullptr;}
 CN_HD const ObservationOwn&own(int side)const{return observationOwn[side];}
 CN_HD const ObservationClock&clock()const{return observationClock;}
};
}
