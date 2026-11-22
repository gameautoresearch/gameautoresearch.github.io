// SPDX-License-Identifier: Apache-2.0
// Source scenario tests for combat.hpp, port attribution is in combat.hpp.
#include "combat.hpp"
#include <cassert>
#include <cstdio>
using namespace clash_native;
struct TestWorld {
 CombatTarget targets[10];int nt=0;CombatHit hits[1000];int nh=0;int buff_targets[32],buff_count=0,push_count=0,next_id=0,pending=0;
 ProjectileChain chains[3];
 int target_count(){return nt;}CombatTarget& target_at(int i){return targets[i];}
 CombatTarget* find_target(int id){for(int i=0;i<nt;++i)if(targets[i].id==id)return &targets[i];return nullptr;}
 void queue_hit(CombatHit h){assert(nh<1000);hits[nh++]=h;}
 void push(int id,int,int,int){assert(find_target(id));++push_count;}
 void apply_buff(int id,int,int,int,int,int){assert(buff_count<32);buff_targets[buff_count++]=id;}
 int next_hit_id(){return ++next_id;}ProjectileChain& chain(int i){assert(i>=0&&i<3);return chains[i];}
 void return_pending(int,int n){pending-=n;}void fault(int){assert(false);}
};
int main(){
 // Circle strictness and building-square overlap, both sides of each boundary.
 CombatTarget target;target.radius=500;target.x=3000;target.y=1000;
 assert(!within_combat_circle(target,0,1000,2500));target.x=2999;assert(within_combat_circle(target,0,1000,2500));
 target.building=true;target.x=3000;assert(!within_combat_circle(target,0,1000,2500));target.x=2999;assert(within_combat_circle(target,0,1000,2500));
 target.x=2500;target.y=2500;assert(!within_combat_circle(target,0,0,2500));
 TestWorld w;w.nt=5;
 for(int i=0;i<5;++i){w.targets[i].id=i+1;w.targets[i].side=i==4?0:1;w.targets[i].hp=1000;w.targets[i].radius=500;w.targets[i].x=9000+i*20;w.targets[i].y=16000;}
 w.targets[2].crown=w.targets[3].crown=true;w.targets[2].tower_slot=w.targets[3].tower_slot=true;w.targets[2].building=w.targets[3].building=true;
 ProjectileConfig fire;fire.damage=688;fire.crown_percent=-77;fire.radius=2500;fire.pushback=1000;
 ProjectileState p;p.id=999;p.aim_x=9000;p.aim_y=16000;projectile_impact(p,fire,w);
 assert(w.nh==3&&w.push_count==3);assert(w.hits[2].amount==159);assert(w.hits[0].dx==0&&w.hits[0].dy==0);
 assert(w.hits[0].projectile&&w.hits[0].buff_id==-1);
 // Zap filter ignores the one-tower-slot area rule and emits nearest-first typed hits.
 w.nh=0;w.push_count=0;w.targets[0].x=10200;w.targets[1].x=9001;
 CombatAreaEffectConfig zap;zap.filter_hits=true;zap.life_ms=1;zap.radius=2500;zap.damage=192;zap.typed_tower_damage=48;zap.buff_id=4;zap.buff_ms=500;
 CombatAreaEffect a;a.id=998;a.side=0;a.x=9000;a.y=16000;a.countdown=1;tick_combat_area(a,zap,w);
 assert(a.countdown==-49&&w.nh==4&&w.buff_count==4);assert(w.hits[0].target_id==2&&w.hits[3].target_id==1);assert(w.hits[1].typed_area&&w.hits[1].amount==48);tick_combat_area(a,zap,w);assert(w.nh==4);
 // Arrows: each of 30 arrows is independently delayed, moves and releases; per-wave chain
 // membership restricts each enemy to exactly one hit in each of three waves.
 TestWorld arrows;arrows.nt=1;arrows.targets[0].id=12;arrows.targets[0].side=1;arrows.targets[0].hp=1000;arrows.targets[0].x=9000;arrows.targets[0].y=16000;arrows.targets[0].radius=500;
 ProjectileState projectiles[30];ProjectileConfig cfg;cfg.speed=1100;cfg.gravity=50;cfg.radius=1400;cfg.damage=122;cfg.crown_percent=-80;cfg.homing=true;
 SpellProjectileConfig spell;spell.count=10;spell.waves=3;spell.interval_ms=50;spell.wave_interval_ms=200;spell.radius=4000;spell.ring=true;BattleRandom random{217};
 assert(cast_spell_projectiles(projectiles,30,arrows.chains,3,0,spell,cfg,1,0,9000,1000,750,9000,16000,1,0,random)==30);
 for(int tick=0;tick<80;++tick)for(int i=0;i<30;++i){projectiles[i].id=1000+i;tick_projectile(projectiles[i],cfg,arrows);}
 assert(arrows.nh==3);for(int i=0;i<30;++i)assert(projectiles[i].released);for(int i=0;i<3;++i)assert(arrows.chains[i].count==1);
 CombatHealth h;h.hp=100;h.maximum=100;DamageContext q;q.reduction_percent=33;auto r=apply_typed_area_damage(h,10,7,-9,q);assert(h.hp==96&&r.amount==4);
 // Refresh keeps the longer remaining duration. Freeze immediately holds attack/move budgets.
 CombatBuffs buffs;CombatBuffConfig freeze;freeze.id=4;freeze.speed_percent=-100;freeze.hit_speed_percent=-100;assert(apply_combat_buff(buffs,freeze,500,998,10));
 auto lookup=[&](int)->const CombatBuffConfig&{return freeze;};assert(combat_buff_speed(buffs,50,true,lookup)==0);assert(combat_buff_speed(buffs,100,false,lookup)==0);
 for(int i=0;i<3;++i)tick_combat_buffs(buffs);assert(buffs.items[0].remaining==350);assert(apply_combat_buff(buffs,freeze,500,998,10));assert(buffs.count==1&&buffs.items[0].remaining==500&&buffs.items[0].total==650);
 for(int i=0;i<10;++i)tick_combat_buffs(buffs);assert(buffs.count==0&&combat_buff_speed(buffs,50,true,lookup)==50);
 CombatPushback push;CombatTarget mover;mover.id=1;mover.x=9000;mover.y=16000;mover.hp=1;assert(request_combat_pushback(push,mover,9000,16000,1000));assert(push.target_x==8000&&push.target_y==16000&&push.budget==225);assert(!request_combat_pushback(push,mover,0,0,1000));int total=0;while(push.in_flight){int b=tick_combat_pushback(push);if(b>0)total+=b;}assert(total==900);
 std::puts("combat source scenarios passed: Fireball, 30-arrow three-wave chains, nearest-first Zap, typed protection, buff refresh, pushback, strict shapes");
}
