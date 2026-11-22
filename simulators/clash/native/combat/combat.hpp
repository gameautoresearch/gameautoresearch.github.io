// SPDX-License-Identifier: Apache-2.0
// Ported from crforge core/src/main/java/org/crforge/core/pathfinding/combat/{DamageApplication,HitPoints,LevelScaling,PackedLevel,AreaDamage}.java,
// core/src/main/java/org/crforge/core/pathfinding/target/DirectHit.java,
// core/src/main/java/org/crforge/core/pathfinding/index/ShapeTests.java,
// core/src/main/java/org/crforge/core/pathfinding/math/{FixedMath,TrigTables}.java,
// core/src/main/java/org/crforge/core/pathfinding/move/{PushbackRequest,MovementVisit}.java,
// core/src/main/java/org/crforge/core/battle/projectile/{ProjectileEntity,ProjectileFlight,ProjectileLauncher,ProjectileChain,ProjectileAmounts}.java,
// core/src/main/java/org/crforge/core/battle/unit/{BattleWorld,AreaEffectEntity,BuffComponent,BuffInstance}.java
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge).
// See ../LICENSE and ../NOTICE. Only the frozen plain fixed8 reachable mechanics are implemented.
#pragma once
#include "../arith.hpp"
#include "../deck.hpp"
#include <cstdint>
namespace clash_native {
CN_HD inline int combat_angle(int x,int y);
CN_HD inline int combat_max(int a,int b){return a>b?a:b;}
CN_HD inline int combat_min(int a,int b){return a<b?a:b;}
// The finite capacities are explicit faults, never silently shortened victim/dedupe lists.
constexpr int COMBAT_MAX_VICTIMS=256, COMBAT_MAX_DEDUPE=64;
struct CombatHealth {
    int hp=0,maximum=0,shield=0,shield_maximum=0,last_hit_heading=0;
    int dedupe_ids[COMBAT_MAX_DEDUPE]={},dedupe_ticks[COMBAT_MAX_DEDUPE]={},dedupe_count=0;
    int decay_step=0,decay_carry=0;
};
struct DamageContext {
    bool forbidden=false,immune=false,hidden=false,untouchable=false,held=false,ended=false,unkillable=false;
    bool target_buffs=true,attacker_buffs=true,crown=false;
    int reduction_percent=0,extra=0,attacker_percent=100,attacker_crown_percent=100,tick=0;
};
struct CombatDamageResult { bool landed=false;int amount=0;bool died=false,accepted=false,overflow=false; };
CN_HD inline int packed_steps(int packed){return int(int8_t(packed));}
CN_HD inline int pack_level(int level,int relative){int hi=sar32(level,8);if(hi==relative)return level;int step=combat_max(add32(hi,packed_steps(level))-relative,0);return (relative<<8)|(step&255);}
// mode: 0 NONE, 1 CARD_DAMAGE, 2 CARD_HP, 3 KING_DAMAGE, 4 KING_HP, 5 TOWER_DAMAGE, 6 TOWER_HP.
struct CombatScaling {int cap=9,per[7]={0,0,0,8,7,8,8},at_cap[7]={0,0,0,10,10,10,10},after_cap[7]={0,0,0,10,10,10,10};};
CN_HD inline int scale_level(int base,int packed,int mode,int relative,const int* rarity_multipliers,int multiplier_count,const CombatScaling& g=CombatScaling{}) {
    int steps=packed_steps(packed);
    if(rarity_multipliers && (mode==1||mode==2)){
        if(steps==0)return base;
        if(steps<1)return base;
        int multiplier;if(steps<=multiplier_count)multiplier=rarity_multipliers[steps-1];else{multiplier=100;for(int k=0;k<steps;++k)multiplier=add32(multiplier,div32(multiplier,10));}
        return div32(mul32(base,multiplier),100);
    }
    int cap=g.cap-(rarity_multipliers?relative:0),multiplier=100;
    for(int k=0;k<steps;++k){int percent=100+(k==cap-1?g.at_cap[mode]:(k<cap?g.per[mode]:g.after_cap[mode]));
        multiplier=multiplier<100000?div32(mul32(percent,multiplier),100):mul32(percent,int(uint32_t(multiplier)/100));}
    return div32(mul32(base,multiplier),100);
}
CN_HD inline int crown_tower_damage(int percent,int amount){return div_or_zero(add32(mul32(combat_max(percent,-100)+100,amount),99),100);}
// Damage reduction is additive percentage (BattleWorld.damageReduction clamps to +/-100).
CN_HD inline int entry_reduced(int amount,const DamageContext& q){return amount<1||!q.target_buffs?amount:combat_max(div32(mul32(100-q.reduction_percent,amount),100),1);}
CN_HD inline int damage_percent(int amount,int percent){return combat_max(div_or_zero(mul32(amount,percent),100),1);}
CN_HD inline CombatDamageResult subtract_damage(CombatHealth& h,int amount,int dx,int dy,const DamageContext& q,bool ignore_holds=false,bool pierces=false) {
    if(!ignore_holds&&q.ended)return {false,0,false,true,false};
    if(h.hp<1)return {true,0,false,true,false};
    int applied=amount;
    if(h.shield>=1){applied=combat_min(h.shield,amount);h.shield=combat_max(sub32(h.shield,amount),0);amount=0;}
    h.hp=sub32(h.hp,amount);if(h.hp<0)applied=add32(applied,h.hp);
    if(!pierces&&q.unkillable&&h.hp<1)h.hp=1;
    bool died=h.hp<1;if(died){h.last_hit_heading=combat_angle(dx,dy);h.hp=combat_max(h.hp,0);}
    // The heading is set by damage_with_heading below to share exact integer trig.
    (void)dx;(void)dy;return {true,applied,died,true,false};
}
CN_HD inline CombatDamageResult apply_damage(CombatHealth& h,int amount,int dedupe,int dx,int dy,const DamageContext& q) {
    if(q.forbidden||q.immune||q.hidden||q.untouchable||amount<1)return {};
    if(q.target_buffs){amount=entry_reduced(amount,q);if(q.extra>0)amount=add32(amount,q.extra);
        if(q.attacker_buffs){amount=damage_percent(amount,q.attacker_percent);if(q.crown)amount=damage_percent(amount,q.attacker_crown_percent);}}
    if(q.held||q.untouchable)return {};
    if(dedupe){for(int k=0;k<h.dedupe_count;++k)if(h.dedupe_ids[k]==dedupe){h.dedupe_ticks[k]=q.tick;return {};}
        if(h.dedupe_count==COMBAT_MAX_DEDUPE)return {false,0,false,false,true};
        h.dedupe_ids[h.dedupe_count]=dedupe;h.dedupe_ticks[h.dedupe_count++]=q.tick;}
    return subtract_damage(h,amount,dx,dy,q);
}
// Typed filter-area pipeline: protection at the type stage, then again at typedHit's entry.
CN_HD inline CombatDamageResult apply_typed_area_damage(CombatHealth& h,int amount,int dx,int dy,const DamageContext& q){
    if(q.forbidden)return {};amount=combat_max(div32(mul32(100-q.reduction_percent,amount),100),0);if(amount==0)return {};
    if(q.held||q.hidden||q.untouchable)return {};amount=entry_reduced(amount,q);return subtract_damage(h,amount,dx,dy,q);
}
CN_HD inline bool tick_decay(CombatHealth& h){h.decay_carry=add32(h.decay_carry,h.decay_step);if(h.decay_carry<100)return false;int whole=div32(h.decay_carry,100);h.decay_carry=sub32(h.decay_carry,mul32(whole,100));h.hp=sub32(h.hp,whole);if(h.hp>0)return false;h.hp=h.decay_carry=0;return true;}
struct CombatTarget {
    int id=0,side=0,kind=0,x=0,y=0,z=0,radius=0,hp=0;
    bool present=true,has_hp=true,building=false,crown=false,tower_slot=false,air=false;
    bool untargetable=false,hidden=false,untouchable=false,area_passed=false;
};
CN_HD inline bool within_combat_circle(const CombatTarget& t,int x,int y,int radius){
    int dx=sub32(x,t.x),dy=sub32(y,t.y);
    if(!t.building){int reach=add32(radius,t.radius);return squared(dx,dy)<mul32(reach,reach);}
    int lowx=sub32(t.x,t.radius),lowy=sub32(t.y,t.radius);
    int cx=combat_max(combat_min(x,add32(lowx,mul32(2,t.radius))),lowx),cy=combat_max(combat_min(y,add32(lowy,mul32(2,t.radius))),lowy);
    dx=sub32(cx,x);dy=sub32(cy,y);if(dx>radius||dx< -radius||dy>radius||dy< -radius)return false;
    return uint32_t(add32(mul32(dx,dx),mul32(dy,dy)))<uint32_t(mul32(radius,radius));
}
struct CombatHit {
    int source_id=0,target_id=0,side=0,amount=0,hit_id=0,dedupe_id=0,dx=0,dy=0;
    int buff_id=-1,buff_ms=0,packed_level=0;bool projectile=false,typed_area=false;
};
struct ProjectileConfig {
    int id=0,speed=0,gravity=0,radius=0,constant_height=0,min_distance=0,projectile_range=0;
    int damage=0,crown_percent=0,packed_level=0,pushback=0;
    int target_buff=-1,buff_ms=0,random_delay_ms=0;
    bool homing=false,only_enemies=true,hits_air=true,hits_ground=true,consider_z=false,buff_before_damage=false;
};
struct ProjectileState {
    int id=0,source_id=0,target_id=0,side=0,config_index=0;
    int x=0,y=0,z=0,start_x=0,start_y=0,start_z=0,aim_x=0,aim_y=0,aim_z=0;
    int delay_ms=0,group_id=0,chain_index=-1,ring_x=0,ring_y=0,cast_id=-1,deploy_index=-1;
    bool released=false,pending_registered=false,first_visit=true;
};
struct ProjectileChain {int x=0,y=0,radius=0,ids[COMBAT_MAX_VICTIMS]={},count=0;};
// World: target_count(), target_at(i) (CombatTarget& in holder ID order), find_target(id),
// queue_hit(CombatHit) (NO immediate drain), push(id,from_x,from_y,distance),
// apply_buff(id,buff_id,ms,source_id,side,packed_level), next_hit_id(), chain(i),
// return_pending(target_id,amount), fault(code). fault must terminate/refuse the transition.
struct CombatArea {
    int x=0,y=0,radius=0,damage=0,tower_damage=0,hit_id=0,limit=0,pushback=0;
    bool own_side=false,hits_air=true,hits_ground=true,split=false,affects_hidden=false,ignore_buildings=false,tower_limit=true;
};
CN_HD inline bool combat_area_accepts(const CombatTarget& t,const CombatArea& a,int owner_id,int side){
    return t.present&&t.id!=owner_id&&t.has_hp&&!t.untargetable&&(a.own_side||t.side!=side)&&
      (!t.air||a.hits_air)&&(t.air||a.hits_ground)&&(!a.ignore_buildings||!t.building)&&
      (!t.hidden||a.affects_hidden)&&!t.area_passed&&!t.untouchable&&within_combat_circle(t,a.x,a.y,a.radius);
}
template<class World> CN_HD int collect_area(World& w,const CombatArea& a,int owner_id,int side,ProjectileChain* chain,int out[COMBAT_MAX_VICTIMS]){
    int count=0;bool tower=false;
    for(int k=0;k<w.target_count();++k){const CombatTarget& t=w.target_at(k);if(!combat_area_accepts(t,a,owner_id,side))continue;
        if(chain){if(chain->radius>=1&&!within_combat_circle(t,chain->x,chain->y,chain->radius))continue;
            bool seen=false;for(int j=0;j<chain->count;++j)if(chain->ids[j]==t.id){seen=true;break;}if(seen)continue;
            if(chain->count==COMBAT_MAX_VICTIMS){w.fault(101);return -1;}chain->ids[chain->count++]=t.id;}
        if(a.tower_limit&&t.tower_slot&&tower)continue;if(count==COMBAT_MAX_VICTIMS){w.fault(102);return -1;}
        out[count++]=t.id;tower|=t.tower_slot;if(a.limit>=1&&count>=a.limit)break;
    }return count;
}
template<class World> CN_HD void queue_area(World& w,const CombatArea& a,const CombatHit& base,ProjectileChain* chain=nullptr){
    int victims[COMBAT_MAX_VICTIMS];int n=collect_area(w,a,base.source_id,base.side,chain,victims);if(n<0)return;
    int damage=a.damage,tower=a.tower_damage;if(a.split&&n){damage=div32(add32(damage,n-1),n);tower=div32(add32(tower,n-1),n);}
    for(int k=0;k<n;++k){CombatTarget* t=w.find_target(victims[k]);if(!t||!t->has_hp)continue;
        CombatHit h=base;h.target_id=t->id;h.amount=t->crown?tower:damage;h.hit_id=a.hit_id;h.dx=h.dy=0;
        if(h.amount>=1)w.queue_hit(h);if(a.pushback>=1)w.push(t->id,a.x,a.y,a.pushback);
    }
}
// DirectHit.apply for the frozen deck's single-target melee attacks. The target is the stored
// reference if still locked; a missed reference deals nothing. Hit delivery remains deferred.
template<class World> CN_HD bool queue_direct_hit(World& w,int source_id,int side,int target_id,int damage,int crown_percent,int source_x,int source_y,int reference_x,int reference_y,bool missed=false){
    CombatTarget* target=w.find_target(target_id);if(!target||missed)return false;
    CombatHit h;h.source_id=source_id;h.side=side;h.target_id=target_id;h.amount=target->crown?crown_tower_damage(crown_percent,damage):damage;h.hit_id=w.next_hit_id();h.dx=sub32(reference_x,source_x);h.dy=sub32(reference_y,source_y);w.queue_hit(h);return true;
}
CN_HD inline void launch_projectile(ProjectileState& p,const ProjectileConfig& c,int source_id,int target_id,int side,int sx,int sy,int sz,int hx,int hy){
    p.source_id=source_id;p.target_id=target_id;p.side=side;p.x=p.start_x=sx;p.y=p.start_y=sy;p.z=p.start_z=sz;
    p.aim_x=hx;p.aim_y=hy;p.aim_z=0;p.delay_ms=0;p.released=false;p.first_visit=true;p.pending_registered=false;
    if(c.min_distance){int dx=hx-sx,dy=hy-sy;if(distance(dx,dy)<c.min_distance){normalize(dx,dy,c.min_distance);p.aim_x=add32(sx,dx);p.aim_y=add32(sy,dy);}}
    if(c.constant_height){p.z=p.start_z=p.aim_z=c.constant_height;}
}
CN_HD inline void launch_unit_projectile(ProjectileState& p,const ProjectileConfig& c,int source_id,int target_id,int side,int x,int y,int z,int height_offset,int start_radius,int start_z,int y_offset,int hx,int hy){
    int dx=sub32(hx,x),dy=sub32(hy,y);normalize(dx,dy,start_radius);
    launch_projectile(p,c,source_id,target_id,side,add32(x,dx),add32(add32(y,dy),side? -y_offset:y_offset),add32(add32(start_z,z),height_offset),hx,hy);
}
CN_HD inline int projectile_arc_height(const ProjectileState& p,const ProjectileConfig& c,int x,int y){
    int speed=combat_max(c.speed,1),total=combat_max(div_or_zero(distance(sub32(p.aim_x,p.start_x),sub32(p.aim_y,p.start_y)),speed),1);
    int t=div_or_zero(distance(sub32(x,p.start_x),sub32(y,p.start_y)),speed),climb=sub32(p.aim_z,p.start_z),gravity_half=div_or_zero(mul32(c.gravity,-500),1000);
    return add32(add32(mul32(mul32(sub32(t,total),t),gravity_half),div_or_zero(mul32(climb,t),total)),p.start_z);
}
template<class World> CN_HD void projectile_impact(ProjectileState& p,const ProjectileConfig& c,World& w){
    int hit_id=w.next_hit_id();CombatHit base;base.source_id=p.id;base.side=p.side;base.hit_id=hit_id;base.dedupe_id=p.group_id;base.projectile=true;
    base.buff_id=c.buff_before_damage?-1:c.target_buff;base.buff_ms=c.buff_ms;base.packed_level=c.packed_level;
    int damage=c.damage,tower=crown_tower_damage(c.crown_percent,damage);
    if(c.radius>=1){CombatArea a;a.x=p.chain_index>=0?p.ring_x:p.aim_x;a.y=p.chain_index>=0?p.ring_y:p.aim_y;a.radius=c.radius;a.damage=damage;a.tower_damage=tower;a.hit_id=hit_id;a.own_side=!c.only_enemies;a.hits_air=c.hits_air;a.hits_ground=c.hits_ground;a.pushback=c.pushback;
        ProjectileChain* chain=p.chain_index<0?nullptr:&w.chain(p.chain_index);
        if(c.buff_before_damage&&c.target_buff>=0){int ids[COMBAT_MAX_VICTIMS];int n=collect_area(w,a,base.source_id,p.side,nullptr,ids);for(int i=0;i<n;++i)w.apply_buff(ids[i],c.target_buff,c.buff_ms,p.id,p.side,c.packed_level);}
        if(damage>0)queue_area(w,a,base,chain);
    }else{CombatTarget* target=w.find_target(p.target_id);if(!target)return;
        if(c.buff_before_damage&&c.target_buff>=0)w.apply_buff(target->id,c.target_buff,c.buff_ms,p.id,p.side,c.packed_level);
        if(damage<1||!target->has_hp)return;base.target_id=target->id;base.amount=target->crown?tower:damage;base.dx=sub32(p.aim_x,p.start_x);base.dy=sub32(p.aim_y,p.start_y);w.queue_hit(base);
    }
}
template<class World> CN_HD void tick_projectile(ProjectileState& p,const ProjectileConfig& c,World& w){
    if(p.released)return;p.first_visit=false;
    if(p.delay_ms>=1){p.delay_ms=combat_max(p.delay_ms,50)-50;return;}
    CombatTarget* target=w.find_target(p.target_id);
    if(target&&c.homing){p.aim_x=target->x;p.aim_y=target->y;p.aim_z=target->z;}
    int remaining=distance(sub32(p.x,p.aim_x),sub32(p.y,p.aim_y));if(c.consider_z)remaining=add32(remaining,abs32(sub32(p.aim_z,p.z)));
    if(remaining<=c.speed){if(c.homing&&p.pending_registered){w.return_pending(p.target_id,c.damage);p.pending_registered=false;}
        p.released=true;p.x=p.aim_x;p.y=p.aim_y;p.z=combat_max(c.constant_height,0);projectile_impact(p,c,w);return;}
    int nx=add32(p.x,div_or_zero(mul32(sub32(p.aim_x,p.x),c.speed),remaining));int ny=add32(p.y,div_or_zero(mul32(sub32(p.aim_y,p.y),c.speed),remaining));
    p.z=c.consider_z?add32(p.z,div_or_zero(mul32(sub32(p.aim_z,p.z),c.speed),remaining)):projectile_arc_height(p,c,nx,ny);p.x=nx;p.y=ny;
}
struct SpellProjectileConfig {int count=1,waves=1,interval_ms=0,wave_interval_ms=0,radius=0;bool ring=false;};
// Trig is the same integer table as FixedMath/TrigTables; no floating point in native ticks.
CN_HD inline int combat_sine_quarter(int index){const int table[]={
    0, 18, 36, 54, 71, 89, 107, 125, 143, 160, 178, 195, 213, 230, 248, 265, 282, 299, 316, 333,
    350, 367, 384, 400, 416, 433, 449, 465, 481, 496, 512, 527, 543, 558, 573, 587, 602, 616, 630,
    644, 658, 672, 685, 698, 711, 724, 737, 749, 761, 773, 784, 796, 807, 818, 828, 839, 849, 859,
    868, 878, 887, 896, 904, 912, 920, 928, 935, 943, 949, 956, 962, 968, 974, 979, 984, 989, 994,
    998, 1002, 1005, 1008, 1011, 1014, 1016, 1018, 1020, 1022, 1023, 1023, 1024, 1024
  };return table[index];}
CN_HD inline int combat_atan_ratio(int index){const int table[]={
    0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 11, 12, 12,
    13, 13, 14, 14, 14, 15, 15, 16, 16, 17, 17, 17, 18, 18, 19, 19, 19, 20, 20, 21, 21, 21, 22, 22,
    22, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31,
    31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 37, 37, 37, 37, 38, 38,
    38, 39, 39, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 43, 44, 44, 44,
    44, 45, 45, 45
  };return table[index];}
CN_HD inline int combat_sine(int degrees){int angle=floor_mod(degrees,360),half=angle>=180?angle-180:angle;int value=combat_sine_quarter(combat_min(half,180-half));return angle>=180?-value:value;}
CN_HD inline void combat_rotate(int& x,int& y,int degrees){int c=combat_sine(degrees+90),s=combat_sine(degrees);int nx=sub32(mul32(c,x),mul32(s,y)),ny=add32(mul32(s,x),mul32(c,y));x=sar32(add32(nx,nx<0?1023:0),10);y=sar32(add32(ny,ny<0?1023:0),10);}
CN_HD inline int combat_angle(int x,int y){if(x==0&&y==0)return 0;int ax=abs32(x),ay=abs32(y);int a=ax>=ay?combat_atan_ratio(div_or_zero(mul32(ay,128),ax)):90-combat_atan_ratio(div_or_zero(mul32(ax,128),ay));if(x<0)a=180-a;if(y<0)a=360-a;return floor_mod(a,360);}
CN_HD inline CombatDamageResult damage_with_heading(CombatHealth& h,int amount,int dedupe,int dx,int dy,const DamageContext& q){auto r=apply_damage(h,amount,dedupe,dx,dy,q);if(r.died)h.last_hit_heading=combat_angle(dx,dy);return r;}
CN_HD inline CombatDamageResult drain_health(CombatHealth& h,int amount,const DamageContext& q){if(q.untouchable)return {};auto r=subtract_damage(h,amount,0,0,q,true);if(r.died)h.last_hit_heading=0;return r;}
// Caller allocates count*waves states and waves chains and assigns entity ids in creation order.
// Returns number written, or -1 on insufficient capacity. One RNG draw per arrow, including centre.
template<class Random> CN_HD int cast_spell_projectiles(ProjectileState* output,int capacity,ProjectileChain* chains,int chain_capacity,int first_chain,const SpellProjectileConfig& spell,const ProjectileConfig& c,int king_id,int side,int king_x,int king_y,int king_radius,int x,int y,int cast_id,int play,Random& random){
    int count=combat_max(spell.count,1),waves=combat_max(spell.waves,1);if(capacity<count*waves||(spell.ring&&chain_capacity<waves))return -1;
    int radius=spell.radius,step=count>1?div32(360,count-1):0,r90=div32(mul32(radius,90),100),r90sq=mul32(r90,r90),written=0;
    for(int wave=0;wave<waves;++wave){if(spell.ring){chains[wave]=ProjectileChain{};chains[wave].x=x;chains[wave].y=y;chains[wave].radius=radius;}
        for(int k=0;k<count;++k){int ox=0,oy=0,rx=x,ry=y;if(k&&spell.ring){ox=sub32(radius,div32(mul32(c.radius,60),100));combat_rotate(ox,oy,mul32(step,k-1));rx=add32(x,ox);ry=add32(y,oy);}
            int tx=x,ty=y;if(spell.ring){ox=0;oy=div32(mul32(c.radius,60),100);combat_rotate(ox,oy,random.next(359));ox=add32(ox,sub32(rx,x));oy=add32(oy,sub32(ry,y));if(add32(mul32(ox,ox),mul32(oy,oy))>r90sq)normalize(ox,oy,r90);tx=add32(x,ox);ty=add32(y,oy);}
            ProjectileState& p=output[written++];launch_projectile(p,c,king_id,0,side,add32(king_x,sar32(ox,2)),add32(king_y,oy),mul32(3,king_radius),tx,ty);
            p.delay_ms=add32(mul32(wave,spell.wave_interval_ms),mul32(k,spell.interval_ms));p.chain_index=spell.ring?first_chain+wave:-1;p.ring_x=rx;p.ring_y=ry;p.cast_id=cast_id;p.deploy_index=play;
        }
    }return written;
}
// A single row suffices for the frozen task's Zap stun. Multiple slots retain normal source/row
// refresh semantics and speed/damage extrema, without heap allocation or virtual dispatch.
struct CombatBuffConfig {int id=-1,speed_percent=0,hit_speed_percent=0,spawn_speed_percent=0,damage_percent=0,crown_percent=0;bool specific_source=false,individual=false;};
struct CombatBuff {int id=-1,source=0,remaining=0,total=0,packed_level=0;};
struct CombatBuffs {CombatBuff items[8]={};int count=0;};
CN_HD inline bool apply_combat_buff(CombatBuffs& b,const CombatBuffConfig& cfg,int ms,int source,int packed_level){
    if(!cfg.individual){bool created=true;for(int i=b.count-1;i>=0;--i){CombatBuff& old=b.items[i];if(old.id==cfg.id&&(!cfg.specific_source||old.source==source)){if(ms>old.remaining&&old.remaining!=-1){old.total=add32(old.total,ms-old.remaining);old.remaining=ms;}if(packed_steps(old.packed_level)<packed_steps(packed_level))old.packed_level=packed_level;created=false;}}if(!created)return true;}
    if(b.count==8)return false;b.items[b.count++]={cfg.id,source,ms,ms,packed_level};return true;
}
CN_HD inline void tick_combat_buffs(CombatBuffs& b){for(int i=b.count-1;i>=0;--i){CombatBuff& v=b.items[i];if(v.remaining!=-1)v.remaining=combat_max(v.remaining,50)-50;if(v.remaining==0){for(int j=i;j+1<b.count;++j)b.items[j]=b.items[j+1];--b.count;}}}
template<class Lookup> CN_HD int combat_buff_speed(const CombatBuffs& b,int base,bool attack,Lookup& lookup){int boost=100,slow=0;for(int i=0;i<b.count;++i){const CombatBuffConfig& c=lookup(b.items[i].id);int v=attack?c.hit_speed_percent:c.speed_percent;if(v>=1)boost=combat_max(boost,v);else if(v<0)slow=combat_max(slow,-v);}return div_or_zero(mul32(div_or_zero(mul32(boost,base),100),combat_max(100-slow,0)),100);}
struct CombatPushback {int target_x=0,target_y=0,budget=0;bool in_flight=false,attack=false;};
CN_HD inline bool request_combat_pushback(CombatPushback& m,const CombatTarget& target,int from_x,int from_y,int length,bool ignores=false,bool keep_longer=false,bool subtract=false,bool attack=false){
    if((m.in_flight&&!keep_longer)||ignores||target.hidden||target.hp<1)return false;
    int dx=sub32(target.x,from_x),dy=sub32(target.y,from_y),d=distance(dx,dy);if(d==0){d=1;dx=(target.id&1)?-1:1;}
    length=subtract?combat_max(length-d,0):length;length=combat_min(length,40000);if(m.in_flight&&keep_longer&&length<m.budget)return true;
    m.attack=attack;m.target_x=target.x;m.target_y=target.y;
    if(length>=1){int step=0,total=0;do{step+=25;total+=step;}while(total<length);m.budget=step;m.target_x=add32(m.target_x,div32(mul32(length,dx),d));m.target_y=add32(m.target_y,div32(mul32(length,dy),d));m.in_flight=true;}else{m.in_flight=false;m.attack=false;}return true;
}
// This returns the exact per-visit displacement budget. The owning movement solver applies
// collision pushes, relocation and terrain-grid displace at the target before ending the route.
CN_HD inline int tick_combat_pushback(CombatPushback& m){m.budget=sub32(m.budget,25);m.in_flight=m.budget>=0;return m.budget;}
// Plain Zap is an AreaEffectEntity, not a projectile. Its buff is applied immediately in the
// area's post-hook after queueing damage; projectile buffs instead ride the deferred damage drain.
struct CombatAreaEffectConfig {int radius=0,max_radius=0,life_ms=0,hit_speed_ms=0,offset_ms=0,damage=0,crown_percent=0,pushback=0,limit=0,buff_id=-1,buff_ms=0,packed_level=0;int typed_tower_damage=0;bool filter_hits=false;bool only_enemies=true,hits_air=true,hits_ground=true,ignore_buildings=false,shared=false,cap_buff_time=false,only_own_troops=false;};
struct CombatAreaEffect {int id=0,side=0,x=0,y=0,countdown=0;};
CN_HD inline int combat_area_radius(const CombatAreaEffect& p,const CombatAreaEffectConfig& c){if(!c.max_radius||!c.life_ms)return c.radius;int r=sub32(c.max_radius,div32(mul32(p.countdown,sub32(c.max_radius,c.radius)),c.life_ms));return r>c.radius?combat_min(r,c.max_radius):c.radius;}
template<class World> CN_HD void tick_combat_area(CombatAreaEffect& p,const CombatAreaEffectConfig& c,World& w){
    if(p.countdown<0)return;int start=sub32(c.life_ms,p.countdown);p.countdown=sub32(p.countdown,50);int end=sub32(c.life_ms,p.countdown),hits=0;
    if(c.hit_speed_ms>0)hits=sub32(div32(add32(c.offset_ms,end),c.hit_speed_ms),div32(add32(c.offset_ms,start),c.hit_speed_ms));else if(c.hit_speed_ms==0)hits=start>0?0:1;
    if(c.filter_hits){
        int step=div32(start,50),offset=div32(c.offset_ms,50),every=div32(c.hit_speed_ms,50);bool due=step==offset||(every>=1&&step-offset>0&&(step-offset)%every==0);if(!due)return;
        CombatArea area;area.tower_limit=false;area.x=p.x;area.y=p.y;area.radius=combat_area_radius(p,c);area.own_side=!c.only_enemies;area.hits_air=c.hits_air;area.hits_ground=c.hits_ground;area.ignore_buildings=c.ignore_buildings;
        int ids[COMBAT_MAX_VICTIMS],n=collect_area(w,area,p.id,p.side,nullptr,ids);if(n<0)return;
        // Stable nearest-first sort is observable in the queued hit order.
        for(int i=1;i<n;++i){int id=ids[i],j=i;CombatTarget* t=w.find_target(id);int d=squared(sub32(t->x,p.x),sub32(t->y,p.y));while(j>0){CombatTarget* before=w.find_target(ids[j-1]);int b=squared(sub32(before->x,p.x),sub32(before->y,p.y));if(b<=d)break;ids[j]=ids[j-1];--j;}ids[j]=id;}
        if(c.limit>0)n=combat_min(n,c.limit);
        for(int i=0;i<n;++i){CombatTarget* t=w.find_target(ids[i]);if(!t)continue;if(c.pushback>0)w.push(t->id,p.x,p.y,c.pushback);
            CombatHit h;h.source_id=p.id;h.target_id=t->id;h.side=p.side;h.amount=t->crown?c.typed_tower_damage:c.damage;h.dx=sub32(t->x,p.x);h.dy=sub32(t->y,p.y);h.packed_level=c.packed_level;h.typed_area=true;if(h.amount>0)w.queue_hit(h);
            int ms=c.cap_buff_time?combat_min(c.buff_ms,p.countdown+c.hit_speed_ms):c.buff_ms;if(c.buff_id>=0&&ms>=1)w.apply_buff(t->id,c.buff_id,ms,p.id,p.side,c.packed_level);
        }return;
    }
    CombatArea a;a.x=p.x;a.y=p.y;a.radius=combat_area_radius(p,c);a.damage=c.damage;a.tower_damage=crown_tower_damage(c.crown_percent,c.damage);a.pushback=c.pushback;a.limit=c.limit;a.own_side=!c.only_enemies;a.hits_air=c.hits_air;a.hits_ground=c.hits_ground;a.ignore_buildings=c.ignore_buildings;a.split=c.shared;
    CombatHit base;base.source_id=p.id;base.side=p.side;base.packed_level=c.packed_level;
    for(int i=0;i<hits;++i){if(c.damage>=1)queue_area(w,a,base);int ms=c.cap_buff_time?combat_min(c.buff_ms,p.countdown+c.hit_speed_ms):c.buff_ms;
        if(c.buff_id>=0&&ms>=1){for(int j=0;j<w.target_count();++j){CombatTarget& t=w.target_at(j);if(!t.present||!t.has_hp||t.hidden||t.untouchable||t.area_passed||t.untargetable||(!c.hits_air&&t.air)||(!c.hits_ground&&!t.air)||(!c.hits_ground&&t.building)||(c.ignore_buildings&&t.building))continue;
            if(c.only_own_troops?t.side!=p.side:(c.only_enemies&&t.side==p.side))continue;if(!within_combat_circle(t,p.x,p.y,a.radius))continue;w.apply_buff(t.id,c.buff_id,ms,p.id,p.side,c.packed_level);}}
    }
}
} // namespace clash_native
