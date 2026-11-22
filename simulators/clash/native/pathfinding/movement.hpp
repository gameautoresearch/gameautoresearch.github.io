// SPDX-License-Identifier: Apache-2.0
// Ported from crforge core/src/main/java/org/crforge/core/pathfinding/move/{SpeedBudget,
// MovementGates,ContactRule,AvoidanceHandler,PushPass,AlignedStaticCheck,GridMove,
// DirectionInitializer,Displacement,WaypointSelector,RouteRetention,RouteBeyondReference,
// RoutePreparation,RouteFollower,MovementVisit,PushbackRequest}.java
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge)
#pragma once
#include "grid.hpp"
#include "spatial.hpp"
#include "trig.hpp"
namespace clash_native {
CN_HD inline int speed_modifier(const int*percents,int count,int base){int boost=100,slow=0;
 for(int i=0;i<count;i++){int p=percents[i];if(p>=1)boost=pf_max(p,boost);else if(p<0)slow=pf_max(slow,sub32(0,p));}
 int remaining=pf_max(pf_min(sub32(100,slow),100),0);
 return div_or_zero(mul32(div_or_zero(mul32(boost,base),100),remaining),100);
}
CN_HD inline int speed_budget(const PfEntity&e,const PfMovement&m,const PfTargeting&t,
 const PfMovementConfig&c,const int*percents=nullptr,int count=0){
 if(e.flags&(e.flagBits.noMove|e.flagBits.noMoveAllowAttract))return 0;
 if(t.targetingComponentActive&&(t.dashWindupMs>0||t.resumeDelayMs>0))return 0;
 if(e.blockCountdownMs>0||pf_in_mask(e.state,0x3415))return 0;
 if(e.state==Dashing||e.state==Jumping)return c.jumpSpeed;
 if(e.state==SpawnPathfind)return c.spawnPathfindSpeed;
 if(e.state==IngamePathfind)return c.ingamePathfindSpeed;
 // Fixed8 cannot create clones; assert rather than apply unknown clone globals.
 assert(e.state!=CloneSetup);
 if(t.targetingComponentActive&&t.specialLoadPending)return 0;
 int speed=speed_modifier(percents,count,c.speed);
 return !e.movementComponent||m.chargeProgress<10000?speed:div_or_zero(mul32(c.chargeSpeedMultiplier,speed),100);
}
CN_HD inline int facing_gate(const PfEntity&e,const PfTargeting&t){return !pf_in_mask(e.state,0x3515)&&
 !(t.targetingComponentActive&&t.dashWindupMs>0)&&e.blockCountdownMs<1;}
CN_HD inline int avoidance_gate(const PfEntity&e,const PfTargeting&t){return !(e.flags&e.flagBits.noCheckAvoidance)&&
 !pf_in_mask(e.state,0x35ed)&&!(t.targetingComponentActive&&(t.specialLoadPending||t.dashWindupMs>0))&&e.blockCountdownMs<1;}
CN_HD inline int push_gate(const PfEntity&e,const PfTargeting&t,const PfMovementConfig&c){
 if((t.targetingComponentActive&&t.dashWindupMs>0)||(e.flags&e.flagBits.noCheckCollisions)||e.blockCountdownMs>0)return 0;
 if(uint32_t(e.state)<=13){if(pf_in_mask(e.state,0x3140))return 0;
 if(e.state==IngamePathfind)return c.ingamePathfindVisible;}return 1;
}
CN_HD inline int collides(const PfEntity&e){
 if((e.flags&e.flagBits.noCheckCollisions)||e.attached)return 0;
 if(e.state<=12&&((uint32_t(1)<<(uint32_t(e.state)&31))&((1<<3)|(1<<5)|(1<<12))))return 0;
 if(e.state==Deploying&&e.clone)return 0;return !e.besideBuilding;
}
CN_HD inline int avoidable(const PfEntity&e){return e.state!=12&&e.state!=13&&!(e.flags&e.flagBits.noCheckAvoidance)&&collides(e);}
CN_HD inline void grid_move(const GridView&g,int&x,int&y,int dx,int dy,const PfEntity*e,int flag,
 bool spawnWaterPermission=false){
 int cx=div_or_zero(x,500),cy=div_or_zero(y,500),ox=sub32(x,mul32(cx,500)),oy=sub32(y,mul32(cy,500));
 x=add32(x,dx);y=add32(y,dy);int water=flag&1;
 if(e&&e->type==5&&!(flag&1))water=e->state==Deploying?!e->air:0;
 auto blocks=[&](int c,int r){if(e&&e->type==5&&e->state==SpawnPathfind){
 if(spawnWaterPermission||water==0)return false;return g.water(c,r)!=0;}
 return water!=0&&g.water(c,r)!=0;};
 if(dx>0&&add32(ox,dx)>=500){int next=cx+1;if(next<0||cy<0||g.columns()<=next||g.rows()<=cy||blocks(next,cy))x=cx*500+499;}
 else if(dx<0&&add32(ox,dx)<0){int next=cx-1;if(next<0||cy<0||g.columns()<cx||g.rows()<=cy||blocks(next,cy))x=cx*500;}
 if(dy>0&&add32(oy,dy)>=500){int next=cy+1;if(next<0||cx<0||g.columns()<=cx||g.rows()<=next||blocks(cx,next))y=cy*500+499;}
 else if(dy<0&&add32(oy,dy)<0){int next=cy-1;if(next<0||cx<0||g.columns()<=cx||g.rows()<cy||blocks(cx,next))y=cy*500;}
}
CN_HD inline void direction_init(PfMovement&m,const PfEntity&e,int width){
 if(m.route.empty()){m.routeDirX=m.routeDirY=0;return;}int n=m.route.last(),row=div32(n,width),col=n-row*width;
 int x=sub32(col*500+250,e.x),y=sub32(row*500+250,e.y);normalize(x,y,256);m.routeDirX=x;m.routeDirY=y;
}
template<class Neighbours,class Q> CN_HD inline void avoidance(PfMovement&m,const PfEntity&e,
 const Neighbours&others,Q&q,int width){
 if(e.flags&(e.flagBits.disablePhysical|e.flagBits.avoidanceAsObstacle))return;
 int lastObstacleSide=1,obstacles=0,blockers=0,blockerSide=1;
 for(int i=0;i<others.count;i++){int idx=others.indices[i];const PfEntity&o=q.entity(idx);
 if(o.id==e.id||(e.zTotal()>0)==(o.zTotal()<1))continue;
 if(!avoidable(o)||(o.flags&o.flagBits.disablePhysical))continue;
 int cross=add32(mul32(e.dirX,sub32(e.y,o.y)),mul32(e.dirY,sub32(o.x,e.x)));
 bool moving=o.movementActive&&!(o.flags&o.flagBits.avoidanceAsObstacle);
 if(moving){int dot=add32(mul32(o.dirX,e.dirX),mul32(o.dirY,e.dirY));
 if(o.targetable&1){if(o.state==CloneSetup||o.state==Standing||o.state==Attacking||o.state==Casting)dot=0;
 if(q.targeting(idx).specialLoadPending)dot=0;}
 if(m.chargeProgress>=10000&&e.mass>o.mass)continue;if(dot>0)continue;
 blockers++;int blend=q.movement(idx).avoidanceBlend;blockerSide=blend!=0?(blend>0?1:0):(cross<0?1:0);continue;}
 if(m.route.size()>=2){int n=m.route.last(),r=div32(n,width),wx=(n-r*width)*500+250,wy=r*500+250;
 if(squared(sub32(wx,o.x),sub32(wy,o.y))<mul32(o.collisionRadius,o.collisionRadius))m.route.pop();}
 obstacles++;lastObstacleSide=cross<0?1:0;}
 if(obstacles+blockers<=0)return;int side=obstacles>0?lastObstacleSide:blockerSide;
 if(!m.avoidanceBlend)m.avoidanceBlend=(side&1)?200:-200;
 else if(obstacles>=1)m.avoidanceBlend=pf_max(pf_min(add32(m.avoidanceBlend,(lastObstacleSide&1)?20:-20),200),-200);
}
template<class Neighbours,class Q> CN_HD inline int aligned_static(const PfEntity&e,const Neighbours&near,Q&q){
 int matches=0;for(int i=0;i<near.count;i++){const PfEntity&o=q.entity(near.indices[i]);
 if(o.id==e.id||(e.zTotal()>0)==(o.zTotal()<1)||o.movementComponent)continue;
 if(o.x==e.x||o.y==e.y){if(matches>0)return 1;matches++;}}return 0;
}
template<class Q> CN_HD inline void push_pass(PfMovement&m,const PfEntity&e,Q&q){
 int radius=e.collisionRadius;if(radius==0||!collides(e)||(e.flags&(e.flagBits.disablePhysical|e.flagBits.avoidanceAsObstacle)))return;
 auto near=q.near(e.x,e.y,add32(radius,20));if(!near.count)return;int statics=0,sr=pf_min(radius,500);
 for(int i=0;i<near.count;i++){const PfEntity&o=q.entity(near.indices[i]);
 if(o.id==e.id||(e.zTotal()>0)==(o.zTotal()<1)||!collides(o)||(o.flags&o.flagBits.disablePhysical)||!o.alive)continue;
 uint64_t mask;if(o.side!=e.side){if(e.flags&e.flagBits.noPushedByEnemy)continue;mask=o.flagBits.noPushedByEnemy;}
 else{if(e.flags&e.flagBits.noPushedByAlly)continue;mask=o.flagBits.noPushedByAlly;}
 int mt=(o.flags&mask)?-1:o.mass,rt=o.movementComponent?radius:sr;if(!o.movementComponent)statics++;
 int dx=sub32(e.x,o.x),dy=sub32(e.y,o.y),reach=add32(o.collisionRadius,rt);
 if(abs32(dx)>reach||abs32(dy)>reach)continue;int square=add32(mul32(dx,dx),mul32(dy,dy));
 if(square==0){dy=e.side==0?-1:1;square=1;}if(square>mul32(reach,reach))continue;assert(square>=0);
 int d=int(isqrt64(uint32_t(square))),safe=pf_max(d,1),mag=pf_max(pf_min(sub32(reach,d),300),0);
 if(mt!=-1)mag=div_or_zero(mul32(mag,mt),e.mass);mag=add32(pf_min(mag,299),1);
 m.pushY=add32(m.pushY,div_or_zero(mul32(mag,dy),safe));m.pushCount=add32(m.pushCount,1);
 m.pushX=add32(m.pushX,div_or_zero(mul32(mag,dx),safe));}
 if(statics>0){auto aligned=q.near(e.x,e.y,1000);if(aligned_static(e,aligned,q)){
 if(m.pushX&&m.pushY)return;if(m.pushX)m.pushY=m.pushX;else m.pushX=m.pushY;}}
}
template<class Q> CN_HD inline int displace(PfMovement&m,PfEntity&e,const GridView&g,
 const PfMovementConfig&c,Q&q,int tx,int ty,int budget,int updateFacing,int attackFlag=0){
 (void)attackFlag;int sx=e.x,sy=e.y,dx=sub32(tx,sx),dy=sub32(ty,sy),d=distance(dx,dy),safe=pf_max(d,1),
 step=pf_min(pf_min(budget,safe),250);
 int scaledX=mul32(div_or_zero(s32(uint32_t(dx)<<8),safe),step),
 scaledY=mul32(div_or_zero(s32(uint32_t(dy)<<8),safe),step);
 if(!q.facingUpdateSuppressed()&&!m.pushbackInFlight&&(updateFacing&1)){
 int x=dx,y=dy;if(normalize(x,y,256)){e.dirX=x;e.dirY=y;}}
 int px=div_or_zero(scaledX,256),py=div_or_zero(scaledY,256);
 if(m.avoidanceBlend){int blend=pf_max(pf_min(m.avoidanceBlend,256),-256),forward=256-abs32(blend);
 m.workX=add32(sar32(mul32(forward,px),8),sar32(mul32(blend,py),8));
 m.workY=add32(sar32(mul32(forward,py),8),sar32(sub32(0,mul32(px,blend)),8));
 normalize(m.workX,m.workY,step);px=m.workX;py=m.workY;}
 int stuck=0;if(m.pushCount>=1){m.workX=div_or_zero(m.pushX,m.pushCount);m.workY=div_or_zero(m.pushY,m.pushCount);
 if(!m.pushUnclamped&&squared(m.workX,m.workY)>=22501)normalize(m.workX,m.workY,150);
 stuck=m.pushStuck!=0;px=add32(px,m.workX);py=add32(py,m.workY);m.clearPush();}
 // Ground-pushed water nudge is disabled by standard 1v1's mode. Buff pull's stuck remains.
 grid_move(g,e.x,e.y,px,py,&e,stuck,c.entersWaterWhileSpawnPathfinding);
 int remainingX=sub32(tx,e.x),remainingY=sub32(ty,e.y),projection=add32(
 div_or_zero(mul32(m.routeDirX,remainingX),256),div_or_zero(mul32(m.routeDirY,remainingY),256));
 int threshold=e.state==SpawnPathfind?c.spawnPathfindSpeed:e.state==IngamePathfind?c.ingamePathfindSpeed:1000;
 m.waypointReached=projection<=threshold;assert(m.chargeProgress==-1);return step;
}
CN_HD inline bool overlay_change_affects_routes(const GridView&g,const Route<>&old,const Route<>&fresh){
 if(old.empty()||fresh.empty())return true;int n=g.columns()*g.rows();
 for(int i=0;i<old.size();i++){int v=old.nodes[i];if(v>=0&&v<n&&g.previous[v]==0&&g.current[v]>0)return true;}
 for(int i=0;i<fresh.size();i++){int v=fresh.nodes[i];if(v>=0&&v<n&&g.previous[v]>=1&&g.current[v]==0)return true;}return false;
}
CN_HD inline int route_beyond_reference(PfMovement&m,const PfEntity&e,PfPoint ref,int width){
 if(m.route.size()<2)return 0;m.workX=sub32(ref.x,e.x);m.workY=sub32(ref.y,e.y);
 int baseline=distance(m.workX,m.workY);for(int i=m.route.size()-1;i>=0;i--){int n=m.route.nodes[i],r=div_or_zero(n,width),c=n-r*width;
 m.workX=sub32(c*500+250,ref.x);m.workY=sub32(r*500+250,ref.y);if(distance(m.workX,m.workY)>baseline)return 1;}return 0;
}
template<int CAP> CN_HD inline void route_copy_live(Route<CAP>&destination,const Route<CAP>&source){
 if(&destination==&source)return;destination.clear();
 for(int i=0;i<source.size();i++)destination.add(source.nodes[i]);
}
// Same previous=-1 and adjacent-only suppression as RoutePreparation's fresh container loop.
// Every stored value already passed Route.add's nonnegative/round-trip checks.
template<int CAP> CN_HD inline void route_deduplicate_adjacent(Route<CAP>&route){
 int write=0,previous=-1,count=route.size();for(int read=0;read<count;read++){
 int node=route.nodes[read];if(node==previous)continue;previous=node;route.nodes[write++]=static_cast<RouteNode>(node);}
 route.count=write;
}
template<class Q> CN_HD inline void prepare_route(PfMovement&m,const PfEntity&e,const GridView&g,Q&q){
 int gc,gr;bool hasRef=q.referenceAvailable();PfPoint ref=q.referencePoint();
 if(m.explicitX>=0&&m.explicitY>=0){gc=m.explicitX/500;gr=m.explicitY/500;}
 else if(hasRef){int packed=q.endpoint(ref.x/500,ref.y/500,q.attackRange());
 if(packed<0){gc=ref.x;gr=ref.y;}else{gc=sar32(packed,16);gr=packed&65535;}}
 else{int packed=q.gameModeGoal();if(packed<0)return;gc=sar32(packed,16);gr=packed&65535;}
 bool changed=uint32_t(e.side)<=1&&g.changeFlags[e.side]!=0,goalChanged=true,ground=!e.air;
 if(!m.route.empty()){int n=m.route.nodes[0];if(gc==n%g.columns()&&gr==n/g.columns()){
 if(!(changed&&ground))return;goalChanged=false;}}
 m.routeLeadsAway=0;if(!ground){m.route.clear();m.route.add(gr*g.columns()+gc);direction_init(m,e,g.columns());return;}
 if constexpr(requires {q.search_into(0,0,0,0,1,m.route);q.saved_route_scratch();}){
  // Fixed8 searches are not reentrant: endpoint/search perform no actions. Saved route scratch
  // belongs to this game, separate from every entity and the grid search working arrays.
  Route<>*saved=nullptr;
  if(!goalChanged){saved=&q.saved_route_scratch();assert(saved!=&m.route);route_copy_live(*saved,m.route);}
  q.search_into(div32(e.x,500),div32(e.y,500),gc,gr,1,m.route);
  route_deduplicate_adjacent(m.route);
  if(saved&&saved->size()>0&&!overlay_change_affects_routes(g,*saved,m.route))route_copy_live(m.route,*saved);
 }else{
  // Original value-returning control, retained for generic query fixtures/adapters.
  Route<>saved{};if(!goalChanged)saved=m.route;
  auto found=q.search(div32(e.x,500),div32(e.y,500),gc,gr,1);m.route.clear();int previous=-1;
  for(int i=0;i<found.size();i++){int n=found.nodes[i];if(n==previous)continue;previous=n;m.route.add(n);}
  if(saved.size()>0&&!overlay_change_affects_routes(g,saved,m.route))m.route=saved;
 }
 if(hasRef)m.routeLeadsAway=route_beyond_reference(m,e,ref,g.columns())&1;
 if(m.route.size()>=1)direction_init(m,e,g.columns());
}
template<class Q> CN_HD inline PfPoint select_waypoint(PfMovement&m,const PfEntity&e,const GridView&g,Q&q){
 if(m.route.empty()){if(q.referenceAvailable()&&!q.referenceInRange())return q.specialWaypoint();return {e.x,e.y};}
 int n=m.route.last(),row=div_or_zero(n,g.columns()),col=n-row*g.columns();return {col*500+250,row*500+250};
}
template<class Q> CN_HD inline void route_follow(PfMovement&m,PfEntity&e,const PfMovementConfig&c,const GridView&g,Q&q){
 if(e.flags&(e.flagBits.noMove|e.flagBits.noMoveAllowAttract)){
 assert(m.chargeProgress==-1);m.route.clear();m.routeLeadsAway=0;
 if(e.state==Moving||e.state==Dashing||e.state==Jumping||e.state==RouteFollowingAlternate)q.requestState(e,Standing);
 if(e.flags&e.flagBits.noMove)return;}
 if(q.avoidanceGate()){auto near=q.near(add32(e.x,e.dirX),add32(e.y,e.dirY),pf_min(e.collisionRadius,500));avoidance(m,e,near,q,g.columns());}
 int blend=m.avoidanceBlend;m.avoidanceBlend=blend>=1?pf_max(blend,10)-10:pf_min(blend,-10)+10;
 if(q.pushGate())push_pass(m,e,q);int speed=q.speedBudget();
 assert(e.state!=Dashing&&e.state!=Jumping);m.moveTimeMsCopy=m.moveTimeMs;
 if(speed>=1){int increment=q.hasModifierComponent()?div_or_zero(q.scaledTimeStep(),2):50;
 m.moveTimeMs=add32(m.moveTimeMs,increment);
 if(c.stopMovementAfterMs>=1&&m.moveTimeMs>c.stopMovementAfterMs){
 if(sub32(m.moveTimeMs,add32(c.stopMovementAfterMs,c.waitMs))<0)speed=0;
 else{m.moveTimeMs=sub32(m.moveTimeMs,add32(c.stopMovementAfterMs,c.waitMs));m.moveTimeMsCopy=m.moveTimeMs;}}}
 if(m.route.empty()){PfPoint p=select_waypoint(m,e,g,q);m.workX=p.x;m.workY=p.y;
 displace(m,e,g,c,q,p.x,p.y,speed,q.facingGate()&1);return;}
 if(speed<-249)return;int iteration=0,remaining=speed,limit=div_or_zero(speed,250);
 while(true){PfPoint p=select_waypoint(m,e,g,q);m.workX=p.x;m.workY=p.y;
 int nextRemaining=sub32(remaining,250),step=speed>249?pf_min(remaining,250):remaining;
 displace(m,e,g,c,q,p.x,p.y,step,q.facingGate()&1);
 if(m.waypointReached){m.route.pop();direction_init(m,e,g.columns());}
 if(iteration<limit&&!m.route.empty()){iteration++;remaining=nextRemaining;continue;}return;}
}
CN_HD inline void set_position(PfEntity&e,int x,int y,int z){if(e.x==INT32_MAX){e.prevX=x;e.prevY=y;e.prevZ=z;}e.x=x;e.y=y;e.z=z;}
template<class Q> CN_HD inline void movement_visit(PfMovement&m,PfEntity&e,const PfMovementConfig&c,const GridView&g,Q&q,
 bool noCheckCollisions=false){
 assert(!e.attached);if(e.state==Morphing)return;
 if(m.blockCountdown>=1){if(m.pushbackBudget>=1)push_pass(m,e,q);
 displace(m,e,g,c,q,m.targetX,m.targetY,m.pushbackBudget,0);m.blockCountdown=sub32(m.blockCountdown,1);
 if(!m.blockCountdown)m.pushbackBudget=0;return;}
 if(m.pushbackInFlight){if(!noCheckCollisions&&m.pushbackBudget>=1){push_pass(m,e,q);
 if(g.cellBlocked(e.x,e.y)&&!e.air&&!c.hovering){int p=relocate(g,e.x,e.y);set_position(e,p&65535,sar32(p,16),e.z);}}
 m.pushbackBudget=sub32(m.pushbackBudget,25);int budget=m.pushbackBudget,was=m.pushbackInFlight;
 displace(m,e,g,c,q,m.targetX,m.targetY,budget,1,m.attackPushback);m.pushbackInFlight=budget>=0;
 if(was&&budget<0){m.route.clear();m.routeLeadsAway=0;}return;}
 if(q.routeRequest()&1)prepare_route(m,e,g,q);route_follow(m,e,c,g,q);
}
CN_HD inline void set_pushback(PfMovement&m,const PfEntity&e,int x,int y,int amount,bool attack=false,
 bool subtract=false,bool keepLonger=false,int cap=40000){
 int dx=sub32(e.x,x),dy=sub32(e.y,y),n=add32(mul32(dx,dx),mul32(dy,dy));assert(n>=0);
 int length=int(isqrt64(uint32_t(n)));if(!length){length=1;dx=(e.id&1)?-1:1;}
 if(subtract){int left=sub32(amount,length);amount=left>0?pf_min(left,cap):0;}else amount=pf_min(amount,cap);
 if(m.pushbackInFlight&&keepLonger&&amount<m.pushbackBudget)return;
 m.attackPushback=attack;m.targetX=e.x;m.targetY=e.y;
 if(length>=1&&amount>=1){int step=0,total=0;do{step=add32(step,25);total=add32(total,step);}while(total<amount);
 m.pushbackBudget=step;m.targetX=add32(m.targetX,div32(mul32(amount,dx),length));
 m.targetY=add32(m.targetY,div32(mul32(amount,dy),length));m.pushbackInFlight=1;}
 else{m.pushbackInFlight=0;m.attackPushback=0;}
}
}
