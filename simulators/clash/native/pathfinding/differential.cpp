// SPDX-License-Identifier: Apache-2.0
// Differential probes for crforge core/src/main/java/org/crforge/core/pathfinding/
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge)
// Test-only C ABI; production simulation uses header-only templates.
#include "pathfinding.hpp"
using namespace clash_native;
struct ProbeQueries {
 // Neighbour probes read only the avoidance blend, not 64 full maximum-capacity routes.
 // Keep the independent owner's full movement state so default-int fixtures fit a JNI stack.
 struct NeighbourMovement {int avoidanceBlend=0;};
 PfEntity entities[64]{};NeighbourMovement movements[64]{};PfMovement ownerMovement{};PfTargeting targets[64]{};int nearCount=0;
 int scaled=100;bool held=false,valid=true,selectTarget=true,targetAccept=true;int hits=0,starts=0,turns=0;
 CN_HD PfEntity&entity(int i){return entities[i];}
 CN_HD NeighbourMovement&movement(int i){return movements[i];}
 CN_HD PfTargeting&targeting(int i){return targets[i];}
 CN_HD bool attackTimersHeld(){return held;}
 CN_HD int scaleTimeStep(int step){return div32(mul32(step,scaled),100);}
 CN_HD bool facingUpdateSuppressed(){return false;}
 CN_HD bool validate(int i,int){return valid&&i>=0;}
 CN_HD bool acceptsAttacker(int){return targetAccept;}
 CN_HD bool pendingDamageKeeps(int,int,int){return true;}
 CN_HD int select(PfTargetOutcome&){return selectTarget?1:-1;}
 CN_HD bool stopsWithoutTarget(){return false;}
 CN_HD bool buildingKeepsAttacking(){return false;}
 CN_HD bool abilityInProgress(){return false;}
 CN_HD void setState(PfEntity&e,int s){if(e.deployCountdown<=0)e.state=s;}
 CN_HD void requestState(PfEntity&e,int s){setState(e,s);}
 CN_HD void turnToward(int){turns++;}
 CN_HD void onStartingAttack(){starts++;}
 CN_HD bool hit(int){hits++;return false;}
 CN_HD int avoidanceGate(){return 0;}
 CN_HD int pushGate(){return 0;}
 CN_HD bool referenceAvailable(){return false;}
 CN_HD PfPoint referencePoint(){return {0,0};}
 CN_HD int endpoint(int,int,int){return -1;}
 CN_HD int attackRange(){return 0;}
 CN_HD int gameModeGoal(){return -1;}
 CN_HD Route<> search(int,int,int,int,int){return {};}
 CN_HD bool referenceInRange(){return false;}
 CN_HD PfPoint specialWaypoint(){return {0,0};}
 CN_HD int speedBudget(){return scaled;}
 CN_HD bool hasModifierComponent(){return false;}
 CN_HD int scaledTimeStep(){return 100;}
 CN_HD int facingGate(){return 1;}
 CN_HD int routeRequest(){return 0;}
 CN_HD int entityCount(){return nearCount;}
 CN_HD PfNeighbours<64> near(int,int,int){PfNeighbours<64> out;for(int i=0;i<nearCount;i++)out.add(i);return out;}
};
struct ProbePrepareQueries:ProbeQueries {
 Route<> found;
 CN_HD Route<>search(int,int,int,int,int){return found;}
};
struct ProbePrepareScratchQueries:ProbePrepareQueries {
 Route<> saved;
 CN_HD Route<>&saved_route_scratch(){return saved;}
 CN_HD void search_into(int,int,int,int,int,Route<>&route){route_copy_live(route,found);}
};
extern "C" int pf_probe(int op,const int*a,int*out){
 int o=0;
 if(op==1){int x=a[0],y=a[1];out[o++]=sine1024(a[2]);out[o++]=angle_of_vector(x,y);
 rotate1024(x,y,a[2]);out[o++]=x;out[o++]=y;}
 else if(op==2||op==18){int width=a[0],height=a[1];RouteSearchBuffers<>b;
 auto cost=[&](int c,int r){return a[10+r*width+c];};
 if(op==2){auto result=route_search(width,height,cost,a[2],a[3],b,a[4],a[5],bool(a[6]),bool(a[7]),bool(a[8]),a[9]);
 out[o++]=result.route.count;for(int i=0;i<result.route.count;i++)out[o++]=result.route.nodes[i];
 for(int i:result.counters)out[o++]=i;out[o++]=result.remainingBudget;}
 else{Route<>route;auto stats=route_search_into(width,height,cost,a[2],a[3],b,route,a[4],a[5],bool(a[6]),bool(a[7]),bool(a[8]),a[9]);
 out[o++]=route.count;for(int i=0;i<route.count;i++)out[o++]=route.nodes[i];for(int n:stats.counters)out[o++]=n;out[o++]=stats.remainingBudget;}}
 else if(op==3){CellCosts c;out[o++]=cell_cost(a[0],a[1],a[2],a[3],a[4],c,a[5],a[6],a[7],a[8],a[9],a[10],a[11]);}
 else if(op==4){GridView g;g.width=a[0];g.height=a[1];g.tiles=a+12;PfEntity e;
 int x=a[2],y=a[3];e.type=a[6];e.state=a[7];e.air=a[8];grid_move(g,x,y,a[4],a[5],&e,a[9],a[10]);out[o++]=x;out[o++]=y;}
 else if(op==5){PfTargetConfig c;c.hitSpeed=a[0];c.loadTime=a[1];c.hitSpeedMultiplier=a[2];
 PfTargeting t;t.reference=a[3];t.attackSequenceIndex=a[4];t.attackTimerMs=a[5];t.loadTimerMs=a[6];
 t.attackStartDelayMs=a[7];t.chargeStrike=a[8];t.instantHit=a[9];t.burstProgressMs=a[10];
 ProbeQueries q;q.held=a[11];q.scaled=a[12];attack_timer_advance(t,c,q);
 out[o++]=t.attackTimerMs;out[o++]=t.loadTimerMs;out[o++]=t.attackStartDelayMs;out[o++]=t.chargeStrike;
 out[o++]=t.instantHit;out[o++]=t.burstProgressMs;out[o++]=t.hitInProgressWithoutReference;out[o++]=t.attackTimeRoundedUp;}
 else if(op==6){GridView g;g.width=36;g.height=64;int tiles[2304]{};g.tiles=tiles;
 PfEntity e;e.x=a[0];e.y=a[1];e.dirX=a[2];e.dirY=a[3];e.state=a[4];
 PfMovement m;m.routeDirX=a[5];m.routeDirY=a[6];m.avoidanceBlend=a[7];m.pushX=a[8];m.pushY=a[9];m.pushCount=a[10];
 m.pushStuck=a[11];m.pushUnclamped=a[12];PfMovementConfig c;ProbeQueries q;
 int step=displace(m,e,g,c,q,a[13],a[14],a[15],a[16]);
 out[o++]=e.x;out[o++]=e.y;out[o++]=e.dirX;out[o++]=e.dirY;out[o++]=m.waypointReached;
 out[o++]=m.pushX;out[o++]=m.pushY;out[o++]=m.pushCount;out[o++]=step;}
 else if(op==7){PfEntity e;e.x=a[0];e.y=a[1];e.state=a[2];e.deployCountdown=a[3];PfEntity r;r.x=a[4];r.y=a[5];r.collisionRadius=a[6];
 PfTargetConfig c;c.range=a[7];c.collisionRadius=a[8];c.hitSpeed=a[9];c.loadTime=a[10];c.hasProjectile=a[11];c.isBuilding=a[12];
 PfTargeting t;t.reference=a[13];t.attackTimerMs=a[14];t.loadTimerMs=a[15];t.hitInProgress=a[16];t.attackBlockTimerMs=a[17];
 t.visitHoldMs=a[18];t.resumeDelayMs=a[19];t.resumeDelayElapsedMs=a[20];t.visitSuspended=a[21];
 ProbeQueries q;q.entities[1]=r;q.valid=a[22];q.selectTarget=a[23];q.held=a[24];q.scaled=a[25];
 PfMovement m;m.pushbackInFlight=a[26];m.blockCountdown=a[27];PfTargetOutcome outc;
 targeting_visit(t,e,&m,c,q,outc);
 out[o++]=t.reference;out[o++]=t.attackTimerMs;out[o++]=t.loadTimerMs;out[o++]=t.hitInProgress;
 out[o++]=t.attackBlockTimerMs;out[o++]=t.visitHoldMs;out[o++]=t.resumeDelayMs;out[o++]=t.resumeDelayElapsedMs;
 out[o++]=e.state;out[o++]=outc.resumeRequested;out[o++]=q.hits;out[o++]=q.starts;out[o++]=q.turns;
 out[o++]=t.hitInProgressWithoutReference;out[o++]=t.attackTimeRoundedUp;}
 else if(op==8){GridView g;g.width=36;g.height=64;int tiles[2304]{};g.tiles=tiles;
 PfEntity e;e.x=a[0];e.y=a[1];e.state=a[2];e.dirX=a[3];e.dirY=a[4];
 PfMovement m;m.routeDirX=a[5];m.routeDirY=a[6];m.moveTimeMs=a[7];m.avoidanceBlend=a[8];
 PfMovementConfig c;c.stopMovementAfterMs=a[9];c.waitMs=a[10];for(int i=0;i<a[11];i++)m.route.add(a[13+i]);
 ProbeQueries q;q.scaled=a[12];movement_visit(m,e,c,g,q);
 out[o++]=e.x;out[o++]=e.y;out[o++]=e.dirX;out[o++]=e.dirY;out[o++]=m.waypointReached;
 out[o++]=m.route.count;for(int i=0;i<m.route.count;i++)out[o++]=m.route.nodes[i];
 out[o++]=m.routeDirX;out[o++]=m.routeDirY;out[o++]=m.moveTimeMs;out[o++]=m.moveTimeMsCopy;out[o++]=m.avoidanceBlend;}
 else if(op==9||op==10||op==11||op==24){
 ProbeQueries q;q.nearCount=a[0];int input=(op==11||op==24)?9:5+a[3];
 for(int i=0;i<q.nearCount;i++){PfEntity&e=q.entities[i];const int*v=a+input+i*16;
 e.x=v[0];e.y=v[1];e.z=v[2];e.dirX=v[3];e.dirY=v[4];e.collisionRadius=v[5];e.mass=v[6];e.state=v[7];
 e.movementActive=v[8];e.movementComponent=v[9];e.targetable=v[10];q.targets[i].specialLoadPending=v[11];
 q.movements[i].avoidanceBlend=v[12];e.alive=v[13];e.id=v[14];e.side=v[15];e.crownTower=v[14]%3==0;e.building=!e.movementComponent;}
 if(op==11||op==24){PfSpatialIndex<64> index;index.rebuild(36,64,q);PfSpatialQuery query;
 query.x=a[1];query.y=a[2];query.radius=a[3];query.halfHeight=a[4];query.kingsLast=a[5];query.buildingAware=a[6];
 query.typeMask=a[7];query.excludeTeam=a[8];PfNeighbours<64>result;
 if(op==11)result=index.query(query,q);else{int keys[64];index.query_into(query,q,result,keys);}out[o++]=result.count;
 for(int i=0;i<result.count;i++)out[o++]=result.indices[i];}
 else{auto&m=q.ownerMovement;m.avoidanceBlend=a[1];m.chargeProgress=a[2];for(int i=0;i<a[3];i++)m.route.add(a[5+i]);
 if(op==9){auto near=q.near(0,0,0);avoidance(m,q.entities[0],near,q,a[4]);
 out[o++]=m.avoidanceBlend;out[o++]=m.route.count;for(int i=0;i<m.route.count;i++)out[o++]=m.route.nodes[i];}
 else{push_pass(m,q.entities[0],q);out[o++]=m.pushX;out[o++]=m.pushY;out[o++]=m.pushCount;}}
 }
 else if(op>=12&&op<=15){GridView g;g.width=a[0];g.height=a[1];g.tiles=a+12;
 int overlay[2304]{};for(int i=0;i<g.width*g.height;i++)overlay[i]=a[12+g.width*g.height+i];g.current=overlay;
 if(op==12){int tx=a[4],ty=a[5];auto ds=[&](int x,int y){return squared_distance(tx,ty,x,y);};
 out[o++]=select_endpoint(g,a[2],a[3],tx/500,ty/500,a[6],a[7],ds);}
 else if(op==13)out[o++]=lane_assignment(g,a[2],a[3],a[4],a[5]);
 else if(op==14){int packed=0;bool fit=rasterize_footprint(g,a[2],a[3],a[4],a[5],a[6],packed);
 out[o++]=fit;out[o++]=fit?packed:0;for(int i=0;i<g.width*g.height;i++)out[o++]=overlay[i];}
 else out[o++]=relocate(g,a[2],a[3],a[4]);}
 else if(op==16){PfEntity e;e.x=a[0];e.y=a[1];e.id=a[2];PfMovement m;
 m.pushbackInFlight=a[3];m.pushbackBudget=a[4];m.targetX=a[5];m.targetY=a[6];
 set_pushback(m,e,a[7],a[8],a[9],a[10],a[11],a[12]);
 out[o++]=m.pushbackInFlight;out[o++]=m.pushbackBudget;out[o++]=m.targetX;out[o++]=m.targetY;out[o++]=m.attackPushback;}
 else if(op==17){ProbeQueries q;PfEntity e;e.id=5000007;auto&r=q.entities[1];r.id=5000008;r.side=1;
 r.state=a[0];r.alive=a[1];r.crownTower=a[2];r.flags=a[3]?1:0;r.flagBits.hidden=1;q.targetAccept=a[4];
 PfTargeting t;t.aliveCheckBypass=a[5];PfTargetConfig c;out[o++]=validate_reference(t,e,c,1,a[6],q);}
 else if(op==19||op==20){int width=a[0],height=a[1];RouteSearchBuffers<>b;
 auto cost=[&](int c,int r){return c>=0&&r>=0&&c<width&&r<height?a[7+r*width+c]:-1;};
 if(op==20){auto result=grid_search(width,height,cost,a[2],a[3],a[4],a[5],a[6],b);
 out[o++]=result.route.count;for(int i=0;i<result.route.count;i++)out[o++]=result.route.nodes[i];
 for(int i:result.counters)out[o++]=i;out[o++]=result.remainingBudget;}
 else{Route<>route;auto stats=grid_search_into(width,height,cost,a[2],a[3],a[4],a[5],a[6],b,route);
 out[o++]=route.count;for(int i=0;i<route.count;i++)out[o++]=route.nodes[i];for(int n:stats.counters)out[o++]=n;out[o++]=stats.remainingBudget;}}
 else if(op==21){Route<>route;for(int i=0;i<a[0];i++)route.add(a[1+i]);route_deduplicate_adjacent(route);
 out[o++]=route.count;for(int i=0;i<route.count;i++)out[o++]=route.nodes[i];}
 else if(op==22||op==23){GridView grid;grid.width=a[0];grid.height=a[1];int tiles[2304]{},current[2304]{},previous[2304]{};
 grid.tiles=tiles;grid.current=current;grid.previous=previous;grid.changeFlags[0]=a[3];
 if(a[6]>=0){current[a[6]]=a[7];previous[a[6]]=a[8];}
 PfEntity e;e.x=a[9];e.y=a[10];PfMovement m;m.explicitX=(a[2]%grid.width)*500+250;m.explicitY=(a[2]/grid.width)*500+250;m.routeLeadsAway=1;
 for(int i=0;i<a[4];i++)m.route.add(a[11+i]);
 if(op==22){ProbePrepareQueries q;for(int i=0;i<a[5];i++)q.found.add(a[11+a[4]+i]);prepare_route(m,e,grid,q);}
 else{ProbePrepareScratchQueries q;for(int i=0;i<a[5];i++)q.found.add(a[11+a[4]+i]);prepare_route(m,e,grid,q);}
 out[o++]=m.route.count;for(int i=0;i<m.route.count;i++)out[o++]=m.route.nodes[i];
 out[o++]=m.routeDirX;out[o++]=m.routeDirY;out[o++]=m.routeLeadsAway;}
 return o;
}
