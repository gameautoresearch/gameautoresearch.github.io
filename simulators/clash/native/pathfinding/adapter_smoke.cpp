// SPDX-License-Identifier: Apache-2.0
// Oracle API integration smoke for the crforge pathfinding port, pinned
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge).
#include "world_queries.hpp"
#include <cstdio>
using namespace clash_native;
struct SmokeWorld {
 PfEntity entities[3]{};PfMovement moves[3]{};PfTargeting targets[3]{};
 PfTargetConfig tc[3]{};PfMovementConfig mc[3]{};
 int tiles[2304]{},overlay[2304]{},previous[2304]{};
 GridView g{36,64,tiles,overlay,previous};RouteSearchBuffers<>scratch;PfSpatialIndex<8>index;
 int hits=0;
#ifdef CN_TEST_SPATIAL_SCRATCH
 PfNeighbours<8> spatialResult{};int spatialKeys[8]{};
 PfNeighbourView spatial_query_view(PfSpatialQuery query){index.query_into(query,*this,spatialResult,spatialKeys);return{spatialResult.indices,spatialResult.count};}
#endif
#ifdef CN_TEST_ROUTE_SCRATCH
 Route<>savedScratch{};
 Route<>&saved_route_scratch(){return savedScratch;}
#endif
 PfEntity&entity(int i){return entities[i];}PfTargeting&targeting(int i){return targets[i];}
 PfMovement*movement(int i){return i==0?&moves[i]:nullptr;}
 const PfTargetConfig&target_config(int i){return tc[i];}
 const PfMovementConfig&movement_config(int i){return mc[i];}
 GridView&grid(){return g;}RouteSearchBuffers<>&search_scratch(){return scratch;}
 int entityCount(){return 3;}
 PfNeighbours<8>spatial_query(PfSpatialQuery query){return index.query(query,*this);}
 int enemy_king(int){return 2;}PfNeighbours<2>enemy_princess_towers(int){PfNeighbours<2>r;r.add(1);return r;}
 const int*speed_percents(int){return nullptr;}int speed_percent_count(int){return 0;}
 int scale_hit_time(int,int base){return base;}bool match_ended(){return false;}
 void prepare_route(int i){prepare_fixed8_route(*this,i);}
 void set_unit_state(int i,int state){PfEntity&e=entity(i);if(e.state==state||e.deployCountdown>0)return;
 if(e.state==Attacking)targeting(i).targetLostTimerMs=0;e.state=state;
 if(movement(i)){if(state==Standing||state==Attacking){moves[i].route.clear();moves[i].routeLeadsAway=0;}
 else if(state==Moving)prepare_route(i);}}
 void resume_unit(int i){set_unit_state(i,entity(i).building?Standing:Moving);}
 bool apply_attack(int,int target){assert(target>=0);hits++;return false;}
};
int main(int argc,char**){SmokeWorld w;auto&e=w.entity(0);e.id=5000007;e.x=9000;e.y=1000;e.state=Moving;
 e.movementActive=e.movementComponent=true;e.targetable=1;e.collisionRadius=500;e.mass=6;e.hp=1000;
 w.moves[0].routeDirY=256;w.tc[0].range=1200;w.tc[0].sightRange=5500;w.tc[0].collisionRadius=500;
 w.tc[0].hitSpeed=1200;w.tc[0].loadTime=700;w.mc[0].speed=60;
 for(int i=1;i<3;i++){auto&r=w.entity(i);r.id=5000000+i;r.x=i==1?9000:14000;r.y=25000;
 r.collisionRadius=750;r.mass=20;r.hp=4000;r.side=1;r.building=r.crownTower=true;r.targetable=1;w.tc[i].isBuilding=true;}
 w.targets[0].movementComponentActive=true;initialize_fixed8_targeting(w.targets[0],e,w.tc[0]);
 for(int i=0;i<600;i++){w.index.rebuild(36,64,w);run_fixed8_targeting(w,0);run_fixed8_movement(w,0);
 if(argc>1){auto&t=w.targets[0];auto&m=w.moves[0];
 std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",e.x,e.y,e.dirX,e.dirY,e.state,t.reference,
 t.attackTimerMs,t.loadTimerMs,m.route.count,m.routeDirX,m.routeDirY,w.hits);}}
 assert(w.hits>0);assert(e.state==Attacking);assert(e.y>20000);assert(e.y<25000);
 std::printf("{\"status\":\"passed\",\"ticks\":600,\"hits\":%d,\"position\":[%d,%d]}\n",w.hits,e.x,e.y);
}
