// SPDX-License-Identifier: Apache-2.0
// Full-capacity caller-scratch control for crforge pathfinding/index/{SpatialIndex,ShapeTests}.java
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge).
#include "spatial.hpp"
#include <cstdio>
using namespace clash_native;
struct Q{PfEntity e[512];int entityCount(){return 512;}const PfEntity&entity(int i){return e[i];}};
int main(){Q q;PfSpatialIndex<512>index;PfNeighbours<512>out;int keys[512];
 for(int i=0;i<512;i++){auto&e=q.e[i];e.id=5000000+i;e.x=1000+20*i;e.y=1000+40*i;
 e.collisionRadius=250+(i%4)*250;e.movementActive=i&1;e.side=i&1;e.crownTower=i%3==0;e.building=i%7==0;}
 index.rebuild(36,64,q);PfSpatialQuery all{9000,16000,40000,0,true,false,0,-1};
 index.query_into(all,q,out,keys);assert(out.count==512);
 uint32_t random=8811;auto next=[&](){random^=random<<13;random^=random>>17;random^=random<<5;return random;};
 int checks=0;for(int test=0;test<3000;test++){
 if(test%7==0){for(int i=0;i<512;i++){q.e[i].x=add32(q.e[i].x,int(next()%101)-50);q.e[i].y=add32(q.e[i].y,int(next()%101)-50);}}
 if(test%11==0)index.rebuild(36,64,q);
 PfSpatialQuery query{int(next()%24000)-3000,int(next()%38000)-3000,int(next()%12000),
 test%3?0:int(next()%6000),bool(next()&1),bool(next()&1),test%5?0:1<<5,int(next()%3)-1};
 auto expected=index.query(query,q);index.query_into(query,q,out,keys);assert(expected.count==out.count);
 for(int i=0;i<out.count;i++)assert(expected.indices[i]==out.indices[i]);++checks;}
 std::printf("{\"status\":\"passed\",\"queries\":%d,\"capacity\":512,\"all512_retained\":true,\"reused_scratch\":true,\"snapshot_live_geometry\":true}\n",checks);
}
