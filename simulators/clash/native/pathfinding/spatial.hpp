// SPDX-License-Identifier: Apache-2.0
// Ported from crforge core/src/main/java/org/crforge/core/pathfinding/index/{SpatialIndex,
// SpatialQuery,ShapeTests}.java @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de
// https://github.com/voonhous/crforge
#pragma once
#include "types.hpp"
namespace clash_native {
CN_HD inline int pf_team(const PfEntity&e){return e.side==100?2:e.side&1;}
CN_HD inline bool within_circle(int ex,int ey,int er,int x,int y,int radius){int r=add32(er,radius);
 return squared(sub32(x,ex),sub32(y,ey))<mul32(r,r);}
CN_HD inline int clamped_offset(int value,int low,int size){return sub32(pf_max(pf_min(value,add32(low,size)),low),value);}
CN_HD inline bool within_box(int ex,int ey,int radius,int left,int top,int width,int height){
 int dx=clamped_offset(ex,left,width);if(dx>radius||dx<sub32(0,radius))return false;
 int dy=clamped_offset(ey,top,height);if(dy>radius||dy<sub32(0,radius))return false;
 return uint32_t(add32(mul32(dx,dx),mul32(dy,dy)))<uint32_t(mul32(radius,radius));}
CN_HD inline bool within_circle_shape(const PfEntity&e,int x,int y,int radius){
 if(!e.building)return within_circle(e.x,e.y,e.collisionRadius,x,y,radius);
 return within_box(x,y,radius,sub32(e.x,e.collisionRadius),sub32(e.y,e.collisionRadius),
 mul32(2,e.collisionRadius),mul32(2,e.collisionRadius));}
CN_HD inline bool box_overlap(const PfEntity&e,int x,int y,int radius,int halfHeight){int r=e.collisionRadius;
 return !(add32(e.x,r)<sub32(x,radius)||add32(e.y,r)<sub32(y,halfHeight)||
 sub32(e.x,r)>=add32(x,radius))&&sub32(e.y,r)<add32(y,halfHeight);}
struct PfSpatialQuery {int x=0,y=0,radius=0,halfHeight=0;bool kingsLast=false,buildingAware=false;
 int typeMask=0,excludeTeam=-1;};
template<int CAP=1024> struct PfNeighbours {int indices[CAP]{};int count=0;
 CN_HD void add(int i){assert(count<CAP);indices[count++]=i;}};
// Borrowed only until the next query on the same game. Fixed8's query consumers complete
// their list traversal before another spatial query; generic by-value controls remain below.
struct PfNeighbourView {const int*indices=nullptr;int count=0;};
// Stores snapshot insertion extents, and computes each entity's first visited bucket directly.
// Sorting this first bucket key reproduces the original bucket/dedup order exactly. Geometry is
// deliberately tested against live state, while bucket extents remain those of rebuild().
template<int CAP=1024> struct PfSpatialIndex {
 int width=0,high=0,count=0;int indices[CAP]{},lowX[CAP]{},highX[CAP]{},lowY[CAP]{},highY[CAP]{};
 CN_HD void clear(){count=0;}
 CN_HD void dimensions(int cellsWide,int cellsHigh){width=(cellsWide*500+1023)>>10;high=(cellsHigh*500+1023)>>10;assert(high<=64);}
 CN_HD void insert(int index,const PfEntity&e){if(e.collisionRadius<1)return;assert(count<CAP);
 int i=count++,m=add32(e.collisionRadius,e.movementActive?250:0);indices[i]=index;
 lowX[i]=sar32(sub32(e.x,m),10);highX[i]=sar32(add32(e.x,m),10);
 lowY[i]=sar32(sub32(e.y,m),10);highY[i]=sar32(add32(e.y,m),10);}
 template<class Q> CN_HD void rebuild(int cw,int ch,Q&q){dimensions(cw,ch);clear();
 for(int i=0;i<q.entityCount();i++)insert(i,q.entity(i));}
 template<class Q> CN_HD PfNeighbours<CAP> query(PfSpatialQuery query,Q&q)const{
 PfNeighbours<CAP> out{};int keys[CAP]{};
 int xl=sar32(sub32(query.x,query.radius),10),xh=sar32(add32(query.x,query.radius),10),
 yl=sar32(sub32(query.y,query.radius),10),yh=sar32(add32(query.y,query.radius),10);
 if(xl>xh||yl>yh)return out;
 for(int i=0;i<count;i++){
 int firstX=pf_max(pf_max(xl,lowX[i]),0),lastX=pf_min(pf_min(xh,highX[i]),width-1),
 firstY=pf_max(pf_max(yl,lowY[i]),0),lastY=pf_min(pf_min(yh,highY[i]),high-1);
 if(firstX>lastX||firstY>lastY)continue;const PfEntity&e=q.entity(indices[i]);
 if(query.typeMask>=1&&((uint32_t(query.typeMask)>>(uint32_t(e.type)&31))&1)==0)continue;
 if(query.excludeTeam!=-1&&pf_team(e)==query.excludeTeam)continue;
 bool accepted;
 if(query.halfHeight!=0){accepted=query.buildingAware&&e.building?
 box_overlap(e,query.x,query.y,query.radius,query.halfHeight):
 within_box(e.x,e.y,e.collisionRadius,sub32(query.x,query.radius),sub32(query.y,query.halfHeight),
 mul32(2,query.radius),mul32(2,query.halfHeight));}
 else accepted=query.buildingAware?within_circle_shape(e,query.x,query.y,query.radius):
 within_circle(e.x,e.y,e.collisionRadius,query.x,query.y,query.radius);
 if(!accepted)continue;int key=firstX*high+firstY,pos=out.count;
 while(pos>0&&key<keys[pos-1]){out.indices[pos]=out.indices[pos-1];keys[pos]=keys[pos-1];pos--;}
 out.indices[pos]=indices[i];keys[pos]=key;out.count++;}
 // Oracle removes crowns by scanning backward and appends each: reversed crown subsequence.
 if(query.kingsLast){PfNeighbours<CAP> partition{};
 for(int i=0;i<out.count;i++)if(!q.entity(out.indices[i]).crownTower)partition.add(out.indices[i]);
 for(int i=out.count-1;i>=0;i--)if(q.entity(out.indices[i]).crownTower)partition.add(out.indices[i]);return partition;}
 return out;
 }
 // Same bucket order/filter/ties as query(), with no per-thread arrays or return copies.
 // keys has CAP entries. Once ranking is finished it is reused for the crown subsequence.
 template<class Q> CN_HD void query_into(PfSpatialQuery query,Q&q,PfNeighbours<CAP>&out,int*keys)const{
 out.count=0;
 int xl=sar32(sub32(query.x,query.radius),10),xh=sar32(add32(query.x,query.radius),10),
 yl=sar32(sub32(query.y,query.radius),10),yh=sar32(add32(query.y,query.radius),10);
 if(xl>xh||yl>yh)return;
 for(int i=0;i<count;i++){
 int firstX=pf_max(pf_max(xl,lowX[i]),0),lastX=pf_min(pf_min(xh,highX[i]),width-1),
 firstY=pf_max(pf_max(yl,lowY[i]),0),lastY=pf_min(pf_min(yh,highY[i]),high-1);
 if(firstX>lastX||firstY>lastY)continue;const PfEntity&e=q.entity(indices[i]);
 if(query.typeMask>=1&&((uint32_t(query.typeMask)>>(uint32_t(e.type)&31))&1)==0)continue;
 if(query.excludeTeam!=-1&&pf_team(e)==query.excludeTeam)continue;
 bool accepted;
 if(query.halfHeight!=0){accepted=query.buildingAware&&e.building?
 box_overlap(e,query.x,query.y,query.radius,query.halfHeight):
 within_box(e.x,e.y,e.collisionRadius,sub32(query.x,query.radius),sub32(query.y,query.halfHeight),
 mul32(2,query.radius),mul32(2,query.halfHeight));}
 else accepted=query.buildingAware?within_circle_shape(e,query.x,query.y,query.radius):
 within_circle(e.x,e.y,e.collisionRadius,query.x,query.y,query.radius);
 if(!accepted)continue;assert(out.count<CAP);int key=firstX*high+firstY,pos=out.count;
 while(pos>0&&key<keys[pos-1]){out.indices[pos]=out.indices[pos-1];keys[pos]=keys[pos-1];pos--;}
 out.indices[pos]=indices[i];keys[pos]=key;out.count++;}
 if(query.kingsLast){int ordinary=0,crowns=0,total=out.count;
 for(int i=0;i<total;i++){int index=out.indices[i];if(q.entity(index).crownTower)keys[crowns++]=index;
 else out.indices[ordinary++]=index;}
 for(int i=crowns-1;i>=0;i--)out.indices[ordinary++]=keys[i];
 assert(ordinary==total);}
 }
};
}
