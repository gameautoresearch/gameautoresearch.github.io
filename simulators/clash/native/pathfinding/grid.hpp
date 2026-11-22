// SPDX-License-Identifier: Apache-2.0
// Ported from crforge core/src/main/java/org/crforge/core/pathfinding/grid/{TileMap,
// CellCosts,CellTests,CellCostField,Route,RouteSearchHeap,RouteSearch,RouteSearchWrapper,
// GridSearchService,ReferenceEndpoint,LaneAssignment,FootprintOverlay,Relocation}.java
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge)
#pragma once
#include "types.hpp"
namespace clash_native {
struct GridView {
 int width=0,height=0; const int* tiles=nullptr; int* current=nullptr;int* previous=nullptr;
 int active=0,changeFlags[2]{},hash[2]{},changed[2]{};bool dynamicEnabled=true;int buildingCost=50;
 CN_HD constexpr int columns()const{
#if defined(CN_CUDA_FIXED8_GRID) && defined(__CUDA_ARCH__)
 return 36;
#else
 return width;
#endif
 }
 CN_HD constexpr int rows()const{
#if defined(CN_CUDA_FIXED8_GRID) && defined(__CUDA_ARCH__)
 return 64;
#else
 return height;
#endif
 }
 CN_HD bool contains(int c,int r)const{return c>=0&&r>=0&&c<columns()&&r<rows();}
 CN_HD int bits(int c,int r)const{assert(contains(c,r));return tiles[r*columns()+c]&127;}
 CN_HD int water(int c,int r)const{return (bits(c,r)>>5)&1;}
 CN_HD int overlayBlocks(int c,int r)const{return contains(c,r)&&dynamicEnabled&&current[r*columns()+c]>=buildingCost;}
 CN_HD int cellBlocked(int x,int y)const{if(x<0||y<0||x>=columns()*500||y>=rows()*500)return 1;
 return (bits(x/500,y/500)&(16|32|64))!=0;}
 CN_HD void swap(){int* older=previous;previous=current;current=older;
 for(int i=0;i<columns()*rows();i++)current[i]=0;}
};
struct CellCosts {int water=7,blocked=50,building=50,ordinary=8,road=5,matchingRoad=5;};
CN_HD inline int cell_cost(int width,int height,int col,int row,int bits,const CellCosts& c,
 bool present,bool water,bool alternateWater,int state,int lane,bool dynamic,int overlay){
 if(col<0||row<0||col>=width||row>=height)return -1;
 if(bits&32)return present&&(water||alternateWater)?c.water:c.blocked;
 if(bits&64)return c.blocked;
 if(present&&(state==6||state==7||state==15))return c.ordinary;
 int road=bits&3,cost=road?(present&&lane==road?c.matchingRoad:c.road):c.ordinary;
 return dynamic?pf_max(cost,overlay):cost;
}
struct GridCostLookup {
 GridView grid;CellCosts costs{};int state=0,lane=0;bool waterPermission=false,alternateWaterPermission=false;
 CN_HD int operator()(int col,int row)const{if(!grid.contains(col,row))return -1;
 return cell_cost(grid.columns(),grid.rows(),col,row,grid.bits(col,row),costs,true,
 waterPermission,alternateWaterPermission,state,lane,grid.active!=0,grid.current[row*grid.columns()+col]);}
};
CN_HD inline int route_heuristic(int dx,int dy,int method,bool accumulated){
 if(method==3&&!accumulated){int n=add32(mul32(dx,dx),mul32(dy,dy));assert(n>=0);return mul32(10,int(isqrt64(n)));}
 int ax=abs32(dx),ay=abs32(dy);
 if(accumulated||method==0)return mul32(10,pf_max(ax,ay));
 assert(method==1||method==2);return add32(mul32(10,pf_max(ax,ay)),mul32(4,pf_min(ax,ay)));
}
template<int CAP=4096> struct RouteSearchBuffers {
 int state[CAP]{},parent[CAP]{},carried[CAP]{},priority[CAP]{},heap[CAP]{},cost[CAP]{};int heapSize=0;
 CN_HD void prepare(int cells){assert(cells<=CAP);heapSize=0;for(int i=0;i<cells;i++){
 state[i]=parent[i]=carried[i]=priority[i]=0;cost[i]=INT32_MIN;}}
 CN_HD void siftUp(int i){while(i>0){int p=(i-1)/2;if(priority[heap[i]]>=priority[heap[p]])break;
 int x=heap[p];heap[p]=heap[i];heap[i]=x;i=p;}}
 CN_HD void add(int node){assert(heapSize<CAP);heap[heapSize++]=node;siftUp(heapSize-1);}
 CN_HD int pop(){assert(heapSize>0);int top=heap[0],last=heap[--heapSize];if(heapSize){heap[0]=last;int i=0;
 while(true){int best=i,right=2*i+2,left=2*i+1;
 if(right<heapSize&&priority[heap[best]]>priority[heap[right]])best=right;
 if(left<heapSize&&priority[heap[best]]>priority[heap[left]])best=left;
 if(best==i)break;int x=heap[i];heap[i]=heap[best];heap[best]=x;i=best;}}return top;}
 CN_HD void refresh(int node){int i=0;while(i<heapSize&&heap[i]!=node)i++;assert(i<heapSize);siftUp(i);}
};
template<int CAP=4096> struct RouteSearchResult {Route<CAP> route{};int counters[4]{};int remainingBudget=0;};
// Search metrics independent of route storage, allowing the full-capacity route to stay in
// the caller's global per-game storage rather than becoming a CUDA thread-local return value.
struct RouteSearchStats {int counters[4]{};int remainingBudget=0;};
template<int CAP,class Lookup> CN_HD inline void route_expand(RouteSearchBuffers<CAP>& b,Lookup& lookup,
 int width,int height,int node,int goalCol,int goalRow,int method,int weight,bool accumulated,bool refresh,
 bool reopen,int* counters){
 constexpr int dc[8]={0,0,-1,1,-1,-1,1,1},dr[8]={-1,1,0,0,-1,1,1,-1};
 int col=node%width,row=node/width;
 for(int i=0;i<8;i++){int c=col+dc[i],r=row+dr[i];if(c<0||c>=width||r<0||r>=height)continue;
 int target=r*width+c,cost=b.cost[target];if(cost==INT32_MIN){cost=lookup(c,r);b.cost[target]=cost;}
 if(cost==-1)continue;int state=b.state[target];if(state==2&&!reopen)continue;if(state==1&&!refresh)continue;
 int carried=add32(b.carried[node],mul32(cost,i<4?10:14));
 int estimate=add32(carried,mul32(weight,route_heuristic(goalCol-c,goalRow-r,method,accumulated)));
 if(state!=0&&estimate>=b.priority[target])continue;
 b.parent[target]=node;b.carried[target]=accumulated?estimate:carried;b.priority[target]=estimate;
 if(state==1){b.refresh(target);counters[1]++;}else{b.state[target]=1;b.add(target);
 counters[state==2?2:0]++;counters[3]=pf_max(counters[3],b.heapSize);}}
}
template<int CAP,class Lookup> CN_HD inline RouteSearchResult<CAP> route_search(int width,int height,
 Lookup lookup,int start,int goal,RouteSearchBuffers<CAP>& b,int method=1,int weight=5,
 bool accumulated=false,bool refresh=true,bool reopen=false,int budget=0){
 int cells=width*height;assert(width>0&&height>0&&cells<=CAP&&start>=0&&start<cells&&goal>=0&&goal<cells);
 b.prepare(cells);b.parent[start]=b.parent[goal]=-1;RouteSearchResult<CAP> out{};out.remainingBudget=budget;
 route_expand(b,lookup,width,height,start,goal%width,goal/width,method,weight,accumulated,refresh,reopen,out.counters);
 b.state[start]=2;if(!b.heapSize){out.route.add(start);return out;}
 while(b.heapSize){int n=b.pop();b.state[n]=2;
 route_expand(b,lookup,width,height,n,goal%width,goal/width,method,weight,accumulated,refresh,reopen,out.counters);
 if(out.remainingBudget>0&&--out.remainingBudget==0)break;if(b.state[goal]==2)break;}
 int n=goal;while(b.parent[n]!=-1){out.route.add(n);n=b.parent[n];assert(out.route.size()<=cells);}return out;
}
template<int CAP,class Lookup> CN_HD inline RouteSearchResult<CAP> grid_search(int width,int height,
 Lookup lookup,int sc,int sr,int gc,int gr,int adjust,RouteSearchBuffers<CAP>& b){
 if(sc<0||sr<0||sc>=width||sr>=height)return {};
 if(lookup(gc,gr)==-1&&(adjust&1)){
 int dx=sub32(gc,sc),dy=sub32(gr,sr),reach=int(isqrt64(uint32_t(add32(mul32(dx,dx),mul32(dy,dy)))));
 int lc=pf_max(pf_min(gc-reach,width),0),lr=pf_max(pf_min(gr-reach,height),0);
 int hc=pf_max(pf_min(gc+reach,width),0),hr=pf_max(pf_min(gr+reach,height),0);
 if(hc<=lc||hr<=lr)return {};int best=INT32_MAX,bc=-1,br=-1;
 for(int r=lr;r<hr;r++)for(int c=lc;c<hc;c++){int d=add32(mul32(c-gc,c-gc),mul32(r-gr,r-gr));
 if(lookup(c,r)!=-1&&d<best){best=d;bc=c;br=r;}}
 if(bc==-1)return {};gc=bc;gr=br;}
 if(lookup(gc,gr)==-1)return {};return route_search(width,height,lookup,sr*width+sc,gr*width+gc,b);
}
// Source-equivalent caller-storage variants. The original value-returning APIs above remain
// independent controls for differential testing of nodes, all counters, and remaining budget.
template<int CAP,int ROUTE_CAP,class Lookup> CN_HD inline RouteSearchStats route_search_into(int width,int height,
 Lookup lookup,int start,int goal,RouteSearchBuffers<CAP>&b,Route<ROUTE_CAP>&route,int method=1,int weight=5,
 bool accumulated=false,bool refresh=true,bool reopen=false,int budget=0){
 int cells=width*height;assert(width>0&&height>0&&cells<=CAP&&start>=0&&start<cells&&goal>=0&&goal<cells);
 route.clear();b.prepare(cells);b.parent[start]=b.parent[goal]=-1;RouteSearchStats out{};out.remainingBudget=budget;
 route_expand(b,lookup,width,height,start,goal%width,goal/width,method,weight,accumulated,refresh,reopen,out.counters);
 b.state[start]=2;if(!b.heapSize){route.add(start);return out;}
 while(b.heapSize){int n=b.pop();b.state[n]=2;
 route_expand(b,lookup,width,height,n,goal%width,goal/width,method,weight,accumulated,refresh,reopen,out.counters);
 if(out.remainingBudget>0&&--out.remainingBudget==0)break;if(b.state[goal]==2)break;}
 int n=goal;while(b.parent[n]!=-1){route.add(n);n=b.parent[n];assert(route.size()<=cells);}return out;
}
template<int CAP,int ROUTE_CAP,class Lookup> CN_HD inline RouteSearchStats grid_search_into(int width,int height,
 Lookup lookup,int sc,int sr,int gc,int gr,int adjust,RouteSearchBuffers<CAP>&b,Route<ROUTE_CAP>&route){
 route.clear();if(sc<0||sr<0||sc>=width||sr>=height)return {};
 if(lookup(gc,gr)==-1&&(adjust&1)){
 int dx=sub32(gc,sc),dy=sub32(gr,sr),reach=int(isqrt64(uint32_t(add32(mul32(dx,dx),mul32(dy,dy)))));
 int lc=pf_max(pf_min(gc-reach,width),0),lr=pf_max(pf_min(gr-reach,height),0);
 int hc=pf_max(pf_min(gc+reach,width),0),hr=pf_max(pf_min(gr+reach,height),0);
 if(hc<=lc||hr<=lr)return {};int best=INT32_MAX,bc=-1,br=-1;
 for(int r=lr;r<hr;r++)for(int c=lc;c<hc;c++){int d=add32(mul32(c-gc,c-gc),mul32(r-gr,r-gr));
 if(lookup(c,r)!=-1&&d<best){best=d;bc=c;br=r;}}
 if(bc==-1)return {};gc=bc;gr=br;}
 if(lookup(gc,gr)==-1)return {};return route_search_into(width,height,lookup,sr*width+sc,gr*width+gc,b,route);
}
template<class Distance> CN_HD inline int select_endpoint(const GridView& g,int ux,int uy,int tc,int tr,
 int range,bool avoidanceRuleFlag,Distance targetDistanceSquared){
 int extent=range/500+1,lc=pf_max(tc-extent,0),hc=pf_min(tc+extent,g.columns()-1),
 lr=pf_max(tr-extent,0),hr=pf_min(tr+extent,g.rows()-1),best=-1,rank=0,bestDistance=INT32_MAX;
 bool waterRank=!avoidanceRuleFlag,overlayRank=!avoidanceRuleFlag,leftToRight=ux<g.columns()*250;
 for(int r=lr;r<=hr;r++)for(int i=0;i<=hc-lc;i++){
 int c=leftToRight?lc+i:hc-i,x=c*500+250,y=r*500+250;
 int distance=add32(mul32(x-ux,x-ux),mul32(y-uy,y-uy));
 if(rank==2&&distance>=bestDistance)continue;
 if(targetDistanceSquared(x,y)>mul32(range,range))continue;
 int candidateRank=((waterRank&&g.water(c,r))||(overlayRank&&g.overlayBlocks(c,r)))?1:2;
 if(candidateRank>rank||(candidateRank==rank&&distance<bestDistance)){
 best=s32((uint32_t(c)<<16)|uint32_t(r));rank=candidateRank;bestDistance=distance;}}
 return bestDistance==INT32_MAX?-1:best;
}
CN_HD inline int lane_assignment(const GridView& g,int x,int y,int referenceX=-1,int flag=0){
 if(g.columns()<1)return 0;int col=div32(x,500),row=div32(y,500),shifted=div32(add32(x,5),500),
 rc=div32(referenceX,500),half=g.columns()/2,best=0,bd=INT32_MAX,ref=-1,rd=INT32_MAX;bool opposite=false;
 for(int c=0;c<g.columns();c++)for(int r=0;r<g.rows();r++){int road=g.bits(c,r)&3;if(!road)continue;
 int dy2=mul32(r-row,r-row),d=add32(dy2,mul32(c-col,c-col));
 if(referenceX==-1||(flag&1)){if(d<bd){best=road;bd=d;}continue;}
 int dref=add32(dy2,mul32(c-rc,c-rc));if(dref<rd){ref=road;rd=dref;}
 bool closer=d<=bd;if(d<bd){best=road;bd=d;}
 if(closer&&!opposite){if(referenceX>x)opposite=rc>half&&shifted<=half;
 else if(referenceX==x)opposite=false;else opposite=rc<half&&shifted>=half;}}
 if(opposite&&best==ref)return ref==1?2:ref==2?1:ref;return best;
}
// Returns false for a footprint that does not fit; caller must still hash that occluder.
CN_HD inline bool rasterize_footprint(GridView&g,int x,int y,int hw,int hh,int cost,int&packed){
 int ax=add32(mul32(div32(sub32(x,1),500),500),500),ay=add32(mul32(div32(sub32(y,1),500),500),500);
 int lx=sub32(ax,hw),ly=sub32(ay,hh),hx=add32(ax,hw),hy=add32(ay,hh);
 if(lx<0||ly<0||hx>=g.columns()*500||hy>=g.rows()*500)return false;
 int fc=lx/500,lc=div32(hx-1,500),fr=ly/500,lr=div32(hy-1,500);
 packed=s32((uint32_t(lc)<<16)|(uint32_t(fc)<<24)|(uint32_t(fr)<<8)|uint32_t(lr));
 for(int r=fr;r<=lr;r++)for(int c=fc;c<=lc;c++){int&i=g.current[r*g.columns()+c];if(cost>i)i=cost;}return true;
}
template<class Q> CN_HD inline void build_overlay(GridView&g,Q&q){int hashes[2]{};
 if(g.dynamicEnabled){g.active=1;for(int i=0;i<q.entityCount();i++){const PfEntity&e=q.entity(i);
 if(e.type!=5||!e.occludesCells())continue;int packed;
 rasterize_footprint(g,e.x,e.y,e.collisionRadius,e.collisionRadius,g.buildingCost,packed);
 if(uint32_t(e.side)<=1){uint32_t h=uint32_t(hashes[e.side]);h=(h>>31)|(h<<1);
 hashes[e.side]=add32(add32(s32(h),e.id),9);}}}
 for(int i=0;i<2;i++){g.changed[i]=hashes[i]!=g.hash[i];g.hash[i]=hashes[i];}
}
CN_HD inline int relocate(const GridView&g,int x,int y,int referenceY=-1){
 int cx=pf_min(pf_max(x,250),g.columns()*500-250),cy=pf_min(pf_max(y,250),g.rows()*500-250);
 if(!g.water(cx/500,cy/500))return s32(uint32_t(cx)|(uint32_t(cy)<<16));
 int first=referenceY==-1?-5:cy<referenceY?1:-5,last=referenceY==-1?5:cy<referenceY?5:cy>referenceY?0:5;
 int best=INT32_MAX,bx=cx,by=cy;
 for(int ro=first;ro<=last;ro++){int yy=ro*500+cy+250;if(yy<0)continue;
 for(int dx=-5500;dx<0;dx+=500){int xx=cx+dx+3250;if(xx<0||xx>=g.columns()*500||yy>=g.rows()*500)continue;
 if(g.water(xx/500,yy/500))continue;int d=approximate_distance(dx+3250,yy-cy);
 if(d<best){best=d;bx=xx;by=yy;}}}return s32(uint32_t(bx)|(uint32_t(by)<<16));
}
}
