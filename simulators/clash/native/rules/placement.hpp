// SPDX-License-Identifier: Apache-2.0
// Ported from crforge core/src/main/java/org/crforge/core/battle/deploy/DeployMask.java,
// PlacementSearch.java, MapCheck.java, CardPlacement.java, InitialDelay.java, Formation.java,
// and core/src/main/java/org/crforge/core/pathfinding/grid/LaneAssignment.java
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge).
// Native/LICENSE and native/NOTICE apply. Plain fixed8 only: max formation count is two.
#pragma once
#include "../arith.hpp"
namespace clash_native {
struct ArenaView {
    int width=0,height=0;const int32_t* cells=nullptr;
    CN_HD int bits(int col,int row)const{return col>=0&&row>=0&&col<width&&row<height?(cells[row*width+col]&127):0x50;}
};
struct CardRule {
    int cost=0,productionStopMs=0,unitType=-1,count=0;
    int summonRadius=0,summonWidth=0,summonDelayMs=0,deployTimeMs=0;
    int collisionRadius=0,spawnRadius=0,spawnAngleShift=0,flyingHeight=0,spawnPathfindSpeed=0,tileSizeOverride=0;
    int margin=0,startY=0,endY=0;
    bool enemyOk=false,onBuildings=false,onWater=false,fullLane=false,building=false,spellAsDeploy=false;
    CN_HD bool spell()const{return unitType<0;}
};
struct MaskEntity {bool alive=false;int team=0,x=0,y=0;bool summonerTower=false,summoner=false,building=false;int noDeploySizeW=0,noDeploySizeH=0,footprint=0;};
struct Point {int x=0,y=0;bool found=false;};
struct PlacementUnit {int index=0,unitType=-1,dx=0,dy=0,x=0,y=0,lane=0,state=-1,waitMs=-1,deployMs=0;};
struct Placement {int code=0,x=0,y=0,originLane=0,count=0,intervalLo=0,intervalHi=0;bool hasInterval=false;PlacementUnit units[2]{};};
constexpr int RULES_MAX_TILES=18*32;
CN_HD inline int placement_footprint(const CardRule& c){return c.tileSizeOverride>0?c.tileSizeOverride:(c.collisionRadius+499)/500+1;}
CN_HD inline void mask_box(uint8_t* grid,int stride,int rows,int x,int y,int w,int h){
    if(w==0)return;int r0=(y-500*h)/1000,r1=(y+500*h)/1000,c0=(x-500*w)/1000,c1=(x+500*w)/1000;
    for(int r=r0;r<r1;++r)if(r>=0&&r<rows)for(int col=c0;col<c1;++col)if(col>=0&&col<stride)grid[r*stride+col]=0;
}
CN_HD inline void deploy_mask(const ArenaView& a,const CardRule& c,int side,const MaskEntity* entities,int n,uint8_t* out){
    int stride=a.width>>1,rows=a.height>>1;for(int i=0;i<stride*rows;++i)out[i]=1;
    if(c.endY!=c.startY){for(int row=0;row<rows;++row)if((c.startY>=1&&row<c.startY)||(c.endY!=0&&row>=c.endY))for(int col=0;col<stride;++col)out[row*stride+col]=0;}
    for(int i=0;i<n;++i){const auto&e=entities[i];if(!e.alive||(!e.building&&!e.summoner))continue;
        bool same=c.startY==c.endY,enemy=((e.team==0)!=((side&1)==0));bool noDeploy=false;
        if(c.onBuildings){bool toBox=e.summoner?(same&&enemy):(same&&enemy&&e.summonerTower);
            if(toBox&&!c.enemyOk)noDeploy=true;else if(e.building||e.noDeploySizeW>0)continue;
        }
        if(same&&enemy&&e.noDeploySizeW>=1&&!c.enemyOk)noDeploy=true;
        mask_box(out,stride,rows,e.x,e.y,noDeploy?e.noDeploySizeW:e.footprint,noDeploy?e.noDeploySizeH:e.footprint);
    }
    if(c.fullLane)for(int row=0;row<rows;++row){bool whole=true;for(int col=0;col<stride;++col)whole=whole&&out[row*stride+col];if(!whole)for(int col=0;col<stride;++col)out[row*stride+col]=0;}
}
CN_HD inline bool cell_ok(const ArenaView& a,const CardRule& c,int col,int row){int bits=a.bits(col,row);return (bits&0x20)?c.onWater:((bits&0x50)==0||c.onBuildings);}
CN_HD inline int map_check(const ArenaView& a,const CardRule& c,int x,int y){
    if(c.spell())return 0;if(x< -499)return 15;int row=y/500;if(row<0)return 16;int col=x/500;if(col>=a.width)return 17;if(row>=a.height)return 18;return cell_ok(a,c,col,row)?0:19;
}
CN_HD inline int placement_snap(const CardRule& c,int v){int t=v/1000*1000;return !c.spell()&&!c.spellAsDeploy&&c.building&&!(placement_footprint(c)&1)?t:t+500;}
CN_HD inline bool placement_legal(const ArenaView& a,const CardRule& c,const uint8_t* mask,int px,int py){
    int hw=a.width>>1,hh=a.height>>1,fp=!c.spell()&&!c.spellAsDeploy&&c.building?placement_footprint(c):1,half=500*fp;
    int xlo=(px-half)/1000,xhi=(px+half)/1000,ylo=(py-half)/1000,yhi=(py+half)/1000;
    if(xlo>=xhi)return true;int available=(hh>ylo?hh:ylo)-ylo;
    for(int col=xlo;col<xhi;++col){if(ylo>=yhi)continue;if(col<c.margin||col>=hw-c.margin||py-half< -999)return false;
        int left=available;for(int row=ylo;row<yhi;++row){if(left==0||row<0||row>=hh)return false;if(!mask[row*hw+col])return false;
            for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)if(!cell_ok(a,c,2*col+dx,2*row+dy))return false;--left;}}
    return true;
}
CN_HD inline void landing_tiles(const ArenaView& a,const CardRule& c,const uint8_t* mask,uint8_t* out){
    int hw=a.width>>1,hh=a.height>>1;for(int row=0;row<hh;++row)for(int col=0;col<hw;++col)out[row*hw+col]=placement_legal(a,c,mask,placement_snap(c,col*1000+500),placement_snap(c,row*1000+500));
}
CN_HD inline Point placement_find(const ArenaView& a,const CardRule& c,int x,int y,int side,const uint8_t* mask,bool symmetricalSnap){
    int cx=x>0?(x<a.width*500?x:a.width*500-1):0,cy=y>0?(y<a.height*500?y:a.height*500-1):0;
    int sx=placement_snap(c,cx),sy=placement_snap(c,cy),best=INT32_MAX;Point out{};
    for(int r=0;r<=30;++r){int count=r==0?1:2*r;for(int j=0;j<count;++j)for(int k=0;k<(r==0?1:4);++k){int dx,dy;
        if(!(k&1)){dx=j-r;dy=-r;}else{dx=-r;dy=r-j;}if(k>1){dx=-dx;dy=-dy;}
        int px=sx+dx*1000,py=sy+dy*1000;if(!placement_legal(a,c,mask,px,py))continue;int d=squared_distance(px,py,cx,cy);if(d<best){best=d;out={px,py,true};}}
        if(out.found)break;
    }
    if(out.found&&symmetricalSnap&&!c.spell()&&c.flyingHeight<=0&&!c.building&&c.spawnPathfindSpeed==0){out.x-=cx<a.width*250?1:0;out.y-=side&1;}return out;
}
// Fixed8 formation counts are one and two, all zero angle shift and width. Fail
// closed for a table outside that pinned task rather than invent extra mechanics.
CN_HD inline int fixed8_formation(const CardRule& c,int k,int side,int lane,bool laneSequence,int& dx,int& dy){
    if(c.count==1){dx=dy=0;return 0;}if(c.count!=2||c.summonWidth!=0||c.spawnAngleShift!=0)return -1;
    int radius=c.summonRadius?c.summonRadius:(c.spawnRadius?c.spawnRadius:c.collisionRadius);
    dx=k==0?-radius:radius;dy=0;if(laneSequence&&lane==1)dx=-dx;(void)side;return 0;
}
CN_HD inline int placement_lane(const ArenaView& a,int x,int y,int referenceX=-1,int flag=0){
    if(a.width<1)return 0;int col=x/500,row=y/500,shifted=(x+5)/500,ref=referenceX/500,half=a.width/2;
    int nearest=0,best=INT32_MAX,nearRef=-1,bestRef=INT32_MAX;bool opposite=false;
    for(int sc=0;sc<a.width;++sc)for(int sr=0;sr<a.height;++sr){int road=a.bits(sc,sr)&3;if(road<1)continue;
        int dy=sr-row,dist=dy*dy+(sc-col)*(sc-col),rd=dy*dy+(sc-ref)*(sc-ref);bool closer=dist<=best;
        if(dist<best){best=dist;nearest=road;}if(rd<bestRef){bestRef=rd;nearRef=road;}
        if(closer&&!opposite&&referenceX!=-1&&!(flag&1))opposite=referenceX>x?(ref>half&&shifted<=half):referenceX<x?(ref<half&&shifted>=half):false;
    }
    if(opposite&&nearest==nearRef){if(nearRef==1)return 2;if(nearRef==2)return 1;}return nearest;
}
CN_HD inline Placement place_card(const ArenaView& a,const CardRule& c,int x,int y,int side,const MaskEntity* entities,int n,bool symmetry,bool laneSequence){
    Placement out{};out.code=map_check(a,c,x,y);if(out.code)return out;uint8_t mask[RULES_MAX_TILES];deploy_mask(a,c,side,entities,n,mask);
    Point point=placement_find(a,c,x,y,side,mask,symmetry);if(!point.found){out.code=23;return out;}out.x=point.x;out.y=point.y;if(c.spell())return out;
    CardRule off=c;off.onBuildings=false;off.startY=off.endY=0;deploy_mask(a,off,side,entities,n,mask);
    int col=(point.x/500)>>1,lo=INT32_MAX,hi=INT32_MIN,stride=a.width>>1,rows=a.height>>1;bool found=false,odd=side&1;
    for(int row=0;row<rows;++row)if(mask[row*stride+col]){int l=row*1000+(odd?500:0),h=row*1000+(odd?0:500);if(l<lo)lo=l;if(h>hi)hi=h;found=true;}
    if(odd)--lo;out.hasInterval=found&&hi-lo<a.height*250;out.intervalLo=lo;out.intervalHi=hi;
    out.originLane=placement_lane(a,point.x,point.y);if(c.count>2||c.count<1){out.code=-1;return out;}
    for(int k=0;k<c.count;++k){auto&u=out.units[k];u.index=k;u.unitType=c.unitType;
        if(fixed8_formation(c,k,side,out.originLane,laneSequence,u.dx,u.dy)){out.code=-1;return out;}
        int ux=point.x+u.dx,uy=point.y+u.dy;if(out.hasInterval&&c.flyingHeight<=0)uy=uy>lo?(uy<hi?uy:hi):lo;
        u.x=ux<250?250:(ux>a.width*500-250?a.width*500-250:ux);u.y=uy<250?250:(uy>a.height*500-250?a.height*500-250:uy);
        u.lane=placement_lane(a,u.x,u.y,point.x,c.fullLane?1:0);if(u.lane<=0)u.lane=out.originLane;u.deployMs=c.deployTimeMs;
        if(c.deployTimeMs>=1){if(c.summonDelayMs>=1&&(c.building||k!=0)){u.state=11;u.waitMs=c.summonDelayMs*(c.building?1:k);}else{u.state=4;u.waitMs=-1;}}
    }
    out.count=c.count;return out;
}
}
