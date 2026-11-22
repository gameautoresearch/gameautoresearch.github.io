// SPDX-License-Identifier: Apache-2.0
// Ported from crforge core/src/main/java/org/crforge/core/pathfinding/math/{FixedMath,TrigTables}.java
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge)
#pragma once
#include "types.hpp"
namespace clash_native {
// Local constexpr tables are CUDA callable without host global-memory access.
CN_HD inline int pf_sine_quarter(int index){constexpr int table[]={
    0, 18, 36, 54, 71, 89, 107, 125, 143, 160, 178, 195, 213, 230, 248, 265, 282, 299, 316, 333,
    350, 367, 384, 400, 416, 433, 449, 465, 481, 496, 512, 527, 543, 558, 573, 587, 602, 616, 630,
    644, 658, 672, 685, 698, 711, 724, 737, 749, 761, 773, 784, 796, 807, 818, 828, 839, 849, 859,
    868, 878, 887, 896, 904, 912, 920, 928, 935, 943, 949, 956, 962, 968, 974, 979, 984, 989, 994,
    998, 1002, 1005, 1008, 1011, 1014, 1016, 1018, 1020, 1022, 1023, 1023, 1024, 1024
  };return table[index];}
CN_HD inline int pf_atan_ratio(int numerator,int denominator){constexpr int table[]={
    0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 11, 12, 12,
    13, 13, 14, 14, 14, 15, 15, 16, 16, 17, 17, 17, 18, 18, 19, 19, 19, 20, 20, 21, 21, 21, 22, 22,
    22, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31,
    31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 37, 37, 37, 37, 38, 38,
    38, 39, 39, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 43, 44, 44, 44,
    44, 45, 45, 45
  };
 int i=div32(s32(uint32_t(numerator)<<7),denominator);assert(i>=0&&i<=128);return table[i];}
CN_HD inline int sine1024(int degrees){int a=floor_mod(degrees,360),h=a>=180?a-180:a;
 int v=pf_sine_quarter(pf_min(h,180-h));return a>=180?-v:v;}
CN_HD inline void rotate1024(int&x,int&y,int degrees){int c=sine1024(add32(degrees,90)),s=sine1024(degrees);
 int nx=sub32(mul32(c,x),mul32(s,y)),ny=add32(mul32(s,x),mul32(c,y));
 x=sar32(add32(nx,nx<0?1023:0),10);y=sar32(add32(ny,ny<0?1023:0),10);}
CN_HD inline int angle_of_vector(int x,int y){
 if((x|y)==0)return 0;
 if(x>=1&&y>=0)return y<x?pf_atan_ratio(y,x):90-pf_atan_ratio(x,y);
 int ax=abs32(x);
 if(x<=0&&y>=1)return ax<y?pf_atan_ratio(ax,y)+90:180-pf_atan_ratio(y,ax);
 int ay=abs32(y);
 if(x<0&&y<=0){if(ay<ax)return pf_atan_ratio(ay,ax)+180;return y==0?0:270-pf_atan_ratio(ax,ay);}
 if(ax<ay)return pf_atan_ratio(ax,ay)+270;
 if(x==0)return 0;
 int wrapped=360-pf_atan_ratio(ay,ax);if(wrapped==360)return 0;return wrapped<0?wrapped+360:wrapped;
}
}
