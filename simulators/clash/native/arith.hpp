// SPDX-License-Identifier: Apache-2.0
// Ported from crforge core/src/main/java/org/crforge/core/pathfinding/math/FixedMath.java
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de
// https://github.com/voonhous/crforge ; see LICENSE and NOTICE.
#pragma once
#include <cstdint>
#include <limits>
#include <algorithm>
#include <bit>
#include <array>
namespace clash_native {
#ifdef __CUDACC__
#define CN_HD __host__ __device__
#define CN_NOINLINE __noinline__
#else
#define CN_HD
#define CN_NOINLINE
#endif
CN_HD inline int32_t s32(uint64_t x) { return static_cast<int32_t>(static_cast<uint32_t>(x)); }
CN_HD inline int32_t add32(int32_t a,int32_t b) { return s32(uint64_t(uint32_t(a))+uint32_t(b)); }
CN_HD inline int32_t sub32(int32_t a,int32_t b) { return s32(uint64_t(uint32_t(a))-uint32_t(b)); }
CN_HD inline int32_t mul32(int32_t a,int32_t b) { return s32(uint64_t(uint32_t(a))*uint32_t(b)); }
CN_HD inline int32_t sar32(int32_t a,unsigned bits) {
    bits&=31; if (!bits) return a;
    uint32_t u=uint32_t(a)>>bits;
    if (a<0) u|=(~uint32_t(0))<<(32-bits);
    return int32_t(u);
}
CN_HD inline int32_t div32(int32_t a,int32_t b) {
    // Java's MIN_VALUE / -1 wraps instead of invoking C++ signed-overflow UB.
    if (a==INT32_MIN && b==-1) return INT32_MIN;
    return a/b;
}
CN_HD inline int32_t div_or_zero(int32_t a,int32_t b) { return b==0?0:div32(a,b); }
CN_HD inline int32_t abs32(int32_t a) { return a<0?sub32(0,a):a; }
CN_HD inline int32_t floor_mod(int32_t a,int32_t b) { int32_t r=a%b;return r<0?r+b:r; }
CN_HD inline uint64_t isqrt64(uint64_t n) {
    if(n<2)return n;
    // Restoring binary square root: bounded 32 iterations, exact on CPU/CUDA.
    uint64_t root=0,bit=uint64_t(1)<<62;
    while(bit>n)bit>>=2;
    while(bit) { if(n>=root+bit){n-=root+bit;root=(root>>1)+bit;}else root>>=1;bit>>=2; }
    return root;
}
CN_HD inline int32_t squared(int32_t dx,int32_t dy) {
    if(dx < -46340 || dx>46340 || dy < -46340 || dy>46340)return INT32_MAX;
    int32_t a=dx*dx,b=dy*dy;
    return b<INT32_MAX-a?a+b:INT32_MAX;
}
CN_HD inline int32_t squared(int32_t dx,int32_t dy,int32_t dz) {
    if(dz < -46340 || dz>46340)return INT32_MAX;
    int32_t xy=squared(dx,dy);if(xy==INT32_MAX)return xy;
    int32_t z=dz*dz;return z<INT32_MAX-xy?xy+z:INT32_MAX;
}
CN_HD inline int32_t distance(int32_t dx,int32_t dy) { int32_t d=squared(dx,dy);return d==INT32_MAX?65535:int32_t(isqrt64(uint32_t(d))); }
CN_HD inline int32_t squared_distance(int32_t px,int32_t py,int32_t x,int32_t y) { return squared(sub32(x,px),sub32(y,py)); }
CN_HD inline int32_t normalize(int32_t &x,int32_t &y,int32_t scale) {
    int32_t d=distance(x,y);if(d){x=div32(mul32(x,scale),d);y=div32(mul32(y,scale),d);}return d;
}
CN_HD inline int32_t approximate_distance(int32_t dx,int32_t dy) {
    int32_t a=abs32(dx),b=abs32(dy),low=a<b?a:b,high=a>b?a:b;
    return add32(high,sar32(mul32(low,53),7));
}
}
