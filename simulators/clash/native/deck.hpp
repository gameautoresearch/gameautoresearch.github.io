// SPDX-License-Identifier: Apache-2.0
// Ported from crforge core/src/main/java/org/crforge/core/battle/BattleRandom.java,
// core/src/main/java/org/crforge/core/battle/match/DeckShuffle.java and Hand.java
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge).
#pragma once
#include "arith.hpp"
namespace clash_native {
struct BattleRandom {
    int32_t state;
    CN_HD int32_t next(int32_t range) {
        if(range<1)return 0;
        int32_t s=state==0?-1:state;
        s=int32_t(uint32_t(s)^(uint32_t(s)<<13));s=int32_t(uint32_t(s)^uint32_t(sar32(s,17)));s=int32_t(uint32_t(s)^(uint32_t(s)<<5));
        state=s;uint32_t magnitude=s<0?uint32_t(0)-uint32_t(s):uint32_t(s);
        return int32_t(magnitude%uint32_t(range));
    }
};
struct DeckShuffle {
    int32_t words[624];int index=0;
    CN_HD explicit DeckShuffle(int32_t seed) {
        words[0]=seed;
        for(int i=1;i<624;++i){int32_t x=words[i-1]^sar32(words[i-1],30);words[i]=add32(mul32(x,0x6c078965),0x6c078965);}
    }
    CN_HD int next(int bound) {
        if(bound<1)return 0;
        if(index==0)for(int i=0;i<624;++i){
            int32_t n=words[i==623?0:i+1],far=words[i<227?i+397:i-227];
            int32_t y=int32_t((uint32_t(words[i])&0x80000000u)|(uint32_t(n)&0x7ffffffeu));
            y=far^sar32(y,1);words[i]=(n&1)?int32_t(uint32_t(y)^0x9908b0dfu):y;
        }
        int32_t y=words[index];y^=sar32(y,11);y=int32_t(uint32_t(y)^((uint32_t(y)<<7)&0x9d2c5680u));y=int32_t(uint32_t(y)^((uint32_t(y)<<15)&0xefc60000u));y^=sar32(y,18);
        uint32_t result=uint32_t(y)&0x7fffffffu;index=(index+1)%624;return int(result%uint32_t(bound));
    }
    CN_HD void fixed_eight(int32_t output[8]) {
        // None of the frozen plain eight cards forces/omits the opening hand.
        int rest[8]={0,1,2,3,4,5,6,7},left=8;
        for(int i=0;i<8;++i){int k=next(left);output[i]=rest[k];for(int j=k;j+1<left;++j)rest[j]=rest[j+1];--left;}
    }
};
struct Hand {
    int32_t slots[4]={-1,-1,-1,-1},queue[8]={},count=0,cooldown_ms=0;
    CN_HD void deal(const int32_t order[8]) { for(int k=0;k<4;++k)slots[k]=order[k];for(int k=0;k<4;++k)queue[k]=order[k+4];count=4;cooldown_ms=0; }
    CN_HD int remove(int slot) {
        if(slot<0||slot>=4||slots[slot]<0)return -1;
        int card=slots[slot];queue[count++]=card;slots[slot]=-1;return card;
    }
    CN_HD int refill(int next_cooldown_ms) {
        int left=cooldown_ms-50;cooldown_ms=left>0?left:0;if(left>0)return -1;
        for(int k=0;k<4;++k)if(slots[k]<0){if(count){slots[k]=queue[0];for(int j=0;j+1<count;++j)queue[j]=queue[j+1];--count;}cooldown_ms=next_cooldown_ms;return k;}return -1;
    }
};
}
