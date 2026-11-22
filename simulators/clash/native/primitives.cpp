// SPDX-License-Identifier: Apache-2.0; source provenance is in the included headers.
#include "deck.hpp"
extern "C" {
void cn_random(int32_t seed,const int32_t* bounds,int count,int32_t* draws,int32_t* states) {
    clash_native::BattleRandom random{seed};
    for(int i=0;i<count;++i){draws[i]=random.next(bounds[i]);states[i]=random.state;}
}
void cn_deck(int32_t seed,int32_t* order) { clash_native::DeckShuffle shuffle(seed);shuffle.fixed_eight(order); }
void cn_arithmetic(const int32_t* xy,int count,int32_t* out) {
    for(int i=0;i<count;++i){int32_t x=xy[2*i],y=xy[2*i+1];out[4*i]=clash_native::squared(x,y);out[4*i+1]=clash_native::distance(x,y);out[4*i+2]=clash_native::approximate_distance(x,y);out[4*i+3]=clash_native::div_or_zero(x,y);}
}
}
