// SPDX-License-Identifier: Apache-2.0
// Isolated CPU/CUDA diagnostics of the native port of crforge
// core/src/main/java/org/crforge/core/battle/BattleRandom.java,
// core/src/main/java/org/crforge/core/battle/match/{DeckShuffle,Hand,LadderMatch}.java
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge).
// This is a diagnostic only: no upstream source, frozen data, or production state is modified.
#include "rules/match.hpp"
#include <cstddef>
#include <cstdint>
#ifndef __CUDACC__
#include "world.hpp" // layout metadata only; no NativeWorld methods are called or linked.
#endif
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define RNG_NOINLINE __noinline__
#else
#if defined(__GNUC__) || defined(__clang__)
#define RNG_NOINLINE __attribute__((noinline))
#else
#define RNG_NOINLINE
#endif
#endif
using namespace clash_native;
constexpr int RNG_DIAG_STRIDE=192;

CN_HD inline void diag_words(const DeckShuffle&deck,int*out){
 constexpr int samples[8]={0,1,2,226,227,396,397,623};
 for(int i=0;i<8;i++)out[i]=deck.words[samples[i]];
}
CN_HD RNG_NOINLINE void diag_one_deck_noinline(int seed,int*out){
 DeckShuffle deck(seed);out[0]=deck.index;diag_words(deck,out+1);
 deck.fixed_eight(out+9);out[17]=deck.index;diag_words(deck,out+18);
}
CN_HD inline void diag_one_deck_inline(int seed,int*out){
 DeckShuffle deck(seed);out[0]=deck.index;diag_words(deck,out+1);
 deck.fixed_eight(out+9);out[17]=deck.index;diag_words(deck,out+18);
}
// Modes 0..5 share initialization and identical complete output layout.
// 0: two DeckShuffle constructors in one loop (the suspected Match.init shape).
// 1: unrolled lexical constructor scopes; 2: an explicit noinline helper per side.
// 3: reuse by assignment from a fresh constructor; 4: inline helper in a loop.
// 5: explicit index=0 after construction (diagnostic control, not a mechanics change).
CN_HD void rng_deck_probe_one(const int*input,int*out,int mode){
 for(int i=0;i<RNG_DIAG_STRIDE;i++)out[i]=-7777777;
 int seed=input[0],warm=input[3];BattleRandom random{seed};
 out[0]=seed;out[1]=input[1];out[2]=input[2];out[3]=warm;
 for(int i=0;i<warm;i++)random.next(1);out[4]=random.state;
 BattleRandom draws=random;
 for(int i=0;i<16;i++){out[16+2*i]=draws.next(0x0fffffff);out[17+2*i]=draws.state;}
 int draw0=random.next(0x0fffffff),state0=random.state,draw1=random.next(0x0fffffff);
 int seeds[2]={add32(draw0,input[1]),add32(draw1,input[2])};
 out[5]=draw0;out[6]=state0;out[7]=draw1;out[8]=random.state;out[9]=seeds[0];out[10]=seeds[1];out[11]=mode;
 if(mode==0||mode==5){for(int side=0;side<2;side++){DeckShuffle deck(seeds[side]);
 if(mode==5)deck.index=0;
 int*o=out+64+side*32;o[0]=deck.index;diag_words(deck,o+1);deck.fixed_eight(o+9);o[17]=deck.index;diag_words(deck,o+18);}}
 else if(mode==1){
 {DeckShuffle first(seeds[0]);int*o=out+64;o[0]=first.index;diag_words(first,o+1);first.fixed_eight(o+9);o[17]=first.index;diag_words(first,o+18);}
 {DeckShuffle second(seeds[1]);int*o=out+96;o[0]=second.index;diag_words(second,o+1);second.fixed_eight(o+9);o[17]=second.index;diag_words(second,o+18);}}
 else if(mode==2){diag_one_deck_noinline(seeds[0],out+64);diag_one_deck_noinline(seeds[1],out+96);}
 else if(mode==3){DeckShuffle deck(seeds[0]);for(int side=0;side<2;side++){
 if(side)deck=DeckShuffle(seeds[side]);int*o=out+64+side*32;o[0]=deck.index;diag_words(deck,o+1);
 deck.fixed_eight(o+9);o[17]=deck.index;diag_words(deck,o+18);}}
 else if(mode==4){for(int side=0;side<2;side++)diag_one_deck_inline(seeds[side],out+64+side*32);}
}
CN_HD void rng_match_probe_one(const int*input,int*out,int mode){
 for(int i=0;i<RNG_DIAG_STRIDE;i++)out[i]=-7777777;
 RulesConfig config{};config.startingElixir=5;config.maxMana=10;config.startCooldownTicks=20;
 config.endDelayMs=3000;config.sectionCount=1;config.sectionLengths[0]=180000;
 config.rateCount=1;config.fullBarMs[0]=28000;config.cooldownCount=1;config.cooldownMs[0]=1000;
 BattleRandom random{input[0]};for(int i=0;i<input[3];i++)random.next(1);
 int words[2]={input[1],input[2]};LadderMatch match{};
 if(mode==0)match.init(config,random,words);
 else if(mode==1){int playerWords[2]={0,0};match.init(config,random,playerWords);}
 else if(mode==2){match=LadderMatch{};match.config=config;
 // Source-equivalent unrolled constructor control, useful to distinguish loop-local lifetime.
 match.shuffleDraws[0]=random.next(0x0fffffff);DeckShuffle a(add32(match.shuffleDraws[0],words[0]));int order0[8];a.fixed_eight(order0);match.sides[0].hand.deal(order0);match.sides[0].elixir=config.startingElixir*10000;
 match.shuffleDraws[1]=random.next(0x0fffffff);DeckShuffle b(add32(match.shuffleDraws[1],words[1]));int order1[8];b.fixed_eight(order1);match.sides[1].hand.deal(order1);match.sides[1].elixir=config.startingElixir*10000;}
 out[0]=input[0];out[1]=words[0];out[2]=words[1];out[3]=input[3];out[4]=random.state;
 out[5]=match.shuffleDraws[0];out[6]=match.shuffleDraws[1];out[7]=match.tick;out[8]=match.clockMs;
 out[9]=match.config.startingElixir;out[10]=int(sizeof(LadderMatch));out[11]=int(sizeof(RulesConfig));
 for(int side=0;side<2;side++){int*o=out+64+side*32;const MatchSide&s=match.sides[side];
 for(int i=0;i<4;i++)o[i]=s.hand.slots[i];for(int i=0;i<8;i++)o[4+i]=s.hand.queue[i];
 o[12]=s.hand.count;o[13]=s.hand.cooldown_ms;o[14]=s.elixir;o[15]=s.productionStopMs;o[16]=s.lastPlayed;o[17]=s.lastPlayedCopy;}
}
extern "C" {
int rng_diag_stride(){return RNG_DIAG_STRIDE;}
void rng_cpu_decks(const int*inputs,int*out,int count,int mode){for(int i=0;i<count;i++)rng_deck_probe_one(inputs+4*i,out+RNG_DIAG_STRIDE*i,mode);}
void rng_cpu_match(const int*inputs,int*out,int count,int mode){for(int i=0;i<count;i++)rng_match_probe_one(inputs+4*i,out+RNG_DIAG_STRIDE*i,mode);}
#ifndef __CUDACC__
void rng_cpu_world_layout(uint64_t*out){out[0]=sizeof(NativeWorld);out[1]=offsetof(NativeWorld,match);out[2]=offsetof(NativeWorld,random);
 out[3]=sizeof(LadderMatch);out[4]=offsetof(LadderMatch,shuffleDraws);out[5]=offsetof(LadderMatch,sides);out[6]=sizeof(MatchSide);out[7]=sizeof(Hand);}
#endif
}
#ifdef __CUDACC__
__global__ void rng_decks_kernel(const int*inputs,int*out,int count,int mode){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)rng_deck_probe_one(inputs+4*i,out+RNG_DIAG_STRIDE*i,mode);}
__global__ void rng_match_kernel(const int*inputs,int*out,int count,int mode){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)rng_match_probe_one(inputs+4*i,out+RNG_DIAG_STRIDE*i,mode);}
// Initialize the actual GLOBAL-memory offsets of a NativeWorld match/random member without
// compiling NativeWorld::init or linking its transition runtime. This distinguishes local Match
// lifetimes from member storage while leaving the owning cached world snapshot untouched.
__global__ void rng_match_member_kernel(const int*inputs,uint8_t*states,uint64_t stride,uint64_t matchOffset,
 uint64_t randomOffset,int*out,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;const int*input=inputs+4*i;int*o=out+RNG_DIAG_STRIDE*i;
 for(int k=0;k<RNG_DIAG_STRIDE;k++)o[k]=-7777777;
 auto&m=*reinterpret_cast<LadderMatch*>(states+stride*i+matchOffset);
 auto&r=*reinterpret_cast<BattleRandom*>(states+stride*i+randomOffset);r=BattleRandom{input[0]};
 for(int j=0;j<input[3];j++)r.next(1);
 RulesConfig c{};c.startingElixir=5;c.maxMana=10;c.startCooldownTicks=20;c.endDelayMs=3000;
 c.sectionCount=1;c.sectionLengths[0]=180000;c.rateCount=1;c.fullBarMs[0]=28000;c.cooldownCount=1;c.cooldownMs[0]=1000;
 int words[2]={input[1],input[2]};m.init(c,r,words);
 o[0]=input[0];o[1]=words[0];o[2]=words[1];o[3]=input[3];o[4]=r.state;
 o[5]=m.shuffleDraws[0];o[6]=m.shuffleDraws[1];o[7]=m.tick;o[8]=m.clockMs;o[9]=m.config.startingElixir;
 o[10]=int(sizeof(LadderMatch));o[11]=int(sizeof(RulesConfig));
 for(int side=0;side<2;side++){int*v=o+64+side*32;const MatchSide&s=m.sides[side];
 for(int j=0;j<4;j++)v[j]=s.hand.slots[j];for(int j=0;j<8;j++)v[4+j]=s.hand.queue[j];
 v[12]=s.hand.count;v[13]=s.hand.cooldown_ms;v[14]=s.elixir;v[15]=s.productionStopMs;v[16]=s.lastPlayed;v[17]=s.lastPlayedCopy;}
}
__global__ void rng_metadata_kernel(const uint8_t*states,uint64_t stride,uint64_t matchOffset,uint64_t randomOffset,int*out,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;const uint8_t*raw=states+stride*i;
 const auto&m=*reinterpret_cast<const LadderMatch*>(raw+matchOffset);const auto&r=*reinterpret_cast<const BattleRandom*>(raw+randomOffset);
 int*o=out+32*i;for(int k=0;k<32;k++)o[k]=-7777777;o[0]=r.state;o[1]=m.shuffleDraws[0];o[2]=m.shuffleDraws[1];o[3]=m.tick;
 for(int side=0;side<2;side++){for(int j=0;j<4;j++)o[4+side*12+j]=m.sides[side].hand.slots[j];
 for(int j=0;j<8;j++)o[8+side*12+j]=m.sides[side].hand.queue[j];}}
extern "C" {
int rng_cuda_decks(const int*in,int*out,int n,int mode,void*stream){rng_decks_kernel<<<(n+31)/32,32,0,static_cast<cudaStream_t>(stream)>>>(in,out,n,mode);return int(cudaGetLastError());}
int rng_cuda_match(const int*in,int*out,int n,int mode,void*stream){rng_match_kernel<<<(n+31)/32,32,0,static_cast<cudaStream_t>(stream)>>>(in,out,n,mode);return int(cudaGetLastError());}
int rng_cuda_match_member(const int*in,void*states,uint64_t stride,uint64_t mo,uint64_t ro,int*out,int n,void*stream){rng_match_member_kernel<<<(n+31)/32,32,0,static_cast<cudaStream_t>(stream)>>>(in,static_cast<uint8_t*>(states),stride,mo,ro,out,n);return int(cudaGetLastError());}
int rng_cuda_metadata(const void*states,uint64_t stride,uint64_t mo,uint64_t ro,int*out,int n,void*stream){rng_metadata_kernel<<<(n+31)/32,32,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t*>(states),stride,mo,ro,out,n);return int(cudaGetLastError());}
int rng_cuda_set_stack(int bytes){return int(cudaDeviceSetLimit(cudaLimitStackSize,bytes));}
const char*rng_cuda_error(int code){return cudaGetErrorString(cudaError_t(code));}
}
#endif
