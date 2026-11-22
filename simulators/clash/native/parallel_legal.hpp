// SPDX-License-Identifier: Apache-2.0
// Scheduling port of crforge env/src/main/java/org/crforge/env/LadderEnv.java mask(),
// landingTiles(), gameTiles(), plus core/src/main/java/org/crforge/core/battle/deploy/
// DeployMask.java and CardPlacement.java @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de.
// https://github.com/voonhous/crforge. All mask mechanics call the original native controls.
#pragma once
#include "rules/actions.hpp"
namespace clash_native {
// One caller-owned plan per game block. Production CUDA uses aligned shared raw storage, so
// no constructor or per-thread initialization runs for the maximum-capacity entity array.
template<int CAP> struct LegalMaskScratch {
 MaskEntity entities[CAP];int entityCount;int cardForSlot[8];uint8_t baseMasks[8][RULES_MAX_TILES];
};
template<int CAP,class World> CN_HD inline void prepare_parallel_legal(World&w,LegalMaskScratch<CAP>&scratch){
 // World.legal builds this list before the done/cooldown tests. It is read-only and identical
 // for both seats, so the cooperative plan materializes it once and preserves holder order.
 scratch.entityCount=w.mask_entities(scratch.entities);assert(scratch.entityCount<=CAP);
 for(int k=0;k<8;k++)scratch.cardForSlot[k]=-1;
 bool done=w.match.is_over(w)||w.match.truncated||w.match.fault;
 if(done||w.match.tick<w.match.config.startCooldownTicks)return;
 for(int side=0;side<2;side++)for(int slot=0;slot<4;slot++){
  Giving tried=w.commands.try_card(w.match,w,w.spec->cards,side,slot);if(!tried.given())continue;
  int card=w.match.sides[side].hand.slots[slot],index=side*4+slot;scratch.cardForSlot[index]=card;
  deploy_mask(w.arena_view(),w.spec->cards[card],side,scratch.entities,scratch.entityCount,scratch.baseMasks[index]);
 }
}
template<int CAP,class World> CN_HD inline void parallel_legal_slice(const World&w,const LegalMaskScratch<CAP>&scratch,
 uint8_t*out,int lane,int lanes){
 for(int flat=lane;flat<2*FIXED8_ACTIONS;flat+=lanes){
  int side=flat/FIXED8_ACTIONS,action=flat-side*FIXED8_ACTIONS;uint8_t legal=action==0?1:0;
  if(action>=1&&action<2305){int encoded=action-1,slot=encoded/576,tile=encoded%576,index=side*4+slot;
   int card=scratch.cardForSlot[index];if(card>=0){const CardRule&c=w.spec->cards[card];
    // Exactly landing_tiles()[gameTile]: retain the side transform, row's snapping, fullLane
    // deploy mask, placement margin, building footprint and all four frozen map-cell tests.
    int x=placement_snap(c,action_game_x(side,tile)),y=placement_snap(c,action_game_y(side,tile));
    legal=placement_legal(w.arena_view(),c,scratch.baseMasks[index],x,y)?1:0;
   }
  }
  out[flat]=legal;
 }
}
}
