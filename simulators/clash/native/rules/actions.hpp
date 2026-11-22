// SPDX-License-Identifier: Apache-2.0
// Ported from crforge env/src/main/java/org/crforge/env/LadderEnv.java mask(),
// landingTiles(), gameTiles(), and env/src/main/java/org/crforge/env/Schema.java
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge).
#pragma once
#include "match.hpp"
namespace clash_native {
constexpr int FIXED8_ACTIONS=2307;
CN_HD inline int action_game_x(int side,int tile){int x=(tile%18)*1000+500;return side==0?18000-x:x;}
CN_HD inline int action_game_y(int side,int tile){int y=(tile/18)*1000+500;return side==0?y:32000-y;}
// seen entities must be filtered to SeatView.sees before passing them here.
// Fixed8 has only visible crown buildings; hidden troop state never affects this mask.
template<class World> CN_HD inline void legal_actions(const LadderMatch&m,const World&w,const PlayerCommands&commands,const CardRule cards[8],const ArenaView&a,const MaskEntity*seen,int n,int side,bool done,uint8_t*out){
    for(int i=0;i<FIXED8_ACTIONS;++i)out[i]=0;out[0]=1;
    if(done||m.tick<m.config.startCooldownTicks)return;
    uint8_t mask[RULES_MAX_TILES],landing[RULES_MAX_TILES];
    for(int slot=0;slot<4;++slot){Giving tried=commands.try_card(m,w,cards,side,slot);if(!tried.given())continue;int card=m.sides[side].hand.slots[slot];
        deploy_mask(a,cards[card],side,seen,n,mask);landing_tiles(a,cards[card],mask,landing);
        for(int t=0;t<576;++t){int gameTile=(action_game_y(side,t)/1000)*18+action_game_x(side,t)/1000;out[1+slot*576+t]=landing[gameTile];}
    }
}
}
