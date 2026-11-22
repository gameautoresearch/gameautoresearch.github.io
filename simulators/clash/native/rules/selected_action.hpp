// SPDX-License-Identifier: Apache-2.0
// Selected-byte evaluation of the mask port from crforge env/src/main/java/org/crforge/env/
// LadderEnv.java mask(), landingTiles(), gameTiles(); uses core/src/main/java/org/crforge/core/
// battle/{unit/PlayerCommands,deploy/DeployMask,deploy/CardPlacement}.java rules unchanged.
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge).
// Full legal_actions remains the independent actor/control implementation.
#pragma once
#include "actions.hpp"
namespace clash_native {
template<class World> CN_HD inline bool selected_action_legal(const LadderMatch&m,const World&w,
 const PlayerCommands&commands,const CardRule cards[8],const ArenaView&a,const MaskEntity*seen,int n,
 int side,int action,bool done){
 if(action<0||action>=FIXED8_ACTIONS)return false;
 if(action==0)return true;
 // Fixed8 has no champion/evolution actions: the final two source mask bytes stay zero.
 if(action>=2305||done||m.tick<m.config.startCooldownTicks)return false;
 int encoded=action-1,slot=encoded/576,tile=encoded%576;
 Giving tried=commands.try_card(m,w,cards,side,slot);if(!tried.given())return false;
 int card=m.sides[side].hand.slots[slot];const CardRule&c=cards[card];uint8_t mask[RULES_MAX_TILES];
 deploy_mask(a,c,side,seen,n,mask);
 // Exact selected landing_tiles byte, including the seat transform and row's snapping.
 return placement_legal(a,c,mask,placement_snap(c,action_game_x(side,tile)),
                              placement_snap(c,action_game_y(side,tile)));
}
}
