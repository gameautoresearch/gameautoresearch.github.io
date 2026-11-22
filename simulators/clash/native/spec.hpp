// SPDX-License-Identifier: Apache-2.0
// Frozen-data configuration for the independent fixed8 runtime. Constants are
// compiled by data_codegen.py from pinned crforge records, not battle traces.
#pragma once
#include "rules/rules.hpp"
#include "rules/unit_state.hpp"
#include "pathfinding/pathfinding.hpp"
#include "combat/combat.hpp"
namespace clash_native {
struct NativeUnitConfig {
 PfMovementConfig movement{};PfTargetConfig targeting{};StateVisitConfig state{};
 int maximumHp=0,damage=0,packedLevel=0,rowVocab=0,projectile=-1;
 int collisionRadius=0,mass=0,projectileStartRadius=0,projectileStartZ=0,projectileYOffset=0,
 noDeploySizeW=0,noDeploySizeH=0,tileSizeOverride=0,crownPercent=0;
 uint64_t rowTags=0;
 bool king=false,princess=false,building=false,hasMovement=false,ignorePushback=false,occluder=false;
};
struct NativeTags {
 PfFlagBits path{};uint64_t inactive=0,activating=0,keepsTargetingOff=0,noDamage=0,
 hidden=0,unkillable=0,attacking=0,noBuffs=0;
};
struct NativeTaskSpec {
 RulesConfig rules{};CardRule cards[8]{};NativeUnitConfig units[8]{};
 ProjectileConfig projectiles[6]{};SpellProjectileConfig spells[8]{};
 CombatAreaEffectConfig zap{};CombatBuffConfig stun{};NativeTags tags{};
 int arenaCells[2304]{},cardVocab[8]{},projectileVocab[6]{},zapVocab=0;
 int cardProjectile[8]{},cardArea[8]{},towerX[6]{},towerY[6]{},towerRow[6]{};
};
}
