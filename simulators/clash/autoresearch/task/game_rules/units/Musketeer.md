# Musketeer

One ground ranged troop that can attack ground and air, using a homing MusketeerProjectile. Its long range and projectile travel make hit cadence different from immediate damage. No evolution reload mechanic or champion ability is admitted.

## Effective level-11 statistics

| Property | Value |
|---|---|
| Native unit row ID | 2 |
| Observation vocabulary ID | 959 |
| Reachability | normal gameplay |
| Maximum HP | 721 |
| Damage per normal hit | 217 |
| Raw sustained DPS | 217.000 |
| Attack period (ms) | 1000 |
| Load time field (ms) | 300 |
| Deployment time field (ms) | 1000 |
| Range (world units) | 6000 |
| Sight range (world units) | 6000 |
| Collision radius (world units) | 500 |
| Movement speed field | 60 |
| Mass | 5 |
| Target restriction | ordinary allowed enemy targets |
| Ground attacks | yes |
| Air attacks | yes |
| Projectile | [MusketeerProjectile](../projectiles/MusketeerProjectile.md) |
| Pushback immunity | no |

Range is not simply center-to-center distance: the attacker and target collision radii extend reach. Sight/target acquisition, first-hit timing, pending projectile damage and retarget rules are described in [combat and targeting](../rules/combat.md). A base `damage=0` in the raw row does not mean a ranged unit deals zero damage: the projectile supplies it. Raw HP/damage values below are pre-scaling table values; use the effective table above during this level-11 mode.

## Complete exported unit definition

The table preserves every exported property, including disabled hooks. `none`, `no`, `0` and empty lists describe absent mechanics, not undocumented abilities. Referenced projectiles have their own complete encyclopedia pages.

| Pinned field | Value |
|---|---|
| `name` | Musketeer |
| `speed` | 60 |
| `range` | 6000 |
| `sightRange` | 6000 |
| `collisionRadius` | 500 |
| `mass` | 5 |
| `hitSpeedMs` | 1000 |
| `loadTimeMs` | 300 |
| `deployTimeMs` | 1000 |
| `attacksGround` | yes |
| `attacksAir` | yes |
| `air` | no |
| `building` | no |
| `characterBuilding` | no |
| `king` | no |
| `summonerTower` | no |
| `hitpoints` | 282 |
| `damage` | 0 |
| `crownTowerDamagePercent` | 0 |
| `rarity.name` | Common |
| `rarity.relativeLevel` | 0 |
| `rarity.levelCount` | 16 |
| `rarity.powerLevelMultiplier` | [110,121,133,146,160,176,193,212,233,256,281,309,339,372,409,450,495,545,600] |
| `projectile` | MusketeerProjectile |
| `customFirstProjectile` | none |
| `projectileStartRadius` | 450 |
| `projectileStartZ` | 450 |
| `projectileYOffset` | 0 |
| `multipleProjectiles` | 0 |
| `areaDamageRadius` | 0 |
| `selfAsAoeCenter` | no |
| `overrideAttackFinishTime` | no |
| `attackFinishTimeMs` | 0 |
| `resetHitTimerWhenNoTarget` | no |
| `spawnRadius` | 800 |
| `spawnAngleShift` | 0 |
| `flyingHeight` | 0 |
| `flyDirectPaths` | no |
| `spawnPathfindSpeed` | 0 |
| `spawnPathfindMorph` | none |
| `ingamePathfindSpeed` | 0 |
| `ingamePathfindVisible` | no |
| `ingamePathfindStopDeploys` | no |
| `groupProjectiles` | no |
| `spawnPushback` | 0 |
| `spawnPushbackRadius` | 0 |
| `tileSizeOverride` | 0 |
| `noDeploySizeW` | 0 |
| `noDeploySizeH` | 0 |
| `attackPushBack` | 0 |
| `ignorePushback` | no |
| `deathDamage` | 0 |
| `deathDamageRadius` | 0 |
| `deathPushBack` | 0 |
| `deathSpawnCharacter` | none |
| `deathSpawnCount` | 0 |
| `deathSpawnCharacter2` | none |
| `deathSpawnCount2` | 0 |
| `deathSpawnRadius` | 0 |
| `deathSpawnDeployTimeMs` | 0 |
| `deathAreaEffect` | none |
| `deathSpawnPushback` | no |
| `spawnConstPriority` | no |
| `deathSpawnMinRadius` | 0 |
| `deathSpawnProjectile` | none |
| `unmodelledDeathColumns` | [] |
| `champion` | no |
| `ability` | none |
| `globalId` | 34000014 |
| `lifeTimeMs` | 0 |
| `targetOnlyBuildings` | no |
| `attackSequence.mode` | 0 |
| `attackSequence.order` | [0] |
| `attackSequence.entries[0].damage` | 0 |
| `attackSequence.entries[0].projectile` | MusketeerProjectile |
| `attackSequence.entries[0].variableDamageTime` | 0 |
| `attackSequence.entries[0].hitSpeedMultiplier` | 100 |
| `attackSequence.entries[0].customRange` | -1 |
| `attackSequence.entries[0].customSightRange` | -1 |
| `attackSequence.entries[0].customMinimumRange` | -1 |
| `attackSequence.entries[0].customProjectileStartZ` | -1 |
| `attackSequence.entries[0].customProjectileStartRadius` | -1 |
| `attackSequence.entries[0].meleePushback` | 0 |
| `attackSequence.entries[0].meleePushbackAll` | no |
| `attackSequence.entries[0].doAttackAction` | none |
| `attackSequence.entries[0].customFirstProjectile` | none |
| `attackSequence.entries[0].customMultipleTargets` | -1 |
| `attackSequence.entries[0].customRememberMultipleTargets` | -1 |
| `attackSequence.entries[0].customOnAttackAction` | none |
| `attackSequence.entries[0].attackStartDelay` | 0 |
| `attackSequence.entries[0].customBuffOnDamage` | none |
| `attackSequence.entries[0].customBuffOnDamageTime` | -1 |
| `shieldLostAction` | none |
| `minimumRange` | 0 |
| `sightClip` | 1000 |
| `sightClipSide` | 0 |
| `loadFirstHit` | no |
| `spawnCharacter` | none |
| `spawnCharacter2` | none |
| `spawnCharacter3` | none |
| `spawnNumber` | 0 |
| `spawnIntervalMs` | 0 |
| `spawnPauseTimeMs` | 0 |
| `spawnStartTimeMs` | 0 |
| `spawnLimit` | 0 |
| `destroyAtLimit` | no |
| `spawnCharacterWithDeploy` | no |
| `untargetableWhenSpawned` | no |
| `gameTagsToSet` | 0 |
| `manaCollectAmount` | 0 |
| `manaGenerateTimeMs` | 0 |
| `manaOnDeathForOpponent` | 0 |
| `manaOnDeath` | 0 |
| `ignoreBuffs` | [] |
| `shieldHitpoints` | 0 |
| `stopMovementAfterMs` | 0 |
| `waitMs` | 0 |
| `deathInheritIgnoreList` | no |
| `spawnAttach` | no |
| `spawnMaxAngle` | 0 |
| `spawnAttachMaxRotation` | 0 |
| `chargeRange` | 0 |
| `chargeSpeedMultiplier` | 0 |
| `damageSpecial` | 0 |
| `keepChargingAfterAttack` | no |
| `jumpEnabled` | no |
| `jumpHeight` | 0 |
| `jumpSpeed` | 0 |
| `kamikaze` | no |
| `kamikazeTimeMs` | 0 |
| `multipleTargets` | 0 |
| `allTargetsHit` | no |
| `uniqueMultipleTargets` | no |
| `rememberMultipleTargets` | no |
| `buffOnDamage` | none |
| `buffOnDamageTimeMs` | 0 |
| `groupMaxSize` | 0 |
| `buffAfterHits` | [] |
| `buffAfterHitsCounts` | [] |
| `buffAfterHitsTimesMs` | [] |
| `dashCooldown` | 0 |
| `dashMinRange` | 0 |
| `dashMaxRange` | 0 |
| `dashDamage` | 0 |
| `dashRadius` | 0 |
| `dashPushBack` | 0 |
| `dashLandingTimeMs` | 0 |
| `dashConstantTimeMs` | 0 |
| `dashImmuneToDamageTimeMs` | 0 |
| `dashToTargetRadius` | no |
| `dashCount` | 0 |
| `dashSecondaryRange` | 0 |
| `backDashRadius` | 0 |
| `targetOnlyTroops` | no |
| `ignoreTargetsWithBuff` | none |
| `deprioritizeTargetsWithBuff` | no |
| `hovering` | no |
| `buffWhenNotAttacking` | none |
| `buffWhenNotAttackingTimeMs` | 0 |
| `buffWhenNotAttackingUseAttackRange` | no |
| `startWithBuffWhenNotAttacking` | yes |
| `startingBuff` | none |
| `startingBuffTimeMs` | 0 |
| `allowAreaDamageWhenInvisible` | no |
| `areaEffectOnHit` | none |
| `keepTargetWithPendingDamage` | yes |
| `hidesWhenNotAttacking` | no |
| `hideTimeMs` | 0 |
| `upTimeMs` | 0 |
| `ignoreClone` | no |
| `ignoreResurrect` | no |
| `occluder` | no |
| `clonedVersion` | none |
| `reflectedAttackBuff` | none |
| `reflectedAttackBuffDurationMs` | 0 |
| `reflectedAttackRadius` | 0 |
| `reflectedAttackDamage` | 0 |
| `reflectAttackCrownTowerDamage` | 0 |
| `specialRange` | 0 |
| `specialMinRange` | 0 |
| `specialLoadTimeMs` | 0 |
| `projectileSpecial` | none |
| `specialIgnoreBuildings` | no |
| `unmodelledColumns` | [] |

[Wiki home](../README.md)
