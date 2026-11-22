# KingTower

Static crown king tower. It starts inactive, awakens after taking damage or after an own Princess tower is removed, and completes a staged 3,300 ms activation action before normal attacks. Its exact activation tag retention can add a holder pass at boundaries.

## Effective level-11 statistics

| Property | Value |
|---|---|
| Native unit row ID | 5 |
| Observation vocabulary ID | 797 |
| Reachability | normal gameplay |
| Maximum HP | 4824 |
| Damage per normal hit | 109 |
| Raw sustained DPS | 109.000 |
| Attack period (ms) | 1000 |
| Load time field (ms) | 500 |
| Deployment time field (ms) | 0 |
| Range (world units) | 7000 |
| Sight range (world units) | 7000 |
| Collision radius (world units) | 1400 |
| Movement speed field | 0 |
| Mass | 20 |
| Target restriction | ordinary allowed enemy targets |
| Ground attacks | yes |
| Air attacks | yes |
| Projectile | [KingProjectile](../projectiles/KingProjectile.md) |
| Pushback immunity | no |

Range is not simply center-to-center distance: the attacker and target collision radii extend reach. Sight/target acquisition, first-hit timing, pending projectile damage and retarget rules are described in [combat and targeting](../rules/combat.md). A base `damage=0` in the raw row does not mean a ranged unit deals zero damage: the projectile supplies it. Raw HP/damage values below are pre-scaling table values; use the effective table above during this level-11 mode.

## Complete exported unit definition

The table preserves every exported property, including disabled hooks. `none`, `no`, `0` and empty lists describe absent mechanics, not undocumented abilities. Referenced projectiles have their own complete encyclopedia pages.

| Pinned field | Value |
|---|---|
| `name` | KingTower |
| `speed` | 0 |
| `range` | 7000 |
| `sightRange` | 7000 |
| `collisionRadius` | 1400 |
| `mass` | 20 |
| `hitSpeedMs` | 1000 |
| `loadTimeMs` | 500 |
| `deployTimeMs` | 0 |
| `attacksGround` | yes |
| `attacksAir` | yes |
| `air` | no |
| `building` | yes |
| `characterBuilding` | no |
| `king` | yes |
| `summonerTower` | no |
| `hitpoints` | 2400 |
| `damage` | 0 |
| `crownTowerDamagePercent` | 0 |
| `rarity.name` | Common |
| `rarity.relativeLevel` | 0 |
| `rarity.levelCount` | 16 |
| `rarity.powerLevelMultiplier` | [110,121,133,146,160,176,193,212,233,256,281,309,339,372,409,450,495,545,600] |
| `projectile` | KingProjectile |
| `customFirstProjectile` | none |
| `projectileStartRadius` | 750 |
| `projectileStartZ` | 3500 |
| `projectileYOffset` | 400 |
| `multipleProjectiles` | 0 |
| `areaDamageRadius` | 0 |
| `selfAsAoeCenter` | no |
| `overrideAttackFinishTime` | no |
| `attackFinishTimeMs` | 0 |
| `resetHitTimerWhenNoTarget` | no |
| `spawnRadius` | 0 |
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
| `tileSizeOverride` | 2 |
| `noDeploySizeW` | 18 |
| `noDeploySizeH` | 16 |
| `attackPushBack` | 0 |
| `ignorePushback` | no |
| `onStartingAction` | KingTower_StartingGroup |
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
| `globalId` | 35000000 |
| `lifeTimeMs` | 0 |
| `targetOnlyBuildings` | no |
| `attackSequence.mode` | 0 |
| `attackSequence.order` | [0] |
| `attackSequence.entries[0].damage` | 0 |
| `attackSequence.entries[0].projectile` | KingProjectile |
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
| `sightClip` | 0 |
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
| `ignoreBuffs` | ["Event_Berserk_OnStartBuff"] |
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
