# TowerPrincessProjectile

Fired by PrincessTower in the native world; the exported TowerPrincess helper references the same projectile.

| Property | Value |
|---|---|
| Observation vocabulary ID | 1331 |
| Damage after level-11 scaling | 109 |
| Crown-tower damage per hit | 109 |
| Speed per 50 ms visit (world units) | 600 |
| Homing | yes |
| Area radius (world units) | 0 |
| Area hits air | no |
| Area hits ground | no |
| Only enemies | yes |
| Pushback (world units) | 0 |
| Crown damage percentage adjustment | 0 |

A homing projectile tracks its live target and registers pending damage for target-selection logic. On target removal, it retains the last position and clears the target reference. Direct-target projectiles do not become splash attacks when the target disappears. Spell projectiles use their aimed impact point and area victim filtering; Arrows additionally uses one shared victim chain per wave. Damage is enqueued, then applied at the holder hit-drain phase. Crown percentages are an additive adjustment to 100%, rounded **up**; `-80` means 20% damage, not negative damage.

## Complete exported projectile definition

| Pinned field | Value |
|---|---|
| `name` | TowerPrincessProjectile |
| `rarity.name` | Common |
| `rarity.relativeLevel` | 0 |
| `rarity.levelCount` | 16 |
| `rarity.powerLevelMultiplier` | [110,121,133,146,160,176,193,212,233,256,281,309,339,372,409,450,495,545,600] |
| `speed` | 600 |
| `gravity` | 60 |
| `homing` | yes |
| `homingTimeMs` | 0 |
| `homingMinDistance` | 0 |
| `damage` | 50 |
| `crownTowerDamagePercent` | 0 |
| `damageMode` | TOWER_DAMAGE |
| `radius` | 0 |
| `aoeToAir` | no |
| `aoeToGround` | no |
| `onlyEnemies` | yes |
| `projectileRadius` | 0 |
| `projectileRange` | 0 |
| `checkCollisions` | no |
| `considerZDistance` | no |
| `alwaysApplyPushback` | no |
| `minDistance` | 0 |
| `circleScatter` | no |
| `lineScatter` | no |
| `pushback` | 0 |
| `spawnCharacter` | none |
| `spawnCharacterCount` | 0 |
| `spawnCharacterDeployTimeMs` | 0 |
| `deflectedCharacterSpawn` | none |
| `spawnConstPriority` | no |
| `radiusY` | 0 |
| `projectileRadiusY` | 0 |
| `projectileStartExtraRadius` | 0 |
| `pushbackAll` | no |
| `spawnProjectile` | none |
| `spawnChain` | 0 |
| `chainIsNewProjectile` | no |
| `constantHeight` | 0 |
| `targetBuff` | none |
| `applyBuffBeforeDamage` | no |
| `applyBuffEvenIfImmuneToDamage` | no |
| `buffTimeMs` | 0 |
| `buffTimeIncreasePerLevel` | 0 |
| `maximumTargets` | 1000 |
| `onlyOwnTroops` | no |
| `spawnCount` | 0 |
| `spawnRadius` | 0 |
| `chainedHitRadius` | 0 |
| `chainedHitCount` | 0 |
| `pingpongVisualTimeMs` | 0 |
| `randomDelayMs` | 0 |
| `spawnAreaEffectObject` | none |
| `ignoreReflectedAttack` | no |
| `dragBackSpeed` | 0 |
| `dragSelfSpeed` | 0 |
| `dragMargin` | 0 |
| `dragBackAsAttractor` | no |
| `allowResetTarget` | yes |
| `deflectBehaviour` | 0 |
| `deflectRadius` | 0 |
| `deflectGroup` | yes |
| `actionOnDeflector` | none |
| `customDeflectAction` | none |
| `useCustomMovement` | no |
| `spawnAxisY` | no |
| `initialCollisionCheckFilter` | none |
| `unmodelledColumns` | [] |

[Wiki home](../README.md)
