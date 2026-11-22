# FireballSpell

A single projectile is launched from the casting king toward the selected point. The explosion hits enemy ground/air units and buildings in its 2.5-tile radius; the crown-tower reduction is applied separately. The projectile has travel time and can push movable eligible troops by 1 tile. It does not damage friendly units or friendly towers.

| Property | Value |
|---|---|
| Observation vocabulary ID | 453 |
| Damage after level-11 scaling | 688 |
| Crown-tower damage per hit | 159 |
| Speed per 50 ms visit (world units) | 600 |
| Homing | no |
| Area radius (world units) | 2500 |
| Area hits air | yes |
| Area hits ground | yes |
| Only enemies | yes |
| Pushback (world units) | 1000 |
| Crown damage percentage adjustment | -77 |

A homing projectile tracks its live target and registers pending damage for target-selection logic. On target removal, it retains the last position and clears the target reference. Direct-target projectiles do not become splash attacks when the target disappears. Spell projectiles use their aimed impact point and area victim filtering; Arrows additionally uses one shared victim chain per wave. Damage is enqueued, then applied at the holder hit-drain phase. Crown percentages are an additive adjustment to 100%, rounded **up**; `-80` means 20% damage, not negative damage.

## Complete exported projectile definition

| Pinned field | Value |
|---|---|
| `name` | FireballSpell |
| `rarity.name` | Common |
| `rarity.relativeLevel` | 0 |
| `rarity.levelCount` | 16 |
| `rarity.powerLevelMultiplier` | [110,121,133,146,160,176,193,212,233,256,281,309,339,372,409,450,495,545,600] |
| `speed` | 600 |
| `gravity` | 50 |
| `homing` | no |
| `homingTimeMs` | 0 |
| `homingMinDistance` | 0 |
| `damage` | 269 |
| `crownTowerDamagePercent` | -77 |
| `damageMode` | CARD_DAMAGE |
| `radius` | 2500 |
| `aoeToAir` | yes |
| `aoeToGround` | yes |
| `onlyEnemies` | yes |
| `projectileRadius` | 0 |
| `projectileRange` | 0 |
| `checkCollisions` | no |
| `considerZDistance` | no |
| `alwaysApplyPushback` | no |
| `minDistance` | 0 |
| `circleScatter` | no |
| `lineScatter` | no |
| `pushback` | 1000 |
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
| `deflectBehaviour` | 8 |
| `deflectRadius` | 700 |
| `deflectGroup` | yes |
| `actionOnDeflector` | none |
| `customDeflectAction` | none |
| `useCustomMovement` | no |
| `spawnAxisY` | no |
| `initialCollisionCheckFilter` | none |
| `unmodelledColumns` | [] |

[Wiki home](../README.md)
