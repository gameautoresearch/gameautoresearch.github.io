# Zap area and ZapFreeze

A short-lived area effect appears at the chosen point without a traveling projectile. It delivers typed damage to enemy ground/air units and buildings in a 2.5-tile radius and applies ZapFreeze for 500 ms. The area lifetime is 1 ms (visited on the 50 ms engine clock), not the stun duration. Stun is applied in the area post-hook; damage is queued and drained later in the same holder process. Damage to crown towers uses the explicit typed tower amount, not an inferred generic spell percentage.

## Effective fixed-mode behavior

- Typed normal damage: **192**. Typed crown-tower damage: **48**.
- Radius: **2,500 world units / 2.5 tiles**. Only enemy units/buildings, ground and air.
- Effect object lifetime: **1 ms**, consumed on its first 50 ms engine visit; the independent buff lasts **500 ms**.
- ZapFreeze changes movement and attack speed by **−100%**. Its row also carries spawn-speed −100%; the fixed8 state config does not scale ordinary deployment countdown by troop speed.
- Stunned units lose/clear target references through the combat gate; their attack timing restarts through the existing null-reference/loading path. It is more than “reduce damage for half a second.”
- Reapplying the same non-individual stun refreshes to the longer remaining duration, rather than stacking duplicate independent −100% slows. Expiry decrements by 50 ms visits.
- Damage victim processing is stable nearest-first for filter hits. Damage is queued; the stun is applied immediately during the area post-hook. That ordering can affect targeting within a tick.

## Complete exported area definition

| Pinned field | Value |
|---|---|
| `name` | Zap |
| `rarity.name` | Common |
| `rarity.relativeLevel` | 0 |
| `rarity.levelCount` | 16 |
| `rarity.powerLevelMultiplier` | [110,121,133,146,160,176,193,212,233,256,281,309,339,372,409,450,495,545,600] |
| `lifeDurationMs` | 1 |
| `radius` | 2500 |
| `maxRadius` | 0 |
| `hitSpeedMs` | 0 |
| `hitSpeedOffsetMs` | 0 |
| `damage` | 0 |
| `crownTowerDamagePercent` | 0 |
| `hitsAir` | no |
| `hitsGround` | no |
| `onlyEnemies` | no |
| `ignoreBuildings` | no |
| `pushback` | 0 |
| `pushbackAll` | no |
| `relativePushback` | no |
| `continuousPushback` | no |
| `maximumTargets` | 0 |
| `sharedDamage` | no |
| `buff` | ZapFreeze |
| `buffTimeMs` | 500 |
| `capBuffTimeToAreaEffectTime` | no |
| `onlyOwnTroops` | no |
| `spawnAreaEffectObject` | none |
| `projectile` | none |
| `hitBiggestTargets` | no |
| `projectileStartHeight` | 0 |
| `targetProjectiles` | yes |
| `affectsHidden` | no |
| `controlsBuff` | no |
| `cloning` | no |
| `oneHitPerTarget` | no |
| `expireOnTrigger` | no |
| `followsParent` | no |
| `followsTarget` | no |
| `deflectsProjectiles` | no |
| `spawnCharacter` | none |
| `spawnIntervalMs` | 0 |
| `spawnInitialDelayMs` | 0 |
| `spawnTimeMs` | 0 |
| `spawnMaxCount` | 0 |
| `spawnMinRadius` | 0 |
| `spawnRandomizeSequence` | no |
| `spawnClones` | no |
| `stayAfterParentDies` | no |
| `linkToInstigatorLife` | no |
| `shaped` | no |
| `shapeWidth` | 0 |
| `shapeHeight` | 0 |
| `shapeRadius` | 0 |
| `damageType` | none |
| `filter` | CommonAreaDamageFilter |
| `filterHits` | yes |
| `typedDamage.name` | none |
| `typedDamage.baseDamage` | 75 |
| `typedDamage.towerDamage` | 19 |
| `unmodelledColumns` | [] |

## Mechanically relevant buff fields

| Pinned field | Value |
|---|---|
| `HitSpeedMultiplier` | -100 |
| `Name` | ZapFreeze |
| `Rarity` | Common |
| `SpawnSpeedMultiplier` | -100 |
| `SpeedMultiplier` | -100 |

[Wiki home](../README.md)
