# Arrows

Three waves of ten projectiles are launched toward the selected point with 200 ms between waves. Their scatter/ring construction covers a 3.5-tile spell footprint. Each projectile has a 1.4-tile local impact radius, but a per-wave chain de-duplicates victims: one target is not damaged ten times by overlapping arrows in a wave. A target present for all three waves takes three hits; moving into/out of the footprint can change how many waves connect. Crown-tower damage is rounded on each wave independently.

## Fixed level-11 summary

| Property | Value |
|---|---|
| Elixir | 3 |
| Damage per connecting wave/projectile chain | 122 |
| Crown-tower damage per connecting wave | 25 |
| Maximum waves hitting one continuously present target | 3 |
| Maximum total damage to one target | 366 |
| Maximum total crown-tower damage | 75 |
| Impact radius | 1.4 tiles |
| Spell ring radius | 3.5 tiles |
| Wave interval | 200 ms |
| Projectile definition | [ArrowsSpell](../projectiles/ArrowsSpell.md) |

Damage/DPS above is the pinned native value. DPS is damage divided by the attack period; it excludes deployment, first-hit loading, travel, retargets, stun and movement. The match does not expose a time-continuous DPS control. All actions first pass through the common [queued deployment rules](../rules/deployment.md).

## Complete plain-card deployment and cast fields

| Pinned field | Value |
|---|---|
| `name` | Arrows |
| `count` | 0 |
| `secondary` | none |
| `secondaryCount` | 0 |
| `summonRadius` | 0 |
| `summonWidth` | 0 |
| `summonDeployDelayMs` | 0 |
| `summonDeployDelaySecondMs` | 0 |
| `canDeployOnEnemySide` | yes |
| `canPlaceOnBuildings` | yes |
| `canPlaceOnWater` | yes |
| `fullLaneDeploy` | no |
| `touchdownLimitedDeploy` | no |
| `deployWTileMargin` | 0 |
| `deployStartY` | 0 |
| `deployEndY` | 0 |
| `projectile` | ArrowsSpell |
| `areaEffect` | none |
| `searchUnit` | none |
| `spellAsDeploy` | no |
| `radius` | 3500 |
| `multipleProjectiles` | 10 |
| `projectileWaves` | 3 |
| `projectileWaveIntervalMs` | 200 |
| `projectileIntervalMs` | 0 |
| `listed` | [] |
| `listOffsetsXMirrored` | no |
| `group` | no |
| `namesCharacter` | no |

## Match-card fields relevant to this mode

| Pinned field | Value |
|---|---|
| `name` | Arrows |
| `cost` | 3 |
| `forceToStartingHand` | no |
| `omitFromStartingHand` | no |
| `elixirProductionStopTimeMs` | 0 |
| `mirror` | no |
| `variant` | none |
| `mirrorUsesRootSpell` | yes |
| `darkElixirCost` | 0 |
| `form` | 0 |

Evolved/hero spell rows that appear in the upstream data are deliberately excluded: every slot flag is zero and the card form is plain. Those rows are not an alternative action in this task.

[Wiki home](../README.md)
