# Zap

A short-lived area effect appears at the chosen point without a traveling projectile. It delivers typed damage to enemy ground/air units and buildings in a 2.5-tile radius and applies ZapFreeze for 500 ms. The area lifetime is 1 ms (visited on the 50 ms engine clock), not the stun duration. Stun is applied in the area post-hook; damage is queued and drained later in the same holder process. Damage to crown towers uses the explicit typed tower amount, not an inferred generic spell percentage.

## Fixed level-11 summary

| Property | Value |
|---|---|
| Elixir | 2 |
| Troop/building damage | 192 |
| Crown-tower damage | 48 |
| Radius | 2.5 tiles |
| Stun duration | 500 ms |
| Area lifetime | 1 ms, one engine visit |
| Effect definition | [Zap area and stun](../effects/Zap.md) |

Damage/DPS above is the pinned native value. DPS is damage divided by the attack period; it excludes deployment, first-hit loading, travel, retargets, stun and movement. The match does not expose a time-continuous DPS control. All actions first pass through the common [queued deployment rules](../rules/deployment.md).

## Complete plain-card deployment and cast fields

| Pinned field | Value |
|---|---|
| `name` | Zap |
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
| `projectile` | none |
| `areaEffect` | Zap |
| `searchUnit` | none |
| `spellAsDeploy` | no |
| `radius` | 2500 |
| `multipleProjectiles` | 0 |
| `projectileWaves` | 0 |
| `projectileWaveIntervalMs` | 0 |
| `projectileIntervalMs` | 0 |
| `listed` | [] |
| `listOffsetsXMirrored` | no |
| `group` | no |
| `namesCharacter` | no |

## Match-card fields relevant to this mode

| Pinned field | Value |
|---|---|
| `name` | Zap |
| `cost` | 2 |
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
