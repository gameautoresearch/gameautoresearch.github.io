# Knight

One ground melee troop. It attacks opposing ground units and buildings, never air. Each hit is single-target direct damage, not an area strike. It has no ability, shield, charge, spawned unit or death effect in this mode.

## Fixed level-11 summary

| Property | Value |
|---|---|
| Elixir | 3 |
| Troops per card | 1 |
| HP per troop | 1766 |
| Damage per hit per troop | 202 |
| Sustained raw DPS per troop | 168.333 |
| Attack period | 1200 ms |
| First-hit load field | 700 ms; see targeting rules for timer/retarget behavior |
| Raw attack range | 1200 world units = 1.2 placement tiles |
| Movement speed field | 60 |
| Deployment countdown | 1000 ms after entity creation |
| Targets | ground |
| Full entity definition | [Knight](../units/Knight.md) |

Damage/DPS above is the pinned native value. DPS is damage divided by the attack period; it excludes deployment, first-hit loading, travel, retargets, stun and movement. The match does not expose a time-continuous DPS control. All actions first pass through the common [queued deployment rules](../rules/deployment.md).

## Complete plain-card deployment and cast fields

| Pinned field | Value |
|---|---|
| `name` | Knight |
| `count` | 1 |
| `secondary` | none |
| `secondaryCount` | 0 |
| `summonRadius` | 0 |
| `summonWidth` | 0 |
| `summonDeployDelayMs` | 0 |
| `summonDeployDelaySecondMs` | 0 |
| `canDeployOnEnemySide` | no |
| `canPlaceOnBuildings` | no |
| `canPlaceOnWater` | no |
| `fullLaneDeploy` | no |
| `touchdownLimitedDeploy` | no |
| `deployWTileMargin` | 0 |
| `deployStartY` | 0 |
| `deployEndY` | 0 |
| `projectile` | none |
| `areaEffect` | none |
| `searchUnit` | none |
| `spellAsDeploy` | no |
| `radius` | 0 |
| `multipleProjectiles` | 0 |
| `projectileWaves` | 0 |
| `projectileWaveIntervalMs` | 0 |
| `projectileIntervalMs` | 0 |
| `listed` | [] |
| `listOffsetsXMirrored` | no |
| `group` | no |
| `namesCharacter` | yes |

## Match-card fields relevant to this mode

| Pinned field | Value |
|---|---|
| `name` | Knight |
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
