# Giant

One slow, high-health ground melee troop that selects buildings only. Enemy troops can attack it but are not ordinary attack targets for it. It deals single-target direct damage. The exported immunity-to-pushback flag is recorded on its unit page; do not assume a generic troop-sized push rule.

## Fixed level-11 summary

| Property | Value |
|---|---|
| Elixir | 5 |
| Troops per card | 1 |
| HP per troop | 3968 |
| Damage per hit per troop | 253 |
| Sustained raw DPS per troop | 168.667 |
| Attack period | 1500 ms |
| First-hit load field | 1000 ms; see targeting rules for timer/retarget behavior |
| Raw attack range | 1200 world units = 1.2 placement tiles |
| Movement speed field | 52 |
| Deployment countdown | 1000 ms after entity creation |
| Targets | buildings only |
| Full entity definition | [Giant](../units/Giant.md) |

Damage/DPS above is the pinned native value. DPS is damage divided by the attack period; it excludes deployment, first-hit loading, travel, retargets, stun and movement. The match does not expose a time-continuous DPS control. All actions first pass through the common [queued deployment rules](../rules/deployment.md).

## Complete plain-card deployment and cast fields

| Pinned field | Value |
|---|---|
| `name` | Giant |
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
| `name` | Giant |
| `cost` | 5 |
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
