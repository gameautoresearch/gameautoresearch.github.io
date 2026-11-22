# Archer

Two separate ground ranged troops are created for one card. They can target ground and air. Both use ArcherArrow; the second formation member first waits 100 ms, then deploys. Killing one does not remove the other. No evolution attack or champion ability exists in this mode.

## Fixed level-11 summary

| Property | Value |
|---|---|
| Elixir | 3 |
| Troops per card | 2 |
| HP per troop | 304 |
| Damage per hit per troop | 112 |
| Sustained raw DPS per troop | 124.444 |
| Attack period | 900 ms |
| First-hit load field | 400 ms; see targeting rules for timer/retarget behavior |
| Raw attack range | 5000 world units = 5 placement tiles |
| Movement speed field | 60 |
| Deployment countdown | 1000 ms after entity creation |
| Targets | ground and air |
| Full entity definition | [Archer](../units/Archer.md) |

Damage/DPS above is the pinned native value. DPS is damage divided by the attack period; it excludes deployment, first-hit loading, travel, retargets, stun and movement. The match does not expose a time-continuous DPS control. All actions first pass through the common [queued deployment rules](../rules/deployment.md).

## Complete plain-card deployment and cast fields

| Pinned field | Value |
|---|---|
| `name` | Archer |
| `count` | 2 |
| `secondary` | none |
| `secondaryCount` | 0 |
| `summonRadius` | 0 |
| `summonWidth` | 0 |
| `summonDeployDelayMs` | 100 |
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
| `name` | Archer |
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
