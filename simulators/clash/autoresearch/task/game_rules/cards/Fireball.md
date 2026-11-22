# Fireball

A single projectile is launched from the casting king toward the selected point. The explosion hits enemy ground/air units and buildings in its 2.5-tile radius; the crown-tower reduction is applied separately. The projectile has travel time and can push movable eligible troops by 1 tile. It does not damage friendly units or friendly towers.

## Fixed level-11 summary

| Property | Value |
|---|---|
| Elixir | 4 |
| Damage per connecting wave/projectile chain | 688 |
| Crown-tower damage per connecting wave | 159 |
| Maximum waves hitting one continuously present target | 1 |
| Maximum total damage to one target | 688 |
| Maximum total crown-tower damage | 159 |
| Impact radius | 2.5 tiles |
| Spell ring radius | 0 tiles |
| Wave interval | 0 ms |
| Projectile definition | [FireballSpell](../projectiles/FireballSpell.md) |

Damage/DPS above is the pinned native value. DPS is damage divided by the attack period; it excludes deployment, first-hit loading, travel, retargets, stun and movement. The match does not expose a time-continuous DPS control. All actions first pass through the common [queued deployment rules](../rules/deployment.md).

## Complete plain-card deployment and cast fields

| Pinned field | Value |
|---|---|
| `name` | Fireball |
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
| `projectile` | FireballSpell |
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
| `namesCharacter` | no |

## Match-card fields relevant to this mode

| Pinned field | Value |
|---|---|
| `name` | Fireball |
| `cost` | 4 |
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
