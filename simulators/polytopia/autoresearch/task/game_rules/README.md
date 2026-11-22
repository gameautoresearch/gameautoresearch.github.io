# Polytopia Bardur1v1 — offline game wiki

This wiki covers the relevant **Polytopia2.17.3 native Bardur mirror mode**, not
all tribes or the retired simplified research ruleset. It is written so an agent
without internet can inspect the actual entities, formulas, legal choices and
exceptions. The common system prompt is separate; this folder is the game reference.

## Start with the mode, then look up rules

1. [Mode, starting position and maps](rules/mode-and-maps.md)
2. [Map parameter ranges](rules/map-parameters.md)
3. [Terrain, resources and tile concepts](rules/terrain-and-resources.md)
4. [Stars, research, recruitment, city levels and reward choices](rules/cities-and-economy.md)
5. [Every construction rule, adjacency, monuments and economic routes](rules/buildings-and-connections.md)
6. [Movement costs, stopping rules, exploration and hidden units](rules/movement-and-fog.md)
7. [Ports, passengers, upgrades, landing and disembarkation](rules/naval.md)
8. [Every relevant unit skill, recovery, promotion and push](rules/unit-skills.md)
9. [Exact integer damage, retaliation, area damage, conversion and infiltration](rules/combat.md)
10. [Capture, elimination, ruins, explorer and score](rules/capture-ruins-and-score.md)
11. [Meetings, peace, cooldown, embassies and war](rules/diplomacy.md)
12. [All17 action families, conditional masks and exact flattening](rules/actions.md)
13. [Every tensor block, scalar, normalization and visible-history field](rules/observations.md)
14. [Off-mode content, caps, failure handling and completeness boundaries](rules/scope-and-limits.md)

## Complete declared-mode catalogs

- [All18 native-supported unit forms](units/README.md), with exact HP, attack,
  defense, movement, range, nominal cost, skills and actual acquisition paths.
- [All26 native technology records](technologies/README.md), including Basic/Hunting,
  their dependencies, city-scaled prices and relevant unlocks.
- [All25 relevant improvement/map-object/relation definitions](structures/README.md),
  including all seven native task monuments, Port/Bridge/naval economics and Embassy.
- [Terrain/resource vocabulary](rules/terrain-and-resources.md), with irrelevant
  shared-enum values explicitly separated from available Bardur actions.

## Scope

See [scope and limitations](scope-and-limitations.md) for mode boundaries and documented catalogs.

**Important source discrepancy:** the benchmark declaration names a10000 value
`max_actions_per_turn`, but current native code counts accepted actions across the
whole episode without a turn reset. [Scope/limits](rules/scope-and-limits.md)
spells out that behavior and separates it from the score-adjudicated round limit.

This wiki does not expose seed banks, opponents, trained weights or privileged
state, and does not modify the engine, a live session or a frozen checkpoint.
