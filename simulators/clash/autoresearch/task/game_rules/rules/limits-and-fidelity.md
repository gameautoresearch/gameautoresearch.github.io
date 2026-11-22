# Failure guards and known fidelity boundaries

The simulator implements the reachable plain fixed-eight task independently. It does not certify commercial-game parity or other cards, levels, evolutions, champions, tower troops or modes. The fixed 16.635.8 tables and specific crforge pin are authoritative for this task; live balance sites may differ.

## Finite capacities

| Storage | Capacity | Overflow handling |
|---|---|---|
| Native world holder | 512 objects | explicit fault, not dropped object |
| Queued player commands | 32 | CapacityFault |
| Queued combat hits | 2,048 | fault 211 |
| Spell shared chains | 128 | explicit fault |
| One area/chain victim list | 256 | explicit combat fault |
| Per-health hit de-duplication list | 64 | explicit overflow fault |
| Per-unit active buff list | 8 | explicit fault |
| Visible entities per seat | 200 | counted observation overflow |
| Supported formation count | 1 or 2 | unsupported placement fault |
| Projectiles created in one fixed spell | at most 30 | explicit fault beyond this |

The native engine also refuses unsupported unit states, extra projectile/effect rows, unexpected buffs, stale data/binary identities and unsupported promise-expiry ambiguity. The full local source lists exact numeric codes. Faults are a separate outcome class. No finite-capacity shortcut grants a player a win or silently removes an inconvenient effect.

## Readability versus executable precision

This wiki gives readable mode rules, complete exported per-entity field tables, every named observation offset, exact level-11 summaries and source provenance. Integer geometry, target selection, holder order and route tie-breaks remain algorithmic: prose and the atlas explain them, while pinned implementation references provide the exact code. This is not a proof that every possible emergent multi-entity trajectory has been tested. Existing differential qualification is separate evidence, not recreated by documentation generation.

The exported TowerPrincess unit row is documented as a helper, not falsely presented as a deployable creature. Generic unused data fields, evolvedSpells and the global vocabulary can contain off-mode content; the wiki indexes only relevant rows and explicitly records inactive fields rather than claiming all upstream content is playable.

[Wiki home](../README.md)
