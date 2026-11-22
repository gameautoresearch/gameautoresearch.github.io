# Scope and limitations

[Wiki home](README.md)

polytopia-2.17.3 native Bardur1v1,11x11,fog,drylands maps only,score-at-cap at the round limit; diplomacy enabled,resign disabled

## Documented content

| Catalog | Entries |
|---|---:|
| supported unit forms | 18 |
| native technology records including basic | 26 |
| relevant structure mapobject relation records | 25 |
| hierarchical action families | 17 |
| observation v3 board channels | 151 |
| observation v3 global entries | 525 |

## Limits

- 10000 action field is mislabeled per-turn in benchmark declaration; native implementation uses a global accepted-action count. No engine change performed.
- Native random/path/push/deferred tie ordering is source-defined; wiki explains major rules but is not a separately proved executable specification.
- The drylands maps come from a synthetic preset, not the real game's map generator; real-game parity evidence is primarily Bardur Drylands14x14, and no commercial-generation parity is claimed for this preset or board size.
- Shared observation enums contain off-mode IDs; availability catalogs filter native-supported content.
- Resource ID8 starfish is not separately represented by baseline resource0..7 one-hot projection; raw public view/masks retain the applicable information.

The written rules apply to the pinned mode identified above. Catalog membership does not by itself establish that every combination of effects has been independently tested.
