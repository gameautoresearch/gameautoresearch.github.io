# Scope and limitations

[Wiki home](README.md)

generals-research-1v1-medium-v2: two-player fog, neutral castles of 10..15 army, independent 12..15 rectangles padded 16, general-trade, 1000 ticks

## Documented content

| Catalog | Entries |
|---|---:|
| cell structure categories | 8 |
| directions | 4 |
| split choices | 2 |
| action ids | 2049 |
| observation channels | 14 |
| board shapes | 16 |

## Limits

- No new exhaustive simulator execution or parity proof; source-derived prose and existing public readout only.
- Map generator has explicit rare fallback behavior; universal requested-distance compliance is not claimed.
- Malformed direct low-level inputs, generic team/FFA/surrender APIs and disabled modifiers are outside mode scope.

The written rules apply to the pinned mode identified above. Catalog membership does not by itself establish that every combination of effects has been independently tested.
