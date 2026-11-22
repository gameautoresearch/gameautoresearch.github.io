# Scope and limitations

[Wiki home](README.md)

AutoTavern 37.0.2 implemented-content, four players, regular tavern tiers 1–4, 3 random types, 2 hero choices, 7 board / 10 hand, 30 recruit action cap; no global Timewarp/experimental mode.

The catalog below documents the whole pinned card set. In this form only tier 1–4 minions (and tier-7 pool cards through special effects) are in the shop pool; tier 5–6 definitions remain listed because some effects name or create them. Card effects written for the eight-player game with six tiers (for example "Discover a Tier 6 minion") run as the engine implements them against this smaller pool; that interaction has not been audited card by card.

## Documented content

| Catalog | Entries |
|---|---:|
| Shop minions | 244 |
| Heroes | 116 |
| Tavern spells | 70 |
| Trinkets | 234 |
| Dark Gifts | 47 |
| Timewarp | 253 |
| Tier 7 | 12 |

## Limits

- Static dependency closure is conservative and not a formal all-sequences reachability proof.
- A printed card description plus trigger/target summary does not enumerate every cross-card event interaction.
- Unresolved direct registry tuples are explicitly identified per card; helper-generated and hard-coded logic remain owning source.
- Commercial exact parity is unqualified; unpublished offer weights and simultaneous-death ties use declared emulator assumptions.

The written rules apply to the pinned mode identified above. Catalog membership does not by itself establish that every combination of effects has been independently tested.
