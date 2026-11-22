# Generals.io — offline mid-size 1v1 wiki

This wiki describes **`generals-research-1v1-medium-v2`**, the exact two-player fog-of-war benchmark. All game rules needed for ordinary play, all cell/structure types, action IDs and visible feature channels are written here. No internet or knowledge of a different Generals mode is needed. There are no collectible unit classes: every mobile army is an integer stack on a tile.

## Read the game

1. [Mode, objective and exclusions](mode.md)
2. [Every board, cell and structure type](board-and-structures.md)
3. [Map generation and its exact constraints/fallbacks](map-generation.md)
4. [Movement, battle, capture and army math](movement-and-combat.md)
5. [Joint-tick order and general trades](joint-order-and-trades.md)
6. [Growth clock and game endings](clock-and-outcomes.md)
7. [Fog, public knowledge and all 14 input channels](observations.md)
8. [All action IDs, directions and legal-mask rules](actions.md)
9. [Worked edge cases](worked-examples.md)
10. [Benchmark scoring, learning boundaries and validity](benchmark-boundary.md)
11. [Glossary](glossary.md)

See [scope and limitations](scope-and-limitations.md) for the exact mode, catalog counts and known limits.

The mode is fixed to independent 12–15 rows/columns padded to 16×16, neutral castles of 10–15 army, general trade enabled and a 1,000-joint-tick cap. Build-castles, deathtouch, teams, FFA, surrender actions and the old 8×8 task are not admitted. No private schedule, opponent weights, evaluation seeds, game artwork or credentials are included.
