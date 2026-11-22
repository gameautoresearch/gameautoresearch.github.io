# Arena and coordinate system

The battlefield is 18 placement tiles wide ×32 long, or 18,000×32,000 world units. A placement tile is 1,000 units; routing cells are 500 units, giving 36×64 cells. Troop collision and spell radii are integer world units, not pixels. All distances in the entity pages state their units.

## Tower coordinates

| Side | King center | Princess center A | Princess center B |
|---|---|---|---|
| 0 | (9000,3000) | (3500,6500) | (14500,6500) |
| 1 | (9000,29000) | (3500,25500) | (14500,25500) |

World coordinates are transformed into the acting player's observation/action orientation as described in [deployment](deployment.md). “Left” and “right” tower headers are ordered after this player-relative transform; they are not fixed global x labels.

Terrain records carry lane, blocker and water flags; crown buildings add dynamic occlusion/cost/deployment overlays. The river and bridge corridors are fixed by the pinned arena, not randomly generated each game. No maps, walls, extra buildings or alternate arena layouts are sampled for this task. A destroyed Princess tower changes both listed tower state and available deployment space.

The [routing atlas](arena-atlas.md) lists all 2,304 base cell values row by row with a legend. It is a textual rules grid, not game artwork. This gives offline exact terrain availability in addition to the normal legal-action mask. Do not use it to reveal hidden hand/order or RNG information.

[Wiki home](../README.md)
