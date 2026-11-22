# Board and complete cell/structure catalog

[Wiki home](README.md) · [Map generation](map-generation.md)

Coordinates are zero-based `(row,column)`. Increasing row goes down; increasing column goes right. Movement uses orthogonal neighbors only. The actual rectangle occupies the top-left of a 16×16 array; the remainder is mountain padding. There is no wrapping, diagonal move, teleport, ranged attack or independent troop-type system.

## Complete gameplay categories

| Category | Passable? | Ownership/initial army | Behavior |
|---|---|---|---|
| Ordinary neutral ground | Yes | Neutral; initially 0 | First strictly winning attack claims it. Never produces while neutral. |
| Owned ordinary ground | Yes | Seat 0 or seat 1; integer army ≥0 | Friendly incoming armies merge. Produces one army at each active 50-tick boundary. Ownership can persist at zero after an equal attack. |
| General | Yes | One live general per seat, starts owned with army 1 | Produces one army at even active ticks and the usual 50-tick land increment. Losing it ordinarily eliminates its owner. |
| Neutral castle | Yes | Neutral; initial army 10–15 | Defending army must be exceeded to capture. Does not grow while neutral. Counts as an obstacle silhouette under fog, but is not an impassable mountain. |
| Owned castle | Yes | One player owns it; army changes normally | Produces like a general: one army every second tick, so a castle captured from N defenders has produced N armies again about 2N ticks later. Remains a castle when recaptured. |
| Captured former general | Yes | Capturer owns surplus on the captured tile | Ordinary general capture clears the general flag and sets the castle flag. Then behaves as a castle. |
| Mountain | No | No player owns it; army 0 | Blocks orthogonal movement. It cannot be attacked, demolished, built over or captured in this mode. Appears as a silhouette when unseen. |
| Padding mountain | No | Outside actual rectangle, no army | Same impassability as mountains; must never become an action source/destination. |

Ownership, army count and structure flags are separate state fields. “Castle” does not imply neutral or owned. “General” does not encode a different troop statistic. A troop stack of 30 on open land and a stack of 30 on a general have identical attack/defense subtraction; the general differs by production and defeat consequences. There are no HP bars, combat RNG, armor, damage multipliers or unit recruitment costs.

## Initial numeric-grid encoding

The generator passes a numeric grid into state creation. `-2` is a mountain, `0` empty ground, `1` seat 0's general, `2` seat 1's general, and values greater than 2 are castles with that many defending troops. `create_initial_state` turns the general markers into army 1, not armies 1 and 2. Every passable nongeneral cell is initially neutral. Castle markers initialize their stated troop count; other neutral ground has zero troops. Time is zero, winner is -1 and both players are alive.

## Structures and visibility

A castle or mountain outside sight appears only in `structures_in_fog`, which intentionally does not distinguish its type or reveal army/ownership. A hidden enemy general is **not** included in that silhouette category; it lies in ordinary fog until seen. Entering sight reveals the normal typed masks. See [all observation fields](observations.md) for exactly what can be known.
