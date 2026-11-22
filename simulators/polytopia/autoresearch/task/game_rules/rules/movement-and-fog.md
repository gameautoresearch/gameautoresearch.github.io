# Movement, terrain, exploration and hidden units

## Grid geometry and movement budget

Tiles use `i=y*11+x`. Adjacency includes eight directions; range and local sight use
Chebyshev distance `max(abs(dx),abs(dy))`. Ordinary land steps cost2 half-units,
friendly road-to-road steps1. Budget is `2*(movement_stat + boosted_flag)`.
Moving once sets moved; a fresh Dash unit can still attack after normal movement.
Movement is not repeated by spending the remaining path budget as several actions.

A tile must be explored and terrain-accessible. Fishing permits shallow water access;
Sailing permits ocean; Climbing permits mountains. A Bridge bypasses certain terrain
access requirements. Non-Float units enter water through own/peace ports, causing
embarkation. Float units move on water but still obey technology/access masks.
A visible hostile occupant is not an ordinary passable endpoint. Friendly and peace
units may be traversed under the corresponding rules, but an occupied tile is not
a normal destination. Hide units can pass enemy-held tiles without ending there.

## Movement-stopping conditions

Entering a mountain, entering forest without a usable road edge (unless Creep),
entering a port with a non-Float unit, or landing a Carry unit on land stops further
travel. Visible nonpeace enemy units exert adjacent zone of control; Creep,
Infiltrate or an already invisible mover bypasses it. The optimized native flag
path applies raw nonhidden enemy adjacency even where the enemy is not currently
explored: use legal masks, not a reconstructed secret unit map. Forest entry costs
at least the entire budget, so a forest road tile reached off-road can end a move.
Friendly road connectivity must hold on both endpoints for the road discount.

The legal reachability solver runs eight synchronous relaxation rounds. A legal
multi-tile movement then uses the pinned A* path to select its route and reveals
from every traversed tile. The unit's final facing uses the last path step; this
can affect later deterministic push behavior. Do not replace route tie order with
a different shortest path merely because both reach the same endpoint.

## Cities, peace and bridges

An Infiltrate unit cannot finish a movement on an opponent city. No unit finishes
on a peace/cooldown partner's city. Other hostile cities may be occupied first;
capture is a later idle-unit action, normally on the next turn. Without Fishing,
a north-south Bridge cannot be entered straight from east/west, and vice versa;
diagonal entry is not automatically prohibited. See [naval play](naval.md).

## Exploration and visibility

Revealed tiles stay explored. The view reports terrain/structures/ownership and
nonhidden units on explored tiles; it is not a transient line-of-sight visibility
circle that re-fogs everything behind a moving army. Unexplored tile fields are
−1. Normal move/train/convert sight is radius1; mountain or Scout sight is2.
Capital start exploration is radius2. New tiles add5 score; first reveal of a
lighthouse can add1 population to the current capital. StartTurn itself does not
invent a fresh reveal; action-driven updates control it.

Cloak/Dinghy Hide can make units invisible to opponents that are not at peace.
The public `detect` bit on each own unit reports whether any neighboring hidden
enemy exists, without reporting its exact tile/type. Attempting a legal move onto
a hidden enemy reveals that enemy and leaves the mover at the source with its
movement still available. Merely passing a hidden unit on the chosen path does
not reveal it. Hiding, friendly observation and peace visibility are distinct.

Crop is hidden until Organization/Farming/Construction, metal until Climbing/
Mining/Smithery, starfish until Sailing/Navigation. A resource under an improvement
is not an ordinary harvest opportunity. The tensor encoder has its own feature
projection limits: starfish ID8 has no ordinary resource-one-hot channel in the
baseline0..7 resource vocabulary; the raw view and legal action masks still matter.

[Wiki home](../README.md)
