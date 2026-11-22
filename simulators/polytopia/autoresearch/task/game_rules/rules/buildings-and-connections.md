# Buildings, adjacency, roads and connections

## Construction legality

Ordinary work occurs on owned tiles inside an own city's territory. The tile's
ruling city matters for per-city limits, even if territory rows overlap. A visible
hostile occupant blocks work; own units, hidden units and peace partners do not.
Structures additionally require enough stars, a valid/unoccupied improvement slot,
the correct terrain/resource, researched unlock, adjacency and unique restrictions.
Roads are separate flags: they can coexist with improvements but cannot be added
again or placed on a village/city that already counts as a road. Eligible explored
neutral roads and bridges are explicitly allowed. Most other economic buildings
require the normal owned-territory condition. Fungi kind37 is rejected by the native
profile even if a generic resource branch might otherwise list it.

Farm/Mine consume their corresponding crop/metal opportunity; LumberHut uses forest.
Harvest actions instead consume animal/fish/fruit for2 stars and add1 population,
requiring Hunting/Fishing/Organization respectively. An improvement blocks ordinary
resource harvest at that location. Forest actions: burn with Construction costs3,
changes forest to field and creates crop; clear with Forestry changes forest to
field and gains1 star; grow with Spiritualism costs5 and changes field to forest
without leaving a resource. Forest actions permit empty or Road-improved tiles.

## Adjacency and population

Windmill counts adjacent same-owner Farms; Sawmill counts LumberHuts; Forge counts
Mines. Each is limited to one per ruling city. Population contribution is its base
population times calculated level:1 per Windmill/Sawmill neighbor and2 per Forge
neighbor. Adjacency includes all eight neighboring tiles. Compatible neighbors can
belong to another city of the same owner; ownership and the receiver's ruling city
control where population accrues. Changed contributions are applied as deltas.

Market sums the levels of adjacent same-owner Windmill/Sawmill/Forge, each treated
as at least1, capped at8. Each level pays1 star. Start-of-turn Market work reads
neighbors' stored levels before queued refresh-level actions; do not collapse income
and all later adjacency updates into an arbitrary simultaneous recomputation.

## Temples and task monuments

Temple variants add1 population and100 score when built. At owner turn start their
score level is `max(old, clamp(1+floor((round-built_round)/2),1,5))`; each gained level
adds100 score. Seven native tasks create free once-per-player monuments with3
population and400 score. Tasks can be remembered after meeting their conditions;
destroying or losing a monument does not reset the player's already-built unique flag.
The [structure catalog](../structures/README.md) spells out every task and building.

## Roads and routes

Land road-to-road movement costs1 in half-movement units instead of2, when the
network tile is neutral/own/at peace. Boats and Creep units never receive this
speedup. A city/village and Bridge count as connectable tiles even without a road
flag. A Port opens eligible water/ocean/wetland route searches of range5; Bridge
opens water span range1. Bardur uses no Outpost or Mycelium route opener.

Route searches require explored terrain and exclude hostile-owned tiles. Entering
water from land requires a usable own/peace port and movement access. Router path
cost is5 onto road/Bridge/city/already established route,10 otherwise, with a
range*10 limit, A* priority distance+cost, no closed set and1000-pop safety bound.
Route creation order matters because later searches can follow earlier routes.
Transport-step flood then links adjacent connectable pairs or adjacent route pairs,
subject to ownership/peace checks. The connection network starts only at the
player's original capital while still owned; it does not silently choose any other
city as a network hub after that capital is captured.

Each newly connected own noncapital city gains1 population and the capital gains1;
disconnection removes those contributions. Capturing a connected city updates both
players' networks with preserved connection state and correct hub population
transfers. Loss of the original capital disconnects all that owner's connections.
Capture, border changes, roads, ports, bridges and relevant diplomatic/exploration
updates can trigger route/connection recalculation.

## Destruction

The Destroy action requires Chivalry (ID3), an eligible economic tile and a positive
improvement other than city, ruin or lighthouse. It removes the improvement, subtracts
its current score, updates adjacency population and routes/connections. Road flags
are cleared when destroying Road/Bridge and otherwise preserved when appropriate.
There is no automatic full construction refund. Per-city room may become available,
but a built unique monument remains used for that player.

[Wiki home](../README.md)
