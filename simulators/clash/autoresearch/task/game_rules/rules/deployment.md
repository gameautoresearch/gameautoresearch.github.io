# Actions, command queue and deployment

## Flat action codec

`0` means wait. For action IDs 1 through 2,304:

- `encoded = action - 1`
- `hand_slot = encoded // 576`, from 0 through 3
- `tile = encoded % 576`
- `column = tile % 18`, `row = tile // 18`
- Local tile-center coordinate is `(column*1000+500, row*1000+500)`.

The two final IDs 2,305 and 2,306 are reserved champion actions and always illegal. At reset and after match ending, wait remains legal. Before tick 90 only wait is open. The supplied mask is authoritative; do not infer that all 576 target tiles for an affordable card are legal.

Seat 0's world x is `18000-local_x` and world y is local y. Seat 1's world x is local x and world y is `32000-local_y`. Observation coordinates use the same transforms. Actor outputs therefore use player-oriented coordinates, not raw global positions. The legal mask is mask version 2: **legal landing positions**, not merely positions that might snap somewhere acceptable.

## Accepted command versus execution

On acceptance, `givenTick=max(currentTick,1)` and `runTick=givenTick+20`; a command has an expiry reference at `currentTick+60`. It is validated for match state, living King, card in hand, no duplicate pending card and spendable whole elixir after promised costs. Candidate placement is snapped/validated and a command record is inserted. It does not immediately spend/cycle or create a troop.

Commands execute in insertion order before the match update when their run tick is due. Execution rechecks match/King/hand/elixir and does a second placement search against the evolved board. Only a successful execution spends elixir, removes/cycles the card, increments the deployment counter and creates entities. The generated play record captures the deployment counter before its increment. The public opponent play event is derived from execution/reveal, not from reading the opponent command queue.

A prior promise can be sure, expiring or fully released. If affordability would differ depending on the unmodelled expiry window, native admission returns `UnsupportedPromiseExpiry`; it does not guess. Failed/expired commands are not equivalent to illegal neural choices. Queue capacity 32 is a hard fault boundary.

## Troop placement and formation

The arena is 36×64 half-tile routing cells, corresponding to 18×32 placement tiles (1,000 world units/tile). Ground troops cannot be placed on blocked terrain/water or enemy-protected deployment regions. Crown tower no-deploy rectangles and footprints are recomputed from live visible buildings; loss of a Princess can open territory. Spells explicitly allow enemy side, water and building coordinates. No player-deployable building card is in the deck.

Placement clamps the requested point to the arena, snaps to tile centers, and searches nearest legal positions in expanding rings up to radius 30 placement tiles. Equal-distance selection follows the fixed ring traversal; a hand-coded nearest-cell tie breaker may differ. Symmetric snap subtracts one world-unit at particular seat/half-arena boundaries for ordinary ground troops. Formation members are clamped within 250 world units of the arena edge and lane-assigned from the pinned lane map.

All troop cards except Archer create one unit at the found origin. Archer creates two, horizontally offset by ±the unit collision radius (500 world units). Formation order may reverse in lane 1 under lane sequencing. The second Archer first enters WaitingToDeploy for 100 ms. After the wait, both ordinary deployment states use a 1,000 ms countdown. No troop attacks or moves normally until its deployment/state gate opens. The full post-acceptance latency therefore includes 1,000 ms command delay plus deployment, and Archer's extra stagger where applicable.

## Refusal codes

| GivingRefusal | Meaning |
|---|---|
| 0 None | Accepted command |
| 1 MatchOver | Match no longer accepts plays |
| 2 KingDead | Own King is dead |
| 3 NotInHand | Slot/card is unavailable |
| 4 StartCooldown | Opening 4,500 ms gate |
| 5 CardPending | That card already has an unexecuted command |
| 6 NotEnoughElixir | Not affordable after reservations |
| 7 NoTile | No legal placement found |
| 100 CapacityFault | Command capacity exhausted; fault |
| 101 UnsupportedPromiseExpiry | Ambiguous promise-expiry boundary; unsupported fault |

The raw last refusal and last execution refusal are visible diagnostics. Refused runs, illegal action totals and simulator faults remain distinct in assessment.

[Wiki home](../README.md)
