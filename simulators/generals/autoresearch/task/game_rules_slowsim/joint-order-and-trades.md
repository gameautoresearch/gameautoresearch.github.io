# Joint-tick order and general trades

[Wiki home](README.md) · [Movement](movement-and-combat.md)

Both players choose against their own current views. This does **not** mean attacks are all calculated simultaneously from a frozen board. The engine first derives a deterministic order from the original state/actions, then executes each move against the state left by previous moves, with the explicit mutual-general-trade exception.

## Base priority

The original-state sort priority is lexicographic:

1. Nonpasses before passes.
2. Defensive moves before nondefensive moves. “Defensive” here means the destination is held by the mover's own team, not a general strategic interpretation.
3. Within the same defensive class, moves whose destination is **not any general tile** before moves onto any general tile. A friendly merge onto your own general counts as a general-target move for this comparison.
4. Larger **source army count** first, not larger sent count after splitting.
5. Equal remaining keys: lower seat index first at even `state.time`; higher index first at odd `state.time`.

Initial `time=0`, so seat 0 wins an otherwise equal first-step ordering tie. The next such tie favors seat 1. This parity uses time before the step increments. The priority keys are not recomputed after the first move.

## Chasing dependency

If move J enters move I's source, I waits for J to resolve first, **unless the two moves are a head-on swap**, meaning each enters the other's source. Thus an army cannot always evade an incoming attack simply because its base sort key is earlier. The engine queues the best-ranked unqueued move whose dependencies are all already queued. If every remaining move is blocked (relevant in general upstream N-player cycles), it picks the best-ranked unqueued move anyway. In this two-player task the head-on exemption prevents a two-move dependency deadlock.

After ordering, each move checks source ownership, remaining army, destination and elimination on the current state. The same destination may receive two attacks in sequence; the second sees the first's resulting owner/count. There is no hidden random initiative draw.

## Mutual general capture test

General trade is enabled. When the order walk reaches an unconsumed player's action, it checks a later unconsumed opposing player's action. A trade is recognized only if:

- Both moves are valid on the board at that point.
- Each destination is the other's current live general.
- Each attack sends **strictly more** than the defender's general garrison after the defender's departure, if its own source is that general.

Let player A send `MA` at B's general and B send `MB` at A's. Let `GB` be B's general's remaining reserve if B moves from it, otherwise its current army; define `GA` analogously. Trade needs `MA>GB` and `MB>GA`. An equality fails the trade test. A pass, wrong destination, vanished source or noncapturing attack is not a trade.

## Trade transformation

Both attacking stacks depart. Each player lands on the other's general with attack surplus:

- A's new general is B's former general tile, army `MA-GB`.
- B's new general is A's former general tile, army `MB-GA`.
- Every **other** territory cell of A goes to B, and vice versa.
- Each such nongeneral cell's remaining army is halved rounded up. If an attacking source was a nongeneral tile, its departure reserve is what is halved during transfer.
- General flags remain at both positions; each player's recorded general coordinate swaps. The two general tiles are not converted to castles.
- Neither player is eliminated. Both actions are consumed and do not execute again.

The ordinary step then increments time and, since the game is ongoing, pays any due growth to the resulting owners. A general trade may therefore exchange the strategic map rather than end the game. Do not score it as a double loss, draw or ordinary capture.

## Example

Adjacent generals each hold 6 troops and both choose a whole move at the other. Each sends 5 and leaves 1. Both tests `5>1` pass. Each lands on the other's old general with 4 troops. All other territories swap and halve rounded up. If old A's remote tile had 5 troops, it is now B's with 3. The generals remain live and play continues; later due income applies normally.
