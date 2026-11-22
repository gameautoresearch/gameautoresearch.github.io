# Scope and limitations

## What this mode is

13x13 Hex with no swap rule, evaluation games played as seat-swapped pairs, and MoHex 2.0 as the strong
reference fixed at 0 Elo. The method is that of Andy Jones's *Scaling Scaling Laws with Board Games*
(2021), whose largest board was 9x9; the 2-ply openings follow boardlaw, the paper's code, whose
reference evaluation played MoHex against itself from every 2-ply opening.

## What it leaves out

- **The swap rule.** Tournament Hex lets the second player take over the first stone; this mode does
  not, as in the paper. Fairness comes from playing every opening from both seats instead.
- **Other board sizes.** Only 13x13. The engine's geometry is fixed at 13.
- **Time controls.** There is no game clock. Your policy has a move clock per call: 0.1 seconds plus
  0.01 seconds for each position in it; a late call plays random moves instead. MoHex levels are
  bounded by simulated games per move.
- **Resignation and passing.** Neither exists. Every game is played until one side connects.

## What the anchor is and is not

MoHex's own default is a 10-second clock per move. Here the anchor is the same program with the paper's
solver settings and a fixed 50,000 simulated games per move instead of a clock, so that its strength
does not depend on how fast or busy the evaluation machine is.

On 9x9 that program is close to perfect play. On 13x13 it is not: it is a strong reference, and its
0 Elo is a fixed point of the scale, not a ceiling. More simulations buy little at this size: when this
mode was built, MoHex at 10,000 games per move won 14 of 32 games against the anchor, and MoHex at
2,000 won 21 of 96. A policy can exceed a 50% win rate against the anchor.

The anchor's solver runs in two extra threads, so its choice in a position can depend on thread timing.
The weaker levels search on one thread and repeat their choice for a given position.

## What the engine guarantees

The engine is tested against an independent breadth-first winner check on thousands of complete random
games and after every single move, on the neighbour table, the symmetries, the canonical frame and on
MoHex's own judgement of finished boards. Hex has no hidden information and no randomness, so the board
you receive is the whole state.
