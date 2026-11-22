# Deck, roles, deal and objective

[Wiki home](../README.md) · [Combination catalog](../combinations/README.md)

## Full content catalog

| Printed rank | Engine integer | Copies | Relative strength |
|---|---:|---:|---:|
| 3 | 3 | 4 | 1 (lowest) |
| 4 | 4 | 4 | 2 |
| 5 | 5 | 4 | 3 |
| 6 | 6 | 4 | 4 |
| 7 | 7 | 4 | 5 |
| 8 | 8 | 4 | 6 |
| 9 | 9 | 4 | 7 |
| 10 | 10 | 4 | 8 |
| J | 11 | 4 | 9 |
| Q | 12 | 4 | 10 |
| K | 13 | 4 | 11 |
| A | 14 | 4 | 12 |
| 2 | 17 | 4 | 13 |
| Small joker (`X`) | 20 | 1 | 14 |
| Big joker (`D`) | 30 | 1 | 15 (highest) |

There are 54 physical cards, but actions represent **rank multiplicities**, not
suit-specific identities. Ordinary ranks have four copies; jokers are distinct
single cards. Ordinary cards have no special powers, HP, mana cost or suits in the
policy vocabulary. Each ordinary rank can form pairs/triples/bombs when sufficient
copies are held. Only 3 through A can be consecutive sequence ranks: the numeric
jump from A=14 to 2=17 prevents a sequence from continuing through 2 or a joker.

## Fixed roles and teams

| Role | Clockwise role ID | Starting cards | Team |
|---|---:|---:|---|
| `landlord` | 0 | 20 | landlord alone |
| `landlord_down` | 1 | 17 | peasants |
| `landlord_up` | 2 | 17 | peasants |

The landlord starts. The peasants cooperate and share a win/loss outcome, but each
acts from its own hand and public information. A peasant cannot directly choose
its teammate's action or see the teammate's private cards. Role assignment occurs
before play, with no bidding, calling, raising, robbing-landlord or redealing phase.
Names in old upstream dictionaries sometimes list `landlord_up` before
`landlord_down`; that storage order does not change the action order above.

## Deal and public bottom cards

The integration constructs the known deck, then uses the source-bound deterministic
Fisher–Yates shuffle driven by its declared split/seed/deal-index domain. This is
ordinary game setup, not information exposed to the player. Positions `[0:20]`
are landlord, `[20:37]` landlord_up and `[37:54]` landlord_down. Shuffle positions
`[17:20]` are the three public bottom cards and are **already included in the
landlord's twenty**, not three additional cards. Hands are sorted by engine rank.

The deal validator requires exactly the three role fields plus bottom-card field,
20/17/17 hand sizes, the exact full-deck multiset, and three bottom cards contained
in the landlord hand. Playing a landlord card matching a remaining public bottom
rank consumes a matching bottom marker. The public `bottom_remaining` field thus
shrinks over play. No extra economic or action rule is created by that marker.
The supplied network encoding does not use this separate bottom-card field.

## Objective and natural end

A player sheds cards by playing legal combinations. The first hand to empty ends
the game immediately; there is no final response round. If the landlord empties,
landlord wins and both peasants lose. If either peasant empties, both peasants win
and landlord loses. Passing never sheds a card and cannot directly end a game.
There is no natural draw state in ordinary valid play, no score-at-horizon winner,
and no penalty for finishing with a particular combination. The last combination
can still be a bomb/rocket and count toward the economic multiplier before ending.
