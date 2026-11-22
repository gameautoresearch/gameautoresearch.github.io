# Every player-visible observation and feature slice

[Wiki home](../README.md) · [Action interface](action-interface.md)

## Public observation fields

| Field | Meaning |
|---|---|
| `role` | The active role string |
| `hand` | Its own sorted rank multiset; the only private hand allowed |
| `remaining` | Hand sizes ordered landlord, down, up |
| `played` | All cards publicly played by each role in the same role order |
| `history` | Full public action sequence, including passes |
| `last_by_role` | Each role's latest action, including an empty pass |
| `last_move` | Engine-derived current rival; empty when free lead resets |
| `last_pid` | Last role to play a nonempty action |
| `bombs` | Number of exact four-card bombs/rockets actually played |
| `bottom_remaining` | Public bottom-card rank markers not yet matched by landlord play |
| `legal` | Complete sorted-unique canonical legal card multisets |

The union of opponents' remaining cards is reconstructed as known deck minus own
hand minus all publicly played cards. Its **partition** between the two other roles
is hidden. This union is public inference, not an engine hand leak. The observation
contains no deal ID, seed/index, RNG state, all-hand dictionary, engine handle or
private hand partitions. The upstream internal InfoSet does have privileged fields;
those do not pass through the integration's immutable public boundary.

## The common 54-value card encoding

Every hand, cumulative played pile, recent action and candidate uses the same int8
rank-multiplicity vector. For ordinary rank position r, four consecutive entries
are thermometer bits: count 0=0000, 1=1000, 2=1100, 3=1110, 4=1111. This is not
four independently named suits. The rank's four slots are `[4*r:4*r+4]`.

| Rank | Slots (zero-based, end-exclusive) |
|---|---|
| 3 | [0:4] |
| 4 | [4:8] |
| 5 | [8:12] |
| 6 | [12:16] |
| 7 | [16:20] |
| 8 | [20:24] |
| 9 | [24:28] |
| 10 | [28:32] |
| J | [32:36] |
| Q | [36:40] |
| K | [40:44] |
| A | [44:48] |
| 2 | [48:52] |
| Small joker | [52:53] |
| Big joker | [53:54] |

An empty pass is all 54 zeros. Integer card codes are not the vector offsets;
2's code is 17 but its thermometer starts at slot 48. For example pair 5 has
ones only at slots 8 and 9; rocket has ones only at 52 and 53.

## Landlord static vector: 319 int8 values

| Slice | Width | Content |
|---|---:|---|
| [0:54] | 54 | Own hand |
| [54:108] | 54 | Unseen-card union |
| [108:162] | 54 | Current rival/last_move |
| [162:216] | 54 | Cumulative landlord_up played cards |
| [216:270] | 54 | Cumulative landlord_down played cards |
| [270:287] | 17 | landlord_up hand-size one-hot |
| [287:304] | 17 | landlord_down hand-size one-hot |
| [304:319] | 15 | Bomb-count one-hot |

## Either peasant static vector: 430 int8 values

“Teammate” means up for down, and down for up. The same slice meanings are used
for either peasant, but the referenced teammate changes.

| Slice | Width | Content |
|---|---:|---|
| [0:54] | 54 | Own hand |
| [54:108] | 54 | Unseen-card union |
| [108:162] | 54 | Cumulative landlord played cards |
| [162:216] | 54 | Cumulative teammate played cards |
| [216:270] | 54 | Current rival/last_move |
| [270:324] | 54 | Landlord's latest action, even a pass |
| [324:378] | 54 | Teammate's latest action, even a pass |
| [378:398] | 20 | Landlord hand-size one-hot |
| [398:415] | 17 | Teammate hand-size one-hot |
| [415:430] | 15 | Bomb-count one-hot |

Hand-size n activates position n−1 within its block: e.g. landlord hand20 activates
local19; peasant hand17 activates local16. Hand-size blocks describe live states;
an empty hand immediately ends the game and yields no next player observation.
Bomb count b activates local position b in its 15-slot block. All these values
are 0/1, without signed-log, normalization or hidden-card ordering features.

## Public history: [5,162] int8

Take the last fifteen actions in oldest-to-newest order; prepend empty zero actions
until there are fifteen. Encode each with 54 values to make [15,54], then row-major
reshape into [5,162]. Each row concatenates three successive actions. Rightmost
history positions are most recent. Passes consume positions but are all zero,
so the tensor alone does not distinguish an old left-padding row from an actual
pass. Full `history` in the public observation retains that distinction.

The baseline encoder does not separately encode `last_pid`, role string, remaining
bottom cards or unlimited older history. Role selects the network/interface; it
is not another undocumented numeric input. These public facts can exist in the
observation without being represented by the baseline network tensors.

## Candidate-conditioned legacy inputs

The upstream `x_batch` appends one 54-value candidate vector to a repeated static
row: 373 values for landlord and 484 for a peasant. The integration factors these
into static/history once per observation plus a ragged candidate matrix, preserving
the exact information. Repeated `z_batch` contains the same [5,162] history for each
candidate. Factoring/chunking is computation plumbing, not permission to drop legal
moves or expose opponent hands.
