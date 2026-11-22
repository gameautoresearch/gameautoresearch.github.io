# Fog, public knowledge and all input channels

[Wiki home](README.md) · [Action mask](actions.md)

## Visibility

A cell is visible if it lies in the 3×3 neighborhood of any owned cell: Chebyshev radius one, including diagonal sight. Sight does **not** stop at a mountain; it is Boolean dilation of ownership, not a line-of-sight ray. Movement remains cardinal. Newly captured land immediately changes which cells are visible for the next observation. Losing land can hide previously seen information again; the default observation stores no remembered map.

All of your cells are visible. An enemy cell is marked enemy only while visible. Army counts in unseen cells are zeroed, hidden general/castle/mountain type masks are zeroed, and neutral ownership is shown only in sight. Unseen mountains and castles instead appear as the combined `structures_in_fog` silhouette. That silhouette reveals an obstacle/structure but not which type, army or owner. Hidden generals are not silhouettes; unless another structure flag existed they look like ordinary fog. The full engine state is never a policy input.

The 1v1 scoreboard intentionally gives both players' **global total land and total armies**, including the opponent's unseen owned cells. This does not reveal the distribution over tiles. Neutral defenders do not contribute to either player's army total. Padded mountains contribute to neither total. Fog does not erase these public scalars.

## Exact network tensor

The submitted actor and value function get the **same** visible float tensor `[batch,16,16,14]` (possibly other leading batching dimensions internally). Axes are row, column, channel. The following normalization is the exact codec.

| Channel | Field | Meaning and encoding |
|---:|---|---|
| 0 | armies | `log(1+visible_army)/6`; zero in fog |
| 1 | generals | 1 for a visible live general, else 0 |
| 2 | castles | 1 for visible castle, including captured former generals, else 0 |
| 3 | mountains | 1 for visible mountain/padding, else 0 |
| 4 | neutral_cells | 1 for visible passable neutral cell, else 0 |
| 5 | owned_cells | 1 for your owned cell, else 0 |
| 6 | opponent_cells | 1 for a visible enemy-owned cell, else 0 |
| 7 | fog_cells | 1 for an invisible non-mountain/non-castle cell, else 0 |
| 8 | structures_in_fog | 1 for an invisible mountain or castle, else 0 |
| 9 | owned_land_count | Your total owned cells divided by `16*16 = 256`, broadcast over all positions |
| 10 | owned_army_count | `log(1+your_total_army)/8`, broadcast |
| 11 | opponent_land_count | Enemy total owned cells divided by 256, broadcast |
| 12 | opponent_army_count | `log(1+enemy_total_army)/8`, broadcast |
| 13 | timestep | Current post-step integer time divided by 1000, broadcast |

`log` is natural log; scales are not clipping thresholds, so large army features can exceed 1. Land normalization uses the **padded tensor area 256**, not the smaller playable area. Boolean masks are float32 zero/one when encoded. Every scalar plane is constant across all 256 spatial positions. General/castle type masks are not ownership masks; combine them with owned/opponent/neutral information where visible.

This benchmark supplies only the fourteen channels above: no teammate planes, hidden critic, seed, RNG key, pool index, true enemy board or simulator object.

## Memory and inference

The current submission interface is stateless greedy inference on this frame plus the legal-action mask. There is no policy memory state or arbitrary history channel. You may not secretly maintain a process-level map or recover hidden engine state. The absence of remembered terrain in the encoder does not make hidden terrain fair input. Architecture/research changes remain within the selected session's admitted interface.

## Useful distinctions

A zero in channel 0 can mean empty ground, a zero-army-owned tile, or fog. Use the visibility and ownership channels rather than interpreting zero as “known empty.” A structure silhouette is not a legal-movement oracle into an unknown distant tile. Immediate cardinal destinations of owned source cells are within the 3×3 sight, so the local legal mask can be computed without inspecting hidden cells.
