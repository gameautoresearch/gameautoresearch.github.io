# Movement, battles and capture

[Wiki home](README.md) · [Joint order](joint-order-and-trades.md) · [Examples](worked-examples.md)

## One move

Choose an owned source and one cardinal neighboring destination. The legal action mask requires source army `S>1`, an in-bounds destination and a nonmountain destination. You may attack neutral/enemy land even if the attack cannot capture it. There is no distance-two movement and no attack that leaves attacking troops on their original square.

Let `S` be the source army **at execution time**. Whole split (`split=0`) sends `M=S-1`; half split (`split=1`) sends `M=floor(S/2)`. The implementation clamps moved troops to `[0,S-1]`; reserve is `R=S-M`. Thus every ordinary legal move leaves at least one troop at source. Full and half both send one when `S=2`; both IDs remain legal.

A joint decision is selected from the public pre-step frame, but moves resolve sequentially. Earlier combat can change a later source's ownership/count. The later move is checked on the updated board, not granted troops that no longer exist. Exact [priority/dependency rules](joint-order-and-trades.md) apply.

## Destination belongs to you

Transfer `M` troops from source and add them to destination: source `S-M`, destination `D+M`. Ownership/structure identity remains. In this 1v1 mode no teammate destination exists. Upstream teammate-only zero-troop handover behavior is not an admitted extra action.

## Destination is enemy or neutral

Subtract moved troops from the source. Compare `M` with the destination army `D`:

| Condition | Destination army after move | Destination owner |
|---|---:|---|
| `M>D` | `M-D` | Attacking player captures |
| `M<D` | `D-M` | Defender/neutral keeps tile |
| `M=D` | 0 | Defender/neutral **keeps tile** |

Capture uses strict greater-than. An equal attack does not capture a zero-army-owned tile; it leaves it under the defender's ownership at zero. A later positive attack can capture it. There is no minimum army floor on destination, random damage, defensive mountain bonus or castle combat multiplier. Mountains are simply illegal targets.

On a successful capture of ordinary neutral land or a neutral castle, neutral ownership clears. A castle keeps its castle flag whether neutral or owned; capturing it does not remove or automatically add troops beyond subtraction. Neutral castles do not regenerate, so unsuccessful attacks can reduce their garrison for later attacks.

## Ordinary general capture

A general is always owned. A strictly winning attack on an enemy general:

1. Gives its tile to the attacker with surplus `M-D`.
2. Clears that tile's live-general flag and sets its castle flag.
3. Marks the defeated player eliminated.
4. Transfers every other cell still owned by the defeated player to the capturer, with each cell's army changed to `ceil(army/2) = (army+1)//2`.
5. Sets the winner when only the capturer's team remains alive, which is immediate in this two-player mode.

The already captured general tile is no longer owned by the defeated player at the transfer step, so its attack surplus is **not halved again**. Any remaining general-flagged transferred cells become castles, though normally there was just one general. A lone troop on transferred territory remains one; zero remains zero.

A later move of the defeated player cannot command confiscated territory. The deciding joint tick completes its time increment but pays no growth. A mutual capture satisfying the special trade test instead uses [general-trade resolution](joint-order-and-trades.md); it is not two sequential ordinary eliminations.

## Invalid input versus changed context

The raw engine masks an out-of-bounds, wrong-owner, impassable or eliminated-player move into no state change. It also has generic zero-army-transfer behavior for `S=1`. The benchmark's policy interface is stricter: select only an ID legal in the supplied mask, which excludes `S≤1`. A policy that outputs an illegal ID violates the submission boundary even if a lower-level engine would no-op it. A move legal when chosen but rendered inapplicable by the opponent's earlier move is ordinary simultaneous-context resolution, not proof that the policy selected an illegal action.
