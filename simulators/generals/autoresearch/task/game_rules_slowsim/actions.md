# All action IDs and legal choices

[Wiki home](README.md) · [Movement math](movement-and-combat.md)

There are **2,049 action IDs**, all stable within this mode: pass plus 16×16 sources ×4 directions ×2 split choices. The network emits one logit per ID; the trusted inference path masks illegal logits and chooses the maximum legal one. No test-time search, external calls or extra action loop is admitted.

## Codec

- ID 0 is pass. Source/direction/split fields are irrelevant for it.
- For `row,column ∈ [0,15]`, direction `d∈[0,3]`, split `s∈[0,1]`:

`ID = 1 + (((row*16 + column)*4 + d)*2 + s)`

To decode any nonzero ID, let `q=ID-1`:

`row=q//128`, `column=(q//8)%16`, `direction=(q//2)%4`, `split=q%2`.

| Direction ID | Name | Row delta | Column delta |
|---:|---|---:|---:|
| 0 | Up | -1 | 0 |
| 1 | Down | +1 | 0 |
| 2 | Left | 0 | -1 |
| 3 | Right | 0 | +1 |

| Split ID | Sent troops from current source S | Reserve |
|---:|---:|---:|
| 0 | `S-1` | 1 |
| 1 | `S//2` | `S-S//2` |

The raw engine action is a five-int vector `[pass,row,column,direction,split]`, with `pass=1` only for wait and `pass=0` for moves. The benchmark does not admit raw build-castle IDs, coordinate commands outside the codec or arbitrary modifier actions.

## Mask

Pass is always legal. A move is legal in the supplied frame if and only if:

1. Source is owned by the acting player.
2. Source army is greater than one.
3. Cardinal destination is in bounds of the padded 16×16 grid.
4. Destination is not a mountain. Actual off-rectangle cells are mountain padding, so this excludes padding without an extra secret board query.

The mask opens **both** splits for each such source/direction. It does not require that the sent army beats the defender and does not add an “enemy general danger” rule. An attack into a neutral castle with a smaller army remains legal. Source ownership and immediate neighbor sight suffice to derive it; no hidden-state mask leak is needed.

Sample/select from this full joint mask. Independently sampling a source, direction and split marginal can join individually legal components into an illegal tuple. For `S=2`, the two split IDs have the same effect but remain two distinct legal IDs. This affects a “uniform legal ID” policy's distribution; deduplicating them would change the declared action sampler.

A move can become inapplicable after the opponent's earlier action changes its source; that is ordinary [joint-step resolution](joint-order-and-trades.md). It is different from intentionally outputting a mask-false ID.

## Enumerated examples

At source `(0,0)`, IDs 1/2 are up whole/half and 3/4 down whole/half; 5/6 left, 7/8 right. Up/left are out of bounds. At `(15,15)`, the final IDs 2047/2048 are right whole/half, which are always out of bounds. Nonzero IDs outside 1–2048 are never a legal action. A row/column lying in padding cannot be owned or provide a move source.
