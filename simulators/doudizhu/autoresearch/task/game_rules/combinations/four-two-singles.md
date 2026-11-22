# Four + two singles

[All combinations](README.md) · [Wiki home](../README.md) · [Pinned quirks](../rules/composite-quirks.md)

**Type ID:** `13`. **Card count:** 6.

## Construction

A quad plus any two physical cards of other ranks. Their ranks may match; both jokers may be used.

## Examples

333345; 333344; 3333XD.

## Exclusions and edge cases

The quad’s rank cannot be reused as an attachment. This whole six-card action is not a bomb for override/scoring.

## Responding and strength

Same family, compare quad rank (sorted-card index2).

For ordinary live nonbomb/nonrocket combinations, any held exact four-card bomb
or rocket is also a legal override. A live bomb admits only a higher bomb, rocket
or pass; a live rocket only pass. Free leads never contain pass. The concrete
canonical legal-action set resolves unusual ambiguous multisets.

The ID above is the source detector/generator family number, **not a stable neural
action index**. Neural indices enumerate current legal rank multisets and change
from one observation to another. Cards are represented by the complete rank encoding
in [observations](../rules/observations.md).
