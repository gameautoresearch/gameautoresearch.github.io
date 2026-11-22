# Triple + single

[All combinations](README.md) · [Wiki home](../README.md) · [Pinned quirks](../rules/composite-quirks.md)

**Type ID:** `6`. **Card count:** 4.

## Construction

One triple and one card of a different rank; the attachment may be a joker.

## Examples

3334; AAA2; 222X.

## Exclusions and edge cases

3333 is emitted as bomb, not as a triple with its own fourth copy.

## Responding and strength

Compare the triple rank, implemented as sorted-card index1. Wing rank does not ordinarily matter.

For ordinary live nonbomb/nonrocket combinations, any held exact four-card bomb
or rocket is also a legal override. A live bomb admits only a higher bomb, rocket
or pass; a live rocket only pass. Free leads never contain pass. The concrete
canonical legal-action set resolves unusual ambiguous multisets.

The ID above is the source detector/generator family number, **not a stable neural
action index**. Neural indices enumerate current legal rank multisets and change
from one observation to another. Cards are represented by the complete rank encoding
in [observations](../rules/observations.md).
