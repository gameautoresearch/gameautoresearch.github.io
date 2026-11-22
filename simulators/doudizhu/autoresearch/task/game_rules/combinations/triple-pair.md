# Triple + pair

[All combinations](README.md) · [Wiki home](../README.md) · [Pinned quirks](../rules/composite-quirks.md)

**Type ID:** `7`. **Card count:** 5.

## Construction

One triple and a pair of a different ordinary rank.

## Examples

33344; AAA22.

## Exclusions and edge cases

A joker pair is impossible; attachments cannot share the triple rank.

## Responding and strength

Compare triple rank, implemented as sorted-card index2.

For ordinary live nonbomb/nonrocket combinations, any held exact four-card bomb
or rocket is also a legal override. A live bomb admits only a higher bomb, rocket
or pass; a live rocket only pass. Free leads never contain pass. The concrete
canonical legal-action set resolves unusual ambiguous multisets.

The ID above is the source detector/generator family number, **not a stable neural
action index**. Neural indices enumerate current legal rank multisets and change
from one observation to another. Cards are represented by the complete rank encoding
in [observations](../rules/observations.md).
