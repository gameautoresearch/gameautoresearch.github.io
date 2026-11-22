# Four + two pairs

[All combinations](README.md) · [Wiki home](../README.md) · [Pinned quirks](../rules/composite-quirks.md)

**Type ID:** `14`. **Card count:** 8.

## Construction

A quad plus two distinct outside ranks with two cards each.

## Examples

33334455; AAAA3322.

## Exclusions and edge cases

33334444 is detector-recognized type14 but not normally generator-emitted: one other quad is only one pair rank.

## Responding and strength

Same family, compare the rank with four occurrences. See the detector-only two-quad ambiguity in quirks.

For ordinary live nonbomb/nonrocket combinations, any held exact four-card bomb
or rocket is also a legal override. A live bomb admits only a higher bomb, rocket
or pass; a live rocket only pass. Free leads never contain pass. The concrete
canonical legal-action set resolves unusual ambiguous multisets.

The ID above is the source detector/generator family number, **not a stable neural
action index**. Neural indices enumerate current legal rank multisets and change
from one observation to another. Cards are represented by the complete rank encoding
in [observations](../rules/observations.md).
