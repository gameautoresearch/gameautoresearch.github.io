# Pass

[All combinations](README.md) · [Wiki home](../README.md) · [Pinned quirks](../rules/composite-quirks.md)

**Type ID:** `0`. **Card count:** 0.

## Construction

Empty multiset; legal only when there is a live nonempty rival. It removes no card.

## Examples

Empty tuple `()` / empty list `[]`.

## Exclusions and edge cases

Passing on a free lead is not generated.

## Responding and strength

No rank comparison; it yields the turn without beating the rival.

For ordinary live nonbomb/nonrocket combinations, any held exact four-card bomb
or rocket is also a legal override. A live bomb admits only a higher bomb, rocket
or pass; a live rocket only pass. Free leads never contain pass. The concrete
canonical legal-action set resolves unusual ambiguous multisets.

The ID above is the source detector/generator family number, **not a stable neural
action index**. Neural indices enumerate current legal rank multisets and change
from one observation to another. Cards are represented by the complete rank encoding
in [observations](../rules/observations.md).
