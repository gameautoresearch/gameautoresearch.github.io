# Four-card bomb

[All combinations](README.md) · [Wiki home](../README.md) · [Pinned quirks](../rules/composite-quirks.md)

**Type ID:** `4`. **Card count:** 4.

## Construction

Exactly four copies of one ordinary rank.

## Examples

3333; AAAA; 2222.

## Exclusions and edge cases

A quad with attachments is a composite and does not count as this bomb action.

## Responding and strength

Beats every nonbomb/nonrocket; higher quad rank beats lower quad; rocket beats it.

For ordinary live nonbomb/nonrocket combinations, any held exact four-card bomb
or rocket is also a legal override. A live bomb admits only a higher bomb, rocket
or pass; a live rocket only pass. Free leads never contain pass. The concrete
canonical legal-action set resolves unusual ambiguous multisets.

The ID above is the source detector/generator family number, **not a stable neural
action index**. Neural indices enumerate current legal rank multisets and change
from one observation to another. Cards are represented by the complete rank encoding
in [observations](../rules/observations.md).
