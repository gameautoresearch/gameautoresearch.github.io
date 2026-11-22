# Airplane + single wings

[All combinations](README.md) · [Wiki home](../README.md) · [Pinned quirks](../rules/composite-quirks.md)

**Type ID:** `11`. **Card count:** 4n, n≥2.

## Construction

n consecutive triple ranks, plus n outside physical cards. Remove every backbone rank from candidate wings; outside wing ranks can repeat.

## Examples

33344456; 33344455; 333444555777.

## Exclusions and edge cases

A backbone rank cannot supply its leftover fourth copy as a wing. Not every emitted multiset gets an ordinary type11 detector label.

## Responding and strength

Response generation uses rival length n, then compares highest rank with exactly three copies in each multiset. See quirks for triple wings and detector-wrong shapes.

For ordinary live nonbomb/nonrocket combinations, any held exact four-card bomb
or rocket is also a legal override. A live bomb admits only a higher bomb, rocket
or pass; a live rocket only pass. Free leads never contain pass. The concrete
canonical legal-action set resolves unusual ambiguous multisets.

The ID above is the source detector/generator family number, **not a stable neural
action index**. Neural indices enumerate current legal rank multisets and change
from one observation to another. Cards are represented by the complete rank encoding
in [observations](../rules/observations.md).
