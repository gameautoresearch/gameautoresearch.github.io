# Pair straight

[All combinations](README.md) · [Wiki home](../README.md) · [Pinned quirks](../rules/composite-quirks.md)

**Type ID:** `9`. **Card count:** 6–20 in reachable hands.

## Construction

At least three consecutive ranks, two copies of each, from3 throughA.

## Examples

334455; QQKKAA.

## Exclusions and edge cases

AA22 cannot continue through2; three nonconsecutive pairs are not a pair straight.

## Responding and strength

Same number of paired ranks and higher starting rank.

For ordinary live nonbomb/nonrocket combinations, any held exact four-card bomb
or rocket is also a legal override. A live bomb admits only a higher bomb, rocket
or pass; a live rocket only pass. Free leads never contain pass. The concrete
canonical legal-action set resolves unusual ambiguous multisets.

The ID above is the source detector/generator family number, **not a stable neural
action index**. Neural indices enumerate current legal rank multisets and change
from one observation to another. Cards are represented by the complete rank encoding
in [observations](../rules/observations.md).
