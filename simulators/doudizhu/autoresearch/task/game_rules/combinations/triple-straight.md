# Bare triple straight

[All combinations](README.md) · [Wiki home](../README.md) · [Pinned quirks](../rules/composite-quirks.md)

**Type ID:** `10`. **Card count:** 6–18 in reachable hands.

## Construction

At least two consecutive ranks, three copies of each, from3 throughA.

## Examples

333444; QQQKKKAAA.

## Exclusions and edge cases

Triples of A and2 are not consecutive. A single triple uses family3.

## Responding and strength

Same number of triple ranks and higher starting rank; detector checks this before airplane-wing special cases.

For ordinary live nonbomb/nonrocket combinations, any held exact four-card bomb
or rocket is also a legal override. A live bomb admits only a higher bomb, rocket
or pass; a live rocket only pass. Free leads never contain pass. The concrete
canonical legal-action set resolves unusual ambiguous multisets.

The ID above is the source detector/generator family number, **not a stable neural
action index**. Neural indices enumerate current legal rank multisets and change
from one observation to another. Cards are represented by the complete rank encoding
in [observations](../rules/observations.md).
