# Airplane + pair wings

[All combinations](README.md) · [Wiki home](../README.md) · [Pinned quirks](../rules/composite-quirks.md)

**Type ID:** `12`. **Card count:** 5n, n≥2.

## Construction

n consecutive triples and n distinct outside pair ranks. Each pair may be selected from a held triple or quad, but pair ranks must be distinct.

## Examples

3334445566; 4445557788.

## Exclusions and edge cases

One quad cannot supply two distinct pair-wing ranks; jokers cannot form a pair wing.

## Responding and strength

Same backbone length n, then highest rank whose multiplicity is three; ordinary wings do not have count3.

For ordinary live nonbomb/nonrocket combinations, any held exact four-card bomb
or rocket is also a legal override. A live bomb admits only a higher bomb, rocket
or pass; a live rocket only pass. Free leads never contain pass. The concrete
canonical legal-action set resolves unusual ambiguous multisets.

The ID above is the source detector/generator family number, **not a stable neural
action index**. Neural indices enumerate current legal rank multisets and change
from one observation to another. Cards are represented by the complete rank encoding
in [observations](../rules/observations.md).
