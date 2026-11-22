# Single straight

[All combinations](README.md) · [Wiki home](../README.md) · [Pinned quirks](../rules/composite-quirks.md)

**Type ID:** `8`. **Card count:** 5–12.

## Construction

At least five consecutive distinct ranks from3 throughA.

## Examples

34567; 10JQKA; 3456789.

## Exclusions and edge cases

JQKA2 and A2XD cannot be sequences; a four-card run is too short.

## Responding and strength

Same number of ranks and higher starting rank. A six-card straight does not answer a five-card straight.

For ordinary live nonbomb/nonrocket combinations, any held exact four-card bomb
or rocket is also a legal override. A live bomb admits only a higher bomb, rocket
or pass; a live rocket only pass. Free leads never contain pass. The concrete
canonical legal-action set resolves unusual ambiguous multisets.

The ID above is the source detector/generator family number, **not a stable neural
action index**. Neural indices enumerate current legal rank multisets and change
from one observation to another. Cards are represented by the complete rank encoding
in [observations](../rules/observations.md).
