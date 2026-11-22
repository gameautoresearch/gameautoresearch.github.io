# Trick Room condition

[Condition index](README.md) · [Wiki home](../README.md)

## Effect and lifetime

For 5 turns, the Speed of every Pokemon is recalculated for the purposes of determining turn order. During the effect, each Pokemon's Speed is considered to be (10000 - its normal Speed), and if this value is greater than 8191, 8192 is subtracted from it. If this move is used during the effect, the effect ends.

This is the persistent condition created by [Trick Room](../moves/trickroom.md), rather than a separately selectable move.

## Duration and timing details

| Rule detail | Value |
|---|---|
| Duration | 5 |
| Field residual resolution order | 27 |
| Field residual sequence within stage | 1 |

Durations are measured in battle turns. Switching, fainting or the stated effect-specific conditions may end an effect before its maximum duration.
