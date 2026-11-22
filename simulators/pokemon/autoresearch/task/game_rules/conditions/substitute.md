# Substitute condition

[Condition index](README.md) · [Wiki home](../README.md)

## Effect and lifetime

The user takes 1/4 of its maximum HP, rounded down, and puts it into a substitute to take its place in battle. The substitute is removed once enough damage is inflicted on it, if the user switches out or faints, or if any Pokemon uses Tidy Up. Baton Pass can be used to transfer the substitute to an ally, and the substitute will keep its remaining HP. Until the substitute is broken, it receives damage from all attacks made by other Pokemon and shields the user from status effects and stat stage changes caused by other Pokemon. Sound-based moves and Pokemon with the Infiltrator Ability ignore substitutes. The user still takes normal damage from weather and status effects while behind its substitute. If the substitute breaks during a multi-hit attack, the user will take damage from any remaining hits. If a substitute is created while the user is trapped by a binding move, the binding effect ends immediately. Fails if the user does not have enough HP remaining to create a substitute without fainting, or if it already has a substitute.

This is the persistent condition created by [Substitute](../moves/substitute.md), rather than a separately selectable move.

## Duration and timing details

| Rule detail | Value |
|---|---|
| Try primary hit resolution priority | -1 |

Durations are measured in battle turns. Switching, fainting or the stated effect-specific conditions may end an effect before its maximum duration.
