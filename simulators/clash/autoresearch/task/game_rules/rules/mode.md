# Exact game mode and boundaries

This is `clash-fixed8-v1`: two players, a full symmetric arena, the same eight-card plain deck, card/tower level 11, Princess towers, paused simultaneous 2 Hz control (ten 50 ms ticks per decision) and the short match form: 60 s regulation + 30 s overtime (see [time and outcomes](time-and-outcomes.md)). The native C++ CPU/CUDA implementation is independent; the pinned Java crforge implementation is a differential oracle. This is a simulator benchmark, not control of the live commercial game.

The deck is Knight, Archer, Musketeer, Giant, MiniPekka, Fireball, Arrows, Zap. No deck building, substitutions, champion abilities, hero variants, evolutions, alternate tower troops, emotes or other cards enter play. Upstream tables may mention thousands of objects; only the reachable subset documented here belongs to this mode. All eight cards are plain forms, with zero slot flags.

## Objective

Each player starts with one King and two Princess crown towers. Destroying an enemy Princess gives a crown; destroying the enemy King gives three crowns and decides the match. If the King remains, crowns are `2 - min(2, number of listed enemy Princess towers)`. Listing/cleanup timing is therefore significant. Regulation, sudden-death overtime and natural tiebreak are in [time and endings](time-and-outcomes.md).

## Scope versus release versions

The rules/content described here are the fixed8 game shared by the baseline and later fast/evaluator contracts. Their model-size limits, teaching/reference pools, resource contracts, policy admission and evaluation schedules can differ. The wiki does not replace those versioned contracts. The baseline `benchmark declaration` is preserved as a public contract reference, not an authorization to launch a historical campaign. Hardware and experiment budgets are not game rules.

## Mode-specific absence

There are no air troops to spawn from the fixed deck, although ranged troops/towers/spells preserve their air-target flags. There are no player buildings beyond crown towers, no healing, shields, lifetime-decay troops, death spawns, charge, jump, dash, cloak, clone or champion actions in the reachable fixed8 rows. Fireball pushback and Zap stun are the meaningful spell control effects. These absences reduce the required encyclopedia to eight cards, five ordinary troop types, two tower types, six projectile types and one area/buff pair.

[Wiki home](../README.md)
