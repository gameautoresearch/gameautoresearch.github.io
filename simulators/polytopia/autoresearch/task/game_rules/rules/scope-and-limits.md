# Scope, outcome taxonomy and known limits

## Relevant content boundary

The native profile admits18 supported unit forms and26 technology records. Shared
tables also contain special-tribe dragons, Cymanti/Aquarion/Polaris units, technologies
and improvements. They are not reachable Bardur choices and are intentionally not
promoted into the primary catalogs. Zero/None placeholder rows are not units.
The ChurchOfConverts structure row exists but has no native monument task bit;
it is not a relevant obtainable monument. The Embassy record is a relation-backed
special command rather than an ordinary tile construction option.

Raw observations may retain full enum vocabulary, including off-mode one-hot IDs.
A channel existing is not evidence of a legal acquisition path. Special terrain/
resources are similarly distinguished from the actual generated map pool.

## Cap and guard distinction

Natural elimination occurs when a player loses its last city. At the benchmark's
round limit ([mode and maps](mode-and-maps.md)), the arena awards the higher in-game score a win and equal score
a draw, recording it as capped adjudication. This is not a natural Domination rule.
The starter trainer scores a capped game by the same rule (+1 for the higher score,
−1 for the lower,0 for equal scores): training and evaluation use one outcome rule.

**Source discrepancy:** benchmark declaration calls its10000 value `max_actions_per_turn`,
but the current native step counter increases once per accepted action and does not reset it in EndTurn. Reaching that native limit truncates
the episode globally. Treat it as a total accepted-action safety cap unless a
versioned wrapper explicitly supplies another interpretation. The wiki preserves
the source behavior and does not silently change the engine to fit the field name.

Status0 is accepted,1 invalid action,2 unsupported rule/state,3 stale state version,
4 episode already done. Rejected actions leave game state unchanged. Native
termination/cap checks override ordinary validation appropriately. A stale version,
illegal output or unsupported state is not a natural loss/draw. Resignation exists
in the catalog but is disabled by this task's mask option. No research code may
change these restrictions or private grading rules.

## What completeness means here

Every relevant declared entity has a readable entry with exact pinned fields;
chapters cover the implemented economy, movement, combat, discovery, diplomacy,
capture, score and public interface. Source files are inventoried and hashed in
scope and limitations. Source-derived documentation does not independently prove every
emergent interaction or commercial-game parity. Exact A*/push/hash/deferred-event
implementations remain source-defined; major rules and ordering are described here,
but this is not an executable reimplementation. Synthetic water-map parity is less
qualified than Bardur Drylands. Runtime freezes/checkpoint identities and private
banks remain separate and unchanged by this wiki.

[Wiki home](../README.md)
