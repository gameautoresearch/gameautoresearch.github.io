# Catalog reachability and exclusions

[Wiki home](../README.md) · [Catalog IDs](../catalog-index.md)

The wiki's universe comes from this frozen mode, not the entire multi-generation
Showdown Dex. The base set table has 509 records and 348 distinct movepool IDs.
Every one has a readable page. Struggle is added as an automatic fallback.

## Why the move catalog does not include the entire National Dex

The actual pool contains Sleep Talk and Transform. Sleep Talk invokes an eligible
move already in its user's generated/copied moveset; it does not sample arbitrary
Dex moves. Transform/Imposter copy the opposing Pokémon's current moves from this
same battle, with their explicit restrictions/PP handling. Magic Bounce and Dancer
re-execute/reflect the triggering admitted move; they do not create a new arbitrary
move ID. Relic Song, Tera Blast, Tera Starstorm, Revelation Dance, Weather Ball and
Aura Wheel can change properties/formes but keep their indexed move identity.

The following are **absent from every pinned singles template**: Metronome, Assist,
Copycat, Nature Power, Mimic, Mirror Move, Sketch, Instruct and Me First. Thus the
usual mechanisms that call a move outside the already generated current sets do
not justify silently expanding this mode's ordinary move universe. Metronome the
move and a held item with a similar name must not be conflated. Species-required
moves and Tera-role enforcement use moves already represented by the union.

## Abilities and formes

Initial role templates name 203 abilities. All are documented even when subsequent
ability culling rejects a particular combination. The actual relevant battle-form
records introduce six additional abilities: the four Ogerpon Embody Aspects, Tera
Shell and Teraform Zero. They have their own pages and are distinguished from
initial choices, giving 209 documented abilities.

Trace copies an ability from a currently present target; Imposter applies the
Transform rules. Neither samples the arbitrary full-Dex ability catalog. Doodle,
Role Play, Simple Beam, Worry Seed, Skill Swap and Entrainment are all absent from
the initial move union. Species-specific form transitions in the public native
metadata have 26 form records, all included among the wiki's species entries.
The 68 additional species pages also include generated cosmetic/start-form variants.
Cosmetic appearance choices retain the selected template's ability; they do not
unlock every ability listed in a different-form Dex record. For this reason unrelated
Friend Guard/Hospitality/Run Away/Sweet Veil/Symbiosis options are not mislabeled
as new battle-induced abilities merely because a related cosmetic form lists them.

## Items

The 59-item set is a conservative union of item literals in the two singles selector
methods plus required form items, with the explicitly doubles-only Clear Amulet
branch and the initially unselectable Gluttony berry array removed. Required Arceus
items use only the pinned generator's first-entry Plate convention. Empty item on
an Acrobatics set is an intentional state. This is a static candidate superset,
not proof that every selector branch can be realized with a current role/moveset.

Trick, Switcheroo, Knock Off, Thief-like effects, item-stealing abilities, Harvest
and similar effects transfer, remove, consume or restore existing held items under
their described conditions. They do not establish an arbitrary externally supplied
item pool. The initial item universe is kept explicit so the static reachability
claim can be audited against the generator instead of inferred from finite sampled
teams. No random-team generation or gameplay was run to produce the inventory.

## Strength of the claim

These are direct input and named-effect reachability checks, not a formal proof of
all event-program paths. A full transitive symbolic analysis of every effect rule,
including possible condition/effect aliases, remains an explicit semantic-audit
limit in [scope and limitations](../scope-and-limitations.md). The wiki includes exact effect rules and
source hashes to retain exceptional behavior rather than claim that a static ID
count proves all mechanics. Engine/format changes require regenerating and reviewing
this scope instead of treating this document as a future-version promise.
