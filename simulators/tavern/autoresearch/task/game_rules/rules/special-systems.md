# Trinkets, Dark Gifts and conditional systems

[Wiki home](../README.md)

## Trinkets

Ordinary lesser and greater offers occur at turns 6 and 9, respectively, four
choices each. Each offered trinket has a visible price, and choosing it applies
its persistent/scripted effect; acquiring a trinket differs from casting a spell.
The full current catalog is split into [lesser](../catalog/trinket-lesser.md) and
[greater](../catalog/trinket-greater.md), with any copied/transformed extra definitions
kept in related catalogs. Some heroes/effects change offer timing, add trinkets,
copy an owned instance or replace one; those exceptions are in their entries.

Type trinkets ordinarily need two matching minions (lesser) or three (greater),
counting board plus hand. Association and lobby subset must also fit. Discountable
“pivot” trinkets can be offered when outside the type with a two-Gold price cut;
not every type trinket is eligible for this exception. Menagerie requires three
distinct board types; an offer has at most one menagerie trinket and also respects
the source's two exclusive groups. Per-card `offer_ok`, hero exclusions, ownership
and dynamic offer parameters further constrain choices. The catalog's type metadata
is only one part of those rules.

Unlisted base offer weights are 100. Source overrides include Rusty's trinket IDs
with weights 25/50/150/200/250, and the “more likely” flags multiply by default
`trinket_likely_pct=200%`. That multiplier is an emulator assumption inferred from
client offers, not a located published server probability. Offer preparation
parameterizes newly unowned choices from the current visible board/hand, rather
than reusing stale parameters from an earlier declined offer. Viewing a prepared
menu itself does not consume new randomness.

## Dark Gift button and offers

The button unlocks at turn 3, costs three Gold, can be used once per turn and
three times per game, and requires hand room. It pays immediately and opens a
three-option Discover of minions bearing different attached Dark Gifts. The gift
is an attachment/effect, not simply a free-standing statless card to buy later.
Availability is checked against the gift universe and current turn window.

| Recruit turn | Offered minion tiers by option | Type affinity |
| ---: | --- | --- |
| 1–2 | none | button unavailable |
| 3 | 2 / 2 / 2 | unrestricted |
| 4 | 2 / 2 / 3 | unrestricted |
| 5 | 3 / 3 / 3 | unrestricted |
| 6 | 3 / 4 / 4 | first most-common type |
| 7 | 4 / 4 / 4 | first two most-common type |
| 8 | 4 / 5 / 5 | first two most-common type |
| 9 | 4-or-5 / 5 / 6 | first two most-common type |
| 10+ | 5 / 5-or-6 / 5-or-6 | first two most-common type |

The ambiguous turn-9 tier and later tier-6 choices use 50% source assumptions.
Four rare gifts use 50% ordinary weight by default; Harpy's Talons has a twofold
Rally affinity multiplier. Gift/card compatibility and exclusions further filter
options. All [47 gift effects](../catalog/dark-gifts.md) have entries with their
window/keyword conditions. Exact probabilities should not be treated as live-server
facts. Gilding, persistent rebirth and attached effects are not interchangeable
with ordinary triples or normal Reborn.

## Deities and Aberration content

The lobby chooses one of the two Deity definitions and one eligible Dark Paradox
variant. A player's Deity is a persistent stat/effect state, distinct from an
ordinary always-on-board minion. Four friendly deaths is the default awakening
threshold. Cards can buff Deity stats before awakening; after it is active in combat,
`buff_deity` also updates the summoned body. C'Thun's awakening distributes its
attack/health stats across other friendly minions, evenly with random distinct
recipients receiving remainder points. A golden multiplier and trinket-specific
Reborn can alter that process. Individual Deity definitions and linked tokens are
in the complete card index.

## Timewarp is conditionally relevant

The global two scheduled Timewarp visits are disabled for this benchmark. However,
Morchie/Murozond powers still open their own visit at their **printed fixed game
turn** (not a delay after power acquisition), and effects may discover Timewarp
content. Thus [253 Timewarp definitions](../catalog/conditional/timewarp.md) are
part of an offline reference, explicitly marked conditional. The visit uses Chronum,
a prepared offer menu and a leave option; it is not a normal three-Gold shop refresh.
Default visit grant is two Chronum; relevant type thresholds are two/three for
minor/major offers. Saved Chronum and visit-only hand/leave rules are represented
by the owning power and pending-menu source.

## Tier 7, Buddies, quests and generated cards

Regular tier upgrades stop at four in this form. An effect that unlocks tier 7 in the
eight-player game raises the cap by one here (to tier 5), and a tier-5 tavern draws only
the tier 1–4 pool. [Tier-7 definitions](../catalog/conditional/tier-7.md)
remain potentially accessible through supported special effects; mere pool metadata
does not mean a tier-one player can buy them. The global experimental Quest/Buddy
system is disabled. Nonetheless an enabled hero/trinket can discover a Buddy or
produce a special reward/card. The related catalog records explicit reachable data
and conservative script dependencies without claiming all appear in a normal shop.
No whole alternate game mode is silently enabled to make those definitions relevant.

