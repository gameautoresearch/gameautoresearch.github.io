# Mode and lobby setup

[Wiki home](../README.md)

The benchmark is a **four-player** solo Battlegrounds lobby in the implemented-content mode:
normal offered content must have a registered effect script or a fully generic
keyword/vanilla implementation. `AllVanilla`, which ignores missing printed effects,
is not the task. A lobby starts at hero selection, turn one, regular tavern tier one.
Regular tavern upgrades stop at **tier 4** (`max_regular_tier` 4): tier 5 and 6 minions
are not in the pool. Players are offered two heroes. A warband has at most seven minions; a hand at most
ten cards. There is no policy control during ordinary combat.

## Selecting the lobby's types

Three types are chosen uniformly without replacement from the engine's ten-type
normal lobby list: Beast, Demon, Dragon, Elemental, Mech (`MECHANICAL` in data),
Murloc, Pirate, Quilboar, Undead and Aberration. The type bitmask also has Naga
for generated/content effects, but **Naga is not in the lobby’s optional normal
lobby-type draw**. Do not mistake every defined type for a normally selected type.
All-type cards count as intersecting any type. Typeless minions remain eligible
unless their separate association/subset conditions require a missing type.
Aberration is not forced by default. No debug `lobby_types` override is implied.

A minion's tribe, association and lobby-required subset are distinct metadata.
Ordinary minions need a relevant type/association, implementation, allowed pool tag
(`pool_tag <= 1`), and pool membership; the `strict_pool_tag` rule removes unpublished
nonstandard candidates such as Rot Hide Gnoll from normal offers. The docs distinguish
its presence as an engine definition from actual ordinary-shop availability.

## Hero availability

The 116-hero catalog is the cross-lobby candidate set, not 116 choices in a game.
Type-associated heroes need a matching lobby type; `lobby_req` adds a separate
subset constraint. Powers must be scripted. Hero-type filters and lobby bans are
part of eligibility, not model choices. Guaranteed new heroes Drest'agath and
Kith'ix are moved into distinct players' offers only when eligible; no incompatible
hero is conjured merely to meet that guarantee. Exact hero HP, armor and powers
are recorded per entry, not assumed equal across heroes.

Lobby exclusions encoded in `flow.rs`: Murozond, Morchie, Sylvanas and Rafaam are
banned with Aberration; Gallywix and Zephrys with Elemental; Greybough with Mech;
Tamsin and Illidan with Dragon; Mutanus with Naga. In an Aberration/Y'Shaarj lobby,
additional bans apply to the listed hero IDs in the source. Hero-pool debug overrides
bypass some filters, but are not this benchmark's setup.

## Dark Paradox and Deity selection

One Dark Paradox variant is selected per lobby. Forms are not freely interchangeable:
the tier-5 form is excluded with Undead; tiers 5 and 6 are excluded with Y'Shaarj
in Aberration lobbies. Variant tier is read from data, not inferred from its ID
suffix. A Deity is drawn from the source's two Deity definitions; availability and
awakening interact with Aberration content. See [special systems](special-systems.md).

## Global systems and mode boundaries

Global Timewarped Tavern visits are off (`timewarp_enabled=false`), as is the
experimental Quest/Buddy/extra-action mode. Enabled heroes still may create their
own Timewarp visit, discover a Buddy or produce cards outside the normal shop.
Tier 7 is not a regular upgrade, but relevant hero/effect access is
not erased. Those definitions are kept in conditional catalogs rather than presented
as normal offers. Cosmetic, historical and unrelated database definitions are
excluded from the documented dependency set.

