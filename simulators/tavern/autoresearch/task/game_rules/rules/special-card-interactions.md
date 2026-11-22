# Special card interactions and contextual choices

[Wiki home](../README.md)

Some generated entries are menu selections or temporary labels. Their gameplay meaning depends on the power or choice that created them. The following sections explain those interactions.

## Combat-retention Dragons

**Tarecgosa (`BG21_015`)** and **Persistent Poet (`BG29_813`)** retain gains through a dedicated combat-retention rule. Tarecgosa retains its own
combat stat/Bonus-Keyword gains. Poet grants retention to adjacent Dragons.
Timewarped Poet affects all friendly Dragons, including itself. A golden retention
source doubles positive combat stat gains; negative gains are unchanged. Doubling
applies immediately to the fighting body as well as its retained recruit copy.
Multiple retention sources take the **largest multiplier**, not their product.
The Tarecgosa's Blessing Dark Gift also supplies multiplier two. The ordinary/golden
catalog entries print the exact base stats and their separate effect text.

## Generated second-power cards

`BG34_HeroPowerSpell_*` “Power of …” Timewarp tokens activate when bought and replace/add the second
hero-power slot with the power named in their text. They are neither normal shop
hero offers nor ordinary minion bodies. The catalog resolves these loop registrations
and links to their power definitions. Some powers have extra timing/cooldown state;
acquisition initializes it through the same attached-power machinery.

## Galewing flightpath choices

`BG20_HERO_283p_t1` Westfall and `BG20_HERO_283p_t2` Ironforge are **menu choices**.
The Discover choice consumes the selection, stores the path and starts a
one-/two-turn counter. At zero, Westfall gets a random one-cost Tavern spell at
or below the current tavern tier; Ironforge gains two Gold. The three-turn path
instead Discovers a minion at the current tavern tier. They do not stay in hand
as castable spells. Flight Trainer doubles path rewards (golden triples), using
the largest applicable Buddy multiplier.

## Bru'kan invocation choices

`BG22_HERO_001p_t1_s` through `_t4_s` are selection labels for Earth, Fire, Water
and Lightning. The source's Discover continuation consumes the label and stores
its ordinal in Bru'kan's power. The power/companion logic later applies it:
Earth gives four random friendly minions the printed elemental-summoning Deathrattle;
Fire doubles leftmost attack; Water gives rightmost Divine Shield/Taunt; Lightning
deals one to five random enemy bodies. These labels do not require independent
castable-spell registrations. Source owners: `content/mod.rs`, `heroes_combat.rs`,
`buddies_a.rs` and the associated hero-power entry.

## Cariel improvement choices

These labels change stored Conviction power values, rather than remaining cards:
`BG21_HERO_000p_t1` adds one target; `BG21_HERO_000_Buddyt`, `...Buddyt3`,
`...Buddy_Gt` add 1/2/4 attack; `...Buddyt2`, `...Buddyt4`, `...Buddyt_Gt2`
add 1/2/4 health. Each Conviction improvement adds to the corresponding stored counter.
Captain Fairmount's end-turn effect picks random improvements (golden twice).
The generated card's printed effect and its parameter are therefore more informative
than looking for a standalone `spell` effect rule.

## Prediction labels and diagnostic aliases

Murloc Holmes's `BG23_HERO_303pt` “No Minions” is a guess label, not an obtainable
minion. Correct guesses give a Tavern Coin; the No-Minions label must not be copied
as a minion by the Buddy. `TB_BaconShop_HP_008a` is the Gold Coin label used by
power/context handling; its one-Gold effect is not a normal shop minion or generic
printed bundle mana cost. Actual enabled Tavern Coin definitions have their
registered gain-Gold effect.

`BG36_360` / `_G` Dark Paradox base definitions describe the family. Lobby generation
selects one explicit tier/stat/Gift variant; the base placeholder itself must not
be read as a freely available additional shop minion. `BG_OldGod` is the secret
Deity UI/diagnostic representation. The actual selected Deity state/awakening
system is described in [special systems](special-systems.md), not implemented by
casting that UI definition.

## Explicitly out-of-scope dependency

Timewarped Ultralisk `BG34_Treasure_994` and its golden form appear in source
StarCraft classification tables, but the current Timewarp implementation explicitly
marks Ultralisk outside the enabled offer pool. The conservative dependency walker
therefore excludes those two bodies from the game-mode catalog. A string constant
being referenced by a filter is not proof it is reachable in an admitted game.
