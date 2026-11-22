# Related / Trinket Lesser

[Wiki home](../../README.md) · [Complete card index](../index.md)

3 definitions, generated from pinned client 37.0.2.256589 / rules 37.0.2.

Card text below is the pinned English rule description, with markup removed. Counts/statistics are exact bundle fields; dynamic buffs and scripted exceptions can change them. Minion bundle `cost` is not its Tavern buy price (base buy price is 3 Gold). Conditional/related entries are not automatically ordinary shop offers.

## Blessing Portrait — `BG32_MagicItem_894`

### bg32-magicitem-894

**Access:** bundle reachable/generated relation. **Kind:** trinket_lesser. **Card embedding index:** 1822 (database index 1821).

| Tier | Attack | Health | Armor | Bundle cost | Minion types |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 0 | 2 | none |

**Effect:** Get a Natural Blessing. At the start of each turn, get another. Your Natural Blessings also affect minions in hand.

**Keywords/mechanics:** AURA, TRIGGER_VISUAL.

**Offer/effect conditions:** assoc=['MURLOC'].

## Putricide Sticker — `BG32_MagicItem_300`

### bg32-magicitem-300

**Access:** bundle reachable/generated relation. **Kind:** trinket_lesser. **Card embedding index:** 1764 (database index 1763).

| Tier | Attack | Health | Armor | Bundle cost | Minion types |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 0 | 0 | none |

**Effect:** Craft a custom Undead. At the start of every 2 turns, repeat this. (2 turns left!)

**Keywords/mechanics:** TRIGGER_VISUAL.

**Offer/effect conditions:** assoc=['UNDEAD'].

**When it applies:** When effect/power is attached; Updates numbers displayed in card text; At next recruit turn start.

## Spell-powered Wrench — `BG32_MagicItem_170`

### bg32-magicitem-170

**Access:** bundle reachable/generated relation. **Kind:** trinket_lesser. **Card embedding index:** 1739 (database index 1738).

| Tier | Attack | Health | Armor | Bundle cost | Minion types |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 0 | 0 | none |

**Effect:** After you play a Magnetic minion, cast Repair Job on a random friendly Mech.

**Keywords/mechanics:** TRIGGER_VISUAL.

**Offer/effect conditions:** assoc=['MECHANICAL'].

**When it applies:** On subscribed engine events.

**Subscribed events:** CARD_PLAYED.
