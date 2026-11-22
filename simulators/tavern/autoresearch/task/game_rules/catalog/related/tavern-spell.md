# Related / Tavern Spell

[Wiki home](../../README.md) · [Complete card index](../index.md)

5 definitions, generated from pinned client 37.0.2.256589 / rules 37.0.2.

Card text below is the pinned English rule description, with markup removed. Counts/statistics are exact bundle fields; dynamic buffs and scripted exceptions can change them. Minion bundle `cost` is not its Tavern buy price (base buy price is 3 Gold). Conditional/related entries are not automatically ordinary shop offers.

## The Goldenizer — `BG26_813t`

### bg26-813t

**Access:** bundle reachable/generated relation. **Kind:** tavern_spell. **Card embedding index:** 651 (database index 650).

| Tier | Attack | Health | Armor | Bundle cost | Minion types |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 0 | 0 | none |

**Effect:** Make a friendly minion Golden.

**When it applies:** When its spell resolves.

**Target classes:** FriendlyNonGoldenMaxTier. Exact legality remains mask-controlled.

## Pointy Arrow — `EBG_Spell_014`

### ebg-spell-014

**Access:** bundle reachable/generated relation. **Kind:** tavern_spell. **Card embedding index:** 3475 (database index 3474).

| Tier | Attack | Health | Armor | Bundle cost | Minion types |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0 | 0 | 0 | 1 | none |

**Effect:** Give a minion +4 Attack.

**When it applies:** Updates numbers displayed in card text.

## Arcane Absorption — `BG35_911`

### bg35-911

**Access:** bundle reachable/generated relation. **Kind:** tavern_spell. **Card embedding index:** 2645 (database index 2644).

| Tier | Attack | Health | Armor | Bundle cost | Minion types |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 0 | 0 | 0 | 1 | none |

**Effect:** Give a friendly Elemental half the stats of the highest-Health minion in the Tavern.

**Offer/effect conditions:** assoc=['ELEMENTAL']; lobby_req=['ELEMENTAL'].

**When it applies:** When its spell resolves.

**Target classes:** FriendlyRace. Exact legality remains mask-controlled.

## Conflagration — `BG35_910`

### bg35-910

**Access:** bundle reachable/generated relation. **Kind:** tavern_spell. **Card embedding index:** 2644 (database index 2643).

| Tier | Attack | Health | Armor | Bundle cost | Minion types |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 0 | 0 | 0 | 2 | none |

**Effect:** Give a minion +4/+4. (Improved by each Elemental you played this turn!)

**Offer/effect conditions:** assoc=['ELEMENTAL']; lobby_req=['ELEMENTAL'].

**When it applies:** Updates numbers displayed in card text; When its spell resolves.

**Target classes:** Friendly. Exact legality remains mask-controlled.

## Knockoff Wisdomball — `BG30_802`

### bg30-802

**Access:** bundle reachable/generated relation. **Kind:** tavern_spell. **Card embedding index:** 1148 (database index 1147).

| Tier | Attack | Health | Armor | Bundle cost | Minion types |
| ---: | ---: | ---: | ---: | ---: | --- |
| 6 | 0 | 0 | 0 | 4 | none |

**Effect:** Your next 2 Refreshes are helpful! (2 left!)

**Keywords/mechanics:** TRIGGER_VISUAL.

**Offer/effect conditions:** quest_goal=2.

**When it applies:** When its spell resolves.
