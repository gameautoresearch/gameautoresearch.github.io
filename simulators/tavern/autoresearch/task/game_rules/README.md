# Battlegrounds offline game-mode wiki

This is the **AutoTavern 37.0.2 implemented-content four-player small benchmark**
(four players, regular tavern tiers 1–4, three minion types per lobby),
client data 37.0.2.256589. It is a frozen emulator variant, not the current live
Blizzard service. Everything needed to read this wiki is local; no Internet is
required. Start with the mode and core rules, then look up individual entities.

## Read the game

1. [Mode, lobby and supported content](rules/mode-and-lobby.md)
2. [Recruit economy, pool, triples and card play](rules/recruit-and-economy.md)
3. [Trinkets, Dark Gifts, Deities and conditional systems](rules/special-systems.md)
4. [Combat, damage, pairing and elimination](rules/combat.md)
5. [Keywords, trigger stages and effect timing](rules/keywords-and-events.md)
6. [Visible observations and every action ID](rules/agent-interface.md)
7. [Results, guard failures and release differences](rules/outcomes-and-guards.md)
8. [Contextual choices and hard-coded card exceptions](rules/special-card-interactions.md)
9. [Provenance, generation and coverage limits](scope-and-limitations.md)

## Look up every relevant definition

- [Complete searchable card index](catalog/index.md): normal forms, golden forms,
  powers, tokens, generated cards, conditional pools and embedding indices.
- Ordinary shop minions: [tier 1](catalog/minions/tier-1.md),
  [tier 2](catalog/minions/tier-2.md), [tier 3](catalog/minions/tier-3.md),
  [tier 4](catalog/minions/tier-4.md). Tiers [5](catalog/minions/tier-5.md) and
  [6](catalog/minions/tier-6.md) are above this form's regular tier cap and are not in
  the shop pool; they are kept because effects that name them exist.
- [Heroes and their linked powers](catalog/heroes.md)
- [Tavern spells](catalog/tavern-spells.md)
- [Lesser trinkets](catalog/trinket-lesser.md) and
  [greater trinkets](catalog/trinket-greater.md)
- [Dark Gifts](catalog/dark-gifts.md)
- Conditional content: [Timewarp](catalog/conditional/timewarp.md) and
  [tier 7](catalog/conditional/tier-7.md). A disabled **global** system is not proof
  that its cards cannot be produced by an enabled hero or card effect.
- Related/generated definitions: [minions and golden forms](catalog/related/minion.md),
  [hero powers](catalog/related/hero-power.md), [special spells](catalog/related/spell.md),
  [additional Tavern spells](catalog/related/tavern-spell.md),
  [lesser](catalog/related/trinket-lesser.md) / [greater](catalog/related/trinket-greater.md)
  trinkets, [special reward](catalog/related/quest-reward.md).

Each entry has its string ID, visible embedding index, kind, tier, attack, health,
armor, bundle cost, minion types, pinned English effect description, keywords,
conditions and linked normal/golden/power/token relations. Static registry annotations
add trigger stages, targets and exact implementation locations. Base minion purchase
price is **3 Gold**, regardless of its unrelated bundle cost metadata.

## Scope

See [scope and limitations](scope-and-limitations.md) for catalog counts and known limits. Ordinary offers, conditional effects and related definitions remain clearly distinguished.
