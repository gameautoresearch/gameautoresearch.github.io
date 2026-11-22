# Keywords, trigger stages and exceptions

[Wiki home](../README.md)

## Generic combat keywords

| Keyword | Pinned engine meaning |
| --- | --- |
| Taunt | Ordinary attackers choose among visible legal Taunts before other visible targets. |
| Divine Shield | Positive damage removes one shield hit instead of health; when its count reaches zero ShieldLost fires. No damage means no Venomous consumption from that hit. Some effects give multiple shield hits. |
| Reborn | After death, summon a fresh normal/golden definition at one HP plus applicable “wherever it is” aura HP, without ordinary enchantments, Magnetic parts or Dark Gifts; remove Reborn on replacement. Shield can reset on its fresh body. |
| Persisting/Perfect Reborn | Special effect retains full stats, keywords and attachments; this is explicitly different from normal Reborn. |
| Poisonous | Positive unshielded damage marks the victim destroyed, independent of its remaining HP. The Poisonous keyword is not consumed like Venomous. |
| Venomous | Positive unshielded damage destroys the victim and ordinarily consumes the source's keyword. Cleave can retain it for all targets in one exchange. |
| Windfury / Mega-Windfury | Two / four attacks in one attack turn, with fresh targets and survivability checks. |
| Stealth | Cannot be chosen as an ordinary target. Drops from an attacker before Rally/attack triggers. A genuine Stealth targeting deadlock can tie naturally. |
| Cleave | The attack also damages adjacent defending minions. It is not a separate attack turn against each neighbor. |
| Immune | Damage to the body does nothing, rather than consuming a shield. |
| Immune while attacking | Suppresses return damage while the body attacks; it does not automatically make it immune while defending. |
| Magnetic | Can attach through the separate Magnetize action to a legal host, carrying stats/keywords/attached effects as specified. |

The engine also recognizes mechanic tags such as Battlecry, Deathrattle, Avenge,
Rally, Spellcraft and Activate. A metadata tag alone is not a substitute for a
supported script. The card catalogs retain the pinned English text and list all
metadata keywords, including informational tags not themselves combat flags.

## Trigger vocabulary

- **Battlecry:** ordinary play or an explicit Battlecry-trigger effect. It may need
  a target or Choose-One branch. Summoning a minion is not by itself playing it.
- **Deathrattle:** after its body dies in the death-resolution sequence; attached
  and gained deathrattles can contribute. Effects can multiply/order triggers.
- **Rally:** before the attacking body's damage, receiving the defender as target
  context. Host and attached Rally scripts run with their own golden identity.
- **Avenge (N):** friendly deaths advance the listener's counter to its normal or
  golden threshold. A summon, sale or destroyed hand card is not automatically a
  combat friendly death for every listener.
- **Start of Combat:** runs in the staged grouping described in [combat](combat.md),
  with separate hand/board hooks and the first attacker's side first per group.
- **Start/End of Turn:** recruit lifecycle hooks, not an arbitrary neural step.
- **Activate:** a paid board action with its cost, target requirement and activation
  counters. Not every minion has this action; the legal mask admits only usable ones.
- **Persistent player event:** held trinket/reward/power effects subscribe even
  without a normal board body. Their relevant player-state counters differ from
  per-minion instance counters.
- **On attach/create/merge:** initialization or transfer logic. Adding a buff,
  summoning, playing, creating a golden entity and acquiring a trinket are different
  events with potentially different listeners.

## Events and state distinctions

Played, Summoned, Bought, Sold, SpellCast, Refreshed, Attacks, AttackDone, Damaged,
Died, Reborn and ShieldLost are distinct events, not synonyms. Scripts subscribe
to masks; the per-card Runtime triggers/Subscribed events annotations summarize
those registrations. Effects that listen in hand are explicitly marked in source;
there is no universal rule that all card text is active in every zone. Damage
triggers are deferred during a simultaneous combat exchange then flushed. Nested
summons/deaths/Battlecries/discovers can open follow-up effects or menus.

“Permanent,” “this turn,” “this combat,” “wherever it is,” tavern bonus, attachment,
hand bonus and player-wide bonus are separate storage concepts. A normal Reborn
loses enchantments while a persistent aura may immediately add stats back. Golden
forms inherit a normal script only if they lack an explicit golden override;
exact golden card numbers are printed separately. Card text that says set a value
can lower an existing value (Armor Stash sets armor to five, not “gain until five”).

## Source-defined examples worth retaining

- Volumizer attachments do not keep receiving later Volumizer bonuses by default.
- Elemental of Surprise can supply an Elemental triple wildcard, not a universal
  third copy for unrelated types.
- Avenge/Deathrattle/Rally multipliers are separate, and attached scripts carry
  their own golden status rather than blindly taking the host's status.
- A spell cast by an effect does not count as a player hand cast by default.
- A new offer's dynamic parameter is captured during offer preparation; merely
  looking at an existing prepared menu does not draw fresh random parameters.

These rules explain the engine's supported semantics. The wiki does not promise
that every interaction among 1,000+ related bodies has a hand-written bespoke
explanation; per-card effects plus source stage ordering are the offline basis.
