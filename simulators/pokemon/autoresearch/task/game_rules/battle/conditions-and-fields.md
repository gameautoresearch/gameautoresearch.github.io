# Status, hazards, weather, terrain and side effects

[Wiki home](../README.md) · [All conditions](../conditions/README.md)

## Major status and volatile effects

A Pokémon normally has at most one major status: burn, paralysis, sleep, freeze,
poison or bad poison. Volatile effects such as confusion, Leech Seed, Taunt,
Substitute, Encore and trapping coexist separately. Major statuses generally
persist on switching; most volatiles clear unless a specific passing effect says
otherwise. Type/ability/terrain/status-prevention checks can block induction.

| Effect | Ordinary Gen 9 behavior |
|---|---|
| Burn | 1/16 base-max-HP residual chip; half physical move damage except applicable Guts/Facade exceptions |
| Paralysis | Half effective Speed; 25% chance action fails |
| Poison | 1/8 base-max-HP residual chip |
| Toxic | Increasing 1/16-based chip, stages 1–15; escalation resets on switching |
| Sleep | Ordinarily 1–3 blocked opportunities; waking checked before action; Rest/Sleep Talk and Early Bird have explicit rules |
| Freeze | 20% thaw chance before action; Fire/defrost effects can thaw |
| Confusion | Volatile; 33% self-hit check at 40 typeless physical power while timer remains |
| Flinch | Prevents an action still pending that turn; does not undo a completed action |

Fire normally blocks burn, Electric paralysis, Ice freeze, and Poison/Steel poison;
abilities such as Corrosion and individual effects change some of these rules.
Grass types, Overcoat and Safety Goggles can block powder effects. Dark targets
block relevant opponent Prankster-boosted status moves. Safeguard, Misty/Electric
Terrain and abilities impose separate status checks; no single immunity flag
substitutes for their conditions. See each indexed condition/effect for exceptions.

## Entry hazards

Entry hazards belong to one side and trigger on incoming Pokémon under their
respective conditions. Stealth Rock damage is 1/8 max HP times Rock effectiveness.
Spikes only affect grounded targets and deal 1/8, 1/6 or 1/4 at one/two/three layers.
Toxic Spikes have one or two layers, poison/badly poison grounded susceptible targets,
and are absorbed by a grounded Poison-type entrant under the normal rules.
Sticky Web lowers a grounded entrant's Speed by one stage. Flying/Levitate,
forced grounding, Heavy-Duty Boots and item/ability suppression alter applicability.

Rapid Spin, Mortal Spin, Defog, Tidy Up, Court Change and other removal/transfer
mechanics are not equivalent: they differ in targets, damage, blocked-hit conditions,
field/screen effects and which side's hazards change. Their move pages specify each.
Magic Bounce can reflect supported hazard placement; it does not remove already
present hazards merely by switching in.

## Weather

Normal rain, sun, sandstorm and snow last five turns; relevant rocks extend to eight
when the weather-setting rules use them. A new setting can replace weather; suppression
such as Cloud Nine/Air Lock differs from ending weather and can disable weather-based
modifiers while a weather timer still exists.

- Rain: Water damage ×1.5; Fire ×.5. Relevant Thunder/Hurricane accuracy,
  Electro Shot charging, Swift Swim, Rain Dish, Dry Skin and Hydration change.
- Sun: Fire ×1.5; Water ×.5. Solar charging moves, Growth, Synthesis-style healing,
  Chlorophyll, Solar Power, Harvest, Protosynthesis and other effects change.
- Sandstorm: 1/16 chip to susceptible Pokémon and Rock Special Defense ×1.5;
  type/ability/item immunities and Sand Rush/Force/Veil matter.
- Snow: Ice Defense ×1.5; **no old hail chip** in this Gen 9 mode. Blizzard,
  Aurora Veil, Slush Rush, Ice Body and Ice Face interact with it.

Weather Ball changes type/power based on weather and its suppression rules.
Ancient primal weather conditions present in shared engine data are not generated
by this ordinary Gen 9 singles pool.

## Terrain and global rooms

Terrain applies its specified effects to grounded Pokémon; Flying typing, Levitate,
Air Balloon, Gravity, Iron Ball and grounding moves can change eligibility. Most
terrain lasts five turns, extended by Terrain Extender to eight where applicable.

- Electric boosts grounded Electric attacks ×1.3, blocks sleep of grounded targets,
  and activates Quark Drive; individual moves get additional changes.
- Grassy boosts grounded Grass attacks ×1.3, heals grounded Pokémon by 1/16 each
  residual turn and weakens the specified Earthquake/Bulldoze/Magnitude-style moves.
- Psychic boosts grounded Psychic attacks ×1.3 and prevents supported opposing
  priority attacks against grounded targets; it is not a universal ban on priority.
- Misty prevents new major status/confusion for grounded targets and halves supported
  Dragon damage against grounded targets; it does not cure an existing status.

Trick Room reverses Speed ordering within priority brackets for its five-turn duration;
Magic Room suppresses held-item effects; Wonder Room swaps the applicable Defense/
Special Defense treatment; Gravity changes grounding and accuracy and restricts
certain moves. The indexed move pages give exact duration, priority and failure rules.
Screens and Aurora Veil reduce appropriate damage (normally half in singles), with
critical, infiltration, screen-breaking and Light Clay exceptions. Tailwind doubles
its side's Speed for four turns. Reflect/Light Screen last five turns, or eight
with Light Clay. Safeguard, Mist, Wish and delayed attack effects retain their own
timers and event stages; do not assume every side condition lasts five turns.
