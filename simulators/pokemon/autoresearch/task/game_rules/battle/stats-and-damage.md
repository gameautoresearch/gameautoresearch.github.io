# Stats, accuracy, damage and integer rounding

[Wiki home](../README.md) · [Type chart](type-chart.md)

## Stat construction

For ordinary species at level L, base stat B, IV I and EV E:

- HP = floor((2B + I + floor(E/4)) × L / 100) + L + 10.
- Other stat = floor((floor((2B + I + floor(E/4)) × L / 100) + 5) × nature).
- Nature is 1.1, .9 or 1 where applicable. The pinned normal singles generator
  does not assign a nonneutral nature. Species-specific fixed-HP exceptions use
  their rule rather than the ordinary HP formula.

Generated EV/IV adjustments are documented in [team generation](../mode/team-generation.md).
Actual opposing stats are hidden even though species base stats are public data.
Stat stages are clamped −6 through +6. At nonnegative stage s, ordinary Attack,
Defense, SpA, SpD and Speed use (2+s)/2; at negative s use 2/(2−s), with engine
integer rounding. Critical hits ignore detrimental attacker stages and beneficial
defender stages where applicable; Unaware and move overrides differ from critical
hits and have their own rules. Effective Speed additionally includes paralysis,
Tailwind, weather abilities, Choice Scarf, boosts and other event modifiers.

## Accuracy

The baseline move's percentage is not its final hit probability. Accuracy/evasion
stages use the analogous three-based factors: (3+s)/3 for nonnegative net stage,
3/(3−s) for negative. Ability/item/weather exceptions and always-hit moves can
bypass or change this calculation. A move with `accuracy: true` bypasses ordinary
accuracy calculation but can still fail through immunity, protection or a move-specific
condition. Some multi-hit moves recheck accuracy per strike; others do not. Read
individual Population Bomb, Triple Axel and similar entries rather than assuming
one shared multi-hit rule.

## Ordinary damage pipeline

First test relevant immunities and special damage paths. OHKO, level damage,
fixed-damage, percentage/current-HP moves and effect rules can replace the normal formula.
For a normal damaging move, derive current base power, actual offensive/defensive
Pokémon and stat, and critical chance. Default crit chance is 1/24; raised crit
stages give 1/8, 1/2 and guaranteed critical hits, subject to crit-prevention effects.
Physical usually uses Attack versus Defense; special uses SpA versus SpD. Body
Press, Foul Play, Psyshock and related moves explicitly override a participant/stat.

The engine's premodifier base damage is:

`floor(floor(floor(floor(2*L/5+2) * power * attack) / defense) / 50)`

Then add two. Relevant modifiers occur in this order in Gen 9:

1. Special multi-hit/spread branches where applicable (ordinary singles has no
   spread-target penalty).
2. Weather damage event (typically rain/sun Fire/Water factors).
3. Critical multiplier, normally 1.5, with truncation.
4. Random damage roll: an integer percentage from 85 through 100.
5. STAB/Tera STAB using fixed-point modifiers.
6. Type effectiveness, repeatedly doubling or halving with truncation.
7. Burn halves physical damage unless its exception applies.
8. Final ModifyDamage event, including relevant item/ability/screen effects.
9. Explicit protect-bypass modifier if applicable, minimum-damage behavior and
   final 16-bit truncation.

The individual effect can modify base power, stats or damage at a different stage;
Life Orb's final damage modifier is not equivalent to adding 30% to the base stat.
Showdown uses 4,096-based fixed-point chained modifiers and Pokémon-specific
rounding in its modify helpers; do not replace all of these with one floating-point
multiply and final floor. A nonimmune ordinary damaging hit is normally at least
one damage after final modifiers, but the late 16-bit truncation has its own edge
behavior. Effect pages retain exact effect rule factors and guards for offline inspection.

## HP costs, healing and indirect damage

Drain is based on HP actually lost, not unconstrained calculated damage. Recoil,
Substitute, Belly Drum, status damage, Life Orb, healing and hazard damage use their
own base/max/current-HP quantity and rounding. Some use half-up or half-down fixed
point; others floor; minimum one damage/heal and suppression exceptions matter.
The move/item/ability's full description states the relevant fraction and exception.
The attack damage formula must not be applied to all indirect damage.
