# Combat, conversion, splash and infiltration

## Exact integer damage calculation

HP is converted to tenths (rounded), attack to hundredths (rounded), defense to
tenths and the defense bonus to tenths (floored with the source's small epsilon).
Use integer truncating division in the following calculation; this preserves
breakpoints that the familiar floating approximation can miss.

| Step | Exact arithmetic rule |
|---|---|
| Scale attack | Let A be the attack stat multiplied by 100 and rounded to an integer. |
| Scale defense bonus | Let B be the floor of ten times the defense bonus plus 0.001. |
| Scale defense | Let D be the defense stat multiplied by ten, rounded, then multiplied by B. |
| Scale health | Multiply each current and maximum HP by ten and round; call these hₐ, mₐ, hᵈ and mᵈ. |
| Attacking force | Fₐ is the floor of hₐ × A × 100 ÷ mₐ; use at least 1 for the divisor. |
| Defending force | Fᵈ is the floor of D × hᵈ × 100 ÷ mᵈ; use at least 1 for the divisor. |
| Total force | T = Fₐ + Fᵈ. If T is zero or negative, all damage is zero. |
| Main hit in HP tenths | First floor A × Fₐ × 450 ÷ (T × 1,000), then round to the nearest multiple of ten, with ties upward. |
| Retaliation in HP tenths | Zero if the main hit kills the defender or B is zero or negative. Otherwise first floor Fᵈ × D × 4,500 ÷ (T × 1,000 × B), then round to the nearest multiple of ten, ties upward. |
| Ordinary HP damage | Divide the rounded main hit in tenths by ten. |
| Area HP damage | Floor the main hit in tenths divided by twenty; this is the area damage in HP. |

If total force≤0, all damage is0. Rounding to the nearest multiple of ten converts tenths to whole HP, with ties upward. Defender forces use its pre-hit current health;
retaliation is suppressed on a killing hit. Example: full10HP Warrior(2/2) versus
full10HP Warrior with no bonus gives5 damage and5 retaliation. Against Fortify
city bonus1.5 the same attack deals4 and takes5. Always use the pinned integer
calculation for exact comparisons.

## Defense and retaliation

Own-city Fortify gives1.5, or4 after CityWall. The terrain branch uses1.5 in water
with Aquatism, forest with Archery, mountain with Climbing; terrain branch selection
can take precedence over the fallback city branch. Poison halves the resulting
bonus. Non-Fortify Swordsman/Giant do not magically gain city Fortify.

A defender retaliates only if alive, not Stiff, within its own attack range, and
its owner has explored the attacker's tile. Surprise suppresses retaliation.
Normal attacks require an own un-attacked unit, with no prior movement unless Dash,
and an explored unhidden unprotected foreign target within range. There is no
projectile travel-time control in this tile engine.

On target death, melee range<2 attackers advance if terrain permits and the attacker
survives. Persist/Escape update flags after the resulting position is known. Kills
remove5*(unit nominal cost+passenger nominal cost) score from the victim's owner,
record casualty/kill counters and give eligible attacker XP. A converted unit's
score transfers rather than being treated as a kill.

## Splash and Stomp

Each adjacent hostile unprotected unit is evaluated independently using that unit's
health and defense bonus, then receives floored half-hit damage. All damage amounts
are determined before deaths are removed. Splash centers on the attacked tile;
Stomp on arrival. Protected peace/cooldown partners are excluded. The LIFO action
ordering and x-outer neighbor order update facing; splash retaliation occurs after
the main/splash resolution where applicable. Do not multiply a main hit by one
constant for every nearby defender regardless of its defense.

## MindBender conversion

A legal conversion has no retaliation. It transfers ownership, full nominal unit/
passenger score, and assigns the converter's current-capital home except Independent
units, which use no home. Converted units are exhausted, retain their current unit
properties, and reveal the new owner's normal/scout sight. Converter is exhausted.
Capturing a supported unit does not unlock its prerequisite tech for free.

## Cloak/Dinghy infiltration

The target must be a foreign unprotected city, not already in riot, without a unit
owned by a third party rather than that city owner. Cloak is consumed. The attacker
gains the city's nonnegative raw production (including original-capital+1, without
negative-progress reduction), inflicts one full-health Dagger-based hit on its
hostile occupant if any, starts riot and creates up to `min(city_level,5)` Daggers
or Pirates on eligible empty explored tiles ruled by that city.

Spawn priority is deterministic: small hash tie noise in[0,.1), +2 for city tile,
−.1 water, +.5 for a known resource, +1 for accessible defensive terrain, −.5 for
poison improvement, then stable tile-index tie order. Candidates need appropriate
mountain/water technology. Water spawns Pirate, land Dagger; no home is assigned.
Riot ends at the end of that city's owner's turn and suppresses its city work while
active. This is infiltration, not immediate city ownership capture.

[Wiki home](../README.md)
