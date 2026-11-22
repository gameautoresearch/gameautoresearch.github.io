# Unit skills and special commands

Only skills on the18 native-supported forms are relevant. The catalog records
exact skill membership; this chapter gives their actual behavior.

| Skill | Rule in this mode |
|---|---|
| Dash | Fresh ordinary movement can leave attack available; not after embark/disembark |
| Escape | Attack can leave movement available; no unlimited movement/actions |
| Persist | Kill may allow another attack if another target exists from new location |
| Fortify | Own-city defense multiplier1.5, or4 with CityWall |
| Stiff | Cannot retaliate; does not mean immune to ordinary damage |
| Static | Cannot be promoted; no new kill XP for a Static attacker |
| Float | Water movement/transport-access handling; no road movement bonus |
| Carry | Boat/passenger score and health model, landing transformations |
| Scout | Sight radius2 instead of1 |
| Creep | Ignores normal forest stopping and zone of control; no road speed bonus |
| Hide | Hides after eligible voluntary movement; first hiding can consume attack |
| Infiltrate | City-target action replaces normal damage and excludes ordinary city endpoints |
| Independent | No city support home (Daggers) |
| Surprise | Defender cannot retaliate against the attack |
| Splash | Adjacent-to-target hostile damage from half of each computed combat hit |
| Stomp | Adjacent-to-arrival hostile damage; suppressed for forced push |
| Convert | MindBender changes target ownership instead of ordinary damage |
| HealOthers | MindBender adjacent-own healing/poison-clear command |

## Recover, passive recovery and promotion

Recover requires an own idle damaged unit and exhausts it. On own or peace territory,
heal4; elsewhere heal2, capped at max HP. A poisoned unit instead keeps its current
HP and loses poison. Idle units get the same recovery/poison-clear handling when
their owner ends turn. MindBender HealOthers heals each adjacent own unpoisoned
unit by4, clears poison on poisoned units without healing them in that application,
and exhausts the MindBender. The action's legal test requires an adjacent damaged
own unit; a healthy poisoned neighbor alone does not necessarily open it.

Promotion requires3 kills, not veteran, and a non-Static judged kind. For a boat,
judge its ordinary passenger when present. Promotion adds5 to max HP and fully heals
to the new maximum; it has no idle requirement in native validation. It does not
spend stars or reset the unit's moved/attacked flags. Boats/Static killers do not
gain new XP, but an ordinary passenger may carry pre-embark XP.

## Push and superunit occupied-space behavior

Creating a Giant or certain ruin units on an occupied tile pushes the occupant.
Its facing direction is tried first, then alternating offsets−1,+1,−2,+2,−3,+3,−4;
another owner's facing is reversed. An ordinary legal unoccupied explored neighbor
is required. A boat keeps a reachable land tile as last-resort landing. If none
works, the occupant is removed. Push is deterministic and uses no random draw.
The moved/attacked flags are preserved unless an embark/disembark transformation
exhausts it. Hidden hostile occupants are revealed before pushing. Facing is a
public observation feature because it can change this result.

[Wiki home](../README.md)
