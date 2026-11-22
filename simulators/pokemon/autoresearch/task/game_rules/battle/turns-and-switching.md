# Turns, action order, switching and move execution

[Wiki home](../README.md) · [Move catalog](../moves/README.md)

## Requests and commitment

A normal move request offers the active Pokémon's legal moves and legal reserve
switches. Both sides usually commit before resolution. Switching after a faint,
a pivot or an effect can create an intermediate forced-switch request for only
one side. A waiting side supplies no action. One adapter phase is one request
submission, not necessarily one full battle turn. Successful prediction of an
opponent's action does not reveal its private submitted choice to the policy.

## Queue order

The pinned action queue orders action families before priority/Speed. In the
relevant mode: start/instant replacement and before-turn events run first, then
Revival Blessing replacement processing where requested; voluntary switches precede
Terastallization; Tera precedes ordinary moves; residual processing follows moves.
Within ordinary moves, higher priority goes first, then effective Speed, then the
queue's tie resolution. Trick Room reverses ordinary Speed ordering without turning
negative-priority moves into positive-priority ones. Abilities/items/fields can
change priority or Speed, block a priority attack, or defer an action; consult
those entries. Gen 9 can update queue ordering after relevant mid-turn Speed changes.
A raw base Speed comparison is therefore insufficient.

Ordinary switches preserve party identity, HP, status, remaining PP and held-item
changes; they normally clear stat boosts and volatile effects. Certain passing
moves preserve specified effects instead. A new occupant can take entry-hazard
and switch-in ability/item effects before the next normal action. A switched-out
Pokémon can heal/cure status via Regenerator/Natural Cure. Automatic form reversions
and item/ability changes follow their individual pages.

## Move pipeline

After move/target and active-state checks, engine events can alter priority, type,
base power, category, target and move properties. Before-move status and ability
checks can consume the action. PP is deducted under PP-pressure rules; an already
selected move can be unusable by the time it would execute. Charging, two-turn
semi-invulnerability, target disappearance, protection, immunity, accuracy and
hit-prevention events can cause a failure or no damage without an illegal policy
output. Multi-hit moves apply their own hit count, per-hit checks and effects.
Damage, drain/recoil, secondary effects, item/ability hit reactions, fainting,
forced movement and after-move effect rules resolve at distinct stages. Their exact
order is recorded in each entry's event priorities and exceptions.

A hit can be immune even when the initial type-chart feature suggests otherwise:
Levitate, absorb abilities, item grounding and ability suppression can matter.
Protect-family moves, Substitute, sound/infiltration rules, Magic Bounce and
contact effects are not interchangeable defenses. The move's `flags` and the
receiving effect determine applicability. Sleep Talk calls another usable known
move; Transform copies relevant public battle properties under its own restrictions.
Neither expands this mode into arbitrary team construction or Metronome battle.

## Restrictions and replacement

Disabled moves, zero PP, Encore, Taunt, Assault Vest, Choice locks, Imprison,
Torment, recharge and charge continuations can constrain available choices. When
no ordinary move is usable the engine may offer Struggle; this is not a fixed
fifth moveslot. Trapping affects voluntary switches, while faint replacements and
move-forced pivots have their own escape rules. Ghost type and applicable items/
abilities can defeat some trapping effects. The policy may not inspect an unrevealed
trapping ability to pre-mask a switch: Showdown can refuse an unavailable choice
and request a legal retry, which remains distinct from an illegal masked output.

Switching and fainting can expose more requests before the turn completes. Revival
Blessing revives a selected fainted teammate at half HP through its dedicated
request; the v4 adapter's target convention is in [mode boundaries](../mode/README.md).
Sacrificial moves, recoil, self-KO and delayed attacks can produce double faints;
engine faint-queue and win checks determine a natural result, rather than a wrapper
inventing a winner from summed HP.

## End of turn

Weather, status chip/healing, delayed attacks, draining/trapping, items, wish effects,
ability state updates, timer expiry and other effects use ordered residual events.
`onResidualOrder` and `onResidualSubOrder` specify ordering, with Speed/tie handling
for effects at the same order as the engine defines. Fainting can occur between
these events; it is not safe to sum all residual HP changes and apply them at once.
See [the condition catalog](../conditions/README.md) for duration/priority fields.
