# Fixed8 native combat

`combat.hpp` implements the projectile, spell impact, queued hit, health, level scaling,
stun and Fireball pushback request mechanics reachable in the frozen plain level-11 task.
The runtime functions are header-only `CN_HD`, use integer Java arithmetic, allocate no
heap memory and never call Java. The header records the full paths and exact upstream
commit read. See the parent native LICENSE and NOTICE for Apache-2.0 attribution.

## World adapter and phases

The owner calls `tick_projectile` in the projectile post-hook and `tick_combat_area` in
the area post-hook, in holder order. `queue_hit` **only appends**; it must not lower hit
points until the owner's post-hook damage drain. Fireball push requests happen immediately
after each queued area hit; Zap's stun is applied immediately after each queued typed hit.
A projectile's after-damage buff (none of this task's projectile rows set one) accompanies
the hit to the drain and applies only if `accepted` is true. Projectiles spawned/launched
during a visit enter the closing cleanup and cannot take a flight step in that same visit.

The World template supplies:

- `target_count()`, `target_at(index) -> CombatTarget&`: current-tick present targets in
  holder identity order, retaining dead objects until the owning cleanup.
- `find_target(id) -> CombatTarget*`: current retained target or null after removal.
- `queue_hit(const CombatHit&)`: deferred hit queue; the source is the projectile or area
  entity ID, or the troop for a direct hit. The projectile keeps its launcher as `source_id`.
- `push(target_id, from_x, from_y, distance)`: immediate push request and row/state gates.
- `apply_buff(target_id, buff_id, ms, source_id, side, packed_level)`: immediate apply/refresh.
- `next_hit_id()`, `chain(index) -> ProjectileChain&`, `return_pending(target_id, amount)`.
- `fault(code)`: fail the transition closed, rather than continue after a capacity overflow.

Use `damage_with_heading` to drain ordinary hits, `apply_typed_area_damage` for Zap's typed
area hits, and `drain_health` for the match tiebreaker. Ordinary attacks retain their stored
reference position for the death heading; `queue_direct_hit` packages that direction.
The owner handles death consequences, king activation, target invalidation and removal.

`request_combat_pushback` creates the exact target and triangular speed budget. Call
`tick_combat_pushback` in the movement visit, then the owning movement/grid displacement
with that budget. This module intentionally does not replace the owning movement solver's
collision push, terrain relocation, route reset or grid clamping. Giant's IgnorePushback
must be passed as `ignores=true`; towers have no movement and never call the request.

## Frozen table mapping

Compile config from the pinned source data, rather than hardcode observed final damage.
`scale_level` takes source base, packed level, mode, rarity relative level and multiplier
array. Projectile `damageMode` determines card, King or Princess scaling. A unit's level
must be rebased on the projectile's rarity via `pack_level` (Musketeer is Rare but its
projectile is Common). `crown_tower_damage` rounds the scaled amount upward using the
row's difference percent. Launcher start uses its launch radius, start height, live height,
height offset and side-mirrored Y offset; `launch_unit_projectile` implements this placement.

All six frozen projectile rows map directly to `ProjectileConfig`: speed, gravity, homing,
radius, `aoeToAir/Ground`, `onlyEnemies`, considerZDistance, constantHeight, minDistance,
pushback, targetBuff and applyBuffBeforeDamage. Their range, collision, scatter, limited
homing, chain spawn, reflection, special runs, collision body and custom movement are zero
or disabled. Those unrelated mechanics are outside this task and have no runtime fallback.
The caller supplies the source-scaled `damage` and packed projectile level.

Arrows uses `SpellProjectileConfig` from DeployCard count, waves, per-projectile interval,
wave interval and spell radius, with `ring=true`. Allocate all count×waves states and one
chain per wave before calling `cast_spell_projectiles`, then assign entity IDs in creation
order. Every arrow takes one battle RNG draw; its jittered aim controls flight duration,
while its ring point controls impact. Each wave's chain independently collects an entity
once and also enforces the full spell circle. No arrow/wave is collapsed into an instant hit.

Zap uses `AreaEffectEntity`'s filter form. Compile `CommonAreaDamageFilter` to enemy targets
including ground, air and towers; filter Hidden, Underground, NoHitpointComponent and
DashImmune through the target snapshot flags. Raw hit-switch booleans are false and do not
specify this filter. Set `filter_hits=true`, `only_enemies=true`, `hits_air=hits_ground=true`.
Scale typedDamage baseDamage=75 and towerDamage=19 separately at the area's Common level;
supply the latter as `typed_tower_damage`. It does not use the projectile crown percent rule.
ZapFreeze's speed/hit/spawn multipliers are -100; its application lasts 500 ms and same-row
refresh keeps the longer duration. Filter hits are nearest first, stable on equal distance;
there is no one-King-slot restriction in this filter path. Zap's life 1 ms gives exactly one
post-hook hit, then countdown -49 and removal.

`COMBAT_MAX_VICTIMS=256`, `COMBAT_MAX_DEDUPE=64` and eight buff slots are fixed capacities;
exhaustion is a simulator fault, never a legal mask or observation truncation. The complete
world adapter must preserve every live supported-task entity and overflow independently
from the 200-row observation boundary.

## Verification

Run from repository root using the existing game venv and a new evidence directory:

```
.venv-clash/bin/python integrations/clash/native/combat/test_combat.py \
  evidence/clash-implementation/native-combat-20261108/another-check
```

This compiles test-only native ABI wrappers, calls the pinned Java helpers and runs native
source scenarios. It checks integer trig, overflow-sensitive scaling, crown rounding,
health/shield/dedupe and damage guards, Fireball/Arrows flight against real data, moving
Archer/Musketeer/King/Princess homing trajectories, and independently launched Arrows RNG
and wave positions. Native scenarios check delayed queues, Fireball pushes, strict shapes,
three separate Arrows wave chains, nearest-first typed Zap, immediate stun, refresh and
pushback budgets. Correctness checks run without timing claims or performance measurement.

These component checks are not a whole-battle parity result. The owning whole-world
stepping, targeting, movement, deployment, observer and queue drain must be integrated and
differentially checked independently. A CPU compiler check does not establish CUDA device
execution; `CN_HD` compatibility is an implementation property pending an actual CUDA build.
