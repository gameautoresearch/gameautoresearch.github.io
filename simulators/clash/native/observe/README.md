# Native fixed8 observer

`observe.hpp` ports the player-visible observation and seat bookkeeping from crforge's
pinned `LadderEnv`, `SeatIds`, `SeatView`, `ElixirCount`, `BuffSlots` and `Schema` Java files.
Each module names the full source path and commit. Runtime observation functions are
header-only `CN_HD`, use flat PODs and fixed arrays, allocate no heap memory, and call no JVM.

The native schema is version 4: 224 header integers, 200×34 entity integers, 7,024 integers
per seat. `schema.hpp` constants were generated from the frozen task-data schema and checked
against original Java outputs by the differential test. Masks belong to the owning deployment
module, not this observer. Values reserved by the schema remain zero.

## World adapter

Supply the native state in `ObservationEntity`, `ObservationOwn` and `ObservationClock`:

- `entity_count()` and `entity_at(i) -> const ObservationEntity&` enumerate **all retained
  live holder entities in holder order**, without truncating to the 200 observation rows.
- `find_entity(battle_id) -> const ObservationEntity*` returns the target's raw view, or null.
- `own(seat) -> const ObservationOwn&` exposes only that seat's private hand, deck, queue,
  commands, promised costs, refusals and public tower bars. `observe_seat` never queries the
  other seat's `own` data. Fill deck/hand/queue entries with vocabulary IDs, not card indices.
- `clock() -> const ObservationClock&` supplies the public match clock, result, crowns and
  environment diagnostics. Keep natural outcomes, truncations, faults and illegal actions
  separate. Fixed8's absent champion slots have champion/unit IDs and charges all -1.
- `fault(code)` marks the transition as a simulator fault and must refuse further publication.

All positions and IDs passed to the observer are raw world values. The observer transforms
positions, gives first-sight identities, redacts unseen targets and hidden aims, and orders
rows itself. Do not pretransform positions, preassign seat IDs or omit waiting own units.

Global tower creation order is King then global x3500 and x14500 princesses. Seat 0 maps
x to `18000-x` and keeps y; seat 1 keeps x and maps y to `32000-y`. Entity rows retain holder
order, so seat 0's first princess entity is drawn at x14500. Header tower slots instead sort
left/right in each seat's screen. Supply `tower_hp/max_hp` as own King/left/right then enemy
King/left/right; retain fallen towers for this header even after holder removal.

Entity `air/building/king` are the source **row** flags, as `writeWorld` reads them. `alive`
is the live GridEntity flag. Set no_move/no_attack/charged from the native equivalents of
view tags. `waiting` means WAITING_TO_DEPLOY, not ordinary deployment. `bar_always` is true
for every plain fixed8 row; other health-bar rules require their persistent damage watcher.
Buff instances are supplied oldest first with vocabulary buff IDs and remaining durations.
The observer selects up to three by stop/slow/timed/forever rank, retaining the full count and
sum of speed percentages. More than eight supplied buffs is an explicit fault, not truncation.

For projectiles, supply raw aim, target, level and deploy_index. For area effects, supply the
current countdown and radius. Projectile visibility is always true, including delay waits;
enemy launched-spell aims stay unknown until the first of that cast's projectiles lands.

## Lifecycle and public bookkeeping

1. Call `ObserverState<IdCapacity,PlayCapacity>::reset(starting_elixir_scaled)` at episode
   reset. Default capacities are 8,192 first-sight identities per seat and 2,048 plays per
   side; increase the template capacity if required. Exhaustion becomes a simulator fault.
   IDs never recycle during an episode and no unseen creation changes an assigned seat ID.
2. After each engine tick, track **placed** plays in the order they ran. Call `on_play(PublicPlay)`
   with the card vocabulary ID, actual cast ID, run tick, final landing point, public cost,
   production stop, kind and side's placed-play ordinal (`deploy_index`). Set projectile_count
   to how many admitted projectiles actually match that cast at tracking time. A launched
   play with no matched projectile shows its point immediately, as Java does.
3. When the first projectile of a launched cast releases, call
   `projectile_released(side, deploy_index, battle_tick-1)`. The point and all remaining
   arrows' aims become public together. Header point age is consequently 1 immediately
   after that tick, not 0. Plain troop and Zap points show at the run tick.
4. Then call `tick(full_bar_ms, maximum_mana, public_spend0, public_spend1, public_gain0,
   public_gain1)` after **every** completed engine tick. Supply a zero rate on a tick that
   did not update entities (end-screen or tiebreaker holds). Costs enter through placed plays,
   then public ability spend, public gains and regeneration are counted in that order.
   Pass only gains the game shows; enemy current elixir, private promises and private refunds
   never enter this state. The fixed8 deck has no private-cost tunnel or refund mechanic.
5. At a decision boundary, call `observe(state, world, int[14048], int[12], float[2])` once.
   This advances movement history once for both seats, writes both views, info and rewards.
   Calling `observe_seat` directly requires one explicit increment of `state.serial` for the
   observation being written. Observe reset before the first action so all six towers take
   the same initial first-sight IDs as Java.

`PublicPlay` rejects tunnel, evolved, hero or Mirror plays as unsupported task state. Those
cards are outside fixed8, and silently revealing their hidden costs would violate the observer
boundary. A tunnel-like entity view can still be redacted as a trail in source tests; this does
not claim support for tunnelling cards' gameplay or delayed cost-reconstruction logic.

## Validation

Use the game virtualenv and a new evidence directory:

```
.venv-clash/bin/python integrations/clash/native/observe/test_observe.py \
  evidence/clash-implementation/native-observe-20261108/new-check
```

The test drives the pinned Java engine at its original five-tick decision boundaries,
extracts raw Java entity/match/own-seat objects into PODs, and compares every native seat
integer, info field and reward to Java's independently encoded direct buffers. Placed play
and release events are tracked after every individual tick; public elixir is independently
reconstructed from these events rather than copied from the enemy observation. Native
source scenarios cover unseen identity gaps, target redaction, movement, trail redaction,
launched-spell aim secrecy, public-elixir stops/caps, row overflow and terminal results.

This validates native observation logic over oracle game states. It does not validate the
owning native whole-world transitions or establish actual CUDA execution. Correctness checks
contain no timing comparisons; performance measurements require the shared performance lock.
