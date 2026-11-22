# Independent native fixed8 rules

These headers execute native C++ integer rules without calling Java. They port the
plain eight-card level-11 task only, from crforge commit
`f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de`; each module identifies its owning source
paths. The parent native directory retains Apache-2.0 `LICENSE` and `NOTICE`.

`rules.hpp` includes timeline, hand/elixir match state, full placement/landing masks,
queued player commands, action masks and Ladder end/tiebreak orchestration.
`unit_state.hpp` separately includes fixed8 state transitions and visits, combat
switching and the king's starting-action activation pipeline. All per-tick state
is POD/fixed storage; headers mark computation functions `CN_HD`.

The owning native world supplies all entity combat, holder pass ordering, tower
listing/cleanup, circle hit queuing and spell/unit creation. Rules never replace
those mechanics with trajectory lookups. A failure or capacity exhaustion is a
fault, independently from a natural winner or a caller-imposed truncation.

## World adapter contract

`LadderMatch.step(world)` reads `king_alive(side)`, `princess_count(side)` (listed
princess towers, including a dead one until cleanup), and `lowest_tower_hp(side)`.
It invokes `execute_due_commands(tick)`, `tick_entities(tick)`, `cleanup()`,
`end_match()`, `hold_hits(bool)`, `clear_field()`, `drain_towers(side,amount)`,
`tiebreak_step_end()`, `arena_length()` and `circle(side,radius)` in the original
order. `end_match` sets the match-ended flag independently from the tiebreaker's
hits-held flag. The clear-field callback must preserve the holder's removal walk
and queued-damage timing; the circle callback queues kills until the damage drain.

The world calls `MatchSide.visit(config,timeline)` in each king's post-hook, even
for a dead king that remains listed. The native reset must make the two
`addPlayerData(1)` battle RNG draws before `LadderMatch.init`; init itself makes
only the two deck-shuffle seed draws, in side order.

`PlayerCommands.execute_due` performs the second placement search and spends and
cycles before invoking `world.deploy_card(side,card,deployIndex,placement)`.
`placement` contains all fixed8 formation points, lanes, deployment states and
stagger waits. The callback receives the deployment counter's value **before**
the play increments it. Commands execute in insertion order. The 32-entry
pending-command store is compacted only after promises are released; an overflow
sets an explicit fault. A refused promise within the oracle's unmodelled expiry
window returns `UnsupportedPromiseExpiry`, preserving its refusal boundary.

`legal_actions` emits all 2307 mask entries, with wait always open and the two
unreachable champion ability actions closed. Pass only seat-visible mask entities.
Seat 0 mirrors x and seat 1 mirrors y, matching `Schema.gameX/gameY`.

`state_set/state_resume/state_visit` take a query adapter that supplies route
preparation, the immediate combat gate and row-specific entry/exit callbacks.
`has_hit_points()` means the HP object exists, not that the entity is alive.
`state_visit` returns 1 only when the visit-tail combat gate should run, 0 on an
early stagger/deployment return and -1 for an unsupported state. State setters
return 1 for a change, 0 for a guard/no-op and -1 for unsupported states.

`KingActivation.pending_pass()` runs in all three pending phases; `run_pass`
runs once after components. Fold `tags(inactiveMask,activatingMask)` at pre-hook,
before phase 1. Finished action runs remain listed until the next run pass, so
their tags keep the component off for the extra tick. The initial wait is queued,
then phase 1 starts it. Its activating duration comes from the frozen action row.

## Constants and checks

`data_codegen.emit_rules(data)` accepts the source-bound exported table data
including `match_cards` and `unit_ids` and produces only constant/configuration
functions. The owner includes `rules.hpp` before the emitted namespace block.
The generated arena array is host storage; a CUDA owner copies those constants to
the device and supplies a device `ArenaView` rather than passing the host pointer.

Run focused checks with the game's environment, each into a new directory:

```sh
.venv-clash/bin/python integrations/clash/native/rules/test_rules.py NEW_OUTPUT
.venv-clash/bin/python integrations/clash/native/rules/test_state.py NEW_OUTPUT
.venv-clash/bin/python integrations/clash/native/rules/test_codegen.py \
  integrations/clash/native/generated-v2/task-data.json \
  evidence/clash-implementation/native-rules-20261108/differential-9/result.json NEW_OUTPUT
```

Retained evidence under `evidence/clash-implementation/native-rules-20261108/`:

- `differential-9`: 15,010 timeline rows, 7,000 hand/elixir rows, 16,880 tower-only
  full match rows, 500 full fixed8 placements, 600 independent command execution
  rows, 402 command givings on real evolving Java states, 81 entire action masks.
- `state-3`: 1,000 visit vectors, 1,000 setter vectors, 1,000 combat gate vectors,
  120 generic action-holder rows and 120 actual king starting-action rows.
- `codegen-2`: all eight cards and 72 configuration fields match independently
  read oracle data; generated arena/configuration compiles with warnings as errors.

The test programs use Java solely as a read-only oracle. Their native shared
libraries never link or delegate simulation to it. These focused results do not
establish whole-world trajectory, observation, CUDA compilation or CUDA runtime
parity; those require the owning native world qualification suite.

`world_state_adapter.hpp` supplies the complete `Fixed8StateQueries<World>` adapter
and `fixed8_set_state`, `fixed8_resume`, `fixed8_post_state`. Its documented World
interface keeps route/reference setters native, synchronizes the targeting switch
and preserves troop versus tower post-hook gate arguments. `adapter-1` exercises
staggered deployment, route preparation, repeated stuns, inactivity, death and the
unsupported-state fault boundary.

`test_holder.py` compiles the adjacent Java `HolderOracle.java` as a separate test
helper against the pinned jars; it never writes the oracle source tree. `holder-1`
retains 340 exactly matching callback events across three scenarios: complete
component and action phases, registered and deferred additions, pending and live
removals, cascading removals, and required-owner admission/release. The test uses
entity labels: it does not qualify an ID-sensitive custom add-tag-fold callback.
The fixed8 row/buff/action tag fold does not read the ID.
