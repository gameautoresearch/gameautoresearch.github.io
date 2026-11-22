# Pokémon native Gen 9 rewrite — development readout

The independent native rewrite is **unfinished and excluded from production**.
Full `gen9randombattle` admission remains false. The user has stopped GPU-environment
work and redirected development to the native C++ CPU engine, targeting measured
multiple-fold improvements. This readout records intermediate evidence, not completion.

The original full task and frozen upstream pin remain unchanged:
`567c1cedddeaa2f35576005d3bfd526cf4f8a13e`.
Showdown is an independent oracle; native stepping/generation do not call it.

## Verified components

All paths below are relative to
`evidence/pokemon-implementation/native-rewrite-20261108/`.
Each receipt preserves its own source, fixtures and build. Do not treat later
working-tree additions as validated by earlier binaries.

| Component receipt | Exact cases | Oracle transitions |
|---|---:|---:|
| `recovery-2` — Roost, Rest, weather healing, Wish, Sucker Punch | 304 | 9,374 |
| `sleep-call-1` — Sleep Talk, Chesto, called effects/Pressure | 436 | 19,639 |
| `modifier-2` — damage/stat abilities, Eviolite, Wide Lens | 480 | 8,165 |
| `paradox-2` — Protosynthesis, Quark Drive, Booster Energy | 304 | 7,955 |
| `boost-ability-1` — stat changes, Sheer Force, Sturdy, crit protection | 192 | 4,762 |
| `immunity-boost-3` — absorption, Unaware, Prankster | 224 | 6,877 |
| `numeric-moves-3` — weight power, Freeze-Dry, screens, Haze, HP effects | 344 | 9,367 |
| `heal-sound-block-1` — Psychic Noise, Throat Chop and healing interactions | 316 | 10,081 |
| `locked-moves-1` — Outrage, Thrash, Petal Dance | 288 | 4,299 |
| `powder-axel-2` — powders, Triple Axel, Fake Out/First Impression | 212 | 4,167 |
| `trick-secondary-2` — item exchange and multiple/random secondaries | 296 | 9,039 |
| `type-stat-3` — type changes, accuracy and additional stat abilities | 304 | 7,936 |

These are constructed differential development cases with exact tested mechanics,
masks, PRNG and public observations. They are not an exhaustive interaction proof.
Failed intermediate receipts are retained. Regression receipts record later
source checks separately.

`materialize-2` separately matches 12,246 independently constructed Pokémon
records (6,102 successful standalone sets plus 1,024 six-Pokémon teams): full
HP/stats/types/ability/item/PP and unchanged constructor battle RNG. It does
not dispatch battle-start callbacks or qualify the resident random reset.

`teamgen-full-2` independently reproduces complete frozen set/team generation:
6,108 set attempts (six expected source failures), 1,024 complete teams, exact
ordered results and final RNG states. Additional checks cover pools, forms,
compatibility, move counters/culling, moveset and ability/item selection.
Actual CUDA generation repeats these checks in `resident-3` and `resident-4`.
Generator fixture SHA-256:
`b01188261ca32e9e665b2e4e20f6683d6fd89cc3dcb68a87099ea335f18bc66d`.

`coverage-4` statically lowers 258 of 348 catalog moves. Of 1,024 diagnostic
teams, eight contain only statically lowerable moves/abilities/items. This is
**an implementation inventory, not a completion percentage, admitted training
subset, or proof that those complete battles are correct**. The random-team
sampling distribution must not be filtered to those teams.

## Actual L4 environment/learner validation

`resident-4`, app `ap-FBdyOoDEdg3CVFzwtOpZUj`, used one NVIDIA L4, four fixed CPUs
and 8 GiB RAM. Its frozen source predates the later Unaware/Prankster, numeric,
heal/sound-block and rampage additions.

- Complete native generator CPU/CUDA results match the oracle fixture above.
- 408 constructed battle cases; 33,864 complete CPU/CUDA state/mask comparisons;
  10,957 oracle transitions; 15,911,000 exact float64 public feature values,
  with exact float32 casting on GPU.
- Real existing Policy/PPO: each backend collected 15,450 then 15,392 samples;
  341 then 460 natural outcomes, zero then two truncations, zero faults.
  Parameter L2 change: 1.3019992433342844.
- Full environment, model, optimizer and CPU/CUDA RNG checkpoint continuation
  matches exactly; CPU/CUDA learning is bitwise identical on the tested run.
- Five alternating profiles at 2,048 constructed games: median native CPU
  195,831 and GPU 226,513 decision samples/s including collection and PPO.
  These are component workloads, not full Gen 9 speedup claims or throughput
  comparisons against previous corpora.
- Measured worker body 209.749 seconds, client invocation 219.191 seconds after
  object preparation. Inner work 199.435 seconds; before-archive body 203.895.
  Image/mount preparation precedes these measurements. Provider billing remains
  unverified. Peak CUDA allocation: 1,670,958,080 bytes.
- Downloaded archive SHA-256 verified against its receipt. The app completed.

The earlier full-task Showdown optimization result remains approximately 1.394×
against the already optimized four-CPU baseline. The roughly 6.06× result used
8× CPUs and 4× RAM. Neither may be multiplied by these component measurements.

## Expanded actual L4 qualification

`resident-5`, app `ap-CYKi5dkbbtVcovsUg5NmrF`, repeats generation and
constructor checks on actual CUDA: 6,108 set attempts including six expected
failures, 1,024 teams, and 12,246 exact constructed Pokémon records. It also
passes 464 constructed battle cases, 38,512 complete CPU/CUDA state/mask
comparisons, 11,666 oracle transitions, 16,982,000 exact public feature values
and GPU casts. It includes the type/stat, Trick, powder, rampage and heal/sound
families above; it predates the subsequent phazing changes.

CPU/CUDA PPO is bitwise identical and full optimizer/model/environment/RNG resume
passes. Updates collect 15,676 then 15,586 samples, 334 then 445 natural outcomes,
zero then two truncations, and zero faults; parameter L2 change is
1.4696669609614414. At 2,048 component games, five alternating profiles give
182,980 CPU and 222,034 GPU samples/s. These are development component timings.
The complete worker body is 187.681 seconds (inner work 177.245; before archive
182.482), and client invocation after preparation is 196.764 seconds. Peak CUDA
allocation is 1,679,575,040 bytes. Archive, CPU executable and source-file hashes
all verify; the app completed. Provider billing remains unverified.

## Actual L4 generation-to-completion qualification

`resident-6`, app `ap-zj1Fk2zfrHJZSbnlXL7qmL`, completes actual CUDA
qualification of native random reset, including all 1,024 attempted pairs and
1,005 explicit unknown-effect faults. The 38 implemented generated battles finish
naturally with 2,439 oracle transitions, 8,852 complete CPU/CUDA state/mask checks,
and 3,467,800 exact public feature values plus GPU float32 casts. There is no
resampling or filtered production admission.

The constructed corpus passes 1,028 cases, 99,716 complete state/mask checks,
35,789 oracle transitions and 51,543,800 exact public feature values. Generator
and constructor parity still cover 6,108 set attempts, 1,024 teams and 12,246
Pokémon records. CPU/CUDA PPO and complete optimizer/model/environment/RNG resume
match bitwise. At 2,048 constructed games the GPU median is 269,470 decision
samples/s including PPO; this is a component workload, not a full-task speedup.
Worker body is 176.384 seconds (inner work 162.883; before archive 168.412),
client invocation after preparation is 388.440 seconds. Peak CUDA allocation is
1,682,062,848 bytes. Archive, CPU binary and frozen source hashes verify. The
app is stopped with zero tasks. Provider billing remains unverified.

This source predates the later items, reaction/trigger abilities and scheduler
corrections. Those additions require their own actual-CUDA qualification.

## Native generated reset and first complete trajectories

`phaze-2` passes 348 cases and 17,310 transitions, including forced switching,
Suction Cups and Parting Shot. `conditional-3` adds Payback, Avalanche/Revenge,
Lash Out, Dragon Energy, terrain-dependent attacks, Triage and Gale Wings:
216 cases and 6,813 transitions pass. The combined regenerated `gpu-corpus-9`
passes 1,028 constructed cases and 35,789 transitions on CPU.

`startup.hpp` now independently generates both teams, constructs every Pokémon,
and dispatches the native startup event queue. It does not filter or resample
unknown teams. `generated-rollout-4` attempts all 1,024 retained seed pairs:
1,005 produce explicit unsupported-effect faults. The other 19 whole-team pairs
match Showdown from reset through natural completion on two battle seeds each:
38 battles, 2,439 transitions, exact mechanics, masks, RNG and all 700 public
features per player at each decision. These are component differential tests,
not admission of a filtered training distribution. The resident random-reset
kernel is undergoing actual CUDA qualification separately.

Later `items-2` passes 528 cases and 14,813 transitions. It covers White Herb,
Flame/Toxic Orbs, Weakness Policy, Rocky Helmet, Light Ball, Soul Dew and the
18 ordinary type-boosting items. The frozen observer's White Herb and Trick
message-order behavior is preserved. These additions postdate `resident-5` and
the currently frozen `resident-6` source, so they have CPU evidence only.

## Later CPU qualification and generated collector wiring

`gpu-corpus-11` passes 2,864 constructed cases and 84,431 transitions on CPU.
It includes `react-ability-2` (460 cases / 10,326 transitions),
`trigger-ability-2` (512 / 10,012), and `misc-moves-3` (336 / 13,491).
The scheduler preserves volatile insertion order and distinguishes stable
left-to-right hit events from events that shuffle speed ties. Additional
fixes cover Struggle contact, simultaneous faint order, blocked Sitrus
consumption and Intimidate visibility with no living opposing slot.

`generated-rollout-7` attempts 1,024 retained seed pairs: 905 explicitly fault;
119 implemented whole-team pairs match complete battles on two battle seeds,
238 natural outcomes and 13,069 transitions. `task-seeds-1` additionally uses
the original worker's SHA-256 domain-separated task seeds and independent
teams. All 4,096 attempts are retained; 58 implemented battles (paired seat
orders) match through natural completion, 3,205 transitions and 4,568,200 public
feature values. These diagnostic successes do not authorize filtering the
training distribution.

`random_env.hpp` and `random_resident.py` connect native generation/startup to
finite automatic-reset schedules. Checkpoints include the exact schedule and
per-environment reset cursors. Missing mechanics fault without resampling;
exhausted schedules fault separately with code 4. `random-env-cpu-2` verifies
complete state/cursor resume, rewards, final truncation views, unsupported reset
faults and schedule exhaustion against independently advanced/reset CPU arenas.

`resident-7` is a retained **failed** actual-L4 attempt. Its generator, generated
battle and task-domain checks returned before an illegal GPU memory access in
the newly added generated collector. A CPU collector checkpoint was saved; the
GPU collector is not qualified. `random-debug-1`, using that exact frozen source
and executable with synchronous launches, reproduces the fault in the first
`pkn_random_env_step_device` call. Kernel/resource diagnosis continues; failed
attempts and their receipts remain retained.

CPU additions after the frozen `resident-7` source: `failure-moves-2` passes
316 cases / 2,765 transitions for crash damage, Explosion and previous-move
failure power; `room-revive-2` passes 160 / 18,375 for Trick Room and Revival
Blessing; `curse-salt-1` passes 168 / 6,109 for Curse and Salt Cure. These do not
inherit older CUDA or whole-task qualification.

## Additional CPU qualification and CUDA stack repair

- `air-accuracy-2`: 272 cases / 4,727 transitions, No Guard, Power Spot,
  Plus/Minus, Air Balloon and Scope Lens.
- `mold-2`: 216 / 6,930, Mold Breaker/Teravolt/Turboblaze, temporary suppression
  through phazing, restoration and immunity Update callbacks. Breakable flags
  are compiled from the pinned Dex by `ability_flags.py`.
- `simple-abilities-2`: 252 / 8,027, Scrappy, Mind's Eye, Keen Eye, Big Pecks,
  Prism Armor, Surge Surfer, Liquid Voice, Pixilate/Galvanize, Corrosion,
  Unseen Fist, Good as Gold, Queenly Majesty and Analytic.
- `misc2-1`: 180 / 6,796, Sparkling Aria, Shell Side Arm, Bleakwind Storm,
  Blood Moon and Gigaton Hammer selection restrictions.
- `react2-2`: 270 / 4,008, Gooey/Tangling Hair, Toxic Debris, Seed Sower,
  Aroma Veil, Flower Veil, Poison Heal, Dry Skin and Super Luck. Struggle's
  default crit ratio is applied before ability/item increments.
- `guard-1`: 144 / 5,468, Magic Guard indirect-damage distinctions. Ten cases
  deliberately reach their fixture bound; 134 have natural outcomes.

The frozen `resident-7` binary's generated collector required a larger dynamic
CUDA stack than the default 1,024 bytes. `random-debug-4` still fails at 8 KiB.
`random-debug-2` passes its first 32 exact CPU/CUDA phases at 256 KiB but then
runs out of GPU memory. **`random-debug-5` passes all 96 phases and three PPO
updates at 64 KiB**, including 384 generated resets, exact full states/cursors,
observations/rewards/flags and zero faults. Its body wall time is 54.232679 s;
this is a diagnostic on the immutable resident-7 binary, not validation of
newer mechanics or full checkpoint qualification. All attempts are retained.
The runtime now provisions this stack explicitly through `cuda_runtime.py`.

`resident-8` lost its client connection before a function-call ID returned;
allocation remains unknown, and stopped/zero tasks was independently observed.
`resident-9` is retained as a failed broad CUDA qualification attempt. Its expanded
3,996-case retained corpus is streamed in shards of 256 under the fixed 8 GiB
RAM limit; profile templates are every sixteenth fixture, at most 256. Older
fixture token tables are accepted only as exact append-only prefixes; any
existing-ID token change remains an error and checkpoint bindings stay strict.
Its CPU/CUDA state agreement held at the failing generated step, while the
oracle RNG differed. The cause was skipping an accuracy draw after Wide Lens
raised a numeric accuracy above 100; source numeric rolls still consume RNG.
`generated-regression-2` now passes all 238 retained generated battles / 13,069
transitions and `task-regression-1` passes all 58 independent task-seed battles /
3,205 transitions. `gpu-corpus-12-cpu-regression-2` passes 3,996 / 123,337.
`resident-9` is stopped with zero tasks; archive, CPU binary and all frozen
source hashes were verified, as were the earlier failed resident-7 artifacts.
Neither pending nor failed jobs constitute a full-task speedup.

`double-shock-2` passes 96 complete games / 3,876 transitions, including
Protean/Libero, Electric Tera and the removed Electric type. `charge-1` passes
144 / 9,033 for Solar Beam, Meteor Beam and Power Herb, including suspended
move locks, status aborts, weather and boosted charge turns. The complete
`gpu-corpus-13-cpu` regression passes **5,082 cases / 160,545 transitions**, of
which 5,008 games naturally complete; bounded/truncated fixtures remain distinct.
`resident-10` attempted a fresh frozen CUDA run of this exact 5,082-case corpus,
but its connection failed during setup before a function-call ID returned. The
sharded-upload resident-11 attempt was subsequently stopped at the user’s request.
Later live additions do not inherit its pending qualification.

## Current direction: native C++ CPU

GPU work was stopped at the user's request. The latest owned app
`ap-3L6WysUp82vMSGRHbsVUAH` is stopped; its local upload process was terminated.
`gpu-work-stopped-20261108.json` retains the cleanup inventory. The interrupted
resident-10/11 setup attempts have unknown allocation, not assumed zero spend.
Do not restart CUDA qualification without a new user request.

The supported full-format backend remains the optimized Showdown CPU simulator.
`recipes/cpu-optimized.json` provides shared observations, lazy decoding and batched
scripted decisions on CPU. Its ordinary CLI smoke completed one learning update,
138 policy samples and four natural endings with zero GPU execution. Both engine
contracts and all 15 Python RL/efficiency tests pass; the fast-engine checks pass.
This smoke is correctness evidence, not a new speed measurement.

The final pre-redirection native CPU regression, `cpu-wrap-regression`, passes
5,390 constructed cases / 166,113 transitions, with 5,316 natural games and the
remaining fixtures explicitly bounded. `faint-3` adds Moxie, Chilling/Grim Neigh
and Soul-Heart (128 complete games / 2,277 transitions). `berries-2` adds Unburden,
Cheek Pouch, Unnerve, Harvest, Leppa Berry and Throat Spray (180 complete games /
3,291 transitions). `generated-regression-3` retains 238 complete generated games /
13,069 transitions. `cpu-wrap-reset` verifies rewards, final truncation views,
complete state/cursor resume and distinct unsupported/exhaustion faults.

`cpu_env.py` is the CPU-focused collector path: persistent native workers write
observations, final views and rewards directly into host tensors, with no Node
IPC or CUDA library. `ResidentPPO` now derives its device from the environment;
the CPU canary uses a CPU model, optimizer and RNG throughout. Qualification and
matched CPU measurements will be retained separately. No full-format speedup is
claimed until the remaining mechanics and matched workloads qualify.

## CPU-only qualification and first matched profile

`cpu-only-1` runs the native random-reset collector and PPO entirely on CPU.
CUDA initialization/runtime access is trapped during the check. Serial and three
native worker threads produce identical models, optimizer states, Torch RNG,
environment bytes and reset cursors. Each variant collects 1,949 then 1,957
samples, exercises 128 generated resets and reproduces a saved continuation
exactly. Parameter L2 change is 0.6480899638; CUDA remains uninitialized. This is
a learning/checkpoint canary on a finite diagnostic schedule, not policy strength.

`cpu-profile-1` holds `evidence/local-performance.lock` and compares the same
96 complete original-task-seed diagnostic games, 9,700 player decisions per
repetition, with three simulator workers and one main thread. The reference is
full Showdown with the existing optimized shared-memory/lazy CPU path. Both
paths generate/reset teams, simulate and produce float32 model-input observations;
24,595,200 public feature values and natural outcomes match exactly. Five
counterbalanced repetitions on macOS 26.6 arm64 give median 0.995101 seconds
for Showdown versus 0.040099 seconds for C++: **24.816× on this selected
simulation/reset/encoding workload**. PPO and one-time compilation/setup are
excluded and separately recorded. This diagnostic selection contains implemented
mechanics only; it is not full-format distribution throughput or a training gain.
`cpu_learning_profile.py` measures the paired complete CPU collection/GAE/PPO
loop separately and requires exact paired models, optimizer states, RNG and views.

`cpu-learning-1` is a retained failed paired learner check. The synchronous
`cpu-shadow-1` trace found a missing user stat secondary when an attack hits a
Substitute (Rapid Spin's Speed boost). The corrected generic self-secondary path
passes `sub-self-1`: 144 complete games / 7,848 transitions. `cpu-shadow-2` then
matches full Showdown observations, masks, rewards, outcome flags and final
truncation views for **192 phases of real CPU PPO**, six updates, 96 environments,
242 natural outcomes and two truncations, with no CUDA initialization or faults.

`cpu-learning-2` holds the shared local performance lock and runs five alternating
matched full-loop repetitions after an exact warmup. Both paths use CPU inference,
collection, generation/resets, GAE and PPO (96 games, 32-step rollouts, 2 PPO epochs,
minibatch 4,096, three simulator workers, one Torch thread). All paired models,
optimizers, Torch RNGs, reset cursors and final observations match exactly after
every update. Median throughput is **6,743.92 Showdown vs 34,011.89 C++ decisions/s,
5.043× on this diagnostic learning workload**. Setup is separately recorded and
excluded from steady-state medians. These are implemented-team diagnostic seeds;
this is neither a complete-format admission claim nor proof of policy strength.
No speedups from different experiments are multiplied.

`cpu-abilities-3` passes 160 complete games / 6,768 transitions for Comatose,
Slow Start, Bad Dreams, Supreme Overlord and Liquid Ooze. Slow Start preserves
its retained ability state while inactive and starts affecting incoming speed
only at the source switch-in timing. `ruin-2` passes 192 complete games / 2,883
transitions for all four Ruin abilities, including equal-speed listener ties
with Water Bubble. `cpu-volatiles-2` passes 144 complete games / 6,209 transitions
for Glaive Rush, No Retreat and Magnet Rise; the public request omits trapping
when there is no living bench alternative. `cpu-startup-1` repeats all 1,024
unfiltered reset attempts: 529 explicitly fault and 495 implemented team pairs
match startup on two battle seeds (990 checks). This is startup qualification,
not evidence that all 495 pairs complete arbitrary battles correctly.

`plates-2` adds the explicit frozen Multitype identity, Judgment, all 17 plates,
plate damage boosts and source/holder-dependent item-removal rules. Qualification
is retained separately from previous speed measurements. Unburden's TakeItem
callback runs before an item can reject removal, including unsuccessful Trick;
restoring the held item does not undo that hidden volatile state.

## CPU stopping point — user-requested wrap

Development is wrapped at the user's request. No new mechanics or experiments
are being started. `cpu-wrap-2/summary.json` indexes the final receipts and their
hashes. GPU work remains stopped; this continuation used local CPU execution only.

The final `cpu-regression-16` binary matches **7,055 cases / 231,508 transitions**,
including 6,981 natural oracle games and 74 bounded cases. The constructor-only
check `cpu-materialize-2` still matches all 12,246 frozen Pokémon records.
`cpu-generated-replay-2` fixes all nine previous counterexamples in 246 retained
original-task battles. The next unfiltered campaign retains all 1,024 attempts:
664 explicitly unsupported and 360 implemented battles. Its initial 18
counterexamples remain in `cpu-campaign-2`; the corrected actual native
generation/reset path matches all **360 natural battles / 18,971 transitions**
in `cpu-generated-replay-4`. No failed attempt or source receipt was overwritten.

This continuation adds actual Arena Trap, Magnet Pull and Shadow Tag, including
hidden switch rejection, pending opponent choices and a new request without
advancing the battle RNG. `trapping-2` and `trapping-protocol-1` match 112 complete
games. The final `protocol-resume-3` checks 118 rejected boundaries, 110 with a
pending opposing choice and eight dual rejections: continuation is exact and
every rejected decision still respects the truncation cap.

`ignore-orbs-1` matches 154 complete games for Moongeist Beam, Sunsteel Strike,
Photon Geyser, the legendary cores/orbs and Rusted Sword/Shield item-removal rules.
Crowned forms are materialized at native BattleStart, separately from the
unchanged constructor contract. The generated campaign also found and fixed
Reckless's crash-move boost, crash damage when no target remains, and fresh
Slow Start state on a switch-in that faints before its Start callback.
`public-data-1.json` verifies 18 public trapping-species entries, intrinsic flags
for all 1,518 pinned Dex species and both crowned-form metadata records.

`cpu-shadow-3` matches all observations, masks, rewards, flags and final boundary
views for 192 phases / six real CPU PPO updates on the newer diagnostic schedule:
241 natural outcomes, three truncations, zero faults and no CUDA initialization.
The final `cpu-only-3` independently verifies exact serial/three-worker learning
and complete model/optimizer/RNG/environment/reset-cursor checkpoint continuation.
Both thread configurations perform 128 actual resets.

`cpu-workers-1` compares five native work-chunk variants under the shared local
performance lock, seven rotating repetitions at 96 and 384 environments. Complete
state, observations, rewards, flags, reset cursors and truncation views match for
all 256 phases of every variant. Four-battle chunks increase median native
simulation/reset/encoding throughput by **19.7% at 96 environments** and **6.4% at
384** versus 32-battle chunks on these diagnostic workloads. Four is now the
CPU default; the worker ticket counter uses relaxed atomic ordering under the
existing dispatch mutex/condition-variable synchronization.

The final locked `cpu-learning-3` comparison uses the new schedule, 96 environments,
three simulation workers, one Torch thread and five alternating measured updates
after an exact warmup. Median full CPU collection/inference/GAE/PPO throughput is
**6,908.78 optimized Showdown vs 34,331.43 native decisions/s: 4.969×**.
Models, optimizer states, Torch RNG, reset cursors and final observations match
exactly after every paired update. One-time setup is recorded separately
(0.264 s Showdown / 1.201 s native) and excluded from steady-state throughput.
This diagnostic result is not a full-format speedup, policy-strength result or
CPU-versus-GPU comparison. It is not multiplied by the older component speedups.

## Remaining CPU obligations

Complete the remaining move/ability/item callbacks and their event order;
dispatch the remaining battle-start effects; finish generated collector qualification and full-task admission; verify complete
random-battle trajectories and public encoding on CPU; then qualify
ordinary CLI training/evaluation and exact checkpoint continuation. Existing
clauses, natural outcomes, truncations, illegal actions and simulator faults
must remain distinct. Full admission stays closed until all requirements pass.

The current static move inventory still rejects 24 of the 348 frozen move records.
Other abilities, items and interactions also remain unimplemented; these counts
are not a semantic completion percentage. Existing CUDA evidence is historical
and does not qualify later CPU additions. Resume only from this documented state.
