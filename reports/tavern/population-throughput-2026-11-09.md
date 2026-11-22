# Battlegrounds population trainer throughput — November 9, 2026

The actual trained-bot population pipeline now reaches **7,633 newly accepted
learner decisions per full allocated worker second** in a fresh three-update
check. Matched 48-lobby training improves **2.274× and 2.263×**, with exact
training-state parity. A separate 256-lobby trainer-window profile reaches
14,618/s; this is **not** a full-allocation rate.

The supported simulator remains the Rust CPU engine. All five owned GPU workers
have provider end timestamps; cumulative allocation is **888.815 of
1,200 L4 seconds**, including failed and deliberately stopped attempts. The task
is closed. Historical campaigns and ledgers remain unchanged.

## Workload and implementation

Every arm explicitly imports the admitted quarter157 trained policy, optimizer,
RNG and bounded historical population through the existing source-verified fork.
The parent checkpoint and its exact archived source remain unchanged. Each lobby
keeps four rotating current-policy learners, three frozen visible tempo/economy/
synergy scripts and one retained/history neural opponent. Native admission remains
`217577b83fc4525e`, with the same rules, player-visible arrays, legal masks, guards
and deferred-action admission protocol. Cohorts start at turn one and run every
learner seat to natural termination. Faulted lobbies cannot enter the loss.

The old population trainer already emitted only ready observations and grouped
historical decisions by network. These paths are retained. The new package:

- Batches ordinary script decisions with NumPy, retaining all duplicate, tribe,
  board-order and action-mask scoring. Pending choices use the unchanged frozen
  scalar oracle. No hidden native game state is read.
- Connects the existing retained-device rollout to this trainer's own PPO loss.
  Actor inputs already transferred for inference are gathered in the original
  accepted seat/decision order, removing per-decision NumPy copies and their
  second H2D transfer. Deferred/speculative and faulted rows stay excluded.
- Encodes numeric inputs once per update, using 8,192-row chunks to bound scratch.
  Card embeddings remain trainable. PPO objectives, epochs, KL stopping, optimizer,
  network, opponent schedule and checkpoint cadence are unchanged in matched pairs.
- Enables the existing exact-shape CUDA graphs for current and grouped historical
  actors, with at most four cached shapes per network.
- Records simulation, observation encoding, current/history/script decisions,
  transfer diagnostics, rollout storage, PPO assembly/transfer, optimization,
  population setup, state saves and checkpoint IO. Production avoids unconditional
  stage synchronization; synchronized profiles provide phase attribution.

The new canonical recipe is
`simulators/tavern/reference_training/strong/curriculum-throughput.json`.
CUDA qualification uses PyTorch 2.9.1 / NumPy 2.3.5, one L4, four fixed physical
CPU cores and 16 GiB RAM. Larger cohorts change the sample batch per update;
only same-size pairs establish exact training identity. No policy-strength
promotion or 100k/s capability is claimed.

## Matched training and scaling

The final repaired suite is `l4-bounded-v4/collected/suite/` under the private
evidence root. Each main arm runs two real PPO updates from the same trained
parent. Both 48-lobby pairs match policy, Adam, population, all RNG, complete
training history, full native states, next cohort seed, update and lineage.

| Lobbies / arm | Accepted learner decisions | Trainer seconds | Decisions/s |
|---|---:|---:|---:|
| 48, original, trial 0 | 135,716 | 40.491 | 3,352 |
| 48, optimized, trial 0 | 135,716 | 17.808 | 7,621 |
| 48, optimized, trial 1 | 135,716 | 17.793 | 7,627 |
| 48, original, trial 1 | 135,716 | 40.262 | 3,371 |
| 96, optimized | 273,287 | 28.296 | 9,658 |
| 192, optimized | 531,837 | 41.683 | 12,759 |
| 256, optimized | 704,655 | 48.204 | 14,618 |

These trainer windows include parent/source checks, population setup, complete
cohorts, PPO and initial/per-update checkpoints. They share one interpreter and
exclude interpreter startup, final result-file writes and outer teardown. Initial
v1/v2 receipts used the misleading field name
`complete_process_samples_per_second`; it means this trainer window. Final
receipts name it `trainer_session_samples_per_second`.

The 256-lobby arm completes 2,048 natural learner episodes with zero faults or
rejected actions. Of its accepted decisions, 37.3% are at turn 10 or later and
74.7% have at least five own minions. Peak allocated CUDA memory is 14.37 GiB on
a 22.06-GiB device. This is full-from-start trained-policy collection, not an
early-game or constructed continuation profile.

## Fresh complete-worker check

`l4-selected-final` reuses the exact qualified image, with production timers and
three 256-lobby updates. It accepts **1,057,544 decisions in 138.553 provider
allocated seconds: 7,632.76/s**. This includes interpreter/CUDA startup, parent
checks, cold graphs, simulation, transfers, PPO, checkpointing, idle time and
teardown. Its trainer window is 129.499 seconds, or 8,166.45/s.

The fresh worker is materially slower even inside its trainer window than the
14.6k/s shared-process suite. The difference is **not explained by startup alone**;
worker and timing-mode effects were not isolated. Therefore 7.6k/s is the
established fresh full-allocation result, and 14.6k/s remains a profile result.
The historical approximately 2k/s population runs used a different checkpoint,
lease and instrumentation; their gap is not a matched speedup estimate.

All 3,072 learner episodes finish naturally, with zero simulator faults or
rejected actions. The retained full checkpoint matches its recorded SHA-256,
all 31 bound source files, finite policy/Adam tensors and eight population hashes.
All 256 full native states restore and serialize exactly on the local admitted
Rust engine. No inherited samples or PPO epoch reuse enter the throughput numerator.

## Where time goes

The synchronized final 48-lobby baseline spends 18.869 seconds on Python scripts,
10.306 on historical decisions and 0.751 on Rust simulation, in 40.491 seconds.
The optimized arm spends 3.159, 5.392 and 0.751 seconds respectively, in 17.808
seconds. Rollout storage drops from 0.669 to 0.194 seconds and PPO assembly/transfer
from 1.111 to 0.293. This attributes the package gain primarily to scripts and
historical inference, with reduced duplicate input work.

On the fresh worker, inference/dispatch takes 65.353 seconds, PPO optimization
31.180, observation encoding 12.977, native simulation 5.437 and checkpointing
2.751. Native simulation is about 4.2% of the trainer window. GPU rules rewriting
is not the measured next bottleneck. Historical inference, PPO and observation
transport remain useful performance targets.

Actor sub-timings overlap their parent inference stage. CUDA-event H2D staging
and compute spans plus host D2H/compute wait are diagnostic spans, not additive
pure transfer-only measurements. Production stage timers do not force GPU
synchronization.

## Correctness, failures and accounting

The focused local suites pass 45 tests with seven CUDA-dependent skips. After
the bounded-scratch repair, the ten strong-trainer checks pass again. Public
full-game and trained-cohort fixtures compare every encountered script row
across all three styles to the immutable scalar oracle, including late boards
and pending choices. Retained input tensors, PPO metrics, weights and Adam state
match exactly. Local timing comparisons hold `evidence/local-performance.lock`.
The earlier local CPU pairs improved 1.688× and 1.634×; they are not GPU rates.

Actual-CUDA split-run resume passes with graphs and the repaired learner, including
RNG/population/native state. A changed-source checkpoint is rejected. New rollout
code is included in strict checkpoint binding; no old source identity is relabeled.

Retained attempts:

- `l4-v1`: exact CUDA pairs and 96/192-lobby profiles completed. The first 256-lobby
  update completed, then whole-cohort numeric preprocessing ran out of GPU memory
  on update two. That incomplete arm is excluded from the qualified table.
- `l4-graphs-v2`: exact resume and one matched graph pair passed. The operator
  stopped this older source before its known memory-unsafe large-cohort path.
- `l4-bounded-v3`: after queuing, new memory telemetry failed before training
  because CUDA peak statistics were reset before context initialization. Its
  10.122 allocated seconds remain charged. Initialization order was corrected.
- `l4-bounded-v4`: the repaired ten-arm suite completed, including two-update
  256-lobby collection, PPO, exact pairs and exact CUDA resume.
- `l4-selected-final`: the fresh complete-worker check completed all three updates.

Every GPU interval is settled, with zero active owned workers or unresolved GPU
holds. The conservative ledger books $2.526 charged-or-reserved resources,
including $2.050 of retained unpriced image-build holds; this is not a reconciled
provider invoice. Pricing reservations use the checked [Modal pricing](https://modal.com/pricing).
External model calls/liability are zero. No private final bank, publication,
commit, push, old campaign continuation or GPU simulator rewrite was performed.

Private source, inputs, checkpoint and raw receipts remain under
`evidence/tavern-population-throughput-20261109/` and the matching private Modal
Volume. `CLOSED.json` blocks further dispatch; unused capacity is not a new
training allowance. The [machine-readable report](population-throughput-2026-11-09.json)
contains the stage profiles, exact-state checks, checkpoint audit and provider
intervals. The previously retained incumbent remains the selected bot.


## Follow-up: controlled gap attribution and bounded cache fix

This is a **separately authorized successor task**. The original results above,
original `CLOSED.json`, authorization and SQLite ledger remain unchanged. New
private records are under `evidence/tavern-throughput-attribution-20261109/`.

**Result:** raising the existing per-network CUDA graph cache from four to sixteen
shapes improves controlled cold-process trainer time by **6.3–8.2%**. The fresh
full-worker checks measured **7,709/s before and
9,587/s after**, including imports, startup, cold captures, real
PPO, checkpoints and teardown. Their 1.244× cross-worker ratio is larger than
the controlled cache gain and is **not all attributed to the fix**.

The original 14.6 k versus 8.2 k trainer-window gap is **not fully explained**.
Controlled warmup and timer settings account for only about 5% on the observed
worker. Substantial runtime variability is directly measured, but the original
fast worker lacked enough host telemetry to identify its remaining cause.
No new 14.6 k/s full-allocation claim is made.

### Controls and runtime verification

All comparisons use the same quarter 157 parent, optimizer/RNG, eight-member
historical pool, four rotating learners, three frozen visible scripts and one
retained/history opponent, with 256 full-from-start lobbies, identical rules,
PPO settings and source/versions. No trained policy is promoted or replaced.
The immutable numerical image/source is the original `l4-bounded-v4` runtime;
all 31 checkpoint-bound numerical files still match canonical source exactly.
Sidecar hooks only observe stage counters and graph-cache decisions.

Provider receipts confirm L 4, CPU request/max 4000 milliCPU and RAM request/max
16384 MiB. Guest affinity exposes four CPU slots. Runtime is Torch 2.9.1+cu 128,
NumPy 2.3.5, NVIDIA driver 610.57.04 and an L 4 with 58 SMs. The sandbox hides the
physical host CPU model and cgroup quota/throttling files. Virtual CPU MHz/model
fields are not treated as reliable physical-host evidence. GPU clocks/power are
recorded throughout, along with per-update phase costs and actual OpenBLAS threads.

On one fixed worker (`crossover-01`):

| Same-source arm, two updates | Trainer seconds | Decisions/s |
|---|---:|---:|
| First production run, cold process/context |80.330|8,772|
| Warm process, synchronized profiler |76.487|9,213|
| Warm process, production timers |76.734|9,183|

The first update drops from 43.022 s cold to 39.476 s warm; second-update times are
37.301 and 37.250 s. The warm synchronized/production difference is 0.3%, not the
original approximately 79% rate gap. These arms have exact policy/Adam/population/
RNG/history/native-state parity. Graph decisions are identical:36 captures,
2,006 hits and 8,539 eager fallbacks. Capture/cache-shape churn therefore does not
explain their warm/cold difference.

The first attempt's fresh-child phase ran out of VRAM because the parent CUDA
allocator still reserved unused memory. This was an **attribution-harness error**,
not an engine fault. Its full 257.346 allocated seconds remain charged. The harness
now releases unused parent allocations equally before cold children; subsequent
paired cases use independent fresh processes with no parent CUDA model. No failed
attempt or partial cohort is erased.

### Justified fix: bounded graph-cache capacity

A fresh-process four/sixteen/four crossover on one worker (`cache-02`) includes
cold imports, new models and graph captures in every arm. The repeated baseline
controls physical-GPU warming and timing drift:

| Cached shapes/network | Trainer seconds | Decisions/s | Eager fallback |
|---|---:|---:|---:|
|4, before|80.809|8,720|80.7%|
|16, candidate|76.026|9,269|33.7%|
|4, after|82.257|8,566|80.7%|

Historical dispatch drops from 18.236/18.687 s to 12.102 s. Extra cold graph capture
cost is included:2.421 s versus 1.125/0.857 s. Peak allocated memory increases only
from 14.367 to 14.434 GiB, about 68 MiB. All 704,655 accepted rows and complete policy,
Adam, population, RNG, history and native states match exactly. Actual-CUDA
four-update versus split two-plus-two resume, changed-source rejection and strict
inference export all pass with sixteen shapes.

### Rejected thread-setting attribution and measured variability

`threads-03` compares independent fresh processes with four/one/four OMP, MKL and
OpenBLAS settings. Runtime symbols verify actual OpenBLAS thread counts 4/1/4;
Torch threads remain 1 and native workers remain 4. Trainer times are 89.279,
75.204 and 70.709 seconds. The one-thread result lies between the repeat baselines,
so there is **no reliable causal thread-setting win**. No single-thread performance
claim or default change is adopted.

The identical four-thread baselines differ by 26.3% in throughput on the same
worker, with exact training-state parity. Their first updates take 50.430 versus
37.576 s and their second updates 38.843 versus 33.127 s. This directly establishes
both cold-phase and continuing execution variation; startup amortization alone
is insufficient. The underlying host/scheduler mechanism remains unobserved.

### Fresh full-provider verification and reusable recipe

Both final workers start cold, have no warmup prelude or proof workload in their
timed training allocation, and execute exactly two 256-lobby updates with the same
704,655 accepted decisions. Both explicitly set library thread variables to 4;
only cache capacity changes:

| Fresh worker | Cache | Trainer seconds | Full allocated seconds | Decisions/allocated second |
|---|---:|---:|---:|---:|
| Before |4|81.411|91.409|7,709|
| After |16|66.684|73.500|9,587|

The final checkpoints have exact model, Adam, RNG, population, full history,
next-seed and native-state parity. All 256 candidate native states restore and
serialize exactly on the admitted local Rust engine. Each worker completes 2,048
natural learner episodes with zero simulator faults or rejected actions. Captured
phase windows and full provider intervals are separate; no warm-loop or epoch-reuse
numerator is substituted for allocated worker time.

`curriculum-throughput.json` now selects sixteen shapes and explicitly declares
four-thread OMP/MKL/OpenBLAS settings to prevent implicit runtime defaults. It
retains Torch threads 1, native workers 4, the original PPO objective and opponent
mixture. `reference_training.start_strong` applies the environment before importing
NumPy/Torch, then invokes the unchanged numerical trainer. A local bootstrap/API
comparison matches all exact-state groups. Ten focused local tests pass. Changing
cache/configuration is not an ordinary exact resume; existing strict source/config/
input checks remain in force. The earlier CPU engine, supported rules and policy
architecture are unchanged; no GPU rules rewrite is justified by these profiles.

The successor used **1003.124/1,200 full allocated L 4 seconds** across five
workers, including the harness failure, diagnostics, qualification, checkpoints,
idle time and teardown. Every GPU interval is provider-settled; zero owned workers,
controllers or unresolved GPU holds remain. Conservative resource accounting is
**$0.537**, with zero external-model liability, not a reconciled
invoice. The successor `CLOSED.json` blocks further dispatch. No original unused
allowance was reused, and there was no private grading, long training, cross-game
change, commit, push or publication. The selected trained bot remains unchanged.
