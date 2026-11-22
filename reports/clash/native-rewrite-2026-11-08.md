# Independent Clash rewrite, 2026-11-08

The frozen `clash-fixed8-v1` task now has independent C++ CPU and CUDA simulation,
device-resident neural PPO, native evaluation/replay, lossless full-state save/resume,
and cumulative allocation accounting. Java is the external differential oracle and
historical comparator. It executes no native training transition.

The correctness and environment handoff is complete for the frozen eight-card
task. The user accepted the current performance and stopped further optimization.
The latest matched H100 result is **6.448×** end to end; an earlier paired H100
allocation measured **7.045×**. The roughly 10× target was not achieved and is no
longer being pursued in this handoff. No commercial-game fidelity, other-deck
coverage or learned policy strength is claimed.

## Final correctness and runnable defaults

|Recipe|Simulation|Neural training|Declared allocation|
|---|---|---|---|
|`native-cpu-smoke.json`|C++ CPU|CPU|Local correctness smoke|
|`native-l4.json`|C++ CPU|CUDA FP32|L4 + 16 CPU cores + 64 GiB RAM|
|`native-cuda-l4.json`|CUDA|CUDA FP32|L4 + 16 CPU cores + 64 GiB RAM|
|`native-h100.json`|CUDA|CUDA FP32|H100 + 32 CPU cores + 128 GiB RAM|

The H100 recipe reproduces the measured 2,048-environment, 32-update control.
Recipes do not allocate hardware. Use the module entry with an explicit recipe
and a new output directory; native evaluation/replay commands are in the native
README. Both simulators are independent of Java transitions; Java is the oracle.

The final owning runtime files match the successful actual-GPU qualification
archive byte for byte (71 checked files). Current-source CPU live/truncated PPO
resume passes, including all model/Adam/RNG/output/counter state and every raw
world byte except four documented rebound pointer addresses; seven source/data/
recipe/backend/runtime tamper controls reject. The earlier complete oracle and
H100 suites retain 14,729 Java/native frames and 5,288 CPU/CUDA frames. Allocation,
lossless snapshot and native evaluation/replay controls also pass. Checkpoints
remain source-bound; historical source archives and all failed attempts remain.

The packed-head option passed actual GPU math, graph, production and tail resume
checks, but remains disabled because it was slightly slower. CPU neural and live-
entity encoder prototypes are unintegrated; GPU encoder qualification remains
pending. They are documented research artifacts, not shipped defaults. Rejected
samplers and the marginal lazy-route candidate were not promoted. All optimization
work and GPU allocations have ended.

Final evidence: `structural-20261108/final-qualified-runtime-identity-1.json`,
`final-wrap-cpu-resume-1/result.json`, and the completed matrices below.

## Latest complete H100 comparison

The final six-arm comparison completed two reverse-order seeded trials on one
H100/32-core/128-GiB worker. Every arm completed 4,194,304 learner rows and 8,192
Adam steps at 2,048 environments and 32 updates, without faults, illegal actions
or truncation.

|Implementation|Combined complete-process seconds|Learner rows/s|Original-relative speed|
|---|---:|---:|---:|
|Original Java integration|806.493|10,401|1.000×|
|Native CUDA, strict FP32 default|125.068|67,072|6.448×|
|Native CUDA, experimental packed head|127.590|65,747|6.321×|

The packed mode was about 2% slower than the default. Per-process spans include
initialization, source archive, rollout, inference, transfers, GAE, PPO, reset,
checkpoint and cleanup. The complete research function additionally charged
setup, correctness/profiling work, idle gaps and commits: 1,247.29 seconds. It
used the exact source archive SHA
`de6b61ccc1744740981ca456ffc2834154ba31ff78623ff96714f30710516607`.
This is configured-workload throughput, not a strength or cross-Java trajectory
identity claim. The earlier 7.045× is a separate allocation, not a multiplier.

Evidence: `structural-20261108/native-packed-2/summary.json`.

## Earlier H100 matrix and backend choice

One H100/32-core/128-GiB worker completed all eight planned arms: two fresh
seeded trials in reverse order, 2,048 environments, 32 updates, otherwise the same
Policy64/e12, 32-step, two-epoch, minibatch 1,024 workload. Every arm completed
4,194,304 learner rows and 8,192 Adam steps without fault, illegal action or
truncation. Each native run included roughly 2,000 natural matches and more than
one million late-game player rows.

|Implementation|Combined arm seconds|Learner rows/s|Original-relative speed|
|---|---:|---:|---:|
|Original Java integration|910.023|9,218|1.000×|
|Native CUDA, strict FP32|129.178|64,939|7.045×|
|Native CUDA, BF16 neural forward|127.628|65,727|7.130×|
|Native CPU simulation + CUDA learner|136.690|61,370|6.658×|

The Java arms varied substantially: 591.736 s and 318.287 s. Native CUDA FP32
was 64.339 s and 64.839 s. The first-seed 9.20× ratio is not the combined result.
Both native FP32 backends have exactly matching natural/tick/activity counters
in both trials; Java/native trajectory identity and policy strength remain
separate from this configured-workload comparison. The complete research
function used 1,638.06 allocated seconds, including fresh H100 qualification,
setup, idle gaps and commits, in addition to the per-arm process spans above.

Fresh H100 qualification passed 5,288 CPU/CUDA frames, five natural games,
features/gradients/capture, BF16 numeric controls, production and tail exact
resume, and exact CPU/GPU-backend neural/public-state agreement.

Use CPU simulation plus CUDA training for the measured 256-environment L4
recipe; it is now `configs/native-l4.json`. The explicit CUDA alternative is
`configs/native-cuda-l4.json`. At this measured H100 scale CUDA simulation is
slightly faster. CPU simulation here is independent C++, never Java. The performance target remained unmet; subsequent wrapping ended optimization
at the user’s request.

Evidence: `structural-20261108/native-h100-4/summary.json`. The first H100 launch
failed after a controller connection reset; two later launches failed before
function submission. All attempts and charges remain in the ledger. The first
packed-head launch used stale transport source and executed no candidate work;
it is explicitly invalid. The launcher now creates immutable content-addressed
archives before mount hydration and requires the exact source hash at worker
entry and result validation. Its replacement run has a new directory.

## Earlier complete L4 matrix

One L4/16-core/64-GiB allocation ran two fresh seeded trials in reverse order,
256 environments, 32 updates, 32 rollout decisions, two epochs, minibatch 1024, Policy64/e12,
sparse terminal self-play and all-row policy loss. Each arm completed 524,288 learner
rows and 1,024 Adam steps with no illegal actions, faults, overflow or truncation.

|Implementation|Complete-process learner rows/s|Original-relative speed|
|---|---:|---:|
|Original Java integration|11,101|1.000×|
|Native CUDA, strict FP32|22,188|1.999×|
|Native CPU simulation + CUDA learner, strict FP32|27,444|2.472×|
|Native CUDA + compiled FP32 loss|11,869|1.069×|
|Native CUDA, TF32 neural math|23,186|2.089×|
|Native CUDA, BF16 neural forward|23,696|2.134×|

Rates are total samples divided by total measured subprocess span, including
initialization, source archive, rollouts, inference, transfers, GAE, PPO, resets,
checkpoint and cleanup. Common worker setup/compilation/qualification, idle gaps
and volume commits are additionally charged in the complete allocation receipt;
these per-arm rates are not whole research-campaign throughput. Native libraries
may reuse the qualified build cache. No cold-compilation cost is discarded from
the parent allocation. The whole function lasted 1,051.08 s.

Compiled FP32 improved warm iteration time from about 0.55 s to 0.48 s, but arm times
were 56.2 s/32.1 s versus 23.7 s/23.6 s for the default. Compilation/capture startup
therefore made it a poor default for this short workload. TF32/BF16 are explicitly
separate neural numerical modes. Defaults remain strict FP32; integer simulation,
observation widths and legal rules do not change.

Native CPU/CUDA and native CUDA used identical actions, rollout data, model, Adam,
all exposed RNG streams and public continuations in the dedicated backend proof.
Their complete benchmark natural/tick/activity counters also match exactly in
both trials. Historical/current Java samplers and seed/RNG layouts differ from
native; cross-Java trajectory identity and equal policy strength are not claimed.

Evidence: `evidence/clash-implementation/structural-20261108/native-completion-1/summary.json`.
The exact immutable source overlay, checkpoints and complete receipts are retained
on private Modal volume `clash-structural-artifacts-20261108`.

## Earlier matched allocations and batch scaling

A previous two-trial L4/16-core/64-GiB allocation measured 5,536 original versus
17,640 native GPU learner rows/s (**3.186×**) at 256 environments. A separate matched
checkpoint study measured 6,308 original versus 18,941 native GPU (**3.003×**).
These are distinct allocations, not factors to multiply. CPU host performance
varies materially; the best earlier ratio is not the latest-worker result.

At 1,024 environments, two complete 32-update trials measured 15,946 original versus
32,269 native GPU rows/s (**2.024×**). Both arms completed 2,097,152 rows/4,096 Adam steps
per trial. The larger batch increased absolute native throughput but benefited
the original integration more, so it did not improve the relative gain. Both
variants played about 1,000 natural games/trial and substantial late-game rows;
this was not a short empty-board timing.

See `native-bucket-1`, `native-codec-1` and `native-batch-scaling-1` beneath the
same structural evidence root. Separately scoped mature raw CPU timing measured
199,354 environment decisions/s versus 80,970 for pinned Java with four threads
(**2.462×**), using a trained action tape, reverse-order repeats and the local
performance lock. This component ratio is not multiplied by any total ratio.

## Independent simulation and checkpoint qualification

The current world preserves command admission/queueing, deployment, elixir/deck
RNG, pathfinding, movement, targeting, projectiles, damage, buffs, king activation,
public observations/legal masks and natural endings. Full 512-entity storage and
original 200-row/7,024-int observation/2,307-action schema remain. Overflow is an
explicit fault, never a reduced rule or observation cap.

After the final padding/constructor fixes, the full CPU oracle compared 14,729 frames,
206,912,992 observation integers and 67,959,606 mask bytes, covering all eight cards,
16 natural outcomes and four intentional truncations. Every public output and
repeated read matched pinned Java; no fault/overflow occurred. Selected-action
validation separately checks all 2,307 mask entries and the original admission path.

The actual CUDA binary passed 5,288 CPU/CUDA frames with five natural games,
complete outputs, raw restore and neural feature/gradient/capture checks. All 25
launch/render combinations pass their complete-output tests. Production Policy64/e12,
1,024-row minibatches, all four 32/64/96/200 neural buckets and historical 48/16 tails
have exact in-mode live/truncated PPO resume checks. They compare model, Adam,
all Python/NumPy/Torch/CUDA RNG streams, graph reconstruction, continued losses,
counters, outputs, 32 future transitions and the complete raw world image, normalizing
only four documented rebound pointer addresses. No padding bytes are excluded.

A prior CPU/CUDA-learner raw checkpoint check failed despite identical semantic
training/public results. Exact GCC 13.3 analysis classified 14 nonpointer bytes as
padding, semantic/unknown 0. Named zero-initialized reserved fields preserve every
semantic offset/size; value-initialized CPU scalar/array allocations also make
nested padding deterministic. Actual allocator-poison tests on both local and
GCC 13.3 constructors passed 768 raw-equal frames, 30 resets and 28 restores. The final
CPU full-oracle and exact-resume tests pass. Both failures and diagnoses remain.
World bytes remain 4,195,384; no serialized bytes are discarded.

Lossless zlib1 encoding reduces the 256-world checkpoint from 1,091.9 MB to 39.4 MB,
preserving all 1,074,018,304 simulator bytes. A two-trial raw-vs-encoded comparison
measured a **2.70% complete-process gain**; copy/encode/save took 1.67–1.72 s versus
2.36–2.44 s raw. Shape, size, corruption, trailing-data and checksum failures reject.

Each new checkpoint binds source/data/rules, compiler/binary, runtime, hardware
and neural helper libraries. Drift/tamper checks fail before continuation. The
module CLI retains a locked cumulative allocation sidecar, including final-save
and cleanup tails. It rejects account forks, corruption, hardware/budget drift
and silent missing-ledger resets. GPU-allocation gaps remain charged when clocks
are comparable; unknown moved/abandoned tails are labelled. Provider full-allocation
receipts remain authoritative; iteration-boundary budgets are not hard cutoffs.

## Native evaluation and remaining performance work

Native CPU and actual CUDA evaluation forbid Java startup. CUDA-produced action
tapes replay 96 public transitions exactly on native CPU; reordered schedules keep
each game's policy RNG/tape unchanged. A busy raw CUDA snapshot restores every
byte and continues exactly. CPU scripted-vs-wait also reaches a natural outcome.
Diagnostic truncations/illegal actions/simulator faults remain distinct and have
no official score. These public development checks are not private grading.

The latest optional compiled/TF32/BF16 modes pass explicit numeric, master-parameter,
capture-after-Adam and exact separate-process resume checks. A five-candidate actual L4 study compared stack/register changes and constant
reads for the unchanged 36×64 arena. Each candidate first passed every output on
546,816 exact compared frames from the trained tape, including all replicas, then ran
three timed repeats under the performance lock. Baseline raw simulation measured
49,793 environment decisions/s; the promoted constant-dimension candidate measured
52,555 environment decisions/s (**1.0555×**). Actions were preloaded; neural work and
host output transfer are excluded from this component timing. It is not multiplied
by the complete-training ratios.

Only the four-file dimension specialization is promoted. CPU/generic dimensions
remain dynamic, and every semantic offset plus the 4,195,384-byte world size is
unchanged. Post-promotion CPU checks include 33,448 helper comparisons, the full
14,729-frame Java oracle and exact live/truncated training resume. Other candidates
remain evidence-only: lower register counts did not change measured occupancy, and
combined changes gained only about 0.4% over the selected candidate.

Evidence: `structural-20261108/native-kernel-variants-1/summary.json` and
`native-pathfinding-20261108/candidate-C-promotion-1/summary.json`.

Current runs preserve full source overlays before executing. The older
optimized-1/baselines-1 overlay was a provider function mount and could not be
recovered through its available API; its manifest hashes and receipts remain,
and no guessed source is substituted. Historical allowances/failures are unchanged.
The user removed old campaign spending/attempt ceilings; provider billing remains
separate from conservative estimates and is provisional while intervals settle.
