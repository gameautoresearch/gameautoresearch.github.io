# AutoTavern neural research baseline

Eight-player **Hearthstone Battlegrounds**, with the Rust AutoTavern simulator retained.
The source pin is `MCG-NJU/AutoTavern@1c2086a997afde46d68c6c5992cab194ddcfce88`,
client 37.0.2.256589 / server 37.0.2. Local changes add strict batch stop diagnostics
and optional immutable observation views. This is an emulator benchmark; registry
coverage and wrapper tests do not establish exact Blizzard-game fidelity.

The November 9 population-trainer performance work retains this Rust CPU engine
and uses an L4 learner. Ready observations, retained learner inputs, batched
visible scripts and grouped historical-policy CUDA graphs now run through the
actual trained-bot PPO path. The new performance recipe is
`reference_training/strong/curriculum-throughput.json` (256 lobbies, unchanged
four-learner / three-script / one-history mixture). Matched 48-lobby CUDA pairs
preserve exact training state; larger batches are separate scaling experiments.
Read the [population throughput readout](../../reports/tavern/population-throughput-2026-11-09.md)
and [trainer instructions](reference_training/strong/README.md). Historical
campaigns remain closed, and the GPU simulator rewrite remains paused.

The separately qualified [H100 pipeline follow-up](../../reports/tavern/h100-pipeline-2026-11-09.md)
reaches **18,824 accepted learner decisions/s** over a fresh full allocation,
versus 15,238/s in its repeated H100 baseline. A same-worker bracket establishes
1.38–1.42× code speedup with exact state parity; the 2× target was missed.
`reference_training/strong/curriculum-h100-throughput.json` selects the verified
collector/PPO graph path on H100/4 CPUs/16 GiB, PyTorch2.9.1+cu128 and NumPy2.3.5.
This bounded pass is closed; its unused allowance does not authorize training.

From the repository root:

```sh
uv pip install --python .venv-tavern/bin/python torch==2.15.0 gymnasium==1.4.0
uv pip install --python .venv-tavern/bin/python --editable vendor/engines/AutoTavern
.venv-tavern/bin/python simulators/tavern/run.py train \
  --config simulators/tavern/recipes/starter-local.json \
  --out evidence/tavern-implementation/my-train
.venv-tavern/bin/python simulators/tavern/run.py evaluate \
  --config simulators/tavern/recipes/heldout-example.json \
  --checkpoint evidence/tavern-implementation/my-train/latest.pt \
  --out evidence/tavern-implementation/my-eval
.venv-tavern/bin/python simulators/tavern/run.py profile \
  --config simulators/tavern/recipes/starter-local.json \
  --out evidence/tavern-implementation/my-profile
.venv-tavern/bin/python -m pytest -q simulators/tavern/tests
```

Use a **new** output directory for every invocation. Add `--build` to any command to
include release compilation within its worker-time accounting window. Standalone
setup/build/development evaluation must also be charged by the owning allocation's
outer timer; command-local accounting cannot observe time spent outside its process.
`requirements-local-lock.txt` captures this local environment, including the editable
engine path. A Linux CUDA host needs the matching official CUDA PyTorch build; the
macOS wheel is not a CUDA installation recipe.

The default recipe is a small runnable starter, not a strong trained policy.
`reference-gpu-template.json` is a longer all-seat self-play recipe, also **untrained**.
Configure the exact GPU, attached CPU/RAM, hours and seed before using an approved
allocation. `run.py` itself does not launch cloud compute; the separately authorized bounded Modal workflow below does. CPU smoke results do not measure
GPU throughput. `device` accepts `cpu`, `cuda:0` or `mps`; unavailable CUDA fails closed.

## CPU restructuring v1

The native simulator now uses compact copy-on-write minion data and board history,
direct batch stepping, grouped observation writers and parallel lobby initialization.
The full eight-player rules and public observation schema are retained. The local
128-lobby paired replay measures 1.083× simulation/observation throughput; a multiple
overall speedup has not been established. See
[`cpu-restructuring-2026-11-08.md`](../../reports/tavern/cpu-restructuring-2026-11-08.md)
for measured scope, correctness and retained evidence. Historical source-bound
checkpoints remain tied to their original source; rebuild the native extension
before starting a fresh run on this revision. The complete GPU simulator rewrite
remains paused. `gpu-throughput.json` still uses the native CPU simulator with the
existing CUDA learner; this CPU revision has no fresh CUDA qualification.

## Learning and opponents

The network concatenates 3,227 signed-log numeric features and position-preserving
embeddings for the 124 card slots. A shared MLP produces 245 masked policy logits and
a scalar value. PPO uses GAE, clipped policy loss, entropy, gradient clipping and
Adam. Tensors are staged to the chosen device once per update, then reused across
minibatches and epochs. Inference batches every ready learner seat in the cohort.
Only ready seats create transitions, losses and sample counts. Terminal rewards
received while waiting accumulate on that seat's last real decision.

`learning_seats` selects any distinct seats from 0–7. `opponent="scripted"` leaves
other seats in the Rust bot driver. `selfplay` controls all eight seats, uses the
current neural policy for other seats and trains only the selected learning seats.
`frozen` controls all eight but uses `opponent_checkpoint` for nonlearning seats.
Periodic `opponent-NNNNNN.pt` snapshots can form a reproducible frozen population;
select the checkpoint explicitly in a new recipe. Automatic population matchmaking
is not implemented. All snapshots contain the architecture, policy and optimizer.
Load only trusted local PyTorch checkpoints.

Collection uses complete cohorts with one fixed policy. It discards **all collected
trajectories for a lobby** if any controlled seat exposes a simulator guard while
that cohort remains active. Surviving trajectories end naturally and bootstrap zero;
there is no invented bootstrap from an automatically reset lobby. Fresh cohorts at
each PPO update avoid stale partial trajectories. This deliberately leaves a cohort
tail cost, reported in the complete-loop profile. It is a baseline, not a claim of
optimal asynchronous collection efficiency.

The configured 30-action turn cap is an explicit benchmark convention, not a Blizzard
rule. It prevents infinite move/freeze loops. `max_turns=100` remains a safety bound;
reaching it is a fault, never a health-ranked win. Combat/effect cutoffs likewise
produce zero placement reward, a truncation and a stop code. Batch API stop codes:
0 ongoing, 1 natural, 2 no-progress, 3 driver guard, 4 turn limit, 5 combat cutoff,
6 recruit cutoff. The optional `adapter.TavernEnv` subclasses `gymnasium.Env`,
declares spaces and reports both termination and truncation correctly. The custom
trainer uses `BatchEnv` directly, not the upstream Gym-style wrapper.

## Reproducibility, resume and budget

Each run writes an immutable `manifest.json`, effective `recipe.json` and
`source.tar.gz`, with per-file SHA-256 hashes, upstream commit, card data, full rules,
loaded native binary/build fingerprints, observation/action schema, seed derivation,
runtime and configured/observed hardware. Mutable results are separate JSON/JSONL.
The harness refuses a stale native build. Defaults are expanded from the pinned
engine in the manifest; the source archive includes Cargo.lock and the card data.

`latest.pt` and periodic snapshots are complete update-boundary checkpoints. Resume:

```sh
.venv-tavern/bin/python simulators/tavern/run.py train \
  --config simulators/tavern/recipes/starter-local.json \
  --resume evidence/tavern-implementation/my-train/opponent-000002.pt \
  --out evidence/tavern-implementation/my-resume
```

The original recipe and source hashes must match. Model, optimizer, Python/NumPy/
Torch/device RNG, next cohort seed, previous elapsed allocation and exact Rust game
states (including engine RNG) are saved. Restoring checks every engine state; training
continues at the next complete cohort, not halfway through a discarded rollout.
Cohort seed is `(seed + update_index*1000003) mod 2^64`; Rust derives each lobby seed
from cohort seed, game index and episode index using splitmix64. Native thread count
does not affect simulator results. Hardware/runtime changes can affect neural numerics.

`allocated_worker_hours` is optional for local smoke. For benchmark use, set a fixed
allowance and the exact machine allocation. Elapsed wall time counts CPU simulation,
GPU-idle periods, compilation, transfers, inference, updates, development evaluation,
checkpointing and startup. The total carries forward on resume; there is no sample-only
score or separate CPU budget. A deadline during rollout discards that incomplete
cohort and preserves the last complete checkpoint. An in-flight optimizer/checkpoint
operation may finish past the deadline; the actual overshoot remains charged. Fixed
CPU/RAM/GPU reservations must be enforced by the allocation provider; local metadata
does not pretend to reserve hardware. Report all command sessions and outer-worker
provisioning/development time in the experiment ledger.

## Evaluation and performance

`starter-local.json` has public practice seeds. `heldout-example.json` fixes a separate
four-seed × eight-seat × four-lobby schedule. It is a reproducible public **example**
of a held-out protocol; truly hidden grader seeds/opponents must be generated and
kept outside the participant workspace. Do not tune on the grading schedule.
Evaluation rotates one policy through every seat against seven scripted or frozen
opponents. Eight identical policies' mean placement would be forced to 4.5.
Primary outputs are mean placement, normalized return `1-2*(placement-1)/7`, top-four
rate and per-seat means. Every episode and fault is retained. A schedule with any
fault has `complete_valid_schedule=false` and is not an admission-quality score.
Fresh-seed retraining and uncertainty across independently trained seeds are required
before claiming an improved or strong reference.

`profile` performs real neural rollouts and PPO updates with synchronized device stage
timers. `profile.json` separates environment initialization, observation, inference /
transfer / sampling, rollout storage, simulator step, PPO transfer/update and IO;
`accounting.json` includes startup/import/build and all residual wall time. Ready player
decisions, accepted samples, batch calls and completed placements are distinct units.
Set `copy_observations=true` for the original copying Python path; false uses immutable
Python-owned byte-buffer views (safe across subsequent calls). Rust still constructs
and serializes buffers. `native_threads` is configurable; scoped worker creation and
copying remain profiling targets. No GPU simulator rewrite is warranted by the old
random-action probe alone.

Controlled local performance ablation (run without other local training/build jobs):

```sh
.venv-tavern/bin/python simulators/tavern/benchmark.py \
  --trials 3 --out evidence/tavern-implementation/my-ablation
```

It interleaves copying/view modes and one/four native threads on identical neural
rollouts and updates, then requires bitwise-equal final weights, episode records and
sample counts. It records each session and the encompassing worker time. An earlier
checkpoint resumed after a completed session charges that entire prior session's
ledger, including work after that snapshot; rewinding a checkpoint never refunds time.

On the inspected local CPU workload, four native simulator threads on an already
configured four-thread allocation improved the complete loop; observation-copy removal
alone did not improve end-to-end time. `throughput-local.json` selects the measured
four-thread/copying configuration. Repeat profiling on the actual allocated machine
and a stronger policy before choosing GPU settings or rewriting thread/buffer code.
The dated measurements and their limits are in `reports/tavern/implementation-2026-11-08.md`.


## Historical NVIDIA CUDA campaign

The earlier campaign used a $12/two-hour allowance. Its six attempts, failed work,
reservations, receipts and source payloads remain unchanged. The later Nov 8 user
instruction superseded those dollar/hour/attempt limits for performance engineering.
Use `structural_launch.py` below for this authorization; `launch_gpu.py` retains the
historical gated workflow and is not the new campaign launcher.

Provider billing refreshes require a **new** output path:

```sh
/Users/K.Z./.local/share/uv/tools/modal/bin/python simulators/tavern/collect_billing.py \
  --out evidence/tavern-implementation/my-new-billing-receipt.json
```

Current defaults selected for this CUDA path are in `recipes/gpu-small.json`:
32 lobbies, a 128-hidden-unit/8-dimensional embedding model, one Torch CPU thread,
four persistent simulator workers, 1,024-sample PPO minibatches, packed inference
returns, deferred learner metrics and no stepping/encoding of finished cohort lobbies.
The rule/action/observation semantics and ready-seat losses stay the same. The new
learning recipe uses gamma=1 so negative terminal placement does not incentivize
prolonging a losing episode. Baseline and optimized arms use the same gamma/model,
seeds, game settings and workload.

`worker_backend="scoped"` retains the original native threading path for controlled
comparisons; `pooled` reuses guarded 64 MiB-stack Rayon workers. `skip_finished_lobbies`
freezes already-finished cohort lobbies instead of repeatedly simulating irrelevant
new episodes while other lobbies finish. `fast_inference` uses the checked ready/mask
invariant and one packed D2H result instead of repeated scalar/mask synchronizations.
`defer_learner_metrics` reads aggregate metrics at the end of an update. It preserves
finite-loss checks and all actual gradient steps.

The optional `start_turn=8` profile constructs deterministic, valid scripted-bot
continuations with populated boards, saves state hashes and refuses already-eliminated
learning seats or guards. It is explicitly a busy continuation workload, not a full
from-start policy-strength evaluation. Normal learning/evaluation starts at turn one.
Profiles separate first-update cold setup from later warmed update timings, include
CUDA event diagnostics and allocator peaks, and keep overlapping diagnostic times
separate from the non-overlapping full-loop stage totals. `nvidia-smi.csv` records
utilization/memory as diagnostics; all allocated idle GPU time still counts.

To compare locally, the following command automatically acquires the shared exclusive
advisory performance lock. Other local timed workloads must use that lock too:

```sh
.venv-tavern/bin/python simulators/tavern/paired.py --device cpu --trials 2 \
  --checkpoint evidence/tavern-implementation/learning-64/latest.pt \
  --out evidence/tavern-implementation/my-local-optimized-pairs
```

The old pilot's native fingerprint predates scheduling changes. The explicit
`migrate_performance_checkpoint.py SOURCE DESTINATION` utility verifies every original
mechanics source and the card-data SHA-256, allows only the named scheduling/binding
files to differ, keeps all weights untouched and writes a migration receipt. It never
silently accepts arbitrary rules/data drift. Ordinary exact resume still requires the
same immutable source and recipe.

See `reports/tavern/cuda-performance-2026-11-08.md` for actual CUDA results, costs,
failure receipts, learning limits and remaining work. A strong reference policy and
commercial-game fidelity remain separate validation requirements.


The optional `validated_sampler=true` path reproduces the pinned PyTorch single-sample
Exp(1)/argmax algorithm while removing redundant device probability-check kernels.
CPU readiness/masks and finite returned logp/value checks remain enforced. It is
qualified on CUDA PyTorch 2.9.1 only and rejects other CUDA versions until requalified.
Two paired profile repeats plus a full 64-update learning pair preserve weights, CUDA
RNG, samples and placements exactly. Its initial/full-training gains did not generalize
to busy continuations, so it is **off by default**. This is a measured optional
optimization, not part of the three-seed headline speedup. See the sampler receipt
under `cuda/cuda-sampling-v6/sampling/sampling-summary.json`.


## Structural performance campaign

The Nov 8 performance continuation uses `structural_launch.py` and separate
`evidence/tavern-implementation/structural-20261108/<run-id>/` records. It preserves the
historical campaign, uses one L4 with four fixed physical CPU cores and 16 GiB RAM per
experiment, zero warm pool/retries, a bounded one-hour function and unique immutable
source payloads. Full worker time, cold image builds, evaluation, idle time and
provider costs remain part of accounting. No other game's jobs are managed.

```sh
.venv-tavern/bin/python simulators/tavern/structural_launch.py \
  --phase probe --run-id my-structural-probe
.venv-tavern/bin/python simulators/tavern/structural_launch.py \
  --phase learning --run-id my-structural-qualification
.venv-tavern/bin/python simulators/tavern/native_profile.py \
  --trials 5 --out evidence/tavern-implementation/my-scripted-native-profile
```

The native profiler automatically takes `evidence/local-performance.lock`, records
a public scripted-policy workload with occupied boards and late combat, then
interleaves exact static/dynamic replays. It hashes all observations/masks/results and
checks final engine/RNG JSON. Neural train/profile timings and native-only timings
are different measurements. `structural_experiment.py` also acquires that lock.

New explicit recipe switches (off unless selected by a measured recipe):

- `native_scheduler="dynamic"`: one independent lobby per Rayon task; `static` keeps
  worker-sized blocks for comparison. Mechanics, seed domains and seat order are unchanged.
  At most one idle pool size is cached; live environments retain their own references.
- `preencode_learner=true`: compute signed-log numeric inputs once per PPO update.
  Card embeddings remain trainable on every forward; update/gradient semantics match.
- `graph_inference=true`: cache exact-ready-count CUDA graphs, preserving live CUDA
  sampling RNG and finite output checks. Cold capture is charged. `actor_graph_max_shapes`
  defaults to 64; rare additional shapes use eager inference, bounding graph lifetime.
- `graph_learner=true`: one graph for full minibatch forward/backward/gradient clipping;
  Adam and short remainder minibatches remain eager. No padding or duplicate samples.
- `actor_device="cpu"`: explicitly separate CPU actor from CUDA learner. Copy one
  frozen policy per cohort without consuming caller RNG. This changes device numerics
  and sampling streams and must receive its own learning comparison.
- `fused_adam=true`: CUDA fused Adam kernels with the same optimizer hyperparameters;
  floating-point numerics differ and are separately reported.

CUDA graph/sampler paths require the qualified PyTorch 2.9.1 runtime and CUDA;
unsupported combinations fail preflight. Normal exact resume remains source/recipe
bound. Graph caches are reconstructed from a checkpoint with capture RNG restored.

## Complete GPU simulator development

`device/core/` contains the independent CUDA-compatible complete rules source;
`vendor/engines/AutoTavern` provides the supported native simulator. The complete GPU
rewrite is **paused by user instruction** while targeted CPU restructuring and
correctness checks are completed. Retained B200 builds passed bounded differential,
adversarial, GAE, PPO and resume gates; later wide replay exposed failures in
experimental launch/cache configurations. Those experiments are retained separately
and are not enabled for normal training. Normal training uses the native simulator
and compact transport recipe.

The current stock LLVM NVPTX builder and bounded developer launcher freeze each
attempt in a new evidence directory:

```sh
.venv-tavern/bin/python simulators/tavern/native_build.py \
  --out evidence/tavern-implementation/device-port/my-device-build
PATH=/Users/K.Z./.local/bin:$PATH \
  /Users/K.Z./.local/share/uv/tools/modal/bin/python \
  simulators/tavern/native_launch.py --run-id my-device-attempt \
  --build evidence/tavern-implementation/device-port/my-device-build
```

The local builder uses the pinned nightly-2026-04-02 stock Rust/LLVM toolchain,
including rust-src, llvm-tools and llvm-bitcode-linker. The retained qualified
compiler host is macOS ARM64. The older SDK/NVVM launcher is preserved as development
history. Separate relocatable device assembly/linking avoids its previous
whole-program assembly bottleneck. Portable stable sorting preserves CPU tie order.

The GPU launcher requires the separately authorized paid Modal workflow. It builds SM100 code
for one B200 with four fixed CPU cores and 16 GiB RAM, zero warm containers/retries,
and a 1,800-second GPU function limit. CPU image compilation/assembly, failures,
transfers and all GPU allocation time remain charged. The split builds simulation
and checkpoint modules from one shared state type, verifies their GPU ABI layouts,
and validates parsed checkpoints before replacing running games. It does not omit
supported content or execute transitions through the CPU oracle.

The latest native/hybrid results and direct original-source comparisons are in
`reports/tavern/structural-performance-2026-11-08.md`. Original native simulation and
the earliest complete neural trainer are separate baselines; historical ratios
must not be multiplied. Neither the direct native results nor the GPU correctness
qualification establishes a 10× full-task speedup.

## Local replay viewer

The shared Replay Room includes three complete eight-player scripted-bot demos:
<http://127.0.0.1:5180/?game=tavern>. Run `npm run dev -- --port 5180 --strictPort`
in `replay-site/` if it is not already running. The viewer has seat selection,
combat playback, round seeking, card inspection and lobby standings. Only fresh
public-demo projections are bundled; no private games or checkpoints are exposed.
See the shared viewer README for export and verification details.

## Autonomous research contract

The full eight-player research task, deliberately small editable PPO starter and
bounded DeepSeek smoke now live in [`autoresearch/`](autoresearch/README.md).
See [`autoresearch/benchmark.json`](autoresearch/benchmark.json) for discoverable
settings and [`autoresearch/CONTRACT.md`](autoresearch/CONTRACT.md) for source-bound
submissions, host-private placement grading and allocated-worker accounting.
Preparation is offline; paid execution requires an explicit `--execute` flag.
The new smoke has its own retained600-second/$2 ledgers and does not reset historical
campaigns. Reference normalization remains gated and the GPU-rules rewrite paused.

## Versioned joint-action admission

The November8–9 overnight campaign exposed a recruit-phase cross-seat interaction:
an earlier player's elimination can open a new pending choice for a later player
after that later player's response was chosen. The admission-v2 driver preserves
the original legal masks, rejects inputs illegal in the observed frame, and defers
responses whose context changed. `BatchEnv.step` reports `applied` and `deferred`;
unexecuted responses do not enter PPO trajectories, and rewards received while
waiting credit the previous actual decision. Pending-menu identity is checked as
well as the current mask.

This has a new native fingerprint and source binding. Ordinary old checkpoint
resume/inference remains rejected; historical releases and records are retained.
Fresh B2 controls and independent researchers use the qualified repaired release.
The old control/anchor failures are not replaced by new scores. Qualification,
budgets and live results are in
[`overnight-2026-11-08.md`](../../reports/tavern/overnight-2026-11-08.md).
