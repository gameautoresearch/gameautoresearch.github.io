# Pokémon autonomous RL baseline

This game-local harness runs pinned Pokémon Showdown **gen9randombattle** mechanics in
persistent CPU Node processes and batches the player-visible observations for a PyTorch
masked policy/value network. It does not connect to a server, use a network port, or need
poke-env/WebSockets. No commercial-client parity or competitive reference strength is implied.

Run all commands from `/Users/K.Z./Desktop/projects/game-autoresearch-bench` (the workspace root). The reference-only Polytopia checkout is not used.

For future source-only recipe experiments, use the
[recipe v3 successor](autoresearch/recipe_campaign_v3/README.md) and its
[incident-hardening readout](../../reports/pokemon/recipe-v3-hardening-2026-11-11.md).
It preserves the completed recipe4h campaign's frozen source and requires a new
explicit v3 authorization and exact-source live qualification before paid use.

For new four-model research campaigns, use the
[v2 host controller and transport-recovery guide](autoresearch/pilot4h_v2/README.md).
It isolates inference billing failures between models and preserves interrupted
sessions. The completed November 9 pilot and its source-bound runtime remain
frozen; this successor does not restart its paid runs or renew its allowances.

## Three-Pokémon form — November 14, 2026

The current harness form is **`pokemon-3v3-v1`**: `gen9randombattle@@@Max Team Size = 3`, the same game with three
Pokémon a side. Its native certificate, admitted runtime and release are under `evidence/pokemon-3v3-20261113/`
and `evidence/releases-20261114-3v3b/pokemon`. See the [readout](../../reports/pokemon/pokemon-3v3-form-2026-11-13.md)
and `native/README.md` (team size). The six-Pokémon sections below are history.

## Selected autoresearch release — November 9, 2026

The selected successor is **`pokemon-gen9-random-6v6-v4`**. Read the
[release catalog](autoresearch/releases/catalog.json),
[v4 workflow](autoresearch/releases/v4/README.md) and
[current campaign readout](../../reports/pokemon/strong-bots-five-hour-2026-11-09-evaluator.md).
It adopts qualified native completion with 4,096 phases / 16 unavailable-choice retries,
keeps 700 observations / 14 actions, and grades against pinned power/retained-neural/
public-information Foul Play opposition. Sole published legal actions are dispatched
before search and counted separately. Revival Blessing targets use an explicit
public-stat policy, also logged separately. A real L4 smoke verified exact full-state
resume, explicit immutable submission/finalization and independent private grading.
The weak starter is randomly initialized; strong-reference and human-strength
calibration remain unqualified. Old v1 source, declarations, 512-phase results,
checkpoints and closed banks, plus frozen v2/v3 releases, remain preserved.

The frozen v4 workflow completed 24 natural public/independent private probe games
without faults or caps after its real L4 resume proof. Its revival policy was
checked against the captured failure history and a separate native mechanic
fixture. The preserved v3 opponent's unsupported revival request remains documented.
Sustained GLM128 continuation/control has priority for both training GPU slots.

The active five-hour campaign uses the separate explicit uncapped Modal ledger
under `evidence/strong-bots-20261109-0516Z/`, with zero model API spending.
Preparation does not authorize another campaign. This campaign uses the supported
full Showdown CPU backend and L4 policy training. The separate native CPU
qualification is documented below.

## Current direction

The frozen `gen9randombattle` **native C++ CPU backend is qualified and admitted**
through a source/data/evidence-bound certificate. Its ordinary training CLI, full
checkpoint continuation and neural export checks pass. The native hot path does
not invoke Node or Showdown. See [`native/README.md`](native/README.md) for the
admitted build/run commands and the [completion readout](../../reports/pokemon/native-cpu-completion-2026-11-09.md)
for source-specific correctness and performance evidence. The measured L4 warm
learner loop is 83.7k new both-seat rows/s; the 100k target remains unmet. New GPU
simulator work remains stopped.

The existing Showdown-backed CLI and campaign defaults below remain available.
For those CPU optimizations, use `recipes/cpu-optimized.json`. Native admission
does not restart campaigns or qualify private scores or expert strength.

## Setup and commands

The inspected local environment is `.venv-pokemon`, Python 3.12.9, Node 24.18.1, NumPy
2.5.4, PyTorch 2.15.0. Recreate with `uv venv --python 3.12.9 .venv-pokemon` and
`uv pip install --python .venv-pokemon/bin/python -r simulators/pokemon/requirements.txt`.
Use a CUDA-enabled PyTorch wheel on the specified GPU host. Build the already checked-out
Showdown pin with `node build` inside `vendor/engines/pokemon-showdown`; its minimal runtime
needs the pinned checkout's `ts-chacha20` dependency. For a fresh install use the checkout's
package lock and `npm ci --omit=optional`, then `node build`. Engine pins and every actual
executed JS/data/dependency hash are checked/recorded at run creation.

```sh
# Short, bounded local learning smoke (CPU only).
.venv-pokemon/bin/python -m integrations.pokemon.cli train \
  --config simulators/pokemon/recipes/smoke.json \
  --output evidence/pokemon-implementation/my-smoke

# Longer local smoke with development evaluation charged to the same clock.
.venv-pokemon/bin/python -m integrations.pokemon.cli train \
  --config simulators/pokemon/recipes/learning-smoke.json \
  --output evidence/pokemon-implementation/my-learning-smoke

# Resume at a completed vector-batch boundary; only iterations/time may change.
.venv-pokemon/bin/python -m integrations.pokemon.cli train \
  --config simulators/pokemon/recipes/smoke.json --iterations 6 \
  --resume evidence/pokemon-implementation/my-smoke/checkpoint.pt \
  --output evidence/pokemon-implementation/my-resume

# Frozen public practice, deterministic learner vs two seeded scripted opponents.
# Omit --checkpoint to evaluate the seeded untrained neural initialization.
.venv-pokemon/bin/python -m integrations.pokemon.cli evaluate \
  --config simulators/pokemon/recipes/smoke.json \
  --checkpoint evidence/pokemon-implementation/my-smoke/checkpoint.pt \
  --schedule simulators/pokemon/recipes/practice.json \
  --output evidence/pokemon-implementation/my-practice

# Held-out schedule. Never use its results for recipe/hyperparameter selection.
.venv-pokemon/bin/python -m integrations.pokemon.cli evaluate \
  --config simulators/pokemon/recipes/smoke.json \
  --checkpoint evidence/pokemon-implementation/my-smoke/checkpoint.pt \
  --schedule simulators/pokemon/recipes/heldout.json \
  --output evidence/pokemon-implementation/my-heldout

# Controlled complete-loop profile: scalar inference, batch inference, then cached
# encoding/terminal-message elision; identical decisions and canonical traces required.
.venv-pokemon/bin/python -m integrations.pokemon.cli profile \
  --config simulators/pokemon/recipes/smoke.json --repeats 3 \
  --checkpoint evidence/pokemon-implementation/my-learning-smoke/checkpoint.pt \
  --output evidence/pokemon-implementation/my-profile

node simulators/pokemon/tests/test-engine.cjs
.venv-pokemon/bin/python -m unittest integrations.pokemon.tests.test_rl -v
```

Each output directory must be new. `manifest.json` and its `source/` snapshot are exclusive
creation/read-only files; retain their hashes with results. Runs write `metrics.jsonl`,
`checkpoint.pt`, `results.json`, `accounting.json`, and `failure.json` on failure. Checkpoints
are trusted local PyTorch files: do not load arbitrary untrusted pickle checkpoints.
An evaluation opponent can be a frozen neural checkpoint: supply `--opponent-checkpoint PATH`;
this overrides the schedule opponents and records that file's SHA256. Both learner seats are
already represented in the schedules. Frozen historical policies live in the training checkpoint.

## Rules, actions and visibility

- Showdown pin: `567c1cedddeaa2f35576005d3bfd526cf4f8a13e`; poke-env provenance pin:
  `dd35fb672bb1cb3f00ee67769e3a9d86688d8d54` (not on the hot path). No upstream edits required.
- Battle and each random team's PRNG receive separate domain-separated 64-bit seeds:
  `SHA256("pokemon-v1:<integer>:battle|team1|team2")`, first eight bytes as four little-endian
  uint16 words. Policy, opponent and team/battle seeds are distinct. NumPy seeding alone is
  not used as a substitute for engine/team seeding.
- The adapter copies upstream `getPlayerStreams` channel routing. It never exposes the
  omniscient stream or the `end` payload, which includes both private teams. The encoder
  accepts only one player's own requests plus that player's visible protocol. Dex lookups
  use public rules data. Opponent unrevealed slots/items/abilities/moves remain unknown.
- Stable 14 actions: 0–3 move slots, 4–7 corresponding Terastallize moves, 8–13 team switches
  in the current private request order. Disabled moves, trapping, exhausted tera, forced
  switches, waiting seats and Revival Blessing requests are handled. Secret trapping can
  cause an upstream `[Unavailable choice]` despite a player-visible legal mask; it prompts
  a retry and is logged separately. Other errors abort the run and produce a categorized failure record (policy illegal action, engine rejection, simulator stall, simulator/adapter fault, or infrastructure failure).
- The encoder uses 12 team slots × 12 categorical tokens and 24 numeric features, 40 global
  features, four global tokens, and 14 × (three tokens + 12 numeric features) action features.
  Shared 16,384-bucket FNV-1a embeddings encode names categorically. Hash collisions are a
  documented representation tradeoff, not an ordered numeric species/card encoding.
  Public type effectiveness/action power is a feature, not an oracle for hidden ability/item
  effects. This is a feed-forward partial-observation baseline, not a complete Showdown
  client state tracker: ambiguous Illusion/Transform identities and all special event effects
  are not reconstructed perfectly. Raw protocol boundary remains player-visible.
- Natural win/loss/tie yields +1/−1/0. Decision/allocation limits are **truncations**, bootstrap
  the final visible state's value for PPO, and never become tie/loss labels. Policy samples
  come from ready current-policy seats only. Waiting/opponent seats have no policy loss.
  Complete-episode batches avoid cutting GAE across resets; time-limited batches bootstrap.
- Optional `faint_reward` adds a small change in public faint-count differential/6. It is
  an auxiliary training reward, not claimed to preserve the unshaped optimum. Starter and
  reference GPU recipes use sparse terminal reward only. Evaluation always uses win/draw score.
- `trace` hashes normalize only Showdown's presentation-only `|t:|<wallclock>` timestamps.
  Deterministic replay tests deliberately change wall-clock time; all other player-stream
  bytes and explicit action commands must match upstream. No game RNG is patched.

## Training, evaluation and allocation

PPO uses vectorized rollouts, categorical embeddings, action-conditioned masked logits,
GAE, clipped surrogate loss, entropy/value losses, gradient clipping, Adam, GPU-capable
inference and updates. Historical frozen networks and latest-policy self-play are supported;
`population-smoke.json` exercises both. Checkpoints contain an executable-source signature, model/Adam state, NumPy/Torch/
CUDA/Python RNGs, iteration/next seed, and historical opponents. Resume occurs between vector
batches with identical executable source and hardware configuration; it does not silently drop in-flight simulator state.

`starter.json` and `reference.json` are **runnable recipe templates, not trained strong
references**. Edit their GPU machine/device, fixed attached CPU/RAM, and allocation duration
before running. Placeholders/CUDA-unavailable configurations fail explicitly. The original 2026-11-08 performance extension used a US$12 / two-hour cap. The user explicitly
superseded those dollar, GPU-hour and attempt limits for the follow-on structural campaign.
Historical attempts remain immutable; the new launcher uses a separate attributed ledger and
records all allocation and costs. Each individual function still has a finite timeout and no retries.
CPU affinity can optionally be specified on Linux as `hardware.cpu_affinity`; Node workers
inherit it. Scheduler/container resource reservations must enforce the stated attached CPU/RAM
allocation. The trainer verifies Torch thread count + Node worker count fit that CPU count.

The budget is **elapsed allocated worker time**, including startup/source verification, CPU
simulation, IPC/transfer, inference, PPO updates, snapshots/checkpointing and development
evaluation—even when a GPU is idle. There is no utilization discount, sample-only replacement,
or separate CPU allowance. `accounting.json` measures from the Python entrypoint through worker
teardown and aggregates prior resume allocation. `allocated_worker_hours` is seconds/3600;
local CPU evidence is not a measured GPU-hour run. A real grader must charge the outer machine
allocation (including provisioning/interpreter overhead and any idle between commands), and
aggregate external development/profile/evaluation commands with the training allocation; these
commands each emit their full wall-clock debit. In-process development evaluation is already
inside the trainer clock. A single in-flight phase/update/checkpoint can exceed a deadline;
actual time/overshoot is reported, never clipped to the requested limit.

Practice seeds start at 900,000,000; held-out examples start at 1,900,000,000, disjoint from
training. Scripted opponents have per-battle/seat RNGs independent of vector grouping. The
published held-out schedule is reproducible, not secret: a submission grader must substitute
private seeds and frozen opponent checkpoints. The primary score is win + 0.5×draw across
**the entire** scheduled panel; any truncation/incomplete panel makes `score=null`. The
completed-only score remains a labeled diagnostic. The random and greedy public-power bots
are starter opponents, not expert Showdown policies. Add frozen trained-policy panels and
fresh-seed retraining before treating this as a qualified research benchmark.

## Evidence and remaining validation

Read `reports/pokemon/implementation-2026-11-08.md` for executed local results, full reproduction
commands and remaining qualification. The initial shared audit probes are unchanged. Actual CUDA receipts and their limits are in
`reports/pokemon/cuda-performance-2026-11-08.md`. Multi-seed convergence, commercial-rule parity
and a strong reference policy remain unqualified. Preserve upstream MIT notices/licenses; Pokémon names/data have their own provenance.


## CUDA performance extension

The staging runner uses one **NVIDIA L4**, four requested/hard-limited physical CPU cores and
8 GiB requested/hard-limited RAM. It has one container, no application retries, no minimum or
buffer containers, single-use shutdown and a 900-second input timeout. Only the distinct
`autoresearch-pokemon-cuda-20261108` app and its temporary artifact volume are managed. Existing
apps/jobs are not modified. A retained attempt ledger includes failed startup/delivery work.

```sh
# Install the SDK alongside Torch so client-side checkpoint metadata can be decoded.
uv pip install --python .venv-pokemon/bin/python 'modal[api-proxy-support]==1.5.2'

# Clean git clones, free of unrelated running server/audit sockets, are kept under
# evidence/pokemon-implementation/cuda-20261108/pinned-source/{pokemon-showdown,poke-env}.
# Their exact commits/source hashes are verified remotely before running.

# Original paired CUDA/CPU matrix and two-seed actual learning; unique labels required.
.venv-pokemon/bin/python -m modal run --env staging simulators/pokemon/modal_cuda.py \
  --mode matrix --label REPLACE_WITH_UNIQUE_LABEL

# Final matrix adds bulk wire encoding, static graph buckets and CPU-actor/CUDA-learner
# comparison, plus real public train/resume/evaluate/profile CLI subprocesses on CUDA.
.venv-pokemon/bin/python -m modal run --env staging simulators/pokemon/modal_cuda.py \
  --mode refine --label REPLACE_WITH_UNIQUE_LABEL
```

The runner refuses further work after seven total attempts or conservative prospective budget
reservations exceed this game's cap. The ledger must not be reset to spend more under the same
authorization. Cold image build, CUDA setup, failed attempts, serialization/persistence, CPU
simulation and evaluation are charged in receipts; no idle-GPU discount is applied. A temporary
Modal volume makes partial/final checkpoints recoverable after a client failure. Local archived
copies and hashes are verified before deleting only that new volume at the end of the campaign.

`optimization` selects:

| Mode | Actor / learner | Implementation |
|---|---|---|
| `baseline` | configured device / configured device | JSON observations; separate feature and result copies; normal PyTorch validation |
| `packed` | configured device / configured device | lossless float64 wire, native bulk copies, reused pinned float32 staging, one H2D / one combined D2H, CPU mask validation and batched learner diagnostics |
| `graph` | CUDA / CUDA | graph inference; deterministic ready counts use static power-of-two buckets; sampling retains exact ready shape |
| `hybrid` | CPU / CUDA when configured for CUDA | actor parameters copied once per rollout; CPU batched decisions; GPU PPO updates |
| `stream` | configured device / configured device | bounded binary frames, immutable contiguous observations, bulk staging |
| `shared` | configured device / configured device | worker mmap slabs, immutable snapshots, bulk actor and learner packing |

`packed` retains stochastic actor/learner semantics and has explicit trajectory/weight parity
checks. `graph` is a profiling option; stochastic graph training is not a qualified default.
`hybrid` changes the CPU/CUDA RNG stream and floating-point execution, so compare genuine
learning across seeds rather than claiming bitwise learning identity. GPU allocation remains
charged throughout CPU acting. Use the measured report to choose defaults for this small model.

`recipes/cuda-small.json` and `recipes/hybrid-small.json` describe the tested modest model and
resource allocation. They run on an already allocated CUDA host via the ordinary `train`,
`evaluate`, `profile` commands. They do not provision compute themselves. The CLI profiler also
includes a `batch_packed` variant. The full matrix records completed games, accepted decisions,
learner samples, updates, late states, faults/truncations, CUDA/host timing, CPU worker encode/
protocol/serialization/bytes, peak memory and utilization diagnostics. The `local_cpu_compare`
command holds the shared exclusive advisory flock for its whole timed comparison:

```sh
.venv-pokemon/bin/python -m integrations.pokemon.local_cpu_compare \
  --output evidence/pokemon-implementation/NEW_CPU_COMPARISON
```


## Structural performance follow-on (2026-11-08)

Read `reports/pokemon/structural-performance-2026-11-08.md` for the full ablations,
worker/batch scaling, simulator-only V8 profile, paired CUDA learning and rejected changes.
The historical CUDA readout above is now complete. New receipts live under
`evidence/pokemon-implementation/structural-20261108/` and
`evidence/pokemon-implementation/structural-cuda-20261108/`; old artifacts are untouched.

`optimization: "stream"` uses framed binary f64le observations, read-only contiguous NumPy
views and bulk inference staging. It preserves float64 public features for scripted choices
and the same float32 cast for the neural network. A 128 MiB reply limit and configurable
`worker_timeout_seconds` (default 60) bound worker reads; EOF/broken pipes are infrastructure
faults, and worker stdin closure tears down the child. Frame offsets/widths are checked.
`recipes/stream-small.json` supplies this opt-in candidate with the established model/PPO
settings. The report determines whether its small gains justify a default on your host.

Optional `mon_cache: true` memoizes per-Pokémon encoded rows using every visible input field
read by the encoder, in each player's own View. Optional `fast_callbacks: true` specializes
only upstream getCallback's non-SwitchIn member-lookup path; SwitchIn delegates to the pinned
method. Neither modifies vendored source/data or gives the encoder engine-object access.
Both remain off by default because native timing did not establish a reliable gain.

Optional `rollout_scheduler: "queue"` plus `episodes_per_update >= vector size` immediately
reuses completed slots. It keeps one fixed policy until the scheduled episodes finish, sorts
samples/results by seed schedule, and computes GAE within each episode. Live resets are
rejected. Natural endings, cutoffs and unstarted allocation-limited episodes stay distinct.
Checkpoint config/source signatures bind scheduler, episode count, optimization and hardware;
resume stays between completed collection batches. Queue scheduling changes stochastic
ready shapes/RNG consumption and episode count changes the PPO batch: qualify those recipes
separately. The tested queue did not reliably beat static collection and stays experimental.

All these local comparison entrypoints acquire the shared exclusive performance lock:

```sh
.venv-pokemon/bin/python -m integrations.pokemon.structural_profile \
  --repeats 5 --batches 2 --output evidence/pokemon-implementation/NEW_ABLATION
.venv-pokemon/bin/python -m integrations.pokemon.structural_profile --scaling \
  --repeats 3 --batches 2 --output evidence/pokemon-implementation/NEW_SCALING
.venv-pokemon/bin/python -m integrations.pokemon.native_profile \
  --output evidence/pokemon-implementation/NEW_NATIVE
.venv-pokemon/bin/python -m integrations.pokemon.queue_profile \
  --repeats 3 --output evidence/pokemon-implementation/NEW_QUEUE
node simulators/pokemon/tests/test-fast.cjs
.venv-pokemon/bin/python -m unittest integrations.pokemon.tests.test_rl \
  integrations.pokemon.tests.test_efficiency -v
```

The ordinary CLI `profile` now includes `batch_stream` and reports canonical trace parity.
When timing that generic command locally, the caller must hold `evidence/local-performance.lock`.
The entrypoints above already do so; do not nest a second flock around them.

Fresh authorized remote experiments use the new staging app, one L4, four fixed physical
CPU cores and 8 GiB RAM, zero retries, no minimum/buffer containers and single-use shutdown:

```sh
.venv-pokemon/bin/python -m modal run --env staging simulators/pokemon/modal_structural.py \
  --mode transport --label UNIQUE_TRANSPORT_LABEL
.venv-pokemon/bin/python -m modal run --env staging simulators/pokemon/modal_structural.py \
  --mode final --label UNIQUE_FINAL_LABEL
.venv-pokemon/bin/python -m modal run --env staging simulators/pokemon/modal_structural.py \
  --mode graph --label UNIQUE_GRAPH_LABEL
```

Each invocation has a 3,600 s safety timeout; this is a lifetime bound, not a renewed campaign
allowance or a spend target. All setup, simulation, transfer, compilation/capture, evaluation,
checkpoint/archive/persistence and idle time remains part of allocation accounting. Unique
labels and retained checkpoints/receipts permit recovery after client delivery faults.

## Larger full-task vectors and native development

`recipes/shared-small.json` enables shared transport, lazy observation views and batched
scripted decisions on the established 24-game / L4 / four-CPU allocation. These options retain
the same observations and scripted RNG/actions; correctness tests compare full traces and PPO
updates. The shared slab is copied to immutable bytes before a later step can overwrite it.
`evaluation_vector_for` sizes a practice vector to its schedule, so a 32-game practice panel
does not start a 992-game training topology.

`recipes/massive.json` is an **experimental different-resource recipe**: one L4, 32 physical
CPU cores, 32 GiB RAM, 992 games and PPO minibatches of 4096. It demonstrated real learning
and exact ordinary-CLI CUDA model/RNG resume. Its approximately 6× paired samples per case
allocation-hour uses 8× the attached CPUs, 4× RAM and changed PPO/episode batch sizes. The
matched four-CPU comparison of 24 versus 192 games gave approximately 1.39×. Neither is a
10× same-resource improvement or a competitive-strength qualification. Read the structural
report for natural endings, allocation cutoff batches, practice scores and costs.

The new `native/` directory contains a real shared CPU/CUDA simulation core and persistent
native CPU workers. Native transitions never invoke Showdown. It is **unfinished for the full
frozen Gen 9 random-battle task** and remains excluded from production/default training.
`pkn_gen9_complete()` returns zero and `Native.require_full_task()` refuses admission.
Constructed component parity and CUDA PPO checks are development evidence only. Unsupported
effects must remain explicit faults; never substitute the oracle or simplified rules to make
a full-task check pass. See [native/README.md](native/README.md).

Development correctness checks (new output directories required):

```sh
.venv-pokemon/bin/python -m integrations.pokemon.native.validate \
  --output evidence/pokemon-implementation/NEW_NATIVE_RNG_DUELS
.venv-pokemon/bin/python -m integrations.pokemon.native.mechanics_check \
  --output evidence/pokemon-implementation/NEW_NATIVE_MECHANICS
.venv-pokemon/bin/python -m integrations.pokemon.native.effects_check \
  --output evidence/pokemon-implementation/NEW_NATIVE_EFFECTS
```

These are correctness checks, not simulator performance comparisons. Every native build
records source/binary hashes and an immutable source snapshot. Native checkpoints also bind
rules and executable signatures. Existing historical checkpoints retain their recorded source;
do not bypass a source mismatch after changing integration code.
