# Clash fixed-eight-card research task

The default module entry now trains with an independent C++/CUDA rewrite of the frozen
eight-card simulator. Java remains a differential oracle and historical baseline;
it executes no game transitions in the native training path. Native commands and
qualification details are in [native/README.md](native/README.md).

The frozen-task environment handoff is complete. Use `configs/native-l4.json`
for C++ CPU simulation with CUDA training, or `configs/native-h100.json` for
the measured CUDA simulation/training recipe. Experimental encoders and CPU
neural prototypes are not installed by either default. See the final
[rewrite readout](../../reports/clash/native-rewrite-2026-11-08.md).

This does not control the commercial game. The private upstream recorded-match corpus is absent;
commercial-game fidelity is unverified. Passing masks, fixed-deck matches, or registry checks
does not establish coverage of other cards, evolutions, champions, tower troops, or levels.

## Autonomous research benchmark

**Current: `clash-autoresearch-short-v1`** ([contract](autoresearch/CONTRACT-short-v1.md),
[readout](../../reports/clash-short-form-2026-11-13.md)). Matches are 60 s of regulation plus
30 s of overtime (elixir x2 from 40 s, x3 from 75 s) at ten ticks, i.e. 2 decisions, per
simulated second, so a match is about 130–200 decisions instead of 700–1,250. Deck, levels,
arena, observations, masks, natural endings and scoring are unchanged. The match form is
compiled into its own native library ([native/match_forms.py](native/match_forms.py));
`full` remains the pinned form and the default of every native entry point. The panel is the
four scripted bots; fast-v4's learned opponents were trained on 300 s matches and are dropped.
`python -m harness prepare clash --out NEW_RELEASE` needs no base release. Everything below about
fast-v4 and earlier describes the full-length form (git tag `full-size-20261113`).

The final checkpoint cleanup is ready: one atomic serialization, complete state
and source bindings, 73 local tests, exact CUDA/full-profile resume and focused
public broker verification. Matched H100/4CPU/16GiB runs measured 36,527 accepted
new decisions/full-worker-second. Read the [cleanup follow-up](../../reports/clash/fast-starter-requalification-2026-11-09.md) for its separate closed 600-second account.

New work selects the [fast v4 starter](autoresearch/benchmark-v4.json) through
`python -m harness prepare clash` (see [the harness guide](../../harness/README.md)). Its editable source
reuses the qualified native collector and learner, starts with random weights,
and defaults to H100/4 CPUs/16 GiB with 1024 environments. The historical
2048-environment/32-CPU/128-GiB throughput recipe remains separate. Read the
[November 9 requalification](../../reports/clash/fast-starter-requalification-2026-11-09.md)
and current commands in [autoresearch/README.md](autoresearch/README.md).


Research sessions run through [the single harness](../../harness/README.md) with
Terminus-2 as the researcher. [autoresearch/](autoresearch/README.md) holds the
minimal editable PPO starter, the [contract layers](autoresearch/CONTRACT.md) and
the trainer and evaluator entrypoints. The full allowance is 7,200 allocated H100
worker seconds; the smoke allowance is 600. A session without `--execute` only
prints its plan. Use `.venv-clash-ar` for the harness and `.venv-clash` for local
simulator checks. The fast-v4 contract has protected-public practice only: its
private final remains closed, so there is no host-private grade for Clash yet.
Reference-normalized scores remain null until full-budget multi-seed calibration.

## Frozen task

The November 9 [evaluator readiness readout](../../reports/clash/evaluation-readiness-2026-11-09.md)
and [v2 contract supplement](autoresearch/CONTRACT-evaluation-v2.md) document the
visible-state teacher/intermediate panel, isolated neural admission, paired
qualification and lifecycle/export regressions. The panel is development-qualified;
strong-reference calibration and superhuman evidence remain unavailable. Reference
training has separate ownership under `reference_training/`.

The active five-hour campaign's [evaluator readout](../../reports/clash/strong-bots-five-hour-2026-11-09-evaluator.md)
tracks the explicit [v3.1 successor](autoresearch/benchmark-v3.1.json), isolated
parallel-pair grading, public cross-play and request/execution/deployment
diagnostics. Its public trajectory-parity check passed; capacity/workflow/final
assessment evidence is reported separately. Private final seeds stay closed
until checkpoint selection freezes. No expert or superhuman qualification is
implied by the new version.

`clash-fixed8-v1` uses Knight, Archer, Musketeer, Giant, MiniPekka, Fireball, Arrows and Zap,
all at level 11, level-11 Princess towers, and no evolution/hero slots. The engine commit is
`f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de`. Tables are data version `16.635.8`, content SHA
`062a11e0500c03cc041f123413b1ef1846b539a4`. `rules.lock.json` records every table file,
including tilemaps, engine source, and installed Java jar SHA-256. Runtime checks refuse drift
in source, installed Python wrappers, jars, rules, or tables. Do not overwrite the table cache.

One environment decision asks both players for an action and advances up to **five 50-ms
ticks**; a natural terminal can stop early. The schema is version 4, 7,024 int32 entries per
seat, 2,307 actions, mask rule 2 (landing tiles), entity-ID rule 2. The upstream environment
exposes at most 200 entities; overflow is logged. Model trimming never imposes a lower cap.

Only each seat's own structured observation goes into its policy. No omniscient lookahead,
private enemy hand, or engine RNG state is fed to the neural policy. The deck is fixed and
public by task definition. The legacy Java comparator retains upstream `VectorBattle`
direct buffers and its thread pool. The native path implements the rules independently
and preserves the same two-seat observations, masks and outcomes.
crforge's Apache-2.0 LICENSE, NOTICE and attribution request remain in the source archive.

## Environment and quick commands

Run from `/Users/K.Z./Desktop/projects/game-autoresearch-bench`. The default native entry is:

```sh
.venv-clash/bin/python -m integrations.clash \
  --config simulators/clash/configs/native-cpu-smoke.json \
  --out evidence/clash-implementation/my-native-smoke
```

The commands below are the preserved Java reference integration:

```sh
export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
.venv-clash/bin/python simulators/clash/tests/test_contract.py
.venv-clash/bin/python simulators/clash/runner.py train \
  --config simulators/clash/configs/smoke.json \
  --out evidence/clash-implementation/my-smoke
.venv-clash/bin/python simulators/clash/runner.py evaluate \
  --config simulators/clash/configs/smoke.json \
  --checkpoint evidence/clash-implementation/my-smoke/checkpoint.pt \
  --schedule simulators/clash/configs/eval-practice.json \
  --out evidence/clash-implementation/my-practice
.venv-clash/bin/python simulators/clash/runner.py evaluate \
  --config simulators/clash/configs/smoke.json \
  --checkpoint evidence/clash-implementation/my-smoke/checkpoint.pt \
  --schedule simulators/clash/configs/eval-heldout.json \
  --out evidence/clash-implementation/my-heldout
.venv-clash/bin/python simulators/clash/runner.py profile \
  --config simulators/clash/configs/profile-trimmed.json \
  --out evidence/clash-implementation/my-profile
```

Output directories must be new. `manifest.json` and `source.tar.gz` are written once and
made read-only; permissions are an accident guard, not a security boundary. Each manifest
contains exact recipe, source/data/rules hashes, engine metadata, device, runtime, seed contract
and hardware declaration. The source archive includes the integration and tracked crforge
source at that run, including uncommitted changes. A pinned JVM distribution is preserved at
`evidence/clash-implementation/pinned-env-dist.tar.gz`, hashed by each manifest. Tables are
separately acquired game data; the cache is retained locally rather than redistributed.

The existing `.venv-clash` now includes PyTorch 2.9.1. For a separate local environment, install
`requirements-local.txt` using uv, then the pinned checkout's `python/` package. crforge's
`docs/game-tables.md` documents table acquisition. Use Java 21 and the pinned distribution;
`./gradlew :env:installDist` rebuilds it, but a differently produced binary must be qualified
and pinned as a new artifact rather than silently rewriting this task's hashes. Do not fetch
latest tables. The Python package and Java distribution must come from the same engine pin.

## Learner and recipes

`model.py` embeds categorical card/unit IDs, buffs, entity kind/side/state, and binary flags.
It scales numeric amounts, pools only valid entities, and retains ordered hand/deck/history
card embeddings. A context-conditioned hand query scores each card's 18×32 placement grid.
The policy normalizes tiles within each hand slot before forming one legal flat categorical
distribution, so hundreds of placements do not overwhelm the initial wait probability. Ability
logits exist in the schema but the qualified plain deck has no champions. All policy actions
use upstream masks. Values are separate from mask normalization.

`runner.py` implements batched masked PPO with GAE, clipped policy loss, value regression,
entropy regularization, gradient clipping and KL early stopping. Optional tower-health
potential shaping is `coefficient * (gamma * phi(next) - phi(now))`, with terminal potential
zero. Natural wins/losses use upstream +/-1; draws use zero. A time limit bootstraps the final
observation and stops the GAE chain. Rollout boundaries bootstrap without ending the match.
Faults or unsupported actions fail training closed and never become policy wins/losses. The
helper GAE also excludes invalid transitions. An evaluation containing any fault, unsupported
episode, truncation or illegal action has a null official score; valid-only diagnostics remain
visible. Fault signatures and details are retained independently from game outcomes.

Configurations:

- `smoke.json`: short CPU teacher initialization followed by 100 PPO updates versus ScriptedBot.
- `starter.json`: pure PPO, sparse terminal reward, scripted opponent; no shipped trained weights.
- `reference.json`: runnable teacher-initialized PPO with tower shaping and historical opponent
  population. It is a **recipe candidate**, not an actually trained strong reference.
- `gpu-reference-template.json`: GPU-capable settings. Supply an exact visible GPU model,
  CPU/RAM allocation and elapsed budget. The template deliberately refuses an unspecified
  GPU model. It does not allocate, buy or launch compute.
- `profile-dense.json` / `profile-trimmed.json`: identical self-play workloads with only entity
  trimming changed. They include rollouts, copies/transfers, masks, inference, simulation,
  resets, PPO updates and checkpointing.

The optional teacher initialization collects legal ScriptedBot actions on real engine states,
trains on nontrivial decisions, and logs dataset hash, examples and before/after NLL in
`warmstart.json`. It is training-data fit, not held-out strength. PPO follows on the same worker
budget. `initial.pt` is the untrained initial checkpoint. Total counters include teacher work;
`ppo_counts` subtracts teacher collection. Metrics also record actual optimizer steps and
nonzero parameter change. `teacher_player_decisions` and `learner_decisions` are distinct.

Opponent modes are `scripted`, `random`, `wait`, `population`, `frozen`, and `selfplay`.
Population samples scripted or historical snapshots once per episode and preserves an assigned
snapshot until that match ends. A bounded recent population plus in-use snapshots is retained.
Both learner seats alternate across completed episodes. `selfplay` learns from both seats,
using one batched inference call. Frozen-network construction preserves sampling RNG state.

## Resume and elapsed allocation accounting

```sh
.venv-clash/bin/python simulators/clash/runner.py train \
  --config simulators/clash/configs/smoke.json \
  --resume evidence/clash-implementation/my-smoke/checkpoint.pt \
  --updates 200 --budget-seconds 240 \
  --out evidence/clash-implementation/my-resumed
```

Changing update/time ceilings is allowed; changing the rest of the recipe, source snapshot,
rules or vocabulary is refused. Checkpoints include Adam, model, torch/NumPy RNGs, historical
opponents, per-lane learner seat, engine seeds and **both players' complete current-episode
action histories**. Resume replays independent histories on the actual engine, checks a digest
of observations/masks/info, then continues. Reconstruction time is charged. This avoids a fake
resume that silently starts fresh games. `tests/resume_check.py` compares separate processes
for bit-identical final model/optimizer/RNG/engine states.

The clock begins before framework import/setup and JVM initialization. Charge all elapsed
allocated-worker time, including CPU simulation, idle GPU time, transfers, updates, source
snapshots, reconstruction, checkpointing and development evaluation. This is **not GPU
utilization or sample-only accounting**. CPU smoke runs report `allocated_gpu_hours: null`.
A configured one-GPU run records elapsed worker seconds divided by 3,600 as allocated GPU-hours,
even when the GPU is idle. GPU ledgers also charge elapsed gaps between development commands
while that allocation remains live; an external scheduler owns allocation release. The runtime checks exact visible GPU identity and one visible device.
CPU count and JVM heap are configurable; a real scheduler must enforce the fixed attached
CPU/RAM allocation. Python threads/JVM active CPU settings do not establish OS isolation.

Every run writes `allocation-ledger.json`. Evaluation/resume of a checkpoint defaults to the
checkpoint directory's ledger; multiple sequential development commands can share an explicit
`--charge-to /absolute/path/allocation-ledger.json`. Total time must fit the same candidate
budget, with no reset for development evaluations. A file lock refuses concurrent use of an
existing ledger. Checkpoints carry the original ledger path, including across resume chains; their saved elapsed
time is also a lower bound if the ledger is moved. The manifest records the ledger path. Independent candidates get separate
ledgers. No extra CPU budget exists. The budget is checked at decision/minibatch boundaries;
initialization, final checkpoint and an in-flight operation can overrun. Actual overrun is
reported, and an external allocation controller is required for a strict wall-time cutoff.

Public practice seeds are `1000000..1000007`; held-out-demo seeds are `2000000..2000007`.
Training resets are restricted to `0..999999`. Evaluations pair every engine seed and opponent
with both learner seats. Each job/seat has an independent policy RNG, so scheduling completion
does not change its action stream. Stochastic neural evaluation matches the trained sampling
policy. Optional greedy inference decodes slot marginal mass, then tile. The evaluator accepts
frozen opponents as `checkpoint:/absolute/path/checkpoint.pt` in the opponent list, records
their hashes and checks rules/vocabulary. Random/scripted/wait panels are public diagnostics,
not strong frozen policies. The published held-out file is a reproducible seed-separation
demonstration, **not a secret grader**. Final grading needs private seeds and independently
trained frozen opponents; no secret split or strong checkpoint is claimed here.

For evaluation, `environment_decisions` counts all vector lanes actually advanced. Finished
lanes are parked/reset while others finish; padding decisions and ticks are charged and logged,
but padding actions do not enter policy scores. `player_decisions` counts active evaluated
players and opponents, `padding_wait_decisions` counts idle lane waits, and `ticks` counts
actual advanced ticks, including padding. In training, every lane is active and
`player_decisions = 2 * environment_decisions`; only controlled learner decisions enter PPO.

## Validation and limits

Read `reports/clash/implementation.md` and machine-readable results in
`evidence/clash-implementation`. The contract tests cover real-table cross-thread determinism,
direct-buffer stability, masks, variable-length episode reconstruction, dense/trimmed output
agreement, natural-terminal/time-limit GAE, and injected fault separation. A real-state teacher
learning diagnostic checks that the optimizer can reduce NLL. The first CPU-only pass is recorded in that report. Actual bounded CUDA follow-up, costs and
current performance/learning limits are recorded in `reports/clash/cuda-efficiency-2026-11-08.md`.
No changes were pushed. Strong-policy qualification and recorded-commercial-game fidelity remain
outstanding.

Repeat the bounded integration validation and controlled optimization study with new evidence
roots (the profile helper can also continue an interrupted study without deleting completed runs):

```sh
.venv-clash/bin/python simulators/clash/tests/validate_local.py \
  evidence/clash-implementation/my-validation
.venv-clash/bin/python simulators/clash/tests/profile_compare.py \
  evidence/clash-implementation/my-profile-study
```

Bit-identical resume evidence applies to checkpoints after completed PPO updates. A deadline
can stop partway through rollout collection, teacher initialization or an update; the final
checkpoint retains the resulting model, optimizer, RNG and live engine state. Resume starts
its next rollout from that saved state. It does not finish discarded in-flight rollout data or
remaining minibatches of the interrupted iteration. A strict allocation controller and
complete-iteration checkpoint boundaries are needed for an uninterrupted-equivalence audit.

## Historical authorized CUDA follow-up

This section records the earlier bounded campaign. Later structural work and
the completed native environment handoff are documented in the rewrite readout.

The 2026-11-08 follow-up uses the existing Modal staging profile with an exclusive Clash cap of
US$12 and two allocated GPU-hours. `modal_app.py` enforces a bounded serial stage matrix, one L4,
four physical CPU cores, 16 GiB RAM, zero retries, one container maximum, no warm pool, and a
900-second function limit. Reservations include cold load, teardown and build/egress headroom.
`billing_receipts.py` fetches provider billing rows for this app only. Source/data versions and
checkpoints are included in each receipt. See the CUDA readout for executed commands and results.

`configs/cuda-eager.json` uses trimmed entities, pinned staging/device buffers, combined policy
readback and full-rollout GPU caching. `configs/cuda-compiled.json` is an experimental comparator,
not an assumed faster default. The initial bucketed compiled runs were slower than eager reuse.
`configs/cuda-learning.json` adds a longer scripted-opponent PPO pilot, a longer GAE trace and
policy losses normalized over non-forced decisions. Value regression still uses all valid states.
A post-teacher checkpoint is saved separately so imitation gains cannot be attributed to PPO.


### Local replay viewer

The shared Replay Room now includes three complete public-demo Clash games at
<http://127.0.0.1:5180/?game=clash>. From the repository root, run
`npm run dev --prefix replay-site -- --port 5180 --strictPort` if the shared
local server is not already running. Reuse an existing server on that port.
The viewer provides original schematic arena art, troop/projectile movement,
tower health, elixir, hands, crowns, command-log seeking, speed controls and
simulator-public Blue/Red views. Downloadable completed movies include both
hands and spectator state. These are scripted/random demos, not policy scores.

```sh
.venv-clash/bin/python simulators/clash/export_demo_replays.py \
  --output evidence/clash-replays/new-public-demo
```

The CPU-only exporter uses fresh public seeds, no JVM or neural checkpoint,
checks every selected action against the mask and rejects faults/truncations.
It independently replays every complete public output and visual frame before
writing a movie. Existing evidence directories are never overwritten. Copy
only the movies and merge their manifest entries into the shared local site;
keep verification receipts and private benchmark artifacts outside the site.
See `replay-site/README.md` for playback details and provenance.
