# Generals full-sized 1v1 RL environment

The 8×8 task `generals-research-1v1-8x8-v2` stood in for this one on 2026-11-13 and 14 (last at
commit `48f226ee`). Its contract [benchmark-8x8-v2.json](benchmark-8x8-v2.json) and recipe
`recipes/8x8-smoke.json` are kept as its record; they do not bind to the active ruleset.

Latest reference readout: the [detached AverageJoe continuation](../../reports/generals/averagejoe-continuation-2026-11-09.md)
completed 220 additional PPO updates in 53m 07s of H100 allocation. Stage-0
random wins improved from 4.69% to 53.13%; the policy remains below the promotion
gate and is not a full-distance or expert benchmark anchor. Final artifacts and
four viewing games were independently verified; all owned workers have stopped.

The separate [AverageJoe reference trainer](averagejoe/README.md) now runs the
actual upstream 15.35M-parameter model and curriculum. Its November 9
[bounded Modal smoke](../../reports/generals/averagejoe-smoke-2026-11-09.md)
passed engineering checks but remains at stage 0 with unqualified strength.
It does not change the frozen entrant task described below.

The current standard is **`generals-research-1v1-full-v2`**: independently sampled
17–23 rows and 17–23 columns, fog of war, and a **2,048-joint-step episode cap**.
All maps are padded with impassable mountains to 24×24 for batched execution.
This replaces the earlier 8×8 research task. Its historical recipes, manifests,
source archives and performance receipts remain unchanged. They do not qualify
this larger task. Current contract: [benchmark-full-v2.json](benchmark-full-v2.json).

The pinned upstream MIT engine remains unmodified at
`954927f508aa47899019d66317cc313af3e9cc19`. The full-size public API supports
training, paired evaluation, profiling, policy-independent replay and exact-state
checkpoints. Performance measurements use the same small width 16 CNN as the old
campaign, not the released 15M-parameter AverageJoe transformer. Neither model
strength nor reproduction of that agent is claimed by throughput qualification.

## Frozen full rules and information

* Two players, ordinary neutral castles, simultaneous player decisions resolved
  in the pinned engine's order, general trade enabled, build-castles/deathtouch
  disabled. Capturing a general gives ±1; reaching 2048 without capture gives a cap
  draw and bootstraps the actual final visible value during training. Natural
  endings, truncations, illegal actions and simulator faults remain distinct.
* Full-distance maps use minimum walking/BFS spawn separation 17, terrain-density
  input .21, castle count 9–11 inclusive and castle army range [40,51), preserving
  the pinned generator's distinct count/army bound semantics. There is no implicit small-map
  or spawn-distance curriculum. Source parameters follow the released full-size
  [training configuration](https://github.com/strakam/AverageJoe/blob/main/configs/custom/L_7d_gae90.yaml),
  with unrestricted maximum spawn distance for the full task.
* Map dimensions come from an independent folded seed key, then the original
  seed key generates that exact rectangle through the pinned upstream generator.
  `build_pool()` groups all 49 shapes into fixed eight-key setup batches and
  restores input seed order. Upstream `init_state()` by itself always uses the
  maximum size and must not be used to represent this mixed-size task.
* [seeds.json](seeds.json) remains the unchanged public 256-training/16-practice/
  32-heldout partition. Geometry changes because the rules version changes;
  new map manifests/receipts go to new output directories. The generator also
  accepts arbitrary seed lists. The default finite training pool is not the
  released agent's 200k-map refreshed pool.
* Actor and critic receive the same upstream player-visible observations:
  nine spatial planes plus five broadcast public scalars, 24×24×14. Public
  opponent army/land totals and structure silhouettes are intentional. Hidden
  map state and privileged features never enter the network. Persistent policy
  memory is not added in this environment change.
* Each player has **4,609 action IDs**: one pass plus 24×24×4 directions×2 splits.
  Padded sources and destinations are illegal. Land features/potential normalize
  by the padded tensor area; time normalizes by 2048. The packed representation
  reconstructs exactly the same visible float planes and mask.

## Setup and ordinary commands

Use the game-local `.venv-generals`. After a fresh clone, restore offline upstream
metadata before setup/pin checks, as described by the project handoff. Source is
vendored; do not reset the upstream checkout.

```sh
simulators/generals/setup.sh
.venv-generals/bin/python -m pytest simulators/generals/tests -q
.venv-generals/bin/python -m integrations.generals train \
  --recipe simulators/generals/recipes/full-smoke.json \
  --output evidence/generals-implementation/new-full-smoke
.venv-generals/bin/python -m integrations.generals evaluate \
  --recipe simulators/generals/recipes/full-smoke.json \
  --checkpoint evidence/generals-implementation/new-full-smoke/latest.msgpack \
  --split practice --opponents random,expansion \
  --output evidence/generals-implementation/new-full-evaluation
.venv-generals/bin/python -m integrations.generals.replay \
  evidence/generals-implementation/new-full-evaluation/replay.npz \
  --output evidence/generals-implementation/new-full-evaluation/replay-check.json
```

The CLI default is `full-smoke.json`; `full-cpu-fast.json` retains the
CPU slice/compact execution path. GPU recipes are `full-l4.json` (128 games) and
`full-h100.json` (2,048 games). Each declares one GPU, four physical CPU cores and 16 GiB RAM.
A scheduler must enforce those limits. On an already allocated matching worker:

```sh
GENERALS_ALLOCATION_CONFIRMED=1 .venv-generals/bin/python -m integrations.generals.lease \
  --recipe simulators/generals/recipes/full-h100.json \
  --ledger evidence/generals-implementation/new-full-allocation.json -- \
  .venv-generals/bin/python -m integrations.generals train --device gpu \
  --recipe simulators/generals/recipes/full-h100.json \
  --output evidence/generals-implementation/new-full-training
```

The lease provisions no hardware. Full allocation time includes imports, map
setup, compile, simulation, transfers, inference, PPO, evaluation, checkpoint IO
and GPU idle time. Use new outputs or explicit `--resume`; never overwrite old
logs. Resume requires the original recipe, source/rules binding and checksum.
The 8×8 checkpoints deliberately fail this new rules binding; use their archived
source to inspect them. No implicit checkpoint migration is performed.

## Preserved speed architecture

GPU states, reset pools, raw visible rollout data, legal masks, historical
policies and PPO remain resident on the device. Packed binary planes/masks are
expanded only for the learner minibatch. Sparse map/history refresh has an exact
dense fallback for cohorts larger than capacity. Final cap states and pre-reset
learner seats remain available for correct GAE bootstrapping. Indexed checkpoints
verify bitwise reconstruction, fall back on fingerprint collisions and preserve
complete game/RNG/optimizer/history state.

The historical CNN's Pallas/Triton backend is now tiled over 64 pixels per block,
so it handles 24×24 maps and partial tiles. It retains BF16 output/bias rounding
and float32 named parameters. PPO remains Flax/cuDNN. The direct/slice backend
is retained as a reference. Gumbel sampling remains default; inverse-CDF is an
explicit experimental option. The CPU-producer/GPU-learner control remains
available through `simulation_device: cpu`, with packed compact inference.

## Full-size qualification and benchmarking

New outputs are under `evidence/generals-implementation/fullsize-20261108/`.
The new campaign measures this task directly; the old 8×8 rates are historical.
Selected warm PPO throughput is **334k learner samples/s on H100** and **76k/s
on L4**. Three-seed provider-allocation throughput is **184k/s /28k/s**,
including setup, JIT, evaluation, saving, replay and startup/shutdown. These
qualify execution speed; a strong playing policy remains unqualified.
Each Modal worker has one L4 or H100 and the same four-core/16-GiB attachment,
zero warm containers and zero configured application-error retries. The H100 launcher requests
`H100!` to prevent an automatic H200 upgrade. Receipts record the actual GPU.
Use unique tags:

```sh
modal run --env staging simulators/generals/full_modal.py \
  --tag new-full-validation --mode validate --gpu H100
modal run --env staging simulators/generals/full_modal.py \
  --tag new-full-profile --mode profile --gpu H100 --batches 128,512,2048,8192
modal run --env staging simulators/generals/full_modal.py \
  --tag new-full-learning --mode learn --gpu H100 --batches 2048 \
  --seeds 42,43,44 --seconds 600 --variant optimized --minibatch-size 4096
```

Validation covers every 17–23 rectangle, exact upstream/CPU/CUDA maps, all state
fields and observations on full 2048-step action streams, natural outcomes/caps,
packed inputs, tiled inference, baseline/optimized rollout, source-bound restore
and fresh CLI training/resume/evaluation/profile/replay. Profiles precondition
with 68 actual PPO updates, crossing a natural 2048 boundary before measurement.
Full-worker learning and provider allocation are reported separately from warm
rates. The user explicitly superseded historical spending/attempt caps; those
old ledgers stay unchanged and failed new attempts stay recorded.

Current readout: [full-size qualification](../../reports/generals/fullsize-performance-2026-11-08.md).

The [autoresearch harness](autoresearch/README.md) adds the minimal
editable starter, agent workspace, budget broker, bots, immutable submissions and
private grading in a sibling package, preserving historical checkpoint bindings.
Read [its preparation report](../../reports/generals/autoresearch-preparation-2026-11-08.md)
for live smoke evidence and reference qualification limits.

The new [frozen benchmark and bounded DeepSeek smoke](../../reports/generals/benchmark-contract-smoke-2026-11-08.md)
adds the discoverable machine/human contract and a separate cumulative
600-second GPU campaign ledger covering training, practice and hidden grading.
Its corrected autonomous session passed within 312.681146 allocated H100 seconds;
raw private panel points were 37.5/100, with reference normalization unqualified.

The active [overnight campaign readout](../../reports/generals/overnight-2026-11-08.md)
records the three-seed recipe comparison, independent researcher trials, qualified
H100 development endpoint, invalid CPU assessments and cumulative accounting.

Historical readouts:
[8×8 structural campaign](../../reports/generals/structural-performance-2026-11-08.md),
[earlier CUDA campaign](../../reports/generals/gpu-efficiency-2026-11-08.md).
