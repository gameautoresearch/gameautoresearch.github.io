# Dou Dizhu autonomous RL integration

The [official pretrained local boot](../../reports/doudizhu/pretrained-local-boot-2026-11-09.md)
adds a separate strict loader for the official demo archive and three CPU
self-play replays on the local Replay Room. It requires no retraining, and does
not alter the frozen evaluation pools or historical source/checkpoint manifests.

This research environment has complete native CPU and GPU-resident simulators for the
pinned fixed-role/no-bidding DouZero task, with three-role Deep Monte Carlo training,
evaluation, replay, source-bound checkpoints and full allocated-worker time metering.
The unchanged [DouZero source](https://github.com/kwai/DouZero/tree/718a5c920bf3361e34178a38f3b80458e176b351)
is the independent rules oracle. No bidding or other game variant is added by the port.

The [autoresearch harness](autoresearch/README.md) adds the Polytopia-style agent workspace,
editable starter, run/practice/submit broker, fixed H100/120-minute allocation, immutable
submissions and private three-role grader. Preparation is offline; official numeric scores
remain blocked until the separately budgeted reference calibration is executed.
Setup validation and retained smoke failures are recorded in the
[autoresearch readout](../../reports/doudizhu/autoresearch-setup-2026-11-08.md).
The [30-minute autonomous trial](../../reports/doudizhu/autoresearch-research30-2026-11-08.md)
records DeepSeek's actual research, practice gains and inconclusive withheld-panel gain.
The new [versioned-contract ten-minute campaign](../../reports/doudizhu/benchmark-contract-smoke-2026-11-08.md)
passed real DeepSeek learning, practice, immutable submission and independent hidden
CPU grading. It used 315.953 allocated H100 seconds under the separate 600-second
ceiling; raw score was 67.1875/100 on 192 natural games. Reference normalization
remains null. See [`benchmark.json`](autoresearch/benchmark.json) and
[`CONTRACT.md`](autoresearch/CONTRACT.md) for the current discoverable task contract.

Use `native-local.json` locally, `native-l4-fp32.json` for the strong CPU-simulation/GPU-learning
control, or `resident-l4-fp32.json` for device-resident simulation and training. These
recipes retain the full 4,486,659-parameter network. The resident backend keeps private
state, complete ragged legal actions, public features, selected actions, trajectory rings
and learner samples on GPU. It uploads pinned deals once per wave and exports scalar
turn controls, completed outcomes and a bounded replay audit. It requires CUDA/NVCC;
the native backend requires a C++17 compiler. First compilation is measured and cached
with source/binary/compiler receipts. No simulator transition calls the Python oracle
in the native or CUDA hot loop; bounded replay checks run after collection.

Measured results, numerical recipe distinctions and remaining limits are in
[`structural-performance-2026-11-08.md`](../../reports/doudizhu/structural-performance-2026-11-08.md)
and [`structural-summary.json`](../../reports/doudizhu/structural-summary.json).
Explicit `engine_backend="upstream", encoding_backend="upstream"` selects the original
simulator/codec. `view` retains the intermediate Python optimization. The low-level
CUDA interface is `CudaBatch`; `collect()` dispatches the common worker to device buffers.
Object-oriented `VectorGames.reset/step` is for CPU engines.

`resident-l4-tf32.json` is an explicitly different neural arithmetic recipe.
`resident-l4-bf16.json` uses the qualified BF16 path and 4,096-game batches. BF16 compute
also keeps FP32 master weights and the scalar Q head; its qualification is reported
separately. Neither precision change is an exact substitute for the matched FP32 path.
Legacy NumPy actor/shuffle streams remain available. Resident recipes explicitly select
`splitmix-counter-v1` actor RNG and GPU Torch shuffle; resume across those sampling
protocols is refused. Deal generation and frozen rules/data pins are unchanged.

Older CUDA graphs, forced-action bypass and packed Python workers remain opt-in.
The graph matrix did not preserve all trained-policy hashes; it is not the recommended
exact path. Recommended native/resident recipes use eager factored inference.

## Setup and source pins

From `/Users/K.Z./Desktop/projects/game-autoresearch-bench`:

```sh
sh simulators/doudizhu/setup.sh
.venv-doudizhu/bin/python -m pytest simulators/doudizhu/tests -q \
  --basetemp evidence/doudizhu-implementation/test-reproduction-tmp
```

Requires `git`, `uv`, and Python 3.12.9 (uv can provision Python). Setup refuses to
modify a different existing source revision. The lock pins and hashes dependencies
across platforms; the local installed-package snapshot is also preserved. The Linux
CUDA dependency set is resolved in the universal lock and was installed and executed
on the L4 container (PyTorch 2.10.0+cu128, CUDA 12.8, driver 610.57.04).
Do not install the upstream `requirements.txt`: its RLCard/GitPython dependencies are
unnecessary for this integration and its package initializers eagerly import the legacy
logger. The pinned model equivalence test imports the upstream model file directly.

Every process verifies source commit and source file hashes before importing the
engine. `source.lock.json`, `rules.lock.json`, and `data-seeds.lock.json` freeze the
source, rules, observation/action codec, and deal algorithm. Per-run `manifest.json`
adds actual package/hardware versions, recipe, and integration source hashes.

## Frozen game and observation boundary

Three players: `landlord`, then `landlord_down`, then `landlord_up`. The two peasants
cooperate and share their team win. Roles and 20/17/17-card deals are fixed before play;
**there is no bidding**. The three public bottom cards are already included in the
landlord's 20. No spring or bid multiplier is modeled. Bombs and the rocket double score.
The exact wing/composite-action conventions are those of the pinned simulator; this
does not claim parity with every commercial Dou Dizhu variant.

`Game` owns all deal/engine state. `Observation` contains immutable public fields plus
the acting player's hand. It carries no seed, deal ID, other players' hand partitions,
RNG state, or engine handle. The network gets only the encoded public observation and
legal candidate actions. The union of unseen cards is reconstructed by subtracting
own and publicly played cards from the known deck. It is inferable information, not
an opponent-hand partition. Hidden-hand-repartition tests cover all roles.

The upstream 54-bit rank multiplicity action encoding is retained. Static features are
319 for landlord and 430 for each peasant; public history is 5×162 (last 15 actions).
Each legal card multiset is sorted and deduplicated; actions have deterministic
lexicographic indices **within the current observation only**. No global fixed action
vocabulary or legal-action truncation is used. `max_candidates` bounds each neural
head/transfer chunk, not the number of available actions. The network currently ignores
the extra public bottom-card field, matching upstream features exactly.

Natural termination occurs when any hand empties. Nonterminal return is zero; the
terminal target is assigned to every selected learner action in the completed game.

| Objective | Landlord terminal target | Each peasant target |
|---|---|---|
| `wp` | +1 if landlord wins, otherwise −1 | negative of landlord target |
| `adp` | ±2^bombs | negative of landlord target |
| `logadp` | ±(bombs+1) | negative of landlord target |

These are upstream DMC training conventions. Economic score is separately reported:
landlord ±2×2^bombs, each peasant ∓2^bombs, summing to zero. WP win rate and economic
average score are different objectives and must not be conflated or pooled across
unequally weighted seats. Simulator guards, illegal actions, crashes and budget-aborted
waves are unscored failures, never ordinary losses or wins.

## Train, resume, evaluate, profile

Use the **launcher** to count interpreter startup and teardown as well as the loop.
All relative recipe, output and checkpoint paths are relative to the workspace root.
Choose a fresh output directory for each new run/evaluation.

```sh
.venv-doudizhu/bin/python simulators/doudizhu/launch.py train \
  --recipe simulators/doudizhu/recipes/native-local.json \
  --out evidence/doudizhu-implementation/my-learning

.venv-doudizhu/bin/python simulators/doudizhu/launch.py train \
  --recipe simulators/doudizhu/recipes/native-local.json \
  --resume evidence/doudizhu-implementation/my-learning/latest.pt \
  --out evidence/doudizhu-implementation/my-learning

.venv-doudizhu/bin/python simulators/doudizhu/launch.py evaluate \
  --recipe simulators/doudizhu/recipes/eval-practice.json \
  --checkpoint evidence/doudizhu-implementation/my-learning/latest.pt \
  --out evidence/doudizhu-implementation/my-practice

.venv-doudizhu/bin/python simulators/doudizhu/structural_profile.py \
  --engine native --native-fast --workers 4 --codec direct --games 256 \
  --out evidence/doudizhu-implementation/my-profile

.venv-doudizhu/bin/python simulators/doudizhu/run.py replay \
  --trajectory evidence/doudizhu-implementation/my-learning/private-first-wave-trajectories.json
```

On an allocated CUDA/NVCC worker, replace `native-local.json` with the selected L4 recipe.
Use `cuda_modal.py --stage validate` for bounded actual-device parity/resume qualification;
`--stage final` additionally reproduces the latest three-seed full-size BF16 packing check
against the retained private precision campaign. This final stage requires that Volume
evidence. All Modal stages are finite paid allocations with durable source/build/config
receipts and checkpoints.

Resume permits changing total episode/time caps, worker/thread count and execution
device, while preserving model/optimizer/objective/sampling semantics. Increase the
total cap to do more work; the cap is cumulative, not renewed per invocation. The
launcher recovers cumulative elapsed time from the allocation ledger beside the resume
checkpoint. It preserves spent time even after a hard deadline or failed invocation.
Checkpointing is atomic at completed-wave boundaries; hard interruption restarts from
the last committed wave. Bit-exact resume is tested on CPU with a history population.
CUDA midpoint save/resume was executed in all three 2,048-game smokes. A separate
three-pair CUDA continuation check matches trajectories, ordered samples, resulting
weights and evaluation hashes across baseline and optimized worker settings.
Cross-device bit-exact training is not claimed.

Checkpoints contain all three role networks, optimizers, RNG states, counters, next deal
index, used deal hashes, and immutable historical-opponent descriptors. Load uses
`torch.load(weights_only=True)`, strict shapes/keys, rules/source checks, and SHA256 for
frozen opponents. Never change it to unsafe pickle loading. Keep historical artifacts
with checkpoints when moving a run; descriptors contain absolute paths and hashes.
Missing/modified opponents cause an error. Loading opponents does not alter learner RNG.
New checkpoints also bind the complete source/rules/deal lockfiles by SHA256. Legacy
format-1 checkpoints retain their original source/rules checks.

`opponents.selfplay_probability=1` trains all seats. Other waves rotate one controlled
learner role and use a single consistent frozen three-role policy for both other seats.
The frozen/history pool is sampled uniformly; historical network cache capacity follows
the configured population. Changing the population requires a fresh run. Frozen
descriptors are `{"kind":"heuristic"}`, `{"kind":"random"}`, or
`{"kind":"checkpoint","path":".../latest.pt","sha256":"<actual file hash>"}`.
The local heuristic is a transparent cooperating shedding bot, not the upstream RLCard
heuristic or a competitive reference.

## Deals, evaluation and evidence

SHA256 domains `train`, `practice`, and `heldout` produce deterministic independent deals
using explicitly implemented Fisher-Yates and Python 3.12's MT19937. Training reserves
monotone indices. Budget-aborted waves contribute no targets and their reserved indices
are skipped. Actual completed train/eval deal hashes are checked for overlap.

Evaluation reuses each deal with the candidate separately controlling landlord and each
peasant role. Each other seat uses the selected panel opponent, including the candidate's
teammate for peasant tests. Thus the score measures individual-role performance with a
fixed teammate as well as competition; it is not evaluation of a jointly controlled
two-peasant team. Random opponents have independent per-game streams unaffected by
batch scheduling. Checkpoint comparisons use identical schedules and cluster bootstrap
by deal, retaining role/opponent dependence:

```sh
.venv-doudizhu/bin/python simulators/doudizhu/compare.py \
  evidence/doudizhu-implementation/eval-initial-heldout/games.json \
  evidence/doudizhu-implementation/eval-trained-heldout/games.json \
  --out evidence/doudizhu-implementation/my-comparison.json
```

The supplied heldout recipe is a **public example**, inspected in local validation. A
blind grader must select fresh host-private seeds, freeze opponents before evaluating,
and keep its seed file and trajectories outside the researcher's container. An optional
evaluator-only `--seed-file /private/grader/seeds.json` accepts:

```json
{"split":"heldout","deal_seed":123456789,"policy_seed":987654321,"deals":10000}
```

The emitted manifest redacts these seeds and records the seed file hash. Full deal and
action trajectories remain evaluator-side artifacts and must not be sent to the policy.
This Python object boundary prevents accidental feature leakage; it is not an adversarial
Python-code sandbox. A hostile submission requires separate process/container isolation
in the external grader.

Artifacts include `initial.pt`, `latest.pt`, historical snapshots, `events.jsonl`, optional
in-budget development-evaluation rows, `result.json`, `manifest.json`, and the launcher's
`allocation.json`. Training results count games, actual player decisions, legal candidates,
per-role learner samples/updates, stage timings and full worker elapsed time. Evaluation
emits per-game rows and role-specific win rates, Wilson intervals and economic scores.
Private replay JSON records private deal + public observation hashes + actual card
actions; replay requires exact public hashes and terminal returns.

## Compute budget and recipes

Budget is **full allocated worker wall time**, including CPU simulation, startup,
compilation/initialization, transfers, idle GPU, inference, updates, checkpoint I/O and
development evaluation. There is no independent CPU budget or sample-only scoring.
`max_episodes` is a bounded-work safety cap; experiments compared on compute should set
it high and use the same full time/hardware allocation. `allocation.gpus=1` charges the
entire elapsed time as GPU-hours, even when `device=cpu`. Local CPU-only evidence records
zero GPU-hours and reports its actual CPU worker time separately.

The launcher enforces a whole-process deadline and terminates the worker process group;
any teardown overrun is measured and charged. Cooperative checks avoid most partial
waves. Use `--allocated-start-unix <lease-start-unix>` to include setup/transfer time in
an allocation begun before launching Python. Do not double-count overlapping lease
intervals. The external allocator remains the authority for time before/after this worker.
Separate profiling/evaluation invocations also consume allocations and must be included
in the experiment's total ledger; in-loop development evaluation is already charged.

`starter-gpu.json` uses the upstream 512-wide, five-layer heads and LSTM-128 with pure
self-play. `reference-gpu.json` adds a rolling history/frozen mixture and in-budget practice
evaluation. ADP counterparts are provided. Both are reproducible **recipes**, not trained
strong baselines. The small local network is for fast functional smoke, not a benchmark
parameter cap. Architecture, batch size, epsilon, workers, objective and budget remain
configurable. Replace `SET-EXACT-SINGLE-GPU-SKU`, pin attached CPU/RAM, choose fresh seeds,
then execute on an independently approved allocated machine. Nothing here provisions one.

The local full-loop profiler compares duplicate versus factored history inference,
cached public features/views, and zero versus two persistent CPU worker processes on
identical deal/action trajectories.
It includes startup, rollout, learner updates, checkpoint write and worker teardown.
Candidate-transfer memory is chunked. Larger worker counts should be adopted only if a
matched profile on the actual GPU/CPU allocation supports them. Local evidence is on a
shared Mac host, without affinity or resource reservation. The final Mac comparison
held the shared exclusive advisory flock at `evidence/local-performance.lock`.

`structural_profile.py` automatically holds that lock across its child process and
measures simulator-only natural games separately from training, checkpointing and
practice evaluation. It supports `--small` (retained trained small model), `--checkpoint`,
`--workers`, `--device`, `--inference graph`, `--skip-forced`, and `--compact-workers`.
Use a fresh `--out`. CUDA profiles require an actual single-GPU allocation with its
CPU/RAM recorded; running the CLI alone does not provision a GPU.

The user authorized a new structural campaign, superseding the old dollar/hour and
attempt restrictions. `structural_modal.py` uses unique new receipts and leaves the
historical ledger untouched. Its finite matrices run on one single-use L4 + four
physical CPUs + 16 GiB each, with no warm pool or retries. Full app lifecycle time,
including failed work and artifact packaging, remains charged in the new report.

```sh
modal run --env staging simulators/doudizhu/structural_modal.py --mode structural
modal run --env staging simulators/doudizhu/structural_modal.py --mode graph
modal run --env staging simulators/doudizhu/structural_modal.py --mode packed
modal run --env staging simulators/doudizhu/structural_modal.py --mode learning
```

The learning matrix includes an explicitly different 512-game-wave recipe; its
sampling/update cadence is a research change and must be assessed with learning
results separately from exact trajectory/sample/weight-preserving optimizations.

## Executed CUDA experiments

This section describes the earlier small-model wrapper campaign. The structural report
above supersedes its simulator and throughput conclusions; original receipts remain unchanged.

`validated-l4-smoke.json` records the tested small learner: 471,171 parameters across
the three role networks, 128-wide three-layer heads, LSTM-64, batch 64 games, learner
batch 256, one Torch thread, and the measured four-worker CUDA setting. The full
upstream architecture has 4,486,659 parameters and remains available in the reference
recipes; the small model is not a benchmark parameter limit.

The dedicated Modal app used one L4, four physical CPU cores (request=limit), 16 GiB
RAM (request=limit), max one container, no warm pool, two-second minimum scale-down,
zero retries, and 15-minute function timeouts. Four bounded invocations include two
failed attempts, the three-seed sweep, and the final paired learning check. Every own
app is stopped. The cost ledger preserves failures and refuses repeated stages or
additional launches beyond the finite matrix/dollar/hour caps.

Executed remote commands, using the existing staging authentication and Modal 1.5.2:

```sh
modal run --env staging simulators/doudizhu/modal_gpu.py --stage followup
modal run --env staging simulators/doudizhu/modal_gpu.py --stage parity
```

These commands describe the retained run. A new reproduction needs its own approved
budget scope; do not erase the existing cost ledger to bypass its guards. The parity
stage uses the prior stage's locally downloaded, content-checked trained checkpoint.
Use `gpu/followup/learning-seed-*/latest.pt` for resume/evaluation; the parity stage's
checkpoint files are validation snapshots for its special deal schedule.

Actual CUDA evidence: three independent 2,048-game learning smokes, midpoint resume,
paired initial/final evaluation in every role, 24 matched timing trials on a trained
policy, operator-level CPU/CUDA traces, utilization/memory diagnostics, and three
paired 512-game continuations with exact trajectory/sample/weight/evaluation hashes.
The final continuations deliver 1.69–1.71× more learner samples per timed GPU worker
hour, including worker setup, checkpointing and development evaluation. Conservatively
charging the entire app lifetime and equally apportioning all common startup, artifact
transfer and teardown overhead still gives 1.37–1.38× more samples per allocated hour.
Public feature/view
caching removes duplicate CPU work; four persistent workers parallelize legal move
generation. History factorization saves memory and helps CPU, but by itself did not
improve this small model's CUDA throughput. At that stage the GPU spent most time waiting
for CPU encoding/simulation. The complete native/CUDA rewrite and resident training above
supersede that performance conclusion.

The three training seeds improved role-balanced win rate by 29.7–40.1 percentage points
against the weak random/heuristic panel. One seed regressed in the downstream-peasant
role. The 32-deal panel is inspected development data, not a blind competitive grade.
Longer full-size reference training and stronger/fresh opponent panels remain unrun.

Regenerate the machine-readable readout from retained receipts:

```sh
.venv-doudizhu/bin/python evidence/doudizhu-implementation/summarize.py
```

See `reports/doudizhu/implementation-2026-11-08.md` in the workspace for actual results
and the remaining GPU, multiple-seed and strong-reference validation work.
