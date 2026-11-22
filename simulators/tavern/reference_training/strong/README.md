# Tavern retained-incumbent continuation and self-play

This is the November 9 five-hour campaign trainer. Read the current game README,
`HANDOFF.md`, authorization and `reports/tavern/strong-bots-five-hour-2026-11-09-rl.md`.
The Modal launcher is intentionally bound to that campaign and its deadlines;
its closure never renews a historical allowance.

## November 9 H100 pipeline follow-up

The separately authorized H100/4-CPU/16-GiB pass uses the same quarter157
parent, 256 full-from-start lobbies, population mixture, 620,278-parameter
policy and PPO minibatches/objective. The baseline–candidate–baseline control
takes 42.99 / 30.34 / 41.99 trainer seconds for the same 713,793 accepted
decisions: 1.38–1.42× with exact complete training-state parity. The 2× target
was missed. Read the [H100 readout](../../../../reports/tavern/h100-pipeline-2026-11-09.md)
for fresh full-worker rates, provider accounting and qualification limits.

`curriculum-h100-throughput.json` enables `batched_policy_dispatch` and
`ppo_backward_graph`, with at most 64 actor-graph shapes per network. The
collector combines neural inputs into pinned staging, retains learner inputs,
queues each network's original shapes/RNG order, runs CPU scripts while GPU
work is pending, then reads one packed result. Full PPO minibatches reuse a
forward/backward/clipping graph; Adam, remainder minibatches, finite-loss
checks before optimizer steps and actual KL stopping remain in the loop.
No fused Adam, mixed precision, padding, extra accepted samples or changed
opponents are used. The separately checked sampling option stays off because
its independent performance gain was not established.

This CUDA qualification requires PyTorch 2.9.1 and the declared H100 allocation;
unqualified CUDA graph/sampler runtime versions fail preflight. CPU collector
fallback remains supported. The older L4 recipe is unchanged. Exact resume
requires the same source, configuration and input identity; use the explicit
verified-parent fork for an older source, or retain its archived runtime.
The successor launcher/runner are `reference_training.h100_launch` and
`reference_training.h100_run`. Its separate evidence, ledger and close guard
are under `evidence/tavern-h100-pipeline-20261109/`. This performance pass does
not promote a policy or authorize a long training campaign.

## November 9 population throughput

The separately reopened performance task improves this actual trainer around
the existing Rust CPU engine. Read the
[throughput readout](../../../../reports/tavern/population-throughput-2026-11-09.md).
`curriculum-throughput.json` keeps the quarter-rate PPO objective, model, scripts
and rotating four-learner / three-script / one-history mixture, with 256 lobbies.
It enables retained-device rollout inputs, batched visible scripts, once-per-update
numeric encoding with bounded scratch, and at most sixteen CUDA-graph shapes for
each current/history policy. CUDA qualification is on PyTorch 2.9.1 / L4 with
four physical CPU cores and 16 GiB RAM. CPU use of the collector remains supported;
CUDA graph parity is not qualified on other PyTorch versions.

The separately authorized gap-attribution follow-up retains the original numerical
trainer and raises the cache from four to sixteen shapes. A fresh-process
four/sixteen/four crossover improves trainer time by 6–8%, with exact policy,
Adam, RNG, history and native-state parity. Host library settings are explicit
four-thread values; pinning them to one did not beat both repeat baselines.
For a future authorized CUDA run, apply these settings before NumPy/Torch import:

```sh
python -m simulators.tavern.reference_training.start_strong \
  --config simulators/tavern/reference_training/strong/curriculum-throughput.json \
  --inputs VERIFIED_PARENT_INPUTS --out NEW_OUTPUT --seconds FINITE_SECONDS
```

`start_strong` applies the recipe's `host_thread_environment` and invokes the same
trainer. API callers must apply that environment before imports. Strict exact
resume still requires identical source/configuration/inputs; changing cache size
or the declared environment is not an ordinary exact resume. The original 14.6k/s
profile is not a promised fresh-worker rate. Follow-up receipts live separately
in `evidence/tavern-throughput-attribution-20261109/`; neither closed ledger is
reopened by unused capacity. `reference_training.attribution_launch` provides
bounded same-source crossover and fresh-worker checks under a successor ledger.

At 48 lobbies the package preserves exact policy, Adam, RNG, population, full
native states and complete training history against the original path. Larger
cohorts change the update's sample batch and are performance options, not evidence
of improved bot strength. The retained incumbent is unchanged. Exact resume still
requires the same source, configuration and inputs. Use the existing explicit
fork verifier for an old parent; never relabel its source identity.

For a local matched check, the command automatically holds the shared timing lock:

```sh
.venv-tavern/bin/python -m simulators.tavern.reference_training.strong.throughput \
  --inputs evidence/tavern-population-throughput-20261109/inputs-quarter157 \
  --out evidence/tavern-population-throughput-20261109/NEW_LOCAL_CHECK \
  --device cpu --trials 2 --updates 2 --sizes 48 --seconds 240 --proof
```

The input directory is a private locally retained fixture. A fresh checkout needs
the admitted full parent, inference export and its exact runtime, processed with
`prepare_fork.py`. CUDA profiling uses the separate finite
`reference_training.throughput_launch` / `throughput_single` tools and a fresh
task ledger; it never uses the stopped five-hour launcher's old credit. The
completed task's close marker blocks further dispatch, including unused capacity.
Its exact qualified source, inputs, profiles, failures and provider intervals stay
in `evidence/tavern-population-throughput-20261109/` and the private matching Volume.

`policy.py` preserves the original DeepSeek-r3 two-Tanh-layer architecture exactly:
620,278 parameters, width128/embedding8, no LayerNorm. Its imported policy, Adam
moments and Python/NumPy/Torch/CUDA RNG are checked against the original full
checkpoint and NPZ. Import creates a declared new logical seed/update domain.
Ordinary exact resume separately requires identical source, config, input manifest
and native states. Source hashes are never rewritten to admit an old checkpoint.

Recipes:

- `plain.json`: original PPO continuation, 192 lobbies, one learning seat against
  seven native Rust bots, LR0.0004.
- `curriculum.json`: 48 lobbies, four rotating current-policy learners, three frozen
  visible tempo/economy/synergy scripts and one retained-incumbent/history neural
  seat, LR0.0004. Only actual learner decisions enter PPO.
- `curriculum-quarter.json`: same curriculum with LR0.0001, new paired seed531902.
- `curriculum-anchor.json`: same quarter-rate recipe plus0.2 times incumbent-to-new
  masked action-distribution KL. The frozen imported-parent model is a training
  regularizer; exported inference has one ordinary neural policy and no teacher.

The first two execute from the archived `proof-01/source` release. Later canonical
source contains optional anchoring and explicit own-parent migration and needs its
own <=600-second CUDA proof before scaling. Long launch admission verifies the
exact tested executable source, architecture and exercised import branch. Always
pass `--source-release` when resuming; never use changed canonical code to export
an old source-bound checkpoint.

Each full checkpoint contains policy, optimizer, RNG, historical population,
complete history, native states, next cohort seed, config, source/admission binding
and parent lineage. Outputs use atomic checkpoint replacement. Incomplete deadline
cohorts are discarded; simulator faults, illegal actions and natural outcomes remain
separate. All provider time, including collection, idle GPU, snapshots and teardown,
counts against the full allocation.

Within the active authorized campaign, a first proof is launched with the host
`.venv-tavern-ar` environment:

```sh
MODAL_DISABLE_API_PROXY=1 .venv-tavern-ar/bin/python -m \
  simulators.tavern.reference_training.strong.launch \
  --name UNIQUE --arm curriculum --recipe curriculum-anchor.json --seconds 600 --proof
```

Collect with `strong.status --name UNIQUE --critical-only`; use `--collect` for all
retained files. After the exact proof and export pass, use `--source-release
PATH_TO_QUALIFIED_STAGE/source` and at most3400 seconds for a long stage. Finite
GPU/observer deadlines run without a laptop connection. All reservations use the
explicit new `OpenCreditBudget` ledger, with two L4 workers per game at most.
No external model APIs are used.

`snapshot --name STAGE --label UNIQUE` hard-links one complete checkpoint inside
an existing reservation, exports with that worker's exact frozen source and
collects the immutable inference files. A collection-only recovery reuses that
same snapshot; it never reruns training or starts another allocation. Snapshot
IO remains inside the original full worker interval.

`develop --bundle BUNDLE --out NEW_DIR [--extended]` uses only the evaluator's
published public panel and all eight seats. `compare` pairs whole seed groups;
`behavior` measures conditional policy drift on fixed public visible observations.
Neither utility consumes a private final bank. Public development gains require
confirmation on the extended panel; uncertain results retain the incumbent.

`prepare_fork` accepts an exact own strong checkpoint, its unchanged export and
its archived runtime. It copies and checks every parent-bound source file,
checkpoint/NPZ arrays, optimizer/RNG and population, and records nested parent
lineage. It resets only explicitly declared logical/cohort domains. Its optional
import branch must pass a CUDA proof before long use, and the evaluator must
independently admit the recursive lineage. Local fixtures alone do not authorize
promotion or establish policy strength.

The native simulator remains CPU Rust admission-v2, fingerprint217577b83fc4525e.
Rules, player-visible arrays and legal masks stay frozen. No GPU simulator rewrite,
commercial ladder action, publication or push is part of this work.
