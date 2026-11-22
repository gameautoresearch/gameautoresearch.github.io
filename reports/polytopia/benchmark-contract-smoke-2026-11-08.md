# Polytopia contract and autonomous smoke — 2026-11-08

The new full-allocation benchmark and real DeepSeek smoke passed. This demonstrates
an operational research workflow; it does not qualify playing strength.

## Frozen task and implementation

`simulators/polytopia/autoresearch/benchmark.json` and `CONTRACT.md` freeze two Bardur
seats on **14×14**, fog, the existing ten-type `proposed-v1` mix, 200 rounds and
10,000 actions per turn. Both seats use human-equivalent rules. Diplomacy remains
enabled by the current engine default; the historical prompt's “no peace treaties”
claim was incorrect. Resignation stays disabled. Score adjudication at the harness
cap remains, with capped games counted separately from natural endings.

The minimal plain PPO starter is 2×32, random initialization, with 228,579 policy
and critic parameters each (caps 600k/800k). No research options or pretrained
weights were added. CUDA training and Rust grading retain their frozen engine/data
and player-visible observation/legal-mask paths. The new `v1/` controller exposes
inspect/edit/run/evaluate/submit/budget/status/finalize to the real model, with exact
source/config/checkpoint manifests and immutable submissions.

The new 7,200-second full entrypoint is plan-only unless explicitly executed. It
allows variable experiment lengths and repeated submissions; the smoke uses a
separate 600-second cumulative ledger. Provider allocation start-to-finish charges
setup, compilation, training, transfer, attached idle and teardown. Failed jobs
retain charges, unresolved lifetimes retain reservations, only one GPU can run,
and both provider and host deadlines stop the worker. Historical harnesses and
ledgers were not reset or used to fund this smoke.

FULL practice executes 32 public maps per opponent; FULL private grading executes
128 fresh hidden maps per opponent, paired seats, in bounded CPU shards. Four
retained source-compatible baseline checkpoints (random, starter-5m/10m/20m) have
host-only inventory/hash provenance. All four reload and perform masked inference.
Candidate and opponent inference use distinct unprivileged UIDs in a network-blocked
CPU container; trusted Rust state and hidden schedules remain root-only. Candidate
Python is never imported into verifier globals. Cap adjudication reads the actual
final state through non-resetting Rust step; incomplete schedules fail closed.

## Executed smoke

DeepSeek `deepseek/deepseek-v4-flash` inspected the actual starter, changed epochs
1→2 and entropy .01→.015, requested learning, evaluated public practice, submitted
its immutable checkpoint and finalized. The transcript is retained privately.

- **359 updates; 5,881,856 samples; 17 policy tensors changed**, all weights finite.
- Final policy loss .00114448, value loss .00082194; zero illegal engine statuses.
- Training recorded 904 natural outcomes and 1,337 caps.
- Public practice: **2/2 natural wins**, one paired-seat public map.
- Independent private grading: **4/4 natural wins**, two fresh hidden maps, against
  an independently initialized weak opponent. Raw points **100/100**, both seats;
  zero caps, illegal actions or simulator faults. Reference normalization is null.

The two-map bootstrap interval is [100,100]; this tiny weak-opponent panel supplies
no credible calibrated precision or strong-play evidence. The full four-opponent
128-map release was packaged and checked, not run.

The executed immutable release is `release-v5` (SHA
`4a040a41b398b1829f1612ddf71214516d9a22f8504867149157cddc752635ee`).
Later full-session, full-panel, portable-host and bounded-read refinements are in
`release-final-v2`; the JSON report records its exact hash. No second successful
GPU experiment was launched merely to retest packaging refinements.

## Accounting, checks and limits

The strict provider SKU was **H100!**; the guest reported NVIDIA H100 80GB HBM3.
Fixed resources were **16 physical CPU cores and 64 GiB RAM**. The GPU worker's
provider lifetime was **207.974167 seconds / 600**, including all attached phases;
unresolved reservation is zero and the ledger is closed. Five owned workers,
including the failed CPU preflight and CPU practice/grading, are authoritatively
finished. CPU-only allocated lifetime totals 68.778542 seconds.

OpenRouter reported **$0.000823071942**, with conservative retained model-call
liability **$0.0322273224 / $2**. List-price estimates are approximately **$0.4481**
for the GPU worker and **$0.5208** including the recorded CPU workers. They exclude
image builds, egress/storage and plan credits/discounts; provider invoice settlement
is unverified. Rates were checked against [Modal pricing](https://modal.com/pricing),
including Sandbox CPU/memory rates, on Nov 8.

Checks: 16 new contract/accounting/manifest/schedule/checkpoint tests, nine historical
harness tests, four retained-panel reload/inference checks, and a remote CPU gate
covering a real legal Rust transition and denial of private-file access. An initial
CPU gate failed because the unprivileged interpreter could not traverse its install
path; the fixed gate passed before any GPU launch. A failed free model-catalog GET
was also retained; no paid completion or GPU allocation resulted from it.

Exact training/environment resume is **not implemented** in this fresh-start
starter. Checkpoint inference reload, optimizer tensor persistence and controller/
budget restart semantics are checked. Commercial fidelity evidence is strongest on
Bardur Drylands 14×14; synthetic water variants have weaker real-game evidence.
Full-budget reference calibration remains unrun.

## Reproduction and evidence

From the repository root, set up the dedicated host environment:

```sh
uv venv .venv-polytopia-ar --python 3.12
uv pip install --python .venv-polytopia-ar/bin/python -r simulators/polytopia/autoresearch/requirements-host.txt
cd simulators/polytopia
../../.venv-polytopia-ar/bin/python autoresearch/manage.py session --release RELEASE --campaign NEW_CAMPAIGN
```

The last command is a verified offline plan. `benchmark.json.commands` supplies
preparation, explicit paid-smoke and read-only grading commands. The initial live
attempt used an already installed host interpreter; launch logs preserve that exact
command. Permanent commands use `.venv-polytopia-ar`; local JAX checks use
`.venv-polytopia`.

Evidence: `evidence/polytopia-autoresearch/contract-smoke-20261108/`, including all
failed/prepared snapshots, GPU/model ledgers, transcript, learning deltas, sealed
submission, host-private grading and provider cleanup audit. Private seeds, weights,
state and credentials were not published. No commits or pushes were made.
