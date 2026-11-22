# Generals autoresearch preparation — 2026-11-08

The end-to-end research harness lives in `integrations/generals_autoresearch/`.
It follows Polytopia's minimal editable starter, host experiment broker,
immutable submissions, milestone grading and hidden paired-map evaluation.
The full-sized Generals simulator remains frozen. Reference strength is not
qualified; fixed-panel game points are the primary score.

## Contract and initial setup

- Game: `generals-research-1v1-full-v2`, independent 17–23 sides, fog,
  padded 24×24, capture outcomes and a 2,048-joint-tick cap. The pinned MIT
  upstream commit remains `954927f508aa47899019d66317cc313af3e9cc19`.
- Default research allocation: 120 full provider-allocated H100 minutes,
  four physical CPU cores and 16 GiB RAM. Setup, compile, practice, simulation,
  training, transfers, save, idle, failure and teardown all count. There are no
  infrastructure-failure time refunds. Standardized host grading and the CPU
  research box are separately reported infrastructure costs.
- Editable starter: `policy.py`, `learner.py`, `train.py` and JSON recipe.
  Width-16 CNN, shared visible critic, plain PPO/GAE, current-policy self-play,
  outcome reward, BF16 compute with float32 parameters. Packed inputs,
  device-resident scans and fused updates preserve the speed work. Cap values
  bootstrap the actual final visible state before reset.
- Submission: safe finite float32 NPZ, at most 600,000 total actor/value
  parameters, exact architecture/source/config/weight hashes, latest explicit
  submission at allocation thirds and finalization. Finalization closes research.
- Inputs: stateless visible 24×24×14 features and 4,609-action legal mask.
  No hidden critic input, inference search or checkpoint pretraining.

The offline release contains 256 public training maps, 16 public practice maps,
128 new private grading maps and 128 independent private calibration maps.
A complete 528-board audit found no duplicate boards, invalid initial states or
private seed overlap with historical public partitions. All starts have BFS
separation at least 17; castles number 9–11 with armies in [40,51).
The public training bank covers all 49 rectangular sizes. Private seeds and
individual hidden outcomes remain local.

## Bots and scoring

The five-opponent development panel is uniform legal random, visible expansion,
pinned MIT Hunter, a strictly compatible early starter checkpoint and a trained
full-sized checkpoint. Practice exposes the first two and the early checkpoint.
Historical checkpoint import uses the original strict source-bound loader.
These bots are development opponents; their competitive strength is unqualified.

Every map is played in both physical seats against every opponent. Capture win
scores 1, loss 0, cap draw 0.5. Mean game points are the reward, with
`panel_points = 100 * reward`. Confidence intervals resample complete maps,
keeping seats and opponents paired. Capture wins/losses, caps, per-seat and
per-opponent results remain separate. Incomplete schedules receive no partial
score. Policy faults invalidate a submission; simulator/panel faults abort the
run rather than becoming capture losses.

Optional starter=0/reference=1 normalization requires equal full-budget training
of three paired seeds, at least a five-point mean gap, positive paired bootstrap
lower bound and positive gap for every training-seed pair. Independent calibration
and grading schedules must both pass. The small pilot cannot qualify this gate;
normalization remains null until the gate passes.

## Isolation and evidence

The CPU research sandbox has no GPU, network, OpenRouter key or Modal token.
The host alone owns experiment budgets, provider timestamps, source snapshots and
submitted artifacts. Submitted inference runs under a verified separate UID,
with no new privileges, root-only private data, CPU/resource limits and bounded
binary visible-input transport. Each grading process handles only one physical
seat of distinct maps. Neural forwards are evaluated independently per example.
Serialization headers are checked before allocation. This is process isolation
and an auditable code contract, not proof of arbitrary source-code compliance.

Local tests cover failed/uncertain allocation charging, immutable budgets,
single-job scheduling, milestone/finalization behavior, cash ceilings, safe
weights, visible-only wire inputs, complete paired scoring, anchor gates,
full-cap bootstrap parity and repeated policy requests. Provider read-only
status calls retry only the observed closed-channel race, within the same job;
no retry creates another allocation or refunds a failure.

The terminal task context, Dockerfile and external-broker verifier transport are
included. The standalone DeepSeek driver is the live smoke path. A separate
Harbor launch is not claimed as tested.

## Live smoke and retained failures

This is an **operator-assisted development smoke**, not a clean unattended
research-trial result or a full-budget strength qualification. Receipts and
source archives are under `evidence/generals-autoresearch/smoke-20261108/`;
release/validation records are under
`evidence/generals-autoresearch/preparation-20261108/`.

DeepSeek V4 Flash actually edited source, checked parameters, trained from scratch
on H100 and explicitly submitted immutable policies. Completed training runs:

| Attempt | Actual policy change | Parameters | Learner transitions | Allocated H100 seconds |
|---|---|---:|---:|---:|
| 2 | LayerNorm after the three convolutions | 7,210 | 20,316,160 | 107.52 |
| 3 | Width-16 residual trunk | 7,114 | 21,692,416 | 116.46 |
| 5 | Width-32 residual trunk | 23,979 | 19,038,208 | 104.22 |

All reported zero illegal training actions. This demonstrates real learning
execution, not improved playing strength. Attempt 5's exact submitted package is
`792365d08fd8cbe2085548724a643747695508b371e97bf1bb50afc1a733f849`.
Its added value projection is unused by its returned value; its actual source
is preserved rather than presenting the agent's architecture description as fact.

Earlier attempts exposed a missing host proxy dependency, SDK transport failures,
a policy-pipe failure and short practice deadlines. Attempt 2's old `complete`
label was incorrect: practice never succeeded. Its zero-game grade is an
infrastructure diagnostic, not measured zero strength. Later failed unattended
trials remain failed/finalized. Failed allocations, interrupted starts, source
snapshots and all model reservations are retained; no research allowance was
reset. There is no completed starter/reference pilot and no full-budget
reference calibration.

The full traceback narrowed a recurring failure to Modal's `TaskExecWait` RPC
inside sandbox filesystem staging. The pinned 1.5.2 adapter now polls the same
existing exec ID, including waits inside filesystem helpers; it does not restart
commands or allocations. Immutable image uploads are resolved separately before
`SandboxCreate`, allowing safe content-addressed retries without duplicating a
worker. Policy pipe writes also use blocking atomic chunks. A real isolated
worker passed 20 sequential requests. Public practice batches its complete
schedule together; private batches remain separated by opponent and physical
seat. Small turn loops and trusted bots stay on attached CPUs; the final configuration
uses H100 only for the isolated submitted neural forward. All H100 wall time,
including idle time and standardized post-submission assessment, appears in the
overall study GPU total and cash accounting. Agent-requested practice counts
against its research allowance. The earlier successful public diagnostic used
CPU inference on its fully charged H100 allocation. Future two-map smoke
requests allow three minutes, rather than the failed 90-second allowance.

Using the unchanged attempt-5 submission, the corrected operator qualification
played **all 12 public practice games**, both seats against all three practice
bots. All reached the genuine 2,048-tick cap: 12 cap draws, zero captures,
zero illegal actions or simulator faults, 50 panel points. The whole H100
allocation was **126.82 seconds**. This separate diagnostic did not reopen or
extend the failed research session or train another policy. The final private
20-game schedule also completed: **40/100 panel points**, with 16 real cap draws,
four capture losses to Hunter and no capture wins. It checked both seats against
all five bots, with zero illegal actions or simulator faults. The trusted game
and bots ran on CPU, and submitted inference ran as the separate UID on H100.
Evaluation took 98.48 seconds; full provider allocation and setup are recorded
in the receipts. Two maps are a workflow smoke, not a strength estimate.
The preceding CPU-only/short-deadline grading attempts remain failed records;
no partial games were assigned capture losses or a score.

The entire study retains one immutable **$5 cash cap** and **$1 maximum for
model calls**, across all attempts and debugging. Cash reservations are admission
ceilings, not an invoice. Completed provider receipts can release only unused
grant seconds, retaining actual failed time, the original 120-second overhead
allowance and 10% resource margin. A proven failure before any provider connection
retains that overhead margin. Model work is permanently closed: reported charges
plus every unknown/failed call's complete reservation and a margin stay held;
the unused maximum cannot be spent on future model calls. Original rows and
settlement proofs remain inspectable. Provider hourly billing can lag recent
allocations; report reconciled billing and remaining conservative holds separately.

Local qualification passes **17 harness tests**, the complete **528-board audit**,
visible-only transport checks, safe preallocation NPZ validation and actual
unprivileged inference. Both historical grading adapters reproduce original
logits and values exactly on four tested public positions. The historical
simulator/checkpoint binding remains
`b1dab24ad2463a4314a190e92970e2fb5cd05b24c656ab20523c7eb1fb4771f0`,
and its current checkpoint loads through the unchanged strict loader.
Release v10 binds the task/Docker/verifier metadata as well as public and frozen
files. Its frozen runtime is the successfully tested v9 runtime: fused trusted
preparation/advance, private progress diagnostics and isolated CUDA inference.
A differential test matches trusted actions, masks, visible inputs and the native
cap boundary. Every release preserves the same game and private map commitments. No controlled timing
comparison is claimed for this preparation step.

## Costs and cleanup

All **15 H100 allocations** are included: **1378.21 allocated seconds**,
including failures, interrupted starts, practice, diagnostic grading and the final
successful grading. All owned allocated workers are stopped. Provider-reported
gross usage as of `2026-11-08T10:32:56.153946+00:00` was **$2.04475241**;
OpenRouter reported **$0.03718805**, with two unknown/failed call
reservations still conservatively covered. These are reported usage amounts,
not a promise that delayed hourly billing is final. Effective conservative cash
holds are **$4.844424**, below the unchanged $5 ceiling.

Stopped apps whose complete billing intervals are available can close their
resource holds against provider gross usage plus 10%. This releases unused
reservation overhead, while preserving original rows, paid failed time, invoice
proofs and allocation records. No app or reservation can be closed twice.
Unbilled or unresolved resources keep their holds; model calls remain closed.

[Machine-readable result](autoresearch-results-2026-11-08.json) contains the
aggregate validation and resource totals. Private seeds, per-game grading rows,
weights and full billing receipts remain local in ignored evidence directories.
The full 128-map/1,280-game assessment and six full-budget reference training runs
were not executed. A clean unattended rerun of the corrected driver is also not
claimed; the successful workflow check was operator assisted.

## Reproduction

See [the harness README](../../integrations/generals_autoresearch/README.md) for
setup, release creation/audit, agent commands and bounded smoke launch. Preparation
and audit allocate no GPU workers. Paid commands default to plan-only unless
`--execute` is supplied. Reusing `--resume` retains the same campaign cash/model
ledgers and writes a new attempt directory; it never resets an allowance.

The previous simulator qualification and throughput results remain in
[the full-size readout](fullsize-performance-2026-11-08.md). The small new smoke
is not a new controlled speed benchmark. Full-budget reference calibration and
competitive-strength qualification remain separate work.
