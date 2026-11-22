# Generals autonomous research contract

Benchmark `generals-autoresearch-v1` uses the source-bound
`generals-research-1v1-medium-v2` simulator. Sessions run under session contract
`generals-autoresearch-v2` (leases, frozen worker prices and two budget modes, below);
the policy-package format is still `generals-autoresearch-v1`, so panel
opponents sealed earlier still verify. `benchmark.json` is the discoverable
machine contract; the immutable prepared release records every public/frozen
file hash, private schedule commitment and exact pinned engine source. Results
are comparable only within the same release and panel commitment.

Two players play independent uniform integer sides 12–15 inclusive, padded to
16×16. Terrain-density input is 0.20, neutral castle count 4–6 inclusive,
castle armies [10,16), minimum BFS spawn separation 10, and no maximum distance.
Fog and general trade are enabled; castle construction and deathtouch are off.
Both decisions are submitted per joint tick and resolved in native order.
The horizon is 1,000 joint ticks. Capture on the boundary takes precedence over
truncation. A cap is a draw, never adjudicated by land or army.
Between 2026-11-13 and 2026-11-14 this contract described a small 8×8 task
(`generals-research-1v1-8x8-v2`: 400 ticks, 513 actions; last at commit
`48f226ee`); its releases and results do not bind to this ruleset.

Actor and critic receive the same visible 16×16×14 tensor and 2,049-action legal
mask. Public scoreboard totals and fog structure silhouettes are intentional.
Pass is action zero; other actions encode cell, direction and whole/half split.
Padding is impassable. No hidden critic, pretraining, inference search or memory
interface is admitted. Submitted inference is stateless greedy legal selection.
Parameters are finite float32, stored up to 32,000,000 elements. The limit is compute per move: one
forward pass for one board at most 1,000,000,000 FLOPs, counted from its jaxpr (`flops.py`).
Exact policy/learner/trainer source, JSON config and safe NPZ weights are sealed
with source/rules hashes; drift and incompatible historical bindings fail closed.

Editable files are starter/policy.py, starter/learner.py, starter/train.py and
recipes/starter.json, staged as agent/* and recipes/* in the CPU research box.
Each `ar run` initializes randomly. The default minimal width-16 CNN has three 3×3
convolutions, a spatial action head and shared visible value head. PPO/GAE uses
current-policy self-play, outcome reward and final-visible-state cap bootstrap.
Training architecture and algorithm changes must remain within the contract.

In dollar mode the full research allowance is **$10.00 of worker time** (`budget_usd`), charged per
worker second at the frozen prices below, with a 5-hour wall-clock limit on the
session. The training worker is one strict L40S, four physical CPU cores and
16 GiB RAM; a recipe's fresh final training is 3,600 seconds of it. The researcher has no `ar run`: it trains on
leases, for which the box's `training/launch.py` does the plain case in that shape; the fixed training run
described below is the host's. Setup, JIT, transfers, simulation, optimization, practice,
saving, idle, failed allocations and teardown count. The broker uses provider
timestamps; an unknown end retains its reservation and its place among the
active jobs. A training run the researcher stops, or whose time limit cuts it
off, is packaged from the network its trainer last saved: with a valid package
it is complete (marked `stopped_early` or `cut_short`) and can be practised and
submitted; without one it is stopped or failed. Its worker time is charged
either way. The last 5 % of a training run's grant (at least 40 s, at most 60 s)
is reserved for ending the trainer and exporting its network as one archive; the
worker's own timeout stays the grant. The shortest run or practice is 120
allocated seconds. Historical ledgers are immutable. Standard full-release host
grading is separately reported compute.

Session contract `generals-autoresearch-v2` has leases and one dollar budget. It
replaces v1 (a seconds budget of one worker, one job at a time), whose results are
not comparable with v2 sessions:

- **Leases.** `ar lease` holds a worker of a chosen shape (0 or 1 L40S, 1-32 CPU
  cores, 2-64 GiB) on which the researcher runs their own commands as an
  unprivileged user, with no network. All leases of a session mount one scratch
  Volume at `/scratch`; training runs and fresh final training do not.
- **Concurrency.** Up to 4 active jobs, at most 2 with a GPU and 64 CPU cores in
  total.
- **Prices.** A worker second is charged `cost(shape)` in US dollars, from
  Modal's sandbox list prices read on 2026-11-13 and frozen in `contract.py`
  `PRICE`: $0.000542 per GPU second, $0.00003942 per CPU core second,
  $0.00000667 per GiB second. Training and public practice use the training
  shape, about $2.90 an hour, so $10.00 is about 3.4 hours of it, which one
  worker can spend within the 5-hour session; workers running side by side
  spend it proportionally faster. These prices are the contract's unit of
  account; Modal's invoice can differ, and the researcher's box, private
  grading and tournaments are outside the budget.
- **Earlier sessions.** A ledger opened under the seconds budget is refused by
  this code: it cannot be continued or graded here.
- **Calibration.** A reference-calibration run gets what the whole budget buys
  of one training worker (about 12,400 seconds).
- **Provenance.** A network packaged from a lease is sealed by the same
  root-side export as a training run, under the same parameter and size caps.
  Weights may descend from earlier work in the same session through `/scratch`,
  never from outside.
- **What a lease can reach.** The worker image holds the pinned engine, the
  control modules and the public training maps, exactly what a training run
  already reads. Private maps and opponent checkpoints are never in the image.

**Budget modes.** Every session runs in one of two research budget modes
(`harness/budget.py`), chosen per session and recorded in its ledger
(`budget_mode`); a ledger refuses to continue in the other mode, and a ledger
that names no mode is a dollar one.

- **seconds** (the default): one job at a time (a training run, a practice or a
  lease), within `budget_seconds` of the training worker: 14,400 seconds
  (smoke 360), the allocation of the last seconds-budget contract. A training or
  practice worker is charged its wall seconds; a lease of another shape is
  charged its wall seconds x `cost(shape) / cost(training shape)` at the frozen
  prices above. The ledger holds the allocation as `budget_seconds x cost(training
  shape)` dollars, so both modes share one accounting path. The wall-clock limit
  follows from the allocation (2 x allocation + 1 hour, at most the harness cap).
- **usd**: the v2 dollar contract described above, $10.00 (smoke $0.50), up to 4
  jobs at once (2 with a GPU, 64 CPU cores), a 5-hour (smoke 80-minute) session.

A recipe's fresh final training is one 3,600-second run of the training worker
in either mode. Leases, the scratch Volume, stop/`stopped_early`/`cut_short`,
provider-receipt accounting and uncharged never-started workers are identical in
both modes.

The separately authorized bounded smoke has a cumulative **600 GPU-second**
campaign limit, including host GPU grading, all attempts and teardown. Its
allocation ledger reserves the sandbox timeout plus 30 seconds for teardown,
settles only authoritative completed L40S receipts, and rejects overlap or an
unfunded allocation. Provider timeout and host termination bound each worker;
an unresolved provider end keeps its entire hold. A provider overrun is recorded
in full and forbids further work; it is never hidden by clipping charged time.
Model calls have an independent immutable $2 charged-plus-reserved ceiling.

The standard release has 256 public training maps, 16 practice maps, 128 fresh
private grading maps and an independent 128-map calibration schedule. Smoke
preparation may use two private maps per schedule. Fresh private schedules are
disjoint from all public/historical partitions and each other. Grading uses
random, expansion and Hunter, plus any sealed checkpoint bots a release imports
from an earlier release of this ruleset; practice exposes random, expansion and
the first such checkpoint. A release prepared with a sealed panel
(`prepare(panel=...)`, built by `ladder.py`) adds learned opponents that keep
their own network code: each runs in its own unprivileged process under a uid
different from the submitted policy's, and each carries an Elo rating on one
ladder with the scripted opponents (Hunter is 0). Practice then exposes
random, expansion and the learned opponents that panel marks for practice. The
sealed panel of this ruleset holds four networks submitted in campaign
`suite-20261113`, rated +350 to +482 on that ladder.
These opponents have no expert-strength qualification.

Every map is played against every panel opponent in both physical seats. Capture
win/loss/cap score 1/0/0.5, with equal map/opponent/seat weights and primary
panel_points = 100 × mean game points. A 95% bootstrap interval clusters whole
maps, preserving opponent/seat pairs. Where the panel is rated, a grade also
reports `panel_elo`: one rating fitted to the games against the rated opponents
(a cap draw is half a win), with the same whole-map interval. Per-seat/opponent results, capture counts,
caps, illegal actions and faults remain separate. Incomplete schedules receive
no score; policy faults invalidate the submission, and simulator/panel faults
abort grading rather than becoming losses. Isolated unprivileged submitted
inference receives only visible arrays and the legal mask; private state,
seeds, bot weights and credentials remain host-owned.

Only explicit completed source-bound submissions count. The latest submitted
policy at allocation thirds is frozen for each milestone; a missing submission
remains missing. Finalization requires resolved allocations, records the latest
explicit final submission and permanently closes research. Grading is cached.
Reference-normalized score remains null until three paired equal-full-budget
training seeds qualify a ≥5-point mean gap, positive paired bootstrap lower
bound and positive gap for every seed on calibration and grading schedules.
The smoke establishes neither calibrated difficulty nor research strength.
