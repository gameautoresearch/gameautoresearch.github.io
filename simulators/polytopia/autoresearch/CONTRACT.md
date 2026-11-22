# Polytopia Bardur duel, 11×11 drylands, v1

Machine contract: `benchmark.json`; game brief: `task/brief.md`; rules: `task/game_rules/`.
Sessions run through the repository harness (`harness/README.md`, backend
`harness/games/polytopia/`). This is the full-allocation benchmark. Historical source
snapshots, receipts, old GPU-time ledgers and the cancelled optimization campaign remain
historical and unchanged. This preparation does not authorize 120 minutes.

The frozen task is `polytopia-bardur-1v1-11-drylands-v1`: two Bardur seats, 11×11, fog,
drylands maps only (one synthetic land map type), a round limit of 30 rounds,
no resignation, max 10,000 actions per turn. One outcome rule serves training and
evaluation (`game_outcomes` in the worker package's `rl/infra.py`, called by the starter
trainer and by `v1/grade.py`): +1/−1 when a player is eliminated; at the round limit the
higher in-game score wins and equal scores draw (0). The starter rewards nothing else. The board, the map type and the round limit
are defined once, in `starter/src/polytopia_bot/rl/mode.py`; the trainer's defaults, the
evaluator, the host's configuration check and the release constructor read them there.
This small form replaced the 14×14, ten-type-mix, 200-round mode on 2026-11-13 because its
results were too noisy; that mode is the repository at git tag `full-size-20261113`. Diplomacy is the current enabled engine
default. CUDA engine training and Rust CPU grading use frozen 2.17.3 data. Native
outcomes and score-adjudicated caps are separate counters. Illegal actions or engine
faults invalidate evaluation, rather than silently producing a win/draw.

Session contract `polytopia-autoresearch-v2` (`contract.py`, `ledger.py`), built the way
Generals' and Hex's are. The benchmark id above did not change, so sealed bundles and
practice opponents still verify. A training run is one L40S (strict provider request
L40S), 4 fixed CPUs and 16 GiB RAM (16 CPUs and 64 GiB until 2026-11-14). Fresh final training is 3,600 seconds
of the training worker, one run. The researcher has no `ar run`: it trains on leases, for which the box's
`training/launch.py` does the plain case in the training worker's shape; the fixed training run is the host's,
for a recipe's final training. The research budget is in one of the harness's two modes
(`harness/budget.py`), chosen per session and recorded in the session's ledger; a
resumed session keeps its mode and refuses another:

- **seconds** (the default): **14,400 seconds of the training worker** (smoke 600), one
  lease at a time, with the harness's wall clock (2 × the allocation + 1 hour). A lease is
  charged its worker seconds × its shape's list price relative to the training worker's
  (`budget.seconds_weight`), so a second of the training worker is always a second of
  allocation.
- **usd**: **$23.00 of worker time** (smoke $1.00) with a 5-hour wall-clock limit. Every
  worker second of a lease is charged what its shape costs at the Modal sandbox list prices
  frozen in `contract.PRICE` (GPU $0.000542, CPU core $0.00003942, GiB of memory
  $0.00000667 per second, read 2026-11-13): the training worker is about $2.90 an hour, so
  the budget is about 8 hours of one. (The $23.00 was set as the 14,400 seconds the small
  form was given, at the earlier 16-CPU, 64 GiB worker's $5.76 an hour.) Up to 4 jobs run
  at once (leases and practice together), at most 2 of them with a GPU and 64 CPU cores in
  total.

These prices are the contract's unit of account; Modal's invoice may differ. Provider
task started_at through finished_at charges ALL phases, including failures, idle time
and teardown; no warm pool or automatic retry. A request holds its whole grant, in the
session's unit, from the moment it is queued until its worker's provider record settles
it, so grants can never add up to more than the budget; an unresolved provider lifetime
retains its whole grant (and, in seconds mode, the one job slot) and blocks finalization. Sandbox timeout is shorter than its grant, and an independent
host timer terminates it before the grant's end. A training run is at least 300
seconds, a lease at least 60.

A lease is a worker of a shape the researcher chooses (0 or 1 L40S, 1-32 CPU cores,
2-64 GiB) that runs the researcher's own commands as an unprivileged user (uid 65534,
no network, no credentials), with the session's one scratch Volume at `/scratch`.
`/work` holds a writable copy of the worker's Python package and launch script with
the pushed editable files linked into their trainer paths. `ar package` reads
`final.npz` and `final.json` from a directory on the lease as that user and seals them
with the current policy source and architecture as a run's bundle; the host checks the
checkpoint as data (its record's digest, the 500k policy cap, finite values) and
imports nothing. A packaged checkpoint is not required to differ from an initial one.
The worker image holds only what a training run already reads: the practice opponents
and private schedules never enter it. In a recipe session leases are exploration only.
A session opened under the earlier contract (v1: 14,400 seconds of one worker with its own
ledger files) cannot be continued or graded with this code.

The plain 2×32 starter learns from random weights; no pretraining, research toggles
or extra baseline options. Policy cap 500k, critic cap 500k. Source/config/checkpoint
manifests bind submissions. Run names are immutable; sealed submission bytes are
read-only and verified again for grading. Milestones refer only to already submitted
artifacts; finalization closes further paid work and seals the last submission.
Training runs deliberately start afresh. The starter has no exact environment/optimizer
continuation API: checkpoint inference reload is supported, exact training resume is
NOT qualified. This limitation must remain explicit until a complete state test exists.

A training run the researcher stops (`ar stop`) or one cut off at its time limit keeps
the last complete policy it wrote and ends `complete`, flagged `stopped_early` or
`cut_short`; a run that wrote none fails. The starter writes whole exports
(`exports/update-N/`, renamed into place) after its first update and every 60 seconds.
On a stop the host creates a request file in the worker and the trainer's loop ends at
its next step with its final checkpoint; a trainer still running after 180 seconds, and
one cut off at its limit, is packaged from its newest whole export. The checks that the
kept policy differs from the initial one and that the logged losses are finite apply
unchanged. A session being closed ends its workers at once and keeps nothing. A fresh
final training that is cut off but had exported counts as trained (`cut_short`).

Raw game points = 100 × mean(win=1, draw=.5, loss=0), balanced across paired seats and
opponents. Higher score wins at the cap, equal score draws. Report natural/capped,
seat and opponent breakdowns with map-cluster bootstrap intervals (2,000 resamples).
Private smoke uses two fresh random host maps and an independently initialized weak
opponent; public practice one separate paired map. Never expose private schedules or
opponent weights to the agent. Private result details remain ignored host evidence.
Untrusted network code executes in unprivileged separate inference processes in a
network-blocked CPU container; it receives only visible features and legal masks.
Root-only trusted Rust state, schedule files and opposing policy are outside its UID.
Full-release grading implements 128 maps × four retained baseline opponents;
stronger reference qualification remains incomplete.
Reference normalization stays null until independent full-budget multi-seed calibration.

The researcher is run by the repository harness; this contract holds no model client
and no model budget. No credentials, private seeds or policies enter the researcher's
workspace or the public release.

Known fidelity boundary: real-game evidence is primarily Bardur Drylands 14×14; this
task's maps come from the synthetic `drylands` preset at 11×11, for which no commercial
map-generation parity is claimed.
No claims of full commercial parity, medium/strong play or model ranking follow from smoke.

Full-session grading implements the four-opponent, 128-map schedule,
using the starter-200m/starter-1h/research-mid/research-top checkpoints of this game mode (stronger
panel since 2026-11-15; random/starter-20m/65m/200m before) and inventoried
with hashes in the prepared host release. A release prepared with `panel=false` has no
opponents: it trains (that is how the opponents are made) and refuses full practice and grading. Provider.image excludes host-private-panel.
Each pair is played through UID-isolated CPU inference in four-map shards, up to eight at once; final
points weight opponents equally and bootstrap whole map clusters. This operational
panel replaces the earlier proposed count; its strength remains uncalibrated.
A session is plan-only unless `python -m harness run ... --execute` is given.

FULL public practice uses 64 fixed public maps per retained opponent (32 before 2026-11-15), paired seats;
SMOKE practice uses one public map against an independent weak random initialization.
The full practice and private panels execute from host-only checkpoints; neither
weights nor hidden schedules are added to agent input. Private grading and CPU-only
practice have separate recorded CPU lifetimes and no attached GPU; neither is charged
to the research budget, and one practice job runs at a time in either mode (in seconds
mode beside the one training run or lease).
