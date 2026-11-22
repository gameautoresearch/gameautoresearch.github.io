# Polytopia autoresearch

The game-side substrate of the Polytopia research task: Bardur 1v1 on an 11×11 board with fog, drylands maps only,
a round limit of 30 rounds (higher in-game score wins at the limit), frozen 2.17.3 rules, benchmark
`polytopia-bardur-1v1-11-drylands-v1`. Sessions are run by the repository harness: read
[`harness/README.md`](../../../harness/README.md) for how a session works and
[`harness/games/polytopia/`](../../../harness/games/polytopia/) for this game's backend (controller, Modal
provider, provider-receipt transport). Earlier harnesses are in git history at `ba5c0d5`.

This small form replaced the earlier mode (14×14, the ten-type `proposed-v1` map mix, 200 rounds) on 2026-11-13,
because results at that size were too noisy. The earlier mode is the repository at git tag `full-size-20261113`.

## The game mode is defined once

[`starter/src/polytopia_bot/rl/mode.py`](starter/src/polytopia_bot/rl/mode.py) holds the board size, the number
of players, the map type, the round limit (`ROUND_LIMIT`), the training map seeds and the map table's file name.
It is part of the worker package (fixed, not editable; the researcher reads it), and everything else reads it:
the trainer's `LearnerConfig` defaults, `v1/grade.py`, `v1/policy.py`, `v1/preflight.py`, the host's frozen-field
check (`v1/core.py`, which loads the file by path), `build_context.py`, `prepare.py` and the backend's `SPEC`.

To change the round limit, change `ROUND_LIMIT` there and the written copies that cannot import it:
`recipes/starter.json` and `recipes/smoke.json` (`max_turns`), `benchmark.json` (`task.round_limit`),
`task/brief.md`, `CONTRACT.md`, this file and `task/game_rules/rules/mode-and-maps.md` (with its two copies under
`wiki_maintenance/`). `tests/test_contract_v1.py` fails while any of them disagrees. The practice opponents were
trained under the old limit and must be retrained; the map table does not depend on it.

## What is here

| Path | What |
|---|---|
| `benchmark.json`, `CONTRACT.md` | The machine contract and its semantics: mode, hardware, budgets, caps, scoring |
| `contract.py`, `ledger.py` | The session contract `polytopia-autoresearch-v2`: the research budget in either mode (seconds of the training worker, the default, or dollars), worker shapes and frozen list prices, lease and concurrency limits; and the session's SQLite ledger (jobs, charges, submissions, milestone points) |
| `prepare.py` | The release constructor, `simulators.polytopia.autoresearch.prepare:prepare` (no options) |
| `build_context.py` | Stages a release's worker runtime: both engine source trees and the Python package pruned to the import closure of the trainer and the worker entrypoints (37 modules), the starter overlaid, research comments scrubbed; fails if a research module or hidden recipe would be staged. Also checks the two local inputs of a release (`map_table()`, `panel()`) |
| `v1/core.py` | Host side of the benchmark: the frozen game mode, sealed manifests, frozen configuration fields |
| `v1/train.py`, `v1/grade.py`, `v1/policy.py`, `v1/preflight.py` | Worker entrypoints. The provider mounts `v1/` at `/opt/v1` and names these paths: training launch, the trusted Rust evaluator, the unprivileged policy process, the CPU isolation gate |
| `starter/` | The minimal starter: `rl/network.py`, `rl/ppo.py`, `rl/gpu_resident.py`, `experiments/gpu_resident_train.py` (editable) and `rl/mode.py`, `rl/infra.py`, `rl/data.py` (fixed: the game mode and plumbing, including `infra.game_outcomes`, the one outcome rule the starter trains on and `v1/grade.py` scores with: elimination, or the higher in-game score at the round limit, equal scores a draw) |
| `recipes/` | Public configurations: `starter.json`, `smoke.json`, `starter_arch.json` |
| `hidden/` | Reference recipes; never staged into a release |
| `task/brief.md`, `task/game_rules/` | The game brief and the Markdown rulebook the researcher receives |
| `base.Dockerfile` | The worker base image (CUDA, Python, torch, JAX, both engines). Kept byte-identical so the built image stays valid; its header comment predates this harness |
| `requirements-host.txt`, `requirements-host.lock.txt` | The host interpreter `.venv-polytopia-ar` (pinned Modal 1.5.2 and NumPy; no JAX) |
| `tests/` | Contract, provider-pin and staged-runtime tests |
| `wiki_maintenance/` | Maintainer-only rulebook sources and generators |
| `replays/` | Local CPU exhibition exporter for retained finalized submissions |
| `overnight/2026-11-07/` | Frozen record: imported researcher submissions and their manifest |

Git-ignored local state: `.cache/` (the training map table and the four practice opponents of the current mode;
also the earlier mode's `table-proposed-v1-14-2p-200000-300000.npz` and `practice/`, which nothing reads now),
`jobs/` and `task/environment/` (staged trees of earlier harnesses; nothing reads them now).

## Workspace mapping

| Researcher file | Native trainer path |
|---|---|
| `training/policy.py` | `src/polytopia_bot/rl/network.py` |
| `training/learner.py` | `src/polytopia_bot/rl/ppo.py` |
| `training/train.py` | `src/polytopia_bot/rl/gpu_resident.py` |
| `training/collector.py` | `experiments/gpu_resident_train.py` |
| `configs/train.json` or `configs/smoke.json` | `recipe.json` |
| `configs/architecture.json` | `arch.json` |

## Preparing a release

```sh
.venv-polytopia-ar/bin/python -m harness prepare polytopia --out NEW_RELEASE
```

Preparation is offline. It needs two local inputs under the git-ignored `.cache/`, both made for the current
game mode:

- `table-drylands-11-2p-200000-300000.npz`, the training map table (100,000 layouts);
- `practice-drylands-11/{starter-200m,starter-1h,research-mid,research-top}/`, the practice opponents (the
  stronger panel of 2026-11-15; see `build_context.PROVENANCE`), each with
  `final.npz`, `final.json`, `network.py`, `ar_arch.json`.

It stops with a message if either is missing, if the table is not this mode's (seed range, map type, tile count),
or if an opponent's checkpoint does not match its recorded digest or was trained for another board. A release
holds `runtime/` (worker source), `maps/`, `v1/`, `public/` (recipes and the researcher workspace),
`base.Dockerfile` with `base-context/`, the host-only `host-private-panel/` and a sealed `manifest.json`.

### The map table

Built once per mode on CPU workers in the project's Modal environment, then fetched (from `simulators/polytopia/`):

```sh
MODAL_PROFILE=duper MODAL_ENVIRONMENT=game-autoresearch-project \
  ../../.venv-polytopia-ar/bin/modal run --env game-autoresearch-project experiments/modal_map_table.py::build \
  --mix drylands --size 11 --players 2 --lo 200000 --hi 300000
MODAL_PROFILE=duper ../../.venv-polytopia-ar/bin/modal volume get --env game-autoresearch-project \
  polytopia-map-tables /maps/table-drylands-11-2p-200000-300000.npz \
  autoresearch/.cache/table-drylands-11-2p-200000-300000.npz
```

### Practice opponents

The opponents are starter runs of the current mode, so the first release of a mode cannot contain them. A
bootstrap release, prepared with `--option panel=false`, has no opponents: it trains and runs smoke sessions, and
refuses full practice and grading. The opponents are made with it (from the repository root):

```sh
PY=.venv-polytopia-ar/bin/python
$PY -m harness prepare polytopia --out BOOTSTRAP_RELEASE --option panel=false
# One host training of the unmodified starter per checkpoint, ended after a fixed number of updates. An update is
# 65,536 samples (the starter's `batch`): 305 updates = 20.0 M samples, 992 = 65.0 M, 3052 = 200.0 M.
# Without --execute each prints its plan only.
$PY -m harness retrain polytopia --release BOOTSTRAP_RELEASE --starter --seed 801 --label starter-20m \
    --final-train-seconds 840 --backend-option training_updates=305 --out NEW_DIR_20M --execute
$PY -m harness retrain polytopia --release BOOTSTRAP_RELEASE --starter --seed 801 --label starter-65m \
    --final-train-seconds 2100 --backend-option training_updates=992 --out NEW_DIR_65M --execute
$PY -m harness retrain polytopia --release BOOTSTRAP_RELEASE --starter --seed 801 --label starter-200m \
    --final-train-seconds 6000 --backend-option training_updates=3052 --out NEW_DIR_200M --execute
```

`--final-train-seconds` is only a ceiling: the trainer ends at its update count and the worker's real lifetime is
what is charged. The ceilings were set before the board's speed was known (for 50,000 samples a second). Measured
on 2026-11-13: the starter compiles for about 127 s and then trains about 105,000 samples a second; the three
training loops took 332, 765 and 2,068 s and the workers were charged 369, 816 and 2,141 s. With the 130 s the
host keeps of a grant, the ceilings 840, 2,100 and 6,000 s leave ample room.

Each run's network is `NEW_DIR/final/jobs/final/bundle/`. Check `learning.json` beside it (`jobs/final/`): `updates`
must be the number asked for and `ended` must be `complete`; a run that reached its time limit first has fewer
updates and needs a longer `--final-train-seconds`. Then place the files:

```sh
P=simulators/polytopia/autoresearch/.cache/practice-drylands-11
for n in 20m:NEW_DIR_20M 65m:NEW_DIR_65M 200m:NEW_DIR_200M; do
  d=$P/starter-${n%%:*}; b=${n#*:}/final/jobs/final/bundle; mkdir -p $d
  cp $b/final.npz $b/final.json $b/network.py $b/ar_arch.json $d/
done
# `random` is the untrained network every run starts from: the initial checkpoint of one of them.
b=NEW_DIR_20M/final/jobs/final/bundle; mkdir -p $P/random
cp $b/initial.npz $P/random/final.npz; cp $b/initial.json $P/random/final.json
cp $b/network.py $b/ar_arch.json $P/random/
```

After that `harness prepare polytopia --out NEW_RELEASE` builds the full release.

**The stronger panel (2026-11-15).** The panel above was saturated: every research network scored 100 against
`random` and about 96 against `starter-20m`, so half the grade carried no signal. Since 2026-11-15 the panel is
`starter-200m` (above) plus three networks copied unchanged (`final.npz`, `final.json`, `network.py`,
`ar_arch.json`) from sealed bundles of earlier sessions. Each folder's `SOURCE` file names its bundle:

| Opponent | Source bundle | Open-tournament Elo vs starter-200m |
|---|---|---|
| `starter-1h` | `polytopia-recipe-20261114` Sol final (unmodified starter recipe, host-trained 3,600 s) | +86 |
| `research-mid` | `polytopia-20261114` Sol research submission (48×3 normalized-v2) | +128 |
| `research-top` | `polytopia-20261114` Astra research submission (48×3 context-v3) | +190 |

Each opponent runs in its own UID-isolated policy process with its own `network.py`, as tournament entrants do.
Grades on this panel are not comparable with grades on the old one. The old folders stay in the cache.

## What the backend enforces

- One research budget pays for leases, in the session's budget mode (`harness/budget.py`, recorded in the
  ledger): by default 14,400 seconds of the training worker (smoke 600), or with `--budget-mode usd` /
  `--research-usd` $23.00 of worker time (about 8 hours of the training worker; smoke $1.00). The training
  worker, which `training/launch.py` leases by default and a recipe's final training gets, is an L40S with
  4 CPUs and 16 GiB, about $2.90 an hour at the list prices frozen in `contract.py`; a lease is a worker of a
  shape the researcher chooses, charged at that shape's price (in seconds mode: its seconds weighted by that
  price relative to the training worker's). A request
  holds its whole grant from the moment it is queued, and the worker's own provider start-to-finish lifetime is
  what is charged, including failures. A worker queued for a GPU is waited for uncharged; one the provider never
  starts is terminated and settled at nothing (harness/workers.py). An abandoned worker that did start is
  terminated and charged from its receipt.
- Seconds mode: one training run or lease at a time. Dollar mode: up to 4 jobs at once, each on its own worker
  thread, at most 2 with a GPU and 64 CPU cores in total. Either way at most one public practice job (practice
  runs on CPU outside the budget and never blocks training).
- Leases (`ar lease`, `exec`, `poll`, `push`, `sync`, `package`): researcher commands run as uid 65534 in
  `/work`, a writable copy of the worker's `src/` and `experiments/` in which the four editable trainer paths are
  links to the pushed `training/` files, with the session's scratch Volume at `/scratch`
  (`research/scratch-volume.json` names it; it is not deleted). `ar package LEASE NAME --path DIR` reads
  `DIR/final.npz` and `DIR/final.json` as that user and seals them, with the current `training/policy.py` and
  `configs/architecture.json`, as a run that practice and `ar submit` accept; the host checks the checkpoint's
  digest, the 500,000 policy cap and finite values without importing anything.
- Before any GPU time, a CPU worker builds the networks and checks the 500,000 / 500,000 parameter caps.
- A grant of G seconds is divided like this (`schedule` in the provider; `jobs/<run>/schedule.json` records it):
  the worker ends at G − 15; the trainer command is cut off at G − 80, which leaves at least 60 s to collect a
  cut-off run; the trainer is told to end its loop 115 s before the worker's end, counted from when its command
  starts, so one that ends by itself (about 25 s to start, 3 s to exit) leaves about 80 s. For 900 s that is a loop
  of about 760 s; for 3,600 s, about 3,460 s. A run is collected as two compressed archive streams, each with its
  own time limit: the run itself (initial and final policy, critic, training log, the last 256 KiB of the trainer's
  output; about 4.9 MB after an hour, 7–13 s at the 0.4–0.7 MB/s measured), then the 5- and 10-minute clock
  snapshots (5.1 MB), which are taken only if time is left and are not needed for the run to count. The trainer's output
  goes to files in the worker and is never read through its command's stream. A failed collection records why in
  the job's error and in `jobs/<run>/export-error.json` (time limit reached with how many bytes, or the collection
  command's own message); `jobs/<run>/export.json` and `trainer.json` record the timings of a successful one.
- A training run the researcher stops, or one cut off at its time limit, keeps the last complete policy it wrote
  and is `complete`, flagged `stopped_early` or `cut_short` (on the job in `ar status` and `ar log`); a run that
  wrote none fails. `ar stop` creates a request file in the worker (`$POLYTOPIA_STOP_REQUEST`), at which the
  starter's loop ends with its final checkpoint; a trainer still running 180 seconds later, and one whose command
  reaches its time limit, is packaged from its newest whole periodic export (`exports/update-N/`). The kept policy
  must still differ from the initial one, with finite logged losses. Closing a session ends workers at once and
  keeps nothing. `training_updates=N` is a host-only backend option that ends training after N updates.
- Public practice and private grading run on network-blocked 4-core CPU workers and are not charged to the
  budget. Submitted network code runs in a separate unprivileged process.
- The private schedule (128 maps, 2 in a smoke) is drawn when a session opens and stays in the session's
  `private/` directory. Only the host can grade.
- Smoke: $1.00, one successful run, the 2×32 architecture, one public practice map.
- A session directory of the earlier contract (`gpu-ledger.json`, `state.json`; seconds of one worker) is refused.

## Tests

```sh
.venv-polytopia-ar/bin/python -m unittest harness.tests.test_polytopia        # backend, provider stubbed
.venv-polytopia/bin/python -m pytest -q simulators/polytopia/autoresearch/tests   # contract and runtime (JAX)
```

Both are offline. They do not prove live isolation, accounting or playing strength.
