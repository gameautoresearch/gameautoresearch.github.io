# The research harness

One harness runs every research session in this repository. The default
researcher is Harbor's **Terminus-2**, working in a terminal inside one uniform
workspace, for all seven games (Generals, Dou Dizhu, Pokémon, Clash, Polytopia,
Tavern, Hex). A game may also enable an installed CLI researcher; Hex, Generals and Polytopia enable
**Claude Code** (see [Researcher agents](#researcher-agents)). There are two
task variants on that same setup:

| Variant | The researcher's answer | What the host does after the session |
|---|---|---|
| `recipe` | The final `training/` source and `configs/` | Trains that recipe from scratch on a fresh worker, then grades the resulting network |
| `network` | A network trained by one of the session's own runs | Grades that network as is |

Everything else is identical between the variants: workspace, tools, budget
accounting, isolation and the backend that trains and evaluates.

## How a session works

```text
harbor run -a terminus-2 -e modal      researcher box: CPU-only, offline, no credentials
        │  ar requests / replies       (/ar/requests, /ar/replies inside the box)
   Broker (harness/broker.py)          host: validates files, applies the variant's rules
        │                              (harness/transport.py reaches into the box)
        │
   Backend (harness/games/<game>/)     host: the game's ledger, trainer, evaluator, workers
```

The researcher's box always looks like this:

```text
/workspace/
  instructions.md   task instructions (common text, game brief, variant, budgets)
  game_rules/       read-only Markdown rules and lookup indexes
  simulator/        read-only public simulator source and engine dependencies
  training/         editable starter network and training code
  configs/          editable experiment configurations (train.json, smoke.json)
  NOTES.md          the researcher's notes
```

Rules, simulator source and instructions are root-owned and read-only; Terminus
runs as an unprivileged user that owns only `training/`, `configs/` and
`NOTES.md`. The box has no network. It has CPU PyTorch (the version of the
game's training workers, `box_torch` in the backend's `SPEC`) and NumPy so the
researcher can import and dry-run their own modules, but no game engine and no
GPU. Only the declared editable files ever leave it, inside `ar` requests, and the host
validates them on every request. Model calls are made by Harbor on the host, so
no key enters the box, the task directory or a command line.

The researcher's only tool besides the shell is `ar`:
`budget`, `status`, `check`, `run NAME --seconds S [--config configs/smoke.json]` (not in a game with leases,
see "Leases"),
`wait`, `log NAME`, `stop NAME`, `eval NAME`, `submit`, `finalize`. `ar` offers
only the options a game acts on: `ar run --seed` and `ar eval --seconds` exist
where the backend's `SPEC` sets `run_seed` and `eval_seconds`. `ar check` asks
the host to validate the files and then, where the game ships
`simulators/<game>/autoresearch/task/dryrun.py`, runs that CPU dry run inside
the box; neither spends allocation. In a
`network` session `ar submit NAME` registers a completed run; in a `recipe`
session `ar submit` registers the current editable files. The latest submission
before `ar finalize` is the answer.

Three rules keep a session from ending with its budget unspent or without an
answer (all in `harness/broker.py`, on for every session except a `--smoke`):

- **Finalize is refused early.** `ar finalize` is refused while a useful share
  of the allocation is unspent: more than 10 % of it, at least one minimum run,
  and the wall-clock limit more than 10 minutes away. `--allow-early-finalize`
  turns this off.
- **A researcher that stops anyway is restarted.** The Harbor task has several
  steps (`--restarts`, default 3 after the first). When a step ends, its verifier
  asks the broker whether the session is unfinished with usable allocation; only
  then does Harbor start the next step: a new Terminus conversation in the same
  box, told to continue from `NOTES.md`. After the wall-clock limit the broker
  accepts no new runs or practice.
- **A session always leaves an answer.** If the host closes a session with
  nothing submitted, the newest complete run of the graded configuration
  (`configs/train.json`; a run that reached its own end is preferred to a stopped
  or cut-off one) is registered as the submission, marked `auto`. An explicit
  submission is never replaced.

Every game's instructions state the same model size limit, `parameter_cap` in
the backend's `SPEC`: 500,000 parameters and buffers, except Pokémon at
1,000,000 (its hashed name-embedding table alone takes 131k-524k).

## Continuing an earlier session

`run ... --continue-from SEED` starts a new session whose workspace holds an earlier session's editable files and
`NOTES.md` instead of the release's starter, and whose instructions open by saying so. A seed is a folder written
by `harness.task.make_seed` (`seed.json` with the files' digests and the prior session's provenance, and
`workspace/` with exactly the release's editable files plus the notes); `task.build` refuses one of another game
or variant, with a missing or extra file, or with changed bytes. The new session has its own ledger, its own new
allocation and its own binding (which records the seed's digest and provenance); nothing the earlier session
submitted carries over. `harness.cloud launch` uploads a local seed folder to `<campaign>/inputs/` and the
coordinator copies it in. Polytopia's `--backend-option scratch_volume=polytopia-scratch-...` makes the research
session adopt an existing Volume (a copy of the earlier session's scratch, never the earlier Volume itself) as
its `/scratch`; a fresh final never mounts scratch.

## Pausing and resuming a session

```sh
$PY -m harness pause --session SESSION [--note WHY]      # returns at once; the coordinator acts within seconds
$PY -m harness resume --session SESSION --execute         # the same session, from where it stopped
$H -m harness.cloud pause --campaign CAMPAIGN --out pokemon/MODEL/main
$H -m harness.cloud launch --campaign CAMPAIGN --game pokemon --secret NAME -- \
    resume --session pokemon/MODEL/main --execute
```

A pause keeps the session open. The coordinator stops answering requests, reads the researcher's editable files and
`NOTES.md` from the box as they are at that moment, stops the researcher, its box and every worker (a training run
keeps what it had saved), settles the ledger and ends. It writes `pauses/NNN/` (a seed of that workspace,
`pause.json`, the result so far) and `pause.json` at the session root with `state: "paused"`, what was left of the
wall clock and the budget. Nothing is finalized and no `session-result.json` is written; while it is paused the
records, packaged networks and worker storage can be pulled and examined, and nothing is running or billed.

`resume` continues that session: the same ledger and remaining budget, the same runs and submissions, the same worker
storage (`/scratch` where the game has it), a new box holding the workspace as it was, and a wall-clock limit of what
was left (`--wall-seconds` sets another). The researcher is a new conversation that is told the session was paused
and to read its notes, `ar status` and `ar budget` first; model, endpoint and agent options are the first run's
unless given again. A session can be paused and resumed any number of times, and a stopped (not paused) session
cannot be resumed. Each part has its own Harbor job folder (`harbor/NAME`, `harbor/NAME-resume001`, ...); the final
result sums their model usage. A cloud coordinator looks for the request in the records Volume every 15 seconds, and
a cloud resume restores the session from the records into a new coordinator.

Resume reopens the game's backend on its existing session root (`resume=True`), the path host grading already
uses. It was run live for Generals on 2026-11-19 (a cloud smoke paused mid-session and resumed to its finalize);
for the other games it is covered only by the offline tests with a stand-in backend.

## Leases

A game whose `SPEC` sets `leases=True` (Hex, Generals and Polytopia) gives the researcher one way to get compute:
workers it leases and drives itself. Such a game has no `ar run` (the client does not offer it and the broker
refuses it). The plain case is the box's `training/launch.py NAME --seconds S`, a researcher-editable script that
calls only `ar`: it checks the files, leases worker `NAMEw` in the game's training shape, runs the starter's
training command there, packages the result as run `NAME` and stops the worker, finishing in the background. It
is built by `task.build` from `harness/templates/launch.py` and the game's
`simulators/<game>/autoresearch/task/launch.json` (training command, output folder, worker shape, time kept back),
and is not one of the editable files that are trained or graded. The fixed training run (`Backend.run`,
`Broker.train`) remains the host's: a recipe's fresh final training and `harness retrain` use it. The other four
games keep `ar run` and have no leases. The lease commands:

| Command | Effect |
|---|---|
| `ar lease NAME --seconds S --gpu 0\|1 --cpu N --memory GIB` | Hold a worker of that shape; the editable files are copied to `/work` |
| `ar exec NAME -- COMMAND`, `ar poll NAME ID` | Run a shell command there as an unprivileged user; read its state and output |
| `ar push NAME` | Copy the editable files again |
| `ar sync NAME [--reload]` | Commit (and reload) the session's scratch Volume, mounted at `/scratch` in every lease |
| `ar package LEASE NAME --path DIR` | Seal a network the lease produced (the files a training run writes) as a run that `ar eval` and `ar submit` accept |

The broker validates names, numbers and the editable files and records every pushed or packaged source
under `sources/`; the backend owns the shape limits, the concurrency limits, the charging rate of each
shape and what a lease can reach. Leases are network-blocked, hold no credentials and are charged from
provider start to end like any other worker. In Hex, one dollar budget pays for training, practice and
leases, each worker second charged at the list price of its shape, and no worker that runs researcher code
contains MoHex. Generals has the same dollar budget (training and practice both use the training shape); its
one worker image holds only what a training run already reads. Polytopia's dollar budget pays for training
runs and leases (its training worker is 4 CPU cores and 16 GiB, about $2.90 an hour, so its $23 is about 8 hours
of it); its practice stays on separate CPU workers outside the budget, and a lease's `/work` is a writable copy
of the worker's Python package with the pushed files linked into it. `ar package` there seals `final.npz` and
`final.json` from a run folder. The scratch
Volume is not deleted when the session ends: collect it for the run archive.

## Researcher agents

`harness/agents.py` is the adapter between the harness and the agent that drives
the box. Terminus-2 is the default everywhere. A game opts into another agent by
naming it in its backend's `SPEC` (`agents=(...)`); a game that does not stays
Terminus-2 only. Hex, Generals and Polytopia currently enable a second agent.

| Agent | Where the model is called | Box network | Credential |
|---|---|---|---|
| `terminus-2` | Harbor on the host, through LiteLLM | offline | stays on the host |
| `claude-code` | Claude Code inside the box | agent phase only | `ANTHROPIC_API_KEY` enters the box |

The adapter adds what an installed agent needs:

- **Image.** Claude Code is installed at image build time, as the `researcher`
  user, so Harbor's agent setup needs no network and the runtime box stays
  offline outside the agent phase.
- **Network.** `[agent].network_mode = "public"` overrides the environment's
  offline baseline for the agent phase only; the verifier phase returns to
  `no-network`.
- **Options.** Harbor's agents do not share option names: the harness maps
  `--api-base` to Terminus-2's `api_base` or Claude Code's `base_url`, drops
  `--model-info` for agents that do not take it, and turns off the installed
  agent's native web tools so the researcher cannot reach the open web.
- **Credential.** `--key-file` / `--key-env` supply the model key; the default
  variable is the agent's (`OPENROUTER_API_KEY` or `ANTHROPIC_API_KEY`).

The tradeoff is explicit: Claude Code runs in the box and therefore needs the
model key there, which Terminus-2 does not. That is why it is opt-in per game
and why the box baseline stays offline. Everything else — workspace, `ar`,
broker, budgets, submission and grading — is identical.

```sh
PY=.venv-hex/bin/python
$PY -m harness run hex network --release NEW_RELEASE --out NEW_SESSION \
    --agent claude-code --model anthropic/claude-sonnet-4-5 \
    --key-file .local/anthropic.key --execute
```

## Commands

Run with the owning game's research interpreter (`catalog` lists them) from the
repository root, with `MODAL_PROFILE=duper`. All cloud work uses the
`game-autoresearch-project` environment.

```sh
PY=.venv-generals-ar/bin/python
$PY -m harness catalog
$PY -m harness prepare generals --out NEW_RELEASE
$PY -m harness verify generals --release NEW_RELEASE
$PY -m harness task generals recipe --release NEW_RELEASE --out NEW_TASK_DIR      # offline: inspect the Harbor task
$PY -m harness run generals recipe --release NEW_RELEASE --out NEW_SESSION \
    --model openrouter/deepseek/deepseek-v4-pro                                    # prints the plan only
$PY -m harness run generals recipe ... --execute                                   # starts the session
$PY -m harness final --session NEW_SESSION        # recipe only, if --skip-final was used
$PY -m harness grade --session NEW_SESSION        # host-private grading
$PY -m harness status --session NEW_SESSION
```

`prepare`, `verify`, `task`, `catalog` and a `run` without `--execute` are
offline and free. `run --execute` spends research worker time (and, for a recipe,
the fresh final training); `grade` spends evaluation worker time. `--smoke`
selects each game's short end-to-end budgets and `configs/smoke.json`.
`--scored` requires a qualification receipt for the exact release.

Harbor is installed once, separately from the game interpreters:

```sh
uv venv .venv-harness --python 3.12
uv pip install --python .venv-harness "harbor[modal]==0.25.0" "modal[api-proxy-support]" socksio
```

(The last two let Harbor's Modal and HTTP clients work when this machine routes
through a SOCKS proxy, as this one does.)

Models are named the way LiteLLM names them. `OPENROUTER_API_KEY` (or the
variable given by `--key-env`, optionally loaded from an owner-only `--key-file`)
is passed to Harbor through its environment. A locally proxied model is reached
with `--api-base http://127.0.0.1:PORT/v1 --model openai/NAME`.

## Running the coordinator in the cloud

The process above (`python -m harness run ... --execute`) is the session's coordinator: Harbor and its
model calls, the broker and the game backend that starts and settles every worker. It must stay alive and
connected for the whole session; on this machine a sleep or a network drop stops its workers.
`harness.cloud` runs the same command in a Modal container instead, so this machine can be closed:

```sh
H=.venv-harness/bin/python
$H -m harness.cloud launch --campaign CAMPAIGN --secret ar-openrouter -- \
    run pokemon recipe --release NEW_RELEASE --out pokemon/MODEL/main \
    --model openrouter/deepseek/deepseek-v4-pro-0813 --research-seconds 10800 --execute
$H -m harness.cloud status --campaign CAMPAIGN
$H -m harness.cloud pull --campaign CAMPAIGN --to runs/cloud      # records, any time, as often as needed
$H -m harness.cloud stop --campaign CAMPAIGN --out pokemon/MODEL/main
```

- `--out` / `--session` are paths inside the campaign; the container runs the session under
  `/work/CAMPAIGN/` and mirrors new files to the `ar-coordinator-records` Volume every minute and at the end.
- `--release` is a local release; it is uploaded once (by binding hash) to `ar-coordinator-releases` and
  verified there by the usual byte and source-identity checks. The image carries the repository's source as
  git lists it, so a release prepared here verifies in the container unchanged.
- Model keys travel only as Modal Secrets in the project environment (`--secret NAME`; create one with
  `modal secret create NAME OPENROUTER_API_KEY=... --env game-autoresearch-project`). `--key-file` and a
  local `--api-base` are refused: the Claude OAuth proxy reads this Mac's keychain and cannot run there.
- `run`, `retrain`, `resume`, `final`, `grade`, `status` and `verify` are accepted; `launch --dry-run` shows the
  rewritten command without building or uploading anything.

Workers hand their results back through a per-session private Volume (`harness/handoff.py`, Pokémon's own
runtime): the worker publishes its exported network and logs and commits the Volume before it ends; the host
reads them afterwards, outside the worker's paid time. A slow or interrupted coordinator link therefore no
longer costs a finished run.

## The seven games

`python -m harness catalog` prints this from each backend's `SPEC`. Hardware is each
game's declared contract (L40S is the repository default where a contract does
not say otherwise).

**Two research budget modes, in every game, chosen per session** (`harness/budget.py`):

| Mode | Jobs | Allocation | Wall clock | Choose with |
|---|---|---|---|---|
| `seconds` | one at a time (a run, a charged practice or a lease) | worker seconds of the training shape; other shapes charged the training-shape seconds their price buys | 2 × allocation + 1 h | `--budget-mode seconds` or `--research-seconds S` |
| `usd` | up to 4 at once (2 GPUs, 64 CPU cores); a queued or running job holds its whole grant | US dollars, each worker second at its shape's frozen list price | the game's `wall_seconds` (5 h) | `--budget-mode usd` or `--research-usd D` |

Without either flag a session gets the game's `budget_mode`, which is `seconds` in every game. Every `SPEC`
declares both modes' defaults;
`ar budget` reports `budget_mode`, `concurrency` and the mode's keys (`*_seconds` or `*_usd`), and the session's
instructions state the mode's rule. Fresh final training is always one run of `final_train_seconds`. A session is
official only in its game's default mode at its default amounts. The table lists each game's default mode.

| Game | Interpreter | Workers | Research / fresh final | Smoke | `prepare` options |
|---|---|---|---|---|---|
| Generals | `.venv-generals-ar` | L40S, 4 CPU, 16 GiB (leases: chosen shape) | $10.00 / 3600 | $0.50 / 120 | `opponent_release`, `panel`, `include_existing`, `private_maps`, `starter` |
| Dou Dizhu | `.venv-doudizhu-ar` | L40S, 4 CPU, 16 GiB | 7200 / 3600 | 600 / 180 | `include_existing` |
| Pokémon | `.venv-pokemon` | L40S, 4 physical CPU, 16 GiB | 14400 / 3600 | 600 / 240 | none |
| Clash | `.venv-clash-ar` | L40S, 4 CPU, 16 GiB | 7200 / 3330 | 600 / 120 | `base_release` (required) |
| Polytopia | `.venv-polytopia-ar` | L40S, 4 CPU, 16 GiB (practice: 4 CPU, no GPU, uncharged; leases: chosen shape) | $23.00 / 3600 | $1.00 / 360 | `panel` |
| Tavern | `.venv-tavern-ar` | L40S, 4 CPU, 16 GiB | 7200 / 3600 | 600 / 240 | `registry` |
| Hex | `.venv-hex` | L40S, 4 CPU, 16 GiB (practice: 8 CPU, no GPU; leases: chosen shape) | $10.00 / 3600 | $0.50 / 180 | none |

Differences a researcher or operator will notice, all stated by `ar budget`,
the game brief or the command's reply:

- **What is charged.** Every game charges training workers. Public practice is
  charged in Generals, Dou Dizhu, Pokémon and Hex (Hex's on an 8-CPU worker
  without a GPU, since MoHex is CPU-only, at that shape's price); in Clash, Polytopia and Tavern it
  runs on separate CPU workers outside the allocation (Clash limits it to six
  per session; Polytopia runs one practice job at a time).
- **Seeds.** Generals, Dou Dizhu and Hex take `ar run --seed`. Pokémon, Clash,
  Polytopia and Tavern take the seed from the config, and their `ar run` has no
  `--seed`. A Pokémon recipe's fresh final training uses host-chosen seeds.
- **Practice time.** `ar eval --seconds` exists in Generals, Dou Dizhu,
  Pokémon and Hex. Clash, Polytopia and Tavern play one fixed schedule.
- **Dry run and live logs.** Pokémon and Hex have a dry run. Pokémon's runs one
  update of the researcher's real `train.py` on CPU against random contract-shaped
  observations, then the real graph export. Hex's engine is plain PyTorch, so its
  dry run trains the real `train.py` against the real engine for about 45 seconds
  on CPU, then asks the package for moves the way the evaluator does. Pokémon's `ar log` also shows a
  running trainer's printed output, read from the session Volume (it trails the
  trainer by the Volume's background commit). Hex's `ar log` shows a running
  trainer's output and progress, copied from the worker every two seconds.
- **Stopped and cut-off runs.** A training run the researcher stops
  (`ar stop`) or that reaches its time limit is packaged from the last complete
  policy it wrote and finishes `complete`, flagged `stopped_early` or
  `cut_short` on its job; it can be practised and submitted, and a fresh final
  that is cut off this way still counts as trained. A run that wrote no
  complete policy ends `stopped` or `failed`. All six original games do this. Closing a
  session ends workers at once and keeps nothing.
- **Small forms.** Four games run a reduced form so that training reaches
  meaningful play sooner: Polytopia
  on 11x11 drylands with a 30-round cap decided by score, Pokémon 3v3, Clash
  with 60+30 s matches at two decisions per second, and Tavern with four players,
  tiers 1-4 and three minion types. Dou Dizhu keeps its full rules. Generals
  returned to its full-size rules on 2026-11-14, graded against learned
  opponents (its 8x8 form was last at commit `48f226ee`). The other
  full-size modes are at git tag `full-size-20261113`.
- **Polytopia's first release of a mode.** Its practice opponents are starter
  runs of the same mode, so the first release is built with
  `prepare --option panel=false` (it trains and smokes, but refuses full
  practice and grading); `retrain --backend-option training_updates=N` then
  trains the starter to an exact sample count for each opponent, and a normal
  `prepare` builds the full release from them.
- **Host evaluation.** Six games have a host-private evaluation for `grade`.
  Clash's fast-v4 contract keeps its private final closed: `grade` plays the
  final once on the release's protected schedule (never shown to the
  researcher; practice uses the public one) and labels it development.
- **Official grades.** A session is official only at the game's declared
  budgets (not smoke, no overrides). Generals and Dou Dizhu also need their
  full-budget calibration before an official grade; otherwise grade as
  development (`--backend-option development=true` for Dou Dizhu). Hex's
  grade is its raw panel score anchored on MoHex and needs no calibration run.
- **Local inputs.** Some constructors read retained local data and fail closed
  without it: Generals' default opponents (use `opponent_release=` or
  `include_existing=false`), Dou Dizhu's default (needs `.venv-doudizhu` for
  torch, or `include_existing=false`), Pokémon's native admission certificate
  and evaluator registry under `evidence/` (or `base_release=` an earlier
  sealed release), Clash's base release, Polytopia's map table and practice
  opponents in its ignored `.cache/`.

## What a session leaves behind

```text
SESSION.harness-run.json   immutable binding: game, variant, release, model, budgets
SESSION/
  plan.json, task/         exactly what the researcher was given
  harbor/, harbor.log      Harbor's job folder: Terminus trajectory, recordings, saved artifacts
  NOTES.md                 the researcher's final notes (copied from the newest researcher step)
  session.json             variant state and every submission
  requests.jsonl           every `ar` request and its outcome
  sources/<run>/           the editable files each training run used
  recipes/<n>/             each submitted recipe (recipe variant)
  research/                the backend's session: ledger, jobs, artifacts
  final/                   the fresh final training session (recipe variant)
  session-result.json, final-train.json, grade.json, completion.json
```

`session-result.json` names the researcher's records (`researcher_records`): the
whole trajectory and the notes of each researcher step are under
`harbor/SESSION/task__*/steps/STEP/` (`agent/trajectory.json`,
`artifacts/workspace/NOTES.md`). Read those. The trial's own
`harbor/SESSION/task__*/agent/trajectory.json` is an earlier snapshot that lacks
the last turns, and `research/agent_workspace/NOTES.md` is the host's copy of the
editable files and is always empty.

A run is complete only when research, worker settlement, evaluation and the
verified private archive are each evidenced (`completion.json`); see
`docs/huggingface-run-archives.md`.

## Releases

`prepare` calls the game's constructor, which builds the sealed release
including the workspace (`harness/workspace.py`), then binds it to the current
harness and game research source (`RELEASE.harness.json`). A session only starts
from a bound release whose bytes and source identity still match. Releases made
by earlier harnesses are not adopted.

## Adding or changing a game backend

`harness/games/<game>/__init__.py` exports `SPEC` (plain data) and `Backend`
(`harness/backend.py` documents every method). A backend wraps the game's own
ledger, trainer, evaluator and worker dispatch; it never calls a language model
and never reads the box. The broker gives it validated files through
`harness.workspace` (`self.sync(config)` copies them into the native trainer
tree) and plain arguments. `harness/games/generals/` is the reference.

Game-side substrate stays with the game in `simulators/<game>/autoresearch/`:
starter, configs (`recipes/`), `task/brief.md`, `task/game_rules/`, benchmark
contract, constructor, trainer/evaluator entrypoints, bundle and ledger code.

## Tests

```sh
python3 -m unittest harness.tests.test_core                       # no game dependencies
.venv-<game>-ar/bin/python -m unittest harness.tests.test_<game>  # each backend, provider stubbed
```

These are offline. They do not prove live isolation, accounting or playing
strength: a real Terminus session on Modal is the remaining qualification.
