# Generals research substrate

The game-side half of Generals research sessions: the starter agent, the
contract, the release constructor, the worker entrypoints and the evaluator for
`generals-research-1v1-full-v2` (full-size fog-of-war 1v1: sides 17–23 padded to 24×24, 2,048
ticks, 4,609 actions; the 8×8 task that stood in for it on 2026-11-13 and 14 was last at commit
`48f226ee`). Sessions themselves
are run by the single research harness; read [harness/README.md](../../../harness/README.md)
for the workspace, the `recipe` and `network` variants and the commands. The
Generals backend is `harness/games/generals/` (session controller and Modal
provider). Earlier harnesses that lived here are in git history at `ba5c0d5`.

## Setup

```sh
python3 tools/restore_upstream_metadata.py        # after a fresh clone
sh simulators/generals/autoresearch/setup.sh       # creates .venv-generals-ar
.venv-generals-ar/bin/python -m pytest simulators/generals/autoresearch/tests -q
.venv-generals-ar/bin/python -m unittest harness.tests.test_generals
```

Run everything from the repository root. `integrations/generals_autoresearch` is
a compatibility link to this directory; staged worker trees keep that package
name.

## What is here

| Path | Purpose |
|---|---|
| `benchmark.json`, `CONTRACT.md` | The benchmark contract `generals-autoresearch-v1` (machine and prose forms) and session contract `generals-autoresearch-v2` |
| `contract.py` | Session defaults (`DEFAULT`), editable file list, engine pin and source binding, the public train/practice seeds (`public_seeds`), worker shapes, prices and lease limits |
| `starter/`, `recipes/` | The starter policy, learner and trainer, and its `starter.json` / `smoke.json` configs |
| `starter_empty/`, `recipes/empty*.json`, `task/brief-empty.md` | The empty starter (`prepare(starter='empty')`): a linear placeholder network, an empty learner and self-play plumbing with no learning, its configs and its own brief |
| `flops.py`, `learned.py` | The two checks on a submitted policy: compute per move (FLOPs of one forward pass), and that it is learned (it must choose differently once its weights are re-initialized; a hand-written player in `policy.py` is rejected) |
| `task/brief.md`, `task/game_rules/` | The game brief and Markdown rulebook staged into every workspace |
| `prepare.py` | Release constructor (`prepare`), `audit_release`, `audit_map_banks` |
| `ledger.py` | SQLite session ledger: the budget mode, allocations in dollars at each worker shape's price (a seconds-mode budget is held as dollars of the training worker), concurrency limits, packaged networks, submissions, milestones, finalization |
| `bundle.py` | Sealed policy packages: safe weight files, parameter cap, source-bound verification |
| `rollout.py` | Fixed self-play collection used by the starter trainer |
| `train_entry.py`, `training_limits.py`, `export_entry.py` | Worker side of a training run: unprivileged supervised trainer, then root-side export |
| `evaluate_entry.py`, `arena.py`, `policy_worker.py`, `wire.py`, `grading.py` | Worker side of practice and grading: isolated policy process, paired-seat panel games, scoring |
| `manage.py` | Host-private `grade_session`, reference `calibrate`, `panel_paths` |
| `ladder.py` | Host-only panel calibration: `fit` an Elo ladder to recorded tournament and grading games, `seal` a rated panel folder of learned opponents for `prepare(panel=...)` |
| `transport.py` | Guard for the pinned Modal 1.5.2 gRPC transport used by the provider |
| `hidden/` | Reference learner and recipe for anchor calibration; staged only into a release's private tree |
| `strong_eval/` | Full-size (`full-v2`, 24×24) cross-play evaluator with the AverageJoe reference class ([README](strong_eval/README.md)) |
| `official_maps.py` | Local pilot importer for official 17–23 ladder maps; not used by releases |
| `export_demo_replays.py` | Demo replay exporter bound to one full-size smoke policy |
| `wiki_maintenance/` | Maintainer sources and provenance for `task/game_rules/` |
| `setup.sh`, `requirements-host.txt`, `pytest.ini`, `tests/` | Interpreter setup and this directory's tests |

## Releases

```sh
.venv-generals-ar/bin/python -m harness prepare generals --out NEW_RELEASE \
  --option panel=runs/generals/panels/suite-20261113-top4
```

`prepare(out, include_existing=True, private_maps=128, opponent_release=None, panel=None, starter='ppo')` is
offline and starts no worker. `starter='empty'` (`--option starter=empty`) builds the from-nothing form: the box
and every unprivileged worker process see only `frozen/unprivileged` (the pinned rule modules, the wrapper without
its scripted players, the weight writer and the two checks), and the image keeps the rest of the frozen tree
(reference policy, scripted players, the engine's reward helpers, example trainer and history) readable by root
only. It writes:

- `frozen/`: the pinned engine, the three simulator runtime files and the
  `CONTROL` modules listed in `prepare.py`, which become the worker image;
- `public/`: starter, configs, contract, the 256-map training bank, 16 practice
  maps (the practice seeds of `simulators/generals/seeds.json`: `contract.public_seeds`),
  practice opponent names and the researcher workspace
  (`harness.workspace.stage`);
- `private/`: fresh grading and calibration map schedules disjoint from every
  public seed, the opponent panel and its bundles, the reference recipe and the
  calibration plan.

`opponent_release` copies the sealed panel bundles of an earlier release after
verifying them against this release's engine and rules. Without it and without
`panel`, `include_existing=True` looks for two historical checkpoints under
`evidence/generals-implementation/`: where they are still present the
conversion fails closed (they bind to an older simulator source hash), where
they are not the panel is scripted only. `include_existing=false` builds
the scripted-only panel (random, expansion, Hunter) explicitly. `private_maps`
below 128 gives a release that is not `grading_ready`
for official grades.

`panel` is a sealed, rated panel folder: learned opponents that keep their own
`policy.py` (panel kind `policy`, up to `contract.PANEL_TENSOR_CAP` parameters)
and an Elo ladder that also rates the scripted opponents. It is built from
recorded games, with no worker:

```sh
.venv-generals-ar/bin/python -m simulators.generals.autoresearch.ladder fit \
  --tournament TOURNAMENT_RESULT_DIR --players a,b --graded a=GRADE_PRIVATE.json \
  --graded b=GRADE_PRIVATE.json --anchor hunter=0 --out NEW_LADDER.json
.venv-generals-ar/bin/python -m simulators.generals.autoresearch.ladder seal \
  --ladder NEW_LADDER.json --member learned-1=a=BUNDLE_DIR --member learned-2=b=BUNDLE_DIR \
  --practice learned-1 --out NEW_PANEL_DIR
```

The tournament folder gives policy-against-policy games (`tools.tournament.generals`),
each `--graded` file a policy's games against the scripted opponents; both must
come from one release, so they share its private maps. `prepare` copies the
packages into `private/bots/`, writes the ratings into `private/panel.json` and
refuses any package that is not bound to the release's own engine and rules.
Grades and practice then report `panel_elo` beside panel points. The one sealed
panel so far, `runs/generals/panels/suite-20261113-top4` (private, not in git),
holds the four strongest networks of campaign `suite-20261113`: see
`reports/generals/learned-panel-2026-11-14.md`.

## Workers

Training runs, public practice, private grading and leases all use the one image built from `frozen/`.
A lease is a worker of a shape the researcher chooses that runs their own commands as uid 65534, with
the session's scratch Volume at `/scratch` (named in the session's `research/scratch-volume.json`; it
is cloud-only until collected for the run archive). Every command and its last output are recorded
under `research/jobs/<lease>/exec/`. `ar package` seals `weights.npz` and `model.json` from a
directory on the lease with the source sent, through the same root-side `export_entry` a training run
gets.

## Limits of what is established

Offline tests cover the ledger, bundles, scoring, the isolated policy process
and the constructor. Reference-normalized scores stay `null` until
`manage.calibrate` passes with three paired full-budget training seeds. Live
worker isolation and accounting are qualified only by a real session. Leases, the scratch Volume and
parallel jobs have run only against stubbed Modal: no lease has started on Modal for this game.
